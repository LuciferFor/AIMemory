import {
  buildExtractionMessages,
  buildCategorySelectionMessages,
  buildCleanCompactionTranscript,
  buildCleanMemoryQueryFromTurn,
  extractInboundText,
  extractTextFromLlmResult,
  fetchMemoryContext,
  fetchCategories,
  fetchWritePolicy,
  hasExplicitRememberIntent,
  isAllowedTurn,
  maskSecret,
  normalizeExtractedMemories,
  parseSelectedCategory,
  resolveConfig,
  writeMemory,
} from "./core.mjs";

function getLogger(api, config) {
  const logger =
    api?.runtime?.logging?.getChildLogger?.({ plugin: "aimemory" }, { level: "info" }) ||
    api?.logger ||
    console;
  const enabled = config.logging !== false;
  const wrap = (method) => (...args) => {
    if (!enabled) {
      return;
    }
    const fn = logger?.[method] || logger?.log || console.log;
    fn.call(logger, ...args);
  };
  return {
    info: wrap("info"),
    warn: wrap("warn"),
    error: wrap("error"),
  };
}

function getPluginConfig(event = {}) {
  return event.context?.pluginConfig || event.pluginConfig || {};
}

function resolveHookConfig(event = {}, options = {}) {
  return resolveConfig(getPluginConfig(event), options);
}

function registerHook(api, name, handler, options) {
  if (typeof api.on === "function") {
    api.on(name, handler, options);
    return;
  }
  if (typeof api.registerHook === "function") {
    api.registerHook([name], handler, options);
    return;
  }
  throw new Error("OpenClaw hook registration API is unavailable");
}

const RECENT_USER_INPUT_TTL_MS = 120000;

function memoryTurnKey(event = {}, ctx = {}) {
  return String(
    ctx.turnId ||
      ctx.messageId ||
      ctx.chatId ||
      ctx.threadId ||
      ctx.conversationId ||
      event.turnId ||
      event.messageId ||
      event.chatId ||
      event.threadId ||
      event.conversationId ||
      event.message?.id ||
      event.message?.messageId ||
      event.inbound?.id ||
      event.inbound?.messageId ||
      "default",
  );
}

function rememberRecentUserInput(cache, event = {}, ctx = {}, text = "") {
  const value = String(text || "").trim();
  if (!value) {
    return;
  }
  cache.set(memoryTurnKey(event, ctx), { text: value, at: Date.now() });
  cache.set("default", { text: value, at: Date.now() });
}

function consumeRecentUserInput(cache, event = {}, ctx = {}, maxChars = 1500) {
  const key = memoryTurnKey(event, ctx);
  const item = cache.get(key) || cache.get("default");
  if (!item) {
    return "";
  }
  cache.delete(key);
  cache.delete("default");
  if (Date.now() - item.at > RECENT_USER_INPUT_TTL_MS) {
    return "";
  }
  return item.text.slice(-maxChars).trim();
}

async function extractAndWriteMemories(api, event, ctx, config, sourceText, reason, logger) {
  if (!sourceText.trim()) {
    return { extracted: 0, written: 0 };
  }
  if (!api.runtime?.llm?.complete) {
    logger.warn("aimemory.extract skipped: api.runtime.llm.complete unavailable");
    return { extracted: 0, written: 0 };
  }

  const policy = await fetchWritePolicy(config);
  const result = await api.runtime.llm.complete({
    messages: buildExtractionMessages(policy, sourceText, reason),
    purpose: "aimemory.extract",
    maxTokens: 1400,
    temperature: 0.1,
  });
  const extractedText = extractTextFromLlmResult(result);
  const memories = normalizeExtractedMemories(extractedText);
  let written = 0;
  for (const memory of memories) {
    await writeMemory(config, memory);
    written += 1;
  }
  logger.info("aimemory.extract done", { reason, extracted: memories.length, written });
  return { extracted: memories.length, written };
}

async function selectMemoryCategory(api, config, query, options, logger) {
  const categories = await fetchCategories(config, options);
  if (!categories.length) {
    logger.info("aimemory.category skipped: no categories");
    return "";
  }
  if (!api.runtime?.llm?.complete) {
    logger.warn("aimemory.category skipped: api.runtime.llm.complete unavailable");
    return "";
  }
  const result = await api.runtime.llm.complete({
    messages: buildCategorySelectionMessages(categories, query),
    purpose: "aimemory.category",
    maxTokens: 120,
    temperature: 0,
  });
  const category = parseSelectedCategory(result, categories);
  if (!category) {
    logger.info("aimemory.category empty");
    return "";
  }
  logger.info("aimemory.category selected", { category });
  return category;
}

export function registerAIMemoryRuntime(api, options = {}) {
  const recentUserInputs = new Map();

  registerHook(
    api,
    "before_prompt_build",
    async (event = {}, ctx = {}) => {
      const config = resolveHookConfig(event, options);
      const logger = getLogger(api, config);
      if (!isAllowedTurn(event, ctx, config)) {
        return {};
      }
      const query =
        buildCleanMemoryQueryFromTurn(event, ctx, 1500) ||
        consumeRecentUserInput(recentUserInputs, event, ctx, 1500);
      if (!query) {
        return {};
      }
      try {
        const category = await selectMemoryCategory(api, config, query, options, logger);
        if (!category) {
          return {};
        }
        const result = await fetchMemoryContext(config, query, { ...options, category });
        if (!result.contextText) {
          logger.info("aimemory.context empty", { category, items: result.items.length });
          return {};
        }
        logger.info("aimemory.context injected", {
          category,
          items: result.items.length,
          chars: result.contextText.length,
        });
        return {
          prependContext: result.contextText,
        };
      } catch (error) {
        logger.warn("aimemory.context failed", {
          error: error instanceof Error ? error.message : String(error),
          baseUrl: config.baseUrl,
          apiKey: maskSecret(config.apiKey),
        });
        return {};
      }
    },
    { priority: 60, timeoutMs: 5000 },
  );

  registerHook(
    api,
    "message_received",
    async (event = {}, ctx = {}) => {
      const config = resolveHookConfig(event, options);
      const logger = getLogger(api, config);
      if (!isAllowedTurn(event, ctx, config)) {
        return;
      }
      const text = extractInboundText(event);
      rememberRecentUserInput(recentUserInputs, event, ctx, text);
      if (!config.saveOnExplicitRemember) {
        return;
      }
      if (!hasExplicitRememberIntent(text)) {
        return;
      }
      try {
        await extractAndWriteMemories(api, event, ctx, config, text, "explicit_remember", logger);
      } catch (error) {
        logger.warn("aimemory.explicit_save failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
    { priority: 40, timeoutMs: 30000 },
  );

  registerHook(
    api,
    "before_compaction",
    async (event = {}, ctx = {}) => {
      const config = resolveHookConfig(event, options);
      const logger = getLogger(api, config);
      if (!config.saveBeforeCompaction || !isAllowedTurn(event, ctx, config)) {
        return {};
      }
      const transcript = buildCleanCompactionTranscript(event, 12000, {
        includeUnstructuredTranscript: config.includeUnstructuredTranscriptForCompaction,
      });
      if (!transcript) {
        return {};
      }
      try {
        await extractAndWriteMemories(api, event, ctx, config, transcript, "before_compaction", logger);
      } catch (error) {
        logger.warn("aimemory.compaction_save failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
      return {};
    },
    { priority: 40, timeoutMs: 60000 },
  );
}

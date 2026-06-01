import {
  buildExtractionMessages,
  buildCategorySelectionMessages,
  buildCleanCompactionTranscript,
  buildCleanCompactionTranscriptFromSessionFile,
  buildCleanMemoryQueryFromTurn,
  extractInboundText,
  extractTextFromLlmResult,
  aimemoryRequest,
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
import { homedir } from "node:os";
import path from "node:path";
import { readdirSync, readFileSync } from "node:fs";

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

function compactionSaveKey(event = {}, ctx = {}) {
  return String(
    ctx.sessionId ||
      event.sessionId ||
      ctx.sessionKey ||
      event.sessionKey ||
      event.key ||
      "default",
  );
}

function markCompactionSave(savedCompactions, key, ttlMs = 120000) {
  savedCompactions.add(key);
  const timer = setTimeout(() => savedCompactions.delete(key), ttlMs);
  timer.unref?.();
}

function inferSessionFile(event = {}, ctx = {}) {
  const explicit = event.sessionFile || ctx.sessionFile;
  if (explicit) {
    return String(explicit);
  }
  const sessionId = ctx.sessionId || event.sessionId;
  if (!sessionId) {
    return "";
  }
  const agentId = ctx.agentId || event.agentId || event.context?.agentId || "main";
  return path.join(homedir(), ".openclaw", "agents", String(agentId), "sessions", `${sessionId}.jsonl`);
}

function buildCompactionTranscript(event = {}, ctx = {}, config = {}, logger, options = {}) {
  let transcript = buildCleanCompactionTranscript(event, 12000, {
    includeUnstructuredTranscript: config.includeUnstructuredTranscriptForCompaction,
  });
  if (transcript) {
    return transcript;
  }

  const sessionFile = inferSessionFile(event, ctx);
  if (!sessionFile) {
    return "";
  }
  transcript = buildCleanCompactionTranscriptFromSessionFile(sessionFile, 12000, {
    readFile: options.readFile,
  });
  if (transcript) {
    logger?.info?.("aimemory.compaction transcript loaded", {
      source: event.sessionFile || ctx.sessionFile ? "sessionFile" : "inferredSessionFile",
      chars: transcript.length,
    });
  }
  return transcript;
}

function readJsonFile(file, options = {}) {
  const readFile = options.readFile || readFileSync;
  return JSON.parse(readFile(file, "utf8"));
}

function listCodexSessionStores(options = {}) {
  const root = options.openclawHome || path.join(homedir(), ".openclaw");
  const agentsDir = path.join(root, "agents");
  let entries = [];
  try {
    entries = readdirSync(agentsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({
      agentId: entry.name,
      file: path.join(agentsDir, entry.name, "sessions", "sessions.json"),
    }));
}

function readSessionEntries(store, options = {}) {
  let data;
  try {
    data = readJsonFile(store.file, options);
  } catch {
    return [];
  }
  const sessions = data?.sessions && typeof data.sessions === "object" ? data.sessions : data;
  if (!sessions || typeof sessions !== "object") {
    return [];
  }
  return Object.entries(sessions).map(([sessionKey, entry]) => ({
    sessionKey,
    agentId: store.agentId,
    entry: entry && typeof entry === "object" ? entry : {},
  }));
}

function startCodexCompactionWatcher(api, initialConfig, logger, savedCompactions, options = {}) {
  const seenCounts = new Map();
  let running = false;
  let stopped = false;
  let initialized = false;
  const intervalMs = initialConfig.compactionWatcherIntervalMs;

  const tick = async () => {
    if (running || stopped) {
      return;
    }
    running = true;
    try {
      const config = resolveConfig(initialConfig, options);
      if (!config.enabled || !config.saveBeforeCompaction || !config.watchCodexCompaction) {
        return;
      }
      for (const store of listCodexSessionStores(options)) {
        for (const { sessionKey, agentId, entry } of readSessionEntries(store, options)) {
          const sessionId = entry.sessionId || entry.id || "";
          const key = `${agentId}:${sessionKey}:${sessionId}`;
          const count = Number(entry.compactionCount || 0);
          const previous = seenCounts.get(key);
          seenCounts.set(key, count);
          if (!initialized || previous === undefined || count <= previous) {
            continue;
          }
          logger.info("aimemory.compaction watcher detected", {
            agentId,
            sessionId,
            sessionKey,
            previous,
            count,
          });
          const saveKey = compactionSaveKey({ sessionId, sessionKey }, { sessionId, sessionKey });
          if (savedCompactions.has(saveKey)) {
            continue;
          }
          markCompactionSave(savedCompactions, saveKey);
          const saved = await saveCompactionMemories(
            api,
            {
              sessionFile: entry.sessionFile,
              sessionId,
              sessionKey,
              messageCount: 0,
            },
            {
              agentId,
              sessionId,
              sessionKey,
              trigger: "codex_compaction_watcher",
            },
            config,
            "codex_compaction_watcher",
            logger,
            options,
          );
          if (!saved) {
            savedCompactions.delete(saveKey);
          }
        }
      }
      initialized = true;
    } catch (error) {
      logger.warn("aimemory.compaction watcher failed", {
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      running = false;
    }
  };

  const timer = setInterval(tick, intervalMs);
  timer.unref?.();
  tick();
  logger.info("aimemory.compaction watcher started", { intervalMs });
  return () => {
    stopped = true;
    clearInterval(timer);
  };
}

async function saveCompactionMemories(api, event = {}, ctx = {}, config, reason, logger, options = {}) {
  const transcript = buildCompactionTranscript(event, ctx, config, logger, options);
  if (!transcript) {
    logger.info("aimemory.compaction skipped: empty transcript", {
      hasSessionFile: Boolean(event.sessionFile || ctx.sessionFile),
      hasSessionId: Boolean(event.sessionId || ctx.sessionId),
      messageCount: event.messageCount,
    });
    return false;
  }
  if (config.useBackendExtraction) {
    await extractCompactionViaBackend(config, transcript, reason, event, ctx, logger, options);
    return true;
  }
  await extractAndWriteMemories(api, event, ctx, config, transcript, reason, logger, options);
  return true;
}

async function extractCompactionViaBackend(config, transcript, reason, event, ctx, logger, options = {}) {
  const response = await aimemoryRequest(
    {
      ...config,
      timeoutMs: Math.max(Number(config.timeoutMs) || 0, 120000),
    },
    "POST",
    "/v1/memories/extract",
    {
      agent_id: config.agentId,
      transcript,
      reason,
      metadata: {
        session_id: ctx.sessionId || event.sessionId,
        session_key: ctx.sessionKey || event.sessionKey || event.key,
        trigger: ctx.trigger || event.trigger,
        source: "openclaw_aimemory_plugin",
      },
    },
    options,
  );
  logger.info("aimemory.extract backend done", {
    reason,
    extracted: response?.extracted,
    written: response?.written,
  });
  return response;
}

async function extractAndWriteMemories(api, event, ctx, config, sourceText, reason, logger, options = {}) {
  if (!sourceText.trim()) {
    return { extracted: 0, written: 0 };
  }
  if (!api.runtime?.llm?.complete) {
    logger.warn("aimemory.extract skipped: api.runtime.llm.complete unavailable");
    return { extracted: 0, written: 0 };
  }

  const policy = await fetchWritePolicy(config, options);
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
    await writeMemory(config, memory, options);
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
  const savedCompactions = new Set();
  let stopCompactionWatcher = null;

  const maybeStartCompactionWatcher = (event = {}) => {
    if (stopCompactionWatcher) {
      return;
    }
    const config = resolveHookConfig(event, options);
    const logger = getLogger(api, config);
    if (!config.watchCodexCompaction) {
      return;
    }
    stopCompactionWatcher = startCodexCompactionWatcher(api, config, logger, savedCompactions, options);
  };

  if (
    options.startCompactionWatcher !== false &&
    options.envValues === undefined &&
    options.processEnv === undefined &&
    options.readFile === undefined &&
    options.fetchImpl === undefined
  ) {
    const timer = setTimeout(() => maybeStartCompactionWatcher({}), 0);
    timer.unref?.();
  }

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
        await extractAndWriteMemories(api, event, ctx, config, text, "explicit_remember", logger, options);
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
      const key = compactionSaveKey(event, ctx);
      if (savedCompactions.has(key)) {
        return {};
      }
      markCompactionSave(savedCompactions, key);
      try {
        const saved = await saveCompactionMemories(
          api,
          event,
          ctx,
          config,
          "before_compaction",
          logger,
          options,
        );
        if (!saved) {
          savedCompactions.delete(key);
        }
      } catch (error) {
        savedCompactions.delete(key);
        logger.warn("aimemory.compaction_save failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
      return {};
    },
    { priority: 40, timeoutMs: 60000 },
  );

  registerHook(
    api,
    "after_compaction",
    async (event = {}, ctx = {}) => {
      const config = resolveHookConfig(event, options);
      const logger = getLogger(api, config);
      if (!config.saveBeforeCompaction || !isAllowedTurn(event, ctx, config)) {
        return {};
      }
      const key = compactionSaveKey(event, ctx);
      if (savedCompactions.has(key)) {
        return {};
      }
      markCompactionSave(savedCompactions, key);
      try {
        const saved = await saveCompactionMemories(
          api,
          event,
          ctx,
          config,
          "after_compaction_fallback",
          logger,
          options,
        );
        if (!saved) {
          savedCompactions.delete(key);
        }
      } catch (error) {
        savedCompactions.delete(key);
        logger.warn("aimemory.compaction_save failed", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
      return {};
    },
    { priority: 40, timeoutMs: 60000 },
  );

  registerHook(
    api,
    "gateway_start",
    async (event = {}) => {
      maybeStartCompactionWatcher(event);
    },
    { priority: 20, timeoutMs: 5000 },
  );

  registerHook(
    api,
    "gateway_stop",
    async () => {
      if (stopCompactionWatcher) {
        stopCompactionWatcher();
        stopCompactionWatcher = null;
      }
    },
    { priority: 20, timeoutMs: 5000 },
  );
}

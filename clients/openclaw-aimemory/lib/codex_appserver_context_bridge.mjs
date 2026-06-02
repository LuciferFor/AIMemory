import { readFileSync } from "node:fs";
import {
  aimemoryRequest,
  maskSecret,
  resolveConfig,
} from "./core.mjs";

const DEFAULT_OPENCLAW_CONFIG = "/home/node/.openclaw/openclaw.json";

function containerPath(filePath) {
  const value = String(filePath || "");
  if (value.startsWith("/home/lucifer/.openclaw/")) {
    return value.replace("/home/lucifer/.openclaw/", "/home/node/.openclaw/");
  }
  return value;
}

function readPluginConfig(openclawConfigFile = DEFAULT_OPENCLAW_CONFIG) {
  try {
    const config = JSON.parse(readFileSync(openclawConfigFile, "utf8"));
    const pluginConfig = config?.plugins?.entries?.aimemory?.config || {};
    return {
      ...pluginConfig,
      envFile: containerPath(pluginConfig.envFile),
    };
  } catch {
    return {};
  }
}

function trimQuery(value, maxChars = 1500) {
  return String(value || "").trim().slice(-maxChars).trim();
}

export async function fetchCodexAppServerMemoryContext(params = {}, options = {}) {
  const pluginConfig = options.pluginConfig || readPluginConfig(options.openclawConfigFile);
  const config = resolveConfig(pluginConfig, options);
  if (!config.enabled) {
    return { contextText: "", items: [], skipped: true, reason: "disabled" };
  }

  const query = trimQuery(params.prompt || params.query || params.input);
  if (!query) {
    return { contextText: "", items: [], skipped: true, reason: "empty_query" };
  }

  try {
    const response = await aimemoryRequest(
      config,
      "POST",
      "/v1/memories/context",
      {
        agent_id: config.agentId,
        ...(config.deviceId ? { device_id: config.deviceId } : {}),
        query,
        top_k: config.topK,
        max_chars: config.maxChars,
      },
      options,
    );
    const contextText = String(response.context_text || "");
    const items = Array.isArray(response.items) ? response.items : [];
    console.info("[aimemory] codex app-server context fetched", {
      sessionFile: params.sessionFile,
      chars: contextText.length,
      items: items.length,
    });
    return { contextText, items, skipped: false };
  } catch (error) {
    console.warn("[aimemory] codex app-server context failed", {
      sessionFile: params.sessionFile,
      baseUrl: config.baseUrl,
      apiKey: maskSecret(config.apiKey),
      error: error instanceof Error ? error.message : String(error),
    });
    return {
      contextText: "",
      items: [],
      skipped: true,
      reason: "context_failed",
      error,
    };
  }
}

export async function buildCodexAppServerPromptWithMemory(params = {}, options = {}) {
  const prompt = String(params.prompt || "");
  const result = await fetchCodexAppServerMemoryContext(params, options);
  if (!result.contextText) {
    return prompt;
  }
  return `${result.contextText}\n\n当前用户消息:\n${prompt}`;
}

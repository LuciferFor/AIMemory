import { readFileSync } from "node:fs";
import {
  aimemoryRequest,
  buildCleanCompactionTranscriptFromSessionFile,
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

function sessionFileFromParams(params = {}) {
  return containerPath(params.sessionFile || params.ctx?.sessionFile || "");
}

export async function runCodexAppServerCompactionBridge(params = {}, options = {}) {
  const pluginConfig = options.pluginConfig || readPluginConfig(options.openclawConfigFile);
  const config = resolveConfig(pluginConfig, options);
  if (!config.enabled || !config.saveBeforeCompaction) {
    return { ok: true, skipped: true, reason: "disabled" };
  }

  const sessionFile = sessionFileFromParams(params);
  const transcript = buildCleanCompactionTranscriptFromSessionFile(sessionFile, 12000, options);
  if (!transcript) {
    console.warn("[aimemory] codex app-server compaction skipped: empty transcript", {
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      sessionFile,
    });
    return { ok: true, skipped: true, reason: "empty_transcript" };
  }

  const requestConfig = {
    ...config,
    timeoutMs: Math.max(Number(config.timeoutMs) || 0, 120000),
  };
  try {
    const response = await aimemoryRequest(
      requestConfig,
      "POST",
      "/v1/memories/extract",
      {
        agent_id: config.agentId,
        ...(config.deviceId ? { device_id: config.deviceId } : {}),
        transcript,
        reason: "codex_appserver_compaction",
        metadata: {
          session_id: params.sessionId,
          session_key: params.sessionKey || params.sandboxSessionKey,
          trigger: params.trigger,
          source: "openclaw_codex_appserver",
        },
      },
      options,
    );
    console.info("[aimemory] codex app-server compaction extracted memories", {
      sessionId: params.sessionId,
      extracted: response?.extracted,
      written: response?.written,
    });
    return { ok: true, response };
  } catch (error) {
    console.warn("[aimemory] codex app-server compaction extract failed", {
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      baseUrl: config.baseUrl,
      apiKey: maskSecret(config.apiKey),
      error: error instanceof Error ? error.message : String(error),
    });
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

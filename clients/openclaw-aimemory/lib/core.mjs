import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

export const DEFAULT_CONFIG = Object.freeze({
  enabled: true,
  baseUrl: "http://192.168.31.11:10011",
  agentId: "5df9cbfb-d31b-46dd-972b-05d466d2257c",
  envFile: "~/.openclaw/aimemory.env",
  allowedChatTypes: ["direct", "private", "dm"],
  allowedAgents: [],
  topK: 8,
  maxChars: 3000,
  timeoutMs: 3000,
  saveOnExplicitRemember: true,
  saveBeforeCompaction: true,
  includePromptInMemoryQuery: false,
  includeUnstructuredTranscriptForCompaction: false,
  logging: true,
});

const FORBIDDEN_MEMORY_PATTERNS = [
  /password/i,
  /passwd/i,
  /api[_\s-]?key/i,
  /secret/i,
  /token/i,
  /private[_\s-]?key/i,
  /sudo/i,
  /密码/,
  /密钥/,
  /私钥/,
  /令牌/,
  /口令/,
];

const REMEMBER_PATTERNS = [
  /记住/,
  /记下来/,
  /帮我记/,
  /以后.*记得/,
  /下次.*记得/,
  /remember this/i,
  /please remember/i,
  /keep this in memory/i,
];

export function expandHome(filePath) {
  if (!filePath || typeof filePath !== "string") {
    return filePath;
  }
  if (filePath === "~") {
    return homedir();
  }
  if (filePath.startsWith("~/") || filePath.startsWith("~\\")) {
    return path.join(homedir(), filePath.slice(2));
  }
  return filePath;
}

export function parseEnvText(text) {
  const values = {};
  for (const rawLine of String(text || "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) {
      continue;
    }
    const index = line.indexOf("=");
    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    values[key] = value;
  }
  return values;
}

export function loadEnvFile(envFile = DEFAULT_CONFIG.envFile, readFile = readFileSync) {
  const resolved = expandHome(envFile);
  if (!resolved) {
    return {};
  }
  try {
    return parseEnvText(readFile(resolved, "utf8"));
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return {};
    }
    throw error;
  }
}

export function normalizeBaseUrl(baseUrl) {
  return String(baseUrl || DEFAULT_CONFIG.baseUrl).replace(/\/+$/, "");
}

export function clampInteger(value, fallback, min, max) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

export function normalizeStringArray(value, fallback) {
  if (!Array.isArray(value)) {
    return [...fallback];
  }
  return value.map((item) => String(item).trim()).filter(Boolean);
}

export function resolveConfig(pluginConfig = {}, options = {}) {
  const envValues =
    options.envValues ??
    loadEnvFile(pluginConfig.envFile ?? DEFAULT_CONFIG.envFile, options.readFile);
  const processEnv = options.processEnv ?? process.env;
  const merged = {
    ...DEFAULT_CONFIG,
    ...pluginConfig,
  };

  return {
    ...merged,
    baseUrl: normalizeBaseUrl(
      processEnv.AIMEMORY_BASE_URL || envValues.AIMEMORY_BASE_URL || merged.baseUrl,
    ),
    apiKey: processEnv.AIMEMORY_API_KEY || envValues.AIMEMORY_API_KEY || "",
    agentId: processEnv.AIMEMORY_AGENT_ID || envValues.AIMEMORY_AGENT_ID || merged.agentId,
    allowedChatTypes: normalizeStringArray(
      merged.allowedChatTypes,
      DEFAULT_CONFIG.allowedChatTypes,
    ).map((item) => item.toLowerCase()),
    allowedAgents: normalizeStringArray(merged.allowedAgents, DEFAULT_CONFIG.allowedAgents),
    topK: clampInteger(merged.topK, DEFAULT_CONFIG.topK, 1, 50),
    maxChars: clampInteger(merged.maxChars, DEFAULT_CONFIG.maxChars, 0, 12000),
    timeoutMs: clampInteger(merged.timeoutMs, DEFAULT_CONFIG.timeoutMs, 500, 600000),
    enabled: merged.enabled !== false,
    saveOnExplicitRemember: merged.saveOnExplicitRemember !== false,
    saveBeforeCompaction: merged.saveBeforeCompaction !== false,
    includePromptInMemoryQuery: merged.includePromptInMemoryQuery === true,
    includeUnstructuredTranscriptForCompaction:
      merged.includeUnstructuredTranscriptForCompaction === true,
    logging: merged.logging !== false,
  };
}

export function maskSecret(value) {
  const text = String(value || "");
  if (!text) {
    return "";
  }
  if (text.length <= 12) {
    return "***";
  }
  return `${text.slice(0, 6)}...${text.slice(-4)}`;
}

export function extractText(value) {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(extractText).filter(Boolean).join("\n");
  }
  if (typeof value === "object") {
    if (typeof value.text === "string") {
      return value.text;
    }
    if (typeof value.content === "string") {
      return value.content;
    }
    if (Array.isArray(value.content)) {
      return extractText(value.content);
    }
    if (typeof value.message === "string") {
      return value.message;
    }
  }
  return "";
}

export function extractPromptText(event = {}) {
  return (
    extractText(event.prompt) ||
    extractText(event.input) ||
    extractText(event.userInput) ||
    extractText(event.currentPrompt) ||
    extractText(event.message)
  );
}

export function extractInboundText(event = {}) {
  return (
    extractText(event.content) ||
    extractText(event.text) ||
    extractText(event.body) ||
    extractText(event.message) ||
    extractText(event.inbound)
  );
}

function normalizeRole(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/^role:/, "");
}

function messageRole(message = {}) {
  if (!message || typeof message !== "object") {
    return "";
  }
  return normalizeRole(
    message.role ||
      message.author?.role ||
      message.sender?.role ||
      message.from?.role ||
      message.type ||
      message.kind ||
      "",
  );
}

function isConversationRole(role) {
  return ["user", "human", "assistant", "ai"].includes(normalizeRole(role));
}

function isUserRole(role) {
  return ["user", "human"].includes(normalizeRole(role));
}

function roleLabel(role) {
  return ["assistant", "ai"].includes(normalizeRole(role)) ? "assistant" : "user";
}

function collectConversationMessages(event = {}) {
  const sources = [event.messages, event.history, event.sessionMessages, event.conversation];
  const messages = [];
  for (const source of sources) {
    if (Array.isArray(source)) {
      messages.push(...source);
    }
  }
  return messages;
}

function addUnique(parts, seen, value) {
  const text = extractText(value).trim();
  if (!text || seen.has(text)) {
    return;
  }
  parts.push(text);
  seen.add(text);
}

function addUserMessageCandidate(parts, seen, value, { allowRoleless = false } = {}) {
  if (value == null) {
    return;
  }
  if (typeof value === "string" && !allowRoleless) {
    return;
  }
  if (typeof value === "object" && !Array.isArray(value)) {
    const role = messageRole(value);
    if (role && !isUserRole(role)) {
      return;
    }
    if (!role && !allowRoleless) {
      return;
    }
  }
  addUnique(parts, seen, value);
}

export function extractCurrentUserInputText(event = {}) {
  const parts = [];
  const seen = new Set();
  for (const value of [
    event.userInput,
    event.user_input,
    event.currentUserInput,
    event.current_user_input,
    event.userText,
    event.user_text,
    event.userMessage,
    event.user_message,
    event.userPrompt,
    event.user_prompt,
  ]) {
    addUnique(parts, seen, value);
  }

  for (const value of [
    event.inbound,
    event.currentMessage,
    event.current_message,
    event.input,
    event.currentInput,
    event.current_input,
  ]) {
    addUserMessageCandidate(parts, seen, value);
  }

  for (const value of [event.message]) {
    addUserMessageCandidate(parts, seen, value);
  }
  return parts.join("\n").trim();
}

export function buildCleanMemoryQueryFromTurn(event = {}, ctx = {}, maxChars = 1500, options = {}) {
  const parts = [];
  const seen = new Set();

  addUnique(parts, seen, extractCurrentUserInputText(event));
  return parts.join("\n").slice(-maxChars).trim();
}

export function buildQueryFromTurn(event = {}, maxChars = 1500) {
  const parts = [];
  const prompt = extractPromptText(event);
  if (prompt) {
    parts.push(prompt);
  }
  const messages = Array.isArray(event.messages) ? event.messages : [];
  for (const message of messages.slice(-4)) {
    const text = extractText(message);
    if (text) {
      parts.push(text);
    }
  }
  return parts.join("\n").slice(-maxChars).trim();
}

export function buildCleanCompactionTranscript(event = {}, maxChars = 12000, options = {}) {
  const lines = [];
  const seen = new Set();
  for (const message of collectConversationMessages(event)) {
    const role = messageRole(message);
    if (!isConversationRole(role)) {
      continue;
    }
    const text = extractText(message).trim();
    if (!text || seen.has(`${role}:${text}`)) {
      continue;
    }
    lines.push(`${roleLabel(role)}: ${text}`);
    seen.add(`${role}:${text}`);
  }

  if (!lines.length && options.includeUnstructuredTranscript === true) {
    for (const value of [event.transcript, event.summary]) {
      const text = extractText(value).trim();
      if (text) {
        lines.push(text);
      }
    }
  }

  return lines.join("\n").slice(-maxChars).trim();
}

export function buildTranscriptText(event = {}, maxChars = 12000) {
  const candidates = [
    event.transcript,
    event.summary,
    event.prompt,
    event.messages,
    event.history,
    event.sessionMessages,
  ];
  const text = candidates.map(extractText).filter(Boolean).join("\n");
  return text.slice(-maxChars).trim();
}

export function extractAgentId(event = {}, ctx = {}) {
  return (
    ctx.agentId ||
    ctx.agent?.id ||
    event.agentId ||
    event.agent?.id ||
    event.context?.agentId ||
    event.context?.agent?.id ||
    ""
  );
}

export function extractChatType(event = {}, ctx = {}) {
  const metadata = event.metadata || event.message?.metadata || event.inbound?.metadata || {};
  const value =
    ctx.chatType ||
    ctx.threadType ||
    ctx.messageType ||
    event.chatType ||
    event.threadType ||
    event.conversationType ||
    event.message?.chatType ||
    event.message?.messageType ||
    event.inbound?.chatType ||
    metadata.chatType ||
    metadata.chat_type ||
    metadata.messageType ||
    metadata.message_type ||
    metadata.detailType ||
    metadata.detail_type ||
    metadata.conversationType ||
    metadata.conversation_type ||
    "";
  return String(value || "").trim().toLowerCase();
}

export function isAllowedTurn(event = {}, ctx = {}, config = DEFAULT_CONFIG) {
  if (config.enabled === false) {
    return false;
  }
  const agentId = extractAgentId(event, ctx);
  if (config.allowedAgents?.length && agentId && !config.allowedAgents.includes(agentId)) {
    return false;
  }
  const chatType = extractChatType(event, ctx);
  if (!chatType) {
    // Local app-server sessions often do not carry a channel chat type; allow them.
    return true;
  }
  return config.allowedChatTypes.includes(chatType);
}

export function hasExplicitRememberIntent(text) {
  const value = String(text || "");
  if (!value.trim()) {
    return false;
  }
  return REMEMBER_PATTERNS.some((pattern) => pattern.test(value));
}

export function containsForbiddenMemoryText(...values) {
  const text = values.map((value) => String(value || "")).join("\n");
  return FORBIDDEN_MEMORY_PATTERNS.some((pattern) => pattern.test(text));
}

export function sanitizeExternalId(value) {
  const text = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 180);
  return text || "";
}

export function stableExternalId(memory) {
  const metadata = memory?.metadata && typeof memory.metadata === "object" ? memory.metadata : {};
  const category = sanitizeExternalId(memory?.category || metadata.category || metadata.kind || "memory") || "memory";
  const title = String(memory?.title || "");
  const content = String(memory?.content || "");
  const digest = createHash("sha256").update(`${title}\n${content}`).digest("hex").slice(0, 12);
  return `auto-${category}-${digest}`;
}

export function stripJsonCodeFence(text) {
  const value = String(text || "").trim();
  const fenced = value.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fenced ? fenced[1].trim() : value;
}

export function parseMemoryJson(text) {
  const cleaned = stripJsonCodeFence(text);
  const candidates = [cleaned];
  const arrayStart = cleaned.indexOf("[");
  const arrayEnd = cleaned.lastIndexOf("]");
  if (arrayStart >= 0 && arrayEnd > arrayStart) {
    candidates.push(cleaned.slice(arrayStart, arrayEnd + 1));
  }
  const objectStart = cleaned.indexOf("{");
  const objectEnd = cleaned.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    candidates.push(cleaned.slice(objectStart, objectEnd + 1));
  }

  let lastError;
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) {
        return parsed;
      }
      if (Array.isArray(parsed.items)) {
        return parsed.items;
      }
      if (Array.isArray(parsed.memories)) {
        return parsed.memories;
      }
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("No JSON memory array found");
}

export function normalizeMemoryCandidate(candidate) {
  if (!candidate || typeof candidate !== "object") {
    return null;
  }
  const title = String(candidate.title || "").trim();
  const content = String(candidate.content || "").trim();
  if (!title || !content) {
    return null;
  }
  if (containsForbiddenMemoryText(title, content)) {
    return null;
  }
  const metadata =
    candidate.metadata && typeof candidate.metadata === "object" && !Array.isArray(candidate.metadata)
      ? candidate.metadata
      : {};
  const category = String(candidate.category || metadata.category || metadata.kind || "").trim();
  if (!category) {
    return null;
  }
  const memory = {
    external_id: sanitizeExternalId(candidate.external_id || candidate.externalId || ""),
    category: category.slice(0, 128),
    title: title.slice(0, 512),
    content,
    metadata,
  };
  if (!memory.external_id) {
    memory.external_id = stableExternalId(memory);
  }
  if (candidate.occurred_at || candidate.occurredAt) {
    memory.occurred_at = candidate.occurred_at || candidate.occurredAt;
  }
  return memory;
}

export function normalizeExtractedMemories(value) {
  return parseMemoryJson(value).map(normalizeMemoryCandidate).filter(Boolean);
}

export async function aimemoryRequest(config, method, pathName, payload, options = {}) {
  if (!config.apiKey) {
    throw new Error("AIMEMORY_API_KEY is required");
  }
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("fetch is not available");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetchImpl(`${normalizeBaseUrl(config.baseUrl)}${pathName}`, {
      method,
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json; charset=utf-8",
      },
      body: payload == null ? undefined : JSON.stringify(payload),
      signal: controller.signal,
    });
    const text = await response.text();
    let body = {};
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { raw: text };
      }
    }
    if (!response.ok) {
      const message = typeof body.detail === "string" ? body.detail : response.statusText;
      throw new Error(`AIMemory HTTP ${response.status}: ${message}`);
    }
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchMemoryContext(config, query, options = {}) {
  const category = options.category || config.category;
  if (!category) {
    return { contextText: "", items: [] };
  }
  const response = await aimemoryRequest(
    config,
    "POST",
    "/v1/memories/context",
    {
      agent_id: config.agentId,
      category,
      query,
      top_k: config.topK,
      max_chars: config.maxChars,
    },
    options,
  );
  return {
    contextText: String(response.context_text || ""),
    items: Array.isArray(response.items) ? response.items : [],
  };
}

export async function fetchWritePolicy(config, options = {}) {
  return aimemoryRequest(config, "GET", "/v1/memories/write-policy", null, options);
}

export async function fetchCategories(config, options = {}) {
  const response = await aimemoryRequest(config, "GET", "/v1/memories/categories", null, options);
  return Array.isArray(response.items) ? response.items : [];
}

export async function writeMemory(config, memory, options = {}) {
  return aimemoryRequest(
    config,
    "POST",
    "/v1/memories",
    {
      agent_id: config.agentId,
      external_id: memory.external_id,
      category: memory.category,
      title: memory.title,
      content: memory.content,
      metadata: memory.metadata || {},
      occurred_at: memory.occurred_at,
    },
    options,
  );
}

export function extractTextFromLlmResult(result) {
  if (typeof result === "string") {
    return result;
  }
  if (!result || typeof result !== "object") {
    return "";
  }
  return (
    extractText(result.text) ||
    extractText(result.content) ||
    extractText(result.message) ||
    extractText(result.output) ||
    extractText(result.choices?.[0]?.message?.content) ||
    ""
  );
}

export function buildExtractionMessages(policy, sourceText, reason) {
  const prompt = String(policy?.prompt || "Extract durable long-term memories as JSON.");
  const schema = policy?.output_schema ? JSON.stringify(policy.output_schema, null, 2) : "";
  const categories = Array.isArray(policy?.categories) ? policy.categories : [];
  const categoryText = categories.length
    ? `\n\n已有分类列表:\n${categories
        .map((item) => `- ${item.name}${item.description ? `：${item.description}` : ""}`)
        .join("\n")}`
    : "\n\n已有分类列表为空。没有合适分类时可以创建简短明确的新分类。";
  return [
    {
      role: "system",
      content: `${prompt}${categoryText}\n\n请使用第三方视角提取和改写记忆，写成“用户……”“助手应……”这类表述；不要写成“我喜欢”“我应该”。只输出 JSON 数组，不要输出解释。${schema ? `\n\n推荐格式:\n${schema}` : ""}`,
    },
    {
      role: "user",
      content: `来源: ${reason}\n\n请从以下内容提取值得长期保存的记忆，忽略密码、密钥、token、sudo 密码和一次性闲聊。\n\n${sourceText}`,
    },
  ];
}

export function buildCategorySelectionMessages(categories, query) {
  const list = Array.isArray(categories) ? categories : [];
  const categoryText = list
    .map((item) => `- ${item.name}${item.description ? `：${item.description}` : ""}`)
    .join("\n");
  return [
    {
      role: "system",
      content:
        "你要为本轮用户请求选择一个长期记忆事务分类。只能从已有分类中选择；如果没有明确合适分类，输出 null。只输出 JSON，例如 {\"category\":\"爱吃的水果\"} 或 {\"category\":null}。",
    },
    {
      role: "user",
      content: `已有分类:\n${categoryText || "无"}\n\n当前请求:\n${query}`,
    },
  ];
}

export function parseSelectedCategory(value, categories) {
  const text = stripJsonCodeFence(extractTextFromLlmResult(value));
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    const normalizedText = String(text || "").trim();
    parsed = { category: normalizedText || null };
  }
  const selected = String(parsed?.category || "").trim();
  if (!selected) {
    return "";
  }
  const known = new Set((categories || []).map((item) => String(item.name || "").trim()));
  return known.has(selected) ? selected : "";
}

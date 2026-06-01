import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildCategorySelectionMessages,
  buildCleanCompactionTranscript,
  buildCleanCompactionTranscriptFromSessionFile,
  buildCleanMemoryQueryFromTurn,
  buildExtractionMessages,
  buildQueryFromTurn,
  containsForbiddenMemoryText,
  fetchCategories,
  fetchMemoryContext,
  hasExplicitRememberIntent,
  isAllowedTurn,
  normalizeExtractedMemories,
  parseSelectedCategory,
  parseEnvText,
  parseSessionJsonlMessages,
  resolveConfig,
  stableExternalId,
} from "../lib/core.mjs";
import { registerAIMemoryRuntime } from "../lib/runtime.mjs";

test("env/config merge prefers AIMEMORY env values", () => {
  const config = resolveConfig(
    { baseUrl: "http://config", agentId: "config-agent", topK: 99 },
    {
      envValues: {
        AIMEMORY_BASE_URL: "http://env/",
        AIMEMORY_API_KEY: "aim_test",
        AIMEMORY_AGENT_ID: "env-agent",
      },
      processEnv: {},
    },
  );

  assert.equal(config.baseUrl, "http://env");
  assert.equal(config.apiKey, "aim_test");
  assert.equal(config.agentId, "env-agent");
  assert.equal(config.topK, 50);
  assert.equal(config.includePromptInMemoryQuery, false);
  assert.equal(config.includeUnstructuredTranscriptForCompaction, false);
});

test("parseEnvText handles quotes and comments", () => {
  assert.deepEqual(
    parseEnvText("A=1\n# no\nB='two'\nC=\"three\"\n"),
    { A: "1", B: "two", C: "three" },
  );
});

test("direct/private/dm allowed and group/channel skipped", () => {
  const config = resolveConfig({}, { envValues: { AIMEMORY_API_KEY: "k" }, processEnv: {} });

  assert.equal(isAllowedTurn({ chatType: "direct" }, {}, config), true);
  assert.equal(isAllowedTurn({ chatType: "private" }, {}, config), true);
  assert.equal(isAllowedTurn({ chatType: "dm" }, {}, config), true);
  assert.equal(isAllowedTurn({ chatType: "group" }, {}, config), false);
  assert.equal(isAllowedTurn({ chatType: "channel" }, {}, config), false);
});

test("buildQueryFromTurn uses prompt and recent messages", () => {
  const query = buildQueryFromTurn({
    prompt: "当前问题",
    messages: [
      { role: "user", content: "旧消息" },
      { role: "assistant", content: "旧回答" },
    ],
  });

  assert.match(query, /当前问题/);
  assert.match(query, /旧消息/);
});

test("clean query only uses current user input", () => {
  const query = buildCleanMemoryQueryFromTurn({
    prompt: "IDENTITY.md 永久人设\nSOUL.md 灵魂设定",
    userInput: "用户正在问苹果偏好",
    messages: [
      { role: "system", content: "MEMORY.md 永久记忆" },
      { role: "developer", content: "不要发送给 AIMemory" },
      { role: "user", content: "我喜欢香蕉" },
      { role: "assistant", content: "我记得了" },
    ],
  });

  assert.match(query, /用户正在问苹果偏好/);
  assert.doesNotMatch(query, /我喜欢香蕉/);
  assert.doesNotMatch(query, /我记得了/);
  assert.doesNotMatch(query, /IDENTITY\.md/);
  assert.doesNotMatch(query, /SOUL\.md/);
  assert.doesNotMatch(query, /MEMORY\.md/);
  assert.doesNotMatch(query, /不要发送给 AIMemory/);
});

test("clean query skips prompt-only and assistant-only turns", () => {
  const event = { prompt: "IDENTITY.md 永久人设" };

  assert.equal(buildCleanMemoryQueryFromTurn(event), "");
  assert.equal(buildCleanMemoryQueryFromTurn(event, {}, 1500, { includePrompt: true }), "");
  assert.equal(buildCleanMemoryQueryFromTurn({ input: "dark armor fantasy poster" }), "");
  assert.equal(buildCleanMemoryQueryFromTurn({ input: "入队了，宝贝。 任务号：`123`" }), "");
  assert.equal(
    buildCleanMemoryQueryFromTurn({
      message: { role: "assistant", content: "这是一段模型回复" },
      messages: [{ role: "user", content: "历史用户消息" }],
    }),
    "",
  );
});

test("clean query accepts only explicit current user fields or role user objects", () => {
  assert.equal(buildCleanMemoryQueryFromTurn({ input: "老婆来点福利美照" }), "老婆来点福利美照");
  assert.equal(
    buildCleanMemoryQueryFromTurn({
      input: { role: "user", content: "老婆来点福利美照" },
      currentMessage: { role: "assistant", content: "assistant reply" },
    }),
    "老婆来点福利美照",
  );
  assert.equal(
    buildCleanMemoryQueryFromTurn({
      userInput: "中文用户输入",
      input: "internal generated prompt",
    }),
    "中文用户输入",
  );
});

test("clean compaction transcript uses only structured dialogue by default", () => {
  const transcript = buildCleanCompactionTranscript({
    prompt: "IDENTITY.md 永久人设",
    transcript: "SOUL.md 灵魂设定",
    messages: [
      { role: "system", content: "MEMORY.md 永久记忆" },
      { role: "user", content: "我喜欢苹果" },
      { role: "assistant", content: "好的" },
    ],
  });

  assert.match(transcript, /user: 我喜欢苹果/);
  assert.match(transcript, /assistant: 好的/);
  assert.doesNotMatch(transcript, /IDENTITY\.md/);
  assert.doesNotMatch(transcript, /SOUL\.md/);
  assert.doesNotMatch(transcript, /MEMORY\.md/);
  assert.equal(buildCleanCompactionTranscript({ transcript: "只有纯 transcript" }), "");
  assert.match(
    buildCleanCompactionTranscript(
      { transcript: "只有纯 transcript" },
      12000,
      { includeUnstructuredTranscript: true },
    ),
    /只有纯 transcript/,
  );
});

const codexSessionJsonl = [
  JSON.stringify({ type: "session_meta", timestamp: "2026-06-01T00:00:00Z" }),
  JSON.stringify({
    type: "response_item",
    item: { type: "message", role: "user", content: [{ type: "input_text", text: "我喜欢苹果" }] },
  }),
  JSON.stringify({
    type: "response_item",
    item: { type: "function_call", name: "tool", arguments: "不要进入记忆" },
  }),
  JSON.stringify({
    type: "response_item",
    item: { type: "message", role: "assistant", content: [{ type: "output_text", text: "好的" }] },
  }),
].join("\n");

test("clean compaction transcript can read Codex session JSONL", () => {
  const messages = parseSessionJsonlMessages(codexSessionJsonl);
  const transcript = buildCleanCompactionTranscriptFromSessionFile("/tmp/session.jsonl", 12000, {
    readFile() {
      return codexSessionJsonl;
    },
  });

  assert.equal(messages.length, 2);
  assert.match(transcript, /user: 我喜欢苹果/);
  assert.match(transcript, /assistant: 好的/);
  assert.doesNotMatch(transcript, /不要进入记忆/);
});

test("extraction prompt asks for third-person memory wording", () => {
  const messages = buildExtractionMessages(
    { prompt: "提取长期记忆。", output_schema: { title: "string" }, categories: [{ name: "偏好" }] },
    "user: 我喜欢短回答\nassistant: 好的",
    "before_compaction",
  );

  assert.match(messages[0].content, /第三方视角/);
  assert.match(messages[0].content, /用户……/);
  assert.match(messages[0].content, /助手应……/);
  assert.match(messages[0].content, /不要写成“我喜欢”“我应该”/);
});

test("fetchMemoryContext returns context text and items", async () => {
  const calls = [];
  const config = resolveConfig(
    { topK: 8, maxChars: 3000 },
    {
      envValues: {
        AIMEMORY_BASE_URL: "http://aimemory",
        AIMEMORY_API_KEY: "aim_key",
        AIMEMORY_AGENT_ID: "agent",
      },
      processEnv: {},
    },
  );
  const result = await fetchMemoryContext(config, "偏好", {
    category: "回答偏好",
    fetchImpl: async (url, request) => {
      calls.push({ url, request });
      return new Response(
        JSON.stringify({ context_text: "长期记忆", items: [{ title: "t" }] }),
        { status: 200 },
      );
    },
  });

  assert.equal(result.contextText, "长期记忆");
  assert.equal(result.items.length, 1);
  assert.equal(calls[0].url, "http://aimemory/v1/memories/context");
  const body = JSON.parse(calls[0].request.body);
  assert.equal(body.agent_id, "agent");
  assert.equal(body.category, "回答偏好");
});

test("fetchCategories returns category list", async () => {
  const config = resolveConfig(
    {},
    {
      envValues: {
        AIMEMORY_BASE_URL: "http://aimemory",
        AIMEMORY_API_KEY: "aim_key",
        AIMEMORY_AGENT_ID: "agent",
      },
      processEnv: {},
    },
  );
  const items = await fetchCategories(config, {
    fetchImpl: async (url) => {
      assert.equal(url, "http://aimemory/v1/memories/categories");
      return new Response(JSON.stringify({ items: [{ name: "回答偏好" }] }), { status: 200 });
    },
  });

  assert.equal(items[0].name, "回答偏好");
});

test("runtime context failure does not block prompt build", async () => {
  const handlers = {};
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async () => new Response("nope", { status: 500 }),
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  const result = await handlers.before_prompt_build({ userInput: "hello", chatType: "direct" }, {});

  assert.deepEqual(result, {});
});

test("runtime skips prompt-only turns without calling AIMemory", async () => {
  const handlers = {};
  let fetchCalls = 0;
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async () => {
      fetchCalls += 1;
      return new Response("{}", { status: 200 });
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  const result = await handlers.before_prompt_build({
    prompt: "IDENTITY.md 永久人设",
    chatType: "direct",
  }, {});

  assert.deepEqual(result, {});
  assert.equal(fetchCalls, 0);
});

test("runtime context success injects prependContext", async () => {
  const handlers = {};
  const llmMessages = [];
  const contextBodies = [];
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete(payload) {
          llmMessages.push(payload.messages);
          return '{"category":"回答偏好"}';
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async (url, request) => {
      if (String(url).endsWith("/v1/memories/categories")) {
        return new Response(JSON.stringify({ items: [{ name: "回答偏好" }] }), { status: 200 });
      }
      contextBodies.push(JSON.parse(request.body));
      return new Response(JSON.stringify({ context_text: "记忆上下文", items: [{ title: "x" }] }), {
        status: 200,
      });
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  const result = await handlers.before_prompt_build(
    { prompt: "IDENTITY.md 永久人设", userInput: "hello", chatType: "direct" },
    {},
  );

  assert.equal(result.prependContext, "记忆上下文");
  assert.doesNotMatch(JSON.stringify(llmMessages), /IDENTITY\.md/);
  assert.equal(contextBodies[0].query, "hello");
});

test("runtime uses one-shot message_received user input instead of internal roleless input", async () => {
  const handlers = {};
  const contextBodies = [];
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete() {
          return '{"category":"未分类"}';
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async (url, request) => {
      if (String(url).endsWith("/v1/memories/categories")) {
        return new Response(JSON.stringify({ items: [{ name: "未分类" }] }), { status: 200 });
      }
      contextBodies.push(JSON.parse(request.body));
      return new Response(JSON.stringify({ context_text: "", items: [] }), { status: 200 });
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  await handlers.message_received(
    { text: "老婆来点福利美照", chatType: "direct", chatId: "chat-1" },
    { chatId: "chat-1" },
  );
  await handlers.before_prompt_build(
    { input: "dark armor fantasy poster", chatType: "direct", chatId: "chat-1" },
    { chatId: "chat-1" },
  );
  await handlers.before_prompt_build(
    { input: "another internal prompt", chatType: "direct", chatId: "chat-1" },
    { chatId: "chat-1" },
  );

  assert.equal(contextBodies.length, 1);
  assert.equal(contextBodies[0].query, "老婆来点福利美照");
});

test("runtime uses fallback user input when prompt build has a different turn key", async () => {
  const handlers = {};
  const contextBodies = [];
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete() {
          return '{"category":"未分类"}';
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async (url, request) => {
      if (String(url).endsWith("/v1/memories/categories")) {
        return new Response(JSON.stringify({ items: [{ name: "未分类" }] }), { status: 200 });
      }
      contextBodies.push(JSON.parse(request.body));
      return new Response(JSON.stringify({ context_text: "", items: [] }), { status: 200 });
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  await handlers.message_received(
    { text: "老婆出个兔女郎的", chatType: "direct", messageId: "message-1" },
    { messageId: "message-1" },
  );
  await handlers.before_prompt_build(
    { input: "internal generated prompt", chatType: "direct", turnId: "turn-2" },
    { turnId: "turn-2" },
  );
  await handlers.before_prompt_build(
    { input: "another internal generated prompt", chatType: "direct", turnId: "turn-3" },
    { turnId: "turn-3" },
  );

  assert.equal(contextBodies.length, 1);
  assert.equal(contextBodies[0].query, "老婆出个兔女郎的");
});

test("runtime compaction skips unstructured prompt and transcript", async () => {
  const handlers = {};
  let fetchCalls = 0;
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete() {
          throw new Error("llm should not be called");
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    fetchImpl: async () => {
      fetchCalls += 1;
      return new Response("{}", { status: 200 });
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  const result = await handlers.before_compaction(
    { prompt: "IDENTITY.md 永久人设", transcript: "SOUL.md 灵魂设定", chatType: "direct" },
    {},
  );

  assert.deepEqual(result, {});
  assert.equal(fetchCalls, 0);
});

test("runtime compaction extracts from sessionFile when hook event has no messages", async () => {
  const handlers = {};
  const writeBodies = [];
  let extractionPrompt = "";
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete(request) {
          extractionPrompt = request.messages.at(-1).content;
          return JSON.stringify([
            {
              category: "回答偏好",
              title: "水果偏好",
              content: "用户喜欢苹果。",
            },
          ]);
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    readFile() {
      return codexSessionJsonl;
    },
    fetchImpl: async (url, request = {}) => {
      if (url.endsWith("/v1/memories/write-policy")) {
        return new Response(
          JSON.stringify({
            prompt: "提取长期记忆。",
            categories: [{ name: "回答偏好" }],
            output_schema: { category: "string", title: "string", content: "string" },
          }),
          { status: 200 },
        );
      }
      if (url.endsWith("/v1/memories") && request.method === "POST") {
        writeBodies.push(JSON.parse(request.body));
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      throw new Error(`unexpected request ${request.method || "GET"} ${url}`);
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  const result = await handlers.before_compaction(
    {
      sessionFile: "/tmp/session.jsonl",
      chatType: "direct",
      messageCount: -1,
      context: { pluginConfig: { useBackendExtraction: false } },
    },
    {},
  );

  assert.deepEqual(result, {});
  assert.match(extractionPrompt, /user: 我喜欢苹果/);
  assert.equal(writeBodies.length, 1);
  assert.equal(writeBodies[0].agent_id, "agent");
  assert.equal(writeBodies[0].content, "用户喜欢苹果。");
});

test("runtime compaction infers sessionFile from hook context", async () => {
  const handlers = {};
  const readPaths = [];
  const writeBodies = [];
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete() {
          return JSON.stringify([
            {
              category: "回答偏好",
              title: "水果偏好",
              content: "用户喜欢苹果。",
            },
          ]);
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    readFile(filePath) {
      readPaths.push(filePath);
      return codexSessionJsonl;
    },
    fetchImpl: async (url, request = {}) => {
      if (url.endsWith("/v1/memories/write-policy")) {
        return new Response(
          JSON.stringify({
            prompt: "提取长期记忆。",
            categories: [{ name: "回答偏好" }],
            output_schema: { category: "string", title: "string", content: "string" },
          }),
          { status: 200 },
        );
      }
      if (url.endsWith("/v1/memories") && request.method === "POST") {
        writeBodies.push(JSON.parse(request.body));
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      throw new Error(`unexpected request ${request.method || "GET"} ${url}`);
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  await handlers.before_compaction(
    {
      chatType: "direct",
      messageCount: 12,
      context: { pluginConfig: { useBackendExtraction: false } },
    },
    { agentId: "main", sessionId: "session-123", sessionKey: "agent:main:test" },
  );

  assert.match(readPaths[0], /\.openclaw\/agents\/main\/sessions\/session-123\.jsonl$/);
  assert.equal(writeBodies.length, 1);
});

test("runtime after_compaction fallback writes when before hook had no transcript", async () => {
  const handlers = {};
  const writeBodies = [];
  const api = {
    on(name, handler) {
      handlers[name] = handler;
    },
    runtime: {
      logging: {
        getChildLogger() {
          return { info() {}, warn() {}, error() {} };
        },
      },
      llm: {
        async complete() {
          return JSON.stringify([
            {
              category: "回答偏好",
              title: "水果偏好",
              content: "用户喜欢苹果。",
            },
          ]);
        },
      },
    },
  };
  registerAIMemoryRuntime(api, {
    readFile(filePath) {
      if (String(filePath).includes("missing")) {
        const error = new Error("missing");
        error.code = "ENOENT";
        throw error;
      }
      return codexSessionJsonl;
    },
    fetchImpl: async (url, request = {}) => {
      if (url.endsWith("/v1/memories/write-policy")) {
        return new Response(
          JSON.stringify({
            prompt: "提取长期记忆。",
            categories: [{ name: "回答偏好" }],
            output_schema: { category: "string", title: "string", content: "string" },
          }),
          { status: 200 },
        );
      }
      if (url.endsWith("/v1/memories") && request.method === "POST") {
        writeBodies.push(JSON.parse(request.body));
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      throw new Error(`unexpected request ${request.method || "GET"} ${url}`);
    },
    envValues: {
      AIMEMORY_BASE_URL: "http://aimemory",
      AIMEMORY_API_KEY: "aim_key",
      AIMEMORY_AGENT_ID: "agent",
    },
    processEnv: {},
  });

  await handlers.before_compaction(
    {
      sessionFile: "/tmp/missing.jsonl",
      chatType: "direct",
      context: { pluginConfig: { useBackendExtraction: false } },
    },
    { sessionId: "session-123" },
  );
  await handlers.after_compaction(
    {
      sessionFile: "/tmp/session.jsonl",
      chatType: "direct",
      context: { pluginConfig: { useBackendExtraction: false } },
    },
    { sessionId: "session-123" },
  );

  assert.equal(writeBodies.length, 1);
  assert.equal(writeBodies[0].content, "用户喜欢苹果。");
});

test("explicit remember intent is detected", () => {
  assert.equal(hasExplicitRememberIntent("帮我记住，以后回答短一点"), true);
  assert.equal(hasExplicitRememberIntent("今天天气怎么样"), false);
});

test("memory extraction JSON normalizes and filters forbidden content", () => {
  const memories = normalizeExtractedMemories(
    JSON.stringify([
      {
        external_id: "Preference Answer Style",
        category: "回答偏好",
        title: "回答偏好",
        content: "用户喜欢短回答。",
        metadata: { category: "preference" },
      },
      {
        title: "密码",
        content: "password is abc",
      },
    ]),
  );

  assert.equal(memories.length, 1);
  assert.equal(memories[0].external_id, "preference-answer-style");
  assert.equal(memories[0].category, "回答偏好");
  assert.equal(memories[0].metadata.category, "preference");
});

test("category selection parses only known categories", () => {
  const categories = [{ name: "回答偏好" }, { name: "爱吃的水果" }];

  assert.equal(parseSelectedCategory('{"category":"回答偏好"}', categories), "回答偏好");
  assert.equal(parseSelectedCategory('{"category":"不存在"}', categories), "");
  assert.match(buildCategorySelectionMessages(categories, "用户喜欢苹果")[1].content, /爱吃的水果/);
});

test("stable external id is deterministic", () => {
  const memory = { title: "标题", content: "内容", metadata: { category: "project" } };

  assert.equal(stableExternalId(memory), stableExternalId(memory));
  assert.match(stableExternalId(memory), /^auto-project-[a-f0-9]{12}$/);
});

test("forbidden memory text catches secrets", () => {
  assert.equal(containsForbiddenMemoryText("api key is sk-xxx"), true);
  assert.equal(containsForbiddenMemoryText("用户喜欢中文回答"), false);
});

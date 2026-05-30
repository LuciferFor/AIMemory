import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildCategorySelectionMessages,
  buildCleanCompactionTranscript,
  buildCleanMemoryQueryFromTurn,
  buildQueryFromTurn,
  containsForbiddenMemoryText,
  fetchCategories,
  fetchMemoryContext,
  hasExplicitRememberIntent,
  isAllowedTurn,
  normalizeExtractedMemories,
  parseSelectedCategory,
  parseEnvText,
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

test("clean query excludes static prompt and system messages", () => {
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
  assert.match(query, /我喜欢香蕉/);
  assert.match(query, /我记得了/);
  assert.doesNotMatch(query, /IDENTITY\.md/);
  assert.doesNotMatch(query, /SOUL\.md/);
  assert.doesNotMatch(query, /MEMORY\.md/);
  assert.doesNotMatch(query, /不要发送给 AIMemory/);
});

test("clean query skips prompt-only turns unless compatibility flag is enabled", () => {
  const event = { prompt: "IDENTITY.md 永久人设" };

  assert.equal(buildCleanMemoryQueryFromTurn(event), "");
  assert.match(
    buildCleanMemoryQueryFromTurn(event, {}, 1500, { includePrompt: true }),
    /IDENTITY\.md/,
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

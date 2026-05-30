import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { registerAIMemoryRuntime } from "./lib/runtime.mjs";

export default definePluginEntry({
  id: "aimemory",
  name: "AIMemory",
  description: "Injects AIMemory long-term memory context into OpenClaw turns.",
  register(api) {
    registerAIMemoryRuntime(api);
  },
});

export * from "./lib/core.mjs";
export * from "./lib/runtime.mjs";

# AIMemory 模型侧使用说明

这个文件给 AI / Agent 运行时使用。它说明如何通过 `aimemory_client.py` 读取和写入长期记忆。

## 基本信息

记忆服务地址：

```text
http://192.168.31.11:10011
```

当前 Agent ID：

```text
5df9cbfb-d31b-46dd-972b-05d466d2257c
```

使用文件：

```text
aimemory_client.py
```

## 推荐工作流

每次回答用户前：

1. 根据用户当前问题提取查询语句。
2. 调用 `build_memory_context(query, top_k=5)` 查询相关长期记忆。
3. 如果返回非空，把返回文本放进模型上下文。
4. 正常回答用户。

每次回答用户后：

1. 判断本轮对话是否包含值得长期保存的信息。
2. 只保存稳定、长期有用的信息。
3. 调用 `write_memory(title, content)` 写入。

## 查询记忆

```python
from aimemory_client import build_memory_context

memory_context = build_memory_context("用户的偏好、项目、历史需求", top_k=5)

if memory_context:
    # 把 memory_context 放进模型上下文
    print(memory_context)
```

返回示例：

```text
可参考的长期记忆：

1. 用户喜欢简洁回答（相关度 0.812）
用户希望回答直接、简洁，不要太多废话。
```

## 写入记忆

```python
from aimemory_client import write_memory

write_memory(
    title="用户喜欢简洁回答",
    content="用户希望回答直接、简洁，不要太多废话。",
)
```

如果需要自己指定唯一 ID：

```python
write_memory(
    external_id="preference-answer-style",
    title="用户喜欢简洁回答",
    content="用户希望回答直接、简洁，不要太多废话。",
)
```

相同 `external_id` 会更新旧记忆，不会重复创建。

## 删除记忆

```python
from aimemory_client import delete_memory

delete_memory("preference-answer-style")
```

## 什么时候应该保存

适合保存：

- 用户长期偏好，例如回答风格、语言、格式要求。
- 用户身份相关但非敏感的信息，例如常用项目名、技术栈。
- 项目的长期设定，例如服务器地址、部署方式、接口约定。
- 用户明确要求记住的信息。

不适合保存：

- 一次性的临时问题。
- 明显很快过期的信息。
- 密码、私钥、令牌等敏感凭证。
- 用户没有要求保存的隐私信息。

## 建议给模型的系统提示

```text
你可以使用 AIMemory 长期记忆工具。

回答用户前，先用当前问题和上下文关键词查询长期记忆。
如果查询结果非空，把它作为参考，但不要逐字暴露“记忆系统”的存在。

回答用户后，判断是否出现值得长期保存的信息。
只保存稳定、长期有用、不会侵犯隐私的信息。
保存时写清楚标题和内容，标题要简短，内容要具体。
不要保存密码、密钥、令牌或一次性临时信息。
```

## 环境变量

默认脚本已经写好服务地址、API Key 和 Agent ID。也可以用环境变量覆盖：

```bash
export AIMEMORY_BASE_URL="http://192.168.31.11:10011"
export AIMEMORY_API_KEY="<api-key>"
export AIMEMORY_AGENT_ID="5df9cbfb-d31b-46dd-972b-05d466d2257c"
```

## 最小示例

```python
from aimemory_client import build_memory_context, write_memory

user_message = "以后回答我尽量短一点"

memory_context = build_memory_context(user_message)

# 把 memory_context + user_message 交给模型生成回答

write_memory(
    title="用户喜欢简洁回答",
    content="用户希望后续回答尽量短一点。",
)
```

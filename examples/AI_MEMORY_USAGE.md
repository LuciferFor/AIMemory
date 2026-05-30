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
2. 先从已有分类中选择一个事务分类，例如“爱吃的水果”“回答偏好”“项目部署”。
3. 优先调用 `build_server_memory_context(query, category, top_k=8)` 获取服务端统一生成的长期记忆提示词。
4. 如果返回非空，把返回文本放进模型 system/developer 上下文。
5. 正常回答用户。

每次回答用户后：

1. 判断本轮对话是否包含值得长期保存的信息。
2. 如果上下文即将压缩，可以先调用 `get_write_policy()` 获取统一提取规则。
3. 从 `get_write_policy()` 返回的 `categories` 里优先选择已有分类；没有合适分类时再创建新分类。
4. 只保存稳定、长期有用的信息。
5. 调用 `write_memory(title, content, category)` 写入。

## 查询记忆

```python
from aimemory_client import build_server_memory_context

memory_context = build_server_memory_context(
    "用户的偏好、项目、历史需求",
    category="回答偏好",
    top_k=8,
)

if memory_context:
    # 把 memory_context 放进模型上下文
    print(memory_context)
```

返回示例：

```text
以下是与当前请求可能相关的长期记忆。请只在相关时自然参考，不要告诉用户你读取了记忆，不要逐字复述；如果记忆与用户当前消息冲突，以当前消息为准。

[长期记忆]

1. 用户喜欢简洁回答
用户希望回答直接、简洁，不要太多废话。
```

如果需要本地拼接上下文，也可以继续使用 `build_memory_context()`，它会调用 `/v1/memories/search` 后在本地拼接上下文。注意现在查询必须传入 `category`，服务端只会在这个分类内检索。

## 写入记忆

```python
from aimemory_client import write_memory

write_memory(
    title="用户喜欢简洁回答",
    content="用户希望回答直接、简洁，不要太多废话。",
    category="回答偏好",
)
```

## 写入图片记忆

图片通过 base64 上传，但服务端会解码后保存为数据库二进制附件。图片本身不会被服务端识别；如果希望后续能搜到图片，请提交 `description`、`ocr_text` 或 tags。

```python
from aimemory_client import image_attachment_from_file, write_memory

write_memory(
    external_id="image-reference-ui-error",
    title="UI 报错截图",
    content="一张可复用的界面报错参考图。",
    category="界面截图",
    attachments=[
        image_attachment_from_file(
            "/path/to/error.png",
            description="深色界面里显示连接失败错误。",
            ocr_text="Connection failed",
            metadata={"tags": ["截图", "报错"]},
        )
    ],
)
```

查询结果只返回附件元数据和下载地址，不会把 base64 塞进模型上下文。

如果需要自己指定唯一 ID：

```python
write_memory(
    external_id="preference-answer-style",
    title="用户喜欢简洁回答",
    content="用户希望回答直接、简洁，不要太多废话。",
    category="回答偏好",
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

回答用户前，先用当前问题和上下文关键词调用 build_server_memory_context 查询长期记忆。
查询前必须先选择一个事务分类；如果无法判断分类，不要请求记忆，避免跨类误命中。
如果返回非空，把它作为参考，但不要逐字暴露“记忆系统”的存在。

回答用户后，判断是否出现值得长期保存的信息。
只保存稳定、长期有用、不会侵犯隐私的信息。
保存时写清楚标题和内容，标题要简短，内容要具体。
保存时必须填写 category，优先使用 get_write_policy 返回的已有分类。
不要保存密码、密钥、令牌或一次性临时信息。
```

## 上下文压缩前提取记忆

```python
from aimemory_client import get_write_policy

policy = get_write_policy()
print(policy["prompt"])
print(policy["categories"])
```

模型应按 `policy["output_schema"]` 输出 JSON 数组，每条都必须包含 `category`，再由客户端逐条调用 `write_memory()` 保存。

## 环境变量

默认脚本已经写好服务地址和 Agent ID。API Key 必须通过环境变量提供，不要硬编码到脚本里：

```bash
export AIMEMORY_BASE_URL="http://192.168.31.11:10011"
export AIMEMORY_API_KEY="<api-key>"
export AIMEMORY_AGENT_ID="5df9cbfb-d31b-46dd-972b-05d466d2257c"
```

## 最小示例

```python
from aimemory_client import build_server_memory_context, write_memory

user_message = "以后回答我尽量短一点"
category = "回答偏好"

memory_context = build_server_memory_context(user_message, category=category)

# 把 memory_context + user_message 交给模型生成回答

write_memory(
    title="用户喜欢简洁回答",
    content="用户希望后续回答尽量短一点。",
    category=category,
)
```

import json
import re
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session

from aimemory.models.ai_chat import AiChatMessage
from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.services.openai_compatible import OpenAICompatibleError, chat_completion, token_usage_summary

MAX_HISTORY_MESSAGES = 16
MAX_SQL_QUERIES = 3
MAX_SQL_ROWS = 100
MAX_WRITE_ROWS = 20
SQL_TIMEOUT_MS = 3000
MAX_USER_MESSAGE_CHARS = 12000
MAX_ASSISTANT_CHARS = 24000
MAX_THREAD_TITLE_CHARS = 18
SENSITIVE_COLUMN_RE = re.compile(r"(api.*key|key|password|secret|token|encrypted)", re.I)
CONTROL_SQL_RE = re.compile(
    r"\b(drop|alter|create|truncate|grant|revoke|copy|call|do|merge|execute|vacuum|refresh|listen|notify|reset)\b",
    re.I,
)
DANGEROUS_FUNCTION_RE = re.compile(r"\b(pg_sleep|pg_read_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|dblink)\b", re.I)
SENSITIVE_SQL_RE = re.compile(r"(api[_-]?key|password|secret|token|encrypted)", re.I)
WRITABLE_TABLES = {
    "memories",
    "memory_categories",
    "search_stopwords",
    "memory_agents",
    "memory_devices",
}
WRITE_OPERATIONS = {"insert", "update", "delete"}


class AiChatError(ValueError):
    pass


def make_thread_title(content: str) -> str:
    text_value = " ".join(str(content or "").split())
    if not text_value:
        return "新对话"
    return text_value[:MAX_THREAD_TITLE_CHARS]


def clean_thread_title(value: str, *, fallback: str = "新对话") -> str:
    text_value = strip_json_code_fence(str(value or "")).strip()
    text_value = re.sub(r"^[\"'“”‘’`]+|[\"'“”‘’`]+$", "", text_value)
    text_value = re.sub(r"^(标题|对话标题|简短标题)\s*[:：]\s*", "", text_value, flags=re.I)
    text_value = " ".join(text_value.split())
    text_value = text_value.strip(" -_，。！？、；：,.!?;:")
    return text_value[:MAX_THREAD_TITLE_CHARS] if text_value else fallback


def generate_ai_chat_title_result(config: LlmProviderConfig, api_key: str, user_content: str) -> dict[str, Any]:
    fallback = make_thread_title(user_content)
    result = chat_completion(
        config,
        api_key,
        [
            {
                "role": "system",
                "content": (
                    "你只负责给后台 AI 对话生成极短标题。"
                    "根据用户第一句话总结一个中文短标题，尽可能短，最多 8 个汉字或 18 个字符。"
                    "不要解释，不要加引号，不要加标点，不要输出 JSON。"
                ),
            },
            {"role": "user", "content": str(user_content or "")[:800]},
        ],
        response_format=None,
        max_tokens=32,
        temperature=0,
        timeout_ms=3000,
    )
    return {"title": clean_thread_title(result.content, fallback=fallback), "usage": token_usage_summary(result.usage)}


def generate_ai_chat_title(config: LlmProviderConfig, api_key: str, user_content: str) -> str:
    return str(generate_ai_chat_title_result(config, api_key, user_content).get("title") or make_thread_title(user_content))


def ai_chat_sql_permissions(config: LlmProviderConfig | None) -> dict[str, bool]:
    return {
        "select": bool(getattr(config, "ai_chat_allow_select", True)),
        "insert": bool(getattr(config, "ai_chat_allow_insert", False)),
        "update": bool(getattr(config, "ai_chat_allow_update", False)),
        "delete": bool(getattr(config, "ai_chat_allow_delete", False)),
    }


def format_sql_permissions(permissions: dict[str, bool]) -> str:
    labels = {
        "select": "查询 SELECT",
        "insert": "新增 INSERT",
        "update": "修改 UPDATE",
        "delete": "删除 DELETE",
    }
    return "、".join(f"{label}={'开启' if permissions.get(key) else '关闭'}" for key, label in labels.items())


def build_project_context(db: Session, config: LlmProviderConfig | None = None) -> str:
    permissions = ai_chat_sql_permissions(config)
    can_write = any(permissions.get(key) for key in WRITE_OPERATIONS)
    return (
        "你是 AIMemory 管理后台内置 AI 助手，面向已登录管理员。"
        "AIMemory 是一个给 AI/OpenClaw 提供长期记忆的服务：业务 API 写入、分类、检索和返回记忆上下文；"
        "后台提供用户、接口密钥、分类、记忆、停用词、请求日志、AI 整理和本 AI 对话。"
        "服务端不再使用 embedding，检索依靠分类、关键词、停用词、词形质量过滤和 PostgreSQL 文本/模糊检索。"
        "OpenClaw 插件负责在用户请求前调用 /v1/memories/context，并在显式记住或压缩前写入记忆。"
        "你可以解释配置、排查日志、生成 SQL 操作和解释执行结果。"
        f"当前后台 AI 对话 SQL 权限：{format_sql_permissions(permissions)}。"
        + (
            "如果生成写入 SQL，必须谨慎、只改管理员明确要求的数据，并优先使用软删除字段 deleted_at；不要批量改动无关数据。"
            if can_write
            else "当前未开启写入权限，你绝不能声称已经修改数据库、配置、文件或服务。"
        )
        + "不要要求或暴露 API Key、管理员密码、sudo 密码、token。"
        + "SQL 不能读取或写入 key/password/secret/token/encrypted 相关字段。"
        + "写入 SQL 只允许操作 memories、memory_categories、search_stopwords、memory_agents、memory_devices 这些非密钥业务表。"
        + "UPDATE/DELETE 必须带 WHERE 条件，单次影响行数必须很少；不要使用 DROP/ALTER/CREATE/TRUNCATE/COPY/CALL/DO 等控制语句。"
        + "如果需要操作数据库，请在 sql_queries 里给出最多 3 条 PostgreSQL SQL。"
        + "请优先使用中文回答。\n\n"
        f"数据库结构摘要：\n{schema_summary(db)}"
    )


def schema_summary(db: Session) -> str:
    try:
        inspector = inspect(db.get_bind())
        table_names = [name for name in inspector.get_table_names(schema="public") if not name.startswith("alembic")]
    except Exception:
        return "当前无法读取数据库结构摘要。"

    lines: list[str] = []
    for table_name in sorted(table_names):
        columns = []
        for column in inspector.get_columns(table_name, schema="public"):
            name = str(column.get("name", ""))
            if SENSITIVE_COLUMN_RE.search(name):
                continue
            columns.append(f"{name}:{column.get('type')}")
        if columns:
            lines.append(f"- {table_name}({', '.join(columns)})")
    return "\n".join(lines[:80])


def build_plan_messages(project_context: str, history: list[AiChatMessage]) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                project_context
                + "\n\n你现在要先生成 JSON SQL 计划。只输出 JSON 对象，不要输出 markdown。"
                "格式：{\"assistant_message\":\"先给管理员看的简短说明\","
                "\"sql_queries\":[{\"title\":\"操作标题\",\"purpose\":\"用途\",\"sql\":\"SELECT/INSERT/UPDATE/DELETE ...\"}]}。"
                "不需要操作数据库时 sql_queries 返回空数组。"
                "assistant_message 必须是非空中文文本。"
                "即使管理员要求联网、访问网页或查询外部 URL，你也不能输出空白；"
                "请在 assistant_message 中明确说明后台 AI 没有浏览器或外部联网访问工具。"
            ),
        }
    ]
    for message in history[-MAX_HISTORY_MESSAGES:]:
        if message.role not in {"user", "assistant"}:
            continue
        content = str(message.content or "")[:MAX_USER_MESSAGE_CHARS]
        messages.append({"role": message.role, "content": content})
    return messages


def retry_message_for_empty_plan() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "上一轮模型返回了空白内容。请重新回答，并严格只输出 JSON 对象："
            "{\"assistant_message\":\"非空中文说明\",\"sql_queries\":[]}。"
            "如果管理员问能不能访问外部网站，请说明不能直接访问外部网站，不要输出空格或空字符串。"
        ),
    }


def parse_plan(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(strip_json_code_fence(content))
    except json.JSONDecodeError as exc:
        raise AiChatError("AI 返回的查询计划不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise AiChatError("AI 返回的查询计划必须是 JSON 对象。")

    raw_queries = parsed.get("sql_queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = []
    for raw in raw_queries[:MAX_SQL_QUERIES]:
        if not isinstance(raw, dict):
            continue
        queries.append(
            {
                "title": str(raw.get("title") or "只读查询").strip()[:80],
                "purpose": str(raw.get("purpose") or "").strip()[:300],
                "sql": str(raw.get("sql") or "").strip(),
            }
        )
    return {
        "assistant_message": str(parsed.get("assistant_message") or "").strip()[:2000],
        "sql_queries": [item for item in queries if item["sql"]],
    }


def strip_json_code_fence(value: str) -> str:
    text_value = str(value or "").strip()
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text_value, re.I)
    return fenced.group(1).strip() if fenced else text_value


def normalize_sql(sql: str) -> str:
    value = str(sql or "").strip()
    if not value:
        raise AiChatError("SQL 为空。")
    if ";" in value:
        raise AiChatError("只允许单条 SQL，不能包含分号。")
    normalized = re.sub(r"/\*.*?\*/", " ", value, flags=re.S)
    normalized = re.sub(r"--.*?$", " ", normalized, flags=re.M).strip()
    return normalized


def sql_operation(normalized_sql: str) -> str:
    match = re.match(r"^(select|with|insert|update|delete)\b", normalized_sql, re.I)
    if not match:
        raise AiChatError("只允许 SELECT、WITH ... SELECT、INSERT、UPDATE 或 DELETE。")
    operation = match.group(1).lower()
    if operation == "with":
        if re.search(r"\b(insert|update|delete)\b", normalized_sql, re.I):
            raise AiChatError("WITH 只允许用于只读 SELECT，写入 SQL 请直接以 INSERT/UPDATE/DELETE 开头。")
        return "select"
    return operation


def extract_write_table(normalized_sql: str, operation: str) -> str:
    patterns = {
        "insert": r"^insert\s+into\s+([a-zA-Z_][\w.]*)\b",
        "update": r"^update\s+([a-zA-Z_][\w.]*)\b",
        "delete": r"^delete\s+from\s+([a-zA-Z_][\w.]*)\b",
    }
    match = re.match(patterns[operation], normalized_sql, re.I)
    if not match:
        raise AiChatError("无法识别写入 SQL 的目标表。")
    return match.group(1).split(".")[-1].lower()


def validate_sql(sql: str, permissions: dict[str, bool] | None = None) -> dict[str, Any]:
    normalized = normalize_sql(sql)
    operation = sql_operation(normalized)
    effective_permissions = permissions or {"select": True, "insert": False, "update": False, "delete": False}
    if not effective_permissions.get(operation, False):
        raise AiChatError(f"当前 AI 对话没有 {operation.upper()} 权限。")
    if CONTROL_SQL_RE.search(normalized):
        raise AiChatError("SQL 包含禁止的控制语句。")
    if DANGEROUS_FUNCTION_RE.search(normalized):
        raise AiChatError("SQL 包含禁止的危险函数。")
    if SENSITIVE_SQL_RE.search(normalized):
        raise AiChatError("SQL 不能查询密钥、密码、token 或加密字段。")

    table_name = None
    if operation in WRITE_OPERATIONS:
        table_name = extract_write_table(normalized, operation)
        if table_name not in WRITABLE_TABLES:
            raise AiChatError(f"AI 对话不允许写入 {table_name} 表。")
        if operation in {"update", "delete"} and not re.search(r"\bwhere\b", normalized, re.I):
            raise AiChatError("UPDATE/DELETE 必须包含 WHERE 条件。")
        if operation == "insert":
            if not re.search(r"\bvalues\b", normalized, re.I) or re.search(r"\bselect\b", normalized, re.I):
                raise AiChatError("INSERT 只允许明确 VALUES 写入，不允许 INSERT ... SELECT。")
    return {"sql": normalized, "operation": operation, "table": table_name}


def validate_readonly_sql(sql: str) -> str:
    return str(validate_sql(sql, {"select": True, "insert": False, "update": False, "delete": False})["sql"])


def execute_sql(
    db: Session,
    sql: str,
    *,
    permissions: dict[str, bool] | None = None,
    limit: int = MAX_SQL_ROWS,
    max_write_rows: int = MAX_WRITE_ROWS,
) -> dict[str, Any]:
    validated = validate_sql(sql, permissions)
    clean_sql = validated["sql"]
    operation = validated["operation"]
    bind = db.get_bind()
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    with bind.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(f"SET LOCAL statement_timeout = '{SQL_TIMEOUT_MS}ms'"))
            if operation == "select":
                conn.execute(text("SET TRANSACTION READ ONLY"))
                result = conn.execute(text(f"SELECT * FROM ({clean_sql}) AS aimemory_ai_query LIMIT {int(limit)}"))
                columns = list(result.keys())
                for row in result.mappings().fetchmany(limit):
                    rows.append(serialize_row(row))
                row_count = len(rows)
                trans.rollback()
                return {
                    "operation": operation,
                    "table": validated.get("table"),
                    "columns": columns,
                    "rows": rows,
                    "row_count": row_count,
                    "truncated": len(rows) >= limit,
                    "committed": False,
                }

            result = conn.execute(text(clean_sql))
            row_count = int(result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0)
            if result.returns_rows:
                columns = list(result.keys())
                for row in result.mappings().fetchmany(limit):
                    rows.append(serialize_row(row))
                if row_count <= 0:
                    row_count = len(rows)
            if row_count > max_write_rows:
                trans.rollback()
                raise AiChatError(f"写入影响 {row_count} 行，超过单次上限 {max_write_rows} 行，已回滚。")
            trans.commit()
            return {
                "operation": operation,
                "table": validated.get("table"),
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "truncated": len(rows) >= limit,
                "committed": True,
            }
        except Exception:
            trans.rollback()
            raise


def execute_readonly_sql(db: Session, sql: str, *, limit: int = MAX_SQL_ROWS) -> dict[str, Any]:
    return execute_sql(
        db,
        sql,
        permissions={"select": True, "insert": False, "update": False, "delete": False},
        limit=limit,
    )


def serialize_row(row: RowMapping) -> dict[str, Any]:
    return {str(key): serialize_value(value) for key, value in row.items()}


def serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<bytes {len(value)}>"
    if isinstance(value, uuid.UUID | datetime | date):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serialize_value(item) for item in value]
    return str(value)


def execute_plan_sql(
    db: Session,
    queries: list[dict[str, Any]],
    *,
    permissions: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    summaries = []
    for query in queries[:MAX_SQL_QUERIES]:
        summary = {
            "title": query.get("title") or "SQL 操作",
            "purpose": query.get("purpose") or "",
            "sql": query.get("sql") or "",
            "status": "pending",
            "operation": None,
            "table": None,
            "committed": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": None,
        }
        try:
            result = execute_sql(db, summary["sql"], permissions=permissions)
            summary.update(result)
            summary["status"] = "ok"
        except Exception as exc:
            summary["status"] = "error"
            summary["error"] = str(exc)[:1000]
        summaries.append(summary)
    return summaries


def build_final_messages(
    project_context: str,
    history: list[AiChatMessage],
    plan: dict[str, Any],
    sql_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                project_context
                + "\n\n请根据管理员问题和 SQL 执行结果，输出最终中文回答。"
                "如果写入 SQL committed=true，才可以说明已经提交；如果 SQL 被拒绝、报错或回滚，请解释原因和下一步建议。不要输出 JSON。"
            ),
        }
    ]
    for message in history[-MAX_HISTORY_MESSAGES:]:
        if message.role in {"user", "assistant"}:
            messages.append({"role": message.role, "content": str(message.content or "")[:MAX_USER_MESSAGE_CHARS]})
    messages.append(
        {
            "role": "user",
            "content": "查询计划和执行结果：\n"
            + json.dumps(
                {
                    "assistant_message": plan.get("assistant_message"),
                    "sql_results": compact_sql_results(sql_results),
                },
                ensure_ascii=False,
                indent=2,
            ),
        }
    )
    return messages


def compact_sql_results(sql_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for item in sql_results:
        compacted.append(
            {
                "title": item.get("title"),
                "purpose": item.get("purpose"),
                "sql": item.get("sql"),
                "status": item.get("status"),
                "columns": item.get("columns", []),
                "row_count": item.get("row_count", 0),
                "operation": item.get("operation"),
                "table": item.get("table"),
                "committed": item.get("committed", False),
                "truncated": item.get("truncated", False),
                "error": item.get("error"),
                "rows": item.get("rows", [])[:20],
            }
        )
    return compacted


def generate_ai_chat_reply(
    db: Session,
    *,
    config: LlmProviderConfig,
    api_key: str,
    history: list[AiChatMessage],
) -> dict[str, Any]:
    project_context = build_project_context(db, config)
    plan_messages = build_plan_messages(project_context, history)
    try:
        plan_result = chat_completion(
            config,
            api_key,
            plan_messages,
            response_format={"type": "json_object"},
        )
    except OpenAICompatibleError as exc:
        if "内容为空" not in str(exc):
            raise
        plan_result = chat_completion(
            config,
            api_key,
            plan_messages + [retry_message_for_empty_plan()],
            response_format={"type": "json_object"},
        )
    plan = parse_plan(plan_result.content)
    sql_results = execute_plan_sql(db, plan["sql_queries"], permissions=ai_chat_sql_permissions(config))
    plan_usage = token_usage_summary(plan_result.usage)
    usage = dict(plan_usage)
    final_usage: dict[str, int] = {}

    if sql_results:
        final_result = chat_completion(
            config,
            api_key,
            build_final_messages(project_context, history, plan, sql_results),
            response_format=None,
        )
        content = final_result.content
        final_usage = token_usage_summary(final_result.usage)
        usage = merge_usage(usage, final_usage)
    else:
        content = plan.get("assistant_message") or "我理解了，但这次不需要查询数据库。"

    return {
        "content": content[:MAX_ASSISTANT_CHARS],
        "metadata": {
            "plan": plan,
            "sql_results": compact_sql_results(sql_results),
            "ai_usage": usage,
            "ai_usage_breakdown": {
                "plan": plan_usage,
                "final": final_usage,
            },
        },
        "usage": usage,
    }


def merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    keys = {"prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"}
    return {key: int(left.get(key) or 0) + int(right.get(key) or 0) for key in keys}

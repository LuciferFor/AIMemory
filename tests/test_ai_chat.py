import json

import pytest

from aimemory.services.ai_chat import (
    AiChatError,
    build_plan_messages,
    clean_thread_title,
    make_thread_title,
    parse_plan,
    retry_message_for_empty_plan,
    validate_readonly_sql,
    validate_sql,
)


def test_make_thread_title_truncates_clean_text() -> None:
    assert make_thread_title("  帮我 查一下  请求日志  ") == "帮我 查一下 请求日志"
    assert len(make_thread_title("这是一条很长的后台 AI 对话标题，用来测试截断逻辑是否稳定，后面还有更多内容")) == 18


def test_clean_thread_title_removes_wrapper_text() -> None:
    assert clean_thread_title("标题：查询请求日志。") == "查询请求日志"
    assert clean_thread_title("```json\n分类统计\n```") == "分类统计"


def test_parse_plan_accepts_json_code_fence() -> None:
    plan = parse_plan(
        "```json\n"
        + json.dumps(
            {
                "assistant_message": "我会查一下。",
                "sql_queries": [
                    {
                        "title": "请求日志",
                        "purpose": "查看最近请求",
                        "sql": "select * from request_logs",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )

    assert plan["assistant_message"] == "我会查一下。"
    assert plan["sql_queries"][0]["title"] == "请求日志"
    assert plan["sql_queries"][0]["sql"] == "select * from request_logs"


def test_parse_plan_rejects_invalid_json() -> None:
    with pytest.raises(AiChatError, match="合法 JSON"):
        parse_plan("不是 json")


def test_plan_prompt_forbids_empty_output_for_external_url() -> None:
    system_prompt = build_plan_messages("项目上下文", [])[0]["content"]

    assert "assistant_message 必须是非空中文文本" in system_prompt
    assert "外部 URL" in system_prompt
    assert "不能输出空白" in system_prompt


def test_empty_plan_retry_message_is_strict_json_instruction() -> None:
    retry = retry_message_for_empty_plan()

    assert retry["role"] == "user"
    assert "严格只输出 JSON" in retry["content"]
    assert "不要输出空格" in retry["content"]


def test_validate_readonly_sql_allows_select_and_with() -> None:
    assert validate_readonly_sql("select id from memories").startswith("select")
    assert validate_readonly_sql("with x as (select 1 as n) select * from x").startswith("with")


@pytest.mark.parametrize(
    "sql",
    [
        "update memories set title = 'x'",
        "delete from memories",
        "drop table memories",
        "select 1; select 2",
        "copy memories to stdout",
        "select pg_sleep(10)",
        "set role postgres",
        "select encrypted_api_key from llm_provider_configs",
        "select api_key_prefix from request_logs",
    ],
)
def test_validate_readonly_sql_rejects_unsafe_sql(sql: str) -> None:
    with pytest.raises(AiChatError):
        validate_readonly_sql(sql)


def test_validate_sql_allows_enabled_update_on_business_tables() -> None:
    result = validate_sql(
        "update memories set title = '新标题' where id = '00000000-0000-0000-0000-000000000000'",
        {"select": True, "insert": False, "update": True, "delete": False},
    )

    assert result["operation"] == "update"
    assert result["table"] == "memories"


def test_validate_sql_rejects_disabled_write_permission() -> None:
    with pytest.raises(AiChatError, match="没有 UPDATE 权限"):
        validate_sql(
            "update memories set title = '新标题' where id = '00000000-0000-0000-0000-000000000000'",
            {"select": True, "insert": False, "update": False, "delete": False},
        )


@pytest.mark.parametrize(
    "sql",
    [
        "update memories set title = 'x'",
        "delete from memories",
        "insert into memories select * from memories",
        "update api_keys set label = 'x' where id = '00000000-0000-0000-0000-000000000000'",
        "update llm_provider_configs set model = 'x' where name = 'default'",
    ],
)
def test_validate_sql_rejects_risky_write_sql(sql: str) -> None:
    with pytest.raises(AiChatError):
        validate_sql(sql, {"select": True, "insert": True, "update": True, "delete": True})

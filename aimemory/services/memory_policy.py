import json
import re
from typing import Any


TECHNICAL_MEMORY_SKIP_REASON = "technical_or_troubleshooting_memory_not_allowed"

TECHNICAL_OPERATIONAL_CATEGORY_TERMS = [
    "技术",
    "技术记忆",
    "技术资料",
    "故障",
    "故障排查",
    "配置",
    "修复",
    "部署",
    "日志",
    "接口",
    "数据库",
    "自动化",
    "工作流程",
    "运维",
]

TECHNICAL_OPERATIONAL_TERMS = [
    "配置",
    "修复",
    "故障",
    "故障排查",
    "报错",
    "错误日志",
    "部署",
    "日志",
    "接口",
    "api",
    "数据库",
    "脚本",
    "路径",
    "命令",
    "终端",
    "端口",
    "服务状态",
    "连接失败",
    "登录态",
    "重启",
    "启动",
    "docker",
    "compose",
    "redis",
    "postgres",
    "postgresql",
    "nginx",
    "systemd",
    "ssh",
    "curl",
    "python",
    "node",
    "npm",
    "onebot",
    "openclaw",
    "aimemory",
    "websocket",
    "http",
    "https",
    "server",
    "service",
    "database",
    "endpoint",
    "config",
    "configuration",
    "deploy",
    "deployment",
    "troubleshoot",
    "troubleshooting",
    "bug",
    "error",
    "log",
    "script",
    "path",
    "command",
    "port",
]

HUMAN_MEMORY_ALLOW_TERMS = [
    "人设",
    "设定",
    "人格",
    "性格",
    "说话",
    "语气",
    "口吻",
    "称呼",
    "关系",
    "偏好",
    "喜欢",
    "讨厌",
    "生活",
    "习惯",
    "饮食",
    "水果",
    "情绪",
    "安慰",
    "互动",
    "角色关系",
    "回答风格",
    "角色设定",
    "绯夜",
    "月见绫音",
]

_TECHNICAL_PATH_RE = re.compile(r"(^|\s|[：:])([a-z]:)?[/\\][\w./\\-]+", re.IGNORECASE)
_TECHNICAL_COMMAND_RE = re.compile(
    r"(\b(docker|systemctl|journalctl|ssh|scp|curl|npm|pnpm|node|python|pip|uvicorn|alembic|psql|redis-cli)\b|`[^`]+`)",
    re.IGNORECASE,
)


def normalized_policy_text(*values: object) -> str:
    return "\n".join(str(value or "") for value in values).lower()


def _metadata_policy_text(metadata: object | None) -> str:
    if not isinstance(metadata, dict):
        return str(metadata or "")
    allowed_values: list[object] = []
    for key in ("category", "scope", "priority", "tags", "note", "notes"):
        value = metadata.get(key)
        if value:
            allowed_values.append(value)
    return json.dumps(allowed_values, ensure_ascii=False)


def technical_or_operational_memory_skip_reason(
    *,
    title: str,
    content: str,
    category: str,
    metadata: object | None = None,
) -> str:
    """Return a skip reason for ops/config/fix memories that should not be auto-saved."""

    metadata_text = _metadata_policy_text(metadata)
    category_text = normalized_policy_text(category)
    candidate_text = normalized_policy_text(category, title, content, metadata_text)
    visible_text = normalized_policy_text(category, title, content)
    title_content_text = normalized_policy_text(title, content, metadata_text)

    has_human_allow = any(term.lower() in candidate_text for term in HUMAN_MEMORY_ALLOW_TERMS)
    category_is_technical = any(term.lower() in category_text for term in TECHNICAL_OPERATIONAL_CATEGORY_TERMS)
    term_hits = [term for term in TECHNICAL_OPERATIONAL_TERMS if term.lower() in title_content_text]
    has_structural_technical = bool(_TECHNICAL_PATH_RE.search(visible_text) or _TECHNICAL_COMMAND_RE.search(visible_text))

    if category_is_technical and (term_hits or has_structural_technical or not has_human_allow):
        return TECHNICAL_MEMORY_SKIP_REASON
    if has_structural_technical:
        return TECHNICAL_MEMORY_SKIP_REASON
    if len(term_hits) >= 2:
        return TECHNICAL_MEMORY_SKIP_REASON
    if term_hits and not has_human_allow:
        return TECHNICAL_MEMORY_SKIP_REASON
    return ""

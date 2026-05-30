from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


AgentId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
ExternalId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
Filename = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
MimeType = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
AttachmentText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
AttachmentBase64 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MemoryAttachmentInput(BaseModel):
    filename: Filename
    mime_type: MimeType
    data_base64: AttachmentBase64
    description: AttachmentText | None = None
    ocr_text: AttachmentText | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAttachmentMeta(BaseModel):
    attachment_id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    description: str | None = None
    download_url: str


class MemoryUpsertRequest(BaseModel):
    agent_id: AgentId
    external_id: ExternalId
    title: Title
    content: Content
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime | None = None
    attachments: list[MemoryAttachmentInput] | None = None


class MemoryUpsertResponse(BaseModel):
    memory_id: UUID
    external_id: str
    action: str
    embedding_status: str


class MemorySearchRequest(BaseModel):
    agent_id: AgentId
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int = Field(default=10, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = None
    since: datetime | None = None
    until: datetime | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "MemorySearchRequest":
        if self.since and self.until and self.since > self.until:
            raise ValueError("since must be before until")
        return self


class ScoreParts(BaseModel):
    semantic: float = 0.0
    keyword: float = 0.0
    fuzzy: float = 0.0
    term: float = 0.0
    title: float = 0.0
    content: float = 0.0
    metadata: float = 0.0
    exact: float = 0.0
    recency: float = 0.0


class MemorySearchItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    memory_id: UUID
    external_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    score: float
    score_parts: ScoreParts
    embedding_status: str
    attachments: list[MemoryAttachmentMeta] = Field(default_factory=list)


class MemorySearchResponse(BaseModel):
    items: list[MemorySearchItem]


class MemoryContextRequest(BaseModel):
    agent_id: AgentId
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    top_k: int = Field(default=8, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = None
    since: datetime | None = None
    until: datetime | None = None
    max_chars: int = Field(default=3000, ge=1, le=12000)

    @model_validator(mode="after")
    def validate_window(self) -> "MemoryContextRequest":
        if self.since and self.until and self.since > self.until:
            raise ValueError("since must be before until")
        return self


class MemoryContextItem(BaseModel):
    memory_id: UUID
    external_id: str
    title: str
    score: float
    embedding_status: str


class MemoryContextResponse(BaseModel):
    context_text: str
    items: list[MemoryContextItem]
    usage_hint: dict[str, Any]


class MemoryWritePolicyResponse(BaseModel):
    prompt: str
    output_schema: dict[str, Any]
    required_fields: list[str]
    rules: list[str]
    forbidden: list[str]


class MemoryDeleteRequest(BaseModel):
    agent_id: AgentId
    external_id: ExternalId


class MemoryDeleteResponse(BaseModel):
    deleted: bool

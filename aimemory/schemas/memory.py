from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


AgentId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
ExternalId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class MemoryUpsertRequest(BaseModel):
    agent_id: AgentId
    external_id: ExternalId
    title: Title
    content: Content
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime | None = None


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
    semantic: float
    keyword: float
    fuzzy: float


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


class MemorySearchResponse(BaseModel):
    items: list[MemorySearchItem]


class MemoryDeleteRequest(BaseModel):
    agent_id: AgentId
    external_id: ExternalId


class MemoryDeleteResponse(BaseModel):
    deleted: bool

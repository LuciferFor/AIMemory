from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AIMemory"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://aimemory:aimemory@localhost:5432/aimemory"
    redis_url: str = "redis://localhost:6379/0"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    api_key_prefix: str = "aim_"

    admin_username: str = "admin"
    admin_password: str = "change-me"
    admin_session_secret: str = "change-me-session-secret"
    admin_session_max_age_seconds: int = 43200
    admin_cookie_secure: bool = False

    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 1024
    embedding_include_dimensions: bool = True
    embedding_timeout_seconds: float = 20.0

    celery_task_always_eager: bool = False

    max_memory_title_chars: int = 512
    max_memory_content_chars: int = 20000
    max_agent_id_chars: int = 128
    max_external_id_chars: int = 256
    max_search_top_k: int = Field(default=50, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()

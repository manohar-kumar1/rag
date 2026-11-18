from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    db_pool_min_size: int
    db_pool_max_size: int
    db_pool_max_lifetime_seconds: float
    db_pool_timeout_seconds: float
    embedding_provider: Literal["jina", "local"]
    embedding_model: str
    embedding_dimension: int
    jina_api_key: str
    jina_base_url: str
    jina_batch_size: int
    jina_max_retries: int
    jina_retry_backoff_seconds: float
    jina_timeout_seconds: float
    chunk_size: int
    chunk_overlap: int
    docx_max_uncompressed_bytes: int
    max_upload_bytes: int
    max_pdf_pages: int
    max_extracted_chars: int
    supabase_url: str
    supabase_secret_key: str
    storage_bucket: str
    storage_timeout_seconds: float
    worker_enabled: bool
    worker_poll_interval_seconds: float
    worker_concurrency: int
    worker_stale_after_seconds: float
    cors_origins: str
    docs_enabled: bool
    job_retry_after_seconds: int
    idempotency_ttl_seconds: float
    idempotency_lock_seconds: float
    google_api_key: str
    google_model: str
    google_decision_model: str
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> "Settings":
        if not 0 < self.db_pool_min_size <= self.db_pool_max_size:
            raise ValueError("db pool sizes must satisfy 0 < min <= max")
        if self.db_pool_max_lifetime_seconds <= 0 or self.db_pool_timeout_seconds <= 0:
            raise ValueError("db pool lifetime and timeout must be positive")
        if self.jina_batch_size < 1 or self.jina_max_retries < 0:
            raise ValueError(
                "jina batch size must be at least 1 and retries not negative"
            )
        if self.jina_retry_backoff_seconds < 0 or self.jina_timeout_seconds <= 0:
            raise ValueError(
                "jina backoff must not be negative and timeout must be positive"
            )
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk sizes must satisfy 0 <= overlap < size")
        limits = (
            self.docx_max_uncompressed_bytes,
            self.max_upload_bytes,
            self.max_pdf_pages,
            self.max_extracted_chars,
            self.worker_concurrency,
        )
        if any(limit < 1 for limit in limits):
            raise ValueError("file limits and worker concurrency must be at least 1")
        if self.storage_timeout_seconds <= 0:
            raise ValueError("storage timeout must be positive")
        if (
            self.worker_poll_interval_seconds <= 0
            or self.worker_stale_after_seconds <= 0
        ):
            raise ValueError("worker intervals must be positive")
        if self.job_retry_after_seconds < 1:
            raise ValueError("job retry-after must be at least 1 second")
        if self.idempotency_ttl_seconds <= 0 or self.idempotency_lock_seconds <= 0:
            raise ValueError("idempotency ttl and lock durations must be positive")
        if not self.cors_origin_list:
            raise ValueError("at least one CORS origin is required")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def anthropic_fallback_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()

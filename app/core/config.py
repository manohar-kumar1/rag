from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    db_pool_min_size: int
    db_pool_max_size: int
    db_pool_max_lifetime_seconds: float
    db_pool_timeout_seconds: float
    embedding_model: str
    embedding_dimension: int
    jina_api_key: str
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
        return self

    @property
    def anthropic_fallback_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    embedding_model: str
    google_api_key: str
    google_model: str
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None

    @property
    def anthropic_fallback_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()

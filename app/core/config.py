from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All values must come from environment variables or .env (no defaults)."""

    model_config = SettingsConfigDict(env_file=".env")

    anthropic_api_key: str
    database_url: str
    embedding_model: str


settings = Settings()

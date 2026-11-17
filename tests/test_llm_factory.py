from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings
from app.llm.factory import create_chat_model


def make_settings(**overrides) -> Settings:
    values = {
        "database_url": "postgresql://u:p@h/db",
        "embedding_model": "embed",
        "google_api_key": "g-key",
        "google_model": "google-model",
        "anthropic_api_key": "an-key",
        "anthropic_model": "claude-model",
    }
    return Settings(_env_file=None, **{**values, **overrides})


def test_google_is_primary_and_anthropic_is_fallback():
    model = create_chat_model(make_settings())

    assert isinstance(model.runnable, ChatGoogleGenerativeAI)
    assert model.runnable.model == "google-model"
    assert len(model.fallbacks) == 1
    assert isinstance(model.fallbacks[0], ChatAnthropic)
    assert model.fallbacks[0].model == "claude-model"


def test_no_fallback_without_anthropic_credentials():
    model = create_chat_model(
        make_settings(anthropic_api_key=None, anthropic_model=None)
    )

    assert isinstance(model, ChatGoogleGenerativeAI)

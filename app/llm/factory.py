from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings


def create_primary_model(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=settings.google_model,
        google_api_key=settings.google_api_key,
    )


def create_fallback_model(settings: Settings) -> BaseChatModel:
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
    )


def create_chat_model(settings: Settings) -> Runnable:
    primary = create_primary_model(settings)
    if not settings.anthropic_fallback_enabled:
        return primary
    return primary.with_fallbacks([create_fallback_model(settings)])

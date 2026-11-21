from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenAIClassifier

from app.core.config import Settings
from app.llm.resilience import RetryPolicy, is_retryable, with_retries

GOOGLE_SDK_RETRIES_OFF = 1
ANTHROPIC_SDK_RETRIES_OFF = 0


def retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.llm_max_retries,
        initial_delay_seconds=settings.llm_retry_initial_seconds,
        max_delay_seconds=settings.llm_retry_max_seconds,
    )


def create_primary_model(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=settings.google_model,
        google_api_key=settings.google_api_key,
        max_retries=GOOGLE_SDK_RETRIES_OFF,
        timeout=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def create_fallback_model(settings: Settings) -> BaseChatModel | None:
    if not settings.anthropic_fallback_enabled:
        return None
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_retries=ANTHROPIC_SDK_RETRIES_OFF,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_output_tokens,
    )


def resilient_chain(
    primary: BaseChatModel, fallback: BaseChatModel | None, policy: RetryPolicy
) -> Runnable:
    retried = with_retries(primary, policy)
    if fallback is None:
        return retried
    return retried.with_fallbacks([with_retries(fallback, policy)])


def agent_middleware(
    fallback: BaseChatModel | None, policy: RetryPolicy
) -> list[AgentMiddleware]:
    retry = ModelRetryMiddleware(
        max_retries=policy.max_retries,
        retry_on=is_retryable,
        on_failure="error",
        initial_delay=policy.initial_delay_seconds,
        max_delay=policy.max_delay_seconds,
    )
    if fallback is None:
        return [retry]
    return [ModelFallbackMiddleware(fallback), retry]


def create_judge_model(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=settings.eval_judge_model,
        google_api_key=settings.google_api_key,
        max_retries=GOOGLE_SDK_RETRIES_OFF,
        timeout=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def create_decision_model(settings: Settings) -> Runnable:
    return GoogleGenAIClassifier(
        model=settings.google_decision_model,
        google_api_key=settings.google_api_key,
        timeout=settings.decision_timeout_seconds,
        max_retries=GOOGLE_SDK_RETRIES_OFF,
    )


def create_chat_model(settings: Settings) -> Runnable:
    return resilient_chain(
        create_primary_model(settings),
        create_fallback_model(settings),
        retry_policy(settings),
    )

from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.core.config import Settings
from app.llm.resilience import RetryPolicy, is_retryable, with_retries

SDK_RETRIES_OFF = 0


def retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy(
        max_retries=settings.llm_max_retries,
        initial_delay_seconds=settings.llm_retry_initial_seconds,
        max_delay_seconds=settings.llm_retry_max_seconds,
    )


def create_claude(settings: Settings, model: str) -> BaseChatModel:
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        max_retries=SDK_RETRIES_OFF,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_output_tokens,
    )


def agent_middleware(policy: RetryPolicy) -> list[AgentMiddleware]:
    return [
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise"),
        ModelRetryMiddleware(
            max_retries=policy.max_retries,
            retry_on=is_retryable,
            on_failure="error",
            initial_delay=policy.initial_delay_seconds,
            max_delay=policy.max_delay_seconds,
        )
    ]


def create_base_model(settings: Settings) -> BaseChatModel:
    return create_claude(settings, settings.anthropic_model)


def create_chat_model(settings: Settings) -> Runnable:
    return with_retries(create_base_model(settings), retry_policy(settings))


def create_judge_model(settings: Settings) -> BaseChatModel:
    return create_claude(settings, settings.eval_judge_model)


def create_contextual_model(settings: Settings) -> Runnable:
    return with_retries(create_claude(settings, settings.contextual_model), retry_policy(settings))

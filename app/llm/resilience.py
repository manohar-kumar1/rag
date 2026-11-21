from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.exceptions import ModelError
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.retry import RunnableRetry
from tenacity import AsyncRetrying, Retrying, retry_if_exception


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    initial_delay_seconds: float
    max_delay_seconds: float


@dataclass
class StreamProgress:
    started: bool = False


def is_retryable(error: BaseException) -> bool:
    return isinstance(error, ModelError) and error.is_retryable


class StreamingRetry(RunnableRetry):
    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        progress = StreamProgress()
        for attempt in Retrying(**self._stream_retrying(progress)):
            with attempt:
                for chunk in self.bound.stream(input, config, **kwargs):
                    progress.started = True
                    yield chunk

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        progress = StreamProgress()
        async for attempt in AsyncRetrying(**self._stream_retrying(progress)):
            with attempt:
                async for chunk in self.bound.astream(input, config, **kwargs):
                    progress.started = True
                    yield chunk

    def _stream_retrying(self, progress: "StreamProgress") -> dict[str, Any]:
        def should_retry(error: BaseException) -> bool:
            return not progress.started and self._matches(error)

        return {
            **self._kwargs_retrying,
            "retry": retry_if_exception(should_retry),
            "reraise": True,
        }

    def _matches(self, error: BaseException) -> bool:
        if callable(self.retry_exception_types):
            return self.retry_exception_types(error)
        return isinstance(error, self.retry_exception_types)


def with_retries(model: Runnable, policy: RetryPolicy) -> Runnable:
    return StreamingRetry(
        bound=model,
        kwargs={},
        config={},
        retry_exception_types=is_retryable,
        wait_exponential_jitter=True,
        exponential_jitter_params={
            "initial": policy.initial_delay_seconds,
            "max": policy.max_delay_seconds,
            "jitter": policy.initial_delay_seconds,
        },
        max_attempt_number=policy.max_retries + 1,
    )

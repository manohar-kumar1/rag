import re
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_VECTOR_CANDIDATES = 1000


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
    rrf_k: int
    retrieval_vector_weight: float
    retrieval_fts_weight: float
    retrieval_vector_candidates: int
    retrieval_fts_candidates: int
    retrieval_top_k: int
    docs_enabled: bool
    job_retry_after_seconds: int
    idempotency_ttl_seconds: float
    idempotency_lock_seconds: float
    llm_timeout_seconds: float
    llm_max_output_tokens: int
    llm_max_retries: int
    llm_retry_initial_seconds: float
    llm_retry_max_seconds: float
    decision_timeout_seconds: float
    agent_max_subquestions: int
    agent_max_iterations: int
    agent_max_rewrites: int
    agent_grade_threshold: float
    mask_patterns: dict[str, str]
    groundedness_check_enabled: bool
    groundedness_threshold: float
    conversation_max_turns: int
    conversation_retention_seconds: float
    rerank_enabled: bool
    rerank_candidates: int
    rerank_top_n: int
    contextual_chunking_enabled: bool
    answer_cache_enabled: bool
    answer_cache_store: Literal["postgres", "memory"]
    answer_cache_ttl_seconds: float
    answer_cache_semantic_threshold: float
    langsmith_tracing: bool
    google_api_key: str
    google_model: str
    google_decision_model: str
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    langsmith_endpoint: str | None = None
    eval_judge_model: str | None = None
    eval_max_workers: int | None = None
    rerank_model: str | None = None
    contextual_model: str | None = None

    @model_validator(mode="after")
    def validate_answering_features(self) -> "Settings":
        if min(self.agent_max_subquestions, self.agent_max_iterations) < 1:
            raise ValueError("agent sub-question and iteration caps must be at least 1")
        if self.agent_max_rewrites < 0:
            raise ValueError("agent rewrites must not be negative")
        probabilities = (
            self.agent_grade_threshold,
            self.groundedness_threshold,
            self.answer_cache_semantic_threshold,
        )
        if not all(0 < value < 1 for value in probabilities):
            raise ValueError("grade, groundedness and cache thresholds must be between 0 and 1")
        if self.conversation_max_turns < 1 or self.conversation_retention_seconds <= 0:
            raise ValueError("conversation turns and retention must be positive")
        if not 1 <= self.rerank_top_n <= self.rerank_candidates <= MAX_VECTOR_CANDIDATES:
            raise ValueError("rerank counts must satisfy 1 <= top_n <= candidates")
        if self.rerank_enabled and not self.rerank_model:
            raise ValueError("RERANK_MODEL is required when reranking is enabled")
        if self.contextual_chunking_enabled and not self.contextual_model:
            raise ValueError("CONTEXTUAL_MODEL is required when contextual chunking is enabled")
        if self.answer_cache_ttl_seconds <= 0:
            raise ValueError("answer cache ttl must be positive")
        for name, pattern in self.mask_patterns.items():
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"mask pattern {name} is not a valid regex") from error
        return self

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
        if self.rrf_k < 1:
            raise ValueError("rrf k must be at least 1")
        weights = (self.retrieval_vector_weight, self.retrieval_fts_weight)
        if min(weights) < 0 or max(weights) <= 0:
            raise ValueError("retrieval weights must not be negative and not both zero")
        counts = (
            self.retrieval_vector_candidates,
            self.retrieval_fts_candidates,
            self.retrieval_top_k,
        )
        if min(counts) < 1 or self.retrieval_vector_candidates > MAX_VECTOR_CANDIDATES:
            raise ValueError(
                "retrieval counts must be at least 1 and vector candidates at most "
                f"{MAX_VECTOR_CANDIDATES}"
            )
        if self.eval_max_workers is not None and self.eval_max_workers < 1:
            raise ValueError("eval max workers must be at least 1")
        tracing_values = (
            self.langsmith_api_key,
            self.langsmith_project,
            self.langsmith_endpoint,
        )
        if self.langsmith_tracing and not all(tracing_values):
            raise ValueError(
                "tracing needs LANGSMITH_API_KEY, LANGSMITH_PROJECT and LANGSMITH_ENDPOINT"
            )
        if self.decision_timeout_seconds <= 0:
            raise ValueError("decision timeout must be positive")
        if self.llm_timeout_seconds <= 0 or self.llm_max_output_tokens < 1:
            raise ValueError("llm timeout must be positive and max output tokens at least 1")
        if self.llm_max_retries < 0 or not (
            0 < self.llm_retry_initial_seconds <= self.llm_retry_max_seconds
        ):
            raise ValueError(
                "llm retries must not be negative and 0 < initial delay <= max delay"
            )
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
    def evaluation_configured(self) -> bool:
        return bool(self.eval_judge_model and self.eval_max_workers)

    @property
    def anthropic_fallback_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()

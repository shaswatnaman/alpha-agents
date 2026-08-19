from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "AlphaAgents"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── API Security ─────────────────────────────────────────────────────────
    api_key_header: str = "X-API-Key"
    # Comma-separated list of valid API keys (for demo; use a proper auth service in prod)
    valid_api_keys: str = "dev-key-1,dev-key-2"

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.valid_api_keys.split(",") if k.strip()}

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://alphaagents:alphaagents@localhost:5432/alphaagents"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = 20

    # ── Caching ───────────────────────────────────────────────────────────────
    research_cache_ttl_seconds: int = 3600          # 1 hour
    retrieval_cache_ttl_seconds: int = 1800         # 30 minutes
    market_data_cache_ttl_seconds: int = 300        # 5 minutes
    idempotency_key_ttl_seconds: int = 86400        # 24 hours

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_requests_per_minute: int = 20
    rate_limit_research_per_hour: int = 10

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20          # candidates before reranking
    retrieval_final_k: int = 6         # chunks passed to agents
    mmr_lambda: float = 0.6            # balance relevance vs diversity

    # ── Ingestion ─────────────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    allowed_document_types: list[str] = ["pdf", "txt", "docx"]

    # ── Observability ─────────────────────────────────────────────────────────
    prometheus_enabled: bool = True
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"

    # ── Market Data ───────────────────────────────────────────────────────────
    yfinance_timeout_seconds: int = 15
    news_max_articles: int = 20

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        # Allow empty key for test environments
        return v or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

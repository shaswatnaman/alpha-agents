"""
Provider-agnostic LLM interface.

Every agent uses LLMProvider.complete() — never calls OpenAI directly.
Swapping to Anthropic/Gemini/local requires only a new implementation class.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import openai
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config.settings import get_settings
from app.observability.metrics import llm_cost_counter, llm_latency_histogram, llm_token_counter

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Cost per 1k tokens (USD) — update as pricing changes
_OPENAI_COST_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.00250, "output": 0.01000},
    "gpt-4-turbo": {"input": 0.01000, "output": 0.03000},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _OPENAI_COST_TABLE.get(model, {"input": 0.001, "output": 0.002})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: int
    estimated_cost_usd: float


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[T] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        If response_format is provided, the model uses structured output mode
        and the returned content is a validated JSON string of that type.
        """


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation with:
    - Exponential-backoff retries on transient errors
    - Timeout enforcement
    - Structured output (JSON mode via response_format)
    - Token & cost tracking
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we manage retries ourselves for observability
        )
        self._default_model = settings.openai_model
        self._default_temperature = settings.llm_temperature
        self._default_max_tokens = settings.llm_max_tokens
        self._max_retries = settings.llm_max_retries

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[T] | None = None,
    ) -> LLMResponse:
        resolved_model = model or self._default_model
        resolved_temp = temperature if temperature is not None else self._default_temperature
        resolved_tokens = max_tokens or self._default_max_tokens

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                backoff = 2**attempt
                log.warning(
                    "llm_retry",
                    attempt=attempt,
                    backoff_seconds=backoff,
                    model=resolved_model,
                )
                await asyncio.sleep(backoff)

            try:
                t0 = time.monotonic()
                resp = await self._call_api(
                    messages, resolved_model, resolved_temp, resolved_tokens, response_format
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                usage = resp.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                cost = _estimate_cost(resolved_model, in_tok, out_tok)

                # Emit Prometheus metrics
                llm_token_counter.labels(model=resolved_model, direction="input").inc(in_tok)
                llm_token_counter.labels(model=resolved_model, direction="output").inc(out_tok)
                llm_cost_counter.labels(model=resolved_model).inc(cost)
                llm_latency_histogram.labels(model=resolved_model).observe(latency_ms / 1000)

                content = resp.choices[0].message.content or ""

                log.info(
                    "llm_complete",
                    model=resolved_model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )

                return LLMResponse(
                    content=content,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    model=resolved_model,
                    latency_ms=latency_ms,
                    estimated_cost_usd=cost,
                )

            except (openai.RateLimitError, openai.APITimeoutError) as exc:
                last_exc = exc
                log.warning("llm_transient_error", error=str(exc), attempt=attempt)
                if attempt == self._max_retries:
                    raise
            except openai.AuthenticationError:
                # Non-retriable — fail fast
                log.error("llm_auth_error")
                raise
            except Exception as exc:
                last_exc = exc
                log.error("llm_unexpected_error", error=str(exc), attempt=attempt)
                if attempt == self._max_retries:
                    raise

        raise RuntimeError(
            f"LLM request failed after {self._max_retries + 1} attempts"
        ) from last_exc

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: type[T] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            # Use beta structured outputs (JSON schema enforcement)
            return await self._client.beta.chat.completions.parse(
                **kwargs,
                response_format=response_format,
            )
        return await self._client.chat.completions.create(**kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        settings = get_settings()
        response = await self._client.embeddings.create(
            input=texts,
            model=settings.openai_embedding_model,
        )
        return [item.embedding for item in response.data]


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = OpenAIProvider()
    return _provider

"""
Base class shared by all specialist agents.

Handles:
- Prompt loading from the prompts/ directory
- Structured output via LLMProvider
- Execution timing
- Failure isolation: agents return a failed AgentReport rather than raising
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TypeVar

import structlog
from pydantic import BaseModel

from app.domain.models import AgentReport, AgentRole
from app.llm.provider import LLMProvider, LLMResponse, get_llm_provider
from app.observability.metrics import agent_execution_histogram, agent_failure_counter

log = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

T = TypeVar("T", bound=BaseModel)


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class BaseAgent:
    role: AgentRole
    prompt_file: str

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider or get_llm_provider()
        self._system_prompt = load_prompt(self.prompt_file)

    async def _call(
        self,
        user_message: str,
        response_model: type[T],
    ) -> tuple[T, LLMResponse]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await self._provider.complete(
            messages,
            response_format=response_model,
        )
        parsed = response_model.model_validate_json(response.content)
        return parsed, response

    def _failed_report(
        self,
        research_id: str,
        reason: str,
        elapsed_ms: int,
    ) -> AgentReport:
        agent_failure_counter.labels(
            agent=self.role.value,
            reason=reason[:50],
        ).inc()
        return AgentReport(
            agent=self.role,
            research_id=research_id,
            findings=[],
            confidence=0.0,
            summary=f"Agent failed: {reason}",
            failed=True,
            failure_reason=reason,
            execution_time_ms=elapsed_ms,
        )

"""
Integration tests for FastAPI endpoints.
Uses httpx.AsyncClient against the in-process app.
All external dependencies (LLM, DB, Redis) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def api_headers() -> dict[str, str]:
    return {"X-API-Key": "dev-key-1"}


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestResearchEndpoints:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/research", json={"ticker": "AAPL"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/research",
            json={"ticker": "AAPL"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_create_research_returns_202(
        self, client: AsyncClient, api_headers: dict
    ) -> None:
        with (
            patch("app.api.v1.research.get_db"),
            patch("app.api.v1.research.api_limiter.check", new_callable=AsyncMock),
            patch("app.api.v1.research.research_limiter.check", new_callable=AsyncMock),
            patch(
                "app.services.orchestrator.ResearchOrchestrator.start_research",
                new_callable=AsyncMock,
            ) as mock_start,
        ):
            mock_start.return_value = "test-research-id-123"

            resp = await client.post(
                "/api/v1/research",
                json={"ticker": "AAPL"},
                headers=api_headers,
            )

        assert resp.status_code == 202
        data = resp.json()
        assert data["research_id"] == "test-research-id-123"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, client: AsyncClient, api_headers: dict) -> None:
        with (
            patch("app.api.v1.research.api_limiter.check", new_callable=AsyncMock),
            patch("app.api.v1.research.get_db"),
            patch(
                "app.repositories.research_repository.ResearchRepository.get_request",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            resp = await client.get(
                "/api/v1/research/nonexistent-id",
                headers=api_headers,
            )
        assert resp.status_code == 404

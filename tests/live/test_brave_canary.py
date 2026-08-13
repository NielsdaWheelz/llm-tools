"""Opt-in Brave release canary; absence is ``not_run``, never a passing test."""

from __future__ import annotations

import os

import httpx
import pytest

from llm_tools.web.brave import BraveSearchProvider
from llm_tools.web.contracts import WebSearchRequest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_brave_live_contract_and_cost_ceiling() -> None:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        pytest.fail("not_run: BRAVE_SEARCH_API_KEY is required for the opt-in release canary")

    async with httpx.AsyncClient(trust_env=False) as client:
        response = await BraveSearchProvider(client, api_key=api_key).search(
            WebSearchRequest(query="RFC 8785 JSON canonicalization", limit=1)
        )

    assert response.provider == "brave"
    assert response.attempts <= 2
    assert len(response.results) <= 1
    assert response.provider_request_id is None or (
        isinstance(response.provider_request_id, str) and len(response.provider_request_id) <= 512
    )
    assert all(result.provider == "brave" for result in response.results)
    assert all(
        result.provider_request_id == response.provider_request_id for result in response.results
    )

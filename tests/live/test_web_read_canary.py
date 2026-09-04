"""Opt-in protected Web-reader release canary for an owned public fixture."""

from __future__ import annotations

import json
import os

import pytest

from llm_tools.web.reader import SafeWebReader
from llm_tools.web.tools import bind_web_read

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_owned_https_redirect_fixture_peer_bounds_and_receipt() -> None:
    fixture_url = os.environ.get("LLM_TOOLS_WEB_READ_FIXTURE_URL")
    if not fixture_url:
        pytest.fail(
            "not_run: LLM_TOOLS_WEB_READ_FIXTURE_URL is required when a named production profile "
            "enables web.read"
        )

    reader = SafeWebReader()
    binding = bind_web_read(reader)
    response = await reader.read(fixture_url)

    assert binding.implementation_revision == "llm-tools-web-read-v2"
    assert fixture_url.startswith("https://")
    assert response.value.final_url.startswith("https://")
    assert len(response.hops) >= 2
    assert response.attempts == len(response.hops)
    assert response.attempts <= 8
    assert response.value.evidence.source_uri == fixture_url
    assert response.value.evidence.final_uri == response.value.final_url
    assert len(response.value.evidence.content_sha256) == 64
    locator = json.loads(response.value.evidence.locator)
    assert locator["extraction"] == "html-visible-text-v2"
    assert locator["representation"] == "decoded-entity-bytes"

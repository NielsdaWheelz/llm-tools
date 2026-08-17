"""Brave Search provider tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx
import pytest
import respx

from llm_tools.web.brave import BraveSearchProvider
from llm_tools.web.contracts import (
    WebSearchError,
    WebSearchErrorCode,
    WebSearchRequest,
    WebSearchResultType,
)

BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"


@pytest.fixture
async def httpx_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.mark.asyncio
@respx.mock
async def test_web_search_sends_brave_params_and_returns_normalized_results(
    httpx_client: httpx.AsyncClient,
) -> None:
    route = respx.get(BRAVE_WEB_URL).respond(
        200,
        headers={"x-request-id": "req-123"},
        json={
            "web": {
                "results": [
                    {
                        "title": "Example Result",
                        "url": "HTTPS://Example.COM:443/docs?q=Hello World#section",
                        "description": "Primary snippet",
                        "extra_snippets": ["More context", "", 7],
                        "age": "2026-04-21T10:00:00Z",
                        "profile": {"name": "Example Docs"},
                        "provider_specific": {"must": "not leak"},
                    },
                    {
                        "title": "Duplicate Result",
                        "url": "https://example.com/docs?q=Hello%20World",
                        "description": "Skipped duplicate",
                    },
                    {
                        "title": "Unsafe Result",
                        "url": "javascript:alert(1)",
                        "description": "Skipped",
                    },
                ]
            },
        },
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(
        WebSearchRequest(
            query="  brave   search  ",
            result_type=WebSearchResultType.WEB,
            limit=3,
            freshness_days=7,
            allowed_domains=("https://Example.com/path",),
            blocked_domains=("Spam.Example",),
        )
    )

    assert route.called
    request = route.calls.last.request
    assert request.headers["X-Subscription-Token"] == "test-key"
    assert request.headers["Accept"] == "application/json"
    assert request.url.params["q"] == "brave search site:example.com -site:spam.example"
    assert request.url.params["freshness"] == "pw"
    assert request.url.params["result_filter"] == "web"
    assert request.url.params["count"] == "3"
    assert request.url.params["extra_snippets"] == "true"

    assert response.provider == "brave"
    assert response.provider_request_id == "req-123"
    assert datetime.fromisoformat(response.retrieved_at.replace("Z", "+00:00"))
    assert len(response.results) == 1
    result = response.results[0]
    assert result.provider == "brave"
    assert result.provider_request_id == "req-123"
    assert result.rank == 1
    assert result.result_ref.startswith("brave:web:")
    assert result.url == "https://example.com/docs?q=Hello%20World"
    assert result.display_url == "example.com/docs"
    assert result.title == "Example Result"
    assert result.snippet == "Primary snippet"
    assert result.extra_snippets == ("More context",)
    assert result.published_at == "2026-04-21"
    assert result.source_name == "Example Docs"
    assert not hasattr(result, "provider_specific")


@pytest.mark.asyncio
@respx.mock
async def test_news_search_uses_news_endpoint(httpx_client: httpx.AsyncClient) -> None:
    route = respx.get(BRAVE_NEWS_URL).respond(
        200,
        json={
            "results": [
                {
                    "title": "News Result",
                    "url": "https://news.example/story",
                    "description": "News snippet",
                    "source": "Example News",
                    "age": "2 hours ago",
                }
            ],
        },
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(
        WebSearchRequest(
            query="market update",
            result_type=WebSearchResultType.NEWS,
            limit=2,
            freshness_days=1,
        )
    )

    request = route.calls.last.request
    assert request.url.params["freshness"] == "pd"
    assert "result_filter" not in request.url.params
    assert len(response.results) == 1
    assert response.results[0].result_ref.startswith("brave:news:")
    assert response.results[0].source_name == "Example News"
    assert response.results[0].published_at == "2 hours ago"


@pytest.mark.asyncio
@respx.mock
async def test_mixed_search_respects_brave_mixed_order(httpx_client: httpx.AsyncClient) -> None:
    respx.get(BRAVE_WEB_URL).respond(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": "Web One",
                        "url": "https://example.com/one",
                        "description": "Web one",
                    },
                    {
                        "title": "Web Two",
                        "url": "https://example.com/two",
                        "description": "Web two",
                    },
                ]
            },
            "news": {
                "results": [
                    {
                        "title": "News One",
                        "url": "https://news.example/one",
                        "description": "News one",
                    }
                ]
            },
            "mixed": {
                "main": [
                    {"type": "news", "index": 0},
                    {"type": "web", "index": 1},
                    {"type": "web", "index": 0},
                ]
            },
        },
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(
        WebSearchRequest(query="mixed order", result_type=WebSearchResultType.MIXED, limit=3)
    )

    assert [result.title for result in response.results] == ["News One", "Web Two", "Web One"]
    assert [result.rank for result in response.results] == [1, 2, 3]


@pytest.mark.asyncio
@respx.mock
async def test_host_limit_twenty_is_forwarded_and_caps_normalized_results(
    httpx_client: httpx.AsyncClient,
) -> None:
    route = respx.get(BRAVE_WEB_URL).respond(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": f"Result {index}",
                        "url": f"https://example.com/{index}",
                        "description": f"Snippet {index}",
                    }
                    for index in range(1, 22)
                ]
            }
        },
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(
        WebSearchRequest(
            query="host owned breadth",
            result_type=WebSearchResultType.WEB,
            limit=20,
        )
    )

    assert route.calls.last.request.url.params["count"] == "20"
    assert len(response.results) == 20
    assert [result.rank for result in response.results] == list(range(1, 21))
    assert response.results[-1].title == "Result 20"


@pytest.mark.asyncio
@respx.mock
async def test_retries_retryable_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: httpx.AsyncClient,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("llm_tools.web.brave.asyncio.sleep", fake_sleep)
    route = respx.get(BRAVE_WEB_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(200, json={"web": {"results": []}}),
        ]
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(WebSearchRequest(query="retry me"))

    assert response.results == ()
    assert route.call_count == 2
    assert delays == [0.01]


@pytest.mark.asyncio
@respx.mock
async def test_custom_base_url_is_normalized_and_api_key_is_trimmed(
    httpx_client: httpx.AsyncClient,
) -> None:
    route = respx.get("https://proxy.example/brave/web/search").respond(
        200,
        json={"web": {"results": []}},
    )
    provider = BraveSearchProvider(
        httpx_client,
        api_key=" test-key ",
        base_url="HTTPS://Proxy.Example:443/brave/",
    )

    response = await provider.search(WebSearchRequest(query="base url"))

    assert response.results == ()
    assert route.called
    assert route.calls.last.request.headers["X-Subscription-Token"] == "test-key"


@pytest.mark.asyncio
async def test_provider_validates_base_url_and_timeout(httpx_client: httpx.AsyncClient) -> None:
    with pytest.raises(ValueError, match="HTTP"):
        BraveSearchProvider(httpx_client, api_key="test-key", base_url="api.search.brave.com")
    with pytest.raises(ValueError, match="HTTPS"):
        BraveSearchProvider(httpx_client, api_key="test-key", base_url="http://example.com")
    with pytest.raises(ValueError, match="credentials"):
        BraveSearchProvider(httpx_client, api_key="test-key", base_url="https://u:p@example.com")
    with pytest.raises(ValueError, match="query or fragment"):
        BraveSearchProvider(httpx_client, api_key="test-key", base_url="https://example.com?q=1")
    with pytest.raises(ValueError, match="positive"):
        BraveSearchProvider(httpx_client, api_key="test-key", timeout_seconds=0)
    for invalid in (True, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError), match="timeout_seconds"):
            BraveSearchProvider(httpx_client, api_key="test-key", timeout_seconds=invalid)


@pytest.mark.asyncio
async def test_provider_never_follows_a_client_configured_cross_origin_redirect() -> None:
    seen: list[httpx.Request] = []

    def transcript(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.test/steal"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript),
        follow_redirects=True,
    ) as client:
        with pytest.raises(WebSearchError):
            await BraveSearchProvider(client, api_key="SECRET").search(
                WebSearchRequest(query="redirect credentials", max_attempts=1)
            )

    assert len(seen) == 1
    assert seen[0].url.host == "api.search.brave.com"
    assert seen[0].headers["X-Subscription-Token"] == "SECRET"


@pytest.mark.asyncio
@respx.mock
async def test_timeout_after_bounded_retries_maps_to_web_search_error(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: httpx.AsyncClient,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("llm_tools.web.brave.asyncio.sleep", fake_sleep)
    route = respx.get(BRAVE_WEB_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    with pytest.raises(WebSearchError) as exc_info:
        await provider.search(WebSearchRequest(query="timeout please"))

    assert exc_info.value.code == WebSearchErrorCode.TIMEOUT
    assert route.call_count == 2
    assert len(delays) == 1


@pytest.mark.asyncio
@respx.mock
async def test_unauthorized_maps_to_invalid_key_without_retry(
    httpx_client: httpx.AsyncClient,
) -> None:
    route = respx.get(BRAVE_WEB_URL).respond(403, json={"error": {"message": "forbidden"}})
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    with pytest.raises(WebSearchError) as exc_info:
        await provider.search(WebSearchRequest(query="auth failure"))

    assert exc_info.value.code == WebSearchErrorCode.INVALID_KEY
    assert exc_info.value.status_code == 403
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_provider_down_retries_then_fails(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: httpx.AsyncClient,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("llm_tools.web.brave.asyncio.sleep", fake_sleep)
    route = respx.get(BRAVE_WEB_URL).respond(503, json={"error": "unavailable"})
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    with pytest.raises(WebSearchError) as exc_info:
        await provider.search(WebSearchRequest(query="provider down"))

    assert exc_info.value.code == WebSearchErrorCode.PROVIDER_DOWN
    assert exc_info.value.status_code == 503
    assert route.call_count == 2
    assert delays == [0.25]


@pytest.mark.asyncio
@respx.mock
async def test_retry_after_delay_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    httpx_client: httpx.AsyncClient,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("llm_tools.web.brave.asyncio.sleep", fake_sleep)
    route = respx.get(BRAVE_WEB_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "inf"}),
            httpx.Response(200, json={"web": {"results": []}}),
        ]
    )
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    response = await provider.search(WebSearchRequest(query="retry bounded"))

    assert response.results == ()
    assert route.call_count == 2
    assert delays == [0.25]


@pytest.mark.asyncio
@respx.mock
async def test_malformed_json_maps_to_bad_response(httpx_client: httpx.AsyncClient) -> None:
    respx.get(BRAVE_WEB_URL).respond(200, content=b"not json")
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    with pytest.raises(WebSearchError) as exc_info:
        await provider.search(WebSearchRequest(query="bad json"))

    assert exc_info.value.code == WebSearchErrorCode.BAD_RESPONSE


@pytest.mark.asyncio
@respx.mock
async def test_non_object_payload_maps_to_bad_response(httpx_client: httpx.AsyncClient) -> None:
    respx.get(BRAVE_WEB_URL).respond(200, json=[])
    provider = BraveSearchProvider(httpx_client, api_key="test-key")

    with pytest.raises(WebSearchError) as exc_info:
        await provider.search(WebSearchRequest(query="bad payload"))

    assert exc_info.value.code == WebSearchErrorCode.BAD_RESPONSE

    oversized = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"Content-Length": str(2 * 1_024 * 1_024 + 1)},
            content=b"{}",
        )
    )
    async with httpx.AsyncClient(transport=oversized) as client:
        with pytest.raises(WebSearchError) as oversized_error:
            await BraveSearchProvider(client, api_key="test-key").search(
                WebSearchRequest(query="oversized response", max_attempts=1)
            )
    assert oversized_error.value.code == WebSearchErrorCode.BAD_RESPONSE
    assert oversized_error.value.attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "headers"),
    [
        ({"web": []}, {}),
        (
            {
                "web": {
                    "results": [
                        {
                            "title": "Malformed profile",
                            "url": "https://example.com/",
                            "profile": "not-an-object",
                        }
                    ]
                }
            },
            {},
        ),
        ({"web": {"results": []}, "request_id": {"bad": "type"}}, {}),
        ({"web": {"results": []}}, {"x-request-id": "x" * 513}),
    ],
)
async def test_nested_malformed_payload_maps_to_bad_response(
    httpx_client: httpx.AsyncClient,
    payload: object,
    headers: dict[str, str],
) -> None:
    del httpx_client
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=payload, headers=headers)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = BraveSearchProvider(client, api_key="test-key")
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search(WebSearchRequest(query="bad nested payload"))

    assert exc_info.value.code == WebSearchErrorCode.BAD_RESPONSE
    assert exc_info.value.attempts == 1


def test_request_validates_query_limit_domains_and_safe_search() -> None:
    invalid_safe_search: Any = "loose"

    with pytest.raises(ValueError, match="too short"):
        WebSearchRequest(query=" ")
    with pytest.raises(ValueError, match="between 1 and 20"):
        WebSearchRequest(query="valid", limit=21)
    with pytest.raises(ValueError, match="registrable"):
        WebSearchRequest(query="valid", allowed_domains=("localhost",))
    with pytest.raises(ValueError, match="safe_search"):
        WebSearchRequest(query="valid", safe_search=invalid_safe_search)

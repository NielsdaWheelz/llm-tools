"""Brave Search API provider."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx

from web_search_tool.types import (
    WebSearchError,
    WebSearchErrorCode,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResultItem,
    WebSearchResultType,
)

_PROVIDER = "brave"
_DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1"
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 0.25


class BraveSearchProvider:
    """Async Brave Search provider with normalized results."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        if not self._api_key:
            raise WebSearchError(
                WebSearchErrorCode.INVALID_KEY,
                "Brave Search API key is not configured",
                provider=_PROVIDER,
            )

        last_error: WebSearchError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._client.get(
                    self._endpoint_for(request),
                    params=self._params_for(request),
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                    timeout=httpx.Timeout(self._timeout_seconds, connect=5.0),
                )
                response.raise_for_status()
                data = response.json()
                return self._response_from_json(
                    data,
                    request,
                    response.headers.get("x-request-id")
                    or response.headers.get("request-id")
                    or data.get("request_id"),
                )
            except httpx.TimeoutException as exc:
                last_error = WebSearchError(
                    WebSearchErrorCode.TIMEOUT,
                    "Brave Search request timed out",
                    provider=_PROVIDER,
                )
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise last_error from exc
            except httpx.HTTPStatusError as exc:
                error = _error_from_response(exc.response)
                if error.code not in (
                    WebSearchErrorCode.RATE_LIMITED,
                    WebSearchErrorCode.PROVIDER_DOWN,
                ):
                    raise error from exc
                last_error = error
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise last_error from exc
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_error = WebSearchError(
                    WebSearchErrorCode.PROVIDER_DOWN,
                    "Brave Search network error",
                    provider=_PROVIDER,
                )
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise last_error from exc
            except ValueError as exc:
                raise WebSearchError(
                    WebSearchErrorCode.BAD_RESPONSE,
                    "Brave Search returned malformed JSON",
                    provider=_PROVIDER,
                ) from exc

            retry_after = last_error.retry_after if last_error is not None else None
            await asyncio.sleep(
                retry_after if retry_after is not None else _RETRY_BACKOFF_SECONDS * (attempt + 1)
            )

        raise last_error or WebSearchError(
            WebSearchErrorCode.PROVIDER_DOWN,
            "Brave Search failed",
            provider=_PROVIDER,
        )

    def _endpoint_for(self, request: WebSearchRequest) -> str:
        if request.result_type == WebSearchResultType.NEWS:
            return f"{self._base_url}/news/search"
        if request.result_type in (WebSearchResultType.WEB, WebSearchResultType.MIXED):
            return f"{self._base_url}/web/search"
        raise ValueError(f"Unsupported web search result type: {request.result_type}")

    def _params_for(self, request: WebSearchRequest) -> dict[str, str | int]:
        query = request.query
        if request.allowed_domains:
            query = f"{query} " + " ".join(f"site:{domain}" for domain in request.allowed_domains)
        if request.blocked_domains:
            query = f"{query} " + " ".join(f"-site:{domain}" for domain in request.blocked_domains)

        params: dict[str, str | int] = {
            "q": query,
            "count": request.limit,
            "country": request.country.lower(),
            "search_lang": request.search_lang,
            "safesearch": request.safe_search,
            "spellcheck": 1,
            "extra_snippets": "true",
        }
        if request.freshness_days:
            params["freshness"] = _freshness_window(request.freshness_days)
        if request.result_type == WebSearchResultType.WEB:
            params["result_filter"] = "web"
        return params

    def _response_from_json(
        self,
        data: dict[str, Any],
        request: WebSearchRequest,
        provider_request_id: str | None,
    ) -> WebSearchResponse:
        results: list[WebSearchResultItem] = []
        seen_urls: set[str] = set()

        for raw in _ordered_results(data, request.result_type):
            if len(results) >= request.limit:
                break
            item = _result_item_from_json(
                raw,
                len(results) + 1,
                provider_request_id,
                request.result_type,
            )
            if item is None or item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            results.append(item)

        return WebSearchResponse(
            results=tuple(results),
            provider=_PROVIDER,
            provider_request_id=provider_request_id,
        )


def _error_from_response(response: httpx.Response) -> WebSearchError:
    if response.status_code in (401, 403):
        code = WebSearchErrorCode.INVALID_KEY
    elif response.status_code == 429:
        code = WebSearchErrorCode.RATE_LIMITED
    elif 400 <= response.status_code < 500:
        code = WebSearchErrorCode.INVALID_REQUEST
    else:
        code = WebSearchErrorCode.PROVIDER_DOWN

    error = WebSearchError(
        code,
        f"Brave Search returned HTTP {response.status_code}",
        provider=_PROVIDER,
        status_code=response.status_code,
    )
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            error.retry_after = float(retry_after)
        except ValueError:
            pass
    return error


def _result_item_from_json(
    raw: Any,
    rank: int,
    provider_request_id: str | None,
    request_type: WebSearchResultType,
) -> WebSearchResultItem | None:
    if not isinstance(raw, dict):
        return None

    title = str(raw.get("title") or "").strip()
    url = _normalize_http_url(raw.get("url"))
    if not title or url is None:
        return None

    source_name = str(
        (raw.get("profile") or {}).get("name") or raw.get("source") or ""
    ).strip() or _hostname(url)
    published_at = (
        raw.get("age") or raw.get("page_age") or raw.get("published") or raw.get("published_time")
    )
    ref_type = "news" if request_type == WebSearchResultType.NEWS else "web"

    return WebSearchResultItem(
        result_ref=f"brave:{ref_type}:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:32],
        title=title[:512],
        url=url,
        display_url=_display_url(url),
        snippet=str(raw.get("description") or "").strip()[:1000],
        extra_snippets=tuple(
            snippet.strip()[:1000]
            for snippet in (raw.get("extra_snippets") or [])[:5]
            if isinstance(snippet, str) and snippet.strip()
        ),
        published_at=_published_at(published_at),
        source_name=source_name[:256] if source_name else None,
        rank=rank,
        provider=_PROVIDER,
        provider_request_id=provider_request_id,
    )


def _ordered_results(data: dict[str, Any], result_type: WebSearchResultType) -> list[Any]:
    web_results = list((data.get("web") or {}).get("results") or [])
    if result_type == WebSearchResultType.WEB:
        return web_results

    news_results = list((data.get("news") or {}).get("results") or data.get("results") or [])
    if result_type == WebSearchResultType.NEWS:
        return news_results

    if result_type == WebSearchResultType.MIXED:
        ordered: list[Any] = []
        for item in (data.get("mixed") or {}).get("main") or []:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            source_index = item["index"]
            if item.get("type") == "web" and 0 <= source_index < len(web_results):
                ordered.append(web_results[source_index])
            elif item.get("type") == "news" and 0 <= source_index < len(news_results):
                ordered.append(news_results[source_index])
        return ordered or [*web_results, *news_results]

    raise ValueError(f"Unsupported web search result type: {result_type}")


def _normalize_http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None

    parsed = urlsplit(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username or parsed.password:
        return None

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{hostname}:{port}"

    return urlunsplit(
        (
            scheme,
            netloc,
            quote(parsed.path or "", safe="/%:@"),
            urlencode(parse_qsl(parsed.query, keep_blank_values=True), doseq=True, quote_via=quote),
            "",
        )
    )


def _hostname(url: str) -> str:
    return urlsplit(url).hostname or ""


def _display_url(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    if path and path != "/":
        return f"{parsed.netloc}{path}"
    return parsed.netloc


def _freshness_window(days: int) -> str:
    if days <= 1:
        return "pd"
    if days <= 7:
        return "pw"
    if days <= 31:
        return "pm"
    return "py"


def _published_at(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return text[:128]

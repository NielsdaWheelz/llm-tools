"""Brave Search API binding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx

from llm_tools.web.contracts import (
    WebSearchError,
    WebSearchErrorCode,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResultItem,
    WebSearchResultType,
)

_PROVIDER = "brave"
_DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1"
_RETRY_BACKOFF_SECONDS = 0.25
_MAX_RETRY_AFTER_SECONDS = 2.0
# httpx decodes Content-Encoding before yielding response bytes.
_MAX_DECODED_RESPONSE_BYTES = 2 * 1_024 * 1_024
_MAX_ERROR_BODY_BYTES = 4 * 1_024


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
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError("Brave Search API key must not be empty")
        self._base_url = _normalize_base_url(base_url)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("Brave Search timeout_seconds must be numeric")
        self._timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self._timeout_seconds) or self._timeout_seconds <= 0:
            raise ValueError("Brave Search timeout_seconds must be positive and finite")

    async def search(
        self,
        request: WebSearchRequest,
        *,
        attempt_started: Callable[[], None] | None = None,
    ) -> WebSearchResponse:
        last_error: WebSearchError | None = None
        for attempt in range(request.max_attempts):
            if attempt_started is not None:
                attempt_started()
            try:
                async with self._client.stream(
                    "GET",
                    self._endpoint_for(request),
                    params=self._params_for(request),
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                    follow_redirects=False,
                    timeout=httpx.Timeout(self._timeout_seconds, connect=5.0),
                ) as response:
                    if response.status_code == 422 and await _invalid_token_rejected(response):
                        raise WebSearchError(
                            WebSearchErrorCode.CREDENTIAL_REJECTED,
                            "Brave Search rejected its subscription token",
                            provider=_PROVIDER,
                            status_code=422,
                            attempts=attempt + 1,
                        )
                    response.raise_for_status()
                    data = json.loads(await _bounded_response_body(response))
                if not isinstance(data, dict):
                    raise WebSearchError(
                        WebSearchErrorCode.BAD_RESPONSE,
                        "Brave Search returned malformed JSON",
                        provider=_PROVIDER,
                        attempts=attempt + 1,
                    )
                return self._response_from_json(
                    data,
                    request,
                    _provider_request_id(response, data),
                    attempts=attempt + 1,
                )
            except httpx.TimeoutException as exc:
                last_error = WebSearchError(
                    WebSearchErrorCode.TIMEOUT,
                    "Brave Search request timed out",
                    provider=_PROVIDER,
                    attempts=attempt + 1,
                )
                if attempt + 1 >= request.max_attempts:
                    raise last_error from exc
            except httpx.HTTPStatusError as exc:
                error = _error_from_response(exc.response)
                error.attempts = attempt + 1
                if error.code not in (
                    WebSearchErrorCode.RATE_LIMITED,
                    WebSearchErrorCode.PROVIDER_DOWN,
                ):
                    raise error from exc
                last_error = error
                if attempt + 1 >= request.max_attempts:
                    raise last_error from exc
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_error = WebSearchError(
                    WebSearchErrorCode.PROVIDER_DOWN,
                    "Brave Search network error",
                    provider=_PROVIDER,
                    attempts=attempt + 1,
                )
                if attempt + 1 >= request.max_attempts:
                    raise last_error from exc
            except (
                AttributeError,
                httpx.DecodingError,
                RecursionError,
                TypeError,
                ValueError,
            ) as exc:
                raise WebSearchError(
                    WebSearchErrorCode.BAD_RESPONSE,
                    "Brave Search returned malformed JSON",
                    provider=_PROVIDER,
                    attempts=attempt + 1,
                ) from exc

            retry_after = last_error.retry_after if last_error is not None else None
            delay = (
                retry_after if retry_after is not None else _RETRY_BACKOFF_SECONDS * (attempt + 1)
            )
            await asyncio.sleep(delay)

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
        *,
        attempts: int,
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
            retrieved_at=datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            attempts=attempts,
        )


async def _bounded_response_body(
    response: httpx.Response, *, max_bytes: int = _MAX_DECODED_RESPONSE_BYTES
) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        if not content_length.isdigit() or int(content_length) > max_bytes:
            raise ValueError("Brave Search response exceeds the wire limit")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError("Brave Search response exceeds the wire limit")
    return bytes(body)


async def _invalid_token_rejected(response: httpx.Response) -> bool:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, member in pairs:
            if name in value:
                raise ValueError("Brave Search error contains duplicate JSON members")
            value[name] = member
        return value

    try:
        body = await _bounded_response_body(response, max_bytes=_MAX_ERROR_BODY_BYTES)
        data = json.loads(body, object_pairs_hook=unique_object)
    except (httpx.DecodingError, httpx.TransportError, UnicodeDecodeError, ValueError):
        return False
    error = data.get("error") if isinstance(data, dict) else None
    return (
        isinstance(error, dict)
        and error.get("code") == "SUBSCRIPTION_TOKEN_INVALID"
        and type(error.get("status")) is int
        and error["status"] == 422
    )


def _provider_request_id(response: httpx.Response, data: dict[str, Any]) -> str | None:
    if "x-request-id" in response.headers:
        value: object = response.headers["x-request-id"]
    elif "request-id" in response.headers:
        value = response.headers["request-id"]
    else:
        value = data.get("request_id")
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError("Brave Search request id is invalid")
    return value


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
            parsed = float(retry_after)
            if math.isfinite(parsed):
                error.retry_after = min(max(parsed, 0.0), _MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return error


def _normalize_base_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Brave Search base_url must not be empty")

    parsed = urlsplit(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Brave Search base_url must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("Brave Search base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Brave Search base_url must not contain query or fragment")

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Brave Search base_url must include a host")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Brave Search base_url has an invalid port") from exc

    netloc = hostname
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{hostname}:{port}"

    path = quote(parsed.path.rstrip("/"), safe="/%:@")
    return urlunsplit((scheme, netloc, path, "", ""))


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
    extra_snippets = raw.get("extra_snippets") or []
    if not isinstance(extra_snippets, list):
        raise ValueError("Brave extra snippets must be an array")
    ref_type = "news" if request_type == WebSearchResultType.NEWS else "web"

    return WebSearchResultItem(
        result_ref=f"brave:{ref_type}:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:32],
        title=title[:512],
        url=url,
        display_url=_display_url(url),
        snippet=str(raw.get("description") or "").strip()[:1000],
        extra_snippets=tuple(
            snippet.strip()[:1000]
            for snippet in extra_snippets[:5]
            if isinstance(snippet, str) and snippet.strip()
        ),
        published_at=_published_at(published_at),
        source_name=source_name[:256] if source_name else None,
        rank=rank,
        provider=_PROVIDER,
        provider_request_id=provider_request_id,
    )


def _ordered_results(data: dict[str, Any], result_type: WebSearchResultType) -> list[Any]:
    web_results = _nested_results(data, "web")
    if result_type == WebSearchResultType.WEB:
        return web_results

    news_results = _nested_results(data, "news", fallback=data.get("results"))
    if result_type == WebSearchResultType.NEWS:
        return news_results

    if result_type == WebSearchResultType.MIXED:
        ordered: list[Any] = []
        mixed = data.get("mixed")
        if mixed is None:
            mixed_main: object = []
        elif isinstance(mixed, dict):
            mixed_main = mixed.get("main", [])
        else:
            raise ValueError("Brave mixed results must be an object")
        if not isinstance(mixed_main, list):
            raise ValueError("Brave mixed result order must be an array")
        for item in mixed_main:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            source_index = item["index"]
            if item.get("type") == "web" and 0 <= source_index < len(web_results):
                ordered.append(web_results[source_index])
            elif item.get("type") == "news" and 0 <= source_index < len(news_results):
                ordered.append(news_results[source_index])
        return ordered or [*web_results, *news_results]

    raise ValueError(f"Unsupported web search result type: {result_type}")


def _nested_results(
    data: dict[str, Any],
    key: str,
    *,
    fallback: object = None,
) -> list[Any]:
    container = data.get(key)
    if container is None:
        results = [] if fallback is None else fallback
    elif isinstance(container, dict):
        results = container.get("results", [] if fallback is None else fallback)
    else:
        raise ValueError(f"Brave {key} results must be an object")
    if not isinstance(results, list):
        raise ValueError(f"Brave {key} result list must be an array")
    return results


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

    normalized = urlunsplit(
        (
            scheme,
            netloc,
            quote(parsed.path or "", safe="/%:@"),
            urlencode(parse_qsl(parsed.query, keep_blank_values=True), doseq=True, quote_via=quote),
            "",
        )
    )
    return normalized if len(normalized) <= 4_096 else None


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

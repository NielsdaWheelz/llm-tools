"""Brave Search API binding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from llm_tools.web.contracts import (
    DEFAULT_WEB_SEARCH_POLICY,
    WebProviderQuery,
    WebSearchError,
    WebSearchErrorCode,
    WebSearchOmission,
    WebSearchPolicy,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResultItem,
    WebSearchResultType,
)

_PROVIDER = "brave"
_RETRY_BACKOFF_SECONDS = 0.25
_MAX_RETRY_AFTER_SECONDS = 2.0
# httpx decodes Content-Encoding before yielding response bytes.
_MAX_DECODED_RESPONSE_BYTES = 2 * 1_024 * 1_024


class BraveSearchProvider:
    """Async Brave Search provider with normalized results."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        policy: WebSearchPolicy = DEFAULT_WEB_SEARCH_POLICY,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._client = client
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError("Brave Search API key must not be empty")
        self.policy = policy
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
            return self.policy.endpoint.removesuffix("/web/search") + "/news/search"
        if request.result_type in (WebSearchResultType.WEB, WebSearchResultType.MIXED):
            return self.policy.endpoint
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
            "offset": request.offset,
            "country": request.country.lower(),
            "search_lang": request.search_lang,
            "safesearch": request.safe_search,
            "spellcheck": int(request.spellcheck),
            "extra_snippets": str(request.extra_snippets).lower(),
        }
        if request.freshness is not None:
            params["freshness"] = request.freshness
        if request.result_filter is not None:
            params["result_filter"] = request.result_filter
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
        omissions: list[WebSearchOmission] = []
        for source_kind, provider_index, raw in _ordered_results(data, request.result_type):
            try:
                item = _result_item_from_json(
                    raw, len(results) + 1, provider_request_id, source_kind
                )
            except _InvalidItem as exc:
                omissions.append(
                    WebSearchOmission(
                        source_kind=source_kind, provider_index=provider_index, reason=exc.reason
                    )
                )
                continue
            results.append(item)
        query = data.get("query")
        if query is None:
            query = {}
        if not isinstance(query, dict):
            raise ValueError("Brave query metadata must be an object")
        more = query.get("more_results_available")
        if more is not None and type(more) is not bool:
            raise ValueError("Brave pagination flag must be boolean")
        if any(
            query.get(key) is not None and not isinstance(query[key], str)
            for key in ("original", "altered", "cleaned")
        ):
            raise ValueError("Brave query strings must be exact text")
        return WebSearchResponse(
            results=tuple(results),
            provider=_PROVIDER,
            provider_request_id=provider_request_id,
            retrieved_at=datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            attempts=attempts,
            more_results_available=more,
            provider_query=WebProviderQuery(
                original=query.get("original"),
                altered=query.get("altered"),
                cleaned=query.get("cleaned"),
            ),
            omissions=tuple(omissions),
        )


async def _bounded_response_body(response: httpx.Response) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        if not content_length.isdigit() or int(content_length) > _MAX_DECODED_RESPONSE_BYTES:
            raise ValueError("Brave Search response exceeds the wire limit")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_DECODED_RESPONSE_BYTES:
            raise ValueError("Brave Search response exceeds the wire limit")
    return bytes(body)


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
    elif response.status_code == 402:
        code = WebSearchErrorCode.PAYMENT_REQUIRED
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


class _InvalidItem(Exception):
    def __init__(
        self, reason: Literal["invalid_item", "missing_url", "unrepresentable_url"]
    ) -> None:
        self.reason: Literal["invalid_item", "missing_url", "unrepresentable_url"] = reason


def _result_item_from_json(
    raw: Any, rank: int, provider_request_id: str | None, source_kind: Literal["web", "news"]
) -> WebSearchResultItem:
    if not isinstance(raw, dict):
        raise _InvalidItem("invalid_item")
    url = raw.get("url")
    if not isinstance(url, str) or not url:
        raise _InvalidItem("missing_url")
    try:
        parsed = urlsplit(url)
        if (
            len(url) > 4096
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or any(ord(c) < 33 or ord(c) == 127 for c in url)
        ):
            raise ValueError
        url.encode("utf-8")
        _ = parsed.port
    except (ValueError, UnicodeError):
        raise _InvalidItem("unrepresentable_url") from None
    title = raw.get("title")
    description = raw.get("description")
    extra = raw.get("extra_snippets")
    profile = raw.get("profile")
    if description is None:
        description = ""
    if extra is None:
        extra = []
    if profile is None:
        profile = {}
    if (
        not isinstance(title, str)
        or not title
        or not isinstance(description, str)
        or not isinstance(extra, list)
        or any(not isinstance(v, str) for v in extra)
        or not isinstance(profile, dict)
    ):
        raise _InvalidItem("invalid_item")
    source_name = profile.get("name")
    if source_name is None:
        source_name = raw.get("source")
    if source_name is None:
        source_name = parsed.hostname
    published = None
    for field in ("age", "page_age", "published", "published_time"):
        if raw.get(field) is not None:
            published = raw[field]
            break
    if not isinstance(source_name, str) or (
        published is not None and not isinstance(published, str)
    ):
        raise _InvalidItem("invalid_item")
    values = {
        "title": title,
        "display_url": parsed.netloc + parsed.path.rstrip("/"),
        "snippet": description,
        "published_at": published,
        "source_name": source_name,
    }
    limits = {
        "title": 512,
        "display_url": 4096,
        "snippet": 1000,
        "published_at": 128,
        "source_name": 256,
    }
    truncated = [
        name for name, value in values.items() if value is not None and len(value) > limits[name]
    ]
    if len(extra) > 5 or any(len(v) > 1000 for v in extra):
        truncated.append("extra_snippets")
    return WebSearchResultItem(
        result_ref=f"brave:{source_kind}:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:32],
        title=title[:512],
        url=url,
        display_url=(parsed.netloc + parsed.path.rstrip("/"))[:4096],
        snippet=description[:1000],
        extra_snippets=tuple(value[:1000] for value in extra[:5]),
        published_at=published[:128] if published is not None else None,
        source_name=source_name[:256],
        rank=rank,
        provider=_PROVIDER,
        provider_request_id=provider_request_id,
        source_kind=source_kind,
        truncated_fields=tuple(truncated),
    )


def _ordered_results(
    data: dict[str, Any], result_type: WebSearchResultType
) -> list[tuple[Any, int, Any]]:
    collections = {
        "web": _nested_results(data, "web"),
        "news": _nested_results(data, "news", fallback=data.get("results")),
    }
    if result_type != WebSearchResultType.MIXED:
        kind = "news" if result_type == WebSearchResultType.NEWS else "web"
        return [(kind, i, raw) for i, raw in enumerate(collections[kind])]
    mixed = data.get("mixed")
    if mixed is None:
        mixed = {}
    if not isinstance(mixed, dict):
        raise ValueError("Brave mixed metadata must be an object")
    main = mixed.get("main")
    if main is None:
        main = []
    if not isinstance(main, list):
        raise ValueError("Brave mixed order must be an array")
    ordered = []
    seen = set()
    for item in main:
        if not isinstance(item, dict):
            raise ValueError("Brave mixed reference must be an object")
        kind, index = item.get("type"), item.get("index")
        if kind not in collections:
            continue
        all_items = item.get("all", False)
        if type(all_items) is not bool:
            raise ValueError("Brave mixed all flag must be a boolean")
        if all_items:
            indices = range(len(collections[kind]))
        elif type(index) is int and 0 <= index < len(collections[kind]):
            indices = (index,)
        else:
            continue
        for index in indices:
            if (kind, index) not in seen:
                ordered.append((kind, index, collections[kind][index]))
                seen.add((kind, index))
    for kind, items in collections.items():
        ordered.extend((kind, i, raw) for i, raw in enumerate(items) if (kind, i) not in seen)
    return ordered


def _nested_results(data: dict[str, Any], key: str, *, fallback: object = None) -> list[Any]:
    container = data.get(key)
    if container is None:
        results = [] if fallback is None else fallback
    elif isinstance(container, dict):
        results = container.get("results", [] if fallback is None else fallback)
    else:
        raise ValueError("Brave result container must be an object")
    if not isinstance(results, list):
        raise ValueError("Brave result collection must be an array")
    return results

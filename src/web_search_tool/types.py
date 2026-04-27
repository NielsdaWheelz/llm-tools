"""Public web-search contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol
from urllib.parse import urlsplit

_MIN_QUERY_LENGTH = 2
_MAX_QUERY_LENGTH = 400
_MAX_QUERY_WORDS = 50
_MAX_LIMIT = 10


class WebSearchResultType(StrEnum):
    WEB = "web"
    NEWS = "news"
    MIXED = "mixed"


class WebSearchErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_KEY = "invalid_key"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_DOWN = "provider_down"
    BAD_RESPONSE = "bad_response"


class WebSearchError(Exception):
    """Normalized web-search provider error."""

    def __init__(
        self,
        code: WebSearchErrorCode,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retry_after: float | None = None
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WebSearchRequest:
    query: str
    result_type: WebSearchResultType = WebSearchResultType.MIXED
    limit: int = 10
    freshness_days: int | None = None
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    country: str = "US"
    search_lang: str = "en"
    safe_search: Literal["off", "moderate", "strict"] = "moderate"

    def __post_init__(self) -> None:
        try:
            result_type = WebSearchResultType(self.result_type)
        except ValueError as exc:
            raise ValueError("Web search result_type is invalid") from exc

        if self.safe_search not in ("off", "moderate", "strict"):
            raise ValueError("Web search safe_search is invalid")

        query = " ".join(self.query.split())
        if len(query) < _MIN_QUERY_LENGTH:
            raise ValueError("Web search query is too short")
        if len(query) > _MAX_QUERY_LENGTH:
            raise ValueError("Web search query is too long")
        if len(query.split()) > _MAX_QUERY_WORDS:
            raise ValueError("Web search query has too many words")

        if self.limit < 1 or self.limit > _MAX_LIMIT:
            raise ValueError(f"Web search limit must be between 1 and {_MAX_LIMIT}")

        if self.freshness_days is not None and self.freshness_days < 1:
            raise ValueError("Web search freshness_days must be positive")

        country = self.country.strip().upper()
        if len(country) != 2 or not country.isalpha():
            raise ValueError("Web search country must be a 2-letter code")

        search_lang = self.search_lang.strip().lower()
        if len(search_lang) < 2 or not search_lang.replace("-", "").isalpha():
            raise ValueError("Web search language must be a language code")

        object.__setattr__(self, "query", query)
        object.__setattr__(self, "result_type", result_type)
        object.__setattr__(self, "country", country)
        object.__setattr__(self, "search_lang", search_lang)
        object.__setattr__(
            self,
            "allowed_domains",
            tuple(_normalize_search_domain(domain) for domain in self.allowed_domains),
        )
        object.__setattr__(
            self,
            "blocked_domains",
            tuple(_normalize_search_domain(domain) for domain in self.blocked_domains),
        )


@dataclass(frozen=True, slots=True)
class WebSearchResultItem:
    result_ref: str
    title: str
    url: str
    display_url: str
    snippet: str
    extra_snippets: tuple[str, ...]
    published_at: str | None
    source_name: str | None
    rank: int
    provider: str
    provider_request_id: str | None


@dataclass(frozen=True, slots=True)
class WebSearchResponse:
    results: tuple[WebSearchResultItem, ...]
    provider: str
    provider_request_id: str | None
    retrieved_at: str


class WebSearchProvider(Protocol):
    async def search(self, request: WebSearchRequest) -> WebSearchResponse: ...


def _normalize_search_domain(value: str) -> str:
    raw = value.strip().lower().rstrip(".")
    if not raw:
        raise ValueError("Domain filter must not be empty")

    if "://" in raw:
        raw = urlsplit(raw).hostname or ""
    elif "/" in raw:
        raw = raw.split("/", 1)[0]

    if "@" in raw:
        raise ValueError("Domain filter must not contain credentials")

    if ":" in raw:
        host, _, maybe_port = raw.rpartition(":")
        if host and maybe_port.isdigit():
            raw = host

    try:
        normalized = raw.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Domain filter is not a valid hostname") from exc

    labels = normalized.split(".")
    if len(labels) < 2:
        raise ValueError("Domain filter must include a registrable domain")
    for label in labels:
        if not label or label.startswith("-") or label.endswith("-"):
            raise ValueError("Domain filter is not a valid hostname")
        if not all(character.isalnum() or character == "-" for character in label):
            raise ValueError("Domain filter is not a valid hostname")

    return normalized

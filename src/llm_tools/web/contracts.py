"""Owned programmatic and model-visible contracts for the Web tool family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_tools.evidence import EvidenceReceipt

_MIN_QUERY_LENGTH = 2
_MAX_QUERY_LENGTH = 400
_MAX_QUERY_WORDS = 50
_MAX_LIMIT = 20
_MAX_QUERY_WORDS_PATTERN = r"^\s*\S+(?:\s+\S+){0,49}\s*$"
_MIN_PROGRAMMATIC_QUERY_LENGTH = 1


WEB_READ_PROJECTIONS = {
    "application/json": "web-json-source-v2",
    "application/xhtml+xml": "html-visible-text-v2",
    "text/html": "html-visible-text-v2",
    "text/plain": "web-plain-source-v3",
}


class WebSearchResultType(StrEnum):
    WEB = "web"
    NEWS = "news"
    MIXED = "mixed"


class WebSearchErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_KEY = "invalid_key"
    PAYMENT_REQUIRED = "payment_required"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_DOWN = "provider_down"
    BAD_RESPONSE = "bad_response"


class WebSearchError(Exception):
    """Normalized programmatic Web-search provider error."""

    def __init__(
        self,
        code: WebSearchErrorCode,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        self.code = code
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.attempts = attempts
        self.retry_after: float | None = None
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WebSearchRequest:
    """Broader host-owned Brave request; the model sees only query and freshness."""

    query: str
    result_type: WebSearchResultType = WebSearchResultType.MIXED
    limit: int = 10
    freshness: str | None = None
    offset: int = 0
    spellcheck: bool = True
    extra_snippets: bool = True
    result_filter: str | None = None
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    country: str = "US"
    search_lang: str = "en"
    safe_search: Literal["off", "moderate", "strict"] = "moderate"
    max_attempts: int = 2

    def __post_init__(self) -> None:
        try:
            result_type = WebSearchResultType(self.result_type)
        except ValueError as exc:
            raise ValueError("Web search result_type is invalid") from exc

        if self.safe_search not in ("off", "moderate", "strict"):
            raise ValueError("Web search safe_search is invalid")

        query = " ".join(self.query.split())
        if len(query) < _MIN_PROGRAMMATIC_QUERY_LENGTH:
            raise ValueError("Web search query is too short")
        if len(query) > _MAX_QUERY_LENGTH:
            raise ValueError("Web search query is too long")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("Web search limit must be an integer")
        if self.limit < 1 or self.limit > _MAX_LIMIT:
            raise ValueError(f"Web search limit must be between 1 and {_MAX_LIMIT}")
        _validate_freshness(self.freshness)
        if type(self.spellcheck) is not bool or type(self.extra_snippets) is not bool:
            raise TypeError("search boolean options must be booleans")
        if type(self.offset) is not int or not 0 <= self.offset <= 9:
            raise ValueError("Web search offset must be between 0 and 9")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 2
        ):
            raise ValueError("Web search max_attempts must be 1 or 2")

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
    source_kind: Literal["web", "news"] = "web"
    truncated_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WebSearchResponse:
    results: tuple[WebSearchResultItem, ...]
    provider: str
    provider_request_id: str | None
    retrieved_at: str
    attempts: int
    more_results_available: bool | None = None
    provider_query: WebProviderQuery | None = None
    omissions: tuple[WebSearchOmission, ...] = ()


class WebSearchProvider(Protocol):
    policy: WebSearchPolicy

    async def search(
        self,
        request: WebSearchRequest,
        *,
        attempt_started: Callable[[], None] | None = None,
    ) -> WebSearchResponse:
        """Report attempts before dispatch and propagate task cancellation unchanged."""

        ...


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebSearchSource(_StrictModel):
    provider: Literal["brave"]
    endpoint: str


class WebSearchScope(_StrictModel):
    source: WebSearchSource
    query: str
    freshness: str | None
    country: str
    search_lang: str
    safe_search: Literal["off", "moderate", "strict"]
    spellcheck: bool
    extra_snippets: bool
    result_filter: str | None
    count: Literal[20]


class WebProviderQuery(_StrictModel):
    original: str | None
    altered: str | None
    cleaned: str | None


class WebSearchOmission(_StrictModel):
    source_kind: Literal["web", "news"]
    provider_index: Annotated[int, Field(strict=True, ge=0)]
    reason: Literal["invalid_item", "missing_url", "unrepresentable_url"]


@dataclass(frozen=True, slots=True)
class WebSearchPolicy:
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    country: str = "us"
    search_lang: str = "en"
    safe_search: Literal["off", "moderate", "strict"] = "moderate"
    spellcheck: bool = True
    extra_snippets: bool = True
    result_filter: str | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("search endpoint must be an exact HTTPS URL without credentials")
        object.__setattr__(self, "endpoint", str(httpx.URL(self.endpoint)))
        if (
            self.country != self.country.lower()
            or len(self.country) != 2
            or not self.country.isalpha()
        ):
            raise ValueError("country must be normalized lowercase")
        if (
            self.search_lang != self.search_lang.lower()
            or len(self.search_lang) < 2
            or not self.search_lang.replace("-", "").isalpha()
        ):
            raise ValueError("language must be normalized lowercase")
        if (
            self.safe_search not in {"off", "moderate", "strict"}
            or type(self.spellcheck) is not bool
            or type(self.extra_snippets) is not bool
        ):
            raise ValueError("invalid fixed search policy")
        if self.result_filter is not None and (
            not isinstance(self.result_filter, str) or not self.result_filter
        ):
            raise ValueError("invalid fixed result filter")


DEFAULT_WEB_SEARCH_POLICY = WebSearchPolicy()


def _validate_freshness(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError("freshness must be a string or null")
    if value in {"pd", "pw", "pm", "py"}:
        return
    if len(value) != 22 or value[10:12] != "to":
        raise ValueError("invalid freshness filter")
    start, end = date.fromisoformat(value[:10]), date.fromisoformat(value[12:])
    if start.isoformat() != value[:10] or end.isoformat() != value[12:] or start > end:
        raise ValueError("invalid freshness date range")


class WebSearchInput(_StrictModel):
    query: Annotated[
        str,
        Field(
            min_length=2,
            max_length=400,
            pattern=_MAX_QUERY_WORDS_PATTERN,
            description="Public-Web search query; use concrete source, entity, and time terms.",
        ),
    ]
    freshness: Annotated[
        str | None,
        Field(pattern=r"^(pd|pw|pm|py|[0-9]{4}-[0-9]{2}-[0-9]{2}to[0-9]{4}-[0-9]{2}-[0-9]{2})$"),
    ]
    page_size: Annotated[int, Field(strict=True, ge=1, le=20)]
    cursor: Annotated[str | None, Field(max_length=16_384)]

    @field_validator("freshness")
    @classmethod
    def valid_freshness(cls, value: str | None) -> str | None:
        _validate_freshness(value)
        return value

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < _MIN_QUERY_LENGTH or len(normalized.split()) > _MAX_QUERY_WORDS:
            raise ValueError("query must contain 2-400 characters and at most 50 words")
        return normalized


class WebSearchHit(_StrictModel):
    result_ref: Annotated[str, Field(min_length=1, max_length=128)]
    title: Annotated[str, Field(min_length=1, max_length=512)]
    url: Annotated[str, Field(min_length=1, max_length=4_096)]
    display_url: Annotated[str, Field(min_length=1, max_length=4_096)]
    snippet: Annotated[str, Field(max_length=1_000)]
    extra_snippets: Annotated[
        tuple[Annotated[str, Field(max_length=1_000)], ...], Field(max_length=5)
    ]
    published_at: Annotated[str | None, Field(max_length=128)]
    source_name: Annotated[str | None, Field(max_length=256)]
    rank: Annotated[int, Field(strict=True, ge=1)]
    source_kind: Literal["web", "news"]
    truncated_fields: tuple[
        Literal["title", "display_url", "snippet", "extra_snippets", "published_at", "source_name"],
        ...,
    ]

    @field_validator("truncated_fields")
    @classmethod
    def unique_truncations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate truncated field")
        return value

    provider: Literal["brave"]
    provider_request_id: Annotated[str | None, Field(max_length=512)]


class WebSearchEvidence(_StrictModel):
    type: Literal["web.search"] = "web.search"
    provider: Literal["brave"] = "brave"
    provider_request_id: Annotated[str | None, Field(max_length=512)]


class WebSearchSuccess(_StrictModel):
    scope: WebSearchScope
    results: Annotated[tuple[WebSearchHit, ...], Field(max_length=20)]
    provider_offset: Annotated[int, Field(strict=True, ge=0, le=9)]
    provider_count: Literal[20]
    more_results_available: bool | None
    provider_query: WebProviderQuery
    omissions: tuple[WebSearchOmission, ...]
    next_cursor: str | None
    boundary: Literal["more", "exhausted", "provider_limit", "unknown"]
    provider: Literal["brave"]
    provider_request_id: Annotated[str | None, Field(max_length=512)]
    observed_at: datetime
    evidence: WebSearchEvidence


class WebReadInput(_StrictModel):
    url: Annotated[
        str,
        Field(
            min_length=1,
            max_length=4_096,
            description=(
                "One public HTTP(S) URL without credentials or a fragment; private destinations "
                "are rejected at every hop."
            ),
        ),
    ]
    offset: Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
    max_bytes: Annotated[int, Field(strict=True, ge=1, le=32_768)]
    expected_digest: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def digest_for_continuation(self) -> WebReadInput:
        if self.offset and self.expected_digest is None:
            raise ValueError("continuation requires expected_digest")
        return self


class WebReadSource(_StrictModel):
    source_uri: Annotated[str, Field(min_length=1, max_length=4_096)]
    final_url: Annotated[str, Field(min_length=1, max_length=4_096)]
    media_type: Annotated[str, Field(min_length=1, max_length=128)]


class WebReadSuccess(_StrictModel):
    source: WebReadSource
    projection: str
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    observed_at: datetime
    offset: Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
    end_offset: Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
    total_bytes: Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
    text: str
    next_offset: Annotated[int | None, Field(strict=True, ge=0, le=2**53 - 1)]
    title: Annotated[str | None, Field(max_length=512)]
    evidence: EvidenceReceipt


@dataclass(frozen=True, slots=True)
class WebDocument:
    source: WebReadSource
    title: str | None
    text: str
    projection: str
    evidence: EvidenceReceipt


class InvalidUrl(_StrictModel):
    type: Literal["InvalidUrl"] = "InvalidUrl"


class UnsafeDestination(_StrictModel):
    type: Literal["UnsafeDestination"] = "UnsafeDestination"


class UnsupportedContent(_StrictModel):
    type: Literal["UnsupportedContent"] = "UnsupportedContent"


class TooLarge(_StrictModel):
    type: Literal["TooLarge"] = "TooLarge"


class RateLimited(_StrictModel):
    type: Literal["RateLimited"] = "RateLimited"


class UpstreamUnavailable(_StrictModel):
    type: Literal["UpstreamUnavailable"] = "UpstreamUnavailable"


class InvalidUpstreamResponse(_StrictModel):
    type: Literal["InvalidUpstreamResponse"] = "InvalidUpstreamResponse"


class InvalidCursor(_StrictModel):
    type: Literal["InvalidCursor"] = "InvalidCursor"


class StaleCursor(_StrictModel):
    type: Literal["StaleCursor"] = "StaleCursor"


class SourceChanged(_StrictModel):
    type: Literal["SourceChanged"] = "SourceChanged"


type WebSearchToolError = (
    RateLimited | UpstreamUnavailable | InvalidUpstreamResponse | InvalidCursor | StaleCursor
)
type WebReadToolError = (
    InvalidUrl
    | UnsafeDestination
    | UnsupportedContent
    | TooLarge
    | RateLimited
    | UpstreamUnavailable
    | InvalidUpstreamResponse
    | SourceChanged
)


@dataclass(frozen=True, slots=True)
class WebReadHop:
    uri: str
    peer_address: str
    status_code: int


@dataclass(frozen=True, slots=True)
class WebReadResponse:
    value: WebDocument
    attempts: int
    hops: tuple[WebReadHop, ...]


class WebReadFailure(Exception):
    """Private reader failure carrying a safe declared error and exact spend."""

    def __init__(self, error: WebReadToolError, *, attempts: int) -> None:
        self.error = error
        self.attempts = attempts
        super().__init__(error.type)


class WebReadDeadline(Exception):
    """Private reader deadline signal carrying exact request spend."""

    def __init__(self, *, attempts: int) -> None:
        self.attempts = attempts
        super().__init__("Web reader deadline exceeded")


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

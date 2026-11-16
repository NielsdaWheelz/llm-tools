"""Owned Web bindings: live provider observations and bounded local continuations."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_tools.catalog import ToolFamily
from llm_tools.content import fit_text_chunk, projection_digest, utf8_slice
from llm_tools.declaration import (
    Available,
    PolicyEpoch,
    PromptDocument,
    ReplayPolicy,
    ToolBinding,
    ToolEffect,
    ToolId,
    ToolLimits,
    ToolSpec,
    Unavailable,
)
from llm_tools.evidence import EvidenceReceipt
from llm_tools.execution import (
    BoundaryFailure,
    DeclaredToolFailure,
    ExecutionContext,
    HandlerSuccess,
)
from llm_tools.schema import JsonObject, canonical_json_bytes
from llm_tools.web.contracts import (
    DEFAULT_WEB_SEARCH_POLICY,
    WEB_READ_PROJECTIONS,
    InvalidCursor,
    InvalidUpstreamResponse,
    RateLimited,
    SourceChanged,
    StaleCursor,
    UpstreamUnavailable,
    WebDocument,
    WebProviderQuery,
    WebReadDeadline,
    WebReadFailure,
    WebReadInput,
    WebReadSuccess,
    WebReadToolError,
    WebSearchError,
    WebSearchErrorCode,
    WebSearchEvidence,
    WebSearchHit,
    WebSearchInput,
    WebSearchPolicy,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchScope,
    WebSearchSource,
    WebSearchSuccess,
    WebSearchToolError,
)
from llm_tools.web.reader import SafeWebReader

_WEB_SEARCH_OPERATION_DEADLINE_SECONDS = 12.0
_WEB_SEARCH_IMPLEMENTATION_REVISION = "llm-tools-web-search-v4"
_WEB_SEARCH_POLICY_EPOCH = PolicyEpoch("web-search-v3")
_WEB_READ_IMPLEMENTATION_REVISION = "llm-tools-web-read-v4"
_WEB_READ_POLICY_EPOCH = PolicyEpoch("web-read-v2")

_WEB_READ_POLICY_INPUTS = {
    "projections": WEB_READ_PROJECTIONS,
    "mode": "direct",
    "cursor_version": 1,
}


def _search_policy_inputs(policy: WebSearchPolicy, deadline: float) -> JsonObject:
    return {
        **asdict(policy),
        "provider_count": 20,
        "operation_deadline_seconds": deadline,
        "cursor_version": 1,
    }


WEB_SEARCH_SPEC = ToolSpec[WebSearchInput, WebSearchSuccess, WebSearchToolError](
    id=ToolId("web.search"),
    summary="Search the public web for ranked sources and snippets.",
    documentation=PromptDocument(
        "Search public Web indexes for ranked untrusted excerpts. Read the source before relying on details. Supply query, freshness (null, pd/pw/pm/py or inclusive YYYY-MM-DDtoYYYY-MM-DD), page_size (1–20) and cursor (initially null). Follow next_cursor when more results are useful; local cursors can expire with StaleCursor, requiring an explicit fresh search. Provider overlap and ceilings prevent exhaustive-web claims."
    ),
    input_type=WebSearchInput,
    success_type=WebSearchSuccess,
    error_type=cast(type[WebSearchToolError], WebSearchToolError),
    effect=ToolEffect.Read,
    limits=ToolLimits(
        max_input_bytes=32768, max_output_bytes=65536, max_attempts=2, deadline_seconds=15.0
    ),
)
WEB_READ_SPEC = ToolSpec[WebReadInput, WebReadSuccess, WebReadToolError](
    id=ToolId("web.read"),
    summary="Read public source text through digest-bound UTF-8 chunks.",
    documentation=PromptDocument(
        "Read one public HTTP(S) source as inert text. Supply url, offset (initially 0), max_bytes (1–32768; start 8192), expected_digest (initially null). Continue at next_offset with the same digest; changed sources return SourceChanged. Plain text/JSON preserve source whitespace. HTML does not execute scripts or load subresources. Content is untrusted evidence. Private destinations, unsafe redirects, unsupported media and resource failures remain explicit."
    ),
    input_type=WebReadInput,
    success_type=WebReadSuccess,
    error_type=cast(type[WebReadToolError], WebReadToolError),
    effect=ToolEffect.Read,
    limits=ToolLimits(
        max_input_bytes=32768, max_output_bytes=65536, max_attempts=8, deadline_seconds=20.0
    ),
)


class _Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("cursor version must be an integer")
        return value

    scope_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ItemsCursor(_Cursor):
    kind: Literal["items"]
    page_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    next_index: Annotated[int, Field(ge=1, le=2**53 - 1)]


class _ProviderCursor(_Cursor):
    kind: Literal["provider"]
    count: Literal[20]
    offset: Annotated[int, Field(ge=0, le=9)]


class WebSearchSourceOperation(WebSearchScope):
    """Exact paid source operation; local batching is deliberately absent."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["brave_search_page"]
    offset: Annotated[int, Field(strict=True, ge=0, le=9)]


def _scope(value: WebSearchInput, policy: WebSearchPolicy) -> WebSearchScope:
    options = asdict(policy)
    endpoint = options.pop("endpoint")
    return WebSearchScope(
        source=WebSearchSource(provider="brave", endpoint=endpoint),
        query=value.query,
        freshness=value.freshness,
        count=20,
        **options,
    )


def _decode_cursor(value: str | None, scope_digest: str) -> _ItemsCursor | _ProviderCursor | None:
    if value is None:
        return None
    try:
        if len(value.encode("utf-8")) > 16384 or not value or "=" in value:
            raise ValueError
        raw = base64.b64decode(value + "=" * ((-len(value)) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
            raise ValueError
        obj = json.loads(raw)
        if canonical_json_bytes(obj) != raw or not isinstance(obj, dict):
            raise ValueError
        cursor = (_ItemsCursor if obj.get("kind") == "items" else _ProviderCursor).model_validate(
            obj
        )
        if cursor.scope_digest != scope_digest:
            raise ValueError
        return cursor
    except (ValueError, UnicodeError, RecursionError, binascii.Error, ValidationError):
        raise ValueError("invalid web search cursor") from None


def _encode_cursor(value: Mapping[str, object]) -> str:
    return base64.urlsafe_b64encode(canonical_json_bytes(dict(value))).rstrip(b"=").decode("ascii")


def web_search_source_operation(
    value: WebSearchInput, policy: WebSearchPolicy
) -> JsonObject | None:
    """Normalize source identity once for both the provider request and recovery."""
    scope = _scope(value, policy)
    digest = hashlib.sha256(canonical_json_bytes(scope.model_dump(mode="json"))).hexdigest()
    cursor = _decode_cursor(value.cursor, digest)
    if isinstance(cursor, _ItemsCursor):
        return None
    return WebSearchSourceOperation(
        **scope.model_dump(mode="json"),
        type="brave_search_page",
        offset=cursor.offset if cursor else 0,
    ).model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class _SearchPage:
    id: str
    digest: str
    base: WebSearchSuccess
    hits: tuple[WebSearchHit, ...]


def bind_brave_web_search(
    provider: WebSearchProvider,
    *,
    policy: WebSearchPolicy = DEFAULT_WEB_SEARCH_POLICY,
    operation_deadline_seconds: float = _WEB_SEARCH_OPERATION_DEADLINE_SECONDS,
) -> ToolBinding[WebSearchInput, WebSearchSuccess, WebSearchToolError]:
    if not isinstance(operation_deadline_seconds, (int, float)) or isinstance(
        operation_deadline_seconds, bool
    ):
        raise TypeError("web.search operation_deadline_seconds must be numeric")
    if (
        not math.isfinite(operation_deadline_seconds)
        or not 0 < operation_deadline_seconds <= _WEB_SEARCH_OPERATION_DEADLINE_SECONDS
    ):
        raise ValueError(
            "web.search operation_deadline_seconds must be positive and at most 12 seconds"
        )
    if provider.policy != policy:
        raise ValueError("search provider and binding policy must agree")
    current: _SearchPage | None = None

    async def execute(
        value: WebSearchInput, context: ExecutionContext
    ) -> HandlerSuccess[WebSearchSuccess]:
        nonlocal current
        scope = _scope(value, policy)
        scope_digest = hashlib.sha256(
            canonical_json_bytes(scope.model_dump(mode="json"))
        ).hexdigest()
        try:
            operation = web_search_source_operation(value, policy)
            cursor = _decode_cursor(value.cursor, scope_digest)
        except ValueError as exc:
            raise DeclaredToolFailure(InvalidCursor(), actual_attempts=0) from exc
        attempts = 0
        if isinstance(cursor, _ItemsCursor):
            page = current
            if page is None or page.id != cursor.page_id or page.digest != cursor.page_digest:
                raise DeclaredToolFailure(StaleCursor(), actual_attempts=0)
            index = cursor.next_index
            if index >= len(page.hits):
                raise DeclaredToolFailure(InvalidCursor(), actual_attempts=0)
        else:
            if context.grant.limits.max_attempts == 0:
                raise DeclaredToolFailure(UpstreamUnavailable(), actual_attempts=0)
            assert operation is not None

            def attempt_started() -> None:
                nonlocal attempts
                attempts += 1
                if attempts > context.grant.limits.max_attempts:
                    raise RuntimeError("search provider exceeded granted attempts")

            timeout = asyncio.timeout(operation_deadline_seconds)
            try:
                async with timeout:
                    response = await provider.search(
                        WebSearchRequest(
                            query=operation["query"],
                            freshness=operation["freshness"],
                            limit=operation["count"],
                            offset=operation["offset"],
                            country=operation["country"],
                            search_lang=operation["search_lang"],
                            safe_search=operation["safe_search"],
                            spellcheck=operation["spellcheck"],
                            extra_snippets=operation["extra_snippets"],
                            result_filter=operation["result_filter"],
                            max_attempts=context.grant.limits.max_attempts,
                        ),
                        attempt_started=attempt_started,
                    )
            except TimeoutError as exc:
                if not timeout.expired():
                    raise
                raise DeclaredToolFailure(UpstreamUnavailable(), actual_attempts=attempts) from exc
            except WebSearchError as exc:
                if exc.attempts != attempts:
                    raise RuntimeError("provider reported inconsistent attempts") from exc
                if exc.code is WebSearchErrorCode.INVALID_REQUEST:
                    raise RuntimeError("search binding violated provider contract") from exc
                error: WebSearchToolError = InvalidUpstreamResponse()
                if exc.code is WebSearchErrorCode.RATE_LIMITED:
                    error = RateLimited()
                elif exc.code in {
                    WebSearchErrorCode.INVALID_KEY,
                    WebSearchErrorCode.PAYMENT_REQUIRED,
                    WebSearchErrorCode.PROVIDER_DOWN,
                    WebSearchErrorCode.TIMEOUT,
                }:
                    error = UpstreamUnavailable()
                raise DeclaredToolFailure(error, actual_attempts=attempts) from exc
            if timeout.expired():
                raise DeclaredToolFailure(UpstreamUnavailable(), actual_attempts=attempts)
            if response.attempts != attempts:
                raise RuntimeError("provider reported inconsistent attempts")
            try:
                hits = tuple(WebSearchHit(**asdict(item)) for item in response.results)
                base = WebSearchSuccess(
                    scope=scope,
                    results=(),
                    provider_offset=operation["offset"],
                    provider_count=20,
                    more_results_available=response.more_results_available,
                    provider_query=response.provider_query
                    or WebProviderQuery(original=None, altered=None, cleaned=None),
                    omissions=response.omissions,
                    next_cursor=None,
                    boundary="unknown",
                    provider="brave",
                    provider_request_id=response.provider_request_id,
                    observed_at=datetime.fromisoformat(
                        response.retrieved_at.replace("Z", "+00:00")
                    ),
                    evidence=WebSearchEvidence(provider_request_id=response.provider_request_id),
                )
                page = _SearchPage(
                    id=hashlib.sha256(str(context.position).encode("utf-8")).hexdigest(),
                    digest=hashlib.sha256(
                        canonical_json_bytes(
                            {
                                "base": base.model_dump(mode="json"),
                                "hits": [h.model_dump(mode="json") for h in hits],
                            }
                        )
                    ).hexdigest(),
                    base=base,
                    hits=hits,
                )
                if not _success_fits(base, context.grant.limits.max_output_bytes):
                    if (
                        context.grant.limits.max_output_bytes
                        < WEB_SEARCH_SPEC.limits.max_output_bytes
                    ):
                        raise BoundaryFailure("BudgetExceeded", actual_attempts=attempts)
                    raise ValueError("page metadata does not fit")
            except (ValueError, TypeError, UnicodeError) as exc:
                raise DeclaredToolFailure(
                    InvalidUpstreamResponse(), actual_attempts=attempts
                ) from exc
            current = page
            index = 0
        assert page is not None
        best: WebSearchSuccess | None = None
        for end in range(index, min(len(page.hits), index + value.page_size) + 1):
            if end == index and end < len(page.hits):
                continue
            if end < len(page.hits):
                next_cursor = _encode_cursor(
                    {
                        "version": 1,
                        "kind": "items",
                        "scope_digest": scope_digest,
                        "page_id": page.id,
                        "page_digest": page.digest,
                        "next_index": end,
                    }
                )
                boundary = "more"
            elif page.base.more_results_available is False:
                next_cursor = None
                boundary = "exhausted"
            elif page.base.provider_offset == 9:
                next_cursor = None
                boundary = "provider_limit"
            else:
                next_cursor = _encode_cursor(
                    {
                        "version": 1,
                        "kind": "provider",
                        "scope_digest": scope_digest,
                        "count": 20,
                        "offset": page.base.provider_offset + 1,
                    }
                )
                boundary = "more" if page.base.more_results_available is True else "unknown"
            candidate = page.base.model_copy(
                update={
                    "results": page.hits[index:end],
                    "next_cursor": next_cursor,
                    "boundary": boundary,
                }
            )
            if _success_fits(candidate, context.grant.limits.max_output_bytes):
                best = candidate
        if best is None:
            if context.grant.limits.max_output_bytes < WEB_SEARCH_SPEC.limits.max_output_bytes:
                raise BoundaryFailure("BudgetExceeded", actual_attempts=attempts)
            raise DeclaredToolFailure(InvalidUpstreamResponse(), actual_attempts=attempts)
        return HandlerSuccess(best, actual_attempts=attempts)

    return ToolBinding(
        spec=WEB_SEARCH_SPEC,
        execute=Available(execute),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision=_WEB_SEARCH_IMPLEMENTATION_REVISION,
        policy_epoch=_WEB_SEARCH_POLICY_EPOCH,
        policy_inputs=_search_policy_inputs(policy, operation_deadline_seconds),
    )


def bind_web_read(
    reader: SafeWebReader,
) -> ToolBinding[WebReadInput, WebReadSuccess, WebReadToolError]:
    current: tuple[str, WebDocument, bytes, str] | None = None

    async def execute(
        value: WebReadInput, context: ExecutionContext
    ) -> HandlerSuccess[WebReadSuccess]:
        nonlocal current
        attempts = 0
        if value.expected_digest is not None and current is not None and current[0] == value.url:
            _, document, content, digest = current
        else:
            if context.grant.limits.max_attempts == 0:
                raise DeclaredToolFailure(UpstreamUnavailable(), actual_attempts=0)
            try:
                response = await reader.read(
                    value.url,
                    max_requests=context.grant.limits.max_attempts,
                    deadline_seconds=context.grant.limits.deadline_seconds,
                )
            except WebReadFailure as exc:
                raise DeclaredToolFailure(exc.error, actual_attempts=exc.attempts) from exc
            except WebReadDeadline as exc:
                raise BoundaryFailure("DeadlineExceeded", actual_attempts=exc.attempts) from exc
            document = response.value
            attempts = response.attempts
            content = document.text.encode("utf-8")
            digest = projection_digest(
                document.source.model_dump(mode="json"),
                document.projection,
                hashlib.sha256(content).hexdigest(),
            )
            current = value.url, document, content, digest
        if value.expected_digest is not None and value.expected_digest != digest:
            raise DeclaredToolFailure(SourceChanged(), actual_attempts=attempts)
        try:
            if value.offset > len(content) or (
                value.offset < len(content) and content[value.offset] & 0xC0 == 0x80
            ):
                raise ValueError("invalid offset")

            def build(text: str, end: int, next_offset: int | None) -> JsonObject:
                locator = json.loads(document.evidence.locator)
                locator.update(
                    {
                        "projection": document.projection,
                        "offset": value.offset,
                        "end_offset": end,
                        "total_bytes": len(content),
                    }
                )
                receipt = EvidenceReceipt(
                    source_uri=document.evidence.source_uri,
                    final_uri=document.evidence.final_uri,
                    observed_at=document.evidence.observed_at,
                    content_sha256=document.evidence.content_sha256,
                    media_type=document.evidence.media_type,
                    locator=canonical_json_bytes(locator).decode("utf-8"),
                )
                return WebReadSuccess(
                    source=document.source,
                    projection=document.projection,
                    digest=digest,
                    observed_at=document.evidence.observed_at,
                    offset=value.offset,
                    end_offset=end,
                    total_bytes=len(content),
                    text=text,
                    next_offset=next_offset,
                    title=document.title,
                    evidence=receipt,
                ).model_dump(mode="json")

            window = utf8_slice(content, value.offset, value.max_bytes)
        except ValueError as exc:
            raise BoundaryFailure("InvalidInput", actual_attempts=attempts) from exc
        try:
            result = fit_text_chunk(
                window,
                offset=value.offset,
                total_bytes=len(content),
                max_bytes=value.max_bytes,
                build=build,
                max_output_bytes=context.grant.limits.max_output_bytes,
            )
        except ValueError as exc:
            raise BoundaryFailure("BudgetExceeded", actual_attempts=attempts) from exc
        return HandlerSuccess(WebReadSuccess.model_validate(result), actual_attempts=attempts)

    return ToolBinding(
        spec=WEB_READ_SPEC,
        execute=Available(execute),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision=_WEB_READ_IMPLEMENTATION_REVISION,
        policy_epoch=_WEB_READ_POLICY_EPOCH,
        policy_inputs=_WEB_READ_POLICY_INPUTS,
    )


def web_family(
    *,
    search: ToolBinding[WebSearchInput, WebSearchSuccess, WebSearchToolError] | None = None,
    read: ToolBinding[WebReadInput, WebReadSuccess, WebReadToolError] | None = None,
) -> ToolFamily:
    search_binding = search or ToolBinding(
        spec=WEB_SEARCH_SPEC,
        execute=Unavailable("Brave credential was not supplied by the host"),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision=_WEB_SEARCH_IMPLEMENTATION_REVISION,
        policy_epoch=_WEB_SEARCH_POLICY_EPOCH,
        policy_inputs=_search_policy_inputs(
            DEFAULT_WEB_SEARCH_POLICY, _WEB_SEARCH_OPERATION_DEADLINE_SECONDS
        ),
    )
    read_binding = read or ToolBinding(
        spec=WEB_READ_SPEC,
        execute=Unavailable("public Web reading is disabled by host policy"),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision=_WEB_READ_IMPLEMENTATION_REVISION,
        policy_epoch=_WEB_READ_POLICY_EPOCH,
        policy_inputs=_WEB_READ_POLICY_INPUTS,
    )
    if search_binding.spec is not WEB_SEARCH_SPEC or read_binding.spec is not WEB_READ_SPEC:
        raise ValueError("Web bindings must own published declarations")
    return ToolFamily(
        namespace="web",
        declarations=(WEB_SEARCH_SPEC, WEB_READ_SPEC),
        bindings=(search_binding, read_binding),
    )


def _success_fits(value: BaseModel, max_output_bytes: int) -> bool:
    return (
        len(canonical_json_bytes({"type": "Success", "value": value.model_dump(mode="json")}))
        <= max_output_bytes
    )

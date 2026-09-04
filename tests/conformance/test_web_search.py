"""Fixed-transcript conformance proof for the public ``web.search`` tool."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime

import httpx
import pytest

from llm_tools.catalog import ToolCatalog
from llm_tools.execution import (
    ExecutionContext,
    InvocationPosition,
    ParsedJson,
    Principal,
    RecoveryRequired,
    Scope,
    ToolExecutor,
)
from llm_tools.profiles import CapabilityProfile, Native, ProfileId, RunLimits, ToolGrant, ToolPlan
from llm_tools.schema import canonical_json_bytes
from llm_tools.testing import (
    InMemoryBudgetState,
    InMemoryPositionRecorder,
    NeverCancelled,
    RecordingTelemetry,
)
from llm_tools.web.brave import BraveSearchProvider
from llm_tools.web.contracts import (
    InvalidUpstreamResponse,
    RateLimited,
    UpstreamUnavailable,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResultType,
)
from llm_tools.web.tools import WEB_SEARCH_SPEC, bind_brave_web_search, web_family


@pytest.fixture
async def search_client() -> AsyncIterator[httpx.AsyncClient]:
    calls = 0

    async def transcript(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/res/v1/web/search"
        assert request.url.params["q"] == "portable tool architecture"
        assert request.url.params["freshness"] == "pw"
        assert request.url.params["count"] == "2"
        assert request.url.params["country"] == "us"
        assert request.url.params["search_lang"] == "en"
        assert request.url.params["safesearch"] == "moderate"
        assert request.headers["X-Subscription-Token"] == "test-key"
        assert calls == 1
        return httpx.Response(
            200,
            headers={"x-request-id": "brave-request-1"},
            json={
                "web": {
                    "results": [
                        {
                            "title": "Portable tools",
                            "url": "HTTPS://Example.COM:443/tools?q=LLM Tools#section",
                            "description": "Primary snippet",
                            "extra_snippets": ["More context", "", 7],
                            "age": "2026-08-12T10:00:00Z",
                            "profile": {"name": "Example Docs"},
                            "provider_specific": {"must": "not leak"},
                        },
                        {
                            "title": "Duplicate",
                            "url": "https://example.com/tools?q=LLM%20Tools",
                            "description": "Skipped duplicate",
                        },
                    ]
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        yield client


def _context(binding, *, tool_limits=None, position: str = "turn-1/search-1"):
    catalog = ToolCatalog.compose((web_family(search=binding),))
    run_limits = RunLimits(
        max_calls=2,
        max_external_attempts=4,
        max_input_bytes=4_096,
        max_output_bytes=65_536,
        max_in_flight=1,
        max_elapsed_seconds=30.0,
    )
    profile = CapabilityProfile(
        id=ProfileId("web-search-proof"),
        grants=(ToolGrant(id=WEB_SEARCH_SPEC.id, limits=tool_limits),),
        run_limits=run_limits,
    ).freeze(catalog)
    plan = ToolPlan(profile=profile.id, exposure=Native()).freeze(catalog, profile)
    recorder = InMemoryPositionRecorder()
    return ExecutionContext(
        plan=plan,
        grant=plan.grant(WEB_SEARCH_SPEC.id),
        catalog_view=plan.catalog_view,
        position=InvocationPosition(position),
        recorder=recorder,
        effect_id=None,
        budgets=InMemoryBudgetState(run_limits),
        principal=Principal("proof"),
        scope=Scope("public-web"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )


def test_model_visible_contract_is_small_closed_and_honest() -> None:
    assert WEB_SEARCH_SPEC.summary == "Search the public web for ranked sources and snippets."
    assert "untrusted" in WEB_SEARCH_SPEC.documentation.text
    assert "read the source" in WEB_SEARCH_SPEC.documentation.text.lower()
    assert WEB_SEARCH_SPEC.input_schema.semantic == {
        "additionalProperties": False,
        "properties": {
            "freshness_days": {"anyOf": [{"minimum": 1, "type": "integer"}, {"type": "null"}]},
            "query": {
                "maxLength": 400,
                "minLength": 2,
                "pattern": r"^\s*\S+(?:\s+\S+){0,49}\s*$",
                "type": "string",
            },
        },
        "required": ["freshness_days", "query"],
        "type": "object",
    }
    assert WEB_SEARCH_SPEC.limits.max_attempts == 2
    assert WEB_SEARCH_SPEC.limits.max_output_bytes == 32 * 1_024
    assert WEB_SEARCH_SPEC.limits.deadline_seconds == 15.0
    result_schema = WEB_SEARCH_SPEC.success_schema.semantic["properties"]["results"]
    assert result_schema["maxItems"] == 10
    assert result_schema["items"]["properties"]["rank"]["maximum"] == 10
    declared = WEB_SEARCH_SPEC.declared_error_schema
    assert declared is not None
    assert {branch["properties"]["type"]["const"] for branch in declared.semantic["anyOf"]} == {
        "RateLimited",
        "UpstreamUnavailable",
        "InvalidUpstreamResponse",
    }


@pytest.mark.asyncio
async def test_model_binding_remains_capped_at_ten_results(
    search_client: httpx.AsyncClient,
) -> None:
    provider = BraveSearchProvider(search_client, api_key="test-key")

    binding = bind_brave_web_search(provider, max_results=10)

    assert binding.policy_inputs["max_results"] == 10
    with pytest.raises(ValueError, match="between 1 and 10"):
        bind_brave_web_search(provider, max_results=11)


@pytest.mark.asyncio
async def test_operation_deadline_is_validated_and_frozen_into_policy_identity(
    search_client: httpx.AsyncClient,
) -> None:
    provider = BraveSearchProvider(search_client, api_key="test-key")

    default = bind_brave_web_search(provider)
    explicit_default = bind_brave_web_search(provider, operation_deadline_seconds=12)
    tightened = bind_brave_web_search(provider, operation_deadline_seconds=11.5)
    unavailable = web_family().bindings[0]

    assert default.policy_inputs["operation_deadline_seconds"] == 12.0
    assert unavailable.policy_inputs["operation_deadline_seconds"] == 12.0
    assert default.implementation_revision == "llm-tools-web-search-v2"
    assert default.policy_epoch == "web-search-v2"
    assert default.policy_revision == explicit_default.policy_revision
    assert default.policy_revision == unavailable.policy_revision
    assert tightened.policy_revision != default.policy_revision
    assert tightened.spec.tool_contract_revision == default.spec.tool_contract_revision
    assert tightened.implementation_revision == default.implementation_revision
    assert _context(tightened).plan.profile.profile_revision != (
        _context(default).plan.profile.profile_revision
    )
    assert _context(tightened).plan.plan_revision != _context(default).plan.plan_revision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ("12", TypeError),
        (0, ValueError),
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (12.001, ValueError),
        (WEB_SEARCH_SPEC.limits.deadline_seconds - 0.001, ValueError),
        (WEB_SEARCH_SPEC.limits.deadline_seconds, ValueError),
        (WEB_SEARCH_SPEC.limits.deadline_seconds + 0.001, ValueError),
    ],
)
async def test_operation_deadline_rejects_invalid_or_non_inner_values(
    search_client: httpx.AsyncClient,
    value: object,
    error: type[Exception],
) -> None:
    provider = BraveSearchProvider(search_client, api_key="test-key")

    with pytest.raises(error, match="operation_deadline_seconds"):
        bind_brave_web_search(
            provider,
            operation_deadline_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_operation_deadline_preserves_guard_before_executor_deadline(
    search_client: httpx.AsyncClient,
) -> None:
    deadline = WEB_SEARCH_SPEC.limits.deadline_seconds - 3.0

    binding = bind_brave_web_search(
        BraveSearchProvider(search_client, api_key="test-key"),
        operation_deadline_seconds=deadline,
    )

    assert binding.policy_inputs["operation_deadline_seconds"] == deadline


@pytest.mark.asyncio
async def test_binding_preserves_rank_identity_provenance_and_attempt_count(
    search_client: httpx.AsyncClient,
) -> None:
    binding = bind_brave_web_search(
        BraveSearchProvider(search_client, api_key="test-key"),
        max_results=2,
    )
    context = _context(binding)

    result = await ToolExecutor.execute(
        binding,
        ParsedJson({"query": "  portable   tool architecture ", "freshness_days": 7}),
        context,
    )

    assert result["type"] == "Success"
    value = result["value"]
    assert value["provider"] == "brave"
    assert value["provider_request_id"] == "brave-request-1"
    assert datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
    assert value["evidence"] == {
        "provider": "brave",
        "provider_request_id": "brave-request-1",
        "type": "web.search",
    }
    assert value["results"] == [
        {
            "display_url": "example.com/tools",
            "extra_snippets": ["More context"],
            "provider": "brave",
            "provider_request_id": "brave-request-1",
            "published_at": "2026-08-12",
            "rank": 1,
            "result_ref": value["results"][0]["result_ref"],
            "snippet": "Primary snippet",
            "source_name": "Example Docs",
            "title": "Portable tools",
            "url": "https://example.com/tools?q=LLM%20Tools",
        }
    ]
    assert value["results"][0]["result_ref"].startswith("brave:web:")
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected_error", "attempts"),
    [
        ([httpx.Response(429), httpx.Response(429)], RateLimited, 2),
        ([httpx.Response(503), httpx.Response(503)], UpstreamUnavailable, 2),
        ([httpx.Response(200, content=b"not-json")], InvalidUpstreamResponse, 1),
        (
            [httpx.Response(503), httpx.Response(200, json=[])],
            InvalidUpstreamResponse,
            2,
        ),
    ],
)
async def test_binding_maps_only_closed_safe_errors(
    responses: list[httpx.Response],
    expected_error: type,
    attempts: int,
) -> None:
    index = 0

    async def transcript(request: httpx.Request) -> httpx.Response:
        nonlocal index
        response = responses[index]
        index += 1
        return response

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        context = _context(binding)
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "safe error", "freshness_days": None}),
            context,
        )

    assert result == {"type": "Failure", "error": {"type": expected_error().type}}
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.settlement is not None
    assert record.settlement.actual_attempts == attempts


@pytest.mark.asyncio
async def test_operation_deadline_terminalizes_instead_of_requiring_owner_recovery() -> None:
    calls = 0

    async def transcript(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("an indefinitely blocked request must be cancelled")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            operation_deadline_seconds=0.05,
        )
        context = _context(binding, position="turn-1/search-operation-deadline")
        raw = ParsedJson({"query": "blocked upstream", "freshness_days": None})

        result = await ToolExecutor.execute(binding, raw, context)
        replayed = await ToolExecutor.execute(binding, raw, context)

    assert result == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}
    assert replayed is result
    assert calls == 1
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is result
    assert record.terminal_commits == 1
    assert record.uncertain is False
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1
    assert isinstance(context.budgets, InMemoryBudgetState)
    assert context.budgets.actual_external_attempts == 1
    assert context.budgets.reserved_external_attempts == 0


@pytest.mark.asyncio
async def test_operation_deadline_reports_both_started_attempts() -> None:
    calls = 0

    async def transcript(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        await asyncio.Event().wait()
        raise AssertionError("the second blocked request must be cancelled")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            operation_deadline_seconds=0.05,
        )
        context = _context(binding, position="turn-1/search-second-attempt-deadline")

        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "second attempt timeout", "freshness_days": None}),
            context,
        )

    assert result == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}
    assert calls == 2
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is result
    assert record.uncertain is False
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 2


@pytest.mark.asyncio
async def test_operation_deadline_covers_retry_backoff_with_accurate_attempts() -> None:
    calls = 0

    async def transcript(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, headers={"Retry-After": "2"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            operation_deadline_seconds=0.05,
        )
        context = _context(binding, position="turn-1/search-backoff-deadline")

        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "bounded backoff", "freshness_days": None}),
            context,
        )

    assert result == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}
    assert calls == 1
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is result
    assert record.uncertain is False
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_unexpected_outer_timeout_still_requires_billed_once_recovery() -> None:
    calls = 0

    async def transcript(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("an indefinitely blocked request must be cancelled")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            operation_deadline_seconds=0.5,
        )
        outer_first = WEB_SEARCH_SPEC.limits.tightened(deadline_seconds=0.05)
        context = _context(
            binding,
            tool_limits=outer_first,
            position="turn-1/search-unexpected-outer-timeout",
        )

        with pytest.raises(RecoveryRequired, match="uncertain"):
            await ToolExecutor.execute(
                binding,
                ParsedJson({"query": "unexpected timeout", "freshness_days": None}),
                context,
            )

    assert calls == 1
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is None
    assert record.settlement is None
    assert record.uncertain is True


@pytest.mark.asyncio
async def test_provider_timeout_still_requires_billed_once_recovery() -> None:
    class UnexpectedTimeoutProvider:
        async def search(
            self,
            request: WebSearchRequest,
            *,
            attempt_started: Callable[[], None] | None = None,
        ) -> WebSearchResponse:
            del request
            assert attempt_started is not None
            attempt_started()
            raise TimeoutError("provider violated the normalized error contract")

    binding = bind_brave_web_search(
        UnexpectedTimeoutProvider(),
        operation_deadline_seconds=0.5,
    )
    context = _context(binding, position="turn-1/search-provider-timeout")

    with pytest.raises(RecoveryRequired, match="uncertain"):
        await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "unexpected timeout", "freshness_days": None}),
            context,
        )

    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is None
    assert record.settlement is None
    assert record.uncertain is True
    assert record.in_flight is False


@pytest.mark.asyncio
async def test_external_cancellation_still_leaves_billed_once_recovery_state() -> None:
    started = asyncio.Event()
    calls = 0

    async def transcript(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("the cancelled request must not resume")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            operation_deadline_seconds=0.5,
        )
        context = _context(binding, position="turn-1/search-cancelled")
        task = asyncio.create_task(
            ToolExecutor.execute(
                binding,
                ParsedJson({"query": "cancelled search", "freshness_days": None}),
                context,
            )
        )
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == 1
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is None
    assert record.settlement is None
    assert record.uncertain is True
    assert record.in_flight is False


@pytest.mark.asyncio
async def test_expired_operation_cannot_return_success_after_suppressing_cancellation() -> None:
    class SuppressingProvider:
        async def search(
            self,
            request: WebSearchRequest,
            *,
            attempt_started: Callable[[], None] | None = None,
        ) -> WebSearchResponse:
            del request
            assert attempt_started is not None
            attempt_started()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            return WebSearchResponse(
                results=(),
                provider="brave",
                provider_request_id=None,
                retrieved_at="2026-09-04T12:00:00Z",
                attempts=1,
            )

    binding = bind_brave_web_search(
        SuppressingProvider(),
        operation_deadline_seconds=0.05,
    )
    context = _context(binding, position="turn-1/search-suppressed-timeout")

    result = await ToolExecutor.execute(
        binding,
        ParsedJson({"query": "suppressed cancellation", "freshness_days": None}),
        context,
    )

    assert result == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.terminal_result is result
    assert record.uncertain is False
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_profile_tightened_attempt_limit_reaches_brave() -> None:
    calls = 0

    async def transcript(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        catalog = ToolCatalog.compose((web_family(search=binding),))
        run_limits = RunLimits(2, 4, 4_096, 65_536, 1, 30.0)
        tightened = WEB_SEARCH_SPEC.limits.tightened(max_attempts=1)
        profile = CapabilityProfile(
            id=ProfileId("one-attempt"),
            grants=(ToolGrant(id=WEB_SEARCH_SPEC.id, limits=tightened),),
            run_limits=run_limits,
        ).freeze(catalog)
        plan = ToolPlan(profile=profile.id, exposure=Native()).freeze(catalog, profile)
        context = ExecutionContext(
            plan=plan,
            grant=plan.grant(WEB_SEARCH_SPEC.id),
            catalog_view=plan.catalog_view,
            position=InvocationPosition("turn-1/one-attempt"),
            recorder=InMemoryPositionRecorder(),
            effect_id=None,
            budgets=InMemoryBudgetState(run_limits),
            principal=Principal("proof"),
            scope=Scope("public-web"),
            cancellation=NeverCancelled(),
            telemetry=RecordingTelemetry(),
        )
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "one attempt", "freshness_days": None}),
            context,
        )

    assert result == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}
    assert calls == 1
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    settlement = context.recorder.record(context.position).settlement
    assert settlement is not None
    assert settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_effective_output_limit_deterministically_truncates_search_results() -> None:
    calls = 0

    async def transcript(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"x-request-id": "bounded-request"},
            json={
                "web": {
                    "results": [
                        {
                            "title": "First result",
                            "url": "https://example.com/first",
                            "description": "first snippet " * 20,
                        },
                        {
                            "title": "Second result",
                            "url": "https://example.com/second",
                            "description": "second snippet " * 20,
                        },
                    ]
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key="test-key"),
            max_results=2,
        )
        full = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "bounded output", "freshness_days": None}),
            _context(binding, position="turn-1/search-full"),
        )
        assert full["type"] == "Success"
        one_result_value = {**full["value"], "results": full["value"]["results"][:1]}
        one_result_bytes = len(canonical_json_bytes({"type": "Success", "value": one_result_value}))
        tightened = WEB_SEARCH_SPEC.limits.tightened(max_output_bytes=one_result_bytes)
        bounded_context = _context(
            binding,
            tool_limits=tightened,
            position="turn-1/search-bounded",
        )
        bounded = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "bounded output", "freshness_days": None}),
            bounded_context,
        )

        too_small = WEB_SEARCH_SPEC.limits.tightened(max_output_bytes=54)
        failure_context = _context(
            binding,
            tool_limits=too_small,
            position="turn-1/search-minimal-too-large",
        )
        failure = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "bounded output", "freshness_days": None}),
            failure_context,
        )

    assert bounded["type"] == "Success"
    assert bounded["value"]["results"] == full["value"]["results"][:1]
    assert len(canonical_json_bytes(bounded)) == one_result_bytes
    assert failure == {"type": "Failure", "error": {"type": "BudgetExceeded"}}
    assert calls == 3
    assert isinstance(failure_context.recorder, InMemoryPositionRecorder)
    settlement = failure_context.recorder.record(failure_context.position).settlement
    assert settlement is not None
    assert settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_oversized_provider_result_url_is_dropped_before_the_public_binding() -> None:
    async def transcript(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Oversized",
                            "url": "https://example.com/" + "x" * 5_000,
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        context = _context(binding, position="turn-1/search-oversized-url")
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "oversized result", "freshness_days": None}),
            context,
        )

    assert result["type"] == "Success"
    assert result["value"]["results"] == []
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.uncertain is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_malformed_provider_item_terminalizes_the_billed_call() -> None:
    async def transcript(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Malformed",
                            "url": "https://example.com/",
                            "extra_snippets": {"not": "an array"},
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        context = _context(binding, position="turn-1/search-malformed-item")
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "malformed result", "freshness_days": None}),
            context,
        )

    assert result == {
        "type": "Failure",
        "error": {"type": "InvalidUpstreamResponse"},
    }
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.uncertain is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_deep_provider_json_terminalizes_the_billed_call() -> None:
    body = (b"[" * 10_000) + b"0" + (b"]" * 10_000)

    async def transcript(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        context = _context(binding, position="turn-1/search-deep-json")
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "deep response", "freshness_days": None}),
            context,
        )

    assert result == {
        "type": "Failure",
        "error": {"type": "InvalidUpstreamResponse"},
    }
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.uncertain is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_invalid_provider_content_encoding_terminalizes_the_billed_call() -> None:
    async def transcript(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            content=b"not gzip",
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(transcript), trust_env=False
    ) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key="test-key"))
        context = _context(binding, position="turn-1/search-invalid-encoding")
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"query": "invalid encoding", "freshness_days": None}),
            context,
        )

    assert result == {
        "type": "Failure",
        "error": {"type": "InvalidUpstreamResponse"},
    }
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.uncertain is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


def test_programmatic_request_keeps_broader_host_policy_surface() -> None:
    request = WebSearchRequest(
        query="  Brave   Search  ",
        result_type=WebSearchResultType.WEB,
        limit=3,
        country="us",
        search_lang="EN",
        allowed_domains=("https://Example.com/path",),
        blocked_domains=("spam.example:443",),
    )

    assert request.query == "Brave Search"
    assert request.result_type is WebSearchResultType.WEB
    assert request.allowed_domains == ("example.com",)
    assert request.blocked_domains == ("spam.example",)


@pytest.mark.asyncio
async def test_empty_brave_credential_is_a_composition_defect() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        with pytest.raises(ValueError, match="must not be empty"):
            BraveSearchProvider(client, api_key="  ")

"""Temporary native contract acceptance through the real executor."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict

import llm_tools as tools
from llm_tools.testing import InMemoryPositionRecorder, NeverCancelled, RecordingTelemetry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class EchoSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def _limits() -> tools.RunLimits:
    return tools.RunLimits(None, None, None, None, 1, None)


def _plan(handler, limits: tools.RunLimits, deadline: float = 1.0):
    spec = tools.ToolSpec(
        id=tools.ToolId("acceptance.echo"),
        summary="Return the supplied value",
        documentation=tools.PromptDocument("Return the supplied value."),
        input_type=EchoInput,
        success_type=EchoSuccess,
        error_type=tools.NoDeclaredError,
        effect=tools.ToolEffect.Read,
        limits=tools.ToolLimits(1_024, 1_024, 1, deadline),
    )
    binding = tools.ToolBinding(
        spec=spec,
        execute=tools.Available(handler),
        replay_policy=tools.ReplayPolicy.ReDispatchable,
        implementation_revision="acceptance-v1",
        policy_epoch=tools.PolicyEpoch("acceptance-v1"),
        policy_inputs={},
    )
    catalog = tools.ToolCatalog.compose((tools.ToolFamily("acceptance", (spec,), (binding,)),))
    profile = tools.CapabilityProfile(
        tools.ProfileId("acceptance"), (tools.ToolGrant(spec.id, None),), limits
    ).freeze(catalog)
    return binding, catalog, tools.ToolPlan(profile.id, tools.Native()).freeze(catalog, profile)


def _context(plan, budgets, recorder, position: str):
    return tools.ExecutionContext(
        plan=plan,
        grant=plan.profile.ordered_grants[0],
        catalog_view=plan.catalog_view,
        position=tools.InvocationPosition(position),
        recorder=recorder,
        effect_id=None,
        budgets=budgets,
        principal=tools.Principal("acceptance"),
        scope=tools.Scope("acceptance"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )


async def test_native_calls_continue_and_terminal_replay_never_dispatches_or_charges_again():
    calls = 0

    async def handler(value, context):
        nonlocal calls
        calls += 1
        return tools.HandlerSuccess(EchoSuccess(value=value.value), actual_attempts=1)

    limits = _limits()
    binding, _, plan = _plan(handler, limits)
    budgets = tools.RunBudgetState(limits)
    recorder = InMemoryPositionRecorder(durable=False)
    results = []
    for ordinal in range(20):
        results.append(
            await tools.ToolExecutor.execute(
                binding,
                tools.ParsedJson({"value": str(ordinal)}),
                _context(plan, budgets, recorder, f"native/{ordinal}"),
            )
        )
    assert all(result["type"] == "Success" for result in results)
    assert (
        await tools.ToolExecutor.execute(
            binding,
            tools.ParsedJson({"value": "0"}),
            _context(plan, budgets, recorder, "native/0"),
        )
        == results[0]
    )
    assert calls == budgets.actual_calls == budgets.actual_external_attempts == 20
    assert budgets.reserved_external_attempts == budgets.reserved_output_bytes == 0
    assert budgets.remaining_elapsed_seconds is None


async def test_per_tool_deadline_still_ends_a_stuck_handler_without_a_run_deadline():
    entered = asyncio.Event()

    async def handler(value, context):
        entered.set()
        await asyncio.Event().wait()

    limits = _limits()
    binding, _, plan = _plan(handler, limits, deadline=0.02)
    budgets = tools.RunBudgetState(limits)
    recorder = InMemoryPositionRecorder(durable=False)
    result = await asyncio.wait_for(
        tools.ToolExecutor.execute(
            binding,
            tools.ParsedJson({"value": "stuck"}),
            _context(plan, budgets, recorder, "native/stuck"),
        ),
        timeout=1.0,
    )
    assert entered.is_set()
    assert result == {"type": "Failure", "error": {"type": "DeadlineExceeded"}}
    assert budgets.actual_calls == budgets.actual_external_attempts == 1
    assert budgets.reserved_external_attempts == 0


async def test_in_flight_stays_finite_and_duplicates_precede_capacity_check():
    budgets = tools.RunBudgetState(_limits())
    reservation = tools.Reservation(1, 20, 2, 100)
    first = tools.InvocationPosition("native/first")
    second = tools.InvocationPosition("native/second")
    assert await budgets.reserve(first, reservation)
    assert await budgets.reserve(first, reservation)
    assert not await budgets.reserve(second, reservation)
    with pytest.raises(ValueError):
        await budgets.reserve(first, replace(reservation, input_bytes=21))
    assert budgets.actual_calls == 1
    assert budgets.reserved_external_attempts == 2
    await budgets.settle(first, tools.Settlement(1, 30))
    await budgets.settle(first, tools.Settlement(1, 30))
    assert await budgets.reserve(second, reservation)
    assert budgets.actual_calls == 2
    assert budgets.actual_external_attempts == 1
    assert budgets.reserved_external_attempts == 2


async def test_finite_job_budget_is_still_enforced_by_the_real_executor():
    calls = 0

    async def handler(value, context):
        nonlocal calls
        calls += 1
        return tools.HandlerSuccess(EchoSuccess(value=value.value), actual_attempts=1)

    limits = tools.RunLimits(1, 1, 1_024, 1_024, 1, 5.0)
    binding, _, plan = _plan(handler, limits)
    budgets = tools.RunBudgetState(limits)
    recorder = InMemoryPositionRecorder(durable=False)
    assert (
        await tools.ToolExecutor.execute(
            binding,
            tools.ParsedJson({"value": "first"}),
            _context(plan, budgets, recorder, "job/first"),
        )
    )["type"] == "Success"
    assert await tools.ToolExecutor.execute(
        binding,
        tools.ParsedJson({"value": "second"}),
        _context(plan, budgets, recorder, "job/second"),
    ) == {"type": "Failure", "error": {"type": "BudgetExceeded"}}
    assert calls == budgets.actual_calls == 1


def test_null_limits_are_canonical_and_authority_tightening_is_directional():
    async def handler(value, context):
        return tools.HandlerSuccess(EchoSuccess(value=value.value), actual_attempts=0)

    _, catalog, unlimited = _plan(handler, _limits())
    assert tools.canonical_json_bytes(unlimited.profile.run_limits.json()) == (
        b'{"max_calls":null,"max_elapsed_seconds":null,"max_external_attempts":null,'
        b'"max_in_flight":1,"max_input_bytes":null,"max_output_bytes":null}'
    )
    limited = tools.CapabilityProfile(
        tools.ProfileId("limited"),
        (tools.ToolGrant(unlimited.profile.ordered_grants[0].id, None),),
        tools.RunLimits(2, 2, 2_048, 2_048, 1, 5.0),
    ).freeze(catalog)
    assert limited.is_tightening_of(unlimited.profile)
    assert unlimited.profile.is_tightening_of(unlimited.profile)
    assert not unlimited.profile.is_tightening_of(limited)


@pytest.mark.parametrize("value", [True, 0.5, -1])
def test_invalid_settlement_never_changes_budget(value):
    with pytest.raises((TypeError, ValueError)):
        tools.Settlement(value, 0)


def test_pure_reservation_predicate_counts_reserved_maxima():
    finite = tools.RunLimits(3, 5, 100, 200, 2, 5.0)
    totals = tools.BudgetTotals(1, 20, 4, 100, 1)
    assert tools.can_reserve(finite, totals, tools.Reservation(1, 20, 1, 100))
    assert not tools.can_reserve(finite, totals, tools.Reservation(1, 20, 2, 100))
    assert not tools.can_reserve(finite, totals, tools.Reservation(1, 20, 1, 101))
    assert not tools.can_reserve(
        finite, replace(totals, in_flight=2), tools.Reservation(1, 1, 0, 1)
    )

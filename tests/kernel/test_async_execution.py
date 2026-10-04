"""Async recorder and budget contract qualification."""

from __future__ import annotations

import asyncio
import inspect
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from llm_tools.budgets import RunBudgetState
from llm_tools.catalog import ToolCatalog, ToolFamily
from llm_tools.declaration import (
    Available,
    NoDeclaredError,
    PolicyEpoch,
    PromptDocument,
    ReplayPolicy,
    ToolBinding,
    ToolEffect,
    ToolId,
    ToolLimits,
    ToolSpec,
)
from llm_tools.execution import (
    BudgetState,
    ExecutionContext,
    HandlerSuccess,
    InvocationPosition,
    ParsedJson,
    PositionRecorder,
    Principal,
    Reservation,
    Scope,
    Settlement,
    ToolExecutor,
)
from llm_tools.profiles import (
    CapabilityProfile,
    Native,
    ProfileId,
    RunLimits,
    ToolGrant,
    ToolPlan,
)
from llm_tools.testing import (
    InMemoryPositionRecorder,
    NeverCancelled,
    RecordingTelemetry,
)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Annotated[str, Field(min_length=1, max_length=20)]


class Success(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


RUN_LIMITS = RunLimits(
    max_calls=1,
    max_external_attempts=1,
    max_input_bytes=1_024,
    max_output_bytes=1_024,
    max_in_flight=1,
    max_elapsed_seconds=5.0,
)


def test_recorder_and_budget_mutation_contracts_are_async_only() -> None:
    for method in (
        "reserve",
        "settle",
    ):
        assert inspect.iscoroutinefunction(getattr(BudgetState, method))
    for method in (
        "occupy",
        "reserve",
        "dispatch_started",
        "dispatch_abandoned",
        "uncertain",
        "terminalize_and_settle",
    ):
        assert inspect.iscoroutinefunction(getattr(PositionRecorder, method))


async def test_executor_awaits_recorder_budget_and_handler_boundaries() -> None:
    trace: list[str] = []

    class YieldingBudgetState(RunBudgetState):
        async def reserve(
            self,
            position: InvocationPosition,
            reservation: Reservation,
        ) -> bool:
            trace.append("budget.reserve.enter")
            await asyncio.sleep(0)
            accepted = await super().reserve(position, reservation)
            trace.append("budget.reserve.exit")
            return accepted

        async def settle(
            self,
            position: InvocationPosition,
            settlement: Settlement,
        ) -> None:
            trace.append("budget.settle.enter")
            await asyncio.sleep(0)
            await super().settle(position, settlement)
            trace.append("budget.settle.exit")

    class YieldingPositionRecorder(InMemoryPositionRecorder):
        async def occupy(self, **kwargs):  # type: ignore[no-untyped-def]
            trace.append("recorder.occupy.enter")
            await asyncio.sleep(0)
            state = await super().occupy(**kwargs)
            trace.append("recorder.occupy.exit")
            return state

        async def reserve(self, **kwargs):  # type: ignore[no-untyped-def]
            trace.append("recorder.reserve.enter")
            await asyncio.sleep(0)
            accepted = await super().reserve(**kwargs)
            trace.append("recorder.reserve.exit")
            return accepted

        async def dispatch_started(self, **kwargs):  # type: ignore[no-untyped-def]
            trace.append("recorder.dispatch_started.enter")
            await asyncio.sleep(0)
            state = await super().dispatch_started(**kwargs)
            trace.append("recorder.dispatch_started.exit")
            return state

        async def terminalize_and_settle(self, **kwargs):  # type: ignore[no-untyped-def]
            trace.append("recorder.terminalize_and_settle.enter")
            await asyncio.sleep(0)
            result = await super().terminalize_and_settle(**kwargs)
            trace.append("recorder.terminalize_and_settle.exit")
            return result

    spec = ToolSpec(
        id=ToolId("test.async_echo"),
        summary="Echo one value",
        documentation=PromptDocument("Echo the validated value."),
        input_type=Input,
        success_type=Success,
        error_type=NoDeclaredError,
        effect=ToolEffect.Read,
        limits=ToolLimits(
            max_input_bytes=1_024,
            max_output_bytes=1_024,
            max_attempts=1,
            deadline_seconds=1.0,
        ),
    )

    async def handler(value: Input, context: ExecutionContext) -> HandlerSuccess[Success]:
        del context
        trace.append("handler.enter")
        await asyncio.sleep(0)
        trace.append("handler.exit")
        return HandlerSuccess(Success(value=value.value), actual_attempts=1)

    binding = ToolBinding(
        spec=spec,
        execute=Available(handler),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision="test-async-v1",
        policy_epoch=PolicyEpoch("v1"),
        policy_inputs={},
    )
    catalog = ToolCatalog.compose((ToolFamily("test", (spec,), (binding,)),))
    profile = CapabilityProfile(
        ProfileId("test"),
        (ToolGrant(spec.id, None),),
        RUN_LIMITS,
    ).freeze(catalog)
    plan = ToolPlan(profile.id, Native()).freeze(catalog, profile)
    budgets = YieldingBudgetState(RUN_LIMITS)
    recorder = YieldingPositionRecorder()
    position = InvocationPosition("turn-1/call-1")
    context = ExecutionContext(
        plan=plan,
        grant=plan.grant(spec.id),
        catalog_view=plan.catalog_view,
        position=position,
        recorder=recorder,
        effect_id=None,
        budgets=budgets,
        principal=Principal("test"),
        scope=Scope("test"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )

    assert await ToolExecutor.execute(binding, ParsedJson({"value": "hello"}), context) == {
        "type": "Success",
        "value": {"value": "hello"},
    }
    assert trace == [
        "recorder.occupy.enter",
        "recorder.occupy.exit",
        "recorder.reserve.enter",
        "budget.reserve.enter",
        "budget.reserve.exit",
        "recorder.reserve.exit",
        "recorder.dispatch_started.enter",
        "recorder.dispatch_started.exit",
        "handler.enter",
        "handler.exit",
        "recorder.terminalize_and_settle.enter",
        "budget.settle.enter",
        "budget.settle.exit",
        "recorder.terminalize_and_settle.exit",
    ]
    record = recorder.record(position)
    assert record.terminal_result == {"type": "Success", "value": {"value": "hello"}}
    assert record.terminal_commits == 1
    assert budgets.actual_external_attempts == 1

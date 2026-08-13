"""Strict tool invocation validation, replay, accounting, and envelopes."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Protocol

from llm_tools.declaration import (
    NoDeclaredError,
    ReplayPolicy,
    ToolBinding,
    ToolEffect,
    ToolId,
    Unavailable,
)
from llm_tools.profiles import EffectiveToolGrant, FrozenToolPlan, PlanCatalogView, RunLimits
from llm_tools.schema import (
    JsonObject,
    JsonValue,
    SchemaDecodeError,
    canonical_json_bytes,
    strict_decode,
    strict_encode,
)


class InvocationPosition(str):
    def __new__(cls, value: str) -> InvocationPosition:
        if not value:
            raise ValueError("invocation position must not be empty")
        return str.__new__(cls, value)


class EffectId(str):
    def __new__(cls, value: str) -> EffectId:
        if not value:
            raise ValueError("effect id must not be empty")
        return str.__new__(cls, value)


class Principal(str):
    pass


class Scope(str):
    pass


@dataclass(frozen=True, slots=True)
class ParsedJson:
    value: JsonValue


@dataclass(frozen=True, slots=True)
class MalformedJson:
    raw_utf8: bytes


type RawToolInput = ParsedJson | MalformedJson
type ToolResult = JsonObject


@dataclass(frozen=True, slots=True)
class Reservation:
    calls: int
    input_bytes: int
    max_attempts: int
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class Settlement:
    actual_attempts: int
    actual_output_bytes: int


@dataclass(frozen=True, slots=True)
class PositionState:
    terminal_result: ToolResult | None
    uncertain: bool
    actual_attempts: int = 0


class BudgetState(Protocol):
    @property
    def limits(self) -> RunLimits: ...

    @property
    def remaining_elapsed_seconds(self) -> float: ...

    def reserve(self, position: InvocationPosition, reservation: Reservation) -> bool: ...

    def settle(self, position: InvocationPosition, settlement: Settlement) -> None: ...


class PositionRecorder(Protocol):
    @property
    def durable(self) -> bool: ...

    def occupy(
        self,
        *,
        position: InvocationPosition,
        tool_id: ToolId,
        tool_contract_revision: str,
        policy_revision: str,
        plan_revision: str,
        input_digest: str,
        replay_policy: ReplayPolicy,
    ) -> PositionState: ...

    def reserve(
        self,
        *,
        position: InvocationPosition,
        budgets: BudgetState,
        reservation: Reservation,
    ) -> bool: ...

    def dispatch_started(
        self,
        *,
        position: InvocationPosition,
        replay_policy: ReplayPolicy,
    ) -> PositionState: ...

    def dispatch_abandoned(
        self,
        *,
        position: InvocationPosition,
        replay_policy: ReplayPolicy,
        actual_attempts: int,
        lease_recovered: bool,
    ) -> None:
        """Re-admit verified abandoned ReDispatchable work outside normal execution."""
        ...

    def uncertain(self, *, position: InvocationPosition) -> None: ...

    def terminalize_and_settle(
        self,
        *,
        position: InvocationPosition,
        budgets: BudgetState,
        result: ToolResult,
        settlement: Settlement,
    ) -> ToolResult: ...


class Cancellation(Protocol):
    @property
    def cancelled(self) -> bool: ...


class Telemetry(Protocol):
    def event(self, name: str, attributes: JsonObject) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    plan: FrozenToolPlan
    grant: EffectiveToolGrant
    catalog_view: PlanCatalogView
    position: InvocationPosition
    recorder: PositionRecorder
    effect_id: EffectId | None
    budgets: BudgetState
    principal: Principal
    scope: Scope
    cancellation: Cancellation
    telemetry: Telemetry


@dataclass(frozen=True, slots=True)
class HandlerSuccess[SuccessT]:
    value: SuccessT
    actual_attempts: int


class DeclaredToolFailure(Exception):
    def __init__(self, error: object, *, actual_attempts: int) -> None:
        self.error = error
        self.actual_attempts = actual_attempts
        super().__init__("handler returned its declared failure")


class BoundaryFailure(Exception):
    """Private signal reserved for library-owned boundary bindings."""

    def __init__(self, error_type: str, *, actual_attempts: int = 0) -> None:
        if error_type not in {
            "InvalidInput",
            "ToolUnavailable",
            "BudgetExceeded",
            "DeadlineExceeded",
        }:
            raise ValueError("boundary failure must use an executor-owned error type")
        if isinstance(actual_attempts, bool) or not isinstance(actual_attempts, int):
            raise TypeError("boundary failure attempt count must be an integer")
        if actual_attempts < 0:
            raise ValueError("boundary failure attempt count must not be negative")
        self.error_type = error_type
        self.actual_attempts = actual_attempts
        super().__init__(error_type)


class PositionConflictDefect(RuntimeError):
    pass


class ExecutorConfigurationDefect(RuntimeError):
    pass


class RecoveryRequired(RuntimeError):
    pass


def raw_input_digest(raw_input: RawToolInput) -> str:
    if isinstance(raw_input, ParsedJson):
        envelope: JsonObject = {"type": "ParsedJson", "value": raw_input.value}
    else:
        envelope = {
            "bytes": len(raw_input.raw_utf8),
            "raw_sha256": hashlib.sha256(raw_input.raw_utf8).hexdigest(),
            "type": "MalformedJson",
        }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


class ToolExecutor:
    @staticmethod
    async def execute[InputT, SuccessT, ErrorT](
        binding: ToolBinding[InputT, SuccessT, ErrorT],
        raw_input: RawToolInput,
        context: ExecutionContext,
    ) -> ToolResult:
        _verify_context(binding, context)
        digest = raw_input_digest(raw_input)
        try:
            position = context.recorder.occupy(
                position=context.position,
                tool_id=binding.spec.id,
                tool_contract_revision=binding.spec.tool_contract_revision,
                policy_revision=binding.policy_revision,
                plan_revision=context.plan.plan_revision,
                input_digest=digest,
                replay_policy=binding.replay_policy,
            )
        except ValueError as exc:
            raise PositionConflictDefect("occupied position has a different invocation") from exc
        if position.terminal_result is not None:
            return position.terminal_result
        if position.uncertain:
            raise RecoveryRequired("tool position has an uncertain outcome")

        raw_bytes = _raw_envelope_bytes(raw_input)
        limits = context.grant.limits
        reservation = Reservation(
            calls=1,
            input_bytes=len(raw_bytes),
            max_attempts=limits.max_attempts,
            max_output_bytes=limits.max_output_bytes,
        )
        try:
            reserved = context.recorder.reserve(
                position=context.position,
                budgets=context.budgets,
                reservation=reservation,
            )
        except ValueError as exc:
            raise PositionConflictDefect("position budget reservation conflicts") from exc
        if not reserved:
            return _terminalize_boundary("BudgetExceeded", context)

        if len(raw_bytes) > limits.max_input_bytes:
            return _terminalize_boundary("BudgetExceeded", context)

        if isinstance(raw_input, MalformedJson):
            return _terminalize_boundary("InvalidInput", context)
        try:
            decoded = strict_decode(
                binding.spec.input_type,
                binding.spec.input_schema,
                raw_input.value,
            )
        except SchemaDecodeError:
            return _terminalize_boundary("InvalidInput", context)

        if isinstance(binding.execute, Unavailable):
            context.telemetry.event(
                "tool.unavailable",
                {"tool_id": str(binding.spec.id)},
            )
            return _terminalize_boundary(
                "ToolUnavailable",
                context,
                actual_attempts=position.actual_attempts,
            )
        remaining_elapsed = context.budgets.remaining_elapsed_seconds
        if not math.isfinite(remaining_elapsed):
            raise ExecutorConfigurationDefect("run budget returned a non-finite deadline")
        if context.cancellation.cancelled or remaining_elapsed <= 0:
            return _terminalize_boundary(
                "DeadlineExceeded",
                context,
                actual_attempts=position.actual_attempts,
            )
        if binding.spec.effect is ToolEffect.Write and (
            context.effect_id is None or not context.recorder.durable
        ):
            raise ExecutorConfigurationDefect(
                "Write tool execution requires a stable effect id and durable recorder"
            )

        if position.actual_attempts > 0 and position.actual_attempts >= limits.max_attempts:
            return _terminalize_boundary(
                "BudgetExceeded",
                context,
                actual_attempts=position.actual_attempts,
            )

        dispatch = context.recorder.dispatch_started(
            position=context.position,
            replay_policy=binding.replay_policy,
        )
        if dispatch.terminal_result is not None:
            return dispatch.terminal_result
        if dispatch.uncertain:
            raise RecoveryRequired("tool position has an uncertain outcome")
        prior_attempts = dispatch.actual_attempts
        if prior_attempts != position.actual_attempts:
            return _raise_dispatch_defect(
                binding,
                context,
                RuntimeError("position attempt accounting changed before dispatch"),
            )
        remaining_attempts = limits.max_attempts - prior_attempts
        dispatch_context = replace(
            context,
            grant=replace(
                context.grant,
                limits=context.grant.limits.tightened(max_attempts=remaining_attempts),
            ),
        )
        try:
            async with asyncio.timeout(min(limits.deadline_seconds, remaining_elapsed)):
                outcome = await binding.execute.handler(decoded, dispatch_context)
        except BoundaryFailure as failure:
            try:
                return _terminalize_boundary(
                    failure.error_type,
                    context,
                    actual_attempts=prior_attempts + failure.actual_attempts,
                )
            except Exception as exc:
                return _raise_dispatch_defect(binding, context, exc)
        except DeclaredToolFailure as failure:
            try:
                if binding.spec.error_type is NoDeclaredError:
                    raise RuntimeError("tool without declared errors raised a declared failure")
                assert binding.spec.declared_error_schema is not None
                encoded_error = strict_encode(
                    binding.spec.error_type,
                    binding.spec.declared_error_schema,
                    failure.error,
                )
                result = _failure_result(encoded_error)
                return _terminalize_result(
                    result,
                    prior_attempts + failure.actual_attempts,
                    context,
                )
            except Exception as exc:
                return _raise_dispatch_defect(binding, context, exc)
        except TimeoutError as exc:
            if binding.replay_policy is ReplayPolicy.BilledOnce:
                context.recorder.uncertain(position=context.position)
                raise RecoveryRequired("BilledOnce tool outcome is uncertain") from exc
            return _terminalize_boundary(
                "DeadlineExceeded",
                context,
                actual_attempts=prior_attempts + remaining_attempts,
            )
        except asyncio.CancelledError:
            if binding.replay_policy is ReplayPolicy.BilledOnce:
                context.recorder.uncertain(position=context.position)
            raise
        except Exception as exc:
            return _raise_dispatch_defect(binding, context, exc)

        try:
            if not isinstance(outcome, HandlerSuccess):
                raise RuntimeError("tool handler returned an invalid private outcome")
            encoded_success = strict_encode(
                binding.spec.success_type,
                binding.spec.success_schema,
                outcome.value,
            )
            result: ToolResult = {"type": "Success", "value": encoded_success}
            return _terminalize_result(
                result,
                prior_attempts + outcome.actual_attempts,
                context,
            )
        except Exception as exc:
            return _raise_dispatch_defect(binding, context, exc)


def _verify_context(
    binding: ToolBinding[object, object, object],
    context: ExecutionContext,
) -> None:
    if context.catalog_view is not context.plan.catalog_view:
        raise ExecutorConfigurationDefect("context must use the plan-owned catalogue view")
    if context.budgets.limits != context.plan.profile.run_limits:
        raise ExecutorConfigurationDefect("budget limits differ from the frozen plan")
    try:
        planned_binding = context.catalog_view.binding(binding.spec.id)
        planned_spec = context.catalog_view.spec(binding.spec.id)
        planned_grant = context.plan.grant(binding.spec.id)
    except KeyError as exc:
        raise ExecutorConfigurationDefect("binding is absent from the frozen plan") from exc
    if planned_binding is not binding:
        raise ExecutorConfigurationDefect("execution requires the exact binding frozen in the plan")
    if (
        planned_binding.policy_revision != binding.policy_revision
        or planned_spec.tool_contract_revision != binding.spec.tool_contract_revision
        or context.grant != planned_grant
        or context.grant.tool_contract_revision != binding.spec.tool_contract_revision
        or context.grant.policy_revision != binding.policy_revision
    ):
        raise ExecutorConfigurationDefect("binding, grant, or revision differs from frozen plan")


def _raw_envelope_bytes(raw_input: RawToolInput) -> bytes:
    if isinstance(raw_input, ParsedJson):
        return canonical_json_bytes({"type": "ParsedJson", "value": raw_input.value})
    return canonical_json_bytes(
        {
            "bytes": len(raw_input.raw_utf8),
            "raw_sha256": hashlib.sha256(raw_input.raw_utf8).hexdigest(),
            "type": "MalformedJson",
        }
    )


def _failure_result(error: JsonValue) -> ToolResult:
    if not isinstance(error, dict):
        raise RuntimeError("encoded error must be an object")
    return {"type": "Failure", "error": error}


def _terminalize_boundary(
    error_type: str,
    context: ExecutionContext,
    *,
    actual_attempts: int = 0,
) -> ToolResult:
    return _terminalize_result(
        {"type": "Failure", "error": {"type": error_type}},
        actual_attempts,
        context,
    )


def _raise_dispatch_defect(
    binding: ToolBinding[object, object, object],
    context: ExecutionContext,
    defect: BaseException,
) -> ToolResult:
    if binding.replay_policy is ReplayPolicy.BilledOnce:
        context.recorder.uncertain(position=context.position)
        raise RecoveryRequired("BilledOnce tool outcome is uncertain") from defect
    raise defect


def _terminalize_result(
    result: ToolResult,
    actual_attempts: int,
    context: ExecutionContext,
) -> ToolResult:
    if (
        isinstance(actual_attempts, bool)
        or not isinstance(actual_attempts, int)
        or not 0 <= actual_attempts <= context.grant.limits.max_attempts
    ):
        raise RuntimeError("handler reported invalid actual attempt accounting")
    output_bytes = len(canonical_json_bytes(result))
    if output_bytes > context.grant.limits.max_output_bytes:
        raise RuntimeError("owned result exceeds its declared output limit")
    try:
        return context.recorder.terminalize_and_settle(
            position=context.position,
            budgets=context.budgets,
            result=result,
            settlement=Settlement(
                actual_attempts=actual_attempts,
                actual_output_bytes=output_bytes,
            ),
        )
    except ValueError as exc:
        raise PositionConflictDefect("terminal position commit conflicts") from exc

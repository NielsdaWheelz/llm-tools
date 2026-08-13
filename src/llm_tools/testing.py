"""Small deterministic doubles for kernel and consumer conformance proofs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from llm_tools.declaration import ReplayPolicy, ToolId
from llm_tools.execution import (
    BudgetState,
    InvocationPosition,
    PositionState,
    Reservation,
    Settlement,
    ToolResult,
)
from llm_tools.profiles import RunLimits
from llm_tools.schema import JsonObject


@dataclass(slots=True)
class _Spend:
    reservation: Reservation
    settlement: Settlement | None = None


class InMemoryBudgetState:
    def __init__(
        self,
        limits: RunLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
    ) -> None:
        self._limits = limits
        self._spend: dict[InvocationPosition, _Spend] = {}
        self._monotonic = monotonic
        self._started_at = monotonic() if started_at is None else started_at

    @property
    def limits(self) -> RunLimits:
        return self._limits

    @property
    def remaining_elapsed_seconds(self) -> float:
        return self._limits.max_elapsed_seconds - (self._monotonic() - self._started_at)

    @property
    def actual_external_attempts(self) -> int:
        return sum(
            spend.settlement.actual_attempts
            for spend in self._spend.values()
            if spend.settlement is not None
        )

    @property
    def actual_calls(self) -> int:
        return sum(spend.reservation.calls for spend in self._spend.values())

    @property
    def actual_input_bytes(self) -> int:
        return sum(spend.reservation.input_bytes for spend in self._spend.values())

    @property
    def actual_output_bytes(self) -> int:
        return sum(
            spend.settlement.actual_output_bytes
            for spend in self._spend.values()
            if spend.settlement is not None
        )

    @property
    def reserved_external_attempts(self) -> int:
        return sum(
            spend.reservation.max_attempts
            for spend in self._spend.values()
            if spend.settlement is None
        )

    @property
    def reserved_output_bytes(self) -> int:
        return sum(
            spend.reservation.max_output_bytes
            for spend in self._spend.values()
            if spend.settlement is None
        )

    def reserve(self, position: InvocationPosition, reservation: Reservation) -> bool:
        existing = self._spend.get(position)
        if existing is not None:
            if existing.reservation != reservation:
                raise ValueError("position budget reservation differs from durable spend")
            return True
        calls = sum(spend.reservation.calls for spend in self._spend.values())
        input_bytes = sum(spend.reservation.input_bytes for spend in self._spend.values())
        attempts = sum(
            spend.settlement.actual_attempts
            if spend.settlement is not None
            else spend.reservation.max_attempts
            for spend in self._spend.values()
        )
        output_bytes = sum(
            spend.settlement.actual_output_bytes
            if spend.settlement is not None
            else spend.reservation.max_output_bytes
            for spend in self._spend.values()
        )
        if (
            calls + reservation.calls > self._limits.max_calls
            or input_bytes + reservation.input_bytes > self._limits.max_input_bytes
            or attempts + reservation.max_attempts > self._limits.max_external_attempts
            or output_bytes + reservation.max_output_bytes > self._limits.max_output_bytes
            or sum(spend.settlement is None for spend in self._spend.values())
            >= self._limits.max_in_flight
        ):
            return False
        self._spend[position] = _Spend(reservation=reservation)
        return True

    def settle(self, position: InvocationPosition, settlement: Settlement) -> None:
        spend = self._spend.get(position)
        if spend is None:
            raise ValueError("position has no durable budget charge")
        if spend.settlement is not None:
            if spend.settlement != settlement:
                raise ValueError("position budget was settled differently")
            return
        if isinstance(settlement.actual_attempts, bool) or isinstance(
            settlement.actual_output_bytes, bool
        ):
            raise TypeError("settlement accounting must use integers")
        if settlement.actual_attempts < 0 or settlement.actual_output_bytes < 0:
            raise ValueError("settlement accounting must not be negative")
        if (
            settlement.actual_attempts > spend.reservation.max_attempts
            or settlement.actual_output_bytes > spend.reservation.max_output_bytes
        ):
            raise ValueError("settlement exceeds reservation")
        spend.settlement = settlement


@dataclass(slots=True)
class PositionRecord:
    tool_id: ToolId
    tool_contract_revision: str
    policy_revision: str
    plan_revision: str
    input_digest: str
    replay_policy: ReplayPolicy
    reservation: Reservation | None = None
    reservation_accepted: bool | None = None
    terminal_result: ToolResult | None = None
    settlement: Settlement | None = None
    dispatches: int = 0
    abandoned_attempts: int = 0
    uncertain: bool = False
    in_flight: bool = False
    terminal_commits: int = 0


class InMemoryPositionRecorder:
    def __init__(self, *, durable: bool = True) -> None:
        self._durable = durable
        self._records: dict[InvocationPosition, PositionRecord] = {}

    @property
    def durable(self) -> bool:
        return self._durable

    def record(self, position: InvocationPosition) -> PositionRecord:
        return self._records[position]

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
    ) -> PositionState:
        existing = self._records.get(position)
        if existing is None:
            existing = PositionRecord(
                tool_id=tool_id,
                tool_contract_revision=tool_contract_revision,
                policy_revision=policy_revision,
                plan_revision=plan_revision,
                input_digest=input_digest,
                replay_policy=replay_policy,
            )
            self._records[position] = existing
        elif (
            existing.tool_id != tool_id
            or existing.tool_contract_revision != tool_contract_revision
            or existing.policy_revision != policy_revision
            or existing.plan_revision != plan_revision
            or existing.input_digest != input_digest
            or existing.replay_policy != replay_policy
        ):
            raise ValueError("occupied position invocation mismatch")
        recovered_billed_dispatch = (
            existing.replay_policy is ReplayPolicy.BilledOnce
            and existing.dispatches > 0
            and existing.terminal_result is None
        )
        return PositionState(
            terminal_result=existing.terminal_result,
            uncertain=existing.uncertain or recovered_billed_dispatch,
            actual_attempts=existing.abandoned_attempts,
        )

    def reserve(
        self,
        *,
        position: InvocationPosition,
        budgets: BudgetState,
        reservation: Reservation,
    ) -> bool:
        record = self._records[position]
        if record.reservation is not None:
            if record.reservation != reservation:
                raise ValueError("position reservation differs from durable record")
            assert record.reservation_accepted is not None
            return record.reservation_accepted
        accepted = budgets.reserve(position, reservation)
        record.reservation = reservation
        record.reservation_accepted = accepted
        return accepted

    def dispatch_started(
        self,
        *,
        position: InvocationPosition,
        replay_policy: ReplayPolicy,
    ) -> PositionState:
        record = self._records[position]
        if record.replay_policy != replay_policy:
            raise ValueError("dispatch replay policy mismatch")
        if record.terminal_result is not None:
            return PositionState(
                terminal_result=record.terminal_result,
                uncertain=False,
                actual_attempts=record.abandoned_attempts,
            )
        if record.uncertain:
            return PositionState(
                terminal_result=None,
                uncertain=True,
                actual_attempts=record.abandoned_attempts,
            )
        if record.in_flight:
            return PositionState(
                terminal_result=None,
                uncertain=True,
                actual_attempts=record.abandoned_attempts,
            )
        record.in_flight = True
        record.dispatches += 1
        return PositionState(
            terminal_result=None,
            uncertain=False,
            actual_attempts=record.abandoned_attempts,
        )

    def dispatch_abandoned(
        self,
        *,
        position: InvocationPosition,
        replay_policy: ReplayPolicy,
        actual_attempts: int,
        lease_recovered: bool,
    ) -> None:
        record = self._records[position]
        if replay_policy is not ReplayPolicy.ReDispatchable or record.replay_policy is not (
            ReplayPolicy.ReDispatchable
        ):
            raise ValueError("only ReDispatchable positions can be re-admitted")
        if record.terminal_result is not None:
            raise ValueError("terminal position cannot be re-admitted")
        if record.uncertain:
            raise ValueError("uncertain position requires explicit reconciliation")
        if not record.in_flight:
            raise ValueError("position has no abandoned dispatch claim")
        if not lease_recovered:
            raise ValueError("dispatch claim requires verified operator or lease recovery")
        if isinstance(actual_attempts, bool) or not isinstance(actual_attempts, int):
            raise TypeError("abandoned dispatch attempts must be an integer")
        if actual_attempts < 0:
            raise ValueError("abandoned dispatch attempts must not be negative")
        if record.reservation is None or (
            record.abandoned_attempts + actual_attempts > record.reservation.max_attempts
        ):
            raise ValueError("abandoned dispatch attempts exceed the position reservation")
        record.abandoned_attempts += actual_attempts
        record.in_flight = False

    def uncertain(self, *, position: InvocationPosition) -> None:
        record = self._records[position]
        if record.terminal_result is not None:
            raise ValueError("terminal position cannot become uncertain")
        record.uncertain = True
        record.in_flight = False

    def terminalize_and_settle(
        self,
        *,
        position: InvocationPosition,
        budgets: BudgetState,
        result: ToolResult,
        settlement: Settlement,
    ) -> ToolResult:
        record = self._records[position]
        if record.terminal_result is not None:
            if record.terminal_result != result or record.settlement != settlement:
                raise ValueError("terminal position commit mismatch")
            return record.terminal_result
        if record.uncertain:
            raise ValueError("uncertain position requires explicit reconciliation")
        if record.reservation_accepted:
            budgets.settle(position, settlement)
        record.terminal_result = result
        record.settlement = settlement
        record.in_flight = False
        record.terminal_commits += 1
        return result


class NeverCancelled:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, JsonObject]] = []

    def event(self, name: str, attributes: JsonObject) -> None:
        self.events.append((name, attributes))

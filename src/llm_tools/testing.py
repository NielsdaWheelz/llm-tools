"""Small deterministic doubles for kernel and consumer conformance proofs."""

from __future__ import annotations

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
from llm_tools.schema import JsonObject


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

    async def occupy(
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

    async def reserve(
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
        accepted = await budgets.reserve(position, reservation)
        record.reservation = reservation
        record.reservation_accepted = accepted
        return accepted

    async def dispatch_started(
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

    async def dispatch_abandoned(
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

    async def uncertain(self, *, position: InvocationPosition) -> None:
        record = self._records[position]
        if record.terminal_result is not None:
            raise ValueError("terminal position cannot become uncertain")
        record.uncertain = True
        record.in_flight = False

    async def terminalize_and_settle(
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
            await budgets.settle(position, settlement)
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

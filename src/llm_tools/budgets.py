"""Optional cumulative quotas and process-local, position-owned accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from llm_tools.execution import InvocationPosition, Reservation, Settlement
from llm_tools.profiles import RunLimits


@dataclass(frozen=True, slots=True)
class BudgetTotals:
    """Accepted calls/input; settled actual plus unsettled reserved attempts/output."""

    calls: int
    input_bytes: int
    external_attempts: int
    output_bytes: int
    in_flight: int

    def __post_init__(self) -> None:
        values = (
            self.calls,
            self.input_bytes,
            self.external_attempts,
            self.output_bytes,
            self.in_flight,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("budget totals must use integers")
        if any(value < 0 for value in values):
            raise ValueError("budget totals must not be negative")


def can_reserve(limits: RunLimits, totals: BudgetTotals, reservation: Reservation) -> bool:
    """Check a new position after the host handles duplicates under its lock/transaction."""

    cumulative = (
        (totals.calls + reservation.calls, limits.max_calls),
        (totals.input_bytes + reservation.input_bytes, limits.max_input_bytes),
        (totals.external_attempts + reservation.max_attempts, limits.max_external_attempts),
        (totals.output_bytes + reservation.max_output_bytes, limits.max_output_bytes),
    )
    return totals.in_flight < limits.max_in_flight and all(
        ceiling is None or total <= ceiling for total, ceiling in cumulative
    )


@dataclass(slots=True)
class _Spend:
    reservation: Reservation
    settlement: Settlement | None = None


class RunBudgetState:
    """One event-loop owner's budget; host recorders own durable atomic settlement."""

    def __init__(
        self,
        limits: RunLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
    ) -> None:
        self._limits = limits
        self._spend: dict[InvocationPosition, _Spend] = {}
        self._totals = BudgetTotals(0, 0, 0, 0, 0)
        self._actual_external_attempts = 0
        self._actual_output_bytes = 0
        self._monotonic = monotonic
        self._started_at = monotonic() if started_at is None else started_at

    @property
    def limits(self) -> RunLimits:
        return self._limits

    @property
    def remaining_elapsed_seconds(self) -> float | None:
        if self._limits.max_elapsed_seconds is None:
            return None
        return self._limits.max_elapsed_seconds - (self._monotonic() - self._started_at)

    @property
    def actual_external_attempts(self) -> int:
        return self._actual_external_attempts

    @property
    def actual_calls(self) -> int:
        return self._totals.calls

    @property
    def actual_input_bytes(self) -> int:
        return self._totals.input_bytes

    @property
    def actual_output_bytes(self) -> int:
        return self._actual_output_bytes

    @property
    def reserved_external_attempts(self) -> int:
        return self._totals.external_attempts - self._actual_external_attempts

    @property
    def reserved_output_bytes(self) -> int:
        return self._totals.output_bytes - self._actual_output_bytes

    async def reserve(self, position: InvocationPosition, reservation: Reservation) -> bool:
        existing = self._spend.get(position)
        if existing is not None:
            if existing.reservation != reservation:
                raise ValueError("position budget reservation differs from its original spend")
            return True
        if not can_reserve(self._limits, self._totals, reservation):
            return False
        totals = self._totals
        self._spend[position] = _Spend(reservation)
        self._totals = BudgetTotals(
            calls=totals.calls + reservation.calls,
            input_bytes=totals.input_bytes + reservation.input_bytes,
            external_attempts=totals.external_attempts + reservation.max_attempts,
            output_bytes=totals.output_bytes + reservation.max_output_bytes,
            in_flight=totals.in_flight + 1,
        )
        return True

    async def settle(self, position: InvocationPosition, settlement: Settlement) -> None:
        spend = self._spend.get(position)
        if spend is None:
            raise ValueError("position has no budget charge")
        if spend.settlement is not None:
            if spend.settlement != settlement:
                raise ValueError("position budget was settled differently")
            return
        if (
            settlement.actual_attempts > spend.reservation.max_attempts
            or settlement.actual_output_bytes > spend.reservation.max_output_bytes
        ):
            raise ValueError("settlement exceeds reservation")
        totals = self._totals
        self._totals = BudgetTotals(
            calls=totals.calls,
            input_bytes=totals.input_bytes,
            external_attempts=(
                totals.external_attempts
                - spend.reservation.max_attempts
                + settlement.actual_attempts
            ),
            output_bytes=(
                totals.output_bytes
                - spend.reservation.max_output_bytes
                + settlement.actual_output_bytes
            ),
            in_flight=totals.in_flight - 1,
        )
        self._actual_external_attempts += settlement.actual_attempts
        self._actual_output_bytes += settlement.actual_output_bytes
        spend.settlement = settlement

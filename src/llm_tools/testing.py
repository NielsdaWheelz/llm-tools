"""Small deterministic doubles for kernel and consumer conformance proofs."""

from __future__ import annotations

from llm_tools.execution import InvocationPosition
from llm_tools.recorders import _InMemoryPositionRecorder, _PositionRecord
from llm_tools.schema import JsonObject


class InMemoryPositionRecorder(_InMemoryPositionRecorder):
    def __init__(self, *, durable: bool = True) -> None:
        super().__init__()
        self._durable = durable

    @property
    def durable(self) -> bool:
        return self._durable

    def record(self, position: InvocationPosition) -> _PositionRecord:
        return self._records[position]


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

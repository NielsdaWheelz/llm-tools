"""Portable tool declarations and separately owned executable bindings."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from llm_tools.schema import (
    CompiledSchema,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    compile_schema,
)

_TOOL_SEGMENT = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*_?"
TOOL_NAMESPACE_PATTERN = rf"^{_TOOL_SEGMENT}$"
TOOL_ID_PATTERN = rf"^{_TOOL_SEGMENT}(?:\.{_TOOL_SEGMENT})+$"
_TOOL_ID = re.compile(TOOL_ID_PATTERN)
_LOCAL_ID = re.compile(r"[a-z][a-z0-9_-]*")


class ToolId(str):
    """Canonical public tool identity."""

    def __new__(cls, value: str) -> ToolId:
        if not _TOOL_ID.fullmatch(value):
            raise ValueError(f"invalid canonical tool id: {value!r}")
        return str.__new__(cls, value)


class PolicyEpoch(str):
    def __new__(cls, value: str) -> PolicyEpoch:
        if not _LOCAL_ID.fullmatch(value):
            raise ValueError(f"invalid policy epoch: {value!r}")
        return str.__new__(cls, value)


@dataclass(frozen=True, slots=True)
class PromptDocument:
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("prompt documentation must not be empty")


class ToolEffect(StrEnum):
    Pure = "Pure"
    Read = "Read"
    Write = "Write"


class ReplayPolicy(StrEnum):
    BilledOnce = "BilledOnce"
    ReDispatchable = "ReDispatchable"


@dataclass(frozen=True, slots=True)
class ToolLimits:
    max_input_bytes: int
    max_output_bytes: int
    max_attempts: int
    deadline_seconds: float

    def __post_init__(self) -> None:
        integer_limits = (self.max_input_bytes, self.max_output_bytes, self.max_attempts)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_limits):
            raise TypeError("tool byte and attempt limits must be integers")
        if isinstance(self.deadline_seconds, bool) or not isinstance(
            self.deadline_seconds, (int, float)
        ):
            raise TypeError("tool deadline must be numeric")
        if not math.isfinite(self.deadline_seconds):
            raise ValueError("tool deadline must be finite")
        if (
            self.max_input_bytes <= 0
            or self.max_output_bytes <= 0
            or self.max_attempts < 0
            or self.deadline_seconds <= 0
        ):
            raise ValueError("tool limits must be positive; attempts may be zero")

    def tightened(self, **changes: int | float) -> ToolLimits:
        return replace(self, **changes)

    def is_tightening_of(self, declared: ToolLimits) -> bool:
        return (
            self.max_input_bytes <= declared.max_input_bytes
            and self.max_output_bytes <= declared.max_output_bytes
            and self.max_attempts <= declared.max_attempts
            and self.deadline_seconds <= declared.deadline_seconds
        )

    def json(self) -> JsonObject:
        return {
            "deadline_seconds": self.deadline_seconds,
            "max_attempts": self.max_attempts,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
        }


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InvalidInput(_BoundaryModel):
    type: Literal["InvalidInput"] = "InvalidInput"


class ToolUnavailable(_BoundaryModel):
    type: Literal["ToolUnavailable"] = "ToolUnavailable"


class BudgetExceeded(_BoundaryModel):
    type: Literal["BudgetExceeded"] = "BudgetExceeded"


class DeadlineExceeded(_BoundaryModel):
    type: Literal["DeadlineExceeded"] = "DeadlineExceeded"


type BoundaryError = InvalidInput | ToolUnavailable | BudgetExceeded | DeadlineExceeded


class NoDeclaredError:
    """Marker for tools whose only failures are executor-owned boundary errors."""

    def __new__(cls) -> NoDeclaredError:
        raise TypeError("NoDeclaredError is a declaration marker and cannot be instantiated")


@dataclass(frozen=True, slots=True)
class ToolSpec[InputT, SuccessT, ErrorT]:
    id: ToolId
    summary: str
    documentation: PromptDocument
    input_type: type[InputT]
    success_type: type[SuccessT]
    error_type: type[ErrorT] | type[NoDeclaredError]
    effect: ToolEffect
    limits: ToolLimits
    input_schema: CompiledSchema = field(init=False, repr=False)
    success_schema: CompiledSchema = field(init=False, repr=False)
    declared_error_schema: CompiledSchema | None = field(init=False, repr=False)
    error_schema: CompiledSchema = field(init=False, repr=False)
    tool_contract_revision: str = field(init=False)
    documentation_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("tool summary must not be empty")
        input_schema = compile_schema(self.input_type)
        success_schema = compile_schema(self.success_type)
        for role, schema in (("input", input_schema), ("success", success_schema)):
            semantic = schema.semantic
            if (
                semantic.get("type") != "object"
                or semantic.get("additionalProperties") is not False
            ):
                raise ValueError(f"tool {role} schema must be a closed object")
        if self.error_type is NoDeclaredError:
            declared_error_schema = None
            error_schema = compile_schema(BoundaryError)
        else:
            declared_error_schema = compile_schema(self.error_type)
            error_schema = compile_schema(BoundaryError | self.error_type)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "success_schema", success_schema)
        object.__setattr__(self, "declared_error_schema", declared_error_schema)
        object.__setattr__(self, "error_schema", error_schema)
        contract = {
            "effect": self.effect.value,
            "error_schema": error_schema.semantic,
            "id": str(self.id),
            "input_schema": input_schema.semantic,
            "limits": self.limits.json(),
            "success_schema": success_schema.semantic,
        }
        documentation = {
            "documentation": self.documentation.text,
            "error_schema": error_schema.presentation,
            "input_schema": input_schema.presentation,
            "success_schema": success_schema.presentation,
            "summary": self.summary,
        }
        object.__setattr__(self, "tool_contract_revision", _revision(contract))
        object.__setattr__(self, "documentation_revision", _revision(documentation))


type ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Available[InputT]:
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class Unavailable:
    private_reason: str

    def __post_init__(self) -> None:
        if not self.private_reason:
            raise ValueError("unavailable binding requires a private reason")


@dataclass(frozen=True, slots=True)
class ToolBinding[InputT, SuccessT, ErrorT]:
    spec: ToolSpec[InputT, SuccessT, ErrorT]
    execute: Available[InputT] | Unavailable
    replay_policy: ReplayPolicy
    implementation_revision: str
    policy_epoch: PolicyEpoch
    policy_inputs: Mapping[str, object]
    policy_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_revision, str):
            raise TypeError("implementation revision must be a string")
        if not self.implementation_revision.strip():
            raise ValueError("implementation revision must not be empty")
        canonical_inputs = {key: self.policy_inputs[key] for key in sorted(self.policy_inputs)}
        canonical_json_bytes(canonical_inputs)
        object.__setattr__(self, "policy_inputs", _freeze_json(canonical_inputs))
        object.__setattr__(
            self,
            "policy_revision",
            _revision(
                {
                    "policy_epoch": str(self.policy_epoch),
                    "policy_inputs": canonical_inputs,
                    "replay_policy": self.replay_policy.value,
                }
            ),
        )


def _revision(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze_json(value: JsonValue) -> object:
    from types import MappingProxyType

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value

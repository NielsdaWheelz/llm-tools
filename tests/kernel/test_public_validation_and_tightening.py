"""Public pure-validation and frozen-authority tightening proofs."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Any, Literal, assert_type

import pytest
from pydantic import BaseModel, ConfigDict, Field

from llm_tools import (
    Available,
    CapabilityProfile,
    HostTable,
    NoDeclaredError,
    PolicyEpoch,
    ProfileId,
    PromptDocument,
    ReplayPolicy,
    RunLimits,
    SchemaDecodeError,
    ToolBinding,
    ToolCatalog,
    ToolEffect,
    ToolFamily,
    ToolGrant,
    ToolId,
    ToolLimits,
    ToolPlan,
    ToolSpec,
    validate_tool_input,
)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: Annotated[int, Field(ge=1, le=10)]


class ChangedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: Annotated[int, Field(ge=2, le=10)]


class Success(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: Literal[True] = True


DECLARED_TOOL_LIMITS = ToolLimits(
    max_input_bytes=1_000,
    max_output_bytes=1_000,
    max_attempts=4,
    deadline_seconds=10.0,
)
MAXIMUM_TOOL_LIMITS = ToolLimits(
    max_input_bytes=800,
    max_output_bytes=800,
    max_attempts=3,
    deadline_seconds=8.0,
)
NARROW_TOOL_LIMITS = ToolLimits(
    max_input_bytes=700,
    max_output_bytes=700,
    max_attempts=2,
    deadline_seconds=7.0,
)
MAXIMUM_RUN_LIMITS = RunLimits(
    max_calls=8,
    max_external_attempts=12,
    max_input_bytes=16_000,
    max_output_bytes=16_000,
    max_in_flight=2,
    max_elapsed_seconds=60.0,
)
NARROW_RUN_LIMITS = RunLimits(
    max_calls=7,
    max_external_attempts=11,
    max_input_bytes=15_000,
    max_output_bytes=15_000,
    max_in_flight=1,
    max_elapsed_seconds=50.0,
)


async def _unused_handler(value: Input, context: object) -> object:
    raise AssertionError(f"validation must not dispatch: {value!r}, {context!r}")


def _binding(
    *,
    tool_id: str = "test.echo",
    input_type: type[BaseModel] = Input,
    implementation_revision: str = "test-echo-v1",
    policy_epoch: str = "v1",
) -> ToolBinding[Any, Success, NoDeclaredError]:
    spec = ToolSpec(
        id=ToolId(tool_id),
        summary="Accept a count",
        documentation=PromptDocument("Accept the strictly validated count."),
        input_type=input_type,
        success_type=Success,
        error_type=NoDeclaredError,
        effect=ToolEffect.Pure,
        limits=DECLARED_TOOL_LIMITS,
    )
    return ToolBinding(
        spec=spec,
        execute=Available(_unused_handler),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision=implementation_revision,
        policy_epoch=PolicyEpoch(policy_epoch),
        policy_inputs={},
    )


def _profile(
    bindings: tuple[ToolBinding[Any, Success, NoDeclaredError], ...],
    *,
    grants: tuple[ToolGrant, ...],
    run_limits: RunLimits,
    profile_id: str,
):
    catalog = ToolCatalog.compose(
        (
            ToolFamily(
                namespace="test",
                declarations=tuple(binding.spec for binding in bindings),
                bindings=bindings,
            ),
        )
    )
    profile = CapabilityProfile(
        id=ProfileId(profile_id),
        grants=grants,
        run_limits=run_limits,
    ).freeze(catalog)
    return catalog, profile


def test_validate_tool_input_is_strict_and_has_no_execution_effects() -> None:
    binding = _binding()

    validated = validate_tool_input(binding, {"count": 3})

    assert validated == Input(count=3)
    for invalid in (
        {"count": "3"},
        {"count": 3, "unknown": True},
        {},
        [3],
    ):
        with pytest.raises(SchemaDecodeError):
            validate_tool_input(binding, invalid)


def test_validate_tool_input_returns_the_declared_owned_type() -> None:
    spec = ToolSpec(
        id=ToolId("test.typed"),
        summary="Accept a count",
        documentation=PromptDocument("Accept the strictly validated count."),
        input_type=Input,
        success_type=Success,
        error_type=NoDeclaredError,
        effect=ToolEffect.Pure,
        limits=DECLARED_TOOL_LIMITS,
    )
    binding: ToolBinding[Input, Success, NoDeclaredError] = ToolBinding(
        spec=spec,
        execute=Available(_unused_handler),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision="test-typed-v1",
        policy_epoch=PolicyEpoch("v1"),
        policy_inputs={},
    )

    assert_type(validate_tool_input(binding, {"count": 3}), Input)


def test_frozen_profile_and_plan_accept_strict_subsets_with_new_identity() -> None:
    binding = _binding()
    _, maximum = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="maximum",
    )
    candidate_catalog, candidate = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, NARROW_TOOL_LIMITS),),
        run_limits=NARROW_RUN_LIMITS,
        profile_id="candidate",
    )
    empty_catalog, empty = _profile(
        (binding,),
        grants=(),
        run_limits=NARROW_RUN_LIMITS,
        profile_id="empty",
    )

    assert candidate.profile_revision != maximum.profile_revision
    assert candidate.is_tightening_of(maximum)
    assert empty.is_tightening_of(maximum)
    plan = ToolPlan(candidate.id, HostTable()).freeze(candidate_catalog, candidate)
    assert isinstance(plan.exposure, HostTable)
    assert plan.is_tightening_of(maximum)
    assert plan.profile.run_limits.max_in_flight == 1
    assert ToolPlan(empty.id, HostTable()).freeze(empty_catalog, empty).is_tightening_of(maximum)


@pytest.mark.parametrize(
    ("field", "wider_value"),
    [
        ("max_input_bytes", 801),
        ("max_output_bytes", 801),
        ("max_attempts", 4),
        ("deadline_seconds", 8.1),
    ],
)
def test_frozen_profile_rejects_every_widened_tool_limit(
    field: str,
    wider_value: int | float,
) -> None:
    binding = _binding()
    _, maximum = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="maximum",
    )
    _, candidate = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, replace(MAXIMUM_TOOL_LIMITS, **{field: wider_value})),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="candidate",
    )

    assert not candidate.is_tightening_of(maximum)


@pytest.mark.parametrize(
    ("field", "wider_value"),
    [
        ("max_calls", 9),
        ("max_external_attempts", 13),
        ("max_input_bytes", 16_001),
        ("max_output_bytes", 16_001),
        ("max_in_flight", 3),
        ("max_elapsed_seconds", 60.1),
    ],
)
def test_frozen_profile_rejects_every_widened_run_limit(
    field: str,
    wider_value: int | float,
) -> None:
    binding = _binding()
    _, maximum = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="maximum",
    )
    _, candidate = _profile(
        (binding,),
        grants=(ToolGrant(binding.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=replace(MAXIMUM_RUN_LIMITS, **{field: wider_value}),
        profile_id="candidate",
    )

    assert not candidate.is_tightening_of(maximum)


def test_frozen_profile_rejects_extra_grants_and_revision_changes() -> None:
    baseline = _binding()
    _, maximum = _profile(
        (baseline,),
        grants=(ToolGrant(baseline.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="maximum",
    )

    extra = _binding(tool_id="test.extra")
    _, extra_grant = _profile(
        (baseline, extra),
        grants=(
            ToolGrant(baseline.spec.id, MAXIMUM_TOOL_LIMITS),
            ToolGrant(extra.spec.id, MAXIMUM_TOOL_LIMITS),
        ),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="extra",
    )
    policy_change = _binding(policy_epoch="v2")
    _, changed_policy = _profile(
        (policy_change,),
        grants=(ToolGrant(policy_change.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="changed-policy",
    )
    implementation_change = _binding(implementation_revision="test-echo-v2")
    _, changed_implementation = _profile(
        (implementation_change,),
        grants=(ToolGrant(implementation_change.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="changed-implementation",
    )
    contract_change = _binding(input_type=ChangedInput)
    _, changed_contract = _profile(
        (contract_change,),
        grants=(ToolGrant(contract_change.spec.id, MAXIMUM_TOOL_LIMITS),),
        run_limits=MAXIMUM_RUN_LIMITS,
        profile_id="changed-contract",
    )

    assert not extra_grant.is_tightening_of(maximum)
    assert not changed_implementation.is_tightening_of(maximum)
    assert not changed_policy.is_tightening_of(maximum)
    assert not changed_contract.is_tightening_of(maximum)

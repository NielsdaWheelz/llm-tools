"""Adversarial frozen-plan integrity and cross-catalog substitution proofs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from llm_tools import (
    TOOL_FAMILY,
    TOOL_READ_SPEC,
    TOOL_SEARCH_SPEC,
    Available,
    CapabilityProfile,
    Discoverable,
    EffectiveToolGrant,
    FrozenToolPlan,
    HostTable,
    Native,
    NoDeclaredError,
    PlanCatalogView,
    PolicyEpoch,
    ProfileId,
    PromptDocument,
    ReplayPolicy,
    RunLimits,
    ToolBinding,
    ToolCatalog,
    ToolEffect,
    ToolFamily,
    ToolGrant,
    ToolId,
    ToolLimits,
    ToolPlan,
    ToolSpec,
    publish_host_table,
)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class ChangedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class Success(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool


TOOL_LIMITS = ToolLimits(
    max_input_bytes=1_024,
    max_output_bytes=2_048,
    max_attempts=1,
    deadline_seconds=3.0,
)
RUN_LIMITS = RunLimits(
    max_calls=8,
    max_external_attempts=8,
    max_input_bytes=16_384,
    max_output_bytes=16_384,
    max_in_flight=1,
    max_elapsed_seconds=20.0,
)


async def _unused_handler(value: object, context: object) -> object:
    raise AssertionError(f"integrity validation must not dispatch: {value!r}, {context!r}")


async def _replacement_handler(value: object, context: object) -> object:
    raise AssertionError(f"replacement must not dispatch: {value!r}, {context!r}")


def _binding(
    tool_id: str = "test.inspect",
    *,
    effect: ToolEffect = ToolEffect.Read,
    input_type: type[BaseModel] = Input,
    limits: ToolLimits = TOOL_LIMITS,
    replay_policy: ReplayPolicy = ReplayPolicy.ReDispatchable,
    implementation_revision: str = "test-inspect-v1",
    policy_epoch: str = "v1",
    policy_inputs: dict[str, object] | None = None,
) -> ToolBinding[Any, Success, NoDeclaredError]:
    spec = ToolSpec(
        id=ToolId(tool_id),
        summary="Inspect a value",
        documentation=PromptDocument("Inspect one value."),
        input_type=input_type,
        success_type=Success,
        error_type=NoDeclaredError,
        effect=effect,
        limits=limits,
    )
    return ToolBinding(
        spec=spec,
        execute=Available(_unused_handler),
        replay_policy=replay_policy,
        implementation_revision=implementation_revision,
        policy_epoch=PolicyEpoch(policy_epoch),
        policy_inputs=policy_inputs or {},
    )


def _catalog(*bindings: ToolBinding[Any, Any, Any]) -> ToolCatalog:
    return ToolCatalog.compose(
        (
            ToolFamily(
                namespace="test",
                declarations=tuple(binding.spec for binding in bindings),
                bindings=bindings,
            ),
        )
    )


def _profile(catalog: ToolCatalog, binding: ToolBinding[Any, Any, Any]):
    return CapabilityProfile(
        id=ProfileId("maximum"),
        grants=(ToolGrant(binding.spec.id, None),),
        run_limits=RUN_LIMITS,
    ).freeze(catalog)


@pytest.mark.parametrize(
    ("replacement", "mismatch"),
    [
        (_binding(effect=ToolEffect.Write), "contract"),
        (_binding(input_type=ChangedInput), "contract"),
        (
            _binding(limits=TOOL_LIMITS.tightened(max_output_bytes=1_024)),
            "contract",
        ),
        (_binding(replay_policy=ReplayPolicy.BilledOnce), "policy"),
        (
            replace(
                _binding(implementation_revision="test-inspect-v2"),
                execute=Available(_replacement_handler),
            ),
            "implementation",
        ),
        (_binding(policy_epoch="v2"), "policy"),
        (_binding(policy_inputs={"audience": "external"}), "policy"),
    ],
    ids=(
        "read-to-write",
        "schema",
        "declaration-limits",
        "replay-policy",
        "implementation-revision",
        "policy-epoch",
        "policy-inputs",
    ),
)
def test_freeze_rejects_cross_catalog_contract_implementation_and_policy_substitution(
    replacement: ToolBinding[Any, Success, NoDeclaredError],
    mismatch: str,
) -> None:
    authorized = _binding()
    authorized_catalog = _catalog(authorized)
    profile = _profile(authorized_catalog, authorized)
    substituted_catalog = _catalog(replacement)

    with pytest.raises(ValueError, match=mismatch):
        ToolPlan(profile.id, HostTable()).freeze(substituted_catalog, profile)


def test_direct_inconsistent_plan_fails_tightening_and_host_publication() -> None:
    authorized = _binding()
    authorized_catalog = _catalog(authorized)
    profile = _profile(authorized_catalog, authorized)
    valid = ToolPlan(profile.id, HostTable()).freeze(authorized_catalog, profile)

    replacement = replace(
        _binding(implementation_revision="test-inspect-v2"),
        execute=Available(_replacement_handler),
    )
    substituted_catalog = _catalog(replacement)
    forged = FrozenToolPlan(
        profile=profile,
        exposure=HostTable(),
        catalog_view=PlanCatalogView.from_catalog(substituted_catalog, (replacement.spec.id,)),
        plan_revision=valid.plan_revision,
    )

    assert not forged.is_tightening_of(profile)
    with pytest.raises(ValueError, match="consistent frozen plan"):
        publish_host_table(forged)


@pytest.mark.parametrize(
    ("implementation_revision", "error"),
    [(7, TypeError), (" ", ValueError)],
)
def test_direct_effective_grant_requires_a_valid_implementation_revision(
    implementation_revision: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="implementation revision"):
        EffectiveToolGrant(
            id=ToolId("test.inspect"),
            limits=TOOL_LIMITS,
            implementation_revision=implementation_revision,
            tool_contract_revision="contract-v1",
            policy_revision="policy-v1",
        )


def test_implementation_revision_rotates_profile_and_plan_revisions() -> None:
    first = _binding(implementation_revision="test-inspect-v1")
    second = _binding(implementation_revision="test-inspect-v2")
    assert first.spec.tool_contract_revision == second.spec.tool_contract_revision
    assert first.policy_revision == second.policy_revision

    first_catalog = _catalog(first)
    second_catalog = _catalog(second)
    first_profile = _profile(first_catalog, first)
    second_profile = _profile(second_catalog, second)
    first_plan = ToolPlan(first_profile.id, HostTable()).freeze(first_catalog, first_profile)
    second_plan = ToolPlan(second_profile.id, HostTable()).freeze(second_catalog, second_profile)

    assert first_profile.profile_revision != second_profile.profile_revision
    assert first_plan.plan_revision != second_plan.plan_revision


@pytest.mark.parametrize(
    "view",
    [
        PlanCatalogView(_specs={}, _bindings={}),
        PlanCatalogView(
            _specs={_binding().spec.id: _binding().spec},
            _bindings={},
        ),
        PlanCatalogView(
            _specs={},
            _bindings={_binding().spec.id: _binding()},
        ),
    ],
    ids=("missing-spec-and-binding", "missing-binding", "missing-specification"),
)
def test_direct_plan_rejects_missing_view_entries(view: PlanCatalogView) -> None:
    binding = _binding()
    catalog = _catalog(binding)
    profile = _profile(catalog, binding)
    valid = ToolPlan(profile.id, HostTable()).freeze(catalog, profile)
    inconsistent = replace(valid, catalog_view=view)

    assert not inconsistent.is_tightening_of(profile)
    with pytest.raises(ValueError, match="consistent frozen plan"):
        publish_host_table(inconsistent)


@pytest.mark.parametrize("extra_side", ["specification", "binding"])
def test_direct_plan_rejects_extra_view_entries(extra_side: str) -> None:
    binding = _binding()
    extra = _binding("test.extra")
    catalog = _catalog(binding, extra)
    profile = _profile(catalog, binding)
    valid = ToolPlan(profile.id, HostTable()).freeze(catalog, profile)
    specs = {binding.spec.id: binding.spec}
    bindings = {binding.spec.id: binding}
    if extra_side == "specification":
        specs[extra.spec.id] = extra.spec
    else:
        bindings[extra.spec.id] = extra
    inconsistent = replace(
        valid,
        catalog_view=PlanCatalogView(_specs=specs, _bindings=bindings),
    )

    assert not inconsistent.is_tightening_of(profile)
    with pytest.raises(ValueError, match="consistent frozen plan"):
        publish_host_table(inconsistent)


def test_direct_plan_rejects_revisions_that_do_not_commit_to_contents() -> None:
    binding = _binding()
    catalog = _catalog(binding)
    profile = _profile(catalog, binding)
    valid = ToolPlan(profile.id, HostTable()).freeze(catalog, profile)

    mismatched_plan_revision = replace(valid, plan_revision="0" * 64)
    assert not mismatched_plan_revision.is_tightening_of(profile)
    with pytest.raises(ValueError, match="consistent frozen plan"):
        publish_host_table(mismatched_plan_revision)

    mismatched_profile = replace(profile, profile_revision="0" * 64)
    mismatched_profile_plan = replace(valid, profile=mismatched_profile)
    assert not mismatched_profile_plan.is_tightening_of(profile)
    with pytest.raises(ValueError, match="consistent frozen plan"):
        publish_host_table(mismatched_profile_plan)


def test_equivalent_catalogs_and_all_exposure_shapes_remain_valid() -> None:
    authorized = _binding()
    authorized_catalog = _catalog(authorized)
    profile = _profile(authorized_catalog, authorized)

    equivalent = _binding()
    equivalent_catalog = _catalog(equivalent)
    host_plan = ToolPlan(profile.id, HostTable()).freeze(equivalent_catalog, profile)
    native_plan = ToolPlan(profile.id, Native()).freeze(equivalent_catalog, profile)
    assert host_plan.is_tightening_of(profile)
    assert native_plan.is_tightening_of(profile)
    assert publish_host_table(host_plan)

    empty_catalog = ToolCatalog.compose(())
    empty_profile = CapabilityProfile(ProfileId("empty"), (), RUN_LIMITS).freeze(empty_catalog)
    empty_plan = ToolPlan(empty_profile.id, HostTable()).freeze(empty_catalog, empty_profile)
    assert empty_plan.is_tightening_of(empty_profile)
    assert publish_host_table(empty_plan)


def test_discoverable_view_is_the_authorized_subset_not_every_profile_grant() -> None:
    target = _binding("test.target")
    hidden = _binding("test.hidden")
    catalog = ToolCatalog.compose(
        (
            TOOL_FAMILY,
            ToolFamily(
                namespace="test",
                declarations=(target.spec, hidden.spec),
                bindings=(target, hidden),
            ),
        )
    )
    profile = CapabilityProfile(
        id=ProfileId("discovery"),
        grants=(
            ToolGrant(TOOL_SEARCH_SPEC.id, None),
            ToolGrant(TOOL_READ_SPEC.id, None),
            ToolGrant(target.spec.id, None),
            ToolGrant(hidden.spec.id, None),
        ),
        run_limits=RUN_LIMITS,
    ).freeze(catalog)

    plan = ToolPlan(
        profile.id,
        Discoverable(targets=(target.spec.id,), max_target_tools_published=1),
    ).freeze(catalog, profile)

    assert plan.is_tightening_of(profile)
    assert plan.catalog_view.spec(target.spec.id) is target.spec
    with pytest.raises(KeyError, match="absent from plan catalogue view"):
        plan.catalog_view.spec(hidden.spec.id)

"""Profile-filtered progressive discovery and reference-host conformance proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from llm_tools.budgets import RunBudgetState
from llm_tools.catalog import ToolCatalog, ToolFamily
from llm_tools.declaration import (
    TOOL_ID_PATTERN,
    TOOL_NAMESPACE_PATTERN,
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
from llm_tools.discovery import (
    TOOL_FAMILY,
    TOOL_READ_SPEC,
    TOOL_SEARCH_SPEC,
    ToolSearchInput,
    ToolSearchSuccess,
    published_tool_ids,
)
from llm_tools.execution import (
    ExecutionContext,
    HandlerSuccess,
    InvocationPosition,
    ParsedJson,
    Principal,
    Scope,
    ToolExecutor,
    ToolResult,
)
from llm_tools.profiles import (
    CapabilityProfile,
    Discoverable,
    FrozenToolPlan,
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


class TargetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needle: Annotated[str, Field(min_length=1, max_length=80, description="Text to find")]


class TargetSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool


class TargetFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["TargetRejected"] = "TargetRejected"


TARGET_LIMITS = ToolLimits(
    max_input_bytes=1_024,
    max_output_bytes=2_048,
    max_attempts=1,
    deadline_seconds=3.0,
)
RUN_LIMITS = RunLimits(
    max_calls=32,
    max_external_attempts=16,
    max_input_bytes=131_072,
    max_output_bytes=1_048_576,
    max_in_flight=1,
    max_elapsed_seconds=60.0,
)


async def _unused_target_handler(
    value: TargetInput,
    context: ExecutionContext,
) -> HandlerSuccess[TargetSuccess]:
    raise AssertionError(f"target must not dispatch in discovery proof: {value!r}, {context!r}")


def _target(
    tool_id: str,
    summary: str,
    *,
    effect: ToolEffect = ToolEffect.Read,
) -> tuple[
    ToolSpec[TargetInput, TargetSuccess, TargetFailure],
    ToolBinding[TargetInput, TargetSuccess, TargetFailure],
]:
    spec = ToolSpec(
        id=ToolId(tool_id),
        summary=summary,
        documentation=PromptDocument(f"Complete documentation for {tool_id}."),
        input_type=TargetInput,
        success_type=TargetSuccess,
        error_type=TargetFailure,
        effect=effect,
        limits=TARGET_LIMITS,
    )
    return spec, ToolBinding(
        spec=spec,
        execute=Available(_unused_target_handler),
        replay_policy=ReplayPolicy.ReDispatchable,
        implementation_revision="test-target-v1",
        policy_epoch=PolicyEpoch("target-v1"),
        policy_inputs={},
    )


@dataclass(frozen=True, slots=True)
class Fixture:
    catalog: ToolCatalog
    plan: FrozenToolPlan
    alpha: ToolSpec[TargetInput, TargetSuccess, TargetFailure]
    zeta: ToolSpec[TargetInput, TargetSuccess, TargetFailure]


def _fixture(
    *,
    publication_cap: int = 1,
    search_output_limit: int | None = None,
    read_output_limit: int | None = None,
) -> Fixture:
    alpha, alpha_binding = _target("alpha.inspect", "Inspect an alpha item")
    hidden, hidden_binding = _target("alpha.hidden", "Private granted metadata")
    zeta, zeta_binding = _target(
        "zeta.compose",
        "Compose a public report",
        effect=ToolEffect.Write,
    )
    ungranted, ungranted_binding = _target("beta.ungranted", "Ungrantable catalogue entry")
    catalog = ToolCatalog.compose(
        (
            TOOL_FAMILY,
            ToolFamily(
                namespace="zeta",
                declarations=(zeta,),
                bindings=(zeta_binding,),
            ),
            ToolFamily(
                namespace="alpha",
                declarations=(hidden, alpha),
                bindings=(hidden_binding, alpha_binding),
            ),
            ToolFamily(
                namespace="beta",
                declarations=(ungranted,),
                bindings=(ungranted_binding,),
            ),
        )
    )
    profile = CapabilityProfile(
        id=ProfileId("research"),
        grants=(
            ToolGrant(
                id=TOOL_SEARCH_SPEC.id,
                limits=(
                    None
                    if search_output_limit is None
                    else TOOL_SEARCH_SPEC.limits.tightened(max_output_bytes=search_output_limit)
                ),
            ),
            ToolGrant(
                id=TOOL_READ_SPEC.id,
                limits=(
                    None
                    if read_output_limit is None
                    else TOOL_READ_SPEC.limits.tightened(max_output_bytes=read_output_limit)
                ),
            ),
            ToolGrant(id=zeta.id, limits=TARGET_LIMITS.tightened(max_output_bytes=1_024)),
            ToolGrant(id=hidden.id, limits=None),
            ToolGrant(id=alpha.id, limits=None),
        ),
        run_limits=RUN_LIMITS,
    ).freeze(catalog)
    plan = ToolPlan(
        profile=profile.id,
        exposure=Discoverable(
            targets=(zeta.id, alpha.id),
            max_target_tools_published=publication_cap,
        ),
    ).freeze(catalog, profile)
    return Fixture(catalog=catalog, plan=plan, alpha=alpha, zeta=zeta)


def _context(
    fixture: Fixture,
    tool_id: ToolId,
    *,
    position: str,
    recorder: InMemoryPositionRecorder | None = None,
    budgets: RunBudgetState | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        plan=fixture.plan,
        grant=fixture.plan.grant(tool_id),
        catalog_view=fixture.plan.catalog_view,
        position=InvocationPosition(position),
        recorder=recorder or InMemoryPositionRecorder(),
        effect_id=None,
        budgets=budgets or RunBudgetState(RUN_LIMITS),
        principal=Principal("principal-1"),
        scope=Scope("scope-1"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )


async def _execute(
    fixture: Fixture,
    tool_id: ToolId,
    value: dict[str, object],
    *,
    position: str,
) -> ToolResult:
    binding = fixture.catalog.binding(tool_id)
    result = await ToolExecutor.execute(
        binding,
        ParsedJson(value),
        _context(fixture, tool_id, position=position),
    )
    return result


def test_discovery_declarations_have_reviewed_closed_contracts_and_content() -> None:
    assert tuple(binding.spec for binding in TOOL_FAMILY.bindings) == (
        TOOL_SEARCH_SPEC,
        TOOL_READ_SPEC,
    )
    assert all(
        binding.replay_policy is ReplayPolicy.ReDispatchable for binding in TOOL_FAMILY.bindings
    )
    assert TOOL_SEARCH_SPEC.summary == "Find tools granted for progressive use"
    assert TOOL_SEARCH_SPEC.documentation.text == (
        "Search the tools granted and discoverable in this frozen plan. Match canonical ids, "
        "families, summaries, and input property names. Use an empty query to list a selected "
        "family. Results describe existing authority; they never grant a tool. Call tool.read "
        "before asking the host to publish a target's native definition."
    )
    assert TOOL_SEARCH_SPEC.input_schema.semantic == {
        "additionalProperties": False,
        "properties": {
            "family": {
                "anyOf": [
                    {
                        "maxLength": 64,
                        "minLength": 1,
                        "pattern": TOOL_NAMESPACE_PATTERN,
                        "type": "string",
                    },
                    {"type": "null"},
                ]
            },
            "limit": {"maximum": 20, "minimum": 1, "type": "integer"},
            "query": {"maxLength": 200, "minLength": 0, "type": "string"},
        },
        "required": ["family", "limit", "query"],
        "type": "object",
    }
    assert TOOL_READ_SPEC.summary == "Read one granted tool declaration"
    assert TOOL_READ_SPEC.documentation.text == (
        "Read the complete declaration of one granted discoverable target. Unknown, ungranted, "
        "and non-target ids are intentionally indistinguishable. A successful read records the "
        "already-granted target for deterministic native publication on the next request, subject "
        "to the plan's publication ceiling."
    )
    assert TOOL_READ_SPEC.input_schema.semantic == {
        "additionalProperties": False,
        "properties": {
            "id": {
                "maxLength": 255,
                "minLength": 3,
                "pattern": TOOL_ID_PATTERN,
                "type": "string",
            }
        },
        "required": ["id"],
        "type": "object",
    }
    for spec in (TOOL_SEARCH_SPEC, TOOL_READ_SPEC):
        assert spec.effect is ToolEffect.Pure
        assert spec.declared_error_schema is None
        assert {
            branch["properties"]["type"]["const"] for branch in spec.error_schema.semantic["anyOf"]
        } == {
            "BudgetExceeded",
            "DeadlineExceeded",
            "InvalidInput",
            "ToolUnavailable",
        }
        assert spec.input_schema.semantic["additionalProperties"] is False
        assert spec.success_schema.semantic["additionalProperties"] is False


def test_discovery_documentation_changes_do_not_change_authority_revision() -> None:
    changed = ToolSpec[ToolSearchInput, ToolSearchSuccess, NoDeclaredError](
        id=TOOL_SEARCH_SPEC.id,
        summary="Find discoverable tools",
        documentation=PromptDocument("Reworded model-facing help."),
        input_type=ToolSearchInput,
        success_type=ToolSearchSuccess,
        error_type=TOOL_SEARCH_SPEC.error_type,
        effect=TOOL_SEARCH_SPEC.effect,
        limits=TOOL_SEARCH_SPEC.limits,
    )

    assert changed.tool_contract_revision == TOOL_SEARCH_SPEC.tool_contract_revision
    assert changed.documentation_revision != TOOL_SEARCH_SPEC.documentation_revision


@pytest.mark.asyncio
async def test_search_reveals_only_targets_with_stable_matching_order_and_truncation() -> None:
    fixture = _fixture(publication_cap=1)

    listing = await _execute(
        fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "", "family": None, "limit": 20},
        position="turn-1/search-all",
    )
    assert listing == {
        "type": "Success",
        "value": {
            "matches": [
                {
                    "effect": "Read",
                    "family": "alpha",
                    "id": "alpha.inspect",
                    "input_synopsis": "needle",
                    "summary": "Inspect an alpha item",
                },
                {
                    "effect": "Write",
                    "family": "zeta",
                    "id": "zeta.compose",
                    "input_synopsis": "needle",
                    "summary": "Compose a public report",
                },
            ],
            "truncated": False,
        },
    }
    assert "alpha.hidden" not in repr(listing)
    assert "beta.ungranted" not in repr(listing)
    assert "tool.read" not in repr(listing)

    family_listing = await _execute(
        fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "", "family": "zeta", "limit": 20},
        position="turn-1/search-family",
    )
    assert [item["id"] for item in family_listing["value"]["matches"]] == ["zeta.compose"]

    property_match = await _execute(
        fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "NEEDLE", "family": "alpha", "limit": 20},
        position="turn-1/search-property",
    )
    assert [item["id"] for item in property_match["value"]["matches"]] == ["alpha.inspect"]

    truncated = await _execute(
        fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "", "family": None, "limit": 1},
        position="turn-1/search-truncated",
    )
    assert truncated["value"] == {
        "matches": [listing["value"]["matches"][0]],
        "truncated": True,
    }


@pytest.mark.asyncio
async def test_read_is_full_and_unknown_ungranted_and_nontarget_are_identical() -> None:
    fixture = _fixture()
    unavailable_results = []
    for index, tool_id in enumerate(
        ("gamma.unknown", "beta.ungranted", "alpha.hidden"),
        start=1,
    ):
        unavailable_results.append(
            await _execute(
                fixture,
                TOOL_READ_SPEC.id,
                {"id": tool_id},
                position=f"turn-2/unavailable-{index}",
            )
        )

    assert unavailable_results == [
        {"type": "Failure", "error": {"type": "ToolUnavailable"}},
        {"type": "Failure", "error": {"type": "ToolUnavailable"}},
        {"type": "Failure", "error": {"type": "ToolUnavailable"}},
    ]

    result = await _execute(
        fixture,
        TOOL_READ_SPEC.id,
        {"id": "alpha.inspect"},
        position="turn-2/read-alpha",
    )
    assert result["type"] == "Success"
    value = result["value"]
    assert value == {
        "declared_limits": TARGET_LIMITS.json(),
        "documentation": fixture.alpha.documentation.text,
        "documentation_revision": fixture.alpha.documentation_revision,
        "effect": "Read",
        "effective_limits": TARGET_LIMITS.json(),
        "error_schema": {
            "presentation_json": _stable_json(fixture.alpha.error_schema.presentation),
            "semantic_json": _stable_json(fixture.alpha.error_schema.semantic),
        },
        "family": "alpha",
        "id": "alpha.inspect",
        "input_schema": {
            "presentation_json": _stable_json(fixture.alpha.input_schema.presentation),
            "semantic_json": _stable_json(fixture.alpha.input_schema.semantic),
        },
        "replay_policy": "ReDispatchable",
        "success_schema": {
            "presentation_json": _stable_json(fixture.alpha.success_schema.presentation),
            "semantic_json": _stable_json(fixture.alpha.success_schema.semantic),
        },
        "summary": fixture.alpha.summary,
        "tool_contract_revision": fixture.alpha.tool_contract_revision,
    }


@pytest.mark.asyncio
async def test_discovery_respects_tightened_output_limits_without_owned_output_defects() -> None:
    fixture = _fixture(search_output_limit=100, read_output_limit=128)

    search = await _execute(
        fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "", "family": None, "limit": 20},
        position="turn-3/bounded-search",
    )
    assert search == {
        "type": "Success",
        "value": {"matches": [], "truncated": True},
    }
    assert await _execute(
        fixture,
        TOOL_READ_SPEC.id,
        {"id": "alpha.inspect"},
        position="turn-3/bounded-read",
    ) == {"type": "Failure", "error": {"type": "BudgetExceeded"}}

    minimum_fixture = _fixture(search_output_limit=54, read_output_limit=54)
    assert await _execute(
        minimum_fixture,
        TOOL_SEARCH_SPEC.id,
        {"query": "", "family": None, "limit": 20},
        position="turn-3/minimum-search",
    ) == {"type": "Failure", "error": {"type": "BudgetExceeded"}}


@dataclass(slots=True)
class ReferenceHost:
    fixture: Fixture
    revealed_targets: set[ToolId] = field(default_factory=set)
    recorder: InMemoryPositionRecorder = field(default_factory=InMemoryPositionRecorder)
    budgets: RunBudgetState = field(default_factory=lambda: RunBudgetState(RUN_LIMITS))
    call_index: int = 0

    @property
    def published(self) -> tuple[ToolId, ...]:
        return published_tool_ids(self.fixture.plan, frozenset(self.revealed_targets))

    async def call(
        self,
        tool_id: ToolId,
        value: dict[str, object],
    ) -> ToolResult:
        self.call_index += 1
        result = await ToolExecutor.execute(
            self.fixture.catalog.binding(tool_id),
            ParsedJson(value),
            _context(
                self.fixture,
                tool_id,
                position=f"reference-host/{self.call_index}",
                recorder=self.recorder,
                budgets=self.budgets,
            ),
        )
        if tool_id == TOOL_READ_SPEC.id and result["type"] == "Success":
            raw_target = value["id"]
            assert isinstance(raw_target, str)
            self.revealed_targets.add(ToolId(raw_target))
        return result


@pytest.mark.asyncio
async def test_reference_host_publishes_only_successfully_read_targets_under_exact_cap() -> None:
    host = ReferenceHost(_fixture(publication_cap=1))
    assert host.published == (ToolId("tool.search"), ToolId("tool.read"))

    await host.call(TOOL_SEARCH_SPEC.id, {"query": "", "family": None, "limit": 20})
    await host.call(TOOL_READ_SPEC.id, {"id": "alpha.hidden"})
    assert host.published == (ToolId("tool.search"), ToolId("tool.read"))

    assert (await host.call(TOOL_READ_SPEC.id, {"id": "zeta.compose"}))["type"] == "Success"
    assert host.published == (
        ToolId("tool.search"),
        ToolId("tool.read"),
        ToolId("zeta.compose"),
    )

    assert (await host.call(TOOL_READ_SPEC.id, {"id": "alpha.inspect"}))["type"] == "Success"
    assert host.published == (
        ToolId("tool.search"),
        ToolId("tool.read"),
        ToolId("alpha.inspect"),
    )


def test_plan_revision_and_publication_ceiling_have_independent_oracles() -> None:
    fixture = _fixture(publication_cap=1)
    exposure = fixture.plan.exposure
    assert isinstance(exposure, Discoverable)
    expected_payload = {
        "exposure": {
            "max_target_tools_published": 1,
            "targets": ["zeta.compose", "alpha.inspect"],
            "type": "Discoverable",
        },
        "profile_revision": fixture.plan.profile.profile_revision,
    }
    expected_revision = hashlib.sha256(
        json.dumps(
            expected_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    assert fixture.plan.plan_revision == expected_revision
    assert _fixture(publication_cap=2).plan.plan_revision != fixture.plan.plan_revision
    with pytest.raises(ValueError, match="outside the discoverable target set"):
        published_tool_ids(fixture.plan, frozenset({ToolId("alpha.hidden")}))


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

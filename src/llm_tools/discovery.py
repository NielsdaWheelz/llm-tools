"""Profile-filtered progressive tool discovery."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from llm_tools.catalog import ToolFamily
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
from llm_tools.execution import BoundaryFailure, ExecutionContext, HandlerSuccess
from llm_tools.profiles import Discoverable, FrozenToolPlan
from llm_tools.schema import canonical_json_bytes


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolSearchInput(_ClosedModel):
    query: Annotated[
        str,
        Field(
            min_length=0,
            max_length=200,
            description="Text matched against tool ids, families, summaries, and input fields.",
        ),
    ]
    family: (
        Annotated[
            str,
            Field(
                min_length=1,
                max_length=64,
                pattern=TOOL_NAMESPACE_PATTERN,
                description="Optional exact family namespace.",
            ),
        ]
        | None
    )
    limit: Annotated[
        int,
        Field(ge=1, le=20, description="Maximum number of stable ordered matches."),
    ]


class ToolSearchMatch(_ClosedModel):
    id: Annotated[str, Field(pattern=TOOL_ID_PATTERN)]
    family: Annotated[str, Field(pattern=TOOL_NAMESPACE_PATTERN)]
    summary: str
    effect: ToolEffect
    input_synopsis: str


class ToolSearchSuccess(_ClosedModel):
    matches: Annotated[tuple[ToolSearchMatch, ...], Field(max_length=20)]
    truncated: bool


class ToolReadInput(_ClosedModel):
    id: Annotated[
        str,
        Field(
            min_length=3,
            max_length=255,
            pattern=TOOL_ID_PATTERN,
            description="Canonical id of one granted discoverable target.",
        ),
    ]


class ToolLimitsDocument(_ClosedModel):
    max_input_bytes: int
    max_output_bytes: int
    max_attempts: int
    deadline_seconds: float


class SchemaDocument(_ClosedModel):
    semantic_json: str
    presentation_json: str


class ToolReadSuccess(_ClosedModel):
    id: Annotated[str, Field(pattern=TOOL_ID_PATTERN)]
    family: Annotated[str, Field(pattern=TOOL_NAMESPACE_PATTERN)]
    summary: str
    documentation: str
    effect: ToolEffect
    replay_policy: ReplayPolicy
    declared_limits: ToolLimitsDocument
    effective_limits: ToolLimitsDocument
    input_schema: SchemaDocument
    success_schema: SchemaDocument
    error_schema: SchemaDocument
    tool_contract_revision: str
    documentation_revision: str


TOOL_SEARCH_SPEC = ToolSpec[ToolSearchInput, ToolSearchSuccess, NoDeclaredError](
    id=ToolId("tool.search"),
    summary="Find tools granted for progressive use",
    documentation=PromptDocument(
        "Search the tools granted and discoverable in this frozen plan. Match canonical ids, "
        "families, summaries, and input property names. Use an empty query to list a selected "
        "family. Results describe existing authority; they never grant a tool. Call tool.read "
        "before asking the host to publish a target's native definition."
    ),
    input_type=ToolSearchInput,
    success_type=ToolSearchSuccess,
    error_type=NoDeclaredError,
    effect=ToolEffect.Pure,
    limits=ToolLimits(
        max_input_bytes=4_096,
        max_output_bytes=16_384,
        max_attempts=0,
        deadline_seconds=2.0,
    ),
)

TOOL_READ_SPEC = ToolSpec[ToolReadInput, ToolReadSuccess, NoDeclaredError](
    id=ToolId("tool.read"),
    summary="Read one granted tool declaration",
    documentation=PromptDocument(
        "Read the complete declaration of one granted discoverable target. Unknown, ungranted, "
        "and non-target ids are intentionally indistinguishable. A successful read records the "
        "already-granted target for deterministic native publication on the next request, subject "
        "to the plan's publication ceiling."
    ),
    input_type=ToolReadInput,
    success_type=ToolReadSuccess,
    error_type=NoDeclaredError,
    effect=ToolEffect.Pure,
    limits=ToolLimits(
        max_input_bytes=4_096,
        max_output_bytes=65_536,
        max_attempts=0,
        deadline_seconds=2.0,
    ),
)


async def _search(
    request: ToolSearchInput,
    context: ExecutionContext,
) -> HandlerSuccess[ToolSearchSuccess]:
    exposure = _discoverable(context.plan)
    terms = tuple(request.query.casefold().split())
    matches: list[ToolSearchMatch] = []
    truncated = False
    for tool_id in sorted(exposure.targets):
        spec = context.catalog_view.spec(tool_id)
        family = str(tool_id).split(".", 1)[0]
        if request.family is not None and family != request.family:
            continue
        input_names = tuple(sorted(spec.input_schema.semantic.get("properties", {})))
        searchable = " ".join((str(tool_id), family, spec.summary, *input_names)).casefold()
        if not all(term in searchable for term in terms):
            continue
        match = ToolSearchMatch(
            id=str(tool_id),
            family=family,
            summary=spec.summary,
            effect=spec.effect,
            input_synopsis=", ".join(input_names) if input_names else "(no input fields)",
        )
        if len(matches) == request.limit or not _success_fits(
            ToolSearchSuccess(matches=(*matches, match), truncated=False),
            context,
        ):
            truncated = True
            break
        matches.append(match)

    result = ToolSearchSuccess(matches=tuple(matches), truncated=truncated)
    if not _success_fits(result, context):
        raise BoundaryFailure("BudgetExceeded")
    return HandlerSuccess(result, actual_attempts=0)


async def _read(
    request: ToolReadInput,
    context: ExecutionContext,
) -> HandlerSuccess[ToolReadSuccess]:
    exposure = _discoverable(context.plan)
    tool_id = ToolId(request.id)
    if tool_id not in exposure.targets:
        raise BoundaryFailure("ToolUnavailable")
    try:
        spec = context.catalog_view.spec(tool_id)
        binding = context.catalog_view.binding(tool_id)
        effective = context.plan.grant(tool_id)
    except KeyError as exc:
        raise RuntimeError("discoverable target is absent from its frozen plan") from exc

    result = ToolReadSuccess(
        id=str(tool_id),
        family=str(tool_id).split(".", 1)[0],
        summary=spec.summary,
        documentation=spec.documentation.text,
        effect=spec.effect,
        replay_policy=binding.replay_policy,
        declared_limits=ToolLimitsDocument(**spec.limits.json()),
        effective_limits=ToolLimitsDocument(**effective.limits.json()),
        input_schema=_schema_document(spec.input_schema.semantic, spec.input_schema.presentation),
        success_schema=_schema_document(
            spec.success_schema.semantic,
            spec.success_schema.presentation,
        ),
        error_schema=_schema_document(spec.error_schema.semantic, spec.error_schema.presentation),
        tool_contract_revision=spec.tool_contract_revision,
        documentation_revision=spec.documentation_revision,
    )
    if not _success_fits(result, context):
        raise BoundaryFailure("BudgetExceeded")
    return HandlerSuccess(result, actual_attempts=0)


def _schema_document(
    semantic: object,
    presentation: object,
) -> SchemaDocument:
    return SchemaDocument(
        semantic_json=canonical_json_bytes(semantic).decode("utf-8"),
        presentation_json=canonical_json_bytes(presentation).decode("utf-8"),
    )


def _success_fits(value: BaseModel, context: ExecutionContext) -> bool:
    envelope = {"type": "Success", "value": value.model_dump(mode="json")}
    return len(canonical_json_bytes(envelope)) <= context.grant.limits.max_output_bytes


def _discoverable(plan: FrozenToolPlan) -> Discoverable:
    if not isinstance(plan.exposure, Discoverable):
        raise RuntimeError("discovery bindings require a frozen Discoverable plan")
    return plan.exposure


def published_tool_ids(
    plan: FrozenToolPlan,
    revealed_targets: Iterable[ToolId],
) -> tuple[ToolId, ...]:
    """Project persisted successful reads into the next request's native set."""

    exposure = _discoverable(plan)
    revealed = frozenset(revealed_targets)
    outside_targets = revealed - frozenset(exposure.targets)
    if outside_targets:
        raise ValueError("revealed tool is outside the discoverable target set")
    # The ceiling is a deterministic prompt-cost cap, not a chronology contract.
    # Canonical ordering makes the projection identical across process recovery.
    published_targets = tuple(sorted(revealed))[: exposure.max_target_tools_published]
    return (TOOL_SEARCH_SPEC.id, TOOL_READ_SPEC.id, *published_targets)


_TOOL_SEARCH_BINDING = ToolBinding[ToolSearchInput, ToolSearchSuccess, NoDeclaredError](
    spec=TOOL_SEARCH_SPEC,
    execute=Available(_search),
    replay_policy=ReplayPolicy.ReDispatchable,
    policy_epoch=PolicyEpoch("discovery-v1"),
    policy_inputs={},
)
_TOOL_READ_BINDING = ToolBinding[ToolReadInput, ToolReadSuccess, NoDeclaredError](
    spec=TOOL_READ_SPEC,
    execute=Available(_read),
    replay_policy=ReplayPolicy.ReDispatchable,
    policy_epoch=PolicyEpoch("discovery-v1"),
    policy_inputs={},
)

TOOL_FAMILY = ToolFamily(
    namespace="tool",
    declarations=(TOOL_SEARCH_SPEC, TOOL_READ_SPEC),
    bindings=(_TOOL_SEARCH_BINDING, _TOOL_READ_BINDING),
)

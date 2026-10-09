"""Independent contract proof for declarations, schemas, catalogues, and plans."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

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
    Unavailable,
)
from llm_tools.profiles import (
    CapabilityProfile,
    Discoverable,
    HostTable,
    Native,
    ProfileId,
    RunLimits,
    ToolGrant,
    ToolPlan,
)
from llm_tools.schema import (
    SchemaDecodeError,
    SchemaEncodeDefect,
    UnsupportedSchema,
    compile_schema,
    strict_decode,
    strict_encode,
)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", title="SearchInput")

    query: Annotated[str, Field(min_length=2, max_length=400, description="Search text")]
    mode: Literal["web", "news"]


class SearchSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    titles: Annotated[list[str], Field(max_length=10)]


class SearchFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["RateLimited"] = "RateLimited"
    retry_after_seconds: int | None


LIMITS = ToolLimits(
    max_input_bytes=4_096,
    max_output_bytes=32_768,
    max_attempts=2,
    deadline_seconds=15.0,
)
RUN_LIMITS = RunLimits(
    max_calls=8,
    max_external_attempts=12,
    max_input_bytes=16_384,
    max_output_bytes=131_072,
    max_in_flight=1,
    max_elapsed_seconds=60.0,
)


async def _unused_handler(value: SearchInput, context: object) -> object:
    raise AssertionError(f"handler should not run: {value!r}, {context!r}")


def _spec(
    *,
    input_type: type[BaseModel] = SearchInput,
    summary: str = "Search the public web",
    documentation: str = "Use this for current public information.",
    tool_id: str = "web.search",
) -> ToolSpec[BaseModel, SearchSuccess, SearchFailure]:
    return ToolSpec(
        id=ToolId(tool_id),
        summary=summary,
        documentation=PromptDocument(documentation),
        input_type=input_type,
        success_type=SearchSuccess,
        error_type=SearchFailure,
        effect=ToolEffect.Read,
        limits=LIMITS,
    )


def _binding(
    spec: ToolSpec[BaseModel, SearchSuccess, SearchFailure],
    *,
    epoch: str = "search-v1",
) -> ToolBinding[BaseModel, SearchSuccess, SearchFailure]:
    return ToolBinding(
        spec=spec,
        execute=Available(_unused_handler),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision="test-search-v1",
        policy_epoch=PolicyEpoch(epoch),
        policy_inputs={"provider": "brave", "safe_search": "moderate"},
    )


def _catalog() -> tuple[ToolCatalog, ToolSpec[BaseModel, SearchSuccess, SearchFailure]]:
    search = _spec()
    web = ToolFamily(namespace="web", declarations=(search,), bindings=(_binding(search),))

    read = _spec(tool_id="tool.read")
    tool = ToolFamily(namespace="tool", declarations=(read,), bindings=(_binding(read),))
    return ToolCatalog.compose((web, tool)), search


def test_schema_normalization_closes_objects_and_separates_annotations() -> None:
    spec = _spec()

    assert spec.input_schema.semantic == {
        "additionalProperties": False,
        "properties": {
            "mode": {"enum": ["news", "web"], "type": "string"},
            "query": {"maxLength": 400, "minLength": 2, "type": "string"},
        },
        "required": ["mode", "query"],
        "type": "object",
    }
    assert spec.input_schema.presentation["title"] == "SearchInput"
    assert spec.input_schema.presentation["properties"]["query"]["description"] == "Search text"
    declared_error_branch = next(
        branch
        for branch in spec.error_schema.semantic["anyOf"]
        if isinstance(branch, dict)
        and isinstance(branch.get("properties"), dict)
        and "retry_after_seconds" in branch["properties"]
    )
    assert declared_error_branch["properties"]["retry_after_seconds"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }


def test_revisions_distinguish_semantics_documentation_and_binding_policy() -> None:
    baseline = _spec()
    description_only = _spec(
        summary="Find public sources",
        documentation="Reworded presentation instructions.",
    )

    class ReorderedInput(BaseModel):
        model_config = ConfigDict(extra="forbid", title="SearchInput")

        query: Annotated[str, Field(min_length=2, max_length=400, description="Search text")]
        mode: Literal["news", "web"]

    class ChangedInput(BaseModel):
        model_config = ConfigDict(extra="forbid", title="SearchInput")

        query: Annotated[str, Field(min_length=2, max_length=400, description="Search text")]
        mode: Literal["web", "images"]

    reordered = _spec(input_type=ReorderedInput)
    changed = _spec(input_type=ChangedInput)

    assert description_only.tool_contract_revision == baseline.tool_contract_revision
    assert description_only.documentation_revision != baseline.documentation_revision
    assert reordered.tool_contract_revision == baseline.tool_contract_revision
    assert changed.tool_contract_revision != baseline.tool_contract_revision
    assert (
        _binding(baseline, epoch="search-v2").policy_revision != _binding(baseline).policy_revision
    )
    unavailable = ToolBinding(
        spec=baseline,
        execute=Unavailable("credential absent"),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision="test-search-v1",
        policy_epoch=PolicyEpoch("search-v1"),
        policy_inputs={"provider": "brave", "safe_search": "moderate"},
    )
    assert unavailable.policy_revision == _binding(baseline).policy_revision
    differently_worded = ToolBinding(
        spec=baseline,
        execute=Unavailable("different private deployment detail"),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision="test-search-v1",
        policy_epoch=PolicyEpoch("search-v1"),
        policy_inputs={"provider": "brave", "safe_search": "moderate"},
    )
    assert differently_worded.policy_revision == unavailable.policy_revision


def test_binding_requires_an_explicit_nonempty_implementation_revision() -> None:
    with pytest.raises(ValueError, match="implementation revision"):
        ToolBinding(
            spec=_spec(),
            execute=Available(_unused_handler),
            replay_policy=ReplayPolicy.BilledOnce,
            implementation_revision=" ",
            policy_epoch=PolicyEpoch("search-v1"),
            policy_inputs={},
        )


def test_no_declared_error_publishes_only_boundary_failures() -> None:
    spec = ToolSpec(
        id=ToolId("tool.read"),
        summary="Read one granted tool declaration",
        documentation=PromptDocument("Read an already granted target."),
        input_type=SearchInput,
        success_type=SearchSuccess,
        error_type=NoDeclaredError,
        effect=ToolEffect.Pure,
        limits=LIMITS,
    )

    assert spec.declared_error_schema is None
    assert {
        branch["properties"]["type"]["const"] for branch in spec.error_schema.semantic["anyOf"]
    } == {"BudgetExceeded", "DeadlineExceeded", "InvalidInput", "ToolUnavailable"}


def test_frozen_revision_inputs_cannot_drift_through_exposed_nested_data() -> None:
    spec = _spec()
    binding = ToolBinding(
        spec=spec,
        execute=Available(_unused_handler),
        replay_policy=ReplayPolicy.BilledOnce,
        implementation_revision="test-search-v1",
        policy_epoch=PolicyEpoch("search-v1"),
        policy_inputs={"nested": {"modes": ["web", "news"]}},
    )
    contract_revision = spec.tool_contract_revision
    policy_revision = binding.policy_revision

    semantic = spec.input_schema.semantic
    semantic["type"] = "string"
    assert spec.input_schema.semantic["type"] == "object"
    assert spec.tool_contract_revision == contract_revision

    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], binding.policy_inputs)["nested"] = {}
    nested = binding.policy_inputs["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], nested)["modes"] = ()
    assert binding.policy_revision == policy_revision


def test_catalogue_composes_families_and_rejects_invalid_publication() -> None:
    catalog, search = _catalog()

    assert catalog.spec(ToolId("web.search")) is search
    assert tuple(catalog.family_names) == ("tool", "web")

    with pytest.raises(ValueError, match="prefix"):
        ToolCatalog.compose(
            (
                ToolFamily(
                    namespace="tool",
                    declarations=(search,),
                    bindings=(_binding(search),),
                ),
            )
        )
    with pytest.raises(ValueError, match="duplicate tool id"):
        ToolCatalog.compose(
            (
                ToolFamily(
                    namespace="web",
                    declarations=(search, search),
                    bindings=(_binding(search),),
                ),
            )
        )
    with pytest.raises(ValueError, match="unbound"):
        ToolCatalog.compose((ToolFamily(namespace="web", declarations=(search,), bindings=()),))


def test_profile_and_plan_freezing_validate_authority_and_exposure() -> None:
    catalog, search = _catalog()
    profile = CapabilityProfile(
        id=ProfileId("research"),
        grants=(
            ToolGrant(id=search.id, limits=LIMITS.tightened(max_output_bytes=16_384)),
            ToolGrant(id=ToolId("tool.read"), limits=None),
        ),
        run_limits=RUN_LIMITS,
    ).freeze(catalog)
    native = ToolPlan(profile=profile.id, exposure=Native()).freeze(catalog, profile)

    assert native.grant(search.id).limits.max_output_bytes == 16_384
    assert native.catalog_view.spec(search.id) is search
    assert native.plan_revision

    with pytest.raises(ValueError, match="absent from the catalogue"):
        CapabilityProfile(
            id=ProfileId("invalid"),
            grants=(ToolGrant(id=ToolId("web.missing"), limits=None),),
            run_limits=RUN_LIMITS,
        ).freeze(catalog)
    with pytest.raises(ValueError, match="tighten"):
        CapabilityProfile(
            id=ProfileId("wide"),
            grants=(ToolGrant(id=search.id, limits=LIMITS.tightened(max_output_bytes=65_536)),),
            run_limits=RUN_LIMITS,
        ).freeze(catalog)
    with pytest.raises(ValueError, match="boundary failure envelope"):
        CapabilityProfile(
            id=ProfileId("unreportable"),
            grants=(
                ToolGrant(
                    id=search.id,
                    limits=LIMITS.tightened(max_output_bytes=53),
                ),
            ),
            run_limits=RUN_LIMITS,
        ).freeze(catalog)
    with pytest.raises(ValueError, match="boundary failure envelope"):
        RunLimits(
            max_calls=1,
            max_external_attempts=1,
            max_input_bytes=1,
            max_output_bytes=53,
            max_in_flight=1,
            max_elapsed_seconds=1.0,
        )
    with pytest.raises(ValueError, match="discovery tools"):
        ToolPlan(
            profile=profile.id,
            exposure=Discoverable(targets=(search.id,), max_target_tools_published=1),
        ).freeze(catalog, profile)
    discovery_profile = CapabilityProfile(
        id=ProfileId("self-discovery"),
        grants=(
            ToolGrant(id=search.id, limits=None),
            ToolGrant(id=ToolId("tool.read"), limits=None),
        ),
        run_limits=RUN_LIMITS,
    ).freeze(catalog)
    with pytest.raises(ValueError, match="separate from discoverable targets"):
        ToolPlan(
            profile=discovery_profile.id,
            exposure=Discoverable(
                targets=(ToolId("tool.read"),),
                max_target_tools_published=1,
            ),
        ).freeze(catalog, discovery_profile)
    with pytest.raises(TypeError, match="exposure"):
        ToolPlan(
            profile=profile.id,
            exposure=cast(Any, (Native(), HostTable())),
        ).freeze(catalog, profile)


def test_discoverable_targets_are_frozen_before_plan_revision() -> None:
    mutable_targets = [ToolId("web.search")]
    exposure = Discoverable(
        targets=cast(tuple[ToolId, ...], mutable_targets),
        max_target_tools_published=1,
    )
    mutable_targets.append(ToolId("tool.read"))
    assert exposure.targets == (ToolId("web.search"),)


def test_family_and_profile_constructor_iterables_are_snapshotted() -> None:
    search = _spec()
    binding = _binding(search)
    declarations = [search]
    bindings = [binding]
    family = ToolFamily(
        namespace="web",
        declarations=cast(tuple[ToolSpec[Any, Any, Any], ...], declarations),
        bindings=cast(tuple[ToolBinding[Any, Any, Any], ...], bindings),
    )
    declarations.clear()
    bindings.clear()

    assert family.declarations == (search,)
    assert family.bindings == (binding,)
    catalog = ToolCatalog.compose((family,))

    grants = [ToolGrant(id=search.id, limits=None)]
    profile = CapabilityProfile(
        id=ProfileId("snapshot"),
        grants=cast(tuple[ToolGrant, ...], grants),
        run_limits=RUN_LIMITS,
    )
    grants.clear()

    assert profile.grants == (ToolGrant(id=search.id, limits=None),)
    assert profile.freeze(catalog).grant(search.id).tool_contract_revision == (
        search.tool_contract_revision
    )


def test_schema_fails_closed_on_unknown_keywords() -> None:
    class UnsupportedInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        query: str = Field(json_schema_extra={"unevaluatedProperties": False})

    with pytest.raises(UnsupportedSchema, match="unevaluatedProperties"):
        _spec(input_type=UnsupportedInput)


def test_schema_rejects_ambiguous_untagged_unions() -> None:
    class AmbiguousInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str | int

    with pytest.raises(UnsupportedSchema, match="ambiguous untagged union"):
        _spec(input_type=AmbiguousInput)
    with pytest.raises(ValueError, match="input schema must be a closed object"):
        ToolSpec(
            id=ToolId("test.scalar"),
            summary="Invalid scalar input",
            documentation=PromptDocument("Rejected at construction."),
            input_type=str,
            success_type=SearchSuccess,
            error_type=NoDeclaredError,
            effect=ToolEffect.Pure,
            limits=LIMITS,
        )


def test_literal_kind_discriminator_round_trips_and_remains_closed() -> None:
    class RecordReference(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["record"]
        id: str

    class RangeReference(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["range"]
        start: int
        count: int

    class References(BaseModel):
        model_config = ConfigDict(extra="forbid")
        values: tuple[Annotated[RecordReference | RangeReference, Field(discriminator="kind")], ...]

    schema = compile_schema(References)
    value = {"values": [{"kind": "record", "id": "one"}, {"kind": "range", "start": 0, "count": 2}]}
    decoded = strict_decode(References, schema, value)
    assert strict_encode(References, schema, decoded) == value
    for invalid in (
        {"values": [{"kind": "unknown", "id": "one"}]},
        {"values": [{"kind": "record", "id": "one", "start": 0}]},
        {"values": [{"kind": "range", "start": 0}]},
    ):
        with pytest.raises(SchemaDecodeError):
            strict_decode(References, schema, invalid)


def test_duplicate_literal_tags_and_open_maps_remain_unsupported() -> None:
    class First(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["same"]
        first: str

    class Second(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["same"]
        second: str

    class Ambiguous(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: First | Second

    class OpenMap(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: dict[str, str]

    with pytest.raises(UnsupportedSchema):
        compile_schema(Ambiguous)
    with pytest.raises(UnsupportedSchema, match="open or map-like"):
        compile_schema(OpenMap)


def test_nullable_literal_union_keeps_each_branch_closed() -> None:
    class Leaf(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["leaf"]
        text: str

    class Branch(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["branch"]
        count: int

    class Result(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: Annotated[Leaf | Branch, Field(discriminator="kind")] | None

    schema = compile_schema(Result)
    for value in (None, {"kind": "leaf", "text": "one"}, {"kind": "branch", "count": 2}):
        wire = {"value": value}
        assert strict_encode(Result, schema, strict_decode(Result, schema, wire)) == wire
    for value in ({"kind": "unknown"}, {"kind": "leaf", "text": "one", "count": 2}):
        with pytest.raises(SchemaDecodeError):
            strict_decode(Result, schema, {"value": value})


def test_nullable_object_and_array_unions_round_trip_strictly() -> None:
    class NestedValue(BaseModel):
        model_config = ConfigDict(extra="forbid")

        label: str

    class NullableContainers(BaseModel):
        model_config = ConfigDict(extra="forbid")

        nested: NestedValue | None
        items: tuple[int, ...] | None

    schema = compile_schema(NullableContainers)
    assert schema.semantic["properties"] == {
        "items": {
            "anyOf": [
                {"items": {"type": "integer"}, "type": "array"},
                {"type": "null"},
            ]
        },
        "nested": {
            "anyOf": [
                {
                    "additionalProperties": False,
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                    "type": "object",
                },
                {"type": "null"},
            ]
        },
    }

    decoded = strict_decode(
        NullableContainers,
        schema,
        {"items": [1, 2], "nested": {"label": "portable"}},
    )
    assert decoded == NullableContainers(
        items=(1, 2),
        nested=NestedValue(label="portable"),
    )
    assert strict_encode(NullableContainers, schema, decoded) == {
        "items": [1, 2],
        "nested": {"label": "portable"},
    }
    assert strict_decode(
        NullableContainers,
        schema,
        {"items": None, "nested": None},
    ) == NullableContainers(items=None, nested=None)

    class PortableMode(StrEnum):
        Exact = "exact"

    class EnumInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        mode: PortableMode

    assert strict_decode(EnumInput, compile_schema(EnumInput), {"mode": "exact"}).mode is (
        PortableMode.Exact
    )


def test_schema_rejects_invalid_keyword_values_and_type_combinations() -> None:
    class InvalidTypeInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(json_schema_extra={"type": "unicorn"})

    class InvalidConstraintInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(json_schema_extra={"minItems": 2})

    class InvalidFormatInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(json_schema_extra={"format": "provider-secret"})

    for input_type in (InvalidTypeInput, InvalidConstraintInput, InvalidFormatInput):
        with pytest.raises(UnsupportedSchema):
            _spec(input_type=input_type)


def test_schema_preserves_valid_string_patterns_in_the_semantic_projection() -> None:
    class CanonicalIdInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        id: str = Field(pattern=TOOL_ID_PATTERN)

    spec = _spec(input_type=CanonicalIdInput)

    assert spec.input_schema.semantic["properties"]["id"] == {
        "pattern": TOOL_ID_PATTERN,
        "type": "string",
    }
    assert TOOL_NAMESPACE_PATTERN == r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*_?$"
    for value in ("web.read", "web.bad__name", "Web.read", "web", "web.read.more"):
        schema_accepts = __import__("re").fullmatch(TOOL_ID_PATTERN, value) is not None
        try:
            ToolId(value)
        except ValueError:
            runtime_accepts = False
        else:
            runtime_accepts = True
        assert schema_accepts is runtime_accepts


def test_compiled_semantic_constraints_are_enforced_independently_of_type_metadata() -> None:
    class SchemaConstrained(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(
            json_schema_extra={"minLength": 4, "pattern": "^safe$"},
        )

    schema = compile_schema(SchemaConstrained)

    with pytest.raises(SchemaDecodeError):
        strict_decode(SchemaConstrained, schema, {"value": "no"})
    with pytest.raises(SchemaEncodeDefect):
        strict_encode(SchemaConstrained, schema, SchemaConstrained(value="no"))
    assert strict_decode(SchemaConstrained, schema, {"value": "safe"}).value == "safe"


def test_schema_rejects_nonportable_patterns_and_impossible_constraints() -> None:
    class LookaheadInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(json_schema_extra={"pattern": "^(?=safe)safe$"})

    class ImpossibleLengthInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str = Field(json_schema_extra={"minLength": 4, "maxLength": 2})

    for input_type in (LookaheadInput, ImpossibleLengthInput):
        with pytest.raises(UnsupportedSchema):
            compile_schema(input_type)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": True},
        {"max_input_bytes": 1.5},
        {"deadline_seconds": float("inf")},
        {"deadline_seconds": float("nan")},
    ],
)
def test_tool_limits_reject_noncanonical_numeric_configuration(
    changes: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "max_input_bytes": 10,
        "max_output_bytes": 10,
        "max_attempts": 1,
        "deadline_seconds": 1.0,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ToolLimits(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_calls": False},
        {"max_in_flight": 1.5},
        {"max_elapsed_seconds": float("inf")},
        {"max_elapsed_seconds": float("nan")},
    ],
)
def test_run_limits_reject_noncanonical_numeric_configuration(
    changes: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "max_calls": 2,
        "max_external_attempts": 2,
        "max_input_bytes": 10,
        "max_output_bytes": 10,
        "max_in_flight": 1,
        "max_elapsed_seconds": 1.0,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        RunLimits(**values)

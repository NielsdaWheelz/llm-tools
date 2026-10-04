"""Closed authority profiles and mutually exclusive exposure plans."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from llm_tools.catalog import ToolCatalog
from llm_tools.declaration import ToolBinding, ToolId, ToolLimits, ToolSpec
from llm_tools.schema import JsonObject, JsonValue, canonical_json_bytes

_PROFILE_ID = re.compile(r"[a-z][a-z0-9_-]*")
_BOUNDARY_FAILURE_MAX_BYTES = len(
    canonical_json_bytes({"type": "Failure", "error": {"type": "DeadlineExceeded"}})
)


class ProfileId(str):
    def __new__(cls, value: str) -> ProfileId:
        if not _PROFILE_ID.fullmatch(value):
            raise ValueError(f"invalid profile id: {value!r}")
        return str.__new__(cls, value)


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_calls: int | None
    max_external_attempts: int | None
    max_input_bytes: int | None
    max_output_bytes: int | None
    max_in_flight: int
    max_elapsed_seconds: float | None

    def __post_init__(self) -> None:
        cumulative_integer_limits = (
            self.max_calls,
            self.max_external_attempts,
            self.max_input_bytes,
            self.max_output_bytes,
        )
        integer_limits = tuple(value for value in cumulative_integer_limits if value is not None)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*integer_limits, self.max_in_flight)
        ):
            raise TypeError("run count and byte limits must be integers")
        if self.max_elapsed_seconds is not None:
            if isinstance(self.max_elapsed_seconds, bool) or not isinstance(
                self.max_elapsed_seconds, (int, float)
            ):
                raise TypeError("run elapsed limit must be numeric")
            if not math.isfinite(self.max_elapsed_seconds):
                raise ValueError("run elapsed limit must be finite")
        if (
            (self.max_calls is not None and self.max_calls <= 0)
            or (self.max_external_attempts is not None and self.max_external_attempts < 0)
            or (self.max_input_bytes is not None and self.max_input_bytes <= 0)
            or (self.max_output_bytes is not None and self.max_output_bytes <= 0)
            or self.max_in_flight <= 0
            or (self.max_elapsed_seconds is not None and self.max_elapsed_seconds <= 0)
        ):
            raise ValueError("run limits must be positive; attempts may be zero")
        if (
            self.max_output_bytes is not None
            and self.max_output_bytes < _BOUNDARY_FAILURE_MAX_BYTES
        ):
            raise ValueError("run limits cannot encode the largest boundary failure envelope")

    def json(self) -> JsonObject:
        return {
            "max_calls": self.max_calls,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "max_external_attempts": self.max_external_attempts,
            "max_in_flight": self.max_in_flight,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class ToolGrant:
    id: ToolId
    limits: ToolLimits | None


@dataclass(frozen=True, slots=True)
class EffectiveToolGrant:
    id: ToolId
    limits: ToolLimits
    implementation_revision: str
    tool_contract_revision: str
    policy_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_revision, str):
            raise TypeError("implementation revision must be a string")
        if not self.implementation_revision.strip():
            raise ValueError("implementation revision must not be empty")

    def json(self) -> JsonObject:
        return {
            "id": str(self.id),
            "implementation_revision": self.implementation_revision,
            "limits": self.limits.json(),
            "policy_revision": self.policy_revision,
            "tool_contract_revision": self.tool_contract_revision,
        }


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    id: ProfileId
    grants: tuple[ToolGrant, ...]
    run_limits: RunLimits

    def __post_init__(self) -> None:
        object.__setattr__(self, "grants", tuple(self.grants))

    def freeze(self, catalog: ToolCatalog) -> FrozenCapabilityProfile:
        effective: dict[ToolId, EffectiveToolGrant] = {}
        ordered: list[EffectiveToolGrant] = []
        for grant in self.grants:
            if grant.id in effective:
                raise ValueError(f"duplicate grant: {grant.id!s}")
            try:
                spec = catalog.spec(grant.id)
                binding = catalog.binding(grant.id)
            except KeyError as exc:
                raise ValueError(f"grant absent from the catalogue: {grant.id!s}") from exc
            limits = grant.limits or spec.limits
            if not limits.is_tightening_of(spec.limits):
                raise ValueError(f"profile limits may only tighten declaration: {grant.id!s}")
            if limits.max_output_bytes < _BOUNDARY_FAILURE_MAX_BYTES:
                raise ValueError("profile cannot encode the largest boundary failure envelope")
            resolved = EffectiveToolGrant(
                id=grant.id,
                limits=limits,
                implementation_revision=binding.implementation_revision,
                tool_contract_revision=spec.tool_contract_revision,
                policy_revision=binding.policy_revision,
            )
            effective[grant.id] = resolved
            ordered.append(resolved)
        revision = _revision(
            {
                "grants": [grant.json() for grant in ordered],
                "id": str(self.id),
                "run_limits": self.run_limits.json(),
            }
        )
        return FrozenCapabilityProfile(
            id=self.id,
            grants=MappingProxyType(effective),
            ordered_grants=tuple(ordered),
            run_limits=self.run_limits,
            profile_revision=revision,
        )


@dataclass(frozen=True, slots=True)
class FrozenCapabilityProfile:
    id: ProfileId
    grants: Mapping[ToolId, EffectiveToolGrant]
    ordered_grants: tuple[EffectiveToolGrant, ...]
    run_limits: RunLimits
    profile_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "grants", MappingProxyType(dict(self.grants)))
        object.__setattr__(self, "ordered_grants", tuple(self.ordered_grants))

    def grant(self, tool_id: ToolId) -> EffectiveToolGrant:
        try:
            return self.grants[tool_id]
        except KeyError as exc:
            raise KeyError(f"tool is not granted: {tool_id!s}") from exc

    def is_tightening_of(self, maximum: FrozenCapabilityProfile) -> bool:
        """Return whether this frozen authority is no wider than ``maximum``.

        Profile identities and revisions may differ; shared grants require the
        same tool-contract, implementation, and policy revisions.
        """

        try:
            self._require_integrity()
            maximum._require_integrity()
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

        if self.run_limits.max_in_flight > maximum.run_limits.max_in_flight:
            return False
        cumulative_limits = (
            (self.run_limits.max_calls, maximum.run_limits.max_calls),
            (self.run_limits.max_external_attempts, maximum.run_limits.max_external_attempts),
            (self.run_limits.max_input_bytes, maximum.run_limits.max_input_bytes),
            (self.run_limits.max_output_bytes, maximum.run_limits.max_output_bytes),
            (self.run_limits.max_elapsed_seconds, maximum.run_limits.max_elapsed_seconds),
        )
        if any(
            ceiling is not None and (value is None or value > ceiling)
            for value, ceiling in cumulative_limits
        ):
            return False
        for grant in self.ordered_grants:
            try:
                maximum_grant = maximum.grants[grant.id]
            except KeyError:
                return False
            if (
                grant.implementation_revision != maximum_grant.implementation_revision
                or grant.tool_contract_revision != maximum_grant.tool_contract_revision
                or grant.policy_revision != maximum_grant.policy_revision
                or not grant.limits.is_tightening_of(maximum_grant.limits)
            ):
                return False
        return True

    def _require_integrity(self) -> None:
        ordered_ids = tuple(grant.id for grant in self.ordered_grants)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("frozen profile has duplicate ordered grants")
        if set(self.grants) != set(ordered_ids):
            raise ValueError("frozen profile grant map differs from its ordered grants")
        if any(self.grants[grant.id] != grant for grant in self.ordered_grants):
            raise ValueError("frozen profile grant values differ from its ordered grants")
        expected_revision = _revision(
            {
                "grants": [grant.json() for grant in self.ordered_grants],
                "id": str(self.id),
                "run_limits": self.run_limits.json(),
            }
        )
        if self.profile_revision != expected_revision:
            raise ValueError("frozen profile revision does not match its contents")


@dataclass(frozen=True, slots=True)
class Native:
    pass


@dataclass(frozen=True, slots=True)
class HostTable:
    pass


@dataclass(frozen=True, slots=True)
class Discoverable:
    targets: tuple[ToolId, ...]
    max_target_tools_published: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if isinstance(self.max_target_tools_published, bool) or not isinstance(
            self.max_target_tools_published, int
        ):
            raise TypeError("discoverable publication ceiling must be an integer")


type Exposure = Native | Discoverable | HostTable


@dataclass(frozen=True, slots=True)
class PlanCatalogView:
    _specs: Mapping[ToolId, ToolSpec[Any, Any, Any]]
    _bindings: Mapping[ToolId, ToolBinding[Any, Any, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_specs", MappingProxyType(dict(self._specs)))
        object.__setattr__(self, "_bindings", MappingProxyType(dict(self._bindings)))

    @classmethod
    def from_catalog(cls, catalog: ToolCatalog, ids: tuple[ToolId, ...]) -> PlanCatalogView:
        return cls(
            _specs=MappingProxyType({tool_id: catalog.spec(tool_id) for tool_id in ids}),
            _bindings=MappingProxyType({tool_id: catalog.binding(tool_id) for tool_id in ids}),
        )

    def spec(self, tool_id: ToolId) -> ToolSpec[Any, Any, Any]:
        try:
            return self._specs[tool_id]
        except KeyError as exc:
            raise KeyError(f"tool absent from plan catalogue view: {tool_id!s}") from exc

    def binding(self, tool_id: ToolId) -> ToolBinding[Any, Any, Any]:
        try:
            return self._bindings[tool_id]
        except KeyError as exc:
            raise KeyError(f"tool absent from plan catalogue view: {tool_id!s}") from exc


@dataclass(frozen=True, slots=True)
class ToolPlan:
    profile: ProfileId
    exposure: Exposure

    def freeze(
        self,
        catalog: ToolCatalog,
        profile: FrozenCapabilityProfile,
    ) -> FrozenToolPlan:
        if self.profile != profile.id:
            raise ValueError("plan references a different frozen profile")
        profile._require_integrity()
        view_ids, exposure_json = _exposure_projection(profile, self.exposure)

        plan = FrozenToolPlan(
            profile=profile,
            exposure=self.exposure,
            catalog_view=PlanCatalogView.from_catalog(catalog, view_ids),
            plan_revision=_plan_revision(profile.profile_revision, exposure_json),
        )
        plan._require_integrity()
        return plan


@dataclass(frozen=True, slots=True)
class FrozenToolPlan:
    profile: FrozenCapabilityProfile
    exposure: Exposure
    catalog_view: PlanCatalogView
    plan_revision: str

    def grant(self, tool_id: ToolId) -> EffectiveToolGrant:
        return self.profile.grant(tool_id)

    def is_tightening_of(self, maximum_profile: FrozenCapabilityProfile) -> bool:
        """Prove a consistent plan's authority is no wider than ``maximum_profile``."""

        try:
            self._require_integrity()
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

        return self.profile.is_tightening_of(maximum_profile)

    def _require_integrity(self) -> None:
        self.profile._require_integrity()
        view_ids, exposure_json = _exposure_projection(self.profile, self.exposure)
        expected_ids = set(view_ids)
        if set(self.catalog_view._specs) != expected_ids:
            raise ValueError("frozen plan catalogue specifications differ from its exposure")
        if set(self.catalog_view._bindings) != expected_ids:
            raise ValueError("frozen plan catalogue bindings differ from its exposure")

        for tool_id in view_ids:
            grant = self.profile.grant(tool_id)
            spec = self.catalog_view.spec(tool_id)
            binding = self.catalog_view.binding(tool_id)
            if spec.id != tool_id or binding.spec.id != tool_id:
                raise ValueError("frozen plan catalogue entry has a mismatched tool id")
            if spec.tool_contract_revision != grant.tool_contract_revision:
                raise ValueError("frozen plan specification differs from its authorized contract")
            if binding.spec.tool_contract_revision != grant.tool_contract_revision:
                raise ValueError("frozen plan binding differs from its authorized contract")
            if binding.spec.documentation_revision != spec.documentation_revision:
                raise ValueError("frozen plan binding uses a stale specification")
            if binding.implementation_revision != grant.implementation_revision:
                raise ValueError("frozen plan binding differs from its authorized implementation")
            if binding.policy_revision != grant.policy_revision:
                raise ValueError("frozen plan binding differs from its authorized policy")
            if not grant.limits.is_tightening_of(spec.limits):
                raise ValueError("frozen plan grant widens its declaration limits")

        expected_revision = _plan_revision(self.profile.profile_revision, exposure_json)
        if self.plan_revision != expected_revision:
            raise ValueError("frozen plan revision does not match its contents")


def _exposure_projection(
    profile: FrozenCapabilityProfile,
    exposure: Exposure,
) -> tuple[tuple[ToolId, ...], JsonObject]:
    if not isinstance(exposure, (Native, Discoverable, HostTable)):
        raise TypeError("plan exposure must be exactly one supported variant")

    grant_ids = tuple(grant.id for grant in profile.ordered_grants)
    if isinstance(exposure, Discoverable):
        targets = exposure.targets
        discovery_ids = {ToolId("tool.search"), ToolId("tool.read")}
        if len(set(targets)) != len(targets):
            raise ValueError("discoverable targets must be unique")
        if discovery_ids.intersection(targets):
            raise ValueError("discovery tools must remain separate from discoverable targets")
        missing_discovery = discovery_ids - set(grant_ids)
        if missing_discovery:
            raise ValueError("Discoverable plans must grant both discovery tools")
        if not set(targets).issubset(profile.grants):
            raise ValueError("discoverable targets must be a subset of profile grants")
        if not 0 <= exposure.max_target_tools_published <= len(targets):
            raise ValueError("discoverable publication ceiling must fit the target set")
        return (
            tuple(dict.fromkeys((ToolId("tool.search"), ToolId("tool.read"), *targets))),
            {
                "max_target_tools_published": exposure.max_target_tools_published,
                "targets": [str(tool_id) for tool_id in targets],
                "type": "Discoverable",
            },
        )
    if isinstance(exposure, Native):
        return grant_ids, {"type": "Native"}
    return grant_ids, {"type": "HostTable"}


def _plan_revision(profile_revision: str, exposure_json: JsonObject) -> str:
    return _revision(
        {
            "exposure": exposure_json,
            "profile_revision": profile_revision,
        }
    )


def _revision(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

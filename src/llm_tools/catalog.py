"""Explicit immutable tool-family aggregation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from llm_tools.declaration import TOOL_NAMESPACE_PATTERN, ToolBinding, ToolId, ToolSpec

_NAMESPACE = re.compile(TOOL_NAMESPACE_PATTERN)


@dataclass(frozen=True, slots=True)
class ToolFamily:
    namespace: str
    declarations: tuple[ToolSpec[Any, Any, Any], ...]
    bindings: tuple[ToolBinding[Any, Any, Any], ...]

    def __post_init__(self) -> None:
        if not _NAMESPACE.fullmatch(self.namespace):
            raise ValueError(f"invalid tool family namespace: {self.namespace!r}")
        object.__setattr__(self, "declarations", tuple(self.declarations))
        object.__setattr__(self, "bindings", tuple(self.bindings))


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    _families: Mapping[str, ToolFamily]
    _specs: Mapping[ToolId, ToolSpec[Any, Any, Any]]
    _bindings: Mapping[ToolId, ToolBinding[Any, Any, Any]]

    @classmethod
    def compose(cls, families: Iterable[ToolFamily]) -> ToolCatalog:
        family_map: dict[str, ToolFamily] = {}
        specs: dict[ToolId, ToolSpec[Any, Any, Any]] = {}
        bindings: dict[ToolId, ToolBinding[Any, Any, Any]] = {}

        for family in families:
            if family.namespace in family_map:
                raise ValueError(f"duplicate family name: {family.namespace}")
            family_map[family.namespace] = family

            declaration_ids: set[ToolId] = set()
            for spec in family.declarations:
                if str(spec.id).split(".", 1)[0] != family.namespace:
                    raise ValueError(
                        f"tool id {spec.id!s} does not match family prefix {family.namespace!r}"
                    )
                if spec.id in declaration_ids or spec.id in specs:
                    raise ValueError(f"duplicate tool id: {spec.id!s}")
                declaration_ids.add(spec.id)
                specs[spec.id] = spec

            family_binding_ids: set[ToolId] = set()
            for binding in family.bindings:
                tool_id = binding.spec.id
                if tool_id in family_binding_ids or tool_id in bindings:
                    raise ValueError(f"duplicate binding for tool id: {tool_id!s}")
                if tool_id not in declaration_ids:
                    raise ValueError(f"binding without declaration: {tool_id!s}")
                if binding.spec is not specs[tool_id]:
                    raise ValueError(f"binding uses a stale declaration: {tool_id!s}")
                family_binding_ids.add(tool_id)
                bindings[tool_id] = binding

            missing = declaration_ids - family_binding_ids
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"unbound declarations: {names}")

        return cls(
            _families=MappingProxyType(dict(sorted(family_map.items()))),
            _specs=MappingProxyType(dict(sorted(specs.items()))),
            _bindings=MappingProxyType(dict(sorted(bindings.items()))),
        )

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(self._families)

    @property
    def tool_ids(self) -> tuple[ToolId, ...]:
        return tuple(self._specs)

    def spec(self, tool_id: ToolId) -> ToolSpec[Any, Any, Any]:
        try:
            return self._specs[tool_id]
        except KeyError as exc:
            raise KeyError(f"unknown tool id: {tool_id!s}") from exc

    def binding(self, tool_id: ToolId) -> ToolBinding[Any, Any, Any]:
        try:
            return self._bindings[tool_id]
        except KeyError as exc:
            raise KeyError(f"unknown tool id: {tool_id!s}") from exc

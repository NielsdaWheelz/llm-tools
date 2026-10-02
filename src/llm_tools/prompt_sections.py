"""Typed, fixed-tag prompt presentation with one escaping boundary."""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from llm_tools.schema import JsonValue, canonical_json_bytes

_NAME = re.compile(r"[a-z][a-z0-9_]*")


class PromptSectionKind(str):
    """A semantic section kind rendered as data, never as a tag name."""

    def __new__(cls, value: str) -> PromptSectionKind:
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise ValueError(f"invalid prompt section kind: {value!r}")
        return str.__new__(cls, value)


class PromptAttributeName(str):
    """A validated attribute identity owned by prompt-construction code."""

    def __new__(cls, value: str) -> PromptAttributeName:
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise ValueError(f"invalid prompt attribute name: {value!r}")
        return str.__new__(cls, value)


type PromptAttributeValue = bool | int | float | str


@dataclass(frozen=True, slots=True)
class PromptAttribute:
    name: PromptAttributeName
    value: PromptAttributeValue

    def __post_init__(self) -> None:
        if not isinstance(self.name, PromptAttributeName):
            raise TypeError("prompt attribute name must be a PromptAttributeName")
        if isinstance(self.value, str):
            _validate_xml_text(self.value)
            return
        if isinstance(self.value, bool) or isinstance(self.value, int):
            return
        if isinstance(self.value, float) and math.isfinite(self.value):
            return
        raise TypeError("prompt attribute value must be a finite JSON scalar")


@dataclass(frozen=True, slots=True)
class PromptText:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("prompt text must be a string")
        _validate_xml_text(self.text)


@dataclass(frozen=True, slots=True, init=False)
class PromptJson:
    """A canonical immutable snapshot of a JSON value."""

    _canonical_text: str

    def __init__(self, value: JsonValue) -> None:
        object.__setattr__(self, "_canonical_text", canonical_json_bytes(value).decode("utf-8"))

    @property
    def value(self) -> JsonValue:
        """Return an independent JSON copy for durable host context."""
        return json.loads(self._canonical_text)


@dataclass(frozen=True, slots=True)
class PromptSection:
    kind: PromptSectionKind
    attributes: tuple[PromptAttribute, ...]
    body: PromptContent | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PromptSectionKind):
            raise TypeError("prompt section kind must be a PromptSectionKind")
        attributes = tuple(self.attributes)
        if not all(isinstance(attribute, PromptAttribute) for attribute in attributes):
            raise TypeError("prompt section attributes must be PromptAttribute values")
        names = [attribute.name for attribute in attributes]
        if PromptAttributeName("kind") in names:
            raise ValueError("kind is reserved by the fixed prompt-section frame")
        if len(set(names)) != len(names):
            raise ValueError("prompt section attribute names must be unique")
        if self.body is not None and not isinstance(
            self.body,
            (PromptText, PromptJson, PromptSections),
        ):
            raise TypeError("prompt section body must be typed prompt content")
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class PromptSections:
    sections: tuple[PromptSection, ...]

    def __init__(self, sections: Iterable[PromptSection]) -> None:
        sections = tuple(sections)
        if not all(isinstance(section, PromptSection) for section in sections):
            raise TypeError("prompt sections must contain PromptSection values")
        object.__setattr__(self, "sections", sections)


type PromptContent = PromptText | PromptJson | PromptSections


def render_prompt(value: PromptSection | PromptContent) -> str:
    """Render a typed prompt value through the sole XML-like escaping boundary."""

    if isinstance(value, PromptSection):
        return _render_section(value)
    if isinstance(value, PromptText):
        return _escape_text(value.text)
    if isinstance(value, PromptJson):
        return _escape_text(value._canonical_text)
    if isinstance(value, PromptSections):
        return "\n".join(_render_section(section) for section in value.sections)
    raise TypeError("render_prompt requires a typed prompt value")


def _render_section(section: PromptSection) -> str:
    attributes = [
        f'kind="{_escape_attribute(section.kind)}"',
        *(
            f'{attribute.name}="{_escape_attribute(_attribute_text(attribute.value))}"'
            for attribute in sorted(section.attributes, key=lambda item: item.name)
        ),
    ]
    opening = f"<section {' '.join(attributes)}"
    if section.body is None:
        return f"{opening} />"
    return f"{opening}>\n{render_prompt(section.body)}\n</section>"


def _attribute_text(value: PromptAttributeValue) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _escape_text(value: str) -> str:
    return html.escape(value, quote=False)


def _escape_attribute(value: str) -> str:
    return (
        html.escape(value, quote=True)
        .replace("\t", "&#9;")
        .replace("\n", "&#10;")
        .replace("\r", "&#13;")
    )


def _validate_xml_text(value: str) -> None:
    for character in value:
        codepoint = ord(character)
        if (
            codepoint in {0xFFFE, 0xFFFF}
            or 0xD800 <= codepoint <= 0xDFFF
            or (codepoint < 0x20 and character not in "\t\n\r")
        ):
            raise ValueError("prompt value contains text outside the XML-like presentation set")

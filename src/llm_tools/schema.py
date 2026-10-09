"""Strict portable JSON Schema compilation and canonical JSON encoding."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import rfc8785
from pydantic import TypeAdapter, ValidationError

type JsonPrimitive = None | bool | int | float | str
# JSON is runtime-validated at every public ingress. Keeping this recursive alias
# open avoids invariant-container false positives in consumers while the
# canonicalizer remains the authority for representability.
type JsonValue = Any
type JsonObject = dict[str, Any]

_ANNOTATION_KEYWORDS = frozenset({"title", "description", "examples"})
_SCALAR_KEYWORDS = frozenset(
    {
        "type",
        "const",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
        "pattern",
    }
)
_STRUCTURAL_KEYWORDS = frozenset(
    {"properties", "required", "additionalProperties", "items", "enum", "anyOf", "oneOf"}
)
_IGNORED_GENERATOR_ANNOTATIONS = frozenset({"default", "discriminator"})
_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
_FORMATS = frozenset({"date", "date-time", "email", "hostname", "ipv4", "ipv6", "uri", "uuid"})


class UnsupportedSchema(ValueError):
    """The generated schema is outside the versioned portable subset."""


class SchemaDecodeError(ValueError):
    """An untrusted JSON value does not satisfy its strict owned type."""


class SchemaEncodeDefect(RuntimeError):
    """An owned value cannot be encoded according to its declared schema."""


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    _semantic: bytes
    _presentation: bytes

    @property
    def semantic(self) -> JsonObject:
        value = json.loads(self._semantic)
        if not isinstance(value, dict):
            raise AssertionError("compiled semantic schema root must remain an object")
        return value

    @property
    def presentation(self) -> JsonObject:
        value = json.loads(self._presentation)
        if not isinstance(value, dict):
            raise AssertionError("compiled presentation schema root must remain an object")
        return value


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode a JSON value with RFC 8785/JCS semantics."""

    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def compile_schema(python_type: Any) -> CompiledSchema:
    """Compile one Python type into semantic and presentation projections."""

    try:
        raw = TypeAdapter(python_type).json_schema()
    except Exception as exc:
        raise UnsupportedSchema(f"cannot generate schema for {python_type!r}") from exc
    if not isinstance(raw, dict):
        raise UnsupportedSchema("schema root must be an object")

    definitions = raw.get("$defs", {})
    if not isinstance(definitions, dict):
        raise UnsupportedSchema("$defs must be an object")
    root = {key: value for key, value in raw.items() if key != "$defs"}
    semantic = _normalize(root, definitions, presentation=False, resolving=())
    presentation = _normalize(root, definitions, presentation=True, resolving=())
    if not isinstance(semantic, dict) or not isinstance(presentation, dict):
        raise UnsupportedSchema("schema root must normalize to an object")
    return CompiledSchema(
        _semantic=canonical_json_bytes(semantic),
        _presentation=canonical_json_bytes(presentation),
    )


def strict_decode(python_type: Any, schema: CompiledSchema, value: JsonValue) -> Any:
    """Decode untrusted JSON without coercion or undeclared omission."""

    _verify_required_shape(schema.semantic, value)
    try:
        return TypeAdapter(python_type).validate_json(canonical_json_bytes(value), strict=True)
    except (ValidationError, ValueError) as exc:
        raise SchemaDecodeError("input does not satisfy its declared schema") from exc


def strict_encode(python_type: Any, schema: CompiledSchema, value: Any) -> JsonValue:
    """Encode an owned result; invalid owned output is a defect."""

    adapter = TypeAdapter(python_type)
    try:
        validated = adapter.validate_python(value, strict=True)
        encoded = adapter.dump_python(validated, mode="json")
        _verify_required_shape(schema.semantic, encoded)
        canonical_json_bytes(encoded)
    except (ValidationError, SchemaDecodeError, TypeError, ValueError) as exc:
        raise SchemaEncodeDefect("owned output violates its declared schema") from exc
    return _as_json_value(encoded)


def _normalize(
    node: object,
    definitions: Mapping[str, object],
    *,
    presentation: bool,
    resolving: tuple[str, ...],
) -> JsonValue:
    if not isinstance(node, dict):
        raise UnsupportedSchema("every schema node must be an object")

    if "$ref" in node:
        if len(node) != 1:
            raise UnsupportedSchema("$ref siblings are unsupported")
        reference = node["$ref"]
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise UnsupportedSchema("only local generated references can be inlined")
        name = reference.removeprefix(prefix)
        if name in resolving:
            raise UnsupportedSchema("recursive $ref is outside the portable subset")
        target = definitions.get(name)
        if target is None:
            raise UnsupportedSchema(f"unknown $ref target: {name}")
        return _normalize(
            target,
            definitions,
            presentation=presentation,
            resolving=(*resolving, name),
        )

    allowed = _SCALAR_KEYWORDS | _STRUCTURAL_KEYWORDS | _ANNOTATION_KEYWORDS
    unknown = set(node) - allowed - _IGNORED_GENERATOR_ANNOTATIONS
    if unknown:
        raise UnsupportedSchema(f"unsupported schema keyword: {sorted(unknown)[0]}")

    normalized: JsonObject = {}
    for key in sorted(node):
        value = node[key]
        if key in _IGNORED_GENERATOR_ANNOTATIONS:
            continue
        if key in _ANNOTATION_KEYWORDS:
            if presentation:
                normalized[key] = _as_json_value(value)
            continue
        if key == "properties":
            if not isinstance(value, dict):
                raise UnsupportedSchema("properties must be an object")
            normalized[key] = {
                str(name): _normalize(
                    child,
                    definitions,
                    presentation=presentation,
                    resolving=resolving,
                )
                for name, child in sorted(value.items())
            }
            continue
        if key == "required":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise UnsupportedSchema("required must contain property names")
            normalized[key] = sorted(set(value))
            continue
        if key == "additionalProperties":
            if value is not False:
                raise UnsupportedSchema("open or map-like objects are outside the portable subset")
            normalized[key] = False
            continue
        if key == "items":
            normalized[key] = _normalize(
                value,
                definitions,
                presentation=presentation,
                resolving=resolving,
            )
            continue
        if key in {"anyOf", "oneOf"}:
            if not isinstance(value, list) or not value:
                raise UnsupportedSchema(f"{key} must contain schema branches")
            branches = [
                _normalize(
                    branch,
                    definitions,
                    presentation=presentation,
                    resolving=resolving,
                )
                for branch in value
            ]
            _validate_union(branches, keyword=key)
            normalized[key] = sorted(branches, key=canonical_json_bytes)
            continue
        if key == "enum":
            if not isinstance(value, list) or not value:
                raise UnsupportedSchema("enum must contain values")
            enum_values = [_as_json_value(item) for item in value]
            normalized[key] = sorted(enum_values, key=canonical_json_bytes)
            continue
        normalized[key] = _as_json_value(value)

    if normalized.get("type") == "object" or "properties" in normalized:
        properties = normalized.get("properties", {})
        if not isinstance(properties, dict):
            raise UnsupportedSchema("object properties must be an object")
        normalized["additionalProperties"] = False
        normalized["required"] = sorted(properties)

    _validate_keywords(normalized)

    return {key: normalized[key] for key in sorted(normalized)}


def _validate_union(branches: list[JsonValue], *, keyword: str) -> None:
    flattened: list[JsonValue] = []
    for branch in branches:
        if (
            isinstance(branch, dict)
            and set(branch) == {keyword}
            and isinstance(branch[keyword], list)
        ):
            flattened.extend(branch[keyword])
        else:
            flattened.append(branch)
    branches[:] = flattened
    branch_objects = [branch for branch in branches if isinstance(branch, dict)]
    if len(branch_objects) != len(branches):
        raise UnsupportedSchema(f"{keyword} branches must be schemas")

    branch_types = [branch.get("type") for branch in branch_objects]
    if len(branches) == 2 and "null" in branch_types:
        other = branch_types[0] if branch_types[1] == "null" else branch_types[1]
        if other in {"string", "integer", "number", "boolean", "object", "array"}:
            return

    properties_by_branch: list[dict[str, JsonValue]] = []
    discriminators: set[str] | None = None
    for branch in branch_objects:
        if branch.get("type") != "object":
            raise UnsupportedSchema("ambiguous untagged union is outside the portable subset")
        properties = branch.get("properties")
        if not isinstance(properties, dict):
            raise UnsupportedSchema("tagged union branch must be an object")
        properties_by_branch.append(properties)
        literal_names = {
            name
            for name, child in properties.items()
            if isinstance(child, dict) and "const" in child
        }
        discriminators = literal_names if discriminators is None else discriminators & literal_names
    assert discriminators is not None
    for name in discriminators:
        tags = {
            canonical_json_bytes(properties[name]["const"]) for properties in properties_by_branch
        }
        if len(tags) == len(branches):
            return
    raise UnsupportedSchema("tagged object union needs a common unique literal discriminator")


def _validate_keywords(schema: JsonObject) -> None:
    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in _TYPES:
        raise UnsupportedSchema(f"unsupported schema type: {schema_type!r}")
    if "format" in schema:
        if schema_type != "string" or schema["format"] not in _FORMATS:
            raise UnsupportedSchema(f"unsupported schema format: {schema['format']!r}")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if schema_type != "string" or not isinstance(pattern, str) or len(pattern) > 512:
            raise UnsupportedSchema("schema pattern must be a bounded string constraint")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise UnsupportedSchema("schema pattern must be valid") from exc
        if any(token in pattern for token in ("(?=", "(?!", "(?<", "(?P", "(?i")) or re.search(
            r"\\[1-9]", pattern
        ):
            raise UnsupportedSchema("schema pattern is outside the portable regex subset")

    compatibility = {
        "properties": {"object"},
        "required": {"object"},
        "additionalProperties": {"object"},
        "items": {"array"},
        "minItems": {"array"},
        "maxItems": {"array"},
        "minLength": {"string"},
        "maxLength": {"string"},
        "pattern": {"string"},
        "minimum": {"integer", "number"},
        "maximum": {"integer", "number"},
        "exclusiveMinimum": {"integer", "number"},
        "exclusiveMaximum": {"integer", "number"},
        "multipleOf": {"integer", "number"},
    }
    for keyword, allowed_types in compatibility.items():
        if keyword in schema and schema_type not in allowed_types:
            raise UnsupportedSchema(
                f"schema keyword {keyword} is incompatible with type {schema_type!r}"
            )
    integer_keywords = {"minLength", "maxLength", "minItems", "maxItems"}
    for keyword in integer_keywords & schema.keys():
        value = schema[keyword]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UnsupportedSchema(f"schema keyword {keyword} must be a nonnegative integer")
    numeric_keywords = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
    for keyword in numeric_keywords & schema.keys():
        value = schema[keyword]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedSchema(f"schema keyword {keyword} must be numeric")
        if not math.isfinite(value):
            raise UnsupportedSchema(f"schema keyword {keyword} must be finite")
    if "multipleOf" in schema and schema["multipleOf"] <= 0:
        raise UnsupportedSchema("schema multipleOf must be positive")
    for minimum, maximum in (("minLength", "maxLength"), ("minItems", "maxItems")):
        if minimum in schema and maximum in schema and schema[minimum] > schema[maximum]:
            raise UnsupportedSchema(f"schema {minimum} must not exceed {maximum}")


def _verify_required_shape(schema: JsonValue, value: object) -> None:
    if not isinstance(schema, dict):
        return
    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list):
            matching = sum(_shape_matches(branch, value) for branch in branches)
            if matching == 0 or (union_key == "oneOf" and matching != 1):
                raise SchemaDecodeError("value does not match the declared union")
            return

    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _value_matches_type(schema_type, value):
        raise SchemaDecodeError(f"value is not a declared {schema_type}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise SchemaDecodeError("value does not equal the declared constant")
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_json_equal(value, member) for member in enum):
        raise SchemaDecodeError("value is outside the declared enum")

    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise SchemaDecodeError("value must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise SchemaDecodeError("compiled object schema is invalid")
        if set(value) != set(properties) or any(name not in value for name in required):
            raise SchemaDecodeError("object keys do not match the closed schema")
        for name, child in properties.items():
            _verify_required_shape(child, value[name])
        return
    items = schema.get("items")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaDecodeError("array is shorter than its declared minimum")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaDecodeError("array is longer than its declared maximum")
    if isinstance(items, dict) and isinstance(value, list):
        for item in value:
            _verify_required_shape(items, item)
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaDecodeError("string is shorter than its declared minimum")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaDecodeError("string is longer than its declared maximum")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise SchemaDecodeError("string does not match its declared pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaDecodeError("number is below its declared minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaDecodeError("number is above its declared maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaDecodeError("number is not above its declared exclusive minimum")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise SchemaDecodeError("number is not below its declared exclusive maximum")
        multiple = schema.get("multipleOf")
        if isinstance(multiple, (int, float)) and not math.isclose(
            value / multiple,
            round(value / multiple),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SchemaDecodeError("number is not a declared multiple")


def _shape_matches(schema: object, value: object) -> bool:
    try:
        _verify_required_shape(_as_json_value(schema), value)
    except SchemaDecodeError:
        return False
    return True


def _value_matches_type(schema_type: str, value: object) -> bool:
    matches: dict[str, bool] = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        "boolean": isinstance(value, bool),
    }
    return matches.get(schema_type, False)


def _json_equal(left: object, right: object) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except ValueError as exc:
        raise SchemaDecodeError("value is not canonical JSON") from exc


def _as_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise UnsupportedSchema("JSON object keys must be strings")
        return {str(key): _as_json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_json_value(child) for child in value]
    raise UnsupportedSchema(f"non-JSON schema value: {value!r}")

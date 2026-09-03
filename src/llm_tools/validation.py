"""Pure strict validation of model-proposed tool arguments."""

from __future__ import annotations

from llm_tools.declaration import ToolBinding
from llm_tools.schema import JsonValue, strict_decode


def validate_tool_input[InputT, SuccessT, ErrorT](
    binding: ToolBinding[InputT, SuccessT, ErrorT],
    arguments: JsonValue,
) -> InputT:
    """Decode arguments through the binding's declared strict input schema."""

    return strict_decode(binding.spec.input_type, binding.spec.input_schema, arguments)

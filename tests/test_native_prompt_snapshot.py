"""Temporary cross-consumer durable prompt snapshot acceptance."""

from llm_tools import PromptJson, PromptSection, PromptSectionKind, PromptSections, render_prompt


def test_durable_prompt_json_uses_public_copy() -> None:
    body = PromptJson({"nested": ["original"]})
    value = body.value
    value["nested"].append("changed")
    assert body.value == {"nested": ["original"]}
    assert "changed" not in render_prompt(
        PromptSections((PromptSection(PromptSectionKind("proof"), (), body),))
    )

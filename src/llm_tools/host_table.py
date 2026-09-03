"""Frozen host-table publication through typed prompt sections."""

from __future__ import annotations

from llm_tools.profiles import FrozenToolPlan, HostTable
from llm_tools.prompt_sections import PromptJson, PromptSection, PromptSectionKind
from llm_tools.schema import JsonObject


def publish_host_table(plan: FrozenToolPlan) -> PromptSection:
    """Publish one exact HostTable plan as immutable model-facing prompt data."""

    if not isinstance(plan.exposure, HostTable):
        raise ValueError("host-table publication requires HostTable exposure")

    tools: list[JsonObject] = []
    for grant in plan.profile.ordered_grants:
        spec = plan.catalog_view.spec(grant.id)
        binding = plan.catalog_view.binding(grant.id)
        tools.append(
            {
                "declared_limits": spec.limits.json(),
                "documentation": spec.documentation.text,
                "documentation_revision": spec.documentation_revision,
                "effect": spec.effect.value,
                "effective_limits": grant.limits.json(),
                "error_schema": spec.error_schema.presentation,
                "id": str(grant.id),
                "input_schema": spec.input_schema.presentation,
                "policy_revision": binding.policy_revision,
                "replay_policy": binding.replay_policy.value,
                "success_schema": spec.success_schema.presentation,
                "summary": spec.summary,
                "tool_contract_revision": spec.tool_contract_revision,
            }
        )

    return PromptSection(
        kind=PromptSectionKind("host_table"),
        attributes=(),
        body=PromptJson(
            {
                "count": len(tools),
                "plan_revision": plan.plan_revision,
                "profile_revision": plan.profile.profile_revision,
                "tools": tools,
            }
        ),
    )

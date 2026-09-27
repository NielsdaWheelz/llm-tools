"""Public package identity and facade tests."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

PUBLIC_FACADE = {
    "Available",
    "BraveSearchProvider",
    "BoundaryFailure",
    "BudgetState",
    "Cancellation",
    "CapabilityProfile",
    "CredentialRejected",
    "DirectConnector",
    "DeclaredToolFailure",
    "Discoverable",
    "EffectId",
    "EffectiveToolGrant",
    "EvidenceReceipt",
    "ExecutionContext",
    "ExecutorConfigurationDefect",
    "Exposure",
    "FrozenCapabilityProfile",
    "FrozenToolPlan",
    "HostTable",
    "HttpApi",
    "HandlerSuccess",
    "InvalidUpstreamResponse",
    "InvalidUrl",
    "InvocationPosition",
    "MalformedJson",
    "Native",
    "NoDeclaredError",
    "ParsedJson",
    "PlanCatalogView",
    "PolicyEpoch",
    "PositionConflictDefect",
    "PositionRecorder",
    "PositionState",
    "Principal",
    "ProfileId",
    "PromptAttribute",
    "PromptAttributeName",
    "PromptDocument",
    "PromptJson",
    "PromptSection",
    "PromptSectionKind",
    "PromptSections",
    "PromptText",
    "RateLimited",
    "RawToolInput",
    "RecoveryRequired",
    "ReplayPolicy",
    "Reservation",
    "RunLimits",
    "SafeWebReader",
    "Scope",
    "SchemaDecodeError",
    "Settlement",
    "SystemResolver",
    "Telemetry",
    "TOOL_FAMILY",
    "TOOL_READ_SPEC",
    "TOOL_SEARCH_SPEC",
    "TooLarge",
    "ToolBinding",
    "ToolCatalog",
    "ToolEffect",
    "ToolExecutor",
    "ToolFamily",
    "ToolGrant",
    "ToolId",
    "ToolLimits",
    "ToolPlan",
    "ToolReadInput",
    "ToolReadSuccess",
    "ToolSearchInput",
    "ToolSearchMatch",
    "ToolSearchSuccess",
    "ToolSpec",
    "ToolResult",
    "Unavailable",
    "UnsafeDestination",
    "UnsupportedContent",
    "UpstreamUnavailable",
    "WEB_READ_SPEC",
    "WEB_SEARCH_SPEC",
    "WebReadLimits",
    "WebReadInput",
    "WebReadSuccess",
    "WebSearchError",
    "WebSearchErrorCode",
    "WebSearchInput",
    "WebSearchProvider",
    "WebSearchRequest",
    "WebSearchResponse",
    "WebSearchResultItem",
    "WebSearchResultType",
    "bind_brave_web_search",
    "bind_web_read",
    "canonical_json_bytes",
    "publish_host_table",
    "published_tool_ids",
    "raw_input_digest",
    "render_prompt",
    "sha256_hex",
    "validate_tool_input",
    "web_family",
}


def test_llm_tools_public_facade() -> None:
    import llm_tools

    assert set(llm_tools.__all__) == PUBLIC_FACADE
    assert llm_tools.BraveSearchProvider.__module__ == "llm_tools.web.brave"
    assert llm_tools.WebSearchRequest.__module__ == "llm_tools.web.contracts"


def test_public_execution_boundary_and_input_digest_are_stable() -> None:
    import llm_tools

    raw = llm_tools.ParsedJson({"query": "one"})
    expected = hashlib.sha256(
        llm_tools.canonical_json_bytes({"type": "ParsedJson", "value": {"query": "one"}})
    ).hexdigest()

    assert llm_tools.raw_input_digest(raw) == expected
    failure = llm_tools.BoundaryFailure("InvalidInput", actual_attempts=0)
    assert (failure.error_type, failure.actual_attempts) == ("InvalidInput", 0)


@pytest.mark.asyncio
async def test_public_facade_is_sufficient_to_author_and_execute_a_binding() -> None:
    import llm_tools
    from llm_tools.testing import (
        InMemoryBudgetState,
        InMemoryPositionRecorder,
        NeverCancelled,
        RecordingTelemetry,
    )

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    class Success(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    async def handler(
        value: Input,
        context: llm_tools.ExecutionContext,
    ) -> llm_tools.HandlerSuccess[Success]:
        del context
        return llm_tools.HandlerSuccess(Success(value=value.value), actual_attempts=0)

    limits = llm_tools.ToolLimits(1_024, 1_024, 0, 1.0)
    spec = llm_tools.ToolSpec(
        id=llm_tools.ToolId("test.echo"),
        summary="Echo one value",
        documentation=llm_tools.PromptDocument("Echo the validated input."),
        input_type=Input,
        success_type=Success,
        error_type=llm_tools.NoDeclaredError,
        effect=llm_tools.ToolEffect.Pure,
        limits=limits,
    )
    binding = llm_tools.ToolBinding(
        spec=spec,
        execute=llm_tools.Available(handler),
        replay_policy=llm_tools.ReplayPolicy.ReDispatchable,
        implementation_revision="test-package-v1",
        policy_epoch=llm_tools.PolicyEpoch("v1"),
        policy_inputs={},
    )
    catalog = llm_tools.ToolCatalog.compose((llm_tools.ToolFamily("test", (spec,), (binding,)),))
    run_limits = llm_tools.RunLimits(1, 0, 1_024, 1_024, 1, 5.0)
    profile = llm_tools.CapabilityProfile(
        llm_tools.ProfileId("test"),
        (llm_tools.ToolGrant(spec.id, None),),
        run_limits,
    ).freeze(catalog)
    plan = llm_tools.ToolPlan(profile.id, llm_tools.Native()).freeze(catalog, profile)
    context = llm_tools.ExecutionContext(
        plan=plan,
        grant=plan.grant(spec.id),
        catalog_view=plan.catalog_view,
        position=llm_tools.InvocationPosition("turn-1/call-1"),
        recorder=InMemoryPositionRecorder(),
        effect_id=None,
        budgets=InMemoryBudgetState(run_limits),
        principal=llm_tools.Principal("test"),
        scope=llm_tools.Scope("test"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )

    assert await llm_tools.ToolExecutor.execute(
        binding,
        llm_tools.ParsedJson({"value": "portable"}),
        context,
    ) == {"type": "Success", "value": {"value": "portable"}}


def test_built_wheel_exposes_only_the_new_package_identity(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for isolated wheel verification")

    project_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "dist"
    environment = tmp_path / "wheel-environment"
    subprocess.run(
        [uv, "build", "--no-sources", "--out-dir", str(wheel_dir)],
        check=True,
        cwd=project_root,
    )
    subprocess.run(
        [uv, "venv", "--python", sys.executable, str(environment)],
        check=True,
        cwd=project_root,
    )
    wheel = next(wheel_dir.glob("llm_tools-*.whl"))
    python = environment / "bin" / "python"
    subprocess.run([uv, "pip", "install", "--python", str(python), str(wheel)], check=True)
    subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util, json, llm_tools; "
            "assert importlib.util.find_spec('web_search_tool') is None; "
            f"assert set(llm_tools.__all__) == set(json.loads({str(__import__('json').dumps(__import__('json').dumps(sorted(PUBLIC_FACADE))))})); "
            "assert llm_tools.BraveSearchProvider.__module__ == 'llm_tools.web.brave'; "
            "assert llm_tools.WebSearchRequest.__module__ == 'llm_tools.web.contracts'",
        ],
        check=True,
    )

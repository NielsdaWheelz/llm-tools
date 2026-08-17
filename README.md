# llm-tools

A provider-neutral typed tool kernel with four portable tools: `web.search`, `web.read`,
`tool.search`, and `tool.read`.

Applications compose immutable tool families, freeze closed capability profiles, and execute calls
through one strict result, budget, and replay boundary. Declarations remain separate from owned
bindings; there is no ambient registry or model-provider dependency.

## Install

```bash
uv add llm-tools
```

## Quick start

This example follows the kernel path: bind an owned implementation, compose a catalogue, grant one
tool through a closed profile, freeze one exposure plan, then execute one canonical call.

```python
import httpx

from llm_tools import (
    BraveSearchProvider,
    CapabilityProfile,
    ExecutionContext,
    InvocationPosition,
    Native,
    ParsedJson,
    Principal,
    ProfileId,
    RunLimits,
    Scope,
    ToolCatalog,
    ToolExecutor,
    ToolGrant,
    ToolPlan,
    WEB_SEARCH_SPEC,
    bind_brave_web_search,
    web_family,
)
from llm_tools.testing import (
    InMemoryBudgetState,
    InMemoryPositionRecorder,
    NeverCancelled,
    RecordingTelemetry,
)


async def search_once(api_key: str) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        binding = bind_brave_web_search(BraveSearchProvider(client, api_key=api_key))
        catalog = ToolCatalog.compose((web_family(search=binding),))
        profile = CapabilityProfile(
            id=ProfileId("demo-web"),
            grants=(ToolGrant(id=WEB_SEARCH_SPEC.id, limits=None),),
            run_limits=RunLimits(
                max_calls=1,
                max_external_attempts=2,
                max_input_bytes=WEB_SEARCH_SPEC.limits.max_input_bytes,
                max_output_bytes=WEB_SEARCH_SPEC.limits.max_output_bytes,
                max_in_flight=1,
                max_elapsed_seconds=15,
            ),
        ).freeze(catalog)
        plan = ToolPlan(profile=profile.id, exposure=Native()).freeze(catalog, profile)

        budgets = InMemoryBudgetState(profile.run_limits)
        result = await ToolExecutor.execute(
            plan.catalog_view.binding(WEB_SEARCH_SPEC.id),
            ParsedJson({"query": "Brave Search API docs", "freshness_days": None}),
            ExecutionContext(
                plan=plan,
                grant=plan.grant(WEB_SEARCH_SPEC.id),
                catalog_view=plan.catalog_view,
                position=InvocationPosition("demo/turn-1/tool-1"),
                recorder=InMemoryPositionRecorder(durable=False),
                effect_id=None,
                budgets=budgets,
                principal=Principal("demo-user"),
                scope=Scope("demo"),
                cancellation=NeverCancelled(),
                telemetry=RecordingTelemetry(),
            ),
        )
        print(result)
```

`llm_tools.testing` contains process-local conformance doubles only. They are not durable replay,
budget, recovery, or production storage implementations. A production host supplies those owners.
Hosts persist `raw_input_digest(...)` as invocation identity; a binding may raise
`BoundaryFailure` only for the four executor-owned boundary outcomes.

## Activation and security

Importing `llm_tools` grants nothing: the package ships no ambient registry or default profile, so
all four tools remain inactive until a host explicitly composes bindings and grants.

- `web.search` requires an explicitly configured provider credential, binding, and profile grant.
  Missing credentials may leave its binding unavailable without preventing process boot.
- `web.read` requires `bind_web_read(SafeWebReader())`, a profile grant, an application-owned
  information-flow policy, and protected live release proof. Its network controls mitigate SSRF;
  they do not decide whether private application data may be disclosed to an external destination.
- `tool.search` and `tool.read` require `TOOL_FAMILY`, grants for both discovery tools and every
  target, and a `Discoverable` plan with explicit targets and a publication cap. Discovery reveals
  existing authority; it never grants authority.

Provider lowering and model wire protocols remain the responsibility of `provider-runtime`.

## Runtime

- Python 3.12+
- `httpx` 0.28+

## Scope

The package owns portable declarations, strict schemas, catalogs/profiles, execution contracts,
progressive discovery, typed prompt sections, a Brave search adapter, and a non-persisting bounded
public-Web reader. Hosts own authorization, credentials, durable storage, provider orchestration,
and application-specific tools. Every tool requires explicit host composition and authority.

See [the hard-cutover contract](docs/cutovers/llm-tools-library-hard-cutover.md) for the complete
security, replay, and ownership rules.

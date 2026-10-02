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
    RunBudgetState,
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
    InMemoryPositionRecorder,
    NeverCancelled,
    RecordingTelemetry,
)


async def search_once(api_key: str) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        binding = bind_brave_web_search(
            BraveSearchProvider(client, api_key=api_key),
            operation_deadline_seconds=12.0,
        )
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

        budgets = RunBudgetState(profile.run_limits)
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

`RunBudgetState` supplies process-local accounting for one event-loop owner. `llm_tools.testing`
contains conformance doubles only; it does not supply durable replay or recovery. A production host
owns its recorder and atomic result/budget commit.
Hosts persist `raw_input_digest(...)` as invocation identity; a binding may raise
`BoundaryFailure` only for the four executor-owned boundary outcomes.

## Host integration

`RunLimits` permits `None` for cumulative calls, external attempts, input/output bytes and elapsed
time. `None` means no ceiling and is encoded as canonical JSON null in frozen revisions. A finite
ceiling tightens an absent one; an absent ceiling cannot tighten a finite one. `max_in_flight` and
every per-tool limit remain finite. The executor uses the tool deadline alone when run elapsed is
absent.

`BudgetTotals` includes accepted calls/input and settled actual plus unsettled reserved maxima for
attempts/output. `can_reserve(limits, totals, reservation)` is the shared pure admission predicate;
the host handles existing positions first under its recorder lock or transaction. `RunBudgetState`
uses the same predicate with running totals and idempotent position settlement. It supplies no
durability claim; database-backed hosts persist result and settlement atomically.

`validate_tool_input(binding, arguments)` is the public pure validation boundary. It returns the
binding's declared input type or raises `SchemaDecodeError`; it has no execution context and cannot
occupy a position, inspect or reserve a budget, touch a recorder, or dispatch a handler.

`FrozenCapabilityProfile.is_tightening_of(maximum)` proves that every candidate grant is present in
the maximum with the same tool-contract, implementation, and policy revisions and no wider tool or
run limit.
`ToolPlan.freeze(catalog, profile)` additionally proves that every specification and binding in the
exposure-filtered catalogue view has exactly the contract, implementation, and policy revision
authorized by its grant, that effective limits remain narrowed, and that the profile and plan
revisions commit to their contents. Every `ToolBinding` must declare a nonempty owner-controlled
`implementation_revision` covering its handler and transitive execution behavior. Behavior-affecting
configuration belongs in `policy_inputs`; an implementation change not represented there requires
a revision bump. It rejects a substituted catalogue before returning a plan.
`FrozenToolPlan.is_tightening_of(maximum_profile)` first revalidates that complete plan integrity,
then applies the authority proof; inconsistent directly constructed plans return `False`. Profile
identity, profile revision, plan revision, and exposure may differ between a valid candidate and
its maximum because narrowing and publication mode are separate concerns. Equivalent independently
composed catalogues remain valid when their deterministic revisions match. A host that requires
serial prompt-published tools must also require `HostTable` exposure and `max_in_flight == 1`.

`publish_host_table(plan)` accepts only a frozen `HostTable` plan and returns an immutable typed
`PromptSection`. `render_prompt(...)` performs the sole XML-like escaping step. The publication
contains the exact ordered grants, documentation, schemas, effects, replay policies, effective
limits, and revisions. It revalidates the frozen plan and therefore cannot publish a mismatched
contract, implementation, policy, filtered view, profile revision, or plan revision. An empty
profile publishes an actual empty `tools` array; no placeholder capability is required. Publication
is exact rather than silently truncated, so the consuming host must reject a table that exceeds its
cumulative model-context limit.

The durable execution boundary is asynchronous end to end. `BudgetState.reserve/settle` and every
mutating `PositionRecorder` operation are `async`; `ToolExecutor.execute` awaits them and the bound
handler. `terminalize_and_settle` still owns one atomic, idempotent terminal-result and budget-
settlement commit. Durable adapters must use nonblocking persistence drivers. This is a hard cut:
there is no synchronous recorder or budget fallback.

Timeout handling is effect-sensitive after dispatch. A `Pure` or `Read` `ReDispatchable` timeout
terminalizes as `DeadlineExceeded`. A `Write` `ReDispatchable` timeout instead leaves its durable
dispatch claim occupied and raises `RecoveryRequired`; the host must reconcile the provider and
may call `dispatch_abandoned` only after proving the effect absent and redispatch safe. `BilledOnce`
timeouts retain their existing uncertain state. Write handlers must let an ambiguous post-dispatch
`TimeoutError` reach the executor. They may normalize it to `BoundaryFailure("DeadlineExceeded")`
only when they can prove no effect occurred; returning a terminal domain failure for an ambiguous
provider outcome is invalid.

`bind_brave_web_search(provider, *, max_results=10, operation_deadline_seconds=12.0)` owns a
whole-search deadline that includes every Brave request and retry delay. The value must be positive,
finite, and at most 12 seconds; host policy may tighten it but cannot consume the deliberate
three-second guard before `web.search`'s 15-second executor deadline. It is included in the binding
policy revision and therefore in frozen profile and plan identity. On expiration, the binding
returns the declared `UpstreamUnavailable` failure with the number of external attempts actually
started; it does not expose the expected inner expiration as an uncertain executor timeout.
Implementations of `WebSearchProvider.search` used by this binding must accept the optional
keyword-only `attempt_started` callback and invoke it synchronously exactly once immediately before
each external attempt. They must also propagate task cancellation unchanged; suppressing or
replacing cancellation violates the provider contract. Unexpected provider `TimeoutError` and
external task cancellation are not normalized by the binding and retain the executor's recovery
semantics.

`bind_web_read(reader)` carries implementation revision `llm-tools-web-read-v2`. Plain-text
extraction decodes only the declared character set and collapses whitespace: entity-looking text
and markup remain literal data. HTML/XHTML extraction delegates its sole entity-decoding pass to
`HTMLParser(convert_charrefs=True)`, then collapses whitespace without re-decoding parser output.
The evidence locator records these algorithms as `plain-text-v2` and `html-visible-text-v2`;
unchanged canonical JSON extraction remains `json-canonical-v1`. The Web-reader policy inputs and
`web-read-v1` policy epoch are unchanged because accepted media and direct-network policy did not
change.

## Activation and security

Importing `llm_tools` grants nothing: the package ships no ambient registry or default profile, so
all four tools remain inactive until a host explicitly composes bindings and grants.

- `web.search` requires an explicitly configured provider credential, binding, and profile grant.
  Missing credentials may leave its binding unavailable without preventing process boot. Its
  portable declaration retains the two-attempt ceiling; a host may tighten the effective grant to
  one attempt without changing the declaration or binding policy.
- `web.read` requires `bind_web_read(SafeWebReader())`, a profile grant, an application-owned
  information-flow policy, and protected live release proof. Its network controls mitigate SSRF;
  they do not decide whether private application data may be disclosed to an external destination.
  Consumers pinned to `llm-tools-web-read-v1` must update that expected implementation revision and
  rebuild their frozen profile and plan after upgrading the package commit.
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

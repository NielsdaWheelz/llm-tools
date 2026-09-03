# LLM Tools Library Hard Cutover

**Status:** APPROVED IMPLEMENTATION SPECIFICATION
**Date:** 2026-08-13
**Type:** upstream repository/package hard cutover; no compatibility period
**Companion:** [Nexus Tool Runtime Hard Cutover](https://github.com/NielsdaWheelz/nexus-web/blob/main/docs/cutovers/nexus-tool-runtime-hard-cutover.md) — coordination only.

This is the authoritative upstream copy. The Nexus coordination copy is frozen
and, during consumer adoption, links this exact upstream commit. The portable
upstream must not pin a consumer commit: that would invert the dependency and
make the cutover graph cyclic. Never maintain two editable specs or edit a uv
cache checkout.

## 1. Decision

Rename `NielsdaWheelz/web-search-tool` to `llm-tools`, the distribution to
`llm-tools`, and the import package to `llm_tools`. Preserve Git history and
start from upstream main `20adb934cdfb6e7a3a9bbffa990e5553541cba2c`.

`llm_tools` is a small provider-neutral tool kernel plus four portable tools:

- `web.search`
- `web.read`
- `tool.search`
- `tool.read`

Applications declare application-specific tools with the same primitives in
their owning repositories. In particular, `llm_tools` contains no `nexus.*`
declaration, Nexus vocabulary, DTO, authorization policy, client, SQL, or
evidence variant.

Keep `provider-runtime` separate. It owns model/provider wire protocols,
continuations, retries, usage, provider-safe names, and the one lowering from a
published portable declaration to its existing native tool-call values. No
provider SDK or `provider-runtime` type enters `llm_tools`.

## 2. Why this shape

| Lesson | Rule adopted here |
|---|---|
| Codapt/Solid host functions | One typed declaration, separately bound to an owned implementation. |
| Capability systems | A catalogue states possibility; an immutable profile grants authority. |
| Native tool calling | Small profiles publish exact native tools directly. |
| Progressive tool discovery | Large profiles initially publish only `tool.search/read`; discovery never grants authority. |
| Host tables | Deterministic application code, or a separately owned Codapt-like Program Agent, consumes a frozen table that is never provider-published. A Program Agent publishes only `run`, never `run` plus native tools. |
| Durable workflows | A call has a stable position and canonical input digest; the application owns durable memoization. |
| Evidence-first retrieval | Retrieval success carries source identity, snapshot identity, and locator with the data. |
| Provider portability | Canonical dotted ids persist; provider-safe aliases exist only at the wire adapter. |

This cutover supplies the architecture seams, not a speculative runtime. OpenAI
Programmatic Tool Calling, provider-native Tool Search, a code sandbox, and a
new workflow engine require a measured consumer win before implementation.

## 3. Goals and non-goals

The goal is one portable kernel and four general tools satisfying the contracts
in §§5–8 and the acceptance criteria in §12.

Non-goals:

- Nexus tools or another application's domain policy;
- provider/model selection, LLM orchestration, MCP transport, authorization, or
  durable storage;
- automatic provider, search-vendor, model, or exposure fallback;
- crawling, browser control, JavaScript execution, PDF/media ingestion, or a
  general extraction platform;
- hidden ambient registries, plugins, entry-point discovery, or a marketplace;
- compatibility wheels, import aliases, dual declarations, or old-state
  readers.

## 4. Package and ownership

```text
src/llm_tools/
  declaration.py       ToolSpec, ToolEffect, ToolLimits
  schema.py            the one strict-schema normalizer
  catalog.py           immutable ToolFamily and composed ToolCatalog
  profiles.py          CapabilityProfile and ToolPlan
  execution.py         validation, budget accounting, result envelope
  evidence.py          portable evidence values and digests
  discovery.py         tool.search/read declarations, bindings, and family
  prompt_sections.py   escaped typed prompt construction
  testing.py           handlers, catalogues, budgets, and execution doubles
  web/
    contracts.py       request/result/error values
    brave.py           preserved Brave provider
    reader.py          safe fetch plus extraction
    tools.py           web.search/read declarations, bindings, and family
```

Use one deliberate package facade, no ambient registry or second declaration/
normalization path, and no dependency on `provider-runtime`. `httpx` remains the
sole Web client unless the connected-peer contract requires a smaller transport.

## 5. Capability contract

The production construction shape is:

```python
ToolSpec[InputT, SuccessT, ErrorT](
    id=ToolId,
    summary=str,
    documentation=PromptDocument,
    input_type=type[InputT],
    success_type=type[SuccessT],
    error_type=type[ErrorT],
    effect=ToolEffect.Pure | ToolEffect.Read | ToolEffect.Write,
    limits=ToolLimits,
)

ToolBinding(
    spec=spec,
    execute=Available(handler) | Unavailable(private_reason),
    replay_policy=ReplayPolicy.BilledOnce | ReplayPolicy.ReDispatchable,
    implementation_revision=str,
    policy_epoch=PolicyEpoch,
    policy_inputs=CanonicalJsonObject,
)

ToolFamily(namespace="web", declarations=(...), bindings=(...))

CapabilityProfile(
    id=ProfileId,
    grants=(ToolGrant(id=ToolId, limits=tightened_limits), ...),
    run_limits=RunLimits,
)

ToolPlan(
    profile=ProfileId,
    exposure=(
        Native()
        | Discoverable(targets=(ToolId, ...), max_target_tools_published=int)
        | HostTable()
    ),
)

ExecutionContext(
    plan=FrozenToolPlan,
    grant=EffectiveToolGrant,
    catalog_view=PlanCatalogView,
    position=InvocationPosition,
    recorder=PositionRecorder,
    effect_id=EffectId | None,
    budgets=BudgetState,
    principal=Principal,
    scope=Scope,
)
```

Each owner exports one immutable `ToolFamily`; every member id begins with that
family's first dotted segment. `ToolCatalog.compose(families)` is the only
aggregation point and creates no global registry. It rejects duplicate family
names or ids, prefix mismatch, unbound publication, schemas outside the strict
subset, and grants absent from the catalogue. A profile may only tighten a
declaration limit. A plan selects exactly one exposure; unsupported composition
defects with no fallback. `Discoverable.targets` must be a subset of its grants,
both discovery tools must also be granted, and
`max_target_tools_published` must fit
the granted set. Authority lives in the profile, not its exposure.

`validate_tool_input(binding, arguments)` is the pure strict schema boundary. It
returns the declaration's owned input type or raises the public
`SchemaDecodeError`. Its signature contains no execution context, so validation
cannot occupy a position, access or reserve a budget, touch a recorder, or
dispatch a handler. `FrozenCapabilityProfile.is_tightening_of(maximum)` proves a
grant subset with exact shared contract, implementation, and policy revisions
and no wider per-tool or run limit. `ToolPlan.freeze(catalog, profile)` proves
before returning that the exact exposure-filtered catalogue view contains
neither missing nor extra entries and that every exposed specification and
binding matches the contract, implementation, and policy revision authorized
by its grant. It also verifies tightened limits and recomputes the profile and
plan revisions from their contents.
`FrozenToolPlan.is_tightening_of(maximum_profile)` repeats that plan-integrity
proof before applying the authority comparison, returning false for an
inconsistent directly constructed plan. Equivalent separately composed
catalogues remain valid when their deterministic revisions match. Exposure
remains a separate publication choice.

`ToolExecutor.execute(binding, raw_input: RawToolInput, context)` accepts one
bounded tagged raw input from the provider adapter:
`ParsedJson(value: JsonValue) | MalformedJson(raw_utf8: bytes)`. The adapter
rejects parsed arguments whose canonical tagged envelope exceeds the effective
input limit. A provider engine that owns raw argument text rejects transport-
oversize/non-UTF-8 text before it becomes a tool invocation; an SDK that
exposes only parsed JSON cannot honestly attest raw transport bytes. The
executor performs exactly:

1. verify the binding id/revisions, plan-filtered catalogue view, and context grant
   against the frozen plan, then derive the effective limits;
2. canonicalize a parsed value as `{"type":"ParsedJson","value":...}` or a
   malformed value as `{"type":"MalformedJson","raw_sha256":...,"bytes":n}`;
   hash that canonical envelope before typed decode;
3. ask the recorder to occupy or inspect the position with that digest: an id/digest mismatch
   defects, a matching terminal result returns unchanged, and an uncertain
   `BilledOnce` position raises the host-only recovery signal without dispatch;
4. reserve the call and canonical input bytes plus maximum attempt/output spend
   in position-owned budget state; if reservation fails, record no budget
   charge and terminalize `BudgetExceeded` at the occupied position without
   dispatch;
5. strictly decode input; terminalize `InvalidInput` through the same position
   when decoding fails;
6. if the binding is unavailable, terminalize `ToolUnavailable` without
   handler/network dispatch or an uncertain transition;
7. validate a pre-persisted stable effect id and durable recorder at runtime for
   `Write`; record dispatch-started with the binding replay policy, then call
   the one bound handler under the effective deadline;
8. accept the handler's private `(declared outcome, actual_attempts)` value,
   strictly encode the public success/failure, and atomically/idempotently commit
   terminal result plus final budget settlement through the recorder.

Raw transport-oversize/non-UTF-8 rejection, where the engine owns raw text, is
a provider request-boundary failure. Every canonical-bounded parsed or
malformed argument entering this API occupies a durable position. An accepted
reservation retains its call/input charge;
malformed JSON, a non-object JSON value, or an object outside the strict schema
terminalizes `InvalidInput`. A rejected reservation records no budget charge,
because a rejection cannot itself exceed the hard run ceiling it reports.
Invalid input, unavailable, budget, and pre-dispatch deadline outcomes use zero
attempts and actual encoded output bytes; settlement refunds unused
attempt/output reservation. A pre-dispatch rejection dispatches nothing. After
dispatch, a timeout or unknown outcome on `BilledOnce` stays uncertain and
raises the host recovery signal; it is not mislabeled as `DeadlineExceeded`. A
timeout on a `Pure` or `Read` `ReDispatchable` binding terminalizes as
`DeadlineExceeded`. A timeout on a `Write` `ReDispatchable` binding instead
leaves its durable dispatch claim occupied and raises `RecoveryRequired`; it
does not terminalize, enter `BilledOnce` uncertainty, or re-admit itself. The
host must reconcile the provider and may call `dispatch_abandoned` only after
proving the effect absent and redispatch safe. Reservation, terminal result,
and settlement are position-owned, so replay of a terminal result observes an
already settled budget and cannot leak or double-settle it. The host recorder
owns any transactional enlistment needed to commit a Write result with its
domain mutation. Catalogue composition validates declared metadata only;
runtime and the binding owner's proof establish effect-id presence, durability,
and atomic commit behavior.

Expected failures are values:

```json
{"type":"Failure","error":{"type":"..."}}
```

The strict error union is `BoundaryError | ErrorT`. The executor owns
`InvalidInput`, `ToolUnavailable`, `BudgetExceeded`, and `DeadlineExceeded`;
the declaration's `error_type` owns only intentional handler/domain failures.
The normalizer compiles both into the published error schema. A handler raises
one internal declared-failure wrapper to return `ErrorT`; it never throws a
model-visible payload directly. A library-owned binding may raise the private
boundary-failure signal only for an executor-owned outcome with exact attempt
accounting. Discovery uses it for indistinguishable availability and output
budget failures; Web uses it for deadline and output-budget failures.
`tool.read` maps both an unknown and an ungranted id to identical
`ToolUnavailable` before returning any catalogue fact.

A `Write` handler must propagate an ambiguous post-dispatch `TimeoutError` so
the executor preserves the occupied claim and signals host recovery. It may
normalize a timeout to `BoundaryFailure("DeadlineExceeded")` only when it can
prove no effect occurred. A false terminal boundary or domain failure would
erase the ambiguity and is a handler defect.

Success is:

```json
{"type":"Success","value":{}}
```

The envelope does not impose an open-ended evidence union. A retrieval tool's
strict success type owns a required evidence field. `llm_tools` owns the Web
receipt; an application owns its local evidence type without changing the
kernel.

Unknown handler exceptions, invalid owned output, impossible registry states,
and exhausted required invariants are defects and raise. Arbitrary handler JSON
is never recursively rewritten.

An unavailable binding keeps catalogue/profile construction deterministic but
returns `ToolUnavailable` before budgeted handler or network dispatch. Its
private reason is telemetry only. `llm_tools` never reads credentials or the
environment. Missing optional credentials do not prevent process boot,
catalogue construction, or unrelated tool use; invalid credentials that a host
explicitly configured are host defects.

### Strict schema

The declaration owns Python types; `schema.py` owns the only JSON Schema
normalization path. It produces two canonical projections:

- `semantic_schema` contains only validation-affecting keywords and values;
- `presentation_schema` adds the allowed annotation vocabulary, including
  titles, descriptions, examples, and help text.

Both projections recursively:

- closes every object with `additionalProperties: false`;
- marks every property required, representing intentional nullable public
  fields with JSON `null` rather than importing any application's absence type;
- inlines or rejects references, normalizes tagged unions on `type`, and rejects
  unsupported keywords or ambiguous unions;
- canonicalize object keys and set-like arrays whose order has no validation
  meaning, including `required`, `enum`, and normalized union branches; and
- fail closed outside one versioned portable subset.

Input, success, and error schemas are all compiled and validated. Raw Pydantic
output is never sent directly to a model provider. `llm_tools` proves the
portable subset; each `provider-runtime` adapter owns proof that it lowers that
subset without semantic loss and decodes returned names to canonical ids.

### Identity and revisions

Canonical ids use lower-case dotted segments. A segment matches
`[a-z][a-z0-9_]*` and may not contain `__`. `provider-runtime` replaces `.` with
`__`, rejects alias collisions or provider length/grammar violations, and
reverses the mapping. Only canonical ids serve as application identity or keys
for authorization, dispatch, replay, evidence, telemetry, or storage. A host
may retain one bounded raw provider name as non-authoritative rejection/audit
payload, but it never becomes a `ToolId` or identity key.

Each declaration's `tool_contract_revision` hashes its id, semantic input,
success, and error schemas, effect, and limits. `documentation_revision` hashes
presentation-only schema annotations plus summaries and prompt/help text. Each
binding declares a nonempty owner-controlled `implementation_revision`, which
covers its handler and transitive execution behavior; the library does not
infer identity from a callable. Behavior-affecting configuration belongs in
`policy_inputs`; an implementation change not represented there, including a
transitive dependency change, requires an implementation-revision bump. The
`policy_revision` hashes the explicit owner-controlled policy epoch, canonical
policy inputs, and replay policy; availability is live deployment state, not
authority. A profile's `profile_revision` hashes its run limits, ordered grants,
effective limits, and granted tool, implementation, and binding-policy
revisions.
A plan's `plan_revision` hashes its profile revision and the full tagged
exposure payload, including discoverable targets and publication ceiling.
Plan construction and every public proof or HostTable publication recompute
these revisions and cross-check the filtered specifications and bindings against
their grants; a different catalogue cannot substitute an effect, schema,
declared limit, implementation, replay policy, or policy input under an
authorized revision.
Presentation-only edits never invalidate authority or durable replay, while the
plan still requires its view specification and bound specification to share the
same documentation revision.

## 6. General tools

Every tool also exposes the common `BoundaryError` union:
`InvalidInput | ToolUnavailable | BudgetExceeded | DeadlineExceeded`.

| Id | Effect / replay | Input | Success | Closed `ErrorT` | Default ceiling |
|---|---|---|---|---|---|
| `web.search` | Read / `BilledOnce` | query plus required-nullable freshness days | ranked normalized results plus provider/request identity | `RateLimited | UpstreamUnavailable | InvalidUpstreamResponse` | query 2–400 chars/50 words; 10 results; 2 provider attempts; 15 s; 32 KiB output |
| `web.read` | Read / `ReDispatchable` | one normalized public `http` or `https` URL | final URL, title, media type, extracted text, Web evidence | `InvalidUrl | UnsafeDestination | UnsupportedContent | TooLarge | RateLimited | UpstreamUnavailable | InvalidUpstreamResponse` | URL 4,096 chars; 5 redirects; 8 total network requests, including at most 2 pre-body transport retries overall; 2 MiB wire; 4 MiB decoded; 64 KiB text; 20 s |
| `tool.search` | Pure / `ReDispatchable` | query 0–200 chars, required-nullable family prefix, limit 1–20 | matching granted ids, families, summaries, effects, and input synopsis | no declared domain failure | 20 results; 16 KiB; 2 s |
| `tool.read` | Pure / `ReDispatchable` | one canonical id | full granted declaration, semantic/presentation schemas, documentation, and effective limits | no declared domain failure | 64 KiB; 2 s |

The model tool deliberately exposes only `query` and `freshness_days`; mixed
results, moderate safe search, US/en locale, and the effective profile limit are
host policy. Preserve the broader programmatic `WebSearchRequest`, current Brave
request/result semantics, URL normalization, deduplication, provenance, and
bounded two-attempt retry behavior from upstream main. A concrete Brave binding
requires an explicitly supplied credential; a host may instead bind the
declared tool as unavailable. Retry policy has one library owner; SDK/client
retries remain disabled. Invalid input is the common boundary error, and raw
provider messages never enter a model-visible failure.

`tool.search/read` inspect an immutable view filtered by both the profile grants
and that plan's frozen `Discoverable.targets`. Search
is deterministic over canonical id, family, summary, and input property names;
an empty query lists the selected family. They never reveal ungranted ids and
never mutate a profile. `tool.read` collapses unknown, ungranted, and granted-
but-nontarget ids to the identical `ToolUnavailable` boundary failure. A
`Discoverable` plan references a profile granting
these two tools and its discoverable targets; the model initially sees only the
two discovery tools. Successful `tool.read(id)` records that already-granted id
in the next request's published set. `max_target_tools_published` counts only
revealed targets; the two always-published discovery tools are separate. The
host persists the revealed-target set and publishes deterministically. The
reference-host conformance proof is their v1 consumer.

All three currently ungranted tools—`web.read`, `tool.search`, and `tool.read`—
are disabled-by-default exports. Shipping and conformance-testing them is not a
claim that a production profile enables them. A production consumer must opt in
with its own authority, information-flow policy, budgets, and release proof.

## 7. Web-reader security and evidence

`web.read` is a non-persisting external read. It never creates application
records, queues ingest, joins a library, sends credentials, executes scripts,
loads subresources, or follows a page-authentication challenge.

Before each connection and redirect hop, the reader:

1. normalizes the URL and rejects userinfo, fragments, non-HTTP schemes, and
   malformed or ambiguous hosts;
2. resolves every address and rejects loopback, private, link-local, multicast,
   reserved, documentation, carrier-grade NAT, and cloud-metadata destinations;
3. connects to a pinned admitted address using that hop URL's normalized
   hostname for Host and TLS SNI, then verifies the connected peer is still
   admitted;
4. re-runs the full policy for every redirect, including a changed authority;
   and
5. enforces deadline, attempt, redirect, compressed, decompressed, MIME, and
   extracted-text ceilings while streaming.

The initial request, every redirect request, and every transport retry each
consume one attempt. The eight-request ceiling is one initial request plus up
to five redirects and no more than two pre-body retries total across the whole
logical dispatch; no hop receives a fresh retry allowance. Wire, decoded-byte,
and elapsed-time ceilings are aggregate across the entire dispatch, not reset
per request or redirect.

V1 is direct mode and must pin/verify the actual peer. Proxy support is out of
scope; ambient proxy variables, cookies, authorization headers, and shared
browser state are disabled.

`reader.py` owns one narrow resolver/connected-transport boundary. The
production implementation returns both resolved candidates and the actual peer;
the policy, not the transport, decides admission. Tests may substitute only
this OS/network boundary, which lets the same production policy prove rebinding
and peer mismatch without weakening the real call path.

V1 accepts HTML/XHTML, plain text, and JSON. Extraction emits bounded text and
metadata, never active markup. Every success includes:

```text
EvidenceReceipt(
  source_uri, final_uri, observed_at,
  content_sha256, media_type, locator
)
```

Retrieved content is untrusted data. `web.read` prevents network-boundary abuse;
it cannot decide whether a host may disclose private information to an external
destination. Any application granting external reads beside private data owns
that information-flow policy explicitly.

## 8. Budgets, replay hooks, and prompts

`ToolLimits` declares per-call input, output, attempt, and deadline maxima.
`RunLimits` declares total calls, external attempts, input bytes, output bytes,
maximum in-flight calls, and elapsed host-call time. Before dispatch, execution
charges the known input/call and reserves the maximum output and attempt spend;
it settles/refunds unused reservation only in the recorder's atomic terminal
commit. Uncertainty retains the reservation until explicit reconciliation
terminalizes that position; neither an exception nor process recovery refunds
it independently. Elapsed time is measured against the run deadline. A
rejected reservation records no budget charge and dispatches nothing; its
position-owned `BudgetExceeded` result is replayed without repeated admission.
Every dispatched attempt counts; a retry is not free.

The library defines `InvocationPosition` and the canonical input digest. The
host supplies principal/scope, budget state, cancellation, telemetry, and a
position recorder through `ExecutionContext`. The library stores no replay
state. A durable host must memoize terminal results by position and reject a
different tool or input digest at an occupied position.

`ToolBinding.replay_policy` is mandatory. `BilledOnce` forbids automatic
redispatch after dispatch begins: an uncertain outcome suspends for explicit
recovery. `ReDispatchable` permits the same canonical invocation at the same
durable position only after the host has fenced and explicitly re-admitted a
verified abandoned dispatch claim. The normal execute path never releases its
own claim after a defect or cancellation. Library-owned HTTP retries are
attempts inside one logical dispatch and consume its attempt budget; a host
crash never restarts a `BilledOnce` dispatch.

The same recovery gate applies when a `Write` `ReDispatchable` handler times
out: `ToolExecutor` raises `RecoveryRequired` while preserving the nonterminal
occupied dispatch. A replay before reconciliation cannot dispatch. The host
may re-admit only through `dispatch_abandoned` after external evidence proves
the effect absent and repeating it safe. `Pure` and `Read` `ReDispatchable`
timeouts remain terminal `DeadlineExceeded` results.

`PositionRecorder.dispatch_abandoned` is the narrow host recovery hook. It is
not a model-call path: only the durable owner calls it after proving lease or
operator recovery, and it carries the known attempts already spent so the next
dispatch receives only the remaining ceiling. It rejects `BilledOnce`,
terminal, uncertain, unclaimed, unfenced, and over-attempt positions.

`BudgetState.reserve/settle` and every mutating `PositionRecorder` operation are
asynchronous. `ToolExecutor.execute` awaits those operations and the handler;
there is no synchronous persistence fallback. The recorder still owns atomic,
idempotent terminal-result plus budget settlement. A durable adapter uses a
nonblocking persistence driver and preserves its own transaction boundary.

A `HostTable` plan is never provider-published. Deterministic application code
may call it directly; a separately owned Program Agent may consume it behind
its sole `run` tool. This library implements neither a Program Agent nor
Programmatic Tool Calling.

`publish_host_table(plan)` accepts only a frozen `HostTable` plan and returns a
typed immutable prompt section for `render_prompt`. It publishes the exact
ordered grants, documentation, schemas, effects, replay policies, effective
limits, implementation identity, and revisions. Before rendering, it repeats
the complete plan-integrity check, so a directly constructed plan, substituted
catalogue view, or forged profile/plan revision fails closed. An empty profile
produces a real empty table with no dummy tool. Publication never truncates
silently; the consuming host enforces its cumulative context bound before
provider I/O.

Every granted `Write` binding, including HostTable calls, declares its durable
effect requirements; `ToolExecutor.execute` requires the invocation to carry a
durable position recorder and stable effect id at runtime. Misconfiguration
defects before handler dispatch. The host owns approval policy,
idempotency/reconciliation, commit ordering, and recovery; no write executes
from in-memory-only replay state.
`ExecutionContext.effect_id` is allocated and persisted before dispatch and is
stable across retries, workers, and provider continuations. It never derives
from a provider call id, provider alias, tool id, attempt, or worker identity.

Canonical tool results are JSON. When a prompt needs a readable wrapper,
`prompt_sections.py` builds typed, fixed-tag sections and escapes attributes and
text. Dynamic tag names and raw interpolation are impossible. XML-like prompt
text is presentation only, never the execution or persistence format.

## 9. Hard cut and consumers

The rename is one hard cut across source and known consumers. The stopped world
is dependency resolution, image construction, and deployment: no build or
deployment may resolve the old URL or name after the rename. An already-running
immutable predecessor does not resolve its dependency at runtime and may remain
healthy only where the consumer's release controller requires it for verified
handoff and rollback.

1. Prepare and prove one final commit that renames the repository source,
   distribution, wheel package, imports, build metadata, CI, and release
   workflow. Copy this spec into that repository and rewrite its
   cross-repository links to immutable targets. Publish no import alias, shim,
   or newly built old-name artifact. Reconfirm and reserve the GitHub and PyPI
   names before the stopped window; if either is unavailable, stop and respec
   instead of publishing an alternate identity.
2. Inventory every deployed Nexus/Ariel process or image containing the old
   dependency and record its immutable artifact SHA. Stop it before the rename
   when its release owner supports a safe maintenance stop. Where the verified
   release controller requires a healthy predecessor, leave that immutable
   predecessor running, prohibit rebuild/restart through the old identity, and
   replace it through the controller after the new artifact is ready. Rename
   the GitHub repository and publish the new distribution. The hosting redirect
   is not a supported source and no build, lock, image, or deployment may use it.
3. Pin, deploy, and behavior-check Nexus from
   `ce911344a2e31900eaee3039b5d61109ef54bcb1` to the exact final `llm-tools`
   commit and migrate its imports. The cut is incomplete until no running Nexus
   artifact contains the old distribution/import.
4. Inventory whether Ariel has a deployed runtime. Pin and behavior-check its
   selected clean implementation checkout (currently `/opt/ariel`) from
   `20adb934cdfb6e7a3a9bbffa990e5553541cba2c` to the same final commit and
   migrate its imports. If deployed, stop and replace every old artifact and
   prove the exact new artifact; if source-only, record that fact and prove its
   exact pinned build. The cut is incomplete until no running Ariel artifact
   contains the old distribution/import. Do not edit the unrelated dirty Ariel
   checkout.
5. After both source and running-artifact inspections prove every known
   consumer migrated, retire any controllable old remote redirect and active
   old-name release, CI, and package documentation surfaces. Historical Git
   objects and immutable releases remain history, not executable compatibility.
   Update the current project identity/links in the known niels-lab portfolio
   entry; preserve explicitly dated historical claims as history.

No consumer may use a local head, editable install, uv cache path, branch ref,
version range, or old-name redirect as release evidence.

## 10. Non-overlapping work packages

| Lane | Exclusive owner | Depends on | Exit proof |
|---|---|---|---|
| L0 rename/baseline | renamed source/package/build metadata and this spec move; no alias or old-name publication | none | wheel imports only `llm_tools`; history preserved |
| L1 kernel | declaration, schema, catalogue, profiles, execution, evidence, testing | L0 | kernel contract proof |
| L2 Web | Web contracts, Brave adapter, safe reader | L1 | search and real-network-seam conformance |
| L3 discovery/prompt | discovery and prompt-section modules | L1 | profile-filtered discovery and escaping proof |
| L4 provider integration | `llm-calling` adapter/optional dependency and tests only; no provider engine or registry changes | final L1–L3 library commit | provider values/aliases are structurally exact |
| L5 consumers | Nexus and Ariel pins/imports/deployed artifacts in their own repositories | L4 | exact-source and running-artifact behavior proof |
| L6 legacy retirement | controllable old remote redirect plus active old-name release/CI/package docs and the niels-lab current project identity; no executable source or consumer edits | L5 | source/runtime/current-doc residue audit; no supported old identity |

L2 and L3 may run in parallel after L1. Each lane owns its tests. L4 and L5
never edit the library. Freeze the final library commit before provider and
consumer pinning; L6 changes no executable code.

## 11. Red / green / refactor and proof

Follow red/green/refactor per the companion Nexus testing standard: target proof
first, meaningful red fingerprint, green, representative sensitivity for each
critical/replacement proof, then refactor without changing the proof.

| Boundary | Primary proof and independent oracle |
|---|---|
| declaration/schema/catalog/profile | `tests/kernel/test_tool_contract.py`: hand-authored semantic/presentation schemas and profile table; description-only and enum-order-only edits preserve contract revision, description edits bump documentation revision, semantic enum edits bump contract revision; composes separate `web`/`tool` families and rejects mixed prefixes, duplicates, malformed/unbound grants, stale binding policy, over-budget, and mixed exposure. `tests/kernel/test_frozen_plan_integrity.py`: cross-catalog Read-to-Write, schema, limit, implementation, replay, and policy substitution; direct inconsistent plan, missing/extra view, and forged revision rejection; equivalent-catalogue, narrowed Discoverable, Native, HostTable, and empty-plan validity. |
| execution/result/prompt | `tests/kernel/test_execution_and_prompt.py`: hand-authored recorder trace proves parsed/malformed raw-envelope digest before decode, occupied mismatch, completed replay, malformed/nonobject/schema-invalid terminalization, budget/unavailable terminalization, unavailable-without-uncertain, dispatch transition by replay policy, uncertain `BilledOnce`, terminal `Pure`/`Read` `ReDispatchable` timeout, recovery-gated `Write` `ReDispatchable` timeout and redispatch, effect-id rejection, direct inconsistent-plan rejection before recorder access, atomic/idempotent result-plus-settlement and crash/replay, exact attempt/output accounting, reviewed envelopes/escaping, and arbitrary guest JSON preservation |
| discovery | `tests/kernel/test_discovery.py`: fixed granted/target/nontarget/ungranted catalogue; search reveals only targets, unknown/ungranted/nontarget reads are identical, the target-only publication cap and plan revision are exact, and only successfully read targets publish on the next reference-host turn |
| Web search | `tests/conformance/test_web_search.py`: fixed Brave transcripts; normalized identity, attempts, limits, and errors |
| Web read | `tests/conformance/test_web_read.py`: test-owned loopback servers and resolver/peer fixtures for cross-authority redirect Host/SNI, rebinding, private address, peer mismatch, MIME, size, compression, timeout, evidence, and ambient-proxy rejection |
| provider lowering | `llm-calling/tests/test_tool_adapter.py`: `provider-runtime` owns immutable request-scoped native values, alias grammar/collision rejection, reverse decoding, dotted-id round-trip, and semantic-schema fidelity across every supported engine against the exact `llm-tools` dependency |
| packaging | `tests/test_package.py`: built-wheel isolated install proves old import absent and public facade complete |
| consumers | Nexus owns `python/tests/llm_tools_contract/test_pinned_llm_tools.py`; Ariel owns `tests/test_llm_tools_contract.py`; both materialize the exact SHA and prove import plus one public behavior |

PR proof is deterministic and socket-denied except for test-owned local servers.
One opt-in `tests/live/test_brave_canary.py` release canary calls the real Brave
API and checks the minimal contract, credential/quota behavior, provenance, and
a fixed cost ceiling. When a named production profile enables `web.read`, its
protected release also runs `tests/live/test_web_read_canary.py` against an
owned public HTTPS redirect fixture and checks the actual peer, hop, bounds, and receipt.
Missing credentials or fixture availability mean `not_run`, never pass; a
required enablement gate cannot promote on `not_run`.
Protected live owners override the default socket denial only for the exact
canary node with `--force-enable-socket`; ordinary package tests allow only
Unix sockets and loopback.

```bash
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen pyright src tests
uv run --frozen pytest -q
uv build --no-sources
uv run --frozen pip-audit
```

After all consumers are green, run a one-time residue audit for
`web_search_tool` and `web-search-tool`. Do not retain a tombstone grep test.

## 12. Acceptance criteria

1. The repository, distribution, and import are `llm-tools` / `llm_tools`; Git
   history is preserved, and no current source, built artifact, or known
   consumer executes the old names. Immutable historical releases remain
   history only.
2. One strict declaration/binding/executor path owns all four general tools.
3. Every published tool has strict input, success, and declared-error schemas,
   an effect, mandatory replay policy, explicit limits, and separate contract/
   binding-policy/documentation revisions. Explicitly unavailable bindings
   preserve catalogue shape and fail before dispatch without blocking boot.
4. Profiles are closed authority; Native, Discoverable, and HostTable plans
   select one mutually exclusive exposure and defect without fallback. Freeze,
   tightening proof, execution, and HostTable publication reject any catalogue
   view, contract/implementation/policy grant, effective limit, or profile/plan
   revision inconsistency before provider or handler I/O.
5. `tool.search/read` reveal only the frozen intersection of grants and
   Discoverable targets; unknown, ungranted, and nontarget ids are
   indistinguishable, and the reference host proves target-capped
   discovery-to-next-request publication without authority expansion.
6. `web.search` preserves current shipped behavior and provenance and is
   `BilledOnce` across host recovery.
7. `web.read` is non-persisting, credential-free, bounded, and proven at its
   actual peer/redirect/streaming seam against SSRF and resource exhaustion;
   its identical durable invocation is `ReDispatchable`.
8. Every bounded raw call is durably digested before typed decode; expected
   boundary failures terminalize at that position except ambiguous dispatched
   Writes, which preserve their claim for reconciliation. Terminal result plus
   budget settlement is atomic/idempotent, and unexpected failures raise.
9. The provider-runtime integration proves reversible aliases and structurally
   exact native values; only canonical dotted ids are executable identity, with
   a bounded raw rejected name permitted solely as non-authoritative audit data.
10. Nexus and Ariel run against one exact immutable library commit, with no uv
    cache, editable checkout, branch, compatibility import, or fallback.
11. Closest-seam proofs have independent oracles, recorded red/green evidence,
    and demonstrated sensitivity in proportion to risk.
12. No Nexus, MCP, provider-orchestration, PTC, crawler, or workflow-engine
    policy has entered the portable kernel.

## 13. References

- [Nexus testing standards](https://github.com/NielsdaWheelz/nexus-web/blob/823b371332e51244a1fc31632a3234f794288eb4/docs/local-rules/testing-standards.md)
- [Nexus boundary rules](https://github.com/NielsdaWheelz/nexus-web/blob/823b371332e51244a1fc31632a3234f794288eb4/docs/rules/boundaries.md)
- [Nexus generated-text rules](https://github.com/NielsdaWheelz/nexus-web/blob/823b371332e51244a1fc31632a3234f794288eb4/docs/rules/generated-text.md)
- [Provider Runtime contract](https://github.com/NielsdaWheelz/llm-calling/blob/f080686aee5889b341304108b796ed2b1de1d614/README.md)
- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [JSON Schema 2020-12](https://json-schema.org/draft/2020-12)

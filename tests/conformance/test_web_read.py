"""Real-stream conformance proof for the fail-closed ``web.read`` boundary."""

from __future__ import annotations

import asyncio
import gzip
import json
import ssl
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import trustme

from llm_tools.budgets import RunBudgetState
from llm_tools.catalog import ToolCatalog
from llm_tools.declaration import ToolLimits
from llm_tools.execution import (
    ExecutionContext,
    InvocationPosition,
    ParsedJson,
    Principal,
    Scope,
    ToolExecutor,
)
from llm_tools.profiles import CapabilityProfile, Native, ProfileId, RunLimits, ToolGrant, ToolPlan
from llm_tools.schema import canonical_json_bytes
from llm_tools.testing import (
    InMemoryPositionRecorder,
    NeverCancelled,
    RecordingTelemetry,
)
from llm_tools.web.contracts import (
    InvalidUpstreamResponse,
    RateLimited,
    TooLarge,
    UnsafeDestination,
    UnsupportedContent,
    UpstreamUnavailable,
    WebReadFailure,
    WebReadResponse,
)
from llm_tools.web.reader import (
    ConnectedStream,
    DirectConnector,
    SafeWebReader,
    WebReadLimits,
)
from llm_tools.web.tools import WEB_READ_SPEC, bind_web_read, web_family

PUBLIC_A = "93.184.216.34"
PUBLIC_B = "142.250.72.14"


@dataclass(slots=True)
class StaticResolver:
    addresses: dict[str, tuple[str, ...]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        return self.addresses[hostname]


@dataclass(slots=True)
class MappedConnector:
    ports: dict[str, int]
    peer_addresses: dict[str, str] = field(default_factory=dict)
    ssl_context: ssl.SSLContext | None = None
    failures_before_connect: int = 0
    calls: list[tuple[str, int, str, bool]] = field(default_factory=list)

    async def connect(
        self,
        address: str,
        port: int,
        *,
        hostname: str,
        tls: bool,
        timeout_seconds: float,
    ) -> ConnectedStream:
        self.calls.append((address, port, hostname, tls))
        if self.failures_before_connect:
            self.failures_before_connect -= 1
            raise OSError("test-owned pre-body transport failure")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                "127.0.0.1",
                self.ports[hostname],
                ssl=self.ssl_context if tls else None,
                server_hostname=hostname if tls else None,
            ),
            timeout=timeout_seconds,
        )
        return ConnectedStream(
            reader=reader,
            writer=writer,
            peer_address=self.peer_addresses.get(hostname, address),
        )


class MustNotReadReader(SafeWebReader):
    async def read(
        self,
        raw_url: str,
        *,
        max_requests: int | None = None,
        deadline_seconds: float | None = None,
    ) -> WebReadResponse:
        raise AssertionError(f"must not read: {raw_url!r} {max_requests!r} {deadline_seconds!r}")


Response = bytes | Callable[[bytes], bytes | Awaitable[bytes | None] | None]


@dataclass(slots=True)
class LoopbackServer:
    response: Response
    ssl_context: ssl.SSLContext | None = None
    requests: list[bytes] = field(default_factory=list)
    _server: asyncio.Server | None = None

    async def __aenter__(self) -> LoopbackServer:
        self._server = await asyncio.start_server(
            self._handle,
            "127.0.0.1",
            0,
            ssl=self.ssl_context,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def port(self) -> int:
        assert self._server is not None
        sockets = self._server.sockets
        assert sockets
        return int(sockets[0].getsockname()[1])

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1.0)
            self.requests.append(request)
            response = self.response(request) if callable(self.response) else self.response
            if isinstance(response, Awaitable):
                response = await response
            if response is not None:
                writer.write(response)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def _response(
    body: bytes,
    *,
    status: str = "200 OK",
    content_type: str = "text/plain; charset=utf-8",
    headers: tuple[tuple[str, str], ...] = (),
) -> bytes:
    fields = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Connection", "close"),
        *headers,
    ]
    encoded = "".join(f"{name}: {value}\r\n" for name, value in fields).encode("ascii")
    return f"HTTP/1.1 {status}\r\n".encode() + encoded + b"\r\n" + body


def _redirect(location: str) -> bytes:
    return _response(
        b"",
        status="302 Found",
        headers=(("Location", location),),
    )


def _execution_context(
    binding: Any,
    *,
    tool_limits: ToolLimits | None = None,
    run_limits: RunLimits | None = None,
    position: str = "turn-1/read-1",
) -> ExecutionContext:
    catalog = ToolCatalog.compose((web_family(read=binding),))
    effective_run_limits = run_limits or RunLimits(
        2,
        16,
        64 * 1_024,
        1 * 1_024 * 1_024,
        1,
        30.0,
    )
    profile = CapabilityProfile(
        id=ProfileId(position.replace("/", "-")),
        grants=(ToolGrant(id=WEB_READ_SPEC.id, limits=tool_limits),),
        run_limits=effective_run_limits,
    ).freeze(catalog)
    plan = ToolPlan(profile=profile.id, exposure=Native()).freeze(catalog, profile)
    return ExecutionContext(
        plan=plan,
        grant=plan.grant(WEB_READ_SPEC.id),
        catalog_view=plan.catalog_view,
        position=InvocationPosition(position),
        recorder=InMemoryPositionRecorder(),
        effect_id=None,
        budgets=RunBudgetState(effective_run_limits),
        principal=Principal("proof"),
        scope=Scope("public-web"),
        cancellation=NeverCancelled(),
        telemetry=RecordingTelemetry(),
    )


def test_model_visible_read_contract_states_security_and_non_persistence() -> None:
    assert WEB_READ_SPEC.summary == "Read one public web page as bounded inert text with evidence."
    documentation = WEB_READ_SPEC.documentation.text.lower()
    assert "does not persist" in documentation
    assert "untrusted" in documentation
    assert "javascript" in documentation
    assert "subresources" in documentation
    assert WEB_READ_SPEC.input_schema.semantic == {
        "additionalProperties": False,
        "properties": {"url": {"maxLength": 4096, "minLength": 1, "type": "string"}},
        "required": ["url"],
        "type": "object",
    }
    assert WEB_READ_SPEC.limits.max_attempts == 8
    assert WEB_READ_SPEC.limits.max_input_bytes == 24_616
    assert WEB_READ_SPEC.limits.max_output_bytes == 512 * 1_024
    assert WEB_READ_SPEC.limits.deadline_seconds == 20.0
    declared = WEB_READ_SPEC.declared_error_schema
    assert declared is not None
    assert {branch["properties"]["type"]["const"] for branch in declared.semantic["anyOf"]} == {
        "InvalidUrl",
        "UnsafeDestination",
        "UnsupportedContent",
        "TooLarge",
        "RateLimited",
        "UpstreamUnavailable",
        "InvalidUpstreamResponse",
    }
    for deadline in (True, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            WebReadLimits(deadline_seconds=deadline)  # type: ignore[arg-type]


def test_web_read_binding_revision_records_entity_extraction_behavior() -> None:
    available = bind_web_read(SafeWebReader(resolver=StaticResolver({})))
    unavailable = web_family().bindings[1]

    assert available.implementation_revision == "llm-tools-web-read-v3"
    assert unavailable.implementation_revision == "llm-tools-web-read-v3"
    assert available.policy_revision == unavailable.policy_revision


@pytest.mark.asyncio
async def test_every_schema_valid_4096_character_url_reaches_typed_url_validation() -> None:
    binding = bind_web_read(SafeWebReader(resolver=StaticResolver({})))
    context = _execution_context(
        binding,
        position="turn-1/max-url",
    )
    raw_url = "é" * 4_096

    result = await ToolExecutor.execute(binding, ParsedJson({"url": raw_url}), context)

    assert result == {"type": "Failure", "error": {"type": "InvalidUrl"}}
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    assert context.recorder.record(context.position).reservation_accepted is True


@pytest.mark.asyncio
async def test_normalized_unicode_url_over_public_output_limit_is_invalid_before_network() -> None:
    resolver = StaticResolver({})
    binding = bind_web_read(SafeWebReader(resolver=resolver))
    context = _execution_context(binding, position="turn-1/expanded-url")
    raw_url = "http://public.test/" + "é" * 2_000

    result = await ToolExecutor.execute(binding, ParsedJson({"url": raw_url}), context)

    assert result == {"type": "Failure", "error": {"type": "InvalidUrl"}}
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_cross_authority_redirect_revalidates_dns_host_and_evidence() -> None:
    body = (
        b"<html><head><title> Example &amp; title </title><script>steal()</script></head>"
        b"<body><h1>Hello</h1><p>portable tools</p></body></html>"
    )
    async with LoopbackServer(_redirect("http://b.test/final")) as first:
        async with LoopbackServer(
            _response(body, content_type="text/html; charset=utf-8")
        ) as second:
            resolver = StaticResolver({"a.test": (PUBLIC_A,), "b.test": (PUBLIC_B,)})
            connector = MappedConnector({"a.test": first.port, "b.test": second.port})
            result = await SafeWebReader(resolver=resolver, connector=connector).read(
                "HTTP://A.TEST:80/start"
            )

    assert resolver.calls == [("a.test", 80), ("b.test", 80)]
    assert connector.calls == [
        (PUBLIC_A, 80, "a.test", False),
        (PUBLIC_B, 80, "b.test", False),
    ]
    assert b"Host: a.test\r\n" in first.requests[0]
    assert b"Host: b.test\r\n" in second.requests[0]
    assert result.attempts == 2
    assert [hop.peer_address for hop in result.hops] == [PUBLIC_A, PUBLIC_B]
    assert result.value.final_url == "http://b.test/final"
    assert result.value.title == "Example & title"
    assert result.value.media_type == "text/html"
    assert result.value.text == "Example & title Hello portable tools"
    assert result.value.evidence.source_uri == "http://a.test/start"
    assert result.value.evidence.final_uri == "http://b.test/final"
    assert result.value.evidence.content_sha256 == __import__("hashlib").sha256(body).hexdigest()
    locator = json.loads(result.value.evidence.locator)
    assert locator == {
        "content_encoding": "identity",
        "extraction": "html-visible-text-v2",
        "representation": "decoded-entity-bytes",
        "text_truncated": False,
        "text_utf8_bytes": len(result.value.text.encode()),
        "type": "web.document",
        "version": 1,
    }


@pytest.mark.asyncio
async def test_unicode_query_is_canonically_escaped_in_request_and_evidence() -> None:
    async with LoopbackServer(_response(b"unicode")) as server:
        result = await SafeWebReader(
            resolver=StaticResolver({"unicode.test": (PUBLIC_A,)}),
            connector=MappedConnector({"unicode.test": server.port}),
        ).read("http://unicode.test/search?q=café&lang=日本語")

    target = b"/search?q=caf%C3%A9&lang=%E6%97%A5%E6%9C%AC%E8%AA%9E"
    assert server.requests[0].startswith(b"GET " + target + b" HTTP/1.1\r\n")
    assert result.value.final_url == (
        "http://unicode.test/search?q=caf%C3%A9&lang=%E6%97%A5%E6%9C%AC%E8%AA%9E"
    )
    assert result.value.evidence.final_uri == result.value.final_url


@pytest.mark.asyncio
async def test_public_binding_strictly_encodes_required_evidence_and_exact_attempts() -> None:
    async with LoopbackServer(_response(b"bound page")) as server:
        binding = bind_web_read(
            SafeWebReader(
                resolver=StaticResolver({"bound.test": (PUBLIC_A,)}),
                connector=MappedConnector({"bound.test": server.port}),
            )
        )
        context = _execution_context(binding)
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"url": "http://bound.test/"}),
            context,
        )

    assert result["type"] == "Success"
    evidence = result["value"]["evidence"]
    assert evidence["source_uri"] == "http://bound.test/"
    assert evidence["final_uri"] == "http://bound.test/"
    assert datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00"))
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_private_resolution_and_peer_mismatch_send_no_http_request() -> None:
    async with LoopbackServer(_response(b"private")) as server:
        private_resolver = StaticResolver({"private.test": ("127.0.0.1",)})
        connector = MappedConnector({"private.test": server.port})
        with pytest.raises(WebReadFailure) as private:
            await SafeWebReader(resolver=private_resolver, connector=connector).read(
                "http://private.test/secret"
            )
        assert isinstance(private.value.error, UnsafeDestination)
        assert private.value.attempts == 0
        assert connector.calls == []

        public_resolver = StaticResolver({"rebind.test": (PUBLIC_A,)})
        peer_mismatch = MappedConnector(
            {"rebind.test": server.port},
            peer_addresses={"rebind.test": PUBLIC_B},
        )
        with pytest.raises(WebReadFailure) as rebound:
            await SafeWebReader(resolver=public_resolver, connector=peer_mismatch).read(
                "http://rebind.test/secret"
            )
        assert isinstance(rebound.value.error, UnsafeDestination)
        assert rebound.value.attempts == 1

    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://safe.test/ok\r\nX-Smuggled: yes",
        "http://safe.test/path\x00tail",
        "http://-bad.test/",
        "http://bad_.test/",
        "http://[2001:4860:4860::8888]/",
    ],
)
async def test_ambiguous_or_smuggling_urls_are_rejected_before_dns(url: str) -> None:
    resolver = StaticResolver({"safe.test": (PUBLIC_A,)})
    connector = MappedConnector({"safe.test": 1})
    with pytest.raises(WebReadFailure):
        await SafeWebReader(resolver=resolver, connector=connector).read(url)
    assert resolver.calls == []
    assert connector.calls == []


@pytest.mark.asyncio
async def test_redirect_to_private_authority_is_rejected_before_second_request() -> None:
    async with LoopbackServer(_response(b"private")) as private_server:
        async with LoopbackServer(
            _redirect(f"http://127.0.0.1:{private_server.port}/metadata")
        ) as public_server:
            resolver = StaticResolver({"public.test": (PUBLIC_A,), "127.0.0.1": ("127.0.0.1",)})
            connector = MappedConnector({"public.test": public_server.port})
            with pytest.raises(WebReadFailure) as rejected:
                await SafeWebReader(resolver=resolver, connector=connector).read(
                    "http://public.test/start"
                )

    assert isinstance(rejected.value.error, UnsafeDestination)
    assert rejected.value.attempts == 1
    assert len(public_server.requests) == 1
    assert private_server.requests == []


@pytest.mark.asyncio
async def test_pre_body_retries_share_one_logical_attempt_ceiling() -> None:
    async with LoopbackServer(_response(b"eventually")) as server:
        connector = MappedConnector(
            {"retry.test": server.port},
            failures_before_connect=2,
        )
        result = await SafeWebReader(
            resolver=StaticResolver({"retry.test": (PUBLIC_A,)}),
            connector=connector,
        ).read("http://retry.test/")

    assert result.value.text == "eventually"
    assert result.attempts == 3
    assert len(connector.calls) == 3
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_effective_attempt_ceiling_tightens_reader_retry_policy() -> None:
    async with LoopbackServer(_response(b"must not reach")) as server:
        connector = MappedConnector({"cap.test": server.port}, failures_before_connect=2)
        with pytest.raises(WebReadFailure) as failed:
            await SafeWebReader(
                resolver=StaticResolver({"cap.test": (PUBLIC_A,)}),
                connector=connector,
            ).read("http://cap.test/", max_requests=1)

    assert isinstance(failed.value.error, UpstreamUnavailable)
    assert failed.value.attempts == 1
    assert len(connector.calls) == 1
    assert server.requests == []


@pytest.mark.asyncio
async def test_profile_zero_attempt_grant_fails_without_reader_dispatch() -> None:
    binding = bind_web_read(MustNotReadReader())
    run_limits = RunLimits(1, 0, 8_192, WEB_READ_SPEC.limits.max_output_bytes, 1, 30.0)
    context = _execution_context(
        binding,
        tool_limits=WEB_READ_SPEC.limits.tightened(max_attempts=0),
        run_limits=run_limits,
        position="turn-1/zero-attempt",
    )

    assert await ToolExecutor.execute(
        binding,
        ParsedJson({"url": "http://never.test/"}),
        context,
    ) == {"type": "Failure", "error": {"type": "UpstreamUnavailable"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "content_type", "expected_text", "expected_title"),
    [
        (
            (
                b"literal &amp;amp; | named &amp; | decimal &#38; | "
                b"nested-decimal &amp;#38; | hex &#x26; | nested-hex &amp;#x26; | "
                b"<script>plain()</script> <b>literal</b>"
            ),
            "text/plain",
            (
                "literal &amp;amp; | named &amp; | decimal &#38; | "
                "nested-decimal &amp;#38; | hex &#x26; | nested-hex &amp;#x26; | "
                "<script>plain()</script> <b>literal</b>"
            ),
            None,
        ),
        (
            (
                b"<html><head><title>Nested &amp;amp; title</title></head><body>"
                b"<p>literal &amp;amp; | named &amp; | decimal &#38; | "
                b"nested-decimal &amp;#38; | hex &#x26; | nested-hex &amp;#x26; | "
                b"encoded &amp;lt;script&amp;gt;still text&amp;lt;/script&amp;gt;</p>"
                b"<script>discard()</script></body></html>"
            ),
            "text/html",
            (
                "Nested &amp; title literal &amp; | named & | decimal & | "
                "nested-decimal &#38; | hex & | nested-hex &#x26; | "
                "encoded &lt;script&gt;still text&lt;/script&gt;"
            ),
            "Nested &amp; title",
        ),
    ],
)
async def test_entity_decoding_is_media_appropriate_and_single_pass(
    body: bytes,
    content_type: str,
    expected_text: str,
    expected_title: str | None,
) -> None:
    async with LoopbackServer(_response(body, content_type=content_type)) as server:
        result = await SafeWebReader(
            resolver=StaticResolver({"entities.test": (PUBLIC_A,)}),
            connector=MappedConnector({"entities.test": server.port}),
        ).read("http://entities.test/")

    assert result.value.text == expected_text
    assert result.value.title == expected_title
    assert result.value.media_type == content_type
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "content_type", "encoding", "expected_text", "expected_extraction"),
    [
        (
            gzip.compress(b"compressed text"),
            "text/plain",
            "gzip",
            "compressed text",
            "plain-text-v2",
        ),
        (
            zlib.compress(b'{"z":1,"a":2}'),
            "application/json",
            "deflate",
            '{"a":2,"z":1}',
            "json-canonical-v1",
        ),
    ],
)
async def test_bounded_streaming_decodes_supported_content(
    body: bytes,
    content_type: str,
    encoding: str,
    expected_text: str,
    expected_extraction: str,
) -> None:
    async with LoopbackServer(
        _response(body, content_type=content_type, headers=(("Content-Encoding", encoding),))
    ) as server:
        result = await SafeWebReader(
            resolver=StaticResolver({"content.test": (PUBLIC_A,)}),
            connector=MappedConnector({"content.test": server.port}),
        ).read("http://content.test/")

    assert result.value.text == expected_text
    assert json.loads(result.value.evidence.locator)["extraction"] == expected_extraction


@pytest.mark.asyncio
async def test_decoded_and_extracted_text_limits_are_independent() -> None:
    compressed_bomb = gzip.compress(b"z" * 65)
    html_body = b"<p>abcdefghijk</p>"

    async with LoopbackServer(
        _response(
            compressed_bomb,
            content_type="text/plain",
            headers=(("Content-Encoding", "gzip"),),
        )
    ) as bomb_server:
        with pytest.raises(WebReadFailure) as bomb:
            await SafeWebReader(
                resolver=StaticResolver({"bomb.test": (PUBLIC_A,)}),
                connector=MappedConnector({"bomb.test": bomb_server.port}),
                limits=WebReadLimits(max_wire_bytes=128, max_decoded_bytes=64),
            ).read("http://bomb.test/")
    assert isinstance(bomb.value.error, TooLarge)
    assert bomb.value.attempts == 1

    async with LoopbackServer(_response(html_body, content_type="text/html")) as text_server:
        result = await SafeWebReader(
            resolver=StaticResolver({"truncate.test": (PUBLIC_A,)}),
            connector=MappedConnector({"truncate.test": text_server.port}),
            limits=WebReadLimits(max_text_bytes=5),
        ).read("http://truncate.test/")
    assert result.value.text == "abcde"
    locator = json.loads(result.value.evidence.locator)
    assert locator["text_truncated"] is True
    assert locator["text_utf8_bytes"] == 5
    assert (
        result.value.evidence.content_sha256 == __import__("hashlib").sha256(html_body).hexdigest()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "content_type", "error_type"),
    [
        (b"payload", "text/plain; charset=rot_13", UnsupportedContent),
        (
            (b"[" * 1_500) + b"0" + (b"]" * 1_500),
            "application/json",
            InvalidUpstreamResponse,
        ),
    ],
)
async def test_malformed_content_is_a_declared_failure(
    body: bytes,
    content_type: str,
    error_type: type,
) -> None:
    async with LoopbackServer(_response(body, content_type=content_type)) as server:
        with pytest.raises(WebReadFailure) as failed:
            await SafeWebReader(
                resolver=StaticResolver({"malformed.test": (PUBLIC_A,)}),
                connector=MappedConnector({"malformed.test": server.port}),
            ).read("http://malformed.test/")

    assert isinstance(failed.value.error, error_type)
    assert failed.value.attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        b"HTTP/1.1 302 Found\r\nLocation: http://[\r\nContent-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: \xb2\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\n",
    ],
)
async def test_malformed_redirect_and_length_terminalize_at_the_binding(response: bytes) -> None:
    async with LoopbackServer(response) as server:
        binding = bind_web_read(
            SafeWebReader(
                resolver=StaticResolver({"malformed.test": (PUBLIC_A,)}),
                connector=MappedConnector({"malformed.test": server.port}),
            )
        )
        context = _execution_context(binding, position="turn-4/malformed-response")
        result = await ToolExecutor.execute(
            binding,
            ParsedJson({"url": "http://malformed.test/"}),
            context,
        )

    assert result == {
        "type": "Failure",
        "error": {"type": "InvalidUpstreamResponse"},
    }
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_peer_close_error_cannot_override_a_terminal_binding_result() -> None:
    class CloseErrorWriter:
        def write(self, _data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            raise OSError("peer reset while closing")

    class CloseErrorConnector:
        async def connect(self, *args: object, **kwargs: object) -> ConnectedStream:
            del args, kwargs
            reader = asyncio.StreamReader()
            reader.feed_data(_response(b"ok"))
            reader.feed_eof()
            return ConnectedStream(reader, CloseErrorWriter(), PUBLIC_A)  # type: ignore[arg-type]

    binding = bind_web_read(
        SafeWebReader(
            resolver=StaticResolver({"close.test": (PUBLIC_A,)}),
            connector=CloseErrorConnector(),  # type: ignore[arg-type]
        )
    )
    context = _execution_context(binding, position="turn-4/close-error")

    result = await ToolExecutor.execute(
        binding,
        ParsedJson({"url": "http://close.test/"}),
        context,
    )

    assert result["type"] == "Success"
    assert result["value"]["text"] == "ok"
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    record = context.recorder.record(context.position)
    assert record.in_flight is False
    assert record.settlement is not None
    assert record.settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_redirect_and_total_request_ceilings_are_global() -> None:
    def redirect_to_self(request: bytes) -> bytes:
        path = request.split(b" ", 2)[1].decode()
        number = int(path.removeprefix("/")) if path != "/" else 0
        return _redirect(f"/{number + 1}")

    async with LoopbackServer(redirect_to_self) as server:
        with pytest.raises(WebReadFailure) as failed:
            await SafeWebReader(
                resolver=StaticResolver({"loop.test": (PUBLIC_A,)}),
                connector=MappedConnector({"loop.test": server.port}),
                limits=WebReadLimits(max_redirects=2, max_requests=8),
            ).read("http://loop.test/0")

    assert isinstance(failed.value.error, InvalidUpstreamResponse)
    assert failed.value.attempts == 3
    assert len(server.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (_response(b"pdf", content_type="application/pdf"), UnsupportedContent),
        (_response(b"busy", status="429 Too Many Requests"), RateLimited),
        (_response(b"x" * 33, content_type="text/plain"), TooLarge),
        (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Content-Length: 1\r\nContent-Length: 2\r\n\r\nx",
            InvalidUpstreamResponse,
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n" + b"f" * 17 + b"\r\n",
            TooLarge,
        ),
    ],
)
async def test_mime_rate_size_and_ambiguous_headers_fail_closed(
    response: bytes,
    error_type: type,
) -> None:
    async with LoopbackServer(response) as server:
        reader = SafeWebReader(
            resolver=StaticResolver({"failure.test": (PUBLIC_A,)}),
            connector=MappedConnector({"failure.test": server.port}),
            limits=WebReadLimits(max_wire_bytes=32, max_decoded_bytes=64),
        )
        with pytest.raises(WebReadFailure) as failed:
            await reader.read("http://failure.test/")

    assert isinstance(failed.value.error, error_type)
    assert failed.value.attempts == 1


@pytest.mark.asyncio
async def test_deadline_and_ambient_proxy_are_owned_by_direct_reader() -> None:
    proxy_response = _response(b"proxy must not receive target")

    async def hang(_: bytes) -> None:
        await asyncio.sleep(1)
        return None

    async with LoopbackServer(proxy_response) as proxy:
        async with LoopbackServer(hang) as target:
            resolver = StaticResolver({"deadline.test": (PUBLIC_A,)})
            connector = MappedConnector({"deadline.test": target.port})
            binding = bind_web_read(
                SafeWebReader(
                    resolver=resolver,
                    connector=connector,
                    limits=WebReadLimits(deadline_seconds=0.05),
                )
            )
            context = _execution_context(binding, position="turn-1/read-deadline")
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
                monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{proxy.port}")
                result = await ToolExecutor.execute(
                    binding,
                    ParsedJson({"url": "http://deadline.test/"}),
                    context,
                )

    assert result == {"type": "Failure", "error": {"type": "DeadlineExceeded"}}
    assert isinstance(context.recorder, InMemoryPositionRecorder)
    settlement = context.recorder.record(context.position).settlement
    assert settlement is not None
    assert settlement.actual_attempts == 1
    assert len(target.requests) == 1
    assert proxy.requests == []


@pytest.mark.asyncio
async def test_effective_output_limit_trims_read_text_and_locator_or_returns_budget_failure() -> (
    None
):
    body = ('é<&\\"' * 200).encode()
    async with LoopbackServer(_response(body)) as server:
        binding = bind_web_read(
            SafeWebReader(
                resolver=StaticResolver({"bounded-read.test": (PUBLIC_A,)}),
                connector=MappedConnector({"bounded-read.test": server.port}),
            )
        )
        bounded_context = _execution_context(
            binding,
            tool_limits=WEB_READ_SPEC.limits.tightened(max_output_bytes=700),
            position="turn-1/read-bounded",
        )
        bounded = await ToolExecutor.execute(
            binding,
            ParsedJson({"url": "http://bounded-read.test/"}),
            bounded_context,
        )
        minimal_context = _execution_context(
            binding,
            tool_limits=WEB_READ_SPEC.limits.tightened(max_output_bytes=54),
            position="turn-1/read-minimal",
        )
        minimal = await ToolExecutor.execute(
            binding,
            ParsedJson({"url": "http://bounded-read.test/"}),
            minimal_context,
        )

    assert bounded["type"] == "Success"
    assert len(canonical_json_bytes(bounded)) <= 700
    assert len(bounded["value"]["text"].encode()) < len(body)
    locator = json.loads(bounded["value"]["evidence"]["locator"])
    assert locator["text_truncated"] is True
    assert locator["text_utf8_bytes"] == len(bounded["value"]["text"].encode())
    assert bounded["value"]["evidence"]["content_sha256"] == (
        __import__("hashlib").sha256(body).hexdigest()
    )
    assert minimal == {"type": "Failure", "error": {"type": "BudgetExceeded"}}
    assert len(server.requests) == 2
    recorder = minimal_context.recorder
    assert isinstance(recorder, InMemoryPositionRecorder)
    settlement = recorder.record(minimal_context.position).settlement
    assert settlement is not None
    assert settlement.actual_attempts == 1


@pytest.mark.asyncio
async def test_tls_uses_original_hostname_for_sni_and_host(tmp_path: Path) -> None:
    certificate_path = tmp_path / "certificate.pem"
    key_path = tmp_path / "key.pem"
    ca_path = tmp_path / "ca.pem"
    ca = trustme.CA()
    certificate = ca.issue_cert("a.test", "b.test")
    certificate.cert_chain_pems[0].write_to_path(certificate_path)
    certificate.private_key_pem.write_to_path(key_path)
    ca.cert_pem.write_to_path(ca_path)
    seen_sni: list[str | None] = []
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate_path, key_path)
    server_context.sni_callback = lambda _socket, hostname, _context: seen_sni.append(hostname)
    client_context = ssl.create_default_context(cafile=str(ca_path))

    async with LoopbackServer(_response(b"secure"), ssl_context=server_context) as server:
        result = await SafeWebReader(
            resolver=StaticResolver({"a.test": (PUBLIC_A,)}),
            connector=MappedConnector(
                {"a.test": server.port},
                ssl_context=client_context,
            ),
        ).read("https://a.test/secure")

    assert seen_sni == ["a.test"]
    assert b"Host: a.test\r\n" in server.requests[0]
    assert result.hops[0].peer_address == PUBLIC_A


@pytest.mark.asyncio
async def test_production_connector_reports_the_real_connected_peer() -> None:
    async with LoopbackServer(_response(b"unused")) as server:
        connection = await DirectConnector().connect(
            "127.0.0.1",
            server.port,
            hostname="localhost",
            tls=False,
            timeout_seconds=1.0,
        )
        try:
            assert connection.peer_address == "127.0.0.1"
        finally:
            connection.writer.close()
            await connection.writer.wait_closed()

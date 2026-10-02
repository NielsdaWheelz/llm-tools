"""Temporary real-stream proof for ignored metadata and strict consumed fields."""

import pytest

from conformance.test_web_read import (
    PUBLIC_A,
    LoopbackServer,
    MappedConnector,
    StaticResolver,
    _response,
)
from llm_tools.web.contracts import InvalidUpstreamResponse, WebReadFailure
from llm_tools.web.reader import SafeWebReader


@pytest.mark.asyncio
async def test_unused_repeated_response_fields_do_not_reject_body() -> None:
    async with LoopbackServer(
        _response(
            b"a public observation",
            headers=(
                ("X-Powered-By", "first"),
                ("X-Powered-By", "second"),
                ("Set-Cookie", "ignored-one=1"),
                ("Set-Cookie", "ignored-two=2"),
                ("Vary", "Accept-Encoding"),
                ("Vary", "Accept-Language"),
            ),
        )
    ) as server:
        reader = SafeWebReader(
            resolver=StaticResolver({"metadata.test": (PUBLIC_A,)}),
            connector=MappedConnector({"metadata.test": server.port}),
        )
        result = await reader.read("http://metadata.test/")
    assert result.value.text == "a public observation"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    (
        (("Content-Length", "4"),),
        (("Content-Type", "text/plain"),),
        (("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "chunked")),
        (("Content-Encoding", "identity"), ("Content-Encoding", "gzip")),
        (("Location", "/one"), ("Location", "/two")),
    ),
)
async def test_repeated_consumed_fields_remain_invalid(fields) -> None:
    async with LoopbackServer(_response(b"body", headers=fields)) as server:
        reader = SafeWebReader(
            resolver=StaticResolver({"ambiguous.test": (PUBLIC_A,)}),
            connector=MappedConnector({"ambiguous.test": server.port}),
        )
        with pytest.raises(WebReadFailure) as failure:
            await reader.read("http://ambiguous.test/")
    assert isinstance(failure.value.error, InvalidUpstreamResponse)

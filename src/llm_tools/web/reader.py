"""Fail-closed direct Web reader with pinned DNS, peer, and resource policy."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import socket
import ssl
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from llm_tools.evidence import EvidenceReceipt, sha256_hex
from llm_tools.schema import canonical_json_bytes
from llm_tools.web.contracts import (
    InvalidUpstreamResponse,
    InvalidUrl,
    RateLimited,
    TooLarge,
    UnsafeDestination,
    UnsupportedContent,
    UpstreamUnavailable,
    WebReadDeadline,
    WebReadFailure,
    WebReadHop,
    WebReadResponse,
    WebReadSuccess,
)

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_ACCEPTED_MEDIA = frozenset(
    {"application/json", "application/xhtml+xml", "text/html", "text/plain"}
)
_DENIED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "64:ff9b:1::/48",
        "100::/64",
        "2001:db8::/32",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)
_TOKEN = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_MAX_URL_CHARS = 4_096


@dataclass(frozen=True, slots=True)
class WebReadLimits:
    max_redirects: int = 5
    max_requests: int = 8
    max_pre_body_retries: int = 2
    max_wire_bytes: int = 2 * 1_024 * 1_024
    max_decoded_bytes: int = 4 * 1_024 * 1_024
    max_text_bytes: int = 64 * 1_024
    deadline_seconds: float = 20.0

    def __post_init__(self) -> None:
        integers = (
            self.max_redirects,
            self.max_requests,
            self.max_pre_body_retries,
            self.max_wire_bytes,
            self.max_decoded_bytes,
            self.max_text_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("Web reader limits must be integers")
        if (
            any(value < 0 for value in integers)
            or min(
                self.max_requests,
                self.max_wire_bytes,
                self.max_decoded_bytes,
                self.max_text_bytes,
            )
            <= 0
        ):
            raise ValueError("Web reader limits must be positive; redirects/retries may be zero")
        if isinstance(self.deadline_seconds, bool) or not isinstance(
            self.deadline_seconds, (int, float)
        ):
            raise TypeError("Web reader deadline must be numeric")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("Web reader deadline must be positive and finite")


@dataclass(frozen=True, slots=True)
class ConnectedStream:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    peer_address: str


class Resolver(Protocol):
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]: ...


class Connector(Protocol):
    async def connect(
        self,
        address: str,
        port: int,
        *,
        hostname: str,
        tls: bool,
        timeout_seconds: float,
    ) -> ConnectedStream: ...


class SystemResolver:
    """Resolve every candidate through the event-loop OS resolver."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        infos = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
        if not addresses:
            raise OSError("hostname resolved to no addresses")
        return addresses


class DirectConnector:
    """Open a direct socket to one admitted address while retaining Host/SNI."""

    async def connect(
        self,
        address: str,
        port: int,
        *,
        hostname: str,
        tls: bool,
        timeout_seconds: float,
    ) -> ConnectedStream:
        context = ssl.create_default_context() if tls else None
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                address,
                port,
                ssl=context,
                server_hostname=hostname if tls else None,
            ),
            timeout=timeout_seconds,
        )
        peer = writer.get_extra_info("peername")
        if not isinstance(peer, tuple) or not peer or not isinstance(peer[0], str):
            writer.close()
            await writer.wait_closed()
            raise OSError("connected socket has no IP peer")
        return ConnectedStream(reader=reader, writer=writer, peer_address=peer[0])


@dataclass(frozen=True, slots=True)
class _NormalizedUrl:
    uri: str
    scheme: str
    hostname: str
    port: int
    authority: str
    target: str


@dataclass(frozen=True, slots=True)
class _ResponseHead:
    status_code: int
    headers: Mapping[str, str]


class SafeWebReader:
    """Read one URL without proxy/cookie/auth/browser state or persistence."""

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        connector: Connector | None = None,
        limits: WebReadLimits | None = None,
    ) -> None:
        self._resolver = resolver or SystemResolver()
        self._connector = connector or DirectConnector()
        self._limits = limits or WebReadLimits()

    async def read(
        self,
        raw_url: str,
        *,
        max_requests: int | None = None,
        deadline_seconds: float | None = None,
    ) -> WebReadResponse:
        request_limit = min(
            self._limits.max_requests if max_requests is None else max_requests,
            self._limits.max_requests,
        )
        deadline_limit = min(
            self._limits.deadline_seconds if deadline_seconds is None else deadline_seconds,
            self._limits.deadline_seconds,
        )
        if request_limit <= 0 or deadline_limit <= 0:
            raise ValueError("effective Web reader attempts and deadline must be positive")
        deadline = time.monotonic() + deadline_limit
        attempts = [0]
        try:
            async with asyncio.timeout(deadline_limit):
                return await self._read(raw_url, deadline, request_limit, attempts)
        except TimeoutError as exc:
            raise WebReadDeadline(attempts=attempts[0]) from exc
        except asyncio.CancelledError as exc:
            if time.monotonic() >= deadline:
                raise WebReadDeadline(attempts=attempts[0]) from exc
            raise

    async def _read(
        self,
        raw_url: str,
        deadline: float,
        max_requests: int,
        attempts_tracker: list[int],
    ) -> WebReadResponse:
        source = _normalize_url(raw_url)
        current = source
        attempts = 0
        retries = 0
        redirects = 0
        wire_bytes = 0
        decoded_bytes = 0
        hops: list[WebReadHop] = []

        while True:
            try:
                addresses = await self._resolver.resolve(current.hostname, current.port)
            except (OSError, socket.gaierror) as exc:
                raise WebReadFailure(UpstreamUnavailable(), attempts=attempts) from exc
            if not addresses or any(not _is_public(address) for address in addresses):
                raise WebReadFailure(UnsafeDestination(), attempts=attempts)

            connection: ConnectedStream | None = None
            connected_address = ""
            while connection is None:
                if attempts >= max_requests:
                    raise WebReadFailure(UpstreamUnavailable(), attempts=attempts)
                attempts += 1
                attempts_tracker[0] = attempts
                connected_address = addresses[(attempts - 1) % len(addresses)]
                try:
                    connection = await self._connector.connect(
                        connected_address,
                        current.port,
                        hostname=current.hostname,
                        tls=current.scheme == "https",
                        timeout_seconds=_remaining(deadline),
                    )
                except (OSError, ssl.SSLError, asyncio.IncompleteReadError) as exc:
                    if retries >= self._limits.max_pre_body_retries:
                        raise WebReadFailure(UpstreamUnavailable(), attempts=attempts) from exc
                    retries += 1
                    continue

            try:
                if not _is_public(connection.peer_address) or not _same_address(
                    connection.peer_address, connected_address
                ):
                    raise WebReadFailure(UnsafeDestination(), attempts=attempts)
                request = (
                    f"GET {current.target} HTTP/1.1\r\n"
                    f"Host: {current.authority}\r\n"
                    "Accept: text/html, application/xhtml+xml, text/plain, application/json\r\n"
                    "Accept-Encoding: gzip, deflate, identity\r\n"
                    "Connection: close\r\n"
                    "User-Agent: llm-tools/0.1\r\n\r\n"
                ).encode("ascii")
                connection.writer.write(request)
                await connection.writer.drain()
                try:
                    head_bytes = await connection.reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
                    raise WebReadFailure(InvalidUpstreamResponse(), attempts=attempts) from exc
                if len(head_bytes) > 32 * 1_024:
                    raise WebReadFailure(InvalidUpstreamResponse(), attempts=attempts)
                head = _parse_head(head_bytes)
                hops.append(
                    WebReadHop(
                        uri=current.uri,
                        peer_address=connection.peer_address,
                        status_code=head.status_code,
                    )
                )
                if head.status_code in _REDIRECTS:
                    location = head.headers.get("location")
                    if not location or redirects >= self._limits.max_redirects:
                        raise WebReadFailure(InvalidUpstreamResponse(), attempts=attempts)
                    redirects += 1
                    try:
                        redirect_url = urljoin(current.uri, location)
                    except (UnicodeError, ValueError) as exc:
                        raise WebReadFailure(InvalidUpstreamResponse(), attempts=attempts) from exc
                    try:
                        current = _normalize_url(redirect_url)
                    except WebReadFailure as failure:
                        raise WebReadFailure(failure.error, attempts=attempts) from failure
                    continue
                if head.status_code == 429:
                    raise WebReadFailure(RateLimited(), attempts=attempts)
                if head.status_code < 200 or head.status_code >= 300:
                    raise WebReadFailure(UpstreamUnavailable(), attempts=attempts)

                media_type, charset = _parse_content_type(head.headers.get("content-type", ""))
                if media_type not in _ACCEPTED_MEDIA:
                    raise WebReadFailure(UnsupportedContent(), attempts=attempts)
                encoding = head.headers.get("content-encoding", "identity").lower().strip()
                if encoding not in {"identity", "gzip", "deflate"}:
                    raise WebReadFailure(UnsupportedContent(), attempts=attempts)
                declared_length = _content_length(head.headers)
                if declared_length is not None:
                    if wire_bytes + declared_length > self._limits.max_wire_bytes:
                        raise WebReadFailure(TooLarge(), attempts=attempts)

                body, consumed = await _read_body(
                    connection.reader,
                    head.headers,
                    max_bytes=self._limits.max_wire_bytes - wire_bytes,
                )
                wire_bytes += consumed
                decoded = _decode_content_encoding(
                    body, encoding, self._limits.max_decoded_bytes - decoded_bytes
                )
                decoded_bytes += len(decoded)
                text, title, extraction = _extract(decoded, media_type, charset)
                bounded_text, truncated = _truncate_utf8(text, self._limits.max_text_bytes)
                locator = canonical_json_bytes(
                    {
                        "content_encoding": encoding,
                        "extraction": extraction,
                        "representation": "decoded-entity-bytes",
                        "text_truncated": truncated,
                        "text_utf8_bytes": len(bounded_text.encode("utf-8")),
                        "type": "web.document",
                        "version": 1,
                    }
                ).decode("utf-8")
                observed_at = datetime.now(tz=UTC)
                receipt = EvidenceReceipt(
                    source_uri=source.uri,
                    final_uri=current.uri,
                    observed_at=observed_at,
                    content_sha256=sha256_hex(decoded),
                    media_type=media_type,
                    locator=locator,
                )
                return WebReadResponse(
                    value=WebReadSuccess(
                        final_url=current.uri,
                        title=title,
                        media_type=media_type,
                        text=bounded_text,
                        evidence=receipt,
                    ),
                    attempts=attempts,
                    hops=tuple(hops),
                )
            except WebReadFailure as failure:
                if failure.attempts == attempts:
                    raise
                raise WebReadFailure(failure.error, attempts=attempts) from failure
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                ConnectionError,
                OSError,
            ) as exc:
                raise WebReadFailure(InvalidUpstreamResponse(), attempts=attempts) from exc
            finally:
                try:
                    connection.writer.close()
                    await connection.writer.wait_closed()
                except OSError:
                    pass


def _normalize_url(raw_url: str) -> _NormalizedUrl:
    if not isinstance(raw_url, str) or not raw_url or len(raw_url) > _MAX_URL_CHARS:
        raise WebReadFailure(InvalidUrl(), attempts=0)
    try:
        parsed = urlsplit(raw_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise ValueError
        if parsed.hostname is None or parsed.hostname.endswith("."):
            raise ValueError
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        if any(ord(character) < 33 or ord(character) == 127 for character in raw_url):
            raise ValueError
        labels = hostname.split(".")
        if (
            not hostname
            or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or any(not (character.isalnum() or character == "-") for character in label)
                for label in labels
            )
            or len(hostname) > 253
        ):
            raise ValueError
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            error = InvalidUrl() if _is_public(hostname) else UnsafeDestination()
            raise WebReadFailure(error, attempts=0)
        scheme = parsed.scheme.lower()
        port = parsed.port or (443 if scheme == "https" else 80)
        if not 1 <= port <= 65_535:
            raise ValueError
        default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
        authority = hostname if default_port else f"{hostname}:{port}"
        path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        query = quote(parsed.query, safe="!$&'()*+,;=:/?@%-._~")
        target = path + (f"?{query}" if query else "")
        uri = urlunsplit((scheme, authority, path, query, ""))
        if len(uri) > _MAX_URL_CHARS:
            raise ValueError
        return _NormalizedUrl(uri, scheme, hostname, port, authority, target)
    except WebReadFailure:
        raise
    except (UnicodeError, ValueError):
        raise WebReadFailure(InvalidUrl(), attempts=0) from None


def _is_public(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_global and not any(address in network for network in _DENIED_NETWORKS)


def _same_address(left: str, right: str) -> bool:
    try:
        return ipaddress.ip_address(left) == ipaddress.ip_address(right)
    except ValueError:
        return False


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _parse_head(raw: bytes) -> _ResponseHead:
    lines = raw.split(b"\r\n")
    if not lines or len(lines[0]) > 256:
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
    parts = lines[0].split(b" ", 2)
    if len(parts) < 2 or parts[0] != b"HTTP/1.1" or len(parts[1]) != 3 or not parts[1].isdigit():
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(b":")
        if not separator or not _TOKEN.fullmatch(name):
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
        key = name.decode("ascii").lower()
        if key not in {
            "content-length",
            "transfer-encoding",
            "content-type",
            "content-encoding",
            "location",
        }:
            continue
        if key in headers:
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
        try:
            headers[key] = value.decode("latin-1").strip()
        except UnicodeDecodeError:
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1) from None
    if "transfer-encoding" in headers and "content-length" in headers:
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
    return _ResponseHead(status_code=int(parts[1]), headers=headers)


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdecimal():
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
    digits = raw.lstrip("0") or "0"
    if len(digits) > 20:
        raise WebReadFailure(TooLarge(), attempts=1)
    return int(digits)


async def _read_body(
    reader: asyncio.StreamReader,
    headers: Mapping[str, str],
    *,
    max_bytes: int,
) -> tuple[bytes, int]:
    transfer = headers.get("transfer-encoding", "").lower()
    if transfer:
        if transfer != "chunked":
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
        return await _read_chunked(reader, max_bytes=max_bytes)
    length = _content_length(headers)
    if length is not None:
        if length > max_bytes:
            raise WebReadFailure(TooLarge(), attempts=1)
        return await reader.readexactly(length), length
    data = bytearray()
    while True:
        chunk = await reader.read(min(65_536, max_bytes - len(data) + 1))
        if not chunk:
            return bytes(data), len(data)
        data.extend(chunk)
        if len(data) > max_bytes:
            raise WebReadFailure(TooLarge(), attempts=1)


async def _read_chunked(reader: asyncio.StreamReader, *, max_bytes: int) -> tuple[bytes, int]:
    data = bytearray()
    while True:
        line = await reader.readuntil(b"\r\n")
        raw_size = line[:-2].split(b";", 1)[0]
        if not raw_size or any(byte not in b"0123456789abcdefABCDEF" for byte in raw_size):
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
        if len(raw_size) > 16:
            raise WebReadFailure(TooLarge(), attempts=1)
        size = int(raw_size, 16)
        if size == 0:
            trailer = await reader.readuntil(b"\r\n")
            if trailer != b"\r\n":
                raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
            return bytes(data), len(data)
        if len(data) + size > max_bytes:
            raise WebReadFailure(TooLarge(), attempts=1)
        data.extend(await reader.readexactly(size))
        if await reader.readexactly(2) != b"\r\n":
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)


def _decode_content_encoding(body: bytes, encoding: str, remaining: int) -> bytes:
    try:
        if encoding == "identity":
            if len(body) > remaining:
                raise WebReadFailure(TooLarge(), attempts=1)
            return body
        window_bits = zlib.MAX_WBITS | 16 if encoding == "gzip" else zlib.MAX_WBITS
        decoder = zlib.decompressobj(window_bits)
        decoded = bytearray()
        pending = body
        while pending:
            chunk = decoder.decompress(pending, remaining - len(decoded) + 1)
            decoded.extend(chunk)
            if len(decoded) > remaining:
                raise WebReadFailure(TooLarge(), attempts=1)
            pending = decoder.unconsumed_tail
            if not pending:
                break
        tail = decoder.flush(remaining - len(decoded) + 1)
        decoded.extend(tail)
        if len(decoded) > remaining:
            raise WebReadFailure(TooLarge(), attempts=1)
        if not decoder.eof or decoder.unused_data:
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1)
        return bytes(decoded)
    except zlib.error:
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1) from None


def _parse_content_type(value: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in value.split(";")]
    media_type = parts[0].lower()
    charset: str | None = None
    for part in parts[1:]:
        key, separator, raw = part.partition("=")
        if separator and key.lower().strip() == "charset":
            charset = raw.strip().strip('"').lower()
    return media_type, charset


def _decode_text(content: bytes, charset: str | None) -> str:
    encoding = (charset or "utf-8").replace("_", "-").lower()
    aliases = {
        "ascii": "ascii",
        "iso-8859-1": "iso-8859-1",
        "latin-1": "iso-8859-1",
        "us-ascii": "ascii",
        "utf-8": "utf-8",
        "utf8": "utf-8",
    }
    encoding = aliases.get(encoding, "")
    if not encoding:
        raise WebReadFailure(UnsupportedContent(), attempts=1)
    try:
        return content.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        raise WebReadFailure(UnsupportedContent(), attempts=1) from None


def _extract(content: bytes, media_type: str, charset: str | None) -> tuple[str, str | None, str]:
    if media_type == "application/json":
        try:
            value = json.loads(_decode_text(content, charset))
            text = canonical_json_bytes(value).decode("utf-8")
        except (json.JSONDecodeError, RecursionError, ValueError, UnicodeDecodeError):
            raise WebReadFailure(InvalidUpstreamResponse(), attempts=1) from None
        return text, None, "json-canonical-v1"
    decoded = _decode_text(content, charset)
    if media_type == "text/plain":
        return _collapse_text(decoded), None, "plain-text-v2"
    parser = _VisibleTextParser()
    try:
        parser.feed(decoded)
        parser.close()
    except Exception:
        raise WebReadFailure(InvalidUpstreamResponse(), attempts=1) from None
    return _collapse_text(" ".join(parser.text)), parser.title, "html-visible-text-v2"


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self._excluded = 0
        self._in_title = False
        self._title: list[str] = []

    @property
    def title(self) -> str | None:
        value = _collapse_text(" ".join(self._title))
        return value[:512] if value else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "template", "noscript"}:
            self._excluded += 1
        elif lowered == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "template", "noscript"} and self._excluded:
            self._excluded -= 1
        elif lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._excluded:
            return
        if self._in_title:
            self._title.append(data)
        self.text.append(data)


def _collapse_text(value: str) -> str:
    return " ".join(value.split())


def _truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    bounded = encoded[:max_bytes]
    while True:
        try:
            return bounded.decode("utf-8"), True
        except UnicodeDecodeError as exc:
            bounded = bounded[: exc.start]

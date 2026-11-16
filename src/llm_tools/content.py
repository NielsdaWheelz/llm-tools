"""Pure UTF-8 projection and envelope operations shared by source readers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from html.parser import HTMLParser

from llm_tools.schema import JsonObject, canonical_json_bytes


def projection_digest(source: JsonObject, projection: str, text_sha256: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"source": source, "projection": projection, "text_sha256": text_sha256}
        )
    ).hexdigest()


def utf8_slice(content: bytes, offset: int, max_bytes: int) -> bytes:
    """Slice at character boundaries; reject ranges that cannot make progress."""
    if (
        type(offset) is not int
        or type(max_bytes) is not int
        or not 0 <= offset <= len(content)
        or max_bytes < 1
        or (offset < len(content) and content[offset] & 0xC0 == 0x80)
    ):
        raise ValueError("invalid UTF-8 range")
    end = min(len(content), offset + max_bytes)
    while end > offset and end < len(content) and content[end] & 0xC0 == 0x80:
        end -= 1
    if end == offset and offset < len(content):
        raise ValueError("range cannot contain the next character")
    result = content[offset:end]
    result.decode("utf-8", errors="strict")
    return result


def fit_text_chunk(
    window: bytes,
    *,
    offset: int,
    total_bytes: int,
    max_bytes: int,
    build: Callable[[str, int, int | None], JsonObject],
    max_output_bytes: int,
) -> JsonObject:
    """Fit a bounded window beginning at offset into the complete success envelope."""
    if (
        type(offset) is not int
        or type(total_bytes) is not int
        or not 0 <= offset <= total_bytes <= 2**53 - 1
        or len(window) > total_bytes - offset
    ):
        raise ValueError("invalid text window")
    bounded = utf8_slice(window, 0, max_bytes)
    if not bounded and offset != total_bytes:
        raise ValueError("text window cannot make progress")
    text = bounded.decode("utf-8")
    end = offset + len(bounded)
    complete = build(text, end, end if end < total_bytes else None)
    if len(canonical_json_bytes({"type": "Success", "value": complete})) <= max_output_bytes:
        return complete
    low, high = (1 if text else 0), len(text) - 1
    best: JsonObject | None = None
    while low <= high:
        count = (low + high) // 2
        candidate = text[:count]
        end = offset + len(candidate.encode("utf-8"))
        value = build(candidate, end, end if end < total_bytes else None)
        if len(canonical_json_bytes({"type": "Success", "value": value})) <= max_output_bytes:
            best = value
            low = count + 1
        else:
            high = count - 1
    if best is None:
        raise ValueError("text envelope cannot make progress")
    return best


def html_visible_text(decoded: str) -> tuple[str, str | None]:
    """Project HTML inertly, without fetching or executing any content."""
    parser = _VisibleTextParser()
    parser.feed(decoded)
    parser.close()
    return " ".join(" ".join(parser.text).split()), parser.title


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self._excluded = 0
        self._in_title = False
        self._title: list[str] = []

    @property
    def title(self) -> str | None:
        value = " ".join(" ".join(self._title).split())
        return value[:512] if value else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "template", "noscript"}:
            self._excluded += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "template", "noscript"} and self._excluded:
            self._excluded -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._excluded:
            return
        if self._in_title:
            self._title.append(data)
        self.text.append(data)

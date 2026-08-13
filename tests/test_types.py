"""Web-search type tests."""

from __future__ import annotations

from typing import Any

import pytest

from llm_tools.web.contracts import WebSearchRequest, WebSearchResultType


def test_request_normalizes_simple_fields() -> None:
    request = WebSearchRequest(
        query="  Brave   Search  ",
        result_type=WebSearchResultType.WEB,
        country="us",
        search_lang="EN",
        allowed_domains=("https://Example.com/path",),
        blocked_domains=("spam.example:443",),
    )

    assert request.query == "Brave Search"
    assert request.result_type == WebSearchResultType.WEB
    assert request.country == "US"
    assert request.search_lang == "en"
    assert request.allowed_domains == ("example.com",)
    assert request.blocked_domains == ("spam.example",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"limit": True}, "limit"),
        ({"freshness_days": True}, "freshness"),
        ({"max_attempts": 1.5}, "attempt"),
    ],
)
def test_request_rejects_noncanonical_integer_policy(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        WebSearchRequest(query="valid query", **changes)

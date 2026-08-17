"""Web-search type tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from llm_tools.web.contracts import WebSearchInput, WebSearchRequest, WebSearchResultType


def test_request_normalizes_simple_fields() -> None:
    request = WebSearchRequest(
        query="  Brave   Search  ",
        result_type=WebSearchResultType.WEB,
        limit=20,
        country="us",
        search_lang="EN",
        allowed_domains=("https://Example.com/path",),
        blocked_domains=("spam.example:443",),
    )

    assert request.query == "Brave Search"
    assert request.result_type == WebSearchResultType.WEB
    assert request.limit == 20
    assert request.country == "US"
    assert request.search_lang == "en"
    assert request.allowed_domains == ("example.com",)
    assert request.blocked_domains == ("spam.example",)


def test_programmatic_request_is_broader_than_model_visible_input() -> None:
    one_character = WebSearchRequest(query="x")
    many_words = WebSearchRequest(query=" ".join(f"term-{index}" for index in range(51)))

    assert one_character.query == "x"
    assert len(many_words.query.split()) == 51
    with pytest.raises(ValidationError):
        WebSearchInput(query="x", freshness_days=None)
    with pytest.raises(ValidationError):
        WebSearchInput(query=many_words.query, freshness_days=None)


def test_request_rejects_limit_above_the_host_owned_ceiling() -> None:
    with pytest.raises(ValueError, match="between 1 and 20"):
        WebSearchRequest(query="valid query", limit=21)


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

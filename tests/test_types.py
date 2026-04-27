"""Web-search type tests."""

from __future__ import annotations

from web_search_tool.types import WebSearchRequest, WebSearchResultType


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

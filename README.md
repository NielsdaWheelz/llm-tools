# web-search-tool

Small async Python package for public web-search provider access.

The first release supports Brave Search. Callers own HTTP client lifecycle so the package can be
used inside web apps, workers, and scripts without creating hidden connection pools.

## Install

```bash
uv add web-search-tool
```

## Example

```python
import httpx

from web_search_tool.brave import BraveSearchProvider
from web_search_tool.types import WebSearchRequest


async def main() -> None:
    async with httpx.AsyncClient() as client:
        provider = BraveSearchProvider(client, api_key="...")
        response = await provider.search(WebSearchRequest(query="Brave Search API docs"))
        print("retrieved at", response.retrieved_at)
        for result in response.results:
            print(result.title, result.url)
```

## Supported Runtime

- Python 3.12+
- `httpx` 0.28+

## Scope

This package normalizes search requests, results, retrieval timestamps, and provider errors. It does
not persist results, render citations, cache responses, scrape pages, summarize pages, or manage API
keys.

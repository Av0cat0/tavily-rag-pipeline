import os
from tavily import AsyncTavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Any, Callable, Coroutine, Optional
import asyncio

TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")


class TavilySearchError(Exception):
    """Custom exception raised for Tavily search-related errors."""
    pass


def retry_async(*dargs: Any, **dkwargs: Any) -> Callable:
    """
    Decorator to apply retry logic to async functions using tenacity.

    Wraps the function in `asyncio.coroutine()` so `tenacity.retry` can be applied to it.
    """
    def wrapper(fn: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        return retry(*dargs, **dkwargs)(asyncio.coroutine(fn))
    return wrapper


@retry_async(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TavilySearchError)
)
async def tavily_search(query: str, search_depth: str = "basic") -> str:
    """
    Performs an async search using the Tavily API with retries.

    Args:
        query: The search query.
        search_depth: Either "basic" or "advanced", determines depth of results.

    Returns:
        A string representing formatted search results context.
    """
    if len(query) > 400:
        raise TavilySearchError("Query is too long. Max query length is 400 characters.")
    if not TAVILY_API_KEY:
        raise TavilySearchError("Tavily API key not found.")
    try:
        client = AsyncTavilyClient(TAVILY_API_KEY)
        response = await client.search(
            query=query,
            search_depth=search_depth,  # "advanced" ideal for better quality
            include_answer=True,
            include_raw_content=False,
            max_results=10,
        )
        return format_context_from_tavily(response)
    except Exception as e:
        raise TavilySearchError(f"[Tavily SDK Error] {e}")


def format_context_from_tavily(response: dict, score_max_diff: float = 0.08, max_results: int = 4) -> str:
    """
    Filters and formats search results from Tavily response.

    Args:
        response: The response dictionary from Tavily.
        score_max_diff: Maximum score drop allowed from top result.
        max_results: Number of top results to include.

    Returns:
        A formatted string containing relevant content snippets.
    """
    results = response.get("results", [])
    if not results:
        return "No relevant information found."

    sorted_results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    top_score = sorted_results[0].get("score", 0)
    filtered = [r for r in sorted_results if (top_score - r.get("score", 0)) <= score_max_diff][:max_results]
    if not filtered:
        return "No high-confidence sources available."

    context_parts = [f"{r['title']}:\n{r['content']}\n\n" for r in filtered]
    return "\n\n".join(context_parts)

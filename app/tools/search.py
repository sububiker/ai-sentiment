import logging

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search_web(query: str) -> str:
    logger.info("Searching web: %s", query)

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return f"No web results found for: {query}"

    formatted = "\n".join(
        f"- {r['title']}: {r['body']}" for r in results
    )
    logger.info("search_web returned %d results", len(results))
    return formatted[:3000]

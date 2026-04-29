import logging

import anthropic

from config import settings
from models.schemas import PredictResponse
from tools.fetcher import fetch_content, is_url
from agents.analyzer import analyze_sentiment
from agents.critic import reflect_on_sentiment

logger = logging.getLogger(__name__)


def run_pipeline(client: anthropic.Anthropic, input_text: str) -> PredictResponse:
    """
    3-step pipeline:
      1. Fetch  — scrape URL if input is a link, otherwise use raw text
      2. Analyze — classify sentiment with Claude
      3. Reflect — re-evaluate if score lands in the uncertainty band (0.40–0.60)
    """
    # ── Step 1: Fetch ────────────────────────────────────────────────────────
    source = "text"
    content = input_text.strip()

    if is_url(content):
        logger.info("Step 1: Input is a URL — fetching content")
        content = fetch_content(content)
        source = "url"
    else:
        logger.info("Step 1: Input is plain text — skipping fetch")

    # ── Step 2: Analyze ──────────────────────────────────────────────────────
    logger.info("Step 2: Running sentiment analysis")
    result = analyze_sentiment(client, content)

    # ── Step 3: Reflect ──────────────────────────────────────────────────────
    reflected = False
    if settings.reflection_low <= result["score"] <= settings.reflection_high:
        logger.info("Step 3: Score %.3f is uncertain — triggering reflection", result["score"])
        result = reflect_on_sentiment(client, content, result)
        reflected = True
    else:
        logger.info("Step 3: Score %.3f is confident — skipping reflection", result["score"])

    return PredictResponse(
        label=result["label"],
        score=result["score"],
        reasoning=result.get("reasoning", ""),
        reflected=reflected,
        source=source,
    )

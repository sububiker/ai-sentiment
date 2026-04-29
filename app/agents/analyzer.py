import json
import logging

import anthropic

from config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise sentiment analysis assistant. "
    "Analyze the sentiment of the provided text and respond with a JSON object containing:\n"
    '- "label": either "positive" or "negative"\n'
    '- "score": float 0.0-1.0 (above 0.5 = positive, below 0.5 = negative)\n'
    '- "reasoning": one sentence citing the key signal driving your classification\n'
    "Respond with only the JSON object and no other text."
)


def analyze_sentiment(client: anthropic.Anthropic, text: str) -> dict:
    logger.info("Running sentiment analysis (chars=%d)", len(text))
    content = text[: settings.max_content_chars]

    response = client.messages.create(
        model=settings.model,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": content}],
    )

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("Claude returned no text block in analyzer response")

    result = json.loads(text_block.text)
    logger.info("Analyzer: label=%s score=%.3f", result["label"], result["score"])
    return result

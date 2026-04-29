import json
import logging

import anthropic

from config import settings

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = (
    "You are a strict sentiment critic. A first-pass analysis returned an uncertain result "
    "with a score too close to the decision boundary (0.40–0.60). "
    "Re-examine the text more carefully. Look for stronger signals — specific word choices, "
    "emotional tone, comparisons, qualifiers, or context clues that clearly indicate "
    "positive or negative sentiment. Be decisive.\n\n"
    "Respond with a JSON object:\n"
    '- "label": either "positive" or "negative"\n'
    '- "score": float 0.0-1.0, must be outside the 0.40-0.60 uncertainty band\n'
    '- "reasoning": one sentence citing the specific signal that made you more confident\n'
    "Respond with only the JSON object and no other text."
)


def reflect_on_sentiment(
    client: anthropic.Anthropic,
    original_text: str,
    initial_result: dict,
) -> dict:
    logger.info(
        "Triggering reflection — initial score %.3f is within uncertainty band [%.2f, %.2f]",
        initial_result["score"],
        settings.reflection_low,
        settings.reflection_high,
    )

    content = (
        f"Original text:\n{original_text[:settings.max_content_chars]}\n\n"
        f"Initial analysis — "
        f"label: {initial_result['label']}, "
        f"score: {initial_result['score']:.3f}, "
        f"reasoning: {initial_result.get('reasoning', 'N/A')}"
    )

    response = client.messages.create(
        model=settings.model,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _CRITIC_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": content}],
    )

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise ValueError("Claude returned no text block in critic response")

    result = json.loads(text_block.text)
    logger.info(
        "Critic: label=%s score=%.3f (was %.3f)",
        result["label"],
        result["score"],
        initial_result["score"],
    )
    return result

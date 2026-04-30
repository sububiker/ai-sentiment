import json
import logging

import anthropic

from config import settings
from models.schemas import PredictResponse
from tools.fetcher import fetch_content
from tools.search import search_web
from tools.history import analyze_history
from agents.tools_def import SYSTEM_PROMPT, TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

TOOL_MAP = {
    "fetch_url":      lambda url: fetch_content(url),
    "search_web":     lambda query: search_web(query),
    "analyze_history": lambda topic: analyze_history(topic),
}


def run_agent(client: anthropic.Anthropic, user_input: str) -> PredictResponse:
    """
    Fully agentic loop — Claude drives the flow.
    It decides which tools to call, how many times, and when it has enough
    information to return a final answer.
    """
    messages = [{"role": "user", "content": user_input}]
    tools_called: list[str] = []

    for iteration in range(MAX_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        response = client.messages.create(
            model=settings.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        )

        logger.info("stop_reason=%s", response.stop_reason)

        # ── Claude is done ────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            text_block = next((b for b in response.content if b.type == "text"), None)
            if text_block is None:
                raise ValueError("Agent returned no final text block")

            result = json.loads(text_block.text)
            logger.info(
                "Agent finished: label=%s score=%.3f tools_called=%s",
                result["label"], result["score"], tools_called,
            )
            return PredictResponse(
                label=result["label"],
                score=result["score"],
                reasoning=result.get("reasoning", ""),
                source=result.get("source", "text"),
                tools_called=tools_called,
            )

        # ── Claude wants to use tools ─────────────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                logger.info("Claude calling tool: %s(%s)", tool_name, tool_input)
                tools_called.append(tool_name)

                try:
                    result = TOOL_MAP[tool_name](**tool_input)
                except Exception as exc:
                    result = f"Tool error: {exc}"
                    logger.warning("Tool %s failed: %s", tool_name, exc)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)[:4000],  # cap to avoid token overflow
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

    raise ValueError(f"Agent exceeded maximum iterations ({MAX_ITERATIONS})")

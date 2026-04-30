"""
Fully agentic MCP-enabled runner.

Claude drives the tool-use loop.
Tools are executed via MCP servers (separate processes) not direct function calls.

Flow:
  1. Start 3 MCP server processes (fetch, search, history)
  2. Connect MCP client to each server
  3. Discover tools dynamically from each server
  4. Claude decides which tools to call — loop until end_turn
  5. Tool calls are routed to the correct MCP server via protocol
  6. Results returned to Claude; loop continues
"""
import asyncio
import json
import logging
import os
import re

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings
from models.schemas import PredictResponse

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


def _parse_response(text: str) -> dict:
    """Extract JSON from Claude's response, handling prose wrappers or empty strings."""
    text = text.strip()
    if not text:
        raise ValueError("Claude returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Claude sometimes wraps JSON in markdown fences or prose — find the first {...}
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse Claude response as JSON: {text[:300]}")


# Absolute paths so subprocesses can be found inside the Docker container
_APP_DIR = os.path.dirname(os.path.dirname(__file__))

MCP_SERVERS = [
    StdioServerParameters(
        command="python",
        args=[os.path.join(_APP_DIR, "mcp_servers", "fetch_server.py")],
    ),
    StdioServerParameters(
        command="python",
        args=[os.path.join(_APP_DIR, "mcp_servers", "search_server.py")],
    ),
    StdioServerParameters(
        command="python",
        args=[os.path.join(_APP_DIR, "mcp_servers", "history_server.py")],
    ),
]

SYSTEM_PROMPT = (
    "You are an expert sentiment analysis agent. "
    "Your ONLY job is to return a JSON sentiment result — nothing else.\n\n"
    "CRITICAL RULES:\n"
    "- NEVER ask the user questions or explain what you need\n"
    "- NEVER respond conversationally\n"
    "- For ANY input — even a single word like 'test' — return the JSON immediately\n"
    "- If the input is ambiguous or short, make a best-effort assessment and return JSON\n\n"
    "Tool guidelines (optional — only call if genuinely needed):\n"
    "- Input contains a URL → call fetch_url first\n"
    "- Input is a topic needing context → call search_web\n"
    "- User asks about past/trends → call analyze_history\n"
    "- Clear plain text → skip tools, return JSON directly\n\n"
    "Your entire response must be exactly this JSON and nothing else:\n"
    "{\n"
    '  "label": "positive" or "negative",\n'
    '  "score": float 0.0-1.0,\n'
    '  "reasoning": "one sentence citing the key signal",\n'
    '  "source": "text" | "url" | "web_search" | "combined"\n'
    "}"
)


async def _run_agent_async(
    client: anthropic.Anthropic,
    user_input: str,
) -> tuple[dict, list[str]]:
    """
    Connect to all MCP servers, discover their tools, then run the agent loop.
    Claude calls tools via MCP protocol — not direct Python function calls.
    """
    anthropic_tools: list[dict] = []
    tool_to_session: dict[str, ClientSession] = {}
    sessions: list = []
    tools_called: list[str] = []

    # ── Step 1: Connect to each MCP server and discover tools ────────────────
    for server_params in MCP_SERVERS:
        logger.info("Connecting to MCP server: %s", server_params.args)
        transport = stdio_client(server_params)
        read, write = await transport.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        mcp_tools = await session.list_tools()          # ← discover tools
        logger.info(
            "Server %s exposes tools: %s",
            server_params.args,
            [t.name for t in mcp_tools.tools],
        )

        for tool in mcp_tools.tools:
            # Convert MCP tool format → Anthropic tool format
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )
            tool_to_session[tool.name] = session       # ← remember which server owns it

        sessions.append((transport, session))

    # ── Step 2: Agent loop — Claude drives from here ─────────────────────────
    messages = [{"role": "user", "content": user_input}]

    try:
        for iteration in range(MAX_ITERATIONS):
            logger.info("Agent iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = client.messages.create(
                model=settings.model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                tools=anthropic_tools,              # ← tools discovered from MCP servers
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

            # Claude is done
            if response.stop_reason == "end_turn":
                text_block = next(
                    (b for b in response.content if b.type == "text"), None
                )
                if text_block is None:
                    raise ValueError("Agent returned no final text block")
                result = _parse_response(text_block.text)
                logger.info(
                    "Agent done: label=%s score=%.3f tools=%s",
                    result["label"], result["score"], tools_called,
                )
                return result, tools_called

            # Claude wants to call tools
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input
                    logger.info(
                        "Claude → MCP call: %s(%s)", tool_name, tool_input
                    )
                    tools_called.append(tool_name)

                    # ── Route tool call to correct MCP server via protocol ──
                    session = tool_to_session[tool_name]
                    try:
                        mcp_result = await session.call_tool(tool_name, tool_input)
                        content = mcp_result.content[0].text if mcp_result.content else ""
                    except Exception as exc:
                        content = f"Tool error: {exc}"
                        logger.warning("MCP tool %s failed: %s", tool_name, exc)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content[:4000],
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                continue

            raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

        raise ValueError(f"Agent exceeded {MAX_ITERATIONS} iterations")

    finally:
        # ── Step 3: Cleanly shut down all MCP server connections ─────────────
        for transport, session in reversed(sessions):
            try:
                await session.__aexit__(None, None, None)
                await transport.__aexit__(None, None, None)
            except Exception:
                pass


def run_agent(client: anthropic.Anthropic, user_input: str) -> PredictResponse:
    """Sync wrapper — FastAPI calls this; asyncio runs the MCP client internally."""
    result, tools_called = asyncio.run(_run_agent_async(client, user_input))
    return PredictResponse(
        label=result["label"],
        score=result["score"],
        reasoning=result.get("reasoning", ""),
        source=result.get("source", "text"),
        tools_called=tools_called,
    )

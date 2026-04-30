"""
MCP SERVER — analyze_history tool

Runs as a separate process.
Exposes the analyze_history tool to any MCP client via stdio transport.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from tools.history import analyze_history, init_db

init_db()
server = Server("history-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="analyze_history",
            description=(
                "Look up past sentiment predictions stored in the database. "
                "Use this when the user asks about trends or historical sentiment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Keyword or topic to look up in history",
                    }
                },
                "required": ["topic"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "analyze_history":
        result = analyze_history(arguments["topic"])
        return [types.TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

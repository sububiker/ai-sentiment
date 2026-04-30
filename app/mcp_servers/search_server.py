"""
MCP SERVER — search_web tool

Runs as a separate process.
Exposes the search_web tool to any MCP client via stdio transport.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from tools.search import search_web

server = Server("search-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_web",
            description=(
                "Search the web for reviews, news, or opinions about a topic. "
                "Use this to gather external context when input alone is insufficient."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_web":
        result = search_web(arguments["query"])
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

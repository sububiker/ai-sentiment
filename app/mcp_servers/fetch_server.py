"""
MCP SERVER — fetch_url tool

Runs as a separate process.
Exposes the fetch_url tool to any MCP client via stdio transport.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from tools.fetcher import fetch_content

server = Server("fetch-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_url",
            description=(
                "Fetch and extract readable text from a URL. "
                "Use this whenever the user provides a link to analyze."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "fetch_url":
        result = fetch_content(arguments["url"])
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

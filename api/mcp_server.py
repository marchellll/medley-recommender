"""MCP server implementation."""

from typing import Any, Optional

from api.models import AddSongRequest, SearchRequest
from api.routes import add_song, search_songs
from src.database.db import AsyncSessionLocal


class MCPServer:
    """MCP server for medley recommender."""

    def __init__(self):
        """Initialize MCP server."""
        self.tools = [
            {
                "name": "search_songs",
                "description": "Search for worship/praise songs by lyrics similarity and BPM filter",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (lyrics or description)",
                        },
                        "bpm_min": {
                            "type": "number",
                            "description": "Minimum BPM filter (optional)",
                        },
                        "bpm_max": {
                            "type": "number",
                            "description": "Maximum BPM filter (optional)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "add_song",
                "description": "Add a new worship/praise song to the system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Song title"},
                        "youtube_url": {
                            "type": "string",
                            "description": "YouTube URL for the song",
                        },
                        "lyrics": {"type": "string", "description": "Song lyrics"},
                    },
                    "required": ["title", "youtube_url", "lyrics"],
                },
            },
        ]

    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Handle MCP tool call.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        async with AsyncSessionLocal() as session:
            if tool_name == "search_songs":
                request = SearchRequest(**arguments)
                response = await search_songs(request, session)
                return {
                    "results": [
                        {
                            "song_id": r.song_id,
                            "title": r.title,
                            "bpm": r.bpm,
                            "similarity_score": r.similarity_score,
                            "youtube_url": r.youtube_url,
                        }
                        for r in response.results
                    ],
                    "total": response.total,
                }
            elif tool_name == "add_song":
                request = AddSongRequest(**arguments)
                response = await add_song(request, session)
                return {
                    "success": response.success,
                    "song_id": response.song_id,
                    "message": response.message,
                }
            else:
                raise ValueError(f"Unknown tool: {tool_name}")


# Global MCP server instance
mcp_server = MCPServer()







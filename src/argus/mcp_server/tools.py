"""Tool callables, re-exported for testing without spinning up the MCP transport.

FastMCP's ``@mcp.tool()`` decorator registers the function with the server *and*
returns it unchanged, so these are the exact same callables the MCP server exposes —
tests import from here rather than server.py to make that reuse explicit.
"""

from __future__ import annotations

from argus.mcp_server.server import (
    get_hazard_exposure,
    list_high_risk_districts,
    search_regulatory_corpus,
)

__all__ = ["get_hazard_exposure", "list_high_risk_districts", "search_regulatory_corpus"]

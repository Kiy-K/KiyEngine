import os
import asyncio
from fastmcp import Client as MCPClient

# MCP Server URL from environment
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://med-mcp.fastmcp.app/mcp")


async def call_mcp_tool_async(tool_name: str, arguments: dict) -> str:
    """
    Call a tool on the FastMCP server asynchronously.
    Returns the result as a string.
    """
    try:
        async with MCPClient(MCP_SERVER_URL) as client:
            result = await client.call_tool(tool_name, arguments)
            # Result can be a list of content items or a single value
            if isinstance(result, list):
                # Typically result is a list of TextContent or similar
                return "\n".join(str(item) for item in result)
            return str(result)
    except Exception as e:
        return f"MCP call failed: {str(e)}"


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Synchronous wrapper for MCP tool calls.
    Use this in Streamlit which runs in sync context.
    """
    return asyncio.run(call_mcp_tool_async(tool_name, arguments))


async def list_mcp_tools_async() -> list:
    """
    List available tools from the MCP server.
    """
    try:
        async with MCPClient(MCP_SERVER_URL) as client:
            tools = await client.list_tools()
            return tools
    except Exception as e:
        return [f"Error listing tools: {str(e)}"]


def list_mcp_tools() -> list:
    """
    Synchronous wrapper for listing MCP tools.
    """
    return asyncio.run(list_mcp_tools_async())


# Legacy PubMed function (kept for fallback)
def search_pubmed(query: str) -> str:
    """
    Search medical literature via MCP server.
    This is a convenience wrapper that calls the MCP tool.
    """
    # Try to use MCP first, with tool name guessed as 'search' or 'pubmed_search'
    # The actual tool name depends on the MCP server implementation
    result = call_mcp_tool("search", {"query": query})
    
    if "MCP call failed" in result:
        # Fallback: try different tool names
        result = call_mcp_tool("pubmed_search", {"query": query})
    
    return result

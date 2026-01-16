import os
import asyncio
from fastmcp import Client

# Load URL from env, default to the one that works with FastMCP Client
# Note: FastMCP Client usually expects the base URL (e.g., .../mcp or .../sse)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://med-mcp.fastmcp.app/sse")

async def _call_tool_async(tool_name: str, args: dict):
    """
    Internal async function to call the MCP tool using FastMCP Client.
    """
    try:
        # Initialize client with the URL
        client = Client(MCP_SERVER_URL)
        
        async with client:
            # 1. Ping to ensure connection (Optional but good for debug)
            # await client.ping()
            
            # 2. Call the tool
            result = await client.call_tool(tool_name, args)
            
            # 3. Handle result (FastMCP returns list of content or direct value)
            if isinstance(result, list):
                # Extract text from TextContent objects if present
                texts = [item.text for item in result if hasattr(item, 'text')]
                if texts:
                    return "\n".join(texts)
                # Fallback for list of strings
                return "\n".join([str(item) for item in result])
            
            return str(result)

    except Exception as e:
        return f"❌ MCP Connection Error: {str(e)}"

def call_mcp_tool(tool_name: str, args: dict):
    """
    Synchronous wrapper for Streamlit to call async MCP tools.
    """
    return asyncio.run(_call_tool_async(tool_name, args))

# --- EXPORTED TOOLS ---

def search_pubmed(query: str):
    """
    Search PubMed via the MCP Server.
    Maps to the 'search_pubmed' tool on the remote server.
    """
    return call_mcp_tool("search_pubmed", {"query": query})

def get_available_tools():
    """
    Helper to list tools for debugging
    """
    async def _list():
        client = Client(MCP_SERVER_URL)
        async with client:
            tools = await client.list_tools()
            return [t.name for t in tools]
    try:
        return asyncio.run(_list())
    except:
        return ["Error listing tools"]

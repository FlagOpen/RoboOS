from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("robots")

# task
# navigate to customerTable and bring the basket to kitchenTable.
@mcp.tool()
async def navigate(target: str) -> str:
    """Navigate to target
    Args:
        target: String, Represents the navigation destination.
    """
    return f"Navigate to {target} success"

@mcp.tool()
async def grasp_object(object: str) -> str:
    """Grasp the object for bring
    Args:
        object: String, Represents which to grasp.
    """
    return f"Grasp {object} success"

@mcp.tool()
async def place_where(affordance: str) -> str:
    """Place the object to affordance, the object has been grasped
    Args:
        affordance: String, Represents where the object to place.
    """
    return f"Place success"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")

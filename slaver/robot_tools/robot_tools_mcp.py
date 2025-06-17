from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("robots")

@mcp.tool()
async def navigate_to_target(target: str) -> str:
    """Navigate to target
    Args:
        target: String, Represents the navigation destination.
    """
    ret = f"Navigate to {target} success"
    print(ret)
    return ret

@mcp.tool()
async def grasp_object(object: str) -> str:
    """Pick up the object
    Args:
        object: String, Represents which to grasp.
    """
    ret = f"Grasp {object} success"
    print(ret)
    return ret

@mcp.tool()
async def place_to_affordance(affordance: str, object: str=None) -> str:
    """Place the grasped object in affordance
    Args:
        affordance: String, Represents where the object to place.
        object: String, Represents the object has been grasped.
    """
    ret = f"Place {object} on {affordance} success."
    print(ret)
    return ret

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")

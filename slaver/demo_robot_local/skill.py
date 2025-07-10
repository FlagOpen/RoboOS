from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("robots")


@mcp.tool()
async def navigate_to_target(target: str) -> tuple[str, dict]:
    """Navigate to target, do not call when Navigation to target has been successfully performed.
    Args:
        target: String, Represents the navigation destination.
    """
    result = f"Navigation to {target} has been successfully performed."
    print(result)
    return result, {"position": f"{target}"}


@mcp.tool()
async def grasp_object(object: str) -> tuple[str, dict]:
    """Pick up the object, do not call when object has been successfully grasped.
    Args:
        object: String, Represents which to grasp.
    """
    result = f"{object} has been successfully grasped."
    print(result)
    return result, {"grasped": f"{object}"}


@mcp.tool()
async def place_to_affordance(affordance: str, object: str = None) -> tuple[str, dict]:
    """Place the grasped object in affordance, do not call when object has been successfully placed on affordance."
    Args:
        affordance: String, Represents where the object to place.
        object: String, Represents the object has been grasped.
    """
    result = f"{object} has been successfully placed on {affordance}."
    print(result)
    return result, {"grasped": f"None"}


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")

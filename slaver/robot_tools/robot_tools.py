def navigate_to_target(target: str) -> str:
    """Navigate to target, do not call when Navigation to target has been successfully performed.
    Args:
        target: String, Represents the navigation destination.
    """
    ret = f"Navigation to {target} has been successfully performed."
    print(ret)
    return ret

def grasp_object(object: str) -> str:
    """Pick up the object, do not call when object has been successfully grasped.
    Args:
        object: String, Represents which to grasp.
    """
    ret = f"{object} has been successfully grasped."
    print(ret)
    return ret

def place_to_affordance(affordance: str, object: str=None) -> str:
    """Place the grasped object in affordance, do not call when object has been successfully placed on affordance."
    Args:
        affordance: String, Represents where the object to place.
        object: String, Represents the object has been grasped.
    """
    ret = f"{object} has been successfully placed on {affordance}."
    print(ret)
    return ret

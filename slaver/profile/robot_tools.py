def navigate_to_target(target: str) -> str:
    """Navigate to target
    Args:
        target: String, Represents the navigation destination.
    """
    ret = f"Navigate to {target} success"
    print(ret)
    return ret

def grasp_object(object: str) -> str:
    """Pick up the object
    Args:
        object: String, Represents which to grasp.
    """
    ret = f"Grasp {object} success"
    print(ret)
    return ret

def place_to_affordance(affordance: str) -> str:
    """Place the grasped object in affordance
    Args:
        affordance: String, Represents where the object to place.
    """
    ret = f"Place on {affordance} success."
    print(ret)
    return ret
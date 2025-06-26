import functools

import yaml
from tools.utils import communicator, config

robot_profile = yaml.safe_load(open(config["profile"]["PATH"], "r", encoding="utf-8"))


def record_state(action_name: str):
    """Decorator: Automatically records status when calling tools"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            key = f"ROBOT_INFO_{robot_profile['robot_name']}"
            if action_name == "navigated":
                communicator.update_json_field_py(
                    key=key,
                    field_path="current_position",
                    new_value=list(kwargs.values())[0],
                )
            elif action_name == "grasped":
                communicator.update_json_field_py(
                    key=key,
                    field_path="robot_holding",
                    new_value=list(kwargs.values())[0],
                )
            elif action_name == "placed":
                communicator.update_json_field_py(
                    key=key, field_path="robot_holding", new_value=[]
                )
            return result

        return wrapper

    return decorator

from typing import Dict, List

import yaml
from prompt.prompts import (
    MASTER_PLANNING_PLANNING,
    ROBOT_POSITION_INFO_TEMPLATE,
    ROBOT_TOOLS_INFO_TEMPLATE,
    SCENE_OBJECTS_INFO_TEMPLATE,
)


# generate the robot position information based on the robot profiles
def get_robot_position_info(robot_info: List) -> str:
    """
    Generate the robot position information based on the robot profiles.

    Args:
        robot_info (List): List of robot profiles.

    Returns:
        str: Formatted string with robot position information.
    """
    robot_position_info = []

    for robot in robot_info:
        robot_name = robot["robot_name"]
        initial_pos = robot["current_position"]
        target_pos = (
            robot["navigate_position"]
            if isinstance(robot["navigate_position"], List)
            else [robot["navigate_position"]]
        )

        robot_position_info.append(
            ROBOT_POSITION_INFO_TEMPLATE.format(
                robot_name=robot_name, initial_pos=initial_pos, target_pos=target_pos
            )
        )

    return "\n".join(robot_position_info)


# generate the robot tools information based on the robot profiles
def get_robot_tools_info(robot_info: List) -> str:
    """
    Generate the robot tools information based on the robot profiles.

    Args:
        robot_info (List): List of robot profiles.

    Returns:
        str: Formatted string with robot tools information.
    """
    robot_tools_info = []

    for robot in robot_info:
        robot_name = robot["robot_name"]
        tools_list = (
            robot["robot_tool"]
            if isinstance(robot["robot_tool"], List)
            else [robot["robot_tool"]]
        )

        robot_tools_info.append(
            ROBOT_TOOLS_INFO_TEMPLATE.format(
                robot_name=robot_name, tool_list=tools_list
            )
        )

    return "\n".join(robot_tools_info)


# generate the scene objects information based on the scene profiles
def get_scene_objects_info(scene_info: List) -> str:
    """
    Generate the scene objects information based on the scene profiles.

    Args:
        scene_info (List): List of scene profiles.

    Returns:
        str: Formatted string with scene objects information.
    """
    scene_objects_info = []

    for scene in scene_info:
        recep_name = scene["recep_name"]
        object_list = (
            scene["recep_object"]
            if isinstance(scene["recep_object"], List)
            else [scene["recep_object"]]
        )

        scene_objects_info.append(
            SCENE_OBJECTS_INFO_TEMPLATE.format(
                recep_name=recep_name, object_list=object_list
            )
        )

    return "\n".join(scene_objects_info)


# gather all the information and generate the master planning prompt
def get_master_planning_prompt(robot_profile: List, scene_profile: List, task) -> str:
    """
    Generate the master planning prompt for task decomposition.

    Args:
        robot_profile (Dict): Dict of robot profiles.
        scene_profile (Dict): Dict of scene profiles.
        task (str): The task to be completed.

    Returns:
        str: Formatted master planning prompt.
    """

    robot_position_info = get_robot_position_info(robot_profile)
    robot_tools_info = get_robot_tools_info(robot_profile)
    scene_objects_info = get_scene_objects_info(scene_profile)

    robot_name_list = [robot["robot_name"] for robot in robot_profile]
    recep_name_list = [scene["recep_name"] for scene in scene_profile]

    prompt = MASTER_PLANNING_PLANNING.format(
        robot_name_list=robot_name_list,
        recep_name_list=recep_name_list,
        robot_position_info=robot_position_info,
        robot_tools_info=robot_tools_info,
        scene_object_info=scene_objects_info,
        task=task,
    )

    return prompt


# read from yaml file
def read_yaml_file(robot_profile_path: str = None, scene_profile_path=None) -> Dict:
    """
    Read the YAML file and return the data as a Dictionary.

    Args:
        robot_profile_path (str): Path to the robot profile YAML file.
        scene_profile_path (str): Path to the scene profile YAML file.

    Returns:
        Dict: Data from the YAML files.
    """
    if robot_profile_path is not None:
        with open(robot_profile_path, "r") as f:
            robot_profile = yaml.safe_load(f)
    else:
        robot_profile = None

    if scene_profile_path is not None:
        with open(scene_profile_path, "r") as f:
            scene_profile = yaml.safe_load(f)
    else:
        scene_profile = None

    return {
        "robot_profile": robot_profile["robot"] if robot_profile else [],
        "scene_profile": scene_profile["scene"] if scene_profile else [],
    }

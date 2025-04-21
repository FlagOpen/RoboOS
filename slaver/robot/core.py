import json
import random
from robot.error_definitions import ERROR_DEFINITIONS
from robot.error_handler import ErrorHandler
from utils import convert_yaml_to_json, communicator, config
from robot.base import IMechanical
from tools.monitoring import (
    AgentLogger,
    LogLevel,
)


class Robot:
    def __init__(self, robot: IMechanical):
        self.robot = robot
        self.error_handler = ErrorHandler(self.robot, ERROR_DEFINITIONS)
        self.robot_profile = convert_yaml_to_json(config["profile"]["PATH"])
        self.robot_name = self.robot_profile["robot_name"]
        self.communicator = communicator
        self.logger = AgentLogger(level=LogLevel.INFO, log_file="./.log/agent.log")

    def perform_grasp(self, target):
        if config["tool"]["DISABLE_ARM"]:
            if random.random() < config["tool"]["ERROR_PROBABILITY"]:
                error_pool = ERROR_DEFINITIONS.get("graspObjectErrors", {})
                error_code, error_info = random.choice(list(error_pool.items()))
                result = {
                    "status": "error",
                    "error_code": error_code,
                    "error_name": error_info["name"],
                    "description": error_info["description"],
                }
            else:
                result = {"status": "success", "message": "success"}

        else:
            result = self.robot.grasp(target)
        status = result["status"]
        if status == "error":
            error_code = result["error_code"]
            self.error_handler.handle_error(error_code)

        robot_info = self.communicator.retrieve(f"ROBOT_INFO_{self.robot_name}")
        robot_info["grasp_object"] = target
        current_position = robot_info["current_position"]
        scenc_profile = self.communicator.retrieve(f"SCENE_INFO_{current_position}")
        if scenc_profile:
            scenc_profile["recep_object"] = [
                recept_object
                for recept_object in scenc_profile.get("recep_object", [])
                if recept_object != target
            ]
            self.communicator.register(
                f"SCENE_INFO_{current_position}", json.dumps(scenc_profile)
            )
            self.communicator.register(
                f"ROBOT_INFO_{self.robot_name}",
                json.dumps(robot_info),
                expire_second=60,
            )

        return json.dumps(result, ensure_ascii=False)

    def perform_place(self, target):
        if config["tool"]["DISABLE_ARM"]:
            if random.random() < config["tool"]["ERROR_PROBABILITY"]:
                error_pool = ERROR_DEFINITIONS.get("graspObjectErrors", {})
                error_code, error_info = random.choice(list(error_pool.items()))
                result = {
                    "status": "error",
                    "error_code": error_code,
                    "error_name": error_info["name"],
                    "description": error_info["description"],
                }
            else:
                result = {"status": "success", "message": "success"}

        else:
            try:
                self.robot.place(target)
            except Exception as e:
                self.logger.log(f"[Robot] : {e}", LogLevel.ERROR)
                self.error_handler.handle_error("E101")
        robot_info = self.communicator.retrieve(f"ROBOT_INFO_{self.robot_name}")
        current_position = robot_info["current_position"]
        grasp_object = robot_info["grasp_object"]
        scenc_profile = self.communicator.retrieve(f"SCENE_INFO_{current_position}")
        scenc_profile["recep_object"].append(grasp_object)
        self.communicator.register(
            f"SCENE_INFO_{current_position}", json.dumps(scenc_profile)
        )
        return result

    def perform_navigate(self, target):
        if config["tool"]["DISABLE_CHASSIS"]:
            if random.random() < config["tool"]["ERROR_PROBABILITY"]:
                error_pool = ERROR_DEFINITIONS.get("navigationErrors", {})
                error_code, error_info = random.choice(list(error_pool.items()))
                result = {
                    "status": "error",
                    "error_code": error_code,
                    "error_name": error_info["name"],
                    "description": error_info["description"],
                }
            else:
                result = {"status": "success", "message": "success"}
        else:
            try:
                result = self.robot.navigate(target)
            except Exception as e:
                self.logger.log(f"[Robot] : {e}", LogLevel.ERROR)
                self.error_handler.handle_error("E101")
        self.communicator.update_json_field_py(
            f"ROBOT_INFO_{self.robot_name}", "current_position", target
        )

        return result

    def perform_detect(self, target):
        result = {"status": "success", "message": "success"}
        return result

    def Attempt_Other_Path(self, target) -> bool:
        # TODO: Attempt to find an alternative path to the target.
        pass

    def Ask_Other_Robot_for_Help():
        # TODO: Ask another robot for assistance in reaching the target.
        pass

    def Request_Manual_Assistance():
        # TODO: Request manual assistance from a human operator.
        pass

    def Ask_RoboBrain_for_Replanning(self, robot, task):
        # TODO: Ask RoboBrain for replanning assistance.
        pass

    def Use_Vision_Navigation(self, robot, target):
        # TODO: Use vision-based navigation to reach the target.
        pass

    def Move_to_Near_Position(self, robot, offset=0.1):
        # TODO: Move to a position near the target with a specified offset.
        pass

    def Recognize_Similar_Object(self, robot, target):
        # TODO: Recognize a similar object to the target.
        pass

    def Move_to_Other_Position(self, robot, new_target):
        # TODO: Move to a different position.
        pass

    def Adjust_EEF_Orientation(self, robot):
        # TODO: Adjust the end-effector's orientation.
        pass

    def Move_to_Candidate_Position(self, robot, candidates):
        # TODO: Move to a candidate position.
        pass


class RobotFactory:
    @staticmethod
    def create_robot():
        return Robot(IMechanical)

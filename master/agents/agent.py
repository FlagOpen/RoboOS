import os
import logging
import json
import yaml
import uuid
import threading
from collections import defaultdict
from typing import Dict, Tuple
from agents.planner import GlobalTaskPlanner
from flag_scale.flagscale.agent.communication import Communicator


class GlobalAgent:
    def __init__(self, config_path="config.yaml"):
        """Initialize GlobalAgent"""
        self._init_config(config_path)
        self._init_logger(self.config["logger"])
        self.planner = GlobalTaskPlanner(self.config)
        self.communicator = Communicator(
            host=self.config["communicator"]["HOST"],
            port=self.config["communicator"]["PORT"],
            db=self.config["communicator"]["DB"],
            clear=self.config["communicator"]["CLEAR"],
            password=self.config["communicator"]["PASSWORD"],
        )
        # TODO This is only for mocking when ROBOT_PROFILE_ENABLE set 'true', it should be removed in the future
        for robot_info in self.planner.global_memory["robot_profile"]:
            robot_name = robot_info["robot_name"]
            self.communicator.register(
                f"ROBOT_REGISTER_{robot_name}", json.dumps(robot_info)
            )
            self.communicator.register(
                f"ROBOT_INFO_{robot_name}", json.dumps(robot_info)
            )

        # TODO This is only for mocking when SCENE_PROFILE_ENABLE set 'true', it should be removed in the future
        for scene_info in self.planner.global_memory["scene_profile"]:
            recep_name = scene_info["recep_name"]
            self.communicator.register(
                f"SCENE_INFO_{recep_name}", json.dumps(scene_info)
            )

        self.logger.info(f"Configuration loaded from {config_path} ...")
        self.logger.info(f"Master Configuration:\n{self.config}")
        self._start_listener()

    def _init_logger(self, logger_config):
        """Initialize an independent logger for GlobalAgent"""
        self.logger = logging.getLogger(logger_config["MASTER_LOGGER_NAME"])
        logger_file = logger_config["MASTER_LOGGER_FILE"]
        os.makedirs(os.path.dirname(logger_file), exist_ok=True)
        file_handler = logging.FileHandler(logger_file)

        # Set the logging level
        if logger_config["MASTER_LOGGER_LEVEL"] == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        elif logger_config["MASTER_LOGGER_LEVEL"] == "INFO":
            self.logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
        elif logger_config["MASTER_LOGGER_LEVEL"] == "WARNING":
            self.logger.setLevel(logging.WARNING)
            file_handler.setLevel(logging.WARNING)
        elif logger_config["MASTER_LOGGER_LEVEL"] == "ERROR":
            self.logger.setLevel(logging.ERROR)
            file_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _init_config(self, config_path="config.yaml"):
        """Initialize configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _handle_register(self, data: Dict) -> None:
        """Listen for robot registrations."""
        self.logger.info("Received registration data:", data)
        robot_name = data.get("robot_name")
        self.logger.info(f"robot_registration: {robot_name} \n {json.dumps(data)}")
        self.communicator.register(f"ROBOT_REGISTER_{robot_name}", json.dumps(data))

        # Register functions for processing robot execution results in the brain
        channel_r2b = f"{robot_name}_to_roboos"
        threading.Thread(
            target=lambda: self.communicator.listen(channel_r2b, self._handle_result),
            daemon=True,
            name=channel_r2b,
        ).start()

        self.logger.info(
            f"RoboOS has listened to [{robot_name}] by channel [{channel_r2b}]"
        )

    def _handle_result(self, data: Dict):
        """Handle results from agents."""
        robot_name = data.get("robot_name")
        subtask_handle = data.get("subtask_handle")
        subtask_result = data.get("subtask_result")

        # TODO: Task result should be refered to the next step determination.
        if robot_name and subtask_handle and subtask_result:
            self.logger.info(
                f"================ Received result from {robot_name} ================"
            )
            self.logger.info(f"Subtask: {subtask_handle}\nResult: {subtask_result}")
            self.logger.info(
                "===================================================================="
            )
            self.communicator.register(f"ROBOT_SUBTASK_{robot_name}", json.dumps(data))
            self.communicator.update_json_field_py(
                f"ROBOT_INFO_{robot_name}", "robot_state", "idle"
            )

        else:
            self.logger.warning("[WARNING] Received incomplete result data")
            self.logger.info(
                f"================ Received result from {robot_name} ================"
            )
            self.logger.info(f"Subtask: {subtask_handle}\nResult: {subtask_result}")
            self.logger.info(
                "===================================================================="
            )

    def _extract_json(self, input_string):
        """Extract JSON from a string."""
        start_marker = "```json"
        end_marker = "```"
        try:
            start_idx = input_string.find(start_marker)
            end_idx = input_string.find(end_marker, start_idx + len(start_marker))
            if start_idx == -1 or end_idx == -1:
                self.logger.warning("[WARNING] JSON markers not found in the string.")
                return None
            json_str = input_string[start_idx + len(start_marker) : end_idx].strip()
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"[WARNING] JSON cannot be extracted from the string.\n{e}"
            )
            return None

    def _group_tasks_by_order(self, tasks):
        """Group tasks by topological order."""
        grouped = defaultdict(list)
        for task in tasks:
            grouped[int(task.get("subtask_order", 0))].append(task)
        return dict(sorted(grouped.items()))

    def _start_listener(self):
        """Start listen in a background thread."""
        threading.Thread(
            target=lambda: self.communicator.listen(
                "robot_registration", self._handle_register
            ),
            daemon=True,
        ).start()
        self.logger.info("Started listening for robot registrations...")

    def publish_global_task(self, task: str) -> Tuple:
        """Publish a global task to all Agents"""
        self.logger.info(f"Publishing global task: {task}")
        current_robot_info = self.communicator.gat_all_values("ROBOT_INFO_*")
        current_scene_info = self.communicator.gat_all_values("SCENE_INFO_*")
        current_memory = {
            "robot_profile": current_robot_info,
            "scene_profile": current_scene_info,
        }
        self.logger.debug(f"Current Agents:\n{current_robot_info}")
        self.logger.debug(f"Current Scenes:\n{current_robot_info}")
        response = self.planner.forward(task, current_memory)
        reasoning_and_subtasks = self._extract_json(response)

        # Retry if JSON extraction fails
        if reasoning_and_subtasks is None:
            for attempt in range(self.config["model"]["MODEL_RETRY_PLANNING"]):
                self.logger.warning(
                    f"Attempt {attempt + 1} to extract JSON failed. Retrying..."
                )
                response = self.planner.forward(task, current_memory)
                reasoning_and_subtasks = self._extract_json(response)
                if reasoning_and_subtasks is not None:
                    break
            self.logger.warning(
                f"[WARNING] JSON extraction failed after {self.config['model']['MODEL_RETRY_PLANNING']} attempts."
            )
            self.logger.error(
                f"[ERROR] Task ({task}) failed to be decomposed into subtasks, it will be ignored."
            )
            return False

        self.logger.info(f"Received reasoning and subtasks:\n{reasoning_and_subtasks}")
        subtask_list = reasoning_and_subtasks.get("subtask_list", [])
        grouped_tasks = self._group_tasks_by_order(subtask_list)
        task_id = str(uuid.uuid4()).replace("-", "")
        order_flag = "false" if len(grouped_tasks.keys()) == 1 else "true"
        for task_count, (order, task_group) in enumerate(grouped_tasks.items()):
            self.logger.info(f"Sending task group {order}:\n{task_group}")
            for task in task_group:
                robot_name = task.get("robot_name")
                subtask_data = {
                    "task_id": task_id,
                    "task": task["subtask"],
                    "order": order_flag,
                }
                self.communicator.send(
                    f"roboos_to_{robot_name}", json.dumps(subtask_data)
                )
                self.communicator.update_json_field_py(
                    f"ROBOT_INFO_{robot_name}", "robot_state", "busy"
                )
            # wait for all channels response
            if task_count + 1 < len(grouped_tasks.keys()):
                channels = [
                    f"{task.get('robot_name')}_to_roboos" for task in task_group
                ]
                self.communicator.wait_for_all_channels_response(
                    channels=channels, task_id=task_id
                )
        self.logger.info(f"Task_id ({task_id}) [{task}] has been sent to all agents.")
        return reasoning_and_subtasks

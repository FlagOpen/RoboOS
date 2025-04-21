# -*- coding: utf-8 -*-

import json
import os
import time
import importlib
import threading

from flask import Flask
from datetime import datetime
from typing import Dict, List
from utils import convert_yaml_to_json, communicator, config
from agents.models import OpenAIServerModel, AzureOpenAIServerModel
from agents.slaver_agent import ToolCallingAgent


app = Flask(__name__)


class RobotManager:
    """Centralized robot management system with task handling and communication"""

    def __init__(self, communicator, model="robobrain"):
        """
        Args:
            communicator: Message broker interface
            model_initializer: AI model initialization function
            config: System configuration dictionary
        """
        self.communicator = communicator
        self.heartbeat_interval = 60
        self.lock = threading.Lock()
        self.model = self._gat_model_info_from_config()
        self.register_robot()

    def _gat_model_info_from_config(self):
        """Initial model"""
        for candidate in config["model"]["MODEL_LIST"]:
            if candidate["CLOUD_MODEL"] in config["model"]["MODEL_SELECT"]:
                if candidate["CLOUD_TYPE"] == "azure":
                    model_client = OpenAIServerModel(
                        model_id=config["model"]["MODEL_SELECT"],
                        azure_endpoint=candidate["AZURE_ENDPOINT"],
                        azure_deployment=candidate["AZURE_DEPLOYMENT"],
                        api_key=candidate["AZURE_API_KEY"],
                        api_version=candidate["AZURE_API_VERSION"],
                    )
                elif candidate["CLOUD_TYPE"] == "default":
                    model_client = AzureOpenAIServerModel(
                        api_key=candidate["CLOUD_API_KEY"],
                        api_base=candidate["CLOUD_SERVER"],
                        model_id=candidate["CLOUD_MODEL"],
                    )
                else:
                    raise ValueError(
                        f"Unsupported cloud type: {candidate['CLOUD_TYPE']}"
                    )
                return model_client

    def handle_task(self, data: Dict) -> None:
        """Process incoming tasks with thread-safe operation"""
        task_data = {
            "task": data.get("task"),
            "task_id": data.get("task_id"),
            "order_flag": data.get("order_flag", "false"),
        }
        with self.lock:
            self._execute_task(task_data)

    def _execute_task(self, task_data: Dict) -> None:
        """Internal task execution logic"""

        robot_tool = self._get_tools()

        os.makedirs("./.log", exist_ok=True)
        agent = ToolCallingAgent(
            tools=robot_tool,
            verbosity_level=2,
            model=self.model,
            log_file="./.log/agent.log",
            robot_name=self.robot_profile["robot_name"],
            communicator=self.communicator,
        )
        result = agent.run(task=task_data["task"])
        self._send_result(
            robot_name=self.robot_profile["robot_name"],
            task=task_data["task"],
            task_id=task_data["task_id"],
            result=result,
            tool_call=agent.tool_call,
        )

    def _get_tools(self) -> List:
        """Get toolset based on configured brand"""
        robot_tool = self.robot_profile["robot_tool"]
        tools = []
        for robot in robot_tool:
            module_path, class_name = robot["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            tools.append(getattr(module, class_name)())
        return tools

    def _send_result(
        self, robot_name: str, task: str, task_id: str, result: Dict, tool_call: List
    ) -> None:
        """Send task results to communication channel"""
        channel = f"{robot_name}_to_roboos"
        payload = {
            "robot_name": robot_name,
            "subtask_handle": task,
            "subtask_result": json.dumps(result),
            "tools": tool_call,
            "task_id": task_id,
        }
        self.communicator.send(channel, json.dumps(payload))

    def register_robot(self) -> None:
        """Complete robot registration with thread management"""

        self.robot_profile = convert_yaml_to_json(config["profile"]["PATH"])
        robot_name = self.robot_profile["robot_name"]
        register = {
            "robot_name": robot_name,
            "robot_type": self.robot_profile["robot_type"],
            "robot_tool": [
                tool["tool_name"] for tool in self.robot_profile["robot_tool"]
            ],
            "current_position": self.robot_profile["current_position"],
            "navigate_position": self.robot_profile["navigate_position"],
            "robot_state": "idle",
            "timestamp": int(datetime.now().timestamp()),
        }
        self.robot_profile["robot_state"] = "idle"
        with self.lock:
            # Registration thread
            self.communicator.send("robot_registration", json.dumps(register))
            self.communicator.register(
                f"ROBOT_INFO_{robot_name}", json.dumps(register), expire_second=60
            )
            # Heartbeat thread
            threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                args=(robot_name,),
                name=f"heartbeat_{robot_name}",
            ).start()

            # Command listener thread
            channel_b2r = f"roboos_to_{robot_name}"
            threading.Thread(
                target=lambda: self.communicator.listen(channel_b2r, self.handle_task),
                daemon=True,
                name=channel_b2r,
            ).start()

    def _heartbeat_loop(self, robot_name) -> None:
        """Continuous heartbeat signal emitter"""
        key = f"ROBOT_INFO_{robot_name}"
        while True:
            self.communicator.set_ttl(key, seconds=60)
            time.sleep(30)


if __name__ == "__main__":
    # start the Flask app
    robot_manager = RobotManager(communicator)
    app.run(host="0.0.0.0", port=5001)

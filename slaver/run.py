# -*- coding: utf-8 -*-

import asyncio
import json
import os
import signal
import sys
import threading
import time
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Dict, List, Optional

import yaml
from agents.models import AzureOpenAIServerModel, OpenAIServerModel
from agents.slaver_agent import ToolCallingAgent
from flag_scale.flagscale.agent.collaboration import Collaborator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from tools.utils import Config

config = Config.load_config()
collaborator = Collaborator.from_config(config=config["collaborator"])


class RobotManager:
    """Centralized robot management system with task handling and collaboration"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.collaborator = collaborator
        self.heartbeat_interval = 60
        self.lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.model, self.model_path = self._gat_model_info_from_config()
        self.tools = None
        self.threads = []
        self.loop = asyncio.get_event_loop()
        self.robot_name = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()

    async def _safe_cleanup(self):
        if hasattr(self, "session") and self.session:
            await self.cleanup()

        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

    def _gat_model_info_from_config(self):
        """Initial model"""
        candidate = config["model"]["model_dict"]
        if candidate["cloud_model"] in config["model"]["model_select"]:
            if candidate["cloud_type"] == "azure":
                model_client = AzureOpenAIServerModel(
                    model_id=config["model"]["model_select"],
                    azure_endpoint=candidate["azure_endpoint"],
                    azure_deployment=candidate["azure_deployment"],
                    api_key=candidate["azure_api_key"],
                    api_version=candidate["azure_api_version"],
                    support_tool_calls=config["tool"]["support_tool_calls"],
                    profiling=config["profiling"],
                )
                model_name = config["model"]["model_select"]
            elif candidate["cloud_type"] == "default":
                model_client = OpenAIServerModel(
                    api_key=candidate["cloud_api_key"],
                    api_base=candidate["cloud_server"],
                    model_id=candidate["cloud_model"],
                    support_tool_calls=config["tool"]["support_tool_calls"],
                    profiling=config["profiling"],
                )
                model_name = config["model"]["model_select"]
            else:
                raise ValueError(f"Unsupported cloud type: {candidate['cloud_type']}")
            return model_client, model_name
        raise ValueError(f"Unsupported model: {config['model']['model_select']}")

    def handle_task(self, data: str) -> None:
        """Process incoming tasks with thread-safe operation"""
        if self._shutdown_event.is_set():
            return

        data = json.loads(data)
        task_data = {
            "task": data.get("task"),
            "task_id": data.get("task_id"),
            "refresh": data.get("refresh"),
            "order_flag": data.get("order_flag", "false"),
        }
        with self.lock:
            future = asyncio.run_coroutine_threadsafe(
                self._execute_task(task_data), self.loop
            )
            future.result()

    async def _execute_task(self, task_data: Dict) -> None:
        """Internal task execution logic"""
        if self._shutdown_event.is_set():
            return

        os.makedirs("./.log", exist_ok=True)
        agent = ToolCallingAgent(
            tools=self.tools,
            verbosity_level=2,
            model=self.model,
            model_path=self.model_path,
            log_file="./.log/agent.log",
            robot_name=self.robot_name,
            collaborator=self.collaborator,
            tool_executor=self.session.call_tool,
        )
        task = task_data["task"]
        result = await agent.run(task)
        self._send_result(
            robot_name=self.robot_name,
            task=task,
            task_id=task_data["task_id"],
            result=result,
            tool_call=agent.tool_call,
        )

    def _send_result(
        self, robot_name: str, task: str, task_id: str, result: Dict, tool_call: List
    ) -> None:
        """Send task results to collaboration channel"""
        if self._shutdown_event.is_set():
            return

        channel = f"{robot_name}_to_RoboOS"
        payload = {
            "robot_name": robot_name,
            "subtask_handle": task,
            "subtask_result": result,
            "tools": tool_call,
            "task_id": task_id,
        }
        self.collaborator.send(channel, json.dumps(payload))

    def _heartbeat_loop(self, robot_name) -> None:
        """Continuous heartbeat signal emitter"""
        key = robot_name
        while not self._shutdown_event.is_set():
            try:
                self.collaborator.agent_heartbeat(key, seconds=60)
                time.sleep(30)
            except Exception as e:
                if not self._shutdown_event.is_set():
                    print(f"Heartbeat error: {e}")
                break

    async def connect_to_robot(self):
        """Connect to an MCP server"""

        call_type = config["robot"]["call_type"]

        if call_type == "local":
            server_params = StdioServerParameters(
                command="python", args=[config["robot"]["path"] + "/skill.py"], env=None
            )
            mcp_client = stdio_client(server_params)

        if call_type == "remote":
            mcp_client = streamablehttp_client(config["robot"]["path"] + "/mcp")

        stdio_transport = await self.exit_stack.enter_async_context(mcp_client)
        if call_type == "local":
            self.stdio, self.write = stdio_transport
        if call_type == "remote":
            self.stdio, self.write, _ = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = [
            {
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                },
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        print("Connected to robot with tools:", str(self.tools))

        """Complete robot registration with thread management"""
        robot_name = config["robot"]["name"]
        self.robot_name = robot_name
        register = {
            "robot_name": robot_name,
            "robot_tool": self.tools,
            "robot_state": "idle",
            "timestamp": int(datetime.now().timestamp()),
        }
        with self.lock:
            # Registration thread
            self.collaborator.register_agent(
                robot_name, json.dumps(register), expire_second=60
            )

            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                args=(robot_name,),
                name=f"heartbeat_{robot_name}",
            )
            heartbeat_thread.start()
            self.threads.append(heartbeat_thread)

            # Command listener thread
            channel_b2r = f"roboos_to_{robot_name}"
            listener_thread = threading.Thread(
                target=lambda: self.collaborator.listen(channel_b2r, self.handle_task),
                daemon=True,
                name=channel_b2r,
            )
            listener_thread.start()
            self.threads.append(listener_thread)

    async def cleanup(self):
        """Clean up resources"""
        self._shutdown_event.set()
        await self.exit_stack.aclose()


async def main():
    robot_manager = RobotManager()
    try:
        print("connecting to robot...")
        await robot_manager.connect_to_robot()
        print("connection success")

        while not robot_manager._shutdown_event.is_set():
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await robot_manager._safe_cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
        sys.exit(0)

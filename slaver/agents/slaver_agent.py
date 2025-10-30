#!/usr/bin/env python
# coding=utf-8
import json
import os
import sys
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Union

from agents.models import ChatMessage
from mcp import ClientSession
from rich.panel import Panel
from rich.text import Text

from tools.agent_status_manager import AgentStatusManager
from tools.long_term_memory import LongTermMemory
from tools.memory import ActionStep, AgentMemory, SceneMemory, ShortTermMemory
from tools.monitoring import AgentLogger, LogLevel, Monitor

# Import flagscale last to avoid path conflicts
from flag_scale.flagscale.agent.collaboration import Collaborator


logger = getLogger(__name__)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
    """

    def __init__(
        self,
        tools: List[Dict[str, str]],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        model_path: str,
        collaborator: Collaborator,
        tool_executor: ClientSession,
        robot_name: str,
        max_steps: int = 20,
        verbosity_level: LogLevel = LogLevel.INFO,
        step_callbacks: Optional[List[Callable]] = None,
        log_file: Optional[str] = None,
    ):
        self.tools = tools
        self.model = model
        self.model_path = model_path
        self.collaborator = collaborator
        self.robot_name = robot_name
        self.tool_executor = tool_executor
        self.task_id = None
        self.max_steps = max_steps
        self.step_number = 0
        self.state = {}
        self.memory = AgentMemory()
        self.scene = SceneMemory(collaborator)
        self.logger = AgentLogger(level=verbosity_level, log_file=log_file)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

        self.optimization_enabled = os.getenv("ROBOOS_DISABLE_OPTIMIZATION", "false").lower() != "true"

        redis_host = getattr(collaborator, 'redis_host', 'localhost')
        redis_port = getattr(collaborator, 'redis_port', 6379)
        if self.optimization_enabled:
            self.status_manager = AgentStatusManager(redis_host, redis_port)
        else:
            self.status_manager = None

        if self.optimization_enabled:
            self.short_term_memory = ShortTermMemory(capacity=20)
        else:
            self.short_term_memory = None

        if self.optimization_enabled:
            try:
                self.long_term_memory = LongTermMemory(redis_host, redis_port)
                self.logger.log("Long-term memory initialized for storage only (optimization enabled)", LogLevel.DEBUG)
            except Exception as e:
                self.logger.log(f"Long-term memory initialization failed: {e}", LogLevel.INFO)
                self.long_term_memory = None
        else:
            self.long_term_memory = None
            self.logger.log("Running in BASELINE mode (optimizations disabled)", LogLevel.INFO)

    async def run(
        self,
        task: str,
        reset: bool = True,
        images: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task

        if ":" in task:
            self.task_id, subtask_desc = task.split(":", 1)
        else:
            self.task_id = "unknown"


        if reset:
            self.memory.reset()
            if self.short_term_memory:
                self.short_term_memory.reset()
            self.step_number = 1

            if self.status_manager:
                self.status_manager.clear_status(self.robot_name)
                self.logger.log("Agent status and memory cleared for new task", LogLevel.DEBUG)

        if self.short_term_memory:
            task_id = f"task_{int(time.time())}"
            self.short_term_memory.start_task(task_id, task)

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        while self.step_number <= max_steps:
            step_start_time = time.time()
            step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            answer = await self.step(step)
            if answer == "final_answer":
                if self.optimization_enabled:
                    self._save_to_long_term_memory(success=True)
                return "Mission accomplished"

            if self.status_manager:
                self.status_manager.record_status(self.robot_name, answer)
            else:
                self.collaborator.record_agent_status(self.robot_name, answer)

            step.end_time = time.time()
            self.step_number += 1

        if self.optimization_enabled:
            self._save_to_long_term_memory(success=False)
        return "Maximum number of attempts reached, Mission not completed"

    def _save_to_long_term_memory(self, success: bool):
        """Save current subtask execution result to long-term memory

        Args:
            success: Whether the subtask succeeded
        """
        if not hasattr(self, 'long_term_memory') or not self.long_term_memory:
            return

        try:
            if self.short_term_memory and self.short_term_memory.current_context:
                current_context = self.short_term_memory.current_context

                recent_actions = current_context.get_recent_actions(1)
                if recent_actions:
                    latest_action = recent_actions[0]

                    observation_data = {
                        "subtask": current_context.task_text,
                        "execution_result": latest_action.tool_result_summary,
                        "success": success,
                        "timestamp": latest_action.timestamp,
                        "execution_agent": self.robot_name
                    }

                    self.long_term_memory.store_observation(observation_data)

                    self.logger.log(
                        f"Subtask observation saved to long-term memory (success={success})",
                        LogLevel.DEBUG
                    )
        except Exception as e:
            self.logger.log(f"Failed to save observation to long-term memory: {e}", LogLevel.INFO)

    def _save_tool_call_to_json(self, tool_name: str, tool_arguments: dict):
        """Save tool call data to JSON file"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.log')
        os.makedirs(log_dir, exist_ok=True)

        json_file = os.path.join(log_dir, f"slaver_data_{self.task_id}.json")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tool_call_data = {
            "task_id": self.task_id,
            "task": self.task,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "type": "tool_call"
        }

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {"tasks": []}
        else:
            data = {"tasks": []}

        current_task_data = None
        for task_data in data["tasks"]:
            if task_data.get("task_id") == self.task_id:
                current_task_data = task_data
                break

        if current_task_data is None:
            current_task_data = {
                "task_id": self.task_id,
                "task": self.task,
                "tool_calls": [],
                "tool_results": [],
                "reasoning": []
            }
            data["tasks"].append(current_task_data)

        current_task_data["tool_calls"].append(tool_call_data)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_tool_result_to_json(self, tool_name: str, observation: str):
        """Save tool result data to JSON file"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.log')
        os.makedirs(log_dir, exist_ok=True)

        json_file = os.path.join(log_dir, f"slaver_data_{self.task_id}.json")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tool_result_data = {
            "task_id": self.task_id,
            "task": self.task,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "observation": observation,
            "type": "tool_result"
        }

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {"tasks": []}
        else:
            data = {"tasks": []}

        current_task_data = None
        for task_data in data["tasks"]:
            if task_data.get("task_id") == self.task_id:
                current_task_data = task_data
                break

        if current_task_data is None:
            current_task_data = {
                "task_id": self.task_id,
                "task": self.task,
                "tool_calls": [],
                "tool_results": [],
                "reasoning": []
            }
            data["tasks"].append(current_task_data)

        current_task_data["tool_results"].append(tool_result_data)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def step(self) -> Optional[Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        raise NotImplementedError


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: List[Dict[str, str]],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        model_path: str,
        collaborator: Collaborator,
        robot_name: str,
        **kwargs,
    ):
        self.tool_call = []
        super().__init__(
            tools=tools,
            model=model,
            model_path=model_path,
            collaborator=collaborator,
            robot_name=robot_name,
            **kwargs,
        )

    async def _execute_tool_call(
        self, tool_name: str, tool_arguments: dict, memory_step: ActionStep
    ) -> Union[str, None]:
        call_start_time = time.time()


        parsed_args = json.loads(tool_arguments) if isinstance(tool_arguments, str) else tool_arguments
        self._save_tool_call_to_json(tool_name, parsed_args)

        self.logger.log(
            Panel(
                Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")
            ),
            level=LogLevel.INFO,
        )
        observation = await self.tool_executor(tool_name, json.loads(tool_arguments))
        observation = observation.content[0].text


        self._save_tool_result_to_json(tool_name, observation)

        self.logger.log(
            f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
            level=LogLevel.INFO,
        )


        # Construct memory input
        memory_input = {
            "tool_name": tool_name,
            "arguments": tool_arguments,
            "result": observation,
        }
        try:
            await self.memory_predict(memory_input)
        except Exception:
            pass

        if hasattr(self, 'short_term_memory') and self.short_term_memory:
            self.short_term_memory.add_action(
                step_number=self.step_number,
                tool_name=tool_name,
                tool_arguments=json.loads(tool_arguments) if isinstance(tool_arguments, str) else tool_arguments,
                tool_result=observation,
                success=not ("error" in str(observation).lower()),
                duration=time.time() - call_start_time,
                error_msg=observation if "error" in str(observation).lower() else None
            )

        return observation

    async def memory_predict(self, memory_input: dict) -> str:
        """
        Use the model to predict the scene-level effect of the current tool execution.
        Possible effects: add_object, remove_object, move_object, position.
        """
        prompt = self.scene.get_action_type_prompt(memory_input)

        model_message: ChatMessage = self.model(
            task=prompt,
            current_status="",
            model_path=self.model_path,
        )

        action_type = model_message.content.strip().lower()

        self.scene.apply_action(action_type, json.loads(memory_input["arguments"]))

    def _get_current_scene_info(self) -> dict:
        """Get current robot position information"""
        try:
            robot_info = self.collaborator.read_environment("robot")
            if robot_info:
                return {"robot": robot_info}
            else:
                return {"robot": {"position": None, "holding": None, "status": "idle"}}
        except Exception as e:
            self.logger.log(f"Failed to get robot info: {e}", LogLevel.DEBUG)
            return {"robot": {"position": None, "holding": None, "status": "idle"}}

    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)

        if self.status_manager:
            current_status = self.status_manager.read_latest_status(self.robot_name)
        else:
            current_status = self.collaborator.read_agent_status(self.robot_name)

        scene_info = self._get_current_scene_info()

        model_message: ChatMessage = self.model(
            task=self.task,
            current_status=current_status,
            model_path=self.model_path,
            tools_to_call_from=self.tools,
            stop_sequences=["Observation:"],
            scene_info=scene_info,
        )
        memory_step.model_output_message = model_message

        if model_message.content and model_message.content.strip():
            reasoning_content = model_message.content.strip()
            if reasoning_content.startswith("Since") or "no action is needed" in reasoning_content or "no further action" in reasoning_content:
                self._save_reasoning_to_json(reasoning_content)

        self.logger.log_markdown(
            content=(
                model_message.content
                if model_message.content
                else str(model_message.raw)
            ),
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        if model_message.content and ("no action is needed" in model_message.content.lower() or
                                    "no action required" in model_message.content.lower() or
                                    "no tool calls" in model_message.content.lower() or
                                    "no tool will be called" in model_message.content.lower()):
            return "final_answer"

        if model_message.tool_calls:
            tool_call = model_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments
        else:
            return "final_answer"

        current_call = {"tool_name": tool_name, "tool_arguments": tool_arguments}

        if self.tool_call and self.tool_call[-1] == current_call:
            return "final_answer"
        else:
            self.tool_call.append(current_call)

        return await self._execute_tool_call(tool_name, tool_arguments, memory_step)

    def _save_reasoning_to_json(self, reasoning_content: str):
        """Save Slaver reasoning process to JSON file"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.log')
        os.makedirs(log_dir, exist_ok=True)

        json_file = os.path.join(log_dir, f"slaver_data_{self.task_id}.json")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reasoning_data = {
            "task_id": self.task_id,
            "task": self.task,
            "timestamp": timestamp,
            "reasoning_content": reasoning_content,
            "type": "reasoning"
        }

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {"tasks": []}
        else:
            data = {"tasks": []}

        current_task_data = None
        for task_data in data["tasks"]:
            if task_data.get("task_id") == self.task_id:
                current_task_data = task_data
                break

        if current_task_data is None:
            current_task_data = {
                "task_id": self.task_id,
                "task": self.task,
                "tool_calls": [],
                "tool_results": [],
                "reasoning": []
            }
            data["tasks"].append(current_task_data)

        current_task_data["reasoning"].append(reasoning_data)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

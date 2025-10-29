import asyncio
import json
import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Dict

import yaml
from agents.planner import GlobalTaskPlanner

# Import flagscale last to avoid path conflicts
from flag_scale.flagscale.agent.collaboration import Collaborator


class GlobalAgent:
    def __init__(self, config_path="config.yaml"):
        """Initialize GlobalAgent"""
        self._init_config(config_path)
        self._init_logger(self.config["logger"])
        self.collaborator = Collaborator.from_config(self.config["collaborator"])
        self.planner = GlobalTaskPlanner(self.config)

        self.logger.info(f"Configuration loaded from {config_path} ...")
        self.logger.info(f"Master Configuration:\n{self.config}")

        self._init_scene(self.config["profile"])
        self._start_listener()

    def _init_logger(self, logger_config):
        """Initialize an independent logger for GlobalAgent"""
        self.logger = logging.getLogger(logger_config["master_logger_name"])
        logger_file = logger_config["master_logger_file"]
        os.makedirs(os.path.dirname(logger_file), exist_ok=True)
        file_handler = logging.FileHandler(logger_file)

        # Set the logging level
        if logger_config["master_logger_level"] == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        elif logger_config["master_logger_level"] == "INFO":
            self.logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
        elif logger_config["master_logger_level"] == "WARNING":
            self.logger.setLevel(logging.WARNING)
            file_handler.setLevel(logging.WARNING)
        elif logger_config["master_logger_level"] == "ERROR":
            self.logger.setLevel(logging.ERROR)
            file_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _init_config(self, config_path="config.yaml"):
        """Initialize configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _init_scene(self, scene_config):
        """Initialize scene object"""
        path = scene_config["path"]
        if not os.path.exists(path):
            self.logger.error(f"Scene config file {path} does not exist.")
            raise FileNotFoundError(f"Scene config file {path} not found.")
        with open(path, "r", encoding="utf-8") as f:
            self.scene = yaml.safe_load(f)

        scenes = self.scene.get("scene", [])
        for scene_info in scenes:
            scene_name = scene_info.pop("name", None)
            if scene_name:
                self.collaborator.record_environment(scene_name, json.dumps(scene_info))
            else:
                self.logger.warning("Warning: Missing 'name' in scene_info: %s", scene_info)

    def _handle_register(self, robot_name: Dict) -> None:
        """Listen for robot registrations."""
        robot_info = self.collaborator.read_agent_info(robot_name)
        self.logger.info(
            f"AGENT_REGISTRATION: {robot_name} \n {json.dumps(robot_info)}"
        )

        # Register functions for processing robot execution results in the brain
        channel_r2b = f"{robot_name}_to_RoboOS"
        threading.Thread(
            target=lambda: self.collaborator.listen(channel_r2b, self._handle_result),
            daemon=True,
            name=channel_r2b,
        ).start()

        self.logger.info(
            f"RoboOS has listened to [{robot_name}] by channel [{channel_r2b}]"
        )

    def _handle_result(self, data: str):
        data = json.loads(data)

        """Handle results from agents."""
        robot_name = data.get("robot_name")
        subtask_handle = data.get("subtask_handle")
        subtask_result = data.get("subtask_result")

        if robot_name and subtask_handle and subtask_result:
            self.logger.info(
                f"================ Received result from {robot_name} ================"
            )
            self.logger.info(f"Subtask: {subtask_handle}\nResult: {subtask_result}")
            self.logger.info(
                "===================================================================="
            )
            self.collaborator.update_agent_busy(robot_name, False)

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
            target=lambda: self.collaborator.listen(
                "AGENT_REGISTRATION", self._handle_register
            ),
            daemon=True,
        ).start()
        self.logger.info("Started listening for robot registrations...")

    def reasoning_and_subtasks_is_right(self, reasoning_and_subtasks: dict) -> bool:
        """
        Verify if all robots mentioned in the task decomposition exist in the system registry

        Args:
            reasoning_and_subtasks: Task decomposition dictionary with format:
                {
                    "reasoning_explanation": "...",
                    "subtask_list": [
                        {"robot_name": "xxx", ...},
                        {"robot_name": "xxx", ...}
                    ]
                }

        Returns:
            bool: True if all robots are registered, False if any invalid robots found
        """
        # Check if input has correct structure
        if not isinstance(reasoning_and_subtasks, dict):
            return False

        if "subtask_list" not in reasoning_and_subtasks:
            return False

        # Extract all unique robot names from subtask_list
        try:
            worker_list = {
                subtask["robot_name"]
                for subtask in reasoning_and_subtasks["subtask_list"]
                if isinstance(subtask, dict) and "robot_name" in subtask
            }

            # Read list of all registered robots from the collaborator
            robots_list = set(self.collaborator.read_all_agents_name())

            # Check if all workers are registered
            return worker_list.issubset(robots_list)

        except (TypeError, KeyError):
            return False

    def _save_task_data_to_json(self, task_id: str, task: str, reasoning_and_subtasks: dict):
        """Save task data to JSON file - single file stores all tasks"""
        import os
        from datetime import datetime

        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.log')
        os.makedirs(log_dir, exist_ok=True)

        json_file = os.path.join(log_dir, f"master_data_{task_id}.json")
        current_task_data = {
            "task_id": task_id,
            "task": task,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "reasoning_explanation": reasoning_and_subtasks.get("reasoning_explanation", ""),
            "subtask_list": reasoning_and_subtasks.get("subtask_list", []),
            "prompt_content": self._get_last_prompt_content()
        }

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {"tasks": []}
        else:
            data = {"tasks": []}

        data["tasks"].append(current_task_data)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_last_prompt_content(self) -> str:
        """Get the last prompt content"""
        if hasattr(self.planner, 'last_prompt_content'):
            return self.planner.last_prompt_content
        return ""

    def _store_task_to_long_term_memory(self, task_id: str, task: str, reasoning_and_subtasks: dict):
        """Store task decomposition results to long-term memory

        Args:
            task_id: Task ID
            task: Original task description
            reasoning_and_subtasks: Task decomposition results
        """
        if not hasattr(self.planner, 'long_term_memory'):
            self.logger.warning(f"Planner does not have long_term_memory attribute, cannot store task {task_id}")
            return
        if not self.planner.long_term_memory:
            self.logger.warning(f"Planner's long_term_memory is None, cannot store task {task_id}")
            return

        self.logger.info(f"[LongTermMemory] Storing task {task_id} to long-term memory: {task[:50]}")

        try:
            import time
            import sys
            import os
            import importlib.util

            # Import TaskContext and CompactActionStep from slaver memory module
            _slaver_path = os.path.join(os.path.dirname(__file__), '..', '..', 'slaver')
            sys.path.insert(0, _slaver_path)
            try:
            from tools.memory import TaskContext, CompactActionStep
            except ImportError:
                # Fallback to direct file loading
                _memory_file = os.path.join(_slaver_path, 'tools', 'memory.py')
                spec = importlib.util.spec_from_file_location('memory_module', _memory_file)
                memory_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(memory_module)
                TaskContext = memory_module.TaskContext
                CompactActionStep = memory_module.CompactActionStep

            subtask_list = reasoning_and_subtasks.get("subtask_list", [])

            tool_sequence = []
            for subtask in subtask_list:
                subtask_desc = subtask.get("subtask", "")
                if "Navigate" in subtask_desc:
                    tool_sequence.append("Navigate")
                elif "Grasp" in subtask_desc:
                    tool_sequence.append("Grasp")
                elif "Place" in subtask_desc:
                    tool_sequence.append("Place")

            start_time = time.time()

            actions = []
            for i, subtask in enumerate(subtask_list, 1):
                subtask_desc = subtask.get("subtask", "")
                action = CompactActionStep(
                    step_number=i,
                    timestamp=start_time + i,
                    tool_name=subtask_desc.split()[0] if subtask_desc else "unknown",
                    tool_arguments={},
                    tool_result_summary=f"Subtask: {subtask_desc}",
                    success=True,
                    duration=1.0,
                    error_msg=None
                )
                actions.append(action)

            task_context = TaskContext(
                task_id=task_id,
                task_text=task,
                start_time=start_time,
                actions=actions,
                end_time=start_time + len(subtask_list),
                success=True
            )
            stored_id = self.planner.long_term_memory.store_task_episode(task_context)
            self.logger.info(f"[LongTermMemory] ✅ Task {task_id} stored to long-term memory as {stored_id}")
            self.logger.info(f"[LongTermMemory] ✅ Task {task_id} stored to long-term memory")

        except Exception as e:
            error_msg = f"Failed to store task {task_id} to long-term memory: {e}"
            self.logger.warning(error_msg)
            self.logger.warning(f"[LongTermMemory] ❌ {error_msg}")
            import traceback
            self.logger.warning(traceback.format_exc())

    def publish_global_task(self, task: str, refresh: bool, task_id: str) -> Dict:
        """Publish a global task to all Agents"""
        self.logger.info(f"[TASK_START:{task_id}] {task}")

        response = self.planner.forward(task)
        reasoning_and_subtasks = self._extract_json(response)

        # Retry if JSON extraction fails
        attempt = 0
        while (not self.reasoning_and_subtasks_is_right(reasoning_and_subtasks)) and (
            attempt < self.config["model"]["model_retry_planning"]
        ):
            response = self.planner.forward(task)
            reasoning_and_subtasks = self._extract_json(response)
            attempt += 1

        if reasoning_and_subtasks is None:
            reasoning_and_subtasks = {"error": "Failed to extract valid task decomposition"}
        self.logger.info(f"[MASTER_RESPONSE:{task_id}] {json.dumps(reasoning_and_subtasks, ensure_ascii=False)}")

        self._save_task_data_to_json(task_id, task, reasoning_and_subtasks)
        if reasoning_and_subtasks and "error" not in reasoning_and_subtasks:
            self._store_task_to_long_term_memory(task_id, task, reasoning_and_subtasks)

        subtask_list = reasoning_and_subtasks.get("subtask_list", [])
        grouped_tasks = self._group_tasks_by_order(subtask_list)

        task_id = task_id or str(uuid.uuid4()).replace("-", "")

        try:
            from subtask_analyzer import SubtaskAnalyzer
            import os
            log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.log')
            analyzer = SubtaskAnalyzer(log_dir=log_dir)
            if isinstance(task, list):
                task_str = task[0] if task else str(task)
            else:
                task_str = str(task)

            decomposition_record = analyzer.record_decomposition(
                task_id=task_id,
                original_task=task_str,
                reasoning_and_subtasks=reasoning_and_subtasks
            )
            self.logger.info(f"Subtask decomposition recorded: {decomposition_record.decomposition_quality}")
            self.logger.info(f"Decomposition details: {len(subtask_list)} subtasks")
            for i, subtask in enumerate(subtask_list, 1):
                self.logger.info(f"  {i}. [{subtask.get('robot_name', 'unknown')}] {subtask.get('subtask', '')}")
        except Exception as e:
            self.logger.warning(f"Failed to record subtask: {e}")

        threading.Thread(
            target=asyncio.run,
            args=(self._dispath_subtasks_async(task, task_id, grouped_tasks, refresh),),
            daemon=True,
        ).start()

        return reasoning_and_subtasks

    async def _dispath_subtasks_async(
        self, task: str, task_id: str, grouped_tasks: Dict, refresh: bool
    ):
        order_flag = "false" if len(grouped_tasks.keys()) == 1 else "true"
        for task_count, (order, group_task) in enumerate(grouped_tasks.items()):
            self.logger.info(f"Sending task group {order}:\n{group_task}")
            working_robots = []
            for tasks in group_task:
                robot_name = tasks.get("robot_name")
                subtask_data = {
                    "task_id": task_id,
                    "task": tasks["subtask"],
                    "order": order_flag,
                }
                if refresh:
                    self.collaborator.clear_agent_status(robot_name)
                self.collaborator.send(
                    f"roboos_to_{robot_name}", json.dumps(subtask_data)
                )
                working_robots.append(robot_name)
                self.collaborator.update_agent_busy(robot_name, True)
            self.logger.info(f"Tasks sent to {len(working_robots)} agents, executing asynchronously...")
        self.logger.info(f"Task_id ({task_id}) [{task}] has been sent to all agents.")

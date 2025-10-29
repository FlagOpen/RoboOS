from typing import Any, Dict, Optional, Union
import logging

import yaml
from agents.prompts import MASTER_PLANNING_PLANNING
from openai import AzureOpenAI, OpenAI

import os
import sys
import importlib.util

# Import LongTermMemory from slaver module
_slaver_path = os.path.join(os.path.dirname(__file__), '..', '..', 'slaver')
sys.path.insert(0, _slaver_path)
try:
    from tools.long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

# Import flagscale last to avoid path conflicts
from flag_scale.flagscale.agent.collaboration import Collaborator


class GlobalTaskPlanner:
    """A tool planner to plan task into sub-tasks."""

    def __init__(
        self,
        config: Union[Dict, str] = None,
    ) -> None:
        self.logger = logging.getLogger("GlobalTaskPlanner")
        self.collaborator = Collaborator.from_config(config["collaborator"])

        self.global_model: Any
        self.model_name: str
        self.global_model, self.model_name = self._get_model_info_from_config(
            config["model"]
        )

        self.profiling = config["profiling"]

        self.long_term_memory: Optional[LongTermMemory] = None
        memory_config = config.get("long_term_memory", {})
        if memory_config.get("enabled", False) and LongTermMemory is not None:
            try:
                self.long_term_memory = LongTermMemory(
                    redis_host=memory_config.get("redis_host", "127.0.0.1"),
                    redis_port=memory_config.get("redis_port", 6379)
                )
            except Exception:
                self.long_term_memory = None

        self.memory_enabled = memory_config.get("enabled", False) and self.long_term_memory is not None
        self.memory_similarity_threshold = memory_config.get("similarity_threshold", 0.6)
        self.memory_max_tasks = memory_config.get("max_historical_tasks", 3)
        self.memory_filter_success = memory_config.get("filter_success_only", True)

    def _get_model_info_from_config(self, config: Dict) -> tuple:
        """Get the model info from config."""
        candidate = config["model_dict"]
        if candidate["cloud_model"] in config["model_select"]:
            if candidate["cloud_type"] == "azure":
                model_name = config["model_select"]
                model_client = AzureOpenAI(
                    azure_endpoint=candidate["azure_endpoint"],
                    azure_deployment=candidate["azure_deployment"],
                    api_version=candidate["azure_api_version"],
                    api_key=candidate["azure_api_key"],
                )
            elif candidate["cloud_type"] == "default":
                model_client = OpenAI(
                    base_url=candidate["cloud_server"],
                    api_key=candidate["cloud_api_key"],
                )
                model_name = config["model_select"]
            else:
                raise ValueError(f"Unsupported cloud type: {candidate['cloud_type']}")
            return model_client, model_name
        raise ValueError(f"Unsupported model: {config['model_select']}")

    def _init_config(self, config_path="config.yaml"):
        """Initialize configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def display_profiling_info(self, description: str, message: any):
        """
        Outputs profiling information if profiling is enabled.

        :param message: The content to be printed. Can be of any type.
        :param description: A brief title or description for the message.
        """
        if self.profiling:
            module_name = "master"
            self.logger.info(f" [{module_name}] {description}:")
            self.logger.info(message)

    def _format_scene_info(self, scene_info: dict) -> str:
        """Format scene information for better readability and to avoid confusion"""
        if not scene_info:
            return "No scene information available."

        formatted_lines = []
        formatted_lines.append("=== CURRENT SCENE STATE ===")

        if 'robot' in scene_info:
            robot = scene_info['robot']
            formatted_lines.append(f"🤖 ROBOT STATUS:")
            formatted_lines.append(f"   • Position: {robot.get('position', 'unknown')}")
            formatted_lines.append(f"   • Holding: {robot.get('holding', 'nothing')}")
            formatted_lines.append(f"   • Status: {robot.get('status', 'unknown')}")
            formatted_lines.append("")

        formatted_lines.append("📍 LOCATIONS AND OBJECTS:")
        for location, info in scene_info.items():
            if location == 'robot':
                continue

            location_type = info.get('type', 'unknown')
            contains = info.get('contains', [])

            formatted_lines.append(f"   📍 {location} ({location_type}):")
            if contains:
                for obj in contains:
                    formatted_lines.append(f"      - {obj}")
            else:
                formatted_lines.append(f"      - (empty)")

        formatted_lines.append("")
        formatted_lines.append("KEY INFORMATION:")

        all_objects = {}
        for location, info in scene_info.items():
            if location == 'robot':
                continue
            contains = info.get('contains', [])
            for obj in contains:
                if obj not in all_objects:
                    all_objects[obj] = []
                all_objects[obj].append(location)

        if all_objects:
            formatted_lines.append("   Objects and their locations:")
            for obj, locations in all_objects.items():
                formatted_lines.append(f"   • {obj}: {', '.join(locations)}")
        else:
            formatted_lines.append("   No objects found in scene")

        formatted_lines.append("")
        formatted_lines.append("TASK ANALYSIS GUIDANCE:")
        formatted_lines.append("   • Check if target objects are already at their destinations")
        formatted_lines.append("   • Verify robot's current position and holding status")
        formatted_lines.append("   • Confirm all required objects and locations exist")
        formatted_lines.append("   • Skip unnecessary movements if objects are already in place")

        formatted_lines.append("================================")
        return "\n".join(formatted_lines)

    def _format_historical_experiences(self, task: str) -> str:
        """Query similar historical tasks from long-term memory and format as prompt text

        Args:
            task: Current task description

        Returns:
            Formatted historical experience text, or empty string if no relevant history
        """
        if not self.memory_enabled or not self.long_term_memory:
            return ""

        try:
            similar_tasks = self.long_term_memory.search_similar_tasks(
                query=task,
                limit=self.memory_max_tasks,
                filter_success=self.memory_filter_success
            )

            if not similar_tasks:
                return ""

            filtered_tasks = [
                t for t in similar_tasks
                if t.get("score", 0.0) >= self.memory_similarity_threshold
            ]

            if not filtered_tasks:
                return ""

            formatted_lines = []
            formatted_lines.append("=== HISTORICAL EXPERIENCES ===")
            formatted_lines.append("The following are similar tasks that were previously executed:")
            formatted_lines.append("")

            for i, task_info in enumerate(filtered_tasks, 1):
                metadata = task_info.get("metadata", {})
                score = task_info.get("score", 0.0)
                task_text = metadata.get("task_text", "N/A")
                success = metadata.get("success", False)
                tool_sequence = metadata.get("tool_sequence", "N/A")
                duration = metadata.get("duration", 0)

                formatted_lines.append(f"{i}. Similarity: {score:.2%}")
                formatted_lines.append(f"   Task: {task_text}")
                formatted_lines.append(f"   Result: {'✅ Successful' if success else '❌ Failed'}")
                if tool_sequence and tool_sequence != "N/A":
                    formatted_lines.append(f"   Tool sequence: {tool_sequence}")
                if duration > 0:
                    formatted_lines.append(f"   Duration: {duration:.1f}s")
                formatted_lines.append("")

            formatted_lines.append("Please refer to these historical experiences when decomposing the current task.")
            formatted_lines.append("Learn from successful cases and avoid mistakes from failed cases.")
            formatted_lines.append("================================")

            return "\n".join(formatted_lines)

        except Exception as e:
            self.logger.warning(f"[LongTermMemory] Failed to query historical experiences: {e}")
            return ""

    def forward(self, task: str) -> str:
        """Get the sub-tasks from the task."""

        all_robots_name = self.collaborator.read_all_agents_name()
        all_robots_info = self.collaborator.read_all_agents_info()
        all_environments_info = self.collaborator.read_environment()

        formatted_scene_info = self._format_scene_info(all_environments_info)

        historical_experiences = self._format_historical_experiences(task)
        if historical_experiences:
            formatted_scene_info = formatted_scene_info + "\n\n" + historical_experiences
            self.logger.info(f"[LongTermMemory] Historical experiences added to prompt ({len(historical_experiences)} chars)")
        else:
            self.logger.info(f"[LongTermMemory] No historical experiences found (or below threshold)")

        content = MASTER_PLANNING_PLANNING.format(
            robot_name_list=all_robots_name, robot_tools_info=all_robots_info, task=task, scene_info=formatted_scene_info
        )
        self.logger.info(f"[PROMPT_START] {content}")
        self.last_prompt_content = content

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            },
        ]

        self.display_profiling_info("messages", messages)

        from datetime import datetime

        start_inference = datetime.now()
        response = self.global_model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=2048,
            seed=42,
        )
        end_inference = datetime.now()

        self.display_profiling_info(
            "inference time",
            f"inference start:{start_inference} end:{end_inference} during:{end_inference-start_inference}",
        )
        self.display_profiling_info("response", response)
        self.display_profiling_info("response.usage", response.usage)

        self.logger.info(f"[PROMPT_END] {response.choices[0].message.content}")

        return response.choices[0].message.content

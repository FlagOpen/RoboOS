from typing import Any, Dict, Optional, Union

import yaml
from openai import AzureOpenAI, OpenAI
from prompt.utils import get_master_planning_prompt, read_yaml_file


class GlobalTaskPlanner:
    """A tool planner to plan task into sub-tasks."""

    def __init__(
        self,
        config: Union[Dict, str] = None,
        name: Optional[str] = "GlobalTaskPlanner0",
    ) -> None:
        """Initialize the GlobalTaskPlanner."""
        self.name = name
        if config is None:
            config = {}

        if isinstance(config, str):
            config = self._init_config(config)

        if not isinstance(config, Dict):
            raise TypeError("config must be a dictionary or path string")
        # Initialize the planner with the config

        profile_section: Dict = config.get("profile", {})
        logger_section: Dict = config.get("logger", {})

        self.robot_profile_path: Optional[str] = (
            profile_section.get("ROBOT_PROFILE_PATH")
            if profile_section.get("ROBOT_PROFILE_ENABLE", False)
            else None
        )
        self.scene_profile_path: Optional[str] = (
            profile_section.get("SCENE_PROFILE_PATH")
            if profile_section.get("SCENE_PROFILE_ENABLE", False)
            else None
        )
        self.robot_memory_path: str = logger_section["ROBOT_MEMORY_YAML"]
        self.scene_memory_path: str = logger_section["SCENE_MEMORY_YAML"]

        self.global_memory: Dict = (
            read_yaml_file(self.robot_profile_path, self.scene_profile_path) or {}
        )
        self.global_model: Any
        self.model_name: str
        self.global_model, self.model_name = self._gat_model_info_from_config(
            config["model"]
        )

    def _get_prompt_from_memory(self, task: str, global_memory: Dict = None) -> str:
        """Get the prompt from memory."""
        if global_memory is not None:
            assert isinstance(
                global_memory, Dict
            ), "global_memory should be a dictionary."
            assert (
                "scene_profile" in global_memory
            ), "global_memory should contain scene_profile."
            self.global_memory = global_memory

        scene_profile = self.global_memory["scene_profile"]

        # Filter the robot profile to get the idle robots
        idle_robot_profile = []

        prompt = get_master_planning_prompt(idle_robot_profile, scene_profile, task)
        return prompt

    def _gat_model_info_from_config(self, config: Dict) -> tuple:
        """Get the model info from config."""
        candidate = config["MODEL_DICT"]
        if candidate["CLOUD_MODEL"] in config["MODEL_SELECT"]:
            if candidate["CLOUD_TYPE"] == "azure":
                model_name = config["MODEL_SELECT"]
                model_client = AzureOpenAI(
                    azure_endpoint=candidate["AZURE_ENDPOINT"],
                    azure_deployment=candidate["AZURE_DEPLOYMENT"],
                    api_version=candidate["AZURE_API_VERSION"],
                    api_key=candidate["AZURE_API_KEY"],
                )

            elif candidate["CLOUD_TYPE"] == "default":
                model_client = OpenAI(
                    base_url=candidate["CLOUD_SERVER"],
                    api_key=candidate["CLOUD_API_KEY"],
                )
                model_name = config["MODEL_SELECT"]
            else:
                raise ValueError(f"Unsupported cloud type: {candidate['CLOUD_TYPE']}")
            return model_client, model_name

        raise ValueError(f"Unsupported model: {config['MODEL_SELECT']}")

    def _init_config(self, config_path="config.yaml"):
        """Initialize configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def forward(self, task: str, global_memory: Dict = {}) -> str:
        """Get the sub-tasks from the task."""
        prompt = self._get_prompt_from_memory(task, global_memory)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        response = self.global_model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=8192,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    planner = GlobalTaskPlanner(config="config.yaml")
    task = "Take basket to kitchenTable, and put apple and knife into basket, and then take them back to customTable."
    response = planner.forward(task)
    print(response)

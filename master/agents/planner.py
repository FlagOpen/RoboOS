from typing import Any, Dict, Union

import yaml
from agents.prompts import MASTER_PLANNING_PLANNING
from flag_scale.flagscale.agent.collaboration import Collaborator
from openai import AzureOpenAI, OpenAI


class GlobalTaskPlanner:
    """A tool planner to plan task into sub-tasks."""

    def __init__(
        self,
        config: Union[Dict, str] = None,
    ) -> None:
        self.collaborator = Collaborator.from_config(config["collaborator"])

        self.global_model: Any
        self.model_name: str
        self.global_model, self.model_name = self._gat_model_info_from_config(
            config["model"]
        )

        self.profiling = config["profiling"]

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

    def display_profiling_info(self, description: str, message: any):
        """
        Outputs profiling information if profiling is enabled.

        :param message: The content to be printed. Can be of any type.
        :param description: A brief title or description for the message.
        """
        if self.profiling:
            module_name = "master"  # Name of the current module
            print(f" [{module_name}] {description}:")
            print(message)

    def forward(self, task: str) -> str:
        """Get the sub-tasks from the task."""

        all_robots_name = self.collaborator.retrieve_all_agents_name()
        all_robots_info = self.collaborator.retrieve_all_agents()

        content = MASTER_PLANNING_PLANNING.format(
            robot_name_list=all_robots_name, robot_tools_info=all_robots_info, task=task
        )

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

        return response.choices[0].message.content

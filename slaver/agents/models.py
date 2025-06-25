###############################################################
# Copyright 2025 BAAI. All rights reserved.
###############################################################
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.util
import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from tools.utils import config

logger = logging.getLogger(__name__)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]),
                    id=tc["id"],
                    type=tc["type"],
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class Model:
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self.last_input_token_count = None
        self.last_output_token_count = None

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!

    def to_dict(self) -> Dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "top_p",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary


@dataclass
class FunctionCall:
    name: str
    arguments: str
    description: str = None


@dataclass
class ToolCall:
    id: str
    type: str
    function: FunctionCall


def convert_chat_message(original_message):
    content = original_message.content

    json_match = re.search(r"```json\n(.*?)\n```", content, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        parsed_data = json.loads(json_str)
        if isinstance(parsed_data, dict) and "name" in parsed_data:
            return ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex}",
                        type="function",
                        function=FunctionCall(
                            name=parsed_data["name"],
                            arguments=json.dumps(parsed_data.get("arguments", {})),
                            description=None,
                        ),
                    )
                ],
                raw=None,
            )

    return ChatMessage(role="assistant", content=content, tool_calls=[], raw=None)


class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "robobrain").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        if importlib.util.find_spec("openai") is None:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            )
        super().__init__(flatten_messages_as_text=flatten_messages_as_text, **kwargs)
        self.model_id = model_id
        self.custom_role_conversions = custom_role_conversions
        self.client_kwargs = client_kwargs or {}
        self.client_kwargs.update(
            {
                "api_key": api_key,
                "base_url": api_base,
                "organization": organization,
                "project": project,
            }
        )
        self.client = self.create_client()

    def create_client(self):
        import openai

        return openai.OpenAI(**self.client_kwargs)

    def __call__(
        self,
        task: str,
        current_status: str,
        model_path: str,
        stop_sequences: Optional[List[str]] = None,
        tools_to_call_from: Optional[List[str]] = None,
    ) -> ChatMessage:
        content = task
        if len(current_status) > 0:
            content += " Currently completing the following actions: "
            for current_statu in current_status:
                content += f"{current_statu} "
        completion_kwargs = {
            "messages": [{"role": "user", "content": content}],
            "model": model_path,
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 8192,
        }

        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences

        if tools_to_call_from:
            completion_kwargs["tools"] = tools_to_call_from

        print(completion_kwargs)
        response = self.client.chat.completions.create(**completion_kwargs)
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        print(response)

        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            )
        )
        if config["tool"]["SUPPORT_TOOL_CALLS"] == False:
            first_message = convert_chat_message(first_message)
        first_message.raw = response
        return first_message


class AzureOpenAIServerModel(OpenAIServerModel):
    """This model connects to an Azure OpenAI deployment.

    Parameters:
        model_id (`str`):
            The model deployment name to use when connecting (e.g. "robobrain").
        azure_endpoint (`str`, *optional*):
            The Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`. If not provided, it will be inferred from the `AZURE_OPENAI_ENDPOINT` environment variable.
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, it will be inferred from the `AZURE_OPENAI_API_KEY` environment variable.
        api_version (`str`, *optional*):
            The API version to use. If not provided, it will be inferred from the `OPENAI_API_VERSION` environment variable.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the AzureOpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Azure OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        if importlib.util.find_spec("openai") is None:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIServerModel: `pip install 'smolagents[openai]'`"
            )
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
            }
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        import openai

        return openai.AzureOpenAI(**self.client_kwargs)


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "OpenAIServerModel",
    "AzureOpenAIServerModel",
    "ChatMessage",
]

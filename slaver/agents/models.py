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
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from tools.utils import (
    encode_image_base64,
    make_image_url,
)

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
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [
                ChatMessageToolCall.from_hf_api(tool_call)
                for tool_call in message.tool_calls
            ]
        return cls(
            role=message.role, content=message.content, tool_calls=tool_calls, raw=raw
        )

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


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(
                f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
            )

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image":
                    assert (
                        not flatten_messages_as_text
                    ), f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": make_image_url(
                                        encode_image_base64(element.pop("image"))
                                    )
                                },
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if (
            len(output_message_list) > 0
            and message["role"] == output_message_list[-1]["role"]
        ):
            assert isinstance(message["content"], list), "Error: wrong content:" + str(
                message["content"]
            )
            Observation = "," + message["content"][0]["text"]["Observation"]
            if type(output_message_list[-1]["content"]) is str:
                output_message_list[-1]["content"] += ". Current Status: " + Observation
            else:
                output_message_list[-1]["content"][0]["text"] += Observation
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list

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

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[str]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar
            
        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs["tools"] = tools_to_call_from

        completion_kwargs["n"] = kwargs.get("n", 1)
        completion_kwargs["temperature"] = kwargs.get("temperature", 0.0)
        completion_kwargs["top_p"] = kwargs.get("top_p", 1.0)
        completion_kwargs["max_tokens"] = kwargs.get("max_tokens", 8192)

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
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
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            tool_choice="auto",
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens

        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            )
        )
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

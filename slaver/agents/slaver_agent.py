#!/usr/bin/env python
# coding=utf-8
import inspect
import time
from collections import deque
from logging import getLogger
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

from jinja2 import StrictUndefined, Template
from rich.panel import Panel
from rich.text import Text
from tools.memory import (
    ActionStep,
    AgentMemory,
    TaskStep,
    ToolCall,
)


from agents.models import (
    ChatMessage,
    MessageRole,
)
from tools.monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from tools.utils import (
    AgentError,
    AgentGenerationError,
    AgentMaxStepsError
)

from tools.memory import Message


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
        tools_path: str,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        max_steps: int = 20,
        verbosity_level: LogLevel = LogLevel.INFO,
        step_callbacks: Optional[List[Callable]] = None,
        log_file: Optional[str] = None
    ):
        self.tools = tools
        self.tools_path = tools_path
        self.model = model
        self.max_steps = max_steps
        self.step_number = 0
        self.state = {}
        self.memory = AgentMemory()
        self.logger = AgentLogger(level=verbosity_level, log_file=log_file)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics) 

    
    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
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
        max_steps = max_steps or  self.max_steps
        self.task = task

        if reset:
            self.memory.reset()
            self.step_number = 1


        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))
        if stream:
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        return deque(
            self._run_stream(task=self.task, max_steps=max_steps, images=images), maxlen=1
        )[0]
        
    def _run_stream(self, task: str, max_steps: int, images: Optional[List[str]] = None)  -> Generator[ActionStep, None, None]:
        final_answer = None
        while not final_answer and self.step_number <= max_steps:
            step_start_time = time.time()
            step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                final_answer = self.step(step)

            except AgentError as e:
                step.error = e
            finally:
                step.end_time = time.time()
                self.memory.steps.append(step)
                yield step
                self.step_number += 1
        
        if not final_answer:
            final_answer = self._handle_max_steps_reached(task, images, step_start_time)
            yield final_answer
        yield final_answer
                
    def _handle_max_steps_reached(
        self, task: str, images: List[str], step_start_time: float
    ) -> Any:
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - step_start_time
        self.memory.steps.append(final_memory_step)
        for callback in self.step_callbacks:
            callback(final_memory_step) if len(
                inspect.signature(callback).parameters
            ) == 1 else callback(final_memory_step, agent=self)
        return final_answer
    
    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = []
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": task
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"
    

    def step(self) -> Optional[Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        raise NotImplementedError
    
    def execute_tool_call(
        self, tool_name: str, arguments: Union[Dict[str, str], str]
    ) -> Any:
        import importlib.util
        from pathlib import Path
        
        file_path = Path(self.tools_path).absolute()
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        func = getattr(module, tool_name)
        import json
        result = func(**(json.loads(arguments)))
            
        return result


    def write_memory_to_messages(
        self,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """   
        messages = []
        
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages())
        return messages


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
        tools_path: str,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        communicator=None,
        robot_name: str = None,
        **kwargs,
    ):
        self.tool_call = []
        self.communicator = communicator
        self.robot_name = robot_name
        super().__init__(
            tools=tools,
            tools_path=tools_path,
            model=model,
            **kwargs,
        )

    def _handle_final_answer(self, tool_arguments: str, memory_step: ActionStep) -> Union[str, None]:
        if isinstance(tool_arguments, dict):
            if "answer" in tool_arguments:
                answer = tool_arguments["answer"]
            else:
                answer = tool_arguments
        else:
            answer = tool_arguments
        if (
            isinstance(answer, str) and answer in self.state.keys()
        ):  # if the answer is a state variable, return the value
            final_answer = self.state[answer]
            self.logger.log(
                f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                level=LogLevel.INFO,
            )
        else:
            final_answer = answer
            self.logger.log(
                Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                level=LogLevel.INFO,
            )

        memory_step.action_output = final_answer
        return final_answer


    def _handle_tool_call(self, tool_name: str, tool_arguments: dict, memory_step: ActionStep) -> Union[str, None]:
        self.logger.log(
            Panel(
                Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")
            ),
            level=LogLevel.INFO,
        )
        observation = self.execute_tool_call(tool_name, tool_arguments)            
        self.tool_call.append({"tool_name": tool_name, "result": observation})
        self.logger.log(
            f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
            level=LogLevel.INFO,
        )
        memory_step.observations = str(observation).strip()
        return None

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)

        memory_messages = self.write_memory_to_messages()
        self.input_messages = memory_messages
        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()
        model_message: ChatMessage = self.model(
            memory_messages,
            tools_to_call_from=self.tools,
            stop_sequences=["Observation:"],
        )
        memory_step.model_output_message = model_message
        self.logger.log_markdown(
            content=model_message.content
            if model_message.content
            else str(model_message.raw),
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )
        if model_message.tool_calls:
            tool_call = model_message.tool_calls[0]
            tool_name, tool_call_id = tool_call.function.name, tool_call.id
            tool_arguments = tool_call.function.arguments
        else:
            
            final_answer = model_message.content
            tool_arguments = {"answer": final_answer}
            tool_name = "final_answer"
            
        if tool_name == "final_answer":
            return self._handle_final_answer(tool_arguments, memory_step)
        memory_step.tool_calls = [
            ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
        ]
        return self._handle_tool_call(tool_name, tool_arguments, memory_step)
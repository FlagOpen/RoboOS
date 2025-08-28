#!/usr/bin/env python
# coding=utf-8

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
import json
import logging
from enum import IntEnum
from typing import List, Optional, Union

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from tools.utils import escape_code_brackets

__all__ = ["AgentLogger", "LogLevel", "Monitor"]


class Monitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if (
            getattr(self.tracked_model, "last_input_token_count", "Not found")
            != "Not found"
        ):
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """Update the metrics of the monitor.

        Args:
            step_log ([`MemoryStep`]): Step log to update the monitor with.
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console_outputs = (
            f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"
        )

        if getattr(self.tracked_model, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            console_outputs += f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)


class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"


class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO, log_file: str = "agent.log"):
        """
        Initialize the logger.

        Args:
            level (LogLevel): Logging level. Defaults to LogLevel.INFO.
            log_file (str): Path to the log file. Defaults to "agent.log".
        """
        self.level = level
        self.console = Console()

        # Configure file logging
        if log_file:
            self.log_file = log_file
            self._init_file_logger(log_file)

    def _init_file_logger(self, log_file):
        """Initialize the file logger."""
        self.file_logger = logging.getLogger(log_file.split(".")[-2])
        self.file_logger.setLevel(logging.DEBUG)  # File logger captures all levels

        # Create a file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        # Set log format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.file_logger.addHandler(file_handler)

    def log(self, *args, level: Union[str, LogLevel] = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log2file(self, log_message, level: LogLevel = LogLevel.INFO) -> None:
        """Logs a message to the file logger."""
        if hasattr(self, "file_logger"):
            if level == LogLevel.DEBUG:
                self.file_logger.debug(log_message)
            elif level == LogLevel.INFO:
                self.file_logger.info(log_message)
            elif level == LogLevel.ERROR:
                self.file_logger.error(log_message)

    def log_error(self, error_message: str) -> None:
        self.log(
            escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR
        )
        self.log2file(error_message, level=LogLevel.ERROR)

    def log_markdown(
        self,
        content: str,
        title: Optional[str] = None,
        level=LogLevel.INFO,
        style=YELLOW_HEX,
    ) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
            # self.log2file("\n=====> Markdown <=====\n" + title + "\n" + content.split("'function':")[-1].split("function=")[-1] + "\n=====> Markdown <=====", level=level)
        else:
            self.log(markdown_content, level=level)

        self.log2file(
            "\n=====> Markdown <=====\n"
            + content.split("'function':")[-1].split("function=")[-1]
            + "\n=====> Markdown <=====",
            level=level,
        )

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )
        self.log2file(
            "\n=====> Code <=====\n" + title + "\n" + content + "\n=====> Code <=====",
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )
        self.log2file(
            "\n=====> Step <=====\n" + title + "\n=====> Step <=====", level=level
        )

    def log_task(
        self,
        content: str,
        subtitle: str,
        title: Optional[str] = None,
        level: int = LogLevel.INFO,
    ) -> None:
        self.log(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )
        if title and subtitle:
            self.log2file(
                "\n=====> Task <=====\n"
                + title
                + " -> "
                + subtitle
                + "\n"
                + content
                + "\n=====> Task <=====",
                level=level,
            )
        elif title:
            self.log2file(
                "\n=====> Task <=====\n"
                + title
                + "\n"
                + content
                + "\n=====> Task <=====",
                level=level,
            )
        else:
            self.log2file(
                "\n=====> Task <=====\n" + content + "\n=====> Task <=====", level=level
            )

    def log_messages(self, messages: List) -> None:
        messages_as_string = "\n".join(
            [json.dumps(dict(message), indent=4) for message in messages]
        )
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            )
        )
        self.log2file(
            "\n=====> Message <=====\n" + messages_as_string + "\n=====> Message <====="
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(
                    name, getattr(tool, "description", str(tool)), "\n".join(args)
                )

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: Optional[str] = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(
                        get_agent_headline(managed_agent, name)
                    )
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(
                        f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}"
                    )
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboOS Memory Module

This module provides a complete memory management system, including:
- Short-term memory: Structured message storage with CRUD operations and state persistence
- Long-term memory: Semantic search and multi-weighted intelligent search based on mem0 and Qdrant vector database
"""

from .base import MemoryBase, LongTermMemoryBase, MemoryMessage
from .short_term import ShortTermMemory as StructuredShortTermMemory
from .long_term import LongTermMemory
from .memory_manager import MemoryManager

# Import from old system for backward compatibility
import importlib.util
import os
_old_memory_path = os.path.join(os.path.dirname(__file__), '..', 'memory.py')
if os.path.exists(_old_memory_path):
    spec = importlib.util.spec_from_file_location('old_memory', _old_memory_path)
    old_memory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_memory_module)
    ActionStep = old_memory_module.ActionStep
    AgentMemory = old_memory_module.AgentMemory
    SceneMemory = old_memory_module.SceneMemory
    ShortTermMemory = old_memory_module.ShortTermMemory  # Use the compact version with capacity
else:
    # If old file doesn't exist, provide placeholders
    ActionStep = None
    AgentMemory = None
    SceneMemory = None
    ShortTermMemory = StructuredShortTermMemory  # Fallback to structured version

__all__ = [
    "MemoryBase",
    "LongTermMemoryBase",
    "MemoryMessage",
    "ShortTermMemory",
    "StructuredShortTermMemory",
    "LongTermMemory",
    "MemoryManager",
    "ActionStep",
    "AgentMemory",
    "SceneMemory",
]

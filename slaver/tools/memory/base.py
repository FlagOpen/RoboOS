#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Module Base Classes

Defines base interfaces for short-term and long-term memory to ensure modular design and consistency.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class MemoryMessage:
    """Memory message data structure"""
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMessage":
        """Create instance from dictionary"""
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata")
        )


class MemoryBase:
    """Memory module base class"""

    @abstractmethod
    async def add(self, message: Union[MemoryMessage, List[MemoryMessage]]) -> None:
        """Add message to memory"""
        pass

    @abstractmethod
    async def delete(self, message_id: Union[str, List[str]]) -> None:
        """Delete message from memory"""
        pass

    @abstractmethod
    async def update(self, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update message in memory"""
        pass

    @abstractmethod
    async def get(self, message_id: str) -> Optional[MemoryMessage]:
        """Get message by ID"""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryMessage]:
        """Search messages in memory"""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get memory size"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear memory"""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load memory from state dictionary"""
        pass


class LongTermMemoryBase(MemoryBase):
    """Long-term memory base class"""

    @abstractmethod
    async def record_to_memory(self, thinking: str, content: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Record important information to long-term memory"""
        pass

    @abstractmethod
    async def retrieve_from_memory(self, keywords: List[str], limit: int = 5, **kwargs: Any) -> List[str]:
        """Retrieve information from long-term memory based on keywords"""
        pass

    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 5) -> List[MemoryMessage]:
        """Semantic search"""
        pass

    @abstractmethod
    async def multi_weight_search(self, query: str, weights: Dict[str, float], limit: int = 5) -> List[MemoryMessage]:
        """Multi-weighted intelligent search"""
        pass

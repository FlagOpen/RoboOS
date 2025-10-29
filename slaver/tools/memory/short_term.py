#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short-term Memory Module Implementation

Short-term memory based on structured message storage with CRUD operations and state persistence.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict

from .base import MemoryBase, MemoryMessage


class ShortTermMemory(MemoryBase):
    """Short-term memory implementation class

    Features:
    - In-memory structured message storage
    - Support for CRUD operations
    - Support for state save and restore
    - Support for time-based sorting and search
    """

    def __init__(self, max_size: int = 1000):
        """Initialize short-term memory

        Args:
            max_size: Maximum number of messages to store, oldest messages will be deleted when exceeded
        """
        self.max_size = max_size
        self.messages: OrderedDict[str, MemoryMessage] = OrderedDict()
        self._message_index: Dict[str, List[str]] = {}  # Content index for fast search

    async def add(self, message: Union[MemoryMessage, List[MemoryMessage]]) -> None:
        """Add message to short-term memory"""
        if isinstance(message, MemoryMessage):
            messages = [message]
        else:
            messages = message

        for msg in messages:
            # Check if already exists
            if msg.id in self.messages:
                continue

            # Add to message storage
            self.messages[msg.id] = msg

            # Update index
            self._update_index(msg)

            # Check size limit
            if len(self.messages) > self.max_size:
                # Delete oldest message
                oldest_id = next(iter(self.messages))
                await self.delete(oldest_id)

    async def delete(self, message_id: Union[str, List[str]]) -> None:
        """Delete message from short-term memory"""
        if isinstance(message_id, str):
            message_ids = [message_id]
        else:
            message_ids = message_id

        for msg_id in message_ids:
            if msg_id in self.messages:
                msg = self.messages[msg_id]
                # Remove from index
                self._remove_from_index(msg)
                # Remove from message storage
                del self.messages[msg_id]

    async def update(self, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update message in memory"""
        if message_id not in self.messages:
            raise ValueError(f"Message with id {message_id} not found")

        old_msg = self.messages[message_id]

        # Remove old message from index
        self._remove_from_index(old_msg)

        # Create new message
        new_msg = MemoryMessage(
            id=message_id,
            role=old_msg.role,
            content=content,
            timestamp=old_msg.timestamp,
            metadata=metadata or old_msg.metadata
        )

        # Update message storage
        self.messages[message_id] = new_msg

        # Update index
        self._update_index(new_msg)

    async def get(self, message_id: str) -> Optional[MemoryMessage]:
        """Get message by ID"""
        return self.messages.get(message_id)

    async def search(self, query: str, limit: int = 10) -> List[MemoryMessage]:
        """Search messages in memory"""
        query_lower = query.lower()
        results = []

        # Simple text search
        for msg in self.messages.values():
            if query_lower in msg.content.lower():
                results.append(msg)

        # Sort by time (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    async def size(self) -> int:
        """Get memory size"""
        return len(self.messages)

    async def clear(self) -> None:
        """Clear memory"""
        self.messages.clear()
        self._message_index.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization"""
        return {
            "max_size": self.max_size,
            "messages": [msg.to_dict() for msg in self.messages.values()]
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load memory from state dictionary"""
        self.max_size = state_dict.get("max_size", 1000)
        self.messages.clear()
        self._message_index.clear()

        for msg_data in state_dict.get("messages", []):
            msg = MemoryMessage.from_dict(msg_data)
            self.messages[msg.id] = msg
            self._update_index(msg)

    def _update_index(self, message: MemoryMessage) -> None:
        """Update message index"""
        words = message.content.lower().split()
        for word in words:
            if word not in self._message_index:
                self._message_index[word] = []
            if message.id not in self._message_index[word]:
                self._message_index[word].append(message.id)

    def _remove_from_index(self, message: MemoryMessage) -> None:
        """Remove message from index"""
        words = message.content.lower().split()
        for word in words:
            if word in self._message_index and message.id in self._message_index[word]:
                self._message_index[word].remove(message.id)
                if not self._message_index[word]:
                    del self._message_index[word]

    async def get_recent_messages(self, limit: int = 10) -> List[MemoryMessage]:
        """Get recent messages"""
        messages = list(self.messages.values())
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        return messages[:limit]

    async def get_messages_by_role(self, role: str, limit: int = 10) -> List[MemoryMessage]:
        """Get messages by role"""
        messages = [msg for msg in self.messages.values() if msg.role == role]
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        return messages[:limit]

    async def get_messages_by_time_range(self, start_time: datetime, end_time: datetime) -> List[MemoryMessage]:
        """Get messages by time range"""
        messages = [
            msg for msg in self.messages.values()
            if start_time <= msg.timestamp <= end_time
        ]
        messages.sort(key=lambda x: x.timestamp)
        return messages

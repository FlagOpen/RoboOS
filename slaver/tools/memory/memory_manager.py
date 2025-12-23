#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Manager

Integrates short-term and long-term memory, providing a unified memory management interface.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from logging import getLogger

from .base import MemoryMessage
from .short_term import ShortTermMemory
from .long_term import LongTermMemory

logger = getLogger(__name__)


class MemoryManager:
    """Memory Manager

    Features:
    - Unified management of short-term and long-term memory
    - Automatic decision on message storage location
    - Provides unified memory operation interface
    - Support for memory synchronization and migration
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        short_term_max_size: Optional[int] = None,
        long_term_config: Optional[Dict[str, Any]] = None,
        auto_migrate_threshold: Optional[int] = None,
        migration_age_hours: Optional[int] = None
    ):
        """Initialize memory manager

        Args:
            config: Configuration dictionary from config.yaml
            short_term_max_size: Maximum size of short-term memory (overrides config)
            long_term_config: Long-term memory configuration (overrides config)
            auto_migrate_threshold: Auto-migration threshold (overrides config)
            migration_age_hours: Migration age threshold (hours) (overrides config)
        """
        # Load configuration from config.yaml or use provided parameters
        if config:
            memory_config = config.get("memory", {})
            short_term_config = memory_config.get("short_term", {})
            long_term_config_from_file = memory_config.get("long_term", {})

            # Use config values as defaults, but allow parameter overrides
            self.short_term_max_size = short_term_max_size or short_term_config.get("max_size", 1000)
            self.auto_migrate_threshold = auto_migrate_threshold or short_term_config.get("auto_migrate_threshold", 100)
            self.migration_age_hours = migration_age_hours or short_term_config.get("migration_age_hours", 24)

            # Prepare long-term memory configuration
            if long_term_config is None:
                long_term_config = {}

            # Merge file config with provided config
            merged_long_term_config = {**long_term_config_from_file, **long_term_config}

            # Set OpenAI API key from environment if not provided
            if "mem0" in merged_long_term_config and "openai_api_key" in merged_long_term_config["mem0"]:
                if merged_long_term_config["mem0"]["openai_api_key"] is None:
                    import os
                    merged_long_term_config["mem0"]["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            # Use provided parameters or defaults
            self.short_term_max_size = short_term_max_size or 1000
            self.auto_migrate_threshold = auto_migrate_threshold or 100
            self.migration_age_hours = migration_age_hours or 24
            merged_long_term_config = long_term_config or {}

        # Initialize memory components
        self.short_term_memory = ShortTermMemory(max_size=self.short_term_max_size)
        self.long_term_memory = LongTermMemory(**merged_long_term_config)

        logger.info("Memory manager initialized")

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_long_term: bool = False
    ) -> str:
        """Add message to memory system

        Args:
            role: Message role
            content: Message content
            metadata: Metadata
            force_long_term: Force storage to long-term memory

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        msg = MemoryMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )

        if force_long_term:
            await self.long_term_memory.add(msg)
            logger.debug(f"Message {message_id} added to long-term memory")
        else:
            await self.short_term_memory.add(msg)
            logger.debug(f"Message {message_id} added to short-term memory")

            # Check if automatic migration is needed
            await self._check_auto_migration()

        return message_id

    async def get_message(self, message_id: str) -> Optional[MemoryMessage]:
        """Get message by ID"""
        # First search in short-term memory
        msg = await self.short_term_memory.get(message_id)
        if msg:
            return msg

        # Then search in long-term memory
        msg = await self.long_term_memory.get(message_id)
        return msg

    async def search_messages(
        self,
        query: str,
        limit: int = 10,
        search_short_term: bool = True,
        search_long_term: bool = True
    ) -> List[MemoryMessage]:
        """Search messages"""
        results = []

        if search_short_term:
            short_results = await self.short_term_memory.search(query, limit)
            results.extend(short_results)

        if search_long_term:
            long_results = await self.long_term_memory.search(query, limit)
            results.extend(long_results)

        # Deduplicate and sort by time
        seen_ids = set()
        unique_results = []
        for msg in results:
            if msg.id not in seen_ids:
                seen_ids.add(msg.id)
                unique_results.append(msg)

        # Sort by time (newest first)
        unique_results.sort(key=lambda x: x.timestamp, reverse=True)

        return unique_results[:limit]

    async def delete_message(self, message_id: str) -> bool:
        """Delete message"""
        # Try to delete from short-term memory
        msg = await self.short_term_memory.get(message_id)
        if msg:
            await self.short_term_memory.delete(message_id)
            logger.debug(f"Message {message_id} deleted from short-term memory")
            return True

        # Try to delete from long-term memory
        msg = await self.long_term_memory.get(message_id)
        if msg:
            await self.long_term_memory.delete(message_id)
            logger.debug(f"Message {message_id} deleted from long-term memory")
            return True

        return False

    async def update_message(
        self,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update message"""
        # Try to update short-term memory
        msg = await self.short_term_memory.get(message_id)
        if msg:
            await self.short_term_memory.update(message_id, content, metadata)
            logger.debug(f"Message {message_id} updated in short-term memory")
            return True

        # Try to update long-term memory
        msg = await self.long_term_memory.get(message_id)
        if msg:
            await self.long_term_memory.update(message_id, content, metadata)
            logger.debug(f"Message {message_id} updated in long-term memory")
            return True

        return False

    async def migrate_to_long_term(self, message_ids: Optional[List[str]] = None) -> int:
        """Migrate messages to long-term memory"""
        if message_ids is None:
            # Migrate all eligible short-term memory messages
            messages = await self.short_term_memory.get_recent_messages(limit=1000)
            cutoff_time = datetime.now() - timedelta(hours=self.migration_age_hours)
            messages_to_migrate = [
                msg for msg in messages
                if msg.timestamp < cutoff_time
            ]
        else:
            # Migrate messages with specified IDs
            messages_to_migrate = []
            for msg_id in message_ids:
                msg = await self.short_term_memory.get(msg_id)
                if msg:
                    messages_to_migrate.append(msg)

        # Execute migration
        migrated_count = 0
        for msg in messages_to_migrate:
            try:
                await self.long_term_memory.add(msg)
                await self.short_term_memory.delete(msg.id)
                migrated_count += 1
                logger.debug(f"Message {msg.id} migrated to long-term memory")
            except Exception as e:
                logger.error(f"Failed to migrate message {msg.id}: {e}")

        logger.info(f"Migrated {migrated_count} messages to long-term memory")
        return migrated_count

    async def _check_auto_migration(self) -> None:
        """Check if automatic migration is needed"""
        short_term_size = await self.short_term_memory.size()

        if short_term_size >= self.auto_migrate_threshold:
            logger.info("Auto-migration triggered due to size threshold")
            await self.migrate_to_long_term()

    async def record_important_info(
        self,
        thinking: str,
        content: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Record important information to long-term memory"""
        return await self.long_term_memory.record_to_memory(thinking, content, **kwargs)

    async def retrieve_important_info(
        self,
        keywords: List[str],
        limit: int = 5,
        **kwargs: Any
    ) -> List[str]:
        """Retrieve important information from long-term memory"""
        return await self.long_term_memory.retrieve_from_memory(keywords, limit, **kwargs)

    async def semantic_search(self, query: str, limit: int = 5) -> List[MemoryMessage]:
        """Semantic search"""
        return await self.long_term_memory.semantic_search(query, limit)

    async def multi_weight_search(
        self,
        query: str,
        weights: Dict[str, float],
        limit: int = 5
    ) -> List[MemoryMessage]:
        """Multi-weighted intelligent search"""
        return await self.long_term_memory.multi_weight_search(query, weights, limit)

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        short_term_size = await self.short_term_memory.size()
        long_term_size = await self.long_term_memory.size()

        return {
            "short_term_size": short_term_size,
            "long_term_size": long_term_size,
            "total_size": short_term_size + long_term_size,
            "auto_migrate_threshold": self.auto_migrate_threshold,
            "migration_age_hours": self.migration_age_hours
        }

    async def clear_all_memory(self) -> None:
        """Clear all memory"""
        await self.short_term_memory.clear()
        await self.long_term_memory.clear()
        logger.info("All memory cleared")

    def save_state(self, filepath: str) -> None:
        """Save memory state to file"""
        state = {
            "short_term": self.short_term_memory.state_dict(),
            "long_term": self.long_term_memory.state_dict(),
            "config": {
                "auto_migrate_threshold": self.auto_migrate_threshold,
                "migration_age_hours": self.migration_age_hours
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info(f"Memory state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load memory state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.short_term_memory.load_state_dict(state["short_term"])
        self.long_term_memory.load_state_dict(state["long_term"])

        config = state.get("config", {})
        self.auto_migrate_threshold = config.get("auto_migrate_threshold", 100)
        self.migration_age_hours = config.get("migration_age_hours", 24)

        logger.info(f"Memory state loaded from {filepath}")

    async def get_recent_conversation(self, limit: int = 20) -> List[MemoryMessage]:
        """Get recent conversation records"""
        # Get recent messages from short-term memory
        short_messages = await self.short_term_memory.get_recent_messages(limit)

        # Get recent messages from long-term memory (implemented through search)
        long_messages = await self.long_term_memory.search("", limit)

        # Merge and sort
        all_messages = short_messages + long_messages
        all_messages.sort(key=lambda x: x.timestamp, reverse=True)

        return all_messages[:limit]

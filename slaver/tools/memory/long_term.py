#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-term Memory Module Implementation

Long-term memory based on mem0 and Qdrant vector database for semantic search and multi-weighted intelligent search.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import asyncio
from logging import getLogger

from .base import LongTermMemoryBase, MemoryMessage

logger = getLogger(__name__)


class LongTermMemory(LongTermMemoryBase):
    """Long-term memory implementation class

    Features:
    - Based on mem0 and Qdrant vector database
    - Support for semantic search
    - Support for multi-weighted intelligent search
    - Support for persistent storage
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        user_name: Optional[str] = None,
        run_name: Optional[str] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        mem0_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        """Initialize long-term memory

        Args:
            agent_name: Agent name
            user_name: User name
            run_name: Run name
            vector_store_config: Vector store configuration
            mem0_config: mem0 configuration
        """
        super().__init__()

        # Storage identifiers
        self.agent_id = agent_name
        self.user_id = user_name
        self.run_id = run_name

        # Initialize mem0 and Qdrant
        self._init_mem0(vector_store_config, mem0_config, **kwargs)

        # Local message cache (for fast access)
        self._local_cache: Dict[str, MemoryMessage] = {}

    def _init_mem0(self, vector_store_config: Optional[Dict[str, Any]],
                   mem0_config: Optional[Dict[str, Any]], **kwargs: Any) -> None:
        """Initialize mem0 and vector store"""
        try:
            import mem0
            from mem0.configs.base import MemoryConfig
            from mem0.vector_stores.configs import VectorStoreConfig
            import os

            # Create default configuration
            if mem0_config is None:
                mem0_config = {}

            # Set vector store configuration
            if vector_store_config is None:
                vector_store_config = {
                    "provider": "qdrant",
                    "config": {
                        "on_disk": True,
                        "collection_name": f"roboos_memory_{self.agent_id or 'default'}"
                    }
                }

            # Prepare mem0 configuration
            mem0_data = {}

            # Configure embedder
            if "embedder" in mem0_config:
                embedder_config = mem0_config["embedder"].copy()
                embedder_provider = embedder_config.get("provider", "openai")
                embedder_model_config = embedder_config.get("config", {})

                # Set API keys from environment variables if not provided
                if embedder_provider == "openai" and embedder_model_config.get("api_key") is None:
                    embedder_model_config["api_key"] = os.getenv("OPENAI_API_KEY")
                elif embedder_provider == "cohere" and embedder_model_config.get("api_key") is None:
                    embedder_model_config["api_key"] = os.getenv("COHERE_API_KEY")
                elif embedder_provider == "azure_openai" and embedder_model_config.get("api_key") is None:
                    embedder_model_config["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

                mem0_data["embedder"] = {
                    "provider": embedder_provider,
                    "config": embedder_model_config
                }

            # Configure LLM (optional)
            if "llm" in mem0_config:
                llm_config = mem0_config["llm"].copy()
                llm_provider = llm_config.get("provider", "openai")
                llm_model_config = llm_config.get("config", {})

                # Set API keys from environment variables if not provided
                if llm_provider == "openai" and llm_model_config.get("api_key") is None:
                    # For local vLLM service, use a dummy API key
                    if llm_model_config.get("base_url", "").startswith("http://localhost"):
                        llm_model_config["api_key"] = "EMPTY"
                    else:
                        llm_model_config["api_key"] = os.getenv("OPENAI_API_KEY")
                elif llm_provider == "azure_openai" and llm_model_config.get("api_key") is None:
                    llm_model_config["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

                mem0_data["llm"] = {
                    "provider": llm_provider,
                    "config": llm_model_config
                }

            # Create MemoryConfig with proper structure
            config_kwargs = {}

            # Add embedder configuration
            if "embedder" in mem0_data:
                config_kwargs["embedder"] = mem0_data["embedder"]

            # Add LLM configuration
            if "llm" in mem0_data:
                config_kwargs["llm"] = mem0_data["llm"]

            # Add vector store configuration
            config_kwargs["vector_store"] = vector_store_config

            # Create MemoryConfig
            config = MemoryConfig(**config_kwargs)

            # Initialize async memory instance
            self.long_term_memory = mem0.AsyncMemory(config)

            embedder_provider = mem0_config.get("embedder", {}).get("provider", "openai")
            logger.info(f"Long-term memory initialized with {embedder_provider} embedder and Qdrant vector store")

        except ImportError as e:
            logger.error("Failed to import mem0. Please install: pip install mem0ai")
            raise ImportError("mem0ai package is required for long-term memory") from e
        except Exception as e:
            logger.warning(f"Failed to initialize long-term memory: {e}")
            logger.warning("Long-term memory will be disabled. Check your configuration and API keys.")
            self.long_term_memory = None

    async def add(self, message: Union[MemoryMessage, List[MemoryMessage]]) -> None:
        """Add message to long-term memory"""
        if self.long_term_memory is None:
            logger.warning("Long-term memory is not initialized. Message will only be stored in local cache.")

        if isinstance(message, MemoryMessage):
            messages = [message]
        else:
            messages = message

        for msg in messages:
            # Add to local cache
            self._local_cache[msg.id] = msg

            # Add to mem0 if available
            if self.long_term_memory is not None:
                try:
                    await self.long_term_memory.add(
                        messages=[{
                            "role": msg.role,
                            "content": msg.content,
                            "name": msg.role
                        }],
                        agent_id=self.agent_id,
                        user_id=self.user_id,
                        run_id=self.run_id,
                        metadata={
                            "message_id": msg.id,
                            "timestamp": msg.timestamp.isoformat(),
                            **(msg.metadata or {})
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to add message to long-term memory: {e}")
                    # Remove from local cache
                    self._local_cache.pop(msg.id, None)
                    raise

    async def delete(self, message_id: Union[str, List[str]]) -> None:
        """Delete message from long-term memory"""
        if isinstance(message_id, str):
            message_ids = [message_id]
        else:
            message_ids = message_id

        for msg_id in message_ids:
            # Remove from local cache
            self._local_cache.pop(msg_id, None)

            # Remove from mem0 (find corresponding memory ID through search)
            try:
                # This needs to be implemented based on actual mem0 API
                # Since mem0's delete API may be different, here's a basic implementation
                logger.warning(f"Delete operation for message {msg_id} not fully implemented")
            except Exception as e:
                logger.error(f"Failed to delete message from long-term memory: {e}")

    async def update(self, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update message in long-term memory"""
        if message_id not in self._local_cache:
            raise ValueError(f"Message with id {message_id} not found")

        old_msg = self._local_cache[message_id]

        # Create new message
        new_msg = MemoryMessage(
            id=message_id,
            role=old_msg.role,
            content=content,
            timestamp=old_msg.timestamp,
            metadata=metadata or old_msg.metadata
        )

        # Update local cache
        self._local_cache[message_id] = new_msg

        # Update mem0 (delete old record, add new record)
        try:
            await self.delete(message_id)
            await self.add(new_msg)
        except Exception as e:
            logger.error(f"Failed to update message in long-term memory: {e}")
            raise

    async def get(self, message_id: str) -> Optional[MemoryMessage]:
        """Get message by ID"""
        return self._local_cache.get(message_id)

    async def search(self, query: str, limit: int = 10) -> List[MemoryMessage]:
        """Search messages in long-term memory"""
        if self.long_term_memory is None:
            logger.warning("Long-term memory is not initialized. Returning empty search results.")
            return []

        try:
            # Use mem0 for search
            results = await self.long_term_memory.search(
                query=query,
                agent_id=self.agent_id,
                user_id=self.user_id,
                run_id=self.run_id,
                limit=limit
            )

            # Convert to MemoryMessage objects
            messages = []
            if results and "results" in results:
                for item in results["results"]:
                    memory_data = item.get("memory", "")
                    metadata = item.get("metadata", {})

                    # Get message ID from metadata
                    msg_id = metadata.get("message_id", str(uuid.uuid4()))

                    # Create MemoryMessage
                    msg = MemoryMessage(
                        id=msg_id,
                        role=metadata.get("role", "assistant"),
                        content=memory_data,
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        metadata=metadata
                    )
                    messages.append(msg)

            return messages

        except Exception as e:
            logger.error(f"Failed to search long-term memory: {e}")
            return []

    async def size(self) -> int:
        """Get long-term memory size"""
        return len(self._local_cache)

    async def clear(self) -> None:
        """Clear long-term memory"""
        self._local_cache.clear()
        # Note: mem0's clear operation may require special handling
        logger.warning("Clear operation for mem0 not fully implemented")

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "local_cache": {k: v.to_dict() for k, v in self._local_cache.items()}
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load long-term memory from state dictionary"""
        self.agent_id = state_dict.get("agent_id")
        self.user_id = state_dict.get("user_id")
        self.run_id = state_dict.get("run_id")

        self._local_cache.clear()
        for msg_id, msg_data in state_dict.get("local_cache", {}).items():
            self._local_cache[msg_id] = MemoryMessage.from_dict(msg_data)

    async def record_to_memory(self, thinking: str, content: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Record important information to long-term memory"""
        try:
            # Merge thinking process and content
            full_content = [thinking] + content if thinking else content

            # Create message
            msg = MemoryMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content="\n".join(full_content),
                timestamp=datetime.now(),
                metadata=kwargs
            )

            # Add to long-term memory
            await self.add(msg)

            return {
                "success": True,
                "message_id": msg.id,
                "content": msg.content
            }

        except Exception as e:
            logger.error(f"Failed to record to memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def retrieve_from_memory(self, keywords: List[str], limit: int = 5, **kwargs: Any) -> List[str]:
        """Retrieve information from long-term memory based on keywords"""
        try:
            results = []
            for keyword in keywords:
                messages = await self.search(keyword, limit)
                for msg in messages:
                    results.append(msg.content)

            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve from memory: {e}")
            return []

    async def semantic_search(self, query: str, limit: int = 5) -> List[MemoryMessage]:
        """Semantic search"""
        return await self.search(query, limit)

    async def multi_weight_search(self, query: str, weights: Dict[str, float], limit: int = 5) -> List[MemoryMessage]:
        """Multi-weighted intelligent search

        Args:
            query: Search query
            weights: Weight configuration, e.g., {"content": 0.7, "metadata": 0.3}
            limit: Limit on number of results to return
        """
        try:
            # More complex multi-weighted search logic can be implemented here
            # For now, use basic semantic search
            results = await self.semantic_search(query, limit)

            # Adjust result sorting based on weights
            # More complex weight calculations can be implemented based on actual needs
            return results

        except Exception as e:
            logger.error(f"Failed to perform multi-weight search: {e}")
            return []

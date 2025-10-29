"""
Long-Term Memory - Lightweight experience memory system based on Redis
Uses Redis directly for storing and retrieving task experiences without external APIs
"""

import json
import time
from typing import List, Dict, Optional, Any
import redis
from collections import defaultdict


class LongTermMemory:
    """Lightweight long-term memory system based on Redis

    Uses Redis directly for storing task experiences:
    1. Store task execution episodes (TaskContext)
    2. Retrieve historical tasks based on keyword matching
    3. Provide execution plan suggestions for new tasks
    4. Learn from failure cases

    Redis structure:
    - Hash: task_episodes:{task_id} - Store single task details
    - Sorted Set: task_episodes_index - Time-ordered task ID index
    - Set: task_episodes_success - Successful task ID set
    - Set: task_episodes_failed - Failed task ID set
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize long-term memory system

        Args:
            redis_host: Redis host address
            redis_port: Redis port
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.prefix = "task_episodes"

    def store_task_episode(self, task_context: 'TaskContext') -> str:
        """Store task execution episode to long-term memory (Redis)

        Args:
            task_context: Task context from ShortTermMemory

        Returns:
            Task ID
        """
        task_id = task_context.task_id

        episode_data = {
            "task_id": task_id,
            "task_text": task_context.task_text,
            "success": str(task_context.success),
            "tool_sequence": ",".join(task_context.get_tool_sequence()),
            "duration": str((task_context.end_time - task_context.start_time) if task_context.end_time else 0),
            "timestamp": str(task_context.start_time),
            "num_steps": str(len(task_context.actions)),
            "actions": json.dumps([action.to_dict() for action in task_context.actions])
        }

        self.redis_client.hset(f"{self.prefix}:{task_id}", mapping=episode_data)
        self.redis_client.zadd(f"{self.prefix}_index", {task_id: task_context.start_time})

        if task_context.success:
            self.redis_client.sadd(f"{self.prefix}_success", task_id)
        else:
            self.redis_client.sadd(f"{self.prefix}_failed", task_id)

        return task_id

    def search_similar_tasks(self, query: str, limit: int = 3,
                            filter_success: bool = True) -> List[Dict[str, Any]]:
        """Search similar historical tasks based on keyword matching

        Queries both task_episodes and observations to ensure finding all historical experiences

        Args:
            query: Query text (usually task description)
            limit: Number of results to return
            filter_success: Whether to return only successful cases

        Returns:
            List of similar tasks, sorted by similarity
        """
        query_words = set(query.lower().split())
        results = []

        if filter_success:
            task_ids = self.redis_client.smembers(f"{self.prefix}_success")
        else:
            task_ids = self.redis_client.zrevrange(f"{self.prefix}_index", 0, -1)

        for task_id in task_ids:
            episode = self.redis_client.hgetall(f"{self.prefix}:{task_id}")
            if not episode:
                continue

            task_text = episode.get("task_text", "").lower()
            task_words = set(task_text.split())

            overlap = len(query_words & task_words)
            if overlap > 0:
                results.append({
                    "task_id": task_id,
                    "metadata": {
                        "task_id": episode.get("task_id"),
                        "task_text": episode.get("task_text"),
                        "success": episode.get("success") == "True",
                        "tool_sequence": episode.get("tool_sequence", ""),
                        "duration": float(episode.get("duration", 0)),
                        "timestamp": float(episode.get("timestamp", 0)),
                        "num_steps": int(episode.get("num_steps", 0))
                    },
                    "score": overlap / max(len(query_words), 1)
                })

        if len(results) < limit:
            if filter_success:
                obs_ids = self.redis_client.smembers(f"{self.prefix}:observations_success")
            else:
                obs_ids = self.redis_client.zrevrange(f"{self.prefix}:observations_index", 0, -1)

            max_obs_to_check = 100
            obs_ids_to_check = list(obs_ids)[:max_obs_to_check]

            for obs_id in obs_ids_to_check:
                obs_data = self.redis_client.hgetall(f"{self.prefix}:observations:{obs_id}")
                if not obs_data:
                    continue

                subtask_full = obs_data.get("subtask", "")

                if ":" in subtask_full:
                    _, subtask_desc = subtask_full.split(":", 1)
                else:
                    subtask_desc = subtask_full

                execution_result = obs_data.get("execution_result", "")
                match_text = (subtask_desc + " " + execution_result).lower()
                match_words = set(match_text.split())

                overlap = len(query_words & match_words)
                if overlap > 0:
                    if ":" in subtask_full:
                        task_id_from_subtask = subtask_full.split(":")[0]
                    else:
                        task_id_from_subtask = obs_id

                    if not any(r["task_id"] == task_id_from_subtask for r in results):
                        results.append({
                            "task_id": task_id_from_subtask,
                            "metadata": {
                                "task_id": task_id_from_subtask,
                                "task_text": subtask_desc.strip(),
                                "success": obs_data.get("success") == "True",
                                "tool_sequence": "",
                                "duration": 0.0,
                                "timestamp": float(obs_data.get("timestamp", 0)),
                                "num_steps": 1
                            },
                            "score": overlap / max(len(query_words), 1)
                        })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def get_task_background_suggestion(self, task_text: str, similarity_threshold: float = 0.6) -> Optional[str]:
        """Provide task background description based on historical experience

        Args:
            task_text: Current task description
            similarity_threshold: Similarity threshold, below which no suggestion is returned

        Returns:
            Historical task background description, or None if no similar history or similarity too low
        """
        similar_tasks = self.search_similar_tasks(task_text, limit=3, filter_success=True)

        if not similar_tasks:
            return None

        best_match = similar_tasks[0]
        similarity_score = best_match.get("score", 0.0)

        if similarity_score < similarity_threshold:
            return None

        task_description = best_match["metadata"].get("task_text", "")
        success = best_match["metadata"].get("success", False)

        if not task_description:
            return None

        background = f"Previously executed similar task: '{task_description}'"
        if success:
            background += " (successful)"
        else:
            background += " (failed)"

        return background

    def learn_from_failure(self, task_context: 'TaskContext'):
        """Learn from failure cases

        Failed cases are already stored in the failed set via store_task_episode.
        This method performs additional failure analysis and marking.

        Args:
            task_context: Failed task context
        """
        if task_context.success:
            return

        failure_info = {
            "failure_type": self._classify_failure(task_context),
            "failure_points": self._identify_failure_points(task_context)
        }

        self.redis_client.hset(
            f"{self.prefix}:{task_context.task_id}",
            "failure_info",
            json.dumps(failure_info)
        )

    def _classify_failure(self, task_context: 'TaskContext') -> str:
        """Classify failure type"""
        if len(task_context.actions) == 0:
            return "no_action"

        failure_count = sum(1 for a in task_context.actions if not a.success)

        if failure_count == 0:
            return "incomplete"
        elif failure_count == len(task_context.actions):
            return "all_failed"
        else:
            return "partial_failed"

    def _identify_failure_points(self, task_context: 'TaskContext') -> str:
        """Identify failure points

        Args:
            task_context: Task context

        Returns:
            Failure point description
        """
        failures = []
        for action in task_context.actions:
            if not action.success:
                failures.append(f"Step {action.step_number}: {action.tool_name}")

        return "; ".join(failures) if failures else "Unknown"

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics from Redis

        Returns:
            Statistics dictionary containing total, success count, failure count, and success rate
        """
        try:
            total = self.redis_client.zcard(f"{self.prefix}_index")
            successes = self.redis_client.scard(f"{self.prefix}_success")
            failures = self.redis_client.scard(f"{self.prefix}_failed")

            return {
                "total_episodes": total,
                "successful": successes,
                "failed": failures,
                "success_rate": successes / total if total > 0 else 0.0
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_episodes": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0
            }

    def get_recent_episodes(self, limit: int = 10, filter_success: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get recent task episodes

        Args:
            limit: Number of results to return
            filter_success: None=all, True=only success, False=only failed

        Returns:
            Task list, sorted by time descending
        """
        task_ids = self.redis_client.zrevrange(f"{self.prefix}_index", 0, limit - 1)

        episodes = []
        for task_id in task_ids:
            episode = self.redis_client.hgetall(f"{self.prefix}:{task_id}")
            if not episode:
                continue

            success = episode.get("success") == "True"

            if filter_success is not None and success != filter_success:
                continue

            episodes.append({
                "task_id": episode.get("task_id"),
                "task_text": episode.get("task_text"),
                "success": success,
                "tool_sequence": episode.get("tool_sequence"),
                "duration": float(episode.get("duration", 0)),
                "timestamp": float(episode.get("timestamp", 0))
            })

        return episodes

    def store_observation(self, observation_data: Dict[str, Any]) -> str:
        """Store subtask execution observation to long-term memory

        Args:
            observation_data: Data containing subtask, execution result, success status, timestamp, execution agent

        Returns:
            Observation ID
        """
        try:
            observation_id = f"obs_{int(time.time() * 1000)}"
            observation_record = {
                "observation_id": observation_id,
                "subtask": str(observation_data.get("subtask", "")),
                "execution_result": str(observation_data.get("execution_result", "")),
                "success": str(observation_data.get("success", False)),
                "timestamp": str(observation_data.get("timestamp", time.time())),
                "execution_agent": str(observation_data.get("execution_agent", "unknown"))
            }

            key = f"{self.prefix}:observations:{observation_id}"
            self.redis_client.hset(key, mapping=observation_record)

            self.redis_client.zadd(
                f"{self.prefix}:observations_index",
                {observation_id: observation_record["timestamp"]}
            )

            if observation_record["success"]:
                self.redis_client.sadd(f"{self.prefix}:observations_success", observation_id)
            else:
                self.redis_client.sadd(f"{self.prefix}:observations_failed", observation_id)

            return observation_id

        except Exception as e:
            return ""

    def get_recent_observations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent observations

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent observations
        """
        try:
            observation_ids = self.redis_client.zrevrange(
                f"{self.prefix}:observations_index",
                0, limit - 1
            )

            observations = []
            for obs_id in observation_ids:
                key = f"{self.prefix}:observations:{obs_id}"
                observation = self.redis_client.hgetall(key)
                if observation:
                    observations.append({
                        "observation_id": observation.get("observation_id"),
                        "subtask": observation.get("subtask"),
                        "execution_result": observation.get("execution_result"),
                        "success": observation.get("success") == "True",
                        "timestamp": float(observation.get("timestamp", 0)),
                        "execution_agent": observation.get("execution_agent")
                    })

            return observations

        except Exception:
            return []

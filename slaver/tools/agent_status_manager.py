"""
Agent Status Manager - Independent Redis status manager
Each agent is fully isolated by robot_name with a 30-second time window, keeping only the latest entries
"""

import json
import time
from typing import List
import redis


class AgentStatusManager:
    """Independent agent status manager with 30-second window and latest entry

    Important: Each agent's status is fully isolated by robot_name
    Redis key format: agent_status:{robot_name}
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize Redis connection

        Args:
            redis_host: Redis server address
            redis_port: Redis port
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.time_window = 30

    def _get_key(self, robot_name: str) -> str:
        """Generate agent-specific Redis key to ensure agent state isolation

        Args:
            robot_name: Robot name (e.g., "grx_robot")

        Returns:
            Redis key string
        """
        return f"agent_status:{robot_name}"

    def record_status(self, robot_name: str, status: str):
        """Record agent status with timestamp

        Args:
            robot_name: Robot name
            status: Status string (e.g., "Successfully navigated to kitchenTable")
                   If empty string, clears all status for this agent
        """
        if not status:
            self.redis_client.delete(self._get_key(robot_name))
            return

        status_entry = {
            "status": status,
            "timestamp": time.time(),
            "robot_name": robot_name
        }

        key = self._get_key(robot_name)
        self.redis_client.lpush(key, json.dumps(status_entry))
        self.redis_client.ltrim(key, 0, 4)

    def read_latest_status(self, robot_name: str) -> List[str]:
        """Read latest status (recent 3 within 30-second window)

        Returns only statuses belonging to current robot_name to ensure agent isolation

        Args:
            robot_name: Robot name

        Returns:
            List containing 0-3 status strings
            - Returns up to 3 recent statuses if within 30 seconds: ["status1", "status2", "status3"]
            - Returns empty list if none or timeout: []
        """
        key = self._get_key(robot_name)
        entries = self.redis_client.lrange(key, 0, -1)

        if not entries:
            return []

        current_time = time.time()
        valid_entries = []

        for entry_json in entries:
            try:
                entry = json.loads(entry_json)

                if entry.get("robot_name") != robot_name:
                    continue

                if current_time - entry["timestamp"] <= self.time_window:
                    valid_entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue

        valid_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_entries = valid_entries[:3] if len(valid_entries) >= 3 else valid_entries

        return [entry["status"] for entry in recent_entries]

    def clear_status(self, robot_name: str):
        """Clear status for specified agent

        Args:
            robot_name: Robot name
        """
        self.redis_client.delete(self._get_key(robot_name))

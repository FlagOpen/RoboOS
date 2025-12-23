# RoboOS Memory System

A comprehensive memory management system for RoboOS that provides both short-term and long-term memory capabilities for intelligent task planning and execution.

## Overview

The RoboOS Memory System consists of multiple memory components that work together to store, retrieve, and utilize historical task experiences to improve future task planning and execution.

## Architecture

```
Memory System
├── Short-Term Memory (STM)
│   ├── Agent Status Manager
│   └── Scene Memory
├── Long-Term Memory (LTM)
│   ├── Task Episodes Storage
│   ├── Similarity Search
│   └── Historical Experience Retrieval
└── Memory Manager
    ├── Message Management
    ├── Memory Migration
    └── Unified Interface
```

## Components

### 1. Short-Term Memory (`short_term.py`)

**Purpose**: Manages recent task context and agent status within a limited time window.

**Key Features**:
- Capacity-limited circular buffer (default: 20 items)
- Task context tracking
- Agent status monitoring
- Automatic cleanup of old entries

**Usage**:
```python
from tools.memory import ShortTermMemory

# Initialize with custom capacity
memory = ShortTermMemory(capacity=20)

# Start a new task
memory.start_task("task_123", "Navigate to kitchen")

# Add observations
memory.add_observation("Current location: living room")
memory.add_observation("Navigation started")

# Get current context
context = memory.current_context
```

### 2. Long-Term Memory (`long_term.py`)

**Purpose**: Stores and retrieves historical task experiences using Redis for persistent storage.

**Key Features**:
- Redis-based persistent storage
- Semantic similarity search
- Task episode management
- Success/failure filtering
- Configurable similarity thresholds

**Usage**:
```python
from tools.long_term_memory import LongTermMemory

# Initialize with Redis connection
memory = LongTermMemory(redis_host='127.0.0.1', redis_port=6379)

# Store a task episode
memory.store_task_episode(task_context)

# Search for similar tasks
similar_tasks = memory.search_similar_tasks(
    query="Navigate to kitchen",
    limit=5,
    filter_success=True
)
```

### 3. Agent Status Manager (`agent_status_manager.py`)

**Purpose**: Tracks individual agent status with 30-second time windows and latest entry retention.

**Key Features**:
- Robot-specific status isolation
- 30-second sliding window
- Latest status retention
- Redis-based storage

**Usage**:
```python
from tools.agent_status_manager import AgentStatusManager

# Initialize
status_manager = AgentStatusManager(redis_host='127.0.0.1', redis_port=6379)

# Record agent status
status_manager.record_status("robot_1", "Navigating to kitchen")

# Read latest status
latest_status = status_manager.read_latest_status("robot_1")
```

### 4. Memory Manager (`memory_manager.py`)

**Purpose**: Provides a unified interface for managing both short-term and long-term memory.

**Key Features**:
- Unified memory operations
- Automatic message migration
- Memory lifecycle management
- Error handling and logging

**Usage**:
```python
from tools.memory_manager import MemoryManager

# Initialize with configuration
memory_manager = MemoryManager(
    short_term_capacity=20,
    redis_host='127.0.0.1',
    redis_port=6379
)

# Add message to appropriate memory
memory_manager.add_message(message)

# Retrieve messages
messages = memory_manager.get_messages(limit=10)
```

### 5. Base Classes (`base.py`)

**Purpose**: Defines abstract base classes and data structures for the memory system.

**Key Components**:
- `MemoryBase`: Abstract base class for memory implementations
- `LongTermMemoryBase`: Abstract base class for long-term memory
- `MemoryMessage`: Data structure for memory messages
- `TaskContext`: Data structure for task episodes

## Configuration

### Master Configuration (`master/config.yaml`)

```yaml
long_term_memory:
  enabled: false  # Set to true to enable long-term memory
  redis_host: "127.0.0.1"
  redis_port: 6379
  similarity_threshold: 0.6
  max_historical_tasks: 3
  filter_success_only: true
```

### Slaver Configuration (`slaver/config.yaml`)

```yaml
optimization:
  enabled: true  # Memory optimization enabled by default
  short_term_capacity: 20
  long_term_memory:
    enabled: true  # Long-term memory enabled by default
    redis_host: "127.0.0.1"
    redis_port: 6379
```

## Integration

### Master Agent Integration

The Master Agent uses long-term memory to:
1. Store task decomposition results
2. Query historical experiences
3. Enhance planning prompts with relevant past experiences

```python
# In master/agents/planner.py
historical_experiences = self._format_historical_experiences(task)
if historical_experiences:
    formatted_scene_info = formatted_scene_info + "\n\n" + historical_experiences
```

### Slaver Agent Integration

The Slaver Agent uses memory to:
1. Track task execution context
2. Store successful task episodes
3. Learn from past experiences

```python
# In slaver/agents/slaver_agent.py
if self.optimization_enabled:
    self.short_term_memory = ShortTermMemory(capacity=20)
    self.long_term_memory = LongTermMemory(redis_host, redis_port)
```

## Data Flow

1. **Task Planning**: Master queries long-term memory for similar historical tasks
2. **Task Execution**: Slaver tracks execution context in short-term memory
3. **Task Completion**: Successful tasks are stored in long-term memory
4. **Future Planning**: Historical experiences are retrieved and integrated into planning prompts

## Redis Schema

### Task Episodes
- **Key**: `task_episodes:{task_id}`
- **Type**: Hash
- **Fields**:
  - `task_text`: Original task description
  - `success`: Boolean success status
  - `tool_sequence`: List of tools used
  - `duration`: Execution time in seconds
  - `timestamp`: Creation timestamp

### Task Index
- **Key**: `task_episodes_index`
- **Type**: Sorted Set
- **Score**: Timestamp
- **Member**: Task ID

### Agent Status
- **Key**: `agent_status:{robot_name}`
- **Type**: List
- **Content**: Recent status messages (30-second window)

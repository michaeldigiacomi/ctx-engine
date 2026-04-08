# Two-Tier Memory System

## Overview

The two-tier memory system provides separate storage optimized for different access patterns:

- **Working Memory**: Session-scoped, fast access, no embeddings
- **Reference Memory**: Long-term, semantic search with embeddings

## Use Cases

### Local Agent (Fast, Conversational)

```python
manager = MemoryManager(model_type="local-8k")

# Quick session context
manager.working.set_session_context("current_topic", "auth")

# Check for ready tasks
tasks = manager.working.get_tasks(status="ready")
for task in tasks:
    process_task(task)
```

### Remote Agent (Powerful, Batch)

```python
manager = MemoryManager(model_type="claude-opus")

# Rich context with cross-project learning
context = manager.get_context(
    "authentication patterns",
    max_tokens=12000,
    include_namespaces=["*"]
)

# Execute complex task
result = process_with_context(context)
manager.working.update_task(task_id, status="done", result=result)
```

## Token Budgets

Default budgets by model type:

| Model Type | Default Budget |
|------------|----------------|
| local-8k | 4000 |
| local-32k | 8000 |
| claude-haiku | 6000 |
| claude-sonnet | 8000 |
| claude-opus | 12000 |
| gpt-4o | 8000 |

## API Reference

### MemoryManager

```python
class MemoryManager:
    def __init__(self, config=None, model_type="claude-sonnet")
    def get_context(query, max_tokens=None, include_namespaces=None) -> str
    def remember(content, tier="reference", **kwargs) -> str
    def save_task(description, **kwargs) -> str
    def get_ready_tasks() -> List[Dict]
    def update_task(task_id, **kwargs) -> bool
    def close()
```

### WorkingMemory

```python
class WorkingMemory:
    def set_session_context(key, value, priority=5, ttl_minutes=60)
    def get_session_context() -> Dict[str, str]
    def save_task(description, **kwargs) -> str
    def get_tasks(status=None, limit=50) -> List[Dict]
    def update_task(task_id, **kwargs) -> bool
    def save_decision(content, context=None, category="decision", ttl_minutes=480) -> int
    def get_recent_decisions(limit=10) -> List[Dict]
    def cleanup_expired() -> int
    def close()
```

## CLI Commands

```bash
# Working memory
ctx-engine working set key value --priority 8 --ttl 120
ctx-engine working get
ctx-engine working tasks --status ready
ctx-engine working add-task "Task description" --priority 8
```

## Database Schema

### working.session_context

| Column | Type | Description |
|--------|------|-------------|
| key | TEXT PK | Context key |
| value | TEXT | Context value |
| priority | INTEGER | 1-10, higher = keep longer |
| ttl_minutes | INTEGER | Expiration time |
| last_accessed | TIMESTAMP | Last access time |
| created_at | TIMESTAMP | Creation time |

### working.tasks

| Column | Type | Description |
|--------|------|-------------|
| task_id | TEXT PK | Unique task ID |
| description | TEXT | Task description |
| plan | JSONB | Array of steps |
| status | TEXT | planning, ready, executing, done, error |
| assigned_to | TEXT | Agent identifier |
| priority | INTEGER | Task priority |
| result | JSONB | Task output |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update |

### working.recent_decisions

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Decision ID |
| content | TEXT | Decision text |
| category | TEXT | Decision category |
| context | TEXT | Context that prompted decision |
| ttl_minutes | INTEGER | Expiration time |
| created_at | TIMESTAMP | Creation time |

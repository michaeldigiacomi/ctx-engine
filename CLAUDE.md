# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PGVector Context Engine is a Python library providing semantic memory/storage using PostgreSQL + pgvector. It enables AI agents to store and retrieve context via vector embeddings with namespace isolation for multi-project use.

## Development Commands

### Installation
```bash
pip install -e .           # Basic install
pip install -e ".[dev]"    # With test dependencies
pip install -e ".[openai]" # With OpenAI embedding support
```

### Testing
```bash
python run_tests.py              # Run all tests
python run_tests.py --unit       # Unit tests only (no DB required)
python run_tests.py --integration  # Integration tests (requires PostgreSQL)
python run_tests.py --coverage   # With coverage report

# Or with pytest directly:
pytest tests/test_unit.py tests/test_cli.py -v
pytest tests/test_integration.py -v -m integration
```

### CLI Usage
```bash
ctx-engine init                    # Initialize database schema
ctx-engine save "content"          # Save a memory
ctx-engine search "query"          # Semantic search
ctx-engine get-context "query"     # Get token-budgeted context
ctx-engine list                    # List memories
```

## Architecture

### Core Components

```
src/context_engine/
├── __init__.py      # Public API exports
├── core.py          # ContextEngine - main class for memory operations
├── config.py        # ContextEngineConfig - env/file-based configuration
├── providers.py     # Embedding providers (Ollama, OpenAI)
├── schema.py        # SchemaManager - database schema/migrations
├── cli.py           # CLI tool entry point
└── agent.py         # ContextAgent - base class for AI agents
```

### Data Flow

```
Text Input → Embedding (Ollama/OpenAI) → 768-dim vector → pgvector similarity search → Token-budget filter → Formatted context
```

### Key Classes

**ContextEngine** (core.py)
- Main entry point for all memory operations
- Lazy database connection initialization
- Methods: `save()`, `get_context()`, `search()`, `list()`, `delete()`, `save_conversation()`
- Uses `doc_id` (SHA256 hash) for idempotency

**ContextEngineConfig** (config.py)
- Dataclass with env var fallbacks (CTX_* prefix)
- Loads from `~/.config/context_engine/config.json`
- Properties: `conn_string` builds PostgreSQL connection URL

**ContextAgent** (agent.py)
- Abstract base class for AI agents
- Implements `process()` method
- Built-in methods: `remember()`, `recall()`, `remember_interaction()`, `get_relevant_context()`

### Configuration Hierarchy

1. Constructor arguments (highest priority)
2. Environment variables (CTX_DB_HOST, CTX_DB_USER, etc.)
3. Config file at `~/.config/context_engine/config.json`
4. Defaults (localhost:5432, namespace="default")

### Database Schema

**memories table** (migrations/001_initial.sql):
- `doc_id` - SHA256 hash of content (unique constraint)
- `embedding` - pgvector vector(768)
- `namespace` - Project isolation key
- `category`, `importance`, `ttl_days`, `session_key` - Organization/weighting
- Access tracking: `access_count`, `last_accessed`

### Testing Strategy

- **Unit tests** (test_unit.py): Mock database and embedding provider, test business logic
- **CLI tests** (test_cli.py): Mock stdout/stderr, test argument parsing
- **Integration tests** (test_integration.py): Require PostgreSQL with pgvector extension
- Fixtures in conftest.py: `mock_embedding`, `test_config`, `postgres_available`, `ollama_available`

## Important Patterns

### Idempotent Saves
Saves use `ON CONFLICT (doc_id) DO UPDATE` where `doc_id` is derived from content hash. Saving identical content updates the existing record.

### Lazy Initialization
Database connection and schema initialization are deferred until first operation. Use `auto_init=False` to disable automatic schema creation.

### Namespace Isolation
Memories are scoped to namespaces (default: "default"). Different namespaces cannot see each other's data. Use this for multi-project or multi-agent isolation.

### Token Budget Context
`get_context()` respects `max_tokens` by approximating 4 chars/token and truncating results while prioritizing by similarity score (>0.5 threshold).

### Context Manager Support
Both `ContextEngine` and `ContextAgent` support `with` statements for automatic cleanup.

## Common Tasks

### Run a single test
```bash
pytest tests/test_unit.py::test_save_basic -v
```

### Test with real database
Set environment variables:
```bash
export CTX_DB_HOST=localhost
export CTX_DB_USER=your_user
export CTX_DB_PASS=your_password
pytest tests/test_integration.py -v
```

### Check database connection
```python
from context_engine.schema import SchemaManager
from context_engine.config import ContextEngineConfig

config = ContextEngineConfig()
schema = SchemaManager(config)
success, error = schema.verify_connection()
```

# PGVector Context Engine

A reusable semantic memory/context engine using PostgreSQL + pgvector. Store and retrieve relevant context across projects using vector embeddings.

## Features

- **Semantic Search** - Query memories by meaning, not keywords
- **Project Isolation** - Namespace-based separation for multi-project use
- **Externalized Config** - All credentials via environment variables or config file
- **Pluggable Embeddings** - Ollama (default) or OpenAI
- **TTL & Importance** - Temporary memories and priority scoring
- **Category Organization** - Filter by source/category
- **Agent Integration** - Built-in base class for AI agents

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

**Option A: Interactive setup**
```bash
./samples/setup_context_engine.sh
```

**Option B: Environment variables**
```bash
export CTX_DB_HOST=localhost
export CTX_DB_PORT=5432
export CTX_DB_NAME=context_engine
export CTX_DB_USER=your_user
export CTX_DB_PASS=your_password
export CTX_NAMESPACE=my-project
export CTX_OLLAMA_URL=http://localhost:11434
```

**Option C: Config file** (`~/.config/context_engine/config.json`)
```json
{
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "context_engine",
  "db_user": "your_user",
  "ollama_url": "http://localhost:11434",
  "embedding_model": "nomic-embed-text",
  "namespace": "my-project"
}
```

### 3. Initialize Database

```bash
ctx-engine init
```

### 4. Use It

```bash
# Save memories
ctx-engine save "Deployed to k8s cluster" --category infrastructure --importance 8
ctx-engine save "User prefers terse responses" --category preference

# Search
ctx-engine search "What was I working on?"

# Get context for a task (token-budget aware)
ctx-engine get-context "Current session initialization" --max-tokens 3000

# List
ctx-engine list --category infrastructure
```

## Python API

```python
from context_engine import ContextEngine

# Initialize (reads from env/config)
ctx = ContextEngine()

# Save a memory
ctx.save(
    content="Deployed WebMonsters to k3s",
    category="infrastructure",
    importance=8.0,
    ttl_days=30
)

# Get relevant context
context = ctx.get_context(
    query="What was I working on?",
    max_memories=10,
    max_tokens=4000
)

# Search
results = ctx.search("k8s deployment", limit=5, min_similarity=0.6)
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content']}")

# Cleanup
ctx.cleanup_expired()
ctx.close()
```

## Agent Integration

For AI agents, use the built-in `ContextAgent` base class:

```python
from context_engine.agent import ContextAgent

class MyAgent(ContextAgent):
    def process(self, message):
        # Context automatically retrieved
        context = self.get_relevant_context(message)
        
        # Your LLM call here
        response = call_your_llm(context, message)
        
        # Auto-saved to memory
        self.remember_interaction(message, response)
        return response

# Run immediately (zero config if context engine is set up)
agent = MyAgent("MyBot")
agent.run()
```

See [AGENT_SETUP.md](AGENT_SETUP.md) for quick agent setup (when context engine is already configured).
See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for detailed integration patterns.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CTX_DB_HOST` | `localhost` | PostgreSQL host |
| `CTX_DB_PORT` | `5432` | PostgreSQL port |
| `CTX_DB_NAME` | `context_engine` | Database name |
| `CTX_DB_USER` | (none) | Database user |
| `CTX_DB_PASS` | (none) | Database password |
| `CTX_DB_SSLMODE` | `disable` | SSL mode |
| `CTX_OLLAMA_URL` | `http://localhost:11434` | Ollama URL |
| `CTX_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CTX_NAMESPACE` | `default` | Project namespace |

## Namespace Isolation

Namespaces keep memories separate per project:

```bash
# Project A
CTX_NAMESPACE=project-a ctx-engine save "Working on auth"

# Project B
CTX_NAMESPACE=project-b ctx-engine save "Refactoring API"

# Search only returns memories from current namespace
CTX_NAMESPACE=project-a ctx-engine search "auth"  # Returns project-a memory
CTX_NAMESPACE=project-b ctx-engine search "auth"  # Returns project-b memory
```

## Database Setup

### PostgreSQL + pgvector

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-14-pgvector

# Enable extension (as superuser)
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Create Database

```sql
CREATE DATABASE context_engine;
CREATE USER ctx_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE context_engine TO ctx_user;
\c context_engine
-- Run migrations/001_initial.sql
```

## Architecture

```
Query Text
    ↓
[Embedding] → 768-dim vector (Ollama: nomic-embed-text)
    ↓
[pgvector Search] → Top-K similar memories (by namespace)
    ↓
[Token Budget Filter] → Context within limit
    ↓
LLM Context
```

## File Structure

```
src/context_engine/
├── __init__.py      # Public API
├── config.py        # Configuration (env + config file)
├── providers.py     # Embedding providers (Ollama, OpenAI)
├── schema.py        # Database schema management
├── core.py          # Main ContextEngine class
└── cli.py           # CLI tool
```

## Examples

See `examples/claude_integration.py` for Claude Code integration patterns:

- Per-project context initialization
- Saving architectural decisions
- User preference memory
- Session-based conversation storage

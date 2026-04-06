# Agent Setup (Context Engine Already Configured)

If the context engine is already set up (PostgreSQL + pgvector running, database created), agents can start using it immediately with zero additional configuration.

## For Agent Developers

### Option 1: Zero-Config (Reads from existing config)

```python
from context_engine import ContextEngine

# Automatically loads config from ~/.config/context_engine/config.json
# or from environment variables
memory = ContextEngine()

# Start using immediately
memory.save("User prefers dark mode", category="preference")
context = memory.get_context("What are user preferences?")
```

### Option 2: Explicit Namespace (Recommended for multi-agent)

```python
from context_engine import ContextEngine
from context_engine.config import ContextEngineConfig

# Use existing DB config but isolate this agent's memories
config = ContextEngineConfig(namespace="customer-support-bot")
memory = ContextEngine(config=config)
```

### Option 3: Full Agent Class

```python
from context_engine.agent import ContextAgent

class MyAgent(ContextAgent):
    def __init__(self):
        # Just pass namespace - DB config loads automatically
        super().__init__(name="SupportBot", namespace="support-team")
    
    def process(self, message):
        # Context automatically retrieved
        context = self.get_relevant_context(message)
        
        # Your LLM call here
        response = call_llm(context, message)
        
        # Auto-saved to memory
        self.remember_interaction(message, response)
        return response

# Run immediately
agent = MyAgent()
agent.run()
```

## What the Agent Gets For Free

When context engine is pre-configured, agents automatically get:

1. **Database Connection** - No setup needed
2. **Embeddings** - Via configured Ollama/OpenAI
3. **Namespace Isolation** - Each agent can have its own memory space
4. **Session Tracking** - Conversations automatically tracked
5. **TTL Support** - Temporary memories auto-expire

## Configuration Precedence

The agent checks for config in this order:

1. **Explicit config passed** to constructor
2. **Environment variables** (CTX_DB_HOST, etc.)
3. **Config file** at `~/.config/context_engine/config.json`
4. **Defaults** (localhost:5432)

## Quick Test

Verify the agent can connect:

```python
from context_engine import ContextEngine

memory = ContextEngine()

# Test save/retrieve
doc_id = memory.save("Test connection", category="test")
results = memory.search("test connection")

print(f"✓ Connected! Found {len(results)} results")
memory.close()
```

## Common Patterns

### Multi-Agent Shared Database

```python
# Agent A - Customer Support
support_agent = ContextEngine(
    config=ContextEngineConfig(namespace="support")
)

# Agent B - Code Review  
code_agent = ContextEngine(
    config=ContextEngineConfig(namespace="code-review")
)

# Both use same DB, but memories are isolated
```

### Session-Based Agents

```python
import uuid

class SessionAgent:
    def __init__(self, session_id=None):
        self.memory = ContextEngine()
        self.session_id = session_id or str(uuid.uuid4())[:8]
    
    def chat(self, user_msg):
        # Load session context
        history = self.memory.get_session(self.session_id)
        
        # Get relevant context from all sessions
        relevant = self.memory.get_context(user_msg)
        
        # Process and save
        response = self.process(user_msg, history, relevant)
        self.memory.save_conversation(
            session_key=self.session_id,
            user_message=user_msg,
            assistant_response=response
        )
        return response
```

## Environment Variables

If the context engine was set up with a script, these might already be set:

```bash
# Check what's configured
env | grep CTX_

# Typical setup from setup_context_engine.sh
CTX_DB_HOST=localhost
CTX_DB_PORT=5432
CTX_DB_NAME=context_engine
CTX_DB_USER=youruser
CTX_OLLAMA_URL=http://localhost:11434
CTX_NAMESPACE=default
```

## Migration from Markdown/Memory Files

If your agent currently uses markdown files or in-memory storage:

### Before (Markdown)
```python
# Load from file
with open("agent_memory.md") as f:
    context = f.read()

# Save to file  
with open("agent_memory.md", "a") as f:
    f.write(f"\nUser: {msg}\nAgent: {response}")
```

### After (Context Engine)
```python
from context_engine import ContextEngine

memory = ContextEngine()

# Semantic retrieval (not just text search)
context = memory.get_context(user_query, max_tokens=2000)

# Auto-embedded and searchable
memory.save_conversation(
    session_key="session-123",
    user_message=msg,
    assistant_response=response
)
```

## Troubleshooting

### "Database connection failed"

Check if context engine is accessible:

```python
from context_engine.schema import SchemaManager
from context_engine.config import ContextEngineConfig

config = ContextEngineConfig()
schema = SchemaManager(config)

success, error = schema.verify_connection()
if not success:
    print(f"Connection failed: {error}")
    print(f"Tried to connect to: {config.conn_string}")
```

### "No results found"

The database might be empty or using different namespace:

```python
# Check what's in the database
all_memories = memory.list(limit=100)
print(f"Found {len(all_memories)} memories")
print(f"Current namespace: {memory.namespace}")

# Check by category
for mem in all_memories:
    print(f"  [{mem['category']}] {mem['content'][:50]}...")
```

### "Embedding failed"

Check Ollama connection:

```python
import requests

ollama_url = "http://localhost:11434"  # From config
try:
    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
    if response.status_code == 200:
        print("✓ Ollama is running")
    else:
        print(f"✗ Ollama returned {response.status_code}")
except Exception as e:
    print(f"✗ Cannot reach Ollama: {e}")
```

## Summary

For agents, using an existing context engine is **zero-setup**:

1. Import `ContextEngine` or `ContextAgent`
2. Call `ContextEngine()` - config loads automatically
3. Start saving/retrieving memories

The context engine handles all the complexity (embeddings, database, etc.). Agents just use the simple API.

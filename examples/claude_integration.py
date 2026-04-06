#!/usr/bin/env python3
"""
Example: Integrating Context Engine with Claude Code.

This shows how to use the context engine as a memory layer for Claude Code
across multiple projects.

Setup:
    1. Install: pip install pgvector-context-engine
    2. Configure: Set environment variables (see below)
    3. Add to your CLAUDE.md or project setup

Environment variables:
    CTX_DB_HOST      - PostgreSQL host
    CTX_DB_PORT      - PostgreSQL port
    CTX_DB_NAME      - Database name (e.g., context_engine)
    CTX_DB_USER      - Database user
    CTX_DB_PASS      - Database password
    CTX_OLLAMA_URL   - Ollama URL (http://localhost:11434)
    CTX_EMBEDDING_MODEL - Embedding model (nomic-embed-text)
    CTX_NAMESPACE    - Project namespace (auto-detected from git repo)

Usage in Claude Code:

    from context_engine import ContextEngine

    def setup_agent_context():
        # Initialize context engine for this project
        ctx = ContextEngine()

        # Get relevant context for current task
        context = ctx.get_context(
            "What was I working on in this project?",
            max_tokens=3000
        )

        if context:
            return f"""
RELEVANT PROJECT CONTEXT:
{context}

[END CONTEXT]
"""
        return ""

    # Then include in your system prompt
"""

import os
from pathlib import Path

# Auto-detect namespace from git repo
def get_project_namespace() -> str:
    """Get namespace based on git remote or local directory."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from GitHub URL
            if "github.com" in url:
                parts = url.rstrip(".git").split("/")
                return f"{parts[-2]}/{parts[-1]}"
    except Exception:
        pass

    # Fallback to directory name
    return Path.cwd().name


# Example: Per-project configuration
def create_project_context():
    """Create context engine instance for current project."""
    from context_engine import ContextEngine, ContextEngineConfig

    config = ContextEngineConfig(
        namespace=get_project_namespace(),
        # Override with project-specific DB if needed
        # db_host="project-db.internal",
    )

    return ContextEngine(config=config)


# Example: Session initialization
def init_agent_session():
    """
    Call this at the start of an agent session.

    Returns context string to include in system prompt.
    """
    ctx = create_project_context()

    # Get general project context
    project_context = ctx.get_context(
        query="project goals architecture decisions",
        max_memories=5,
        max_tokens=2000
    )

    # Get recent work
    recent_context = ctx.get_context(
        query="recent work history progress",
        max_memories=3,
        max_tokens=1000
    )

    ctx.close()

    combined = []
    if project_context:
        combined.append(f"PROJECT CONTEXT:\n{project_context}")
    if recent_context:
        combined.append(f"RECENT WORK:\n{recent_context}")

    return "\n\n".join(combined) if combined else ""


# Example: Saving important decisions
def save_decision(decision: str, rationale: str, category: str = "decision"):
    """Save an important architectural decision."""
    ctx = create_project_context()
    ctx.save(
        content=f"Decision: {decision}\nRationale: {rationale}",
        category=category,
        importance=8.0,  # High importance for architectural decisions
        ttl_days=365,   # Keep for a year
    )
    ctx.close()


# Example: Saving user preferences
def save_preference(preference: str, context: str = ""):
    """Save a user preference for future reference."""
    ctx = create_project_context()
    content = f"Preference: {preference}"
    if context:
        content += f"\nContext: {context}"

    ctx.save(
        content=content,
        category="preference",
        importance=7.0,
    )
    ctx.close()


if __name__ == "__main__":
    print("Context Engine Claude Integration Examples")
    print("=" * 50)
    print(f"Project namespace: {get_project_namespace()}")
    print()
    print("Functions available:")
    print("  create_project_context() - Create engine for current project")
    print("  init_agent_session()     - Get context for session startup")
    print("  save_decision()          - Save architectural decision")
    print("  save_preference()        - Save user preference")

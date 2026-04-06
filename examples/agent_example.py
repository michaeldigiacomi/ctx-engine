#!/usr/bin/env python3
"""
Example: Context-Aware Agent with pgvector-context-engine

This demonstrates how to integrate the context engine into an AI agent
for persistent, semantic memory across sessions.

Requirements:
    pip install pgvector-context-engine

Environment:
    Set up your database config in ~/.config/context_engine/config.json
    or use environment variables:
        CTX_DB_HOST=localhost
        CTX_DB_PORT=5432
        CTX_DB_NAME=context_engine
        CTX_DB_USER=your_user
        CTX_DB_PASS=your_pass
        CTX_NAMESPACE=my-agent
"""

import os
import uuid
from typing import Optional

from context_engine import ContextEngine
from context_engine.config import ContextEngineConfig


class SimpleContextAgent:
    """
    A simple agent with semantic memory using pgvector-context-engine.

    This agent can:
    - Remember facts and conversations
    - Retrieve relevant context for queries
    - Maintain session history
    - Learn user preferences
    """

    def __init__(self, name: str = "Agent", namespace: Optional[str] = None):
        """
        Initialize the agent with context engine.

        Args:
            name: Agent name for identification
            namespace: Optional namespace override (default: from config)
        """
        self.name = name
        self.session_id = f"{name.lower()}-{uuid.uuid4().hex[:8]}"

        # Initialize context engine
        if namespace:
            config = ContextEngineConfig(namespace=namespace)
            self.memory = ContextEngine(config=config)
        else:
            self.memory = ContextEngine()

        print(f"🤖 {name} initialized with session: {self.session_id}")

        # Load existing preferences at startup
        self._load_preferences()

    def _load_preferences(self):
        """Load user preferences from memory at startup."""
        prefs = self.memory.get_context(
            "user preferences and settings",
            category="preference",
            max_tokens=500
        )
        if prefs:
            print(f"📚 Loaded preferences:\n{prefs}\n")

    def chat(self, message: str) -> str:
        """
        Process a user message with context-aware memory.

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        # Retrieve relevant context for this query
        relevant_context = self.memory.get_context(
            message,
            max_tokens=2000,
            max_memories=10
        )

        # Build the prompt with context
        prompt = self._build_prompt(relevant_context, message)

        # In a real implementation, this would call your LLM
        # response = call_your_llm(prompt)
        response = self._simulate_llm_response(prompt, message)

        # Save this conversation turn
        self.memory.save_conversation(
            session_key=self.session_id,
            user_message=message,
            assistant_response=response
        )

        return response

    def _build_prompt(self, context: str, message: str) -> str:
        """Build a prompt with relevant context."""
        parts = []

        if context:
            parts.append(f"Relevant information from memory:\n{context}")

        parts.append(f"User: {message}")
        parts.append("Assistant:")

        return "\n\n".join(parts)

    def _simulate_llm_response(self, prompt: str, original_message: str) -> str:
        """
        Simulate an LLM response.

        In a real implementation, replace this with your LLM call:

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        """
        # Simple keyword-based responses for demo
        msg_lower = original_message.lower()

        if "hello" in msg_lower or "hi" in msg_lower:
            return f"Hello! I'm {self.name}. How can I help you today?"

        elif "remember" in msg_lower:
            return "I've saved that information to my memory."

        elif "what" in msg_lower and ("know" in msg_lower or "remember" in msg_lower):
            if "python" in prompt.lower():
                return "Based on my memory, you mentioned liking Python!"
            return "I'm checking my memory for relevant information..."

        else:
            return f"I received your message: '{original_message}'. How can I assist?"

    def remember(self, fact: str, category: str = "general",
                 importance: float = 5.0, ttl_days: Optional[int] = None) -> str:
        """
        Explicitly remember a fact.

        Args:
            fact: The information to remember
            category: Category for organization (preference, knowledge, task, etc.)
            importance: Importance score (1-10)
            ttl_days: Optional expiration in days

        Returns:
            Confirmation message
        """
        doc_id = self.memory.save(
            content=fact,
            category=category,
            importance=importance,
            ttl_days=ttl_days
        )

        print(f"💾 Remembered: {fact[:50]}... (id: {doc_id[:8]}...)")
        return f"✓ Remembered: {fact}"

    def recall(self, query: str, limit: int = 5) -> list:
        """
        Search memory for relevant information.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        results = self.memory.search(query, limit=limit)

        print(f"🔍 Found {len(results)} relevant memories:")
        for r in results:
            print(f"  [{r['similarity']:.2f}] {r['content'][:60]}...")

        return results

    def get_session_history(self) -> list:
        """Get the current session's conversation history."""
        return self.memory.get_session(self.session_id)

    def show_stats(self):
        """Display memory statistics."""
        # Get all memories in namespace
        all_memories = self.memory.list(limit=1000)

        # Count by category
        categories = {}
        for m in all_memories:
            cat = m.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n📊 Memory Statistics:")
        print(f"  Total memories: {len(all_memories)}")
        print(f"  By category:")
        for cat, count in sorted(categories.items()):
            print(f"    - {cat}: {count}")
        print()

    def cleanup(self):
        """Clean up resources."""
        self.memory.close()
        print(f"👋 {self.name} shut down.")


def demo():
    """Run a demonstration of the context-aware agent."""
    print("=" * 60)
    print("Context-Aware Agent Demo")
    print("=" * 60)
    print()

    # Create agent
    agent = SimpleContextAgent("HelperBot", namespace="demo-agent")

    # Demonstrate: Learn preferences
    print("\n--- Teaching the Agent ---")
    agent.remember(
        "User prefers Python over JavaScript",
        category="preference",
        importance=8.0
    )
    agent.remember(
        "User is working on a web scraping project",
        category="project",
        importance=7.0
    )
    agent.remember(
        "User likes concise responses",
        category="preference",
        importance=9.0
    )

    # Demonstrate: Chat with context
    print("\n--- Chatting with Context ---")

    messages = [
        "Hello!",
        "What programming language should I use for my project?",
        "Remember that I prefer async/await patterns",
        "What do you know about my preferences?",
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response}")

    # Demonstrate: Search memory
    print("\n--- Searching Memory ---")
    agent.recall("programming languages", limit=3)

    # Show stats
    agent.show_stats()

    # Cleanup
    agent.cleanup()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

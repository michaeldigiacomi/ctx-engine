"""
Base agent class with built-in context engine integration.

This module provides a convenient base class for building AI agents
with persistent semantic memory.

Example:
    from context_engine.agent import ContextAgent

    class MyAgent(ContextAgent):
        def process(self, message):
            context = self.get_relevant_context(message)
            # Your processing logic here
            response = f"Received: {message}"
            self.remember_interaction(message, response)
            return response
"""

import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

from context_engine import ContextEngine
from context_engine.config import ContextEngineConfig


class ContextAgent(ABC):
    """
    Base class for AI agents with semantic memory.

    This class provides a foundation for building context-aware agents
    with automatic memory management. Subclasses only need to implement
    the `process` method.

    Attributes:
        name: Agent identifier
        memory: ContextEngine instance for persistence
        session_id: Unique session identifier

    Example:
        class MyAgent(ContextAgent):
            def process(self, message: str) -> str:
                context = self.get_relevant_context(message)
                # Use context with your LLM
                return "Response"
    """

    def __init__(
        self,
        name: str = "Agent",
        namespace: Optional[str] = None,
        config: Optional[ContextEngineConfig] = None,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name for identification
            namespace: Optional namespace for memory isolation
            config: Optional custom configuration
        """
        self.name = name
        self.session_id = f"{name.lower()}-{uuid.uuid4().hex[:8]}"

        # Initialize context engine
        if config:
            self.memory = ContextEngine(config=config)
        elif namespace:
            cfg = ContextEngineConfig(namespace=namespace)
            self.memory = ContextEngine(config=cfg)
        else:
            self.memory = ContextEngine()

        # Load preferences at startup
        self._preferences = self._load_preferences()

    def _load_preferences(self) -> str:
        """Load user preferences from memory."""
        return self.memory.get_context(
            "user preferences",
            category="preference",
            max_tokens=500
        )

    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        max_memories: int = 10,
        category: Optional[str] = None,
    ) -> str:
        """
        Get relevant context for a query.

        Args:
            query: The current task/query
            max_tokens: Maximum tokens for context
            max_memories: Maximum number of memories
            category: Optional category filter

        Returns:
            Formatted context string
        """
        # Get user preferences
        prefs = self.memory.get_context(
            "user preferences",
            category="preference",
            max_tokens=500
        )

        # Get relevant knowledge
        if category:
            knowledge = self.memory.get_context(
                query,
                category=category,
                max_tokens=max_tokens - 500
            )
        else:
            knowledge = self.memory.get_context(
                query,
                max_tokens=max_tokens - 500,
                max_memories=max_memories
            )

        # Combine
        parts = []
        if prefs:
            parts.append(f"User Preferences:\n{prefs}")
        if knowledge:
            parts.append(f"Relevant Information:\n{knowledge}")

        return "\n\n".join(parts)

    def remember(
        self,
        content: str,
        category: str = "general",
        importance: float = 5.0,
        ttl_days: Optional[int] = None,
    ) -> str:
        """
        Remember a fact.

        Args:
            content: Information to remember
            category: Category for organization
            importance: Importance score (1-10)
            ttl_days: Optional expiration

        Returns:
            Document ID of saved memory
        """
        return self.memory.save(
            content=content,
            category=category,
            importance=importance,
            ttl_days=ttl_days
        )

    def remember_interaction(
        self,
        user_message: str,
        assistant_response: str,
        category: str = "conversation",
    ) -> str:
        """
        Remember a conversation turn.

        Args:
            user_message: User's message
            assistant_response: Agent's response
            category: Conversation category

        Returns:
            Document ID of saved memory
        """
        return self.memory.save_conversation(
            session_key=self.session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            category=category
        )

    def recall(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memory for relevant information.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            category: Optional category filter

        Returns:
            List of matching memories
        """
        return self.memory.search(
            query,
            limit=limit,
            min_similarity=min_similarity,
            category=category
        )

    def get_session_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for current session.

        Args:
            limit: Maximum number of turns

        Returns:
            List of conversation memories
        """
        return self.memory.get_session(self.session_id, limit=limit)

    def list_memories(
        self,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List all memories.

        Args:
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of memories
        """
        return self.memory.list(category=category, limit=limit)

    def forget(self, doc_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete(doc_id)

    def cleanup_expired(self) -> int:
        """
        Remove expired memories.

        Returns:
            Number of memories deleted
        """
        return self.memory.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        memories = self.memory.list(limit=10000)

        categories = {}
        for m in memories:
            cat = m.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_memories': len(memories),
            'by_category': categories,
            'session_id': self.session_id,
            'namespace': self.memory.namespace,
        }

    @abstractmethod
    def process(self, message: str) -> str:
        """
        Process a user message.

        This is the main method subclasses must implement.

        Args:
            message: User's input message

        Returns:
            Agent's response
        """
        pass

    def run(self):
        """
        Run the agent in an interactive loop.

        Type 'exit' to quit, 'stats' to see memory stats.
        """
        print(f"🤖 {self.name} is ready! Type 'exit' to quit.")
        print()

        while True:
            try:
                message = input("You: ").strip()

                if message.lower() == 'exit':
                    break

                if message.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\n📊 Stats: {stats['total_memories']} memories")
                    for cat, count in stats['by_category'].items():
                        print(f"  - {cat}: {count}")
                    print()
                    continue

                if not message:
                    continue

                response = self.process(message)
                print(f"{self.name}: {response}\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

    def close(self):
        """Clean up resources."""
        self.memory.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SimpleAgent(ContextAgent):
    """
    Simple agent implementation that echoes messages.

    Use this as a template for your own agents.
    """

    def process(self, message: str) -> str:
        """Process message with context."""
        # Get relevant context
        context = self.get_relevant_context(message)

        # Here you would call your LLM with the context
        # For demo, we just echo
        if context:
            return f"[With context] Received: {message}"
        return f"Received: {message}"


# Example usage
if __name__ == "__main__":
    # Create a simple agent
    with SimpleAgent("Demo") as agent:
        # Remember something
        agent.remember(
            "User likes Python",
            category="preference",
            importance=8.0
        )

        # Process a message
        response = agent.process("What language should I use?")
        print(response)

        # Show stats
        stats = agent.get_stats()
        print(f"\nTotal memories: {stats['total_memories']}")

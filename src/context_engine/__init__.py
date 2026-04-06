"""
PGVector Context Engine - Reusable semantic memory system

Usage:
    from context_engine import ContextEngine

    ctx = ContextEngine()
    ctx.save("Deployed to k3s cluster", category="infrastructure")
    context = ctx.get_context("What was I working on?")

Agent Integration:
    from context_engine.agent import ContextAgent

    class MyAgent(ContextAgent):
        def process(self, message):
            context = self.get_relevant_context(message)
            # Use context with your LLM
            return "Response"
"""

from context_engine.core import ContextEngine
from context_engine.config import ContextEngineConfig
from context_engine.providers import EmbeddingProvider, OllamaProvider

__all__ = [
    "ContextEngine",
    "ContextEngineConfig",
    "EmbeddingProvider",
    "OllamaProvider",
]

# Agent module available at context_engine.agent

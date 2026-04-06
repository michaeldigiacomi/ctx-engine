"""
PGVector Context Engine - Core semantic memory manager.

This is the main public API for the context engine.
"""

from __future__ import annotations

import hashlib
import psycopg2
import psycopg2.extras
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from context_engine.config import ContextEngineConfig
from context_engine.providers import OllamaProvider, EmbeddingProvider, EmbeddingError


class ContextEngine:
    """
    Semantic context/memory engine using PostgreSQL + pgvector.

    Main features:
        - Store memories with automatic embedding
        - Semantic search via vector similarity
        - Project/namespace isolation
        - TTL-based expiration
        - Importance scoring
        - Category filtering

    Usage:
        ctx = ContextEngine()
        ctx.save("Deployed to k8s", category="infra")
        context = ctx.get_context("What was I working on?")
    """

    def __init__(
        self,
        config: Optional[ContextEngineConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        auto_init: bool = True,
    ):
        """
        Initialize the context engine.

        Args:
            config: Configuration object. If None, loads from env/config file.
            embedding_provider: Custom embedding provider. If None, uses Ollama.
            auto_init: If True, ensure schema exists on first use.
        """
        self.config = config or ContextEngineConfig()
        self.namespace = self.config.namespace

        # Use custom provider or default to Ollama
        if embedding_provider is not None:
            self._embedding = embedding_provider
        else:
            self._embedding = OllamaProvider(
                url=self.config.ollama_url,
                model=self.config.embedding_model,
            )

        self._conn = None
        self._auto_init = auto_init
        self._initialized = False

    def _get_conn(self):
        """Get database connection with lazy initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.config.conn_string)
        return self._conn

    def _ensure_initialized(self):
        """Ensure database schema is ready (called on first operation)."""
        if self._initialized:
            return

        if self._auto_init:
            from context_engine.schema import SchemaManager
            schema = SchemaManager(self.config)
            schema.ensure_database_exists()
            schema.ensure_schema(run_migrations=True)

        self._initialized = True

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            return self._embedding.embed(text)
        except EmbeddingError as e:
            print(f"Embedding failed: {e}")
            # Return zero vector as fallback (will have low similarity)
            return [0.0] * self._embedding.dimension

    def save(
        self,
        content: str,
        category: str = "general",
        importance: float = 1.0,
        ttl_days: Optional[int] = None,
        session_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Save a memory with automatic embedding.

        Args:
            content: The memory content (min 10 chars)
            category: Category for filtering (default: "general")
            importance: 0.1-10.0 importance score (default: 1.0)
            ttl_days: Days until expiration (None = permanent)
            session_key: Optional session identifier
            tags: Optional list of tags
            metadata: Optional additional metadata
            source: Optional source identifier
            doc_id: Optional stable ID (auto-generated from content if not provided)

        Returns:
            The doc_id of the saved memory
        """
        self._ensure_initialized()

        if len(content.strip()) < 10:
            return ""

        # Generate doc_id if not provided
        if not doc_id:
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:32]

        embedding = self._embed(content)

        expires_at = None
        if ttl_days:
            expires_at = datetime.now() + timedelta(days=ttl_days)

        conn = self._get_conn()
        cur = conn.cursor()

        metadata = metadata or {}
        metadata["saved_by"] = "context_engine"
        metadata["saved_at"] = datetime.now().isoformat()

        try:
            cur.execute("""
                INSERT INTO memories
                (doc_id, content, embedding, namespace, category, importance,
                 expires_at, session_key, tags, metadata, source, created_at)
                VALUES (
                    %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    category = EXCLUDED.category,
                    importance = EXCLUDED.importance,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
            """, (
                doc_id, content, embedding, self.namespace, category,
                importance, expires_at, session_key, tags,
                psycopg2.extras.Json(metadata), source
            ))

            mem_id = cur.fetchone()[0]
            conn.commit()
            return doc_id

        except psycopg2.Error as e:
            conn.rollback()
            raise ContextEngineError(f"Failed to save memory: {e}") from e
        finally:
            cur.close()

    def get_context(
        self,
        query: str,
        max_memories: int = 10,
        max_tokens: int = 4000,
        category: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Get relevant context for a query using semantic search.

        Args:
            query: Search query or current task description
            max_memories: Maximum number of memories to retrieve
            max_tokens: Approximate token budget for context
            category: Optional category filter
            namespace: Optional namespace override (defaults to self.namespace)

        Returns:
            Formatted context string with relevant memories
        """
        self._ensure_initialized()

        ns = namespace or self.namespace
        embedding = self._embed(query)

        conn = self._get_conn()
        cur = conn.cursor()

        # Build query with filters
        sql = """
            SELECT id, doc_id, content, category, source, created_at,
                   importance, 1 - (embedding <=> %s::vector) as similarity
            FROM memories
            WHERE namespace = %s
              AND (expires_at IS NULL OR expires_at > NOW())
        """
        params = [embedding, ns]

        if category:
            sql += " AND category = %s"
            params.append(category)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, max_memories])

        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return ""

        # Build context within token budget
        memories = []
        total_chars = 0
        max_chars = int(max_tokens * 4)  # ~4 chars per token

        for row in rows:
            mem_id, doc_id, content, cat, source, created_at, importance, similarity = row

            if similarity < 0.5:
                continue

            date_str = created_at.strftime("%Y-%m-%d") if created_at else "unknown"

            parts = [f"[{cat}]"]
            if source:
                parts.append(f"@{source}")
            parts.append(f"({date_str})")
            parts.append(content)

            formatted = " ".join(parts)

            if len(formatted) + total_chars > max_chars:
                break

            memories.append(formatted)
            total_chars += len(formatted) + 2

            # Update access tracking (fire and forget)
            self._update_access(mem_id)

        return "\n\n".join(memories)

    def _update_access(self, memory_id: int):
        """Update access count and last accessed time."""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = %s
            """, (memory_id,))
            conn.commit()
            cur.close()
        except psycopg2.Error:
            pass  # Non-critical, ignore

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold (0-1)
            category: Optional category filter

        Returns:
            List of matching memories with similarity scores
        """
        self._ensure_initialized()

        embedding = self._embed(query)

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        sql = """
            SELECT doc_id, content, category, source, tags, metadata,
                   importance, created_at, access_count,
                   1 - (embedding <=> %s::vector) as similarity
            FROM memories
            WHERE namespace = %s
              AND (expires_at IS NULL OR expires_at > NOW())
        """
        params = [embedding, self.namespace]

        if category:
            sql += " AND category = %s"
            params.append(category)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, limit * 2])

        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        results = []
        for row in rows:
            row = dict(row)
            if row["similarity"] >= min_similarity:
                row["similarity"] = round(row["similarity"], 4)
                results.append(row)

        return results[:limit]

    def list(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List memories with optional filtering.

        Args:
            category: Optional category filter
            limit: Maximum results
            since: Only show memories created after this time

        Returns:
            List of memories
        """
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        conditions = ["namespace = %s"]
        params = [self.namespace]

        if category:
            conditions.append("category = %s")
            params.append(category)

        if since:
            conditions.append("created_at > %s")
            params.append(since)

        where_clause = " AND ".join(conditions)

        cur.execute(f"""
            SELECT doc_id, content, category, source, tags, metadata,
                   importance, created_at, access_count, expires_at
            FROM memories
            WHERE {where_clause}
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY created_at DESC
            LIMIT %s
        """, (*params, limit))

        rows = cur.fetchall()
        cur.close()

        return [dict(row) for row in rows]

    def delete(self, doc_id: str) -> bool:
        """Delete a memory by doc_id."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM memories
            WHERE doc_id = %s AND namespace = %s
        """, (doc_id, self.namespace))

        deleted = cur.rowcount > 0
        conn.commit()
        cur.close()

        return deleted

    def cleanup_expired(self) -> int:
        """Delete expired memories. Returns count deleted."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM memories
            WHERE expires_at IS NOT NULL AND expires_at <= NOW()
        """)

        count = cur.rowcount
        conn.commit()
        cur.close()

        return count

    def save_conversation(
        self,
        session_key: str,
        user_message: str,
        assistant_response: str,
        category: str = "conversation",
    ) -> str:
        """
        Save a conversation turn and link to session.

        Args:
            session_key: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            category: Memory category

        Returns:
            doc_id of saved memory
        """
        content = f"User: {user_message}\nAssistant: {assistant_response}"
        return self.save(
            content=content,
            category=category,
            importance=1.0,
            ttl_days=30,
            session_key=session_key,
        )

    def get_session(self, session_key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all memories for a specific session."""
        self._ensure_initialized()

        conn = self._get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT doc_id, content, category, created_at, importance
            FROM memories
            WHERE session_key = %s AND namespace = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (session_key, self.namespace, limit))

        rows = cur.fetchall()
        cur.close()

        return [dict(row) for row in rows]

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ContextEngineError(Exception):
    """Raised when a context engine operation fails."""
    pass

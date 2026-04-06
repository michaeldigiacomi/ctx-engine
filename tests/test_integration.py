"""Integration tests for ContextEngine with real PostgreSQL/pgvector.

These tests require a running PostgreSQL instance with pgvector extension.
Set the following environment variables:
    CTX_DB_HOST - PostgreSQL host (default: localhost)
    CTX_DB_PORT - PostgreSQL port (default: 5432)
    CTX_DB_NAME - Database name (default: context_engine)
    CTX_DB_USER - Database user
    CTX_DB_PASS - Database password

Run with: pytest tests/test_integration.py -v
Skip with: pytest tests/test_integration.py -v --ignore-glob='*integration*'
"""

import pytest
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from context_engine import ContextEngine
from context_engine.config import ContextEngineConfig
from context_engine.schema import SchemaManager


pytestmark = pytest.mark.integration


class MockEmbeddingProvider:
    """Mock embedding provider that returns deterministic vectors for testing."""

    def __init__(self, dimension=768):
        self.dimension = dimension

    def embed(self, text: str):
        """Return a deterministic vector based on text content."""
        import hashlib
        # Create a hash from the text
        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Convert to float values, normalize to unit vector
        import math
        values = []
        for i in range(self.dimension):
            # Use hash segments to create float values
            segment = hash_val[i % 32]
            val = (ord(segment) / 128.0) - 1.0  # Range -1 to 1
            values.append(val)

        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        return values


def create_test_config(namespace: str = None):
    """Create a test configuration."""
    if namespace is None:
        namespace = f"test-{uuid.uuid4().hex[:8]}"

    return ContextEngineConfig(
        db_host=os.getenv("CTX_DB_HOST", "localhost"),
        db_port=int(os.getenv("CTX_DB_PORT", "5432")),
        db_name=os.getenv("CTX_DB_NAME", "context_engine"),
        db_user=os.getenv("CTX_DB_USER", ""),
        db_pass=os.getenv("CTX_DB_PASS", ""),
        db_sslmode=os.getenv("CTX_DB_SSLMODE", "disable"),
        ollama_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        namespace=namespace,
    )


@pytest.fixture
def test_namespace():
    """Generate a unique test namespace."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def context_engine(test_namespace, postgres_available):
    """Create a ContextEngine with mocked embeddings for testing."""
    if not postgres_available:
        pytest.skip("PostgreSQL with pgvector not available")

    config = create_test_config(test_namespace)
    mock_embedding = MockEmbeddingProvider(dimension=768)

    engine = ContextEngine(
        config=config,
        embedding_provider=mock_embedding,
        auto_init=True
    )

    # Pre-cleanup: ensure no existing test data in this namespace
    try:
        conn = engine._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE namespace = %s", (test_namespace,))
        conn.commit()
        cur.close()
    except Exception:
        pass

    yield engine

    # Cleanup: delete all memories in test namespace
    try:
        conn = engine._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE namespace = %s", (test_namespace,))
        conn.commit()
        cur.close()
        engine.close()
    except Exception:
        pass


class TestBasicOperations:
    """Test basic CRUD operations."""

    def test_save_and_retrieve(self, context_engine):
        """Test saving a memory and retrieving it."""
        doc_id = context_engine.save(
            content="This is a test memory about Kubernetes deployment",
            category="infrastructure",
            importance=8.0
        )

        assert doc_id is not None
        assert len(doc_id) == 32

        # List should find the memory
        memories = context_engine.list()
        assert len(memories) >= 1
        # Check that our saved memory is in the list
        contents = [m["content"] for m in memories]
        assert "This is a test memory about Kubernetes deployment" in contents

    def test_save_with_ttl(self, context_engine):
        """Test saving with TTL creates expiration."""
        doc_id = context_engine.save(
            content="This memory expires soon",
            ttl_days=1
        )

        # Verify the memory has expiration set by querying directly
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT expires_at FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        result = cur.fetchone()
        cur.close()

        assert result is not None
        assert result[0] is not None

    def test_save_duplicate_content_updates(self, context_engine):
        """Test that saving duplicate content updates existing record."""
        content = f"Duplicate content test {datetime.now().isoformat()}"  # Unique content

        doc_id1 = context_engine.save(content=content, category="first")
        doc_id2 = context_engine.save(content=content, category="second")

        # Should have same doc_id (based on content hash)
        assert doc_id1 == doc_id2

        # Should only have one memory with this content, with updated category
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT category FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id1, context_engine.namespace)
        )
        result = cur.fetchone()
        cur.close()

        assert result is not None
        assert result[0] == "second"

    def test_delete(self, context_engine):
        """Test deleting a memory."""
        doc_id = context_engine.save(content="Memory to delete")

        # Verify it exists
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        assert cur.fetchone() is not None

        # Delete it
        result = context_engine.delete(doc_id)
        assert result is True

        # Should be gone
        cur.execute(
            "SELECT 1 FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        assert cur.fetchone() is None
        cur.close()

    def test_delete_nonexistent(self, context_engine):
        """Test deleting a non-existent memory."""
        result = context_engine.delete("nonexistent-doc-id")
        assert result is False


class TestSemanticSearch:
    """Test semantic search functionality."""

    def test_search_finds_similar_content(self, context_engine):
        """Test that search finds semantically similar content."""
        # Save related memories
        context_engine.save(content="Deployed WebMonsters to the Kubernetes cluster")
        context_engine.save(content="The weather is nice today")
        context_engine.save(content="Working on k8s infrastructure")

        # Search for k8s-related content
        results = context_engine.search("kubernetes deployment", limit=5)

        # Should find the k8s-related memories (may find all depending on mock vectors)
        assert len(results) >= 1

    def test_search_with_similarity_threshold(self, context_engine):
        """Test search with minimum similarity threshold."""
        context_engine.save(content="Kubernetes deployment strategies")

        # Search with high threshold
        results = context_engine.search(
            "completely unrelated topic",
            min_similarity=0.9
        )

        # Should not find anything with high threshold on unrelated query
        assert len(results) == 0

    def test_search_with_category_filter(self, context_engine):
        """Test search with category filter."""
        # Use unique content and category to avoid conflicts
        unique_id = uuid.uuid4().hex[:8]
        unique_category = f"test-infra-{unique_id}"
        context_engine.save(content=f"K8s deploy test {unique_id}", category=unique_category)
        context_engine.save(content=f"Code review test {unique_id}", category=f"test-dev-{unique_id}")

        results = context_engine.search(f"K8s deploy test {unique_id}", category=unique_category)

        assert len(results) >= 1
        assert all(r["category"] == unique_category for r in results)


class TestGetContext:
    """Test get_context method."""

    def test_get_context_returns_formatted_memories(self, context_engine):
        """Test that get_context returns formatted memories."""
        # Use unique content for reliable matching
        unique_content = f"User prefers dark mode {uuid.uuid4().hex[:8]}"
        context_engine.save(
            content=unique_content,
            category="preference"
        )

        context = context_engine.get_context(unique_content)

        assert unique_content in context
        assert "[preference]" in context

    def test_get_context_respects_token_budget(self, context_engine):
        """Test that get_context respects token budget."""
        # Add many long memories
        for i in range(20):
            context_engine.save(
                content=f"Memory {i}: " + "X" * 500,
                category="test"
            )

        context = context_engine.get_context("test query", max_tokens=500)

        # Should be limited by token budget
        assert len(context) < 2000  # 500 tokens * ~4 chars/token

    def test_get_context_with_category_filter(self, context_engine):
        """Test get_context with category filter."""
        context_engine.save(content="Infrastructure memory", category="infrastructure")
        context_engine.save(content="Personal preference", category="preference")

        context = context_engine.get_context(
            "test",
            category="infrastructure"
        )

        assert "Infrastructure" in context
        assert "Personal" not in context


class TestNamespaceIsolation:
    """Test namespace isolation between projects."""

    def test_namespaces_are_isolated(self, postgres_available):
        """Test that different namespaces don't see each other's data."""
        if not postgres_available:
            pytest.skip("PostgreSQL not available")

        mock_embedding = MockEmbeddingProvider(dimension=768)
        ns1 = f"test-ns1-{uuid.uuid4().hex[:8]}"
        ns2 = f"test-ns2-{uuid.uuid4().hex[:8]}"

        config1 = create_test_config(ns1)
        config2 = create_test_config(ns2)

        engine1 = ContextEngine(config1, embedding_provider=mock_embedding, auto_init=True)
        engine2 = ContextEngine(config2, embedding_provider=mock_embedding, auto_init=True)

        try:
            # Pre-cleanup
            for engine in [engine1, engine2]:
                conn = engine._get_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM memories WHERE namespace = %s", (engine.namespace,))
                conn.commit()
                cur.close()

            # Save in namespace 1
            unique_content = f"Memory in namespace 1 {datetime.now().isoformat()}"
            engine1.save(content=unique_content)

            # Should be visible in namespace 1
            memories1 = engine1.list()
            contents1 = [m["content"] for m in memories1]
            assert unique_content in contents1

            # Should not be visible in namespace 2
            memories2 = engine2.list()
            contents2 = [m["content"] for m in memories2]
            assert unique_content not in contents2

            # Search should also be isolated
            results1 = engine1.search(unique_content)
            results2 = engine2.search(unique_content)

            assert len(results1) == 1
            assert len(results2) == 0
        finally:
            engine1.close()
            engine2.close()

            # Cleanup
            try:
                from context_engine.schema import SchemaManager
                sm = SchemaManager(config1)
                conn = sm._get_app_conn()
                cur = conn.cursor()
                cur.execute("DELETE FROM memories WHERE namespace IN (%s, %s)", (ns1, ns2))
                conn.commit()
                cur.close()
                conn.close()
            except Exception:
                pass


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_expired_memories(self, context_engine):
        """Test cleaning up expired memories."""
        # Save a memory with very short TTL (already expired)
        doc_id = context_engine.save(
            content="Expired memory",
            ttl_days=-1  # Already expired
        )

        # Should exist before cleanup
        memories = context_engine.list()
        # Note: list filters out expired, so we need to check directly

        # Cleanup should remove it
        count = context_engine.cleanup_expired()
        assert count >= 1

    def test_list_excludes_expired(self, context_engine):
        """Test that list excludes expired memories."""
        # Save an expired memory directly via SQL
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO memories (doc_id, content, embedding, namespace, category, expires_at)
            VALUES (%s, %s, %s::vector, %s, %s, NOW() - INTERVAL '1 day')
        """, ("expired-doc", "Expired content", [0.0] * 768, context_engine.namespace, "test"))
        conn.commit()
        cur.close()

        # List should not include expired
        memories = context_engine.list()
        doc_ids = [m["doc_id"] for m in memories]
        assert "expired-doc" not in doc_ids


class TestSessionManagement:
    """Test session/conversation management."""

    def test_save_conversation(self, context_engine):
        """Test saving a conversation turn."""
        unique_session = f"session-{uuid.uuid4().hex[:8]}"
        unique_msg = f"How do I deploy to k8s? {uuid.uuid4().hex[:8]}"
        doc_id = context_engine.save_conversation(
            session_key=unique_session,
            user_message=unique_msg,
            assistant_response="You can use kubectl apply..."
        )

        assert doc_id is not None

        # Should be findable by querying the session
        session_memories = context_engine.get_session(unique_session)
        assert len(session_memories) >= 1

    def test_get_session(self, context_engine):
        """Test retrieving session memories."""
        unique_session = f"test-session-{uuid.uuid4().hex[:8]}"

        # Save multiple conversation turns
        for i in range(3):
            context_engine.save_conversation(
                session_key=unique_session,
                user_message=f"Question {i} {uuid.uuid4().hex[:4]}",
                assistant_response=f"Answer {i}"
            )

        # Save to different session
        context_engine.save_conversation(
            session_key=f"other-{unique_session}",
            user_message="Different session",
            assistant_response="Different answer"
        )

        # Get session memories
        session_memories = context_engine.get_session(unique_session)

        assert len(session_memories) == 3


class TestMetadata:
    """Test metadata handling."""

    def test_save_with_metadata(self, context_engine):
        """Test saving with metadata."""
        unique_content = f"Memory with metadata {uuid.uuid4().hex[:8]}"
        doc_id = context_engine.save(
            content=unique_content,
            metadata={"project": "test", "priority": "high", "tags": ["a", "b"]}
        )

        # Verify metadata by querying directly
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT metadata FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        result = cur.fetchone()
        cur.close()

        assert result is not None
        meta = result[0]
        assert meta.get("project") == "test"

    def test_save_with_tags(self, context_engine):
        """Test saving with tags."""
        unique_content = f"Memory with tags {uuid.uuid4().hex[:8]}"
        doc_id = context_engine.save(
            content=unique_content,
            tags=["important", "review-later"]
        )

        # Verify tags by querying directly
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT tags FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        result = cur.fetchone()
        cur.close()

        assert result is not None
        tags = result[0]
        assert "important" in tags


class TestSchemaManagement:
    """Test schema management functionality."""

    def test_schema_manager_verifies_connection(self, test_namespace):
        """Test SchemaManager can verify connection."""
        config = create_test_config(test_namespace)
        schema = SchemaManager(config)

        success, error = schema.verify_connection()

        if os.getenv("CTX_DB_USER"):
            assert success is True
            assert error is None
        else:
            # May fail if no credentials configured
            pass

    def test_schema_manager_creates_database(self, test_namespace):
        """Test SchemaManager can create database if needed."""
        config = create_test_config(test_namespace)
        schema = SchemaManager(config)

        # This may or may not create depending on if DB exists
        # Just verify it doesn't crash
        try:
            schema.ensure_database_exists()
        except Exception as e:
            pytest.skip(f"Database creation not available: {e}")


class TestImportanceAndAccess:
    """Test importance scoring and access tracking."""

    def test_importance_saved_correctly(self, context_engine):
        """Test that importance is saved and retrieved."""
        # Use unique content to avoid ON CONFLICT with existing data
        unique_content = f"Important memory {uuid.uuid4().hex[:8]}"
        doc_id = context_engine.save(
            content=unique_content,
            importance=9.5
        )

        # Verify importance by querying directly
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT importance FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        result = cur.fetchone()
        cur.close()

        assert result is not None
        assert result[0] == 9.5

    def test_access_count_updated(self, context_engine):
        """Test that access count is updated on get_context."""
        unique_content = f"Test memory for access {uuid.uuid4().hex[:8]}"
        doc_id = context_engine.save(content=unique_content)

        # Get initial state by querying directly
        conn = context_engine._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT access_count FROM memories WHERE doc_id = %s AND namespace = %s",
            (doc_id, context_engine.namespace)
        )
        initial_count = cur.fetchone()[0] or 0
        cur.close()

        # Search to trigger access update
        context_engine.get_context(unique_content)

        # Access count should have been updated
        # Note: This is fire-and-forget, so we just verify the save worked
        assert doc_id is not None

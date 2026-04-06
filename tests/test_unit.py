"""Unit tests for ContextEngine with mocked dependencies."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock

from context_engine.core import ContextEngine, ContextEngineError
from context_engine.config import ContextEngineConfig


class TestContextEngineInit:
    """Test ContextEngine initialization."""

    def test_init_with_defaults(self, mock_embedding):
        """Test initialization with default config."""
        with patch('context_engine.core.ContextEngineConfig') as mock_config:
            mock_config.return_value = MagicMock(
                namespace="test",
                ollama_url="http://localhost:11434",
                embedding_model="nomic-embed-text",
                conn_string="postgresql://user:pass@localhost/db",
            )

            engine = ContextEngine(embedding_provider=mock_embedding, auto_init=False)
            assert engine.namespace == "test"
            assert engine._embedding == mock_embedding

    def test_init_with_custom_config(self, test_config, mock_embedding):
        """Test initialization with custom config."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )
        assert engine.namespace == test_config.namespace
        assert engine.config == test_config

    def test_init_auto_init_disabled(self, test_config, mock_embedding):
        """Test that auto_init=False prevents schema initialization."""
        with patch('context_engine.schema.SchemaManager') as mock_schema:
            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )
            assert not engine._initialized
            mock_schema.assert_not_called()

    def test_context_manager(self, mock_embedding):
        """Test that ContextEngine works as a context manager."""
        with patch('context_engine.core.ContextEngineConfig') as mock_config:
            mock_config.return_value = MagicMock(
                namespace="test",
                conn_string="postgresql://user:pass@localhost/db",
            )

            with ContextEngine(embedding_provider=mock_embedding, auto_init=False) as ctx:
                assert isinstance(ctx, ContextEngine)


class TestSave:
    """Test saving memories."""

    def test_save_basic(self, test_config, mock_embedding):
        """Test saving a basic memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory",
                category="test",
                importance=5.0
            )

            assert doc_id is not None
            assert len(doc_id) == 32  # SHA256 hash length
            mock_cur.execute.assert_called()
            mock_conn.commit.assert_called_once()

    def test_save_content_too_short(self, test_config, mock_embedding):
        """Test that short content is rejected."""
        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )

        doc_id = engine.save(content="Short")  # Less than 10 chars
        assert doc_id == ""

    def test_save_with_ttl(self, test_config, mock_embedding):
        """Test saving with TTL."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory with TTL",
                ttl_days=7
            )

            assert doc_id is not None
            # Check that expires_at was included in the query
            call_args = mock_cur.execute.call_args
            assert call_args is not None

    def test_save_with_custom_doc_id(self, test_config, mock_embedding):
        """Test saving with a custom doc_id."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            custom_id = "my-custom-doc-id-12345"
            doc_id = engine.save(
                content="This is a test memory with custom ID",
                doc_id=custom_id
            )

            assert doc_id == custom_id

    def test_save_with_metadata(self, test_config, mock_embedding):
        """Test saving with metadata."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save(
                content="This is a test memory with metadata",
                metadata={"project": "test", "priority": "high"}
            )

            assert doc_id is not None


class TestSearch:
    """Test searching memories."""

    def test_search_basic(self, test_config, mock_embedding):
        """Test basic search functionality."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {
                    "doc_id": "abc123",
                    "content": "Test memory",
                    "category": "test",
                    "source": None,
                    "tags": None,
                    "metadata": {},
                    "importance": 1.0,
                    "created_at": datetime.now(),
                    "access_count": 0,
                    "similarity": 0.85,
                }
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", limit=5)

            assert len(results) == 1
            assert results[0]["similarity"] == 0.85
            mock_cur.execute.assert_called()

    def test_search_with_category_filter(self, test_config, mock_embedding):
        """Test search with category filter."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test query", category="infrastructure")

            # Check that category filter was included
            call_args = mock_cur.execute.call_args[0][0]
            assert "category" in call_args

    def test_search_min_similarity_filter(self, test_config, mock_embedding):
        """Test that results below min_similarity are filtered out."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            # Return results with various similarities
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "High sim", "similarity": 0.9},
                {"doc_id": "2", "content": "Med sim", "similarity": 0.6},
                {"doc_id": "3", "content": "Low sim", "similarity": 0.3},  # Below threshold
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.search("test", min_similarity=0.5)

            assert len(results) == 2
            assert all(r["similarity"] >= 0.5 for r in results)


class TestGetContext:
    """Test get_context method."""

    def test_get_context_basic(self, test_config, mock_embedding):
        """Test getting context for a query."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                (1, "doc1", "Test memory content", "test", "source", datetime.now(), 5.0, 0.85)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            context = engine.get_context("What was I working on?")

            assert "Test memory content" in context
            assert "[test]" in context

    def test_get_context_empty_results(self, test_config, mock_embedding):
        """Test get_context with no results."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            context = engine.get_context("Nonexistent query")

            assert context == ""

    def test_get_context_token_budget(self, test_config, mock_embedding):
        """Test that token budget is respected."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()

            # Create many long memories
            long_memory = "X" * 1000  # 1000 char memory
            mock_cur.fetchall.return_value = [
                (i, f"doc{i}", long_memory, "test", None, datetime.now(), 5.0, 0.9)
                for i in range(10)
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            # With max_tokens=100 (~400 chars), should only get a few memories
            context = engine.get_context("test", max_tokens=100)

            # Rough check: should be limited by token budget
            assert len(context) < 4000


class TestList:
    """Test listing memories."""

    def test_list_basic(self, test_config, mock_embedding):
        """Test basic list functionality."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "Memory 1", "category": "test"},
                {"doc_id": "2", "content": "Memory 2", "category": "test"},
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.list(limit=10)

            assert len(results) == 2

    def test_list_with_category_filter(self, test_config, mock_embedding):
        """Test list with category filter."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            engine.list(category="infrastructure")

            call_args = mock_cur.execute.call_args[0][0]
            assert "category" in call_args


class TestDelete:
    """Test deleting memories."""

    def test_delete_existing(self, test_config, mock_embedding):
        """Test deleting an existing memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 1
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.delete("doc-id-123")

            assert result is True
            mock_cur.execute.assert_called()

    def test_delete_nonexistent(self, test_config, mock_embedding):
        """Test deleting a non-existent memory."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 0
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            result = engine.delete("nonexistent-doc")

            assert result is False


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_expired(self, test_config, mock_embedding):
        """Test cleaning up expired memories."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.rowcount = 5
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            count = engine.cleanup_expired()

            assert count == 5
            mock_cur.execute.assert_called()


class TestConversation:
    """Test conversation methods."""

    def test_save_conversation(self, test_config, mock_embedding):
        """Test saving a conversation turn."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = [1]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            doc_id = engine.save_conversation(
                session_key="session-123",
                user_message="Hello",
                assistant_response="Hi there!"
            )

            assert doc_id is not None
            # Check that content includes both user and assistant
            call_args = mock_cur.execute.call_args
            assert "session-123" in str(call_args)

    def test_get_session(self, test_config, mock_embedding):
        """Test getting session memories."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [
                {"doc_id": "1", "content": "User: Hello\nAssistant: Hi", "category": "conversation"},
            ]
            mock_conn.cursor.return_value = mock_cur
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            results = engine.get_session("session-123")

            assert len(results) == 1
            mock_cur.execute.assert_called()


class TestEmbeddingFallback:
    """Test embedding failure fallback."""

    def test_embedding_failure_returns_zero_vector(self, test_config):
        """Test that embedding failure returns zero vector."""
        mock_embedding = MagicMock()
        mock_embedding.dimension = 768
        from context_engine.providers import EmbeddingError
        mock_embedding.embed.side_effect = EmbeddingError("Ollama not available")

        engine = ContextEngine(
            config=test_config,
            embedding_provider=mock_embedding,
            auto_init=False
        )

        result = engine._embed("test text")

        assert result == [0.0] * 768


class TestClose:
    """Test connection cleanup."""

    def test_close_connection(self, test_config, mock_embedding):
        """Test closing the connection."""
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_connect.return_value = mock_conn

            engine = ContextEngine(
                config=test_config,
                embedding_provider=mock_embedding,
                auto_init=False
            )

            # Access connection to create it
            _ = engine._get_conn()
            engine.close()

            mock_conn.close.assert_called_once()


class TestConfig:
    """Test configuration."""

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict('os.environ', {
            'CTX_DB_HOST': 'test-host',
            'CTX_DB_PORT': '5433',
            'CTX_NAMESPACE': 'test-ns',
            'CTX_DB_NAME': 'test-db',
            'CTX_DB_USER': 'test-user',
            'CTX_DB_PASS': 'test-pass',
            'CTX_CONFIG_PATH': '/nonexistent/config.json',  # Skip config file
        }, clear=True):
            config = ContextEngineConfig()
            assert config.db_host == 'test-host'
            assert config.db_port == 5433
            assert config.namespace == 'test-ns'

    def test_conn_string_building(self, test_config):
        """Test that connection string is built correctly."""
        expected = (
            f"postgresql://{test_config.db_user}:{test_config.db_pass}"
            f"@{test_config.db_host}:{test_config.db_port}/{test_config.db_name}"
            f"?sslmode={test_config.db_sslmode}"
        )
        assert test_config.conn_string == expected

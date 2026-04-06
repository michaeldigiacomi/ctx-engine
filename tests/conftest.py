"""Pytest fixtures and configuration."""

import os
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_embedding():
    """Return a mock embedding provider that returns fixed vectors."""
    mock = MagicMock()
    mock.dimension = 768
    # Return a normalized vector (unit vector)
    mock.embed.return_value = [1.0] + [0.0] * 767
    return mock


@pytest.fixture
def test_namespace():
    """Return a test namespace to isolate test data."""
    return "test-namespace"


@pytest.fixture
def test_config(test_namespace):
    """Create a test configuration."""
    from context_engine.config import ContextEngineConfig

    return ContextEngineConfig(
        db_host=os.getenv("CTX_DB_HOST", "localhost"),
        db_port=int(os.getenv("CTX_DB_PORT", "5432")),
        db_name=os.getenv("CTX_DB_NAME", "context_engine"),
        db_user=os.getenv("CTX_DB_USER", ""),
        db_pass=os.getenv("CTX_DB_PASS", ""),
        db_sslmode=os.getenv("CTX_DB_SSLMODE", "disable"),
        ollama_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        namespace=test_namespace,
    )


@pytest.fixture(scope="session")
def postgres_available():
    """Check if PostgreSQL with pgvector is available for integration tests."""
    import psycopg2

    try:
        conn_string = (
            f"postgresql://{os.getenv('CTX_DB_USER', '')}:{os.getenv('CTX_DB_PASS', '')}"
            f"@{os.getenv('CTX_DB_HOST', 'localhost')}:{os.getenv('CTX_DB_PORT', '5432')}"
            f"/{os.getenv('CTX_DB_NAME', 'context_engine')}"
            f"?sslmode={os.getenv('CTX_DB_SSLMODE', 'disable')}"
        )
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()

        # Check if pgvector extension exists
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        has_vector = cur.fetchone() is not None

        cur.close()
        conn.close()

        return has_vector
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama is available for embedding tests."""
    import requests

    try:
        url = os.getenv("CTX_OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# Testing pgvector-context-engine

This directory contains tests for the PGVector Context Engine.

## Test Structure

- **test_unit.py** - Unit tests with mocked dependencies (database, embedding provider)
- **test_cli.py** - CLI argument parsing and output tests
- **test_integration.py** - Integration tests requiring PostgreSQL with pgvector
- **conftest.py** - Pytest fixtures and configuration

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

### Run Unit Tests Only

```bash
pytest tests/test_unit.py tests/test_cli.py -v
# or
python run_tests.py --unit
```

### Run Integration Tests

Integration tests require:
- PostgreSQL server running
- pgvector extension installed
- Database credentials configured via environment variables

```bash
# Set up environment variables
export CTX_DB_HOST=localhost
export CTX_DB_PORT=5432
export CTX_DB_NAME=context_engine
export CTX_DB_USER=your_user
export CTX_DB_PASS=your_password

# Run integration tests
pytest tests/test_integration.py -v
# or
python run_tests.py --integration
```

### Run All Tests

```bash
pytest tests/ -v
# or
python run_tests.py
```

### Run with Coverage

```bash
pytest tests/ --cov=context_engine --cov-report=term-missing
# or
python run_tests.py --coverage
```

## Test Categories

### Unit Tests

Unit tests mock external dependencies:
- PostgreSQL database connections
- Embedding providers (Ollama/OpenAI)
- File system operations

These tests run quickly and don't require any external services.

### CLI Tests

CLI tests verify:
- Argument parsing
- Output formatting
- Error handling
- Command execution

### Integration Tests

Integration tests verify:
- Database schema creation
- CRUD operations
- Semantic search with real vectors
- Namespace isolation
- TTL/cleanup functionality
- Session management

These tests require a real PostgreSQL database with pgvector extension.

## Environment Variables for Testing

| Variable | Description | Default |
|----------|-------------|---------|
| `CTX_DB_HOST` | PostgreSQL host | localhost |
| `CTX_DB_PORT` | PostgreSQL port | 5432 |
| `CTX_DB_NAME` | Database name | context_engine |
| `CTX_DB_USER` | Database user | (empty) |
| `CTX_DB_PASS` | Database password | (empty) |
| `CTX_DB_SSLMODE` | SSL mode | disable |

## Writing Tests

### Unit Test Example

```python
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

        doc_id = engine.save(content="Test memory")
        assert doc_id is not None
```

### Integration Test Example

```python
@pytest.mark.integration
def test_save_and_retrieve(self, context_engine):
    """Test saving a memory and retrieving it."""
    doc_id = context_engine.save(
        content="Test memory",
        category="test"
    )

    memories = context_engine.list()
    assert len(memories) == 1
    assert memories[0]["content"] == "Test memory"
```

## Skipping Tests

Tests are automatically skipped when dependencies are unavailable:
- Integration tests skip if PostgreSQL is not available
- Tests requiring Ollama skip if Ollama is not running

"""
Database schema management for context engine.
"""

import psycopg2
from typing import Optional
from context_engine.config import ContextEngineConfig


class SchemaManager:
    """Manages database schema creation and migrations."""

    def __init__(self, config: ContextEngineConfig):
        self.config = config

    def _get_conn(self):
        """Get raw database connection without database name (for creation)."""
        # Connect to postgres default db to create our database
        conn_string = (
            f"postgresql://{self.config.db_user}:{self.config.db_pass}"
            f"@{self.config.db_host}:{self.config.db_port}/postgres"
            f"?sslmode={self.config.db_sslmode}"
        )
        return psycopg2.connect(conn_string)

    def _get_app_conn(self):
        """Get connection to the application database."""
        conn_string = self.config.conn_string
        return psycopg2.connect(conn_string)

    def ensure_database_exists(self) -> bool:
        """Create database if it doesn't exist. Returns True if created."""
        if self.config.db_name == "postgres":
            return False  # Won't create postgres itself

        try:
            conn = self._get_conn()
            conn.autocommit = True  # Need autocommit for CREATE DATABASE
            cur = conn.cursor()

            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config.db_name,)
            )
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{self.config.db_name}"')
                print(f"Created database: {self.config.db_name}")
                created = True
            else:
                created = False

            cur.close()
            conn.close()
            return created
        except psycopg2.Error as e:
            print(f"Error creating database: {e}")
            return False

    def ensure_schema(self, run_migrations: bool = True) -> bool:
        """
        Ensure schema is up to date.

        Args:
            run_migrations: If True, run migration files. If False, use inline schema.
        """
        if run_migrations:
            return self._run_migrations()
        else:
            return self._ensure_inline_schema()

    def _run_migrations(self) -> bool:
        """Run migration files from migrations directory."""
        try:
            conn = self._get_app_conn()
            cur = conn.cursor()

            # Create migrations tracking table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _schema_migrations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Get list of applied migrations
            cur.execute("SELECT name FROM _schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

            # Find migration files
            import os
            migrations_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "migrations"
            )
            migrations_dir = os.path.normpath(migrations_dir)

            if os.path.isdir(migrations_dir):
                migration_files = sorted([
                    f for f in os.listdir(migrations_dir)
                    if f.endswith(".sql")
                ])
            else:
                migration_files = []

            # Apply missing migrations
            for mf in migration_files:
                if mf in applied:
                    continue

                with open(os.path.join(migrations_dir, mf)) as f:
                    sql = f.read()

                cur.execute(sql)
                cur.execute(
                    "INSERT INTO _schema_migrations (name) VALUES (%s)",
                    (mf,)
                )
                print(f"Applied migration: {mf}")

            conn.commit()
            cur.close()
            conn.close()
            return True

        except psycopg2.Error as e:
            print(f"Migration error: {e}")
            return False

    def _ensure_inline_schema(self) -> bool:
        """Ensure minimal schema exists using inline SQL (fallback)."""
        try:
            conn = self._get_app_conn()
            cur = conn.cursor()

            # Check if table exists
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name = 'memories'
            """)

            if not cur.fetchone():
                # Create minimal schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id SERIAL PRIMARY KEY,
                        doc_id VARCHAR(64) UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR(768),
                        namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                        category VARCHAR(50) NOT NULL DEFAULT 'general',
                        source VARCHAR(50),
                        filename VARCHAR(255),
                        importance FLOAT DEFAULT 1.0,
                        tags TEXT[],
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        session_key VARCHAR(64),
                        conversation_chain INTEGER[],
                        expires_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_namespace
                        ON memories (namespace)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding_cosine
                        ON memories USING ivfflat (embedding vector_cosine_ops)
                """)

                print("Created memories table")
                conn.commit()

            cur.close()
            conn.close()
            return True

        except psycopg2.Error as e:
            print(f"Schema error: {e}")
            return False

    def verify_connection(self) -> tuple[bool, Optional[str]]:
        """
        Verify database connection works.

        Returns:
            (success, error_message)
        """
        try:
            conn = self._get_app_conn()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return True, None
        except psycopg2.Error as e:
            return False, str(e)

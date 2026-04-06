#!/bin/bash
# Context Engine Setup Script
#
# This sets up the database and config file for Context Engine.
#
# Prerequisites:
#   - PostgreSQL with pgvector extension
#   - Ollama running with nomic-embed-text model
#
# Usage:
#   ./setup_context_engine.sh [namespace]

set -e

NAMESPACE="${1:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/context_engine"

echo "Setting up Context Engine..."
echo "Namespace: $NAMESPACE"
echo "Config dir: $CONFIG_DIR"

# Create config directory
mkdir -p "$CONFIG_DIR"

# Prompt for database credentials
read -p "Database host [localhost]: " DB_HOST
DB_HOST="${DB_HOST:-localhost}"

read -p "Database port [5432]: " DB_PORT
DB_PORT="${DB_PORT:-5432}"

read -p "Database name [context_engine]: " DB_NAME
DB_NAME="${DB_NAME:-context_engine}"

read -p "Database user: " DB_USER
read -s -p "Database password: " DB_PASS
echo ""

read -p "Ollama URL [http://localhost:11434]: " OLLAMA_URL
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Write config file
cat > "$CONFIG_DIR/config.json" << EOF
{
  "db_host": "$DB_HOST",
  "db_port": $DB_PORT,
  "db_name": "$DB_NAME",
  "db_user": "$DB_USER",
  "ollama_url": "$OLLAMA_URL",
  "embedding_model": "nomic-embed-text",
  "namespace": "$NAMESPACE"
}
EOF

echo ""
echo "Config written to $CONFIG_DIR/config.json"
echo ""

# Export env vars for testing
export CTX_DB_HOST="$DB_HOST"
export CTX_DB_PORT="$DB_PORT"
export CTX_DB_NAME="$DB_NAME"
export CTX_DB_USER="$DB_USER"
export CTX_DB_PASS="$DB_PASS"
export CTX_NAMESPACE="$NAMESPACE"

# Test connection and initialize schema
echo "Testing connection and initializing schema..."
cd "$SCRIPT_DIR/.."
pip install -e . --quiet 2>/dev/null || true

python3 -c "
from context_engine import ContextEngine
ctx = ContextEngine()
print('Connection successful!')
ctx._ensure_initialized()
print('Schema initialized.')
ctx.close()
"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set environment variables in your shell:"
echo "     export CTX_DB_HOST=$DB_HOST"
echo "     export CTX_DB_PORT=$DB_PORT"
echo "     export CTX_DB_NAME=$DB_NAME"
echo "     export CTX_DB_USER=$DB_USER"
echo "     export CTX_DB_PASS='$DB_PASS'"
echo "     export CTX_NAMESPACE=$NAMESPACE"
echo ""
echo "  2. Or use the config file at: $CONFIG_DIR/config.json"
echo ""
echo "  3. Try it:"
echo "     ctx-engine save 'My first memory' --category test"
echo "     ctx-engine search 'first memory'"

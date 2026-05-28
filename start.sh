#!/usr/bin/env bash
set -e

# Start the Ollama server in the background.
ollama serve &

# Wait for it to accept connections before the API starts routing.
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done

# Cloud Run injects $PORT (default 8080). The container must listen on it.
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}"

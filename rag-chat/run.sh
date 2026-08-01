#!/usr/bin/env bash
# Convenience script to run both services locally.
# Requires: Java 17+, Maven, Python 3.10+
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] Setting up Python venv..."
cd "$ROOT/pyrag"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo "[2/3] Starting Python RAG microservice on :5005 ..."
nohup uvicorn rag_service:app --host 0.0.0.0 --port 5005 > /tmp/pyrag.log 2>&1 &
PYRAG_PID=$!
echo "  pyrag PID=$PYRAG_PID  (logs: /tmp/pyrag.log)"

echo "  waiting for RAG service to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5005/health >/dev/null 2>&1; then
    echo "  RAG service is ready."
    break
  fi
  sleep 2
done

echo "[3/3] Starting Spring Boot on :8080 ..."
cd "$ROOT"
trap "echo 'Stopping pyrag...'; kill $PYRAG_PID 2>/dev/null || true" EXIT
mvn -q spring-boot:run

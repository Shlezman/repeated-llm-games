#!/usr/bin/env bash
# Run the full local stack (Postgres + app) with docker compose.
# Usage: scripts/run-local.sh [config_path]
#   default config: config/runs/llm_gw_multi.yaml
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run: cp .env.example .env  and fill in the gateway key/url/models." >&2
  exit 1
fi

CONFIG="${1:-config/runs/llm_gw_multi.yaml}"

# Write results as the host user so the bind-mounted ./results stays writable.
# (UID is preset and readonly in bash, so export it rather than reassign.)
mkdir -p results
export UID
GID="$(id -g)"; export GID

echo ">> Building images (uv-based app + TLS Postgres)..."
docker compose build

echo ">> Running study with config: ${CONFIG}"
# Brings up the db dependency (healthcheck-gated), runs the study, then removes the app container.
docker compose run --rm app run --config "${CONFIG}"

echo ">> Done. Results written to ./results/"
echo ">> Stop the database with: docker compose down        (add -v to also drop the cache volume)"

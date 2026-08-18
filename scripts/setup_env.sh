#!/usr/bin/env bash
# Create virtual environment and install dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Run tests: pytest tests/"
echo "Run live pipeline: python -m src.inference.live_pipeline --config configs/pipeline_config.yaml"

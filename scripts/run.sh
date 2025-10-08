#!/usr/bin/env bash
set -euo pipefail

QUERY=${1:-"openai gpt-5"}
LIMIT=${2:-100}
FORMAT=${3:-csv}
OUT=${4:-"outputs/output.${FORMAT}"}

PY_BIN=${PYTHON:-python}
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=${PY_FALLBACK:-python3}
fi
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=/usr/bin/python3
fi

export PYTHONPATH=${PYTHONPATH:-"$(pwd)"}

mkdir -p "$(dirname "$OUT")"

"$PY_BIN" -m twitter_sentiment.cli fetch --query "$QUERY" --limit "$LIMIT" \
  | "$PY_BIN" -m twitter_sentiment.cli analyze --analyzer vader \
  | "$PY_BIN" -m twitter_sentiment.cli export --format "$FORMAT" --output "$OUT"
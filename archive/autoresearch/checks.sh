#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Quiet on success, loud on failure: only the tail is fed back to the agent.
uv run pytest -q 2>&1 | tail -30

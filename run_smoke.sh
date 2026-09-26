#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="$ROOT_DIR/cache/xdg"
export XDG_DATA_HOME="$ROOT_DIR/data"
export XDG_CONFIG_HOME="$ROOT_DIR/config"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
MODE="${1:-fixed}"
case "$MODE" in fixed|free) ;; *) echo 'usage: run_smoke.sh fixed|free [output-directory]' >&2; exit 2;; esac
OUTPUT_DIR="${2:-$ROOT_DIR/results/manual_${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/isaac_smoke.py" --mode "$MODE" --duration 4 --output "$OUTPUT_DIR" > "$OUTPUT_DIR/run.log" 2>&1
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/verify_results.py" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/verification.json"
echo "Video: $OUTPUT_DIR/simulation.mp4"

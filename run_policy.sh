#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$ROOT_DIR/cache/xdg" XDG_DATA_HOME="$ROOT_DIR/data" XDG_CONFIG_HOME="$ROOT_DIR/config"
unset PYTHONPATH
OUTPUT_DIR="${1:-$ROOT_DIR/results/policy_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/isaac_smoke.py" --mode policy --duration 8 --policy-dir "$ROOT_DIR/policy" --output "$OUTPUT_DIR" > "$OUTPUT_DIR/run.log" 2>&1
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/verify_results.py" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/verification.json"
echo "Video: $OUTPUT_DIR/simulation.mp4"

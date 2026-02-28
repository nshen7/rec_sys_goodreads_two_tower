#!/usr/bin/env bash
# Run training in the background from the two_tower/ project root.
# Usage: bash notebooks/run_train.sh  (from anywhere inside the repo)

set -euo pipefail

# Resolve two_tower/ directory regardless of where this script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

EXP="two_tower_baseline_$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="logs/$EXP"
mkdir -p "$LOG_DIR"

nohup micromamba run -n fin_sentiment python notebooks/03_train.py > "$LOG_DIR/train.log" 2>&1 &
echo $! > "$LOG_DIR/train.pid"
echo "Started training — PID $(cat "$LOG_DIR/train.pid")"
echo "Experiment: $EXP"
echo "Tail logs:  tail -f $PROJECT_DIR/$LOG_DIR/train.log"
echo "Check alive: ps -p \$(cat $PROJECT_DIR/$LOG_DIR/train.pid)"
echo "Stop:        kill \$(cat $PROJECT_DIR/$LOG_DIR/train.pid)"

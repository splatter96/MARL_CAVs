#!/usr/bin/env bash
# -------------------------------------------------------------
# run_wandb_agents.sh
#
# Usage:
#   ./run_wandb_agents.sh <NUM_AGENTS>
#
# Example (launch 8 agents):
#   ./run_wandb_agents.sh 8
#
# Each agent runs the command:
#   wandb agent paul-auerbach-barkhausen-institut-ggmbh/weaving/weags4o5
#
# The script creates a sub‑directory “wandb_logs” and stores a log
# file per agent: agent_01.log, agent_02.log, …
# -------------------------------------------------------------

set -euo pipefail   # fail fast on errors, undefined vars, etc.

# ----------------------  Argument check  ----------------------
if [[ $# -ne 1 ]]; then
    echo "Error: Exactly one argument required – the number of agents to launch."
    echo "Usage: $0 <NUM_AGENTS>"
    exit 1
fi

NUM_AGENTS=$1

# Validate that it is a positive integer
if ! [[ "$NUM_AGENTS" =~ ^[0-9]+$ ]] || (( NUM_AGENTS == 0 )); then
    echo "Error: <NUM_AGENTS> must be a positive integer (got '$NUM_AGENTS')."
    exit 1
fi

# ----------------------  Settings  ---------------------------
# WandB command – keep it in a variable for easy editing
WANDB_CMD="wandb agent paul-auerbach-barkhausen-institut-ggmbh/weaving/weags4o5"

# Directory for log files
LOG_DIR="wandb_logs"
mkdir -p "$LOG_DIR"

# Optional: set a nice name for the process group (helps with `ps`/`top`)
echo $$ > "$LOG_DIR/launcher_pid"

# ----------------------  Launch loop  -----------------------
echo "Launching $NUM_AGENTS WandB agent(s)…"

for (( i=1; i<=NUM_AGENTS; i++ )); do
    # Zero‑pad the index for nicer sorting of log files
    IDX=$(printf "%02d" "$i")
    LOG_FILE="${LOG_DIR}/agent_${IDX}.log"

    # Start the agent in background, redirect stdout+stderr to its log
    # `nohup` prevents termination if you close the terminal later
    # `setsid` detaches the process from the controlling terminal
    # The trailing `&` puts it in the background.
    nohup setsid bash -c "$WANDB_CMD" >"$LOG_FILE" 2>&1 &

    # Capture the PID so we can optionally wait or kill later
    AGENT_PID=$!
    echo "  → agent $IDX started (PID=$AGENT_PID), logs → $LOG_FILE"
done

echo "All $NUM_AGENTS agents are now running in background."
echo "Check their output with e.g.:"
echo "  tail -f ${LOG_DIR}/agent_01.log"
echo "To stop them later, you can kill the whole group:"
echo "  pkill -P $$   # kills all children of this launcher script"
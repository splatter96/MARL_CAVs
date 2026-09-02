#!/usr/bin/env bash

# -----------------------------------------------------------------
# Parallel training launcher
# -----------------------------------------------------------------
# Place this script in the same directory as train_new.py (or
# adjust SCRIPT_DIR below).  It will start ten training processes,
# each with a different seed, and run them concurrently.
# -----------------------------------------------------------------

# Experiment tag (same for every run)
EXP_TAG="weaving_seed_test"

# List of seeds to use – change or extend as you like
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Keep track of background PIDs so we can wait for them later
pids=()

echo "Launching ${#SEEDS[@]} training jobs in parallel …"

for SEED in "${SEEDS[@]}"; do
    echo "▶️  Starting seed=$SEED ..."
    # Start the process in the background
    python3 train_new.py logging.exp_tag="${EXP_TAG}" seed="${SEED}" &
    sleep 2s
    # Record its PID
    pids+=($!)
done

# Wait for all background jobs to finish
for pid in "${pids[@]}"; do
    wait "$pid"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "⚠️  Job with PID $pid (seed $SEED) exited with status $rc"
    fi
done

echo "✅ All training runs have completed."

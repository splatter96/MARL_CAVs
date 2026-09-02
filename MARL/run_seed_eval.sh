#!/bin/bash

# Script to run eval.py with 10 different seeds in parallel

# Define seeds (you can customize these)
SEEDS=(42 100 200 300 400 500 600 700 800 900)

# Model path
MODEL="results/Jan_20_07_43_31_pixi-test-horse-new-env-longer_42/models/model_pixi-test-horse-new-env-longer_seed_42.zip"

# Create output directory if it doesn't exist
mkdir -p eval_outputs

# Function to run evaluation for a single seed
run_eval() {
    local seed=$1
    local output_file="eval_outputs/eval_seed_${seed}.txt"
    echo "Running evaluation with seed ${seed}..."
    python3 eval.py "$MODEL" --num-runs 1000 --no-render --seed "$seed" > "$output_file" 2>&1
    echo "Finished evaluation with seed ${seed}. Output saved to ${output_file}"
}

# Export functions and variables so they can be used in subshells
export MODEL

# Run evaluations in parallel (up to 10 at once)
for seed in "${SEEDS[@]}"; do
    run_eval "$seed" &
done

# Wait for all background jobs to complete
wait

echo "All evaluations completed. Results are in eval_outputs/"

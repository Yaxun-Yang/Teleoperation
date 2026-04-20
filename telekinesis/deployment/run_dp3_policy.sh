#!/bin/bash

# DP3 Policy Deployment Script
# Usage: ./run_dp3_policy.sh [checkpoint_path]

# Set default checkpoint path if not provided
if [ $# -eq 0 ]; then
    CHECKPOINT_PATH="/media/yaxun/manipulation1/leaphandproject_ws/checkpoints/latest.ckpt"
else
    CHECKPOINT_PATH="$1"
fi

echo "Starting DP3 Policy Deployment..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Using device: cuda:0"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "cam_K.txt" ]; then
    echo "Error: cam_K.txt not found in current directory"
    exit 1
fi

if [ ! -f "rollout_dp3_policy_real_system.py" ]; then
    echo "Error: rollout_dp3_policy_real_system.py not found"
    exit 1
fi

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the DP3 policy deployment
python3 rollout_dp3_policy_real_system.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --device cuda:0

echo "DP3 Policy Deployment finished."

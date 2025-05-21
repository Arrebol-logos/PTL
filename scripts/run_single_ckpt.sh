#!/bin/bash

# Define the base project directory to use for absolute paths
PROJECT_DIR="/shared_data/zijian/Fast-Pruning"
cd $PROJECT_DIR

# Initial settings - using the original model directly
model="Arrebol-logos/Pruned-Qwen2.5-7B"
data_name="math12k"

echo "=== Starting Single Run ==="

echo "1. Running RL training"

# Change to EasyR1 directory
cd "$PROJECT_DIR/EasyR1"

# Set environment variables
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_MODE='offline'

echo "Starting training with model: $model, using dataset: $data_name"

# Execute RL training
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=$model \
    trainer.n_gpus_per_node=8 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_prune_rl_7b_${data_name} \
    trainer.save_checkpoint_path="$PROJECT_DIR/checkpoints/QWEN_PRUNE/qwen2_5_prnue_7b_${data_name}" \
    trainer.project_name=QWEN_PRUNE

echo "2. Processing all RL trained model checkpoints"
# Find all global_step directories
checkpoint_base_dir="$PROJECT_DIR/checkpoints/QWEN_PRUNE/qwen2_5_prune_7b_${data_name}"
if [ -d "$checkpoint_base_dir" ]; then
    # Find all global_step_* directories that contain an actor subdirectory
    echo "Searching for actor directories in $checkpoint_base_dir"
    actor_dirs=$(find "$checkpoint_base_dir" -type d -path "*/global_step_*/actor")
    
    if [ -z "$actor_dirs" ]; then
        echo "Warning: No actor directories found, using original model for evaluation"
        processed_model=$model
    else
        echo "Found $(echo "$actor_dirs" | wc -l) actor directories to process"
        
        # Process each actor directory
        for actor_dir in $actor_dirs; do
            echo "Processing model directory: $actor_dir"
            
            CUDA_VISIBLE_DEVICES=0,1 python3 scripts/model_merger.py --local_dir $actor_dir
            
            # Remove .pt files to save space
            rm -rf $actor_dir/*.pt
            echo "Removed PyTorch checkpoint files to save space"
        done
        
        # Get the latest global_step for evaluation
        latest_step_file="$checkpoint_base_dir/latest_global_step.txt"
        if [ -f "$latest_step_file" ]; then
            global_step=$(cat "$latest_step_file")
            echo "Using latest global_step: $global_step for evaluation"
            processed_model="$checkpoint_base_dir/global_step_${global_step}/actor/huggingface"
        else
            # If latest_global_step.txt doesn't exist, use the last processed directory
            processed_model=$(echo "$actor_dirs" | tail -n 1)"/huggingface"
        fi
        
        echo "Setting model for evaluation: $processed_model"
    fi
else
    echo "Warning: Checkpoint directory not found, using original model for evaluation"
    processed_model=$model
fi

# Return to original directory
cd "$PROJECT_DIR"

echo "3. Evaluating model"
# Switch to evaluation directory
cd "$PROJECT_DIR/evaluation_r1"

# Execute evaluation
echo "Evaluating model: $processed_model"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python eval_llm.py --save=True --model_name "$processed_model" --template "qwen_math"

# Return to original directory
cd "$PROJECT_DIR"

echo "Single run completed!"

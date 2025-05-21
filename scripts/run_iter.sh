#!/bin/bash

# Define the base project directory to use for absolute paths
PROJECT_DIR="/shared_data/zijian/Fast-Pruning"
cd $PROJECT_DIR

# Initial settings
initial_model="hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo"
samples=10
top_n=16000
prune_amount=1000
iterations=10

current_model=$initial_model

for i in $(seq 1 $iterations); do
    echo "=== Starting Iteration $i ==="
    
    # Set current iteration prefix/dataset name
    current_prefix="math_cot_$i"
    
    echo "1. Detecting neurons for English tasks"
    python neuron_detection_en_qwen.py english $current_model $samples $top_n
    
    echo "2. Detecting neurons for Math tasks"
    python neuron_detection_math_qwen.py --model $current_model --samples $samples --top_n $top_n --dataset $current_prefix
    
    echo "3. Deleting neurons"
    en_output="output_neurons/qwen/english/$top_n/$samples.txt"
    math_output="output_neurons/qwen/$current_prefix/math/$top_n/$samples.txt"
    
    python neuron_deletion_math_qwen.py $prune_amount $current_model $math_output $en_output
    
    # Update model path for RL training - using absolute path
    pruned_model="$PROJECT_DIR/pruned_model/qwen/$current_prefix/${prune_amount}_en_${samples}_${top_n}_math_${samples}_${top_n}"
    
    echo "4. Running RL training"
    # Set dataset name
    data_name="math12k_part$i"
    
    # Change to EasyR1 directory
    cd "$PROJECT_DIR/EasyR1"
    
    # Execute RL training
    export PYTHONUNBUFFERED=1
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export VLLM_USE_V1=0
    export WANDB_MODE='offline'
    
    echo "Starting training with model: $pruned_model, using dataset: $data_name"
    
    python3 -m verl.trainer.main \
        config=examples/config.yaml \
        worker.actor.model.model_path=$pruned_model \
        data.train_files="$PROJECT_DIR/data/${data_name}@train" \
        trainer.n_gpus_per_node=8 \
        worker.rollout.enable_chunked_prefill=false \
        trainer.experiment_name=qwen2_5_simple_rl_7b_${data_name} \
        trainer.save_checkpoint_path="$PROJECT_DIR/checkpoints/QWEN_PRUNE/qwen2_5_simplerl_7b_${data_name}" \
        trainer.project_name=QWEN_PRUNE
    
    echo "5. Processing the RL trained model"
    # Read latest global_step
    latest_step_file="$PROJECT_DIR/checkpoints/QWEN_PRUNE/qwen2_5_simplerl_7b_${data_name}/latest_global_step.txt"
    if [ -f "$latest_step_file" ]; then
        global_step=$(cat "$latest_step_file")
        echo "Retrieved latest global_step: $global_step"
        
        # Process the model
        actor_dir="$PROJECT_DIR/checkpoints/QWEN_PRUNE/qwen2_5_simplerl_7b_${data_name}/global_step_${global_step}/actor"
        echo "Processing model directory: $actor_dir"
        
        CUDA_VISIBLE_DEVICES=0,1 python3 scripts/model_merger.py --local_dir $actor_dir
        
        # Remove .pt files to save space
        rm -rf $actor_dir/*.pt
        echo "Removed PyTorch checkpoint files to save space"
        
        # Get processed model path for next iteration
        if [ $i -lt $iterations ]; then
            processed_model="$actor_dir/huggingface"
            current_model="$processed_model"
            echo "Setting model for next iteration: $current_model"
        fi
    else
        echo "Warning: Could not find latest_global_step.txt, using original pruned model for next iteration"
        if [ $i -lt $iterations ]; then
            current_model=$pruned_model
        fi
    fi
    
    # Return to original directory
    cd "$PROJECT_DIR"
    
    echo "6. Evaluating current model"
    # Switch to evaluation directory
    cd "$PROJECT_DIR/evaluation_r1"
    
    # Execute evaluation
    echo "Evaluating model: $current_model"
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python eval_llm.py --save=True --model_name "$current_model" --template "qwen_math"
    
    # Return to original directory
    cd "$PROJECT_DIR"
    
    # Prepare for next iteration
    if [ $i -lt $iterations ]; then
        top_n=$((top_n - prune_amount))
        echo "Next iteration top_n value: $top_n"
    fi
    
    echo ""
done

echo "All iterations completed!"

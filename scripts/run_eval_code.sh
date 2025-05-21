#!/bin/bash

cd ./lm-evaluation-harness

export HF_ALLOW_CODE_EVAL=1

print_usage() {
    echo "Usage: $0 [model_path] [tp] [dp]"
    echo "  model_path: path to the model (default: meta-llama/Meta-Llama-3-8B)"
    echo "  tp: tensor parallel size (default: 2)"
    echo "  dp: data parallel size (default: 1)"
    echo "Example: $0 meta-llama/Meta-Llama-3-70B 4 2"
    echo "Use -h or --help to show this message"
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Default values
model="meta-llama/Meta-Llama-3-8B"
tp=2
dp=1

# Override defaults if arguments provided
if [[ ! -z "$1" ]]; then
    model=$1
fi

if [[ ! -z "$2" ]]; then
    tp=$2
fi

if [[ ! -z "$3" ]]; then
    dp=$3
fi

echo "Running with model=$model, tp=$tp, dp=$dp"

lm_eval --model vllm \
    --model_args pretrained="$model",tensor_parallel_size=$tp,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$dp \
    --tasks humaneval,mbpp \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --output_path ../results

cd -

echo "Evaluation completed."
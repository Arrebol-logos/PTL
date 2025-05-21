#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_MODE='offline'

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=${DATA}@train \
    trainer.n_gpus_per_node=8 \

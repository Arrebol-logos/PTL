# Prepare for dataset

First, you need to follow the approach used in utils_function/transfer_numina.py to convert plain question-answer pairs into a usable training format. If you plan to use multiple datasets, refer to the method in utils_function/split_dataset.py to process and split the merged datasets. After that, adapt your training data according to the custom dataset format required by the llama-factory library.

## PTL

You can use ptl.yaml to create your environment.

This document outlines the full pipeline to detect, prune, fine-tune, and evaluate neurons in the **Gemma 2-9B** model using math and English datasets.

---

### ✅ Step 1: Detect English-Sensitive Neurons (To Keep)

```bash
# Usage:
python neuron_detection_en_gemma.py <language_file> <model_path> <sample_count> <top_n>

# Example:
CUDA_VISIBLE_DEVICES=4,6 python neuron_detection_en_gemma.py english \
google/gemma-2-9b \
1000 12000
```

- Detects neurons important for English tasks.
- These neurons **must not be deleted**.

---

### ✅ Step 2: Detect Math-Specific Neurons (Candidates for Deletion)

```bash
# Usage:
python neuron_detection_math_gemma.py <model_path> <dataset_split> <sample_count> <top_n> <dataset_flag>

# Example:
CUDA_VISIBLE_DEVICES=4,6 python neuron_detection_math_gemma.py \
google/gemma-2-9b \
train 10 12000 1
```

- Detects neurons used in math tasks.
- These are **potentially removable**, unless also used in English.
- ⚠️ Make sure the correct math dataset is being used.

---

### ✅ Step 3: Prune Neurons or Layers

```bash
# Usage: Prune Neurons
python neuron_deletion_math_gemma.py <delete_count> <model_path> <math_neuron_file> <english_neuron_file>

# Example:
CUDA_VISIBLE_DEVICES=4,6 python neuron_deletion_math_gemma.py 1000 \
google/gemma-2-9b \
output_neurons/gemma2/math_cot_1/math/12000/10.txt \
output_neurons/gemma2/english/12000/999.txt
```

- Removes `delete_count` neurons that are math-specific and **not used in English**.
- Saves a new pruned model checkpoint.

```bash
# Usage: Prune Layers
python layer_deletion_gemma.py <layers_to_remove> <model_path>

# Example:
CUDA_VISIBLE_DEVICES=4,6 python layer_deletion_gemma.py "41" google/gemma-2-9b
```

- Saves a new pruned model checkpoint.

---


### ✅ Step 4: Fine-tune the Pruned Model on Math Dataset

```bash
# Example:
CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train \
examples/train_lora/llama3_pretrain_1000_mixed_1.yaml
```

- Fine-tune using math-specific data (e.g., NuminaMath, MetaMathQA).
- Make sure the YAML config includes:
  - Path to the pruned model
  - Training dataset
  - LoRA or full fine-tuning settings
- Before starting a new training session, ensure that special_tokens_map.json and tokenizer_config.json in your target model directory are replaced with those from the original model (e.g., google/gemma-2-9b).

---

### ✅ Step 5: Evaluate the Pruned Model on Math Benchmarks

```bash
# Example:
CUDA_VISIBLE_DEVICES=4,5 bash ./scripts/run_eval_math.sh \
/disk/pruned_model/llama/math_cot_1/1000_en_1000_12000_math_10_12000_pretrained/checkpoint-2503
```

- Runs evaluation using scripts (e.g., GSM8K, MATH, or custom CoT datasets).
- Make sure the evaluation script:
  - Loads the correct model/tokenizer
  - Uses math benchmark datasets
  - Outputs metrics like accuracy or perplexity

---


## Evaluation

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate model performance on various tasks including code generation and mathematical reasoning.

### Setup

```bash
cd ./lm-evaluation-harness
conda create -n lm-eval python=3.10.16 -y
conda activate lm-eval
pip install -e .
pip install lm_eval[vllm]  # for VLLM backend acceleration
pip install lm_eval[math]  # for math evaluation tasks
```

### Evaluation Scripts

We provide scripts for evaluating code (`run_eval_code.sh`) and math (`run_eval_math.sh`) tasks:

```bash
# Usage: ./scripts/run_eval_[code|math].sh [model_path] [tp] [dp]
#   model_path: path to the model (default: meta-llama/Meta-Llama-3-8B)
#   tp: tensor parallel size (default: 2)
#   dp: data parallel size (default: 1)

# Examples:
./scripts/run_eval_code.sh                                # Run with default settings
./scripts/run_eval_code.sh meta-llama/Meta-Llama-3-70B    # Evaluate a different model
./scripts/run_eval_math.sh local/my-model 4 2             # Custom model with tp=4, dp=2
```

Note: If you want to evaluate Gemma2, you need to add "add_bos_token=True" in line 42 of scripts/run_eval_math(or code).sh
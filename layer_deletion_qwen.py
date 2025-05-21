import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config

def main(argv):
    # Parse arguments
    if len(argv) < 2:
        print("Usage: python layer_deletion.py <layers_to_remove> <model_path>")
        sys.exit(1)
    layers_to_remove = list(map(int, argv[0].split(',')))  # Input format: "24,26"
    model_path = argv[1]
    # Construct suffix string for removed layers
    layer_suffix = '_'.join(map(str, layers_to_remove))

    # Load the model and config (in bfloat16 format)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Set torch_dtype to bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    config = Qwen2Config.from_pretrained(model_path)

    # Validate layer indices
    max_layer = config.num_hidden_layers - 1
    for layer in layers_to_remove:
        if layer < 0 or layer > max_layer:
            raise ValueError(f"Invalid layer index {layer}. Model has {config.num_hidden_layers} layers (valid range: 0-{max_layer}).")

    # Deduplicate and sort layers
    layers_to_remove = sorted(list(set(layers_to_remove)))

    # Remove specified layers and update model structure
    original_layers = model.model.layers
    new_layers = [layer for idx, layer in enumerate(original_layers) if idx not in layers_to_remove]
    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    # Ensure all weights are bfloat16 (double check)
    model = model.to(torch.bfloat16)

    # Build save path, support two input path formats
    original_path = Path(model_path)
    parts = list(original_path.parts)

    # Case 1: Initial pruning, path contains "pruned_model"
    if "pruned_model" in parts:
        # Replace with "layer_removed_model"
        new_parts = [p.replace("pruned_model", "layer_removed_model") for p in parts]
        # Remove checkpoint directory
        base_parts = new_parts[:-1]
        save_path = Path(*base_parts) / layer_suffix
    # Case 2: Further pruning, path already contains "layer_removed_model"
    elif "layer_removed_model" in parts:
        new_parts = parts
        # Remove previous pruning directory and checkpoint
        base_parts = new_parts[:-2]
        prev_pruned_dir = new_parts[-2]
        new_pruned_dir = f"{prev_pruned_dir}_{layer_suffix}"
        save_path = Path(*base_parts) / new_pruned_dir
    else:
        # Other cases: treat as initial pruning
        base_parts = parts[:-1]
        save_path = Path(*base_parts) / layer_suffix

    # Create directory and save model
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}, stored in bfloat16 format.")

if __name__ == "__main__":
    main(sys.argv[1:])

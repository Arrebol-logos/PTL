import os
import sys
import torch
import random
import re
import itertools
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

random.seed(112)

def retrive_neuron(filename):
    """
    Read neuron data from file. Each line is a Python-evaluable expression.
    Returns a list where each element is typically a dictionary with multiple parts
    (e.g., fwd_up, fwd_down, q, k, v), each part being a dict mapping layer index (0~31) to a set of neuron indices.
    """
    activate_neuron = []
    with open(filename, 'r') as file:
        for line in file:
            neuron = eval(line.strip())
            activate_neuron.append(neuron)
    return activate_neuron

def deduplicate(neuron_target, neuron_delete):
    """
    For each part (0: fwd_up, 1: fwd_down, 2: q, 3: k, 4: v),
    remove the corresponding neurons in neuron_delete from neuron_target.
    """
    for part in range(5):
        for key in range(32):
            neuron_target[part][key] = neuron_target[part][key] - neuron_delete[part][key]
    return neuron_target

def remove_ffn(neurons, number):
    """
    For each layer's neuron set to be pruned,
    convert it to a sorted list and select the first `number` as removable indices.
    Returns a dict: {layer: set of neuron indices to remove}.
    """
    new_neurons = {}
    for idx in range(32):
        sorted_list = sorted(neurons[idx])
        removal = set(sorted_list[:min(number, len(sorted_list))])
        new_neurons[idx] = removal
    return new_neurons

def sync_remove(neuron_sets):
    """
    Merge deletion sets from multiple parts into a union to ensure synchronized pruning.
    """
    synchronized_removals = {}
    for layer in range(32):
        synchronized_removals[layer] = set()
        for neuron_set in neuron_sets:
            synchronized_removals[layer] |= neuron_set.get(layer, set())
    return synchronized_removals

def main(argv):
    feed_forward_remove = int(argv[0])
    code_neuron_address = argv[1]
    english_neuron_address = argv[2]

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    config = LlamaConfig.from_pretrained(model_name)

    # Freeze model parameters to prevent accidental updates
    for param in model.parameters():
        param.requires_grad = False

    # Get model parameter dictionary
    params_model = dict(model.named_parameters())

    # Load neuron data (each file is assumed to contain a list of dictionaries)
    code_neuron = retrive_neuron(code_neuron_address)
    english_neuron = retrive_neuron(english_neuron_address)

    # Subtract English neurons from code neurons to get target deactivation set
    deactivate_neuron = deduplicate(code_neuron, english_neuron)
        
    # Merge deletion sets across parts to ensure synchronized pruning
    synchronized_deletions = sync_remove([deactivate_neuron[0], deactivate_neuron[1], deactivate_neuron[2]])
    synchronized_deletions = remove_ffn(synchronized_deletions, feed_forward_remove)

    # Update config.intermediate_size, assuming original size is config.intermediate_size
    orig_inter_size = config.intermediate_size
    new_inter_size = orig_inter_size - feed_forward_remove
    if new_inter_size <= 0:
        raise ValueError("Parameter `feed_forward_remove` is too large: resulting intermediate size must be > 0.")
    config.intermediate_size = new_inter_size

    # Traverse model parameters and prune MLP layers
    for name, param in tqdm(params_model.items()):
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer = int(match.group(1))
            # Row pruning for gate_proj and up_proj
            if 'mlp.gate_proj.weight' in name or 'mlp.up_proj.weight' in name:
                removal = sorted(list(synchronized_deletions.get(layer, set())))
                print(f"Layer {layer} {name} will remove {len(removal)} rows")
                kept_rows = [i for i in range(param.size(0)) if i not in removal]
                if len(kept_rows) != new_inter_size:
                    print(f"Warning: Layer {layer} in {name} has {len(kept_rows)} rows retained, expected {new_inter_size}.")
                params_model[name] = param[kept_rows, :]
            # Column pruning for down_proj
            if 'mlp.down_proj.weight' in name:
                removal = sorted(list(synchronized_deletions.get(layer, set())))
                print(f"Layer {layer} {name} will remove {len(removal)} columns")
                kept_cols = [i for i in range(param.size(1)) if i not in removal]
                if len(kept_cols) != new_inter_size:
                    print(f"Warning: Layer {layer} in {name} has {len(kept_cols)} columns retained, expected {new_inter_size}.")
                params_model[name] = param[:, kept_cols]
                
    # Extract numbers from path
    def extract_number(filepath):
        match = re.search(r"/(\d+)/(\d+)\.txt$", filepath)
        return match.group(1), match.group(2) if match else "0"
    
    code_top_n, code_num = extract_number(code_neuron_address)
    english_top_n, english_num = extract_number(english_neuron_address)

    # Rebuild model and load updated parameters
    DATASET_NAME = 'python_mixed'
    pruned_model_path = os.path.join(
        "/disk/pruned_model",
        model_name.replace("/", "_"),
        DATASET_NAME.replace("/", "_"),
        f"{feed_forward_remove}_en_{english_num}_{english_top_n}_code_{code_num}_{code_top_n}/"
    )
    os.makedirs(pruned_model_path, exist_ok=True)
    pruned_model = LlamaForCausalLM(config)
    pruned_model.load_state_dict(params_model, strict=True)

    pruned_model = pruned_model.to(torch.bfloat16)
    pruned_model.save_pretrained(pruned_model_path)
    # Save tokenizer and config
    tokenizer.save_pretrained(pruned_model_path)

    print(f"Pruned model saved successfully to {pruned_model_path}")
    # pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv[1:])
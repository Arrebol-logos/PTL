import os
import sys
import torch
import random
import re
import itertools
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma2Config, Gemma2ForCausalLM

random.seed(112)

def retrive_neuron(filename):
    """
    Read neuron data from file. Each line is a Python expression that can be evaluated.
    Returns a list, where each element is typically a dictionary with multiple parts 
    (e.g., fwd_up, fwd_down, q, k, v), each part being a dict with layer index (0~41) as key 
    and a set as value.
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
    remove the corresponding neuron sets in neuron_delete from neuron_target.
    """
    for part in range(5):
        for key in range(42):
            neuron_target[part][key] = neuron_target[part][key] - neuron_delete[part][key]
    return neuron_target

def remove_ffn(neurons, number):
    """
    For each layer's neuron set to be pruned,
    convert it to a sorted list, then select the first `number` as the neurons to remove.
    Returns a dictionary: layer index -> set of neuron indices to remove.
    """
    new_neurons = {}
    for idx in range(42):
        sorted_list = sorted(neurons[idx])
        removal = set(sorted_list[:min(number, len(sorted_list))])
        new_neurons[idx] = removal
    return new_neurons

def sync_remove(neuron_sets):
    """
    Merge deletion sets from multiple parts into a union set to ensure synchronized pruning.
    """
    synchronized_removals = {}
    for layer in range(42):
        synchronized_removals[layer] = set()
        for neuron_set in neuron_sets:
            synchronized_removals[layer] |= neuron_set.get(layer, set())
    return synchronized_removals

def main(argv):
    feed_forward_remove = int(argv[0])
    model_name = argv[1]
    math_neuron_address = argv[2]
    english_neuron_address = argv[3]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma2ForCausalLM.from_pretrained(model_name, 
                                              device_map="auto",
                                              attn_implementation="eager",
                                              torch_dtype=torch.bfloat16)
    config = Gemma2Config.from_pretrained(model_name)
    config._attn_implementation = "eager"

    # Freeze model parameters to prevent accidental updates
    for param in model.parameters():
        param.requires_grad = False
    
    # Load original model state dict
    orig_sd = model.state_dict()

    # Load neuron data (each file contains a list of dictionaries)
    math_neuron = retrive_neuron(math_neuron_address)
    english_neuron = retrive_neuron(english_neuron_address)

    # Subtract English neurons from math neurons to get target deactivation set
    deactivate_neuron = deduplicate(math_neuron, english_neuron)
        
    # Merge deletion sets across parts to ensure synchronized pruning
    synchronized_deletions = sync_remove([deactivate_neuron[0], deactivate_neuron[1], deactivate_neuron[2]])
    synchronized_deletions = remove_ffn(synchronized_deletions, feed_forward_remove)

    # Update config.intermediate_size, assuming original size is config.intermediate_size
    orig_inter_size = config.intermediate_size
    new_inter_size = orig_inter_size - feed_forward_remove
    if new_inter_size <= 0:
        raise ValueError("feed_forward_remove is too large, resulting intermediate size must be > 0!")
    config.intermediate_size = new_inter_size
    config.torch_dtype = "bfloat16"

    # Prune MLP layers by modifying gate_proj, up_proj (rows), and down_proj (columns)
    for name, param in tqdm(orig_sd.items()):
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer = int(match.group(1))
            # Row pruning for gate_proj and up_proj
            if 'mlp.gate_proj.weight' in name or 'mlp.up_proj.weight' in name:
                removal = sorted(list(synchronized_deletions.get(layer, set())))
                print(f"Layer {layer} {name} will remove {len(removal)} rows")
                kept_rows = [i for i in range(param.size(0)) if i not in removal]
                if len(kept_rows) != new_inter_size:
                    print(f"Warning: Layer {layer} in {name} has {len(kept_rows)} rows retained, expected {new_inter_size}")
                orig_sd[name] = param[kept_rows, :]
            # Column pruning for down_proj
            if 'mlp.down_proj.weight' in name:
                removal = sorted(list(synchronized_deletions.get(layer, set())))
                print(f"Layer {layer} {name} will remove {len(removal)} columns")
                kept_cols = [i for i in range(param.size(1)) if i not in removal]
                if len(kept_cols) != new_inter_size:
                    print(f"Warning: Layer {layer} in {name} has {len(kept_cols)} columns retained, expected {new_inter_size}")
                orig_sd[name] = param[:, kept_cols]
                
    # Extract numbers from file path
    def extract_number(filepath):
        match = re.search(r"/(\d+)/(\d+)\.txt$", filepath)
        return match.group(1), match.group(2) if match else "0"
    
    dataset = re.search(r"(?<=output_neurons/gemma2/)[^/]+", math_neuron_address).group(0) 
    math_top_n, math_num = extract_number(math_neuron_address)
    english_top_n, english_num = extract_number(english_neuron_address)

    # Reconstruct a new model with pruned config and load modified parameters
    pruned_model_path = os.path.join(
        "/disk/pruned_model",
        "gemma2",
        dataset,
        f"{feed_forward_remove}_en_{english_num}_{english_top_n}_math_{math_num}_{math_top_n}/"
    )
    
    print(config)
    
    os.makedirs(pruned_model_path, exist_ok=True)
    pruned_model = Gemma2ForCausalLM(config)
    pruned_model.load_state_dict(orig_sd, strict=True)

    pruned_model = pruned_model.to(torch.bfloat16)
    pruned_model.save_pretrained(pruned_model_path)
    # Save tokenizer and config
    tokenizer.save_pretrained(pruned_model_path)

    print(f"Pruned model saved successfully to {pruned_model_path}")
    # pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv[1:])
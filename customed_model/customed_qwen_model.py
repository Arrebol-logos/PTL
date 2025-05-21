from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch
import re
from typing import Optional, Dict, List, Tuple, Any
import contextlib

class CustomedQwenForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @contextlib.contextmanager
    def register_hooks(self, candidate_layers: Optional[List[int]], top_n: Optional[int], largest: bool):
        handles = []
        
        def extract_layer_idx(name: str) -> Optional[int]:
            match = re.search(r"layers\.(\d+)\.", name)
            return int(match.group(1)) if match else None

        def hook_fn_factory(container: Dict[int, set], module_name: str, is_down_proj: bool):
            def hook_fn(module, input, output):
                layer_idx = extract_layer_idx(module_name)
                if layer_idx is None:
                    return
                try:
                    hidden = input[0] if is_down_proj else output
                    if hidden.dim() != 3:
                        return
                    last_token_hidden = hidden[:, -1, :]
                    
                    if top_n is not None:
                        values, indices = torch.topk(last_token_hidden, k=top_n, dim=1, largest=largest)
                        batch_size = indices.size(0)
                        if batch_size > 1:
                            sample_sets = [set(indices[i].tolist()) for i in range(batch_size)]
                            common_topn = set.intersection(*sample_sets)
                        else:
                            common_topn = set(indices[0].tolist())
                        container[layer_idx] = common_topn
                    else:
                        active_mask = last_token_hidden > 0
                        active_neurons = active_mask.nonzero(as_tuple=False)
                        features = active_neurons[:, 1].unique().tolist() if active_neurons.size(0) > 0 else []
                        container[layer_idx] = set(features)
                except Exception as e:
                    print(f"[hook error] {module_name}: {str(e)}")
            return hook_fn

        containers = {
            'mlp.gate_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'mlp.up_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'mlp.down_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.q_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.k_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.v_proj': {i: set() for i in range(self.config.num_hidden_layers)}
        }

        for name, module in self.named_modules():
            if "attn" in name or "attention" in name: #跳过attention层
                continue
            if candidate_layers is None or any(f".layers.{i}." in name for i in candidate_layers):
                for key in containers.keys():
                    if key in name:
                        is_down_proj = (key == 'mlp.down_proj')
                        handles.append(
                            module.register_forward_hook(
                                hook_fn_factory(containers[key], name, is_down_proj)
                            )
                        )
                        break

        try:
            yield containers, handles
        finally:
            for h in handles:
                h.remove()

    def generate(self, **kwargs):
        candidate_layers = kwargs.pop("candidate_premature_layers", None)
        top_n = kwargs.pop("top_n", None)
        largest = kwargs.pop("largest", True)
        
        device = next(self.parameters()).device
        input_ids = kwargs["input_ids"].to(device)
        attention_mask = kwargs["attention_mask"].to(device)
        next_input_ids = torch.cat([input_ids, input_ids[:, -1:]], dim=1).to(device)
        next_attention_mask = torch.cat([attention_mask, attention_mask[:, -1:]], dim=1).to(device)

        with self.register_hooks(candidate_layers, top_n=top_n, largest=largest) as (containers, _):
            self.forward(
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                return_dict=True,
                output_hidden_states=False
            )

        return (
            None,
            next_input_ids,
            containers['mlp.gate_proj'],
            containers['mlp.up_proj'],
            containers['mlp.down_proj'],
            containers['self_attn.q_proj'],
            containers['self_attn.k_proj'],
            containers['self_attn.v_proj'],
            None,
            None
        )
        

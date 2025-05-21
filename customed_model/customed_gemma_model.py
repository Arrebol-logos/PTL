import re
import torch
import contextlib
from typing import Optional, List, Tuple, Dict, Any
from transformers import Gemma2ForCausalLM

class CustomedGemmaForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @contextlib.contextmanager
    def register_hooks(
        self,
        candidate_layers: Optional[List[int]],
        top_n: Optional[int],
        largest: bool
    ):
        handles = []

        def extract_layer_idx(name: str) -> Optional[int]:
            m = re.search(r"layers\.(\d+)\.", name)
            return int(m.group(1)) if m else None

        def hook_fn_factory(container: Dict[int, set], module_name: str, is_down_proj: bool):
            def hook_fn(module, inp, output):
                layer_idx = extract_layer_idx(module_name)
                if layer_idx is None:
                    return
                hidden = inp[0] if is_down_proj else output
                if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                    return
                last = hidden[:, -1, :]  # [batch, hidden_size or intermediate_size]
                if top_n is not None:
                    # Limit k to not exceed current dimension size
                    k = min(top_n, last.size(1))
                    if k <= 0:
                        chosen = set()
                    else:
                        vals, idxs = torch.topk(last, k=k, dim=1, largest=largest)
                        if idxs.size(0) > 1:
                            sets = [set(idxs[i].tolist()) for i in range(idxs.size(0))]
                            chosen = set.intersection(*sets)
                        else:
                            chosen = set(idxs[0].tolist())
                else:
                    mask = last > 0
                    nz = mask.nonzero(as_tuple=False)
                    chosen = set(nz[:, 1].tolist()) if nz.numel() > 0 else set()
                container[layer_idx] = chosen
            return hook_fn

        # Initialize all containers (keep attention keys to avoid breaking interface)
        containers = {
            'mlp.gate_proj':    {i: set() for i in range(self.config.num_hidden_layers)},
            'mlp.up_proj':      {i: set() for i in range(self.config.num_hidden_layers)},
            'mlp.down_proj':    {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.q_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.k_proj': {i: set() for i in range(self.config.num_hidden_layers)},
            'self_attn.v_proj': {i: set() for i in range(self.config.num_hidden_layers)},
        }

        # Register hooks only for mlp.*, skip all attention.*
        for name, module in self.named_modules():
            for key in containers:
                # Only apply to mlp branches
                if not key.startswith("mlp."):
                    continue
                if key in name:
                    idx = extract_layer_idx(name)
                    if idx is None or (candidate_layers is not None and idx not in candidate_layers):
                        break
                    is_down = (key == 'mlp.down_proj')
                    handles.append(
                        module.register_forward_hook(
                            hook_fn_factory(containers[key], name, is_down)
                        )
                    )
                    break

        try:
            yield containers
        finally:
            for h in handles:
                h.remove()

    def generate(
        self,
        **kwargs
    ) -> Tuple[
        Any,               # placeholder None
        torch.LongTensor,  # next_input_ids
        Dict[int, set],    # mlp.gate_proj
        Dict[int, set],    # mlp.up_proj
        Dict[int, set],    # mlp.down_proj
        Dict[int, set],    # self_attn.q_proj 
        Dict[int, set],    # self_attn.k_proj 
        Dict[int, set],    # self_attn.v_proj 
        Any,               # placeholder None
        Any                # placeholder None
    ]:
        candidate_layers = kwargs.pop("candidate_premature_layers", None)
        top_n   = kwargs.pop("top_n", None)
        largest = kwargs.pop("largest", True)

        input_ids      = kwargs["input_ids"].to(next(self.parameters()).device)
        attention_mask = kwargs["attention_mask"].to(next(self.parameters()).device)

        next_input_ids = torch.cat([input_ids, input_ids[:, -1:]], dim=1)
        next_attn_mask = torch.cat([attention_mask, attention_mask[:, -1:]], dim=1)

        with self.register_hooks(candidate_layers, top_n=top_n, largest=largest) as containers:
            self.forward(
                input_ids=next_input_ids,
                attention_mask=next_attn_mask,
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

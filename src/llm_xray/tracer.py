import torch


class TransformerTracer:
    """
    Captures intermediate activations from a Llama-style Transformer.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.trace = []

    def clear(self):
        self.trace = []

    def _stats(self, tensor):
        tensor = tensor.detach().float()

        return {
            "shape": list(tensor.shape),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }

    def _hook(self, name):
        def hook(module, inputs, output):
            # Some Transformer modules return tuples.
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            if not torch.is_tensor(tensor):
                return

            self.trace.append({
                "name": name,
                **self._stats(tensor),
            })

        return hook

    def register(self):
        self.remove()

        layers = self.model.model.layers

        for layer_index, layer in enumerate(layers):

            prefix = f"layer_{layer_index}"

            self.handles.append(
                layer.input_layernorm.register_forward_hook(
                    self._hook(f"{prefix}.input_layernorm")
                )
            )

            self.handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    self._hook(f"{prefix}.attention.q_proj")
                )
            )

            self.handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    self._hook(f"{prefix}.attention.k_proj")
                )
            )

            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._hook(f"{prefix}.attention.v_proj")
                )
            )

            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._hook(f"{prefix}.attention.o_proj")
                )
            )

            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self._hook(
                        f"{prefix}.post_attention_layernorm"
                    )
                )
            )

            self.handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    self._hook(f"{prefix}.mlp.gate_proj")
                )
            )

            self.handles.append(
                layer.mlp.up_proj.register_forward_hook(
                    self._hook(f"{prefix}.mlp.up_proj")
                )
            )

            self.handles.append(
                layer.mlp.down_proj.register_forward_hook(
                    self._hook(f"{prefix}.mlp.down_proj")
                )
            )

        self.handles.append(
            self.model.model.norm.register_forward_hook(
                self._hook("final_norm")
            )
        )

        self.handles.append(
            self.model.lm_head.register_forward_hook(
                self._hook("lm_head")
            )
        )

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles = []

    def get_trace(self):
        return self.trace
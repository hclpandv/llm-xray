import torch


class TransformerTracer:
    """
    Records intermediate tensors produced while the model executes.
    """

    def __init__(self, model):
        self.model = model
        self.trace = []
        self.handles = []

    def clear(self):
        self.trace.clear()

    def _preview(self, tensor):
        values = (
            tensor.detach()
            .float()
            .flatten()
            .cpu()
            .tolist()
        )
        return values[:16]

    def record_tensor(self, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return

        self.trace.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "preview": self._preview(tensor),
            }
        )

    def _record(self, name):
        def hook(module, inputs, outputs):
            tensor = outputs

            if isinstance(outputs, tuple):
                tensor = outputs[0]

            self.record_tensor(name, tensor)

        return hook

    def _record_input(self, name):
        def hook(module, inputs):
            if not inputs:
                return

            self.record_tensor(name, inputs[0])

        return hook

    def register(self):
        for layer_index, layer in enumerate(self.model.model.layers):
            prefix = f"layer_{layer_index}"

            self.handles.append(
                layer.register_forward_pre_hook(
                    self._record_input(f"{prefix}.input")
                )
            )

            self.handles.append(
                layer.input_layernorm.register_forward_hook(
                    self._record(f"{prefix}.input_layernorm")
                )
            )

            self.handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    self._record(f"{prefix}.attention.q_proj")
                )
            )

            self.handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    self._record(f"{prefix}.attention.k_proj")
                )
            )

            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._record(f"{prefix}.attention.v_proj")
                )
            )

            self.handles.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._record(f"{prefix}.attention.o_proj")
                )
            )

            self.handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self._record(f"{prefix}.post_attention_layernorm")
                )
            )

            self.handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    self._record(f"{prefix}.mlp.gate_proj")
                )
            )

            self.handles.append(
                layer.mlp.up_proj.register_forward_hook(
                    self._record(f"{prefix}.mlp.up_proj")
                )
            )

            self.handles.append(
                layer.mlp.down_proj.register_forward_hook(
                    self._record(f"{prefix}.mlp.down_proj")
                )
            )

    def get_trace(self):
        return self.trace

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

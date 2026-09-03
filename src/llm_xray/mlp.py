def make_inspection_mlp(tracer, layer_index):
    """
    Create an MLP forward wrapper that mirrors the Hugging Face
    Llama/Qwen MLP computation while capturing the activation and
    gated output tensors.

    The wrapper is only enabled during inspection.
    """

    prefix = f"layer_{layer_index}.mlp"

    def inspection_forward(module, x):
        gate_output = module.gate_proj(x)
        activation = module.act_fn(gate_output)

        tracer.record_tensor(
            f"{prefix}.activation",
            activation,
        )

        up_output = module.up_proj(x)

        gated_output = activation * up_output

        tracer.record_tensor(
            f"{prefix}.gated_output",
            gated_output,
        )

        down_proj = module.down_proj(gated_output)

        return down_proj

    return inspection_forward

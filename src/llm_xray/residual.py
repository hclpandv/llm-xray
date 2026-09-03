def make_inspection_decoder_layer(tracer, layer_index):
    """
    Create a decoder-layer forward wrapper that mirrors the Hugging Face
    Llama/Qwen decoder-layer computation while capturing the two residual
    addition results.

    The wrapper is only enabled during inspection.
    """

    prefix = f"layer_{layer_index}"

    def inspection_forward(
        module,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        # ------------------------------------------------------------
        # Attention block
        # ------------------------------------------------------------

        residual = hidden_states

        hidden_states = module.input_layernorm(
            hidden_states
        )

        hidden_states, _ = module.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        tracer.record_tensor(
            f"{prefix}.attention_residual",
            hidden_states,
        )

        # ------------------------------------------------------------
        # MLP block
        # ------------------------------------------------------------

        residual = hidden_states

        hidden_states = module.post_attention_layernorm(
            hidden_states
        )

        hidden_states = module.mlp(
            hidden_states
        )

        hidden_states = residual + hidden_states

        tracer.record_tensor(
            f"{prefix}.mlp_residual",
            hidden_states,
        )

        return hidden_states

    return inspection_forward

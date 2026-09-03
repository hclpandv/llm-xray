import math

import torch
import torch.nn.functional as F

from transformers.integrations.sdpa_attention import (
    repeat_kv,
    sdpa_attention_forward,
)


def make_inspection_attention(tracer):
    """
    Wrap Hugging Face SDPA attention.

    We inspect the exact Q/K/V tensors supplied to the attention backend,
    derive educational attention tensors, and then delegate the actual
    model computation to the original SDPA implementation.
    """

    def inspection_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        is_causal=None,
        position_bias=None,
        **kwargs,
    ):
        layer_index = getattr(module, "layer_idx", None)

        if layer_index is None:
            layer_name = "attention"
        else:
            layer_name = f"layer_{layer_index}.attention"

        # Hugging Face's SDPA backend expands K/V for GQA when needed.
        num_key_value_groups = getattr(
            module,
            "num_key_value_groups",
            1,
        )

        key_for_inspection = repeat_kv(
            key,
            num_key_value_groups,
        )

        value_for_inspection = repeat_kv(
            value,
            num_key_value_groups,
        )

        # Q @ K^T.
        scores = torch.matmul(
            query,
            key_for_inspection.transpose(2, 3),
        )

        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])

        scores = scores * scaling

        tracer.record_tensor(
            f"{layer_name}.attention_scores",
            scores,
        )

        masked_scores = scores

        # Apply the same causal restriction used by decoder attention.
        effective_is_causal = (
            is_causal
            if is_causal is not None
            else getattr(module, "is_causal", True)
        )

        if attention_mask is not None:
            mask = attention_mask[
                ...,
                : key_for_inspection.shape[-2],
            ]

            masked_scores = masked_scores + mask

        elif effective_is_causal:
            q_length = query.shape[-2]
            kv_length = key_for_inspection.shape[-2]

            causal_mask = torch.triu(
                torch.ones(
                    q_length,
                    kv_length,
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )

            masked_scores = masked_scores.masked_fill(
                causal_mask,
                torch.finfo(scores.dtype).min,
            )

        tracer.record_tensor(
            f"{layer_name}.masked_scores",
            masked_scores,
        )

        attention_weights = F.softmax(
            masked_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        tracer.record_tensor(
            f"{layer_name}.attention_weights",
            attention_weights,
        )

        weighted_values = torch.matmul(
            attention_weights,
            value_for_inspection,
        )

        tracer.record_tensor(
            f"{layer_name}.weighted_values",
            weighted_values,
        )

        # Keep the real model computation on Hugging Face SDPA.
        attn_output, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

        tracer.record_tensor(
            f"{layer_name}.attention_heads",
            attn_output,
        )

        # The SDPA backend returns one vector per attention head:
        # [batch, sequence, heads, head_dim].
        #
        # The attention module then merges those heads into:
        # [batch, sequence, hidden_size].
        attention_output = attn_output.reshape(
            attn_output.shape[0],
            attn_output.shape[1],
            -1,
        )

        tracer.record_tensor(
            f"{layer_name}.attention_output",
            attention_output,
        )

        return attn_output, None

    return inspection_attention_forward

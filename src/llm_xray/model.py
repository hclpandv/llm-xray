import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import make_inspection_attention
from .config import MODEL_NAME
from .schemas import GenerationResponse, TokenInfo
from .tracer import TransformerTracer


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tracer = None
        self.device = self._detect_device()

        self.attention_implementation_name = (
            "llm_xray_inspection"
        )

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def load(self) -> None:
        print(f"Loading model: {MODEL_NAME}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()

        self.tracer = TransformerTracer(self.model)
        self.tracer.register()

        print("Model loaded.")

    def tokenize(self, prompt: str) -> list[TokenInfo]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids
        )

        return [
            TokenInfo(
                position=position,
                token=token,
                token_id=token_id,
            )
            for position, (token, token_id)
            in enumerate(zip(tokens, input_ids))
        ]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> GenerationResponse:

        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        tokens = self.tokenize(prompt)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        output = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = encoded["input_ids"].shape[1]

        generated_ids = output[0][input_length:]

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return GenerationResponse(
            model=MODEL_NAME,
            prompt=prompt,
            tokens=tokens,
            generated_text=generated_text,
        )

    def _enable_attention_inspection(self):
        inspection_attention = make_inspection_attention(
            self.tracer
        )

        ALL_ATTENTION_FUNCTIONS.register(
            self.attention_implementation_name,
            inspection_attention,
        )

        self.original_attention_implementation = (
            self.model.config._attn_implementation
        )

        self.model.config._attn_implementation = (
            self.attention_implementation_name
        )

    def _disable_attention_inspection(self):
        self.model.config._attn_implementation = (
            self.original_attention_implementation
        )

    @torch.inference_mode()
    def inspect_prompt(self, prompt: str) -> dict:
        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        input_ids = encoded["input_ids"]

        embedding_layer = self.model.get_input_embeddings()

        embedding_matrix_shape = list(
            embedding_layer.weight.shape
        )

        embeddings = embedding_layer(input_ids)

        self.tracer.clear()

        self._enable_attention_inspection()

        try:
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
            )
        finally:
            self._disable_attention_inspection()

        trace = self.tracer.get_trace()

        logits = outputs.logits

        final_hidden_state = outputs.hidden_states[-1][
            :,
            -1,
            :,
        ]

        config = self.model.config

        tokenizer_class = type(
            self.tokenizer
        ).__name__

        tokenizer_model_type = None

        try:
            tokenizer_model_type = (
                self.tokenizer
                .backend_tokenizer
                .model
                .__class__
                .__name__
            )
        except AttributeError:
            pass

        tokenizer_vocab_size = len(self.tokenizer)

        input_tokens = (
            self.tokenizer.convert_ids_to_tokens(
                input_ids[0]
            )
        )

        embedding_vectors = (
            embeddings[0]
            .detach()
            .float()
            .cpu()
        )

        tokens = []

        for position, (
            token,
            token_id,
        ) in enumerate(
            zip(
                input_tokens,
                input_ids[0].tolist(),
            )
        ):
            decoded = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            vector = embedding_vectors[position]

            tokens.append(
                {
                    "position": position,
                    "token": token,
                    "token_id": token_id,
                    "decoded": decoded,
                    "embedding": {
                        "row": token_id,
                        "dimension": vector.shape[0],
                        "preview": vector[:16].tolist(),
                    },
                }
            )

        layers = []

        for layer_index, layer in enumerate(
            self.model.model.layers
        ):
            layers.append(
                {
                    "index": layer_index,
                    "components": [
                        {
                            "name": "input_layernorm",
                            "type": "normalization",
                            "shape": [
                                1,
                                input_ids.shape[1],
                                config.hidden_size,
                            ],
                        },
                        {
                            "name": "attention",
                            "type": "attention",
                            "components": [
                                {
                                    "name": "q_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.self_attn.q_proj.out_features,
                                    ],
                                },
                                {
                                    "name": "k_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.self_attn.k_proj.out_features,
                                    ],
                                },
                                {
                                    "name": "v_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.self_attn.v_proj.out_features,
                                    ],
                                },
                                {
                                    "name": "attention_scores",
                                    "shape": [
                                        1,
                                        config.num_attention_heads,
                                        input_ids.shape[1],
                                        input_ids.shape[1],
                                    ],
                                },
                                {
                                    "name": "masked_scores",
                                    "shape": [
                                        1,
                                        config.num_attention_heads,
                                        input_ids.shape[1],
                                        input_ids.shape[1],
                                    ],
                                },
                                {
                                    "name": "attention_weights",
                                    "shape": [
                                        1,
                                        config.num_attention_heads,
                                        input_ids.shape[1],
                                        input_ids.shape[1],
                                    ],
                                },
                                {
                                    "name": "weighted_values",
                                    "shape": [
                                        1,
                                        config.num_attention_heads,
                                        input_ids.shape[1],
                                        getattr(
                                            layer.self_attn,
                                            "head_dim",
                                            config.hidden_size
                                            // config.num_attention_heads,
                                        ),
                                    ],
                                },
                                {
                                    "name": "attention_heads",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        config.num_attention_heads,
                                        getattr(
                                            layer.self_attn,
                                            "head_dim",
                                            config.hidden_size
                                            // config.num_attention_heads,
                                        ),
                                    ],
                                },
                                {
                                    "name": "attention_output",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        config.hidden_size,
                                    ],
                                },
                                {
                                    "name": "o_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.self_attn.o_proj.out_features,
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "post_attention_layernorm",
                            "type": "normalization",
                            "shape": [
                                1,
                                input_ids.shape[1],
                                config.hidden_size,
                            ],
                        },
                        {
                            "name": "mlp",
                            "type": "mlp",
                            "components": [
                                {
                                    "name": "gate_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.mlp.gate_proj.out_features,
                                    ],
                                },
                                {
                                    "name": "up_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.mlp.up_proj.out_features,
                                    ],
                                },
                                {
                                    "name": "down_proj",
                                    "shape": [
                                        1,
                                        input_ids.shape[1],
                                        layer.mlp.down_proj.out_features,
                                    ],
                                },
                            ],
                        },
                    ],
                }
            )

        last_logits = logits[0, -1]

        probabilities = torch.softmax(
            last_logits,
            dim=-1,
        )

        top_probabilities, top_ids = torch.topk(
            probabilities,
            k=10,
        )

        next_tokens = []

        for probability, token_id in zip(
            top_probabilities,
            top_ids,
        ):
            token_id = token_id.item()

            next_tokens.append(
                {
                    "token": self.tokenizer.decode(
                        [token_id]
                    ),
                    "token_id": token_id,
                    "probability": probability.item(),
                }
            )

        return {
            "model": {
                "name": config.name_or_path,
                "layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "vocab_size": config.vocab_size,
            },
            "tokenizer": {
                "name": self.tokenizer.name_or_path,
                "class": tokenizer_class,
                "model_type": tokenizer_model_type,
                "vocab_size": tokenizer_vocab_size,
            },
            "prompt": prompt,
            "tokens": tokens,
            "embedding": {
                "matrix_shape": embedding_matrix_shape,
                "dimension": embedding_matrix_shape[1],
                "sequence_shape": list(
                    embeddings.shape
                ),
            },
            "layers": layers,
            "execution_trace": trace,
            "final_hidden_state": {
                "name": "final_hidden_state",
                "shape": list(
                    final_hidden_state.shape
                ),
            },
            "logits": {
                "name": "next_token_logits",
                "shape": list(logits.shape),
            },
            "next_tokens": next_tokens,
        }

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME
from .schemas import GenerationResponse, TokenInfo
from .tracer import TransformerTracer


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tracer = None
        self.device = self._detect_device()

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

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

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
            raise RuntimeError("Model has not been loaded.")

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

    @torch.inference_mode()
    def inspect_prompt(self, prompt: str) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        input_ids = encoded["input_ids"]

        # ---------------------------------------------------------
        # Embedding
        # ---------------------------------------------------------

        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids)

        # ---------------------------------------------------------
        # Run the actual Transformer
        # ---------------------------------------------------------

        self.tracer.clear()

        outputs = self.model(
            **encoded,
            output_hidden_states=True,
        )

        trace = self.tracer.get_trace()
        logits = outputs.logits

        # Final hidden representation for the last input token.
        final_hidden_state = outputs.hidden_states[-1][:, -1, :]

        # ---------------------------------------------------------
        # Model metadata
        # ---------------------------------------------------------

        config = self.model.config

        # ---------------------------------------------------------
        # Describe the Transformer architecture
        # ---------------------------------------------------------

        layers = []

        for layer_index, layer in enumerate(self.model.model.layers):

            layers.append(
                {
                    "index": layer_index,
                    "components": [
                        {
                            "name": "input_layernorm",
                            "type": "normalization",
                            "shape": [1, input_ids.shape[1], config.hidden_size],
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
                            "shape": [1, input_ids.shape[1], config.hidden_size],
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

        # ---------------------------------------------------------
        # Top next-token predictions
        # ---------------------------------------------------------

        last_logits = logits[0, -1]

        probabilities = torch.softmax(last_logits, dim=-1)

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
                    "token": self.tokenizer.decode([token_id]),
                    "token_id": token_id,
                    "probability": probability.item(),
                }
            )

        # ---------------------------------------------------------
        # Response
        # ---------------------------------------------------------

        return {
            "model": {
                "name": config.name_or_path,
                "layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "vocab_size": config.vocab_size,
            },

            "prompt": prompt,

            "tokens": [
                {
                    "position": position,
                    "token": token,
                    "token_id": token_id,
                }
                for position, (token, token_id)
                in enumerate(
                    zip(
                        self.tokenizer.convert_ids_to_tokens(
                            input_ids[0]
                        ),
                        input_ids[0].tolist(),
                    )
                )
            ],

            "embedding": {
                "name": "token_embeddings",
                "shape": list(embeddings.shape),
            },

            "layers": layers,

            "execution_trace": trace,

            "final_hidden_state": {
                "name": "final_hidden_state",
                "shape": list(final_hidden_state.shape),
            },

            "logits": {
                "name": "next_token_logits",
                "shape": list(logits.shape),
            },

            "next_tokens": next_tokens,
        }

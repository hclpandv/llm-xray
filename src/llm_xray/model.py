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

        # The model's actual embedding layer.
        embedding_layer = self.model.get_input_embeddings()

        embeddings = embedding_layer(input_ids)

        # Run the actual Transformer.
        outputs = self.model(
            **encoded,
            output_hidden_states=True,
        )
        trace = self.tracer.get_trace()

        logits = outputs.logits

        # Hidden state at the final input position.
        final_hidden_state = outputs.hidden_states[-1][:, -1, :]

        return {
            "input_ids": input_ids[0].tolist(),
            "tokens": self.tokenizer.convert_ids_to_tokens(
                input_ids[0]
            ),
            "embedding_shape": list(embeddings.shape),
            "embedding_vectors": embeddings[0].detach().cpu().tolist(),
            "hidden_state_shape": list(final_hidden_state.shape),
            "hidden_state": final_hidden_state[0].detach().cpu().tolist(),
            "logits_shape": list(logits.shape),
            "logits": logits[0, -1].detach().cpu().tolist(),
            "trace": trace
        }

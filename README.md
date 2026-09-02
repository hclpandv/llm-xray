# LLM-Xray

Interactive X-ray visualization and debugger for local Hugging Face language models.

LLM-Xray runs a language model locally and exposes what happens inside the model during inference — from tokenization and embeddings through Transformer layers, logits, probabilities, and generation.

The goal is to make the internals of a language model visible, inspectable, and understandable.

---

## Current status

LLM-Xray currently supports interactive inspection of local Hugging Face causal language models.

The current implementation has been tested with:

* `HuggingFaceTB/SmolLM2-360M-Instruct`
* `Qwen/Qwen2.5-0.5B-Instruct`

The model can be selected at runtime without changing the source code.

---

## What it currently does

### Tokenization

Given a prompt, LLM-Xray shows:

* Token position
* Token string
* Token ID
* Decoded text
* Tokenizer class
* Tokenization algorithm
* Vocabulary size

For example:

```text
The capital of France is

The       → 785
Ġcapital  → 6722
Ġof       → 315
ĠFrance   → 9625
Ġis       → 374
```

The `Ġ` character is displayed by some Hugging Face tokenizers to represent a preceding space.

---

### Token embeddings

LLM-Xray exposes the relationship between a token ID and the model's embedding matrix:

```text
Token
  ↓
Token ID
  ↓
Embedding matrix
  ↓
Selected row
  ↓
Embedding vector
```

For example, with Qwen:

```text
Embedding matrix
151,936 × 896

Token ID
9625

Selected row
9625

896-dimensional vector
```

The interface displays the first values of the actual embedding vector.

---

### Transformer architecture

The interface displays the model's architecture dynamically, including:

* Number of Transformer layers
* Hidden dimension
* Vocabulary size
* Embedding dimensions
* Final hidden-state shape
* Logit tensor shape

The pipeline adapts to the loaded model rather than assuming a fixed architecture size.

For example:

```text
Token IDs
[5]
↓
Token Embeddings
[1 × 5 × 896]
↓
24 Transformer Layers
Hidden size: 896
↓
Final RMSNorm
[1 × 896]
↓
LM Head
[1 × 5 × 151936]
↓
Softmax
151,936 probabilities
```

---

### Transformer execution trace

During a forward pass, LLM-Xray captures actual intermediate tensors from the Transformer layers.

For the currently supported Llama-style architectures, each layer exposes:

```text
Input
↓
Input RMSNorm
↓
Q Projection
K Projection
V Projection
↓
Attention
↓
O Projection
↓
Residual
↓
Post-Attention RMSNorm
↓
Gate Projection
Up Projection
↓
SiLU × Gate
↓
Down Projection
↓
Residual
↓
Layer output
```

The trace currently captures 10 real operations per Transformer layer:

1. Layer input
2. Input RMSNorm
3. Q projection
4. K projection
5. V projection
6. O projection
7. Post-attention RMSNorm
8. Gate projection
9. Up projection
10. Down projection

Each captured tensor can be inspected in the UI.

The tensor inspector shows:

* Operation name
* Tensor shape
* Data type
* Device
* First 16 tensor values

For example:

```text
layer_0.mlp.down_proj

[1 × 5 × 896]

dtype: torch.float32
device: mps:0
```

---

### Next-token probabilities

After the forward pass, LLM-Xray calculates the probability distribution for the next token.

The interface displays the top predicted tokens and their probabilities.

For example:

```text
Paris       30.22%
______      12.31%
:            6.60%
...
```

This makes the connection between the model's final hidden state, logits, and predicted next token visible.

---

### Local generation

LLM-Xray can also generate text from the local model.

Generation is performed directly through the Hugging Face model rather than through an external LLM API.

---

## Supported models

The project is designed around Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`.

Current testing includes:

### SmolLM2

```text
Model:
HuggingFaceTB/SmolLM2-360M-Instruct

Transformer layers: 32
Hidden size:        960
Vocabulary:         49,152
```

### Qwen2.5

```text
Model:
Qwen/Qwen2.5-0.5B-Instruct

Transformer layers: 24
Hidden size:        896
Vocabulary:         151,936
```

Testing multiple models is important because their architectures and dimensions differ.

---

## Runtime model selection

The model is configured through the `LLM_XRAY_MODEL` environment variable.

The default model is:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

To run Qwen instead:

```bash
LLM_XRAY_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run llm-xray
```

This allows different models to be tested without modifying the source code.

---

## Technology

LLM-Xray currently uses:

* Python
* FastAPI
* Uvicorn
* PyTorch
* Hugging Face Transformers
* Hugging Face Tokenizers
* HTML / CSS / JavaScript
* `uv` for Python environment and dependency management

On Apple Silicon, PyTorch's MPS backend is used when available.

The runtime device selection is:

```text
MPS → CUDA → CPU
```

---

## Running locally

### Requirements

* Python 3.11+
* `uv`
* A machine capable of running the selected Hugging Face model locally

### Install

Clone the repository and install the environment:

```bash
uv sync
```

For Apple Silicon development, Python 3.12 is currently used:

```bash
uv python install 3.12
uv python pin 3.12
uv sync
```

### Start LLM-Xray

```bash
uv run llm-xray
```

The server starts on:

```text
http://127.0.0.1:8000
```

Open that address in a browser.

---

## Project structure

```text
llm-xray/
│
├── src/
│   └── llm_xray/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── schemas.py
│       ├── server.py
│       └── tracer.py
│
├── web/
│   └── index.html
│
├── tests/
│
├── pyproject.toml
└── README.md
```

---

## Architecture

The application is split into a small Python backend and browser-based frontend.

```text
Browser
   │
   │ HTTP
   ▼
FastAPI
   │
   ├── /inspect
   │       │
   │       ▼
   │   ModelManager
   │       │
   │       ├── Tokenizer
   │       ├── Embeddings
   │       ├── Transformer model
   │       ├── TransformerTracer
   │       ├── Logits
   │       └── Probabilities
   │
   └── /generate
           │
           ▼
       Local Hugging Face model
```

The `TransformerTracer` uses PyTorch forward hooks to capture tensors while the model executes.

---

## Roadmap

The long-term goal is to turn LLM-Xray into a more complete interactive debugger for Transformer models.

Planned areas include:

* Attention visualization
* More detailed attention internals
* Hidden-state visualization
* Token-by-token generation tracing
* Activation inspection
* Activation patching
* Model comparison
* Broader Hugging Face architecture support
* More detailed tensor visualization
* Interactive execution controls

Eventually, the interface should allow you to pause the model's computation and inspect what is happening at each stage.

---

## Current limitations

LLM-Xray is intentionally still an early-stage project.

The current execution tracer assumes a Llama-style Transformer module structure for several internal components, including modules such as:

```text
model.layers
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

This works for the models currently tested, including SmolLM2 and Qwen2.5, but different Hugging Face architectures may use different internal module layouts.

Making the tracer genuinely architecture-independent is a future goal.

The project currently focuses on **visibility and understanding rather than model output quality**.

---

## Philosophy

LLMs are often presented as:

```text
Input → Output
```

LLM-Xray explores the much more interesting question:

```text
What happens between the input and the output?
```

Instead of treating a language model as a black box, the project attempts to expose the computation happening inside it.

The project is intended primarily as an educational and research tool for understanding modern language models.

---

## License

License to be decided.

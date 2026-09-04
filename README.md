# LLM-Xray 🔬

**See inside a language model while it runs.**

LLM-Xray is an interactive visualization and debugging tool for **local Hugging Face causal language models**. It makes the path from prompt → tokens → embeddings → Transformer layers → logits → probabilities → generation visible and inspectable.

The goal is simple: **make language-model internals easier to see, understand, and explore.**

---

## 🌐 Live Demo

**[Open the LLM-Xray GitHub Pages demo](https://hclpandv.github.io/llm-xray/)**

The public demo is a **static, pre-recorded inspection** of a real model run.

It does **not** download a model or run inference in the browser. Instead, the inspection data is captured locally and stored in the repository, allowing anyone to explore the visualization directly from GitHub Pages.

For the full interactive experience — including running different local Hugging Face models — run LLM-Xray locally.

---

## ✨ What LLM-Xray Shows

LLM-Xray exposes several stages of a language model's forward pass:

```text
Prompt
  ↓
Tokenization
  ↓
Token IDs
  ↓
Embeddings
  ↓
Transformer layers
  ↓
Hidden states
  ↓
Logits
  ↓
Next-token probabilities
  ↓
Generation
```

Instead of treating a language model as a black box, LLM-Xray lets you inspect what is happening inside.

---

## 🔍 Features

### 1. Tokenization

Given a prompt, LLM-Xray shows how the tokenizer converts text into tokens.

The interface exposes:

* Token position
* Token string
* Token ID
* Decoded text
* Tokenizer class
* Tokenization algorithm
* Vocabulary size

For example:

```text
The        → 785
Ġcapital   → 6722
Ġof        → 315
ĠFrance    → 9625
Ġis        → 374
```

Some Hugging Face tokenizers use `Ġ` to represent a preceding space.

Seeing the tokenization step makes it easier to understand what the model actually receives as input.

---

### 2. Token Embeddings

LLM-Xray makes the relationship between a token ID and its embedding vector visible:

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

For example, with Qwen2.5:

```text
Embedding matrix
151,936 × 896

Token ID
9625

Selected row
9625

Embedding vector
896 dimensions
```

The interface exposes the beginning of the actual embedding vector so you can inspect the numerical representation passed into the Transformer.

---

### 3. Transformer Architecture

The UI derives model information dynamically rather than assuming a fixed model size.

It exposes information such as:

* Number of Transformer layers
* Hidden dimension
* Vocabulary size
* Embedding dimensions
* Final hidden-state shape
* Logit tensor shape

For the current Qwen2.5 example:

```text
Token IDs
[5]
   ↓
Token embeddings
[1 × 5 × 896]
   ↓
24 Transformer layers
Hidden size: 896
   ↓
Final hidden state
   ↓
LM head
[1 × 5 × 151936]
   ↓
Next-token probabilities
151,936 values
```

This makes the relationship between model architecture and tensor dimensions explicit.

---

### 4. Transformer Execution Trace

During a real forward pass, LLM-Xray captures intermediate tensors from the Transformer.

For the Llama-style architectures currently supported, a layer can be followed through operations such as:

```text
Layer input
   ↓
Input RMSNorm
   ↓
Q projection
K projection
V projection
   ↓
Attention
   ↓
O projection
   ↓
Post-attention RMSNorm
   ↓
Gate projection
Up projection
   ↓
SiLU × Gate
   ↓
Down projection
   ↓
Layer output
```

The tracer currently captures **10 real operations per Transformer layer**:

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

Each captured tensor can be inspected directly in the UI.

The tensor inspector shows:

* Operation name
* Tensor shape
* Data type
* Device
* Minimum value
* Maximum value
* Mean
* Standard deviation
* First 16 tensor values

For example:

```text
layer_0.mlp.down_proj

shape:  [1 × 5 × 896]
dtype:  torch.float32
device: mps:0
```

This turns abstract architecture diagrams into actual tensors produced during inference.

---

### 5. Next-Token Probabilities

After the forward pass, LLM-Xray exposes the model's next-token probability distribution.

The interface shows the highest-probability candidate tokens and their probabilities:

```text
Paris       30.22%
______      12.31%
:            6.60%
...
```

This connects several otherwise separate concepts:

```text
Final hidden state
       ↓
     Logits
       ↓
     Softmax
       ↓
Token probabilities
       ↓
Predicted next token
```

---

### 6. Token-by-Token Generation

LLM-Xray also visualizes autoregressive generation.

The generation interface shows the basic loop:

```text
Context so far
      ↓
    Model
      ↓
Predicted token
      ↓
Append token
      ↓
Updated context
      ↺
```

The production interface includes a replayable generation animation with:

* Token-by-token progression
* Predicted-token highlighting
* Probability display
* Context updates
* Generation speed control
* Hover-to-pause interaction
* Replay support

The GitHub Pages demo uses the same visualization with **pre-recorded inspection data** instead of running inference.

---

## 🤗 Supported Models

LLM-Xray is built around Hugging Face:

```text
AutoModelForCausalLM
AutoTokenizer
```

The current implementation has been tested with:

### SmolLM2

```text
HuggingFaceTB/SmolLM2-360M-Instruct

Transformer layers: 32
Hidden size:        960
Vocabulary size:    49,152
```

### Qwen2.5

```text
Qwen/Qwen2.5-0.5B-Instruct

Transformer layers: 24
Hidden size:        896
Vocabulary size:    151,936
```

Testing multiple models is important because different model families can use different layer counts, tensor dimensions, and internal module layouts.

---

## ⚙️ Runtime Model Selection

The model is selected using the `LLM_XRAY_MODEL` environment variable.

The default model is:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

To run Qwen2.5 instead:

```bash
LLM_XRAY_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run llm-xray
```

No source-code changes are required to switch models.

---

## 🧠 How It Works

LLM-Xray consists of a small Python backend and a browser-based frontend.

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

The browser then turns those captured values into an interactive inspection interface.

---

## 🛠️ Technology

LLM-Xray currently uses:

* **Python**
* **FastAPI**
* **Uvicorn**
* **PyTorch**
* **Hugging Face Transformers**
* **Hugging Face Tokenizers**
* **HTML / CSS / JavaScript**
* **uv** for Python environment and dependency management

### Device selection

When available, the runtime prefers hardware acceleration in this order:

```text
MPS → CUDA → CPU
```

This allows LLM-Xray to run on Apple Silicon, CUDA-capable systems, or CPU-only machines.

---

## ▶️ Running Locally

### Requirements

* Python 3.11+
* `uv`
* A machine capable of running the selected Hugging Face model locally

### Install

Clone the repository:

```bash
git clone https://github.com/hclpandv/llm-xray.git
cd llm-xray
```

Install the environment:

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

Open that address in your browser.

The local application loads the selected Hugging Face model and performs inference on your machine.

---

## 🗺️ Roadmap

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

Eventually, the interface should make it possible to pause computation and inspect what is happening at each stage of a Transformer.

---

## ⚠️ Current Limitations

LLM-Xray is still an early-stage project.

The current execution tracer assumes a Llama-style internal module structure for several components, including:

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

This works for the models currently tested, including SmolLM2 and Qwen2.5, but other Hugging Face architectures may use different internal module layouts.

Making the tracer genuinely architecture-independent is a future goal.

The project currently prioritizes:

> **Visibility, inspectability, and understanding over model-output benchmarking.**

---

## 🎯 Philosophy

Language models are often presented as:

```text
Input → Output
```

LLM-Xray asks a more interesting question:

```text
What happens between the input and the output?
```

A Transformer is not simply a black box that turns text into text.

It is a sequence of:

* Tensor transformations
* Normalization operations
* Linear projections
* Attention computations
* Nonlinear transformations
* Residual connections
* Logit calculations
* Probability distributions

LLM-Xray attempts to make those computations visible.

The project is intended primarily as an **educational and research tool for understanding modern language models**.

---

## 🤝 Contributing

Contributions, ideas, bug reports, architecture experiments, and UI improvements are welcome.

If you try LLM-Xray with another Hugging Face architecture, feedback about module-layout compatibility is especially useful.

---

## 📄 License

License to be decided.

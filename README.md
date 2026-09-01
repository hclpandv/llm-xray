# LLM-Xray 🔬

**LLM-Xray** is an open-source project for looking inside language models.

The goal is simple:

> Make the internal computation of an LLM visible, interactive, and understandable.

Instead of treating a language model as a black box:

```text
Prompt → LLM → Response
```

LLM-Xray aims to expose what happens inside:

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embeddings
  ↓
Transformer Layers
  ↓
Attention
  ↓
MLP
  ↓
Logits
  ↓
Next Token
  ↓
Response
```

## Current status

🚧 **Early development — Milestone 1**

The current version can:

* Run a small Hugging Face language model locally
* Use PyTorch for inference
* Automatically use Apple MPS when available
* Tokenize input text
* Display token IDs
* Generate text locally
* Provide a simple browser interface

No OpenAI API or other hosted LLM API is required.

## Model

The default model is:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

The model is downloaded from Hugging Face the first time it is used and runs locally on your machine.

You can choose another compatible Hugging Face model with:

```bash
LLM_XRAY_MODEL=your-model-name uv run llm-xray
```

## Requirements

* Python 3.11+
* [uv](https://docs.astral.sh/uv/)
* macOS, Linux, or Windows
* A machine capable of running the selected Hugging Face model

Apple Silicon Macs can use the MPS backend through PyTorch.

## Getting started

Clone the repository and enter the project:

```bash
git clone <repository-url>
cd llm-xray
```

Install dependencies:

```bash
uv sync
```

Start the application:

```bash
uv run llm-xray
```

Then open:

```text
http://127.0.0.1:8000
```

Enter a prompt and click **Run Model**.

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
│       └── server.py
│
├── web/
│   └── index.html
│
├── tests/
│
├── pyproject.toml
└── README.md
```

## Roadmap

The long-term goal is to turn LLM-Xray into an interactive debugger for Transformer models.

Planned areas include:

* Tokenization and token IDs
* Embedding visualization
* Transformer layer inspection
* Attention visualization
* MLP inspection
* Hidden-state inspection
* Logit and probability visualization
* Token-by-token generation
* Execution traces
* Activation inspection
* Activation patching
* Model comparison
* Support for multiple Hugging Face models

Eventually, the interface should allow you to pause the model's computation and inspect what is happening at each stage.

## Philosophy

LLMs are often presented as:

```text
Input → Output
```

LLM-Xray explores the much more interesting question:

```text
What happens between the input and the output?
```

The project is intended primarily as an educational and research tool for understanding modern language models.

## License

License to be decided.

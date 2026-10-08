<div align="center">

# 🔬 LLM-Xray

### See inside a language model while it runs.
**Prompt → Tokens → Embeddings → Transformer layers → Logits → Probabilities → Generation**
Every step visible. Every tensor inspectable.

[![Live Demo](https://img.shields.io/badge/🌐_LIVE_DEMO-Explore_Now-6C47FF?style=for-the-badge)](https://hclpandv.github.io/llm-xray/)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Forward_Hooks-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Transformers-Causal_LMs-FFD21E)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Early_Stage-orange)

</div>

---

## 💡 Why?

Most people see an LLM as a black box:

```
Input  ──▶  ???  ──▶  Output
```

**LLM-Xray opens the box.** It is an interactive visualization and debugging tool for **local Hugging Face causal language models**. It captures real tensors from a real forward pass and turns them into something you can click, inspect and understand.

> 🎓 Built for learners, educators and researchers who want to understand *what actually happens* between the prompt and the answer.

---

## 🌐 Try It in 10 Seconds

👉 **[hclpandv.github.io/llm-xray](https://hclpandv.github.io/llm-xray/)**

The public demo is a **pre-recorded inspection of a real model run**, stored in this repo. It does not download a model or run inference in your browser.
Want the full experience with different models? [Run it locally](#️-run-locally).

---

## ✨ What You Can See

| Stage | What LLM-Xray reveals |
|---|---|
| 🔤 **Tokenization** | Token position, string, ID, decoded text, tokenizer class and algorithm, vocab size |
| 🧮 **Embeddings** | Token ID → embedding-matrix row → the actual vector (e.g. Qwen2.5: `151,936 × 896`) |
| 🏗️ **Architecture** | Layers, hidden size, vocab size, final hidden-state and logit shapes, derived dynamically |
| ⚙️ **Execution trace** | **10 real operations per layer**, captured live with PyTorch hooks |
| 🔬 **Tensor inspector** | Shape, dtype, device, min / max / mean / std, first 16 values |
| 🎲 **Next-token probabilities** | Top candidates with probabilities (e.g. `Paris → 30.22%`) |
| 🔁 **Generation** | Replayable token-by-token animation with speed control and hover-to-pause |

### 🧬 Inside a Transformer layer

```
Layer input
   ↓
Input RMSNorm
   ↓
Q / K / V projections ──▶ Attention ──▶ O projection
   ↓
Post-attention RMSNorm
   ↓
Gate + Up projections ──▶ SiLU × Gate ──▶ Down projection
   ↓
Layer output
```

Real example from the tensor inspector:

```
layer_0.mlp.down_proj
shape:  [1 × 5 × 896]
dtype:  torch.float32
device: mps:0
```

---

## 🤗 Supported Models

Works with any Hugging Face `AutoModelForCausalLM` that uses a Llama-style layout. Tested with:

| Model | Hugging Face ID |
|---|---|
| ✅ SmolLM2 (default) | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| ✅ Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` |

Switch models with one environment variable, with no code changes:

```bash
LLM_XRAY_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run llm-xray
```

---

## ▶️ Run Locally

**Requirements:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv), and a machine that can run your chosen model.

```bash
git clone https://github.com/hclpandv/llm-xray.git
cd llm-xray
uv sync
uv run llm-xray
```

Then open **http://127.0.0.1:8000** 🎉

<details>
<summary>🍎 Apple Silicon tip</summary>

```bash
uv python install 3.12
uv python pin 3.12
uv sync
```

Device preference is automatic: **MPS → CUDA → CPU**.
</details>

---

## 🧠 How It Works

```
Browser ──HTTP──▶ FastAPI
                    ├── /inspect ──▶ ModelManager
                    │                 ├── Tokenizer
                    │                 ├── Embeddings
                    │                 ├── Transformer + TransformerTracer (PyTorch hooks)
                    │                 ├── Logits
                    │                 └── Probabilities
                    └── /generate ─▶ Local Hugging Face model
```

A small Python backend captures tensors with forward hooks, and the frontend turns them into an interactive inspection UI.

**Stack:** Python · FastAPI · Uvicorn · PyTorch · Hugging Face Transformers · HTML/CSS/JS · uv

---

## 🗺️ Roadmap

- [ ] 👁️ Attention visualization and deeper attention internals
- [ ] 🌡️ Hidden-state visualization
- [ ] 🧪 Activation inspection and **activation patching**
- [ ] ⚖️ Model comparison
- [ ] 🌍 Architecture-independent tracer
- [ ] ⏯️ Pause and step through computation

---

## ⚠️ Current Limitations

LLM-Xray is early-stage. The tracer currently assumes a **Llama-style module layout** (`model.layers`, `self_attn.q_proj`, `mlp.gate_proj`, and so on). Other architectures may need adapting.
Feedback on other Hugging Face architectures is especially welcome. 🙏

---

## 🎯 Philosophy

> **Visibility, inspectability, and understanding over benchmarking.**

A Transformer is a chain of tensor transformations, normalizations, projections, attention, nonlinearities and residual connections that ends in a probability distribution. LLM-Xray makes that chain visible.

---

## 🤝 Contributing

Ideas, bug reports, architecture experiments and UI improvements are all welcome. Open an issue or a PR.

## 📄 License

License to be decided.

<div align="center">

**If LLM-Xray helped you understand LLMs a little better, give it a ⭐**

</div>

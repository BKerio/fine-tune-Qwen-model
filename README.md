# Dyniq-Qwen-1
README
**Parameter-efficient fine-tuning and serving of Qwen3-4B on a single commodity GPU.**

Dyniq-Qwen-1 is an end-to-end LLM pipeline: it fine-tunes [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with **QLoRA** (4-bit NF4 quantization plus Low-Rank Adapters) and serves the result through a lightweight, API-key-protected REST endpoint. The full training run fits on a **16 GB NVIDIA T4** (Google Colab free tier). The only artifact it produces is a LoRA adapter of about 130 MB, which loads on top of the unchanged base model at inference time.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.x-yellow)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)
![Flask](https://img.shields.io/badge/Flask-3.x-black)

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Training Pipeline](#training-pipeline)
- [Inference API](#inference-api)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Deployment Notes](#deployment-notes)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Highlights

| | |
|---|---|
| **Base model** | `Qwen/Qwen3-4B` (≈4.06B parameters) |
| **Method** | QLoRA supervised fine-tuning (SFT) with TRL `SFTTrainer` |
| **Trainable parameters** | 33.0M (**0.81%** of total) |
| **Model memory (4-bit)** | ≈2.4 GB, down from ≈8 GB in fp16 |
| **Hardware target** | NVIDIA T4, 16 GB (compute capability 7.5, fp16 only) |
| **Training data** | 3,000 multi-turn conversations sampled from UltraChat 200k |
| **Artifact** | LoRA adapter, ≈130 MB |
| **Serving** | Flask REST API with constant-time API-key auth and Qwen3 "thinking" mode toggle |

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │            TRAINING  (Colab T4)          │
                        │                                          │
  UltraChat 200k ──►  sample 6k ─► token-length filter (≤1024)     │
                        │          ─► 3,000 convos ─► 2,900 / 100  │
                        │                               train/eval │
                        │                                          │
  Qwen3-4B ──► 4-bit NF4 + double quant ──► + LoRA (r=16, all 7    │
                        │                   linear projections)    │
                        │                                          │
                        │   SFTTrainer · fp16 · paged AdamW 8-bit  │
                        │   gradient checkpointing · cosine LR     │
                        └──────────────────┬───────────────────────┘
                                           │  adapter (~130 MB)
                                           ▼  checkpointed to Google Drive
                        ┌──────────────────────────────────────────┐
                        │             SERVING  (app.py)            │
                        │                                          │
   Client ──HTTPS──►  x-api-key auth ─► request validation         │
                        │   ─► Qwen3 chat template ─► generate()   │
                        │      (4-bit base + PeftModel adapter,    │
                        │       serialized by a GPU lock)          │
                        └──────────────────────────────────────────┘
```

---

## Repository Structure

```
llm-api/
├── Dyniq_Qwen3_Fine_Tuning.ipynb   # Reproducible QLoRA training + Colab serving walkthrough
├── app.py                          # Production inference server (Flask)
├── requirements.txt                # Pinned serving dependencies
├── .env.example                    # Configuration template
└── README.md
```

---

## Training Pipeline

The notebook [`Dyniq_Qwen3_Fine_Tuning.ipynb`](Dyniq_Qwen3_Fine_Tuning.ipynb) is a linear, reproducible experiment. A single configuration cell holds every setting, and all randomness uses a fixed seed (`42`).

### 1. Quantization (the "Q" in QLoRA)

| Setting | Value | Rationale |
|---|---|---|
| Quant type | NF4 | Information-theoretically optimal 4-bit format for normally distributed weights |
| Double quantization | Enabled | Also quantizes the scaling constants, saving a little more memory |
| Compute dtype | `float16` | T4 (SM 7.5) lacks native bf16; bf16 there is emulated and slow |
| Attention | `sdpa` | FlashAttention-2 requires Ampere+; PyTorch SDPA is the fastest option on T4 |

### 2. Data Engineering

- **Source:** [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft` split), already in the `messages` (role/content) schema that both the Qwen chat template and TRL expect.
- **Sampling:** 6,000 conversations are drawn with a fixed seed, then filtered to those that fit in **≤1,024 tokens** after applying the chat template. The first 3,000 survivors are kept. Capping sequence length is the main defence against OOM errors on a 16 GB card.
- **Split:** 2,900 train / 100 held-out eval, used to monitor eval loss and catch overfitting.

### 3. LoRA Adapter

| Hyperparameter | Value |
|---|---|
| Rank `r` | 16 |
| `lora_alpha` | 32 (2 × r) |
| Dropout | 0.05 |
| Bias | none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

All seven linear projections in every transformer block are adapted. Covering both attention and MLP layers consistently beats attention-only LoRA for behavioural and style tuning.

### 4. Optimization

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 (effective batch = **16**) |
| Learning rate | 2e-4, cosine schedule, 10 warmup steps |
| Optimizer | `paged_adamw_8bit` (can page optimizer state to CPU RAM instead of failing with OOM) |
| Precision | fp16 mixed precision (trainable LoRA weights explicitly kept in fp32) |
| Gradient checkpointing | Enabled (about 30% slower, much lower activation memory) |
| Eval / checkpoint cadence | Every 50 steps; keeps the latest 2 checkpoints |

**Fault tolerance:** checkpoints are written to Google Drive, and the training cell detects existing `checkpoint-*` directories and resumes automatically. Free-tier Colab sessions that get preempted lose no progress.

**Numerical stability note:** with fp16 AMP, some LoRA parameters are created in bf16, which breaks the fp16 gradient scaler. The pipeline therefore casts every trainable parameter to fp32 before training starts.

---

## Inference API

[`app.py`](app.py) is a single-file inference server built for clarity and safe defaults:

- **Hardware-aware loading:** on a GPU it loads the model in 4-bit NF4 with the same quantization config used in training and picks bf16 or fp16 from the device's compute capability. Without a GPU it falls back to fp32 on CPU.
- **Optional adapter:** if `ADAPTER_PATH` is set, the LoRA adapter is attached with `PeftModel` and the service reports itself as `dyniq-qwen-1`. Otherwise it serves the base model.
- **Authentication:** every `/chat` request must carry an `x-api-key` header, compared in constant time with `secrets.compare_digest` to prevent timing attacks.
- **Input validation:** the server checks message structure and roles and coerces numeric parameters. `max_new_tokens` is capped at 1,024 to bound latency and memory.
- **Concurrency control:** a process-wide lock serializes `generate()` calls so concurrent requests cannot exhaust GPU memory on a single device.
- **Qwen3 thinking mode:** clients can switch Qwen3's `<think>` reasoning block on or off per request.
- **Recommended sampling defaults:** `temperature=0.7`, `top_p=0.8`, `top_k=20`, following Qwen's guidance for non-thinking mode. `temperature=0` switches to greedy decoding.
- **No extra dependencies for config:** a minimal built-in `.env` loader means `python-dotenv` is not needed.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An NVIDIA GPU with CUDA is recommended (≥8 GB VRAM for 4-bit inference). CPU-only works but is slow.
- A trained adapter directory (optional; see [Training](#training-pipeline))

### Installation

```bash
git clone <your-repo-url> llm-api
cd llm-api

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** install a CUDA-enabled PyTorch build that matches your driver (see [pytorch.org](https://pytorch.org/get-started/locally/)) before running `pip install -r requirements.txt`. `bitsandbytes` 4-bit quantization needs a CUDA GPU.

### Configure

```bash
cp .env.example .env
# Generate a strong key, e.g.:
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set `DYNIQ_API_KEY` in `.env`, and point `ADAPTER_PATH` at your fine-tuned adapter if you have one.

### Run

```bash
python app.py
# Loading Qwen/Qwen3-4B ...
# Applying adapter from ./dyniq-qwen-1-adapter
#  * Running on http://0.0.0.0:5001
```

### Train Your Own Adapter

1. Open `Dyniq_Qwen3_Fine_Tuning.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or better).
3. Run Steps 1–11 in order. The adapter is saved to `MyDrive/dyniq/Dyniq-Qwen-1/adapter`.
4. Download the adapter folder and set `ADAPTER_PATH` to its location.

---

## Configuration

All configuration is read from environment variables or from a `.env` file next to `app.py`. Real environment variables take precedence over `.env`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DYNIQ_API_KEY` | ✅ | — | Shared secret that clients send in the `x-api-key` header. The server refuses to start without it. |
| `BASE_MODEL` | | `Qwen/Qwen3-4B` | Hugging Face model ID or local path of the base model. |
| `ADAPTER_PATH` | | *(empty)* | Directory containing the LoRA adapter. If empty, the base model is served. |
| `PORT` | | `5001` | Port the HTTP server listens on. |

---

## API Reference

### `GET /health`

Liveness check. Does not require authentication.

```bash
curl http://localhost:5001/health
```

```json
{ "status": "ok", "model": "dyniq-qwen-1" }
```

### `POST /chat`

Generates an assistant reply for a conversation.

**Headers**

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `x-api-key` | Your `DYNIQ_API_KEY` |

**Request body**

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | **required** | Non-empty list of `{ "role": "system" \| "user" \| "assistant", "content": string }` |
| `max_new_tokens` | int | `512` | Maximum tokens to generate (capped at `1024`) |
| `temperature` | float | `0.7` | Sampling temperature; `0` gives deterministic greedy decoding |
| `thinking` | bool | `false` | Turns on Qwen3's `<think>` reasoning mode |

**Example**

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: $DYNIQ_API_KEY" \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a concise, helpful assistant."},
          {"role": "user",   "content": "Explain QLoRA in two sentences."}
        ],
        "max_new_tokens": 256,
        "temperature": 0.7
      }'
```

```python
import os, requests

resp = requests.post(
    "http://localhost:5001/chat",
    headers={"x-api-key": os.environ["DYNIQ_API_KEY"]},
    json={"messages": [{"role": "user", "content": "Explain QLoRA in two sentences."}]},
    timeout=120,
)
resp.raise_for_status()
print(resp.json()["reply"])
```

**Response `200 OK`**

```json
{
  "model": "dyniq-qwen-1",
  "reply": "QLoRA fine-tunes a large language model by ...",
  "tokens": 87
}
```

**Errors**

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "messages must be a non-empty list of {role, content}"}` | Missing or malformed `messages` |
| `400` | `{"error": "max_new_tokens and temperature must be numbers"}` | Non-numeric generation parameters |
| `401` | `{"error": "Unauthorized"}` | Missing or invalid `x-api-key` |

---

## Deployment Notes

- **Development server:** `app.py` uses Flask's built-in server, which is fine for development and internal testing. For production, run it behind a WSGI server (for example `gunicorn -w 1 -b 0.0.0.0:5001 app:app`) and a TLS-terminating reverse proxy. Use **one worker per GPU**: each worker loads its own copy of the model.
- **Colab tunnelling (testing only):** Step 12 of the notebook serves the API from a Colab runtime through a Cloudflare quick tunnel. The URL is ephemeral and ends when the session ends.
- **Secrets:** never commit `.env`. It is already listed in `.gitignore`. In Colab, provide the key through the **Secrets** panel, not in a notebook cell.
- **Throughput:** generation is serialized per process, so throughput is about one request at a time. For higher concurrency, see the [Roadmap](#roadmap).

---

## Limitations

- **Quantitative evaluation is still pending.** The current check is a qualitative before/after comparison on fixed prompts plus held-out eval loss. No benchmark scores (MMLU, MT-Bench, and so on) have been reported yet.
- **General-domain data only.** Version 1 is trained only on UltraChat, so it mostly shifts conversational style. It does not inject new domain knowledge.
- **Short context.** Training sequences were capped at 1,024 tokens. Behaviour on much longer conversations has not been tuned.
- **No streaming.** Responses are returned only after generation finishes.

---

## Roadmap

- [ ] Mix in curated reasoning, coding and maths SFT data
- [ ] Systematic evaluation harness (`lm-eval-harness`, MT-Bench, LLM-as-judge)
- [ ] Experiment tracking (Weights & Biases / MLflow)
- [ ] Token streaming via Server-Sent Events
- [ ] OpenAI-compatible `/v1/chat/completions` endpoint
- [ ] High-throughput serving with vLLM (multi-LoRA, continuous batching)
- [ ] Merged and GGUF exports for CPU and edge inference
- [ ] Docker image and CI

---

## Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen3 model family
- [Hugging Face](https://huggingface.co/) for `transformers`, `peft`, `trl`, `datasets` and `accelerate`
- [HuggingFaceH4](https://huggingface.co/HuggingFaceH4) for the UltraChat 200k dataset
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)

Use of the base model and dataset is subject to their respective licenses.

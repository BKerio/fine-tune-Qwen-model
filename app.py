"""Dyniq chat API: serves Qwen3 (optionally with our LoRA adapter) behind an API key.

Config comes from environment variables (or a .env file next to this script):
  DYNIQ_API_KEY   required; clients send it in the "x-api-key" header
  BASE_MODEL      default "Qwen/Qwen3-4B"
  ADAPTER_PATH    optional; folder with the fine-tuned LoRA adapter
  PORT            default 5001
"""

import os
import secrets
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_dotenv(path=Path(__file__).with_name(".env")):
    """Minimal .env reader so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

API_KEY = os.environ.get("DYNIQ_API_KEY", "")
if not API_KEY:
    raise SystemExit("DYNIQ_API_KEY is not set. Put it in .env or the environment.")

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
PORT = int(os.environ.get("PORT", "5001"))
MAX_NEW_TOKENS_LIMIT = 1024

# ---------------------------------------------------------------- model
print(f"Loading {BASE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    # 4-bit on GPU, same settings as training. fp16 because T4 has no native bf16.
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
        dtype=dtype,
    )
else:
    print("No GPU found: running on CPU (slow).")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

if ADAPTER_PATH:
    from peft import PeftModel

    print(f"Applying adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()
MODEL_NAME = "dyniq-qwen-1" if ADAPTER_PATH else BASE_MODEL
generate_lock = threading.Lock()  # one generation at a time on a single GPU

# ---------------------------------------------------------------- api
app = Flask(__name__)


def authorized(req):
    return secrets.compare_digest(req.headers.get("x-api-key", ""), API_KEY)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.post("/chat")
def chat():
    """Body: {"messages": [{"role": "user", "content": "..."}, ...],
              "max_new_tokens": 512, "temperature": 0.7, "thinking": false}"""
    if not authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages or not all(
        isinstance(m, dict)
        and m.get("role") in ("system", "user", "assistant")
        and isinstance(m.get("content"), str)
        for m in messages
    ):
        return jsonify({"error": "messages must be a non-empty list of {role, content}"}), 400

    try:
        max_new_tokens = min(int(data.get("max_new_tokens", 512)), MAX_NEW_TOKENS_LIMIT)
        temperature = float(data.get("temperature", 0.7))
    except (TypeError, ValueError):
        return jsonify({"error": "max_new_tokens and temperature must be numbers"}), 400

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(data.get("thinking", False)),
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
    else:
        gen_kwargs["do_sample"] = False

    with generate_lock, torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return jsonify({"model": MODEL_NAME, "reply": reply, "tokens": len(new_tokens)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)

import os
import sys
from pathlib import Path


def load_env(path=Path(__file__).parent / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

from transformers import AutoModelForCausalLM, AutoTokenizer

token = os.environ.get("HF_TOKEN")

OPEN_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

GATED_MODELS = [
    "google/gemma-3-1b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

failed = []

for model_id in OPEN_MODELS:
    print(f"downloading model {model_id} ...")
    try:
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id)
        print(f"ok")
    except Exception as e:
        print(f"failed: {e}")
        failed.append(model_id)

if token:
    for model_id in GATED_MODELS:
        print(f"downloading {model_id} ...")
        try:
            AutoTokenizer.from_pretrained(model_id, token=token)
            AutoModelForCausalLM.from_pretrained(model_id, token=token)
            print(f"ok")
        except Exception as e:
            print(f"failed (check license acceptance on huggingface.co): {e}")
            failed.append(model_id)
else:
    print(f"skipping gated models, no HF_TOKEN in .env")

if failed:
    print(f"failed: {', '.join(failed)}")
    sys.exit(1)

print("done")

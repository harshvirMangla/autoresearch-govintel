"""
One-time download of Llama 3.2 1B Instruct.

Requirements:
    1. Accept license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
    2. export HF_TOKEN=your_token  (from https://huggingface.co/settings/tokens)

Usage:
    python3 download_model.py
"""

import os
import sys
from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

token = os.environ.get("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN not set.")
    print("  1. Get token:      https://huggingface.co/settings/tokens")
    print("  2. Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("  3. Run:            export HF_TOKEN=your_token")
    sys.exit(1)

print(f"Downloading {MODEL_ID} ...")
path = snapshot_download(MODEL_ID, token=token)
print(f"Cached at: {path}")
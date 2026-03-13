"""
Quick test: verify the Fireworks API key works and the model outputs
the correct DESCRIPTION + python block format that agent.py expects.

Usage:
    python3 test_api.py
"""

import os
import re
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

from openai import OpenAI

MODEL   = "accounts/fireworks/models/minimax-m2p5"
API_KEY = os.environ.get("FIREWORKS_API_KEY")

if not API_KEY:
    sys.exit("FIREWORKS_API_KEY not found in .env")

client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=API_KEY)

MINI_TRAIN = """\
MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
TIME_BUDGET   = 300
LORA_RANK     = 16
LEARNING_RATE = 2e-4
BATCH_SIZE    = 2
"""

PROMPT = (
    f"Current train.py (abbreviated):\n```python\n{MINI_TRAIN}\n```\n\n"
    "Propose one small experiment (e.g. change LORA_RANK to 32).\n\n"
    "Your response MUST follow this exact format, no other text:\n\n"
    "DESCRIPTION: <one line>\n"
    "```python\n"
    "<complete new train.py>\n"
    "```"
)

print(f"Testing {MODEL} ...")
print("-" * 50)

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an ML research assistant."},
            {"role": "user",   "content": PROMPT},
        ],
        max_tokens=512,
        temperature=0.7,
    )
except Exception as e:
    sys.exit(f"API call failed: {e}")

content = resp.choices[0].message.content
print("Raw response:")
print(content)
print("-" * 50)

desc = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", content)
code = re.search(r"```python\n(.*?)```", content, re.DOTALL)

print(f"DESCRIPTION parsed: {'YES - ' + desc.group(1).strip() if desc else 'NO (missing)'}")
print(f"python block:       {'YES (' + str(len(code.group(1))) + ' chars)' if code else 'NO (missing)'}")

if desc and code:
    print("\nFormat check: PASSED")
else:
    print("\nFormat check: FAILED - the model did not follow the required format.")
    print("The agent prompt may need adjustment.")
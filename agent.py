"""
Autonomous experiment loop for autoresearch using Fireworks AI.

Usage:
    export FIREWORKS_API_KEY=your_key
    python3 agent.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

MODEL = "accounts/fireworks/models/qwen3p5-397b-a17b"
TRAIN_PY = Path("train.py")
PROGRAM_MD = Path("program.md")
RESULTS_TSV = Path("results.tsv")
TIMEOUT_TRAIN = 700
TIMEOUT_EVAL = 60


def sh(cmd, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def run_training():
    r = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=TIMEOUT_TRAIN,
    )
    return r.stdout + r.stderr


def parse_metrics(output):
    bpb = re.search(r"^val_bpb:\s+([\d.]+)", output, re.MULTILINE)
    vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", output, re.MULTILINE)
    if not bpb:
        return None, None
    return float(bpb.group(1)), float(vram.group(1)) / 1024 if vram else 0.0


def log(commit, bpb, mem_gb, status, description):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with RESULTS_TSV.open("a") as f:
        f.write(f"{commit}\t{bpb:.6f}\t{mem_gb:.1f}\t{status}\t{description}\n")


def ask_agent(client, program, train, results):
    prompt = (
        f"Current train.py:\n```python\n{train}\n```\n\n"
        f"Results so far:\n{results or '(none yet)'}\n\n"
        "Propose the next experiment. First write a single-line description "
        "starting with 'Description: ', then output the complete new train.py "
        "inside a ```python``` block."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": program},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8192,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def extract(response):
    desc_m = re.search(r"^Description:\s*(.+)$", response, re.MULTILINE)
    code_m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    description = desc_m.group(1).strip() if desc_m else "no description"
    code = code_m.group(1) if code_m else response
    return description, code


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("Set FIREWORKS_API_KEY environment variable.")

    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=api_key)
    program = PROGRAM_MD.read_text()

    print("=== autoresearch/govintel-legal ===")
    print("Running baseline...")

    sh("git add train.py && git commit -m 'baseline' --allow-empty")
    commit = sh("git rev-parse --short HEAD")
    output = run_training()
    bpb, mem = parse_metrics(output)

    if bpb is None:
        print("Baseline crashed:\n", output[-3000:])
        sys.exit(1)

    log(commit, bpb, mem, "keep", "baseline")
    best_bpb = bpb
    print(f"Baseline val_bpb: {bpb:.6f}")

    while True:
        results = RESULTS_TSV.read_text() if RESULTS_TSV.exists() else ""
        train = TRAIN_PY.read_text()

        print(f"\nBest so far: {best_bpb:.6f} — asking agent...")
        response = ask_agent(client, program, train, results)
        description, new_train = extract(response)
        print(f"Experiment: {description}")

        TRAIN_PY.write_text(new_train)
        sh("git add train.py && git commit -m 'experiment'")
        commit = sh("git rev-parse --short HEAD")

        output = run_training()
        bpb, mem = parse_metrics(output)

        if bpb is None:
            print("Crashed — reverting")
            log(commit, 0.0, 0.0, "crash", description)
            sh("git reset --hard HEAD~1")
            continue

        if bpb < best_bpb:
            log(commit, bpb, mem, "keep", description)
            best_bpb = bpb
            print(f"Improvement: {bpb:.6f}")
        else:
            log(commit, bpb, mem, "discard", description)
            sh("git reset --hard HEAD~1")
            print(f"No improvement ({bpb:.6f}) — reverted")


if __name__ == "__main__":
    main()
"""
Autonomous experiment loop — Fireworks AI (Qwen3.5 397B) + autoresearch.

Usage:
    export FIREWORKS_API_KEY=your_key
    python3 agent.py
"""

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

MODEL = "accounts/fireworks/models/qwen3p5-397b-a17b"
TRAIN_PY = Path("train.py")
PROGRAM_MD = Path("program.md")
RESULTS = Path("results.tsv")
BACKUP_PY = Path("train.py.bak")

TIMEOUT_TRAIN = 800
MAX_RETRIES = 2

ALLOWED_MODULES = {
    "os", "sys", "math", "time", "gc", "random", "re", "json", "copy",
    "dataclasses", "contextlib", "pathlib", "collections", "itertools",
    "functools", "typing", "warnings", "hashlib",
    "torch", "transformers", "peft", "datasets", "numpy", "accelerate",
}


# ---------------------------------------------------------------------------

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
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
    if not RESULTS.exists():
        RESULTS.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with RESULTS.open("a") as f:
        f.write(f"{commit}\t{bpb:.6f}\t{mem_gb:.1f}\t{status}\t{description}\n")


def validate_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in ALLOWED_MODULES:
                    return False, f"Forbidden import: {node.module}"
    return True, None


def extract_code(response):
    m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    return response


def extract_description(response):
    m = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response)
    return m.group(1).strip() if m else "no description"


def ask_agent(client, program, train, results, error_context=""):
    error_section = f"\nThe previous attempt failed with:\n{error_context}\nFix the issue.\n" if error_context else ""
    prompt = (
        f"Current train.py:\n```python\n{train}\n```\n\n"
        f"Results so far:\n{results or '(none yet)'}\n"
        f"{error_section}\n"
        "Propose the next experiment. Available packages: "
        "torch, transformers, peft, datasets, numpy, accelerate (and stdlib).\n\n"
        "Your response MUST follow this exact format — no other text:\n\n"
        "DESCRIPTION: <one line describing the change>\n"
        "```python\n"
        "<complete new train.py>\n"
        "```"
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


# ---------------------------------------------------------------------------

def run_experiment(client, program, description_hint=""):
    results = RESULTS.read_text() if RESULTS.exists() else ""
    orig_code = TRAIN_PY.read_text()

    response = ask_agent(client, program, orig_code, results)
    description = extract_description(response)
    new_code = extract_code(response)

    for attempt in range(MAX_RETRIES):
        ok, err = validate_code(new_code)
        if ok:
            break
        print(f"  Validation failed ({err}), asking agent to fix...")
        response = ask_agent(client, program, orig_code, results, error_context=err)
        description = extract_description(response)
        new_code = extract_code(response)
    else:
        print("  Could not get valid code after retries — skipping.")
        return None, None, description

    shutil.copy(TRAIN_PY, BACKUP_PY)
    TRAIN_PY.write_text(new_code)

    sh("git add train.py && git commit -m 'experiment'")
    commit = sh("git rev-parse --short HEAD")
    print(f"  Commit {commit}: {description}")

    try:
        output = run_training()
    except subprocess.TimeoutExpired:
        output = ""
        print("  Timed out.")

    bpb, mem = parse_metrics(output)

    if bpb is None:
        error_tail = output[-2000:] if output else "(no output)"
        print("  Crashed. Attempting fix...")

        fixed_response = ask_agent(client, program, new_code, results, error_context=error_tail)
        fixed_code = extract_code(fixed_response)
        ok, _ = validate_code(fixed_code)

        if ok:
            TRAIN_PY.write_text(fixed_code)
            sh("git add train.py && git commit -m 'fix crash'")
            commit = sh("git rev-parse --short HEAD")
            try:
                output = run_training()
            except subprocess.TimeoutExpired:
                output = ""
            bpb, mem = parse_metrics(output)

        if bpb is None:
            log(commit, 0.0, 0.0, "crash", description)
            sh("git reset --hard HEAD~1")
            shutil.copy(BACKUP_PY, TRAIN_PY)
            print("  Could not fix — reverted.")
            return None, None, description

    return bpb, mem, description


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("Set FIREWORKS_API_KEY.")

    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=api_key)
    program = PROGRAM_MD.read_text()

    print("=== autoresearch / govintel-legal ===")
    print("Running baseline...")

    sh("git add train.py && git commit -m 'baseline' --allow-empty")
    commit = sh("git rev-parse --short HEAD")

    try:
        output = run_training()
    except subprocess.TimeoutExpired:
        sys.exit("Baseline timed out — check train.py runs correctly first.")

    bpb, mem = parse_metrics(output)
    if bpb is None:
        print(output[-3000:])
        sys.exit("Baseline crashed.")

    log(commit, bpb, mem, "keep", "baseline")
    best_bpb = bpb
    print(f"Baseline val_bpb: {bpb:.6f}\n")

    while True:
        print(f"Best so far: {best_bpb:.6f} — asking agent...")
        bpb, mem, description = run_experiment(client, program)

        if bpb is None:
            continue

        if bpb < best_bpb:
            log(sh("git rev-parse --short HEAD"), bpb, mem, "keep", description)
            best_bpb = bpb
            print(f"  Improvement → {bpb:.6f}")
        else:
            log(sh("git rev-parse --short HEAD"), bpb, mem, "discard", description)
            sh("git reset --hard HEAD~1")
            shutil.copy(BACKUP_PY, TRAIN_PY)
            print(f"  No improvement ({bpb:.6f}) — reverted.")


if __name__ == "__main__":
    main()

import ast
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _load_env(path=Path(__file__).parent / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

from openai import OpenAI

MODEL = "accounts/fireworks/models/minimax-m2p5"
TRAIN_PY = Path("train.py")
PROGRAM_MD = Path("program.md")
RESULTS = Path("results.tsv")
BACKUP_PY = Path("train.py.bak")
LOGS_DIR = Path("logs")

TIMEOUT_TRAIN = 800
MAX_RETRIES = 2

ALLOWED_MODULES = {
    "os", "sys", "math", "time", "gc", "random", "re", "json", "copy",
    "dataclasses", "contextlib", "pathlib", "collections", "itertools",
    "functools", "typing", "warnings", "hashlib", "struct", "io",
    "torch", "transformers", "peft", "datasets", "numpy", "accelerate",
    "scipy", "sklearn", "sentence_transformers", "tqdm",
}

LOGS_DIR.mkdir(exist_ok=True)
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
_master_log = LOGS_DIR / f"run_{_run_id}.log"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, also_print=True):
    line = f"[{_ts()}] {msg}"
    if also_print:
        print(line)
    with _master_log.open("a") as f:
        f.write(line + "\n")


def log_agent_response(response, description, exp_num):
    path = LOGS_DIR / f"exp_{exp_num:03d}_agent_response.txt"
    path.write_text(
        f"experiment {exp_num}\n"
        f"description: {description}\n"
        f"timestamp: {datetime.now().isoformat()}\n\n"
        f"{response}\n"
    )
    log(f"  agent response saved to {path.name}")


def log_training_output(output, commit, exp_num):
    path = LOGS_DIR / f"exp_{exp_num:03d}_{commit}_train.log"
    path.write_text(output)
    log(f"  training output saved to {path.name}")


def log_train_code(code, exp_num):
    path = LOGS_DIR / f"exp_{exp_num:03d}_train.py"
    path.write_text(code)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def run_training(exp_num):
    log("  launching train.py ...")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=TIMEOUT_TRAIN,
    )
    elapsed = time.time() - t0
    output = r.stdout + r.stderr

    lines = output.strip().splitlines()
    step_lines = [l for l in lines if l.startswith("step ")]
    if step_lines:
        log(f"  last step: {step_lines[-1].strip()}")
    summary_start = next((i for i, l in enumerate(lines) if l.strip() == "---"), None)
    if summary_start is not None:
        for l in lines[summary_start:]:
            log(f"  {l}")

    log(f"  finished in {elapsed:.0f}s")
    return output


def parse_metrics(output):
    bpb = re.search(r"^val_bpb:\s+([\d.]+)", output, re.MULTILINE)
    vram = re.search(r"^peak_vram_mb:\s+([\d.]+)", output, re.MULTILINE)
    steps = re.search(r"^num_steps:\s+(\d+)", output, re.MULTILINE)
    rank = re.search(r"^lora_rank:\s+(\d+)", output, re.MULTILINE)
    coverage = re.search(r"^data_coverage:\s+([\d.]+)%", output, re.MULTILINE)
    if not bpb:
        return None, None, None, None, None
    return (
        float(bpb.group(1)),
        float(vram.group(1)) / 1024 if vram else 0.0,
        int(steps.group(1)) if steps else 0,
        int(rank.group(1)) if rank else 0,
        float(coverage.group(1)) if coverage else 0.0,
    )


def record(commit, bpb, mem_gb, coverage, status, description):
    if not RESULTS.exists():
        RESULTS.write_text("commit\tval_bpb\tmemory_gb\tcoverage_pct\tstatus\tdescription\n")
    with RESULTS.open("a") as f:
        f.write(f"{commit}\t{bpb:.6f}\t{mem_gb:.1f}\t{coverage:.1f}\t{status}\t{description}\n")


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
    for pattern in [r"```python\n(.*?)```", r"```\n(.*?)```"]:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1)
    return response


def extract_description(response):
    m = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response)
    return m.group(1).strip() if m else "no description"


def call_api(client, messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=16000,
        temperature=0.7,
        stream=False,
    )
    return resp.choices[0].message.content


def ask_agent(client, program, train_code, results_text, error_context=""):
    error_section = (
        f"\nThe previous attempt failed with this error:\n```\n{error_context}\n```\nFix it.\n"
        if error_context else ""
    )
    prompt = (
        f"Current train.py:\n```python\n{train_code}\n```\n\n"
        f"Results so far:\n{results_text or '(none yet, this will be the baseline run)'}\n"
        f"{error_section}\n"
        f"Available packages: {', '.join(sorted(ALLOWED_MODULES))}.\n\n"
        "Your response MUST follow this exact format, no other text:\n\n"
        "DESCRIPTION: <one line, what changes and why>\n"
        "```python\n"
        "<complete new train.py>\n"
        "```"
    )
    messages = [
        {"role": "system", "content": program},
        {"role": "user", "content": prompt},
    ]
    log("  calling Fireworks API ...")
    try:
        return call_api(client, messages)
    except Exception as e:
        err_str = str(e).lower()
        if any(x in err_str for x in ("credit", "billing", "quota", "payment", "402", "429")):
            log(f"  API credits exhausted: {e}")
            log("  shutting down, results saved to results.tsv and logs/")
            sys.exit(0)
        log(f"  API error: {e}, retrying in 15s ...")
        time.sleep(15)
        return call_api(client, messages)


def run_experiment(client, program, exp_num):
    results_text = RESULTS.read_text() if RESULTS.exists() else ""
    orig_code = TRAIN_PY.read_text()

    try:
        response = ask_agent(client, program, orig_code, results_text)
    except Exception as e:
        log(f"  API failed after retry: {e}, skipping")
        return None, None, None, None, None, "api failure"

    description = extract_description(response)
    new_code = extract_code(response)

    log_agent_response(response, description, exp_num)

    for attempt in range(MAX_RETRIES):
        ok, err = validate_code(new_code)
        if ok:
            break
        log(f"  validation failed: {err}, asking agent to fix (attempt {attempt + 1})")
        try:
            response = ask_agent(client, program, orig_code, results_text, error_context=err)
        except Exception as e:
            log(f"  API failed during fix: {e}, skipping")
            return None, None, None, None, None, description
        description = extract_description(response)
        new_code = extract_code(response)
        log_agent_response(response, f"{description} [fix {attempt + 1}]", exp_num)
    else:
        log("  could not get valid code after retries, skipping")
        return None, None, None, None, None, description

    log_train_code(new_code, exp_num)
    shutil.copy(TRAIN_PY, BACKUP_PY)
    TRAIN_PY.write_text(new_code)

    commit_msg = description[:72].replace("'", "").replace('"', "")
    sh(f"git add train.py && git commit -m '{commit_msg}'")
    commit = sh("git rev-parse --short HEAD")
    log(f"  committed: {commit}")

    try:
        output = run_training(exp_num)
    except subprocess.TimeoutExpired:
        output = ""
        log("  training timed out")

    log_training_output(output, commit, exp_num)
    bpb, mem, steps, rank, coverage = parse_metrics(output)

    if bpb is None:
        error_tail = output[-3000:] if output else "(no output)"
        log("  crashed, sending error to agent for fix ...")
        log(f"  error tail:\n{error_tail}", also_print=False)

        try:
            fix_response = ask_agent(client, program, new_code, results_text, error_context=error_tail)
        except Exception as e:
            log(f"  API failed during crash fix: {e}")
            fix_response = None

        if fix_response:
            fix_code = extract_code(fix_response)
            log_agent_response(fix_response, f"{description} [crash fix]", exp_num)

            ok, _ = validate_code(fix_code)
            if ok:
                TRAIN_PY.write_text(fix_code)
                sh("git add train.py && git commit -m 'fix crash'")
                commit = sh("git rev-parse --short HEAD")
                try:
                    output = run_training(exp_num)
                except subprocess.TimeoutExpired:
                    output = ""
                log_training_output(output, commit, exp_num)
                bpb, mem, steps, rank, coverage = parse_metrics(output)

        if bpb is None:
            record(commit, 0.0, 0.0, 0.0, "crash", description)
            sh("git reset --hard HEAD~1")
            shutil.copy(BACKUP_PY, TRAIN_PY)
            log("  could not fix, reverted")
            return None, None, None, None, None, description

    return bpb, mem, steps, rank, coverage, description


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("FIREWORKS_API_KEY not found in .env")

    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=api_key)
    program = PROGRAM_MD.read_text()

    log(f"starting autoresearch govintel-legal")
    log(f"master log: {_master_log}")
    log(f"model: {MODEL}")
    log("")

    log("baseline run")
    sh("git add train.py && git commit -m 'baseline' --allow-empty")
    commit = sh("git rev-parse --short HEAD")

    try:
        output = run_training(0)
    except subprocess.TimeoutExpired:
        sys.exit("baseline timed out, fix train.py first")

    log_training_output(output, commit, 0)
    bpb, mem, steps, rank, coverage = parse_metrics(output)

    if bpb is None:
        log("baseline crashed. Fix train.py first.")
        log(output[-3000:])
        sys.exit(1)

    record(commit, bpb, mem, coverage, "keep", "baseline")
    best_bpb = bpb
    log(f"baseline: val_bpb={bpb:.6f} mem={mem:.1f}GB steps={steps} coverage={coverage:.1f}% rank={rank}")

    exp_num = 1
    while True:
        log("")
        log(f"experiment {exp_num}  (best so far: {best_bpb:.6f})")

        bpb, mem, steps, rank, coverage, description = run_experiment(client, program, exp_num)

        if bpb is None:
            log("  skipped")
            exp_num += 1
            continue

        commit = sh("git rev-parse --short HEAD")
        log(f"  val_bpb={bpb:.6f} mem={mem:.1f}GB steps={steps} coverage={coverage:.1f}% rank={rank}")
        log(f"  {description}")

        if bpb < best_bpb:
            record(commit, bpb, mem, coverage, "keep", description)
            improvement = best_bpb - bpb
            best_bpb = bpb
            log(f"  improved by {improvement:.6f}, new best: {best_bpb:.6f}")
        else:
            record(commit, bpb, mem, coverage, "discard", description)
            sh("git reset --hard HEAD~1")
            shutil.copy(BACKUP_PY, TRAIN_PY)
            log("  no improvement, reverted")

        exp_num += 1


if __name__ == "__main__":
    main()
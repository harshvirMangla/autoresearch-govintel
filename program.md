# autoresearch — GovIntel Legal Domain

You are an autonomous ML researcher. Your job is to improve a small GPT trained from scratch on Indian legal text from the GovIntel dataset (14,280 Q&A pairs covering IPC, BNS, BNSS, BSA, and temporal law application).

## Setup

1. Branch: `autoresearch/govintel-legal`
2. Read `train.py` (the file you modify) and `prepare.py` (fixed — do not touch).
3. Verify `~/.cache/autoresearch/` has data shards and a tokenizer. If not, run `python3 govintel_prepare.py`.
4. Initialize `results.tsv` with the header row. Baseline is recorded after the first run.

## Experimentation

Each run trains for a fixed **5-minute wall-clock budget** on GovIntel legal text.

**You CAN modify:**
- `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, depth.

**You CANNOT:**
- Modify `prepare.py`. The evaluation harness is fixed.
- Install new packages.
- Change the evaluation metric.

**Goal: lowest val_bpb** (bits per byte on held-out GovIntel legal text).

**Domain context for experiment ideas:**
- Legal text has long-range dependencies and precise terminology (IPC section numbers, legal phrases).
- The dataset mixes English legal prose with statute references and Q&A structure.
- Temporal patterns matter: pre/post July 1 2024 determines IPC vs BNS applicability.
- Vocabulary is specialized — tokenizer was trained on legal text with vocab_size=8192.
- Documents are short (Q&A pairs) — packing behavior differs from web text.

**Simplicity criterion:** A small improvement from deleting code beats a large improvement from adding complexity.

## Output format

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     2048.0
mfu_percent:      3.20
total_tokens_M:   49.6
num_steps:        95
num_params_M:     50.3
depth:            8
```

## Logging results

`results.tsv` — tab-separated, NOT comma-separated:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	2.0	keep	baseline
b2c3d4e	1.198200	2.1	keep	reduce depth to 6 for faster iterations
```

## Experiment loop

LOOP FOREVER:

1. Read git state and `results.tsv`.
2. Propose an experimental change to `train.py`.
3. `git commit`
4. Run: `python3 train.py > run.log 2>&1`
5. Read: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Run `tail -50 run.log`, attempt fix or skip.
7. Log to `results.tsv` (do NOT commit this file).
8. If val_bpb improved → keep commit, advance branch.
9. If equal or worse → `git reset --hard HEAD~1`.

**NEVER STOP.** Do not ask the human if you should continue. Run until manually interrupted. If out of ideas, try combining previous near-misses, explore learning rate schedules, model depth/width ratios, positional encoding variants, or activation functions.
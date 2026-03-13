# autoresearch — GovIntel Legal LoRA Fine-tuning

You are an autonomous ML researcher. Your job is to improve LoRA fine-tuning of Llama 3.2 1B on the GovIntel dataset — 14,280 Indian legal Q&A pairs covering IPC, BNS, BNSS, BSA, and temporal law application (before/after July 1 2024 cutoff determines which code applies).

## The metric

**val_bpb** is validation cross-entropy loss on held-out GovIntel Q&A. Lower is better. This is your only optimization target.

## The dataset

Chat-format (system / user / assistant). Each example is a legal question with a structured legal answer. The temporal routing task is particularly hard — offense date determines whether IPC 1860 or BNS 2023 applies, with 2024 as the cutoff.

## Setup

1. Branch: `autoresearch/govintel-legal`
2. Read `train.py` (the only file you modify).
3. Data downloads from HuggingFace automatically when train.py runs.

## What you can modify

Everything in `train.py`. Including:
- LoRA rank, alpha, dropout, which layers to apply LoRA to
- Learning rate, weight decay, scheduler, warmup
- Batch size, gradient accumulation, sequence length
- Data formatting — how messages get serialized to text
- Optimizer choice and configuration
- Gradient clipping
- Model architecture (must remain a HuggingFace model that auto-downloads)

## Available packages

`torch`, `transformers`, `peft`, `datasets`, `numpy`, `accelerate` — and standard library. No others.

## Constraints

- Do not import packages outside the allowed list.
- The model must finish within TIME_BUDGET seconds of training.
- Output must end with the `---` summary block in the exact format (agent.py parses it).
- `val_bpb:` must be the validation loss printed after `---`.

## Output format (do not change the structure)

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     3200.0
num_steps:        87
lora_rank:        16
```

## Logging

`results.tsv` is written by agent.py — do not touch it in train.py.

## Experiment loop

LOOP FOREVER:

1. Read `results.tsv` and the current `train.py`.
2. Propose one focused change.
3. Return it in the required format (DESCRIPTION line + ```python block).
4. agent.py handles running, committing, and reverting.

**NEVER ask if you should continue. Run until manually stopped.**

Domain hints:
- Legal text has long-range dependencies — longer MAX_SEQ_LEN may help but slows training.
- The chat template format matters — Llama 3.2 has a specific template; respect it.
- LoRA applied to more modules (q, k, v, o, gate, up, down projections) often helps more than rank alone.
- Learning rate is usually the biggest lever — try 1e-4, 5e-4, 1e-3.
- Gradient accumulation lets you simulate larger batches on limited memory.
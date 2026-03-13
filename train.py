"""
LoRA fine-tuning of Llama 3.2 1B on GovIntel legal dataset.
Autoresearch target: modify everything in this file to minimize val_bpb.
Usage: python3 train.py
"""

import contextlib
import gc
import math
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters — agent modifies these
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TIME_BUDGET = 300  # wall-clock training seconds

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_MODULES = ["q_proj", "v_proj"]

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
device_type = device.type
autocast_ctx = (
    torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cuda" else contextlib.nullcontext()
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

t_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_MODULES, bias="none",
))
model = model.to(device)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Device: {device}  |  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ds = load_dataset("aashnasharma/govintel-legal-dataset")


def tokenize(examples):
    texts = []
    for msgs in examples["messages"]:
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = " ".join(m.get("content", "") for m in msgs)
        texts.append(text)
    out = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out


train_ds = ds["train"].map(tokenize, batched=True, remove_columns=["messages"])
val_ds = ds["test"].select(range(min(256, len(ds["test"])))).map(tokenize, batched=True, remove_columns=["messages"])
train_ds.set_format("torch")
val_ds.set_format("torch")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
smooth_loss = 0.0
step = 0

gc.collect();
gc.freeze();
gc.disable()

train_iter = iter(train_loader)

while True:
    t0 = time.time()
    optimizer.zero_grad()
    accum_loss = 0.0

    for _ in range(GRAD_ACCUM):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        with autocast_ctx:
            loss = model(input_ids=ids, attention_mask=mask, labels=lbls).loss
        (loss / GRAD_ACCUM).backward()
        accum_loss += loss.item() / GRAD_ACCUM

    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()

    if math.isnan(accum_loss) or accum_loss > 100:
        print("FAIL");
        exit(1)

    t1 = time.time()
    if step > 2:
        total_training_time += t1 - t0

    smooth_loss = 0.9 * smooth_loss + 0.1 * accum_loss
    debiased = smooth_loss / (1 - 0.9 ** (step + 1))
    progress = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:04d} ({progress:.1f}%) | loss: {debiased:.4f} | remaining: {remaining:.0f}s", end="",
          flush=True)
    step += 1

    if step > 2 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        with autocast_ctx:
            total_loss += model(input_ids=ids, attention_mask=mask, labels=lbls).loss.item()
        n_batches += 1

val_loss = total_loss / max(n_batches, 1)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_mb = (
    torch.mps.current_allocated_memory() / 1024 ** 2 if device_type == "mps" else
    torch.cuda.max_memory_allocated() / 1024 ** 2 if device_type == "cuda" else 0.0
)

print("---")
print(f"val_bpb:          {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_mb:.1f}")
print(f"num_steps:        {step}")
print(f"lora_rank:        {LORA_RANK}")

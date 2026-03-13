"""
Download GovIntel and prepare it for autoresearch.

Creates parquet shards matching prepare.py's expected layout in
~/.cache/autoresearch/data/, then trains a tokenizer on legal text.

Usage:
    python3 govintel_prepare.py
"""

import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from prepare import train_tokenizer, CACHE_DIR, DATA_DIR, VAL_SHARD, VAL_FILENAME

TOKENIZER_DIR = Path(CACHE_DIR) / "tokenizer"
DATA_DIR = Path(DATA_DIR)


def to_text(ex):
    messages = ex.get("messages", [])
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "").strip()
        if role == "user":
            parts.append(f"Question: {content}")
        elif role == "assistant":
            parts.append(f"Answer: {content}")
    return "\n".join(parts)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading aashnasharma/govintel-legal-dataset...")
    ds = load_dataset("aashnasharma/govintel-legal-dataset")

    texts = []
    for split in ds:
        for ex in ds[split]:
            t = to_text(ex)
            if t.strip():
                texts.append(t)

    print(f"Total examples: {len(texts)}")

    random.seed(42)
    random.shuffle(texts)
    n_val = max(200, len(texts) // 10)
    val_texts = texts[-n_val:]
    train_texts = texts[:-n_val]
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

    val_path = DATA_DIR / VAL_FILENAME
    if not val_path.exists():
        pq.write_table(pa.table({"text": val_texts}), val_path)
        print(f"Saved {val_path.name}")

    train_path = DATA_DIR / "shard_00000.parquet"
    if not train_path.exists():
        pq.write_table(pa.table({"text": train_texts}), train_path)
        print(f"Saved {train_path.name}")

    print("\nTraining tokenizer on legal text...")
    train_tokenizer()
    print("Done. Ready to run train.py")


if __name__ == "__main__":
    main()
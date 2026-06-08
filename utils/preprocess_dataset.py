"""
preprocess_dataset.py

Cleans raw datasets and writes JSONL (train/val splits) to datasets/processed/<name>/.
Then tokenizes the JSONL with the Qwen3 tokenizer and writes .bin + metadata.json
to datasets/tokenized/<name>/.

Usage:
    python utils/preprocess_dataset.py --dataset indiccorp
    python utils/preprocess_dataset.py --dataset culturax
    python utils/preprocess_dataset.py --dataset wikipedia

    # Skip tokenization (only produce JSONL):
    python utils/preprocess_dataset.py --dataset wikipedia --skip-tokenize

    # Force reprocess even if processed/ already exists:
    python utils/preprocess_dataset.py --dataset indiccorp --force
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to the repo root, i.e. thalam1/)
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent          # thalam1/
DATASETS   = REPO_ROOT.parent / "datasets"                   # ../datasets/
RAW        = DATASETS / "raw"
PROCESSED  = DATASETS / "processed"
TOKENIZED  = DATASETS / "tokenized"

VAL_RATIO  = 0.01   # 1 % of examples go to val split
DTYPE      = np.uint32   # Qwen vocab > 65535, so uint16 is not enough

# ---------------------------------------------------------------------------
# Tokenizer (loaded once, lazily)
# ---------------------------------------------------------------------------
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print("[tokenizer] Loading Qwen3 tokenizer …")
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True,
        )
    return _tokenizer


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Common cleaning applied to every dataset."""
    text = text.strip()
    # collapse runs of whitespace/newlines to a single space
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid(text: str, min_chars: int = 50) -> bool:
    return len(text) >= min_chars


# ---------------------------------------------------------------------------
# Per-dataset readers  – yield clean text strings one at a time
# ---------------------------------------------------------------------------

def read_indiccorp(raw_dir: Path):
    """IndicCorpV2: single large .txt file, one sentence / paragraph per line."""
    txt_files = list(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    for txt_file in txt_files:
        print(f"  [indiccorp] Reading {txt_file.name} …")
        with open(txt_file, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                line = line.strip()
                if not line:
                    if buffer:
                        text = clean_text(" ".join(buffer))
                        if is_valid(text):
                            yield text
                        buffer = []
                else:
                    buffer.append(line)
            # flush remaining
            if buffer:
                text = clean_text(" ".join(buffer))
                if is_valid(text):
                    yield text


def read_culturax(raw_dir: Path):
    """CulturaX: parquet files with a 'text' column. Streamed in batches to keep RAM low."""
    import pyarrow.parquet as pq

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {raw_dir}")

    for pf in parquet_files:
        print(f"  [culturax] Reading {pf.name} …")
        pf_reader = pq.ParquetFile(pf)
        for batch in pf_reader.iter_batches(batch_size=10_000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not isinstance(text, str):
                    continue
                text = clean_text(text)
                if is_valid(text):
                    yield text


def read_wikipedia(raw_dir: Path):
    """
    Tamil Wikipedia dump processed by WikiExtractor.
    Directory structure: raw/wikipedia/AA/wiki_00, AA/wiki_01, …
    Each file contains <doc …> … </doc> blocks.
    """
    doc_pattern = re.compile(
        r'<doc[^>]*>(.*?)</doc>', re.DOTALL | re.IGNORECASE
    )

    for wiki_file in sorted(raw_dir.rglob("wiki_*")):
        with open(wiki_file, "r", encoding="utf-8") as f:
            content = f.read()
        for match in doc_pattern.finditer(content):
            text = clean_text(match.group(1))
            if is_valid(text):
                yield text


# Add new datasets here in the future:
# def read_books(raw_dir: Path): ...
# def read_news(raw_dir: Path): ...

READERS = {
    "indiccorp": read_indiccorp,
    "culturax":  read_culturax,
    "wikipedia": read_wikipedia,
    # "books": read_books,
    # "news":  read_news,
}


# ---------------------------------------------------------------------------
# JSONL writing
# ---------------------------------------------------------------------------

def write_jsonl_splits(dataset_name: str, force: bool = False):
    """Read raw data → clean → split → write processed/<name>/train.jsonl + val.jsonl"""

    # Normalise name for raw directory lookup — case-insensitive match
    raw_dir = None
    if RAW.exists():
        for candidate in RAW.iterdir():
            if candidate.is_dir() and candidate.name.lower() == dataset_name.lower():
                raw_dir = candidate
                break
    if raw_dir is None:
        sys.exit(f"[ERROR] Raw directory not found for '{dataset_name}'. "
                 f"Expected a folder matching '{dataset_name}' (case-insensitive) inside {RAW}")

    out_dir = PROCESSED / dataset_name
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    if train_path.exists() and val_path.exists() and not force:
        print(f"[skip] Processed JSONL already exists for '{dataset_name}'. "
              "Use --force to reprocess.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    reader = READERS.get(dataset_name)
    if reader is None:
        sys.exit(f"[ERROR] No reader implemented for dataset '{dataset_name}'. "
                 f"Available: {list(READERS.keys())}")

    print(f"[process] {dataset_name}: raw → JSONL …")

    # Stream through data, route to train / val by index
    train_count = val_count = 0
    with open(train_path, "w", encoding="utf-8") as ft, \
         open(val_path,   "w", encoding="utf-8") as fv:
        for i, text in enumerate(reader(raw_dir)):
            record = json.dumps({"text": text}, ensure_ascii=False)
            if i % round(1 / VAL_RATIO) == 0:
                fv.write(record + "\n")
                val_count += 1
            else:
                ft.write(record + "\n")
                train_count += 1

    print(f"  train: {train_count:,} examples → {train_path}")
    print(f"  val  : {val_count:,}  examples → {val_path}")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_split(jsonl_path: Path, bin_path: Path, tokenizer):
    """Tokenize a JSONL file and write token IDs as uint32 .bin.
    Writes in chunks so the full token list never lives in RAM.
    """
    print(f"  [tokenize] {jsonl_path.name} → {bin_path.name} …")

    CHUNK = 50_000   # flush to disk every N documents
    total_tokens = 0
    buf = []

    def _flush(f):
        nonlocal total_tokens
        arr = np.array(buf, dtype=DTYPE)
        arr.tofile(f)
        total_tokens += len(arr)
        buf.clear()

    with open(jsonl_path, "r", encoding="utf-8") as jf, \
         open(bin_path, "wb") as bf:
        for i, line in enumerate(jf):
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(tokenizer.eos_token_id)
            if len(buf) >= CHUNK:
                _flush(bf)
        if buf:
            _flush(bf)

    print(f"    {total_tokens:,} tokens written.")
    return total_tokens


def tokenize_dataset(dataset_name: str, force: bool = False):
    tok_dir = TOKENIZED / dataset_name
    tok_dir.mkdir(parents=True, exist_ok=True)

    train_bin = tok_dir / "train.bin"
    val_bin   = tok_dir / "val.bin"
    meta_path = tok_dir / "metadata.json"

    if train_bin.exists() and val_bin.exists() and not force:
        print(f"[skip] Tokenized .bin already exists for '{dataset_name}'. "
              "Use --force to re-tokenize.")
        return

    proc_dir   = PROCESSED / dataset_name
    train_jsonl = proc_dir / "train.jsonl"
    val_jsonl   = proc_dir / "val.jsonl"

    if not train_jsonl.exists() or not val_jsonl.exists():
        sys.exit(f"[ERROR] Processed JSONL not found for '{dataset_name}'. "
                 "Run without --skip-tokenize first.")

    tokenizer = get_tokenizer()

    print(f"[tokenize] {dataset_name} …")
    train_tokens = tokenize_split(train_jsonl, train_bin, tokenizer)
    val_tokens   = tokenize_split(val_jsonl,   val_bin,   tokenizer)

    metadata = {
        "dataset":      dataset_name,
        "dtype":        "uint32",
        "vocab_size":   tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens":   val_tokens,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata → {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and tokenize a Tamil dataset."
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help=f"Dataset name. Currently supported: {list(READERS.keys())}",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Only produce processed JSONL; skip tokenization.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process / re-tokenize even if outputs already exist.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.lower()

    write_jsonl_splits(dataset_name, force=args.force)

    if not args.skip_tokenize:
        tokenize_dataset(dataset_name, force=args.force)

    print(f"\n[done] '{dataset_name}' preprocessing complete.")


if __name__ == "__main__":
    main()

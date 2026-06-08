"""
data_loader.py

Multi-dataset data loader for training and evaluation.

Usage example:

    from data_loader.data_loader import MultiDatasetLoader, DatasetConfig

    config = DatasetConfig(
        datasets=["indiccorp", "culturax", "wikipedia"],
        weights=[0.5, 0.3, 0.2],   # sampling probabilities (must sum to 1)
        block_size=2048,
        batch_size=8,
    )

    train_loader = MultiDatasetLoader(split="train", config=config)
    val_loader   = MultiDatasetLoader(split="val",   config=config)

    for batch in train_loader:
        # batch: torch.Tensor of shape (batch_size, block_size), dtype=torch.long
        ...

Notes:
- Reads pre-tokenized .bin files (uint32) from ../datasets/tokenized/<name>/<split>.bin
  via np.memmap — the entire dataset never has to fit in RAM.
- Sampling is weighted: at each step a dataset is chosen proportionally to its
  weight, then a random contiguous block is drawn from it.
- Weights are automatically normalised if they don't sum to 1.
- If weights are omitted they default to uniform.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOADER_DIR  = Path(__file__).resolve().parent          # thalam1/data_loader/
REPO_ROOT   = LOADER_DIR.parent                        # thalam1/
TOKENIZED   = REPO_ROOT.parent / "datasets" / "tokenized"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """
    datasets : list of dataset names to use, e.g. ["indiccorp", "culturax"]
    weights  : sampling probability for each dataset (will be normalised).
               If None, uniform weights are used.
    block_size: number of tokens per sample (context length).
    batch_size: number of samples per batch.
    seed     : random seed for reproducibility (None = non-deterministic).
    """
    datasets:   List[str]
    weights:    Optional[List[float]] = None
    block_size: int = 2048
    batch_size: int = 8
    seed:       Optional[int] = None


# ---------------------------------------------------------------------------
# Single-dataset shard
# ---------------------------------------------------------------------------
class _DatasetShard:
    """Memory-mapped view of one tokenized .bin file."""

    def __init__(self, dataset_name: str, split: str):
        bin_path  = TOKENIZED / dataset_name / f"{split}.bin"
        meta_path = TOKENIZED / dataset_name / "metadata.json"

        if not bin_path.exists():
            raise FileNotFoundError(
                f"Tokenized file not found: {bin_path}\n"
                f"Run:  python utils/preprocess_dataset.py --dataset {dataset_name}"
            )

        self.name     = dataset_name
        self.data     = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.n_tokens = len(self.data)

        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        print(f"  [{split}] {dataset_name}: {self.n_tokens:,} tokens  ({bin_path})")

    def sample(self, block_size: int, rng: random.Random) -> np.ndarray:
        """Return a random contiguous block of (block_size + 1) tokens."""
        max_start = self.n_tokens - block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset '{self.name}' has only {self.n_tokens} tokens, "
                f"which is too small for block_size={block_size}."
            )
        start = rng.randint(0, max_start)
        return self.data[start : start + block_size + 1]


# ---------------------------------------------------------------------------
# Multi-dataset loader
# ---------------------------------------------------------------------------
class MultiDatasetLoader:
    """
    Infinite iterator that yields batches drawn from multiple tokenized datasets.

    Parameters
    ----------
    split  : "train" or "val"
    config : DatasetConfig
    """

    def __init__(self, split: str, config: DatasetConfig):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.split  = split
        self.config = config
        self.rng    = random.Random(config.seed)

        print(f"\n[MultiDatasetLoader] split={split}")
        self._shards: List[_DatasetShard] = []
        for name in config.datasets:
            self._shards.append(_DatasetShard(name, split))

        # Normalise weights
        if config.weights is None:
            n = len(self._shards)
            self._weights = [1.0 / n] * n
        else:
            if len(config.weights) != len(config.datasets):
                raise ValueError(
                    f"len(weights)={len(config.weights)} != "
                    f"len(datasets)={len(config.datasets)}"
                )
            total = sum(config.weights)
            self._weights = [w / total for w in config.weights]

        print(f"  Effective sampling weights:")
        for shard, w in zip(self._shards, self._weights):
            print(f"    {shard.name}: {w:.3f}")

    # ------------------------------------------------------------------
    # Core batch generation
    # ------------------------------------------------------------------
    def _get_batch(self) -> torch.Tensor:
        """
        Returns a (batch_size, block_size) LongTensor of input token ids.
        The target is simply batch[:, 1:], i.e. shift by one.
        """
        block_size = self.config.block_size
        batch_size = self.config.batch_size

        inputs  = torch.zeros(batch_size, block_size, dtype=torch.long)

        for i in range(batch_size):
            # Weighted random dataset selection
            shard = self.rng.choices(self._shards, weights=self._weights, k=1)[0]
            chunk = shard.sample(block_size, self.rng)
            # chunk has block_size+1 tokens; inputs = first block_size
            inputs[i] = torch.from_numpy(chunk[:block_size].astype(np.int64))

        return inputs

    # ------------------------------------------------------------------
    # Iteration interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Infinite iterator — call next() or use in a for loop with a manual break."""
        while True:
            yield self._get_batch()

    def get_batch(self) -> torch.Tensor:
        """Convenience method: fetch a single batch."""
        return self._get_batch()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(s.n_tokens for s in self._shards)

    def __repr__(self) -> str:
        lines = [f"MultiDatasetLoader(split={self.split})"]
        for shard, w in zip(self._shards, self._weights):
            lines.append(f"  {shard.name}: {shard.n_tokens:,} tokens  weight={w:.3f}")
        return "\n".join(lines)

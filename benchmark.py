"""BoolQ dataset loading and seeded sampling.

Loads the BoolQ benchmark from HuggingFace, samples a training pool and
test set with deterministic seeding, and derives a category from the
passage title when available.

NOTE: This module is kept for backward compatibility.  New code should use
tasks.py (TaskExample / Task) instead of BoolQExample.
"""

import random
from dataclasses import dataclass
from typing import List


@dataclass
class BoolQExample:
    """A single BoolQ instance with all relevant fields."""
    question_id: str
    question: str
    passage: str
    answer: bool          # True → "yes", False → "no"
    category: str         # derived from title or "general"


def _derive_category(example: dict) -> str:
    """Derive a category string from a BoolQ example.

    Uses the 'title' field if present and non-empty, otherwise falls back
    to 'general'.

    Returns:
        Lowercase category string.
    """
    title = example.get("title", "")
    if title and isinstance(title, str) and title.strip():
        return title.strip().lower()
    return "general"


def _to_boolq_example(idx: int, example: dict) -> BoolQExample:
    """Convert a raw HuggingFace dict to a BoolQExample.

    Args:
        idx: Positional index in the split (used as stable question_id).
        example: Raw row from the HuggingFace dataset.

    Returns:
        A BoolQExample dataclass instance.
    """
    return BoolQExample(
        question_id=str(idx),
        question=example["question"],
        passage=example["passage"],
        answer=bool(example["answer"]),
        category=_derive_category(example),
    )


def load_boolq(n_train: int, n_test: int, seed: int) -> tuple:
    """Load BoolQ and sample train pool + test set.

    Args:
        n_train: Number of examples to sample from the train split as the
                 demonstration pool.
        n_test: Number of examples to sample from the validation split as
                test instances.
        seed: Random seed for reproducible sampling.

    Returns:
        (train_pool, test_set) – two lists of BoolQExample.

    Raises:
        ValueError: If requested sizes exceed available data.
    """
    from datasets import load_dataset

    ds = load_dataset("google/boolq")

    train_split = ds["train"]
    val_split = ds["validation"]

    if n_train > len(train_split):
        raise ValueError(
            f"n_train={n_train} exceeds train split size ({len(train_split)})")
    if n_test > len(val_split):
        raise ValueError(
            f"n_test={n_test} exceeds validation split size ({len(val_split)})")

    rng = random.Random(seed)

    train_indices = rng.sample(range(len(train_split)), n_train)
    test_indices = rng.sample(range(len(val_split)), n_test)

    train_pool: List[BoolQExample] = [
        _to_boolq_example(i, train_split[i]) for i in train_indices
    ]
    test_set: List[BoolQExample] = [
        _to_boolq_example(i, val_split[i]) for i in test_indices
    ]

    return train_pool, test_set

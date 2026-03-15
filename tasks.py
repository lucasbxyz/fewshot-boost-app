"""Generic task abstraction for few-shot evaluation.

Provides a unified Task/TaskExample interface that decouples the evaluation
engine from any specific benchmark.  Ships with loaders for three pre-built
benchmarks (BoolQ, SST-2, AG News) and a builder for custom user tasks.
"""

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskExample:
    id: str
    input_text: str
    context: str
    correct: str
    category: str = "general"


@dataclass
class Task:
    name: str
    instruction: str
    choices: List[str]
    train_pool: List[TaskExample] = field(default_factory=list)
    test_set: List[TaskExample] = field(default_factory=list)


# ── Pre-built benchmark loaders ──────────────────────────────────────────────

BENCHMARK_REGISTRY: dict[str, dict] = {
    "boolq": {
        "label": "BoolQ (Yes/No Comprehension)",
        "description": "Answer yes/no questions based on a Wikipedia passage.",
        "choices": ["yes", "no"],
    },
    "sst2": {
        "label": "SST-2 (Sentiment Analysis)",
        "description": "Classify movie review sentiment as positive or negative.",
        "choices": ["positive", "negative"],
    },
    "agnews": {
        "label": "AG News (Topic Classification)",
        "description": "Classify a news headline into one of four topics.",
        "choices": ["World", "Sports", "Business", "Sci/Tech"],
    },
}


def load_boolq_task(n_train: int = 50, n_test: int = 100, seed: int = 42) -> Task:
    from datasets import load_dataset

    ds = load_dataset("google/boolq")
    rng = random.Random(seed)

    train_idx = rng.sample(range(len(ds["train"])), min(n_train, len(ds["train"])))
    test_idx = rng.sample(
        range(len(ds["validation"])), min(n_test, len(ds["validation"]))
    )

    train_pool = [
        TaskExample(
            id=str(i),
            input_text=ds["train"][i]["question"],
            context=ds["train"][i]["passage"],
            correct="yes" if ds["train"][i]["answer"] else "no",
        )
        for i in train_idx
    ]
    test_set = [
        TaskExample(
            id=str(i),
            input_text=ds["validation"][i]["question"],
            context=ds["validation"][i]["passage"],
            correct="yes" if ds["validation"][i]["answer"] else "no",
        )
        for i in test_idx
    ]

    return Task(
        name="BoolQ",
        instruction="Answer the following yes/no question based on the given passage.",
        choices=["yes", "no"],
        train_pool=train_pool,
        test_set=test_set,
    )


def load_sst2_task(n_train: int = 50, n_test: int = 100, seed: int = 42) -> Task:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2")
    rng = random.Random(seed)
    label_map = {0: "negative", 1: "positive"}

    train_idx = rng.sample(range(len(ds["train"])), min(n_train, len(ds["train"])))
    test_idx = rng.sample(
        range(len(ds["validation"])), min(n_test, len(ds["validation"]))
    )

    train_pool = [
        TaskExample(
            id=str(i),
            input_text=ds["train"][i]["sentence"],
            context="",
            correct=label_map[ds["train"][i]["label"]],
        )
        for i in train_idx
    ]
    test_set = [
        TaskExample(
            id=str(i),
            input_text=ds["validation"][i]["sentence"],
            context="",
            correct=label_map[ds["validation"][i]["label"]],
        )
        for i in test_idx
    ]

    return Task(
        name="SST-2",
        instruction="Classify the sentiment of the following movie review.",
        choices=["positive", "negative"],
        train_pool=train_pool,
        test_set=test_set,
    )


def load_agnews_task(n_train: int = 50, n_test: int = 100, seed: int = 42) -> Task:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    rng = random.Random(seed)
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    train_idx = rng.sample(range(len(ds["train"])), min(n_train, len(ds["train"])))
    test_idx = rng.sample(range(len(ds["test"])), min(n_test, len(ds["test"])))

    train_pool = [
        TaskExample(
            id=str(i),
            input_text=ds["train"][i]["text"],
            context="",
            correct=label_map[ds["train"][i]["label"]],
        )
        for i in train_idx
    ]
    test_set = [
        TaskExample(
            id=str(i),
            input_text=ds["test"][i]["text"],
            context="",
            correct=label_map[ds["test"][i]["label"]],
        )
        for i in test_idx
    ]

    return Task(
        name="AG News",
        instruction="Classify the following news headline into one of the topics.",
        choices=["World", "Sports", "Business", "Sci/Tech"],
        train_pool=train_pool,
        test_set=test_set,
    )


_LOADERS = {
    "boolq": load_boolq_task,
    "sst2": load_sst2_task,
    "agnews": load_agnews_task,
}


def load_task(name: str, n_train: int = 50, n_test: int = 100, seed: int = 42) -> Task:
    if name not in _LOADERS:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from: {list(_LOADERS)}")
    return _LOADERS[name](n_train, n_test, seed)


# ── Custom task builder ──────────────────────────────────────────────────────


def build_custom_task(
    name: str,
    instruction: str,
    choices: List[str],
    examples_text: str,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> Task:
    """Parse user-provided examples and split into train/test.

    Each line in *examples_text* should be:  input_text ||| label
    Lines that are blank or don't contain the delimiter are skipped.
    """
    examples: List[TaskExample] = []
    choice_set = {c.strip().lower() for c in choices}

    for idx, line in enumerate(examples_text.strip().splitlines()):
        if "|||" not in line:
            continue
        parts = line.split("|||", 1)
        text = parts[0].strip()
        label = parts[1].strip()
        if not text or label.lower() not in choice_set:
            continue
        matched = next(c for c in choices if c.lower() == label.lower())
        examples.append(
            TaskExample(id=str(idx), input_text=text, context="", correct=matched)
        )

    if len(examples) < 4:
        raise ValueError(
            f"Need at least 4 valid examples, got {len(examples)}. "
            "Format: input_text ||| label (one per line)."
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    split = max(2, int(len(examples) * train_ratio))
    return Task(
        name=name,
        instruction=instruction,
        choices=choices,
        train_pool=examples[:split],
        test_set=examples[split:],
    )

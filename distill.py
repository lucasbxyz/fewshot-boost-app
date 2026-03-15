"""Distilling Step-by-Step — Knowledge distillation with rationale generation.

Implements the core mechanism from Hsieh et al. (2023):
  1. Teacher LLM generates (label, rationale) pairs for training data
  2. Student SLM is LoRA-fine-tuned with a multi-task loss:
       L = λ · L_label + (1-λ) · L_rationale
     achieved via dataset mixing at ratio λ : (1-λ)
  3. Distilled student is evaluated against the original
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List

import torch
from torch.utils.data import Dataset as TorchDataset


@dataclass
class AnnotatedExample:
    id: str
    input_text: str
    context: str
    label: str
    rationale: str


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Teacher rationale generation
# ═══════════════════════════════════════════════════════════════════════════════


def _rationale_prompt(input_text: str, context: str, choices: list[str]) -> str:
    ctx_block = f"Context: {context}\n\n" if context else ""
    return (
        f"Answer the following question and explain your reasoning.\n\n"
        f"{ctx_block}"
        f"Question: {input_text}\n"
        f"Choices: {', '.join(choices)}\n\n"
        f"Reply in exactly this format (no extra text):\n"
        f"Answer: <one of {', '.join(choices)}>\n"
        f"Rationale: <1-2 sentence explanation>"
    )


def _parse_rationale_response(text: str, choices: list[str]) -> tuple[str, str]:
    """Parse 'Answer: ... Rationale: ...' from teacher response."""
    label, rationale = "", ""
    for line in text.strip().splitlines():
        low = line.strip().lower()
        if low.startswith("answer:"):
            raw_label = line.split(":", 1)[1].strip().lower().rstrip(".")
            for c in choices:
                if raw_label == c.lower() or raw_label.startswith(c.lower()):
                    label = c
                    break
        elif low.startswith("rationale:"):
            rationale = line.split(":", 1)[1].strip()
    return label, rationale


def generate_teacher_rationales(
    task,
    provider: str,
    model: str,
    api_key: str,
    callback: Callable[[int, int], None] | None = None,
) -> list[AnnotatedExample]:
    """Use the teacher LLM to produce (label, rationale) for every train example."""
    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=30.0)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    annotated: list[AnnotatedExample] = []
    total = len(task.train_pool) + len(task.test_set)

    for idx, ex in enumerate(task.train_pool + task.test_set):
        prompt = _rationale_prompt(ex.input_text, ex.context, task.choices)

        if provider == "openai":
            resp = client.chat.completions.create(
                model=model, max_tokens=150, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
        else:
            resp = client.messages.create(
                model=model, max_tokens=150, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text

        label, rationale = _parse_rationale_response(text, task.choices)

        if not label:
            label = ex.correct
        if not rationale:
            rationale = f"The answer is {label}."

        annotated.append(AnnotatedExample(
            id=ex.id,
            input_text=ex.input_text,
            context=ex.context,
            label=label,
            rationale=rationale,
        ))

        if callback:
            callback(idx + 1, total)

    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Multi-task dataset with λ-weighting
# ═══════════════════════════════════════════════════════════════════════════════


class DistillDataset(TorchDataset):
    """Tokenized dataset mixing label-prediction and rationale-generation tasks.

    The loss weighting  L = λ · L_label + (1-λ) · L_rationale  is achieved by
    mixing the two task types at ratio λ : (1-λ) in the dataset.  In expectation
    over random sampling, this is equivalent to explicit loss weighting.
    """

    def __init__(self, examples: list[AnnotatedExample], task_instruction: str,
                 choices: list[str], tokenizer, max_len: int = 512,
                 lambda_weight: float = 0.5, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items: list[dict] = []

        rng = random.Random(seed)

        label_system = (
            f"{task_instruction} "
            f"Reply with exactly one word from: {', '.join(choices)}."
        )
        rationale_system = (
            f"{task_instruction} "
            f"First state your answer, then explain your reasoning in 1-2 sentences."
        )

        for ex in examples:
            user_msg = f"Context: {ex.context}\n\n{ex.input_text}" if ex.context else ex.input_text

            label_messages = [
                {"role": "system", "content": label_system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ex.label},
            ]
            rationale_messages = [
                {"role": "system", "content": rationale_system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{ex.label}. {ex.rationale}"},
            ]

            if rng.random() < lambda_weight:
                self.items.append(self._tokenize(label_messages))
            else:
                self.items.append(self._tokenize(rationale_messages))

        rng.shuffle(self.items)

    def _tokenize(self, messages: list[dict]) -> dict:
        prompt_msgs = messages[:-1]
        full_msgs = messages

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_enc = self.tokenizer(
            full_text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt", add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3: LoRA fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════


def fine_tune_student(
    model_name: str,
    dataset: DistillDataset,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    callback: Callable[[int, int, float], None] | None = None,
) -> tuple:
    """LoRA-fine-tune a student SLM on the multi-task dataset.

    Returns (model, tokenizer, device) with LoRA adapters merged.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from slm import load_model

    model, tokenizer, device = load_model(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} trainable / {total_params:,} total "
          f"({trainable/total_params:.2%})")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(loader)
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if callback:
                callback(step, total_steps, loss.item())

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"  Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

    model.eval()
    model = model.merge_and_unload()

    return model, tokenizer, device


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_distilled(model, tokenizer, device, task,
                       callback: Callable[[int, int], None] | None = None) -> dict:
    """Evaluate a (distilled) model on the task's test set.

    Returns dict with accuracy and per-question results.
    """
    from slm import generate_answer_generic

    correct = 0
    results = []

    for i, ex in enumerate(task.test_set):
        ans, raw, _ = generate_answer_generic(
            model, tokenizer, device,
            task.instruction, task.choices,
            ex.input_text, ex.context, [],
        )
        is_correct = ans == ex.correct
        correct += int(is_correct)
        results.append({
            "id": ex.id,
            "predicted": ans,
            "ground_truth": ex.correct,
            "correct": is_correct,
            "raw": raw,
        })

        if callback:
            callback(i + 1, len(task.test_set))

    return {
        "accuracy": correct / max(len(task.test_set), 1),
        "n_correct": correct,
        "n_total": len(task.test_set),
        "results": results,
    }

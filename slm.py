"""Small Language Model loading and inference.

Loads HuggingFace instruction-tuned models, applies the chat template,
and runs inference with CUDA / MPS acceleration (CPU fallback).
"""

from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Select the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str) -> tuple:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier
            (e.g. 'HuggingFaceTB/SmolLM2-360M-Instruct').

    Returns:
        (model, tokenizer, device) tuple ready for inference.
    """
    device = _get_device()
    # float16 works on CUDA; MPS and CPU need float32
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    log.info("Loading %s on %s (dtype=%s)...", model_name, device, dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}'. Check the model ID and your "
            f"internet connection. Error: {e}"
        ) from e

    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


# ── Generic task support ─────────────────────────────────────────────────────


def build_prompt_generic(instruction: str, choices: List[str],
                         test_input: str, test_context: str,
                         shots: list) -> List[dict]:
    """Build a chat-format prompt for any classification task.

    *shots* is a list of objects with .input_text, .context, and .correct.
    """
    system_msg = (
        f"{instruction} "
        f"Reply with exactly one word from: {', '.join(choices)}."
    )
    messages: List[dict] = [{"role": "system", "content": system_msg}]

    for ex in shots:
        content = f"Context: {ex.context}\n\n{ex.input_text}" if ex.context else ex.input_text
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": ex.correct})

    content = f"Context: {test_context}\n\n{test_input}" if test_context else test_input
    messages.append({"role": "user", "content": content})
    return messages


def normalize_answer_generic(raw: str, choices: List[str]) -> str:
    """Normalize SLM output to one of *choices*, or 'INVALID'."""
    text = raw.strip().lower()
    if not text:
        return "INVALID"
    # Exact match first (handles multi-word choices like "Sci/Tech")
    for choice in choices:
        if text == choice.lower() or text.startswith(choice.lower()):
            return choice
    # Fuzzy: first token match
    first_token = text.split()[0].strip(".,!?;:'\"")
    for choice in choices:
        if first_token == choice.lower():
            return choice
    # Substring search (e.g. "the answer is positive")
    for choice in choices:
        if choice.lower() in text:
            return choice
    return "INVALID"


def generate_answer_generic(model, tokenizer, device,
                            instruction: str, choices: List[str],
                            test_input: str, test_context: str,
                            shots: list) -> tuple:
    """Generate an answer for a generic classification task.

    Returns (normalized_answer, raw_answer, input_token_count).
    """
    messages = build_prompt_generic(
        instruction, choices, test_input, test_context, shots,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return normalize_answer_generic(raw_answer, choices), raw_answer, input_len

"""Small Language Model loading and inference.

Loads HuggingFace instruction-tuned models, applies the chat template,
and runs inference with MPS acceleration on Apple Silicon (CPU fallback).
"""

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import BoolQExample


def _get_device() -> torch.device:
    """Select the best available device.

    Returns:
        torch.device – MPS on Apple Silicon, else CPU.
    """
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
    dtype = torch.float32  # MPS does not support float16 reliably

    print(f"  Loading {model_name} on {device} (dtype={dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    )
    model = model.to(device)
    model.eval()

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def build_prompt(test_example: BoolQExample,
                 shots: List[BoolQExample]) -> List[dict]:
    """Build a chat-format prompt with few-shot demonstrations.

    Constructs a list of message dicts suitable for apply_chat_template.

    Args:
        test_example: The question to answer.
        shots: Few-shot demonstration examples.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    system_msg = (
        "You are a helpful assistant that answers yes/no questions based on "
        "a given passage. Reply with exactly one word: yes or no."
    )

    messages = [{"role": "system", "content": system_msg}]

    for ex in shots:
        label = "yes" if ex.answer else "no"
        messages.append({
            "role": "user",
            "content": f"Passage: {ex.passage}\n\nQuestion: {ex.question}",
        })
        messages.append({
            "role": "assistant",
            "content": label,
        })

    messages.append({
        "role": "user",
        "content": (
            f"Passage: {test_example.passage}\n\n"
            f"Question: {test_example.question}"
        ),
    })

    return messages


def normalize_answer(raw: str) -> str:
    """Normalize SLM output to 'yes', 'no', or 'INVALID'.

    Strips whitespace, lowercases, takes the first token, and maps it.
    Unmappable outputs return 'INVALID'.

    Args:
        raw: Raw text output from the SLM.

    Returns:
        'yes', 'no', or 'INVALID'.
    """
    text = raw.strip().lower()
    if not text:
        return "INVALID"

    first_token = text.split()[0].strip(".,!?;:'\"")

    if first_token in ("yes", "true"):
        return "yes"
    if first_token in ("no", "false"):
        return "no"
    return "INVALID"


def generate_answer(model, tokenizer, device,
                    test_example: BoolQExample,
                    shots: List[BoolQExample]) -> tuple:
    """Generate a yes/no answer from the SLM.

    Applies the chat template, runs greedy generation, and normalizes the
    output.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Corresponding tokenizer.
        device: Torch device the model lives on.
        test_example: The question to answer.
        shots: Few-shot demonstration examples.

    Returns:
        Tuple of (normalized_answer, raw_answer) where normalized_answer is
        'yes', 'no', or 'INVALID' and raw_answer is the decoded model output.
    """
    messages = build_prompt(test_example, shots)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return normalize_answer(raw_answer), raw_answer


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
    first_token = text.split()[0].strip(".,!?;:'\"")
    for choice in choices:
        if first_token == choice.lower() or first_token.startswith(choice.lower()[:4]):
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
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return normalize_answer_generic(raw_answer, choices), raw_answer, input_len

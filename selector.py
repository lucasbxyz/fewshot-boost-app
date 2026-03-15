"""Few-shot example selection strategies.

Provides:
  - RandomSelector: uniformly samples k examples from the training pool.
  - GenericLLMSelector: task-agnostic LLM selector with token tracking,
    retry logic, and optional disk caching.  Supports OpenAI and Anthropic.

Legacy provider-specific classes (AnthropicSelector, OpenAISelector, etc.)
are kept for backward compatibility but new code should use
GenericLLMSelector directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Random selector
# ═══════════════════════════════════════════════════════════════════════════════


class RandomSelector:
    """Selects few-shot examples uniformly at random."""

    def select(self, train_pool, test_example, k: int,
               seed: int) -> list:
        rng = random.Random(seed)
        k = min(k, len(train_pool))
        return rng.sample(train_pool, k) if k > 0 else []


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic task-agnostic selector with token tracking + caching
# ═══════════════════════════════════════════════════════════════════════════════


class GenericLLMSelector:
    """Task-agnostic LLM selector that tracks cumulative token usage.

    Features:
      - Supports OpenAI and Anthropic providers
      - Tracks input/output tokens across all calls
      - Optional disk-based response caching (cache_dir)
      - Exponential backoff with rate-limit awareness
      - Falls back to random selection after MAX_RETRIES (with warning)
    """

    MAX_RETRIES = 5
    BASE_DELAY = 2.0

    def __init__(self, provider: str, model: str, api_key: str,
                 task_instruction: str = "", task_choices: List[str] | None = None,
                 cache_dir: str | Path | None = None):
        self.provider = provider
        self.model = model
        self.task_instruction = task_instruction
        self.task_choices = task_choices or []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Optional disk cache
        self._cache_dir: Path | None = None
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        if provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key, timeout=15.0)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, timeout=15.0)
        else:
            raise ValueError(f"GenericLLMSelector supports 'openai' and 'anthropic', got '{provider}'")

    # ── Caching ──────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(prompt: str, model: str) -> str:
        h = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()[:16]
        return h

    def _cache_get(self, prompt: str) -> str | None:
        if not self._cache_dir:
            return None
        key = self._cache_key(prompt, self.model)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("text")
        return None

    def _cache_put(self, prompt: str, text: str, in_tok: int, out_tok: int):
        if not self._cache_dir:
            return
        key = self._cache_key(prompt, self.model)
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "model": self.model, "text": text,
            "input_tokens": in_tok, "output_tokens": out_tok,
        }))

    # ── Prompt building ──────────────────────────────────────────────────

    def _build_prompt(self, train_pool, test_example, k: int) -> str:
        pool_lines = []
        for i, ex in enumerate(train_pool):
            snippet = ex.input_text[:80]
            ctx = f" | Ctx: {ex.context[:60]}..." if ex.context else ""
            pool_lines.append(f"[{i}] {snippet}{ctx} | A: {ex.correct}")

        choices_str = ", ".join(self.task_choices) if self.task_choices else "N/A"
        test_ctx = f"\nCONTEXT: {test_example.context[:200]}" if test_example.context else ""

        return (
            f"You are selecting few-shot examples for a classification task.\n"
            f"TASK: {self.task_instruction}\n"
            f"CLASSES: {choices_str}\n\n"
            f"TEST INPUT: {test_example.input_text[:200]}{test_ctx}\n\n"
            f"CANDIDATES:\n" + "\n".join(pool_lines) + "\n\n"
            f"Select the {k} most relevant candidates.\n"
            f"Reply with ONLY a JSON array of {k} integers. Example: [3, 17, 42]"
        )

    # ── LLM call ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
        # Check cache first
        cached = self._cache_get(prompt)
        if cached is not None:
            log.debug("Cache hit for selector prompt")
            return cached, 0, 0

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model, max_tokens=100, temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
        else:
            resp = self._client.messages.create(
                model=self.model, max_tokens=100, temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            in_tok = resp.usage.input_tokens if resp.usage else 0
            out_tok = resp.usage.output_tokens if resp.usage else 0

        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        # Persist to cache
        self._cache_put(prompt, text, in_tok, out_tok)

        return text, in_tok, out_tok

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_indices(text: str, pool_size: int, k: int) -> List[int]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        indices = json.loads(text)
        if not isinstance(indices, list) or len(indices) != k:
            raise ValueError(f"Expected list of {k} ints, got: {text[:80]}")
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= pool_size:
                raise ValueError(f"Index {idx} out of range [0, {pool_size})")
        return indices

    # ── Main entry point ─────────────────────────────────────────────────

    def select(self, train_pool, test_example, k: int, seed: int = 42):
        prompt = self._build_prompt(train_pool, test_example, k)
        text = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                text, _, _ = self._call_llm(prompt)
                indices = self._parse_indices(text, len(train_pool), k)
                return [train_pool[i] for i in indices]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    delay = 5.0 + attempt * 5.0
                else:
                    delay = self.BASE_DELAY * (2 ** attempt)
                raw_preview = repr(text[:120]) if text else "N/A"
                log.warning(
                    "LLM selector attempt %d/%d failed: %s | raw: %s. Retrying in %.1fs...",
                    attempt + 1, self.MAX_RETRIES, e, raw_preview, delay,
                )
                time.sleep(delay)

        log.warning("LLM selector: all %d retries exhausted — falling back to random selection.", self.MAX_RETRIES)
        rng = random.Random(seed)
        k = min(k, len(train_pool))
        return rng.sample(train_pool, k) if k > 0 else []


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory + backward-compatible aliases
# ═══════════════════════════════════════════════════════════════════════════════


def create_llm_selector(provider: str, model: str, api_key: str,
                        **kwargs) -> GenericLLMSelector:
    """Instantiate a GenericLLMSelector for the given provider.

    This replaces the old per-provider classes.  All keyword arguments are
    forwarded to GenericLLMSelector.
    """
    if provider not in ("openai", "anthropic"):
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: openai, anthropic")
    return GenericLLMSelector(provider=provider, model=model, api_key=api_key, **kwargs)

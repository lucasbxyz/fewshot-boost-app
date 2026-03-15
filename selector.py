"""Few-shot example selection strategies.

Provides two selector families:
  - RandomSelector: uniformly samples k examples from the training pool.
  - LLM-backed selectors: use a language model to pick the k most useful
    examples for each test instance. Supported providers:
      * AnthropicSelector  (provider: "anthropic")
      * OpenAISelector     (provider: "openai")
      * GoogleSelector     (provider: "google")

Use create_llm_selector() to instantiate the right backend from config.
"""

import json
import random
import time
from typing import List

from benchmark import BoolQExample


class RandomSelector:
    """Selects few-shot examples uniformly at random."""

    def select(self, train_pool: List[BoolQExample],
               test_example: BoolQExample, k: int,
               seed: int) -> List[BoolQExample]:
        """Sample k random examples from the training pool.

        Args:
            train_pool: All available demonstration examples.
            test_example: The test instance (unused by this strategy).
            k: Number of examples to select.
            seed: Random seed for this specific selection.

        Returns:
            List of k BoolQExample instances.
        """
        rng = random.Random(seed)
        return rng.sample(train_pool, k)


class _BaseLLMSelector:
    """Shared prompt-building, parsing, and retry logic for LLM selectors."""

    MAX_RETRIES = 5
    BASE_DELAY = 2.0

    def _build_prompt(self, train_pool: List[BoolQExample],
                      test_example: BoolQExample, k: int) -> str:
        pool_text = ""
        for i, ex in enumerate(train_pool):
            label = "yes" if ex.answer else "no"
            pool_text += (
                f"[{i}] Q: {ex.question} | Topic: {ex.passage[:80]}... | A: {label}\n"
            )

        return (
            f"You are selecting few-shot examples for a yes/no question-answering task.\n\n"
            f"TEST QUESTION: {test_example.question}\n"
            f"CONTEXT: {test_example.passage[:200]}\n\n"
            f"CANDIDATES (index | question | topic | answer):\n{pool_text}\n"
            f"Select the {k} most topically similar candidates.\n"
            f"Reply with ONLY a JSON array of {k} integers. Example: [3, 17, 42]"
        )

    def _parse_indices(self, text: str, pool_size: int, k: int) -> List[int]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        if not text:
            raise ValueError(f"LLM returned an empty response")

        indices = json.loads(text)
        if not isinstance(indices, list) or len(indices) != k:
            raise ValueError(
                f"Expected JSON list of {k} indices, got: {text[:80]}")
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= pool_size:
                raise ValueError(
                    f"Index {idx} out of range [0, {pool_size})")
        return indices

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return raw text. Implemented by subclasses."""
        raise NotImplementedError

    def select(self, train_pool: List[BoolQExample],
               test_example: BoolQExample, k: int,
               seed: int) -> List[BoolQExample]:
        """Select k examples using the LLM, with retries and random fallback."""
        prompt = self._build_prompt(train_pool, test_example, k)
        text = ""

        for attempt in range(self.MAX_RETRIES):
            try:
                text = self._call_llm(prompt)
                indices = self._parse_indices(text, len(train_pool), k)
                return [train_pool[i] for i in indices]
            except Exception as e:
                # For rate-limit errors (429) use a longer fixed pause;
                # for other errors use exponential backoff.
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    delay = 5.0 + attempt * 5.0   # 5s, 10s, 15s, 20s, 25s
                else:
                    delay = self.BASE_DELAY * (2 ** attempt)
                raw_preview = repr(text[:120]) if text else "N/A"
                print(f"  [LLM selector] Attempt {attempt + 1}/{self.MAX_RETRIES} "
                      f"failed: {e} | raw: {raw_preview}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

        print("  [LLM selector] All retries exhausted. Falling back to random.")
        rng = random.Random(seed)
        return rng.sample(train_pool, k)


class AnthropicSelector(_BaseLLMSelector):
    """Uses an Anthropic Claude model to select few-shot examples."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=15.0)
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAISelector(_BaseLLMSelector):
    """Uses an OpenAI model to select few-shot examples."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key, timeout=15.0)
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class QwenSelector(_BaseLLMSelector):
    """Uses an Alibaba Qwen model via DashScope's OpenAI-compatible endpoint."""

    # International endpoint; swap for dashscope.aliyuncs.com inside mainland China
    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str, model: str = "qwen-turbo") -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=15.0,
        )
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class GoogleSelector(_BaseLLMSelector):
    """Uses a Google Gemini model to select few-shot examples."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        model = self._genai.GenerativeModel(
            self.model,
            generation_config=self._genai.GenerationConfig(
                max_output_tokens=100,
                temperature=0.7,
            ),
        )
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15},
        )
        return response.text


# Backward-compatible alias
LLMAssistedSelector = AnthropicSelector

_PROVIDER_MAP = {
    "anthropic": AnthropicSelector,
    "openai": OpenAISelector,
    "qwen": QwenSelector,
    "google": GoogleSelector,
}


def create_llm_selector(provider: str, model: str,
                        api_key: str) -> _BaseLLMSelector:
    """Instantiate the appropriate LLM selector for the given provider.

    Args:
        provider: One of 'anthropic', 'openai', 'google'.
        model: Model identifier string (provider-specific).
        api_key: API key for the chosen provider.

    Returns:
        A configured selector instance.

    Raises:
        ValueError: If provider is not recognised.
    """
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(_PROVIDER_MAP.keys())}")
    return _PROVIDER_MAP[provider](api_key=api_key, model=model)


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic task-agnostic selector with token tracking
# ═══════════════════════════════════════════════════════════════════════════════


class GenericLLMSelector:
    """Task-agnostic LLM selector that tracks cumulative token usage."""

    MAX_RETRIES = 5
    BASE_DELAY = 2.0

    def __init__(self, provider: str, model: str, api_key: str,
                 task_instruction: str = "", task_choices: List[str] | None = None):
        self.provider = provider
        self.model = model
        self.task_instruction = task_instruction
        self.task_choices = task_choices or []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key, timeout=15.0)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, timeout=15.0)
        else:
            raise ValueError(f"GenericLLMSelector supports 'openai' and 'anthropic', got '{provider}'")

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

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """Returns (text, input_tokens, output_tokens)."""
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
        return text, in_tok, out_tok

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
                time.sleep(delay)

        rng = random.Random(seed)
        return rng.sample(train_pool, k)

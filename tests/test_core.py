"""Tests for core David vs Goliath modules.

Run with:  python -m pytest tests/ -v
"""

import json
import random
import tempfile
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
#  tasks.py
# ═══════════════════════════════════════════════════════════════════════════════


from tasks import TaskExample, Task, build_custom_task, BENCHMARK_REGISTRY


class TestTaskExample:
    def test_creation(self):
        ex = TaskExample(id="1", input_text="hello", context="", correct="yes")
        assert ex.id == "1"
        assert ex.category == "general"

    def test_custom_category(self):
        ex = TaskExample(id="2", input_text="q", context="c", correct="no", category="science")
        assert ex.category == "science"


class TestBuildCustomTask:
    EXAMPLES = (
        "I love this movie ||| positive\n"
        "Terrible film ||| negative\n"
        "Great acting ||| positive\n"
        "Worst ever ||| negative\n"
        "Amazing story ||| positive\n"
    )

    def test_basic_build(self):
        task = build_custom_task("Test", "Classify sentiment", ["positive", "negative"],
                                 self.EXAMPLES, train_ratio=0.6, seed=42)
        assert task.name == "Test"
        assert len(task.train_pool) + len(task.test_set) == 5
        assert len(task.train_pool) >= 2
        assert len(task.test_set) >= 1

    def test_too_few_examples(self):
        with pytest.raises(ValueError, match="at least 4"):
            build_custom_task("T", "I", ["a", "b"], "one ||| a\ntwo ||| b")

    def test_invalid_labels_skipped(self):
        examples = (
            "good ||| positive\n"
            "bad ||| negative\n"
            "meh ||| neutral\n"  # not in choices
            "ok ||| positive\n"
            "no ||| negative\n"
        )
        task = build_custom_task("T", "I", ["positive", "negative"], examples)
        total = len(task.train_pool) + len(task.test_set)
        assert total == 4  # neutral skipped

    def test_case_insensitive_labels(self):
        examples = (
            "a ||| Positive\n"
            "b ||| NEGATIVE\n"
            "c ||| positive\n"
            "d ||| Negative\n"
        )
        task = build_custom_task("T", "I", ["positive", "negative"], examples)
        for ex in task.train_pool + task.test_set:
            assert ex.correct in ["positive", "negative"]


class TestBenchmarkRegistry:
    def test_registry_has_expected_keys(self):
        assert "boolq" in BENCHMARK_REGISTRY
        assert "sst2" in BENCHMARK_REGISTRY
        assert "agnews" in BENCHMARK_REGISTRY

    def test_registry_entries_have_required_fields(self):
        for key, entry in BENCHMARK_REGISTRY.items():
            assert "label" in entry
            assert "description" in entry
            assert "choices" in entry
            assert len(entry["choices"]) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
#  pricing.py
# ═══════════════════════════════════════════════════════════════════════════════


from pricing import (
    get_model_price, CostEstimate, CostProjection, project_monthly,
    quick_cost_table, LLM_PRICING,
)


class TestGetModelPrice:
    def test_known_model(self):
        price = get_model_price("gpt-4o-mini")
        assert price["input"] == 0.15
        assert price["output"] == 0.60

    def test_unknown_model_fallback(self):
        price = get_model_price("nonexistent-model")
        assert price["input"] == 0.15  # falls back to gpt-4o-mini pricing


class TestCostEstimate:
    def test_averages(self):
        est = CostEstimate(
            llm_input_tokens=1000, llm_output_tokens=200,
            slm_input_tokens=500, n_questions=10,
        )
        assert est.avg_llm_input_per_q == 100.0
        assert est.avg_llm_output_per_q == 20.0
        assert est.avg_slm_input_per_q == 50.0

    def test_zero_questions(self):
        est = CostEstimate(llm_input_tokens=100, n_questions=0)
        assert est.avg_llm_input_per_q == 100.0  # divides by max(0,1)=1


class TestProjectMonthly:
    def test_basic_projection(self):
        proj = project_monthly(100_000, direct_model="gpt-4o-mini")
        assert proj.monthly_volume == 100_000
        assert proj.pure_llm_cost > 0
        assert proj.hybrid_total >= 0

    def test_savings_are_positive_for_cheap_selector(self):
        proj = project_monthly(
            100_000,
            direct_model="gpt-4o",
            direct_input_tpq=500.0,
            direct_output_tpq=50.0,
            selector_model="gpt-4o-mini",
        )
        assert proj.savings_abs > 0
        assert proj.savings_pct > 0

    def test_measured_overrides(self):
        measured = CostEstimate(
            llm_input_tokens=5000, llm_output_tokens=500,
            n_questions=10,
            accuracy_llm_assisted=0.75,
        )
        proj = project_monthly(100_000, measured=measured)
        assert proj.selector_input_tpq == 500.0  # 5000/10
        assert proj.hybrid_accuracy == 0.75

    def test_break_even(self):
        proj = project_monthly(
            100_000,
            direct_model="gpt-4o",
            selector_model="gpt-4o-mini",
            slm_hosting_key="cloud_cpu",
        )
        assert proj.break_even_volume > 0


class TestQuickCostTable:
    def test_returns_rows(self):
        rows = quick_cost_table("gpt-4o", "gpt-4o-mini", 100_000)
        assert len(rows) > 0
        for row in rows:
            assert "hosting" in row
            assert "pure_llm_cost" in row
            assert "savings_abs" in row


# ═══════════════════════════════════════════════════════════════════════════════
#  selector.py
# ═══════════════════════════════════════════════════════════════════════════════


from selector import RandomSelector, GenericLLMSelector


class TestRandomSelector:
    def test_basic_selection(self):
        pool = [TaskExample(id=str(i), input_text=f"q{i}", context="", correct="yes")
                for i in range(10)]
        test_ex = TaskExample(id="test", input_text="test", context="", correct="no")
        sel = RandomSelector()
        result = sel.select(pool, test_ex, k=3, seed=42)
        assert len(result) == 3
        assert all(r in pool for r in result)

    def test_k_larger_than_pool(self):
        pool = [TaskExample(id="0", input_text="q", context="", correct="yes")]
        test_ex = TaskExample(id="t", input_text="t", context="", correct="no")
        sel = RandomSelector()
        result = sel.select(pool, test_ex, k=5, seed=42)
        assert len(result) == 1  # clamped to pool size

    def test_empty_pool(self):
        sel = RandomSelector()
        result = sel.select([], TaskExample(id="t", input_text="t", context="", correct="no"), k=3, seed=42)
        assert result == []

    def test_deterministic(self):
        pool = [TaskExample(id=str(i), input_text=f"q{i}", context="", correct="yes")
                for i in range(20)]
        test_ex = TaskExample(id="t", input_text="t", context="", correct="no")
        sel = RandomSelector()
        r1 = sel.select(pool, test_ex, k=3, seed=123)
        r2 = sel.select(pool, test_ex, k=3, seed=123)
        assert [r.id for r in r1] == [r.id for r in r2]


class TestGenericLLMSelectorParsing:
    def test_parse_valid_json(self):
        indices = GenericLLMSelector._parse_indices("[0, 3, 7]", pool_size=10, k=3)
        assert indices == [0, 3, 7]

    def test_parse_with_code_block(self):
        text = "```json\n[1, 2, 3]\n```"
        indices = GenericLLMSelector._parse_indices(text, pool_size=10, k=3)
        assert indices == [1, 2, 3]

    def test_parse_wrong_count(self):
        with pytest.raises(ValueError, match="Expected list"):
            GenericLLMSelector._parse_indices("[1, 2]", pool_size=10, k=3)

    def test_parse_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            GenericLLMSelector._parse_indices("[0, 1, 15]", pool_size=10, k=3)

    def test_parse_negative_index(self):
        with pytest.raises(ValueError, match="out of range"):
            GenericLLMSelector._parse_indices("[-1, 0, 1]", pool_size=10, k=3)


class TestGenericLLMSelectorCache:
    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = GenericLLMSelector.__new__(GenericLLMSelector)
            sel._cache_dir = Path(tmpdir)
            sel.model = "test-model"

            # Put
            sel._cache_put("test prompt", "[1,2,3]", 100, 10)

            # Get
            result = sel._cache_get("test prompt")
            assert result == "[1,2,3]"

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sel = GenericLLMSelector.__new__(GenericLLMSelector)
            sel._cache_dir = Path(tmpdir)
            sel.model = "test-model"
            assert sel._cache_get("nonexistent") is None

    def test_no_cache_dir(self):
        sel = GenericLLMSelector.__new__(GenericLLMSelector)
        sel._cache_dir = None
        sel.model = "test-model"
        assert sel._cache_get("anything") is None
        sel._cache_put("prompt", "text", 0, 0)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
#  slm.py (parsing only — model loading requires torch+transformers)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeAnswerGeneric:
    """Test answer normalization without loading models."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from slm import normalize_answer_generic
            self.normalize = normalize_answer_generic
        except ImportError:
            pytest.skip("torch not installed")

    def test_exact_match(self):
        assert self.normalize("yes", ["yes", "no"]) == "yes"
        assert self.normalize("no", ["yes", "no"]) == "no"

    def test_case_insensitive(self):
        assert self.normalize("YES", ["yes", "no"]) == "yes"
        assert self.normalize("Positive", ["positive", "negative"]) == "positive"

    def test_prefix_match(self):
        assert self.normalize("positive.", ["positive", "negative"]) == "positive"

    def test_substring_match(self):
        assert self.normalize("the answer is positive", ["positive", "negative"]) == "positive"

    def test_empty_string(self):
        assert self.normalize("", ["yes", "no"]) == "INVALID"

    def test_gibberish(self):
        assert self.normalize("asdfgh", ["yes", "no"]) == "INVALID"

    def test_multiword_choice(self):
        assert self.normalize("Sci/Tech", ["World", "Sports", "Business", "Sci/Tech"]) == "Sci/Tech"


# ═══════════════════════════════════════════════════════════════════════════════
#  distill.py (parsing only)
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseRationaleResponse:
    @pytest.fixture(autouse=True)
    def _import(self):
        from distill import _parse_rationale_response
        self.parse = _parse_rationale_response

    def test_standard_format(self):
        text = "Answer: yes\nRationale: Because the passage says so."
        label, rationale = self.parse(text, ["yes", "no"])
        assert label == "yes"
        assert "passage" in rationale

    def test_case_insensitive_label(self):
        text = "Answer: Positive\nRationale: It's good."
        label, rationale = self.parse(text, ["positive", "negative"])
        assert label == "positive"

    def test_missing_rationale(self):
        text = "Answer: no"
        label, rationale = self.parse(text, ["yes", "no"])
        assert label == "no"
        assert rationale == ""

    def test_missing_answer(self):
        text = "Rationale: Because reasons."
        label, rationale = self.parse(text, ["yes", "no"])
        assert label == ""
        assert "reasons" in rationale

    def test_label_with_period(self):
        text = "Answer: yes.\nRationale: It is."
        label, rationale = self.parse(text, ["yes", "no"])
        assert label == "yes"

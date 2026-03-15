"""LLM pricing tables and cost estimation for FewShot Boost.

Compares the cost of a pure-LLM approach (every question sent to a large model)
against the hybrid approach (LLM selects examples, SLM answers locally).
"""

from dataclasses import dataclass

# Prices per 1 million tokens (USD), as of early 2025.
LLM_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}

SLM_HOSTING: dict[str, dict] = {
    "self_hosted_cpu": {"monthly": 0, "label": "Own hardware (CPU) — free"},
    "cloud_cpu": {"monthly": 30, "label": "Cloud CPU instance (~$30/mo)"},
    "hf_inference_cpu": {"monthly": 45, "label": "HF Endpoints CPU ($0.06/hr)"},
    "hf_inference_gpu": {"monthly": 450, "label": "HF Endpoints GPU ($0.60/hr)"},
}

AVG_TOKENS_PER_DIRECT_LLM_CALL = 350
AVG_OUTPUT_TOKENS_PER_DIRECT_LLM_CALL = 5


@dataclass
class CostEstimate:
    """Measured stats from a benchmark run + projected costs."""

    llm_input_tokens: int
    llm_output_tokens: int
    slm_input_tokens: int
    n_questions: int

    accuracy_zero_shot: float
    accuracy_random: float
    accuracy_llm_assisted: float

    def _per_question_llm_tokens(self) -> tuple[float, float]:
        if self.n_questions == 0:
            return (0.0, 0.0)
        return (
            self.llm_input_tokens / self.n_questions,
            self.llm_output_tokens / self.n_questions,
        )

    def project_monthly(
        self,
        monthly_volume: int,
        llm_model: str,
        slm_hosting_key: str = "self_hosted_cpu",
        llm_direct_model: str | None = None,
    ) -> dict:
        """Project monthly costs for pure-LLM vs hybrid approach.

        Returns dict with keys:
            pure_llm_cost, hybrid_llm_cost, hybrid_hosting_cost, hybrid_total,
            savings_abs, savings_pct, accuracy_llm_assumed, accuracy_hybrid,
            break_even_volume
        """
        direct_model = llm_direct_model or llm_model
        direct_price = LLM_PRICING.get(direct_model, {"input": 2.50, "output": 10.00})
        selector_price = LLM_PRICING.get(llm_model, {"input": 0.15, "output": 0.60})
        hosting = SLM_HOSTING.get(slm_hosting_key, SLM_HOSTING["self_hosted_cpu"])

        pure_llm_cost = monthly_volume * (
            AVG_TOKENS_PER_DIRECT_LLM_CALL * direct_price["input"] / 1e6
            + AVG_OUTPUT_TOKENS_PER_DIRECT_LLM_CALL * direct_price["output"] / 1e6
        )

        pq_in, pq_out = self._per_question_llm_tokens()
        hybrid_llm_cost = monthly_volume * (
            pq_in * selector_price["input"] / 1e6
            + pq_out * selector_price["output"] / 1e6
        )
        hybrid_hosting = hosting["monthly"]
        hybrid_total = hybrid_llm_cost + hybrid_hosting

        savings_abs = pure_llm_cost - hybrid_total
        savings_pct = (savings_abs / pure_llm_cost * 100) if pure_llm_cost > 0 else 0

        if hybrid_llm_cost > 0 and hybrid_hosting > 0:
            cost_per_call = hybrid_llm_cost / monthly_volume
            break_even = int(hybrid_hosting / (
                AVG_TOKENS_PER_DIRECT_LLM_CALL * direct_price["input"] / 1e6
                + AVG_OUTPUT_TOKENS_PER_DIRECT_LLM_CALL * direct_price["output"] / 1e6
                - cost_per_call
            )) if (AVG_TOKENS_PER_DIRECT_LLM_CALL * direct_price["input"] / 1e6
                   + AVG_OUTPUT_TOKENS_PER_DIRECT_LLM_CALL * direct_price["output"] / 1e6
                   - cost_per_call) > 0 else 0
        else:
            break_even = 0

        return {
            "pure_llm_cost": round(pure_llm_cost, 2),
            "hybrid_llm_cost": round(hybrid_llm_cost, 2),
            "hybrid_hosting_cost": hybrid_hosting,
            "hybrid_total": round(hybrid_total, 2),
            "savings_abs": round(savings_abs, 2),
            "savings_pct": round(savings_pct, 1),
            "accuracy_llm_assumed": 0.90,
            "accuracy_hybrid": self.accuracy_llm_assisted,
            "break_even_volume": break_even,
        }


def quick_cost_table(llm_model: str, monthly_volume: int) -> list[dict]:
    """Generate a comparison table across all hosting tiers for a given LLM."""
    dummy = CostEstimate(
        llm_input_tokens=monthly_volume * 200,
        llm_output_tokens=monthly_volume * 20,
        slm_input_tokens=monthly_volume * 300,
        n_questions=monthly_volume,
        accuracy_zero_shot=0.50,
        accuracy_random=0.52,
        accuracy_llm_assisted=0.58,
    )
    rows = []
    for key, info in SLM_HOSTING.items():
        proj = dummy.project_monthly(monthly_volume, llm_model, key)
        rows.append({"hosting": info["label"], **proj})
    return rows

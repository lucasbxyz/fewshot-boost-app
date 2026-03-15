"""LLM pricing tables and cost estimation for David vs Goliath.

Compares the cost of a pure-LLM approach (every question sent to a large model)
against the hybrid approach (LLM selects examples, SLM answers locally).

All costs are token-based.  Users can override every assumption.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Pricing tables — loaded from config.yaml if available, else defaults
# ═══════════════════════════════════════════════════════════════════════════════

_PRICING_UPDATED = "2025-03"

# Prices per 1 million tokens (USD).
_DEFAULT_LLM_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}

_DEFAULT_SLM_HOSTING: dict[str, dict[str, Any]] = {
    "self_hosted_cpu": {"monthly": 0, "label": "Own hardware (CPU) — free"},
    "cloud_cpu": {"monthly": 30, "label": "Cloud CPU instance (~$30/mo)"},
    "hf_inference_cpu": {"monthly": 45, "label": "HF Endpoints CPU ($0.06/hr)"},
    "hf_inference_gpu": {"monthly": 450, "label": "HF Endpoints GPU ($0.60/hr)"},
    "custom": {"monthly": 0, "label": "Custom (enter your own)"},
}


def _load_config(config_path: Path | None = None) -> dict:
    """Attempt to load config.yaml from project root."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning("Failed to load config.yaml: %s", e)
    return {}


def _check_pricing_staleness():
    """Warn if pricing data is more than 6 months old."""
    try:
        updated = date.fromisoformat(f"{_PRICING_UPDATED}-01")
        age_days = (date.today() - updated).days
        if age_days > 180:
            warnings.warn(
                f"LLM pricing data was last updated {_PRICING_UPDATED} "
                f"({age_days} days ago). Prices may be outdated. "
                f"Update the pricing tables in pricing.py or config.yaml.",
                UserWarning,
                stacklevel=3,
            )
    except Exception:
        pass


# Initialize on import
_config = _load_config()
_check_pricing_staleness()

LLM_PRICING: dict[str, dict[str, float]] = _config.get("llm_pricing", _DEFAULT_LLM_PRICING)
SLM_HOSTING: dict[str, dict[str, Any]] = _config.get("slm_hosting", _DEFAULT_SLM_HOSTING)


def get_model_price(model: str) -> dict[str, float]:
    """Look up per-1M-token prices; falls back to gpt-4o-mini tier."""
    return LLM_PRICING.get(model, {"input": 0.15, "output": 0.60})


# ═══════════════════════════════════════════════════════════════════════════════
#  Measured run data
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CostEstimate:
    """Measured token counts + accuracy from a benchmark run."""

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    slm_input_tokens: int = 0
    n_questions: int = 0

    accuracy_zero_shot: float = 0.0
    accuracy_random: float = 0.0
    accuracy_llm_assisted: float = 0.0

    @property
    def avg_llm_input_per_q(self) -> float:
        return self.llm_input_tokens / max(self.n_questions, 1)

    @property
    def avg_llm_output_per_q(self) -> float:
        return self.llm_output_tokens / max(self.n_questions, 1)

    @property
    def avg_slm_input_per_q(self) -> float:
        return self.slm_input_tokens / max(self.n_questions, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Token-based cost projection
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CostProjection:
    """Result of a monthly cost projection, fully transparent."""

    # Inputs used
    monthly_volume: int = 0
    direct_model: str = ""
    selector_model: str = ""
    slm_hosting_label: str = ""

    # Token assumptions (per question)
    direct_input_tpq: float = 0.0
    direct_output_tpq: float = 0.0
    selector_input_tpq: float = 0.0
    selector_output_tpq: float = 0.0

    # Costs (raw)
    pure_llm_cost: float = 0.0
    hybrid_llm_cost: float = 0.0
    hybrid_hosting_cost: float = 0.0
    hybrid_total: float = 0.0
    savings_abs: float = 0.0
    savings_pct: float = 0.0

    # Accuracy
    direct_accuracy: float = 0.0
    hybrid_accuracy: float = 0.0

    # Accuracy-adjusted costs (cost per correct answer * volume)
    direct_cost_per_correct: float = 0.0
    hybrid_cost_per_correct: float = 0.0
    adjusted_savings_abs: float = 0.0
    adjusted_savings_pct: float = 0.0

    # Error handling: cost of routing misses back to the LLM
    error_routing_cost: float = 0.0
    hybrid_total_with_errors: float = 0.0
    savings_with_errors_abs: float = 0.0
    savings_with_errors_pct: float = 0.0

    # Break-even
    break_even_volume: int = 0


def project_monthly(
    monthly_volume: int,
    *,
    # Pure LLM baseline (what you'd spend without David vs Goliath)
    direct_model: str = "gpt-4o-mini",
    direct_input_tpq: float = 350.0,
    direct_output_tpq: float = 20.0,
    direct_accuracy: float = 0.90,
    # Hybrid: LLM selector cost (measured or estimated)
    selector_model: str = "gpt-4o-mini",
    selector_input_tpq: float | None = None,
    selector_output_tpq: float | None = None,
    # Hybrid: SLM hosting
    slm_hosting_key: str = "self_hosted_cpu",
    custom_hosting_cost: float = 0.0,
    # Hybrid: accuracy
    hybrid_accuracy: float = 0.58,
    # Pre-measured data (overrides selector token estimates)
    measured: CostEstimate | None = None,
) -> CostProjection:
    """Project monthly costs, fully token-based and user-configurable.

    Every assumption can be overridden.  If *measured* is provided, the
    selector token counts per question are taken from actual benchmark data.
    """
    direct_price = get_model_price(direct_model)
    selector_price = get_model_price(selector_model)

    # Resolve selector tokens per question
    if measured and measured.n_questions > 0:
        sel_in = measured.avg_llm_input_per_q
        sel_out = measured.avg_llm_output_per_q
        hybrid_accuracy = measured.accuracy_llm_assisted
    else:
        sel_in = selector_input_tpq if selector_input_tpq is not None else 800.0
        sel_out = selector_output_tpq if selector_output_tpq is not None else 60.0

    # Resolve hosting
    if slm_hosting_key == "custom":
        hosting_monthly = custom_hosting_cost
        hosting_label = f"Custom (${custom_hosting_cost:.0f}/mo)"
    else:
        hosting_info = SLM_HOSTING.get(slm_hosting_key, SLM_HOSTING["self_hosted_cpu"])
        hosting_monthly = hosting_info["monthly"]
        hosting_label = hosting_info["label"]

    # Pure LLM: every question goes to the big model
    pure_llm_cost = monthly_volume * (
        direct_input_tpq * direct_price["input"] / 1e6
        + direct_output_tpq * direct_price["output"] / 1e6
    )

    # Hybrid: selector LLM picks examples, SLM answers locally
    hybrid_llm_cost = monthly_volume * (
        sel_in * selector_price["input"] / 1e6
        + sel_out * selector_price["output"] / 1e6
    )
    hybrid_total = hybrid_llm_cost + hosting_monthly

    savings_abs = pure_llm_cost - hybrid_total
    savings_pct = (savings_abs / pure_llm_cost * 100) if pure_llm_cost > 0 else 0.0

    # Break-even: how many calls/month before hybrid is cheaper than pure LLM
    direct_cost_per_q = (
        direct_input_tpq * direct_price["input"] / 1e6
        + direct_output_tpq * direct_price["output"] / 1e6
    )
    hybrid_cost_per_q = (
        sel_in * selector_price["input"] / 1e6
        + sel_out * selector_price["output"] / 1e6
    )
    marginal_saving = direct_cost_per_q - hybrid_cost_per_q
    break_even = int(hosting_monthly / marginal_saving) if marginal_saving > 0 else 0

    # ── Accuracy-adjusted economics ──────────────────────────────────────
    # Cost per correct answer: what you actually pay for each *right* answer
    d_cpc = (pure_llm_cost / (monthly_volume * direct_accuracy)) if direct_accuracy > 0 else 0.0
    h_cpc = (hybrid_total / (monthly_volume * hybrid_accuracy)) if hybrid_accuracy > 0 else 0.0

    # Adjusted savings: comparing cost to get the same number of correct answers
    # = monthly_volume * direct_accuracy correct answers needed
    correct_needed = monthly_volume * direct_accuracy
    direct_cost_for_correct = correct_needed * d_cpc
    hybrid_cost_for_correct = correct_needed * h_cpc
    adj_savings = direct_cost_for_correct - hybrid_cost_for_correct
    adj_savings_pct = (adj_savings / direct_cost_for_correct * 100) if direct_cost_for_correct > 0 else 0.0

    # Error routing: assume hybrid misses get escalated back to the direct LLM
    error_rate = max(0.0, 1.0 - hybrid_accuracy)
    error_volume = monthly_volume * error_rate
    error_cost = error_volume * direct_cost_per_q
    hybrid_with_errors = hybrid_total + error_cost
    savings_we_abs = pure_llm_cost - hybrid_with_errors
    savings_we_pct = (savings_we_abs / pure_llm_cost * 100) if pure_llm_cost > 0 else 0.0

    return CostProjection(
        monthly_volume=monthly_volume,
        direct_model=direct_model,
        selector_model=selector_model,
        slm_hosting_label=hosting_label,
        direct_input_tpq=direct_input_tpq,
        direct_output_tpq=direct_output_tpq,
        selector_input_tpq=sel_in,
        selector_output_tpq=sel_out,
        pure_llm_cost=round(pure_llm_cost, 2),
        hybrid_llm_cost=round(hybrid_llm_cost, 2),
        hybrid_hosting_cost=hosting_monthly,
        hybrid_total=round(hybrid_total, 2),
        savings_abs=round(savings_abs, 2),
        savings_pct=round(savings_pct, 1),
        direct_accuracy=direct_accuracy,
        hybrid_accuracy=hybrid_accuracy,
        direct_cost_per_correct=round(d_cpc * 1000, 4),   # per 1K correct
        hybrid_cost_per_correct=round(h_cpc * 1000, 4),   # per 1K correct
        adjusted_savings_abs=round(adj_savings, 2),
        adjusted_savings_pct=round(adj_savings_pct, 1),
        error_routing_cost=round(error_cost, 2),
        hybrid_total_with_errors=round(hybrid_with_errors, 2),
        savings_with_errors_abs=round(savings_we_abs, 2),
        savings_with_errors_pct=round(savings_we_pct, 1),
        break_even_volume=break_even,
    )


def quick_cost_table(
    direct_model: str,
    selector_model: str,
    monthly_volume: int,
    *,
    direct_input_tpq: float = 350.0,
    direct_output_tpq: float = 20.0,
    direct_accuracy: float = 0.90,
    measured: CostEstimate | None = None,
) -> list[dict]:
    """Generate a comparison table across all hosting tiers."""
    rows = []
    for key, info in SLM_HOSTING.items():
        if key == "custom":
            continue
        proj = project_monthly(
            monthly_volume,
            direct_model=direct_model,
            direct_input_tpq=direct_input_tpq,
            direct_output_tpq=direct_output_tpq,
            direct_accuracy=direct_accuracy,
            selector_model=selector_model,
            slm_hosting_key=key,
            measured=measured,
        )
        rows.append({
            "hosting": info["label"],
            "pure_llm_cost": proj.pure_llm_cost,
            "hybrid_total": proj.hybrid_total,
            "savings_abs": proj.savings_abs,
            "savings_pct": proj.savings_pct,
            "break_even": proj.break_even_volume,
        })
    return rows

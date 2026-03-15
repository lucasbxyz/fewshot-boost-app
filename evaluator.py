"""Evaluation metrics: accuracy, standard deviation, confidence intervals.

Computes per-run accuracy and aggregates across runs with t-distribution
confidence intervals.
"""

from dataclasses import dataclass
from typing import List

import scipy.stats as st


@dataclass
class AggregateResult:
    """Aggregated statistics for one (SLM × strategy) combination."""
    slm_model: str
    strategy: str
    mean_accuracy: float
    std: float
    ci_lower: float
    ci_upper: float
    n_runs: int


def compute_accuracy(predictions: List[dict]) -> float:
    """Compute exact-match accuracy for a single run.

    Args:
        predictions: List of dicts, each with 'correct' (bool) key.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if not predictions:
        return 0.0
    n_correct = sum(1 for p in predictions if p["correct"])
    return n_correct / len(predictions)


def aggregate_runs(run_accuracies: List[float], slm_model: str,
                   strategy: str) -> AggregateResult:
    """Aggregate accuracies across multiple runs.

    Computes mean, standard deviation, and 95% CI using the t-distribution.

    Args:
        run_accuracies: List of accuracy values, one per run.
        slm_model: Name of the SLM model.
        strategy: Selection strategy name.

    Returns:
        AggregateResult with all computed statistics.
    """
    n = len(run_accuracies)
    mean = sum(run_accuracies) / n
    if n < 2:
        return AggregateResult(
            slm_model=slm_model,
            strategy=strategy,
            mean_accuracy=mean,
            std=0.0,
            ci_lower=mean,
            ci_upper=mean,
            n_runs=n,
        )

    variance = sum((x - mean) ** 2 for x in run_accuracies) / (n - 1)
    std = variance ** 0.5

    # 95% CI using t-distribution
    se = std / (n ** 0.5)
    t_crit = st.t.ppf(0.975, df=n - 1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return AggregateResult(
        slm_model=slm_model,
        strategy=strategy,
        mean_accuracy=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_runs=n,
    )

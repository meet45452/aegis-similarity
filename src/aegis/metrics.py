"""Virtual-screening metrics for honest benchmarking.

AEGIS deliberately reports early-enrichment and decision-quality metrics in
addition to global rank quality, because mean AUROC rewards broad
retrospective ranking and may not reflect DMTA value:

* ``bedroc``: Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007),
  normalised to [0, 1] with perfect = 1 and reversed = 0.
* ``enrichment_factor``: EF at a fixed top fraction.
* ``auroc``: global rank quality baseline.
* ``scaffold_hop_recovery``: actives with a *different* Murcko scaffold
  recovered in the top-k - the metric that matters for novel scaffold yield.
* ``topk_scaffold_diversity``: unique generalised frameworks in the top-k.
* ``calibration_by_tier``: does a claimed confidence tier actually succeed
  more often than a lower tier?
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_arrays(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    scores_array = np.asarray(scores, dtype=float)
    labels_array = (np.asarray(labels) == 1).astype(int)
    if scores_array.shape != labels_array.shape:
        raise ValueError("scores and labels must have the same length")
    return scores_array, labels_array


def bedroc(scores: Sequence[float], labels: Sequence[int], alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly, 2007), normalised to [0, 1].

    Implements BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min) where RIE is the
    exponential rank-weighted enrichment of actives and the extrema are the
    perfect and reversed rankings.  Returns NaN when all labels are equal.
    """
    scores_array, labels_array = _as_arrays(scores, labels)
    n = len(labels_array)
    n_actives = int(labels_array.sum())
    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")
    order = np.argsort(-scores_array, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    active_ranks = ranks[labels_array == 1]
    numerator = float(np.sum(np.exp(-alpha * active_ranks / n)))
    numerator_max = float(np.sum(np.exp(-alpha * np.arange(1, n_actives + 1) / n)))
    numerator_min = float(
        np.sum(np.exp(-alpha * np.arange(n - n_actives + 1, n + 1) / n))
    )
    if numerator_max - numerator_min <= 0.0:
        return float("nan")
    return (numerator - numerator_min) / (numerator_max - numerator_min)


def bedroc_random_baseline(n: int, n_actives: int, alpha: float = 20.0) -> float:
    """Analytic expected BEDROC for a random ranking (for context)."""
    active_fraction = n_actives / n
    return float(
        (active_fraction * (1 - np.exp(-alpha))) / (1 - np.exp(-alpha * active_fraction))
    )


def enrichment_factor(
    scores: Sequence[float],
    labels: Sequence[int],
    fraction: float = 0.01,
) -> float:
    """Enrichment factor EF at a fixed top fraction of the ranked list."""
    scores_array, labels_array = _as_arrays(scores, labels)
    n = len(labels_array)
    n_actives = int(labels_array.sum())
    if n == 0 or n_actives == 0:
        return float("nan")
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-scores_array, kind="stable")
    top = labels_array[order][:k]
    return float((top.sum() / k) / (n_actives / n))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve via the rank statistic."""
    scores_array, labels_array = _as_arrays(scores, labels)
    n_actives = int(labels_array.sum())
    n_inactives = len(labels_array) - n_actives
    if n_actives == 0 or n_inactives == 0:
        return float("nan")
    order = np.argsort(scores_array, kind="stable")
    ranks = np.empty(len(labels_array), dtype=float)
    ranks[order] = np.arange(1, len(labels_array) + 1)
    active_rank_sum = float(ranks[labels_array == 1].sum())
    return (active_rank_sum - n_actives * (n_actives + 1) / 2.0) / (n_actives * n_inactives)


def scaffold_hop_recovery(
    scores: Sequence[float],
    labels: Sequence[int],
    scaffold_changed: Sequence[bool],
    top_k: int = 50,
) -> float:
    """Fraction of active scaffold hops recovered in the top-k."""
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels) == 1
    changed = np.asarray(scaffold_changed, dtype=bool)
    order = np.argsort(-scores_array, kind="stable")
    top = order[:top_k]
    hits = int((labels_array[top] & changed[top]).sum())
    n_active_hops = int((labels_array & changed).sum())
    if n_active_hops == 0:
        return float("nan")
    return hits / n_active_hops


def topk_scaffold_diversity(scaffolds: Sequence[str], top_k: int = 50) -> float:
    """Unique generalised frameworks / min(k, list length) in the top-k."""
    subset = list(scaffolds)[:top_k]
    if not subset:
        return 0.0
    return len(set(subset)) / len(subset)


def calibration_by_tier(labels_by_tier: dict[str, list[int]]) -> dict[str, float]:
    """Hit rate per confidence tier - the honesty check for claimed confidence."""
    return {
        tier: float(np.mean(values)) if values else float("nan")
        for tier, values in labels_by_tier.items()
    }

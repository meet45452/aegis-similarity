"""Tests for virtual-screening metrics (validated against analytic baselines)."""

from __future__ import annotations

import numpy as np
import pytest

from aegis.metrics import (
    auroc,
    bedroc,
    bedroc_random_baseline,
    calibration_by_tier,
    enrichment_factor,
    scaffold_hop_recovery,
    topk_scaffold_diversity,
)


def _perfect(n: int = 40, n_actives: int = 6):
    scores = np.zeros(n)
    scores[:n_actives] = 1.0
    labels = np.zeros(n, dtype=int)
    labels[:n_actives] = 1
    return scores, labels


def test_bedroc_extremes():
    scores, labels = _perfect()
    assert bedroc(scores, labels, alpha=20.0) == pytest.approx(1.0)
    assert bedroc(-scores, labels, alpha=20.0) == pytest.approx(0.0)


def test_bedroc_random_matches_analytic_baseline():
    rng = np.random.default_rng(42)
    n, n_actives = 400, 5
    labels = np.zeros(n, dtype=int)
    labels[:n_actives] = 1
    observed = bedroc(rng.random(n), labels, alpha=20.0)
    expected = bedroc_random_baseline(n, n_actives, alpha=20.0)
    assert abs(observed - expected) < 0.05


def test_bedroc_early_enrichment_beats_late():
    labels = np.zeros(100, dtype=int)
    labels[[0, 2, 5, 10, 50, 90]] = 1
    early_scores = -np.arange(100, dtype=float)
    late_scores = np.arange(100, dtype=float)
    assert bedroc(early_scores, labels) > bedroc(late_scores, labels)


def test_bedroc_degenerate():
    assert np.isnan(bedroc([0.1, 0.2], [0, 0]))
    assert np.isnan(bedroc([0.1, 0.2], [1, 1]))


def test_enrichment_factor():
    scores = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    # Perfect early placement: EF equals 1 / active fraction.
    assert enrichment_factor(scores, labels, fraction=0.3) == pytest.approx(1 / 0.3)


def test_auroc_extremes():
    scores, labels = _perfect()
    assert auroc(scores, labels) == pytest.approx(1.0)
    assert auroc(-scores, labels) == pytest.approx(0.0)


def test_scaffold_hop_recovery():
    scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    labels = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    changed = [True, False, True, True, False, True, True, False, True, False]
    # Active scaffold hops sit at ranks 1, 4, 7; the top-5 contains two of them.
    assert scaffold_hop_recovery(scores, labels, changed, top_k=5) == pytest.approx(2 / 3)


def test_topk_scaffold_diversity():
    scaffolds = ["A", "A", "B", "C", "C", "D"]
    assert topk_scaffold_diversity(scaffolds, top_k=4) == pytest.approx(3 / 4)
    assert topk_scaffold_diversity([], top_k=4) == 0.0


def test_calibration_by_tier():
    result = calibration_by_tier({"high": [1, 1, 0], "low": [0, 0]})
    assert result["high"] == pytest.approx(2 / 3)
    assert result["low"] == 0.0

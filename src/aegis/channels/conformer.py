"""Conformer channel: Boltzmann-weighted ensemble similarity.

Implements the AEGIS ensemble principle: two ligands can be functional
analogues even when their dominant solution-state conformers differ, provided
each can access a matching geometry at acceptable energetic cost.

S_ensemble(A, B) aggregates the per-conformer-pair USR similarity matrix with
    logits_ij = (S_ij - alpha * (E_Ai + E_Bj)) / temperature
using a softmax weight over pairs.  As temperature -> 0 this approaches the
soft-maximum over pairs of the design document; the finite temperature keeps
the score smooth and differentiable in practice.
"""

from __future__ import annotations

import numpy as np

from aegis.channels.shape import usr_similarity
from aegis.types import MoleculeRecord


def ensemble_aggregate(
    pair_scores: np.ndarray,
    energy_sums: np.ndarray,
    energy_alpha: float,
    temperature: float,
) -> tuple[float, float]:
    """Return (score, spread) for one pair of conformer ensembles."""
    logits = (pair_scores - energy_alpha * energy_sums) / max(temperature, 1e-6)
    weights = np.exp(logits - logits.max())
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full(pair_scores.shape, 1.0 / pair_scores.size)
    else:
        weights = weights / total
    score = float((weights * pair_scores).sum())
    variance = float((weights * (pair_scores - score) ** 2).sum())
    return score, variance**0.5


class ConformerChannel:
    """Energy-aware conformer-ensemble similarity with a spread diagnostic."""

    name = "conformer"

    def __init__(self, energy_alpha: float = 0.05, temperature: float = 0.15):
        self.energy_alpha = energy_alpha
        self.temperature = temperature

    def _pair_matrices(self, query: MoleculeRecord, cand: MoleculeRecord):
        query_ensemble, cand_ensemble = query.ensemble, cand.ensemble
        if (
            query_ensemble is None
            or cand_ensemble is None
            or not query_ensemble.usr
            or not cand_ensemble.usr
            or query_ensemble.energies is None
            or cand_ensemble.energies is None
        ):
            return None, None
        pair_scores = np.array(
            [
                [usr_similarity(query_moments, cand_moments) for cand_moments in cand_ensemble.usr]
                for query_moments in query_ensemble.usr
            ]
        )
        energy_sums = np.array(
            [
                [query_energy + cand_energy for cand_energy in cand_ensemble.energies]
                for query_energy in query_ensemble.energies
            ]
        )
        return pair_scores, energy_sums

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        pair_scores, energy_sums = self._pair_matrices(query, cand)
        if pair_scores is None:
            return None
        score, _ = ensemble_aggregate(pair_scores, energy_sums, self.energy_alpha, self.temperature)
        return score

    def spread(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        """Conformational ambiguity: weighted std of the pair-score distribution."""
        pair_scores, energy_sums = self._pair_matrices(query, cand)
        if pair_scores is None:
            return None
        _, spread = ensemble_aggregate(pair_scores, energy_sums, self.energy_alpha, self.temperature)
        return spread

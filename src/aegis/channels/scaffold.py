"""Scaffold channel: Bemis-Murcko hierarchy agreement (Bemis & Murcko, 1996)."""

from __future__ import annotations

from rdkit import DataStructs

from aegis.types import MoleculeRecord


class ScaffoldChannel:
    """Multi-level scaffold similarity.

    Compares the exact Murcko framework (weight 0.65) and the
    atom-type-generalised framework skeleton (weight 0.35).  The generic
    level captures "same topology, different heteroatom pattern"
    relationships that matter for scaffold hopping.

    Returns ``None`` when both molecules are acyclic (channel not
    applicable) and 0.0 when exactly one molecule lacks a scaffold.
    """

    name = "scaffold"
    exact_weight = 0.65
    generic_weight = 0.35

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        query_acyclic = not query.scaffold and not query.generic_scaffold
        cand_acyclic = not cand.scaffold and not cand.generic_scaffold
        if query_acyclic and cand_acyclic:
            return None
        if query_acyclic or cand_acyclic:
            return 0.0
        levels = (
            (
                query.scaffold,
                cand.scaffold,
                query.scaffold_fingerprint,
                cand.scaffold_fingerprint,
                self.exact_weight,
            ),
            (
                query.generic_scaffold,
                cand.generic_scaffold,
                query.generic_fingerprint,
                cand.generic_fingerprint,
                self.generic_weight,
            ),
        )
        numerator = 0.0
        denominator = 0.0
        for query_scaffold, cand_scaffold, query_fp, cand_fp, weight in levels:
            if not query_scaffold or not cand_scaffold:
                continue
            if query_fp is None or cand_fp is None:
                continue
            numerator += weight * float(DataStructs.TanimotoSimilarity(query_fp, cand_fp))
            denominator += weight
        if denominator <= 0.0:
            return None
        return numerator / denominator

"""Shape channel: USR ensemble similarity (Ballester & Richards, 2007)."""

from __future__ import annotations

import numpy as np

from aegis.types import MoleculeRecord


def usr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """USR similarity 1 / (1 + d) with size-normalised Euclidean distance."""
    distance = float(np.linalg.norm(a - b) / np.sqrt(len(a)))
    return 1.0 / (1.0 + distance)


class ShapeChannel:
    """3D shape overlap via USR moments over the conformer ensembles.

    The score is the best USR similarity over all kept conformer pairs, so a
    molecule that can *access* a matching geometry scores well even when its
    dominant solution conformer differs - a core AEGIS design goal.
    """

    name = "shape"

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        query_usr = query.ensemble.usr if query.ensemble is not None else None
        cand_usr = cand.ensemble.usr if cand.ensemble is not None else None
        if not query_usr or not cand_usr:
            return None
        best = 0.0
        for query_moments in query_usr:
            for cand_moments in cand_usr:
                value = usr_similarity(query_moments, cand_moments)
                if value > best:
                    best = value
        return float(best)

"""Field channel: USRCAT-style interaction-field similarity.

Instead of describing a molecule as an object, this channel describes the
*interaction field it can present*: per-pharmacophore-class moment blocks
plus global partial-charge statistics (Schreyer & Blundell, 2012).  Similarity
means "can this ligand reproduce the other's interaction opportunities",
not merely "does it occupy the same volume".

Approximation note: features are atom-class moment summaries rather than a
dense 3D ESP grid; the charge statistics use Gasteiger charges.  This keeps
the index compact enough for cascade use; see README for the upgrade path.
"""

from __future__ import annotations

import numpy as np

from aegis.chemistry import ATOM_CLASSES
from aegis.types import MoleculeRecord


def _pair_field_similarity(features_a: dict[str, np.ndarray], features_b: dict[str, np.ndarray]) -> float:
    keys_a = {key for key in features_a if not key.startswith("_")}
    keys_b = {key for key in features_b if not key.startswith("_")}
    similarities: list[float] = []
    for key in keys_a & keys_b:
        vector_a = features_a[key]
        vector_b = features_b[key]
        distance = float(np.linalg.norm(vector_a - vector_b) / np.sqrt(len(vector_a)))
        similarities.append(1.0 / (1.0 + distance))
    if "_charge" in features_a and "_charge" in features_b:
        charge_distance = float(np.mean(np.abs(features_a["_charge"] - features_b["_charge"])))
        similarities.append(1.0 / (1.0 + 2.0 * charge_distance))
    if not similarities:
        return 0.0
    base = float(np.mean(similarities))
    mismatch_penalty = 0.05 * len(keys_a ^ keys_b)
    return float(max(0.0, min(1.0, base - mismatch_penalty)))


class FieldChannel:
    """Interaction-field agreement between conformer ensembles."""

    name = "field"

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        query_fields = query.ensemble.fields if query.ensemble is not None else None
        cand_fields = cand.ensemble.fields if cand.ensemble is not None else None
        if not query_fields or not cand_fields:
            return None
        best = 0.0
        for features_a in query_fields:
            for features_b in cand_fields:
                value = _pair_field_similarity(features_a, features_b)
                if value > best:
                    best = value
        return float(best)

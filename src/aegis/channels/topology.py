"""Topology channel: ECFP Tanimoto similarity (Rogers & Hahn, 2010)."""

from __future__ import annotations

from rdkit import DataStructs

from aegis.types import MoleculeRecord


class TopologyChannel:
    """2D circular-fingerprint similarity.

    Extremely fast and excellent for close-analogue retrieval, but blind to
    many scaffold hops and to 3D presentation - which is exactly why AEGIS
    keeps orthogonal channels alongside it.
    """

    name = "topology"

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        if query.fingerprint is None or cand.fingerprint is None:
            return None
        return float(DataStructs.TanimotoSimilarity(query.fingerprint, cand.fingerprint))

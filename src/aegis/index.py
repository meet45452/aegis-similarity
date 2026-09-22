"""Tier-0 candidate routing: ECFP + student embedding + scaffold families.

Tier 0 never applies expensive physics: it generates a high-recall candidate
set as the union of

* ECFP nearest neighbours (close analogues),
* student-embedding nearest neighbours (compact 2D space),
* members of the query's Murcko scaffold families (explicit diversity
  management and scaffold-transformation neighbourhoods).

The exact ordering does not matter much because Tier 1 rescores everything
that survives; recall at a fixed budget is the design target.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import DataStructs

from aegis.types import MoleculeRecord


class Tier0Index:
    """In-memory union index over a prepared library."""

    def __init__(self, records: list[MoleculeRecord]):
        self.records = records
        self.fingerprints = [record.fingerprint for record in records]
        embeddings = [record.student_embedding for record in records]
        if embeddings and all(embedding is not None for embedding in embeddings):
            self.matrix = np.vstack(embeddings)
        else:
            self.matrix = None
        self.exact_scaffold_map: dict[str, list[int]] = defaultdict(list)
        self.generic_scaffold_map: dict[str, list[int]] = defaultdict(list)
        for position, record in enumerate(records):
            if record.scaffold:
                self.exact_scaffold_map[record.scaffold].append(position)
            if record.generic_scaffold:
                self.generic_scaffold_map[record.generic_scaffold].append(position)

    def best_similarity(self, query: MoleculeRecord) -> float | None:
        """Maximum ECFP Tanimoto similarity of the query to the library."""
        if query.fingerprint is None or not self.fingerprints:
            return None
        similarities = DataStructs.BulkTanimotoSimilarity(query.fingerprint, self.fingerprints)
        return float(max(similarities)) if similarities else None

    def search(
        self,
        query: MoleculeRecord,
        n_ecfp: int = 150,
        n_student: int = 150,
        n_scaffold: int = 100,
        limit: int | None = None,
    ) -> list[int]:
        """Return the union of candidate positions, best-rank ordered."""
        if query.fingerprint is None or not self.records:
            return []
        best_rank: dict[int, int] = {}

        similarities = DataStructs.BulkTanimotoSimilarity(query.fingerprint, self.fingerprints)
        ecfp_order = np.argsort(-np.asarray(similarities), kind="stable")
        for rank, position in enumerate(ecfp_order[:n_ecfp]):
            best_rank[int(position)] = rank

        if self.matrix is not None and query.student_embedding is not None:
            cosines = self.matrix @ query.student_embedding
            student_order = np.argsort(-cosines, kind="stable")
            for rank, position in enumerate(student_order[:n_student]):
                position = int(position)
                best_rank[position] = min(best_rank.get(position, 10**9), rank)

        family = list(self.exact_scaffold_map.get(query.scaffold, []))
        family.extend(self.generic_scaffold_map.get(query.generic_scaffold, []))
        for rank, position in enumerate(family[:n_scaffold]):
            best_rank[position] = min(best_rank.get(position, 10**9), rank)

        candidates = sorted(best_rank, key=lambda position: best_rank[position])
        if limit is not None:
            candidates = candidates[:limit]
        return candidates

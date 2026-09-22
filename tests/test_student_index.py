"""Tests for the Tier-0 student embedding and candidate index."""

from __future__ import annotations

import numpy as np
import pytest

from aegis.library import Library
from aegis.student import StudentEmbedder

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "OC(=O)c1ccccc1O",
    "NC(=O)c1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
]


def test_student_embedding_shape_and_normalisation():
    library = Library.build(SMILES, with_conformers=False)
    assert len(library) == 5
    for record in library.records:
        assert record.student_embedding is not None
        assert record.student_embedding.shape == (256,)
        assert abs(float(np.linalg.norm(record.student_embedding)) - 1.0) < 1e-6


def test_index_self_retrieval():
    library = Library.build(SMILES, with_conformers=False)
    query = library.records[0]
    assert library.index.best_similarity(query) == pytest.approx(1.0)
    candidates = library.index.search(query, n_ecfp=3, n_student=3, n_scaffold=3)
    assert 0 in candidates
    assert all(0 <= position < len(library) for position in candidates)


def test_scaffold_family_neighbours_found():
    library = Library.build(SMILES, with_conformers=False)
    aspirin = next(r for r in library.records if "CC(=O)Oc1" in r.smiles)
    candidates = library.index.search(aspirin, n_ecfp=1, n_student=1, n_scaffold=50)
    # All phenyl-containing molecules belong to the aspirin generic scaffold family.
    assert len(candidates) >= 3


def test_embedder_save_load_roundtrip(tmp_path):
    library = Library.build(SMILES, with_conformers=False)
    path = str(tmp_path / "student.npz")
    library.embedder.save(path)
    loaded = StudentEmbedder.load(path)
    mol = library.records[0].mol
    np.testing.assert_allclose(loaded.transform(mol), library.embedder.transform(mol))

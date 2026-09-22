"""Tests for standardisation, scaffolds, ensembles, and descriptors."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from aegis import chemistry


def test_parse_and_strip_salt():
    mol = chemistry.parse_mol("CC(=O)O.Cl")
    smiles = Chem.MolToSmiles(mol)
    assert "Cl" not in smiles
    assert len(Chem.GetMolFrags(mol)) == 1


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        chemistry.parse_mol("not_a_smiles")


def test_murcko_hierarchy():
    aspirin = chemistry.parse_mol("CC(=O)Oc1ccccc1C(=O)O")
    exact, generic = chemistry.murcko_frameworks(aspirin)
    assert exact  # aspirin retains a substituted ring framework
    assert generic
    assert generic != exact  # generalisation strips heteroatoms/bond orders
    acyclic = chemistry.parse_mol("CN(C)C(=N)N=C(N)N")
    assert chemistry.murcko_frameworks(acyclic) == ("", "")


def test_boltzmann_weights():
    weights = chemistry.boltzmann_weights(np.array([0.0, 1.0, 2.0]))
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert weights[0] >= weights[1] >= weights[2]


def test_usr_moments():
    rng = np.random.default_rng(0)
    moments = chemistry.usr_moments(rng.random((20, 3)))
    assert moments.shape == (12,)
    assert np.all(np.isfinite(moments))
    assert chemistry.usr_moments(rng.random((20, 3)) * 5.0 + 100.0).shape == (12,)


def test_classify_atoms():
    mol = chemistry.parse_mol("CC(=O)Nc1ccc(O)cc1")  # paracetamol
    classes = chemistry.classify_atoms(mol)
    assert "donor" in classes  # amide NH / phenol OH
    assert "acceptor" in classes  # carbonyl oxygen
    assert "aromatic" in classes  # ring carbons


def test_conformer_ensemble():
    mol = chemistry.parse_mol("CC(=O)Oc1ccccc1C(=O)O")
    ensemble = chemistry.generate_conformer_ensemble(mol, n_confs=4, keep=2, seed=42)
    assert ensemble is not None
    assert 1 <= len(ensemble) <= 2
    assert abs(float(ensemble.weights.sum()) - 1.0) < 1e-9
    assert len(ensemble.usr) == len(ensemble)
    assert len(ensemble.fields) == len(ensemble)
    assert all(moments.shape == (12,) for moments in ensemble.usr)


def test_synthetic_complexity_ordering():
    benzoic = chemistry.synthetic_complexity(chemistry.parse_mol("OC(=O)c1ccccc1"))
    caffeine = chemistry.synthetic_complexity(chemistry.parse_mol("CN1C=NC2=C1C(=O)N(C)C(=O)N2C"))
    assert caffeine > benzoic

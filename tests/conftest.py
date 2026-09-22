"""Shared fixtures for the AEGIS test suite."""

from __future__ import annotations

import pytest

from aegis.config import AegisConfig

DEMO_MOLECULES: dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "salicylic_acid": "OC(=O)c1ccccc1O",
    "benzoic_acid": "OC(=O)c1ccccc1",
    "benzamide": "NC(=O)c1ccccc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "adenine": "Nc1ncnc2[nH]cnc12",
    "aminoquinazoline": "Nc1ncnc2ccccc12",
    "metformin": "CN(C)C(=N)N=C(N)N",
    "quinoline": "c1ccc2ncccc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
}


@pytest.fixture(scope="session")
def small_config() -> AegisConfig:
    return AegisConfig(
        n_conformers=4,
        keep_conformers=2,
        tier0_candidates=50,
        tier1_candidates=20,
        tier2_candidates=10,
        default_top_k=10,
    )


@pytest.fixture(scope="session")
def library(small_config):
    from aegis.library import Library

    names = list(DEMO_MOLECULES)
    smiles = list(DEMO_MOLECULES.values())
    return Library.build(smiles, names=names, config=small_config, with_conformers=True)

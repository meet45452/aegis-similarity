"""Shared fixtures for the AEGIS test suite."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

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


def pose_mol(smiles: str, n_confs: int = 3, seed: int = 42) -> Chem.Mol:
    """Embed a molecule with explicit hydrogens and conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    return mol


def pocket_around(mol: Chem.Mol, cutoff_spacing: float = 3.2) -> "object":
    """Build a synthetic binding site around a posed ligand (tests only)."""
    from aegis.channels.interaction import Pocket, Residue

    conformer = mol.GetConformer()
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    residue_defs = [
        ("ALA", 1),
        ("LYS", 2),
        ("ASP", 3),
        ("PHE", 4),
        ("SER", 5),
        ("LEU", 6),
        ("VAL", 7),
        ("THR", 8),
    ]
    residues = []
    for (resname, resid), index in zip(residue_defs, heavy):
        point = conformer.GetAtomPosition(index)
        base = np.array([point.x, point.y, point.z])
        coords = base + np.array(
            [
                [cutoff_spacing, 0.0, 0.0],
                [0.0, cutoff_spacing, 0.0],
                [cutoff_spacing, cutoff_spacing, 0.0],
            ]
        )
        residues.append(Residue(resname=resname, resid=resid, chain="A", coords=coords))
    return Pocket(residues=residues)


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


@pytest.fixture(scope="session")
def aspirin_pose(library) -> Chem.Mol:
    record = next(record for record in library.records if record.name == "aspirin")
    return pose_mol(record.smiles)


@pytest.fixture(scope="session")
def pocket_around_aspirin(aspirin_pose):
    return pocket_around(aspirin_pose)

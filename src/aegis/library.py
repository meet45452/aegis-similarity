"""Library preparation and persistence.

Offline preparation computes, for every compound: the canonical standardised
structure, ECFP fingerprint, Murcko hierarchy (exact + generalised, with their
own fingerprints), physicochemical properties, synthetic complexity, a
Boltzmann-weighted conformer ensemble with cached USR/field features, and a
student embedding.  Online queries then pay none of that cost.
"""

from __future__ import annotations

import csv
import pickle
from typing import Callable, Sequence

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from aegis.chemistry import (
    compute_properties,
    generate_conformer_ensemble,
    murcko_frameworks,
    parse_mol,
    synthetic_complexity,
)
from aegis.config import AegisConfig
from aegis.index import Tier0Index
from aegis.student import StudentEmbedder
from aegis.types import MoleculeRecord

ProgressCallback = Callable[[int, int, str], None]


def build_record(
    smiles: str,
    name: str,
    config: AegisConfig,
    generator: rdFingerprintGenerator.MorganGenerator,
    with_conformers: bool = True,
) -> MoleculeRecord | None:
    """Prepare a single molecule; returns None for unparseable input."""
    try:
        mol = parse_mol(smiles)
    except ValueError:
        return None
    fingerprint = generator.GetFingerprint(mol)
    scaffold, generic = murcko_frameworks(mol)
    scaffold_mol = Chem.MolFromSmiles(scaffold) if scaffold else None
    generic_mol = Chem.MolFromSmiles(generic) if generic else None
    ensemble = None
    if with_conformers:
        ensemble = generate_conformer_ensemble(
            mol,
            n_confs=config.n_conformers,
            keep=config.keep_conformers,
            seed=config.conformer_seed,
            temperature=config.boltzmann_temperature,
        )
    return MoleculeRecord(
        name=name,
        smiles=Chem.MolToSmiles(mol),
        mol=mol,
        fingerprint=fingerprint,
        scaffold=scaffold,
        generic_scaffold=generic,
        scaffold_fingerprint=generator.GetFingerprint(scaffold_mol) if scaffold_mol else None,
        generic_fingerprint=generator.GetFingerprint(generic_mol) if generic_mol else None,
        properties=compute_properties(mol),
        synthetic_complexity=synthetic_complexity(mol),
        ensemble=ensemble,
    )


class Library:
    """A prepared, searchable compound collection."""

    def __init__(
        self,
        records: list[MoleculeRecord],
        config: AegisConfig,
        embedder: StudentEmbedder,
        index: Tier0Index,
    ):
        self.records = records
        self.config = config
        self.embedder = embedder
        self.index = index

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def build(
        cls,
        smiles_list: Sequence[str],
        names: Sequence[str] | None = None,
        config: AegisConfig | None = None,
        with_conformers: bool = True,
        progress: ProgressCallback | None = None,
    ) -> "Library":
        """Prepare a library from an iterable of SMILES."""
        config = config or AegisConfig()
        if names is None:
            names = [f"mol_{i}" for i in range(len(smiles_list))]
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=config.ecfp_radius,
            fpSize=config.ecfp_bits,
        )
        records: list[MoleculeRecord] = []
        total = len(smiles_list)
        for position, (smiles, name) in enumerate(zip(smiles_list, names)):
            record = build_record(
                smiles,
                name,
                config,
                generator,
                with_conformers=with_conformers,
            )
            if record is not None:
                records.append(record)
            if progress is not None:
                progress(position + 1, total, name)
        embedder = StudentEmbedder(
            radius=config.ecfp_radius,
            bits=config.student_bits,
            dim=config.student_dim,
        )
        mols = [record.mol for record in records if record.mol is not None]
        if mols:
            embedder.fit(mols)
            for record in records:
                if record.mol is not None:
                    record.student_embedding = embedder.transform(record.mol)
        return cls(records=records, config=config, embedder=embedder, index=Tier0Index(records))

    @classmethod
    def from_csv(
        cls,
        path: str,
        smiles_col: str = "smiles",
        name_col: str = "name",
        config: AegisConfig | None = None,
        with_conformers: bool = True,
        progress: ProgressCallback | None = None,
    ) -> "Library":
        """Build a library from a CSV with ``smiles`` (and optionally ``name``) columns."""
        smiles_list: list[str] = []
        names: list[str] = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                smiles_list.append((row.get(smiles_col) or "").strip())
                names.append((row.get(name_col) or f"mol_{len(names)}").strip())
        return cls.build(
            smiles_list,
            names=names,
            config=config,
            with_conformers=with_conformers,
            progress=progress,
        )

    def save(self, path: str) -> None:
        """Persist the library with pickle (trusted input only)."""
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> "Library":
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"{path!r} does not contain a Library")
        return obj

    def describe(self) -> dict:
        """Library statistics for sanity checking and UI display."""
        with_conformers = sum(1 for record in self.records if record.ensemble is not None)
        families = {record.generic_scaffold for record in self.records if record.generic_scaffold}
        return {
            "n_records": len(self.records),
            "with_conformer_ensembles": with_conformers,
            "generic_scaffold_families": len(families),
        }

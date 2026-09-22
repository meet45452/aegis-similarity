"""Core data structures shared across the AEGIS cascade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem


@dataclass
class ConformerEnsemble:
    """A Boltzmann-weighted conformer ensemble for one molecule.

    Attributes:
        mol: molecule with explicit hydrogens and embedded conformers.
        conformer_ids: kept conformer IDs (lowest-energy, deduplicated).
        energies: relative conformer energies (kcal/mol).
        weights: Boltzmann weights at ``boltzmann_temperature`` (sum to 1).
        usr: USR moment vectors (12 per conformer, heavy atoms only).
        fields: per-conformer pharmacophore-class field features.
        atom_classes: per-atom interaction classes (None for hydrogens).
        charges: Gasteiger partial charges for all atoms.
    """

    mol: Chem.Mol | None = None
    conformer_ids: list[int] = field(default_factory=list)
    energies: np.ndarray | None = None
    weights: np.ndarray | None = None
    usr: list[np.ndarray] = field(default_factory=list)
    fields: list[dict[str, np.ndarray]] = field(default_factory=list)
    atom_classes: list[str | None] = field(default_factory=list)
    charges: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.conformer_ids)


@dataclass
class MoleculeRecord:
    """A prepared molecule: everything needed for Tier-0/1/2 scoring."""

    name: str
    smiles: str
    mol: Chem.Mol | None = None
    fingerprint: Any = None  # RDKit ExplicitBitVect (ECFP)
    scaffold: str = ""  # exact Bemis-Murcko framework SMILES
    generic_scaffold: str = ""  # atom-type-generalised framework SMILES
    scaffold_fingerprint: Any = None
    generic_fingerprint: Any = None
    properties: dict[str, float] = field(default_factory=dict)
    synthetic_complexity: float = 0.0
    ensemble: ConformerEnsemble | None = None
    student_embedding: np.ndarray | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # metadata / user extras


@dataclass
class ChannelScore:
    """One channel's similarity value, router weight, and contribution."""

    channel: str
    value: float | None = None
    weight: float = 0.0
    contribution: float | None = None
    note: str = ""


@dataclass
class UncertaintyReport:
    """Decomposed uncertainty for one scored pair.

    ``total`` enters the final score as ``S_AEGIS = sum w_k S_k - lambda * U``;
    the components explain *why* the system is uncertain.
    """

    total: float = 0.0
    epistemic: float = 0.0
    conformational: float = 0.0
    ood: float = 0.0
    flags: list[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    """A scored, labelled, and explained candidate molecule."""

    record: MoleculeRecord
    score: float = 0.0
    channel_scores: list[ChannelScore] = field(default_factory=list)
    uncertainty: UncertaintyReport = field(default_factory=UncertaintyReport)
    label: str = ""
    explanation: str = ""
    scaffold_hop: bool = False
    pose_used: bool = False

    def channel_value(self, name: str) -> float | None:
        for channel_score in self.channel_scores:
            if channel_score.channel == name:
                return channel_score.value
        return None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.record.name,
            "smiles": self.record.smiles,
            "score": round(self.score, 4),
            "label": self.label,
            "uncertainty": round(self.uncertainty.total, 4),
            "scaffold": self.record.scaffold,
            "scaffold_hop": self.scaffold_hop,
        }
        for channel_score in self.channel_scores:
            row[channel_score.channel] = (
                None if channel_score.value is None else round(channel_score.value, 4)
            )
        return row

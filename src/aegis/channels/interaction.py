"""Interaction channel: pocket-conditioned interaction similarity (Tier 2).

This channel implements the most biologically defensible definition of
scaffold-hop similarity in the AEGIS design: two molecules score highly if
they generate a *similar interaction pattern in a particular binding site*,
even with unrelated cores:

    S_target(L1, L2 | P) = Sim(Phi(L1, P), Phi(L2, P))

How it works
------------
1. A ``PocketBundle`` pairs a parsed binding site (PDB) with a *posed*
   reference ligand (e.g. from a co-crystal or docking).
2. Each candidate's conformers are transferred into the binding site by
   maximum-common-substructure alignment onto the reference pose
   (``McsPoseTransfer``).
3. For every pose, a residue-level interaction fingerprint is computed:
   (residue, contact-type) counts, where contact types are hydrogen bonds,
   hydrophobic contacts, ionic contacts, and aromatic contacts derived from
   ligand atom classes and amino-acid chemistry.
4. Similarity is the generalised Jaccard overlap of the two fingerprints.

Scientific honesty
------------------
* MCS pose transfer covers analogues and scaffold hops that retain a common
  core.  Full scaffold hops with no shared substructure require an external
  pose generator (docking); plug one in by setting ``record.raw["pose_mol"]``
  to pre-posed molecules.
* No induced fit, no solvation, no contact geometry filtering (H-bond line
  angles).  Treat scores as a reranking signal, not a binding-free-energy
  estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign

from aegis.chemistry import classify_atoms
from aegis.types import MoleculeRecord

#: Sidechain interaction classes for the 20 canonical amino acids.
RESIDUE_SIDECHAIN_CLASSES: dict[str, tuple[str, ...]] = {
    "ALA": ("hydrophobic",),
    "VAL": ("hydrophobic",),
    "LEU": ("hydrophobic",),
    "ILE": ("hydrophobic",),
    "MET": ("hydrophobic",),
    "PRO": ("hydrophobic",),
    "PHE": ("hydrophobic", "aromatic"),
    "TRP": ("hydrophobic", "aromatic"),
    "TYR": ("aromatic", "donor", "acceptor"),
    "SER": ("donor", "acceptor"),
    "THR": ("donor", "acceptor"),
    "ASN": ("donor", "acceptor"),
    "GLN": ("donor", "acceptor"),
    "CYS": ("donor",),
    "LYS": ("donor", "positive"),
    "ARG": ("donor", "positive"),
    "HIS": ("donor", "positive", "aromatic"),
    "ASP": ("acceptor", "negative"),
    "GLU": ("acceptor", "negative"),
    "GLY": (),
}

_CONTACT_TYPES = ("hbond", "hydrophobic", "aromatic", "ionic")
_WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD"}


@dataclass
class Residue:
    """One protein residue with heavy-atom coordinates."""

    resname: str
    resid: int
    chain: str
    coords: np.ndarray  # (n_atoms, 3)

    @property
    def key(self) -> str:
        return f"{self.chain}{self.resid}"

    def classes(self) -> set[str]:
        """Sidechain classes plus backbone donor/acceptor capability."""
        result = set(RESIDUE_SIDECHAIN_CLASSES.get(self.resname.upper(), ()))
        result.add("acceptor")  # backbone carbonyl oxygen
        if self.resname.upper() != "PRO":
            result.add("donor")  # backbone amide NH
        return result


@dataclass
class Pocket:
    """A binding site as an ordered list of residues."""

    residues: list[Residue]

    @classmethod
    def from_pdb(cls, path: str) -> "Pocket":
        """Parse heavy atoms from a PDB file (ATOM/HETATM records)."""
        groups: dict[tuple[str, int, str], list[list[float]]] = {}
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                if resname.upper() in _WATER_RESNAMES:
                    continue
                element = line[76:78].strip().upper()
                if element in ("H", "D"):
                    continue
                if not element and atom_name[:1] in ("H", "D"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                chain = line[21:22].strip() or "A"
                try:
                    resid = int(line[22:26])
                except ValueError:
                    resid = 0
                groups.setdefault((chain, resid, resname), []).append([x, y, z])
        residues = [
            Residue(resname=key[2], resid=key[1], chain=key[0], coords=np.asarray(value, dtype=float))
            for key, value in sorted(groups.items())
        ]
        return cls(residues=residues)


@dataclass
class PocketBundle:
    """A binding site plus a posed reference ligand for Tier-2 scoring."""

    pocket: Pocket
    reference: Chem.Mol  # must carry at least one conformer


def contact_type(ligand_class: str | None, residue: Residue) -> str | None:
    """Classify a ligand-atom/residue contact (distance-based, no geometry filtering)."""
    if ligand_class is None:
        return None
    residue_classes = residue.classes()
    if ligand_class == "donor" and "acceptor" in residue_classes:
        return "hbond"
    if ligand_class == "acceptor" and "donor" in residue_classes:
        return "hbond"
    if ligand_class == "aromatic" and "aromatic" in residue_classes:
        return "aromatic"
    if ligand_class in ("hydrophobe", "aromatic") and "hydrophobic" in residue_classes:
        return "hydrophobic"
    if ligand_class == "cation" and "negative" in residue_classes:
        return "ionic"
    if ligand_class == "anion" and "positive" in residue_classes:
        return "ionic"
    return None


def interaction_fingerprint(
    posed_mol: Chem.Mol,
    pocket: Pocket,
    cutoff: float = 4.5,
    conf_id: int = -1,
) -> dict[tuple[str, str], int]:
    """Residue-level interaction counts for one posed ligand."""
    conformer = posed_mol.GetConformer(conf_id)
    classes = classify_atoms(posed_mol)
    contacts: dict[tuple[str, str], int] = {}
    for atom in posed_mol.GetAtoms():
        index = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            continue
        point = conformer.GetAtomPosition(index)
        ligand_position = np.array([point.x, point.y, point.z])
        for residue in pocket.residues:
            distances = np.linalg.norm(residue.coords - ligand_position, axis=1)
            if float(distances.min()) > cutoff:
                continue
            ctype = contact_type(classes[index], residue)
            if ctype is None:
                continue
            key = (residue.key, ctype)
            contacts[key] = contacts.get(key, 0) + 1
    return contacts


def count_jaccard(
    counts_a: dict[tuple[str, str], int],
    counts_b: dict[tuple[str, str], int],
) -> float:
    """Generalised (weighted) Jaccard overlap of two count dictionaries."""
    keys = set(counts_a) | set(counts_b)
    if not keys:
        return 0.0
    intersection = sum(min(counts_a.get(key, 0), counts_b.get(key, 0)) for key in keys)
    union = sum(max(counts_a.get(key, 0), counts_b.get(key, 0)) for key in keys)
    return intersection / union if union > 0 else 0.0


class McsPoseTransfer:
    """Transfer candidate conformers onto a reference pose via MCS alignment."""

    def __init__(self, timeout: int = 10, min_atoms: int = 3):
        self.timeout = timeout
        self.min_atoms = min_atoms

    def pose(
        self,
        mol: Chem.Mol,
        reference: Chem.Mol,
        conformer_ids: Iterable[int] | None = None,
    ) -> list[tuple[Chem.Mol, int]]:
        """Return (aligned_mol, conf_id) pairs; empty when no usable MCS."""
        probe_base = Chem.RemoveHs(Chem.Mol(mol))
        reference_base = Chem.RemoveHs(Chem.Mol(reference))
        if probe_base.GetNumConformers() == 0 or reference_base.GetNumConformers() == 0:
            return []
        try:
            mcs = rdFMCS.FindMCS(
                [reference_base, probe_base],
                timeout=self.timeout,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
            )
        except Exception:  # noqa: BLE001
            return []
        if mcs.numAtoms < self.min_atoms:
            return []
        pattern = Chem.MolFromSmarts(mcs.smartsString)
        if pattern is None:
            return []
        reference_matches = reference_base.GetSubstructMatches(pattern)
        if not reference_matches:
            return []
        reference_match = reference_matches[0]
        if conformer_ids is None:
            conformer_ids = [c.GetId() for c in probe_base.GetConformers()]
        posed: list[tuple[Chem.Mol, int]] = []
        for conf_id in conformer_ids:
            probe = Chem.Mol(probe_base)
            probe_matches = probe.GetSubstructMatches(pattern)
            if not probe_matches:
                continue
            atom_map = list(zip(probe_matches[0], reference_match))
            try:
                rdMolAlign.AlignMol(probe, reference_base, prbCid=conf_id, atomMap=atom_map)
            except Exception:  # noqa: BLE001
                continue
            posed.append((probe, conf_id))
        return posed


class InteractionChannel:
    """Pocket-conditioned interaction-fingerprint similarity."""

    name = "interaction"

    def __init__(self, cutoff: float = 4.5, mcs_timeout: int = 10):
        self.cutoff = cutoff
        self._transfer = McsPoseTransfer(timeout=mcs_timeout)
        self.bundle: PocketBundle | None = None

    def set_context(self, bundle: PocketBundle) -> None:
        """Attach the pocket + posed reference for subsequent scoring."""
        self.bundle = bundle

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        if self.bundle is None:
            return None
        query_fingerprint = self._fingerprint(query)
        cand_fingerprint = self._fingerprint(cand)
        if query_fingerprint is None or cand_fingerprint is None:
            return None
        return float(count_jaccard(query_fingerprint, cand_fingerprint))

    def _fingerprint(self, record: MoleculeRecord) -> dict[tuple[str, str], int] | None:
        """Best-contact fingerprint for a record (user pose or MCS transfer)."""
        if self.bundle is None:
            return None
        pose_mol = record.raw.get("pose_mol")
        if pose_mol is not None:
            return interaction_fingerprint(pose_mol, self.bundle.pocket, self.cutoff)
        if record.ensemble is None or record.ensemble.mol is None:
            return None
        poses = self._transfer.pose(
            record.ensemble.mol,
            self.bundle.reference,
            record.ensemble.conformer_ids,
        )
        best: dict[tuple[str, str], int] | None = None
        best_total = -1
        for posed_mol, conf_id in poses:
            fingerprint = interaction_fingerprint(posed_mol, self.bundle.pocket, self.cutoff, conf_id)
            total = sum(fingerprint.values())
            if total > best_total:
                best_total = total
                best = fingerprint
        return best

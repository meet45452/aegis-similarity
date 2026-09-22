"""Molecular standardisation, conformer ensembles, and descriptor chemistry.

This module is the scientific foundation of AEGIS:

* salt stripping / largest-fragment standardisation,
* Bemis-Murcko scaffold hierarchy (exact and atom-type-generalised),
* ETKDGv3 conformer generation with MMFF (or UFF) optimisation,
* Boltzmann conformer weights,
* USR shape moments (Ballester & Richards, 2007),
* USRCAT-style pharmacophore-class field features (Schreyer & Blundell, 2012),
* Gasteiger partial charges (documented as an ESP proxy, not ground truth),
* a transparent synthetic-complexity proxy.

Scientific honesty notes are included inline where approximations are made.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, SaltRemover, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from aegis.config import PROPERTY_RANGES
from aegis.types import ConformerEnsemble

RDLogger.DisableLog("rdApp.warning")

#: Boltzmann constant in kcal/(mol K).
BOLTZMANN_RT = 0.0019872041

#: Pharmacophore interaction classes used by the field and pocket channels.
ATOM_CLASSES: tuple[str, ...] = (
    "donor",
    "acceptor",
    "hydrophobe",
    "cation",
    "anion",
    "aromatic",
)

_SALT_REMOVER = SaltRemover.SaltRemover()


def standardize(mol: Chem.Mol) -> Chem.Mol:
    """Sanitise, desalt, and keep the largest organic fragment."""
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - surfaced as ValueError to callers
        raise ValueError(f"could not sanitise molecule: {exc}") from exc
    mol = _SALT_REMOVER.StripMol(mol)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        fragments = sorted(fragments, key=lambda m: m.GetNumAtoms(), reverse=True)
        mol = fragments[0]
        Chem.SanitizeMol(mol)
    return mol


def parse_mol(smiles: str) -> Chem.Mol:
    """Parse and standardise a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return standardize(mol)


def murcko_frameworks(mol: Chem.Mol) -> tuple[str, str]:
    """Return (exact, atom-type-generalised) Bemis-Murcko framework SMILES.

    The generic level converts every atom to carbon and every bond to a single
    bond, enabling scaffold-hop aware hierarchy comparison (Bemis & Murcko,
    1996).  Acyclic molecules return two empty strings.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "", ""
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    if generic is None or generic.GetNumAtoms() == 0:
        return Chem.MolToSmiles(scaffold), ""
    return Chem.MolToSmiles(scaffold), Chem.MolToSmiles(generic)


def compute_properties(mol: Chem.Mol) -> dict[str, float]:
    """Drug-relevant physicochemical descriptors for the property channel."""
    return {
        "molwt": float(Descriptors.MolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": float(Descriptors.NumHDonors(mol)),
        "hba": float(Descriptors.NumHAcceptors(mol)),
        "rotb": float(Descriptors.NumRotatableBonds(mol)),
        "charge": float(Chem.GetFormalCharge(mol)),
        "fsp3": float(Descriptors.FractionCSP3(mol)),
    }


def property_vector(properties: dict[str, float]) -> np.ndarray:
    """Normalise properties to [0, 1] over drug-like ranges."""
    order = ("molwt", "clogp", "tpsa", "hbd", "hba", "rotb", "charge", "fsp3")
    values = []
    for key in order:
        low, high = PROPERTY_RANGES[key]
        value = float(properties.get(key, low))
        values.append(min(max((value - low) / (high - low), 0.0), 1.0))
    return np.asarray(values, dtype=float)


def classify_atoms(mol: Chem.Mol) -> list[str | None]:
    """Assign each atom a primary pharmacophore class (None for hydrogens).

    Priority: formal charge > donor/acceptor > aromatic > hydrophobe.
    Works for both implicit- and explicit-hydrogen molecules.
    """
    classes: list[str | None] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            classes.append(None)
            continue
        charge = atom.GetFormalCharge()
        if charge > 0:
            classes.append("cation")
            continue
        if charge < 0:
            classes.append("anion")
            continue
        atomic_num = atom.GetAtomicNum()
        n_hydrogens = atom.GetTotalNumHs()
        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() == 1:
                n_hydrogens += 1
        if atomic_num in (7, 8) and n_hydrogens > 0:
            classes.append("donor")
        elif atomic_num in (7, 8):
            classes.append("acceptor")
        elif atom.GetIsAromatic() and atom.IsInRing():
            classes.append("aromatic")
        else:
            classes.append("hydrophobe")
    return classes


def gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    """Gasteiger partial charges for all atoms.

    Scientific limitation: Gasteiger charges are a crude ESP proxy.  Higher
    quality charges (e.g. from semi-empirical or DFT calculations) should be
    attached by the caller for production use; this baseline keeps the
    dependency footprint minimal.
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:  # noqa: BLE001 - fall back to neutral charges
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    values = []
    for atom in mol.GetAtoms():
        try:
            value = float(atom.GetProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        values.append(value)
    return np.asarray(values, dtype=float)


def boltzmann_weights(energies: np.ndarray, temperature: float = 298.15) -> np.ndarray:
    """Boltzmann population weights p_i proportional to exp(-E_i / RT)."""
    energies = np.asarray(energies, dtype=float)
    if energies.size == 0:
        return energies
    rt = BOLTZMANN_RT * temperature
    relative = energies - np.min(energies)
    weights = np.exp(-relative / rt)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(energies.shape, 1.0 / energies.size)
    return weights / total


def _reference_points(coords: np.ndarray) -> tuple[np.ndarray, ...]:
    """The four USR reference points: ctd, cst, fct, ftf."""
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    closest = coords[int(np.argmin(distances))]
    farthest = coords[int(np.argmax(distances))]
    distances_from_closest = np.linalg.norm(coords - closest, axis=1)
    farthest_from_closest = coords[int(np.argmax(distances_from_closest))]
    return centroid, closest, farthest, farthest_from_closest


def usr_moments(coords: np.ndarray) -> np.ndarray:
    """The 12 USR moments (Ballester & Richards, 2007).

    For each of the four reference points, the mean, standard deviation, and
    skewness of the distribution of atom distances is recorded.  Only
    translation- and rotation-invariant quantities are used.
    """
    moments: list[float] = []
    for reference in _reference_points(coords):
        distances = np.linalg.norm(coords - reference, axis=1)
        mean = float(distances.mean())
        std = float(distances.std())
        skew = float(np.mean(((distances - mean) / std) ** 3)) if std > 1e-9 else 0.0
        moments.extend([mean, std, skew])
    return np.asarray(moments, dtype=float)


def field_features(
    positions: np.ndarray,
    heavy_classes: list[str],
    charges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-pharmacophore-class moment blocks (USRCAT-style).

    For every interaction class present, distances from the class atoms to the
    four global USR reference points are summarised by mean and standard
    deviation.  A ``_charge`` block stores global partial-charge statistics.
    Classes absent from a molecule are simply missing from the dictionary, and
    the field channel only compares classes present in both molecules.
    """
    references = _reference_points(positions)
    features: dict[str, np.ndarray] = {}
    for cls in ATOM_CLASSES:
        indices = [i for i, c in enumerate(heavy_classes) if c == cls]
        if not indices:
            continue
        class_positions = positions[indices]
        values: list[float] = []
        for reference in references:
            distances = np.linalg.norm(class_positions - reference, axis=1)
            values.extend([float(distances.mean()), float(distances.std())])
        features[cls] = np.asarray(values, dtype=float)
    features["_charge"] = np.asarray(
        [
            float(charges.mean()),
            float(charges.std()),
            float(charges.min()),
            float(charges.max()),
        ],
        dtype=float,
    )
    return features


def _count_stereocenters(mol: Chem.Mol) -> int:
    """Version-proof stereo-centre count (assigned or tagged)."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode") or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            count += 1
    return count


def synthetic_complexity(mol: Chem.Mol) -> float:
    """A transparent synthetic-complexity proxy on a 0-10 scale.

    This is NOT the Ertl SA score (Ertl & Schuffenhauer, 2009): it is a
    documented, auditable complexity heuristic based on heavy-atom count,
    ring count, aromatic rings, spiro atoms, bridgehead atoms, stereocentres,
    and macrocycle presence.  Swap in a full SA score implementation for
    production prioritisation.
    """
    ring_info = mol.GetRingInfo()
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = ring_info.NumRings()
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_stereo = _count_stereocenters(mol)
    has_macrocycle = any(len(ring) > 7 for ring in ring_info.AtomRings())
    score = (
        2.0 * n_heavy / 50.0
        + 0.8 * n_rings
        + 0.4 * n_aromatic
        + 1.0 * n_spiro
        + 1.0 * n_bridgehead
        + 0.5 * n_stereo
        + 1.5 * float(has_macrocycle)
    )
    return float(min(max(score, 0.0), 10.0))


def _conformer_positions(mol: Chem.Mol, conf_id: int, atom_indices: list[int]) -> np.ndarray:
    conformer = mol.GetConformer(conf_id)
    positions = []
    for index in atom_indices:
        point = conformer.GetAtomPosition(index)
        positions.append([point.x, point.y, point.z])
    return np.asarray(positions, dtype=float)


def _rmsd_no_align(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 1e9
    difference = a - b
    return float(np.sqrt((difference * difference).sum() / len(a)))


def generate_conformer_ensemble(
    mol: Chem.Mol,
    n_confs: int = 8,
    keep: int = 4,
    seed: int = 42,
    temperature: float = 298.15,
    rmsd_threshold: float = 0.25,
) -> ConformerEnsemble | None:
    """Generate a deduplicated, Boltzmann-weighted conformer ensemble.

    Conformers are embedded with ETKDGv3 (fixed seed for reproducibility),
    optimised with MMFF94 when parameters are available (UFF fallback), ranked
    by energy, deduplicated by heavy-atom RMSD, and weighted by a Boltzmann
    distribution at the requested temperature.
    """
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    try:
        AllChem.EmbedMultipleConfs(work, numConfs=n_confs, params=params)
    except Exception:  # noqa: BLE001 - fall back to random coordinates
        pass
    if work.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        try:
            AllChem.EmbedMultipleConfs(work, numConfs=n_confs, params=params)
        except Exception:  # noqa: BLE001
            return None
    if work.GetNumConformers() == 0:
        return None

    if AllChem.MMFFHasAllMoleculeParams(work):
        results = AllChem.MMFFOptimizeMoleculeConfs(work, maxIters=500)
    elif AllChem.UFFHasAllMoleculeParams(work):
        results = AllChem.UFFOptimizeMoleculeConfs(work, maxIters=500)
    else:
        results = [(0, 0.0)] * work.GetNumConformers()
    energies = np.asarray([float(e) for _, e in results], dtype=float)
    if energies.size != work.GetNumConformers():
        energies = np.zeros(work.GetNumConformers(), dtype=float)
    energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)

    conformer_ids = [c.GetId() for c in work.GetConformers()]
    heavy_indices = [a.GetIdx() for a in work.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy_indices:
        return None
    classes = classify_atoms(work)
    charges = gasteiger_charges(work)
    heavy_classes = [classes[i] for i in heavy_indices]
    heavy_charges = charges[heavy_indices]

    kept_ids: list[int] = []
    kept_energies: list[float] = []
    kept_positions: list[np.ndarray] = []
    for order_index in np.argsort(energies):
        index = int(order_index)
        conf_id = conformer_ids[index]
        positions = _conformer_positions(work, conf_id, heavy_indices)
        if any(_rmsd_no_align(positions, prev) < rmsd_threshold for prev in kept_positions):
            continue
        kept_ids.append(conf_id)
        kept_energies.append(float(energies[index]))
        kept_positions.append(positions)
        if len(kept_ids) >= keep:
            break
    if not kept_ids:
        return None

    weights = boltzmann_weights(np.asarray(kept_energies), temperature)
    usr = [usr_moments(positions) for positions in kept_positions]
    fields = [field_features(positions, heavy_classes, heavy_charges) for positions in kept_positions]
    return ConformerEnsemble(
        mol=work,
        conformer_ids=kept_ids,
        energies=np.asarray(kept_energies, dtype=float),
        weights=weights,
        usr=usr,
        fields=fields,
        atom_classes=classes,
        charges=charges,
    )

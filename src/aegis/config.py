"""Central configuration for the AEGIS similarity cascade.

Every tunable parameter of the cascade lives in :class:`AegisConfig` so that
experiments are reproducible, auditable, and easy to share between campaigns.
Scientific rationale for each parameter block is documented in the README.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

CHANNEL_NAMES: tuple[str, ...] = (
    "topology",
    "scaffold",
    "shape",
    "field",
    "conformer",
    "interaction",
    "property",
    "synthesis",
)

#: Soft drug-like ranges used to normalise the property channel to [0, 1].
PROPERTY_RANGES: dict[str, tuple[float, float]] = {
    "molwt": (150.0, 550.0),
    "clogp": (-1.0, 6.0),
    "tpsa": (0.0, 150.0),
    "hbd": (0.0, 5.0),
    "hba": (0.0, 12.0),
    "rotb": (0.0, 12.0),
    "charge": (-2.0, 2.0),
    "fsp3": (0.0, 1.0),
}


@dataclass
class AegisConfig:
    """Tunable parameters of the AEGIS cascade.

    Attributes mirror the three cascade tiers; see the README for the mapping
    between parameters, tiers, and the scientific literature.
    """

    # 2D fingerprints
    ecfp_radius: int = 2
    ecfp_bits: int = 2048

    # Tier-0 student embedding
    student_bits: int = 1024
    student_dim: int = 256

    # Conformer ensembles (ETKDGv3 + MMFF/UFF, Boltzmann weights)
    n_conformers: int = 8
    keep_conformers: int = 4
    conformer_seed: int = 42
    boltzmann_temperature: float = 298.15

    # Ensemble similarity (see aegis.channels.conformer)
    ensemble_energy_alpha: float = 0.05
    ensemble_temperature: float = 0.15

    # Pocket-conditioned scoring (Tier 2)
    contact_cutoff: float = 4.5
    mcs_timeout: int = 10

    # Score composition: S = sum_k w_k S_k - lambda * U
    lambda_uncertainty: float = 0.15

    # Cascade budgets
    tier0_candidates: int = 200
    tier1_candidates: int = 50
    tier2_candidates: int = 20
    default_top_k: int = 20

    # Decision labels
    analogue_min_score: float = 0.60
    analogue_min_topology: float = 0.50
    hop_min_score: float = 0.55
    hop_min_3d: float = 0.50
    uncertain_min_score: float = 0.45
    novelty_max_topology: float = 0.30
    confident_max_uncertainty: float = 0.40

    def to_dict(self) -> dict:
        """Return the configuration as a plain dictionary (for logging)."""
        return asdict(self)

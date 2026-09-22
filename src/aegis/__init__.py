"""AEGIS - Adaptive Ensemble of Geometric, Interaction, and Scaffold similarity.

AEGIS is a multi-resolution, uncertainty-aware molecular similarity cascade for
drug-discovery decisions.  Instead of a single universal fingerprint, it routes
queries through a hierarchy of increasingly physics- and target-aware scorers
and returns candidates together with per-channel explanations and calibrated
uncertainty.

References for the underlying methods are collected in the project README.
"""

from __future__ import annotations

from aegis.cascade import AegisArray, SearchResult
from aegis.config import AegisConfig, CHANNEL_NAMES
from aegis.library import Library
from aegis.router import CampaignContext, LearnedRouter, Profile, Router
from aegis.types import CandidateResult, MoleculeRecord

__version__ = "0.1.0"

__all__ = [
    "AegisArray",
    "AegisConfig",
    "CampaignContext",
    "CandidateResult",
    "CHANNEL_NAMES",
    "LearnedRouter",
    "Library",
    "MoleculeRecord",
    "Profile",
    "Router",
    "SearchResult",
    "__version__",
]

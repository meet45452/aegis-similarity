"""Context-conditioned routing of channel weights.

The router is the piece that stops AEGIS from making the classic mistake of
one similarity metric for every medicinal-chemistry question.  A campaign
context (target class, novelty requirement, program stage, pocket availability)
selects and blends a base profile of channel weights; a learned router can
instead fit the weights from labelled preference pairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import numpy as np

from aegis.config import CHANNEL_NAMES


class Profile(str, Enum):
    """Curated channel-weight profiles for common campaign archetypes."""

    GENERAL = "general"
    KINASE_HINGE = "kinase_hinge"
    GPCR_LIPOPHILIC = "gpcr_lipophilic"
    ADMET_LIABILITY = "admet_liability"


BASE_PROFILES: dict[str, dict[str, float]] = {
    # Balanced retrieval across notions of similarity.
    "general": {
        "topology": 0.20,
        "scaffold": 0.10,
        "shape": 0.15,
        "field": 0.15,
        "conformer": 0.10,
        "interaction": 0.10,
        "property": 0.10,
        "synthesis": 0.10,
    },
    # Hinge-binder kinase: donor-acceptor geometry and local electronic
    # features dominate; interaction fields are heavily weighted.
    "kinase_hinge": {
        "topology": 0.10,
        "scaffold": 0.05,
        "shape": 0.10,
        "field": 0.25,
        "conformer": 0.15,
        "interaction": 0.25,
        "property": 0.05,
        "synthesis": 0.05,
    },
    # Lipophilic GPCR pocket: 3D hydrophobic shape and conformational
    # accessibility dominate; protein context is rarely available.
    "gpcr_lipophilic": {
        "topology": 0.15,
        "scaffold": 0.10,
        "shape": 0.25,
        "field": 0.15,
        "conformer": 0.15,
        "interaction": 0.05,
        "property": 0.10,
        "synthesis": 0.05,
    },
    # ADMET/CYP-PXR liability work: developability envelope dominates.
    "admet_liability": {
        "topology": 0.10,
        "scaffold": 0.05,
        "shape": 0.10,
        "field": 0.15,
        "conformer": 0.05,
        "interaction": 0.05,
        "property": 0.40,
        "synthesis": 0.10,
    },
}


@dataclass
class CampaignContext:
    """The campaign context ``c`` in ``S_AEGIS(q, m, c)``.

    Attributes:
        profile: target-class archetype selecting the base weight profile.
        novelty_requirement: 0 = closest analogues, 1 = maximal scaffold hopping.
        stage: "discovery" or "optimization" (optimization boosts property/
            synthesis weights).
        pocket_available: True when Tier-2 pocket scoring will run.
        channel_overrides: per-channel manual overrides applied last.
    """

    profile: Profile | str = Profile.GENERAL
    novelty_requirement: float = 0.3
    stage: str = "discovery"
    pocket_available: bool = False
    channel_overrides: dict[str, float] | None = None

    def profile_name(self) -> str:
        if isinstance(self.profile, Profile):
            return self.profile.value
        return str(self.profile)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile_name(),
            "novelty_requirement": self.novelty_requirement,
            "stage": self.stage,
            "pocket_available": self.pocket_available,
        }


class Router:
    """Maps a campaign context to channel weights summing to 1."""

    def weights(self, context: CampaignContext) -> dict[str, float]:
        base = dict(BASE_PROFILES.get(context.profile_name(), BASE_PROFILES[Profile.GENERAL.value]))
        novelty = float(np.clip(context.novelty_requirement, 0.0, 1.0))
        # Novelty shifts mass away from topology/scaffold toward 3D/field experts.
        for source in ("topology", "scaffold"):
            if source in base:
                base[source] *= 1.0 - 0.8 * novelty
        for destination in ("shape", "field", "conformer", "interaction"):
            if destination in base:
                base[destination] *= 1.0 + 0.8 * novelty
        if context.stage == "optimization":
            for key in ("property", "synthesis"):
                if key in base:
                    base[key] *= 1.5
        if context.pocket_available and base.get("interaction", 0.0) > 0.0:
            base["interaction"] *= 1.5
        if context.channel_overrides:
            for key, value in context.channel_overrides.items():
                if key in base:
                    base[key] = float(value)
        total = sum(value for value in base.values() if value > 0.0)
        if total <= 0.0:
            return {key: 1.0 / len(base) for key in base}
        return {key: max(0.0, value) / total for key, value in base.items()}


class LearnedRouter(Router):
    """Channel weights learned from pairwise preference labels.

    ``fit`` trains a logistic-regression on channel-score *differences*
    between a preferred and a non-preferred candidate for the same query;
    positive coefficients become channel weights.  This grounds the learned
    router in the same expert judgement the base profiles encode by hand.
    """

    def __init__(self, channels: tuple[str, ...] = CHANNEL_NAMES):
        self.channels = list(channels)
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        score_differences: np.ndarray,
        labels: np.ndarray,
        l2: float = 1e-2,
        lr: float = 0.5,
        n_iter: int = 2000,
    ) -> "LearnedRouter":
        x = np.asarray(score_differences, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.channels):
            raise ValueError(f"expected score_differences of shape (n, {len(self.channels)})")
        if len(y) != len(x):
            raise ValueError("labels and score_differences must have equal length")
        design = np.hstack([x, np.ones((len(x), 1))])
        theta = np.zeros(design.shape[1])
        penalty = np.concatenate([np.full(len(self.channels), l2), [0.0]])
        for _ in range(n_iter):
            logits = np.clip(design @ theta, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (probabilities - y) / len(y)
            theta -= lr * (gradient + penalty * theta)
        self.coef_ = theta[: len(self.channels)]
        return self

    def weights(self, context: CampaignContext) -> dict[str, float]:
        if self.coef_ is None:
            return super().weights(context)
        clipped = np.clip(self.coef_, 0.0, None)
        total = float(clipped.sum())
        if total <= 0.0:
            return super().weights(context)
        return {name: float(value / total) for name, value in zip(self.channels, clipped)}

    def save(self, path: str) -> None:
        """Persist the learned coefficients to JSON."""
        payload = {
            "channels": self.channels,
            "coef": None if self.coef_ is None else self.coef_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> "LearnedRouter":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        router = cls(channels=tuple(payload["channels"]))
        coef = payload.get("coef")
        router.coef_ = None if coef is None else np.asarray(coef, dtype=float)
        return router

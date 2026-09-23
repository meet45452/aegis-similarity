"""Weighted multi-channel scoring with an uncertainty penalty.

Implements

    S_AEGIS(q, m, c) = sum_k w_k(q, c) * S_k(q, m) - lambda * U(q, m, c)

where the weights come from the router (conditioned on the campaign context),
channels that cannot score a pair are excluded with renormalisation, and U is
the decomposed uncertainty report.
"""

from __future__ import annotations

from aegis.channels.base import Channel
from aegis.channels.conformer import ConformerChannel
from aegis.channels.field import FieldChannel
from aegis.channels.interaction import InteractionChannel, PocketBundle
from aegis.channels.property import PropertyChannel
from aegis.channels.scaffold import ScaffoldChannel
from aegis.channels.shape import ShapeChannel
from aegis.channels.synthesis import SynthesisChannel
from aegis.channels.topology import TopologyChannel
from aegis.config import AegisConfig
from aegis.router import CampaignContext, Router
from aegis.types import ChannelScore, MoleculeRecord, UncertaintyReport
from aegis.uncertainty import uncertainty_report


class AegisScorer:
    """Combines all similarity channels for a single (query, candidate) pair."""

    def __init__(self, config: AegisConfig | None = None, router: Router | None = None):
        self.config = config or AegisConfig()
        self.router = router or Router()
        self.interaction = InteractionChannel(
            cutoff=self.config.contact_cutoff,
            mcs_timeout=self.config.mcs_timeout,
        )
        self.channels: dict[str, Channel] = {
            "topology": TopologyChannel(),
            "scaffold": ScaffoldChannel(),
            "shape": ShapeChannel(),
            "field": FieldChannel(),
            "conformer": ConformerChannel(
                energy_alpha=self.config.ensemble_energy_alpha,
                temperature=self.config.ensemble_temperature,
            ),
            "interaction": self.interaction,
            "property": PropertyChannel(),
            "synthesis": SynthesisChannel(),
        }

    def score(
        self,
        query: MoleculeRecord,
        cand: MoleculeRecord,
        context: CampaignContext,
        pocket_bundle: PocketBundle | None = None,
        best_neighbor_similarity: float | None = None,
    ) -> tuple[float, list[ChannelScore], UncertaintyReport] | None:
        """Score one pair; returns None when no channel can score it."""
        if pocket_bundle is not None:
            self.interaction.set_context(pocket_bundle)

        values: dict[str, float | None] = {}
        for name, channel in self.channels.items():
            try:
                values[name] = channel.similarity(query, cand)
            except Exception:  # noqa: BLE001 - a channel failure must not kill the search
                values[name] = None

        available = {name: value for name, value in values.items() if value is not None}
        if not available:
            return None

        base_weights = self.router.weights(context)
        weight_sum = sum(base_weights.get(name, 0.0) for name in available)
        if weight_sum > 0.0:
            weights = {name: base_weights.get(name, 0.0) / weight_sum for name in available}
        else:
            weights = {name: 1.0 / len(available) for name in available}

        conformer_spread = None
        try:
            conformer_spread = self.channels["conformer"].spread(query, cand)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            conformer_spread = None

        missing = [name for name, value in values.items() if value is None]
        uncertainty = uncertainty_report(
            values,
            weights,
            conformer_spread=conformer_spread,
            best_neighbor_similarity=best_neighbor_similarity,
            missing_channels=missing,
        )

        score = (
            sum(weights[name] * available[name] for name in available)
            - self.config.lambda_uncertainty * uncertainty.total
        )

        channel_scores: list[ChannelScore] = []
        for name in self.channels:
            weight = weights.get(name, 0.0) if name in available else 0.0
            contribution = weight * available[name] if name in available else None
            channel_scores.append(
                ChannelScore(
                    channel=name,
                    value=values.get(name),
                    weight=weight,
                    contribution=contribution,
                )
            )
        return score, channel_scores, uncertainty

"""The three-tier AEGIS search cascade.

Tier 0 - ultra-fast global routing: union of ECFP, student-embedding, and
    scaffold-family neighbours over the whole library (no physics applied).
Tier 1 - physics-aware refinement: full channel scoring (topology, scaffold,
    USR shape, interaction fields, Boltzmann conformer ensemble, property,
    synthesis) with uncertainty penalty, on the Tier-0 shortlist only.
Tier 2 - target-specific refinement: pocket-conditioned interaction
    reranking (MCS pose transfer onto a posed reference ligand) for finalists,
    followed by decision labelling and explanation.

The system is fast globally because expensive physics is never applied
globally, and high quality locally because final decisions never rely on a
single lossy embedding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem

from aegis.channels.interaction import Pocket, PocketBundle
from aegis.config import AegisConfig
from aegis.explain import explain_result
from aegis.library import Library, build_record
from aegis.router import CampaignContext
from aegis.scoring import AegisScorer
from aegis.types import CandidateResult, MoleculeRecord

LABELS: tuple[str, ...] = (
    "high_confidence_analogue",
    "high_confidence_scaffold_hop",
    "high_upside_uncertain",
    "novelty_candidate",
    "low_confidence",
)


@dataclass
class SearchResult:
    """Everything returned by one cascade run."""

    query: MoleculeRecord
    context: CampaignContext
    results: list[CandidateResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    tier_counts: dict[str, int] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([result.to_row() for result in self.results])

    def summary(self) -> dict:
        return {
            "n_results": len(self.results),
            "labels": {label: sum(1 for r in self.results if r.label == label) for label in LABELS},
            "tier_counts": dict(self.tier_counts),
            "timings_s": {key: round(value, 4) for key, value in self.timings.items()},
        }


class AegisArray:
    """The public search API: ``AegisArray(library).search(...)``."""

    def __init__(self, library: Library, config: AegisConfig | None = None):
        self.library = library
        self.config = config or library.config
        self.scorer = AegisScorer(self.config)

    def _prepare_query(self, query: str | Chem.Mol | MoleculeRecord) -> MoleculeRecord:
        if isinstance(query, MoleculeRecord):
            return query
        if isinstance(query, Chem.Mol):
            smiles = Chem.MolToSmiles(query)
            record = build_record(smiles, "query", self.config, _make_generator(self.config))
        else:
            record = build_record(str(query), "query", self.config, _make_generator(self.config))
        if record is None:
            raise ValueError(f"could not process query: {query!r}")
        return record

    def search(
        self,
        query: str | Chem.Mol | MoleculeRecord,
        context: CampaignContext | None = None,
        top_k: int | None = None,
        pocket: Pocket | None = None,
        reference_pose: Chem.Mol | None = None,
    ) -> SearchResult:
        """Run the full cascade for one query.

        Args:
            query: SMILES, RDKit molecule, or pre-built record.
            context: campaign context for the router.
            top_k: number of results to return.
            pocket: parsed binding site for optional Tier-2 reranking.
            reference_pose: posed *query* ligand (e.g. from a co-crystal or
                docking) whose conformer defines the binding-mode frame.
        """
        context = context or CampaignContext()
        top_k = top_k or self.config.default_top_k
        query_record = self._prepare_query(query)
        timings: dict[str, float] = {}

        # ---- Tier 0: global routing -------------------------------------
        start = time.perf_counter()
        best_similarity = self.library.index.best_similarity(query_record)
        candidates = self.library.index.search(
            query_record,
            limit=self.config.tier0_candidates,
        )
        timings["tier0_routing"] = time.perf_counter() - start

        # ---- Tier 1: physics-aware shortlist refinement ------------------
        start = time.perf_counter()
        scored: list[CandidateResult] = []
        for position in candidates:
            record = self.library.records[position]
            outcome = self.scorer.score(
                query_record,
                record,
                context,
                best_neighbor_similarity=best_similarity,
            )
            if outcome is None:
                continue
            score, channel_scores, uncertainty = outcome
            scored.append(
                CandidateResult(
                    record=record,
                    score=score,
                    channel_scores=channel_scores,
                    uncertainty=uncertainty,
                    scaffold_hop=(
                        bool(record.generic_scaffold)
                        and record.generic_scaffold != query_record.generic_scaffold
                    ),
                )
            )
        scored.sort(key=lambda result: result.score, reverse=True)
        tier1 = scored[: self.config.tier1_candidates]
        timings["tier1_scoring"] = time.perf_counter() - start
        tier_counts = {"tier0_candidates": len(candidates), "tier1_scored": len(tier1)}

        # ---- Tier 2: pocket-conditioned reranking ------------------------
        results = tier1
        if pocket is not None and reference_pose is not None:
            start = time.perf_counter()
            bundle = PocketBundle(pocket=pocket, reference=reference_pose)
            if query_record.raw.get("pose_mol") is None:
                # Default: the reference pose IS the posed query ligand.
                query_record.raw["pose_mol"] = reference_pose
            reranked: list[CandidateResult] = []
            for result in tier1[: self.config.tier2_candidates]:
                outcome = self.scorer.score(
                    query_record,
                    result.record,
                    context,
                    pocket_bundle=bundle,
                    best_neighbor_similarity=best_similarity,
                )
                if outcome is not None:
                    score, channel_scores, uncertainty = outcome
                    result.score = score
                    result.channel_scores = channel_scores
                    result.uncertainty = uncertainty
                    result.pose_used = True
                reranked.append(result)
            reranked.sort(key=lambda result: result.score, reverse=True)
            results = reranked
            timings["tier2_interaction"] = time.perf_counter() - start
            tier_counts["tier2_reranked"] = len(results)

        for result in results:
            result.label = _label(result, self.config)
            result.explanation = explain_result(result, query_record)
        return SearchResult(
            query=query_record,
            context=context,
            results=results[:top_k],
            timings=timings,
            tier_counts=tier_counts,
        )


def _make_generator(config: AegisConfig):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=config.ecfp_radius,
        fpSize=config.ecfp_bits,
    )


def _label(result: CandidateResult, config: AegisConfig) -> str:
    """Assign a decision label using the configured thresholds."""
    topology = result.channel_value("topology") or 0.0
    shape = result.channel_value("shape")
    field = result.channel_value("field")
    three_d: float | None = None
    if shape is not None and field is not None:
        three_d = 0.5 * (shape + field)
    elif shape is not None:
        three_d = shape
    uncertainty = result.uncertainty.total
    if (
        result.score >= config.analogue_min_score
        and topology >= config.analogue_min_topology
        and uncertainty < config.confident_max_uncertainty
    ):
        return "high_confidence_analogue"
    if (
        result.scaffold_hop
        and result.score >= config.hop_min_score
        and (three_d or 0.0) >= config.hop_min_3d
        and uncertainty < config.confident_max_uncertainty
    ):
        return "high_confidence_scaffold_hop"
    if result.scaffold_hop and result.score >= config.uncertain_min_score:
        return "high_upside_uncertain"
    if topology < config.novelty_max_topology:
        return "novelty_candidate"
    return "low_confidence"

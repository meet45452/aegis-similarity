"""Human-readable explanations of AEGIS similarity decisions.

Every score is auditable: this module turns the channel-level contributions,
scaffold relationship, property shifts, and uncertainty flags into the
"why similar" narrative a medicinal chemist needs before acting on a result.
"""

from __future__ import annotations

from aegis.types import CandidateResult, MoleculeRecord

LABEL_DESCRIPTIONS: dict[str, str] = {
    "high_confidence_analogue": (
        "Close analogue: high overall similarity, strong topological agreement, low uncertainty."
    ),
    "high_confidence_scaffold_hop": (
        "Scaffold hop: different Murcko framework but matching 3D/field profile at low uncertainty."
    ),
    "high_upside_uncertain": (
        "High-upside hypothesis: scaffold hop with meaningful similarity but elevated uncertainty - "
        "assay data on this candidate would most improve the model."
    ),
    "novelty_candidate": (
        "Topology-distant candidate retained for novelty / IP-distance exploration."
    ),
    "low_confidence": (
        "Below decision thresholds; inspect the channel breakdown before acting."
    ),
}

CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "topology": "2D ECFP overlap",
    "scaffold": "Murcko framework hierarchy agreement",
    "shape": "USR 3D shape overlap",
    "field": "interaction-field agreement",
    "conformer": "Boltzmann-weighted ensemble agreement",
    "interaction": "pocket-conditioned interaction agreement",
    "property": "physicochemical profile agreement",
    "synthesis": "synthetic-complexity proximity",
}


def explain_result(result: CandidateResult, query: MoleculeRecord) -> str:
    """Build the "why similar" explanation for one candidate."""
    parts: list[str] = []
    label_note = LABEL_DESCRIPTIONS.get(result.label)
    if label_note:
        parts.append(label_note)

    contributions = sorted(
        (channel_score for channel_score in result.channel_scores if channel_score.contribution is not None),
        key=lambda channel_score: channel_score.contribution or 0.0,
        reverse=True,
    )
    top = contributions[:3]
    if top:
        described = ", ".join(
            f"{CHANNEL_DESCRIPTIONS.get(channel_score.channel, channel_score.channel)} "
            f"({channel_score.value:.2f}, weight {channel_score.weight:.2f})"
            for channel_score in top
        )
        parts.append(f"Driven mainly by: {described}.")

    if result.scaffold_hop:
        parts.append(
            f"Murcko scaffold differs from the query "
            f"({query.generic_scaffold or 'acyclic'} -> {result.record.generic_scaffold or 'acyclic'}): "
            "a scaffold-hop candidate."
        )
    else:
        parts.append("Retains the query Murcko scaffold family.")

    query_properties = query.properties or {}
    candidate_properties = result.record.properties or {}
    if query_properties and candidate_properties:
        deltas = []
        for key, fmt in (("molwt", "+.1f"), ("clogp", "+.2f"), ("tpsa", "+.1f")):
            if key in query_properties and key in candidate_properties:
                delta = candidate_properties[key] - query_properties[key]
                deltas.append(f"d{key}={delta:{fmt}}")
        if deltas:
            parts.append("Property shifts: " + ", ".join(deltas) + ".")

    if result.uncertainty.flags:
        parts.append("Uncertainty flags: " + "; ".join(result.uncertainty.flags) + ".")
    parts.append(f"Score {result.score:.3f} with uncertainty {result.uncertainty.total:.3f}.")
    return " ".join(parts)

"""Uncertainty decomposition for AEGIS scores.

Most similarity systems output ``S(A, B) = 0.82``.  A discovery team needs
``S(A, B) = 0.82 +/- 0.20`` together with *why* the uncertainty exists.  This
module decomposes uncertainty into:

* **epistemic**: disagreement between the active similarity channels,
* **ood**: distance of the query from the library distribution (out-of-domain
  risk),
* **conformational**: spread of the ensemble pair-score distribution,

and produces machine-readable flags that downstream decision rules surface
to the user.
"""

from __future__ import annotations

import numpy as np

from aegis.types import UncertaintyReport


def uncertainty_report(
    channel_values: dict[str, float | None],
    weights: dict[str, float],
    conformer_spread: float | None = None,
    best_neighbor_similarity: float | None = None,
    missing_channels: list[str] | None = None,
) -> UncertaintyReport:
    """Build an :class:`UncertaintyReport` for one scored pair."""
    flags: list[str] = []
    for missing in missing_channels or []:
        flags.append(f"channel_unavailable:{missing}")

    available = {
        name: value
        for name, value in channel_values.items()
        if value is not None and weights.get(name, 0.0) > 0.0
    }
    if not available:
        return UncertaintyReport(total=1.0, flags=flags or ["no_channels"])

    names = list(available)
    values = np.asarray([available[name] for name in names], dtype=float)
    weight_vector = np.asarray([weights[name] for name in names], dtype=float)
    weight_vector = weight_vector / weight_vector.sum()

    mean = float((weight_vector * values).sum())
    epistemic = float(np.sqrt((weight_vector * (values - mean) ** 2).sum()))
    if epistemic > 0.25:
        flags.append("high_channel_disagreement")

    topology = channel_values.get("topology")
    shape = channel_values.get("shape")
    field = channel_values.get("field")
    if topology is not None and shape is not None and field is not None:
        if topology < 0.3 and 0.5 * (shape + field) > 0.6:
            flags.append("scaffold_hop_signature")

    if best_neighbor_similarity is None:
        ood = 0.3
        flags.append("no_library_reference")
    else:
        ood = float(np.clip(1.0 - best_neighbor_similarity, 0.0, 1.0))
        if best_neighbor_similarity < 0.2:
            flags.append("out_of_library_distribution")

    conformational = float(np.clip(conformer_spread or 0.0, 0.0, 1.0))
    if conformer_spread is not None and conformer_spread > 0.25:
        flags.append("conformational_ambiguity")

    total = float(np.clip(0.5 * epistemic + 0.3 * ood + 0.2 * conformational, 0.0, 1.0))
    return UncertaintyReport(
        total=total,
        epistemic=epistemic,
        conformational=conformational,
        ood=ood,
        flags=flags,
    )

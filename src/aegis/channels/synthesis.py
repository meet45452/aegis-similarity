"""Synthesis channel: synthetic-complexity proximity."""

from __future__ import annotations

from aegis.types import MoleculeRecord


class SynthesisChannel:
    """Agreement of synthetic complexity between query and candidate.

    Complexity is the transparent proxy from :func:`aegis.chemistry.synthetic_complexity`
    (0-10 scale).  Similarity decays linearly with the absolute difference,
    so analogue work keeps make-on-demand tractability comparable.
    """

    name = "synthesis"
    scale = 10.0

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        if query.synthetic_complexity is None or cand.synthetic_complexity is None:
            return None
        difference = abs(query.synthetic_complexity - cand.synthetic_complexity)
        return float(max(0.0, 1.0 - difference / self.scale))

"""Similarity channels for AEGIS.

Each channel is a small class exposing ``name`` and
``similarity(query, cand) -> float | None``.  Returning ``None`` signals that
the channel is not applicable to a pair (for example the interaction channel
without a pocket, or 3D channels for molecules without conformer ensembles);
the scorer then renormalises router weights over the available channels.
"""

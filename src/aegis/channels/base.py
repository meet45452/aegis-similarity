"""Channel protocol shared by all similarity experts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aegis.types import MoleculeRecord


@runtime_checkable
class Channel(Protocol):
    """A single notion of molecular resemblance.

    ``similarity`` must return a value in [0, 1] or ``None`` when the channel
    cannot score the pair (missing prerequisites, unsupported chemistry).
    """

    name: str

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None: ...

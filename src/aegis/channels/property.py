"""Property channel: physicochemical profile agreement."""

from __future__ import annotations

import numpy as np

from aegis.chemistry import property_vector
from aegis.types import MoleculeRecord


class PropertyChannel:
    """Distance in normalised property space (MW, cLogP, TPSA, HBD, HBA, ...).

    The ADMET-liability router profile heavily weights this channel: two
    molecules presenting a similar developability envelope are interchangeable
    in rescue workflows even when their topology differs.
    """

    name = "property"

    def similarity(self, query: MoleculeRecord, cand: MoleculeRecord) -> float | None:
        if not query.properties or not cand.properties:
            return None
        vector_a = property_vector(query.properties)
        vector_b = property_vector(cand.properties)
        return float(1.0 - np.mean(np.abs(vector_a - vector_b)))

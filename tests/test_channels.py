"""Tests for the individual similarity channels."""

from __future__ import annotations

import numpy as np
import pytest

from aegis.channels.conformer import ConformerChannel
from aegis.channels.field import FieldChannel
from aegis.channels.interaction import (
    InteractionChannel,
    Pocket,
    PocketBundle,
    Residue,
    interaction_fingerprint,
)
from aegis.channels.property import PropertyChannel
from aegis.channels.scaffold import ScaffoldChannel
from aegis.channels.shape import ShapeChannel
from aegis.channels.synthesis import SynthesisChannel
from aegis.channels.topology import TopologyChannel


def _record(library, name):
    for record in library.records:
        if record.name == name:
            return record
    raise KeyError(name)


def test_topology_channel():
    channel = TopologyChannel()
    aspirin = _record(library := pytest.Library if False else None, "aspirin") if False else None

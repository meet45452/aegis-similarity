"""Tests for the individual similarity channels."""

from __future__ import annotations

import pytest

from aegis.channels.conformer import ConformerChannel
from aegis.channels.field import FieldChannel
from aegis.channels.interaction import (
    InteractionChannel,
    PocketBundle,
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


def test_topology_channel(library):
    channel = TopologyChannel()
    aspirin = _record(library, "aspirin")
    salicylic = _record(library, "salicylic_acid")
    caffeine = _record(library, "caffeine")
    assert channel.similarity(aspirin, aspirin) == 1.0
    assert channel.similarity(aspirin, salicylic) > channel.similarity(aspirin, caffeine)


def test_scaffold_channel(library):
    channel = ScaffoldChannel()
    benzamide = _record(library, "benzamide")
    benzoic = _record(library, "benzoic_acid")
    metformin = _record(library, "metformin")
    quinoline = _record(library, "quinoline")
    # Both monocyclic aromatics reduce to the bare phenyl Murcko framework.
    assert channel.similarity(benzamide, benzoic) == 1.0
    # One acyclic molecule -> maximally dissimilar scaffold relationship.
    assert channel.similarity(benzamide, metformin) == 0.0
    # Both acyclic -> channel not applicable.
    assert channel.similarity(metformin, metformin) is None
    # Fused heteroaromatic differs from the phenyl framework.
    assert channel.similarity(benzamide, quinoline) < 1.0


def test_shape_field_conformer_channels(library):
    aspirin = _record(library, "aspirin")
    caffeine = _record(library, "caffeine")
    shape = ShapeChannel()
    field = FieldChannel()
    conformer = ConformerChannel()
    assert shape.similarity(aspirin, aspirin) == pytest.approx(1.0)
    assert field.similarity(aspirin, aspirin) == pytest.approx(1.0)
    assert conformer.similarity(aspirin, aspirin) >= 0.8
    assert shape.similarity(aspirin, caffeine) < 1.0
    assert conformer.spread(aspirin, aspirin) is not None


def test_property_and_synthesis_channels(library):
    property_channel = PropertyChannel()
    synthesis_channel = SynthesisChannel()
    aspirin = _record(library, "aspirin")
    paracetamol = _record(library, "paracetamol")
    caffeine = _record(library, "caffeine")
    assert property_channel.similarity(aspirin, aspirin) == pytest.approx(1.0)
    assert property_channel.similarity(aspirin, paracetamol) > 0.5
    assert synthesis_channel.similarity(aspirin, aspirin) == pytest.approx(1.0)
    assert aspirin.synthetic_complexity != caffeine.synthetic_complexity


def test_interaction_fingerprint_makes_contacts(library, aspirin_pose, pocket_around_aspirin):
    fingerprint = interaction_fingerprint(aspirin_pose, pocket_around_aspirin, cutoff=4.5)
    assert fingerprint  # the synthetic pocket produces contacts
    valid_types = {"hbond", "hydrophobic", "aromatic", "ionic"}
    assert all(contact_type in valid_types for _, contact_type in fingerprint)


def test_interaction_channel_self_similarity(library, aspirin_pose, pocket_around_aspirin):
    aspirin = _record(library, "aspirin")
    channel = InteractionChannel()
    channel.set_context(PocketBundle(pocket=pocket_around_aspirin, reference=aspirin_pose))
    aspirin.raw["pose_mol"] = aspirin_pose
    assert channel.similarity(aspirin, aspirin) == pytest.approx(1.0)


def test_interaction_channel_mcs_transfer_and_abstention(
    library, aspirin_pose, pocket_around_aspirin
):
    aspirin = _record(library, "aspirin")
    salicylic = _record(library, "salicylic_acid")
    metformin = _record(library, "metformin")
    channel = InteractionChannel()
    channel.set_context(PocketBundle(pocket=pocket_around_aspirin, reference=aspirin_pose))
    aspirin.raw["pose_mol"] = aspirin_pose
    # Salicylic acid shares the salicylate core with aspirin -> MCS pose transfer works.
    value = channel.similarity(aspirin, salicylic)
    assert value is not None
    assert 0.0 <= value <= 1.0
    # Metformin shares no 3-atom MCS with aspirin -> the channel abstains.
    assert channel.similarity(aspirin, metformin) is None
    # Without a pocket context the channel always abstains.
    assert InteractionChannel().similarity(aspirin, salicylic) is None

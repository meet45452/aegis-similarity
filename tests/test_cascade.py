"""End-to-end tests for the three-tier AEGIS cascade."""

from __future__ import annotations

import pytest

from aegis.cascade import LABELS, AegisArray
from aegis.router import CampaignContext, Profile

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_self_retrieval(library):
    array = AegisArray(library)
    result = array.search(ASPIRIN, top_k=10)
    assert result.results
    assert result.results[0].record.name == "aspirin"
    scores = [candidate.score for candidate in result.results]
    assert scores == sorted(scores, reverse=True)


def test_labels_and_explanations(library):
    array = AegisArray(library)
    result = array.search(ASPIRIN, top_k=10)
    for candidate in result.results:
        assert candidate.label in LABELS
        assert candidate.explanation
        assert candidate.uncertainty.total >= 0.0
    frame = result.to_dataframe()
    assert len(frame) == 10
    assert {"name", "smiles", "score", "label", "uncertainty"} <= set(frame.columns)


def test_context_changes_ranking(library):
    array = AegisArray(library)
    close = array.search(ASPIRIN, CampaignContext(novelty_requirement=0.0), top_k=10)
    novel = array.search(
        ASPIRIN,
        CampaignContext(profile=Profile.KINASE_HINGE, novelty_requirement=0.9),
        top_k=10,
    )
    assert [c.record.name for c in close.results][0] == "aspirin"
    assert [c.record.name for c in close.results] != [c.record.name for c in novel.results]


def test_tier2_pocket_reranking(library, aspirin_pose, pocket_around_aspirin):
    array = AegisArray(library)
    context = CampaignContext(
        profile=Profile.KINASE_HINGE,
        novelty_requirement=0.5,
        pocket_available=True,
    )
    result = array.search(
        ASPIRIN,
        context=context,
        top_k=10,
        pocket=pocket_around_aspirin,
        reference_pose=aspirin_pose,
    )
    assert "tier2_interaction" in result.timings
    interaction_values = [candidate.channel_value("interaction") for candidate in result.results]
    assert any(value is not None for value in interaction_values)
    scores = [candidate.score for candidate in result.results]
    assert scores == sorted(scores, reverse=True)


def test_invalid_query_raises(library):
    array = AegisArray(library)
    with pytest.raises(ValueError):
        array.search("not_a_smiles")


def test_library_roundtrip(library, tmp_path):
    path = str(tmp_path / "library.aegis")
    library.save(path)
    loaded = type(library).load(path)
    assert len(loaded) == len(library)
    assert loaded.describe() == library.describe()


def test_search_without_conformers(library):
    from aegis.library import Library

    plain = Library.build(
        ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O", "NC(=O)c1ccccc1"],
        with_conformers=False,
    )
    result = AegisArray(plain).search(ASPIRIN, top_k=3)
    assert result.results
    # 3D channels abstained; the scorer renormalised over the 2D channels.
    for candidate in result.results:
        assert candidate.channel_value("shape") is None
        assert candidate.channel_value("topology") is not None

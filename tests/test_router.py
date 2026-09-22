"""Tests for the context-conditioned router."""

from __future__ import annotations

import numpy as np

from aegis.router import CampaignContext, LearnedRouter, Profile, Router


def test_weights_sum_to_one():
    router = Router()
    for profile in Profile:
        weights = router.weights(CampaignContext(profile=profile))
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert all(value >= 0.0 for value in weights.values())


def test_profiles_emphasise_expected_channels():
    router = Router()
    general = router.weights(CampaignContext(profile=Profile.GENERAL))
    kinase = router.weights(CampaignContext(profile=Profile.KINASE_HINGE))
    admet = router.weights(CampaignContext(profile=Profile.ADMET_LIABILITY))
    assert kinase["interaction"] > general["interaction"]
    assert kinase["field"] > general["field"]
    assert admet["property"] == max(admet.values())


def test_novelty_shifts_weight_from_topology():
    router = Router()
    low = router.weights(CampaignContext(novelty_requirement=0.0))
    high = router.weights(CampaignContext(novelty_requirement=1.0))
    assert high["topology"] < low["topology"]
    assert high["field"] > low["field"]


def test_stage_and_pocket_effects():
    router = Router()
    discovery = router.weights(CampaignContext(stage="discovery"))
    optimization = router.weights(CampaignContext(stage="optimization"))
    assert optimization["property"] > discovery["property"]
    no_pocket = router.weights(CampaignContext(pocket_available=False))
    with_pocket = router.weights(CampaignContext(pocket_available=True))
    assert with_pocket["interaction"] > no_pocket["interaction"]


def test_channel_overrides():
    router = Router()
    context = CampaignContext(channel_overrides={"topology": 0.9})
    weights = router.weights(context)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["topology"] > 0.3


def test_learned_router_prefers_informative_channel():
    router = LearnedRouter()
    rng = np.random.default_rng(0)
    # CHANNEL_NAMES index 2 is "shape".
    differences = rng.random((300, 8))
    labels = (differences[:, 2] + 0.05 * rng.random(300) > 0.5).astype(float)
    router.fit(differences, labels)
    weights = router.weights(CampaignContext())
    assert weights["shape"] == max(weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_learned_router_save_load(tmp_path):
    import numpy as np

    router = LearnedRouter()
    rng = np.random.default_rng(1)
    differences = rng.random((100, 8))
    labels = (differences[:, 2] > 0.5).astype(float)
    router.fit(differences, labels)
    path = str(tmp_path / "router.json")
    router.save(path)
    loaded = LearnedRouter.load(path)
    assert np.allclose(loaded.coef_, router.coef_)

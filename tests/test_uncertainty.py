"""Tests for the uncertainty decomposition."""

from __future__ import annotations

from aegis.uncertainty import uncertainty_report


def test_channel_disagreement_raises_epistemic():
    report = uncertainty_report(
        {"topology": 1.0, "shape": 0.0},
        {"topology": 0.5, "shape": 0.5},
    )
    assert report.epistemic > 0.4
    assert "high_channel_disagreement" in report.flags
    assert 0.0 < report.total <= 1.0


def test_agreement_keeps_epistemic_low():
    report = uncertainty_report(
        {"topology": 0.8, "shape": 0.8},
        {"topology": 0.5, "shape": 0.5},
    )
    assert report.epistemic < 0.1


def test_ood_component():
    report = uncertainty_report(
        {"topology": 0.5, "shape": 0.5},
        {"topology": 0.5, "shape": 0.5},
        best_neighbor_similarity=0.05,
    )
    assert report.ood > 0.9
    assert "out_of_library_distribution" in report.flags


def test_missing_library_reference_flag():
    report = uncertainty_report(
        {"topology": 0.5, "shape": 0.5},
        {"topology": 0.5, "shape": 0.5},
        best_neighbor_similarity=None,
    )
    assert "no_library_reference" in report.flags


def test_scaffold_hop_signature_flag():
    report = uncertainty_report(
        {"topology": 0.1, "shape": 0.75, "field": 0.75},
        {"topology": 0.34, "shape": 0.33, "field": 0.33},
    )
    assert "scaffold_hop_signature" in report.flags


def test_conformational_component():
    report = uncertainty_report(
        {"topology": 0.5, "shape": 0.5},
        {"topology": 0.5, "shape": 0.5},
        conformer_spread=0.4,
    )
    assert report.conformational == 0.4
    assert "conformational_ambiguity" in report.flags


def test_no_channels():
    report = uncertainty_report(
        {"topology": None, "shape": None},
        {"topology": 0.5, "shape": 0.5},
        missing_channels=["topology", "shape"],
    )
    assert report.total == 1.0
    assert "channel_unavailable:topology" in report.flags

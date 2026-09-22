# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Eight similarity channels: ECFP topology, Bemis-Murcko scaffold hierarchy
  (exact + generalised), USR shape, USRCAT-style interaction-field moments,
  Boltzmann-weighted conformer-ensemble similarity, pocket-conditioned
  interaction fingerprints (Tier 2), physicochemical property profile, and a
  transparent synthetic-complexity proxy.
- Context-conditioned router with curated campaign profiles (general,
  kinase hinge, GPCR lipophilic, ADMET liability), novelty/stage/pocket
  blending, manual overrides, and a `LearnedRouter` fit from preference pairs.
- Uncertainty decomposition (epistemic channel disagreement, conformational
  spread, out-of-domain risk) with machine-readable flags and an explicit
  score penalty `S = sum w_k S_k - lambda * U`.
- Three-tier cascade: Tier-0 union routing (ECFP + PCA student embedding +
  scaffold families), Tier-1 full channel scoring, Tier-2 MCS pose-transfer
  interaction reranking.
- Decision labels and human-readable "why similar" explanations.
- Virtual-screening metrics: BEDROC (validated against the analytic random
  baseline), enrichment factor, AUROC, scaffold-hop recovery, top-k scaffold
  diversity, and calibration by confidence tier.
- CLI (`aegis prepare`, `aegis search`), Streamlit application, demo library
  of 37 compounds, pytest suite, and CI (ruff + pytest on Python 3.10-3.12).

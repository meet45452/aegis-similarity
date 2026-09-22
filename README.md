# 🛡️ AEGIS

**A**daptive **E**nsemble of **G**eometric, **I**nteraction, and **S**caffold similarity — an uncertainty-aware, multi-resolution molecular similarity cascade for drug discovery.

[![CI](https://github.com/meet45452/aegis-similarity/actions/workflows/ci.yml/badge.svg)](https://github.com/meet45452/aegis-similarity/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

AEGIS stops treating similarity as one universal cosine distance between whole-molecule embeddings. Instead of a single fingerprint, it is a **similarity operating system**: a learned, context-conditioned router blends **eight orthogonal notions of resemblance**, escalates physics only where the decision merits it, penalises scores with **decomposed uncertainty**, and returns candidates with a **human-readable "why similar" explanation** and a decision label.

```text
S_AEGIS(q, m, c) = Σ_k w_k(q, c) · S_k(q, m)  −  λ · U(q, m, c)
```

where `c` is the campaign context (target class, novelty requirement, program stage, pocket availability), `w_k` are context-dependent channel weights, and `U` is decomposed predictive uncertainty.

---

## Architecture: the distilled cascade

```text
                 query molecule + campaign context
                              │
        ┌─────────────────────▼──────────────────────┐
        │ TIER 0 · global routing (milliseconds)      │
        │  ECFP neighbours ∪ student-embedding ANN   │
        │  ∪ Murcko scaffold families                │
        └─────────────────────┬──────────────────────┘
                              │  high-recall candidate set
        ┌─────────────────────▼──────────────────────┐
        │ TIER 1 · physics-aware refinement (seconds)│
        │  topology · scaffold hierarchy · USR shape │
        │  interaction fields · Boltzmann conformer  │
        │  ensembles · properties · synthesis        │
        │  + decomposed uncertainty penalty          │
        └─────────────────────┬──────────────────────┘
                              │  ranked shortlist
        ┌─────────────────────▼──────────────────────┐
        │ TIER 2 · pocket-aware reranking (optional) │
        │  MCS pose transfer onto reference pose      │
        │  residue-level interaction fingerprints     │
        └─────────────────────┬──────────────────────┘
                              ▼
      results + labels + uncertainty + "why similar"
```

The system is fast globally because expensive physics is never applied globally, and high quality locally because final decisions never rely on a single lossy embedding.

## The eight similarity channels

| Channel | Method | Key references / notes |
|---|---|---|
| `topology` | ECFP4 Tanimoto | Rogers & Hahn (2010) — fast close-analogue retrieval |
| `scaffold` | Bemis–Murcko hierarchy (exact + atom-type-generalised framework) | Bemis & Murcko (1996) — series vs scaffold-hop reasoning |
| `shape` | USR moments over conformer ensembles (best conformer pair) | Ballester & Richards (2007) |
| `field` | USRCAT-style per-pharmacophore-class moment blocks + charge statistics | Schreyer & Blundell (2012); Gasteiger charges as ESP proxy |
| `conformer` | Boltzmann-weighted, energy-penalised ensemble similarity | soft aggregation of the design's `SoftMax[S_ij − α(E_Ai+E_Bj)]` |
| `interaction` | Pocket-conditioned residue-level interaction fingerprints (Tier 2) | MCS pose transfer; plug in docking poses via `record.raw["pose_mol"]` |
| `property` | Normalised drug-like property space (MW, cLogP, TPSA, HBD/HBA, rotB, charge, Fsp³) | heavily weighted in the ADMET-liability profile |
| `synthesis` | Transparent synthetic-complexity proxy (0–10) | documented heuristic, swappable for a full SA score |

Channels that cannot honestly score a pair (no pocket, no conformers, no shared substructure) **abstain** by returning `None`; the scorer renormalises router weights over the available channels and flags the gap in the uncertainty report.

## Campaign routing

The router maps a campaign context to channel weights. Curated profiles:

- **`general`** — balanced retrieval.
- **`kinase_hinge`** — donor–acceptor geometry and interaction fields dominate.
- **`gpcr_lipophilic`** — 3D hydrophobic shape and conformational accessibility dominate.
- **`admet_liability`** — the developability envelope dominates.

The **novelty requirement** (0 = closest analogues, 1 = maximal scaffold hopping) shifts weight from topology/scaffold toward shape, field, conformer, and interaction experts. `LearnedRouter` fits weights from labelled preference pairs (logistic regression on channel-score differences) so routing can follow your project's actual assay history instead of hand-set profiles.

## Uncertainty and decision labels

Every score ships with a decomposition:

- **epistemic** — disagreement between active similarity channels,
- **ood** — distance of the query from the library distribution,
- **conformational** — spread of the ensemble pair-score distribution,

plus machine-readable flags (`high_channel_disagreement`, `scaffold_hop_signature`, `out_of_library_distribution`, `channel_unavailable:*`, …). Results are labelled:

| Label | Meaning |
|---|---|
| `high_confidence_analogue` | close analogue, low uncertainty |
| `high_confidence_scaffold_hop` | different Murcko framework, matching 3D/field profile |
| `high_upside_uncertain` | scaffold hop worth an assay precisely because it is uncertain |
| `novelty_candidate` | topology-distant, retained for IP-distance exploration |
| `low_confidence` | below thresholds; inspect the channel breakdown |

## Installation

```bash
pip install -e .            # core library (numpy, pandas, rdkit)
pip install -e ".[app]"     # + streamlit
pip install -e ".[dev]"     # + pytest, ruff
```

## Quickstart

**Python**

```python
from aegis import AegisArray, CampaignContext, Library, Profile

library = Library.from_csv("examples/data/demo_library.csv")
array = AegisArray(library)

result = array.search(
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    CampaignContext(profile=Profile.KINASE_HINGE, novelty_requirement=0.6),
    top_k=10,
)
print(result.summary())
print(result.to_dataframe())
for candidate in result.results[:3]:
    print(candidate.label, "-", candidate.explanation)
```

**CLI**

```bash
aegis prepare --input library.csv --output lib.aegis
aegis search --library lib.aegis --query "CC(=O)Oc1ccccc1C(=O)O" \
    --profile kinase_hinge --novelty 0.6 --top-k 25 --out results.csv
```

**Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

The app bundles a 37-compound demo library and adds result tables, per-candidate explanation panels with channel-contribution charts, structure grids, routing-weight visualisation, and optional Tier-2 uploads (binding-site PDB + posed query ligand).

## Benchmarking honestly

`aegis.metrics` ships the metrics that reflect DMTA value rather than broad retrospective ranking:

- `bedroc` — Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007), normalised to [0, 1]; the test suite verifies perfect/reversed extremes **and** that empirical random rankings match the analytic baseline.
- `enrichment_factor` — EF at a fixed top fraction.
- `auroc` — global rank-quality baseline.
- `scaffold_hop_recovery` — active scaffold hops recovered in the top-k (the novel-active-yield metric).
- `topk_scaffold_diversity` — unique generalised frameworks in the top-k.
- `calibration_by_tier` — does a claimed confidence tier actually succeed more often than a lower tier?

The business metric for a CRO should be *validated novel active scaffold families per screened compound and per dollar* — not retrospective AUROC.

## Project layout

```text
src/aegis/
  chemistry.py       # standardisation, Murcko hierarchy, ETKDG ensembles, USR/field features
  channels/          # eight similarity experts (incl. pocket-conditioned interaction channel)
  router.py          # context-conditioned + learned routing
  uncertainty.py     # epistemic / conformational / OOD decomposition
  student.py         # Tier-0 PCA student embedding
  index.py           # Tier-0 union index
  library.py         # offline preparation and persistence
  scoring.py         # weighted channel combination with uncertainty penalty
  cascade.py         # three-tier search orchestration
  explain.py         # "why similar" narratives
  metrics.py         # BEDROC, EF, AUROC, scaffold-hop recovery, calibration
  cli.py             # aegis prepare / search
app/streamlit_app.py # UI
tests/               # pytest suite (unit + end-to-end, CI-verified)
examples/            # runnable demo + 37-compound library CSV
```

## Scientific honesty / limitations

AEGIS is a research-grade MVP, and the code documents its approximations:

1. **Gasteiger charges** approximate the electrostatic potential; attach higher-quality charges for production electrostatics.
2. **Field features are moment summaries**, not dense 3D grids — compact enough for cascade use, less expressive than true field overlap.
3. **Tier-2 pose transfer uses MCS alignment**, which covers analogues and hops that retain a common core. Full scaffold hops without a shared substructure require an external pose generator (docking); pre-posed molecules are accepted via `record.raw["pose_mol"]`.
4. **The synthesis channel is a complexity proxy**, not the Ertl SA score.
5. The **student embedding is PCA-compressed 2D features**; teacher-score distillation is a documented roadmap item once labelled campaign pairs accumulate.
6. Conformer ensembles use ETKDGv3 + MMFF (UFF fallback) with fixed seeds; force-field quality bounds the ensemble realism.
7. Interaction fingerprints are distance-based, without H-bond angle filtering, induced fit, or solvation — treat them as a reranking signal, not a binding-energy estimate.

## Roadmap

- Teacher-score distillation into the Tier-0 student embedding (train the projection so student similarity mimics the full cascade).
- ANN backends (faiss/HNSW) and quantised vectors for billion-scale Tier 0.
- SE(3)-equivariant interaction-field encodings from atom-centred bases.
- Scaffold-transformation graphs for systematic bioisostere navigation (quinazoline → pyrimidine-style transformations).
- Quantum-response features (Fukui proxies, polarizability, H-bond acidity/basicity) trained on higher-quality electronic labels.
- Closed-loop DMTA integration: assay outcomes feed `LearnedRouter` and calibration checks.

## References

- Bemis, G. W. & Murcko, M. A. (1996). *J. Med. Chem.* 39, 2887–2893 — Bemis–Murcko scaffolds.
- Rogers, D. & Hahn, M. (2010). *J. Chem. Inf. Model.* 50, 742–754 — ECFP.
- Ballester, P. J. & Richards, W. G. (2007). *J. Comput. Chem.* 28, 1711–1723 — USR.
- Schreyer, A. M. & Blundell, T. (2012). *J. Cheminform.* 4, 27 — USRCAT.
- Truchon, J.-F. & Bayly, C. I. (2007). *J. Chem. Inf. Model.* 47, 488–508 — RIE/BEDROC.
- Ertl, P. & Schuffenhauer, A. (2009). *J. Cheminform.* 1, 8 — synthetic accessibility.
- Riniker, S. & Landrum, G. A. (2015). *J. Chem. Inf. Model.* 55, 2562–2574 — ETKDG.
- Landrum, G. A. et al. — RDKit (https://www.rdkit.org).

## License

Apache-2.0 — see [LICENSE](LICENSE).

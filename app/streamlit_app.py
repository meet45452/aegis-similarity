"""AEGIS Streamlit application.

Run locally::

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

The UI walks the same three-tier cascade as the CLI: configure the campaign
context in the sidebar, run the cascade, and inspect results, per-candidate
"why similar" explanations, channel contributions, and routing weights.
"""

from __future__ import annotations

import io
import os
import tempfile

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from aegis.cascade import AegisArray
from aegis.channels.interaction import Pocket
from aegis.config import AegisConfig
from aegis.library import Library
from aegis.router import CampaignContext, Profile

DEMO_LIBRARY: list[tuple[str, str, str]] = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", "NSAID"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "NSAID"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "NSAID"),
    ("ketoprofen", "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1", "NSAID"),
    ("flurbiprofen", "CC(C(=O)O)c1ccccc1-c1cccc(F)c1", "NSAID"),
    ("diclofenac", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "NSAID"),
    ("mefenamic acid", "Cc1cccc(C)c1Nc1ccccc1C(=O)O", "NSAID"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", "analgesic"),
    ("phenacetin", "CCOc1ccc(NC(C)=O)cc1", "analgesic"),
    ("salicylic acid", "OC(=O)c1ccccc1O", "salicylate"),
    ("benzoic acid", "OC(=O)c1ccccc1", "aromatic acid"),
    ("4-aminobenzoic acid", "Nc1ccc(C(=O)O)cc1", "aromatic acid"),
    ("benzamide", "NC(=O)c1ccccc1", "aromatic amide"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", "beta-blocker"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "beta-blocker"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "anticoagulant"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2ccccc21", "benzodiazepine"),
    ("chloroquine", "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12", "antimalarial"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "xanthine"),
    ("nicotine", "CN1CCCC1c1cccnc1", "stimulant"),
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "SSRI"),
    ("diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1", "antihistamine"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C", "local anaesthetic"),
    ("procainamide", "CCN(CC)CC(=O)Nc1ccc(N)cc1", "antiarrhythmic"),
    ("benzocaine", "CCOC(=O)c1ccc(N)cc1", "local anaesthetic"),
    ("sulfanilamide", "Nc1ccc(S(=O)(=O)N)cc1", "sulfonamide"),
    ("sulfamethoxazole", "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", "sulfonamide"),
    ("dapsone", "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "sulfonamide"),
    ("metformin", "CN(C)C(=N)N=C(N)N", "biguanide"),
    ("nifedipine", "COC(=O)C1=C(C)NC(C)=C(C1c2ccccc2[N+](=O)[O-])C(=O)OC", "CCB"),
    ("quinoline", "c1ccc2ncccc2c1", "heteroaromatic"),
    ("isoquinoline", "c1ccc2cnccc2c1", "heteroaromatic"),
    ("indole", "c1ccc2[nH]ccc2c1", "heteroaromatic"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1", "heteroaromatic"),
    ("adenine", "Nc1ncnc2[nH]cnc12", "nucleobase"),
    ("4-aminoquinazoline", "Nc1ncnc2ccccc12", "kinase hinge motif"),
    ("2-aminopyrimidine", "Nc1ncccn1", "kinase hinge motif"),
]

PROFILE_LABELS = {
    "general": "General (balanced)",
    "kinase_hinge": "Kinase hinge binder",
    "gpcr_lipophilic": "GPCR lipophilic pocket",
    "admet_liability": "ADMET liability work",
}

LABEL_BADGES = {
    "high_confidence_analogue": "🟢 analogue",
    "high_confidence_scaffold_hop": "🔵 scaffold hop",
    "high_upside_uncertain": "🟠 uncertain hop",
    "novelty_candidate": "🟣 novelty",
    "low_confidence": "⚪ low confidence",
}

st.set_page_config(page_title="AEGIS similarity", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Preparing demo library (conformers, fields, embeddings)...")
def _build_demo() -> Library:
    config = AegisConfig(n_conformers=6, keep_conformers=3)
    return Library.build(
        [row[1] for row in DEMO_LIBRARY],
        names=[row[0] for row in DEMO_LIBRARY],
        config=config,
        with_conformers=True,
    )


@st.cache_resource(show_spinner="Preparing uploaded library...")
def _build_uploaded(content: bytes) -> Library:
    frame = pd.read_csv(io.BytesIO(content))
    smiles_column = "smiles" if "smiles" in frame.columns else frame.columns[0]
    if "name" in frame.columns:
        names = frame["name"].astype(str).tolist()
    else:
        names = [f"mol_{i}" for i in range(len(frame))]
    config = AegisConfig(n_conformers=6, keep_conformers=3)
    return Library.build(
        frame[smiles_column].astype(str).tolist(),
        names=names,
        config=config,
        with_conformers=True,
    )


def _pocket_from_upload(uploaded) -> Pocket | None:
    content = uploaded.read().decode("utf-8", errors="ignore")
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        return Pocket.from_pdb(path)
    finally:
        os.unlink(path)


def _reference_from_upload(uploaded):
    suffix = uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile("wb", suffix=f".{suffix}", delete=False) as handle:
        handle.write(uploaded.getvalue())
        path = handle.name
    try:
        if suffix == "sdf":
            supplier = Chem.SDMolSupplier(path, removeHs=False)
            return next(iter(supplier), None)
        return Chem.MolFromMolFile(path, removeHs=False)
    finally:
        os.unlink(path)


def main() -> None:
    with st.sidebar:
        st.title("🛡️ AEGIS")
        st.caption("Adaptive Ensemble of Geometric, Interaction, and Scaffold similarity")

        source = st.selectbox("Library", ["Demo library (37 compounds)", "Upload CSV"])
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader("CSV with `smiles` (and optional `name`) columns", type=["csv"])

        st.divider()
        query_smiles = st.text_input("Query SMILES", value="CC(=O)Oc1ccccc1C(=O)O")
        demo_names = {row[0]: row[1] for row in DEMO_LIBRARY}
        pick = st.selectbox("...or pick a demo compound", ["(custom)"] + list(demo_names))
        if pick != "(custom)":
            query_smiles = demo_names[pick]

        st.divider()
        profile = st.selectbox(
            "Campaign profile",
            list(PROFILE_LABELS),
            format_func=lambda value: PROFILE_LABELS[value],
        )
        novelty = st.slider("Novelty requirement", 0.0, 1.0, 0.3, 0.05)
        stage = st.radio("Program stage", ["discovery", "optimization"], horizontal=True)
        top_k = st.slider("Results (top-k)", 5, 50, 20)

        with st.expander("Tier 2: pocket-aware reranking (optional)"):
            st.caption("Provide a binding-site PDB and the posed query ligand to enable interaction-fingerprint reranking.")
            pocket_upload = st.file_uploader("Binding-site PDB", type=["pdb"], key="pocket")
            pose_upload = st.file_uploader("Posed query ligand (SDF/MOL)", type=["sdf", "mol"], key="pose")

        run = st.button("Run cascade", type="primary", use_container_width=True)

    if run:
        try:
            if source == "Upload CSV":
                if uploaded is None:
                    st.error("Upload a CSV first.")
                    st.stop()
                library = _build_uploaded(uploaded.getvalue())
            else:
                library = _build_demo()

            pocket = _pocket_from_upload(pocket_upload) if pocket_upload else None
            reference = _reference_from_upload(pose_upload) if pose_upload else None
            if pocket is not None and (reference is None or reference.GetNumConformers() == 0):
                st.warning("Pocket ignored: the reference ligand must be a 3D SDF/MOL with a conformer.")
                pocket, reference = None, None

            context = CampaignContext(
                profile=Profile(profile),
                novelty_requirement=novelty,
                stage=stage,
                pocket_available=pocket is not None,
            )
            array = AegisArray(library)
            result = array.search(
                query_smiles,
                context=context,
                top_k=top_k,
                pocket=pocket,
                reference_pose=reference,
            )
            st.session_state["aegis_result"] = result
        except ValueError as exc:
            st.error(f"{exc}")

    result = st.session_state.get("aegis_result")
    if result is None:
        st.info("Configure the campaign on the left and press **Run cascade**.")
        st.stop()

    st.title("AEGIS results")
    total_time = sum(result.timings.values())
    columns = st.columns(4)
    columns[0].metric("Candidates routed (Tier 0)", result.tier_counts.get("tier0_candidates", 0))
    columns[1].metric("Scored (Tier 1)", result.tier_counts.get("tier1_scored", 0))
    columns[2].metric("Returned", len(result.results))
    columns[3].metric("Cascade time", f"{total_time:.2f} s")

    frame = result.to_dataframe()
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame["label"] = frame["label"].map(lambda value: LABEL_BADGES.get(value, value))

    tab_results, tab_why, tab_structures, tab_routing, tab_about = st.tabs(
        ["Results", "Why similar", "Structures", "Routing & uncertainty", "About"]
    )

    with tab_results:
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn("AEGIS score", min_value=0.0, max_value=1.0, format="%.3f"),
                "uncertainty": st.column_config.ProgressColumn("Uncertainty", min_value=0.0, max_value=1.0, format="%.3f"),
                "smiles": st.column_config.TextColumn("SMILES", width="medium"),
            },
        )
        st.download_button(
            "Download results (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            "aegis_results.csv",
            "text/csv",
        )
        label_counts = frame["label"].value_counts()
        st.bar_chart(label_counts)

    with tab_why:
        if not result.results:
            st.info("No results to explain.")
        else:
            names = [candidate.record.name for candidate in result.results]
            choice = st.selectbox("Candidate", names)
            candidate = next(c for c in result.results if c.record.name == choice)
            left, right = st.columns([1, 2])
            with left:
                if candidate.record.mol is not None:
                    st.image(Draw.MolToImage(candidate.record.mol, size=(280, 200)))
                st.metric("AEGIS score", f"{candidate.score:.3f}")
                st.metric("Uncertainty", f"{candidate.uncertainty.total:.3f}")
                st.caption(LABEL_BADGES.get(candidate.label, candidate.label))
            with right:
                st.markdown(candidate.explanation)
                contributions = {
                    channel_score.channel: channel_score.contribution
                    for channel_score in candidate.channel_scores
                    if channel_score.contribution is not None
                }
                st.bar_chart(pd.Series(contributions).sort_values(ascending=True))
                uncertainty_frame = pd.DataFrame(
                    {
                        "component": ["epistemic", "conformational", "out-of-domain", "total"],
                        "value": [
                            candidate.uncertainty.epistemic,
                            candidate.uncertainty.conformational,
                            candidate.uncertainty.ood,
                            candidate.uncertainty.total,
                        ],
                    }
                )
                st.dataframe(uncertainty_frame, use_container_width=True, hide_index=True)

    with tab_structures:
        top = result.results[:12]
        if top:
            image = Draw.MolsToGridImage(
                [candidate.record.mol for candidate in top if candidate.record.mol is not None],
                legends=[
                    f"{i + 1}. {candidate.record.name} ({LABEL_BADGES.get(candidate.label, candidate.label)})"
                    for i, candidate in enumerate(top)
                ],
                molsPerRow=4,
                subImgSize=(320, 220),
            )
            st.image(image)

    with tab_routing:
        from aegis.router import Router

        weights = Router().weights(result.context)
        st.subheader("Channel weights for this campaign context")
        st.bar_chart(pd.Series(weights).sort_values(ascending=True))
        st.caption(
            f"Context: profile={result.context.profile_name()}, "
            f"novelty={result.context.novelty_requirement:.2f}, "
            f"stage={result.context.stage}, pocket={result.context.pocket_available}"
        )
        st.subheader("Uncertainty flags across results")
        flag_counts: dict[str, int] = {}
        for candidate in result.results:
            for flag in candidate.uncertainty.flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if flag_counts:
            st.dataframe(
                pd.DataFrame({"flag": list(flag_counts), "count": list(flag_counts.values())}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No uncertainty flags raised.")

    with tab_about:
        st.markdown(
            """
### How this search worked

1. **Tier 0 - global routing.** The union of ECFP neighbours, student-embedding
   neighbours, and Murcko scaffold-family members builds a high-recall
   candidate set. No physics is applied at this stage.
2. **Tier 1 - physics-aware refinement.** Candidates are scored on eight
   orthogonal channels (topology, scaffold hierarchy, USR shape, interaction
   fields, Boltzmann conformer ensemble, pocket interactions, properties,
   synthetic complexity), weighted by the campaign context, and penalised by
   decomposed uncertainty.
3. **Tier 2 - pocket-aware reranking.** When a binding site and a posed
   reference ligand are supplied, finalists are reranked by pocket-conditioned
   interaction fingerprints.

Every result carries a decision label, a channel-level contribution
breakdown, and an uncertainty decomposition, so scores are auditable rather
than opaque. Approximations (Gasteiger charges, USRCAT-style moment fields,
MCS pose transfer, a synthetic-complexity proxy) are documented in the README.
"""
        )


if __name__ == "__main__":
    main()

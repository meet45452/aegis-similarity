"""End-to-end AEGIS example: prepare a library and run a scaffold-hop search.

Run from the repository root::

    python examples/prepare_and_search.py
"""

from __future__ import annotations

from pathlib import Path

from aegis.cascade import AegisArray
from aegis.library import Library
from aegis.router import CampaignContext, Profile

DEMO_CSV = Path(__file__).parent / "data" / "demo_library.csv"
QUERY = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin


def main() -> None:
    library = Library.from_csv(DEMO_CSV)
    print("Library:", library.describe())

    array = AegisArray(library)

    # A kinase-style, novelty-seeking context: shift weight from topology
    # toward interaction fields, shape, and conformer ensembles.
    context = CampaignContext(
        profile=Profile.KINASE_HINGE,
        novelty_requirement=0.6,
        stage="discovery",
    )
    result = array.search(QUERY, context=context, top_k=10)

    print("Summary:", result.summary())
    print(result.to_dataframe().to_string(index=False))

    print("\nTop explanations:")
    for candidate in result.results[:3]:
        print(f"\n[{candidate.record.name}] {candidate.label}")
        print(candidate.explanation)


if __name__ == "__main__":
    main()

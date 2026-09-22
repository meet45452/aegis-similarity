"""Command-line interface: ``aegis prepare`` and ``aegis search``.

Examples
--------

Prepare a library::

    aegis prepare --input library.csv --output lib.aegis --smiles-col smiles

Search with a kinase-hinge profile favouring scaffold hops::

    aegis search --library lib.aegis --query "CC(=O)Oc1ccccc1C(=O)O" \\
        --profile kinase_hinge --novelty 0.6 --top-k 25 --out results.csv

Tier-2 pocket-aware search (pocket PDB + posed query ligand)::

    aegis search --library lib.aegis --query SMILES --pocket pocket.pdb \\
        --reference query_pose.sdf --out results.csv
"""

from __future__ import annotations

import argparse
import sys

from rdkit import Chem

from aegis.cascade import AegisArray
from aegis.channels.interaction import Pocket
from aegis.config import AegisConfig
from aegis.library import Library
from aegis.router import CampaignContext, Profile


def _cmd_prepare(args: argparse.Namespace) -> int:
    config = AegisConfig(
        n_conformers=args.n_confs,
        keep_conformers=min(args.n_confs, 4),
    )

    def progress(done: int, total: int, name: str) -> None:
        if done % 10 == 0 or done == total:
            print(f"[{done}/{total}] {name}", end="\r", flush=True)

    library = Library.from_csv(
        args.input,
        smiles_col=args.smiles_col,
        name_col=args.name_col,
        config=config,
        with_conformers=not args.no_conformers,
        progress=progress,
    )
    library.save(args.output)
    print(f"\nSaved {len(library)} compounds to {args.output}")
    print(library.describe())
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    library = Library.load(args.library)
    pocket = Pocket.from_pdb(args.pocket) if args.pocket else None
    reference = None
    if args.reference:
        if args.reference.lower().endswith(".sdf"):
            supplier = Chem.SDMolSupplier(args.reference, removeHs=False)
            reference = next(iter(supplier), None)
        else:
            reference = Chem.MolFromMolFile(args.reference, removeHs=False)
        if reference is None or reference.GetNumConformers() == 0:
            print("failed to read reference pose (needs a 3D conformer)", file=sys.stderr)
            return 1
    context = CampaignContext(
        profile=Profile(args.profile),
        novelty_requirement=args.novelty,
        stage=args.stage,
        pocket_available=pocket is not None,
    )
    array = AegisArray(library)
    result = array.search(
        args.query,
        context=context,
        top_k=args.top_k,
        pocket=pocket,
        reference_pose=reference,
    )
    frame = result.to_dataframe()
    print(result.summary())
    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"Wrote {len(frame)} rows to {args.out}")
    else:
        print(frame.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="AEGIS: adaptive ensemble of geometric, interaction, and scaffold similarity",
    )
    parser.add_argument("--version", action="store_true", help="print the version")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare a library from a SMILES CSV")
    prepare.add_argument("--input", required=True, help="input CSV path")
    prepare.add_argument("--output", required=True, help="output .aegis pickle path")
    prepare.add_argument("--smiles-col", default="smiles")
    prepare.add_argument("--name-col", default="name")
    prepare.add_argument("--n-confs", type=int, default=8)
    prepare.add_argument("--no-conformers", action="store_true")
    prepare.set_defaults(func=_cmd_prepare)

    search = subparsers.add_parser("search", help="search a prepared library")
    search.add_argument("--library", required=True, help=".aegis pickle path")
    search.add_argument("--query", required=True, help="query SMILES")
    search.add_argument("--profile", default="general", choices=[p.value for p in Profile])
    search.add_argument("--novelty", type=float, default=0.3)
    search.add_argument("--stage", choices=["discovery", "optimization"], default="discovery")
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--pocket", default=None, help="pocket PDB for Tier-2 reranking")
    search.add_argument("--reference", default=None, help="posed query ligand (.mol/.sdf)")
    search.add_argument("--out", default=None, help="output CSV path")
    search.set_defaults(func=_cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from aegis import __version__

        print(__version__)
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

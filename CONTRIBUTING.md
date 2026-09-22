# Contributing to AEGIS

Thank you for considering a contribution. This project sits at the
intersection of software engineering and medicinal chemistry, so both code
quality and scientific validity are reviewed.

## Getting started

```bash
git clone https://github.com/meet45452/aegis-similarity.git
cd aegis-similarity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development workflow

1. Create a feature branch from `main`.
2. Make your change with tests.
3. Run the full local gate:

```bash
ruff check src tests app examples
pytest --cov=aegis
```

4. Open a pull request. CI must pass on all Python versions (3.10-3.12).

## Scientific contribution guidelines

- **Document approximations.** If a channel or feature approximates physics
  (charges, pose transfer, complexity scores), say so in the docstring and
  README rather than implying first-principles accuracy.
- **Validate numerics against analytics or published values.** New metrics
  need tests that check extremes (perfect/reversed rankings) and, where one
  exists, the analytic random baseline.
- **Benchmark with decision-quality metrics** (BEDROC, EF@fraction,
  scaffold-hop recovery, calibration), not only mean AUROC.
- **Keep channels orthogonal and abstaining.** A channel that cannot honestly
  score a pair must return `None`, not a fabricated number.
- **Reproducibility first.** Fixed random seeds, no hidden state, config
  captured in results.

## Code standards

- Type hints throughout; the package ships `py.typed`.
- Docstrings on every public module, class, and function.
- Tests for every new behaviour; bug fixes land with a regression test.
- `ruff` (E, F rules) must pass; line length 100.

## Reporting issues

Please include: the RDKit and Python versions, a minimal reproducible example
(SMILES strings are ideal), and what you expected versus what happened.

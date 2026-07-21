# ML Model Benchmark

A compact, reproducible machine-learning project that compares logistic
regression and random forest classifiers on scikit-learn's breast-cancer
dataset.

## Engineering features

- deterministic train/test split
- preprocessing inside a scikit-learn pipeline
- multiple evaluation metrics
- automatic best-model selection
- persisted JSON metrics and trained model
- command-line interface
- automated tests
- GitHub Actions continuous integration

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
ml-benchmark --output-dir artifacts
cat artifacts/metrics.json
```

Alternatively:

```bash
python -m ml_benchmark.train --output-dir artifacts
```

## Test and lint

```bash
pytest
ruff check .
```

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── src/ml_benchmark/
│   ├── __init__.py
│   └── train.py
├── tests/test_train.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Notes

This repository is intended as a clean portfolio example. The bundled dataset
is suitable for demonstrating a reproducible ML workflow, not for clinical use.

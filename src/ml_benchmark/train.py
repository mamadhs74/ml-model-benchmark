from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the breast-cancer classification dataset as pandas objects."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def build_models() -> dict[str, Any]:
    """Return candidate models with reproducible configuration."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate common binary-classification metrics."""
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def run_training(output_dir: Path) -> dict[str, dict[str, float]]:
    """Train candidate models, persist metrics, and save the best model."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in build_models().items():
        LOGGER.info("Training %s", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)

    best_name = max(results, key=lambda name: results[name]["f1"])
    joblib.dump(fitted_models[best_name], output_dir / "best_model.joblib")

    report = {
        "best_model": best_name,
        "selection_metric": "f1",
        "models": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Best model: %s", best_name)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark reproducible classification models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for metrics and trained-model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run_training(args.output_dir)


if __name__ == "__main__":
    main()

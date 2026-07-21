from pathlib import Path

from ml_benchmark.train import build_models, load_dataset, run_training


def test_dataset_is_not_empty() -> None:
    X, y = load_dataset()
    assert len(X) == len(y)
    assert len(X) > 500
    assert X.shape[1] == 30


def test_expected_models_are_available() -> None:
    assert set(build_models()) == {"logistic_regression", "random_forest"}


def test_training_creates_artifacts(tmp_path: Path) -> None:
    results = run_training(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "best_model.joblib").exists()
    assert all(metrics["f1"] > 0.8 for metrics in results.values())

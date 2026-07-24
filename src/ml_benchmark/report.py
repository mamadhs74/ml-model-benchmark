from __future__ import annotations
from .utils import validate_metrics
import csv
import json
from pathlib import Path


def load_metrics(metrics_path: str = "artifacts/metrics.json") -> dict:
    path = Path(metrics_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return json.loads(path.read_text(encoding="utf-8"))


def print_report(metrics_path: str = "artifacts/metrics.json") -> None:
    data = load_metrics(metrics_path)
    validate_metrics(data)
    print("\nModel Benchmark Results")
    print("=" * 40)

    for model, metrics in data["models"].items():
        print(f"\n{model}")
        for metric, value in metrics.items():
            print(f"  {metric:<10}: {value:.4f}")

    print(f"\nBest model: {data['best_model']}")


def export_csv(
    metrics_path: str = "artifacts/metrics.json",
    output_path: str = "artifacts/results.csv",
) -> None:
    data = load_metrics(metrics_path)
    validate_metrics(data)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["model", "accuracy", "precision", "recall", "f1"])

        for model, metrics in data["models"].items():
            writer.writerow(
                [
                    model,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                ]
            )


if __name__ == "__main__":
    print_report()
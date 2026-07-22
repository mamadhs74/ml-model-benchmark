from __future__ import annotations

import json
from pathlib import Path


def print_report(metrics_path: str = "artifacts/metrics.json") -> None:
    path = Path(metrics_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    data = json.loads(path.read_text())

    print("\nModel Benchmark Results")
    print("=" * 40)

    for model, metrics in data["models"].items():
        print(f"\n{model}")
        for metric, value in metrics.items():
            print(f"  {metric:<10}: {value:.4f}")

    print(f"\nBest model: {data['best_model']}")


if __name__ == "__main__":
    print_report()

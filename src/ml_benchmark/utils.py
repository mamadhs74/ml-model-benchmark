from __future__ import annotations


REQUIRED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
)


def validate_metrics(data: dict) -> None:
    if "models" not in data:
        raise KeyError("Missing 'models' section.")

    if "best_model" not in data:
        raise KeyError("Missing 'best_model'.")

    for model, metrics in data["models"].items():
        missing = [
            metric
            for metric in REQUIRED_METRICS
            if metric not in metrics
        ]

        if missing:
            raise ValueError(
                f"{model} is missing metrics: {', '.join(missing)}"
            )
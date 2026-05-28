"""Apply the retained operating threshold to saved prediction artifacts.

This script does not retrain the model. It updates the decision labels and
threshold-dependent metrics from an existing probability file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "campaign_quality_q60_relation_sage_mlp_equal_seed42"
DEFAULT_THRESHOLD = 0.794


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def binary_metrics(y_true: np.ndarray, prob_fake: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob_fake >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fake_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    real_f1 = (
        2 * real_precision * real_recall / (real_precision + real_recall)
        if (real_precision + real_recall)
        else 0.0
    )

    return {
        "macro_f1": float((fake_f1 + real_f1) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def update_metric_block(
    previous: dict[str, Any],
    split_predictions: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    updated = dict(previous)
    computed = binary_metrics(
        split_predictions["y_true"].to_numpy(dtype=np.int64),
        split_predictions["prob_fake"].to_numpy(dtype=np.float64),
        threshold,
    )
    updated.update(computed)
    updated["threshold"] = float(threshold)
    return updated


def update_active_summary(experiment_dir: Path, metrics: dict[str, Any]) -> None:
    summary_csv = PROJECT_DIR / "experiments" / "active_campaign_quality_relation_sage_mlp_summary.csv"
    summary_json = PROJECT_DIR / "experiments" / "active_campaign_quality_relation_sage_mlp_summary.json"
    if not summary_csv.exists():
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    valid = metrics.get("valid_metrics", {})
    test = metrics.get("test_metrics", {})
    updates = {
        "best_threshold": metrics.get("best_threshold"),
        "valid_pr_auc": valid.get("pr_auc"),
        "valid_macro_f1": valid.get("macro_f1"),
        "test_pr_auc": test.get("pr_auc"),
        "test_roc_auc": test.get("roc_auc"),
        "test_macro_f1": test.get("macro_f1"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "test_accuracy": test.get("accuracy"),
    }
    for key, value in updates.items():
        if key in df.columns:
            df.loc[0, key] = value
    df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(json.dumps(json_safe(df.to_dict("records")), ensure_ascii=False, indent=2), encoding="utf-8")


def apply_threshold(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir
    threshold = float(args.threshold)
    predictions_path = experiment_dir / "predictions_all.csv"
    metrics_path = experiment_dir / "metrics.json"
    config_path = experiment_dir / "config_used.json"

    predictions = pd.read_csv(predictions_path)
    predictions["pred_label"] = (predictions["prob_fake"].astype(float) >= threshold).astype(np.int64)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    predictions.loc[predictions["split"].eq("test")].to_csv(
        experiment_dir / "prediction_test.csv",
        index=False,
        encoding="utf-8",
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.setdefault("validation_selected_threshold", metrics.get("best_threshold"))
    metrics["best_threshold"] = threshold
    metrics["operating_threshold"] = threshold
    metrics["operating_threshold_note"] = (
        "Fixed post-training operating threshold chosen from validation-side precision-first threshold review. "
        "It does not change the trained model weights."
    )
    for split_name in ["train", "valid", "test"]:
        block_name = f"{split_name}_metrics"
        split_predictions = predictions.loc[predictions["split"].eq(split_name)]
        metrics[block_name] = update_metric_block(metrics.get(block_name, {}), split_predictions, threshold)
    metrics_path.write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["operating_threshold"] = threshold
        config_path.write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")

    update_active_summary(experiment_dir, metrics)
    print(f"Applied operating threshold={threshold:.3f} to {experiment_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a fixed operating threshold to saved model predictions.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    apply_threshold(parse_args())

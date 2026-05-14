"""
Run the active topk15 window60 SAGE model only.

Direction-aware topk15 resources are intentionally kept in the workspace as a
future candidate, but they are not part of the default execution path.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
TRAIN_SCRIPT = PROJECT_DIR / "TrainGNN.py"
ACTIVE_EXPERIMENT = {
    "name": "relflag_edge_t15_w60_thr075_sage_inverse_seed42",
    "graph": PROJECT_DIR / "data" / "graph_relflag_edge_t15_w60_thr075" / "graph_rur_custom2.pt",
    "note": "active topk15 window60 SAGE model",
}

COMMON_ARGS = [
    "--model",
    "sage",
    "--hidden-dim",
    "128",
    "--num-layers",
    "2",
    "--dropout",
    "0.5",
    "--lr",
    "0.001",
    "--weight-decay",
    "0.0001",
    "--epochs",
    "200",
    "--patience",
    "30",
    "--seed",
    "42",
    "--class-weight",
    "--mask-mode",
    "split",
    "--early-stop-metric",
    "valid_pr_auc",
    "--threshold-strategy",
    "prevalence_constrained_macro_f1",
]

SUMMARY_COLUMNS = [
    "experiment",
    "status",
    "graph_path",
    "best_epoch",
    "best_threshold",
    "valid_pr_auc",
    "valid_macro_f1",
    "test_pr_auc",
    "test_roc_auc",
    "test_macro_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
    "note",
]


def log(message: str) -> None:
    print(f"[RunExperiments] {message}")


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def command_for_experiment(output_dir: Path, python_exe: str, device: str) -> list[str]:
    return [
        python_exe,
        str(TRAIN_SCRIPT),
        "--graph-path",
        str(ACTIVE_EXPERIMENT["graph"]),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        *COMMON_ARGS,
    ]


def read_metrics_row(output_dir: Path, status: str) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    row = {
        "experiment": ACTIVE_EXPERIMENT["name"],
        "status": status,
        "graph_path": path_for_summary(ACTIVE_EXPERIMENT["graph"]),
        "note": ACTIVE_EXPERIMENT["note"],
    }
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        valid = metrics.get("valid_metrics", {})
        test = metrics.get("test_metrics", {})
        row.update(
            {
                "best_epoch": metrics.get("best_epoch"),
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
        )
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, None)
    return row


def save_summary(row: dict[str, Any]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])[SUMMARY_COLUMNS]
    df.to_csv(EXPERIMENTS_DIR / "active_sage_model_summary.csv", index=False, encoding="utf-8")
    (EXPERIMENTS_DIR / "active_sage_model_summary.json").write_text(
        json.dumps(json_safe(df.to_dict("records")), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output_dir = EXPERIMENTS_DIR / ACTIVE_EXPERIMENT["name"]
    metrics_path = output_dir / "metrics.json"
    if not ACTIVE_EXPERIMENT["graph"].exists():
        log(f"Missing graph: {path_for_summary(ACTIVE_EXPERIMENT['graph'])}")
        save_summary(read_metrics_row(output_dir, "missing_graph"))
        return
    if metrics_path.exists() and not args.force:
        log(f"Skipping existing active model result: {ACTIVE_EXPERIMENT['name']}")
        save_summary(read_metrics_row(output_dir, "skipped_existing"))
        return
    command = command_for_experiment(output_dir, args.python, args.device)
    if args.dry_run:
        log("[dry-run] " + subprocess.list2cmdline(command))
        save_summary(read_metrics_row(output_dir, "dry_run"))
        return
    log(f"Running active model: {ACTIVE_EXPERIMENT['name']}")
    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subprocess_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "subprocess_stderr.log").write_text(result.stderr, encoding="utf-8")
    save_summary(read_metrics_row(output_dir, "completed" if result.returncode == 0 else "failed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the active topk15 window60 SAGE model only.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run TrainGNN.py")
    parser.add_argument("--device", default="xpu", help="Device passed to TrainGNN.py")
    parser.add_argument("--force", action="store_true", help="Retrain even when metrics.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

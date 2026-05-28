"""Build governance-layer artifacts for the active YelpZip GNN model.

The governance layer is intentionally separate from review/campaign
visualization layers. It stores model identity, performance, threshold policy,
data lineage, artifact registry, and Streamlit access guidance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "governance_layer"
DEFAULT_EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "campaign_quality_q60_relation_sage_mlp_equal_seed42"
DEFAULT_GRAPH_DIR = PROJECT_DIR / "data" / "graph_campaign_quality_q60_top3_b6000_s020"
DEFAULT_EDGE_DIR = PROJECT_DIR / "data" / "edges_campaign_quality_q60_top3_b6000_s020"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_row(path: Path, category: str, role: str, mutable: bool) -> dict[str, Any]:
    exists = path.exists()
    return {
        "artifact_path": path_for_summary(path),
        "category": category,
        "role": role,
        "mutable": mutable,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": sha256_file(path) if exists and path.is_file() else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return path_for_summary(value)
    return value


def metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "valid", "test"]:
        block = metrics[f"{split}_metrics"]
        rows.append(
            {
                "split": split,
                "threshold": block.get("threshold"),
                "pr_auc": block.get("pr_auc"),
                "roc_auc": block.get("roc_auc"),
                "macro_f1": block.get("macro_f1"),
                "precision": block.get("precision"),
                "recall": block.get("recall"),
                "accuracy": block.get("accuracy"),
                "tn": block.get("tn"),
                "fp": block.get("fp"),
                "fn": block.get("fn"),
                "tp": block.get("tp"),
            }
        )
    return rows


def build_governance_layer(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(args.experiment_dir / "metrics.json")
    config = read_json(args.experiment_dir / "config_used.json")
    graph = read_json(args.graph_dir / "graph_summary.json")
    edges = read_json(args.edge_dir / "edges_summary.json")

    model_id = args.experiment_dir.name
    generated_at = datetime.now(timezone.utc).isoformat()
    active_graph = args.graph_dir / "graph_rur_custom2.pt"

    model_card = {
        "model_id": model_id,
        "generated_at_utc": generated_at,
        "model_type": metrics["model"],
        "graph_path": path_for_summary(active_graph),
        "operating_threshold": metrics.get("operating_threshold", metrics.get("best_threshold")),
        "validation_selected_threshold": metrics.get("validation_selected_threshold"),
        "threshold_note": metrics.get("operating_threshold_note"),
        "training": {
            "best_epoch": metrics["best_epoch"],
            "early_stop_metric": metrics["early_stop_metric"],
            "threshold_selection": metrics["threshold_selection"],
            "loss_weighting": metrics["loss_weighting"],
            "class_balanced_beta": metrics.get("class_balanced_beta"),
            "hidden_dim": config.get("hidden_dim"),
            "num_layers": config.get("num_layers"),
            "dropout": config.get("dropout"),
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "seed": config.get("seed"),
        },
        "graph": {
            "n_nodes": graph["n_nodes"],
            "numeric_dim": graph["numeric_dim"],
            "text_dim": graph["text_dim"],
            "total_feature_dim": graph["total_feature_dim"],
            "total_directed_edges": edges["total_directed_edges"],
            "overall_isolated_nodes": edges["overall_isolated_nodes"],
            "relation_edge_counts": metrics.get("relation_edge_counts", {}),
            "relation_stats": edges.get("relation_stats", {}),
        },
        "performance": {
            "valid": metrics["valid_metrics"],
            "test": metrics["test_metrics"],
        },
        "leakage_controls": [
            "Raw label/tag and label-rate aggregates are not used as model input features.",
            "Feature scaling is fitted on train_mask only.",
            "Filtered Campaign Pair component threshold is derived from the train mask.",
            "Test split is used for final reporting only, not for model/threshold selection.",
        ],
        "streamlit_usage": {
            "primary_file": "governance_manifest.json",
            "hide_eval_columns_by_default": True,
            "recommended_cache_key": "model_id + operating_threshold + governance_layer_version",
        },
    }

    metrics_rows = metric_rows(metrics)
    threshold_policy = [
        {
            "policy_item": "validation_selected_threshold",
            "value": metrics.get("validation_selected_threshold"),
            "description": "Threshold selected during validation-side prevalence-constrained Macro F1 review.",
        },
        {
            "policy_item": "operating_threshold",
            "value": metrics.get("operating_threshold", metrics.get("best_threshold")),
            "description": "Current fixed operating threshold for dashboard and prediction labels.",
        },
        {
            "policy_item": "threshold_change_rule",
            "value": "admin_approval_required",
            "description": "Streamlit should only simulate threshold changes until an admin records approval.",
        },
        {
            "policy_item": "test_split_policy",
            "value": "reporting_only",
            "description": "Test metrics may be shown in governance view but must not drive threshold selection.",
        },
    ]

    data_lineage = [
        {"stage_order": 1, "stage": "raw_data", "artifact": "data/origin/yelpzip.csv", "description": "Original YelpZip review table."},
        {"stage_order": 2, "stage": "preprocess", "artifact": "data/processed_rur_shock_context/", "description": "Review-node preprocessing and leakage-safe metadata construction."},
        {"stage_order": 3, "stage": "sampling", "artifact": "data/sampled_relative_flags_q75_m2/", "description": "Connected sampled review subgraph artifacts."},
        {"stage_order": 4, "stage": "split", "artifact": "data/splits_relative_flags_q75_m2/", "description": "Train/validation/test masks."},
        {"stage_order": 5, "stage": "text_embedding", "artifact": "data/embeddings_relative_flags_q75_m2/", "description": "TF-IDF/SVD text embedding artifacts."},
        {"stage_order": 6, "stage": "edge_building", "artifact": path_for_summary(args.edge_dir), "description": "R-U-R, Filtered Campaign Pair, and Weak Product Shock edges."},
        {"stage_order": 7, "stage": "graph_building", "artifact": path_for_summary(args.graph_dir), "description": "PyG graph with numeric and text features."},
        {"stage_order": 8, "stage": "training", "artifact": path_for_summary(args.experiment_dir), "description": "Validation-best relation_sage_mlp model and predictions."},
        {"stage_order": 9, "stage": "governance", "artifact": path_for_summary(args.output_dir), "description": "Governance layer consumed by the Streamlit dashboard."},
    ]

    access_policy = [
        {"role": "viewer", "can_view_metrics": True, "can_view_eval_only": False, "can_change_threshold": False, "can_export": False, "notes": "Presentation or read-only business user."},
        {"role": "reviewer", "can_view_metrics": True, "can_view_eval_only": False, "can_change_threshold": False, "can_export": True, "notes": "Operational reviewer; can export governance-safe summaries."},
        {"role": "admin", "can_view_metrics": True, "can_view_eval_only": True, "can_change_threshold": True, "can_export": True, "notes": "Can approve threshold changes and view evaluation-only fields."},
    ]

    artifact_paths = [
        file_row(args.experiment_dir / "metrics.json", "model", "metric source", False),
        file_row(args.experiment_dir / "config_used.json", "model", "training configuration", False),
        file_row(args.experiment_dir / "best_model.pt", "model", "trained model weights", False),
        file_row(args.experiment_dir / "predictions_all.csv", "model", "probability and operating labels", True),
        file_row(args.experiment_dir / "prediction_test.csv", "model", "test split prediction export", True),
        file_row(active_graph, "graph", "active PyG graph", False),
        file_row(args.graph_dir / "graph_summary.json", "graph", "graph metadata", False),
        file_row(args.edge_dir / "edges_summary.json", "edge", "edge design metadata", False),
        file_row(args.edge_dir / "edge_index.npy", "edge", "edge index array", False),
        file_row(args.edge_dir / "edge_type.npy", "edge", "edge type array", False),
        file_row(PROJECT_DIR / "VerifyResults.py", "verification", "headline metric verification", False),
        file_row(PROJECT_DIR / "ApplyOperatingThreshold.py", "operation", "threshold application utility", False),
        file_row(PROJECT_DIR / "BuildGovernanceLayer.py", "governance", "governance layer generator", False),
    ]

    manifest = {
        "governance_layer_version": "2026-05-19.q60.threshold0794",
        "generated_at_utc": generated_at,
        "model_id": model_id,
        "operating_threshold": model_card["operating_threshold"],
        "files": {
            "model_card": "model_card.json",
            "model_metrics": "model_metrics.csv",
            "threshold_policy": "threshold_policy.csv",
            "data_lineage": "data_lineage.csv",
            "artifact_registry": "artifact_registry.csv",
            "access_policy": "access_policy.csv",
        },
        "streamlit_contract": {
            "read_only_files": [
                "model_card.json",
                "model_metrics.csv",
                "threshold_policy.csv",
                "data_lineage.csv",
                "artifact_registry.csv",
                "access_policy.csv",
            ],
            "mutable_operational_tables": [
                "ops/governance_ops.db:threshold_change_log",
                "ops/governance_ops.db:audit_log",
                "ops/governance_ops.db:model_review_notes",
            ],
            "default_role": "viewer",
            "admin_role": "admin",
        },
    }

    (args.output_dir / "model_card.json").write_text(
        json.dumps(json_safe(model_card), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "governance_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "model_metrics.csv",
        metrics_rows,
        ["split", "threshold", "pr_auc", "roc_auc", "macro_f1", "precision", "recall", "accuracy", "tn", "fp", "fn", "tp"],
    )
    write_csv(
        args.output_dir / "threshold_policy.csv",
        threshold_policy,
        ["policy_item", "value", "description"],
    )
    write_csv(
        args.output_dir / "data_lineage.csv",
        data_lineage,
        ["stage_order", "stage", "artifact", "description"],
    )
    write_csv(
        args.output_dir / "artifact_registry.csv",
        artifact_paths,
        ["artifact_path", "category", "role", "mutable", "exists", "size_bytes", "sha256"],
    )
    write_csv(
        args.output_dir / "access_policy.csv",
        access_policy,
        ["role", "can_view_metrics", "can_view_eval_only", "can_change_threshold", "can_export", "notes"],
    )
    print(f"Saved governance layer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build governance layer artifacts for Streamlit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--edge-dir", type=Path, default=DEFAULT_EDGE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    build_governance_layer(parse_args())

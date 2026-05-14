"""
Create train/valid/test masks from sampled YelpZip review nodes only.

Example:
    python MakeSplits.py
    python MakeSplits.py --method random --random-state 42

This script intentionally reads only data/sampled artifacts. It never reads
data/origin/yelpzip.csv or the full data/processed directory.
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_SAMPLED_DIR = PROJECT_DIR / "data" / "sampled"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "splits"


def log(message: str) -> None:
    print(f"[MakeSplits] {message}")


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
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    total = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.12f} "
            f"(train={train_ratio}, valid={valid_ratio}, test={test_ratio})."
        )


def load_sampled_inputs(sampled_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    nodes_path = sampled_dir / "sampled_review_nodes.csv.gz"
    labels_path = sampled_dir / "sampled_node_labels.npy"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing sampled node file: {nodes_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing sampled label file: {labels_path}")

    log(f"Loading sampled nodes: {nodes_path}")
    nodes = pd.read_csv(nodes_path, parse_dates=["date"])
    labels = np.load(labels_path)

    required_columns = ["sampled_node_idx", "date", "is_fake"]
    missing_columns = [col for col in required_columns if col not in nodes.columns]
    if missing_columns:
        raise ValueError(f"sampled_review_nodes.csv.gz is missing required columns: {missing_columns}")

    nodes = nodes.sort_values("sampled_node_idx").reset_index(drop=True)
    validate_sampled_order(nodes, labels)
    return nodes, labels.astype(np.int64, copy=False)


def validate_sampled_order(nodes: pd.DataFrame, labels: np.ndarray) -> None:
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("sampled_review_nodes.csv.gz contains no rows.")

    sampled_idx = nodes["sampled_node_idx"].to_numpy(dtype=np.int64)
    expected_idx = np.arange(n_nodes, dtype=np.int64)
    if not np.array_equal(sampled_idx, expected_idx):
        bad_positions = np.flatnonzero(sampled_idx != expected_idx)
        first_bad = int(bad_positions[0]) if len(bad_positions) else -1
        raise ValueError(
            "sampled_node_idx must be contiguous from 0 to n-1 after sorting. "
            f"First mismatch at row {first_bad}: expected {first_bad}, got {sampled_idx[first_bad]}."
        )

    if len(labels) != n_nodes:
        raise ValueError(
            f"sampled_node_labels.npy length must match sampled nodes. "
            f"labels={len(labels)}, nodes={n_nodes}."
        )

    node_labels = nodes["is_fake"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels.astype(np.int64, copy=False), node_labels):
        mismatch = np.flatnonzero(labels.astype(np.int64, copy=False) != node_labels)
        first_bad = int(mismatch[0])
        raise ValueError(
            "sampled_node_labels.npy does not match sampled_review_nodes.csv.gz is_fake column. "
            f"First mismatch at sampled_node_idx={first_bad}: "
            f"npy={int(labels[first_bad])}, csv={int(node_labels[first_bad])}."
        )

    if nodes["date"].isna().any():
        missing_count = int(nodes["date"].isna().sum())
        raise ValueError(f"sampled_review_nodes.csv.gz has {missing_count} rows with invalid or missing date.")


def split_counts(n_nodes: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    train_count = int(n_nodes * train_ratio)
    valid_count = int(n_nodes * valid_ratio)
    test_count = n_nodes - train_count - valid_count
    if min(train_count, valid_count, test_count) <= 0:
        raise ValueError(
            f"Each split must contain at least one node. Got train={train_count}, "
            f"valid={valid_count}, test={test_count} for n={n_nodes}."
        )
    return train_count, valid_count, test_count


def make_temporal_masks(
    nodes: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(nodes)
    train_count, valid_count, _ = split_counts(n_nodes, train_ratio, valid_ratio)

    ordered_idx = (
        nodes.sort_values(["date", "sampled_node_idx"])["sampled_node_idx"]
        .to_numpy(dtype=np.int64)
    )
    train_idx = ordered_idx[:train_count]
    valid_idx = ordered_idx[train_count : train_count + valid_count]
    test_idx = ordered_idx[train_count + valid_count :]
    return indices_to_masks(n_nodes, train_idx, valid_idx, test_idx)


def make_random_stratified_masks(
    labels: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(labels)
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    rng = np.random.default_rng(random_state)

    for label in sorted(np.unique(labels).tolist()):
        label_idx = np.flatnonzero(labels == label)
        rng.shuffle(label_idx)
        train_count, valid_count, _ = split_counts(len(label_idx), train_ratio, valid_ratio)
        train_parts.append(label_idx[:train_count])
        valid_parts.append(label_idx[train_count : train_count + valid_count])
        test_parts.append(label_idx[train_count + valid_count :])

    train_idx = np.concatenate(train_parts)
    valid_idx = np.concatenate(valid_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)
    return indices_to_masks(n_nodes, train_idx, valid_idx, test_idx)


def indices_to_masks(
    n_nodes: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.zeros(n_nodes, dtype=bool)
    valid_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True
    validate_masks(train_mask, valid_mask, test_mask)
    return train_mask, valid_mask, test_mask


def validate_masks(train_mask: np.ndarray, valid_mask: np.ndarray, test_mask: np.ndarray) -> None:
    if train_mask.dtype != bool or valid_mask.dtype != bool or test_mask.dtype != bool:
        raise ValueError("All split masks must have bool dtype.")
    if not (train_mask.shape == valid_mask.shape == test_mask.shape):
        raise ValueError(
            f"Split mask shapes must match. Got train={train_mask.shape}, "
            f"valid={valid_mask.shape}, test={test_mask.shape}."
        )

    overlap_train_valid = bool(np.any(train_mask & valid_mask))
    overlap_train_test = bool(np.any(train_mask & test_mask))
    overlap_valid_test = bool(np.any(valid_mask & test_mask))
    if overlap_train_valid or overlap_train_test or overlap_valid_test:
        raise ValueError(
            "Split masks overlap. "
            f"train_valid={overlap_train_valid}, train_test={overlap_train_test}, "
            f"valid_test={overlap_valid_test}."
        )

    coverage = train_mask.astype(np.int8) + valid_mask.astype(np.int8) + test_mask.astype(np.int8)
    if not np.all(coverage == 1):
        missing = int(np.sum(coverage == 0))
        duplicated = int(np.sum(coverage > 1))
        raise ValueError(
            "Split masks must include every node exactly once. "
            f"missing={missing}, duplicated={duplicated}."
        )


def summarize_split(nodes: pd.DataFrame, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    subset = nodes.loc[mask]
    split_labels = labels[mask]
    fake_count = int(np.sum(split_labels == 1))
    real_count = int(np.sum(split_labels == 0))
    return {
        "nodes": int(mask.sum()),
        "fake": fake_count,
        "real": real_count,
        "fake_rate": float(fake_count / len(split_labels)) if len(split_labels) else None,
        "date_min": str(subset["date"].min().date()) if len(subset) else None,
        "date_max": str(subset["date"].max().date()) if len(subset) else None,
    }


def save_outputs(
    output_dir: Path,
    nodes: pd.DataFrame,
    labels: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Saving masks to: {output_dir}")
    np.save(output_dir / "train_mask.npy", train_mask)
    np.save(output_dir / "valid_mask.npy", valid_mask)
    np.save(output_dir / "test_mask.npy", test_mask)

    sampled_nodes_path = args.sampled_dir / "sampled_review_nodes.csv.gz"
    sampled_labels_path = args.sampled_dir / "sampled_node_labels.npy"
    train_mask_path = output_dir / "train_mask.npy"
    valid_mask_path = output_dir / "valid_mask.npy"
    test_mask_path = output_dir / "test_mask.npy"
    summary_path = output_dir / "split_summary.json"

    summary = {
        "method": args.method,
        "random_state": int(args.random_state),
        "total_nodes": int(len(nodes)),
        "ratios": {
            "train": float(args.train_ratio),
            "valid": float(args.valid_ratio),
            "test": float(args.test_ratio),
        },
        "splits": {
            "train": summarize_split(nodes, labels, train_mask),
            "valid": summarize_split(nodes, labels, valid_mask),
            "test": summarize_split(nodes, labels, test_mask),
        },
        "inputs": {
            "sampled_nodes": path_for_summary(sampled_nodes_path),
            "sampled_labels": path_for_summary(sampled_labels_path),
        },
        "outputs": {
            "train_mask": path_for_summary(train_mask_path),
            "valid_mask": path_for_summary(valid_mask_path),
            "test_mask": path_for_summary(test_mask_path),
            "summary": path_for_summary(summary_path),
        },
        "notes": [
            "Only data/sampled artifacts are used.",
            "Masks are indexed by sampled_node_idx.",
            "Labels are used only for stratification in random split and summary diagnostics.",
            "Example command: python MakeSplits.py --method temporal",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    log(f"Saved summary: {summary_path}")


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_make_splits(args: argparse.Namespace) -> None:
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    nodes, labels = load_sampled_inputs(args.sampled_dir)

    if args.method == "temporal":
        log("Creating temporal split masks by review date.")
        train_mask, valid_mask, test_mask = make_temporal_masks(nodes, args.train_ratio, args.valid_ratio)
    elif args.method == "random":
        log("Creating random stratified split masks by sampled labels.")
        train_mask, valid_mask, test_mask = make_random_stratified_masks(
            labels,
            args.train_ratio,
            args.valid_ratio,
            args.random_state,
        )
    else:
        raise ValueError(f"Unsupported split method: {args.method}")

    validate_masks(train_mask, valid_mask, test_mask)
    save_outputs(args.output_dir, nodes, labels, train_mask, valid_mask, test_mask, args)
    log(
        "Done. "
        f"train={int(train_mask.sum()):,}, valid={int(valid_mask.sum()):,}, test={int(test_mask.sum()):,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/valid/test masks from data/sampled YelpZip review nodes."
    )
    parser.add_argument("--sampled-dir", type=Path, default=DEFAULT_SAMPLED_DIR, help="Sampled data directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output split directory")
    parser.add_argument(
        "--method",
        choices=["temporal", "random"],
        default="temporal",
        help="Split method. temporal uses date order; random uses stratified labels.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.64, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.16, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for random stratified split")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_make_splits(parse_args())
    except Exception as exc:
        print(f"[MakeSplits][ERROR] {exc}", file=sys.stderr)
        raise

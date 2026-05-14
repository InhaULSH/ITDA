"""
Build PyTorch Geometric graph datasets from sampled YelpZip artifacts.

Example:
    python BuildGraphDataset.py

This script intentionally reads only sampled, split, embedding, and edge
artifacts. Raw IDs, labels, dates, and tags are not included in x.
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
DEFAULT_EMBEDDING_PATH = PROJECT_DIR / "data" / "embeddings" / "sampled_text_tfidf_svd.npy"
DEFAULT_EDGE_DIR = PROJECT_DIR / "data" / "edges"
DEFAULT_SPLIT_DIR = PROJECT_DIR / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "graph"


def log(message: str) -> None:
    print(f"[BuildGraphDataset] {message}")


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


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def import_pyg() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install PyTorch first, then rerun this script. "
            "See: https://pytorch.org/get-started/locally/"
        ) from exc

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError(
            "torch_geometric is not installed. Install PyTorch Geometric for your PyTorch/CUDA version, "
            "then rerun this script. See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        ) from exc

    return torch, Data


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    sampled_dir = args.sampled_dir
    paths = {
        "numeric": sampled_dir / "sampled_node_features_numeric.npy",
        "labels": sampled_dir / "sampled_node_labels.npy",
        "review_ids": sampled_dir / "sampled_node_review_ids.npy",
        "nodes": sampled_dir / "sampled_review_nodes.csv.gz",
        "text": args.embedding_path,
        "edge_index": args.edge_dir / "edge_index.npy",
        "edge_type": args.edge_dir / "edge_type.npy",
        "train_mask": args.split_dir / "train_mask.npy",
        "valid_mask": args.split_dir / "valid_mask.npy",
        "test_mask": args.split_dir / "test_mask.npy",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {name} file: {path}")

    log("Loading graph dataset inputs.")
    numeric = np.load(paths["numeric"])
    labels = np.load(paths["labels"])
    review_ids = np.load(paths["review_ids"])
    text = np.load(paths["text"])
    edge_index = np.load(paths["edge_index"])
    edge_type = np.load(paths["edge_type"])
    train_mask = np.load(paths["train_mask"])
    valid_mask = np.load(paths["valid_mask"])
    test_mask = np.load(paths["test_mask"])
    nodes = pd.read_csv(paths["nodes"])
    nodes = nodes.sort_values("sampled_node_idx").reset_index(drop=True)
    if "is_target_node" in nodes.columns:
        target_mask = nodes["is_target_node"].astype(bool).to_numpy()
    else:
        target_mask = np.ones(len(nodes), dtype=bool)

    inputs = {
        "numeric": numeric,
        "labels": labels,
        "review_ids": review_ids,
        "text": text,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "test_mask": test_mask,
        "target_mask": target_mask,
        "train_target_mask": train_mask & target_mask,
        "valid_target_mask": valid_mask & target_mask,
        "test_target_mask": test_mask & target_mask,
        "nodes": nodes,
        "paths": paths,
    }
    validate_inputs(inputs)
    return inputs


def validate_inputs(inputs: dict[str, Any]) -> None:
    numeric = inputs["numeric"]
    labels = inputs["labels"]
    review_ids = inputs["review_ids"]
    text = inputs["text"]
    edge_index = inputs["edge_index"]
    edge_type = inputs["edge_type"]
    train_mask = inputs["train_mask"]
    valid_mask = inputs["valid_mask"]
    test_mask = inputs["test_mask"]
    target_mask = inputs["target_mask"]
    train_target_mask = inputs["train_target_mask"]
    valid_target_mask = inputs["valid_target_mask"]
    test_target_mask = inputs["test_target_mask"]
    nodes = inputs["nodes"]

    if numeric.ndim != 2:
        raise ValueError(f"Numeric features must be 2D, got shape {numeric.shape}.")
    if text.ndim != 2:
        raise ValueError(f"Text embeddings must be 2D, got shape {text.shape}.")

    n_nodes = numeric.shape[0]
    if n_nodes == 0:
        raise ValueError("No sampled nodes found in numeric features.")
    for name, array in [("labels", labels), ("review_ids", review_ids)]:
        if array.shape != (n_nodes,):
            raise ValueError(f"{name} shape must be ({n_nodes},), got {array.shape}.")
    if text.shape[0] != n_nodes:
        raise ValueError(f"Text embedding rows must match nodes. text={text.shape[0]}, nodes={n_nodes}.")
    if len(nodes) != n_nodes:
        raise ValueError(f"sampled_review_nodes row count must match nodes. rows={len(nodes)}, nodes={n_nodes}.")

    sampled_idx = nodes["sampled_node_idx"].to_numpy(dtype=np.int64)
    expected_idx = np.arange(n_nodes, dtype=np.int64)
    if not np.array_equal(sampled_idx, expected_idx):
        bad_positions = np.flatnonzero(sampled_idx != expected_idx)
        first_bad = int(bad_positions[0]) if len(bad_positions) else -1
        raise ValueError(
            "sampled_review_nodes sampled_node_idx must be contiguous from 0 to n-1. "
            f"First mismatch at row {first_bad}: expected {first_bad}, got {sampled_idx[first_bad]}."
        )
    node_review_ids = nodes["review_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(review_ids.astype(np.int64, copy=False), node_review_ids):
        mismatch = np.flatnonzero(review_ids.astype(np.int64, copy=False) != node_review_ids)
        first_bad = int(mismatch[0])
        raise ValueError(
            "sampled_node_review_ids.npy does not match sampled_review_nodes.csv.gz. "
            f"First mismatch at sampled_node_idx={first_bad}."
        )

    for name, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        if mask.dtype != bool:
            raise ValueError(f"{name}_mask.npy must have bool dtype, got {mask.dtype}.")
        if mask.shape != (n_nodes,):
            raise ValueError(f"{name}_mask.npy shape must be ({n_nodes},), got {mask.shape}.")
    coverage = train_mask.astype(np.int8) + valid_mask.astype(np.int8) + test_mask.astype(np.int8)
    if not np.all(coverage == 1):
        raise ValueError("train/valid/test masks must include every node exactly once without overlap.")
    if int(train_mask.sum()) == 0:
        raise ValueError("train_mask has no nodes; cannot fit feature scaler.")

    target_mask = inputs["target_mask"]
    if target_mask.dtype != bool or target_mask.shape != (n_nodes,):
        raise ValueError(f"target_mask must be bool with shape ({n_nodes},), got {target_mask.dtype}, {target_mask.shape}.")
    for name, mask in [
        ("train_target", train_target_mask),
        ("valid_target", valid_target_mask),
        ("test_target", test_target_mask),
    ]:
        if mask.dtype != bool:
            raise ValueError(f"{name}_mask must have bool dtype, got {mask.dtype}.")
        if mask.shape != (n_nodes,):
            raise ValueError(f"{name}_mask shape must be ({n_nodes},), got {mask.shape}.")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}.")
    if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_type must have shape [num_edges]. edge_index={edge_index.shape}, edge_type={edge_type.shape}."
        )
    if edge_index.shape[1] > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) > n_nodes - 1:
            raise ValueError(f"edge_index values must be in [0, {n_nodes - 1}].")
    invalid_types = sorted(set(edge_type.astype(np.int64).tolist()) - {0, 1, 2})
    if invalid_types:
        raise ValueError(f"edge_type contains invalid relation values: {invalid_types}")


def build_scaled_features(
    numeric: np.ndarray,
    text: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Concatenating numeric features and text embeddings.")
    x = np.concatenate([numeric.astype(np.float32, copy=False), text.astype(np.float32, copy=False)], axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    log("Fitting feature scaler on train nodes only.")
    mean = x[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-12, 1.0, std).astype(np.float32)

    x_scaled = ((x - mean) / std).astype(np.float32)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return x_scaled, mean, std


def add_rating_history_text_features(inputs: dict[str, Any]) -> list[str]:
    nodes = inputs["nodes"]
    train_mask = inputs["train_mask"]
    required = ["word_len", "rating_bucket", "prior_product_review_count", "extreme_rating"]
    missing = [col for col in required if col not in nodes.columns]
    if missing:
        raise ValueError(f"Cannot add rating/history text features; sampled nodes are missing: {missing}")

    train_nodes = nodes.loc[train_mask].copy()
    history_values = pd.to_numeric(train_nodes["prior_product_review_count"], errors="coerce").fillna(0.0)
    quantile_values = history_values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=np.float64)
    cutoffs = np.unique(quantile_values[np.isfinite(quantile_values)])

    def history_bin(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        return np.searchsorted(cutoffs, values, side="right").astype(np.int16)

    nodes = nodes.copy()
    nodes["_history_bin"] = history_bin(nodes["prior_product_review_count"])
    train_nodes = nodes.loc[train_mask].copy()
    train_nodes["_word_len_numeric"] = pd.to_numeric(train_nodes["word_len"], errors="coerce").fillna(0.0)

    global_stats = {
        "q25": float(train_nodes["_word_len_numeric"].quantile(0.25)),
        "median": float(train_nodes["_word_len_numeric"].median()),
    }
    rating_stats = (
        train_nodes.groupby("rating_bucket", observed=True)["_word_len_numeric"]
        .agg(q25=lambda s: float(s.quantile(0.25)), median="median")
        .reset_index()
    )
    group_stats = (
        train_nodes.groupby(["rating_bucket", "_history_bin"], observed=True)["_word_len_numeric"]
        .agg(q25=lambda s: float(s.quantile(0.25)), median="median")
        .reset_index()
    )

    stats = nodes[["rating_bucket", "_history_bin", "word_len", "extreme_rating"]].copy()
    stats["_word_len_numeric"] = pd.to_numeric(stats["word_len"], errors="coerce").fillna(0.0)
    stats = stats.merge(group_stats, on=["rating_bucket", "_history_bin"], how="left")
    stats = stats.merge(rating_stats, on="rating_bucket", how="left", suffixes=("", "_rating"))
    stats["median"] = stats["median"].fillna(stats["median_rating"]).fillna(global_stats["median"])
    stats["q25"] = stats["q25"].fillna(stats["q25_rating"]).fillna(global_stats["q25"])
    median = stats["median"].astype("float32").to_numpy()
    q25 = stats["q25"].astype("float32").to_numpy()
    word_len = stats["_word_len_numeric"].astype("float32").to_numpy()
    short_by_group = (word_len <= q25).astype("float32")
    extreme = pd.to_numeric(stats["extreme_rating"], errors="coerce").fillna(0.0).astype("float32").to_numpy()

    extra = np.stack(
        [
            word_len - median,
            word_len / np.maximum(median, 1.0),
            short_by_group,
            short_by_group * extreme,
        ],
        axis=1,
    ).astype(np.float32)
    inputs["numeric"] = np.concatenate([inputs["numeric"].astype(np.float32, copy=False), extra], axis=1)
    added_columns = [
        "word_len_minus_train_rating_history_median",
        "word_len_ratio_to_train_rating_history_median",
        "is_short_by_train_rating_history_q25",
        "extreme_rating_and_short_by_train_rating_history_q25",
    ]
    inputs["extra_numeric_feature_columns"] = inputs.get("extra_numeric_feature_columns", []) + added_columns
    inputs["text_sufficiency_feature_cutoffs"] = {
        "history_bin_train_quantiles": [float(value) for value in cutoffs.tolist()],
        "global_train_word_len_q25": global_stats["q25"],
        "global_train_word_len_median": global_stats["median"],
    }
    return added_columns


def add_product_prior_context_features(inputs: dict[str, Any]) -> list[str]:
    nodes = inputs["nodes"]
    train_mask = inputs["train_mask"]
    required = [
        "prior_product_review_count",
        "prior_product_rating_std",
        "rating_deviation_from_prior_product_mean",
        "product_reviews_last_7d",
        "product_reviews_last_30d",
    ]
    missing = [col for col in required if col not in nodes.columns]
    if missing:
        raise ValueError(f"Cannot add product-prior context features; sampled nodes are missing: {missing}")

    prior_count = pd.to_numeric(nodes["prior_product_review_count"], errors="coerce").fillna(0.0).astype("float32")
    rating_std = pd.to_numeric(nodes["prior_product_rating_std"], errors="coerce")
    train_std = rating_std.loc[train_mask]
    positive_std = train_std.loc[train_std.gt(0)]
    std_floor = float(positive_std.quantile(0.25)) if not positive_std.empty else 1.0
    std_floor = max(std_floor, 1e-6)
    scale = rating_std.fillna(std_floor).clip(lower=std_floor).astype("float32")

    rating_dev = (
        pd.to_numeric(nodes["rating_deviation_from_prior_product_mean"], errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )
    reviews_7d = pd.to_numeric(nodes["product_reviews_last_7d"], errors="coerce").fillna(0.0).astype("float32")
    reviews_30d = pd.to_numeric(nodes["product_reviews_last_30d"], errors="coerce").fillna(0.0).astype("float32")
    train_prior_count = prior_count.loc[train_mask]
    low_history_cutoff = float(train_prior_count.quantile(0.25))

    standardized_dev = (rating_dev / scale).to_numpy(dtype=np.float32)
    abs_standardized_dev = np.abs(standardized_dev).astype(np.float32)
    count_values = prior_count.to_numpy(dtype=np.float32)
    denom = np.maximum(count_values, 1.0)
    recent_7d_share = (reviews_7d.to_numpy(dtype=np.float32) / denom).astype(np.float32)
    recent_30d_share = (reviews_30d.to_numpy(dtype=np.float32) / denom).astype(np.float32)
    low_history = (count_values <= low_history_cutoff).astype(np.float32)

    extra = np.stack(
        [
            standardized_dev,
            abs_standardized_dev,
            recent_7d_share,
            recent_30d_share,
            low_history,
        ],
        axis=1,
    ).astype(np.float32)
    inputs["numeric"] = np.concatenate([inputs["numeric"].astype(np.float32, copy=False), extra], axis=1)
    added_columns = [
        "standardized_rating_deviation_from_prior_product_mean",
        "abs_standardized_rating_deviation_from_prior_product_mean",
        "product_reviews_last_7d_share_of_prior_count",
        "product_reviews_last_30d_share_of_prior_count",
        "low_product_history_by_train_q25",
    ]
    inputs["extra_numeric_feature_columns"] = inputs.get("extra_numeric_feature_columns", []) + added_columns
    inputs["product_prior_context_feature_cutoffs"] = {
        "train_prior_product_rating_std_q25_positive": std_floor,
        "train_prior_product_review_count_q25": low_history_cutoff,
    }
    return added_columns


def add_behavior_shift_features(inputs: dict[str, Any]) -> list[str]:
    nodes = inputs["nodes"]
    candidate_columns = [
        "same_dir_log_count_lift_4w",
        "total_log_count_lift_4w",
        "direction_concentration_lift_4w",
        "new_user_ratio_lift_4w",
        "short_review_ratio_lift_4w",
        "word_len_drop_ratio_4w",
    ]
    available_columns = [name for name in candidate_columns if name in nodes.columns]
    if not available_columns:
        raise ValueError(
            "Cannot add behavior-shift features; sampled nodes do not contain any of: "
            f"{candidate_columns}"
        )

    features = nodes[available_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_array = features.to_numpy(dtype=np.float32, copy=True)
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    inputs["numeric"] = np.concatenate([inputs["numeric"].astype(np.float32, copy=False), feature_array], axis=1)
    inputs["extra_numeric_feature_columns"] = inputs.get("extra_numeric_feature_columns", []) + available_columns
    inputs["behavior_shift_feature_notes"] = {
        "columns": available_columns,
        "excluded_columns": [
            "abuse_burst_behavior_score_4w",
            "Any label-derived or full-period fake-rate aggregate.",
        ],
        "rationale": (
            "Use individual prior-4-week behavior-shift components instead of a composite score "
            "to avoid arbitrary coefficient-based risk scoring."
        ),
    }
    return available_columns


def numeric_series(nodes: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in nodes.columns:
        return pd.Series(default, index=nodes.index, dtype="float64")
    return pd.to_numeric(nodes[column], errors="coerce").fillna(default).astype("float64")


def binary_series(nodes: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    if column not in nodes.columns:
        return pd.Series(default, index=nodes.index, dtype="float64")
    values = pd.to_numeric(nodes[column], errors="coerce").fillna(default)
    return values.astype("float64")


def train_quantile(values: pd.Series, train_mask: np.ndarray, q: float, default: float = 0.0) -> float:
    train_values = pd.to_numeric(values.loc[train_mask], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if train_values.empty:
        return float(default)
    return float(train_values.quantile(q))


def history_bin_from_train(prior_count: pd.Series, train_mask: np.ndarray) -> tuple[np.ndarray, list[float]]:
    train_values = prior_count.loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if train_values.empty:
        cutoffs = np.array([], dtype=np.float64)
    else:
        cutoffs = np.unique(train_values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=np.float64))
        cutoffs = cutoffs[np.isfinite(cutoffs)]
    values = prior_count.fillna(0.0).to_numpy(dtype=np.float64)
    bins = np.searchsorted(cutoffs, values, side="right").astype(np.int16)
    return bins, [float(value) for value in cutoffs.tolist()]


def grouped_train_stats(
    nodes: pd.DataFrame,
    train_mask: np.ndarray,
    group_columns: list[str],
    value_column: str,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_nodes = nodes.loc[train_mask, group_columns + [value_column]].copy()
    train_nodes[value_column] = pd.to_numeric(train_nodes[value_column], errors="coerce").fillna(0.0)
    global_stats = {
        f"{prefix}_q25": float(train_nodes[value_column].quantile(0.25)) if len(train_nodes) else 0.0,
        f"{prefix}_median": float(train_nodes[value_column].median()) if len(train_nodes) else 0.0,
    }
    stats = (
        train_nodes.groupby(group_columns, observed=True)[value_column]
        .agg(q25=lambda s: float(s.quantile(0.25)), median="median")
        .reset_index()
    )
    stats = stats.rename(columns={"q25": f"{prefix}_q25", "median": f"{prefix}_median"})
    return stats, global_stats


def weak_template_similarity_features(
    nodes: pd.DataFrame,
    text: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if "prod_id" not in nodes.columns or "week" not in nodes.columns or len(nodes) == 0:
        empty = np.zeros((len(nodes), 3), dtype=np.float32)
        return empty, {"train_q90_template_similarity": 0.0}

    embeddings = text.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(embeddings, np.maximum(norms, 1e-12), out=np.zeros_like(embeddings), where=norms > 0)
    max_sim = np.zeros(len(nodes), dtype=np.float32)
    mean_top3 = np.zeros(len(nodes), dtype=np.float32)

    grouped = nodes.groupby(["prod_id", "week"], sort=False, observed=True).indices
    for positions in grouped.values():
        idx = np.asarray(positions, dtype=np.int64)
        if idx.size <= 1:
            continue
        group_vec = normalized[idx]
        sim = np.clip(group_vec @ group_vec.T, -1.0, 1.0)
        np.fill_diagonal(sim, -np.inf)
        finite = np.where(np.isfinite(sim), sim, 0.0)
        max_sim[idx] = np.max(finite, axis=1).astype(np.float32)
        k = min(3, idx.size - 1)
        if k > 0:
            topk = np.partition(finite, kth=finite.shape[1] - k, axis=1)[:, -k:]
            mean_top3[idx] = np.mean(topk, axis=1).astype(np.float32)

    q90 = float(pd.Series(max_sim[train_mask]).quantile(0.90)) if int(train_mask.sum()) else 0.0
    high_flag = (max_sim >= q90).astype(np.float32)
    features = np.stack([max_sim, mean_top3, high_flag], axis=1).astype(np.float32)
    return features, {"train_q90_template_similarity": q90}


def build_eda_logic_compact_features(inputs: dict[str, Any]) -> list[str]:
    nodes = inputs["nodes"].copy()
    text = inputs["text"]
    train_mask = inputs["train_mask"]

    required = [
        "rating",
        "rating_norm",
        "rating_direction",
        "rating_bucket",
        "extreme_rating",
        "word_len",
        "log_word_len",
        "prior_user_review_count",
        "is_new_user_at_review_time",
        "prior_user_avg_rating",
        "prior_product_review_count",
        "prior_product_avg_rating",
        "prior_product_rating_std",
        "rating_deviation_from_prior_product_mean",
        "product_reviews_last_7d",
        "product_reviews_last_30d",
    ]
    missing = [col for col in required if col not in nodes.columns]
    if missing:
        raise ValueError(f"Cannot build eda_logic_compact features; sampled nodes are missing: {missing}")

    prior_product_count = numeric_series(nodes, "prior_product_review_count")
    history_bins, history_cutoffs = history_bin_from_train(prior_product_count, train_mask)
    nodes["_history_bin"] = history_bins

    word_len = numeric_series(nodes, "word_len")
    text_stats, text_global = grouped_train_stats(
        nodes.assign(_word_len_numeric=word_len),
        train_mask,
        ["rating_bucket", "_history_bin"],
        "_word_len_numeric",
        "word_len",
    )
    rating_text_stats, _ = grouped_train_stats(
        nodes.assign(_word_len_numeric=word_len),
        train_mask,
        ["rating_bucket"],
        "_word_len_numeric",
        "word_len_rating",
    )
    text_context = nodes[["rating_bucket", "_history_bin"]].copy()
    text_context = text_context.merge(text_stats, on=["rating_bucket", "_history_bin"], how="left")
    text_context = text_context.merge(rating_text_stats, on=["rating_bucket"], how="left")
    text_median = (
        text_context["word_len_median"]
        .fillna(text_context["word_len_rating_median"])
        .fillna(text_global["word_len_median"])
        .astype("float64")
    )
    text_q25 = (
        text_context["word_len_q25"]
        .fillna(text_context["word_len_rating_q25"])
        .fillna(text_global["word_len_q25"])
        .astype("float64")
    )
    word_len_ratio = word_len / np.maximum(text_median, 1.0)
    is_short_relative = word_len.le(text_q25).astype("float64")
    text_sufficient = word_len.ge(text_median).astype("float64")

    rating = numeric_series(nodes, "rating")
    prior_user_count = numeric_series(nodes, "prior_user_review_count")
    prior_user_avg = numeric_series(nodes, "prior_user_avg_rating")
    user_dev = np.where(prior_user_count.gt(0), rating - prior_user_avg, 0.0)
    user_dev = pd.Series(user_dev, index=nodes.index, dtype="float64")
    user_std = numeric_series(nodes, "prior_user_rating_std", default=np.nan)
    positive_user_std = user_std.loc[train_mask & user_std.gt(0)]
    user_std_floor = float(positive_user_std.quantile(0.25)) if not positive_user_std.empty else 1.0
    user_std_floor = max(user_std_floor, 1e-6)
    standardized_user_dev = user_dev / user_std.fillna(user_std_floor).clip(lower=user_std_floor)

    product_std = numeric_series(nodes, "prior_product_rating_std", default=np.nan)
    positive_product_std = product_std.loc[train_mask & product_std.gt(0)]
    product_std_floor = float(positive_product_std.quantile(0.25)) if not positive_product_std.empty else 1.0
    product_std_floor = max(product_std_floor, 1e-6)
    product_scale = product_std.fillna(product_std_floor).clip(lower=product_std_floor)
    product_dev = numeric_series(nodes, "rating_deviation_from_prior_product_mean")
    standardized_product_dev = product_dev / product_scale
    abs_standardized_product_dev = standardized_product_dev.abs()
    positive_standardized_product_dev = standardized_product_dev.clip(lower=0.0)
    negative_standardized_product_dev = (-standardized_product_dev).clip(lower=0.0)

    prior_product_q25 = train_quantile(prior_product_count, train_mask, 0.25)
    prior_product_q75 = train_quantile(prior_product_count, train_mask, 0.75)
    low_reputation = prior_product_count.le(prior_product_q25).astype("float64")
    high_reputation = prior_product_count.ge(prior_product_q75).astype("float64")
    rating_impact_abs = (product_dev.abs() / (prior_product_count.clip(lower=0.0) + 1.0)).astype("float64")

    reviews_7d = numeric_series(nodes, "product_reviews_last_7d")
    reviews_30d = numeric_series(nodes, "product_reviews_last_30d")
    product_denominator = prior_product_count.clip(lower=1.0)
    recent_7d_share = reviews_7d / product_denominator
    recent_30d_share = reviews_30d / product_denominator

    days_since_user_last = numeric_series(nodes, "days_since_user_last_review", default=-1.0)
    returning_recent_30d = days_since_user_last.between(0, 30).astype("float64")
    has_prior_7d = binary_series(nodes, "has_prior_user_review_7d")
    if "has_prior_user_review_30d" in nodes.columns:
        has_prior_30d = binary_series(nodes, "has_prior_user_review_30d")
    else:
        has_prior_30d = returning_recent_30d

    template_features, template_cutoffs = weak_template_similarity_features(nodes, text, train_mask)

    base_feature_map: dict[str, pd.Series | np.ndarray] = {
        "rating_norm": numeric_series(nodes, "rating_norm"),
        "rating_direction": numeric_series(nodes, "rating_direction"),
        "extreme_rating": binary_series(nodes, "extreme_rating"),
        "positive_rating_direction_flag": numeric_series(nodes, "rating_direction").gt(0).astype("float64"),
        "negative_rating_direction_flag": numeric_series(nodes, "rating_direction").lt(0).astype("float64"),
        "log_word_len": numeric_series(nodes, "log_word_len"),
        "upper_ratio": numeric_series(nodes, "upper_ratio"),
        "exclamation_count": numeric_series(nodes, "exclamation_count"),
        "question_count": numeric_series(nodes, "question_count"),
        "log1p_prior_user_review_count": numeric_series(nodes, "log1p_prior_user_review_count"),
        "is_new_user_at_review_time": binary_series(nodes, "is_new_user_at_review_time"),
        "returning_recent_user_30d_flag": returning_recent_30d,
        "has_prior_user_review_7d": has_prior_7d,
        "has_prior_user_review_30d": has_prior_30d,
        "prior_user_avg_rating": prior_user_avg,
        "prior_user_extreme_ratio": numeric_series(nodes, "prior_user_extreme_ratio"),
        "log1p_prior_product_review_count": numeric_series(nodes, "log1p_prior_product_review_count"),
        "prior_product_avg_rating": numeric_series(nodes, "prior_product_avg_rating"),
        "prior_product_rating_std": numeric_series(nodes, "prior_product_rating_std"),
        "prior_product_extreme_ratio": numeric_series(nodes, "prior_product_extreme_ratio"),
    }
    optional_base_columns = [
        "unique_token_ratio",
        "avg_token_len",
        "numeric_token_flag",
        "log1p_prior_user_active_span_days",
        "prior_user_rating_std",
    ]
    for col in optional_base_columns:
        if col in nodes.columns:
            base_feature_map[col] = numeric_series(nodes, col)

    engineered_feature_map: dict[str, pd.Series | np.ndarray] = {
        "word_len_minus_train_rating_history_median": word_len - text_median,
        "word_len_ratio_to_train_rating_history_median": word_len_ratio,
        "is_short_by_train_rating_history_q25": is_short_relative,
        "text_sufficient_by_train_rating_history_median": text_sufficient,
        "extreme_rating_and_short_by_train_rating_history_q25": is_short_relative * binary_series(nodes, "extreme_rating"),
        "standardized_rating_deviation_from_prior_product_mean": standardized_product_dev,
        "abs_standardized_rating_deviation_from_prior_product_mean": abs_standardized_product_dev,
        "positive_standardized_rating_deviation_from_prior_product_mean": positive_standardized_product_dev,
        "negative_standardized_rating_deviation_from_prior_product_mean": negative_standardized_product_dev,
        "rating_impact_abs_from_prior_product": rating_impact_abs,
        "low_product_reputation_by_train_q25": low_reputation,
        "high_product_reputation_by_train_q75": high_reputation,
        "low_reputation_and_abs_standardized_product_deviation": low_reputation * abs_standardized_product_dev,
        "product_reviews_last_7d_share_of_prior_count": recent_7d_share,
        "product_reviews_last_30d_share_of_prior_count": recent_30d_share,
        "user_rating_deviation_from_prior_mean": user_dev,
        "abs_user_rating_deviation_from_prior_mean": user_dev.abs(),
        "standardized_user_rating_deviation_from_prior_mean": standardized_user_dev,
        "abs_standardized_user_rating_deviation_from_prior_mean": standardized_user_dev.abs(),
        "new_user_and_abs_standardized_product_deviation": binary_series(nodes, "is_new_user_at_review_time")
        * abs_standardized_product_dev,
        "text_sufficient_and_abs_standardized_product_deviation": text_sufficient * abs_standardized_product_dev,
        "returning_recent_user_and_text_sufficient": returning_recent_30d * text_sufficient,
        "weak_template_similarity_max_product_week": template_features[:, 0],
        "weak_template_similarity_top3_mean_product_week": template_features[:, 1],
        "weak_template_similarity_high_by_train_q90": template_features[:, 2],
    }

    feature_names = list(base_feature_map.keys()) + list(engineered_feature_map.keys())
    feature_arrays: list[np.ndarray] = []
    for name in feature_names:
        source = base_feature_map[name] if name in base_feature_map else engineered_feature_map[name]
        values = np.asarray(source, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        feature_arrays.append(values)

    inputs["numeric"] = np.column_stack(feature_arrays).astype(np.float32)
    inputs["numeric_feature_columns"] = feature_names
    inputs["extra_numeric_feature_columns"] = list(engineered_feature_map.keys())
    inputs["feature_profile"] = "eda_logic_compact"
    inputs["feature_design_notes"] = {
        "profile": "eda_logic_compact",
        "rationale": (
            "Rebuild numeric node features from EDA-aligned behavior blocks: reputation incentive, "
            "rating-deviation context, text concreteness, user-history protection, and weak template repetition. "
            "Raw product-week burst size, raw IDs, labels, full-period aggregates, and arbitrary composite risk scores "
            "are excluded."
        ),
        "removed_or_replaced_examples": [
            "short_review_flag is replaced by rating/history-relative text sufficiency features.",
            "log1p_same_text_count_in_product_week is replaced by weak template similarity over train-fitted text embeddings.",
            "raw product_reviews_last_7d/30d counts are replaced by shares relative to prior product history.",
            "raw product rating deviation is replaced by standardized positive/negative deviation and rating impact.",
            "abuse_burst_behavior_score_4w is not used because it is a composite behavior score.",
        ],
        "base_feature_columns": list(base_feature_map.keys()),
        "engineered_feature_columns": list(engineered_feature_map.keys()),
    }
    inputs["text_sufficiency_feature_cutoffs"] = {
        "history_bin_train_quantiles": history_cutoffs,
        **text_global,
    }
    inputs["product_prior_context_feature_cutoffs"] = {
        "train_prior_product_rating_std_q25_positive": product_std_floor,
        "train_prior_product_review_count_q25": prior_product_q25,
        "train_prior_product_review_count_q75": prior_product_q75,
    }
    inputs["user_context_feature_cutoffs"] = {
        "train_prior_user_rating_std_q25_positive": user_std_floor,
    }
    inputs["template_similarity_feature_cutoffs"] = template_cutoffs
    return feature_names


def infer_original_numeric_feature_columns(inputs: dict[str, Any]) -> list[str]:
    n_features = int(inputs["numeric"].shape[1])
    candidate_paths = [
        PROJECT_DIR / "data" / "processed_rur_shock_context" / "feature_columns.json",
        PROJECT_DIR / "data" / "processed" / "feature_columns.json",
        PROJECT_DIR / "data" / "processed_behavior_shift_clip" / "feature_columns.json",
        PROJECT_DIR / "data" / "processed_behavior_shift" / "feature_columns.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        columns = list(payload.get("numeric_feature_columns", []) or [])
        if len(columns) == n_features:
            return columns
    return [f"original_numeric_{idx:03d}" for idx in range(n_features)]


def build_eda_logic_directional_features(inputs: dict[str, Any]) -> list[str]:
    original_numeric = inputs["numeric"].astype(np.float32, copy=True)
    original_columns = infer_original_numeric_feature_columns(inputs)
    if len(original_columns) != original_numeric.shape[1]:
        original_columns = [f"original_numeric_{idx:03d}" for idx in range(original_numeric.shape[1])]

    compact_inputs = dict(inputs)
    compact_inputs["numeric"] = original_numeric.copy()
    compact_inputs["extra_numeric_feature_columns"] = []
    compact_columns = build_eda_logic_compact_features(compact_inputs)
    compact_numeric = compact_inputs["numeric"]
    compact_map = {name: compact_numeric[:, idx] for idx, name in enumerate(compact_columns)}

    nodes = inputs["nodes"]
    train_mask = inputs["train_mask"]
    prior_user_count = numeric_series(nodes, "prior_user_review_count")
    days_since_user_last = numeric_series(nodes, "days_since_user_last_review", default=-1.0)
    returning_recent_30d = days_since_user_last.between(0, 30).astype("float32").to_numpy()
    new_user_or_sparse = (
        binary_series(nodes, "is_new_user_at_review_time").eq(1) | prior_user_count.le(1)
    ).astype("float32").to_numpy()

    abs_user_dev = np.asarray(compact_map.get("abs_standardized_user_rating_deviation_from_prior_mean"), dtype=np.float32)
    train_history_user_mask = train_mask & prior_user_count.gt(0).to_numpy()
    train_history_user_dev = pd.Series(abs_user_dev[train_history_user_mask])
    train_positive_user_dev = train_history_user_dev[train_history_user_dev.gt(0)]
    if not train_positive_user_dev.empty:
        user_dev_consistency_cutoff = float(train_positive_user_dev.quantile(0.25))
    elif not train_history_user_dev.empty:
        user_dev_consistency_cutoff = float(train_history_user_dev.quantile(0.50))
    else:
        user_dev_consistency_cutoff = 0.0
    user_rating_consistent = (abs_user_dev <= user_dev_consistency_cutoff).astype(np.float32)

    positive_dev = np.asarray(
        compact_map["positive_standardized_rating_deviation_from_prior_product_mean"], dtype=np.float32
    )
    negative_dev = np.asarray(
        compact_map["negative_standardized_rating_deviation_from_prior_product_mean"], dtype=np.float32
    )
    abs_product_dev = np.asarray(
        compact_map["abs_standardized_rating_deviation_from_prior_product_mean"], dtype=np.float32
    )
    low_reputation = np.asarray(compact_map["low_product_reputation_by_train_q25"], dtype=np.float32)
    high_reputation = np.asarray(compact_map["high_product_reputation_by_train_q75"], dtype=np.float32)
    extreme = binary_series(nodes, "extreme_rating").astype("float32").to_numpy()
    text_sufficient = np.asarray(compact_map["text_sufficient_by_train_rating_history_median"], dtype=np.float32)
    weak_template = np.asarray(compact_map["weak_template_similarity_max_product_week"], dtype=np.float32)

    directional_extra = {
        "positive_promotion_low_reputation_shock": positive_dev * low_reputation,
        "negative_attack_high_reputation_shock": negative_dev * high_reputation,
        "positive_extreme_rating_shock": positive_dev * extreme,
        "negative_extreme_rating_shock": negative_dev * extreme,
        "disguised_high_effort_sparse_user_shock": text_sufficient * abs_product_dev * new_user_or_sparse,
        "returning_recent_user_consistent_rating_and_text_sufficient": (
            returning_recent_30d * user_rating_consistent * text_sufficient
        ),
        "returning_recent_user_consistent_rating": returning_recent_30d * user_rating_consistent,
    }

    selected_engineered_columns = [
        "word_len_ratio_to_train_rating_history_median",
        "is_short_by_train_rating_history_q25",
        "text_sufficient_by_train_rating_history_median",
        "positive_standardized_rating_deviation_from_prior_product_mean",
        "negative_standardized_rating_deviation_from_prior_product_mean",
        "high_product_reputation_by_train_q75",
        "low_reputation_and_abs_standardized_product_deviation",
        "product_reviews_last_7d_share_of_prior_count",
        "product_reviews_last_30d_share_of_prior_count",
        "standardized_user_rating_deviation_from_prior_mean",
        "abs_standardized_user_rating_deviation_from_prior_mean",
        "returning_recent_user_and_text_sufficient",
        "weak_template_similarity_max_product_week",
    ]
    semantic_duplicates = {
        "word_len_ratio_to_train_rating_history_median": {"word_len_ratio_to_train_rating_history_median"},
        "is_short_by_train_rating_history_q25": {"is_short_by_train_rating_history_q25"},
        "positive_standardized_rating_deviation_from_prior_product_mean": {
            "positive_standardized_rating_deviation_from_prior_product_mean"
        },
        "negative_standardized_rating_deviation_from_prior_product_mean": {
            "negative_standardized_rating_deviation_from_prior_product_mean"
        },
        "product_reviews_last_7d_share_of_prior_count": {"product_reviews_last_7d_share_of_prior_count"},
        "product_reviews_last_30d_share_of_prior_count": {"product_reviews_last_30d_share_of_prior_count"},
    }
    original_set = set(original_columns)
    added_columns: list[str] = []
    added_arrays: list[np.ndarray] = []
    for name in selected_engineered_columns:
        if name not in compact_map:
            continue
        if semantic_duplicates.get(name, set()) & original_set:
            continue
        added_columns.append(name)
        added_arrays.append(np.asarray(compact_map[name], dtype=np.float32))

    for name, values in directional_extra.items():
        added_columns.append(name)
        added_arrays.append(np.asarray(values, dtype=np.float32))

    added_numeric = (
        np.column_stack(added_arrays).astype(np.float32)
        if added_arrays
        else np.empty((original_numeric.shape[0], 0), dtype=np.float32)
    )
    inputs["numeric"] = np.concatenate([original_numeric, added_numeric], axis=1).astype(np.float32)
    feature_names = original_columns + added_columns
    inputs["numeric_feature_columns"] = feature_names
    inputs["extra_numeric_feature_columns"] = added_columns
    inputs["feature_profile"] = "eda_logic_directional"
    inputs["feature_design_notes"] = {
        "profile": "eda_logic_directional",
        "rationale": (
            "Preserve original model stability while adding direction-aware manipulation features and normal-user "
            "protection features. Positive rating shocks are represented as promotion pressure on weak-reputation "
            "products, while negative shocks are represented as attack pressure on high-reputation products. "
            "Returning-user consistency and sufficient explanation are added as false-positive protection context."
        ),
        "kept_original_columns": original_columns,
        "added_engineered_columns": added_columns,
        "not_added_due_to_existing_equivalent": [
            name
            for name in selected_engineered_columns
            if name in compact_map and name not in added_columns
        ],
        "user_consistency_cutoffs": {
            "train_abs_standardized_user_deviation_q25_positive": user_dev_consistency_cutoff,
            "train_abs_standardized_user_deviation_consistency_cutoff": user_dev_consistency_cutoff,
        },
    }
    for key in [
        "text_sufficiency_feature_cutoffs",
        "product_prior_context_feature_cutoffs",
        "user_context_feature_cutoffs",
        "template_similarity_feature_cutoffs",
    ]:
        inputs[key] = compact_inputs.get(key, {})
    inputs["user_context_feature_cutoffs"][
        "train_abs_standardized_user_deviation_q25_positive"
    ] = user_dev_consistency_cutoff
    inputs["user_context_feature_cutoffs"][
        "train_abs_standardized_user_deviation_consistency_cutoff"
    ] = user_dev_consistency_cutoff
    return feature_names


def make_data(
    torch: Any,
    Data: Any,
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
    target_mask: np.ndarray,
    train_target_mask: np.ndarray,
    valid_target_mask: np.ndarray,
    test_target_mask: np.ndarray,
    review_ids: np.ndarray,
) -> Any:
    n_nodes = x.shape[0]
    data = Data(
        x=torch.from_numpy(x.astype(np.float32, copy=False)),
        y=torch.from_numpy(y.astype(np.int64, copy=False)),
        edge_index=torch.from_numpy(edge_index.astype(np.int64, copy=False)),
        edge_type=torch.from_numpy(edge_type.astype(np.int64, copy=False)),
        train_mask=torch.from_numpy(train_mask.astype(bool, copy=True)),
        valid_mask=torch.from_numpy(valid_mask.astype(bool, copy=True)),
        test_mask=torch.from_numpy(test_mask.astype(bool, copy=True)),
        target_mask=torch.from_numpy(target_mask.astype(bool, copy=True)),
        train_target_mask=torch.from_numpy(train_target_mask.astype(bool, copy=True)),
        valid_target_mask=torch.from_numpy(valid_target_mask.astype(bool, copy=True)),
        test_target_mask=torch.from_numpy(test_target_mask.astype(bool, copy=True)),
    )
    data.review_id = torch.from_numpy(review_ids.astype(np.int64, copy=False))
    data.sampled_node_idx = torch.arange(n_nodes, dtype=torch.long)
    return data


def filter_edges(edge_index: np.ndarray, edge_type: np.ndarray, keep_types: set[int]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(edge_type, sorted(keep_types))
    filtered_edge_index = edge_index[:, mask]
    filtered_edge_type = edge_type[mask]
    return filtered_edge_index.astype(np.int64, copy=False), filtered_edge_type.astype(np.int64, copy=False)


def summarize_graph(edge_type: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(edge_type.astype(np.int64), return_counts=True)
    return {
        "n_edges": int(edge_type.shape[0]),
        "edge_type_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
    }


def split_summary(labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    split_labels = labels[mask]
    fake = int(np.sum(split_labels == 1))
    real = int(np.sum(split_labels == 0))
    return {
        "nodes": int(mask.sum()),
        "fake": fake,
        "real": real,
        "fake_rate": float(fake / len(split_labels)) if len(split_labels) else None,
    }


def save_graphs(
    torch: Any,
    Data: Any,
    output_dir: Path,
    inputs: dict[str, Any],
    x: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = inputs["labels"]
    review_ids = inputs["review_ids"]
    edge_index = inputs["edge_index"].astype(np.int64, copy=False)
    edge_type = inputs["edge_type"].astype(np.int64, copy=False)
    train_mask = inputs["train_mask"]
    valid_mask = inputs["valid_mask"]
    test_mask = inputs["test_mask"]
    target_mask = inputs["target_mask"]
    train_target_mask = inputs["train_target_mask"]
    valid_target_mask = inputs["valid_target_mask"]
    test_target_mask = inputs["test_target_mask"]

    rur_custom2_path = output_dir / "graph_rur_custom2.pt"
    scaler_path = output_dir / "feature_scaler_params.npz"
    summary_path = output_dir / "graph_summary.json"

    log("Saving graph_rur_custom2.pt.")
    rc_edge_index, rc_edge_type = filter_edges(edge_index, edge_type, {0, 2})
    rc_data = make_data(
        torch, Data, x, labels, rc_edge_index, rc_edge_type, train_mask, valid_mask, test_mask,
        target_mask, train_target_mask, valid_target_mask, test_target_mask, review_ids
    )
    torch.save(rc_data, rur_custom2_path)

    np.savez(scaler_path, mean=scaler_mean.astype(np.float32), std=scaler_std.astype(np.float32))

    numeric_dim = int(inputs["numeric"].shape[1])
    text_dim = int(inputs["text"].shape[1])
    graph_summaries = {
        "graph_rur_custom2.pt": summarize_graph(rc_edge_type),
    }
    summary = {
        "n_nodes": int(x.shape[0]),
        "numeric_dim": numeric_dim,
        "text_dim": text_dim,
        "total_feature_dim": int(x.shape[1]),
        "graphs": graph_summaries,
        "splits": {
            "train": split_summary(labels, train_mask),
            "valid": split_summary(labels, valid_mask),
            "test": split_summary(labels, test_mask),
        },
        "target_splits": {
            "target_all": split_summary(labels, target_mask),
            "train": split_summary(labels, train_target_mask),
            "valid": split_summary(labels, valid_target_mask),
            "test": split_summary(labels, test_target_mask),
        },
        "feature_profile": inputs.get("feature_profile", "legacy"),
        "numeric_feature_columns": inputs.get("numeric_feature_columns", []),
        "extra_numeric_feature_columns": inputs.get("extra_numeric_feature_columns", []),
        "text_sufficiency_feature_cutoffs": inputs.get("text_sufficiency_feature_cutoffs", {}),
        "product_prior_context_feature_cutoffs": inputs.get("product_prior_context_feature_cutoffs", {}),
        "user_context_feature_cutoffs": inputs.get("user_context_feature_cutoffs", {}),
        "template_similarity_feature_cutoffs": inputs.get("template_similarity_feature_cutoffs", {}),
        "feature_design_notes": inputs.get("feature_design_notes", {}),
        "behavior_shift_feature_notes": inputs.get("behavior_shift_feature_notes", {}),
        "scaling": {
            "fit_on": "train_mask only",
            "params": path_for_summary(scaler_path),
            "nan_inf_policy": "NaN, +inf, and -inf are replaced with 0 before and after scaling.",
        },
        "inputs": {name: path_for_summary(path) for name, path in inputs["paths"].items()},
        "outputs": {
            "graph_rur_custom2": path_for_summary(rur_custom2_path),
            "feature_scaler_params": path_for_summary(scaler_path),
            "summary": path_for_summary(summary_path),
        },
        "notes": [
            "Only sampled artifacts are used.",
            "x is numeric sampled node features concatenated with sampled text embeddings.",
            "Raw label, tag, user_id, prod_id, and date values are not included in x.",
            "edge_index is indexed by sampled_node_idx.",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    log(f"Saved summary: {summary_path}")


def run_build_graph_dataset(args: argparse.Namespace) -> None:
    torch, Data = import_pyg()
    inputs = load_inputs(args)
    if args.feature_profile == "eda_logic_directional":
        added = build_eda_logic_directional_features(inputs)
        log(f"Built eda_logic_directional feature profile with {len(added)} numeric features.")
    elif args.add_rating_history_text_features:
        added = add_rating_history_text_features(inputs)
        log(f"Added rating/history-relative text features: {', '.join(added)}")
        if args.add_product_prior_context_features:
            added = add_product_prior_context_features(inputs)
            log(f"Added product-prior context features: {', '.join(added)}")
        if args.add_behavior_shift_features:
            added = add_behavior_shift_features(inputs)
            log(f"Added prior-4-week behavior-shift features: {', '.join(added)}")
    else:
        if args.add_product_prior_context_features:
            added = add_product_prior_context_features(inputs)
            log(f"Added product-prior context features: {', '.join(added)}")
        if args.add_behavior_shift_features:
            added = add_behavior_shift_features(inputs)
            log(f"Added prior-4-week behavior-shift features: {', '.join(added)}")
    x, scaler_mean, scaler_std = build_scaled_features(inputs["numeric"], inputs["text"], inputs["train_mask"])
    save_graphs(torch, Data, args.output_dir, inputs, x, scaler_mean, scaler_std, args)
    log(f"Done. Saved PyG graphs to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyTorch Geometric Data objects from sampled YelpZip artifacts.")
    parser.add_argument("--sampled-dir", type=Path, default=DEFAULT_SAMPLED_DIR, help="Sampled data directory")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=DEFAULT_EMBEDDING_PATH,
        help="Path to sampled text embedding .npy file",
    )
    parser.add_argument("--edge-dir", type=Path, default=DEFAULT_EDGE_DIR, help="Edge artifact directory")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR, help="Split mask directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Graph output directory")
    parser.add_argument(
        "--feature-profile",
        choices=[
            "legacy",
            "eda_logic_directional",
        ],
        default="legacy",
        help=(
            "legacy keeps sampled numeric features and optional append-only feature flags. "
            "eda_logic_directional adds direction-aware manipulation and normal-user protection features."
        ),
    )
    parser.add_argument(
        "--add-rating-history-text-features",
        action="store_true",
        help="Append train-only word-length sufficiency features by rating bucket and product history bin.",
    )
    parser.add_argument(
        "--add-product-prior-context-features",
        action="store_true",
        help="Append train-only-scaled product prior context features without adding product-context edges.",
    )
    parser.add_argument(
        "--add-behavior-shift-features",
        action="store_true",
        help=(
            "Append individual prior-4-week behavior-shift features already present in sampled nodes. "
            "Composite behavior scores and label-rate aggregates are not added."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_build_graph_dataset(parse_args())
    except Exception as exc:
        print(f"[BuildGraphDataset][ERROR] {exc}", file=sys.stderr)
        raise

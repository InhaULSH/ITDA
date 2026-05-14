"""
Build sampled-node GNN edges for YelpZip.

Relations:
    0 = R-U-R
    1 = product_time_rating_burst
    2 = custom relation: template/text, cold_start, or weak product shock

Example:
    python BuildEdges.py
    python BuildEdges.py --text-edge-mode risk_similarity --text-threshold 0.90
    python BuildEdges.py --custom2-edge-mode cold_start

This script intentionally reads only data/sampled artifacts and the sampled
text embedding file. All edges are indexed by sampled_node_idx.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "edges"

RELATION_NAMES = {
    0: "R-U-R",
    1: "product_time_rating_burst",
    2: "template_text_relation",
}

RELATION1_NAMES = {
    "burst": "product_time_rating_burst",
    "product_prior_context": "product_prior_context",
    "none": "disabled_relation_1",
}

CUSTOM2_RELATION_NAMES = {
    "template": "template_text_relation",
    "risk_text": "template_text_relation",
    "cold_start": "cold_start_risk_cohort_edge",
    "weak_product_shock": "weak_product_rating_shock_edge",
}


def log(message: str) -> None:
    print(f"[BuildEdges] {message}")


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


def load_inputs(sampled_dir: Path, embedding_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    nodes_path = sampled_dir / "sampled_review_nodes.csv.gz"
    relation_path = sampled_dir / "sampled_relation_candidate_keys.csv.gz"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing sampled node file: {nodes_path}")
    if not relation_path.exists():
        raise FileNotFoundError(f"Missing sampled relation candidate file: {relation_path}")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing sampled text embedding file: {embedding_path}")

    log(f"Loading sampled nodes: {nodes_path}")
    nodes = pd.read_csv(nodes_path, parse_dates=["date"])
    log(f"Loading sampled relation keys: {relation_path}")
    relation_keys = pd.read_csv(relation_path, parse_dates=["date", "week"])
    embeddings = np.load(embedding_path, mmap_mode="r")

    nodes = nodes.sort_values("sampled_node_idx").reset_index(drop=True)
    relation_keys = relation_keys.sort_values("sampled_node_idx").reset_index(drop=True)
    validate_inputs(nodes, relation_keys, embeddings)
    return nodes, relation_keys, embeddings


def validate_inputs(nodes: pd.DataFrame, relation_keys: pd.DataFrame, embeddings: np.ndarray) -> None:
    node_required = [
        "sampled_node_idx",
        "user_id",
        "prod_id",
        "date",
        "week",
        "text_hash",
        "rating_direction",
        "short_review_flag",
        "extreme_rating",
    ]
    relation_required = [
        "sampled_node_idx",
        "user_id",
        "prod_id",
        "date",
        "week",
        "rating_direction",
        "rating_bucket",
    ]
    missing_node_columns = [col for col in node_required if col not in nodes.columns]
    missing_relation_columns = [col for col in relation_required if col not in relation_keys.columns]
    if missing_node_columns:
        raise ValueError(f"sampled_review_nodes.csv.gz is missing required columns: {missing_node_columns}")
    if missing_relation_columns:
        raise ValueError(
            f"sampled_relation_candidate_keys.csv.gz is missing required columns: {missing_relation_columns}"
        )

    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("sampled_review_nodes.csv.gz contains no rows.")
    if len(relation_keys) != n_nodes:
        raise ValueError(
            f"Relation candidate row count must match sampled nodes. relation={len(relation_keys)}, nodes={n_nodes}."
        )
    if embeddings.shape[0] != n_nodes:
        raise ValueError(f"Embedding row count must match sampled nodes. embeddings={embeddings.shape[0]}, nodes={n_nodes}.")

    expected_idx = np.arange(n_nodes, dtype=np.int64)
    for name, df in [("sampled_review_nodes", nodes), ("sampled_relation_candidate_keys", relation_keys)]:
        sampled_idx = df["sampled_node_idx"].to_numpy(dtype=np.int64)
        if not np.array_equal(sampled_idx, expected_idx):
            bad_positions = np.flatnonzero(sampled_idx != expected_idx)
            first_bad = int(bad_positions[0]) if len(bad_positions) else -1
            raise ValueError(
                f"{name} sampled_node_idx must be contiguous from 0 to n-1 after sorting. "
                f"First mismatch at row {first_bad}: expected {first_bad}, got {sampled_idx[first_bad]}."
            )

    if nodes["date"].isna().any() or relation_keys["date"].isna().any():
        raise ValueError("Sampled inputs contain invalid or missing date values.")
    if relation_keys["week"].isna().any():
        raise ValueError("sampled_relation_candidate_keys.csv.gz contains invalid or missing week values.")


def add_pair(
    edge_sets: dict[int, set[tuple[int, int]]],
    relation_type: int,
    src: int,
    dst: int,
) -> None:
    if src == dst:
        return
    edge_sets[relation_type].add((int(src), int(dst)))
    edge_sets[relation_type].add((int(dst), int(src)))


def add_directed_pair(
    edge_sets: dict[int, set[tuple[int, int]]],
    relation_type: int,
    src: int,
    dst: int,
) -> None:
    if src == dst:
        return
    edge_sets[relation_type].add((int(src), int(dst)))


def add_group_edges(
    edge_sets: dict[int, set[tuple[int, int]]],
    relation_type: int,
    group: pd.DataFrame,
    max_neighbors_per_node: int,
    sort_by_date: bool = True,
) -> None:
    if len(group) < 2:
        return

    if sort_by_date and "date" in group.columns:
        group = group.sort_values(["date", "sampled_node_idx"])
    else:
        group = group.sort_values("sampled_node_idx")

    node_ids = group["sampled_node_idx"].to_numpy(dtype=np.int64)
    group_size = len(node_ids)

    if group_size <= max_neighbors_per_node + 1:
        for i in range(group_size):
            for j in range(i + 1, group_size):
                add_pair(edge_sets, relation_type, int(node_ids[i]), int(node_ids[j]))
        return

    degree = defaultdict(int)
    for distance in range(1, group_size):
        any_added = False
        for i in range(group_size - distance):
            src = int(node_ids[i])
            dst = int(node_ids[i + distance])
            if degree[src] >= max_neighbors_per_node or degree[dst] >= max_neighbors_per_node:
                continue
            add_pair(edge_sets, relation_type, src, dst)
            degree[src] += 1
            degree[dst] += 1
            any_added = True
        if not any_added and all(degree[int(node)] >= max_neighbors_per_node for node in node_ids):
            break


def add_temporal_group_edges(
    edge_sets: dict[int, set[tuple[int, int]]],
    relation_type: int,
    group: pd.DataFrame,
    max_prior_neighbors: int,
    temporal_mode: str = "recent",
) -> None:
    if len(group) < 2:
        return
    if temporal_mode not in {"recent", "segmented"}:
        raise ValueError(f"Unsupported temporal edge mode: {temporal_mode}")
    ordered = group.sort_values(["date", "sampled_node_idx"])
    node_ids = ordered["sampled_node_idx"].to_numpy(dtype=np.int64)
    for i in range(1, len(node_ids)):
        prior_positions = select_temporal_prior_positions(i, max_prior_neighbors, temporal_mode)
        prior_ids = node_ids[prior_positions]
        for src in prior_ids:
            add_directed_pair(edge_sets, relation_type, int(src), int(node_ids[i]))


def select_temporal_prior_positions(
    current_position: int,
    max_prior_neighbors: int,
    temporal_mode: str,
) -> np.ndarray:
    if current_position <= 0 or max_prior_neighbors <= 0:
        return np.array([], dtype=np.int64)
    if temporal_mode == "recent":
        return np.arange(max(0, current_position - max_prior_neighbors), current_position, dtype=np.int64)
    if temporal_mode != "segmented":
        raise ValueError(f"Unsupported temporal edge mode: {temporal_mode}")

    prior_positions = np.arange(0, current_position, dtype=np.int64)
    if len(prior_positions) <= max_prior_neighbors:
        return prior_positions

    anchors = [int(prior_positions[-1])]
    if max_prior_neighbors >= 2:
        anchors.append(int(prior_positions[len(prior_positions) // 2]))
    if max_prior_neighbors >= 3:
        anchors.append(int(prior_positions[0]))
    if max_prior_neighbors > 3:
        extra = np.linspace(0, len(prior_positions) - 1, num=max_prior_neighbors, dtype=np.int64)
        anchors.extend(int(prior_positions[pos]) for pos in extra)

    deduped = sorted(set(anchors), key=anchors.index)[:max_prior_neighbors]
    return np.array(sorted(deduped), dtype=np.int64)


def build_user_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_neighbors_per_node: int,
    temporal: bool,
    temporal_mode: str,
) -> None:
    log(f"Building relation 0: R-U-R edges. temporal_mode={temporal_mode}")
    for _, group in nodes.groupby("user_id", sort=False):
        if temporal:
            add_temporal_group_edges(edge_sets, 0, group, max_neighbors_per_node, temporal_mode)
        else:
            add_group_edges(edge_sets, 0, group, max_neighbors_per_node=max_neighbors_per_node, sort_by_date=True)


def build_precomputed_context_edges(
    sampled_dir: Path,
    edge_sets: dict[int, set[tuple[int, int]]],
) -> dict[str, Any]:
    edge_path = sampled_dir / "sampled_context_edges.csv.gz"
    if not edge_path.exists():
        raise FileNotFoundError(f"--use-sampled-context-edges requires {edge_path}")

    log(f"Building relations from precomputed context edge plan: {edge_path}")
    context_edges = pd.read_csv(edge_path)
    required = ["src", "dst", "relation_type"]
    missing = [col for col in required if col not in context_edges.columns]
    if missing:
        raise ValueError(f"sampled_context_edges.csv.gz is missing required columns: {missing}")

    stats: dict[str, Any] = {
        "context_edge_plan_rows": int(len(context_edges)),
        "context_edge_plan_added": 0,
        "context_edge_plan_by_relation": {},
    }
    for row in context_edges.itertuples(index=False):
        relation_type = int(row.relation_type)
        if relation_type not in edge_sets:
            raise ValueError(f"Invalid relation_type in precomputed context edge plan: {relation_type}")
        before = len(edge_sets[relation_type])
        add_directed_pair(edge_sets, relation_type, int(row.src), int(row.dst))
        if len(edge_sets[relation_type]) > before:
            stats["context_edge_plan_added"] += 1
            key = str(relation_type)
            stats["context_edge_plan_by_relation"][key] = stats["context_edge_plan_by_relation"].get(key, 0) + 1
    return stats


def build_burst_edges(
    relation_keys: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_neighbors_per_node: int,
    burst_min_group_size: int,
) -> None:
    log("Building relation 1: product_time_rating_burst edges.")
    group_columns = ["prod_id", "week", "rating_bucket"]
    for _, group in relation_keys.groupby(group_columns, sort=False, observed=True):
        if len(group) < burst_min_group_size:
            continue
        add_group_edges(edge_sets, 1, group, max_neighbors_per_node=max_neighbors_per_node, sort_by_date=True)


def build_product_prior_context_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_prior_neighbors: int,
    exclude_same_week: bool,
) -> dict[str, Any]:
    log("Building relation 1: product_prior_context edges.")
    if max_prior_neighbors <= 0:
        raise ValueError(f"--product-context-topk must be positive, got {max_prior_neighbors}.")

    stats = {
        "product_context_edges_directed": 0,
        "product_context_filtered_same_week": 0,
        "product_context_topk": int(max_prior_neighbors),
        "product_context_exclude_same_week": bool(exclude_same_week),
    }

    for _, group in nodes.groupby("prod_id", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["date", "sampled_node_idx"])
        node_ids = ordered["sampled_node_idx"].to_numpy(dtype=np.int64)
        weeks = ordered["week"].to_numpy()
        for i in range(1, len(node_ids)):
            prior_positions = np.arange(0, i, dtype=np.int64)
            if exclude_same_week:
                same_week = weeks[prior_positions] == weeks[i]
                stats["product_context_filtered_same_week"] += int(np.sum(same_week))
                prior_positions = prior_positions[~same_week]
            if len(prior_positions) == 0:
                continue
            for pos in prior_positions[-max_prior_neighbors:]:
                before = len(edge_sets[1])
                add_directed_pair(edge_sets, 1, int(node_ids[pos]), int(node_ids[i]))
                if len(edge_sets[1]) > before:
                    stats["product_context_edges_directed"] += 1

    return stats


def build_exact_template_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_neighbors_per_node: int,
) -> None:
    log("Building relation 2: exact template repeat edges.")
    group_columns = ["prod_id", "week", "text_hash"]
    for _, group in nodes.groupby(group_columns, sort=False, observed=True):
        if len(group) < 2:
            continue
        add_group_edges(edge_sets, 2, group, max_neighbors_per_node=max_neighbors_per_node, sort_by_date=True)


def build_similarity_template_edges(
    nodes: pd.DataFrame,
    embeddings: np.ndarray,
    edge_sets: dict[int, set[tuple[int, int]]],
    template_threshold: float,
    template_topk: int,
) -> dict[str, int]:
    log("Building relation 2: similar template repeat edges.")
    stats = {
        "groups_considered": 0,
        "groups_skipped_too_large": 0,
        "groups_with_similarity_edges": 0,
    }
    if not 0 <= template_threshold <= 1:
        raise ValueError(f"--template-threshold must be between 0 and 1, got {template_threshold}.")
    if template_topk <= 0:
        raise ValueError(f"--template-topk must be positive, got {template_topk}.")

    for _, group in nodes.groupby(["prod_id", "week"], sort=False, observed=True):
        if len(group) < 2:
            continue
        stats["groups_considered"] += 1
        if len(group) > MAX_TEMPLATE_SIM_GROUP_SIZE:
            stats["groups_skipped_too_large"] += 1
            continue

        node_ids = group.sort_values(["date", "sampled_node_idx"])["sampled_node_idx"].to_numpy(dtype=np.int64)
        matrix = np.asarray(embeddings[node_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_matrix = matrix / np.maximum(norms, 1e-12)
        similarity = safe_matrix @ safe_matrix.T
        np.fill_diagonal(similarity, -np.inf)

        group_added = 0
        for row_idx, src in enumerate(node_ids):
            candidates = np.flatnonzero(similarity[row_idx] >= template_threshold)
            if len(candidates) == 0:
                continue
            ordered = candidates[np.argsort(similarity[row_idx, candidates])[::-1]][:template_topk]
            for col_idx in ordered:
                dst = int(node_ids[col_idx])
                before = len(edge_sets[2])
                add_pair(edge_sets, 2, int(src), dst)
                group_added += len(edge_sets[2]) - before
        if group_added:
            stats["groups_with_similarity_edges"] += 1

    return stats


def build_risk_text_similarity_edges(
    nodes: pd.DataFrame,
    embeddings: np.ndarray,
    edge_sets: dict[int, set[tuple[int, int]]],
    text_threshold: float,
    text_topk: int,
    text_candidate_policy: str,
    text_exclude_neutral: bool,
    allow_same_product_week: bool,
    text_search_multiplier: int,
) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    log("Building relation 2: text_similarity_risk_relation edges.")
    if not 0 <= text_threshold <= 1:
        raise ValueError(f"--text-threshold must be between 0 and 1, got {text_threshold}.")
    if text_topk <= 0:
        raise ValueError(f"--text-topk must be positive, got {text_topk}.")
    if text_search_multiplier <= 0:
        raise ValueError(f"--text-search-multiplier must be positive, got {text_search_multiplier}.")

    nodes = nodes.copy()
    nodes["is_risky_text_candidate"] = nodes["short_review_flag"].eq(1) | nodes["extreme_rating"].eq(1)
    risky_count = int(nodes["is_risky_text_candidate"].sum())
    similarities: list[float] = []
    text_edges_by_rating_direction = {"-1": 0, "0": 0, "1": 0}
    filtered_same_product_week_pairs = 0

    for rating_direction, group in nodes.groupby("rating_direction", sort=True):
        direction = int(rating_direction)
        if text_exclude_neutral and direction == 0:
            continue

        source_group = group.loc[group["is_risky_text_candidate"]].copy()
        if text_candidate_policy == "both_risky":
            target_group = source_group.copy()
        elif text_candidate_policy == "anchor_risky":
            target_group = group.copy()
        else:
            raise ValueError(f"Unsupported text candidate policy: {text_candidate_policy}")

        if len(source_group) == 0 or len(target_group) < 2:
            continue

        source_ids = source_group["sampled_node_idx"].to_numpy(dtype=np.int64)
        target_ids = target_group["sampled_node_idx"].to_numpy(dtype=np.int64)
        target_matrix = normalize(np.asarray(embeddings[target_ids], dtype=np.float32), norm="l2", copy=True)
        source_matrix = normalize(np.asarray(embeddings[source_ids], dtype=np.float32), norm="l2", copy=True)

        n_neighbors = min(len(target_ids), max(text_topk * text_search_multiplier + 1, text_topk + 1))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(target_matrix)
        distances, indices = nn.kneighbors(source_matrix, return_distance=True)

        group_meta = group.set_index("sampled_node_idx")[["prod_id", "week"]]
        for row_idx, src in enumerate(source_ids):
            src = int(src)
            src_prod = group_meta.at[src, "prod_id"]
            src_week = group_meta.at[src, "week"]
            added_for_source = 0

            for distance, target_pos in zip(distances[row_idx], indices[row_idx]):
                dst = int(target_ids[int(target_pos)])
                if src == dst:
                    continue
                similarity = 1.0 - float(distance)
                if similarity < text_threshold:
                    continue

                dst_prod = group_meta.at[dst, "prod_id"]
                dst_week = group_meta.at[dst, "week"]
                if not allow_same_product_week and src_prod == dst_prod and src_week == dst_week:
                    filtered_same_product_week_pairs += 1
                    continue

                before = len(edge_sets[2])
                add_pair(edge_sets, 2, src, dst)
                if len(edge_sets[2]) > before:
                    directed_added = len(edge_sets[2]) - before
                    text_edges_by_rating_direction[str(direction)] += directed_added
                    similarities.append(similarity)
                    added_for_source += 1
                if added_for_source >= text_topk:
                    break

    return {
        "risky_candidate_count": risky_count,
        "risky_candidate_share": float(risky_count / len(nodes)) if len(nodes) else 0.0,
        "text_edges_by_rating_direction": text_edges_by_rating_direction,
        "filtered_same_product_week_pairs": int(filtered_same_product_week_pairs),
        "avg_text_similarity_of_edges": float(np.mean(similarities)) if similarities else None,
        "min_text_similarity_of_edges": float(np.min(similarities)) if similarities else None,
        "max_text_similarity_of_edges": float(np.max(similarities)) if similarities else None,
    }


def cold_start_candidate_mask(nodes: pd.DataFrame, max_prior_user_reviews: int) -> pd.Series:
    if "prior_user_review_count" in nodes.columns:
        prior_user = pd.to_numeric(nodes["prior_user_review_count"], errors="coerce")
        low_prior = prior_user.le(max_prior_user_reviews).fillna(False)
    else:
        low_prior = pd.Series(False, index=nodes.index)

    if "is_new_user_at_review_time" in nodes.columns:
        new_user = pd.to_numeric(nodes["is_new_user_at_review_time"], errors="coerce").eq(1).fillna(False)
    else:
        new_user = pd.Series(False, index=nodes.index)

    risky_review = nodes["short_review_flag"].eq(1) | nodes["extreme_rating"].eq(1)
    return (low_prior | new_user) & risky_review


def build_behavior_matrix(candidates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    behavior_columns = [
        "short_review_flag",
        "extreme_rating",
        "is_new_user_at_review_time",
        "prior_user_review_count",
        "prior_product_review_count",
        "rating_deviation_from_prior_product_mean",
    ]
    used_columns = [col for col in behavior_columns if col in candidates.columns]
    if not used_columns:
        return np.zeros((len(candidates), 0), dtype=np.float32), used_columns

    values = candidates[used_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if values.size == 0:
        return values.reshape(len(candidates), 0), used_columns

    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)
    values = np.where(np.isnan(values), medians, values).astype(np.float32)
    scales = np.nanstd(values, axis=0).astype(np.float32)
    scales = np.where(scales < 1e-12, 1.0, scales).astype(np.float32)
    return ((values - medians) / scales).astype(np.float32), used_columns


def build_cold_start_risk_cohort_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_prior_user_reviews: int,
    date_window_days: int,
    topk: int,
    exclude_neutral: bool,
    allow_same_product_week: bool,
) -> dict[str, Any]:
    log("Building relation 2: cold_start_risk_cohort_edge edges.")
    if max_prior_user_reviews < 0:
        raise ValueError(
            "--cold-start-max-prior-user-reviews must be non-negative, "
            f"got {max_prior_user_reviews}."
        )
    if date_window_days < 0:
        raise ValueError(f"--cold-start-date-window-days must be non-negative, got {date_window_days}.")
    if topk <= 0:
        raise ValueError(f"--cold-start-topk must be positive, got {topk}.")

    candidate_mask = cold_start_candidate_mask(nodes, max_prior_user_reviews)
    if exclude_neutral:
        candidate_mask &= nodes["rating_direction"].ne(0)

    candidates = nodes.loc[candidate_mask].copy()
    candidate_count = int(len(candidates))
    stats: dict[str, Any] = {
        "cold_start_candidate_count": candidate_count,
        "cold_start_candidate_share": float(candidate_count / len(nodes)) if len(nodes) else 0.0,
        "cold_start_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "cold_start_filtered_same_user_pairs": 0,
        "cold_start_filtered_same_product_week_pairs": 0,
        "cold_start_behavior_columns": [],
    }
    if candidate_count < 2:
        return stats

    candidates = candidates.sort_values(["rating_direction", "date", "sampled_node_idx"]).reset_index(drop=True)
    behavior_matrix, behavior_columns = build_behavior_matrix(candidates)
    stats["cold_start_behavior_columns"] = behavior_columns

    for rating_direction, group in candidates.groupby("rating_direction", sort=True):
        direction = int(rating_direction)
        if exclude_neutral and direction == 0:
            continue
        if len(group) < 2:
            continue

        positions = group.index.to_numpy(dtype=np.int64)
        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        node_ids = group["sampled_node_idx"].to_numpy(dtype=np.int64)
        user_ids = group["user_id"].to_numpy()
        prod_ids = group["prod_id"].to_numpy()
        weeks = group["week"].to_numpy()
        window_ns = np.timedelta64(int(date_window_days), "D")

        for i, src in enumerate(node_ids):
            lower = dates[i] - window_ns
            upper = dates[i] + window_ns
            left = int(np.searchsorted(dates, lower, side="left"))
            right = int(np.searchsorted(dates, upper, side="right"))
            if right - left <= 1:
                continue

            local_idx = np.arange(left, right, dtype=np.int64)
            not_self = node_ids[local_idx] != int(src)
            same_user = user_ids[local_idx] == user_ids[i]
            stats["cold_start_filtered_same_user_pairs"] += int(np.sum(not_self & same_user))

            keep_mask = not_self & ~same_user
            if not allow_same_product_week:
                same_product_week = (prod_ids[local_idx] == prod_ids[i]) & (weeks[local_idx] == weeks[i])
                stats["cold_start_filtered_same_product_week_pairs"] += int(np.sum(keep_mask & same_product_week))
                keep_mask &= ~same_product_week

            target_idx = local_idx[keep_mask]
            if len(target_idx) == 0:
                continue

            date_diff = np.abs(dates[target_idx] - dates[i]).astype("timedelta64[D]").astype(np.int64)
            if behavior_matrix.shape[1]:
                behavior_distance = np.mean(
                    np.abs(behavior_matrix[positions[target_idx]] - behavior_matrix[positions[i]]),
                    axis=1,
                )
            else:
                behavior_distance = np.zeros(len(target_idx), dtype=np.float32)

            ordered = np.lexsort((node_ids[target_idx], behavior_distance, date_diff))[:topk]
            added_for_source = 0
            for target_pos in target_idx[ordered]:
                dst = int(node_ids[target_pos])
                before = len(edge_sets[2])
                add_pair(edge_sets, 2, int(src), int(dst))
                if len(edge_sets[2]) > before:
                    stats["cold_start_edges_by_rating_direction"][str(direction)] += len(edge_sets[2]) - before
                    added_for_source += 1
                if added_for_source >= topk:
                    break

    stats["cold_start_filtered_same_user_pairs"] = int(stats["cold_start_filtered_same_user_pairs"])
    stats["cold_start_filtered_same_product_week_pairs"] = int(stats["cold_start_filtered_same_product_week_pairs"])
    return stats


def shock_direction_from_deviation(deviation: pd.Series) -> pd.Series:
    return np.select(
        [deviation.gt(0), deviation.lt(0)],
        ["positive_shock", "negative_shock"],
        default="neutral_shock",
    )


SHOCK_BEHAVIOR_SCORE_COLUMN = "abuse_burst_behavior_score_4w"


def add_shock_scale_columns(work: pd.DataFrame, min_scale_floor: float = 1e-6) -> tuple[pd.DataFrame, float]:
    if "prior_product_rating_std" in work.columns:
        std_source = pd.to_numeric(work["prior_product_rating_std"], errors="coerce")
        positive = std_source.loc[std_source.gt(0)]
        std_floor = float(positive.quantile(0.25)) if not positive.empty else 1.0
    else:
        std_source = pd.Series(np.nan, index=work.index)
        std_floor = 1.0
    std_floor = max(std_floor, min_scale_floor)
    work["product_rating_shift_scale"] = std_source.fillna(std_floor).clip(lower=std_floor)
    work["standardized_abs_rating_deviation"] = (
        work["rating_deviation_numeric"].abs() / work["product_rating_shift_scale"]
    )
    return work, std_floor


def build_weak_product_rating_shock_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_prior_product_reviews: int,
    min_abs_rating_dev: float,
    shock_score_mode: str,
    date_window_days: int,
    topk: int,
    exclude_neutral: bool,
    allow_same_product_week: bool,
    same_product_only: bool,
    min_behavior_shift_score: float,
) -> dict[str, Any]:
    log("Building relation 2: weak_product_rating_shock_edge edges.")
    if max_prior_product_reviews < 0:
        raise ValueError(
            "--shock-max-prior-product-reviews must be non-negative, "
            f"got {max_prior_product_reviews}."
        )
    if min_abs_rating_dev < 0:
        raise ValueError(f"--shock-min-abs-rating-dev must be non-negative, got {min_abs_rating_dev}.")
    if shock_score_mode not in {"absolute", "standardized"}:
        raise ValueError(f"Unsupported --shock-score-mode: {shock_score_mode}")
    if date_window_days < 0:
        raise ValueError(f"--shock-date-window-days must be non-negative, got {date_window_days}.")
    if topk <= 0:
        raise ValueError(f"--shock-topk must be positive, got {topk}.")
    if min_behavior_shift_score < 0:
        raise ValueError(
            f"--shock-min-behavior-shift-score must be non-negative, got {min_behavior_shift_score}."
        )

    stats: dict[str, Any] = {
        "shock_candidate_count": 0,
        "shock_candidate_share": 0.0,
        "shock_behavior_score_column": SHOCK_BEHAVIOR_SCORE_COLUMN,
        "shock_behavior_score_available": bool(SHOCK_BEHAVIOR_SCORE_COLUMN in nodes.columns),
        "shock_min_behavior_shift_score": float(min_behavior_shift_score),
        "shock_score_mode": shock_score_mode,
        "shock_scale_floor": None,
        "shock_candidate_behavior_score_quantiles": {},
        "shock_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "shock_edges_by_shock_direction": {
            "negative_shock": 0,
            "neutral_shock": 0,
            "positive_shock": 0,
        },
        "shock_filtered_same_user_pairs": 0,
        "shock_filtered_same_product_week_pairs": 0,
    }

    required_columns = ["prior_product_review_count", "rating_deviation_from_prior_product_mean"]
    missing_columns = [col for col in required_columns if col not in nodes.columns]
    if missing_columns:
        log(f"Skipping weak_product_rating_shock_edge; missing columns: {missing_columns}")
        return stats

    nodes = nodes.copy()
    nodes["prior_product_review_count_numeric"] = pd.to_numeric(
        nodes["prior_product_review_count"], errors="coerce"
    )
    nodes["rating_deviation_numeric"] = pd.to_numeric(
        nodes["rating_deviation_from_prior_product_mean"], errors="coerce"
    )
    nodes["abs_rating_deviation"] = nodes["rating_deviation_numeric"].abs()
    nodes, shock_scale_floor = add_shock_scale_columns(nodes)
    stats["shock_scale_floor"] = float(shock_scale_floor)
    if SHOCK_BEHAVIOR_SCORE_COLUMN in nodes.columns:
        nodes["behavior_shift_score_numeric"] = pd.to_numeric(
            nodes[SHOCK_BEHAVIOR_SCORE_COLUMN], errors="coerce"
        ).fillna(0.0)
    else:
        nodes["behavior_shift_score_numeric"] = 0.0

    shock_score_col = "standardized_abs_rating_deviation" if shock_score_mode == "standardized" else "abs_rating_deviation"
    candidate_mask = (
        nodes["prior_product_review_count_numeric"].le(max_prior_product_reviews)
        & nodes[shock_score_col].ge(min_abs_rating_dev)
    )
    candidate_mask = candidate_mask.fillna(False)
    if exclude_neutral:
        candidate_mask &= nodes["rating_direction"].ne(0)
    if min_behavior_shift_score > 0:
        candidate_mask &= nodes["behavior_shift_score_numeric"].ge(min_behavior_shift_score)

    candidates = nodes.loc[candidate_mask].copy()
    candidates["shock_direction"] = shock_direction_from_deviation(candidates["rating_deviation_numeric"])
    candidate_count = int(len(candidates))
    stats["shock_candidate_count"] = candidate_count
    stats["shock_candidate_share"] = float(candidate_count / len(nodes)) if len(nodes) else 0.0
    if candidate_count:
        stats["shock_candidate_behavior_score_quantiles"] = {
            str(q): float(v)
            for q, v in candidates["behavior_shift_score_numeric"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).items()
        }
    if candidate_count < 2:
        return stats

    sort_columns = ["shock_direction", "rating_direction", "date", "sampled_node_idx"]
    candidates = candidates.sort_values(sort_columns).reset_index(drop=True)
    window_ns = np.timedelta64(int(date_window_days), "D")

    for (shock_direction, rating_direction), group in candidates.groupby(
        ["shock_direction", "rating_direction"], sort=True
    ):
        direction = int(rating_direction)
        if exclude_neutral and direction == 0:
            continue
        if len(group) < 2:
            continue

        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        node_ids = group["sampled_node_idx"].to_numpy(dtype=np.int64)
        user_ids = group["user_id"].to_numpy()
        prod_ids = group["prod_id"].to_numpy()
        weeks = group["week"].to_numpy()
        abs_devs = group["abs_rating_deviation"].to_numpy(dtype=np.float32)
        shock_scores = group[shock_score_col].to_numpy(dtype=np.float32)
        prior_product_counts = group["prior_product_review_count_numeric"].to_numpy(dtype=np.float32)
        behavior_scores = group["behavior_shift_score_numeric"].to_numpy(dtype=np.float32)

        for i, src in enumerate(node_ids):
            lower = dates[i] - window_ns
            upper = dates[i] + window_ns
            left = int(np.searchsorted(dates, lower, side="left"))
            right = int(np.searchsorted(dates, upper, side="right"))
            if right - left <= 1:
                continue

            local_idx = np.arange(left, right, dtype=np.int64)
            not_self = node_ids[local_idx] != int(src)
            same_user = user_ids[local_idx] == user_ids[i]
            stats["shock_filtered_same_user_pairs"] += int(np.sum(not_self & same_user))

            keep_mask = not_self & ~same_user
            if same_product_only:
                same_product = prod_ids[local_idx] == prod_ids[i]
                same_product_week = same_product & (weeks[local_idx] == weeks[i])
                stats["shock_filtered_same_product_week_pairs"] += int(np.sum(keep_mask & same_product_week))
                keep_mask &= same_product & ~same_product_week
            elif not allow_same_product_week:
                same_product_week = (prod_ids[local_idx] == prod_ids[i]) & (weeks[local_idx] == weeks[i])
                stats["shock_filtered_same_product_week_pairs"] += int(np.sum(keep_mask & same_product_week))
                keep_mask &= ~same_product_week

            target_idx = local_idx[keep_mask]
            if len(target_idx) == 0:
                continue

            abs_dev_diff = np.abs(abs_devs[target_idx] - abs_devs[i])
            shock_score_diff = np.abs(shock_scores[target_idx] - shock_scores[i])
            prior_count_diff = np.abs(prior_product_counts[target_idx] - prior_product_counts[i])
            date_diff = np.abs(dates[target_idx] - dates[i]).astype("timedelta64[D]").astype(np.int64)
            pair_behavior_score = np.minimum(behavior_scores[target_idx], behavior_scores[i])
            ordered = np.lexsort(
                (node_ids[target_idx], date_diff, prior_count_diff, abs_dev_diff, shock_score_diff, -pair_behavior_score)
            )[:topk]

            added_for_source = 0
            for target_pos in target_idx[ordered]:
                dst = int(node_ids[target_pos])
                before = len(edge_sets[2])
                add_pair(edge_sets, 2, int(src), dst)
                if len(edge_sets[2]) > before:
                    directed_added = len(edge_sets[2]) - before
                    stats["shock_edges_by_rating_direction"][str(direction)] += directed_added
                    stats["shock_edges_by_shock_direction"][str(shock_direction)] += directed_added
                    added_for_source += 1
                if added_for_source >= topk:
                    break

    stats["shock_filtered_same_user_pairs"] = int(stats["shock_filtered_same_user_pairs"])
    stats["shock_filtered_same_product_week_pairs"] = int(stats["shock_filtered_same_product_week_pairs"])
    return stats


def build_product_baseline_shock_edges(
    nodes: pd.DataFrame,
    edge_sets: dict[int, set[tuple[int, int]]],
    max_prior_product_reviews: int,
    min_abs_rating_dev: float,
    shock_score_mode: str,
    topk: int,
    exclude_neutral: bool,
    exclude_same_week: bool,
    temporal_mode: str,
) -> dict[str, Any]:
    log("Building relation 2: product_baseline_rating_shock_edge edges.")
    if max_prior_product_reviews < 0:
        raise ValueError(
            "--shock-max-prior-product-reviews must be non-negative, "
            f"got {max_prior_product_reviews}."
        )
    if min_abs_rating_dev < 0:
        raise ValueError(f"--shock-min-abs-rating-dev must be non-negative, got {min_abs_rating_dev}.")
    if shock_score_mode not in {"absolute", "standardized"}:
        raise ValueError(f"Unsupported --shock-score-mode: {shock_score_mode}")
    if topk <= 0:
        raise ValueError(f"--shock-baseline-topk must be positive, got {topk}.")
    if temporal_mode not in {"recent", "segmented"}:
        raise ValueError(f"Unsupported --shock-baseline-temporal-mode: {temporal_mode}")

    stats: dict[str, Any] = {
        "shock_candidate_count": 0,
        "shock_candidate_share": 0.0,
        "shock_behavior_score_column": SHOCK_BEHAVIOR_SCORE_COLUMN,
        "shock_behavior_score_available": bool(SHOCK_BEHAVIOR_SCORE_COLUMN in nodes.columns),
        "shock_min_behavior_shift_score": 0.0,
        "shock_score_mode": shock_score_mode,
        "shock_scale_floor": None,
        "shock_candidate_behavior_score_quantiles": {},
        "shock_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "shock_edges_by_shock_direction": {
            "negative_shock": 0,
            "neutral_shock": 0,
            "positive_shock": 0,
        },
        "shock_filtered_same_user_pairs": 0,
        "shock_filtered_same_product_week_pairs": 0,
        "shock_baseline_topk": int(topk),
        "shock_baseline_exclude_same_week": bool(exclude_same_week),
        "shock_baseline_temporal_mode": temporal_mode,
    }

    required_columns = ["prior_product_review_count", "rating_deviation_from_prior_product_mean"]
    missing_columns = [col for col in required_columns if col not in nodes.columns]
    if missing_columns:
        log(f"Skipping product_baseline_rating_shock_edge; missing columns: {missing_columns}")
        return stats

    work = nodes.copy()
    work["prior_product_review_count_numeric"] = pd.to_numeric(
        work["prior_product_review_count"], errors="coerce"
    )
    work["rating_deviation_numeric"] = pd.to_numeric(
        work["rating_deviation_from_prior_product_mean"], errors="coerce"
    )
    work["abs_rating_deviation"] = work["rating_deviation_numeric"].abs()
    work, shock_scale_floor = add_shock_scale_columns(work)
    stats["shock_scale_floor"] = float(shock_scale_floor)
    shock_score_col = "standardized_abs_rating_deviation" if shock_score_mode == "standardized" else "abs_rating_deviation"
    candidate_mask = (
        work["prior_product_review_count_numeric"].between(1, max_prior_product_reviews)
        & work[shock_score_col].ge(min_abs_rating_dev)
    )
    candidate_mask = candidate_mask.fillna(False)
    if exclude_neutral:
        candidate_mask &= work["rating_direction"].ne(0)

    work["shock_direction"] = shock_direction_from_deviation(work["rating_deviation_numeric"])
    candidate_ids = set(work.loc[candidate_mask, "sampled_node_idx"].astype(int).tolist())
    stats["shock_candidate_count"] = int(len(candidate_ids))
    stats["shock_candidate_share"] = float(len(candidate_ids) / len(work)) if len(work) else 0.0
    if not candidate_ids:
        return stats

    for _, group in work.groupby("prod_id", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["date", "sampled_node_idx"])
        node_ids = ordered["sampled_node_idx"].astype(int).to_numpy()
        weeks = ordered["week"].to_numpy()
        users = ordered["user_id"].to_numpy()
        rating_dirs = ordered["rating_direction"].astype(int).to_numpy()
        shock_dirs = ordered["shock_direction"].astype(str).to_numpy()
        for i, dst in enumerate(node_ids):
            if int(dst) not in candidate_ids or i == 0:
                continue
            prior_positions = np.arange(0, i, dtype=np.int64)
            same_user = users[prior_positions] == users[i]
            stats["shock_filtered_same_user_pairs"] += int(np.sum(same_user))
            prior_positions = prior_positions[~same_user]
            if exclude_same_week:
                same_week = weeks[prior_positions] == weeks[i]
                stats["shock_filtered_same_product_week_pairs"] += int(np.sum(same_week))
                prior_positions = prior_positions[~same_week]
            if len(prior_positions) == 0:
                continue
            direction = int(rating_dirs[i])
            shock_direction = str(shock_dirs[i])
            if temporal_mode == "recent":
                selected_positions = prior_positions[-topk:]
            else:
                selected_offsets = select_temporal_prior_positions(len(prior_positions), topk, temporal_mode)
                selected_positions = prior_positions[selected_offsets]
            for pos in selected_positions:
                before = len(edge_sets[2])
                add_directed_pair(edge_sets, 2, int(node_ids[pos]), int(dst))
                if len(edge_sets[2]) > before:
                    stats["shock_edges_by_rating_direction"][str(direction)] += 1
                    stats["shock_edges_by_shock_direction"][shock_direction] += 1

    stats["shock_filtered_same_user_pairs"] = int(stats["shock_filtered_same_user_pairs"])
    stats["shock_filtered_same_product_week_pairs"] = int(stats["shock_filtered_same_product_week_pairs"])
    return stats


def collapse_duplicate_pairs_across_relations(
    edge_sets: dict[int, set[tuple[int, int]]],
) -> tuple[dict[int, set[tuple[int, int]]], int]:
    pair_to_relation: dict[tuple[int, int], int] = {}
    relation_priority = [0, 1, 2]

    for relation_type in relation_priority:
        for src, dst in edge_sets[relation_type]:
            if src == dst:
                continue
            pair_to_relation[(int(src), int(dst))] = relation_type

    collapsed: dict[int, set[tuple[int, int]]] = {relation_type: set() for relation_type in RELATION_NAMES}
    for pair, relation_type in pair_to_relation.items():
        collapsed[relation_type].add(pair)

    original_directed = sum(len(pairs) for pairs in edge_sets.values())
    collapsed_directed = sum(len(pairs) for pairs in collapsed.values())
    removed = original_directed - collapsed_directed
    return collapsed, int(removed)


def build_edge_arrays(edge_sets: dict[int, set[tuple[int, int]]], n_nodes: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    rows: list[tuple[int, int, int]] = []
    count_rows: list[dict[str, Any]] = []
    for relation_type in sorted(RELATION_NAMES):
        pairs = sorted(edge_sets[relation_type])
        count_rows.append(
            {
                "edge_type": relation_type,
                "relation": RELATION_NAMES[relation_type],
                "directed_edges": len(pairs),
                "undirected_edges": len(pairs) // 2,
            }
        )
        rows.extend((src, dst, relation_type) for src, dst in pairs)

    if rows:
        edge_df = pd.DataFrame(rows, columns=["src", "dst", "edge_type"])
        edge_index = edge_df[["src", "dst"]].to_numpy(dtype=np.int64).T
        edge_type = edge_df["edge_type"].to_numpy(dtype=np.int64)
    else:
        edge_df = pd.DataFrame(columns=["src", "dst", "edge_type"])
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_type = np.empty((0,), dtype=np.int64)

    validate_edges(edge_index, edge_type, n_nodes)
    return edge_index, edge_type, pd.DataFrame(count_rows), edge_df


def validate_edges(edge_index: np.ndarray, edge_type: np.ndarray, n_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}.")
    if edge_type.ndim != 1:
        raise ValueError(f"edge_type must be 1D, got {edge_type.shape}.")
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError(
            f"edge_index and edge_type length mismatch. edges={edge_index.shape[1]}, types={edge_type.shape[0]}."
        )
    if edge_index.shape[1] == 0:
        raise ValueError("No edges were created. Check relation thresholds and sampled inputs.")
    if int(edge_index.min()) < 0:
        raise ValueError(f"edge_index minimum must be >= 0, got {int(edge_index.min())}.")
    if int(edge_index.max()) > n_nodes - 1:
        raise ValueError(f"edge_index maximum must be <= {n_nodes - 1}, got {int(edge_index.max())}.")
    invalid_types = sorted(set(edge_type.tolist()) - set(RELATION_NAMES))
    if invalid_types:
        raise ValueError(f"edge_type contains invalid values: {invalid_types}")
    self_loops = int(np.sum(edge_index[0] == edge_index[1]))
    if self_loops:
        raise ValueError(f"edge_index contains {self_loops} self-loops.")

    edge_df = pd.DataFrame({"src": edge_index[0], "dst": edge_index[1], "edge_type": edge_type})
    duplicate_count = int(edge_df.duplicated(["src", "dst", "edge_type"]).sum())
    if duplicate_count:
        raise ValueError(f"edge_index contains {duplicate_count} duplicate directed edges within relation.")


def relation_stats(edge_sets: dict[int, set[tuple[int, int]]], n_nodes: int) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for relation_type, relation_name in RELATION_NAMES.items():
        pairs = edge_sets[relation_type]
        degree = np.zeros(n_nodes, dtype=np.int64)
        for src, dst in pairs:
            degree[src] += 1
            degree[dst] += 1
        summary[str(relation_type)] = {
            "relation": relation_name,
            "directed_edges": int(len(pairs)),
            "undirected_edges": int(len(pairs) // 2),
            "mean_degree": float(degree.mean()),
            "isolated_nodes": int(np.sum(degree == 0)),
        }
    return summary


def save_outputs(
    output_dir: Path,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_df: pd.DataFrame,
    relation_counts: pd.DataFrame,
    relation_summary: dict[str, dict[str, Any]],
    n_nodes: int,
    duplicate_pairs_removed: int,
    text_relation_stats: dict[str, Any],
    cold_start_stats: dict[str, Any],
    shock_stats: dict[str, Any],
    context_edge_stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_index_path = output_dir / "edge_index.npy"
    edge_type_path = output_dir / "edge_type.npy"
    summary_path = output_dir / "edges_summary.json"
    counts_path = output_dir / "relation_edge_counts.csv"
    csv_path = output_dir / "edges.csv.gz"
    parquet_path = output_dir / "edges.parquet"

    np.save(edge_index_path, edge_index.astype(np.int64, copy=False))
    np.save(edge_type_path, edge_type.astype(np.int64, copy=False))
    relation_counts.to_csv(counts_path, index=False, encoding="utf-8")
    edge_df.to_csv(csv_path, index=False, encoding="utf-8", compression="gzip")

    parquet_saved = False
    parquet_error = None
    try:
        edge_df.to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception as exc:
        parquet_error = str(exc)
        log(f"Could not save parquet; saved CSV instead. Reason: {parquet_error}")

    total_degree = np.zeros(n_nodes, dtype=np.int64)
    for src in edge_index[0]:
        total_degree[int(src)] += 1
    for dst in edge_index[1]:
        total_degree[int(dst)] += 1

    summary = {
        "n_nodes": int(n_nodes),
        "total_directed_edges": int(edge_index.shape[1]),
        "total_undirected_edges": int(edge_index.shape[1] // 2),
        "edge_index_shape": [int(edge_index.shape[0]), int(edge_index.shape[1])],
        "edge_type_shape": [int(edge_type.shape[0])],
        "relation_stats": relation_summary,
        "relation1_mode": args.relation1_mode,
        "custom2_edge_mode": args.custom2_edge_mode,
        "rur_temporal": bool(args.rur_temporal),
        "rur_temporal_mode": args.rur_temporal_mode,
        "use_sampled_context_edges": bool(args.use_sampled_context_edges),
        "sampled_context_edge_plan": context_edge_stats,
        "cold_start_max_prior_user_reviews": int(args.cold_start_max_prior_user_reviews),
        "cold_start_date_window_days": int(args.cold_start_date_window_days),
        "cold_start_topk": int(args.cold_start_topk),
        "cold_start_exclude_neutral": bool(args.cold_start_exclude_neutral),
        "cold_start_allow_same_product_week": bool(args.cold_start_allow_same_product_week),
        "cold_start_candidate_count": int(cold_start_stats.get("cold_start_candidate_count", 0)),
        "cold_start_candidate_share": float(cold_start_stats.get("cold_start_candidate_share", 0.0)),
        "cold_start_edges_directed": int(relation_summary["2"]["directed_edges"])
        if args.custom2_edge_mode == "cold_start"
        else 0,
        "cold_start_edges_undirected": int(relation_summary["2"]["undirected_edges"])
        if args.custom2_edge_mode == "cold_start"
        else 0,
        "cold_start_edges_by_rating_direction": cold_start_stats.get(
            "cold_start_edges_by_rating_direction", {"-1": 0, "0": 0, "1": 0}
        ),
        "cold_start_filtered_same_user_pairs": int(cold_start_stats.get("cold_start_filtered_same_user_pairs", 0)),
        "cold_start_filtered_same_product_week_pairs": int(
            cold_start_stats.get("cold_start_filtered_same_product_week_pairs", 0)
        ),
        "shock_max_prior_product_reviews": int(args.shock_max_prior_product_reviews),
        "shock_min_abs_rating_dev": float(args.shock_min_abs_rating_dev),
        "shock_score_mode": args.shock_score_mode,
        "shock_date_window_days": int(args.shock_date_window_days),
        "shock_topk": int(args.shock_topk),
        "shock_exclude_neutral": bool(args.shock_exclude_neutral),
        "shock_allow_same_product_week": bool(args.shock_allow_same_product_week),
        "shock_same_product_only": bool(args.shock_same_product_only),
        "shock_min_behavior_shift_score": float(args.shock_min_behavior_shift_score),
        "shock_behavior_score_column": shock_stats.get("shock_behavior_score_column", SHOCK_BEHAVIOR_SCORE_COLUMN),
        "shock_behavior_score_available": bool(shock_stats.get("shock_behavior_score_available", False)),
        "shock_candidate_behavior_score_quantiles": shock_stats.get("shock_candidate_behavior_score_quantiles", {}),
        "shock_candidate_count": int(shock_stats.get("shock_candidate_count", 0)),
        "shock_candidate_share": float(shock_stats.get("shock_candidate_share", 0.0)),
        "shock_edges_directed": int(relation_summary["2"]["directed_edges"])
        if args.custom2_edge_mode == "weak_product_shock"
        else 0,
        "shock_edges_undirected": int(relation_summary["2"]["undirected_edges"])
        if args.custom2_edge_mode == "weak_product_shock"
        else 0,
        "shock_edges_by_rating_direction": shock_stats.get("shock_edges_by_rating_direction", {"-1": 0, "0": 0, "1": 0}),
        "shock_edges_by_shock_direction": shock_stats.get(
            "shock_edges_by_shock_direction",
            {"negative_shock": 0, "neutral_shock": 0, "positive_shock": 0},
        ),
        "shock_filtered_same_user_pairs": int(shock_stats.get("shock_filtered_same_user_pairs", 0)),
        "shock_filtered_same_product_week_pairs": int(
            shock_stats.get("shock_filtered_same_product_week_pairs", 0)
        ),
        "overall_mean_degree": float(total_degree.mean()),
        "overall_isolated_nodes": int(np.sum(total_degree == 0)),
        "duplicate_pairs_removed_across_relations": int(duplicate_pairs_removed),
        "cross_relation_duplicate_policy": "When the same directed sampled-node pair appears in multiple relations, keep the highest-priority relation: 2 custom edge > 1 relation > 0 R-U-R.",
        "parameters": {
            "max_neighbors_per_node": int(args.max_neighbors_per_node),
            "burst_min_group_size": int(args.burst_min_group_size),
            "use_sampled_context_edges": bool(args.use_sampled_context_edges),
            "relation1_mode": args.relation1_mode,
            "product_context_topk": int(args.product_context_topk),
            "product_context_exclude_same_week": bool(args.product_context_exclude_same_week),
            "rur_temporal": bool(args.rur_temporal),
            "rur_temporal_mode": args.rur_temporal_mode,
            "custom2_edge_mode": args.custom2_edge_mode,
            "shock_edge_style": args.shock_edge_style,
            "text_edge_mode": args.text_edge_mode,
            "text_threshold": float(args.text_threshold),
            "text_topk": int(args.text_topk),
            "text_candidate_policy": args.text_candidate_policy,
            "text_exclude_neutral": bool(args.text_exclude_neutral),
            "allow_same_product_week": bool(args.allow_same_product_week),
            "text_search_multiplier": int(args.text_search_multiplier),
            "cold_start_max_prior_user_reviews": int(args.cold_start_max_prior_user_reviews),
            "cold_start_date_window_days": int(args.cold_start_date_window_days),
            "cold_start_topk": int(args.cold_start_topk),
            "cold_start_exclude_neutral": bool(args.cold_start_exclude_neutral),
            "cold_start_allow_same_product_week": bool(args.cold_start_allow_same_product_week),
            "shock_max_prior_product_reviews": int(args.shock_max_prior_product_reviews),
            "shock_min_abs_rating_dev": float(args.shock_min_abs_rating_dev),
            "shock_score_mode": args.shock_score_mode,
            "shock_date_window_days": int(args.shock_date_window_days),
            "shock_topk": int(args.shock_topk),
            "shock_exclude_neutral": bool(args.shock_exclude_neutral),
            "shock_allow_same_product_week": bool(args.shock_allow_same_product_week),
            "shock_same_product_only": bool(args.shock_same_product_only),
            "shock_min_behavior_shift_score": float(args.shock_min_behavior_shift_score),
            "shock_baseline_topk": int(args.shock_baseline_topk),
            "shock_baseline_exclude_same_week": bool(args.shock_baseline_exclude_same_week),
            "shock_baseline_temporal_mode": args.shock_baseline_temporal_mode,
        },
        "product_prior_context_relation": {
            "relation1_mode": args.relation1_mode,
            "product_context_topk": int(args.product_context_topk),
            "product_context_exclude_same_week": bool(args.product_context_exclude_same_week),
            "product_context_edges_directed": int(relation_summary["1"]["directed_edges"])
            if args.relation1_mode == "product_prior_context"
            else 0,
        },
        "text_relation": {
            "text_edge_mode": args.text_edge_mode,
            "text_threshold": float(args.text_threshold),
            "text_topk": int(args.text_topk),
            "text_candidate_policy": args.text_candidate_policy,
            "text_exclude_neutral": bool(args.text_exclude_neutral),
            "allow_same_product_week": bool(args.allow_same_product_week),
            "risky_candidate_count": int(text_relation_stats.get("risky_candidate_count", 0)),
            "risky_candidate_share": float(text_relation_stats.get("risky_candidate_share", 0.0)),
            "text_edges_directed": int(relation_summary["2"]["directed_edges"]),
            "text_edges_undirected": int(relation_summary["2"]["undirected_edges"]),
            "text_edges_by_rating_direction": text_relation_stats.get("text_edges_by_rating_direction", {"-1": 0, "0": 0, "1": 0}),
            "filtered_same_product_week_pairs": int(text_relation_stats.get("filtered_same_product_week_pairs", 0)),
            "avg_text_similarity_of_edges": text_relation_stats.get("avg_text_similarity_of_edges"),
            "min_text_similarity_of_edges": text_relation_stats.get("min_text_similarity_of_edges"),
            "max_text_similarity_of_edges": text_relation_stats.get("max_text_similarity_of_edges"),
        },
        "cold_start_relation": {
            "custom2_edge_mode": args.custom2_edge_mode,
            "cold_start_max_prior_user_reviews": int(args.cold_start_max_prior_user_reviews),
            "cold_start_date_window_days": int(args.cold_start_date_window_days),
            "cold_start_topk": int(args.cold_start_topk),
            "cold_start_exclude_neutral": bool(args.cold_start_exclude_neutral),
            "cold_start_allow_same_product_week": bool(args.cold_start_allow_same_product_week),
            "cold_start_candidate_count": int(cold_start_stats.get("cold_start_candidate_count", 0)),
            "cold_start_candidate_share": float(cold_start_stats.get("cold_start_candidate_share", 0.0)),
            "cold_start_edges_directed": int(relation_summary["2"]["directed_edges"])
            if args.custom2_edge_mode == "cold_start"
            else 0,
            "cold_start_edges_undirected": int(relation_summary["2"]["undirected_edges"])
            if args.custom2_edge_mode == "cold_start"
            else 0,
            "cold_start_edges_by_rating_direction": cold_start_stats.get(
                "cold_start_edges_by_rating_direction", {"-1": 0, "0": 0, "1": 0}
            ),
            "cold_start_filtered_same_user_pairs": int(
                cold_start_stats.get("cold_start_filtered_same_user_pairs", 0)
            ),
            "cold_start_filtered_same_product_week_pairs": int(
                cold_start_stats.get("cold_start_filtered_same_product_week_pairs", 0)
            ),
            "cold_start_behavior_columns": cold_start_stats.get("cold_start_behavior_columns", []),
        },
        "weak_product_shock_relation": {
            "custom2_edge_mode": args.custom2_edge_mode,
            "shock_edge_style": args.shock_edge_style,
            "shock_max_prior_product_reviews": int(args.shock_max_prior_product_reviews),
            "shock_min_abs_rating_dev": float(args.shock_min_abs_rating_dev),
            "shock_score_mode": args.shock_score_mode,
            "shock_date_window_days": int(args.shock_date_window_days),
            "shock_topk": int(args.shock_topk),
            "shock_exclude_neutral": bool(args.shock_exclude_neutral),
            "shock_allow_same_product_week": bool(args.shock_allow_same_product_week),
            "shock_same_product_only": bool(args.shock_same_product_only),
            "shock_min_behavior_shift_score": float(args.shock_min_behavior_shift_score),
            "shock_baseline_topk": int(args.shock_baseline_topk),
            "shock_baseline_exclude_same_week": bool(args.shock_baseline_exclude_same_week),
            "shock_baseline_temporal_mode": args.shock_baseline_temporal_mode,
            "shock_behavior_score_column": shock_stats.get("shock_behavior_score_column", SHOCK_BEHAVIOR_SCORE_COLUMN),
            "shock_behavior_score_available": bool(shock_stats.get("shock_behavior_score_available", False)),
            "shock_candidate_behavior_score_quantiles": shock_stats.get("shock_candidate_behavior_score_quantiles", {}),
            "shock_candidate_count": int(shock_stats.get("shock_candidate_count", 0)),
            "shock_candidate_share": float(shock_stats.get("shock_candidate_share", 0.0)),
            "shock_edges_directed": int(relation_summary["2"]["directed_edges"])
            if args.custom2_edge_mode == "weak_product_shock"
            else 0,
            "shock_edges_undirected": int(relation_summary["2"]["undirected_edges"])
            if args.custom2_edge_mode == "weak_product_shock"
            else 0,
            "shock_edges_by_rating_direction": shock_stats.get(
                "shock_edges_by_rating_direction", {"-1": 0, "0": 0, "1": 0}
            ),
            "shock_edges_by_shock_direction": shock_stats.get(
                "shock_edges_by_shock_direction",
                {"negative_shock": 0, "neutral_shock": 0, "positive_shock": 0},
            ),
            "shock_filtered_same_user_pairs": int(shock_stats.get("shock_filtered_same_user_pairs", 0)),
            "shock_filtered_same_product_week_pairs": int(
                shock_stats.get("shock_filtered_same_product_week_pairs", 0)
            ),
        },
        "inputs": {
            "sampled_nodes": path_for_summary(args.sampled_dir / "sampled_review_nodes.csv.gz"),
            "sampled_relation_keys": path_for_summary(args.sampled_dir / "sampled_relation_candidate_keys.csv.gz"),
            "text_embeddings": path_for_summary(args.embedding_path),
        },
        "outputs": {
            "edge_index": path_for_summary(edge_index_path),
            "edge_type": path_for_summary(edge_type_path),
            "relation_edge_counts": path_for_summary(counts_path),
            "edges_csv_gz": path_for_summary(csv_path),
            "edges_parquet": path_for_summary(parquet_path) if parquet_saved else None,
            "summary": path_for_summary(summary_path),
        },
        "notes": [
            "Only sampled artifacts are used.",
            "All edges use sampled_node_idx.",
            "Self-loops are removed.",
            "R-U-R, product context, and baseline shock edges may be directed when temporal options are enabled.",
            "Duplicate directed edges are removed within each relation.",
            "is_fake, label and tag are not used as edge construction criteria.",
        ],
    }
    if parquet_error:
        summary["parquet_error"] = parquet_error

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    log(f"Saved edge_index: {edge_index_path}")
    log(f"Saved edge_type: {edge_type_path}")
    log(f"Saved summary: {summary_path}")


def run_build_edges(args: argparse.Namespace) -> None:
    if args.max_neighbors_per_node <= 0:
        raise ValueError(f"--max-neighbors-per-node must be positive, got {args.max_neighbors_per_node}.")
    if args.burst_min_group_size < 2:
        raise ValueError(f"--burst-min-group-size must be at least 2, got {args.burst_min_group_size}.")

    RELATION_NAMES[1] = RELATION1_NAMES[args.relation1_mode]
    RELATION_NAMES[2] = CUSTOM2_RELATION_NAMES[args.custom2_edge_mode]
    if args.custom2_edge_mode == "weak_product_shock" and args.shock_edge_style == "baseline_context":
        RELATION_NAMES[2] = "product_baseline_rating_shock_edge"
    nodes, relation_keys, embeddings = load_inputs(args.sampled_dir, args.embedding_path)
    n_nodes = len(nodes)
    edge_sets: dict[int, set[tuple[int, int]]] = {relation_type: set() for relation_type in RELATION_NAMES}

    context_edge_stats: dict[str, Any] = {}
    if args.use_sampled_context_edges:
        context_edge_stats = build_precomputed_context_edges(args.sampled_dir, edge_sets)
    else:
        build_user_edges(nodes, edge_sets, args.max_neighbors_per_node, args.rur_temporal, args.rur_temporal_mode)
        if args.relation1_mode == "burst":
            build_burst_edges(relation_keys, edge_sets, args.max_neighbors_per_node, args.burst_min_group_size)
        elif args.relation1_mode == "product_prior_context":
            build_product_prior_context_edges(
                nodes,
                edge_sets,
                args.product_context_topk,
                args.product_context_exclude_same_week,
            )
    risky_mask = nodes["short_review_flag"].eq(1) | nodes["extreme_rating"].eq(1)
    text_relation_stats: dict[str, Any] = {
        "risky_candidate_count": int(risky_mask.sum()),
        "risky_candidate_share": float(risky_mask.mean()),
        "text_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "filtered_same_product_week_pairs": 0,
        "avg_text_similarity_of_edges": None,
        "min_text_similarity_of_edges": None,
        "max_text_similarity_of_edges": None,
    }
    cold_start_stats: dict[str, Any] = {
        "cold_start_candidate_count": 0,
        "cold_start_candidate_share": 0.0,
        "cold_start_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "cold_start_filtered_same_user_pairs": 0,
        "cold_start_filtered_same_product_week_pairs": 0,
        "cold_start_behavior_columns": [],
    }
    shock_stats: dict[str, Any] = {
        "shock_candidate_count": 0,
        "shock_candidate_share": 0.0,
        "shock_behavior_score_column": SHOCK_BEHAVIOR_SCORE_COLUMN,
        "shock_behavior_score_available": False,
        "shock_min_behavior_shift_score": float(args.shock_min_behavior_shift_score),
        "shock_candidate_behavior_score_quantiles": {},
        "shock_edges_by_rating_direction": {"-1": 0, "0": 0, "1": 0},
        "shock_edges_by_shock_direction": {
            "negative_shock": 0,
            "neutral_shock": 0,
            "positive_shock": 0,
        },
        "shock_filtered_same_user_pairs": 0,
        "shock_filtered_same_product_week_pairs": 0,
    }

    if args.use_sampled_context_edges:
        pass
    elif args.custom2_edge_mode == "template" and args.text_edge_mode in {"exact", "both"}:
        build_exact_template_edges(nodes, edge_sets, args.max_neighbors_per_node)
    if (not args.use_sampled_context_edges) and (
        args.custom2_edge_mode == "risk_text"
        or (args.custom2_edge_mode == "template" and args.text_edge_mode in {"risk_similarity", "both"})
    ):
        risk_stats = build_risk_text_similarity_edges(
            nodes,
            embeddings,
            edge_sets,
            args.text_threshold,
            args.text_topk,
            args.text_candidate_policy,
            args.text_exclude_neutral,
            args.allow_same_product_week,
            args.text_search_multiplier,
        )
        text_relation_stats.update(risk_stats)
    if (not args.use_sampled_context_edges) and args.custom2_edge_mode == "cold_start":
        cold_start_stats = build_cold_start_risk_cohort_edges(
            nodes,
            edge_sets,
            args.cold_start_max_prior_user_reviews,
            args.cold_start_date_window_days,
            args.cold_start_topk,
            args.cold_start_exclude_neutral,
            args.cold_start_allow_same_product_week,
        )
    if (not args.use_sampled_context_edges) and args.custom2_edge_mode == "weak_product_shock":
        if args.shock_edge_style == "peer":
            shock_stats = build_weak_product_rating_shock_edges(
                nodes,
                edge_sets,
                args.shock_max_prior_product_reviews,
                args.shock_min_abs_rating_dev,
                args.shock_score_mode,
                args.shock_date_window_days,
                args.shock_topk,
                args.shock_exclude_neutral,
                args.shock_allow_same_product_week,
                args.shock_same_product_only,
                args.shock_min_behavior_shift_score,
            )
        elif args.shock_edge_style == "baseline_context":
            shock_stats = build_product_baseline_shock_edges(
                nodes,
                edge_sets,
                args.shock_max_prior_product_reviews,
                args.shock_min_abs_rating_dev,
                args.shock_score_mode,
                args.shock_baseline_topk,
                args.shock_exclude_neutral,
                args.shock_baseline_exclude_same_week,
                args.shock_baseline_temporal_mode,
            )
        else:
            raise ValueError(f"Unsupported --shock-edge-style: {args.shock_edge_style}")

    edge_sets, duplicate_pairs_removed = collapse_duplicate_pairs_across_relations(edge_sets)
    relation_summary = relation_stats(edge_sets, n_nodes)
    edge_index, edge_type, relation_counts, edge_df = build_edge_arrays(edge_sets, n_nodes)
    save_outputs(
        args.output_dir,
        edge_index,
        edge_type,
        edge_df,
        relation_counts,
        relation_summary,
        n_nodes,
        duplicate_pairs_removed,
        text_relation_stats,
        cold_start_stats,
        shock_stats,
        context_edge_stats,
        args,
    )

    log("Relation edge counts:")
    for _, row in relation_counts.iterrows():
        log(
            f"  {int(row['edge_type'])} {row['relation']}: "
            f"{int(row['directed_edges']):,} directed / {int(row['undirected_edges']):,} undirected"
        )
    log(f"Done. Total directed edges: {edge_index.shape[1]:,}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled-node edge_index and edge_type for YelpZip GNN.")
    parser.add_argument("--sampled-dir", type=Path, default=DEFAULT_SAMPLED_DIR, help="Sampled data directory")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=DEFAULT_EMBEDDING_PATH,
        help="Path to sampled text embedding .npy file",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Edge output directory")
    parser.add_argument(
        "--max-neighbors-per-node",
        type=int,
        default=10,
        help="Maximum neighbors per node within each large relation group",
    )
    parser.add_argument(
        "--burst-min-group-size",
        type=int,
        default=3,
        help="Minimum product-week-rating group size for burst edges",
    )
    parser.add_argument(
        "--rur-temporal",
        action="store_true",
        help="Connect same-user reviews from prior reviews to later reviews instead of bidirectional cliques.",
    )
    parser.add_argument(
        "--rur-temporal-mode",
        choices=["recent", "segmented"],
        default="recent",
        help="When --rur-temporal is used, choose recent prior reviews or old/mid/recent segmented prior reviews.",
    )
    parser.add_argument(
        "--use-sampled-context-edges",
        action="store_true",
        help="Use sampled_context_edges.csv.gz from the sampled directory instead of constructing generic relation edges.",
    )
    parser.add_argument(
        "--relation1-mode",
        choices=["burst", "product_prior_context", "none"],
        default="burst",
        help="Relation 1 construction mode.",
    )
    parser.add_argument(
        "--product-context-topk",
        type=int,
        default=3,
        help="Recent prior same-product reviews linked to each current review when relation1-mode is product_prior_context.",
    )
    parser.add_argument(
        "--product-context-exclude-same-week",
        action="store_true",
        help="Exclude same product-week pairs from product_prior_context relation.",
    )
    parser.add_argument(
        "--custom2-edge-mode",
        choices=["template", "risk_text", "cold_start", "weak_product_shock"],
        default="template",
        help="Select which custom edge type is stored as edge_type 2.",
    )
    parser.add_argument(
        "--shock-edge-style",
        choices=["peer", "baseline_context"],
        default="peer",
        help="For weak_product_shock, choose peer shock-review edges or prior product-baseline edges.",
    )
    parser.add_argument(
        "--text-edge-mode",
        choices=["exact", "risk_similarity", "both"],
        default="exact",
        help="Backward-compatible template/text mode used when custom2 edge mode is template.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for risk text similarity edges",
    )
    parser.add_argument(
        "--text-topk",
        type=int,
        default=3,
        help="Maximum accepted text similarity neighbors per risky source node",
    )
    parser.add_argument(
        "--text-candidate-policy",
        choices=["both_risky", "anchor_risky"],
        default="both_risky",
        help="Whether both endpoints or only the source must be risky",
    )
    parser.add_argument(
        "--text-exclude-neutral",
        action="store_true",
        help="Skip rating_direction == 0 groups for text similarity edges",
    )
    parser.add_argument(
        "--allow-same-product-week",
        action="store_true",
        help="Allow text similarity edges inside the same prod_id and week. Default excludes these pairs.",
    )
    parser.add_argument(
        "--text-search-multiplier",
        type=int,
        default=10,
        help="Nearest-neighbor overfetch multiplier before filtering text candidates",
    )
    parser.add_argument(
        "--cold-start-max-prior-user-reviews",
        type=int,
        default=1,
        help="Maximum prior user reviews for cold-start candidates",
    )
    parser.add_argument(
        "--cold-start-date-window-days",
        type=int,
        default=30,
        help="Maximum date difference in days for cold-start cohort edges",
    )
    parser.add_argument(
        "--cold-start-topk",
        type=int,
        default=5,
        help="Maximum cold-start cohort targets per source node",
    )
    parser.add_argument(
        "--cold-start-exclude-neutral",
        action="store_true",
        help="Skip rating_direction == 0 groups for cold-start cohort edges",
    )
    parser.add_argument(
        "--cold-start-allow-same-product-week",
        action="store_true",
        help="Allow cold-start edges inside the same prod_id and week. Default excludes these pairs.",
    )
    parser.add_argument(
        "--shock-max-prior-product-reviews",
        type=int,
        default=30,
        help="Maximum prior product reviews for weak product shock candidates",
    )
    parser.add_argument(
        "--shock-min-abs-rating-dev",
        type=float,
        default=1.0,
        help=(
            "Minimum deviation from prior product rating mean for shock candidates. "
            "It is raw star deviation when --shock-score-mode=absolute and prior-volatility-scaled deviation when standardized."
        ),
    )
    parser.add_argument(
        "--shock-score-mode",
        choices=["absolute", "standardized"],
        default="absolute",
        help="Use raw absolute rating deviation or deviation scaled by prior product rating volatility for shock candidates.",
    )
    parser.add_argument(
        "--shock-date-window-days",
        type=int,
        default=30,
        help="Maximum date difference in days for weak product shock edges",
    )
    parser.add_argument(
        "--shock-topk",
        type=int,
        default=3,
        help="Maximum weak product shock targets per source node",
    )
    parser.add_argument(
        "--shock-exclude-neutral",
        action="store_true",
        help="Skip rating_direction == 0 groups for weak product shock edges",
    )
    parser.add_argument(
        "--shock-allow-same-product-week",
        action="store_true",
        help="Allow weak product shock edges inside the same prod_id and week. Default excludes these pairs.",
    )
    parser.add_argument(
        "--shock-same-product-only",
        action="store_true",
        help="Only connect weak product shock reviews from the same prod_id but different weeks.",
    )
    parser.add_argument(
        "--shock-min-behavior-shift-score",
        type=float,
        default=0.0,
        help=(
            "Minimum prior-4-week behavior shift score for weak product shock candidates. "
            "Use 0.0 to keep all rating-shock candidates while still prioritizing higher scores."
        ),
    )
    parser.add_argument(
        "--shock-baseline-topk",
        type=int,
        default=5,
        help="Recent prior same-product reviews linked to each shock candidate in baseline_context style.",
    )
    parser.add_argument(
        "--shock-baseline-exclude-same-week",
        action="store_true",
        help="Exclude same product-week pairs from baseline_context shock edges.",
    )
    parser.add_argument(
        "--shock-baseline-temporal-mode",
        choices=["recent", "segmented"],
        default="recent",
        help="For baseline_context shock edges, choose recent prior product reviews or old/mid/recent segmented prior reviews.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_build_edges(parse_args())
    except Exception as exc:
        print(f"[BuildEdges][ERROR] {exc}", file=sys.stderr)
        raise

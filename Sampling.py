"""
Sample a dense, graph-ready YelpZip review-node subgraph.

이 스크립트는 전처리된 리뷰 노드 중 상품-주 단위로 밀도와 행동 flag를 만족하는 구간을 선택한다.
아직 엣지는 직접 만들지 않고, 선택된 노드와 이후 relation 생성에 필요한 후보 키만 함께 저장한다.
"""

# Windows/PyCharm/PowerShell 환경에서 한국어 로그가 깨지지 않도록 기본 입출력 인코딩을 맞춘다.
# 샘플링 산출물은 data/sampled 아래에 저장하며, data/processed 원본 산출물은 수정하지 않는다.
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
DEFAULT_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "sampled"

DEFAULT_MIN_REVIEWS_PER_PRODUCT_WEEK = 10
DEFAULT_MIN_USERS_PER_PRODUCT_WEEK = 8
DEFAULT_FLAG_QUANTILE = 0.75
DEFAULT_MIN_FLAGS = 2
DEFAULT_MAX_NODES = 50_000
DEFAULT_SHOCK_MAX_PRIOR_PRODUCT_REVIEWS = 30
DEFAULT_SHOCK_MIN_ABS_RATING_DEV = 1.0
DEFAULT_PRODUCT_CONTEXT_PER_SHOCK = 5
DEFAULT_USER_CONTEXT_PER_SHOCK = 3
DEFAULT_CONTRAST_SEED_PER_ROLE = 0
DEFAULT_DISTRIBUTION_FILL_TARGET = 50_000
WEEK_FREQ = "W-SUN"


# 간단한 로그 출력과 JSON 직렬화 보조 함수다.
# numpy/pandas 타입이 섞여도 summary JSON 저장이 실패하지 않도록 표준 타입으로 바꾼다.
def log(message: str) -> None:
    print(f"[Sampling] {message}")


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


# 전처리된 리뷰 노드 테이블을 읽고, 전처리 단계와 동일한 월요일 시작 주차를 다시 계산한다.
# local_template_repeat_flag 계산을 위해 상품-주 내부 동일 텍스트 반복 여부도 노드 단위로 표시한다.
def load_review_nodes(processed_dir: Path) -> pd.DataFrame:
    node_path = processed_dir / "review_nodes.csv.gz"
    if not node_path.exists():
        raise FileNotFoundError(f"Missing preprocessed node file: {node_path}")

    log(f"Loading review nodes: {node_path}")
    df = pd.read_csv(node_path, parse_dates=["date"])
    df["week"] = df["date"].dt.to_period(WEEK_FREQ).dt.start_time
    df["rating_direction_group"] = np.select(
        [df["rating"].le(2), df["rating"].ge(4)],
        ["low", "high"],
        default="mid",
    )
    df["local_template_repeat"] = df["same_text_count_in_product_week"].ge(2)
    return df


# 상품-주 단위의 샘플링 후보 테이블을 만든다.
# 라벨은 선택 기준에 쓰지 않고, 사후 진단용 fake_rate/fake_count로만 함께 계산한다.
def build_product_week_units(df: pd.DataFrame) -> pd.DataFrame:
    log("Aggregating product-week sampling units.")
    units = (
        df.groupby(["prod_id", "week"], observed=True)
        .agg(
            n_reviews=("node_idx", "size"),
            n_users=("user_id", "nunique"),
            growth_base_30d=("product_reviews_last_30d", "median"),
            new_user_ratio=("is_new_user_at_review_time", "mean"),
            extreme_ratio=("extreme_rating", "mean"),
            short_ratio=("short_review_flag", "mean"),
            prior_product_review_count_median=("prior_product_review_count", "median"),
            mean_abs_rating_dev=("rating_deviation_from_prior_product_mean", lambda s: float(np.mean(np.abs(s)))),
            prior_product_rating_std_median=("prior_product_rating_std", "median"),
            rating_impact_abs_mean=("rating_impact_abs", "mean"),
            extreme_rating_impact_abs_mean=("extreme_rating_impact_abs", "mean"),
            local_template_repeat_flag=("local_template_repeat", "max"),
            local_template_repeat_ratio=("local_template_repeat", "mean"),
            fake_rate=("is_fake", "mean"),
            fake_count=("is_fake", "sum"),
        )
        .reset_index()
    )

    relative_behavior_columns = [
        "same_dir_log_count_lift_4w",
        "total_log_count_lift_4w",
        "direction_concentration_lift_4w",
        "new_user_ratio_lift_4w",
        "short_review_ratio_lift_4w",
        "word_len_drop_ratio_4w",
        "abuse_burst_behavior_score_4w",
    ]
    available_relative_behavior_columns = [col for col in relative_behavior_columns if col in df.columns]
    if available_relative_behavior_columns:
        behavior_units = (
            df.groupby(["prod_id", "week"], observed=True)[available_relative_behavior_columns]
            .median()
            .add_suffix("_median")
            .reset_index()
        )
        units = units.merge(behavior_units, on=["prod_id", "week"], how="left")

    direction_counts = (
        df.groupby(["prod_id", "week", "rating_direction_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    units = units.merge(
        (direction_counts.max(axis=1) / direction_counts.sum(axis=1))
        .rename("rating_direction_concentration")
        .reset_index(),
        on=["prod_id", "week"],
        how="left",
    )

    # 직전 30일 리뷰 수를 주간 기대치로 환산하고 +1 smoothing을 둬 cold-start 상품의 0 나누기를 피한다.
    # weak_product_score는 값이 클수록 작성 시점 이전 상품 리뷰 이력이 약하다는 뜻이 되도록 방향을 뒤집는다.
    units["growth_ratio_30d"] = units["n_reviews"] / (units["growth_base_30d"] / 30 * 7 + 1.0)
    units["weak_product_score"] = 1 / np.log1p(units["prior_product_review_count_median"] + 1)
    units["current_week_share_of_history"] = units["n_reviews"] / (
        units["prior_product_review_count_median"].clip(lower=0) + 1.0
    )
    return units


# 기본 후보 안에서 7개 분위수 flag와 1개 템플릿 반복 flag를 계산한다.
# local_template_repeat_flag는 희소한 이진 사건이므로 분위수 기준 대신 존재 여부를 그대로 사용한다.
def apply_sampling_flags(
    units: pd.DataFrame,
    min_reviews: int,
    min_users: int,
    flag_quantile: float,
    min_flags: int,
    flag_mode: str = "absolute",
) -> tuple[pd.DataFrame, dict[str, float]]:
    log(f"Applying density filters and 8 sampling flags. flag_mode={flag_mode}")
    units = units.copy()
    units["base_candidate"] = units["n_reviews"].ge(min_reviews) & units["n_users"].ge(min_users)

    base = units.loc[units["base_candidate"]].copy()
    if base.empty:
        raise ValueError("No product-week unit satisfies the base candidate thresholds.")

    if flag_mode == "absolute":
        flag_sources = {
            "review_growth_flag": "growth_ratio_30d",
            "new_user_ratio_flag": "new_user_ratio",
            "extreme_rating_ratio_flag": "extreme_ratio",
            "short_review_ratio_flag": "short_ratio",
            "weak_product_flag": "weak_product_score",
            "rating_deviation_flag": "mean_abs_rating_dev",
            "rating_direction_concentration_flag": "rating_direction_concentration",
        }
    elif flag_mode == "relative":
        positive_std = pd.to_numeric(base["prior_product_rating_std_median"], errors="coerce")
        positive_std = positive_std.loc[positive_std.gt(0)]
        std_floor = float(positive_std.quantile(0.25)) if not positive_std.empty else 1.0
        std_floor = max(std_floor, 1e-6)
        scale = pd.to_numeric(units["prior_product_rating_std_median"], errors="coerce").fillna(std_floor)
        scale = scale.clip(lower=std_floor)
        units["standardized_mean_abs_rating_dev"] = units["mean_abs_rating_dev"] / scale
        units["standardized_rating_impact_abs_mean"] = units["rating_impact_abs_mean"] / scale
        units["standardized_extreme_rating_impact_abs_mean"] = units["extreme_rating_impact_abs_mean"] / scale

        relative_defaults = {
            "new_user_ratio_lift_4w_median": "new_user_ratio",
            "short_review_ratio_lift_4w_median": "short_ratio",
            "direction_concentration_lift_4w_median": "rating_direction_concentration",
            "same_dir_log_count_lift_4w_median": "growth_ratio_30d",
            "word_len_drop_ratio_4w_median": "short_ratio",
            "abuse_burst_behavior_score_4w_median": "standardized_mean_abs_rating_dev",
        }
        for relative_col, fallback_col in relative_defaults.items():
            if relative_col not in units.columns:
                units[relative_col] = units[fallback_col]
            units[relative_col] = pd.to_numeric(units[relative_col], errors="coerce").fillna(0.0)

        flag_sources = {
            "review_growth_flag": "same_dir_log_count_lift_4w_median",
            "new_user_ratio_flag": "new_user_ratio_lift_4w_median",
            "extreme_rating_ratio_flag": "abuse_burst_behavior_score_4w_median",
            "short_review_ratio_flag": "short_review_ratio_lift_4w_median",
            "weak_product_flag": "word_len_drop_ratio_4w_median",
            "rating_deviation_flag": "standardized_mean_abs_rating_dev",
            "rating_direction_concentration_flag": "direction_concentration_lift_4w_median",
        }
    else:
        raise ValueError(f"Unsupported --flag-mode: {flag_mode}")

    base = units.loc[units["base_candidate"]].copy()
    thresholds = {
        flag_name: float(base[source_col].quantile(flag_quantile))
        for flag_name, source_col in flag_sources.items()
    }

    for flag_name, source_col in flag_sources.items():
        units[flag_name] = units[source_col].ge(thresholds[flag_name])
    units["local_template_repeat_flag"] = units["local_template_repeat_flag"].astype(bool)

    flag_columns = list(flag_sources.keys()) + ["local_template_repeat_flag"]
    units["sampling_flag_count"] = units[flag_columns].sum(axis=1).astype("int16")
    units["selected"] = units["base_candidate"] & units["sampling_flag_count"].ge(min_flags)
    thresholds["_flag_mode"] = flag_mode
    return units, thresholds


# 선택된 상품-주 구간에 속한 모든 리뷰 노드를 포함한다.
# 원본 node_idx는 original_node_idx로 보존하고, 샘플 내부에서 사용할 sampled_node_idx를 새로 부여한다.
def select_review_nodes(df: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    log("Selecting review nodes from selected product-week units.")
    selected_units = units.loc[units["selected"], ["prod_id", "week"]].copy()
    selected_units["_selected_unit"] = True

    sampled = df.merge(selected_units, on=["prod_id", "week"], how="inner")
    sampled = sampled.sort_values(["node_idx"]).reset_index(drop=True)
    sampled = sampled.rename(columns={"node_idx": "original_node_idx"})
    sampled.insert(0, "sampled_node_idx", np.arange(len(sampled), dtype=np.int64))
    sampled = sampled.drop(columns=["_selected_unit"])
    return sampled


def make_sampled_nodes_from_idx(df: pd.DataFrame, selected_idx: set[int]) -> pd.DataFrame:
    sampled = df.loc[df["node_idx"].isin(selected_idx)].copy()
    sampled = sampled.sort_values(["node_idx"]).reset_index(drop=True)
    sampled = sampled.rename(columns={"node_idx": "original_node_idx"})
    sampled.insert(0, "sampled_node_idx", np.arange(len(sampled), dtype=np.int64))
    return sampled


def deterministic_stratified_take(
    candidates: pd.DataFrame,
    n: int,
    strata_cols: list[str],
    sort_cols: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    if n <= 0 or candidates.empty:
        return candidates.head(0).copy()
    if len(candidates) <= n:
        return candidates.sort_values(sort_cols, ascending=ascending).copy()

    work = candidates.copy()
    available_strata = [col for col in strata_cols if col in work.columns]
    if not available_strata:
        return work.sort_values(sort_cols, ascending=ascending).head(n).copy()

    counts = work.groupby(available_strata, dropna=False, observed=True).size().rename("count").reset_index()
    counts["_raw_quota"] = counts["count"] / counts["count"].sum() * n
    counts["_quota"] = np.floor(counts["_raw_quota"]).astype(int)
    remainder = int(n - counts["_quota"].sum())
    if remainder > 0:
        counts["_fraction"] = counts["_raw_quota"] - counts["_quota"]
        counts = counts.sort_values(["_fraction", "count"], ascending=[False, False]).reset_index(drop=True)
        counts.loc[: remainder - 1, "_quota"] += 1

    parts: list[pd.DataFrame] = []
    for _, row in counts.iterrows():
        quota = int(row["_quota"])
        if quota <= 0:
            continue
        mask = np.ones(len(work), dtype=bool)
        for col in available_strata:
            value = row[col]
            if pd.isna(value):
                mask &= work[col].isna().to_numpy()
            else:
                mask &= work[col].eq(value).to_numpy()
        part = work.loc[mask].sort_values(sort_cols, ascending=ascending).head(quota)
        parts.append(part)

    if not parts:
        return work.sort_values(sort_cols, ascending=ascending).head(n).copy()
    sampled = pd.concat(parts, ignore_index=False).drop_duplicates("node_idx")
    if len(sampled) < n:
        remaining = work.loc[~work["node_idx"].isin(sampled["node_idx"])]
        sampled = pd.concat(
            [sampled, remaining.sort_values(sort_cols, ascending=ascending).head(n - len(sampled))],
            ignore_index=False,
        )
    return sampled.sort_values("node_idx").head(n).copy()


def deterministic_stratified_fill_like(
    candidates: pd.DataFrame,
    n: int,
    reference: pd.DataFrame,
    strata_cols: list[str],
    sort_cols: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    if n <= 0 or candidates.empty:
        return candidates.head(0).copy()
    if len(candidates) <= n:
        return candidates.sort_values(sort_cols, ascending=ascending).copy()

    available_strata = [col for col in strata_cols if col in candidates.columns and col in reference.columns]
    if not available_strata or reference.empty:
        return deterministic_stratified_take(candidates, n, strata_cols, sort_cols, ascending)

    work = candidates.copy()
    ref_counts = reference.groupby(available_strata, dropna=False, observed=True).size().rename("ref_count").reset_index()
    cand_counts = work.groupby(available_strata, dropna=False, observed=True).size().rename("candidate_count").reset_index()
    quotas = cand_counts.merge(ref_counts, on=available_strata, how="left")
    quotas["ref_count"] = quotas["ref_count"].fillna(0.0)
    if float(quotas["ref_count"].sum()) <= 0:
        quotas["_raw_quota"] = quotas["candidate_count"] / quotas["candidate_count"].sum() * n
    else:
        quotas["_raw_quota"] = quotas["ref_count"] / quotas["ref_count"].sum() * n
    quotas["_quota"] = np.floor(quotas["_raw_quota"]).astype(int)
    quotas["_quota"] = np.minimum(quotas["_quota"], quotas["candidate_count"].astype(int))

    parts: list[pd.DataFrame] = []
    for _, row in quotas.iterrows():
        quota = int(row["_quota"])
        if quota <= 0:
            continue
        mask = np.ones(len(work), dtype=bool)
        for col in available_strata:
            value = row[col]
            if pd.isna(value):
                mask &= work[col].isna().to_numpy()
            else:
                mask &= work[col].eq(value).to_numpy()
        part = work.loc[mask].sort_values(sort_cols, ascending=ascending).head(quota)
        parts.append(part)

    sampled = pd.concat(parts, ignore_index=False).drop_duplicates("node_idx") if parts else work.head(0)
    if len(sampled) < n:
        remaining = work.loc[~work["node_idx"].isin(sampled["node_idx"])]
        sampled = pd.concat(
            [sampled, remaining.sort_values(sort_cols, ascending=ascending).head(n - len(sampled))],
            ignore_index=False,
        )
    return sampled.sort_values("node_idx").head(n).copy()


def add_recent_prior_context(
    df: pd.DataFrame,
    seed_nodes: pd.DataFrame,
    selected_idx: set[int],
    group_col: str,
    context_count: int,
) -> int:
    if context_count <= 0 or seed_nodes.empty:
        return 0

    added_before = len(selected_idx)
    needed_keys = seed_nodes[group_col].dropna().unique().tolist()
    seed_idx_by_key = {
        key: set(group["node_idx"].astype(int).tolist())
        for key, group in seed_nodes.groupby(group_col, sort=False)
    }

    for key, group in df.loc[df[group_col].isin(needed_keys)].groupby(group_col, sort=False):
        seed_idx = seed_idx_by_key.get(key)
        if not seed_idx:
            continue
        ordered = group.sort_values(["date", "node_idx"])
        node_ids = ordered["node_idx"].astype(int).to_numpy()
        positions = {int(node_id): pos for pos, node_id in enumerate(node_ids)}
        for node_id in seed_idx:
            pos = positions.get(int(node_id))
            if pos is None or pos == 0:
                continue
            prior_ids = node_ids[max(0, pos - context_count) : pos]
            selected_idx.update(int(prior_id) for prior_id in prior_ids)

    return len(selected_idx) - added_before


def prepare_temporal_group_lookup(
    df: pd.DataFrame,
    group_col: str,
) -> tuple[dict[Any, pd.DataFrame], dict[int, tuple[Any, int]]]:
    groups: dict[Any, pd.DataFrame] = {}
    positions: dict[int, tuple[Any, int]] = {}
    for key, group in df.groupby(group_col, sort=False):
        ordered = group.sort_values(["date", "node_idx"]).reset_index(drop=True)
        groups[key] = ordered
        for pos, node_id in enumerate(ordered["node_idx"].astype(int).tolist()):
            positions[int(node_id)] = (key, pos)
    return groups, positions


def choose_temporal_context_nodes(
    ordered: pd.DataFrame,
    seed_pos: int,
    seed_date: pd.Timestamp,
    seed_week: Any,
    exclude_same_week: bool,
) -> dict[str, int]:
    prior = ordered.loc[ordered.index < seed_pos].copy()
    prior = prior.loc[prior["date"] < seed_date]
    if exclude_same_week and "week" in prior.columns:
        prior = prior.loc[prior["week"] != seed_week]
    if prior.empty:
        return {}

    choices: dict[str, int] = {}
    prior = prior.sort_values(["date", "node_idx"]).reset_index(drop=True)
    recent_idx = int(prior.iloc[-1]["node_idx"])
    choices["recent"] = recent_idx

    if len(prior) >= 3:
        remaining = prior.loc[~prior["node_idx"].astype(int).isin(choices.values())].copy()
        if not remaining.empty:
            middle_pos = len(remaining) // 2
            choices["mid"] = int(remaining.iloc[middle_pos]["node_idx"])
    elif len(prior) >= 2:
        remaining = prior.loc[~prior["node_idx"].astype(int).isin(choices.values())].copy()
        if not remaining.empty:
            choices["mid"] = int(remaining.iloc[0]["node_idx"])

    remaining = prior.loc[~prior["node_idx"].astype(int).isin(choices.values())].copy()
    if not remaining.empty:
        oldest_half_count = max(1, len(prior) // 2)
        old_pool = prior.iloc[:oldest_half_count].copy()
        old_pool = old_pool.loc[~old_pool["node_idx"].astype(int).isin(choices.values())]
        if old_pool.empty:
            old_pool = remaining
        prior_mean = float(prior["rating"].mean())
        old_pool["_rating_distance"] = (old_pool["rating"].astype(float) - prior_mean).abs()
        old_pool = old_pool.sort_values(["_rating_distance", "date", "node_idx"])
        choices["baseline"] = int(old_pool.iloc[0]["node_idx"])

    return choices


def add_seed_context_edges(
    df: pd.DataFrame,
    seed_nodes: pd.DataFrame,
    selected_idx: set[int],
    context_edges: list[dict[str, Any]],
    group_col: str,
    relation_type: int,
    context_kind: str,
    exclude_same_week: bool,
) -> set[int]:
    context_idx: set[int] = set()
    if seed_nodes.empty:
        return context_idx

    groups, positions = prepare_temporal_group_lookup(df, group_col)
    for row in seed_nodes.itertuples(index=False):
        seed_original_idx = int(row.node_idx)
        position = positions.get(seed_original_idx)
        if position is None:
            continue
        key, pos = position
        ordered = groups[key]
        choices = choose_temporal_context_nodes(
            ordered,
            pos,
            pd.Timestamp(row.date),
            row.week,
            exclude_same_week=exclude_same_week,
        )
        for stage, src_original_idx in choices.items():
            selected_idx.add(int(src_original_idx))
            context_idx.add(int(src_original_idx))
            context_edges.append(
                {
                    "src_original_node_idx": int(src_original_idx),
                    "dst_original_node_idx": int(seed_original_idx),
                    "relation_type": int(relation_type),
                    "context_kind": context_kind,
                    "context_stage": stage,
                    "seed_role": getattr(row, "sample_role"),
                }
            )
            if (
                relation_type == 1
                and stage == "baseline"
                and bool(getattr(row, "is_strong_suspicious_seed", False))
            ):
                context_edges.append(
                    {
                        "src_original_node_idx": int(src_original_idx),
                        "dst_original_node_idx": int(seed_original_idx),
                        "relation_type": 2,
                        "context_kind": "product_baseline_shock",
                        "context_stage": stage,
                        "seed_role": getattr(row, "sample_role"),
                    }
                )
    return context_idx


def add_normal_hard_negative_units(
    df: pd.DataFrame,
    units: pd.DataFrame,
    selected_idx: set[int],
    max_nodes: int,
) -> dict[str, Any]:
    base = units.loc[units["base_candidate"]].copy()
    if base.empty:
        return {"candidate_units": 0, "added_units": 0, "added_nodes": 0}

    direction_cutoff = float(base["rating_direction_concentration"].median())
    product_history_cutoff = float(base["prior_product_review_count_median"].median())
    normal_units = base.loc[
        base["rating_direction_concentration"].le(direction_cutoff)
        & base["prior_product_review_count_median"].ge(product_history_cutoff)
    ].copy()
    normal_units = normal_units.sort_values(["n_reviews", "n_users", "prod_id", "week"], ascending=[False, False, True, True])

    added_units = 0
    added_nodes = 0
    for row in normal_units.itertuples(index=False):
        if len(selected_idx) >= max_nodes:
            break
        mask = (df["prod_id"] == row.prod_id) & (df["week"] == row.week)
        unit_idx = set(int(value) for value in df.loc[mask, "node_idx"].tolist())
        new_idx = unit_idx - selected_idx
        if not new_idx:
            continue
        if len(selected_idx) + len(new_idx) > max_nodes:
            continue
        selected_idx.update(new_idx)
        added_units += 1
        added_nodes += len(new_idx)

    return {
        "candidate_units": int(len(normal_units)),
        "added_units": int(added_units),
        "added_nodes": int(added_nodes),
        "direction_concentration_cutoff": direction_cutoff,
        "prior_product_review_count_median_cutoff": product_history_cutoff,
    }


def select_rur_shock_context_nodes(
    df: pd.DataFrame,
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    log("Selecting nodes with R-U-R + weak-shock context strategy.")
    if args.max_nodes <= 0:
        raise ValueError(f"--max-nodes must be positive, got {args.max_nodes}.")

    abs_dev = df["rating_deviation_from_prior_product_mean"].abs()
    impact = df["rating_impact_abs"] if "rating_impact_abs" in df.columns else abs_dev / (df["prior_product_review_count"] + 1)
    candidate_mask = (
        df["prior_product_review_count"].between(1, args.shock_max_prior_product_reviews)
        & abs_dev.ge(args.shock_min_abs_rating_dev)
        & df["rating_direction"].ne(0)
    )
    candidates = df.loc[candidate_mask].copy()
    candidates["_sampling_rating_impact_abs"] = impact.loc[candidates.index].astype("float64")
    if candidates.empty:
        raise ValueError("No weak-product rating-shock candidates were found for rur_shock_context sampling.")

    quantile_grid = [0.0, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    chosen: tuple[float, float, pd.DataFrame, set[int], int, int] | None = None
    for quantile in quantile_grid:
        threshold = float(candidates["_sampling_rating_impact_abs"].quantile(quantile))
        seed_nodes = candidates.loc[candidates["_sampling_rating_impact_abs"].ge(threshold)].copy()
        selected_idx = set(int(value) for value in seed_nodes["node_idx"].tolist())
        product_added = add_recent_prior_context(
            df, seed_nodes, selected_idx, "prod_id", args.product_context_per_shock
        )
        user_added = add_recent_prior_context(
            df, seed_nodes, selected_idx, "user_id", args.user_context_per_shock
        )
        if len(selected_idx) <= args.max_nodes:
            chosen = (quantile, threshold, seed_nodes, selected_idx, product_added, user_added)
            break

    if chosen is None:
        threshold = float(candidates["_sampling_rating_impact_abs"].quantile(0.95))
        seed_nodes = candidates.loc[candidates["_sampling_rating_impact_abs"].ge(threshold)].copy()
        seed_nodes = seed_nodes.sort_values(["_sampling_rating_impact_abs", "date", "node_idx"], ascending=[False, True, True])
        seed_nodes = seed_nodes.head(args.max_nodes)
        selected_idx = set(int(value) for value in seed_nodes["node_idx"].tolist())
        chosen = (0.95, threshold, seed_nodes, selected_idx, 0, 0)

    quantile, threshold, seed_nodes, selected_idx, product_added, user_added = chosen
    hard_negative_stats = add_normal_hard_negative_units(df, units, selected_idx, args.max_nodes)

    sampled_nodes = make_sampled_nodes_from_idx(df, selected_idx)
    selected_unit_keys = sampled_nodes[["prod_id", "week"]].drop_duplicates().copy()
    selected_unit_keys["_selected_unit"] = True
    units = units.drop(columns=["selected"], errors="ignore").merge(
        selected_unit_keys, on=["prod_id", "week"], how="left"
    )
    units["selected"] = units["_selected_unit"].fillna(False).astype(bool)
    units = units.drop(columns=["_selected_unit"])

    details = {
        "strategy": "rur_shock_context",
        "max_nodes": int(args.max_nodes),
        "shock_candidate_count": int(len(candidates)),
        "shock_seed_count": int(len(seed_nodes)),
        "shock_impact_quantile_used": float(quantile),
        "shock_impact_threshold_used": float(threshold),
        "product_context_per_shock": int(args.product_context_per_shock),
        "user_context_per_shock": int(args.user_context_per_shock),
        "product_context_nodes_added": int(product_added),
        "user_context_nodes_added": int(user_added),
        "normal_hard_negative_units": hard_negative_stats,
        "selected_nodes": int(len(sampled_nodes)),
    }
    return sampled_nodes, units, details


# numpy feature/label 배열을 원본 node_idx 순서로 부분 선택한다.
# sampled_node_idx의 행 순서와 feature matrix 행 순서가 정확히 일치하도록 검증한다.
def select_legacy_plus_shock_context_nodes(
    df: pd.DataFrame,
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    log("Selecting nodes with legacy units plus weak-shock context strategy.")
    if args.max_nodes <= 0:
        raise ValueError(f"--max-nodes must be positive, got {args.max_nodes}.")

    legacy_units = units.loc[units["selected"], ["prod_id", "week"]].copy()
    if legacy_units.empty:
        selected_idx: set[int] = set()
        legacy_mask = pd.Series(False, index=df.index)
    else:
        legacy_keys = pd.MultiIndex.from_frame(legacy_units)
        df_keys = pd.MultiIndex.from_frame(df[["prod_id", "week"]])
        legacy_mask = pd.Series(df_keys.isin(legacy_keys), index=df.index)
        selected_idx = set(int(value) for value in df.loc[legacy_mask, "node_idx"].tolist())
    if len(selected_idx) > args.max_nodes:
        raise ValueError(
            f"Legacy selected nodes ({len(selected_idx):,}) exceed --max-nodes ({args.max_nodes:,})."
        )

    abs_dev = df["rating_deviation_from_prior_product_mean"].abs()
    impact = df["rating_impact_abs"] if "rating_impact_abs" in df.columns else abs_dev / (df["prior_product_review_count"] + 1)
    candidate_mask = (
        df["prior_product_review_count"].between(1, args.shock_max_prior_product_reviews)
        & abs_dev.ge(args.shock_min_abs_rating_dev)
        & df["rating_direction"].ne(0)
        & ~df["node_idx"].isin(selected_idx)
    )
    candidates = df.loc[candidate_mask].copy()
    candidates["_sampling_rating_impact_abs"] = impact.loc[candidates.index].astype("float64")

    quantile_grid = [0.0, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
    chosen: tuple[float | None, float | None, pd.DataFrame, int, int] | None = None
    if not candidates.empty and len(selected_idx) < args.max_nodes:
        for quantile in quantile_grid:
            trial_idx = set(selected_idx)
            threshold = float(candidates["_sampling_rating_impact_abs"].quantile(quantile))
            seed_nodes = candidates.loc[candidates["_sampling_rating_impact_abs"].ge(threshold)].copy()
            trial_idx.update(int(value) for value in seed_nodes["node_idx"].tolist())
            product_added = add_recent_prior_context(
                df, seed_nodes, trial_idx, "prod_id", args.product_context_per_shock
            )
            user_added = add_recent_prior_context(
                df, seed_nodes, trial_idx, "user_id", args.user_context_per_shock
            )
            if len(trial_idx) <= args.max_nodes:
                selected_idx = trial_idx
                chosen = (quantile, threshold, seed_nodes, product_added, user_added)
                break

        if chosen is None:
            remaining = args.max_nodes - len(selected_idx)
            seed_nodes = candidates.sort_values(
                ["_sampling_rating_impact_abs", "date", "node_idx"],
                ascending=[False, True, True],
            ).head(remaining)
            selected_idx.update(int(value) for value in seed_nodes["node_idx"].tolist())
            chosen = (None, None, seed_nodes, 0, 0)

    if chosen is None:
        chosen = (None, None, candidates.head(0), 0, 0)

    quantile, threshold, seed_nodes, product_added, user_added = chosen
    hard_negative_stats = add_normal_hard_negative_units(df, units, selected_idx, args.max_nodes)

    sampled_nodes = make_sampled_nodes_from_idx(df, selected_idx)
    selected_unit_keys = sampled_nodes[["prod_id", "week"]].drop_duplicates().copy()
    selected_unit_keys["_selected_unit"] = True
    units = units.drop(columns=["selected"], errors="ignore").merge(
        selected_unit_keys, on=["prod_id", "week"], how="left"
    )
    units["selected"] = units["_selected_unit"].fillna(False).astype(bool)
    units = units.drop(columns=["_selected_unit"])

    details = {
        "strategy": "legacy_plus_shock_context",
        "max_nodes": int(args.max_nodes),
        "legacy_units": int(len(legacy_units)),
        "legacy_nodes": int(legacy_mask.sum()),
        "shock_candidate_count": int(len(candidates)),
        "shock_seed_count": int(len(seed_nodes)),
        "shock_impact_quantile_used": None if quantile is None else float(quantile),
        "shock_impact_threshold_used": None if threshold is None else float(threshold),
        "product_context_per_shock": int(args.product_context_per_shock),
        "user_context_per_shock": int(args.user_context_per_shock),
        "product_context_nodes_added": int(product_added),
        "user_context_nodes_added": int(user_added),
        "normal_hard_negative_units": hard_negative_stats,
        "selected_nodes": int(len(sampled_nodes)),
    }
    return sampled_nodes, units, details


def select_burst_contrast_context_nodes(
    df: pd.DataFrame,
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    log("Selecting nodes with burst-contrast seed/context strategy.")
    if args.max_nodes <= 0:
        raise ValueError(f"--max-nodes must be positive, got {args.max_nodes}.")

    units = units.copy()
    base_mask = units["base_candidate"].astype(bool)
    base = units.loc[base_mask].copy()
    if base.empty:
        raise ValueError("No product-week unit satisfies the base candidate thresholds.")

    growth_cutoff = float(base["growth_ratio_30d"].quantile(0.75))
    growth = base.loc[base["growth_ratio_30d"].ge(growth_cutoff)].copy()
    if growth.empty:
        raise ValueError("No product-week unit satisfies the growth gate.")

    std_source = pd.to_numeric(base["prior_product_rating_std_median"], errors="coerce")
    std_floor_candidates = std_source.loc[std_source.gt(0)]
    std_floor = float(std_floor_candidates.quantile(0.25)) if not std_floor_candidates.empty else 1.0
    std_floor = max(std_floor, 1e-6)

    units["std_for_shift_scale"] = pd.to_numeric(
        units["prior_product_rating_std_median"], errors="coerce"
    ).fillna(std_floor).clip(lower=std_floor)
    units["standardized_rating_shift"] = (
        pd.to_numeric(units["mean_abs_rating_dev"], errors="coerce").fillna(0.0)
        / units["std_for_shift_scale"]
    )

    growth = units.loc[
        base_mask & units["growth_ratio_30d"].ge(growth_cutoff)
    ].copy()
    shift_q50 = float(growth["standardized_rating_shift"].quantile(0.50))
    shift_q75 = float(growth["standardized_rating_shift"].quantile(0.75))
    shift_q90 = float(growth["standardized_rating_shift"].quantile(0.90))
    history_cutoff = float(growth["prior_product_review_count_median"].quantile(0.50))

    other_change_columns = [
        "new_user_ratio",
        "short_ratio",
        "extreme_ratio",
        "rating_direction_concentration",
    ]
    change_thresholds: dict[str, float] = {}
    for col in other_change_columns:
        delta = (growth[col] - growth[col].median()).abs()
        cutoff = float(delta.quantile(0.75))
        change_thresholds[col] = cutoff
        units[f"{col}_delta_from_growth_median"] = (units[col] - growth[col].median()).abs()
        units[f"{col}_changed"] = units[f"{col}_delta_from_growth_median"].gt(cutoff)
    changed_cols = [f"{col}_changed" for col in other_change_columns]
    units["other_review_property_change_count"] = units[changed_cols].sum(axis=1).astype("int16")

    units["burst_growth_gate"] = base_mask & units["growth_ratio_30d"].ge(growth_cutoff)
    units["suspicious_burst_unit"] = (
        units["burst_growth_gate"] & units["standardized_rating_shift"].ge(shift_q75)
    )
    units["strong_suspicious_burst_unit"] = (
        units["burst_growth_gate"] & units["standardized_rating_shift"].ge(shift_q90)
    )
    units["normal_burst_unit"] = (
        units["burst_growth_gate"]
        & units["standardized_rating_shift"].le(shift_q50)
        & units["prior_product_review_count_median"].ge(history_cutoff)
        & units["other_review_property_change_count"].le(1)
    )
    units["background_burst_unit"] = (
        base_mask & ~units["suspicious_burst_unit"] & ~units["normal_burst_unit"]
    )

    unit_role_cols = [
        "prod_id",
        "week",
        "standardized_rating_shift",
        "strong_suspicious_burst_unit",
        "suspicious_burst_unit",
        "normal_burst_unit",
        "background_burst_unit",
        "other_review_property_change_count",
    ]
    role_nodes = df.merge(units[unit_role_cols], on=["prod_id", "week"], how="left")
    role_nodes["year"] = role_nodes["date"].dt.year
    role_nodes["history_bin"] = pd.qcut(
        role_nodes["prior_product_review_count"].rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    ).astype("int16")

    suspicious_candidates = role_nodes.loc[role_nodes["suspicious_burst_unit"].fillna(False)].copy()
    normal_candidates = role_nodes.loc[role_nodes["normal_burst_unit"].fillna(False)].copy()
    background_candidates = role_nodes.loc[role_nodes["background_burst_unit"].fillna(False)].copy()
    if min(len(suspicious_candidates), len(normal_candidates), len(background_candidates)) == 0:
        raise ValueError(
            "burst_contrast_context requires non-empty suspicious, normal, and background candidate groups. "
            f"suspicious={len(suspicious_candidates)}, normal={len(normal_candidates)}, "
            f"background={len(background_candidates)}"
        )

    seed_per_role = min(len(suspicious_candidates), len(normal_candidates), len(background_candidates))
    if args.contrast_seed_per_role > 0:
        seed_per_role = min(seed_per_role, args.contrast_seed_per_role)

    strata_cols = ["year", "rating_direction", "history_bin"]
    suspicious_seed = deterministic_stratified_take(
        suspicious_candidates,
        seed_per_role,
        strata_cols,
        ["strong_suspicious_burst_unit", "standardized_rating_shift", "date", "node_idx"],
        [False, False, True, True],
    )
    normal_seed = deterministic_stratified_take(
        normal_candidates,
        seed_per_role,
        strata_cols,
        ["other_review_property_change_count", "standardized_rating_shift", "date", "node_idx"],
        [True, True, True, True],
    )
    background_seed = deterministic_stratified_take(
        background_candidates,
        seed_per_role,
        strata_cols,
        ["date", "node_idx"],
        [True, True],
    )

    suspicious_seed = suspicious_seed.assign(
        sample_role="suspicious_seed",
        is_strong_suspicious_seed=suspicious_seed["strong_suspicious_burst_unit"].fillna(False).astype(bool),
    )
    normal_seed = normal_seed.assign(sample_role="normal_burst_seed", is_strong_suspicious_seed=False)
    background_seed = background_seed.assign(sample_role="background_seed", is_strong_suspicious_seed=False)
    seed_nodes = pd.concat([background_seed, suspicious_seed, normal_seed], ignore_index=True)
    seed_nodes = seed_nodes.drop_duplicates("node_idx", keep="first")

    selected_idx = set(int(value) for value in seed_nodes["node_idx"].tolist())
    context_edges: list[dict[str, Any]] = []
    user_context_idx = add_seed_context_edges(
        df,
        seed_nodes,
        selected_idx,
        context_edges,
        "user_id",
        0,
        "user",
        exclude_same_week=False,
    )
    product_context_idx = add_seed_context_edges(
        df,
        seed_nodes,
        selected_idx,
        context_edges,
        "prod_id",
        1,
        "product",
        exclude_same_week=True,
    )

    if len(selected_idx) > args.max_nodes:
        raise ValueError(
            f"burst_contrast_context selected {len(selected_idx):,} nodes after context expansion, "
            f"which exceeds --max-nodes={args.max_nodes:,}. Reduce --contrast-seed-per-role."
        )

    sampled_nodes = make_sampled_nodes_from_idx(df, selected_idx)
    seed_role_map = seed_nodes.set_index("node_idx")["sample_role"].to_dict()
    strong_seed_idx = set(
        int(value)
        for value in seed_nodes.loc[seed_nodes["is_strong_suspicious_seed"], "node_idx"].tolist()
    )
    user_context_idx = set(int(value) for value in user_context_idx)
    product_context_idx = set(int(value) for value in product_context_idx)
    sampled_nodes["sample_role"] = sampled_nodes["original_node_idx"].map(seed_role_map).fillna("context_only")
    sampled_nodes["is_target_node"] = sampled_nodes["sample_role"].ne("context_only")
    sampled_nodes["is_background_seed"] = sampled_nodes["sample_role"].eq("background_seed")
    sampled_nodes["is_suspicious_seed"] = sampled_nodes["sample_role"].eq("suspicious_seed")
    sampled_nodes["is_normal_burst_seed"] = sampled_nodes["sample_role"].eq("normal_burst_seed")
    sampled_nodes["is_strong_suspicious_seed"] = sampled_nodes["original_node_idx"].isin(strong_seed_idx)
    sampled_nodes["is_user_context_node"] = sampled_nodes["original_node_idx"].isin(user_context_idx)
    sampled_nodes["is_product_context_node"] = sampled_nodes["original_node_idx"].isin(product_context_idx)
    sampled_nodes["is_context_node"] = sampled_nodes["is_user_context_node"] | sampled_nodes["is_product_context_node"]

    mapping = sampled_nodes[["sampled_node_idx", "original_node_idx"]].copy()
    context_edge_df = pd.DataFrame(context_edges).drop_duplicates()
    if context_edge_df.empty:
        context_edge_df = pd.DataFrame(
            columns=[
                "src_original_node_idx",
                "dst_original_node_idx",
                "relation_type",
                "context_kind",
                "context_stage",
                "seed_role",
            ]
        )
    context_edge_df = context_edge_df.merge(
        mapping.rename(columns={"original_node_idx": "src_original_node_idx", "sampled_node_idx": "src"}),
        on="src_original_node_idx",
        how="inner",
    ).merge(
        mapping.rename(columns={"original_node_idx": "dst_original_node_idx", "sampled_node_idx": "dst"}),
        on="dst_original_node_idx",
        how="inner",
    )
    context_edge_df = context_edge_df[["src", "dst", "relation_type", "context_kind", "context_stage", "seed_role"]]

    selected_unit_keys = sampled_nodes[["prod_id", "week"]].drop_duplicates().copy()
    selected_unit_keys["_selected_unit"] = True
    units = units.drop(columns=["selected"], errors="ignore").merge(
        selected_unit_keys, on=["prod_id", "week"], how="left"
    )
    units["selected"] = units["_selected_unit"].fillna(False).astype(bool)
    units = units.drop(columns=["_selected_unit"])

    role_counts = sampled_nodes["sample_role"].value_counts().to_dict()
    target_labels = sampled_nodes.loc[sampled_nodes["is_target_node"], "is_fake"]
    details = {
        "strategy": "burst_contrast_context",
        "max_nodes": int(args.max_nodes),
        "seed_per_role": int(seed_per_role),
        "selected_nodes": int(len(sampled_nodes)),
        "target_nodes": int(sampled_nodes["is_target_node"].sum()),
        "context_only_nodes": int((~sampled_nodes["is_target_node"]).sum()),
        "role_counts": role_counts,
        "target_fake_rate_for_diagnostics_only": float(target_labels.mean()) if len(target_labels) else None,
        "growth_ratio_q75": growth_cutoff,
        "std_floor_q25": std_floor,
        "standardized_shift_q50": shift_q50,
        "standardized_shift_q75": shift_q75,
        "standardized_shift_q90": shift_q90,
        "history_q50_in_growth": history_cutoff,
        "other_change_thresholds": change_thresholds,
        "candidate_nodes": {
            "suspicious": int(len(suspicious_candidates)),
            "normal_burst": int(len(normal_candidates)),
            "background": int(len(background_candidates)),
        },
        "user_context_nodes": int(len(user_context_idx)),
        "product_context_nodes": int(len(product_context_idx)),
        "precomputed_context_edges": int(len(context_edge_df)),
    }
    return sampled_nodes, units, details, context_edge_df


def select_distribution_preserved_hard_nodes(
    df: pd.DataFrame,
    units: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    log("Selecting nodes with distribution-preserved hard-seed strategy.")
    if args.max_nodes <= 0:
        raise ValueError(f"--max-nodes must be positive, got {args.max_nodes}.")

    units = units.copy()
    base_mask = units["base_candidate"].astype(bool)
    legacy_units = units.loc[units["selected"], ["prod_id", "week"]].copy()
    base_units = units.loc[base_mask, ["prod_id", "week"]].copy()
    if legacy_units.empty:
        raise ValueError("distribution_preserved_hard requires non-empty legacy selected units.")

    df = df.copy()
    df["year"] = df["date"].dt.year
    df["history_bin"] = pd.qcut(
        df["prior_product_review_count"].rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    ).astype("int16")

    df_keys = pd.MultiIndex.from_frame(df[["prod_id", "week"]])
    legacy_keys = pd.MultiIndex.from_frame(legacy_units)
    base_keys = pd.MultiIndex.from_frame(base_units)
    legacy_node_mask = pd.Series(df_keys.isin(legacy_keys), index=df.index)
    base_node_mask = pd.Series(df_keys.isin(base_keys), index=df.index)

    selected_idx: set[int] = set(int(value) for value in df.loc[legacy_node_mask, "node_idx"].tolist())
    if len(selected_idx) > args.max_nodes:
        raise ValueError(
            f"Legacy selected nodes ({len(selected_idx):,}) exceed --max-nodes ({args.max_nodes:,})."
        )

    base = units.loc[base_mask].copy()
    growth_cutoff = float(base["growth_ratio_30d"].quantile(0.75))
    std_source = pd.to_numeric(base["prior_product_rating_std_median"], errors="coerce")
    std_floor_candidates = std_source.loc[std_source.gt(0)]
    std_floor = float(std_floor_candidates.quantile(0.25)) if not std_floor_candidates.empty else 1.0
    std_floor = max(std_floor, 1e-6)
    units["std_for_shift_scale"] = pd.to_numeric(
        units["prior_product_rating_std_median"], errors="coerce"
    ).fillna(std_floor).clip(lower=std_floor)
    units["standardized_rating_shift"] = (
        pd.to_numeric(units["mean_abs_rating_dev"], errors="coerce").fillna(0.0)
        / units["std_for_shift_scale"]
    )

    growth = units.loc[base_mask & units["growth_ratio_30d"].ge(growth_cutoff)].copy()
    shift_q50 = float(growth["standardized_rating_shift"].quantile(0.50))
    shift_q75 = float(growth["standardized_rating_shift"].quantile(0.75))
    shift_q90 = float(growth["standardized_rating_shift"].quantile(0.90))
    history_cutoff = float(growth["prior_product_review_count_median"].quantile(0.50))

    other_change_columns = [
        "new_user_ratio",
        "short_ratio",
        "extreme_ratio",
        "rating_direction_concentration",
    ]
    change_thresholds: dict[str, float] = {}
    for col in other_change_columns:
        delta = (growth[col] - growth[col].median()).abs()
        cutoff = float(delta.quantile(0.75))
        change_thresholds[col] = cutoff
        units[f"{col}_delta_from_growth_median"] = (units[col] - growth[col].median()).abs()
        units[f"{col}_changed"] = units[f"{col}_delta_from_growth_median"].gt(cutoff)

    changed_cols = [f"{col}_changed" for col in other_change_columns]
    units["other_review_property_change_count"] = units[changed_cols].sum(axis=1).astype("int16")
    units["hard_suspicious_unit"] = (
        base_mask
        & units["growth_ratio_30d"].ge(growth_cutoff)
        & units["standardized_rating_shift"].ge(shift_q75)
    )
    units["hard_strong_suspicious_unit"] = (
        base_mask
        & units["growth_ratio_30d"].ge(growth_cutoff)
        & units["standardized_rating_shift"].ge(shift_q90)
    )
    units["hard_normal_burst_unit"] = (
        base_mask
        & units["growth_ratio_30d"].ge(growth_cutoff)
        & units["standardized_rating_shift"].le(shift_q50)
        & units["prior_product_review_count_median"].ge(history_cutoff)
        & units["other_review_property_change_count"].le(1)
    )

    unit_role_cols = [
        "prod_id",
        "week",
        "standardized_rating_shift",
        "hard_suspicious_unit",
        "hard_strong_suspicious_unit",
        "hard_normal_burst_unit",
        "other_review_property_change_count",
    ]
    role_nodes = df.merge(units[unit_role_cols], on=["prod_id", "week"], how="left")
    role_nodes["hard_suspicious_unit"] = role_nodes["hard_suspicious_unit"].fillna(False).astype(bool)
    role_nodes["hard_normal_burst_unit"] = role_nodes["hard_normal_burst_unit"].fillna(False).astype(bool)
    role_nodes["hard_strong_suspicious_unit"] = role_nodes["hard_strong_suspicious_unit"].fillna(False).astype(bool)

    suspicious_nodes = role_nodes.loc[role_nodes["hard_suspicious_unit"]].copy()
    normal_nodes = role_nodes.loc[role_nodes["hard_normal_burst_unit"]].copy()
    hard_idx = set(int(value) for value in suspicious_nodes["node_idx"].tolist())
    hard_idx.update(int(value) for value in normal_nodes["node_idx"].tolist())
    selected_idx.update(hard_idx)

    if len(selected_idx) > args.max_nodes:
        raise ValueError(
            f"Legacy plus hard seeds selected {len(selected_idx):,} nodes, exceeding --max-nodes={args.max_nodes:,}."
        )

    selected_core = df.loc[df["node_idx"].isin(selected_idx)].copy()
    fill_budget = int(args.max_nodes - len(selected_idx))
    remaining_base = df.loc[
        base_node_mask & ~df["node_idx"].isin(selected_idx)
    ].copy()
    fill_nodes = deterministic_stratified_fill_like(
        remaining_base,
        fill_budget,
        selected_core,
        ["year", "rating_direction", "history_bin"],
        ["date", "node_idx"],
        [True, True],
    )
    selected_idx.update(int(value) for value in fill_nodes["node_idx"].tolist())

    sampled_nodes = make_sampled_nodes_from_idx(df, selected_idx)
    suspicious_idx = set(int(value) for value in suspicious_nodes["node_idx"].tolist())
    normal_idx = set(int(value) for value in normal_nodes["node_idx"].tolist())
    fill_idx = set(int(value) for value in fill_nodes["node_idx"].tolist())
    legacy_idx = set(int(value) for value in df.loc[legacy_node_mask, "node_idx"].tolist())

    def role_for_node(node_idx: int) -> str:
        if node_idx in suspicious_idx:
            return "suspicious_seed"
        if node_idx in normal_idx:
            return "normal_burst_seed"
        if node_idx in fill_idx:
            return "distribution_fill"
        if node_idx in legacy_idx:
            return "legacy_core"
        return "selected"

    original_ids = sampled_nodes["original_node_idx"].astype(int)
    sampled_nodes["sample_role"] = [role_for_node(int(value)) for value in original_ids]
    sampled_nodes["is_target_node"] = True
    sampled_nodes["is_legacy_core"] = original_ids.isin(legacy_idx)
    sampled_nodes["is_suspicious_seed"] = original_ids.isin(suspicious_idx)
    sampled_nodes["is_normal_burst_seed"] = original_ids.isin(normal_idx)
    sampled_nodes["is_strong_suspicious_seed"] = original_ids.isin(
        set(int(value) for value in role_nodes.loc[role_nodes["hard_strong_suspicious_unit"], "node_idx"].tolist())
    )

    selected_unit_keys = sampled_nodes[["prod_id", "week"]].drop_duplicates().copy()
    selected_unit_keys["_selected_unit"] = True
    units = units.drop(columns=["_selected_unit"], errors="ignore").merge(
        selected_unit_keys, on=["prod_id", "week"], how="left"
    )
    units["selected"] = units["_selected_unit"].fillna(False).astype(bool)
    units = units.drop(columns=["_selected_unit"])

    role_counts = sampled_nodes["sample_role"].value_counts().to_dict()
    details = {
        "strategy": "distribution_preserved_hard",
        "max_nodes": int(args.max_nodes),
        "legacy_nodes": int(len(legacy_idx)),
        "hard_suspicious_candidate_nodes": int(len(suspicious_nodes)),
        "hard_normal_candidate_nodes": int(len(normal_nodes)),
        "distribution_fill_nodes": int(len(fill_nodes)),
        "selected_nodes": int(len(sampled_nodes)),
        "role_counts": role_counts,
        "selected_fake_rate_for_diagnostics_only": float(sampled_nodes["is_fake"].mean()) if len(sampled_nodes) else None,
        "growth_ratio_q75": growth_cutoff,
        "std_floor_q25": std_floor,
        "standardized_shift_q50": shift_q50,
        "standardized_shift_q75": shift_q75,
        "standardized_shift_q90": shift_q90,
        "history_q50_in_growth": history_cutoff,
        "other_change_thresholds": change_thresholds,
        "notes": [
            "Labels are not used for node selection; fake rate is diagnostic only.",
            "Legacy selected product-week nodes are preserved first.",
            "Suspicious/normal hard nodes are added from the same base gate, then remaining capacity is filled by year/rating_direction/history_bin strata.",
        ],
    }
    return sampled_nodes, units, details


def subset_numpy_artifacts(
    sampled_nodes: pd.DataFrame,
    processed_dir: Path,
    output_dir: Path,
) -> None:
    log("Saving sampled numpy artifacts.")
    selected_idx = sampled_nodes["original_node_idx"].to_numpy(dtype=np.int64)
    features = np.load(processed_dir / "node_features_numeric.npy", mmap_mode="r")
    labels = np.load(processed_dir / "node_labels.npy", mmap_mode="r")
    review_ids = np.load(processed_dir / "node_review_ids.npy", mmap_mode="r")

    sampled_features = np.asarray(features[selected_idx], dtype=np.float32)
    sampled_labels = np.asarray(labels[selected_idx], dtype=np.int64)
    sampled_review_ids = np.asarray(review_ids[selected_idx], dtype=np.int64)

    if not np.array_equal(sampled_labels, sampled_nodes["is_fake"].to_numpy(dtype=np.int64)):
        raise ValueError("Sampled labels do not match sampled node metadata.")
    if not np.array_equal(sampled_review_ids, sampled_nodes["review_id"].to_numpy(dtype=np.int64)):
        raise ValueError("Sampled review_id array does not match sampled node metadata.")

    np.save(output_dir / "sampled_node_features_numeric.npy", sampled_features)
    np.save(output_dir / "sampled_node_labels.npy", sampled_labels)
    np.save(output_dir / "sampled_node_review_ids.npy", sampled_review_ids)
    save_pyg_node_graph(sampled_nodes, sampled_features, sampled_labels, output_dir)


# PyTorch Geometric 형식의 엣지 없는 샘플 그래프를 저장한다.
# 다음 그래프 설계 단계에서 sampled_node_idx 기준 edge_index를 채우면 된다.
def save_pyg_node_graph(
    sampled_nodes: pd.DataFrame,
    sampled_features: np.ndarray,
    sampled_labels: np.ndarray,
    output_dir: Path,
) -> None:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        log("PyTorch Geometric is not available; skipping sampled_review_node_graph_no_edges.pt.")
        return

    data = Data(
        x=torch.from_numpy(sampled_features.astype("float32").copy()),
        y=torch.from_numpy(sampled_labels.astype("int64").copy()),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    data.sampled_node_idx = torch.from_numpy(sampled_nodes["sampled_node_idx"].to_numpy(dtype=np.int64).copy())
    data.original_node_idx = torch.from_numpy(sampled_nodes["original_node_idx"].to_numpy(dtype=np.int64).copy())
    data.review_id = torch.from_numpy(sampled_nodes["review_id"].to_numpy(dtype=np.int64).copy())
    torch.save(data, output_dir / "sampled_review_node_graph_no_edges.pt")
    log("Saved sampled PyG node graph without edges.")


# 선택된 노드와 텍스트, relation 후보 키, mapping, 상품-주 후보 테이블을 저장한다.
# relation 후보 키에도 sampled_node_idx를 붙여 다음 단계에서 바로 edge_index를 만들 수 있게 한다.
def save_tabular_artifacts(
    sampled_nodes: pd.DataFrame,
    units: pd.DataFrame,
    processed_dir: Path,
    output_dir: Path,
    context_edges: pd.DataFrame | None = None,
) -> None:
    log("Saving sampled tabular artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = sampled_nodes[["sampled_node_idx", "original_node_idx", "review_id"]].copy()
    mapping.to_csv(output_dir / "sampled_node_mapping.csv", index=False, encoding="utf-8")

    sampled_nodes.to_csv(
        output_dir / "sampled_review_nodes.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    review_text_path = processed_dir / "review_text.csv.gz"
    if review_text_path.exists():
        review_text = pd.read_csv(review_text_path)
        review_text = review_text.rename(columns={"node_idx": "original_node_idx"})
        sampled_text = mapping.merge(review_text, on=["original_node_idx", "review_id"], how="left")
        sampled_text.to_csv(
            output_dir / "sampled_review_text.csv.gz",
            index=False,
            encoding="utf-8",
            compression="gzip",
        )

    relation_path = processed_dir / "relation_candidate_keys.csv.gz"
    if relation_path.exists():
        relation_keys = pd.read_csv(relation_path)
        relation_keys = relation_keys.rename(columns={"node_idx": "original_node_idx"})
        sampled_relation_keys = mapping.merge(relation_keys, on=["original_node_idx", "review_id"], how="left")
        sampled_relation_keys.to_csv(
            output_dir / "sampled_relation_candidate_keys.csv.gz",
            index=False,
            encoding="utf-8",
            compression="gzip",
        )

    units.to_csv(
        output_dir / "product_week_sampling_units.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    if context_edges is not None:
        context_edges.to_csv(
            output_dir / "sampled_context_edges.csv.gz",
            index=False,
            encoding="utf-8",
            compression="gzip",
        )


# 샘플링 기준, 규모, label 사후 진단, flag 분포를 JSON으로 요약한다.
# 라벨 진단은 표본 품질 확인용이며 샘플 선택 규칙에는 사용하지 않는다.
def save_summary(
    df: pd.DataFrame,
    sampled_nodes: pd.DataFrame,
    units: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
    args: argparse.Namespace,
    sampling_details: dict[str, Any] | None = None,
) -> None:
    log("Saving sampling summary.")
    selected_units = units.loc[units["selected"]].copy()
    flag_columns = [
        "review_growth_flag",
        "new_user_ratio_flag",
        "extreme_rating_ratio_flag",
        "short_review_ratio_flag",
        "weak_product_flag",
        "rating_deviation_flag",
        "rating_direction_concentration_flag",
        "local_template_repeat_flag",
    ]

    product_week_quantiles = {
        "n_reviews": {
            str(q): float(v)
            for q, v in units["n_reviews"].quantile([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995]).items()
        },
        "n_users": {
            str(q): float(v)
            for q, v in units["n_users"].quantile([0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995]).items()
        },
    }

    summary = {
        "input_nodes": int(len(df)),
        "input_products": int(df["prod_id"].nunique()),
        "input_users": int(df["user_id"].nunique()),
        "input_product_week_units": int(len(units)),
        "week_frequency": WEEK_FREQ,
        "strategy": args.strategy,
        "strategy_details": sampling_details or {},
        "base_candidate_rule": {
            "min_reviews_per_product_week": int(args.min_reviews),
            "min_users_per_product_week": int(args.min_users),
        },
        "flag_rule": {
            "quantile_for_7_numeric_flags": float(args.flag_quantile),
            "min_flags_required": int(args.min_flags),
            "local_template_repeat_flag": "True if at least one review in the product-week has same_text_count_in_product_week >= 2.",
        },
        "flag_thresholds": thresholds,
        "product_week_quantiles": product_week_quantiles,
        "base_candidates": {
            "units": int(units["base_candidate"].sum()),
            "nodes": int(units.loc[units["base_candidate"], "n_reviews"].sum()),
        },
        "selected": {
            "product_week_units": int(len(selected_units)),
            "nodes": int(len(sampled_nodes)),
            "node_share_pct": float(len(sampled_nodes) / len(df) * 100),
            "products": int(sampled_nodes["prod_id"].nunique()),
            "users": int(sampled_nodes["user_id"].nunique()),
            "date_min": str(sampled_nodes["date"].min().date()),
            "date_max": str(sampled_nodes["date"].max().date()),
            "label_counts_for_diagnostics_only": sampled_nodes["is_fake"].value_counts().sort_index().to_dict(),
            "fake_rate_for_diagnostics_only": float(sampled_nodes["is_fake"].mean()),
        },
        "selected_flag_prevalence_by_unit": {
            flag: {
                "units": int(selected_units[flag].sum()),
                "share": float(selected_units[flag].mean()),
            }
            for flag in flag_columns
        },
        "notes": [
            "The target label is never used to choose sampled nodes.",
            "The legacy strategy samples product-week units; rur_shock_context samples rating-shock seeds plus causal product/user context and normal hard negatives; legacy_plus_shock_context preserves legacy units first, then adds weak-shock context.",
            "Edges are still not constructed in this step.",
            "sampled_node_idx is the row index for sampled feature arrays; original_node_idx preserves the preprocessed node_idx.",
        ],
    }

    with (output_dir / "sampling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)


# 샘플링 산출물의 연결 구조를 간단히 설명하는 README를 저장한다.
# 보고서와 별개로 다음 코드 단계에서 어떤 파일을 읽어야 하는지 빠르게 확인하기 위한 문서다.
def save_readme(output_dir: Path) -> None:
    readme = """# Sampled YelpZip Review Nodes

이 폴더는 `Sampling.py`가 생성한 상품-주 기반 서브그래프 샘플링 산출물이다.

- `sampled_review_nodes.csv.gz`: 선택된 리뷰 노드 메타데이터와 샘플링용 `sampled_node_idx`.
- `sampled_review_text.csv.gz`: 선택된 리뷰의 원문 텍스트.
- `sampled_relation_candidate_keys.csv.gz`: 선택된 노드의 relation 후보 키. 아직 엣지는 만들지 않았다.
- `sampled_node_features_numeric.npy`: 선택된 노드의 숫자형 피처 행렬.
- `sampled_node_labels.npy`: 선택된 노드의 라벨 배열.
- `sampled_node_review_ids.npy`: 선택된 노드 행과 원본 `review_id`의 매핑.
- `sampled_review_node_graph_no_edges.pt`: PyTorch Geometric `Data` 객체. `edge_index`는 빈 텐서이다.
- `sampled_node_mapping.csv`: `sampled_node_idx`, `original_node_idx`, `review_id` 연결표.
- `product_week_sampling_units.csv.gz`: 전체 상품-주 후보의 flag와 선택 여부.
- `sampling_summary.json`: 샘플링 기준과 결과 요약.

주의: 다음 단계에서 엣지를 만들 때는 `sampled_node_idx`를 기준으로 `edge_index`를 구성해야 한다.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


# 전체 샘플링 파이프라인을 실행한다.
# 상품-주 구간을 선택한 뒤 선택 구간의 모든 리뷰 노드를 샘플 그래프 노드로 저장한다.
def run_sampling(args: argparse.Namespace) -> None:
    processed_dir = args.processed_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_review_nodes(processed_dir)
    units = build_product_week_units(df)
    units, thresholds = apply_sampling_flags(
        units,
        min_reviews=args.min_reviews,
        min_users=args.min_users,
        flag_quantile=args.flag_quantile,
        min_flags=args.min_flags,
        flag_mode=args.flag_mode,
    )
    sampling_details: dict[str, Any] = {}
    context_edges: pd.DataFrame | None = None
    if args.strategy == "legacy":
        sampled_nodes = select_review_nodes(df, units)
        sampling_details = {"strategy": "legacy"}
    elif args.strategy == "rur_shock_context":
        sampled_nodes, units, sampling_details = select_rur_shock_context_nodes(df, units, args)
    elif args.strategy == "legacy_plus_shock_context":
        sampled_nodes, units, sampling_details = select_legacy_plus_shock_context_nodes(df, units, args)
    elif args.strategy == "burst_contrast_context":
        sampled_nodes, units, sampling_details, context_edges = select_burst_contrast_context_nodes(df, units, args)
    elif args.strategy == "distribution_preserved_hard":
        sampled_nodes, units, sampling_details = select_distribution_preserved_hard_nodes(df, units, args)
    else:
        raise ValueError(f"Unsupported sampling strategy: {args.strategy}")

    save_tabular_artifacts(sampled_nodes, units, processed_dir, output_dir, context_edges)
    subset_numpy_artifacts(sampled_nodes, processed_dir, output_dir)
    save_summary(df, sampled_nodes, units, thresholds, output_dir, args, sampling_details)
    save_readme(output_dir)

    log(f"Selected {len(sampled_nodes):,} review nodes ({len(sampled_nodes) / len(df) * 100:.2f}% of input).")
    log("Done. Sampled nodes are ready; edges have not been constructed.")


# 명령행 인자를 정의한다.
# 기본 실행은 `python Sampling.py`이며, 필요하면 threshold와 출력 폴더를 옵션으로 바꿀 수 있다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample YelpZip review nodes by product-week behavioral flags.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="Preprocessed data directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Sampling output directory")
    parser.add_argument(
        "--strategy",
        choices=["legacy"],
        default="legacy",
        help="Retained final strategy: product-week units selected by density gate and behavioral flags.",
    )
    parser.add_argument(
        "--contrast-seed-per-role",
        type=int,
        default=DEFAULT_CONTRAST_SEED_PER_ROLE,
        help="Maximum target seed nodes per role for burst_contrast_context. Use 0 to use the smallest role size.",
    )
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES, help="Maximum sampled review nodes")
    parser.add_argument(
        "--shock-max-prior-product-reviews",
        type=int,
        default=DEFAULT_SHOCK_MAX_PRIOR_PRODUCT_REVIEWS,
        help="Maximum prior product reviews for weak rating-shock sampling seeds",
    )
    parser.add_argument(
        "--shock-min-abs-rating-dev",
        type=float,
        default=DEFAULT_SHOCK_MIN_ABS_RATING_DEV,
        help="Minimum absolute deviation from prior product mean for rating-shock sampling seeds",
    )
    parser.add_argument(
        "--product-context-per-shock",
        type=int,
        default=DEFAULT_PRODUCT_CONTEXT_PER_SHOCK,
        help="Recent prior same-product reviews included around each shock seed",
    )
    parser.add_argument(
        "--user-context-per-shock",
        type=int,
        default=DEFAULT_USER_CONTEXT_PER_SHOCK,
        help="Recent prior same-user reviews included around each shock seed",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=DEFAULT_MIN_REVIEWS_PER_PRODUCT_WEEK,
        help="Minimum reviews in a product-week base candidate",
    )
    parser.add_argument(
        "--min-users",
        type=int,
        default=DEFAULT_MIN_USERS_PER_PRODUCT_WEEK,
        help="Minimum unique users in a product-week base candidate",
    )
    parser.add_argument(
        "--flag-quantile",
        type=float,
        default=DEFAULT_FLAG_QUANTILE,
        help="Quantile threshold for the seven numeric sampling flags",
    )
    parser.add_argument(
        "--flag-mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="Use legacy absolute flag sources or prior-history-relative flag sources.",
    )
    parser.add_argument(
        "--min-flags",
        type=int,
        default=DEFAULT_MIN_FLAGS,
        help="Minimum number of the eight flags required for final selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sampling(parse_args())

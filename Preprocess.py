"""
Build graph-ready review node data from YelpZip.

이 스크립트는 원본 리뷰 테이블을 '리뷰 노드' 중심의 전처리 산출물로 변환한다.
아직 엣지는 직접 연결하지 않고, 이후 relation 생성에 필요한 후보 키만 별도로 저장한다.
"""

# Windows/PyCharm/PowerShell 환경에서 한국어 로그가 깨지지 않도록 기본 입출력 인코딩을 맞춘다.
# 전처리 결과는 data/processed 아래에 저장하며, 원본 CSV는 수정하지 않는다.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.util as pdu


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "origin" / "yelpzip.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
CORE_COLUMNS = ["Unnamed: 0", "user_id", "prod_id", "rating", "label", "date", "text", "tag"]


# 간단한 로그 출력과 JSON 직렬화 보조 함수다.
# numpy/pandas 타입이 섞여도 summary JSON 저장이 실패하지 않도록 표준 타입으로 바꾼다.
def log(message: str) -> None:
    print(f"[Preprocess] {message}")


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


# 원본 CSV를 읽어 필수 컬럼의 타입과 라벨을 표준화한다.
# label=-1은 is_fake=1, label=1은 is_fake=0으로 바꾸고 tag는 검증용으로만 남긴다.
def load_and_standardize(csv_path: Path) -> pd.DataFrame:
    log(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=CORE_COLUMNS)
    df = df.rename(columns={"Unnamed: 0": "review_id"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].fillna("")
    df["is_fake"] = np.where(df["label"].eq(-1), 1, np.where(df["label"].eq(1), 0, np.nan))

    required = ["review_id", "user_id", "prod_id", "rating", "label", "date", "is_fake"]
    before = len(df)
    df = df.dropna(subset=required).copy()
    dropped = before - len(df)
    if dropped:
        log(f"Dropped {dropped:,} rows with missing required fields.")

    df["review_id"] = df["review_id"].astype("int64")
    df["user_id"] = df["user_id"].astype("int64")
    df["prod_id"] = df["prod_id"].astype("int64")
    df["rating"] = df["rating"].astype("float32")
    df["label"] = df["label"].astype("int8")
    df["is_fake"] = df["is_fake"].astype("int8")

    duplicate_mask = df.duplicated(subset=["user_id", "prod_id", "rating", "label", "date", "text"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        df = df.loc[~duplicate_mask].copy()
        log(f"Removed {duplicate_count:,} exact duplicate rows.")

    df = df.sort_values(["date", "review_id"]).reset_index(drop=True)
    df.insert(0, "node_idx", np.arange(len(df), dtype=np.int64))
    return df


# 리뷰 자체에서 직접 계산 가능한 평점과 텍스트 표면 피처를 만든다.
# user_id/prod_id/date 원값은 모델 피처가 아니라 후속 relation 후보 키와 과거 통계 계산에만 사용한다.
def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating review-level features.")
    df["rating_norm"] = ((df["rating"] - 3.0) / 2.0).astype("float32")
    df["rating_direction"] = np.select(
        [df["rating"].le(2), df["rating"].eq(3), df["rating"].ge(4)],
        [-1, 0, 1],
        default=0,
    ).astype("int8")
    df["rating_bucket"] = np.select(
        [df["rating"].le(2), df["rating"].eq(3), df["rating"].ge(4)],
        ["low_1_2", "mid_3", "high_4_5"],
        default="unknown",
    )
    df["extreme_rating"] = df["rating"].isin([1.0, 5.0]).astype("int8")

    text = df["text"].fillna("")
    df["char_len"] = text.str.len().astype("int32")
    df["word_len"] = text.str.split().str.len().fillna(0).astype("int32")
    df["log_word_len"] = np.log1p(df["word_len"]).astype("float32")
    df["short_review_flag"] = df["char_len"].le(100).astype("int8")
    token_lists = text.str.lower().str.findall(r"[a-z0-9']+")
    unique_counts = token_lists.map(lambda tokens: len(set(tokens))).astype("int32")
    df["unique_token_ratio"] = (
        unique_counts / df["word_len"].clip(lower=1)
    ).fillna(0.0).astype("float32")
    df["avg_token_len"] = (
        text.str.replace(r"\s+", "", regex=True).str.len() / df["word_len"].clip(lower=1)
    ).fillna(0.0).astype("float32")
    df["numeric_token_flag"] = text.str.contains(r"\d", regex=True).astype("int8")

    upper_count = text.str.count(r"[A-Z]").astype("int32")
    df["upper_ratio"] = (upper_count / df["char_len"].clip(lower=1)).astype("float32")
    df["exclamation_count"] = text.str.count("!").astype("int16")
    df["question_count"] = text.str.count(r"\?").astype("int16")
    df["text_hash"] = pdu.hash_pandas_object(text, index=False).astype("uint64")

    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    return df


# 사용자별로 작성 시점 이전의 활동 이력만 계산한다.
# 전체 기간 사용자 리뷰 수를 피처로 쓰지 않고, 현재 리뷰보다 과거에 관측된 리뷰만 누적한다.
def add_prior_user_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating prior user-history features.")
    ordered = df.sort_values(["user_id", "date", "review_id"]).copy()
    grouped = ordered.groupby("user_id", sort=False)

    prior_count = grouped.cumcount().astype("int32")
    prior_rating_sum = grouped["rating"].cumsum() - ordered["rating"]
    prior_rating_sq_sum = grouped["rating"].transform(lambda s: (s.astype("float64") ** 2).cumsum()) - (
        ordered["rating"].astype("float64") ** 2
    )
    prior_extreme_sum = grouped["extreme_rating"].cumsum() - ordered["extreme_rating"]
    prev_date = grouped["date"].shift(1)
    first_user_date = grouped["date"].transform("first")

    ordered["prior_user_review_count"] = prior_count
    ordered["is_new_user_at_review_time"] = prior_count.eq(0).astype("int8")
    ordered["prior_user_avg_rating"] = (prior_rating_sum / prior_count.replace(0, np.nan)).fillna(0).astype("float32")
    prior_user_mean_sq = prior_rating_sq_sum / prior_count.replace(0, np.nan)
    prior_user_variance = (prior_user_mean_sq - ordered["prior_user_avg_rating"].astype("float64") ** 2).clip(lower=0)
    ordered["prior_user_rating_std"] = np.sqrt(prior_user_variance).fillna(0).astype("float32")
    ordered["prior_user_extreme_ratio"] = (prior_extreme_sum / prior_count.replace(0, np.nan)).fillna(0).astype("float32")
    ordered["rating_deviation_from_prior_user_mean"] = np.where(
        prior_count.gt(0),
        ordered["rating"] - ordered["prior_user_avg_rating"],
        0,
    ).astype("float32")
    ordered["abs_rating_deviation_from_prior_user_mean"] = (
        np.abs(ordered["rating_deviation_from_prior_user_mean"]).astype("float32")
    )
    ordered["days_since_user_last_review"] = (ordered["date"] - prev_date).dt.days.fillna(-1).astype("int32")
    ordered["has_prior_user_review_7d"] = ordered["days_since_user_last_review"].between(0, 7).astype("int8")
    ordered["has_prior_user_review_30d"] = ordered["days_since_user_last_review"].between(0, 30).astype("int8")
    ordered["prior_user_active_span_days"] = np.where(
        prior_count.gt(0),
        (ordered["date"] - first_user_date).dt.days,
        0,
    ).astype("int32")
    ordered["log1p_prior_user_review_count"] = np.log1p(ordered["prior_user_review_count"]).astype("float32")
    ordered["log1p_days_since_user_last_review"] = np.log1p(
        ordered["days_since_user_last_review"].clip(lower=0)
    ).astype("float32")
    ordered["log1p_prior_user_active_span_days"] = np.log1p(
        ordered["prior_user_active_span_days"].clip(lower=0)
    ).astype("float32")

    cols = [
        "node_idx",
        "prior_user_review_count",
        "is_new_user_at_review_time",
        "prior_user_avg_rating",
        "prior_user_rating_std",
        "prior_user_extreme_ratio",
        "rating_deviation_from_prior_user_mean",
        "abs_rating_deviation_from_prior_user_mean",
        "days_since_user_last_review",
        "has_prior_user_review_7d",
        "has_prior_user_review_30d",
        "prior_user_active_span_days",
        "log1p_prior_user_review_count",
        "log1p_days_since_user_last_review",
        "log1p_prior_user_active_span_days",
    ]
    return df.merge(ordered[cols], on="node_idx", how="left")


# 상품별로 작성 시점 이전의 평판 상태와 최근 활동량을 계산한다.
# 같은 날짜의 리뷰는 선후를 알 수 없으므로 최근 7/30일 카운트에서는 현재 날짜 이전 리뷰만 사용한다.
def add_prior_product_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating prior product-history and recent activity features.")
    ordered = df.sort_values(["prod_id", "date", "review_id"]).copy()
    grouped = ordered.groupby("prod_id", sort=False)

    prior_count = grouped.cumcount().astype("int32")
    prior_rating_sum = grouped["rating"].cumsum() - ordered["rating"]
    prior_rating_sq_sum = grouped["rating"].transform(lambda s: (s.astype("float64") ** 2).cumsum()) - (
        ordered["rating"].astype("float64") ** 2
    )
    prior_extreme_sum = grouped["extreme_rating"].cumsum() - ordered["extreme_rating"]
    first_product_date = grouped["date"].transform("first")

    ordered["prior_product_review_count"] = prior_count
    ordered["prior_product_avg_rating"] = (prior_rating_sum / prior_count.replace(0, np.nan)).fillna(0).astype("float32")
    prior_mean_sq = prior_rating_sq_sum / prior_count.replace(0, np.nan)
    prior_variance = (prior_mean_sq - ordered["prior_product_avg_rating"].astype("float64") ** 2).clip(lower=0)
    ordered["prior_product_rating_std"] = np.sqrt(prior_variance).fillna(0).astype("float32")
    ordered["prior_product_extreme_ratio"] = (prior_extreme_sum / prior_count.replace(0, np.nan)).fillna(0).astype("float32")
    ordered["log1p_prior_product_review_count"] = np.log1p(ordered["prior_product_review_count"]).astype("float32")
    ordered["rating_deviation_from_prior_product_mean"] = np.where(
        prior_count.gt(0),
        ordered["rating"] - ordered["prior_product_avg_rating"],
        0,
    ).astype("float32")
    ordered["rating_impact_signed"] = np.where(
        prior_count.gt(0),
        ordered["rating_deviation_from_prior_product_mean"] / (prior_count + 1),
        0,
    ).astype("float32")
    ordered["rating_impact_abs"] = np.abs(ordered["rating_impact_signed"]).astype("float32")
    ordered["product_age_days"] = np.where(
        prior_count.gt(0),
        (ordered["date"] - first_product_date).dt.days,
        0,
    ).astype("int32")
    ordered["log1p_product_age_days"] = np.log1p(ordered["product_age_days"].clip(lower=0)).astype("float32")

    recent = compute_prior_product_window_counts(ordered, windows=(7, 30))
    ordered = ordered.merge(recent, on="node_idx", how="left")
    ordered["log1p_product_reviews_last_7d"] = np.log1p(ordered["product_reviews_last_7d"]).astype("float32")
    ordered["log1p_product_reviews_last_30d"] = np.log1p(ordered["product_reviews_last_30d"]).astype("float32")

    cols = [
        "node_idx",
        "prior_product_review_count",
        "prior_product_avg_rating",
        "prior_product_rating_std",
        "prior_product_extreme_ratio",
        "log1p_prior_product_review_count",
        "product_reviews_last_7d",
        "product_reviews_last_30d",
        "log1p_product_reviews_last_7d",
        "log1p_product_reviews_last_30d",
        "rating_deviation_from_prior_product_mean",
        "rating_impact_signed",
        "rating_impact_abs",
        "product_age_days",
        "log1p_product_age_days",
    ]
    return df.merge(ordered[cols], on="node_idx", how="left")


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating interpretable interaction features.")
    df["weak_product_prior_count_flag"] = df["prior_product_review_count"].le(30).astype("int8")
    df["new_user_short_extreme_flag"] = (
        df["is_new_user_at_review_time"].eq(1)
        & df["short_review_flag"].eq(1)
        & df["extreme_rating"].eq(1)
    ).astype("int8")
    df["short_weak_product_flag"] = (
        df["short_review_flag"].eq(1) & df["weak_product_prior_count_flag"].eq(1)
    ).astype("int8")
    df["extreme_rating_impact_abs"] = (
        df["extreme_rating"].astype("float32") * df["rating_impact_abs"].astype("float32")
    ).astype("float32")
    df["new_user_rating_impact_abs"] = (
        df["is_new_user_at_review_time"].astype("float32") * df["rating_impact_abs"].astype("float32")
    ).astype("float32")
    return df


# 상품별 최근 7일/30일 리뷰 수를 searchsorted로 계산한다.
# 현재 날짜의 리뷰는 제외해 미래 또는 같은 날짜 배치 정보가 노드 피처에 섞이지 않게 한다.
def compute_prior_product_window_counts(ordered: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    ns_per_day = 24 * 60 * 60 * 1_000_000_000

    for _, group in ordered.groupby("prod_id", sort=False):
        date_days = (group["date"].astype("int64") // ns_per_day).to_numpy()
        node_idx = group["node_idx"].to_numpy()
        data: dict[str, Any] = {"node_idx": node_idx}
        for window in windows:
            left = np.searchsorted(date_days, date_days - window, side="left")
            right = np.searchsorted(date_days, date_days, side="left")
            data[f"product_reviews_last_{window}d"] = (right - left).astype("int32")
        pieces.append(pd.DataFrame(data))

    return pd.concat(pieces, ignore_index=True)


# 상품-시간-평점 방향과 텍스트 반복성 정보를 계산한다.
# 버스트 크기는 노드 피처 행에는 보존하되, 기본 feature matrix에는 넣지 않고 후속 엣지/샘플링 후보 정보로 사용한다.
def add_relation_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Creating relation candidate keys and non-input diagnostic counts.")
    df["product_day_review_count"] = df.groupby(["prod_id", "date"])["node_idx"].transform("size").astype("int32")
    df["product_week_rating_bucket_size"] = (
        df.groupby(["prod_id", "week", "rating_bucket"])["node_idx"].transform("size").astype("int32")
    )
    df["same_text_count_in_product_week"] = (
        df.groupby(["prod_id", "week", "text_hash"])["node_idx"].transform("size").astype("int32")
    )
    df["log1p_same_text_count_in_product_week"] = np.log1p(df["same_text_count_in_product_week"]).astype("float32")
    return df


# 모델에 바로 넣을 수 있는 숫자형 노드 피처 행렬을 만든다.
# raw ID, tag, 전체 기간 집계, 버스트 크기처럼 엣지 후보와 중복되는 정보는 feature matrix에서 제외한다.
BEHAVIOR_SHIFT_FEATURE_COLUMNS = [
    "same_dir_log_count_lift_4w",
    "total_log_count_lift_4w",
    "direction_concentration_lift_4w",
    "new_user_ratio_lift_4w",
    "short_review_ratio_lift_4w",
    "word_len_drop_ratio_4w",
    "abuse_burst_behavior_score_4w",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    numerator_values = numerator.astype("float64").to_numpy()
    denominator_values = denominator.astype("float64").to_numpy()
    return np.divide(
        numerator_values,
        denominator_values,
        out=np.zeros_like(numerator_values, dtype=np.float64),
        where=denominator_values > 0,
    )


def add_prior_week_sums(
    weekly: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    prior_weeks: int,
) -> pd.DataFrame:
    prior_name_by_column = {
        col: f"prior{prior_weeks}_{col}" if col.endswith("_sum") else f"prior{prior_weeks}_{col}_sum"
        for col in value_columns
    }
    prior_columns = [prior_name_by_column[col] for col in value_columns]
    pieces: list[pd.DataFrame] = []

    for key, group in weekly.groupby(group_columns, sort=False, observed=True):
        group = group.sort_values("week").set_index("week")
        full_weeks = pd.date_range(group.index.min(), group.index.max(), freq="7D")
        observed_weeks = group.index
        expanded = group.reindex(full_weeks)

        for col in value_columns:
            values = expanded[col].fillna(0.0)
            expanded[prior_name_by_column[col]] = (
                values.shift(1).rolling(prior_weeks, min_periods=1).sum().fillna(0.0)
            )

        if not isinstance(key, tuple):
            key = (key,)
        for col, value in zip(group_columns, key):
            expanded[col] = value
        expanded["week"] = expanded.index
        pieces.append(expanded.loc[observed_weeks, group_columns + ["week"] + prior_columns].reset_index(drop=True))

    if not pieces:
        return pd.DataFrame(columns=group_columns + ["week"] + prior_columns)
    return pd.concat(pieces, ignore_index=True)


def add_behavior_shift_features(df: pd.DataFrame, prior_weeks: int = 4) -> pd.DataFrame:
    log("Creating prior-4-week product/rating-direction behavior shift features.")
    if prior_weeks <= 0:
        raise ValueError(f"prior_weeks must be positive, got {prior_weeks}.")

    work = df[
        [
            "node_idx",
            "prod_id",
            "week",
            "rating_direction",
            "is_new_user_at_review_time",
            "short_review_flag",
            "word_len",
        ]
    ].copy()

    total_week = (
        work.groupby(["prod_id", "week"], observed=True)
        .agg(product_week_review_count=("node_idx", "size"))
        .reset_index()
    )
    total_prior = add_prior_week_sums(
        total_week,
        group_columns=["prod_id"],
        value_columns=["product_week_review_count"],
        prior_weeks=prior_weeks,
    )

    direction_week = (
        work.groupby(["prod_id", "week", "rating_direction"], observed=True)
        .agg(
            same_dir_review_count=("node_idx", "size"),
            same_dir_new_user_sum=("is_new_user_at_review_time", "sum"),
            same_dir_short_review_sum=("short_review_flag", "sum"),
            same_dir_word_len_sum=("word_len", "sum"),
        )
        .reset_index()
    )
    direction_prior = add_prior_week_sums(
        direction_week,
        group_columns=["prod_id", "rating_direction"],
        value_columns=[
            "same_dir_review_count",
            "same_dir_new_user_sum",
            "same_dir_short_review_sum",
            "same_dir_word_len_sum",
        ],
        prior_weeks=prior_weeks,
    )

    features = direction_week.merge(total_week, on=["prod_id", "week"], how="left")
    features = features.merge(total_prior, on=["prod_id", "week"], how="left")
    features = features.merge(direction_prior, on=["prod_id", "week", "rating_direction"], how="left")
    features = features.fillna(0.0)

    prior_same_count = features[f"prior{prior_weeks}_same_dir_review_count_sum"].astype("float64")
    prior_total_count = features[f"prior{prior_weeks}_product_week_review_count_sum"].astype("float64")
    prior_same_weekly_avg = prior_same_count / float(prior_weeks)
    prior_total_weekly_avg = prior_total_count / float(prior_weeks)

    current_same_count = features["same_dir_review_count"].astype("float64")
    current_total_count = features["product_week_review_count"].astype("float64")
    current_new_ratio = safe_ratio(features["same_dir_new_user_sum"], features["same_dir_review_count"])
    current_short_ratio = safe_ratio(features["same_dir_short_review_sum"], features["same_dir_review_count"])
    current_word_mean = safe_ratio(features["same_dir_word_len_sum"], features["same_dir_review_count"])

    prior_new_ratio = safe_ratio(features[f"prior{prior_weeks}_same_dir_new_user_sum"], prior_same_count)
    prior_short_ratio = safe_ratio(features[f"prior{prior_weeks}_same_dir_short_review_sum"], prior_same_count)
    prior_word_mean = safe_ratio(features[f"prior{prior_weeks}_same_dir_word_len_sum"], prior_same_count)

    current_direction_concentration = safe_ratio(features["same_dir_review_count"], features["product_week_review_count"])
    prior_direction_concentration = safe_ratio(prior_same_count, prior_total_count)

    features["same_dir_log_count_lift_4w"] = np.log1p(current_same_count) - np.log1p(prior_same_weekly_avg)
    features["total_log_count_lift_4w"] = np.log1p(current_total_count) - np.log1p(prior_total_weekly_avg)
    features["direction_concentration_lift_4w"] = current_direction_concentration - prior_direction_concentration
    features["new_user_ratio_lift_4w"] = current_new_ratio - prior_new_ratio
    features["short_review_ratio_lift_4w"] = current_short_ratio - prior_short_ratio
    word_len_drop_ratio = np.divide(
        prior_word_mean - current_word_mean,
        prior_word_mean,
        out=np.zeros_like(prior_word_mean, dtype=np.float64),
        where=prior_word_mean > 0,
    )
    features["word_len_drop_ratio_4w"] = np.clip(word_len_drop_ratio, -1.0, 1.0)
    features["abuse_burst_behavior_score_4w"] = (
        np.maximum(features["same_dir_log_count_lift_4w"], 0)
        + np.maximum(features["direction_concentration_lift_4w"], 0)
        + np.maximum(features["new_user_ratio_lift_4w"], 0)
        + np.maximum(features["short_review_ratio_lift_4w"], 0)
        + np.maximum(features["word_len_drop_ratio_4w"], 0)
    )

    merge_columns = ["prod_id", "week", "rating_direction"] + BEHAVIOR_SHIFT_FEATURE_COLUMNS
    df = df.merge(features[merge_columns], on=["prod_id", "week", "rating_direction"], how="left")
    df[BEHAVIOR_SHIFT_FEATURE_COLUMNS] = df[BEHAVIOR_SHIFT_FEATURE_COLUMNS].fillna(0.0).astype("float32")
    return df


def build_numeric_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns = [
        "rating_norm",
        "rating_direction",
        "extreme_rating",
        "log_word_len",
        "short_review_flag",
        "unique_token_ratio",
        "avg_token_len",
        "numeric_token_flag",
        "upper_ratio",
        "exclamation_count",
        "question_count",
        "log1p_same_text_count_in_product_week",
        "log1p_prior_user_review_count",
        "is_new_user_at_review_time",
        "log1p_days_since_user_last_review",
        "has_prior_user_review_7d",
        "has_prior_user_review_30d",
        "log1p_prior_user_active_span_days",
        "prior_user_avg_rating",
        "prior_user_rating_std",
        "prior_user_extreme_ratio",
        "abs_rating_deviation_from_prior_user_mean",
        "log1p_prior_product_review_count",
        "prior_product_avg_rating",
        "prior_product_rating_std",
        "prior_product_extreme_ratio",
        "log1p_product_reviews_last_7d",
        "log1p_product_reviews_last_30d",
        "rating_deviation_from_prior_product_mean",
        "rating_impact_signed",
        "rating_impact_abs",
        "log1p_product_age_days",
        "weak_product_prior_count_flag",
        "new_user_short_extreme_flag",
        "short_weak_product_flag",
        "extreme_rating_impact_abs",
        "new_user_rating_impact_abs",
    ]
    matrix = df[feature_columns].astype("float32").to_numpy()
    return matrix, feature_columns


# 전처리 산출물을 저장한다.
# 노드 메타데이터, 원문 텍스트, relation 후보 키, 숫자 피처 행렬, 라벨을 분리해 후속 단계에서 재사용하기 쉽게 한다.
def save_outputs(df: pd.DataFrame, feature_matrix: np.ndarray, feature_columns: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    node_columns = [
        "node_idx",
        "review_id",
        "user_id",
        "prod_id",
        "date",
        "rating",
        "rating_norm",
        "rating_direction",
        "rating_bucket",
        "extreme_rating",
        "char_len",
        "word_len",
        "log_word_len",
        "short_review_flag",
        "unique_token_ratio",
        "avg_token_len",
        "numeric_token_flag",
        "upper_ratio",
        "exclamation_count",
        "question_count",
        "text_hash",
        "same_text_count_in_product_week",
        "log1p_same_text_count_in_product_week",
        "prior_user_review_count",
        "is_new_user_at_review_time",
        "prior_user_avg_rating",
        "prior_user_rating_std",
        "prior_user_extreme_ratio",
        "rating_deviation_from_prior_user_mean",
        "abs_rating_deviation_from_prior_user_mean",
        "days_since_user_last_review",
        "has_prior_user_review_7d",
        "has_prior_user_review_30d",
        "prior_user_active_span_days",
        "log1p_prior_user_review_count",
        "log1p_days_since_user_last_review",
        "log1p_prior_user_active_span_days",
        "prior_product_review_count",
        "prior_product_avg_rating",
        "prior_product_rating_std",
        "prior_product_extreme_ratio",
        "log1p_prior_product_review_count",
        "product_reviews_last_7d",
        "product_reviews_last_30d",
        "log1p_product_reviews_last_7d",
        "log1p_product_reviews_last_30d",
        "rating_deviation_from_prior_product_mean",
        "rating_impact_signed",
        "rating_impact_abs",
        "product_age_days",
        "log1p_product_age_days",
        "weak_product_prior_count_flag",
        "new_user_short_extreme_flag",
        "short_weak_product_flag",
        "extreme_rating_impact_abs",
        "new_user_rating_impact_abs",
        *BEHAVIOR_SHIFT_FEATURE_COLUMNS,
        "product_day_review_count",
        "product_week_rating_bucket_size",
        "is_fake",
    ]
    relation_columns = [
        "node_idx",
        "review_id",
        "user_id",
        "prod_id",
        "date",
        "week",
        "rating_direction",
        "rating_bucket",
        "product_day_review_count",
        "product_week_rating_bucket_size",
        "is_fake",
    ]

    log("Saving graph-ready node metadata.")
    df[node_columns].to_csv(output_dir / "review_nodes.csv.gz", index=False, encoding="utf-8", compression="gzip")

    log("Saving review text separately for later embedding/dashboard use.")
    df[["node_idx", "review_id", "text"]].to_csv(
        output_dir / "review_text.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    log("Saving relation candidate keys without constructing edges.")
    df[relation_columns].to_csv(
        output_dir / "relation_candidate_keys.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )

    log("Saving numeric node feature matrix and labels.")
    np.save(output_dir / "node_features_numeric.npy", feature_matrix.astype("float32"))
    np.save(output_dir / "node_labels.npy", df["is_fake"].astype("int64").to_numpy())
    np.save(output_dir / "node_review_ids.npy", df["review_id"].astype("int64").to_numpy())
    save_pyg_node_graph(df, feature_matrix, output_dir)

    feature_spec = {
        "numeric_feature_columns": feature_columns,
        "target_column": "is_fake",
        "node_id_column": "node_idx",
        "review_id_column": "review_id",
        "excluded_from_model_input": [
            "tag",
            "label",
            "user_id",
            "prod_id",
            "date",
            "text_hash",
            "product_day_review_count",
            "product_week_rating_bucket_size",
            "raw text",
            "full-period user/product aggregates",
            "any label-rate aggregate",
        ],
        "deferred_features": {
            "text_embedding": "Fit TF-IDF/SVD or another text encoder after sampling and train/valid/test split to avoid leakage.",
            "edges": "No edge_index is created in this script. Use relation_candidate_keys.csv.gz in the next graph-design step.",
        },
        "pyg_artifact": {
            "file": "review_node_graph_no_edges.pt",
            "description": "PyTorch Geometric Data object with x, y, node_idx, review_id and an empty edge_index.",
        },
        "planned_relations": {
            "basic_relation": "same_user_prior_recent",
            "custom_relation": "weak_product_rating_shock_edge",
            "optional_context_relation": "product_prior_context",
        },
    }
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(feature_spec), f, ensure_ascii=False, indent=2)


# PyTorch Geometric 형식의 엣지 없는 노드 그래프도 함께 저장한다.
# 다음 단계에서 edge_index만 채우면 바로 GNN 입력으로 확장할 수 있다.
def save_pyg_node_graph(df: pd.DataFrame, feature_matrix: np.ndarray, output_dir: Path) -> None:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        log("PyTorch Geometric is not available; skipping review_node_graph_no_edges.pt.")
        return

    data = Data(
        x=torch.from_numpy(feature_matrix.astype("float32").copy()),
        y=torch.from_numpy(df["is_fake"].astype("int64").to_numpy().copy()),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    data.node_idx = torch.from_numpy(df["node_idx"].astype("int64").to_numpy().copy())
    data.review_id = torch.from_numpy(df["review_id"].astype("int64").to_numpy().copy())
    torch.save(data, output_dir / "review_node_graph_no_edges.pt")
    log("Saved PyG node graph without edges.")


# 전처리 결과의 규모와 품질을 요약해 저장한다.
# 이 요약은 샘플링 전 단계에서 노드 데이터가 정상적으로 구성되었는지 확인하는 체크포인트다.
def save_summary(df: pd.DataFrame, feature_matrix: np.ndarray, output_dir: Path) -> None:
    tag_label_crosstab = pd.crosstab(df["label"], df["tag"]).to_dict()
    summary = {
        "rows_as_review_nodes": int(len(df)),
        "feature_matrix_shape": list(feature_matrix.shape),
        "label_counts": df["is_fake"].value_counts().sort_index().to_dict(),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "unique_users": int(df["user_id"].nunique()),
        "unique_products": int(df["prod_id"].nunique()),
        "tag_label_crosstab_for_validation_only": tag_label_crosstab,
        "new_user_review_share_pct": float(df["is_new_user_at_review_time"].mean() * 100),
        "short_review_share_pct": float(df["short_review_flag"].mean() * 100),
        "edge_candidate_diagnostics": {
            "product_week_rating_bucket_size_quantiles": {
                str(q): float(v)
                for q, v in df["product_week_rating_bucket_size"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()
            },
            "product_day_review_count_quantiles": {
                str(q): float(v)
                for q, v in df["product_day_review_count"].quantile([0, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()
            },
        },
        "important_notes": [
            "Each row is one review node.",
            "No edge_index is created yet.",
            "review_node_graph_no_edges.pt contains an empty edge_index with shape [2, 0].",
            "Raw user_id/prod_id/date are saved for relation construction but excluded from numeric model features.",
            "product_week_rating_bucket_size is saved as an edge/sampling candidate signal, not included in node_features_numeric.npy.",
            "Text embeddings are deferred until after sampling and split.",
        ],
    }
    with (output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    log(f"Saved summary: {output_dir / 'preprocess_summary.json'}")


# 산출물 위치와 각 파일의 역할을 설명하는 README를 만든다.
# 다음 단계에서 어떤 파일을 읽어야 하는지 헷갈리지 않게 하기 위한 문서다.
def save_readme(output_dir: Path) -> None:
    readme = """# Processed YelpZip Node Data

이 폴더는 `Preprocess.py`가 생성한 리뷰 노드 중심 전처리 산출물이다.

- `review_nodes.csv.gz`: 리뷰 노드 메타데이터, 라벨, 숫자 파생 피처. 원문 텍스트는 제외.
- `review_text.csv.gz`: `node_idx`, `review_id`, 원문 `text`. 향후 텍스트 임베딩과 대시보드 증거용.
- `relation_candidate_keys.csv.gz`: 아직 엣지는 만들지 않고, 다음 단계에서 relation을 만들기 위한 후보 키만 저장.
- `node_features_numeric.npy`: raw ID와 라벨 누수 정보를 제외한 숫자형 노드 피처 행렬.
- `node_labels.npy`: `is_fake` 타겟 배열.
- `node_review_ids.npy`: 행렬 행과 원본 리뷰 ID의 매핑.
- `review_node_graph_no_edges.pt`: PyTorch Geometric `Data` 객체. `x`, `y`, 빈 `edge_index`만 포함.
- `feature_columns.json`: 숫자 피처 컬럼 순서와 제외한 정보의 이유.
- `preprocess_summary.json`: 전처리 품질과 규모 요약.

주의: 이 단계에서는 샘플링과 엣지 연결을 수행하지 않는다.
다음 단계 순서는 샘플링 -> 그래프 네트워크 설계 -> GNN 모델링 및 최적화이다.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


# 전체 전처리 파이프라인을 순서대로 실행한다.
# 노드 피처 생성과 relation 후보 키 저장까지만 수행하고, 엣지 리스트는 만들지 않는다.
def run_preprocess(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_and_standardize(csv_path)
    df = add_review_features(df)
    df = add_prior_user_features(df)
    df = add_prior_product_features(df)
    df = add_interaction_features(df)
    df = add_relation_candidate_features(df)
    df = add_behavior_shift_features(df)

    feature_matrix, feature_columns = build_numeric_feature_matrix(df)
    save_outputs(df, feature_matrix, feature_columns, output_dir)
    save_summary(df, feature_matrix, output_dir)
    save_readme(output_dir)

    log("Done. Review nodes are ready; edges have not been constructed.")


# 명령행 인자를 정의한다.
# 기본 실행은 `python Preprocess.py`이며, 필요하면 CSV와 출력 폴더를 옵션으로 바꿀 수 있다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess YelpZip into graph-ready review node data.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to yelpzip.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocess(args.csv, args.output)

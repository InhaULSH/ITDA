"""
Build interpretable explanation tags for review-abuse predictions.

The script does not train a model. It attaches EDA/theory-aligned diagnostic
tags to an existing prediction file so reviewers can see why a node is
interesting: reputation leverage, asymmetric rating manipulation, text
specificity gap, disguised high-effort suspicion, weak template repetition, and
normal-user protection context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BuildGraphDataset import history_bin_from_train, numeric_series, weak_template_similarity_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(f"[BuildPredictionExplanations] {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def path_for_summary(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_text_sufficiency(nodes: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.Series, pd.Series]:
    prior_count = numeric_series(nodes, "prior_product_review_count")
    bins, _ = history_bin_from_train(prior_count, train_mask)
    work = nodes[["rating_bucket"]].copy()
    work["_history_bin"] = bins
    work["_word_len"] = numeric_series(nodes, "word_len")
    train = work.loc[train_mask].copy()
    global_q25 = float(train["_word_len"].quantile(0.25)) if len(train) else 0.0
    global_median = float(train["_word_len"].median()) if len(train) else 0.0
    group = (
        train.groupby(["rating_bucket", "_history_bin"], observed=True)["_word_len"]
        .agg(q25=lambda s: float(s.quantile(0.25)), median="median")
        .reset_index()
    )
    rating = (
        train.groupby("rating_bucket", observed=True)["_word_len"]
        .agg(q25_rating=lambda s: float(s.quantile(0.25)), median_rating="median")
        .reset_index()
    )
    stats = work.merge(group, on=["rating_bucket", "_history_bin"], how="left")
    stats = stats.merge(rating, on="rating_bucket", how="left")
    q25 = stats["q25"].fillna(stats["q25_rating"]).fillna(global_q25)
    median = stats["median"].fillna(stats["median_rating"]).fillna(global_median)
    short_gap = work["_word_len"].le(q25)
    sufficient = work["_word_len"].ge(median)
    return short_gap.astype(bool), sufficient.astype(bool)


def build_explanations(args: argparse.Namespace) -> None:
    nodes = pd.read_csv(args.sampled_nodes, parse_dates=["date"])
    predictions = pd.read_csv(args.predictions)
    text = np.load(args.embedding_path)
    train_mask = np.load(args.train_mask).astype(bool)
    summary = load_json(args.graph_summary)

    required = {"sampled_node_idx", "y_true", "prob_fake", "pred_label", "split"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    if len(nodes) != len(predictions):
        # Merge by sampled_node_idx if the prediction file is sorted by split rather than node order.
        merged = predictions.merge(nodes, on="sampled_node_idx", how="left", suffixes=("", "_node"))
    else:
        merged = predictions.merge(nodes, on="sampled_node_idx", how="left", suffixes=("", "_node"))
    if merged["review_id"].isna().any():
        raise ValueError("Some prediction rows could not be matched to sampled nodes.")
    merged = merged.sort_values("sampled_node_idx").reset_index(drop=True)

    product_cutoffs = summary.get("product_prior_context_feature_cutoffs", {})
    user_cutoffs = summary.get("user_context_feature_cutoffs", {})
    prior_product_q25 = float(
        product_cutoffs.get("train_prior_product_review_count_q25", numeric_series(nodes, "prior_product_review_count").loc[train_mask].quantile(0.25))
    )
    prior_product_q75 = float(
        product_cutoffs.get("train_prior_product_review_count_q75", numeric_series(nodes, "prior_product_review_count").loc[train_mask].quantile(0.75))
    )
    product_std_floor = float(product_cutoffs.get("train_prior_product_rating_std_q25_positive", 1.0))
    product_std_floor = max(product_std_floor, 1e-6)

    rating_dev = numeric_series(merged, "rating_deviation_from_prior_product_mean")
    product_std = numeric_series(merged, "prior_product_rating_std", default=product_std_floor)
    standardized_dev = rating_dev / product_std.clip(lower=product_std_floor)
    abs_standardized_dev = standardized_dev.abs()
    prior_product_count = numeric_series(merged, "prior_product_review_count")
    short_gap, text_sufficient = build_text_sufficiency(merged, train_mask)

    template_features, template_cutoffs = weak_template_similarity_features(merged, text, train_mask)
    template_q90 = float(template_cutoffs.get("train_q90_template_similarity", 0.0))

    prior_user_count = numeric_series(merged, "prior_user_review_count")
    days_since_user_last = numeric_series(merged, "days_since_user_last_review", default=-1.0)
    returning_recent = days_since_user_last.between(0, 30)
    new_user = numeric_series(merged, "is_new_user_at_review_time").eq(1)
    rating_direction = numeric_series(merged, "rating_direction")
    user_avg = numeric_series(merged, "prior_user_avg_rating")
    user_dev = np.where(prior_user_count.gt(0), numeric_series(merged, "rating") - user_avg, 0.0)
    if "prior_user_rating_std" in merged.columns:
        user_std_floor = float(user_cutoffs.get("train_prior_user_rating_std_q25_positive", 1.0))
        user_std_floor = max(user_std_floor, 1e-6)
        user_std = numeric_series(merged, "prior_user_rating_std", default=user_std_floor)
        abs_standardized_user_dev = pd.Series(np.abs(user_dev) / user_std.clip(lower=user_std_floor), index=merged.index)
    else:
        abs_standardized_user_dev = pd.Series(np.abs(user_dev), index=merged.index)
    history_user_mask = train_mask & prior_user_count.gt(0).to_numpy()
    fallback_history_user_dev = abs_standardized_user_dev.loc[history_user_mask]
    fallback_positive_user_dev = fallback_history_user_dev[fallback_history_user_dev.gt(0)]
    if not fallback_positive_user_dev.empty:
        fallback_user_consistency_cutoff = float(fallback_positive_user_dev.quantile(0.25))
    elif not fallback_history_user_dev.empty:
        fallback_user_consistency_cutoff = float(fallback_history_user_dev.quantile(0.50))
    else:
        fallback_user_consistency_cutoff = 0.0
    user_consistency_cutoff = float(
        user_cutoffs.get(
            "train_abs_standardized_user_deviation_consistency_cutoff",
            user_cutoffs.get(
                "train_abs_standardized_user_deviation_q25_positive",
                fallback_user_consistency_cutoff,
            ),
        )
    )
    user_consistent = abs_standardized_user_dev.le(user_consistency_cutoff)

    out = predictions.merge(
        merged[
            [
                "sampled_node_idx",
                "review_id",
                "user_id",
                "prod_id",
                "date",
                "rating",
                "rating_direction_group",
                "word_len",
                "prior_user_review_count",
                "prior_product_review_count",
            ]
        ],
        on="sampled_node_idx",
        how="left",
    )
    out["standardized_product_rating_deviation"] = standardized_dev.to_numpy(dtype=np.float32)
    out["abs_standardized_product_rating_deviation"] = abs_standardized_dev.to_numpy(dtype=np.float32)
    out["weak_template_similarity_max_product_week"] = template_features[:, 0]
    out["reputation_leverage_context"] = (
        prior_product_count.le(prior_product_q25) & abs_standardized_dev.ge(args.shock_threshold)
    ).astype(int)
    out["positive_promotion_context"] = (
        rating_direction.gt(0) & standardized_dev.ge(args.shock_threshold)
    ).astype(int)
    out["negative_attack_context"] = (
        rating_direction.lt(0) & (-standardized_dev).ge(args.shock_threshold)
    ).astype(int)
    out["positive_promotion_weak_reputation_context"] = (
        rating_direction.gt(0)
        & standardized_dev.ge(args.shock_threshold)
        & prior_product_count.le(prior_product_q25)
    ).astype(int)
    out["negative_attack_established_product_context"] = (
        rating_direction.lt(0)
        & (-standardized_dev).ge(args.shock_threshold)
        & prior_product_count.ge(prior_product_q75)
    ).astype(int)
    out["text_specificity_gap_context"] = short_gap.astype(int)
    out["disguised_high_effort_context"] = (
        text_sufficient & abs_standardized_dev.ge(args.shock_threshold) & (new_user | prior_user_count.le(1))
    ).astype(int)
    out["weak_template_repetition_context"] = (template_features[:, 0] >= template_q90).astype(int)
    out["normal_user_protection_context"] = (
        returning_recent & text_sufficient & prior_user_count.ge(prior_user_count.loc[train_mask].median())
    ).astype(int)
    out["normal_user_consistent_rating_context"] = (
        returning_recent & text_sufficient & user_consistent & prior_user_count.gt(0)
    ).astype(int)
    out["short_but_returning_user_context"] = (
        short_gap & returning_recent & user_consistent & prior_user_count.gt(0)
    ).astype(int)

    tag_columns = [
        "reputation_leverage_context",
        "positive_promotion_context",
        "negative_attack_context",
        "positive_promotion_weak_reputation_context",
        "negative_attack_established_product_context",
        "text_specificity_gap_context",
        "disguised_high_effort_context",
        "weak_template_repetition_context",
        "normal_user_protection_context",
        "normal_user_consistent_rating_context",
        "short_but_returning_user_context",
    ]
    tag_names = {
        "reputation_leverage_context": "평판취약상품_평점충격",
        "positive_promotion_context": "긍정홍보형_평점상승",
        "negative_attack_context": "부정공격형_평점하락",
        "positive_promotion_weak_reputation_context": "긍정홍보형_취약평판상승",
        "negative_attack_established_product_context": "부정공격형_성숙상품하락",
        "text_specificity_gap_context": "평점대비_설명부족",
        "disguised_high_effort_context": "긴리뷰형_위장가능성",
        "weak_template_repetition_context": "약한템플릿반복",
        "normal_user_protection_context": "정상사용자_보호맥락",
        "normal_user_consistent_rating_context": "정상사용자_일관평점충분설명",
        "short_but_returning_user_context": "짧지만_기존사용자일관맥락",
    }

    def join_tags(row: pd.Series) -> str:
        tags = [tag_names[col] for col in tag_columns if int(row[col]) == 1]
        return "|".join(tags)

    out["explanation_tags"] = out.apply(join_tags, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    counts = {tag_names[col]: int(out[col].sum()) for col in tag_columns}
    payload = {
        "predictions": path_for_summary(args.predictions),
        "sampled_nodes": path_for_summary(args.sampled_nodes),
        "output": path_for_summary(args.output),
        "shock_threshold": args.shock_threshold,
        "prior_product_review_count_q25": prior_product_q25,
        "prior_product_review_count_q75": prior_product_q75,
        "product_std_floor": product_std_floor,
        "user_consistency_cutoff": user_consistency_cutoff,
        "template_similarity_q90": template_q90,
        "tag_counts": counts,
        "notes": [
            "These tags are diagnostic explanations, not training labels.",
            "No label-derived aggregate is used to create the tags.",
        ],
    }
    args.summary_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved explanations: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explanation tags for saved prediction files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sampled-nodes", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--train-mask", type=Path, required=True)
    parser.add_argument("--graph-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--shock-threshold", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    build_explanations(parse_args())

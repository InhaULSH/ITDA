"""Print the headline artifacts and metrics used by the final report.

This script does not train or modify the model. It only reads the retained
validation-best campaign-quality RelationSAGE-MLP artifacts and checks that
the key numbers match the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent

EXPECTED = {
    "preprocess.rows_as_review_nodes": 608_458,
    "preprocess.unique_users": 260_239,
    "preprocess.unique_products": 5_044,
    "preprocess.fake_count": 80_439,
    "sample.nodes": 26_701,
    "sample.product_week_units": 1_957,
    "sample.fake_rate": 0.160443429085053,
    "split.train_nodes": 17_088,
    "split.valid_nodes": 4_272,
    "split.test_nodes": 5_341,
    "graph.n_nodes": 26_701,
    "graph.numeric_dim": 54,
    "graph.text_dim": 128,
    "graph.total_feature_dim": 182,
    "graph.total_edges": 21_304,
    "graph.rur_edges": 11_038,
    "graph.campaign_edges": 6_000,
    "graph.shock_edges": 4_266,
    "graph.isolated_nodes": 17_381,
    "model.best_epoch": 60,
    "model.best_threshold": 0.794,
    "valid.pr_auc": 0.4441336701873743,
    "valid.macro_f1": 0.6600831978094555,
    "test.pr_auc": 0.5288475351597811,
    "test.roc_auc": 0.8051183537691758,
    "test.macro_f1": 0.6962006747836291,
    "test.precision": 0.5273052820053715,
    "test.recall": 0.5148601398601399,
    "test.accuracy": 0.7972289833364539,
    "eda.rating_1_fake_rate": 0.323585,
    "eda.rating_5_fake_rate": 0.156402,
    "eda.fake_char_mean": 474.179,
    "eda.real_char_mean": 652.7374,
    "eda.short_50_char_fake_rate": 0.300071,
    "eda.one_time_user_fake_rate": 0.297861,
    "eda.product_day_10_20_fake_rate": 0.320856,
    "eda.product_day_20_50_fake_rate": 0.516484,
    "eda.product_week_rating_50plus_fake_rate": 0.525641,
    "eda.figure_count": 19,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, float):
        return abs(float(actual) - expected) <= tolerance
    return actual == expected


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def collect_values() -> dict[str, Any]:
    preprocess = read_json(PROJECT_DIR / "data" / "processed_rur_shock_context" / "preprocess_summary.json")
    sampling = read_json(PROJECT_DIR / "data" / "sampled_relative_flags_q75_m2" / "sampling_summary.json")
    split = read_json(PROJECT_DIR / "data" / "splits_relative_flags_q75_m2" / "split_summary.json")
    graph = read_json(PROJECT_DIR / "data" / "graph_campaign_quality_q60_top3_b6000_s020" / "graph_summary.json")
    edges = read_json(PROJECT_DIR / "data" / "edges_campaign_quality_q60_top3_b6000_s020" / "edges_summary.json")
    metrics = read_json(
        PROJECT_DIR / "experiments" / "campaign_quality_q60_relation_sage_mlp_equal_seed42" / "metrics.json"
    )
    eda = read_json(PROJECT_DIR / "data" / "eda" / "eda_summary.json")

    edge_counts = graph["graphs"]["graph_rur_custom2.pt"]["edge_type_counts"]
    valid_metrics = metrics["valid_metrics"]
    test_metrics = metrics["test_metrics"]
    rating_fake_rate = {row["rating_int"]: row["fake_rate"] for row in eda["labels_and_ratings"]["rating_fake_rate"]}
    text_by_label = {row["label"]: row for row in eda["text_features"]["text_summary_by_label"]}
    char_bins = {row["char_bin"]: row for row in eda["text_features"]["char_bin_stats"]}
    user_bins = {row["bin"]: row for row in eda["user_product_concentration"]["user_bins"]}
    product_day_bins = {
        row["bin"]: row for row in eda["burst_patterns"]["product_day_review_level_fake_rate_by_bucket_size"]
    }
    product_week_rating_bins = {
        row["bin"]: row for row in eda["burst_patterns"]["product_week_rating_review_level_fake_rate_by_bucket_size"]
    }
    figure_count = len(list((PROJECT_DIR / "data" / "eda" / "figures").glob("*.png")))

    return {
        "preprocess.rows_as_review_nodes": preprocess["rows_as_review_nodes"],
        "preprocess.unique_users": preprocess["unique_users"],
        "preprocess.unique_products": preprocess["unique_products"],
        "preprocess.fake_count": preprocess["label_counts"]["1"],
        "sample.nodes": sampling["selected"]["nodes"],
        "sample.product_week_units": sampling["selected"]["product_week_units"],
        "sample.fake_rate": sampling["selected"]["fake_rate_for_diagnostics_only"],
        "split.train_nodes": split["splits"]["train"]["nodes"],
        "split.valid_nodes": split["splits"]["valid"]["nodes"],
        "split.test_nodes": split["splits"]["test"]["nodes"],
        "graph.n_nodes": graph["n_nodes"],
        "graph.numeric_dim": graph["numeric_dim"],
        "graph.text_dim": graph["text_dim"],
        "graph.total_feature_dim": graph["total_feature_dim"],
        "graph.total_edges": graph["graphs"]["graph_rur_custom2.pt"]["n_edges"],
        "graph.rur_edges": edge_counts["0"],
        "graph.campaign_edges": edge_counts["1"],
        "graph.shock_edges": edge_counts["2"],
        "graph.isolated_nodes": edges["overall_isolated_nodes"],
        "model.best_epoch": metrics["best_epoch"],
        "model.best_threshold": metrics["best_threshold"],
        "valid.pr_auc": valid_metrics["pr_auc"],
        "valid.macro_f1": valid_metrics["macro_f1"],
        "test.pr_auc": test_metrics["pr_auc"],
        "test.roc_auc": test_metrics["roc_auc"],
        "test.macro_f1": test_metrics["macro_f1"],
        "test.precision": test_metrics["precision"],
        "test.recall": test_metrics["recall"],
        "test.accuracy": test_metrics["accuracy"],
        "eda.rating_1_fake_rate": rating_fake_rate[1],
        "eda.rating_5_fake_rate": rating_fake_rate[5],
        "eda.fake_char_mean": text_by_label[-1]["char_mean"],
        "eda.real_char_mean": text_by_label[1]["char_mean"],
        "eda.short_50_char_fake_rate": char_bins["(-0.001, 50.0]"]["fake_rate"],
        "eda.one_time_user_fake_rate": user_bins["(-0.001, 1.0]"]["fake_rate_reviews"],
        "eda.product_day_10_20_fake_rate": product_day_bins["(10.0, 20.0]"]["fake_rate"],
        "eda.product_day_20_50_fake_rate": product_day_bins["(20.0, 50.0]"]["fake_rate"],
        "eda.product_week_rating_50plus_fake_rate": product_week_rating_bins["(50.0, 100000.0]"]["fake_rate"],
        "eda.figure_count": figure_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify retained validation-best campaign-quality RelationSAGE-MLP report numbers."
    )
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if a value differs.")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="Tolerance for floating point comparisons.")
    args = parser.parse_args()

    values = collect_values()
    failed = []

    print("validation-best campaign-quality RelationSAGE-MLP report check")
    print("=" * 80)
    for key in EXPECTED:
        actual = values[key]
        expected = EXPECTED[key]
        ok = close_enough(actual, expected, args.tolerance)
        status = "OK" if ok else "DIFF"
        print(f"{status:4} {key:34} actual={fmt(actual):>14} expected={fmt(expected):>14}")
        if not ok:
            failed.append(key)

    if failed:
        print("=" * 80)
        print("Mismatched keys:", ", ".join(failed))
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Build TF-IDF + SVD text embeddings from sampled YelpZip review text.

Example:
    python BuildTextEmbeddings.py
    python BuildTextEmbeddings.py --svd-dim 128 --max-features 50000

This script intentionally reads only data/sampled and data/splits artifacts.
TF-IDF and SVD are fit on train_mask rows only, then valid/test rows are
transformed without being used for fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLED_DIR = PROJECT_DIR / "data" / "sampled"
DEFAULT_SPLIT_DIR = PROJECT_DIR / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "embeddings"


def log(message: str) -> None:
    print(f"[BuildTextEmbeddings] {message}")


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


def parse_int_or_float(value: str) -> int | float:
    try:
        int_value = int(value)
    except ValueError:
        try:
            float_value = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Expected int or float, got {value!r}.") from exc
        return float_value

    if str(int_value) == value:
        return int_value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected int or float, got {value!r}.") from exc


def load_sampled_text(sampled_dir: Path) -> pd.DataFrame:
    text_path = sampled_dir / "sampled_review_text.csv.gz"
    mapping_path = sampled_dir / "sampled_node_mapping.csv"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing sampled text file: {text_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing sampled node mapping file: {mapping_path}")

    log(f"Loading sampled text: {text_path}")
    text_df = pd.read_csv(text_path)
    mapping = pd.read_csv(mapping_path)

    required_text_columns = ["sampled_node_idx", "original_node_idx", "review_id", "text"]
    missing_text_columns = [col for col in required_text_columns if col not in text_df.columns]
    if missing_text_columns:
        raise ValueError(f"sampled_review_text.csv.gz is missing required columns: {missing_text_columns}")

    required_mapping_columns = ["sampled_node_idx", "original_node_idx", "review_id"]
    missing_mapping_columns = [col for col in required_mapping_columns if col not in mapping.columns]
    if missing_mapping_columns:
        raise ValueError(f"sampled_node_mapping.csv is missing required columns: {missing_mapping_columns}")

    text_df = text_df.sort_values("sampled_node_idx").reset_index(drop=True)
    mapping = mapping.sort_values("sampled_node_idx").reset_index(drop=True)
    validate_sampled_text_order(text_df, mapping)
    text_df["text"] = text_df["text"].fillna("").astype(str)
    return text_df


def validate_sampled_text_order(text_df: pd.DataFrame, mapping: pd.DataFrame) -> None:
    n_nodes = len(text_df)
    if n_nodes == 0:
        raise ValueError("sampled_review_text.csv.gz contains no rows.")
    if len(mapping) != n_nodes:
        raise ValueError(
            f"sampled_node_mapping.csv row count must match sampled text. "
            f"mapping={len(mapping)}, text={n_nodes}."
        )

    sampled_idx = text_df["sampled_node_idx"].to_numpy(dtype=np.int64)
    expected_idx = np.arange(n_nodes, dtype=np.int64)
    if not np.array_equal(sampled_idx, expected_idx):
        bad_positions = np.flatnonzero(sampled_idx != expected_idx)
        first_bad = int(bad_positions[0]) if len(bad_positions) else -1
        raise ValueError(
            "sampled_node_idx must be contiguous from 0 to n-1 after sorting text. "
            f"First mismatch at row {first_bad}: expected {first_bad}, got {sampled_idx[first_bad]}."
        )

    for column in ["sampled_node_idx", "original_node_idx", "review_id"]:
        left = text_df[column].to_numpy(dtype=np.int64)
        right = mapping[column].to_numpy(dtype=np.int64)
        if not np.array_equal(left, right):
            mismatch = np.flatnonzero(left != right)
            first_bad = int(mismatch[0])
            raise ValueError(
                f"sampled_review_text.csv.gz does not align with sampled_node_mapping.csv on {column}. "
                f"First mismatch at row {first_bad}: text={int(left[first_bad])}, mapping={int(right[first_bad])}."
            )


def load_masks(split_dir: Path, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "train": split_dir / "train_mask.npy",
        "valid": split_dir / "valid_mask.npy",
        "test": split_dir / "test_mask.npy",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} mask file: {path}")

    train_mask = np.load(paths["train"])
    valid_mask = np.load(paths["valid"])
    test_mask = np.load(paths["test"])
    validate_masks(train_mask, valid_mask, test_mask, n_nodes)
    return train_mask, valid_mask, test_mask


def validate_masks(train_mask: np.ndarray, valid_mask: np.ndarray, test_mask: np.ndarray, n_nodes: int) -> None:
    for name, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        if mask.dtype != bool:
            raise ValueError(f"{name}_mask.npy must have bool dtype, got {mask.dtype}.")
        if mask.shape != (n_nodes,):
            raise ValueError(f"{name}_mask.npy shape must be ({n_nodes},), got {mask.shape}.")

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
    if int(train_mask.sum()) == 0:
        raise ValueError("train_mask.npy has no training rows; cannot fit TF-IDF/SVD.")


def build_embeddings(
    texts: pd.Series,
    train_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, tuple[int, int], float]:
    train_texts = texts.loc[train_mask].tolist()
    all_texts = texts.tolist()

    log("Fitting TF-IDF vectorizer on train text only.")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    if train_tfidf.shape[1] == 0:
        raise ValueError("TF-IDF produced zero features from train text. Try lowering min_df.")

    log("Transforming all sampled text with the train-fitted TF-IDF vectorizer.")
    all_tfidf = vectorizer.transform(all_texts)
    if all_tfidf.shape[0] != len(texts):
        raise ValueError(f"TF-IDF row count mismatch: tfidf={all_tfidf.shape[0]}, texts={len(texts)}.")

    if args.svd_dim <= 0:
        raise ValueError(f"--svd-dim must be positive, got {args.svd_dim}.")
    if args.svd_dim > train_tfidf.shape[1]:
        raise ValueError(
            f"--svd-dim ({args.svd_dim}) cannot exceed train TF-IDF feature count ({train_tfidf.shape[1]})."
        )

    log("Fitting TruncatedSVD on train TF-IDF only.")
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.random_state)
    svd.fit(train_tfidf)

    log("Transforming all sampled TF-IDF rows with the train-fitted SVD model.")
    embeddings = svd.transform(all_tfidf).astype(np.float32)
    explained_variance = float(np.sum(svd.explained_variance_ratio_))
    return embeddings, vectorizer, svd, all_tfidf.shape, explained_variance


def save_outputs(
    output_dir: Path,
    embeddings: np.ndarray,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    tfidf_shape: tuple[int, int],
    explained_variance: float,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = output_dir / "sampled_text_tfidf_svd.npy"
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    svd_path = output_dir / "svd_model.joblib"
    summary_path = output_dir / "text_embedding_summary.json"

    log(f"Saving embeddings: {embedding_path}")
    np.save(embedding_path, embeddings.astype(np.float32, copy=False))
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(svd, svd_path)

    summary = {
        "n_nodes": int(embeddings.shape[0]),
        "split_counts": {
            "train": int(train_mask.sum()),
            "valid": int(valid_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "tfidf_shape": [int(tfidf_shape[0]), int(tfidf_shape[1])],
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "svd_dim": int(args.svd_dim),
        "explained_variance_ratio_sum": explained_variance,
        "parameters": {
            "max_features": int(args.max_features),
            "ngram_range": [1, 2],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "random_state": int(args.random_state),
        },
        "inputs": {
            "sampled_text": path_for_summary(args.sampled_dir / "sampled_review_text.csv.gz"),
            "sampled_mapping": path_for_summary(args.sampled_dir / "sampled_node_mapping.csv"),
            "train_mask": path_for_summary(args.split_dir / "train_mask.npy"),
            "valid_mask": path_for_summary(args.split_dir / "valid_mask.npy"),
            "test_mask": path_for_summary(args.split_dir / "test_mask.npy"),
        },
        "outputs": {
            "embeddings": path_for_summary(embedding_path),
            "tfidf_vectorizer": path_for_summary(vectorizer_path),
            "svd_model": path_for_summary(svd_path),
            "summary": path_for_summary(summary_path),
        },
        "notes": [
            "Only data/sampled and data/splits artifacts are used.",
            "TF-IDF is fit on train_mask text only.",
            "TruncatedSVD is fit on train TF-IDF only.",
            "Valid/test text is transformed only, preventing split leakage.",
            "Embedding rows are indexed by sampled_node_idx.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    log(f"Saved summary: {summary_path}")


def run_build_text_embeddings(args: argparse.Namespace) -> None:
    text_df = load_sampled_text(args.sampled_dir)
    n_nodes = len(text_df)
    train_mask, valid_mask, test_mask = load_masks(args.split_dir, n_nodes)

    embeddings, vectorizer, svd, tfidf_shape, explained_variance = build_embeddings(
        text_df["text"],
        train_mask,
        args,
    )
    if embeddings.shape != (n_nodes, args.svd_dim):
        raise ValueError(
            f"Embedding shape mismatch. Expected ({n_nodes}, {args.svd_dim}), got {embeddings.shape}."
        )

    save_outputs(
        args.output_dir,
        embeddings,
        vectorizer,
        svd,
        tfidf_shape,
        explained_variance,
        train_mask,
        valid_mask,
        test_mask,
        args,
    )
    log(f"Done. Saved text embeddings with shape {embeddings.shape}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-fitted TF-IDF + SVD embeddings from sampled YelpZip review text."
    )
    parser.add_argument("--sampled-dir", type=Path, default=DEFAULT_SAMPLED_DIR, help="Sampled data directory")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR, help="Split mask directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Text embedding output directory")
    parser.add_argument("--max-features", type=int, default=50000, help="Maximum TF-IDF vocabulary size")
    parser.add_argument("--svd-dim", type=int, default=128, help="SVD embedding dimension")
    parser.add_argument("--min-df", type=parse_int_or_float, default=2, help="TF-IDF min_df")
    parser.add_argument("--max-df", type=float, default=0.95, help="TF-IDF max_df")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for TruncatedSVD")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_build_text_embeddings(parse_args())
    except Exception as exc:
        print(f"[BuildTextEmbeddings][ERROR] {exc}", file=sys.stderr)
        raise

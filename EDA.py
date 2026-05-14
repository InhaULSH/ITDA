"""
YelpZip EDA script for the ITDA project.

이 스크립트는 data/origin/yelpzip.csv를 읽어 모델링 전 탐색적 데이터 분석(EDA)을 수행한다.
콘솔에는 핵심 표와 통계량을 출력하고, 그래프는 data/eda/figures 폴더에 PNG로 저장한다.
"""

# 기본 라이브러리와 시각화 환경을 설정한다.
# matplotlib은 파일 저장을 기본으로 사용하되, --show 옵션을 주면 화면에도 그래프를 띄운다.
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.util as pdu
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


# Windows 콘솔의 기본 인코딩이 UTF-8이 아닐 때도 한국어 출력이 중단되지 않도록 설정한다.
# PyCharm, PowerShell, Codex 터미널 어디에서 실행해도 같은 로그를 볼 수 있게 한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# 분석 대상 파일과 결과 저장 위치를 정의한다.
# 프로젝트 루트에서 실행하는 상황을 기본으로 하며, CSV 경로는 옵션으로 바꿀 수 있다.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "data" / "origin" / "yelpzip.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "eda"
META_COLUMNS = ["Unnamed: 0", "user_id", "prod_id", "rating", "label", "date", "tag"]


# 콘솔 표 출력과 그래프 저장에 공통으로 쓰는 유틸리티 함수들이다.
# 큰 데이터셋에서도 보기 편하도록 표의 행 수와 소수점 자릿수를 정리한다.
def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table(title: str, table: pd.DataFrame | pd.Series, max_rows: int = 20) -> None:
    print_section(title)
    if isinstance(table, pd.Series):
        table = table.to_frame()
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        30,
        "display.width",
        180,
        "display.float_format",
        "{:,.6f}".format,
    ):
        print(table.head(max_rows).to_string())


def save_fig(fig: plt.Figure, figures_dir: Path, name: str, show: bool = False) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[FIGURE] {path}")
    if show:
        plt.show()
    plt.close(fig)


def pct(x: float) -> float | None:
    if pd.isna(x):
        return None
    return round(float(x) * 100, 4)


def rnd(x: float, digits: int = 6) -> float | None:
    if pd.isna(x):
        return None
    return round(float(x), digits)


def quantile_dict(series: pd.Series) -> dict[str, float | None]:
    quantiles = series.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {str(q): rnd(v, 4) for q, v in quantiles.items()}


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else np.nan


# 범주형 변수와 라벨 사이의 관련성을 Cramer's V로 요약한다.
# 표본 수가 매우 커 p-value만으로는 과장될 수 있으므로 효과크기를 함께 본다.
def cramers_v(table: pd.DataFrame) -> dict[str, float | str | int]:
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n_obs = table.to_numpy().sum()
    rows, cols = table.shape
    denom = min(cols - 1, rows - 1)
    value = math.sqrt((chi2 / n_obs) / denom) if denom > 0 else 0.0
    return {
        "chi2": rnd(chi2, 3),
        "p": "<1e-300" if p_value == 0 else f"{p_value:.3e}",
        "dof": int(dof),
        "cramers_v": rnd(value, 6),
    }


# 연속형 변수의 두 집단 차이를 Cohen's d와 Cliff's delta로 요약한다.
# 큰 표본에서는 검정 유의성보다 실제 차이의 크기가 더 중요하다.
def cohen_d(group_a: pd.Series, group_b: pd.Series) -> float:
    a = group_a.to_numpy()
    b = group_b.to_numpy()
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else np.nan


def cliffs_delta(group_a: pd.Series, group_b: pd.Series, max_n: int = 60000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    a = group_a.to_numpy()
    b = group_b.to_numpy()
    if len(a) > max_n:
        a = rng.choice(a, size=max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, size=max_n, replace=False)

    b_sorted = np.sort(b)
    greater = np.searchsorted(b_sorted, a, side="left").sum()
    less_equal = np.searchsorted(b_sorted, a, side="right").sum()
    less = len(a) * len(b) - less_equal
    return float((greater - less) / (len(a) * len(b)))


# 이진 플래그가 참일 때 가짜 리뷰 odds가 얼마나 변하는지 계산한다.
# Haldane 보정을 넣어 특정 셀이 0이어도 안정적으로 계산되게 한다.
def odds_ratio_for_flag(df: pd.DataFrame, flag: str) -> dict[str, float | int | str | None]:
    table = pd.crosstab(df[flag], df["fake"])
    fake_true = table.loc[1, 1] if 1 in table.index and 1 in table.columns else 0
    real_true = table.loc[1, 0] if 1 in table.index and 0 in table.columns else 0
    fake_false = table.loc[0, 1] if 0 in table.index and 1 in table.columns else 0
    real_false = table.loc[0, 0] if 0 in table.index and 0 in table.columns else 0
    odds_ratio = ((fake_true + 0.5) / (real_true + 0.5)) / ((fake_false + 0.5) / (real_false + 0.5))
    return {
        "flag": flag,
        "odds_ratio": rnd(odds_ratio, 6),
        "flag_true_reviews": int((df[flag] == 1).sum()),
        "flag_true_fake_rate_pct": pct(df.loc[df[flag] == 1, "fake"].mean()),
        "flag_false_fake_rate_pct": pct(df.loc[df[flag] == 0, "fake"].mean()),
    }


# 상품별 평점 분포가 얼마나 다양하게 퍼졌는지 엔트로피로 계산한다.
# 평점이 한쪽에만 몰리면 낮고, 여러 평점에 분산되면 높다.
def rating_entropy(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy()
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


# 원본 CSV의 메타데이터 컬럼을 먼저 읽고 분석용 파생 변수를 만든다.
# label=-1은 fake=1, label=1은 fake=0으로 변환하여 이후 분석의 기준 라벨로 사용한다.
def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=META_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fake"] = (df["label"] == -1).astype(int)
    df["rating_int"] = df["rating"].astype("Int64")
    df["extreme"] = df["rating"].isin([1.0, 5.0]).astype(int)
    df["positive"] = df["rating"].isin([4.0, 5.0]).astype(int)
    df["negative"] = df["rating"].isin([1.0, 2.0]).astype(int)
    df["neutral"] = df["rating"].eq(3.0).astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["dow"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    df["rating_bucket"] = np.select(
        [df["rating"].le(2), df["rating"].eq(3), df["rating"].ge(4)],
        ["low_1_2", "mid_3", "high_4_5"],
        default="unknown",
    )
    return df


# 텍스트 본문은 400MB 이상이므로 청크 단위로 길이, 단어 수, 문장부호, 해시만 추출한다.
# 원문 텍스트를 전부 메모리에 오래 들고 있지 않아도 텍스트 EDA와 중복 점검을 수행할 수 있다.
def extract_text_features(csv_path: Path, chunksize: int = 50000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=chunksize):
        text = chunk["text"].fillna("")
        char_len = text.str.len().to_numpy(np.int32)
        word_len = text.str.split().str.len().fillna(0).to_numpy(np.int32)
        upper_count = text.str.count(r"[A-Z]").to_numpy(np.int32)

        chunks.append(
            pd.DataFrame(
                {
                    "text_missing": chunk["text"].isna().astype(int).to_numpy(np.int8),
                    "char_len": char_len,
                    "word_len": word_len,
                    "upper_ratio": np.divide(upper_count, np.maximum(char_len, 1), dtype=float),
                    "exclam": text.str.count("!").to_numpy(np.int16),
                    "question": text.str.count(r"\?").to_numpy(np.int16),
                    "dollar": text.str.count(r"\$").to_numpy(np.int16),
                    "text_hash": pdu.hash_pandas_object(text, index=False).to_numpy(np.uint64),
                }
            )
        )
    return pd.concat(chunks, ignore_index=True)


# 데이터셋의 규모, 결측, 라벨-태그 일치 여부를 먼저 확인한다.
# 이 단계는 이후 통계 분석이 신뢰 가능한 입력 위에서 수행되는지 점검하는 품질관리 역할을 한다.
def analyze_basic_quality(df: pd.DataFrame, csv_path: Path, output_dir: Path) -> dict[str, object]:
    n_rows = len(df)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    basic = {
        "rows": int(n_rows),
        "columns": header,
        "unique_users": int(df["user_id"].nunique()),
        "unique_products": int(df["prod_id"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "date_parse_na": int(df["date"].isna().sum()),
        "index_is_0_to_n_minus_1": bool((df["Unnamed: 0"].to_numpy() == np.arange(n_rows)).all()),
    }
    missing = df[META_COLUMNS].isna().sum().to_frame("missing_count")
    label_tag = pd.crosstab(df["label"], df["tag"], dropna=False)

    print_section("1. 데이터 기본 구조와 품질 점검")
    print(json.dumps(basic, ensure_ascii=False, indent=2))
    print_table("메타데이터 결측치", missing)
    print_table("label-tag 교차표", label_tag)

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    missing.to_csv(output_dir / "tables" / "basic_missing_counts.csv", encoding="utf-8-sig")
    label_tag.to_csv(output_dir / "tables" / "label_tag_crosstab.csv", encoding="utf-8-sig")
    return basic


# 라벨 불균형과 평점 분포를 요약한다.
# 파이 차트와 막대/선 그래프를 함께 사용해 전체 비율과 평점별 위험 차이를 동시에 보여준다.
def analyze_labels_and_ratings(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    label_counts = df["label"].value_counts().sort_index()
    tag_counts = df["tag"].value_counts(dropna=False).sort_index()
    rating_counts = df["rating_int"].value_counts().sort_index()
    rating_by_label = pd.crosstab(df["rating_int"], df["label"])
    rating_by_label_pct = rating_by_label.div(rating_by_label.sum(axis=0), axis=1) * 100
    rating_fake_rate = (
        df.groupby("rating_int", observed=False)["fake"]
        .agg(reviews="size", fake="sum", fake_rate="mean")
        .reset_index()
    )
    rating_summary = df.groupby("label")["rating"].agg(["count", "mean", "std", "median"])
    extreme_summary = df.groupby("label")[["extreme", "positive", "negative", "neutral"]].mean() * 100

    print_table("2-1. 라벨 분포", label_counts)
    print_table("2-2. tag 분포", tag_counts)
    print_table("2-3. 평점 분포", rating_counts)
    print_table("2-4. 라벨별 평점 요약", rating_summary)
    print_table("2-5. 평점별 가짜 리뷰 비율", rating_fake_rate)
    print_table("2-6. 라벨별 극단/긍정/부정/중립 평점 비율(%)", extreme_summary)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [int((df["fake"] == 1).sum()), int((df["fake"] == 0).sum())],
        labels=["fake", "real"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#c44e52", "#4c72b0"],
    )
    ax.set_title("Label Imbalance")
    save_fig(fig, figures_dir, "01_label_imbalance_pie", show)

    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts.plot(kind="bar", ax=ax, color="#55a868")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("rating")
    ax.set_ylabel("review count")
    save_fig(fig, figures_dir, "02_rating_distribution_bar", show)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=rating_fake_rate, x="rating_int", y="fake_rate", ax=ax, color="#c44e52")
    ax.axhline(df["fake"].mean(), color="black", linestyle="--", linewidth=1, label="overall fake rate")
    ax.set_title("Fake Rate by Rating")
    ax.set_xlabel("rating")
    ax.set_ylabel("fake rate")
    ax.legend()
    save_fig(fig, figures_dir, "03_fake_rate_by_rating", show)

    return {
        "label_counts": label_counts.to_dict(),
        "tag_counts": tag_counts.to_dict(),
        "rating_counts": rating_counts.to_dict(),
        "rating_by_label_counts": rating_by_label.to_dict(),
        "rating_by_label_pct": rating_by_label_pct.round(4).to_dict(),
        "rating_fake_rate": rating_fake_rate.round(6).to_dict("records"),
    }


# 연도, 월, 요일 단위의 리뷰량과 가짜 비율을 살펴본다.
# 시간 순서가 있는 연도 분석은 선 그래프로, 범주형 요일 분석은 막대 그래프로 시각화한다.
def analyze_time_patterns(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    yearly = (
        df.groupby("year")["fake"]
        .agg(reviews="size", fake="sum", fake_rate="mean")
        .reset_index()
    )
    monthly = (
        df.groupby("month")["fake"]
        .agg(reviews="size", fake="sum", fake_rate="mean")
        .reset_index()
    )
    monthly["month_start"] = pd.PeriodIndex(monthly["month"], freq="M").to_timestamp()
    monthly_p90 = monthly["reviews"].quantile(0.90)
    top_months = monthly.sort_values("reviews", ascending=False).head(12)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        df.groupby("dow")["fake"]
        .agg(reviews="size", fake="sum", fake_rate="mean")
        .reindex(dow_order)
        .reset_index()
    )

    print_table("3-1. 연도별 리뷰 수와 가짜 비율", yearly)
    print_table("3-2. 월별 리뷰 수 요약", monthly[["month", "reviews", "fake", "fake_rate"]])
    print_table("3-3. 요일별 리뷰 수와 가짜 비율", dow)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(yearly["year"], yearly["reviews"], color="#4c72b0", alpha=0.75, label="reviews")
    ax1.set_ylabel("review count")
    review_ymax = math.ceil(yearly["reviews"].max() / 50000) * 50000
    fake_ymax = math.ceil(yearly["fake_rate"].max() / 0.05) * 0.05
    tick_count = 6
    ax1.set_ylim(0, review_ymax)
    ax1.set_yticks(np.linspace(0, review_ymax, tick_count))
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["fake_rate"], color="#c44e52", marker="o", label="fake rate")
    ax2.set_ylabel("fake rate")
    ax2.set_ylim(0, fake_ymax)
    ax2.set_yticks(np.linspace(0, fake_ymax, tick_count))
    ax2.grid(False)
    ax1.set_title("Yearly Review Volume and Fake Rate")
    save_fig(fig, figures_dir, "04_yearly_volume_fake_rate", show)

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = np.where(monthly["reviews"] >= monthly_p90, "#c44e52", "#4c72b0")
    ax.bar(monthly["month_start"], monthly["reviews"], width=24, color=colors, alpha=0.85)
    ax.axhline(monthly_p90, color="black", linestyle="--", linewidth=1, label="90th percentile")
    year_ticks = pd.date_range(monthly["month_start"].min(), monthly["month_start"].max(), freq="YS")
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(x.year) for x in year_ticks], rotation=0)
    ax.set_title("Monthly Review Volume Across Entire Period")
    ax.set_xlabel("month")
    ax.set_ylabel("review count")
    ax.legend()
    save_fig(fig, figures_dir, "05_monthly_review_volume_histogram", show)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=dow, x="dow", y="fake_rate", ax=ax, color="#dd8452")
    ax.set_title("Fake Rate by Day of Week")
    ax.set_xlabel("day of week")
    ax.set_ylabel("fake rate")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, figures_dir, "06_day_of_week_fake_rate", show)

    return {
        "yearly": yearly.round(6).to_dict("records"),
        "monthly": monthly[["month", "reviews", "fake", "fake_rate"]].round(6).to_dict("records"),
        "high_volume_month_threshold_p90": rnd(monthly_p90, 4),
        "top_months": top_months[["month", "reviews", "fake", "fake_rate"]].round(6).to_dict("records"),
        "day_of_week": dow.round(6).to_dict("records"),
    }


# 작성일 기준 64:16:20 temporal split을 계산한다.
# 모델링 전에 시간 분할의 라벨 비율과 기간이 안정적인지 점검한다.
def analyze_temporal_split(df: pd.DataFrame, figures_dir: Path, show: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    n_rows = len(df)
    df_sorted = df.sort_values(["date", "Unnamed: 0"]).reset_index(drop=True)
    cut64_date = df_sorted.loc[int(n_rows * 0.64) - 1, "date"]
    cut80_date = df_sorted.loc[int(n_rows * 0.80) - 1, "date"]

    df["split"] = np.select(
        [df["date"] <= cut64_date, (df["date"] > cut64_date) & (df["date"] <= cut80_date), df["date"] > cut80_date],
        ["train", "valid", "test"],
        default="unknown",
    )
    split_summary = (
        df.groupby("split")
        .agg(
            reviews=("fake", "size"),
            fake=("fake", "sum"),
            fake_rate=("fake", "mean"),
            rating_mean=("rating", "mean"),
            start=("date", "min"),
            end=("date", "max"),
        )
        .reindex(["train", "valid", "test"])
        .reset_index()
    )
    split_summary["share"] = split_summary["reviews"] / n_rows

    print_section("4. 작성일 기준 temporal split")
    print(f"64% cutoff date: {cut64_date.date()}")
    print(f"80% cutoff date: {cut80_date.date()}")
    print_table("split별 리뷰 수와 가짜 비율", split_summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=split_summary, x="split", y="fake_rate", ax=ax, color="#c44e52")
    ax.set_title("Fake Rate by Temporal Split")
    ax.set_xlabel("split")
    ax.set_ylabel("fake rate")
    save_fig(fig, figures_dir, "07_temporal_split_fake_rate", show)

    return df, {
        "cut64_date": str(cut64_date.date()),
        "cut80_date": str(cut80_date.date()),
        "split_summary": split_summary.assign(start=split_summary["start"].astype(str), end=split_summary["end"].astype(str))
        .round(6)
        .to_dict("records"),
    }


# 리뷰 본문 길이, 단어 수, 문장부호 사용량 같은 텍스트 표면 피처를 분석한다.
# 길이는 오른쪽 꼬리가 길기 때문에 히스토그램에는 로그 축을 함께 사용한다.
def analyze_text_features(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    text_summary = (
        df.groupby("label")
        .agg(
            reviews=("text_hash", "size"),
            text_missing=("text_missing", "sum"),
            char_mean=("char_len", "mean"),
            char_median=("char_len", "median"),
            char_p90=("char_len", lambda x: np.quantile(x, 0.90)),
            char_p99=("char_len", lambda x: np.quantile(x, 0.99)),
            word_mean=("word_len", "mean"),
            word_median=("word_len", "median"),
            word_p90=("word_len", lambda x: np.quantile(x, 0.90)),
            word_p99=("word_len", lambda x: np.quantile(x, 0.99)),
        )
        .reset_index()
    )

    df["char_bin"] = pd.cut(df["char_len"], bins=[0, 50, 100, 200, 500, 1000, 2000, 5000, 100000], include_lowest=True)
    df["word_bin"] = pd.cut(df["word_len"], bins=[0, 10, 25, 50, 100, 200, 400, 800, 100000], include_lowest=True)
    char_bin = (
        df.groupby("char_bin", observed=False)
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    char_bin["share"] = char_bin["reviews"] / len(df)
    char_bin["char_bin"] = char_bin["char_bin"].astype(str)
    word_bin = (
        df.groupby("word_bin", observed=False)
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    word_bin["share"] = word_bin["reviews"] / len(df)
    word_bin["word_bin"] = word_bin["word_bin"].astype(str)

    print_table("5-1. 라벨별 텍스트 길이 요약", text_summary)
    print_table("5-2. 글자 수 구간별 가짜 비율", char_bin)
    print_table("5-3. 단어 수 구간별 가짜 비율", word_bin)
    print_section("5-4. 텍스트 표면 피처 분위수")
    print(
        json.dumps(
            {
                "char_len": quantile_dict(df["char_len"]),
                "word_len": quantile_dict(df["word_len"]),
                "upper_ratio": quantile_dict(df["upper_ratio"]),
                "exclamation_count": quantile_dict(df["exclam"]),
                "question_count": quantile_dict(df["question"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="char_len", hue="tag", bins=80, stat="density", common_norm=False, ax=ax)
    ax.set_xscale("log")
    ax.set_title("Review Character Length Distribution by Label")
    ax.set_xlabel("character length (log scale)")
    save_fig(fig, figures_dir, "08_text_length_distribution_log", show)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=char_bin, x="char_bin", y="fake_rate", ax=ax, color="#c44e52")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Fake Rate by Character Length Bin")
    ax.set_xlabel("character length bin")
    ax.set_ylabel("fake rate")
    save_fig(fig, figures_dir, "09_fake_rate_by_text_length_bin", show)

    return {
        "text_summary_by_label": text_summary.round(4).to_dict("records"),
        "char_bin_stats": char_bin.round(6).to_dict("records"),
        "word_bin_stats": word_bin.round(6).to_dict("records"),
    }


# 사용자와 상품의 리뷰 수 분포, 집중도, 가짜 리뷰 분포를 분석한다.
# 롱테일 구조가 강하므로 분위수와 로그 스케일 히스토그램을 함께 사용한다.
def analyze_user_product_concentration(df: pd.DataFrame, figures_dir: Path, show: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    product = (
        df.groupby("prod_id")
        .agg(
            reviews=("fake", "size"),
            fake=("fake", "sum"),
            fake_rate=("fake", "mean"),
            avg_rating=("rating", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    user = (
        df.groupby("user_id")
        .agg(
            reviews=("fake", "size"),
            fake=("fake", "sum"),
            fake_rate=("fake", "mean"),
            unique_products=("prod_id", "nunique"),
            avg_rating=("rating", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    product_sorted = product.sort_values("reviews", ascending=False)
    user_sorted = user.sort_values("reviews", ascending=False)
    top_1pct_products = max(1, int(np.ceil(len(product) * 0.01)))
    top_10pct_products = max(1, int(np.ceil(len(product) * 0.10)))
    top_1pct_users = max(1, int(np.ceil(len(user) * 0.01)))
    top_10pct_users = max(1, int(np.ceil(len(user) * 0.10)))

    product_bins = pd.cut(product["reviews"], bins=[0, 5, 10, 30, 100, 300, 1000, 100000], include_lowest=True)
    product_bin_stats = (
        product.assign(bin=product_bins)
        .groupby("bin", observed=False)
        .agg(products=("prod_id", "size"), total_reviews=("reviews", "sum"), fake_reviews=("fake", "sum"))
        .reset_index()
    )
    product_bin_stats["product_share"] = product_bin_stats["products"] / len(product)
    product_bin_stats["fake_rate_reviews"] = product_bin_stats["fake_reviews"] / product_bin_stats["total_reviews"]
    product_bin_stats["bin"] = product_bin_stats["bin"].astype(str)

    user_bins = pd.cut(user["reviews"], bins=[0, 1, 2, 5, 10, 20, 50, 100000], include_lowest=True)
    user_bin_stats = (
        user.assign(bin=user_bins)
        .groupby("bin", observed=False)
        .agg(users=("user_id", "size"), total_reviews=("reviews", "sum"), fake_reviews=("fake", "sum"))
        .reset_index()
    )
    user_bin_stats["user_share"] = user_bin_stats["users"] / len(user)
    user_bin_stats["fake_rate_reviews"] = user_bin_stats["fake_reviews"] / user_bin_stats["total_reviews"]
    user_bin_stats["bin"] = user_bin_stats["bin"].astype(str)

    product_stats = {
        "product_review_count_quantiles": quantile_dict(product["reviews"]),
        "products_with_any_fake": int((product["fake"] > 0).sum()),
        "products_with_any_fake_share_pct": pct((product["fake"] > 0).mean()),
        "products_all_fake": int((product["fake"] == product["reviews"]).sum()),
        "top_1pct_products_review_share_pct": pct(product_sorted.head(top_1pct_products)["reviews"].sum() / len(df)),
        "top_10pct_products_review_share_pct": pct(product_sorted.head(top_10pct_products)["reviews"].sum() / len(df)),
        "top_1pct_products_fake_review_share_pct": pct(product_sorted.head(top_1pct_products)["fake"].sum() / df["fake"].sum()),
    }
    user_stats = {
        "user_review_count_quantiles": quantile_dict(user["reviews"]),
        "users_with_any_fake": int((user["fake"] > 0).sum()),
        "users_with_any_fake_share_pct": pct((user["fake"] > 0).mean()),
        "reviews_by_fake_exposed_users_share_pct": pct(df["user_id"].isin(user.loc[user["fake"] > 0, "user_id"]).mean()),
        "top_1pct_users_review_share_pct": pct(user_sorted.head(top_1pct_users)["reviews"].sum() / len(df)),
        "top_10pct_users_review_share_pct": pct(user_sorted.head(top_10pct_users)["reviews"].sum() / len(df)),
    }

    print_section("6-1. 상품/사용자 집중도 요약")
    print(json.dumps({"product_stats": product_stats, "user_stats": user_stats}, ensure_ascii=False, indent=2))
    print_table("6-2. 리뷰 수 구간별 상품 가짜 비율", product_bin_stats)
    print_table("6-3. 리뷰 수 구간별 사용자 가짜 비율", user_bin_stats)
    print_table("6-4. 가짜 리뷰 수 상위 상품", product.sort_values(["fake", "reviews"], ascending=False).head(10))
    print_table("6-5. 가짜 리뷰 수 상위 사용자", user.sort_values(["fake", "reviews"], ascending=False).head(10))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(product["reviews"], bins=60, ax=axes[0], color="#4c72b0")
    axes[0].set_xscale("log")
    axes[0].set_title("Product Review Count Distribution")
    axes[0].set_xlabel("reviews per product (log scale)")
    sns.histplot(user["reviews"], bins=60, ax=axes[1], color="#55a868")
    axes[1].set_xscale("log")
    axes[1].set_title("User Review Count Distribution")
    axes[1].set_xlabel("reviews per user (log scale)")
    save_fig(fig, figures_dir, "10_user_product_review_count_lognormal", show)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=product_bin_stats, x="bin", y="fake_rate_reviews", ax=axes[0], color="#c44e52")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Fake Rate by Product Review Count Bin")
    axes[0].set_xlabel("product review count bin")
    axes[0].set_ylabel("fake rate")
    sns.barplot(data=user_bin_stats, x="bin", y="fake_rate_reviews", ax=axes[1], color="#dd8452")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Fake Rate by User Review Count Bin")
    axes[1].set_xlabel("user review count bin")
    axes[1].set_ylabel("fake rate")
    save_fig(fig, figures_dir, "11_fake_rate_by_user_product_bins", show)

    return product, user, {
        "product_stats": product_stats,
        "user_stats": user_stats,
        "product_bins": product_bin_stats.round(6).to_dict("records"),
        "user_bins": user_bin_stats.round(6).to_dict("records"),
    }


# 동일 사용자의 시간상 가까운 리뷰 존재 여부를 계산한다.
# 이 relation은 반복 작성자가 무조건 위험하다는 뜻이 아니라 사용자 활동 맥락을 주는 신호로 해석한다.
def analyze_user_recency(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    ordered = df.sort_values(["user_id", "date", "Unnamed: 0"]).copy()
    ordered["prev_date_same_user"] = ordered.groupby("user_id")["date"].shift(1)
    ordered["next_date_same_user"] = ordered.groupby("user_id")["date"].shift(-1)
    ordered["gap_prev_days"] = (ordered["date"] - ordered["prev_date_same_user"]).dt.days
    ordered["gap_next_days"] = (ordered["next_date_same_user"] - ordered["date"]).dt.days
    ordered["has_user_neighbor_7d"] = (
        ordered["gap_prev_days"].between(0, 7) | ordered["gap_next_days"].between(0, 7)
    ).astype(int)
    ordered["has_user_neighbor_30d"] = (
        ordered["gap_prev_days"].between(0, 30) | ordered["gap_next_days"].between(0, 30)
    ).astype(int)

    result: dict[str, object] = {}
    for col in ["has_user_neighbor_7d", "has_user_neighbor_30d"]:
        table = ordered.groupby(col)["fake"].agg(reviews="size", fake="sum", fake_rate="mean").reset_index()
        result[col] = table.round(6).to_dict("records")
        print_table(f"7. {col}별 가짜 리뷰 비율", table)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=table, x=col, y="fake_rate", ax=ax, color="#c44e52")
        ax.set_title(f"Fake Rate by {col}")
        ax.set_ylabel("fake rate")
        save_fig(fig, figures_dir, f"12_{col}_fake_rate", show)

    return result


# 상품-일자 및 상품-주간-평점방향 버스트를 분석한다.
# 초안의 product_time_rating_burst relation이 실제로 의미 있는지 확인하는 핵심 EDA다.
def analyze_burst_patterns(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    product_day = (
        df.groupby(["prod_id", "date"])
        .agg(bucket_reviews=("fake", "size"), bucket_fake=("fake", "sum"))
        .reset_index()
    )
    product_day_key = product_day.set_index(["prod_id", "date"])["bucket_reviews"]
    df["prod_day_reviews"] = pd.MultiIndex.from_frame(df[["prod_id", "date"]]).map(product_day_key)

    product_week_rating = (
        df.groupby(["prod_id", "week", "rating_bucket"])
        .agg(bucket_reviews=("fake", "size"), bucket_fake=("fake", "sum"))
        .reset_index()
    )
    product_week_rating["bucket_fake_rate"] = product_week_rating["bucket_fake"] / product_week_rating["bucket_reviews"]
    pwr_key = product_week_rating.set_index(["prod_id", "week", "rating_bucket"])["bucket_reviews"]
    df["prod_week_rating_reviews"] = pd.MultiIndex.from_frame(df[["prod_id", "week", "rating_bucket"]]).map(pwr_key)

    day_bin = summarize_review_bucket(df, "prod_day_reviews", [0, 1, 2, 5, 10, 20, 50, 100000])
    week_rating_bin = summarize_review_bucket(df, "prod_week_rating_reviews", [0, 1, 2, 5, 10, 20, 50, 100000])
    top_bursts = product_week_rating.sort_values(["bucket_reviews", "bucket_fake"], ascending=False).head(12)
    fake_heavy = (
        product_week_rating[product_week_rating["bucket_reviews"] >= 10]
        .sort_values(["bucket_fake_rate", "bucket_reviews"], ascending=False)
        .head(12)
    )

    print_section("8-1. 상품-시간 버스트 분위수")
    print(
        json.dumps(
            {
                "product_day_bucket_count_quantiles": quantile_dict(product_day["bucket_reviews"]),
                "product_week_rating_bucket_count_quantiles": quantile_dict(product_week_rating["bucket_reviews"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print_table("8-2. 상품-일자 버킷 크기별 가짜 비율", day_bin)
    print_table("8-3. 상품-주간-평점방향 버킷 크기별 가짜 비율", week_rating_bin)
    print_table("8-4. 리뷰 수 기준 상위 상품-주간-평점방향 버스트", top_bursts)
    print_table("8-5. 최소 10건 이상 중 가짜 비율이 높은 버스트", fake_heavy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=day_bin, x="bin", y="fake_rate", ax=axes[0], color="#c44e52")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Fake Rate by Product-Day Bucket Size")
    axes[0].set_xlabel("product-day review count bin")
    axes[0].set_ylabel("fake rate")
    sns.barplot(data=week_rating_bin, x="bin", y="fake_rate", ax=axes[1], color="#8172b2")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Fake Rate by Product-Week-Rating Bucket Size")
    axes[1].set_xlabel("product-week-rating review count bin")
    axes[1].set_ylabel("fake rate")
    save_fig(fig, figures_dir, "13_burst_bucket_fake_rate", show)

    return {
        "product_day_bucket_count_quantiles": quantile_dict(product_day["bucket_reviews"]),
        "product_week_rating_bucket_count_quantiles": quantile_dict(product_week_rating["bucket_reviews"]),
        "product_day_review_level_fake_rate_by_bucket_size": day_bin.round(6).to_dict("records"),
        "product_week_rating_review_level_fake_rate_by_bucket_size": week_rating_bin.round(6).to_dict("records"),
        "top_product_week_rating_bursts": top_bursts.astype({"week": str}).round(6).to_dict("records"),
        "top_product_week_rating_fake_heavy_min10": fake_heavy.astype({"week": str}).round(6).to_dict("records"),
    }


# 리뷰 수준에서 버킷 크기별 가짜 비율을 계산한다.
# 각 리뷰가 속한 버스트의 크기가 커질수록 위험이 커지는지 확인하는 데 사용한다.
def summarize_review_bucket(df: pd.DataFrame, column: str, bins: Iterable[int]) -> pd.DataFrame:
    temp = df.copy()
    temp["bin"] = pd.cut(temp[column], bins=bins, right=True, include_lowest=True)
    out = temp.groupby("bin", observed=False).agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean")).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


# 완전 중복, 정확히 같은 텍스트, 같은 사용자-상품 반복 리뷰 여부를 점검한다.
# 중복 텍스트는 오류일 수도 있지만 템플릿형 어뷰징의 증거일 수도 있으므로 삭제하지 않고 요약한다.
def analyze_duplicates(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    hash_counts = df["text_hash"].value_counts()
    df["text_hash_count"] = df["text_hash"].map(hash_counts)

    structured_fields = df[["user_id", "prod_id", "rating", "label", "date", "text_hash"]].copy()
    structured_duplicate_rows = int(structured_fields.duplicated().sum())

    user_product = (
        df.groupby(["user_id", "prod_id"])
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), first=("date", "min"), last=("date", "max"))
        .reset_index()
    )
    user_product["span_days"] = (user_product["last"] - user_product["first"]).dt.days
    repeated_pairs = user_product[user_product["reviews"] > 1]
    df_repeat = df.merge(
        user_product[["user_id", "prod_id", "reviews"]].rename(columns={"reviews": "user_product_review_count"}),
        on=["user_id", "prod_id"],
        how="left",
    )
    repeat_review_stats = (
        df_repeat.assign(repeat_pair=df_repeat["user_product_review_count"] > 1)
        .groupby("repeat_pair")
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean"))
        .reset_index()
    )

    duplicate_groups = []
    for text_hash, count in hash_counts[hash_counts > 1].head(10).items():
        subset = df[df["text_hash"] == text_hash]
        duplicate_groups.append(
            {
                "count": int(count),
                "fake_rate": rnd(subset["fake"].mean(), 6),
                "n_users": int(subset["user_id"].nunique()),
                "n_products": int(subset["prod_id"].nunique()),
                "ratings": sorted(map(float, subset["rating"].unique().tolist())),
                "date_min": str(subset["date"].min().date()),
                "date_max": str(subset["date"].max().date()),
                "char_len": int(subset["char_len"].iloc[0]),
            }
        )

    duplicate_summary = {
        "text_missing": int(df["text_missing"].sum()),
        "unique_text_hashes": int(df["text_hash"].nunique()),
        "exact_duplicate_text_reviews": int((df["text_hash_count"] > 1).sum()),
        "exact_duplicate_text_review_share_pct": pct((df["text_hash_count"] > 1).mean()),
        "top_exact_duplicate_text_frequencies": [int(x) for x in hash_counts.head(10).tolist()],
        "structured_duplicate_rows_count": structured_duplicate_rows,
        "structured_duplicate_rows_share_pct": pct(structured_duplicate_rows / len(df)),
        "repeated_user_product_pairs": int(len(repeated_pairs)),
        "repeated_user_product_pair_share_pct": pct(len(repeated_pairs) / len(user_product)),
        "reviews_in_repeated_user_product_pairs": int(df_repeat["user_product_review_count"].gt(1).sum()),
    }

    print_section("9-1. 중복 및 반복 리뷰 점검")
    print(json.dumps(duplicate_summary, ensure_ascii=False, indent=2))
    print_table("9-2. 같은 사용자-상품 반복 리뷰 여부별 가짜 비율", repeat_review_stats)
    print_table("9-3. 상위 정확 중복 텍스트 그룹 구조(원문 미출력)", pd.DataFrame(duplicate_groups))

    fig, ax = plt.subplots(figsize=(8, 5))
    top_freq = pd.Series(hash_counts.head(20).to_numpy(), name="frequency")
    sns.barplot(x=np.arange(1, len(top_freq) + 1), y=top_freq, ax=ax, color="#8172b2")
    ax.set_title("Top Exact Duplicate Text Frequencies")
    ax.set_xlabel("duplicate text rank")
    ax.set_ylabel("frequency")
    save_fig(fig, figures_dir, "14_top_duplicate_text_frequencies", show)

    return {
        "duplicate_summary": duplicate_summary,
        "repeat_review_stats": repeat_review_stats.round(6).to_dict("records"),
        "top_duplicate_text_group_stats_no_text": duplicate_groups,
    }


# 일반적인 EDA에서 자주 수행하는 통계검정과 단일 변수 선별 지표를 계산한다.
# p-value는 참고로만 보고, Cramer's V, Cohen's d, AUC, odds ratio를 중심으로 해석한다.
def analyze_statistical_tests(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    fake_char = df.loc[df["fake"] == 1, "char_len"]
    real_char = df.loc[df["fake"] == 0, "char_len"]
    fake_word = df.loc[df["fake"] == 1, "word_len"]
    real_word = df.loc[df["fake"] == 0, "word_len"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char_t = stats.ttest_ind(fake_char, real_char, equal_var=False)
        word_t = stats.ttest_ind(fake_word, real_word, equal_var=False)
        char_mw = stats.mannwhitneyu(fake_char, real_char, alternative="two-sided")
        word_mw = stats.mannwhitneyu(fake_word, real_word, alternative="two-sided")

    tests = {
        "rating_vs_label": cramers_v(pd.crosstab(df["rating_int"], df["fake"])),
        "extreme_rating_vs_label": cramers_v(pd.crosstab(df["extreme"], df["fake"])),
        "year_vs_label": cramers_v(pd.crosstab(df["year"], df["fake"])),
        "char_length_bin_vs_label": cramers_v(pd.crosstab(df["char_bin"], df["fake"])),
        "char_length_fake_minus_real": {
            "fake_mean": rnd(fake_char.mean(), 4),
            "real_mean": rnd(real_char.mean(), 4),
            "fake_median": rnd(fake_char.median(), 4),
            "real_median": rnd(real_char.median(), 4),
            "welch_t": rnd(char_t.statistic, 4),
            "welch_p": "<1e-300" if char_t.pvalue == 0 else f"{char_t.pvalue:.3e}",
            "cohen_d": rnd(cohen_d(fake_char, real_char), 6),
            "mann_whitney_p": "<1e-300" if char_mw.pvalue == 0 else f"{char_mw.pvalue:.3e}",
            "cliffs_delta_approx": rnd(cliffs_delta(fake_char, real_char), 6),
        },
        "word_length_fake_minus_real": {
            "fake_mean": rnd(fake_word.mean(), 4),
            "real_mean": rnd(real_word.mean(), 4),
            "fake_median": rnd(fake_word.median(), 4),
            "real_median": rnd(real_word.median(), 4),
            "welch_t": rnd(word_t.statistic, 4),
            "welch_p": "<1e-300" if word_t.pvalue == 0 else f"{word_t.pvalue:.3e}",
            "cohen_d": rnd(cohen_d(fake_word, real_word), 6),
            "mann_whitney_p": "<1e-300" if word_mw.pvalue == 0 else f"{word_mw.pvalue:.3e}",
            "cliffs_delta_approx": rnd(cliffs_delta(fake_word, real_word), 6),
        },
    }

    feature_rows = []
    for feature in [
        "rating",
        "extreme",
        "positive",
        "negative",
        "char_len",
        "word_len",
        "upper_ratio",
        "exclam",
        "question",
        "dollar",
        "user_total_reviews",
        "prod_total_reviews",
    ]:
        feature_rows.append(
            {
                "feature": feature,
                "pearson_r": rnd(stats.pearsonr(df[feature], df["fake"]).statistic, 6),
                "spearman_r": rnd(stats.spearmanr(df[feature], df["fake"]).statistic, 6),
                "roc_auc_higher_feature_fake": rnd(roc_auc_score(df["fake"], df[feature]), 6),
                "average_precision": rnd(average_precision_score(df["fake"], df[feature]), 6),
            }
        )
    feature_screen = pd.DataFrame(feature_rows)

    df["short_review_le_100_chars"] = (df["char_len"] <= 100).astype(int)
    df["very_long_gt_2000_chars"] = (df["char_len"] > 2000).astype(int)
    df["one_time_user"] = (df["user_total_reviews"] == 1).astype(int)
    df["low_review_product_le_30"] = (df["prod_total_reviews"] <= 30).astype(int)
    odds = pd.DataFrame(
        [
            odds_ratio_for_flag(df, flag)
            for flag in [
                "extreme",
                "negative",
                "positive",
                "short_review_le_100_chars",
                "very_long_gt_2000_chars",
                "one_time_user",
                "low_review_product_le_30",
            ]
        ]
    )

    print_section("10-1. 통계검정과 효과크기")
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    print_table("10-2. 단일 변수 상관 및 예측력 점검", feature_screen)
    print_table("10-3. 주요 이진 플래그의 odds ratio", odds)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = odds.sort_values("odds_ratio", ascending=False)
    sns.barplot(data=plot_df, x="odds_ratio", y="flag", ax=ax, color="#c44e52")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Odds Ratio for Common Risk Flags")
    ax.set_xlabel("odds ratio for fake review")
    ax.set_ylabel("")
    save_fig(fig, figures_dir, "15_odds_ratio_risk_flags", show)

    return {
        "statistical_tests": tests,
        "feature_screen": feature_screen.round(6).to_dict("records"),
        "odds_ratios": odds.round(6).to_dict("records"),
    }


# 상품 수준에서 평점 분산, 극단 평점 비율, 엔트로피와 가짜 비율의 관계를 본다.
# 상품 평균 평점보다 평점 분포의 흔들림이 더 중요한지 확인하는 분석이다.
def analyze_product_rating_structure(product: pd.DataFrame, df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    product_rating = (
        df.groupby("prod_id")
        .agg(
            reviews=("fake", "size"),
            fake_rate=("fake", "mean"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            extreme_rate=("extreme", "mean"),
            rating_entropy=("rating_int", rating_entropy),
        )
        .reset_index()
    )
    product_rating["rating_std"] = product_rating["rating_std"].fillna(0)

    correlations = []
    for feature in ["reviews", "rating_mean", "rating_std", "extreme_rate", "rating_entropy"]:
        correlations.append(
            {
                "feature": feature,
                "spearman_with_product_fake_rate": rnd(stats.spearmanr(product_rating[feature], product_rating["fake_rate"]).statistic, 6),
                "pearson_with_product_fake_rate": rnd(stats.pearsonr(product_rating[feature], product_rating["fake_rate"]).statistic, 6),
            }
        )
    corr_df = pd.DataFrame(correlations)

    print_table("11. 상품 수준 평점 구조와 상품별 가짜 비율의 상관", corr_df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=product_rating, x="rating_std", y="fake_rate", size="reviews", sizes=(10, 180), alpha=0.45, ax=axes[0])
    axes[0].set_title("Product Fake Rate vs Rating Std")
    sns.scatterplot(data=product_rating, x="extreme_rate", y="fake_rate", size="reviews", sizes=(10, 180), alpha=0.45, ax=axes[1])
    axes[1].set_title("Product Fake Rate vs Extreme Rating Share")
    save_fig(fig, figures_dir, "16_product_rating_structure_scatter", show)

    return {"product_rating_structure_correlations": corr_df.round(6).to_dict("records")}


# 시간 분할에 따른 분포 이동과 cold-start 상황을 점검한다.
# 미래 구간에 새 사용자가 얼마나 많은지 확인하면 운영 환경과 모델 입력 설계를 더 현실적으로 잡을 수 있다.
def analyze_temporal_drift_and_cold_start(df: pd.DataFrame, figures_dir: Path, show: bool) -> dict[str, object]:
    split_summary = (
        df.groupby("split")
        .agg(
            reviews=("fake", "size"),
            fake=("fake", "sum"),
            fake_rate=("fake", "mean"),
            rating_mean=("rating", "mean"),
            char_mean=("char_len", "mean"),
            word_mean=("word_len", "mean"),
        )
        .reindex(["train", "valid", "test"])
        .reset_index()
    )
    split_rating = (
        df.groupby(["split", "rating_int"])
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean"))
        .reset_index()
    )

    ks_rows = []
    for feature in ["rating", "char_len", "word_len", "user_total_reviews", "prod_total_reviews"]:
        train_values = df.loc[df["split"] == "train", feature]
        test_values = df.loc[df["split"] == "test", feature]
        test_result = stats.ks_2samp(train_values, test_values)
        ks_rows.append(
            {
                "feature": feature,
                "ks_statistic": rnd(test_result.statistic, 6),
                "p": "<1e-300" if test_result.pvalue == 0 else f"{test_result.pvalue:.3e}",
                "train_mean": rnd(train_values.mean(), 4),
                "test_mean": rnd(test_values.mean(), 4),
                "train_median": rnd(train_values.median(), 4),
                "test_median": rnd(test_values.median(), 4),
            }
        )
    ks_df = pd.DataFrame(ks_rows)

    train_users = set(df.loc[df["split"] == "train", "user_id"])
    train_products = set(df.loc[df["split"] == "train", "prod_id"])
    df["user_seen_in_train"] = df["user_id"].isin(train_users)
    df["prod_seen_in_train"] = df["prod_id"].isin(train_products)
    cold_start = (
        df[df["split"].isin(["valid", "test"])]
        .groupby(["split", "user_seen_in_train", "prod_seen_in_train"])
        .agg(reviews=("fake", "size"), fake=("fake", "sum"), fake_rate=("fake", "mean"))
        .reset_index()
    )

    print_table("12-1. split별 주요 분포 요약", split_summary)
    print_table("12-2. split-평점별 가짜 비율", split_rating)
    print_table("12-3. train-test KS 분포 차이 검정", ks_df)
    print_table("12-4. valid/test의 train 사용자·상품 등장 여부별 가짜 비율", cold_start)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=split_summary, x="split", y="char_mean", ax=ax, color="#4c72b0")
    ax.set_title("Average Review Length by Temporal Split")
    ax.set_xlabel("split")
    ax.set_ylabel("mean character length")
    save_fig(fig, figures_dir, "17_temporal_drift_text_length", show)

    fig, ax = plt.subplots(figsize=(9, 5))
    test_cold = cold_start[cold_start["split"] == "test"].copy()
    test_cold["seen_group"] = test_cold.apply(
        lambda row: f"user_seen={row['user_seen_in_train']}, product_seen={row['prod_seen_in_train']}",
        axis=1,
    )
    sns.barplot(data=test_cold, x="seen_group", y="fake_rate", ax=ax, color="#dd8452")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Test Fake Rate by Cold-Start Group")
    ax.set_xlabel("")
    ax.set_ylabel("fake rate")
    save_fig(fig, figures_dir, "18_test_cold_start_fake_rate", show)

    return {
        "split_summary": split_summary.round(6).to_dict("records"),
        "split_rating": split_rating.round(6).to_dict("records"),
        "ks_train_vs_test": ks_df.to_dict("records"),
        "cold_start": cold_start.round(6).to_dict("records"),
        "test_user_seen_train_share_pct": pct(df.loc[df["split"] == "test", "user_seen_in_train"].mean()),
        "test_product_seen_train_share_pct": pct(df.loc[df["split"] == "test", "prod_seen_in_train"].mean()),
    }


# 전체 기간 라벨 집계처럼 누수가 되는 변수를 진단용으로만 계산한다.
# 높은 성능이 나오더라도 모델 입력으로 사용하면 미래 라벨 정보를 섞는 것이므로 금지해야 한다.
def analyze_leakage_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    user_label_rate = df.groupby("user_id")["fake"].mean().rename("user_label_rate_all_LEAKY")
    product_label_rate = df.groupby("prod_id")["fake"].mean().rename("prod_label_rate_all_LEAKY")
    df = df.join(user_label_rate, on="user_id").join(product_label_rate, on="prod_id")

    leakage = []
    for feature in ["user_label_rate_all_LEAKY", "prod_label_rate_all_LEAKY"]:
        leakage.append(
            {
                "feature": feature,
                "roc_auc": rnd(roc_auc_score(df["fake"], df[feature]), 6),
                "average_precision": rnd(average_precision_score(df["fake"], df[feature]), 6),
                "note": "전체 기간 라벨 집계이므로 진단용이며 모델 입력으로 사용하면 누수다.",
            }
        )
    leakage_df = pd.DataFrame(leakage)
    print_table("13. 라벨 누수 진단용 집계 변수 성능", leakage_df)
    return {"leakage_diagnostics": leakage}


# JSON이 처리하지 못하는 numpy/pandas 타입을 표준 파이썬 타입으로 변환한다.
# 집계 결과에 numpy 정수 키나 Timestamp가 섞여도 요약 파일 저장이 실패하지 않게 한다.
def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# 핵심 요약을 JSON으로 저장한다.
# 콘솔 출력만으로 지나치게 길어지는 것을 방지하고, 보고서 작성 시 재사용할 수 있게 한다.
def save_summary(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eda_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    print(f"\n[SUMMARY] {summary_path}")


# 각 분석 단계에서 필요한 파생 변수와 집계 결과를 순서대로 생성한다.
# 실행 시간이 오래 걸리는 텍스트 분석은 한 번만 수행한 뒤 뒤쪽 분석에서 재사용한다.
def run_eda(csv_path: Path, output_dir: Path, show: bool = False) -> None:
    figures_dir = output_dir / "figures"
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    print_section("YelpZip EDA 시작")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")

    df = load_metadata(csv_path)
    text_features = extract_text_features(csv_path)
    df = pd.concat([df.reset_index(drop=True), text_features], axis=1)

    summary: dict[str, object] = {}
    summary["basic_quality"] = analyze_basic_quality(df, csv_path, output_dir)
    summary["labels_and_ratings"] = analyze_labels_and_ratings(df, figures_dir, show)
    summary["time_patterns"] = analyze_time_patterns(df, figures_dir, show)
    df, summary["temporal_split"] = analyze_temporal_split(df, figures_dir, show)
    summary["text_features"] = analyze_text_features(df, figures_dir, show)
    product, user, summary["user_product_concentration"] = analyze_user_product_concentration(df, figures_dir, show)

    df = df.join(user.set_index("user_id")["reviews"].rename("user_total_reviews"), on="user_id")
    df = df.join(product.set_index("prod_id")["reviews"].rename("prod_total_reviews"), on="prod_id")

    summary["user_recency"] = analyze_user_recency(df, figures_dir, show)
    summary["burst_patterns"] = analyze_burst_patterns(df, figures_dir, show)
    summary["duplicates"] = analyze_duplicates(df, figures_dir, show)
    summary["statistical_tests"] = analyze_statistical_tests(df, figures_dir, show)
    summary["product_rating_structure"] = analyze_product_rating_structure(product, df, figures_dir, show)
    summary["temporal_drift_and_cold_start"] = analyze_temporal_drift_and_cold_start(df, figures_dir, show)
    summary["leakage_diagnostics"] = analyze_leakage_diagnostics(df)

    save_summary(summary, output_dir)
    print_section("YelpZip EDA 완료")
    print(f"그래프 폴더: {figures_dir}")


# 명령행 인자를 처리한다.
# 기본 실행은 python Preprocessing.py 이며, --show를 붙이면 저장과 동시에 그래프 창을 표시한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for YelpZip CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to yelpzip.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for EDA outputs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively after saving them")
    return parser.parse_args()


# 스크립트 진입점이다.
# PyCharm이나 PowerShell에서 직접 실행하면 전체 EDA가 순차적으로 수행된다.
if __name__ == "__main__":
    args = parse_args()
    run_eda(args.csv, args.output, args.show)

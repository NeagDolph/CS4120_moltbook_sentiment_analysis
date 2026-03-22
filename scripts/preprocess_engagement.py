#!/usr/bin/env python3
"""Preprocess Moltbook posts: clean data, compute text features,
engagement scores, and engagement quartile classes.

Reads:  data/moltbook_posts_flat.parquet
Writes: data/moltbook_posts_preprocessed.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "moltbook_posts_flat.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "moltbook_posts_preprocessed.parquet"

ENGAGEMENT_CLASSES = ["lower", "lower_middle", "upper_middle", "upper"]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


# ── Step 1: Clean ──────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    df = df.copy()
    df["content"] = df["content"].fillna("")
    df["title"] = df["title"].fillna("")

    df["has_url"] = df["url"].notna().astype(int)
    df = df.drop(columns=["url"])

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # drop rows where both title and content are empty
    empty_mask = (df["title"].str.strip() == "") & (df["content"].str.strip() == "")
    n_empty = empty_mask.sum()
    df = df[~empty_mask].reset_index(drop=True)

    print(f"Clean: filled nulls, parsed dates, dropped {n_empty} empty-text rows "
          f"({n_before} → {len(df)})")
    return df


# ── Step 2: Text features ─────────────────────────────────────────────────

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["title_char_len"] = df["title"].str.len()
    df["content_char_len"] = df["content"].str.len()
    df["combined_text"] = (df["title"] + " " + df["content"]).str.strip()
    df["combined_char_len"] = df["combined_text"].str.len()

    df["title_word_count"] = df["title"].str.split().str.len().fillna(0).astype(int)
    df["content_word_count"] = df["content"].str.split().str.len().fillna(0).astype(int)
    df["combined_word_count"] = df["title_word_count"] + df["content_word_count"]

    # sentence count: split on .!? followed by whitespace or end-of-string
    df["sentence_count"] = (
        df["combined_text"]
        .str.split(r"[.!?]+\s*")
        .str.len()
        .fillna(0)
        .astype(int)
    )

    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek  # 0=Mon … 6=Sun

    print(f"Text features: added char/word counts, sentence count, "
          f"hour_of_day, day_of_week")
    return df


# ── Step 5: Engagement score ──────────────────────────────────────────────

def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement_score"] = (
        np.log1p(df["upvotes"])
        + np.log1p(df["comment_count"])
        - 0.5 * np.log1p(df["downvotes"])
    )
    print(f"Engagement score: min={df['engagement_score'].min():.3f}, "
          f"median={df['engagement_score'].median():.3f}, "
          f"mean={df['engagement_score'].mean():.3f}, "
          f"max={df['engagement_score'].max():.3f}")
    return df


# ── Step 6: Engagement classes ─────────────────────────────────────────────

def add_engagement_class(df: pd.DataFrame) -> pd.DataFrame:
    """Assign engagement quartile classes.

    Posts with engagement_score <= 0 are assigned to 'lower'.
    The remaining posts are split into tertiles mapped to
    'lower_middle', 'upper_middle', 'upper'.
    """
    df = df.copy()

    zero_mask = df["engagement_score"] <= 0.0
    n_zero = zero_mask.sum()

    classes = pd.Series("", index=df.index, dtype="object")
    classes[zero_mask] = "lower"

    nonzero = df.loc[~zero_mask, "engagement_score"]
    tertile_labels = ["lower_middle", "upper_middle", "upper"]
    tertiles = pd.qcut(nonzero, q=3, labels=tertile_labels)
    classes[~zero_mask] = tertiles.astype(str)

    df["engagement_class"] = pd.Categorical(
        classes, categories=ENGAGEMENT_CLASSES, ordered=True
    )

    class_counts = df["engagement_class"].value_counts().reindex(ENGAGEMENT_CLASSES)
    total = len(df)
    print(f"Engagement classes ({n_zero} zero-or-negative-score → 'lower', "
          f"rest split into tertiles):")
    for cls in ENGAGEMENT_CLASSES:
        n = class_counts[cls]
        print(f"  {cls:>14s}: {n:>6,}  ({n / total * 100:5.1f}%)")

    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Moltbook posts for engagement analysis"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input parquet (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output parquet (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    df = load_raw(args.input)
    df = clean(df)
    df = add_text_features(df)
    df = add_engagement_score(df)
    df = add_engagement_class(df)

    # summary
    print(f"\nFinal schema ({len(df):,} rows, {len(df.columns)} columns):")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    df.to_parquet(args.output, index=False)
    print(f"\nWrote → {args.output}")


if __name__ == "__main__":
    main()

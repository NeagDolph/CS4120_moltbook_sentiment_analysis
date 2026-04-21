"""
Connects to the API and grabs post data
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DATASET_ID = "TrustAIRLab/Moltbook"


def flatten_posts_to_dataframe(ds) -> pd.DataFrame:
    rows = []
    for r in ds:
        p = r.get("post") or {}
        sm = p.get("submolt") or {}
        rows.append(
            {
                "annotation_row_id": r.get("id"),
                "topic_label": r.get("topic_label"),
                "toxic_level": r.get("toxic_level"),
                "post_id": p.get("id"),
                "title": p.get("title"),
                "content": p.get("content"),
                "created_at": p.get("created_at"),
                "comment_count": p.get("comment_count"),
                "upvotes": p.get("upvotes"),
                "downvotes": p.get("downvotes"),
                "url": p.get("url"),
                "submolt_id": sm.get("id"),
                "submolt_name": sm.get("name"),
                "submolt_display_name": sm.get("display_name"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Moltbook dataset into data/")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV copies (larger on disk; parquet is default)",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading posts from {DATASET_ID} …")
    posts = load_dataset(DATASET_ID, "posts", split="train")
    df_posts = flatten_posts_to_dataframe(posts)
    posts_parquet = data_dir / "moltbook_posts_flat.parquet"
    df_posts.to_parquet(posts_parquet, index=False)
    print(f"Wrote {len(df_posts):,} rows → {posts_parquet}")

    print(f"Loading submolts from {DATASET_ID} …")
    submolts = load_dataset(DATASET_ID, "submolts", split="train")
    df_sub = submolts.to_pandas()
    sub_parquet = data_dir / "moltbook_submolts.parquet"
    df_sub.to_parquet(sub_parquet, index=False)
    print(f"Wrote {len(df_sub):,} rows → {sub_parquet}")

    if args.csv:
        df_posts.to_csv(data_dir / "moltbook_posts_flat.csv", index=False)
        df_sub.to_csv(data_dir / "moltbook_submolts.csv", index=False)
        print("Wrote CSV copies.")

    manifest = data_dir / "manifest.json"
    meta = {
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "note": "Per-post comment text is not included in this release; only comment_count.",
        "files": {
            "moltbook_posts_flat.parquet": {"rows": len(df_posts)},
            "moltbook_submolts.parquet": {"rows": len(df_sub)},
        },
    }
    manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()

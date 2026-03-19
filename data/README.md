# Moltbook data (local)

Populate this folder by running from the repo root:

```bash
pip install -r requirements.txt
python scripts/download_moltbook.py
```

Optional CSV export (larger files):

```bash
python scripts/download_moltbook.py --csv
```

## What you get

| File | Description |
|------|-------------|
| `moltbook_posts_flat.parquet` | All annotated posts with flattened fields for `title`, `content`, `topic_label`, `toxic_level`, engagement, and submolt metadata. |
| `moltbook_submolts.parquet` | Submolt (community) metadata aligned with the paper’s release. |
| `manifest.json` | Row counts and source URL. |

Primary text for sentiment (or tone) modeling: combine **`title`** and **`content`** (either may be null).

## Comments

The [TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook) release includes **`comment_count`** per post but **does not ship comment bodies**. For comment-level sentiment you would need another source (e.g. Moltbook API if available to you, or a different dataset).

## Labels (from the dataset card)

- **topic_label**: content category `A`–`I` (Identity, Technology, Socializing, …).
- **toxic_level**: `0`–`4` (Safe through Malicious) — useful as a supervised target or for comparison with your sentiment model.

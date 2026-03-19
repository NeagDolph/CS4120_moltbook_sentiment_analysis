# Moltbook sentiment analysis

Course project workspace for analyzing [TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook) posts.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_moltbook.py
```

Data files are written under `data/`. See `data/README.md` for schema notes and the comment-data limitation.

Example exploration: open `notebooks/moltbook_eda.ipynb` (loads parquet, schema, and summary stats).

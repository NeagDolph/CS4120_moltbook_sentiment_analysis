# Moltbook Sentiment Analysis

CS4120 (NLP) course project on the [TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook) dataset of annotated forum posts. We engineer a small set of linguistic and metadata features for each post (VADER + TextBlob sentiment, POS counts, topic and toxicity one-hots, length and timing stats) and use them to predict an engagement score derived from upvotes, downvotes, and comment counts, both as a continuous regression target and as a 4-class quartile.

Final report: [https://docs.google.com/document/d/1v4D_7jgiEd2RmrFbhV_cFQEBaYzJDioYX89EuA9w6PU/edit?tab=t.0](https://docs.google.com/document/d/1v4D_7jgiEd2RmrFbhV_cFQEBaYzJDioYX89EuA9w6PU/edit?tab=t.0)

The repo has two helper scripts in `scripts/` (`download_moltbook.py`, `preprocess_engagement.py`), four notebooks in `notebooks/` (`feature_engineering.ipynb`, `moltbook_eda.ipynb`, `linear_reg.ipynb`, `FFNN.ipynb`), and `requirements.txt`. The `data/` directory is generated locally by the steps below and is not committed.

## Setup

Python 3.11.

```bash
git clone https://github.com/NeagDolph/CS4120_moltbook_sentiment_analysis.git
cd CS4120_moltbook_sentiment_analysis
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Jupyter is in `requirements.txt`, no other system dependencies.

## Reproducing the results

Run the four steps in order. The first two scripts populate `data/`, then the notebooks read from there:

```bash
python scripts/download_moltbook.py
python scripts/preprocess_engagement.py
jupyter nbconvert --to notebook --execute --inplace notebooks/feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/moltbook_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/linear_reg.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/FFNN.ipynb
```

Or open them in `jupyter lab`. Total runtime is around 5 minutes on CPU; feature engineering (POS tagging) and the FFNN hyperparameter sweep are the longest steps at roughly 2-3 minutes each.

## Notebooks

`feature_engineering.ipynb` reads the preprocessed parquet, adds VADER and TextBlob sentiment plus NLTK part-of-speech counts, one-hot encodes `topic_label` and `toxic_level`, does a 70/15/15 stratified split, and writes the `X_*.npy`, `y_*.npy`, `idx_*.npy`, and `meta.pkl` files to `data/features/` (plus `data/moltbook_posts_with_sentiment.parquet` for the EDA notebook).

`moltbook_eda.ipynb` is the exploratory pass: dataset shape and schema, topic, toxicity, submolt distributions, engagement statistics, and length, sentiment, and temporal patterns.

`linear_reg.ipynb` trains an `sklearn` linear regression on the 29 engineered features against the continuous engagement score, reports MSE and R^2 on val/test, and shows a coefficient bar chart and a feature correlation heatmap.

`FFNN.ipynb` trains a Keras feed-forward classifier on the engagement quartile, sweeps six (architecture, activation, dropout) configurations, picks the best one, retrains for up to 80 epochs, and reports precision, recall, F1, and accuracy along with a confusion matrix on the test set.

## Dataset note

The Moltbook release ships `comment_count` per post but not the comment bodies, so any sentiment work in this repo is on post titles and bodies only. `topic_label` is a categorical A–I (Identity, Technology, Socializing, etc.) and `toxic_level` is 0–4 (Safe through Malicious); both are used as one-hot features. See the [dataset card](https://huggingface.co/datasets/TrustAIRLab/Moltbook) for the full schema.
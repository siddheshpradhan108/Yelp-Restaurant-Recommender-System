# Yelp Restaurant Hybrid Recommender (ALS + LightGBM)

End-to-end offline ranking pipeline on the [Yelp Open Dataset](https://www.yelp.com/dataset). It implements a **two-stage hybrid recommender**: **implicit alternating least squares (ALS)** for collaborative filtering and candidate generation, followed by a **LightGBM LambdaRank** learning-to-rank model over rich features. **Sentence-transformer review embeddings** and **NMF topic features** supply content-side signals that especially help **cold-start and sparse users**.

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Methodology](#methodology)
4. [Data preparation](#data-preparation)
5. [Training pipeline](#training-pipeline)
6. [Evaluation and ablations](#evaluation-and-ablations)
7. [Configuration defaults](#configuration-defaults)
8. [CLI reference](#cli-reference)
9. [Artifacts and outputs](#artifacts-and-outputs)
10. [Requirements and troubleshooting](#requirements-and-troubleshooting)
11. [Repository layout](#repository-layout)
12. [Dataset license](#dataset-license)

---

## Overview

This project targets restaurant recommendation using Yelp reviews and business metadata. It combines:

| Stage | Component | Role |
|--------|-----------|------|
| **Stage 1** | Implicit ALS (`implicit`) | Collaborative filtering on **implicit feedback** (reviews treated as positives above a star threshold). Produces user/item latent factors and **top‑K candidate businesses** per user. |
| **Stage 2** | LightGBM LambdaRank | **Learning-to-rank** reranking on candidates using collaborative features (ALS factors and scores), **content** features (embeddings + topics), and **contextual** features (popularity, geography, seasonality scalars). |

**Research narrative:** Review-derived features (transformer embeddings + topics) tend to raise **NDCG@10** more for **sparse** users (few train interactions) than for **dense** users, because collaborative signals are weaker when history is thin. Example offline figures often reported in similar setups include roughly **+3.2 NDCG@10** for sparse users and **+0.4** for dense users when adding rich text features; **this repository recomputes metrics on your data**—see `artifacts/metrics.json` after training.

---

## Architecture

```mermaid
flowchart LR
  subgraph Data
    R[Reviews JSON]
    B[Business JSON]
    U[User JSON optional]
  end

  subgraph Prep
    F[Filter restaurants]
    T[Time split train / test]
  end

  subgraph Stage1
    M[Implicit matrix]
    ALS[ALS fit]
    Cands[Top-K candidates]
  end

  subgraph Content
    TE[Sentence embeddings + PCA]
    TP[NMF topics]
    UE[User history pooling]
  end

  subgraph Stage2
    FEAT[Feature assembly]
    LGB[LightGBM LambdaRank]
    MET[NDCG@10 + cohorts]
  end

  R --> F
  B --> F
  U --> F
  F --> T
  T --> M
  M --> ALS
  ALS --> Cands
  T --> TE
  T --> TP
  TE --> UE
  Cands --> FEAT
  ALS --> FEAT
  TE --> FEAT
  TP --> FEAT
  UE --> FEAT
  FEAT --> LGB
  LGB --> MET
```

---

## Methodology

### Implicit feedback and ALS

- Reviews with **stars ≥ `implicit_positive_min_stars`** (default **4**) are treated as **implicit positives**.
- A sparse **user × item** confidence matrix is built (`implicit` convention: training uses the **item × user** transpose).
- **ALS** hyperparameters (see [Configuration defaults](#configuration-defaults)) control latent dimensionality, regularization, iterations, and confidence scaling **`alpha`**.

### Candidate generation

- For each training user, **ALS recommend** yields up to **`candidate_k`** items (default **200**) with **`filter_already_liked_items=True`** on the **train** interaction matrix.

### LambdaRank training data

- For each sampled user: **positives** come from their train positives; **negatives** are sampled from ALS candidates that are **not** positives.
- Queries are grouped per user for LambdaRank (**`group`** vector).
- A **random subset of users** can be capped via **`--max-ltr-users`** to control runtime.

### Content features

| Signal | Description |
|--------|-------------|
| **Business sentence embeddings** | All train-period review text per business is aggregated; **`sentence-transformers/all-MiniLM-L6-v2`** encodes each business, then **PCA** reduces dimensionality for the ranker (default **32** dims). |
| **User history embeddings** | For each user, average **PCA business embeddings** over businesses they rated positively (≥ 4 stars) in **train**, pooling cold-start-friendly preferences into a fixed-length vector. |
| **Topic features** | **TF-IDF** over aggregated review text per business, then **NMF** (default **20** topics). Rows are **row-normalized** topic activations. |

### Contextual features (ranker)

Examples assembled in code include:

- **`log_user_train_cnt`** — log of user interaction count in train.
- **`log_biz_review_count`**, **`biz_stars`**, **`biz_price`** (from business attributes when present).
- **`biz_lat`**, **`biz_lon`** — geographic context.
- **`train_month_sin`**, **`train_month_cos`** — cyclic encoding from the user’s mean review month in train (seasonality proxy).

Exact lists depend on enabled **feature ablations** (see below).

### Train / test split

- **Temporal split:** reviews sorted by **`date`**; the **last `test_fraction_time`** fraction (default **15%**) forms the **test** set; the remainder is **train**.
- Evaluation uses **users present in the train index** with **non-empty test positives**, subsampled by **`--max-test-users`** for speed.

---

## Data preparation

### Obtaining data

1. Accept Yelp’s terms and download the academic dataset from [yelp.com/dataset](https://www.yelp.com/dataset).
2. Extract JSON files into a directory (default **`data/yelp/`**).

### Expected files

| File | Required |
|------|----------|
| `yelp_academic_dataset_review.json` | Yes |
| `yelp_academic_dataset_business.json` | Yes |
| `yelp_academic_dataset_user.json` | Optional (loaded if present) |

Paths can be overridden via **`DataConfig`** in code or **`--data-dir`** on the CLI.

### Built-in filters (`prepare-data`)

- Businesses whose **`categories`** contain **`Restaurants`** or **`Food`** (configurable).
- Businesses with at least **`min_reviews_per_business`** reviews (default **5**).
- Users with at least **`min_reviews_per_user`** reviews in the filtered set (default **3**).

Filtered tables are saved under **`artifacts/tables.pkl`** (see [Artifacts](#artifacts-and-outputs)).

---

## Training pipeline

Run commands from the repository root (after [environment setup](#requirements-and-troubleshooting)).

```bash
python run_pipeline.py prepare-data
python run_pipeline.py train
```

**Recommended:** after the first full run, reuse computed embeddings and topics:

```bash
python run_pipeline.py train --reuse-embeddings
```

For faster debugging on smaller slices:

```bash
python run_pipeline.py train \
  --max-train-reviews 50000 \
  --max-ltr-users 2000 \
  --max-test-users 400
```

---

## Evaluation and ablations

### Metrics

- **NDCG@10** (`sklearn.metrics.ndcg_score`) per evaluated user over the union of **ALS candidates** and **held-out test items** for that user.
- Users are split into cohorts by **train interaction count**:
  - **Sparse:** **&lt; 5** train interactions (cold-start–like regime).
  - **Dense:** **≥ 5** train interactions.

### Ablations (three trained rankers)

The training script fits **three** LambdaRank models with disjoint feature toggles via **`FeatureControl`**:

| Key | Collaborative | Contextual | Topic (NMF) | Transformer embeddings |
|-----|---------------|------------|-------------|-------------------------|
| **`collab_context_baseline`** | Yes | Yes | No | No |
| **`collab_context_topics`** | Yes | Yes | Yes | No |
| **`full_hybrid`** | Yes | Yes | Yes | Yes |

### `artifacts/metrics.json` structure

After **`train`**, JSON includes per-variant blocks such as:

- **`ndcg10_all`**, **`ndcg10_sparse_lt5`**, **`ndcg10_dense_ge5`**
- **`n_users`**, **`n_sparse`**, **`n_dense`**

Plus aggregated **deltas**, for example:

- **`deltas_vs_collab_context_baseline`** — full hybrid minus baseline (topics + transformers vs baseline).
- **`deltas_topics_only_vs_baseline`** — topics-only minus baseline.
- **`deltas_transformer_plus_topics_vs_topics_only`** — isolates the **incremental** contribution of **transformer embeddings** once topics are already enabled.

Interpretation: larger lifts on **sparse** users vs **dense** users indicate that **content** features compensate when **collaborative** evidence is weak.

---

## Configuration defaults

Defined in **`yelp_hybrid/config.py`** (edit there or extend `ProjectConfig` in code).

### Paths

| Setting | Default |
|---------|---------|
| **Data directory** | `data/yelp` |
| **Artifacts directory** | `artifacts` |

### Data

| Setting | Default |
|---------|---------|
| **Restaurant keywords** | `Restaurants`, `Food` |
| **min_reviews_per_business** | 5 |
| **min_reviews_per_user** | 3 |
| **implicit_positive_min_stars** | 4 |

### ALS

| Setting | Default |
|---------|---------|
| **factors** | 64 |
| **regularization** | 0.08 |
| **iterations** | 20 |
| **alpha** (implicit confidence) | 40.0 |
| **candidate_k** | 200 |

### Embeddings

| Setting | Default |
|---------|---------|
| **model_name** | `sentence-transformers/all-MiniLM-L6-v2` |
| **batch_size** | 64 |
| **max_chars_per_doc** | 8000 |
| **pca_dim** | 32 |

### Topics

| Setting | Default |
|---------|---------|
| **n_topics** | 20 |
| **max_features_tfidf** | 5000 |

### LambdaRank

| Setting | Default |
|---------|---------|
| **learning_rate** | 0.05 |
| **num_leaves** | 63 |
| **min_data_in_leaf** | 20 |
| **feature_fraction** | 0.85 |
| **bagging_fraction** | 0.8 |
| **bagging_freq** | 1 |
| **max_rounds** | 400 |
| **early_stopping_rounds** | 40 |
| **ndcg_eval_at** | (10,) |

### Split

| Setting | Default |
|---------|---------|
| **test_fraction_time** | 0.15 |

---

## CLI reference

Global arguments (before subcommand):

| Argument | Default | Description |
|----------|---------|-------------|
| **`--data-dir`** | `data/yelp` | Directory containing Yelp JSON files. |
| **`--artifacts-dir`** | `artifacts` | Output directory for pickles, models, metrics. |

### `prepare-data`

Loads JSON, applies filters, writes **`artifacts/tables.pkl`**.

### `train`

| Argument | Default | Description |
|----------|---------|-------------|
| **`--max-train-reviews`** | *(none)* | Random subsample of train reviews (caps preprocessing cost). |
| **`--max-ltr-users`** | **8000** | Cap users used to build LambdaRank training rows. |
| **`--max-test-users`** | **1500** | Cap users used for NDCG evaluation. |
| **`--negatives-per-user`** | **35** | Negatives sampled per user for ranker training. |
| **`--reuse-embeddings`** | off | Load **`content_features.pkl`** if present; skip re-encoding. |

---

## Artifacts and outputs

| Path | Contents |
|------|----------|
| **`artifacts/tables.pkl`** | Filtered **`businesses`**, **`reviews`**, optional **`users`** (`YelpTables`). |
| **`artifacts/als_model.pkl`** | Serialized ALS model and **`IndexMaps`** (user/business id indices). |
| **`artifacts/content_features.pkl`** | Business embeddings (PCA), NMF topics, user pooled embeddings, fitted PCA object, ordered business ids. |
| **`artifacts/ltr_full.pkl`** | LightGBM booster for the **full_hybrid** configuration. |
| **`artifacts/metrics.json`** | NDCG and ablation deltas (see [Evaluation](#evaluation-and-ablations)). |

**Note:** Large artifacts and `data/` are gitignored; commit code and README only unless you intentionally version artifacts.

---

## Requirements and troubleshooting

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependency highlights

| Package | Role |
|---------|------|
| **implicit** | ALS on sparse implicit feedback |
| **lightgbm** | LambdaRank |
| **sentence-transformers**, **torch** | Review embeddings |
| **scikit-learn** | PCA, TF-IDF, NMF, NDCG |
| **pandas**, **numpy**, **scipy** | Data and sparse linear algebra |
| **tqdm** | Progress for embedding batches |

### macOS: LightGBM and `libomp`

If Python raises **`Library not loaded: libomp.dylib`** when importing LightGBM, install OpenMP, for example:

```bash
brew install libomp
```

Alternatively use **conda** and install **`lightgbm`** from conda-forge, which often bundles compatible libraries.

### First embedding run

The sentence-transformer model downloads from Hugging Face on first use; ensure network access. Subsequent runs can use **`--reuse-embeddings`** after **`content_features.pkl`** exists.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py              # CLI: prepare-data, train
└── yelp_hybrid/
    ├── __init__.py
    ├── config.py                # Central hyperparameters
    ├── io_utils.py              # JSONL streaming
    ├── data_pipeline.py         # Load, filter, time split, persist tables
    ├── als_model.py             # Implicit matrix, ALS, recommendations
    ├── embeddings.py            # Sentence embeddings, NMF, user pooling
    ├── ltr.py                   # LambdaRank dataset, FeatureControl, training
    └── evaluation.py            # NDCG@10, sparse/dense cohorts
```

---

## Dataset license

The Yelp Open Dataset is subject to Yelp’s **Dataset Agreement**. Use the data only in compliance with those terms; do not redistribute raw dataset files in this repository.

---

## Citation

If you use this pipeline in academic or portfolio work, cite the **Yelp dataset** per Yelp’s documentation and reference this repository as your implementation of a hybrid ALS + LambdaRank recommender with neural review embeddings and topic features.

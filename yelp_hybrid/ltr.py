from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from yelp_hybrid.als_model import IndexMaps, recommend_candidates_for_users
from yelp_hybrid.config import ProjectConfig


@dataclass
class FeatureControl:
    use_collaborative: bool = True
    use_contextual: bool = True
    """Topic features from NMF over review TF-IDF."""
    use_topic_features: bool = True
    """MiniLM sentence embeddings for business text + user history pooling."""
    use_transformer_embeddings: bool = True


def _als_score(
    user_idx: int,
    item_idx: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> float:
    return float(np.dot(user_factors[user_idx], item_factors[item_idx]))


def assemble_pair_features(
    user_idx: int,
    item_idx: int,
    als_user_factors: np.ndarray,
    als_item_factors: np.ndarray,
    als_score: float,
    user_stats: dict[str, float],
    biz_row: pd.Series | dict,
    business_emb: np.ndarray,
    business_topics: np.ndarray,
    user_hist_emb: np.ndarray | None,
    train_month_mean_for_user: float | None,
    ctrl: FeatureControl,
) -> tuple[list[float], list[str]]:
    names: list[str] = []
    vals: list[float] = []

    if ctrl.use_collaborative:
        names.append("als_score")
        vals.append(als_score)
        for k in range(als_user_factors.shape[1]):
            names.append(f"als_u_{k}")
            vals.append(float(als_user_factors[user_idx, k]))
        for k in range(als_item_factors.shape[1]):
            names.append(f"als_i_{k}")
            vals.append(float(als_item_factors[item_idx, k]))

    if ctrl.use_transformer_embeddings:
        for k in range(business_emb.shape[1]):
            names.append(f"biz_emb_{k}")
            vals.append(float(business_emb[item_idx, k]))
        if user_hist_emb is not None:
            for k in range(user_hist_emb.shape[1]):
                names.append(f"user_hist_emb_{k}")
                vals.append(float(user_hist_emb[user_idx, k]))

    if ctrl.use_topic_features:
        for k in range(business_topics.shape[1]):
            names.append(f"biz_topic_{k}")
            vals.append(float(business_topics[item_idx, k]))

    if ctrl.use_contextual:
        names.extend(
            [
                "log_user_train_cnt",
                "log_biz_review_count",
                "biz_stars",
                "biz_price",
                "biz_lat",
                "biz_lon",
                "train_month_sin",
                "train_month_cos",
            ]
        )
        tm = train_month_mean_for_user if train_month_mean_for_user is not None else 6.0
        vals.extend(
            [
                user_stats["log_train_cnt"],
                float(np.log1p(biz_row["review_count"])),
                float(biz_row["stars"]),
                float(biz_row["attributes_PriceRange"])
                if pd.notna(biz_row["attributes_PriceRange"])
                else -1.0,
                float(biz_row["latitude"]) if pd.notna(biz_row["latitude"]) else 0.0,
                float(biz_row["longitude"]) if pd.notna(biz_row["longitude"]) else 0.0,
                np.sin(2 * np.pi * (tm / 12.0)),
                np.cos(2 * np.pi * (tm / 12.0)),
            ]
        )

    return vals, names


def user_train_stats(train_reviews: pd.DataFrame, maps: IndexMaps) -> pd.DataFrame:
    g = train_reviews.groupby("user_id").agg(
        train_cnt=("business_id", "count"),
        mean_month=("date", lambda s: float(s.dt.month.mean())),
    )
    g["log_train_cnt"] = np.log1p(g["train_cnt"].astype(np.float32))
    return g


def build_ltr_train_matrix(
    cfg: ProjectConfig,
    maps: IndexMaps,
    train_reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    user_item_train: csr_matrix,
    als_model,
    content: dict,
    user_stats_df: pd.DataFrame,
    ctrl: FeatureControl,
    negatives_per_user: int = 40,
    seed: int = 42,
    max_train_users: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    biz_df = businesses.set_index("business_id")

    user_factors = als_model.user_factors
    item_factors = als_model.item_factors

    bids_ordered = content["bids_ordered"]
    emb = content["business_emb"]
    topics = content["business_topics"]
    user_hist_emb = content["user_hist_emb"]

    rows: list[list[float]] = []
    labels: list[float] = []
    groups: list[int] = []

    users_pool = [u for u in maps.idx_to_user_id.values()]
    if max_train_users:
        users_pool = list(rng.choice(users_pool, size=min(max_train_users, len(users_pool)), replace=False))

    feature_names: list[str] | None = None

    pos_mask = train_reviews["stars"] >= cfg.data.implicit_positive_min_stars
    positives = train_reviews[pos_mask].groupby("user_id")["business_id"].apply(set).to_dict()

    for user_id in users_pool:
        if user_id not in maps.user_id_to_idx:
            continue
        uidx = maps.user_id_to_idx[user_id]
        pos_bids = positives.get(user_id)
        if not pos_bids or len(pos_bids) < 1:
            continue

        pos_list = [maps.business_id_to_idx[b] for b in pos_bids if b in maps.business_id_to_idx]
        if not pos_list:
            continue

        # negatives from ALS top-K
        cand_ids, cand_scores = recommend_candidates_for_users(
            als_model,
            np.array([uidx], dtype=np.int32),
            user_item_train,
            k=cfg.als.candidate_k,
        )
        cand_ids = cand_ids[0]
        cand_scores = cand_scores[0]
        neg_pool = [int(i) for i in cand_ids if i not in pos_list]
        if len(neg_pool) < negatives_per_user:
            continue
        negs = rng.choice(neg_pool, size=negatives_per_user, replace=False).tolist()

        # sample up to 10 positives to balance
        k_pos = min(10, len(pos_list))
        pos_pick = rng.choice(pos_list, size=k_pos, replace=False).tolist()

        stats_row = user_stats_df.loc[user_id] if user_id in user_stats_df.index else None
        if stats_row is None:
            user_stat = {"log_train_cnt": 0.0}
            month_mu = 6.0
        else:
            user_stat = {"log_train_cnt": float(stats_row["log_train_cnt"])}
            month_mu = float(stats_row["mean_month"])

        group_n = 0
        for item_idx in pos_pick + negs:
            bid = maps.idx_to_business_id[item_idx]
            br = biz_df.loc[bid]
            score = _als_score(uidx, item_idx, user_factors, item_factors)
            uh = user_hist_emb[uidx]
            feats, names = assemble_pair_features(
                uidx,
                item_idx,
                user_factors,
                item_factors,
                score,
                user_stat,
                br,
                emb,
                topics,
                uh,
                month_mu,
                ctrl,
            )
            if feature_names is None:
                feature_names = names
            label = 1.0 if item_idx in pos_list else 0.0
            rows.append(feats)
            labels.append(label)
            group_n += 1

        if group_n > 0:
            groups.append(group_n)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    group_arr = np.asarray(groups, dtype=np.int32)
    assert feature_names is not None
    return X, y, group_arr, feature_names


def split_train_val_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(groups) <= 1:
        # Degenerate case: reuse train as validation for smoke tests / tiny subsamples.
        return X, y, groups, X, y, groups

    order = np.arange(len(groups))
    rng.shuffle(order)
    n_val = max(1, int(len(groups) * val_fraction))
    val_groups = set(order[:n_val].tolist())

    train_row_idx: list[int] = []
    val_row_idx: list[int] = []
    row = 0
    for gi, sz in enumerate(groups):
        sz = int(sz)
        rows = list(range(row, row + sz))
        if gi in val_groups:
            val_row_idx.extend(rows)
        else:
            train_row_idx.extend(rows)
        row += sz

    tr = np.asarray(train_row_idx, dtype=np.int32)
    va = np.asarray(val_row_idx, dtype=np.int32)

    g_tr = np.asarray([groups[i] for i in range(len(groups)) if i not in val_groups], dtype=np.int32)
    g_va = np.asarray([groups[i] for i in range(len(groups)) if i in val_groups], dtype=np.int32)

    return X[tr], y[tr], g_tr, X[va], y[va], g_va


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_val: np.ndarray,
    feature_names: Sequence[str],
    cfg: ProjectConfig,
) -> lgb.Booster:
    train_ds = lgb.Dataset(X_train, label=y_train, group=group_train, feature_name=list(feature_names))
    val_ds = lgb.Dataset(X_val, label=y_val, group=group_val, feature_name=list(feature_names), reference=train_ds)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": list(cfg.ltr.ndcg_eval_at),
        "learning_rate": cfg.ltr.learning_rate,
        "num_leaves": cfg.ltr.num_leaves,
        "min_data_in_leaf": cfg.ltr.min_data_in_leaf,
        "feature_fraction": cfg.ltr.feature_fraction,
        "bagging_fraction": cfg.ltr.bagging_fraction,
        "bagging_freq": cfg.ltr.bagging_freq,
        "verbosity": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=cfg.ltr.max_rounds,
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg.ltr.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


def predict_scores(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    it = booster.best_iteration if booster.best_iteration is not None else booster.num_trees()
    return booster.predict(X, num_iteration=it)

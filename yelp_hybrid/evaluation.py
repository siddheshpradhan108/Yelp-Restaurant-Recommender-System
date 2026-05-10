from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score

from yelp_hybrid.als_model import IndexMaps, recommend_candidates_for_users
from yelp_hybrid.config import ProjectConfig
from yelp_hybrid.ltr import FeatureControl, assemble_pair_features


@dataclass
class NDCGReport:
    ndcg10_all: float
    ndcg10_sparse: float
    ndcg10_dense: float
    n_users_all: int
    n_users_sparse: int
    n_users_dense: int


def _predict_batch(booster: lgb.Booster | None, X: np.ndarray, als_scores: np.ndarray | None) -> np.ndarray:
    if booster is not None:
        n = booster.best_iteration if booster.best_iteration is not None else booster.num_trees()
        return booster.predict(X, num_iteration=n)
    assert als_scores is not None
    return als_scores


def evaluate_users(
    cfg: ProjectConfig,
    maps: IndexMaps,
    train_reviews: pd.DataFrame,
    test_reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    user_item_train: csr_matrix,
    als_model,
    content: dict,
    user_stats_df: pd.DataFrame,
    ctrl: FeatureControl,
    booster: lgb.Booster | None,
    sparse_threshold: int = 5,
    max_test_users: int | None = None,
    seed: int = 42,
) -> NDCGReport:
    rng = np.random.default_rng(seed)
    biz_df = businesses.set_index("business_id")

    train_counts = train_reviews.groupby("user_id").size()

    test_groups = test_reviews.groupby("user_id")["business_id"].apply(lambda s: set(s.tolist())).to_dict()

    emb = content["business_emb"]
    topics = content["business_topics"]
    user_hist_emb = content["user_hist_emb"]

    user_factors = als_model.user_factors
    item_factors = als_model.item_factors

    users_eval = [u for u in test_groups if u in maps.user_id_to_idx and u in train_counts.index]
    if max_test_users:
        users_eval = list(rng.choice(users_eval, size=min(max_test_users, len(users_eval)), replace=False))

    ndcgs_all: list[float] = []
    ndcgs_sparse: list[float] = []
    ndcgs_dense: list[float] = []

    for user_id in users_eval:
        uidx = maps.user_id_to_idx[user_id]
        rel = test_groups[user_id]
        rel_idx = {maps.business_id_to_idx[b] for b in rel if b in maps.business_id_to_idx}
        if not rel_idx:
            continue

        cand_ids, cand_scores = recommend_candidates_for_users(
            als_model,
            np.array([uidx], dtype=np.int32),
            user_item_train,
            k=cfg.als.candidate_k,
        )
        cand_set = set(int(x) for x in cand_ids[0].tolist())
        cand_set |= rel_idx
        items = sorted(cand_set)

        stats_row = user_stats_df.loc[user_id] if user_id in user_stats_df.index else None
        if stats_row is None:
            user_stat = {"log_train_cnt": 0.0}
            month_mu = 6.0
        else:
            user_stat = {"log_train_cnt": float(stats_row["log_train_cnt"])}
            month_mu = float(stats_row["mean_month"])

        rows: list[list[float]] = []
        als_only: list[float] = []
        for item_idx in items:
            bid = maps.idx_to_business_id[item_idx]
            br = biz_df.loc[bid]
            score = float(np.dot(user_factors[uidx], item_factors[item_idx]))
            als_only.append(score)
            feats, _ = assemble_pair_features(
                uidx,
                item_idx,
                user_factors,
                item_factors,
                score,
                user_stat,
                br,
                emb,
                topics,
                user_hist_emb,
                month_mu,
                ctrl,
            )
            rows.append(feats)

        X = np.asarray(rows, dtype=np.float32)
        if booster is not None:
            scores = _predict_batch(booster, X, None)
        else:
            scores = np.asarray(als_only, dtype=np.float32)

        y_true = np.asarray([1.0 if i in rel_idx else 0.0 for i in items], dtype=np.float32)
        if y_true.sum() < 1:
            continue

        nd = ndcg_score(y_true.reshape(1, -1), scores.reshape(1, -1), k=10)
        ndcgs_all.append(float(nd))

        tc = int(train_counts[user_id]) if user_id in train_counts.index else 0
        if tc < sparse_threshold:
            ndcgs_sparse.append(float(nd))
        else:
            ndcgs_dense.append(float(nd))

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return NDCGReport(
        ndcg10_all=mean(ndcgs_all),
        ndcg10_sparse=mean(ndcgs_sparse),
        ndcg10_dense=mean(ndcgs_dense),
        n_users_all=len(ndcgs_all),
        n_users_sparse=len(ndcgs_sparse),
        n_users_dense=len(ndcgs_dense),
    )

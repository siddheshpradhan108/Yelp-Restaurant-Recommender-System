from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


from yelp_hybrid.config import ALSConfig


@dataclass
class IndexMaps:
    user_id_to_idx: dict[str, int]
    idx_to_user_id: dict[int, str]
    business_id_to_idx: dict[str, int]
    idx_to_business_id: dict[int, str]


def build_index_maps(train_reviews: pd.DataFrame) -> IndexMaps:
    users = sorted(train_reviews["user_id"].unique())
    items = sorted(train_reviews["business_id"].unique())
    user_id_to_idx = {u: i for i, u in enumerate(users)}
    business_id_to_idx = {b: i for i, b in enumerate(items)}
    idx_to_user_id = {i: u for u, i in user_id_to_idx.items()}
    idx_to_business_id = {i: b for b, i in business_id_to_idx.items()}
    return IndexMaps(
        user_id_to_idx=user_id_to_idx,
        idx_to_user_id=idx_to_user_id,
        business_id_to_idx=business_id_to_idx,
        idx_to_business_id=idx_to_business_id,
    )


def build_implicit_matrix(
    train_reviews: pd.DataFrame,
    maps: IndexMaps,
    implicit_positive_min_stars: int,
    als_cfg: ALSConfig,
) -> csr_matrix:
    """User-item confidence matrix C = 1 + alpha * R for implicit ALS."""
    ui = train_reviews["user_id"].map(maps.user_id_to_idx)
    ii = train_reviews["business_id"].map(maps.business_id_to_idx)
    mask = train_reviews["stars"] >= implicit_positive_min_stars
    ui = ui[mask].astype(np.int32)
    ii = ii[mask].astype(np.int32)
    n_users = len(maps.user_id_to_idx)
    n_items = len(maps.business_id_to_idx)
    values = np.ones(len(ui), dtype=np.float32) * als_cfg.alpha
    mat = csr_matrix((values, (ui.to_numpy(), ii.to_numpy())), shape=(n_users, n_items))
    mat.data += 1.0
    return mat


def train_als(item_user: csr_matrix, cfg: ALSConfig) -> AlternatingLeastSquares:
    """implicit expects item-user CSR for training."""
    model = AlternatingLeastSquares(
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        random_state=cfg.random_state,
    )
    model.fit(item_user)
    return model


def recommend_candidates_for_users(
    model: AlternatingLeastSquares,
    user_indices: np.ndarray,
    user_item_train: csr_matrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (user_row_idx, item_indices [n_users, k]) scores optional — implicit recommends returns ids and scores.
    """
    rec_ids = []
    rec_scores = []
    for u in user_indices:
        ids, scores = model.recommend(
            u,
            user_item_train[u],
            N=k,
            filter_already_liked_items=True,
        )
        rec_ids.append(ids)
        rec_scores.append(scores)
    return np.stack(rec_ids, axis=0), np.stack(rec_scores, axis=0)

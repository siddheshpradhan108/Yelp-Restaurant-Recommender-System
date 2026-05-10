from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from tqdm import tqdm

from yelp_hybrid.config import EmbeddingConfig, TopicConfig, ProjectConfig


def aggregate_review_text_per_business(reviews: pd.DataFrame) -> pd.Series:
    g = reviews.groupby("business_id")["text"].apply(lambda s: "\n".join(s.astype(str).tolist()))
    return g


def compute_sentence_embeddings(
    business_ids: list[str],
    text_by_business: pd.Series,
    emb_cfg: EmbeddingConfig,
) -> tuple[np.ndarray, PCA | None]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(emb_cfg.model_name)
    texts = [str(text_by_business.get(bid, ""))[: emb_cfg.max_chars_per_doc] for bid in business_ids]
    embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), emb_cfg.batch_size), desc="Transformer embeddings"):
        batch = texts[i : i + emb_cfg.batch_size]
        v = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs.append(v)
    X = np.vstack(embs).astype(np.float32)
    pca = PCA(n_components=min(emb_cfg.pca_dim, X.shape[1], X.shape[0]), random_state=42)
    Xp = pca.fit_transform(X).astype(np.float32)
    return Xp, pca


def compute_topic_features(
    business_ids: list[str],
    text_by_business: pd.Series,
    topic_cfg: TopicConfig,
) -> np.ndarray:
    texts = [str(text_by_business.get(bid, "")) for bid in business_ids]
    min_df = 3 if len(texts) >= 30 else 1
    vec = TfidfVectorizer(
        max_features=topic_cfg.max_features_tfidf,
        stop_words="english",
        min_df=min_df,
        max_df=0.95,
    )
    X = vec.fit_transform(texts)
    n_topics = min(topic_cfg.n_topics, X.shape[1], X.shape[0])
    nmf = NMF(
        n_components=max(2, n_topics),
        random_state=topic_cfg.random_state,
        max_iter=200,
        init="nndsvda",
    )
    W = nmf.fit_transform(X)
    row_sums = W.sum(axis=1, keepdims=True) + 1e-9
    return (W / row_sums).astype(np.float32)


def user_review_embeddings_train(
    train_reviews: pd.DataFrame,
    business_ids_ordered: list[str],
    business_emb: np.ndarray,
    maps_business_to_idx: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Average pooled business embeddings for businesses the user positively reviewed in train."""
    bid_to_idx = maps_business_to_idx
    user_ids = sorted(train_reviews["user_id"].unique())
    u_to_i = {u: i for i, u in enumerate(user_ids)}
    dim = business_emb.shape[1]
    acc = np.zeros((len(user_ids), dim), dtype=np.float64)
    cnt = np.zeros(len(user_ids), dtype=np.int32)
    pos = train_reviews["stars"] >= 4
    tr = train_reviews[pos]
    for _, row in tr.iterrows():
        uid = row["user_id"]
        bid = row["business_id"]
        if bid not in bid_to_idx:
            continue
        bi = bid_to_idx[bid]
        ui = u_to_i[uid]
        acc[ui] += business_emb[bi]
        cnt[ui] += 1
    cnt_safe = np.maximum(cnt, 1)
    out = (acc / cnt_safe[:, None]).astype(np.float32)
    return out, user_ids


def persist_embeddings(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_embeddings(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def build_all_content_features(
    cfg: ProjectConfig,
    train_reviews: pd.DataFrame,
    maps,
) -> dict:
    """Compute transformer + topic features aligned to maps.business_id_to_idx order."""
    text_agg = aggregate_review_text_per_business(train_reviews)
    bids_ordered = sorted(maps.business_id_to_idx.keys())
    emb, pca = compute_sentence_embeddings(bids_ordered, text_agg, cfg.embedding)
    topics = compute_topic_features(bids_ordered, text_agg, cfg.topics)
    user_hist_emb, user_ids_order = user_review_embeddings_train(
        train_reviews, bids_ordered, emb, maps.business_id_to_idx
    )
    return {
        "business_emb": emb,
        "business_topics": topics,
        "pca": pca,
        "user_hist_emb": user_hist_emb,
        "user_ids_order": user_ids_order,
        "bids_ordered": bids_ordered,
    }

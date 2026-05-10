from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Paths:
    data_dir: Path = Path("data/yelp")
    artifacts_dir: Path = Path("artifacts")


@dataclass
class DataConfig:
    """Paths to Yelp JSONL files (from https://www.yelp.com/dataset)."""

    review_json: str = "yelp_academic_dataset_review.json"
    business_json: str = "yelp_academic_dataset_business.json"
    user_json: str = "yelp_academic_dataset_user.json"
    restaurant_keywords: tuple[str, ...] = ("Restaurants", "Food")
    min_reviews_per_business: int = 5
    min_reviews_per_user: int = 3
    implicit_positive_min_stars: int = 4  # binary implicit positive


@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.08
    iterations: int = 20
    alpha: float = 40.0  # confidence weight for implicit feedback
    random_state: int = 42
    candidate_k: int = 200


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_chars_per_doc: int = 8000
    emb_dim: int = 384  # MiniLM output size
    pca_dim: int = 32  # compress for LTR + faster ablations


@dataclass
class TopicConfig:
    n_topics: int = 20
    max_features_tfidf: int = 5000
    random_state: int = 42


@dataclass
class LTRConfig:
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    max_rounds: int = 400
    early_stopping_rounds: int = 40
    ndcg_eval_at: tuple[int, ...] = (10,)
    label_strategy: Literal["binary", "stars_residual"] = "binary"


@dataclass
class SplitConfig:
    """Time-based split on review date."""

    test_fraction_time: float = 0.15  # last fraction by time


@dataclass
class ProjectConfig:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
    ltr: LTRConfig = field(default_factory=LTRConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

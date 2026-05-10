from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from yelp_hybrid.config import ProjectConfig
from yelp_hybrid.io_utils import iter_json_lines


def _is_restaurant(categories: str | None, keywords: tuple[str, ...]) -> bool:
    if not categories:
        return False
    return any(k in categories for k in keywords)


def load_businesses(cfg: ProjectConfig) -> pd.DataFrame:
    p = cfg.paths.data_dir / cfg.data.business_json
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Download the Yelp Open Dataset and place JSON files in {cfg.paths.data_dir}."
        )
    rows = []
    for b in iter_json_lines(p):
        cats = b.get("categories") or ""
        if not _is_restaurant(cats, cfg.data.restaurant_keywords):
            continue
        rows.append(
            {
                "business_id": b["business_id"],
                "stars": float(b.get("stars") or 0),
                "review_count": int(b.get("review_count") or 0),
                "is_open": int(b.get("is_open") if b.get("is_open") is not None else -1),
                "latitude": float(b["latitude"]) if b.get("latitude") is not None else np.nan,
                "longitude": float(b["longitude"]) if b.get("longitude") is not None else np.nan,
                "attributes_PriceRange": _extract_price(b),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _extract_price(b: dict) -> float:
    attrs = b.get("attributes") or {}
    pr = attrs.get("RestaurantsPriceRange2") or attrs.get("Price Range")
    if pr is None:
        return np.nan
    try:
        return float(pr)
    except (TypeError, ValueError):
        return np.nan


def load_users(cfg: ProjectConfig) -> pd.DataFrame:
    p = cfg.paths.data_dir / cfg.data.user_json
    if not p.exists():
        # Optional in some workflows
        return pd.DataFrame()
    rows = []
    for u in iter_json_lines(p):
        rows.append(
            {
                "user_id": u["user_id"],
                "user_review_count": int(u.get("review_count") or 0),
                "user_yelping_since": u.get("yelping_since"),
            }
        )
    return pd.DataFrame(rows)


def load_reviews(cfg: ProjectConfig, allowed_business_ids: set[str]) -> pd.DataFrame:
    p = cfg.paths.data_dir / cfg.data.review_json
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}.")
    rows = []
    for r in iter_json_lines(p):
        bid = r["business_id"]
        if bid not in allowed_business_ids:
            continue
        rows.append(
            {
                "review_id": r["review_id"],
                "user_id": r["user_id"],
                "business_id": bid,
                "stars": float(r["stars"]),
                "date": pd.to_datetime(r["date"]),
                "text": (r.get("text") or "")[:20000],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class YelpTables:
    businesses: pd.DataFrame
    reviews: pd.DataFrame
    users: pd.DataFrame


def build_tables(cfg: ProjectConfig) -> YelpTables:
    businesses = load_businesses(cfg)
    bid_counts = businesses["business_id"].value_counts()
    businesses = businesses[businesses["business_id"].isin(bid_counts[bid_counts >= 1].index)]

    reviews = load_reviews(cfg, set(businesses["business_id"]))
    rc = reviews.groupby("business_id").size()
    allowed_bids = rc[rc >= cfg.data.min_reviews_per_business].index
    businesses = businesses[businesses["business_id"].isin(allowed_bids)].reset_index(drop=True)
    reviews = reviews[reviews["business_id"].isin(allowed_bids)]

    uc = reviews.groupby("user_id").size()
    allowed_users = uc[uc >= cfg.data.min_reviews_per_user].index
    reviews = reviews[reviews["user_id"].isin(allowed_users)]

    users = load_users(cfg)
    if not users.empty:
        users = users[users["user_id"].isin(set(reviews["user_id"]))]

    return YelpTables(businesses=businesses, reviews=reviews, users=users)


def time_based_split(
    reviews: pd.DataFrame, test_fraction_time: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = reviews.sort_values("date")
    cutoff_idx = int(len(r) * (1.0 - test_fraction_time))
    train = r.iloc[:cutoff_idx].copy()
    test = r.iloc[cutoff_idx:].copy()
    return train, test


def persist_tables(path: Path, tables: YelpTables) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(tables, f)


def load_tables(path: Path) -> YelpTables:
    with path.open("rb") as f:
        return pickle.load(f)

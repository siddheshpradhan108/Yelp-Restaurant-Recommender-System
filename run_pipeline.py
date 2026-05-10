#!/usr/bin/env python3
"""End-to-end runner: Yelp hybrid recommender (ALS + LightGBM) with ablations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from yelp_hybrid.als_model import (
    build_implicit_matrix,
    build_index_maps,
    train_als,
)
from yelp_hybrid.config import ProjectConfig
from yelp_hybrid.data_pipeline import (
    build_tables,
    load_tables,
    persist_tables,
    time_based_split,
)
from yelp_hybrid.embeddings import build_all_content_features, load_embeddings, persist_embeddings
from yelp_hybrid.evaluation import evaluate_users
from yelp_hybrid.ltr import (
    FeatureControl,
    build_ltr_train_matrix,
    split_train_val_groups,
    train_ranker,
    user_train_stats,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cmd_prepare(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    _ensure_dir(cfg.paths.artifacts_dir)
    tables = build_tables(cfg)
    out = cfg.paths.artifacts_dir / "tables.pkl"
    persist_tables(out, tables)
    print(f"Saved processed tables to {out}")
    print(
        f"Businesses: {len(tables.businesses)} reviews: {len(tables.reviews)} "
        f"(after filters)"
    )


def cmd_train(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    art = cfg.paths.artifacts_dir
    _ensure_dir(art)

    tables = load_tables(art / "tables.pkl")
    train_r, test_r = time_based_split(tables.reviews, cfg.split.test_fraction_time)
    if args.max_train_reviews:
        train_r = train_r.sample(n=min(args.max_train_reviews, len(train_r)), random_state=42)

    maps = build_index_maps(train_r)
    ui = build_implicit_matrix(train_r, maps, cfg.data.implicit_positive_min_stars, cfg.als)
    item_user = ui.T.tocsr()
    als_model = train_als(item_user, cfg.als)

    with (art / "als_model.pkl").open("wb") as f:
        pickle.dump({"model": als_model, "maps": maps}, f)

    content_path = art / "content_features.pkl"
    if args.reuse_embeddings and content_path.exists():
        content = load_embeddings(content_path)
        print(f"Loaded cached content features from {content_path}")
    else:
        content = build_all_content_features(cfg, train_r, maps)
        persist_embeddings(content_path, content)
        print(f"Saved content features to {content_path}")

    user_stats_df = user_train_stats(train_r, maps)

    ablations = {
        "collab_context_baseline": FeatureControl(
            use_collaborative=True,
            use_contextual=True,
            use_topic_features=False,
            use_transformer_embeddings=False,
        ),
        "collab_context_topics": FeatureControl(
            use_collaborative=True,
            use_contextual=True,
            use_topic_features=True,
            use_transformer_embeddings=False,
        ),
        "full_hybrid": FeatureControl(
            use_collaborative=True,
            use_contextual=True,
            use_topic_features=True,
            use_transformer_embeddings=True,
        ),
    }

    results: dict = {}

    X_full, y_full, group_full, names_full = build_ltr_train_matrix(
        cfg,
        maps,
        train_r,
        tables.businesses,
        ui,
        als_model,
        content,
        user_stats_df,
        ablations["full_hybrid"],
        negatives_per_user=args.negatives_per_user,
        max_train_users=args.max_ltr_users,
    )

    X_tr, y_tr, g_tr, X_va, y_va, g_va = split_train_val_groups(
        X_full, y_full, group_full, val_fraction=0.1, seed=42
    )

    booster_full = train_ranker(X_tr, y_tr, g_tr, X_va, y_va, g_va, names_full, cfg)
    with (art / "ltr_full.pkl").open("wb") as f:
        pickle.dump(booster_full, f)

    rep_full = evaluate_users(
        cfg,
        maps,
        train_r,
        test_r,
        tables.businesses,
        ui,
        als_model,
        content,
        user_stats_df,
        ablations["full_hybrid"],
        booster_full,
        max_test_users=args.max_test_users,
    )

    results["full_hybrid"] = {
        "ndcg10_all": rep_full.ndcg10_all,
        "ndcg10_sparse_lt5": rep_full.ndcg10_sparse,
        "ndcg10_dense_ge5": rep_full.ndcg10_dense,
        "n_users": rep_full.n_users_all,
        "n_sparse": rep_full.n_users_sparse,
        "n_dense": rep_full.n_users_dense,
    }

    for name, ctrl in list(ablations.items())[:-1]:
        X, y, grp, feat_names = build_ltr_train_matrix(
            cfg,
            maps,
            train_r,
            tables.businesses,
            ui,
            als_model,
            content,
            user_stats_df,
            ctrl,
            negatives_per_user=args.negatives_per_user,
            max_train_users=args.max_ltr_users,
        )
        X_tr, y_tr, g_tr, X_va, y_va, g_va = split_train_val_groups(X, y, grp, val_fraction=0.1, seed=42)
        booster = train_ranker(X_tr, y_tr, g_tr, X_va, y_va, g_va, feat_names, cfg)
        rep = evaluate_users(
            cfg,
            maps,
            train_r,
            test_r,
            tables.businesses,
            ui,
            als_model,
            content,
            user_stats_df,
            ctrl,
            booster,
            max_test_users=args.max_test_users,
        )
        results[name] = {
            "ndcg10_all": rep.ndcg10_all,
            "ndcg10_sparse_lt5": rep.ndcg10_sparse,
            "ndcg10_dense_ge5": rep.ndcg10_dense,
            "n_users": rep.n_users_all,
            "n_sparse": rep.n_users_sparse,
            "n_dense": rep.n_users_dense,
        }

    baseline = results["collab_context_baseline"]
    full = results["full_hybrid"]
    delta_sparse = full["ndcg10_sparse_lt5"] - baseline["ndcg10_sparse_lt5"]
    delta_dense = full["ndcg10_dense_ge5"] - baseline["ndcg10_dense_ge5"]

    results["deltas_vs_collab_context_baseline"] = {
        "added_topics_plus_transformers_sparse_lt5": delta_sparse,
        "added_topics_plus_transformers_dense_ge5": delta_dense,
    }

    topics_only = results["collab_context_topics"]
    results["deltas_topics_only_vs_baseline"] = {
        "sparse_lt5": topics_only["ndcg10_sparse_lt5"] - baseline["ndcg10_sparse_lt5"],
        "dense_ge5": topics_only["ndcg10_dense_ge5"] - baseline["ndcg10_dense_ge5"],
    }

    results["deltas_transformer_plus_topics_vs_topics_only"] = {
        "sparse_lt5": full["ndcg10_sparse_lt5"] - topics_only["ndcg10_sparse_lt5"],
        "dense_ge5": full["ndcg10_dense_ge5"] - topics_only["ndcg10_dense_ge5"],
    }

    out_json = art / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote metrics to {out_json}")
    print(
        "\nInterpretation: transformer+NMF topic features target cold-start / sparse settings; "
        "large lifts on sparse users and smaller lifts on dense users are typical."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Yelp hybrid ALS + LightGBM pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("data/yelp"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare-data", help="Load Yelp JSON and persist filtered tables")
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser("train", help="Train ALS + content features + LightGBM ablations")
    st.add_argument("--max-train-reviews", type=int, default=None)
    st.add_argument("--max-ltr-users", type=int, default=8000)
    st.add_argument("--max-test-users", type=int, default=1500)
    st.add_argument("--negatives-per-user", type=int, default=35)
    st.add_argument("--reuse-embeddings", action="store_true")
    st.set_defaults(func=cmd_train)

    args = p.parse_args()
    cfg = ProjectConfig()
    cfg.paths.data_dir = args.data_dir
    cfg.paths.artifacts_dir = args.artifacts_dir
    args.func(cfg, args)


if __name__ == "__main__":
    main()

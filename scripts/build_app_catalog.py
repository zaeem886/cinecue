"""Build the website's large catalog model from MovieLens 32M.

The notebooks and the report use MovieLens 1M. The website needs newer and more
movies, so this script fits the same pipeline (EASE + MF blend, 20-rating floor,
popularity re-ranking) on a laptop-sized slice of MovieLens 32M:

1. Keep the most-rated movies (--movies) and a seeded sample of users (--users)
   with at least 20 ratings among them.
2. Hold out each sampled user's last --holdout ratings (by time) and tune the EASE
   penalty and blend weight on a subsample of users for NDCG@10. This is a sanity
   check that the settings transfer, not a replacement for the notebook evaluation.
3. Refit on all kept ratings and save artifacts/recommender-32m.pkl, which the
   website loads in preference to the MovieLens 1M model.

Usage:
    python scripts/build_app_catalog.py --archive data/raw/ml-32m/ml-32m.zip
The archive is the official https://files.grouplens.org/datasets/movielens/ml-32m.zip.
Nothing derived from the ratings is committed except the aggregate summary JSON.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.artifacts import save_bundle  # noqa: E402
from cinecue.models import EASE, MatrixFactorization, rmse  # noqa: E402
from cinecue.ranking import accuracy_metrics, blend, top_k  # noqa: E402

MIN_SUPPORT, K, LIKED = 20, 10, 4.0


def read_archive(path):
    with ZipFile(path) as archive:
        prefix = next(name for name in archive.namelist() if name.endswith("ratings.csv")).rsplit("/", 1)[0]
        with archive.open(f"{prefix}/ratings.csv") as source:
            ratings = pd.read_csv(source, dtype={"userId": "int32", "movieId": "int32", "rating": "float32",
                                                 "timestamp": "int64"})
        with archive.open(f"{prefix}/movies.csv") as source:
            movies = pd.read_csv(source)
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]
    movies = movies.rename(columns={"movieId": "movie_id"})
    return ratings, movies


def select(ratings, n_movies, n_users, holdout, seed):
    counts = ratings.movie_id.value_counts()
    keep_movies = counts.index[:n_movies]
    kept = ratings.loc[ratings.movie_id.isin(keep_movies)]
    per_user = kept.user_id.value_counts()
    eligible = per_user.index[per_user >= MIN_SUPPORT + holdout].to_numpy()
    rng = np.random.default_rng(seed)
    users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    return kept.loc[kept.user_id.isin(users)].reset_index(drop=True)


def split_last(ratings, holdout):
    """Each user's last `holdout` ratings (by time, ties by movie id) become the evaluation set."""
    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="stable")
    rank_from_end = ordered.groupby("user_id").cumcount(ascending=False)
    return ordered.loc[rank_from_end >= holdout], ordered.loc[rank_from_end < holdout]


def history_rows(frame, users, model):
    rows = pd.Series(np.arange(len(users)), index=users)
    part = frame.loc[frame.user_id.isin(users)]
    return rows[part.user_id].to_numpy(), model.movies.encode(part.movie_id.to_numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/ml-32m/ml-32m.zip")
    parser.add_argument("--movies", type=int, default=9000)
    parser.add_argument("--users", type=int, default=30000)
    parser.add_argument("--holdout", type=int, default=10)
    parser.add_argument("--tune-users", type=int, default=4000)
    parser.add_argument("--ease-regs", type=int, nargs="+", default=[300, 1000, 3000, 10000, 30000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/recommender-32m.pkl")
    args = parser.parse_args()
    clock = time.perf_counter()
    log = lambda message: print(f"[{time.perf_counter() - clock:6.0f}s] {message}", flush=True)  # noqa: E731

    ratings, movies = read_archive(args.archive)
    log(f"Read {len(ratings):,} ratings and {len(movies):,} movies")
    data = select(ratings, args.movies, args.users, args.holdout, args.seed)
    del ratings
    fit, held = split_last(data, args.holdout)
    log(f"Kept {data.movie_id.nunique():,} movies, {data.user_id.nunique():,} users, {len(data):,} ratings "
        f"({len(held):,} held out)")

    mf_settings = dict(factors=64, reg=0.08, n_iter=10, seed=0)  # notebook 5 selection on MovieLens 1M
    mf = MatrixFactorization(**mf_settings).fit(fit.user_id, fit.movie_id, fit.rating)
    holdout_rmse = rmse(held.rating, mf.predict(held.user_id, held.movie_id))
    log(f"MF fitted; held-out RMSE {holdout_rmse:.4f}")

    # Tuning set: a seeded subsample of users with at least one liked held-out movie.
    liked = held.loc[held.rating >= LIKED].assign(position=lambda d: mf.movies.encode(d.movie_id))
    liked = liked.loc[liked.position >= 0]
    relevant = liked.groupby("user_id").position.apply(set)
    rng = np.random.default_rng(args.seed + 1)
    tune_users = np.sort(rng.choice(relevant.index.to_numpy(), size=min(args.tune_users, len(relevant)),
                                    replace=False))
    relevant = list(relevant.loc[tune_users])
    counts_fit = fit.movie_id.value_counts().reindex(mf.movies.ids).fillna(0).to_numpy()
    mf_scores = mf.user_vectors[mf.users.encode(tune_users)] @ mf.item_aug.T + mf.mean + mf.item_bias

    grid, best = [], None
    for reg in args.ease_regs:
        ease = EASE(reg=reg).fit(fit.user_id, fit.movie_id, dtype=np.float32)
        assert ease.movies.ids.equals(mf.movies.ids)
        rows, positions = history_rows(fit, tune_users, ease)
        X = ease.history_matrix(rows, positions, len(tune_users))
        exclude = X.astype(bool) | (counts_fit < MIN_SUPPORT)[None, :]
        ease_scores = ease.score(X)
        for alpha in [0.0, 0.5, 0.6, 0.75, 0.9]:
            ndcg = float(accuracy_metrics(top_k(blend(ease_scores, mf_scores, alpha), exclude, K), relevant).ndcg.mean())
            grid.append({"ease_reg": reg, "alpha": alpha, "ndcg": ndcg})
            if best is None or ndcg > best["ndcg"]:
                best = grid[-1]
        log(f"EASE reg={reg}: best NDCG@10 so far {best['ndcg']:.4f} (reg={best['ease_reg']}, alpha={best['alpha']})")
    popular = float(accuracy_metrics(top_k(np.tile(counts_fit, (len(tune_users), 1)), exclude, K), relevant).ndcg.mean())
    log(f"Popularity NDCG@10 {popular:.4f}; selected reg={best['ease_reg']}, alpha={best['alpha']}")

    # Final model on every kept rating, as the website serves it.
    final_mf = MatrixFactorization(**mf_settings).fit(data.user_id, data.movie_id, data.rating)
    final_ease = EASE(reg=best["ease_reg"]).fit(data.user_id, data.movie_id, dtype=np.float32)
    assert final_ease.movies.ids.equals(final_mf.movies.ids)
    catalog = (movies.set_index("movie_id").loc[final_mf.movies.ids, ["title", "genres"]]
               .rename_axis("movie_id"))
    catalog["genres"] = catalog.genres.str.replace("(no genres listed)", "", regex=False)
    counts = data.movie_id.value_counts().reindex(final_mf.movies.ids).fillna(0).to_numpy()
    history = data[["user_id", "movie_id", "rating"]].reset_index(drop=True)
    years = pd.to_numeric(catalog.title.str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce")
    settings = {"ease_reg": best["ease_reg"], "alpha": best["alpha"], "rerank_weight": 1.5,
                "dataset": "MovieLens 32M", "movies": int(len(catalog)), "users": int(history.user_id.nunique()),
                "ratings": int(len(history)), "first_year": int(years.min()), "last_year": int(years.max())}
    save_bundle(args.out, (history, catalog, final_mf, final_ease, counts, settings))
    log(f"Saved {args.out} ({args.out.stat().st_size / 2**20:.0f} MB)")

    summary = {"source": "MovieLens 32M (ml-32m.zip, GroupLens)", "selection": vars(args) | {
                   "archive": args.archive.name, "out": args.out.name},
               "catalog": settings, "holdout": {"ratings_per_user": args.holdout, "tuning_users": int(len(tune_users)),
                                                "mf_rmse": holdout_rmse, "popularity_ndcg": popular,
                                                "selected": best, "grid": grid}}
    out = ROOT / "results/app_catalog_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Train and evaluate the weekday/time-of-day context re-ranker on MovieLens 1M.

Question: when a user comes back on a given weekday (and time of day), do the
movies they liked on that same weekday, or the genres they lean to on that weekday,
predict what they will like that day better than the final ranker alone?

Protocol (same splits and settings as notebooks 5-6):
* An *event* is one user-day (UTC) in a period with at least one liked (4-5 star)
  rating of an unseen movie that has 20+ ratings. Its context is that day's weekday
  and the daypart of most of its liked ratings. Users must be known at the cutoff.
* History is everything the user rated before the period. Candidates are the final
  ranker's top 100 for that history; relevant movies are that day's liked ones.
* Validation (models fitted on train): fit the re-ranker's feature weights by
  logistic regression and pick a feature set by 5-fold cross-validation over users.
* Test (models refitted on train + validation): apply the chosen weights unchanged
  and compare NDCG@10 per user against the final ranker, with a paired bootstrap.

Writes results/context_summary.json, which the website reads for the weights.
Usage: python scripts/evaluate_context.py
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.context import (DAYPARTS, FEATURES, WEEKDAYS, context_features,  # noqa: E402
                             fit_logistic, genre_matrix, global_weekday_lift)
from cinecue.data import load_movies, read_interactions, table_path  # noqa: E402
from cinecue.models import EASE, MatrixFactorization  # noqa: E402
from cinecue.ranking import paired_bootstrap  # noqa: E402

K, DEPTH, LIKED, MIN_SUPPORT, FOLDS, SEED = 10, 100, 4, 20, 5, 0
VARIANTS = {
    "final ranker": ["blend"],
    "+ same weekday": ["blend", "same_weekday"],
    "+ same daypart": ["blend", "same_daypart"],
    "+ weekday genres (user)": ["blend", "weekday_genre"],
    "+ weekday genres (everyone)": ["blend", "global_genre"],
    "all context": list(FEATURES),
}
DISCOUNTS = 1.0 / np.log2(np.arange(2, K + 2))


def log(message, clock=time.perf_counter()):
    print(f"[{time.perf_counter() - clock:6.0f}s] {message}", flush=True)


def fit_models(frame, catalog, mf_settings, ease_reg):
    mf = MatrixFactorization(**mf_settings).fit(frame.user_id, frame.movie_id, frame.rating)
    ease = EASE(reg=ease_reg).fit(frame.user_id, frame.movie_id)
    assert ease.movies.ids.equals(mf.movies.ids)
    G, vocabulary = genre_matrix(catalog.loc[mf.movies.ids, "genres"])
    liked = frame.loc[frame.rating >= LIKED]
    lift = global_weekday_lift(liked.weekday_utc, G[mf.movies.encode(liked.movie_id)])
    counts = frame.movie_id.value_counts().reindex(mf.movies.ids, fill_value=0).to_numpy()
    return dict(mf=mf, ease=ease, G=G, vocabulary=vocabulary, lift=lift, low=counts < MIN_SUPPORT)


def find_events(period, history, models):
    known = set(history.user_id)
    liked = period.loc[(period.rating >= LIKED) & period.user_id.isin(known)].copy()
    liked["position"] = models["mf"].movies.encode(liked.movie_id)
    liked = liked.loc[liked.position >= 0]
    liked = liked.loc[~models["low"][liked.position]]
    seen = set(zip(history.user_id, models["mf"].movies.encode(history.movie_id)))
    liked = liked.loc[[pair not in seen for pair in zip(liked.user_id, liked.position)]]
    liked["day"] = liked.rated_at.dt.floor("D")
    liked["part"] = liked.hour_utc // 6
    return [dict(user=user, weekday=int(group.weekday_utc.iloc[0]), part=int(group.part.mode().iloc[0]),
                 relevant=set(group.position)) for (user, _), group in liked.groupby(["user_id", "day"])]


def candidate_table(events, history, models, alpha):
    """Features and labels of each event's top-DEPTH candidates (by the final ranker)."""
    mf, ease = models["mf"], models["ease"]
    by_user = {u: g for u, g in history.groupby("user_id")}
    rows = []
    for number, event in enumerate(events):
        h = by_user[event["user"]]
        positions = mf.movies.encode(h.movie_id.to_numpy())
        features = context_features(ease, mf, alpha, positions, h.rating.to_numpy(float),
                                    h.weekday_utc.to_numpy(), (h.hour_utc // 6).to_numpy(),
                                    event["weekday"], event["part"], models["G"], models["lift"])
        base = features[:, 0].copy()
        base[positions] = -np.inf
        base[models["low"]] = -np.inf
        candidates = np.argpartition(-base, DEPTH - 1)[:DEPTH]
        candidates = candidates[np.argsort(-base[candidates], kind="stable")]
        rows.append(dict(event=number, user=event["user"], X=features[candidates],
                         y=np.array([c in event["relevant"] for c in candidates], float),
                         n_relevant=len(event["relevant"]),
                         has_same_weekday=bool((h.weekday_utc == event["weekday"]).any())))
    return rows


def ndcg(table, weights, columns):
    """NDCG@10 per event when candidates are ordered by X[:, columns] @ weights."""
    index = [FEATURES.index(c) for c in columns]
    values = []
    for row in table:
        order = np.argsort(-(row["X"][:, index] @ weights), kind="stable")[:K]
        ideal = DISCOUNTS[:min(row["n_relevant"], K)].sum()
        values.append(float((row["y"][order] * DISCOUNTS).sum() / ideal))
    return np.array(values)


def fit_weights(table, columns):
    if columns == ["blend"]:
        return np.array([1.0])
    index = [FEATURES.index(c) for c in columns]
    X = np.vstack([row["X"][:, index] for row in table])
    y = np.concatenate([row["y"] for row in table])
    weights, _ = fit_logistic(X, y)
    return weights


def per_user(table, values):
    return pd.Series(values).groupby([row["user"] for row in table]).mean()


def main():
    processed = next(p for p in [ROOT / "data/processed", ROOT / "results"]
                     if table_path(p, "train.csv").is_file())
    splits = {name: read_interactions(table_path(processed, f"{name}.csv"))
              for name in ("train", "validation", "test")}
    catalog = load_movies(processed).set_index("movie_id")
    mf_settings = json.loads((processed / "rating_summary.json").read_text())["mf"]
    selected = json.loads((processed / "ranking_summary.json").read_text())["selected"]
    alpha, ease_reg = float(selected["alpha"]), float(selected["ease_reg"])
    log(f"Loaded splits from {processed.relative_to(ROOT)}; alpha={alpha}, EASE reg={ease_reg:g}")

    # ---- Validation: fit and select on train -> validation.
    train = splits["train"]
    models = fit_models(train, catalog, mf_settings, ease_reg)
    val_events = find_events(splits["validation"], train, models)
    val = candidate_table(val_events, train, models, alpha)
    log(f"Validation: {len(val):,} user-day events from {len({r['user'] for r in val}):,} users")

    users = np.array(sorted({row["user"] for row in val}))
    fold_of = dict(zip(users, np.random.default_rng(SEED).permutation(len(users)) % FOLDS))
    cv, cv_users = {}, {}
    for name, columns in VARIANTS.items():
        scores = np.zeros(len(val))
        for fold in range(FOLDS):
            train_rows = [r for r in val if fold_of[r["user"]] != fold]
            held = [i for i, r in enumerate(val) if fold_of[r["user"]] == fold]
            weights = fit_weights(train_rows, columns)
            scores[held] = ndcg([val[i] for i in held], weights, columns)
        cv_users[name] = per_user(val, scores)
        cv[name] = float(cv_users[name].mean())
        log(f"  CV NDCG@10 {name:<28} {cv[name]:.4f}")
    # A context variant must beat the final ranker on validation users with a 95% interval
    # above zero; otherwise the final ranker stays. This keeps noise from being "selected".
    cv_gain = {name: paired_bootstrap(cv_users[name].to_numpy(), cv_users["final ranker"].to_numpy(), seed=SEED)
               for name in VARIANTS if name != "final ranker"}
    reliable = {name: cv[name] for name, gain in cv_gain.items() if gain["ci_low"] > 0}
    best = max(reliable, key=reliable.get) if reliable else "final ranker"
    columns = VARIANTS[best]
    weights = fit_weights(val, columns)
    full_weights = dict(zip(columns, map(float, weights)))
    log(f"Selected on validation: {best} {full_weights}")

    # ---- Test: refit on train + validation, apply the validation weights unchanged.
    history = pd.concat([train, splits["validation"]], ignore_index=True)
    models = fit_models(history, catalog, mf_settings, ease_reg)
    test_events = find_events(splits["test"], history, models)
    test = candidate_table(test_events, history, models, alpha)
    log(f"Test: {len(test):,} user-day events from {len({r['user'] for r in test}):,} users")

    base = per_user(test, ndcg(test, np.array([1.0]), ["blend"]))
    results = []
    for name, cols in VARIANTS.items():
        w = fit_weights(val, cols)
        ours = per_user(test, ndcg(test, w, cols))
        row = {"variant": name, "validation_cv_ndcg": cv[name], "test_ndcg": float(ours.mean()),
               "weights": dict(zip(cols, map(float, w)))}
        if name in cv_gain:
            row |= {f"cv_vs_final_{k}": v for k, v in cv_gain[name].items()}
        if name != "final ranker":
            row |= {f"vs_final_{k}": v for k, v in paired_bootstrap(ours.to_numpy(), base.to_numpy(),
                                                                    seed=SEED).items()}
        results.append(row)
        log(f"  test NDCG@10 {name:<28} {row['test_ndcg']:.4f}"
            + (f"  diff {row['vs_final_difference']:+.4f} [{row['vs_final_ci_low']:+.4f}, "
               f"{row['vs_final_ci_high']:+.4f}]" if "vs_final_difference" in row else ""))

    # Events where the user has at least one earlier rating on the same weekday.
    same = [r["has_same_weekday"] for r in test]
    chosen = next(r for r in results if r["variant"] == best)
    subset = [i for i, s in enumerate(same) if s]
    sub_table = [test[i] for i in subset]
    sub_gain = paired_bootstrap(per_user(sub_table, ndcg(sub_table, weights, columns)).to_numpy(),
                                per_user(sub_table, ndcg(sub_table, np.array([1.0]), ["blend"])).to_numpy(),
                                seed=SEED) if best != "final ranker" else None

    names = models["vocabulary"]
    summary = {
        "question": "Do same-weekday likes and weekday genre leanings improve top-10 ranking for that day?",
        "protocol": {"event": "user-day (UTC) with a liked unseen movie", "k": K, "candidates": DEPTH,
                     "relevance": f"rating >= {LIKED}", "min_support": MIN_SUPPORT, "cv_folds": FOLDS,
                     "dayparts": {name: f"{6 * i:02d}-{6 * i + 6:02d}h UTC" for i, name in enumerate(DAYPARTS)},
                     "selection": ("5-fold CV over validation users; a context variant is chosen only if its paired "
                                   "bootstrap gain over the final ranker has a 95% interval above zero; "
                                   "test is used once, for reporting")},
        "events": {"validation": len(val), "validation_users": len({r["user"] for r in val}),
                   "test": len(test), "test_users": len({r["user"] for r in test}),
                   "test_share_with_same_weekday_history": float(np.mean(same))},
        "variants": results,
        "selected": {"variant": best, "depth": DEPTH,
                     "weights": {name: full_weights.get(name, 0.0) for name in FEATURES},
                     "test_ndcg": chosen["test_ndcg"],
                     "test_vs_final": {k.removeprefix("vs_final_"): v for k, v in chosen.items()
                                       if k.startswith("vs_final_")} or None,
                     "test_vs_final_same_weekday_history": sub_gain},
        "global_weekday_genre_lift": {WEEKDAYS[d]: {g: float(models["lift"][d, k]) for k, g in enumerate(names)}
                                      for d in range(7)},
        "caveats": ["MovieLens timestamps are rating-submission times in UTC, not local viewing times.",
                    "Many MovieLens users rate in bursts, so weekday history is often one or two days."],
    }
    out = ROOT / "results/context_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

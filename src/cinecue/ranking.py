"""Top-N ranking, accuracy and beyond-accuracy metrics, and popularity-aware re-ranking.

Rankings are arrays of movie positions (columns of a user-by-movie score matrix).
Movies a user already rated are excluded before ranking.
"""

import numpy as np
import pandas as pd


def standardize(scores):
    """Per-row z-scores, so models on different scales can be blended."""
    return (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True)


def blend(ease_scores, mf_scores, alpha):
    """(1 - alpha) * z(EASE) + alpha * z(MF): what users rate, weighed by how they rate it."""
    return (1.0 - alpha) * standardize(ease_scores) + alpha * standardize(mf_scores)


def top_k(scores, exclude, k):
    """Positions of the k highest scores per row, best first, skipping excluded cells."""
    masked = np.where(exclude, -np.inf, scores)
    part = np.argpartition(-masked, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(masked, part, axis=1), axis=1, kind="stable")
    ranked = np.take_along_axis(part, order, axis=1)
    assert np.isfinite(np.take_along_axis(masked, ranked, axis=1)).all(), "Too few candidates."
    return ranked


def rerank_by_popularity(scores, exclude, popularity, weight, k, depth=100):
    """Re-rank each row's `depth` best candidates by score - weight * popularity.

    `popularity` is scaled to [0, 1]. Re-ranking only reorders the model's own top
    candidates, so a large weight cannot surface movies the model scores poorly.
    weight=0 returns the plain top-k.
    """
    candidates = top_k(scores, exclude, depth)
    adjusted = np.take_along_axis(scores, candidates, axis=1) - weight * popularity[candidates]
    order = np.argsort(-adjusted, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(candidates, order, axis=1)


def popularity_scale(counts):
    """log(1 + count) scaled to [0, 1], so penalties act on orders of magnitude."""
    logged = np.log1p(np.asarray(counts, dtype=float))
    return logged / logged.max()


def accuracy_metrics(ranked, relevant):
    """Per-user precision, recall, NDCG and hit rate at k with binary relevance.

    `relevant` holds one set of relevant positions per row of `ranked`.
    """
    k = ranked.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for row, targets in zip(ranked, relevant):
        hits = np.fromiter((position in targets for position in row), dtype=bool, count=k)
        ideal = discounts[:min(len(targets), k)].sum()
        rows.append({"precision": hits.mean(), "recall": hits.sum() / len(targets),
                     "ndcg": (hits * discounts).sum() / ideal, "hit": float(hits.any())})
    return pd.DataFrame(rows)


def beyond_accuracy(ranked, popularity_counts, head, genres, catalog_size):
    """Catalog coverage, long-tail share, mean popularity, and intra-list genre diversity.

    head: boolean flag per position for the most-rated movies. genres: 0/1 matrix with
    one row per position. Diversity is the mean pairwise genre Jaccard distance.
    """
    flat = ranked.ravel()
    G = genres[ranked].astype(float)  # users x k x genres
    overlap = np.einsum("ukg,ulg->ukl", G, G)
    sizes = G.sum(axis=2)
    union = sizes[:, :, None] + sizes[:, None, :] - overlap
    distance = 1.0 - np.divide(overlap, union, out=np.ones_like(overlap), where=union > 0)
    k = ranked.shape[1]
    pairs = np.triu_indices(k, 1)
    return {"coverage": len(np.unique(flat)) / catalog_size,
            "long_tail_share": float(1.0 - head[flat].mean()),
            "mean_ratings_per_recommended_movie": float(np.mean(popularity_counts[flat])),
            "genre_diversity": float(distance[:, pairs[0], pairs[1]].mean())}

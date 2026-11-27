"""Exact, additive explanations for MatrixFactorization scores.

A user vector is a ridge solution, p = A^-1 X^T y, where row j of X is the vector
[q_j, 1] of a rated movie and y_j is that rating minus the movie's baseline. The
personalized part of the score for a target movie t is therefore

    [q_t, 1] . p = sum_j ([q_t, 1] A^-1 [q_j, 1]) * y_j,

one term per rated movie. The terms add up to the score exactly, so an explanation
built from the largest terms describes what the model actually computed.
"""

import numpy as np


def contributions(model, history_positions, history_ratings, target_positions):
    """Contribution of each rated movie (columns) to each target's score (rows)."""
    X, y, A = model.user_system(history_positions, history_ratings)
    influence = np.linalg.solve(A, X.T)  # (factors + 1) x history
    return (model.item_aug[target_positions] @ influence) * y


def personal_score(model, user_vector, target_positions):
    """Score minus the non-personalized part, mean + movie bias."""
    return model.item_aug[target_positions] @ user_vector


def blend_contributions(ease, mf, alpha, history_positions, history_ratings, target_positions,
                        history_weights=None):
    """Split a blended ranking score exactly, for one user.

    Both blend inputs are z-scored per user over all movies: z = (s - mean) / std. EASE
    scores are sums of w_j B[j, t], with w_j a recency weight (1 when not given); MF
    scores are mean + b_t + sum of MF terms. The blend is

        constant + alpha * b_t / std_mf + sum_j [(1 - alpha) w_j B[j, t] / std_ease
                                                 + alpha * mf_term[t, j] / std_mf]

    Returns (per-movie terms: targets x history, general-appeal term per target,
    constant). The three parts add up to `blend` for the target movies.
    """
    weights = np.ones(len(history_positions)) if history_weights is None else np.asarray(history_weights, float)
    history = ease.history_matrix(np.zeros(len(history_positions), int), history_positions, 1, weights)
    ease_scores = ease.score(history)[0]
    mf_vector = mf.fold_in(history_positions, history_ratings)
    mf_scores = mf.score_items(mf_vector)
    ease_scale, mf_scale = (1.0 - alpha) / ease_scores.std(), alpha / mf_scores.std()
    terms = (ease_scale * (weights[:, None] * ease.weights[np.ix_(history_positions, target_positions)]).T
             + mf_scale * contributions(mf, history_positions, history_ratings, target_positions))
    appeal = mf_scale * mf.item_bias[target_positions]
    constant = mf_scale * (mf.mean - mf_scores.mean()) - ease_scale * ease_scores.mean()
    return terms, appeal, constant

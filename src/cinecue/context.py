"""Weekday and time-of-day context for ranking.

A *like* carries the moment it happened: a weekday (Monday = 0) and a daypart
(night 0-6h, morning 6-12h, afternoon 12-18h, evening 18-24h). At ranking time the
current moment is the *context*. ContextRanker re-orders the final ranker's top
candidates with a linear score over five features of each candidate movie:

    blend           the final ranker's score (z-scored EASE + MF blend), unchanged
    same_weekday    pull from only the likes made on the current weekday: the positive part
                    of their z-scored EASE score, softly capped at 2.5 (see squash)
    same_daypart    the same for the likes made in the current daypart
    weekday_genre   how much this user's likes on the current weekday lean to the
                    candidate's genres, as a shrunk log-lift over the user's overall mix
    global_genre    the same lift for everyone's ratings on the current weekday

The weights are fitted by logistic regression on the validation period (scripts/
evaluate_context.py), so the context counts exactly as much as the data supports.
Explicit mood genres chosen in the website are a separate, user-controlled boost.

MovieLens timestamps record when a rating was submitted, in UTC. They are a proxy
for the moment someone engaged with a movie, not proof of when they watched it.
"""

import numpy as np

from .ranking import blend, standardize

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAYPARTS = ("night", "morning", "afternoon", "evening")
FEATURES = ("blend", "same_weekday", "same_daypart", "weekday_genre", "global_genre")
# MovieLens 1M and 32M spell one genre differently.
GENRE_ALIASES = {"Children's": "Children"}


def daypart(hour):
    """Six-hour block of the day: 0 night, 1 morning, 2 afternoon, 3 evening."""
    return int(hour) // 6


def describe(weekday, part):
    return f"{WEEKDAYS[int(weekday)]} {DAYPARTS[int(part)]}"


def genre_names(genres):
    return [GENRE_ALIASES.get(g, g) for g in str(genres).split("|") if g]


def genre_matrix(genre_strings, vocabulary=None):
    """0/1 movies x genres matrix from pipe-separated genre strings, plus its vocabulary."""
    lists = [genre_names(value) for value in genre_strings]
    vocabulary = sorted({g for names in lists for g in names}) if vocabulary is None else list(vocabulary)
    column = {g: k for k, g in enumerate(vocabulary)}
    G = np.zeros((len(lists), len(vocabulary)), dtype=np.float32)
    for row, names in enumerate(lists):
        for g in names:
            if g in column:
                G[row, column[g]] = 1.0
    return G, vocabulary


def log_lift(G_history, in_context, strength=5.0):
    """Per-genre log(share in context / overall share) for one history, shrunk toward 0.

    The overall share uses add-one smoothing; the in-context share is pulled toward it
    with `strength` pseudo-likes, so a single Monday like moves the lift only a little.
    """
    in_context = np.asarray(in_context, dtype=bool)
    overall = (G_history.sum(axis=0) + 1.0) / (G_history.sum() + G_history.shape[1])
    n = in_context.sum()
    if n == 0:
        return np.zeros(G_history.shape[1])
    share = G_history[in_context].sum(axis=0) / max(G_history[in_context].sum(), 1.0)
    shrunk = (n * share + strength * overall) / (n + strength)
    return np.log(shrunk / overall)


def global_weekday_lift(weekdays, G_rated, strength=50.0):
    """7 x genres table: log-lift of each genre among all ratings made on each weekday."""
    weekdays = np.asarray(weekdays)
    return np.vstack([log_lift(G_rated, weekdays == day, strength) for day in range(7)])


def genre_affinity(G_candidates, lift):
    """Mean lift over each candidate's genres (0 for a movie without genres)."""
    sizes = G_candidates.sum(axis=1)
    return np.divide(G_candidates @ lift, sizes, out=np.zeros(len(G_candidates)), where=sizes > 0)


CAP = 2.5


def squash(z, cap=CAP):
    """cap * tanh(z / cap): about linear up to +-2, never beyond +-cap.

    EASE scores from one or two likes have heavy tails (z of 20+ for close neighbours), which
    would let a single like outweigh the whole blend for movies far down the catalogue.
    """
    return cap * np.tanh(np.asarray(z, float) / cap)


def _z_or_zero(scores, mask):
    """Pull from a subset of the likes: squashed positive part of the z-scored EASE score.

    One-sided on purpose: these likes pull their neighbours up but never push a movie down,
    because EASE's negative weights mostly mean "rarely rated together", not "dislike".
    Zeros when the subset is empty.
    """
    if not mask.any():
        return np.zeros(scores.shape[-1])
    return squash(np.maximum(standardize(scores[None, :])[0], 0.0))


def context_features(ease, mf, alpha, positions, stars, like_weekdays, like_dayparts,
                     weekday, part, G, global_lift):
    """Feature matrix (movies x FEATURES) for one profile in one context.

    `positions`, `stars`, `like_weekdays` and `like_dayparts` describe the profile's
    likes; G is the movies x genres matrix aligned with the models' catalog.
    """
    positions = np.asarray(positions)
    like_weekdays, like_dayparts = np.asarray(like_weekdays), np.asarray(like_dayparts)
    rows = np.zeros(len(positions), int)
    ease_all = ease.score(ease.history_matrix(rows, positions, 1))[0]
    base = blend(ease_all[None, :], mf.score_items(mf.fold_in(positions, stars))[None, :], alpha)[0]
    on_day, in_part = like_weekdays == weekday, like_dayparts == part
    same_day = _z_or_zero(np.asarray(ease.weights[positions[on_day]].sum(axis=0), float), on_day)
    same_part = _z_or_zero(np.asarray(ease.weights[positions[in_part]].sum(axis=0), float), in_part)
    liked = stars >= 4
    G_hist = G[positions[liked]]
    user_genre = genre_affinity(G, log_lift(G_hist, on_day[liked])) if liked.any() else np.zeros(len(G))
    everyone = genre_affinity(G, global_lift[int(weekday)])
    return np.column_stack([base, same_day, same_part, user_genre, everyone])


def fit_logistic(X, y, l2=1.0, n_iter=50):
    """L2-regularised logistic regression by Newton's method; returns (weights, intercept).

    Small and dependency-free: the project avoids scikit-learn. Features should be on
    comparable scales; the intercept is not penalised.
    """
    X = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X.shape[1])
    penalty = np.full(X.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        gradient = X.T @ (p - y) + penalty * w
        hessian = (X * (p * (1 - p))[:, None]).T @ X + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


class ContextRanker:
    """Linear re-ranker over FEATURES, applied to the final ranker's top `depth` candidates.

    With only the `blend` weight set (the default), it reproduces the final ranker.
    """

    def __init__(self, weights=None, depth=100):
        self.weights = np.array([1.0, 0, 0, 0, 0]) if weights is None else np.asarray(weights, float)
        self.depth = depth

    @classmethod
    def from_summary(cls, summary):
        selected = summary["selected"]
        return cls([selected["weights"][name] for name in FEATURES], selected.get("depth", 100))

    def score(self, features):
        return features @ self.weights

    def contributions(self, features):
        """Each feature's share of the context score: features x weights (movies x FEATURES)."""
        return features * self.weights

    def rank(self, features, exclude, k, extra=None, penalty=None):
        """Top-k positions: the best `depth` by blend (+ extra), re-ordered by the context score (+ extra).

        `extra` is an explicit boost such as the website's mood genres; it also widens the
        candidate pool, so a strong mood can surface movies the blend alone ranks lower.
        `penalty` (such as the popularity penalty) only re-orders that pool.
        """
        boost = np.zeros(len(features)) if extra is None else np.asarray(extra, float)
        base = np.where(exclude, -np.inf, features[:, 0] + boost)
        depth = min(self.depth, int(np.isfinite(base).sum()))
        if depth == 0:
            return np.array([], dtype=int)
        k = min(k, depth)
        candidates = np.argpartition(-base, depth - 1)[:depth]
        score = self.score(features[candidates]) + boost[candidates]
        if penalty is not None:
            score = score - np.asarray(penalty, float)[candidates]
        return candidates[np.argsort(-score, kind="stable")[:k]]

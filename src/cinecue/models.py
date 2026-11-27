"""Rating models: global mean, regularized biases, biased ALS matrix factorization, and
shrunken context offsets.

Every model is fitted on raw user and movie IDs and predicts for any ID. Unknown users
or movies fall back to the parts of the model that are known, so cold starts never fail.
"""

import numpy as np
import pandas as pd

RATING_MIN, RATING_MAX = 1.0, 5.0


class IdIndex:
    """Map raw user or movie IDs to dense positions; unknown IDs map to -1."""

    def __init__(self, ids):
        self.ids = pd.Index(np.unique(np.asarray(ids)))

    def __len__(self):
        return len(self.ids)

    def encode(self, values):
        return self.ids.get_indexer(np.asarray(values))


def _lookup(values, positions, default=0.0):
    """values[positions] with `default` where the position is unknown (-1)."""
    known = positions >= 0
    out = np.full(len(positions), default, dtype=float)
    out[known] = values[positions[known]]
    return out


def _groups(codes, size):
    """Row numbers of each code, as (order, starts): rows of code g are order[starts[g]:starts[g+1]]."""
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=size))])
    return order, starts


class GlobalMean:
    """Predict the training mean for every pair."""

    def fit(self, users, movies, ratings):
        self.mean = float(np.mean(ratings))
        return self

    def predict(self, users, movies):
        return np.full(len(users), self.mean)


class BiasModel:
    """mean + user bias + movie bias, fitted by alternating regularized averages.

    A larger regularization pulls the biases of sparsely rated users and movies toward 0.
    Unknown users or movies get a bias of 0: an unseen user receives mean + movie bias, an
    unseen movie receives mean + user bias, and an unseen pair receives the mean.
    """

    def __init__(self, user_reg=10.0, item_reg=10.0, n_iter=10):
        self.user_reg, self.item_reg, self.n_iter = user_reg, item_reg, n_iter

    def fit(self, users, movies, ratings):
        self.users, self.movies = IdIndex(users), IdIndex(movies)
        u, i = self.users.encode(users), self.movies.encode(movies)
        r = np.asarray(ratings, dtype=float)
        self.mean = float(r.mean())
        user_counts = np.bincount(u, minlength=len(self.users))
        item_counts = np.bincount(i, minlength=len(self.movies))
        self.user_bias = np.zeros(len(self.users))
        for _ in range(self.n_iter):
            self.item_bias = np.bincount(i, r - self.mean - self.user_bias[u],
                                         minlength=len(self.movies)) / (self.item_reg + item_counts)
            self.user_bias = np.bincount(u, r - self.mean - self.item_bias[i],
                                         minlength=len(self.users)) / (self.user_reg + user_counts)
        return self

    def predict(self, users, movies):
        u, i = self.users.encode(users), self.movies.encode(movies)
        raw = self.mean + _lookup(self.user_bias, u) + _lookup(self.item_bias, i)
        return np.clip(raw, RATING_MIN, RATING_MAX)

    # Online profile hooks (see online_predict): the user bias from a rating history.
    def start_profile(self, movie_positions, ratings):
        residual = np.asarray(ratings, float) - self.mean - self.item_bias[movie_positions]
        return [residual.sum(), len(residual)]

    def profile_add(self, state, movie_positions, ratings):
        state[0] += (np.asarray(ratings, float) - self.mean - self.item_bias[movie_positions]).sum()
        state[1] += len(movie_positions)

    def profile_predict(self, state, movie_positions):
        user_bias = state[0] / (self.user_reg + state[1])
        return self.mean + user_bias + _lookup(self.item_bias, movie_positions)


class MatrixFactorization:
    """Biased matrix factorization fitted by alternating least squares (ALS).

    score(u, i) = mean + b_i + [q_i, 1] . [p_u, b_u]

    Each step solves a ridge regression per user (or per movie) with the penalty
    reg * (number of ratings), the weighted-lambda scheme of Zhou et al. (2008). Every
    iteration ends with the user step, so each stored user vector is exactly
    `fold_in` of that user's ratings. explain.py relies on that to split a score into
    one additive term per rated movie.
    """

    def __init__(self, factors=32, reg=0.05, n_iter=12, seed=0):
        self.factors, self.reg, self.n_iter, self.seed = factors, reg, n_iter, seed

    def fit(self, users, movies, ratings):
        self.users, self.movies = IdIndex(users), IdIndex(movies)
        u, i = self.users.encode(users), self.movies.encode(movies)
        r = np.asarray(ratings, dtype=float)
        self.mean = float(r.mean())
        rng = np.random.default_rng(self.seed)
        k = self.factors
        self.item_factors = rng.normal(0.0, 0.1, (len(self.movies), k))
        self.item_bias = np.zeros(len(self.movies))
        self.user_vectors = np.zeros((len(self.users), k + 1))  # [p_u, b_u]
        by_user = _groups(u, len(self.users))
        by_item = _groups(i, len(self.movies))
        self.history = []
        self._refresh_item_aug()
        self._user_step(u, i, r, by_user)
        for _ in range(self.n_iter):
            self._item_step(u, i, r, by_item)
            self._refresh_item_aug()
            self._user_step(u, i, r, by_user)
            self.history.append(float(np.sqrt(np.mean((self._raw(u, i) - r) ** 2))))
        return self

    def _refresh_item_aug(self):
        """Movie vectors with a trailing 1, so [q_i, 1] . [p_u, b_u] includes the user bias."""
        self.item_aug = np.hstack([self.item_factors, np.ones((len(self.movies), 1))])

    def _solve(self, X, y):
        A = X.T @ X + self.reg * len(y) * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)

    def _user_step(self, u, i, r, groups):
        order, starts = groups
        X_all = self.item_aug
        target = r - self.mean - self.item_bias[i]
        for g in range(len(self.users)):
            rows = order[starts[g]:starts[g + 1]]
            self.user_vectors[g] = self._solve(X_all[i[rows]], target[rows])

    def _item_step(self, u, i, r, groups):
        order, starts = groups
        k = self.factors
        X_all = np.hstack([self.user_vectors[:, :k], np.ones((len(self.users), 1))])
        target = r - self.mean - self.user_vectors[u, k]
        for g in range(len(self.movies)):
            rows = order[starts[g]:starts[g + 1]]
            solution = self._solve(X_all[u[rows]], target[rows])
            self.item_factors[g], self.item_bias[g] = solution[:k], solution[k]

    def user_system(self, movie_positions, ratings):
        """The ridge system behind one user vector: design X, target y, and matrix A."""
        X = self.item_aug[movie_positions]
        y = np.asarray(ratings, dtype=float) - self.mean - self.item_bias[movie_positions]
        A = X.T @ X + self.reg * len(y) * np.eye(X.shape[1])
        return X, y, A

    def fold_in(self, movie_positions, ratings):
        """User vector [p_u, b_u] for a rating history, holding the movie vectors fixed."""
        if len(movie_positions) == 0:
            return np.zeros(self.factors + 1)
        X, y, A = self.user_system(movie_positions, ratings)
        return np.linalg.solve(A, X.T @ y)

    def _raw(self, u, i):
        return self.mean + self.item_bias[i] + np.einsum("ij,ij->i", self.item_aug[i], self.user_vectors[u])

    def score_items(self, user_vector):
        """Unclipped scores of every known movie for one user vector."""
        return self.mean + self.item_bias + self.item_aug @ user_vector

    def predict(self, users, movies):
        u, i = self.users.encode(users), self.movies.encode(movies)
        known_user, known_item = u >= 0, i >= 0
        both = known_user & known_item
        raw = self.mean + _lookup(self.item_bias, i)
        raw[both] += np.einsum("ij,ij->i", self.item_aug[i[both]], self.user_vectors[u[both]])
        # Known user, unseen movie: [0, 1] . [p_u, b_u] leaves just the user bias.
        user_only = known_user & ~known_item
        raw[user_only] += self.user_vectors[u[user_only], self.factors]
        return np.clip(raw, RATING_MIN, RATING_MAX)

    # Online profile hooks (see online_predict): running sums of the fold-in system.
    def start_profile(self, movie_positions, ratings):
        X, y, _ = self.user_system(movie_positions, ratings)
        return [X.T @ X, X.T @ y, len(y)]

    def profile_add(self, state, movie_positions, ratings):
        X, y, _ = self.user_system(movie_positions, ratings)
        state[0] += X.T @ X
        state[1] += X.T @ y
        state[2] += len(y)

    def profile_vector(self, state):
        gram, moment, n = state
        if n == 0:
            return np.zeros(self.factors + 1)
        return np.linalg.solve(gram + self.reg * n * np.eye(len(moment)), moment)

    def profile_predict(self, state, movie_positions):
        vector = self.profile_vector(state)
        known = movie_positions >= 0
        out = np.full(len(movie_positions), self.mean + vector[self.factors])  # unseen movie
        seen = movie_positions[known]
        out[known] = self.mean + self.item_bias[seen] + self.item_aug[seen] @ vector
        return out


def online_predict(model, history, events):
    """Predict each event from the user's history plus their strictly earlier events.

    This mirrors a live service: movie parameters stay at their fitted values while a
    user's profile absorbs each new rating. Events that share a timestamp are predicted
    together before any of them is added, so no rating informs itself or a simultaneous
    one. Ratings of movies the model has never seen cannot update the profile.
    Returns predictions aligned with `events`.
    """
    order = np.lexsort((events.source_row.to_numpy(), events.timestamp.to_numpy()))
    users = events.user_id.to_numpy()[order]
    movies = model.movies.encode(events.movie_id.to_numpy()[order])
    ratings = events.rating.to_numpy(dtype=float)[order]
    stamps = events.timestamp.to_numpy()[order]
    known_history = history.loc[history.movie_id.isin(model.movies.ids) & history.user_id.isin(users)]
    past = {user: (model.movies.encode(group.movie_id.to_numpy()), group.rating.to_numpy(dtype=float))
            for user, group in known_history.groupby("user_id")}
    empty = (np.array([], dtype=int), np.array([], dtype=float))
    predictions = np.empty(len(order))
    by_user = pd.Series(np.arange(len(order))).groupby(users).indices
    for user, rows in by_user.items():
        state = model.start_profile(*past.get(user, empty))
        # Within a user, rows are already in time order; split them at timestamp changes.
        for block in np.split(rows, np.flatnonzero(np.diff(stamps[rows])) + 1):
            predictions[block] = model.profile_predict(state, movies[block])
            seen = block[movies[block] >= 0]
            model.profile_add(state, movies[seen], ratings[seen])
    aligned = np.empty(len(order))
    aligned[order] = predictions
    return np.clip(aligned, RATING_MIN, RATING_MAX)


def session_offsets(events, predictions, gap_seconds=3600, reg=60.0):
    """Rating-session context: the shrunken mean error of the user's earlier ratings this session.

    A rating session is a rating-session proxy, not a viewing session: consecutive ratings
    by one user with no gap longer than `gap_seconds`. Within a session, each event gets
    sum(rating - prediction) / (count + reg) over the user's strictly earlier events in
    that session, so a user who is rating above their profile today is predicted higher.
    Events sharing a timestamp never inform each other, and the first event of a session
    gets 0. `predictions` must be aligned with `events` (for example from online_predict).
    """
    order = np.lexsort((events.source_row.to_numpy(), events.timestamp.to_numpy(), events.user_id.to_numpy()))
    users = events.user_id.to_numpy()[order]
    stamps = events.timestamp.to_numpy()[order]
    errors = (events.rating.to_numpy(dtype=float) - np.asarray(predictions, dtype=float))[order]
    starts = np.r_[True, (users[1:] != users[:-1]) | (np.diff(stamps) > gap_seconds)]
    blocks = np.split(np.arange(len(order)), np.flatnonzero(starts[1:] | (np.diff(stamps) != 0)) + 1)
    offsets = np.zeros(len(order))
    total = count = 0.0
    for block in blocks:
        if starts[block[0]]:
            total = count = 0.0
        offsets[block] = total / (count + reg) if count else 0.0
        total += errors[block].sum()
        count += len(block)
    aligned = np.empty(len(order))
    aligned[order] = offsets
    return aligned


class EASE:
    """EASE item-to-item model for top-N ranking (Steck, 2019).

    It learns from which movies each user chose to rate, not from the stars. The
    weights solve min ||X - XB||^2 + reg ||B||^2 with a zero diagonal in closed form,
    so score(u, i) = sum of B[j, i] over the movies j that u rated: one exact term
    per rated movie.
    """

    def __init__(self, reg=1000.0):
        self.reg = reg

    def fit(self, users, movies):
        self.users, self.movies = IdIndex(users), IdIndex(movies)
        X = self.history_matrix(self.users.encode(users), self.movies.encode(movies), len(self.users))
        P = np.linalg.inv(X.T @ X + self.reg * np.eye(len(self.movies)))
        self.weights = -P / np.diag(P)
        np.fill_diagonal(self.weights, 0.0)
        return self

    def history_matrix(self, rows, movie_positions, n_rows, weights=None):
        """One row per user: 1 (or `weights`) for each rated movie; unknown movies (-1) are skipped.

        Weights let recent ratings count more at scoring time. Training always uses 0/1.
        """
        known = movie_positions >= 0
        values = np.ones(len(movie_positions)) if weights is None else np.asarray(weights, dtype=float)
        X = np.zeros((n_rows, len(self.movies)))
        X[np.asarray(rows)[known], movie_positions[known]] = values[known]
        return X

    def score(self, history):
        return history @ self.weights


class ContextOffsets:
    """Shrunken mean residuals for groups such as (user, weekend) or (movie, daypart).

    Groups are fitted in order, each on what the earlier groups left unexplained:
    offset = sum(residual) / (count + reg). Unseen groups get 0, so the base model's
    prediction is unchanged when no context history exists.
    """

    def __init__(self, groups, reg=25.0):
        self.groups, self.reg = [tuple(group) for group in groups], reg

    def fit(self, frame, residual):
        remaining = np.asarray(residual, dtype=float).copy()
        self.tables = []
        for columns in self.groups:
            stats = (pd.DataFrame({**{c: frame[c].to_numpy() for c in columns}, "residual": remaining})
                     .groupby(list(columns))["residual"].agg(["sum", "count"]))
            table = (stats["sum"] / (stats["count"] + self.reg)).rename("offset")
            self.tables.append(table)
            remaining -= self._offset(frame, columns, table)
        return self

    @staticmethod
    def _offset(frame, columns, table):
        keys = pd.MultiIndex.from_frame(frame[list(columns)].reset_index(drop=True))
        return table.reindex(keys).fillna(0.0).to_numpy()

    def predict(self, frame):
        return sum(self._offset(frame, columns, table) for columns, table in zip(self.groups, self.tables))


class WithContext:
    """A base rating model plus context offsets fitted on its training residuals."""

    def __init__(self, base, offsets):
        self.base, self.offsets = base, offsets

    def fit(self, frame):
        residual = frame.rating.to_numpy() - self.base.predict(frame.user_id, frame.movie_id)
        self.offsets.fit(frame, residual)
        return self

    def predict_frame(self, frame):
        raw = self.base.predict(frame.user_id, frame.movie_id) + self.offsets.predict(frame)
        return np.clip(raw, RATING_MIN, RATING_MAX)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(predicted, float)) ** 2)))


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float))))

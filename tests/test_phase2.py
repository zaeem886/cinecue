"""Small synthetic checks for the Phase 2 code. Run: python -m unittest discover -s tests"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cinecue.explain import blend_contributions, contributions, personal_score  # noqa: E402
from cinecue.models import (EASE, BiasModel, ContextOffsets, MatrixFactorization,  # noqa: E402
                            online_predict, rmse, session_offsets)
from cinecue.ranking import (accuracy_metrics, beyond_accuracy, blend, paired_bootstrap,  # noqa: E402
                             popularity_scale, recency_weights, rerank_by_popularity, top_k)


def low_rank_ratings(n_users=60, n_items=40, rank=3, density=0.5, seed=1):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(0, 0.6, (n_users, rank)), rng.normal(0, 0.6, (n_items, rank))
    item_bias = rng.normal(0, 0.4, n_items)
    users, items = np.nonzero(rng.random((n_users, n_items)) < density)
    ratings = np.clip(3.5 + item_bias[items] + np.einsum("ij,ij->i", P[users], Q[items]), 1, 5)
    return users + 100, items + 500, ratings  # offset raw IDs from positions


class RatingModelTests(unittest.TestCase):
    def test_bias_model_recovers_additive_biases(self):
        users, items = np.meshgrid(np.arange(20), np.arange(15), indexing="ij")
        users, items = users.ravel(), items.ravel()
        user_effect, item_effect = np.linspace(-0.5, 0.5, 20), np.linspace(-1, 1, 15)
        ratings = 3 + user_effect[users] + item_effect[items]
        model = BiasModel(user_reg=0, item_reg=0, n_iter=50).fit(users, items, ratings)
        self.assertLess(rmse(ratings, model.predict(users, items)), 1e-6)

    def test_bias_model_falls_back_for_unknown_ids(self):
        model = BiasModel().fit([1, 1, 2], [10, 11, 10], [5, 4, 3])
        both = model.predict([99], [999])[0]
        self.assertAlmostEqual(both, model.mean)
        movie_only = model.predict([99], [10])[0]
        self.assertAlmostEqual(movie_only, model.mean + model.item_bias[model.movies.encode([10])[0]])

    def test_matrix_factorization_beats_biases_on_low_rank_data(self):
        users, items, ratings = low_rank_ratings()
        bias = BiasModel(user_reg=1, item_reg=1).fit(users, items, ratings)
        mf = MatrixFactorization(factors=3, reg=0.01, n_iter=20).fit(users, items, ratings)
        self.assertLess(rmse(ratings, mf.predict(users, items)), 0.5 * rmse(ratings, bias.predict(users, items)))
        self.assertLessEqual(mf.history[-1], mf.history[0] + 1e-9)

    def test_matrix_factorization_cold_start_fallbacks(self):
        users, items, ratings = low_rank_ratings()
        mf = MatrixFactorization(factors=2, n_iter=3).fit(users, items, ratings)
        u, i = mf.users.encode([users[0]])[0], mf.movies.encode([items[0]])[0]
        self.assertAlmostEqual(mf.predict([-7], [-8])[0], np.clip(mf.mean, 1, 5))
        self.assertAlmostEqual(mf.predict([-7], [items[0]])[0], np.clip(mf.mean + mf.item_bias[i], 1, 5))
        self.assertAlmostEqual(mf.predict([users[0]], [-8])[0],
                               np.clip(mf.mean + mf.user_vectors[u, -1], 1, 5))

    def test_stored_user_vectors_are_fold_ins(self):
        users, items, ratings = low_rank_ratings()
        mf = MatrixFactorization(factors=3, n_iter=4).fit(users, items, ratings)
        rows = users == users[0]
        folded = mf.fold_in(mf.movies.encode(items[rows]), ratings[rows])
        np.testing.assert_allclose(folded, mf.user_vectors[mf.users.encode([users[0]])[0]], atol=1e-10)

    def test_context_offsets_recover_group_effects(self):
        frame = pd.DataFrame({"user_id": np.repeat([1, 2], 400), "is_weekend_utc": np.tile([0, 1], 400)})
        effect = np.where((frame.user_id == 1) & (frame.is_weekend_utc == 1), 0.5, 0.0)
        offsets = ContextOffsets([("user_id", "is_weekend_utc")], reg=0).fit(frame, effect)
        np.testing.assert_allclose(offsets.predict(frame), effect)
        unseen = pd.DataFrame({"user_id": [3], "is_weekend_utc": [1]})
        self.assertEqual(offsets.predict(unseen)[0], 0.0)


class OnlinePredictionTests(unittest.TestCase):
    def setUp(self):
        users, items, ratings = low_rank_ratings()
        self.history = pd.DataFrame({"user_id": users, "movie_id": items, "rating": ratings})
        self.models = [BiasModel(user_reg=2, item_reg=2).fit(users, items, ratings),
                       MatrixFactorization(factors=3, n_iter=4).fit(users, items, ratings)]
        # A known user and a new user (7) rate known movies; two of user 7's events share a time.
        self.events = pd.DataFrame({"user_id": [users[0], users[0], 7, 7, 7, 7],
                                    "movie_id": [items[1], items[2], items[3], items[4], items[5], items[6]],
                                    "rating": [5.0, 1.0, 5.0, 4.0, 1.0, 2.0],
                                    "timestamp": [10, 20, 10, 20, 20, 30], "source_row": np.arange(6)})

    def test_first_event_matches_the_fitted_model(self):
        for model in self.models:
            online = online_predict(model, self.history, self.events)
            static = model.predict(self.events.user_id, self.events.movie_id)
            self.assertAlmostEqual(online[0], static[0])  # known user, nothing earlier
            self.assertAlmostEqual(online[2], static[2])  # new user's first event: fallback

    def test_no_event_informs_itself_or_a_simultaneous_event(self):
        for model in self.models:
            base = online_predict(model, self.history, self.events)
            changed = self.events.assign(rating=self.events.rating.where(self.events.index != 3, 1.0))
            after = online_predict(model, self.history, changed)
            np.testing.assert_allclose(after[:5], base[:5])  # rows 3 and 4 share timestamp 20
            self.assertNotAlmostEqual(after[5], base[5])     # the later event sees the change

    def test_row_order_does_not_matter(self):
        for model in self.models:
            shuffled = self.events.sample(frac=1, random_state=3)
            np.testing.assert_allclose(online_predict(model, self.history, shuffled),
                                       online_predict(model, self.history, self.events)[shuffled.index])


class SessionOffsetTests(unittest.TestCase):
    def setUp(self):
        # User 1: a session at t=0, 10, 10, 20, then a new session after a long gap.
        self.events = pd.DataFrame({"user_id": [1, 1, 1, 1, 1, 2], "timestamp": [0, 10, 10, 20, 9000, 5],
                                    "rating": [5.0, 4.0, 2.0, 3.0, 4.0, 5.0], "source_row": np.arange(6)})
        self.predictions = np.full(6, 3.0)

    def test_offsets_use_only_strictly_earlier_ratings_in_the_session(self):
        offsets = session_offsets(self.events, self.predictions, gap_seconds=100, reg=0.0)
        np.testing.assert_allclose(offsets, [0.0, 2.0, 2.0, (2 + 1 - 1) / 3, 0.0, 0.0])

    def test_shrinkage_and_row_order(self):
        offsets = session_offsets(self.events, self.predictions, gap_seconds=100, reg=1.0)
        self.assertAlmostEqual(offsets[1], 2.0 / 2.0)
        shuffled = self.events.sample(frac=1, random_state=0)
        np.testing.assert_allclose(session_offsets(shuffled, self.predictions[shuffled.index], 100, 1.0),
                                   offsets[shuffled.index])


class ExplanationTests(unittest.TestCase):
    def test_contributions_sum_to_the_personal_score(self):
        users, items, ratings = low_rank_ratings()
        mf = MatrixFactorization(factors=3, n_iter=5).fit(users, items, ratings)
        rows = users == users[3]
        history = mf.movies.encode(items[rows])
        targets = np.arange(len(mf.movies))
        parts = contributions(mf, history, ratings[rows], targets)
        vector = mf.user_vectors[mf.users.encode([users[3]])[0]]
        np.testing.assert_allclose(parts.sum(axis=1), personal_score(mf, vector, targets), atol=1e-10)
        np.testing.assert_allclose(mf.score_items(vector), mf.mean + mf.item_bias + parts.sum(axis=1),
                                   atol=1e-10)

    def test_blend_contributions_rebuild_the_blended_score(self):
        users, items, ratings = low_rank_ratings()
        mf = MatrixFactorization(factors=3, n_iter=5).fit(users, items, ratings)
        ease = EASE(reg=5.0).fit(users, items)
        pd.testing.assert_index_equal(ease.movies.ids, mf.movies.ids)
        rows = users == users[5]
        history, stars = mf.movies.encode(items[rows]), ratings[rows]
        targets = np.arange(len(mf.movies))
        terms, appeal, constant = blend_contributions(ease, mf, 0.4, history, stars, targets)
        ease_scores = ease.score(ease.history_matrix(np.zeros(len(history), int), history, 1))
        mf_scores = mf.score_items(mf.fold_in(history, stars))[None, :]
        np.testing.assert_allclose(constant + appeal + terms.sum(axis=1), blend(ease_scores, mf_scores, 0.4)[0],
                                   atol=1e-9)


    def test_weighted_blend_contributions_rebuild_the_blended_score(self):
        users, items, ratings = low_rank_ratings()
        mf = MatrixFactorization(factors=3, n_iter=5).fit(users, items, ratings)
        ease = EASE(reg=5.0).fit(users, items)
        rows = users == users[9]
        history, stars = mf.movies.encode(items[rows]), ratings[rows]
        weights = np.linspace(0.1, 1.0, len(history))
        targets = np.arange(len(mf.movies))
        terms, appeal, constant = blend_contributions(ease, mf, 0.6, history, stars, targets, weights)
        ease_scores = ease.score(ease.history_matrix(np.zeros(len(history), int), history, 1, weights))
        mf_scores = mf.score_items(mf.fold_in(history, stars))[None, :]
        np.testing.assert_allclose(constant + appeal + terms.sum(axis=1), blend(ease_scores, mf_scores, 0.6)[0],
                                   atol=1e-9)


class EaseTests(unittest.TestCase):
    def test_weights_have_zero_diagonal_and_scores_sum_rows(self):
        users, items, _ = low_rank_ratings()
        ease = EASE(reg=10.0).fit(users, items)
        self.assertTrue(np.all(np.diag(ease.weights) == 0))
        history = ease.history_matrix([0, 0], ease.movies.encode(items[:2]), 1)
        positions = ease.movies.encode(items[:2])
        np.testing.assert_allclose(ease.score(history)[0], ease.weights[positions].sum(axis=0))

    def test_chunked_fit_matches_one_block_and_float32_storage(self):
        users, items, _ = low_rank_ratings()
        whole = EASE(reg=10.0).fit(users, items, chunk_users=10_000)
        blocks = EASE(reg=10.0).fit(users, items, chunk_users=7)
        np.testing.assert_array_equal(blocks.weights, whole.weights)
        small = EASE(reg=10.0).fit(users, items, dtype=np.float32)
        self.assertEqual(small.weights.dtype, np.float32)
        np.testing.assert_allclose(small.weights, whole.weights, atol=1e-6)

    def test_weights_solve_the_constrained_ridge_problem(self):
        users, items, _ = low_rank_ratings(n_users=30, n_items=8)
        ease = EASE(reg=3.0).fit(users, items)
        X = ease.history_matrix(ease.users.encode(users), ease.movies.encode(items), len(ease.users))
        # Stationarity: the off-diagonal gradient of ||X - XB||^2 + reg ||B||^2 vanishes.
        gradient = -X.T @ (X - X @ ease.weights) + 3.0 * ease.weights
        off_diagonal = ~np.eye(len(ease.movies), dtype=bool)
        np.testing.assert_allclose(gradient[off_diagonal], 0, atol=1e-9)


class RankingTests(unittest.TestCase):
    def test_top_k_orders_and_excludes(self):
        scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.8, 0.1]])
        exclude = np.array([[False, True, False, False], [False, False, False, False]])
        np.testing.assert_array_equal(top_k(scores, exclude, 2), [[3, 2], [2, 0]])

    def test_rerank_with_zero_weight_matches_top_k(self):
        rng = np.random.default_rng(0)
        scores, exclude = rng.random((5, 30)), rng.random((5, 30)) < 0.2
        popularity = popularity_scale(rng.integers(1, 100, 30))
        np.testing.assert_array_equal(rerank_by_popularity(scores, exclude, popularity, 0.0, 5, depth=10),
                                      top_k(scores, exclude, 5))

    def test_rerank_demotes_popular_movies(self):
        scores = np.array([[1.0, 0.99, 0.5]])
        popularity = np.array([1.0, 0.0, 0.0])
        ranked = rerank_by_popularity(scores, np.zeros_like(scores, bool), popularity, 0.1, 2, depth=3)
        np.testing.assert_array_equal(ranked, [[1, 0]])

    def test_accuracy_metrics_by_hand(self):
        ranked = np.array([[4, 2, 7]])
        metrics = accuracy_metrics(ranked, [{2, 9}]).iloc[0]
        self.assertAlmostEqual(metrics.precision, 1 / 3)
        self.assertAlmostEqual(metrics.recall, 1 / 2)
        self.assertAlmostEqual(metrics.ndcg, (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertEqual(metrics.hit, 1.0)

    def test_recency_weights_halve_every_half_life(self):
        day = 86400
        np.testing.assert_allclose(recency_weights([10 * day, 8 * day, 6 * day], 10 * day, 2), [1, 0.5, 0.25])
        np.testing.assert_allclose(recency_weights([1, 2], 5, None), [1, 1])
        with self.assertRaises(AssertionError):
            recency_weights([11 * day], 10 * day, 2)

    def test_paired_bootstrap_brackets_the_difference(self):
        rng = np.random.default_rng(4)
        a = rng.random(300)
        result = paired_bootstrap(a + 0.1, a, n_resamples=200)
        self.assertAlmostEqual(result["difference"], 0.1)
        self.assertAlmostEqual(result["ci_low"], 0.1)  # a constant paired shift has no spread
        noisy = paired_bootstrap(a, rng.random(300), n_resamples=500)
        self.assertLess(noisy["ci_low"], noisy["difference"])
        self.assertGreater(noisy["ci_high"], noisy["difference"])

    def test_beyond_accuracy(self):
        ranked = np.array([[0, 1], [0, 2]])
        genres = np.array([[1, 0], [1, 1], [0, 1], [1, 1]])
        head = np.array([True, False, False, False])
        result = beyond_accuracy(ranked, np.array([10, 2, 3, 1]), head, genres, catalog_size=4)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["long_tail_share"], 0.5)
        self.assertAlmostEqual(result["genre_diversity"], (0.5 + 1.0) / 2)


if __name__ == "__main__":
    unittest.main()

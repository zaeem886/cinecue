"""Checks for the weekday/daypart context re-ranker, timed likes, and the context-aware adapter."""

from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cinecue.context import (ContextRanker, context_features, describe, fit_logistic,  # noqa: E402
                             genre_matrix, log_lift)
from cinecue.demo import context_tables, describe_like, recommend, recommend_in_context  # noqa: E402
from cinecue.likes import load_likes, make_like, save_likes  # noqa: E402
from cinecue.models import EASE, MatrixFactorization  # noqa: E402

GENRES = ['Crime|Drama|Thriller', 'Comedy', 'Comedy|Romance', 'Action|Sci-Fi', "Children's|Animation"]


class ContextModelTests(unittest.TestCase):
    def test_logistic_recovers_the_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 2))
        y = (rng.random(4000) < 1 / (1 + np.exp(-(2 * X[:, 0] - 0.0 * X[:, 1])))).astype(float)
        weights, _ = fit_logistic(X, y, l2=0.1)
        self.assertAlmostEqual(weights[0], 2.0, delta=0.25)
        self.assertAlmostEqual(weights[1], 0.0, delta=0.15)

    def test_lift_is_zero_without_context_and_positive_for_its_genre(self):
        G, vocabulary = genre_matrix(['Crime|Drama', 'Comedy', 'Comedy', 'Crime'])
        self.assertTrue(np.allclose(log_lift(G, [False] * 4), 0))
        lift = log_lift(G, [True, False, False, True], strength=1.0)
        self.assertGreater(lift[vocabulary.index('Crime')], 0)
        self.assertLess(lift[vocabulary.index('Comedy')], 0)

    def test_genre_aliases_merge_dataset_spellings(self):
        _, vocabulary = genre_matrix(["Children's|Animation", 'Children'])
        self.assertEqual(vocabulary, ['Animation', 'Children'])

    def test_describe(self):
        self.assertEqual(describe(0, 3), 'Monday evening')
        self.assertEqual(describe_like(0, 3, 0, 3), 'on a Monday evening, like now')
        self.assertEqual(describe_like(0, 1, 0, 3), 'on a Monday morning, also a Monday')
        self.assertEqual(describe_like(4, 1, 0, 3), 'on a Friday morning')


class ContextAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        users, items = np.nonzero(rng.random((80, 40)) < 0.6)
        ratings = rng.integers(1, 6, len(users))
        history = pd.DataFrame(dict(user_id=users, movie_id=items, rating=ratings))
        mf = MatrixFactorization(factors=3, n_iter=3).fit(users, items, ratings)
        ease = EASE(reg=10).fit(users, items)
        catalog = pd.DataFrame({'title': [f'Movie {i}' for i in mf.movies.ids],
                                'genres': [GENRES[i % len(GENRES)] for i in mf.movies.ids]}, index=mf.movies.ids)
        counts = np.bincount(items, minlength=40)
        cls.bundle = history, catalog, mf, ease, counts, {'alpha': 0.75, 'rerank_weight': 1.5}
        cls.tables = context_tables(catalog)

    def likes(self, ids, weekday=0, part=3):
        return [make_like(m, weekday=weekday, part=part) for m in ids]

    def test_blend_only_ranker_matches_the_final_ranker(self):
        """With context weights at zero, the re-ranker returns the plain recommendations."""
        liked = [1, 6, 11]
        plain = recommend(self.bundle, None, k=8, liked_movies=liked)
        timed = recommend_in_context(self.bundle, self.tables, ContextRanker(), likes=self.likes(liked), k=8)
        self.assertEqual([r['movie_id'] for r in plain], [r['movie_id'] for r in timed])

    def test_likes_are_excluded_and_rows_are_complete(self):
        liked = [0, 5, 10]
        rows = recommend_in_context(self.bundle, self.tables, ContextRanker([1, .5, .2, .3, .3]),
                                    likes=self.likes(liked), k=6)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertNotIn(row['movie_id'], liked)
            self.assertTrue(1 <= row['prediction'] <= 5)
            self.assertIsInstance(row['context_tags'], list)

    def test_only_mood_keeps_matching_genres(self):
        rows = recommend_in_context(self.bundle, self.tables, ContextRanker(), likes=self.likes([1, 2]), k=6,
                                    mood={'Crime', 'Thriller'}, mood_strength='only')
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue({'Crime', 'Thriller'} & set(row['genres'].split('|')))
            self.assertTrue(row['mood_match'])

    def test_strong_mood_raises_matching_share(self):
        def share(rows):
            return np.mean(['Comedy' in r['genres'] for r in rows])
        likes = self.likes([3, 8, 13])
        plain = recommend_in_context(self.bundle, self.tables, ContextRanker(), likes=likes, k=10)
        moody = recommend_in_context(self.bundle, self.tables, ContextRanker(), likes=likes, k=10,
                                     mood={'Comedy'}, mood_strength='strong')
        self.assertGreaterEqual(share(moody), share(plain))
        self.assertGreater(share(moody), 0.5)

    def test_same_weekday_likes_steer_the_list(self):
        """A heavy same-weekday weight makes the Monday like drive the explanation on a Monday."""
        likes = self.likes([0], weekday=0) + self.likes([3, 8, 13, 18], weekday=4)
        ranker = ContextRanker([1.0, 3.0, 0, 0, 0])
        rows = recommend_in_context(self.bundle, self.tables, ranker, likes=likes, k=5, weekday=0, part=3)
        self.assertTrue(any(r['because'] == 'Movie 0' for r in rows))
        monday = next(r for r in rows if r['because'] == 'Movie 0')
        self.assertIn('Monday', monday['because_when'])
        self.assertIn('Monday likes', monday['context_tags'])

    def test_profile_and_empty_modes(self):
        rows = recommend_in_context(self.bundle, self.tables, ContextRanker(), user_id=0, k=5)
        seen = set(self.bundle[0].loc[self.bundle[0].user_id == 0, 'movie_id'])
        self.assertEqual(len(rows), 5)
        self.assertFalse(seen & {r['movie_id'] for r in rows})
        popular = recommend_in_context(self.bundle, self.tables, ContextRanker(), likes=[], k=5, mood={'Comedy'})
        self.assertEqual(len(popular), 5)
        self.assertIsNone(popular[0]['prediction'])

    def test_features_shape(self):
        mf, ease = self.bundle[2], self.bundle[3]
        G, _, lift = self.tables
        positions = mf.movies.encode(np.array([1, 2, 3]))
        X = context_features(ease, mf, 0.75, positions, np.full(3, 5.0), np.array([0, 0, 2]),
                             np.array([3, 1, 3]), 0, 3, G, lift)
        self.assertEqual(X.shape, (len(mf.movies), 5))
        self.assertTrue(np.isfinite(X).all())


class LikesStoreTests(unittest.TestCase):
    def test_round_trip_and_bad_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'likes.json'
            self.assertEqual(load_likes(path), [])
            like = make_like(7, when=datetime(2026, 11, 30, 20, 15))  # a Monday evening
            self.assertEqual((like['weekday'], like['part']), (0, 3))
            save_likes(path, [like, {'movie_id': 'x'}])
            self.assertEqual(load_likes(path), [like])
            path.write_text('not json')
            self.assertEqual(load_likes(path), [])


if __name__ == '__main__':
    unittest.main()


class TasteMatchTests(unittest.TestCase):
    """taste_match agrees with Discover's ranking and explains its verdict."""

    @classmethod
    def setUpClass(cls):
        ContextAdapterTests.setUpClass.__func__(cls)  # same synthetic bundle, without rerunning its tests

    likes = ContextAdapterTests.likes

    def test_top_pick_matches_discover_and_ranks_are_consistent(self):
        from cinecue.demo import taste_match
        likes = self.likes([1, 6, 11])
        ranker = ContextRanker([1.0, 1.5, 0.5, 0.75, 0])
        top = recommend_in_context(self.bundle, self.tables, ranker, likes=likes, k=3)
        everything = [m for m in self.bundle[1].index if m not in (1, 6, 11)]
        match = taste_match(self.bundle, self.tables, ranker, likes, everything)
        self.assertEqual(match[top[0]['movie_id']]['rank'], 1)
        self.assertEqual(match[top[0]['movie_id']]['label'], 'Top pick today')
        ranks = sorted(m['rank'] for m in match.values() if not m['low_data'])
        self.assertEqual(ranks[0], 1)
        for m in match.values():
            self.assertTrue(0 <= m['percent'] <= 100)
            self.assertGreaterEqual(m['rank'], 1)

    def test_reasons_and_mood(self):
        from cinecue.demo import taste_match
        likes = self.likes([0, 5])
        match = taste_match(self.bundle, self.tables, ContextRanker(), likes, [10, 3, 0],
                            mood={'Crime', 'Comedy'})
        crime, comedy, liked = match[10], match[3], match[0]
        self.assertEqual(crime['mood_match'], ['Crime'])
        self.assertEqual(crime['mood_miss'], ['Comedy'])
        self.assertTrue(liked['liked'])
        for m in match.values():
            if m['closest']:
                self.assertIn(m['closest']['title'], {'Movie 0', 'Movie 5'})

    def test_no_likes_means_no_match(self):
        from cinecue.demo import taste_match
        self.assertEqual(taste_match(self.bundle, self.tables, ContextRanker(), [], [1, 2]), {})

    def test_day_pull_is_one_sided(self):
        """Same-day likes can lift a movie's day features but never make them negative."""
        mf, ease = self.bundle[2], self.bundle[3]
        G, _, lift = self.tables
        positions = mf.movies.encode(np.array([1, 2, 3]))
        X = context_features(ease, mf, 0.75, positions, np.full(3, 5.0), np.array([0, 0, 2]),
                             np.array([3, 1, 3]), 0, 3, G, lift)
        self.assertTrue((X[:, 1:3] >= 0).all())
        self.assertTrue((X[:, 1:3] <= 2.5).all())

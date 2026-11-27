"""Integration checks for the UI adapter using small, synthetic movie histories."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cinecue.demo import recommend
from cinecue.models import EASE, MatrixFactorization


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        users, items = np.nonzero(rng.random((60, 40)) < 0.7)
        ratings = rng.integers(1, 6, len(users))
        history = pd.DataFrame(dict(user_id=users, movie_id=items, rating=ratings))
        mf = MatrixFactorization(factors=3, n_iter=3).fit(users, items, ratings)
        ease = EASE(reg=10).fit(users, items)
        catalog = pd.DataFrame({'title': [f'Movie {i}' for i in mf.movies.ids],
                                'genres': ['Drama|Comedy'] * len(mf.movies)}, index=mf.movies.ids)
        counts = np.bincount(items, minlength=40)
        cls.bundle = history, catalog, mf, ease, counts, {'alpha': 0.75, 'rerank_weight': 1.5}

    def test_recommendations_exclude_history_and_explain_both_modes(self):
        seen = set(self.bundle[0].loc[self.bundle[0].user_id == 0, 'movie_id'])
        for discover in (False, True):
            rows = recommend(self.bundle, 0, k=5, discover=discover)
            self.assertEqual(len(rows), 5)
            self.assertEqual(len({r['title'] for r in rows}), 5)
            for row in rows:
                self.assertNotIn(int(row['title'].split()[-1]), seen)
                self.assertTrue(row['reason'])
                self.assertTrue(1 <= row['prediction'] <= 5)

    def test_unknown_user_gets_popularity_fallback(self):
        rows = recommend(self.bundle, -1, k=5)
        self.assertEqual(len(rows), 5)
        self.assertIsNone(rows[0]['prediction'])
        self.assertIn('Popular', rows[0]['reason'])

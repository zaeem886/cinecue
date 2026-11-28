"""Checks for the website's title helpers and library search. Run: python -m unittest discover -s tests"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinecue.search import TitleSearch, display_title, label, normalize  # noqa: E402

TITLES = {
    1: "Matrix, The (1999)", 2: "Matrix Reloaded, The (2003)", 3: "Star Wars: Episode IV - A New Hope (1977)",
    4: "Spider-Man (2002)", 5: "WALL·E (2008)", 6: "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
    7: "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
    8: "Lord of the Rings: The Fellowship of the Ring, The (2001)", 9: "Dark Knight, The (2008)",
    10: "Dark Knight Rises, The (2012)", 11: "Schindler's List (1993)",
}


class TitleTests(unittest.TestCase):
    def test_display_title_moves_articles_and_splits_the_year(self):
        self.assertEqual(display_title("Matrix, The (1999)"), ("The Matrix", "1999"))
        self.assertEqual(label("Lord of the Rings: The Fellowship of the Ring, The (2001)"),
                         "The Lord of the Rings: The Fellowship of the Ring (2001)")
        self.assertEqual(display_title("Untitled"), ("Untitled", ""))

    def test_normalize_ignores_case_accents_and_punctuation(self):
        self.assertEqual(normalize("Amélie: Spider-Man's WALL·E"), "amelie spider mans wall e")


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.search = TitleSearch(TITLES)

    def found(self, query):
        return self.search(query)[0]

    def test_every_displayed_label_finds_its_movie(self):
        for movie, title in TITLES.items():
            name, _ = display_title(title)
            self.assertIn(movie, self.found(name), name)
            self.assertIn(movie, self.found(label(title)), label(title))

    def test_forgiving_queries(self):
        for query, movie in [("star wars episode iv", 3), ("spider man", 4), ("wall e", 5), ("amelie", 6),
                             ("harry potter sorcerers stone", 7), ("lord of the rings fellowship", 8),
                             ("dark knight 2008", 9), ("schindlers list", 11), ("MATRIX the", 1)]:
            self.assertIn(movie, self.found(query), query)

    def test_prefix_matches_rank_first_then_popularity(self):
        ids, fuzzy = self.search("matrix", popularity={2: 50, 1: 10})
        self.assertEqual(ids[:2], [2, 1])  # neither label starts with "matrix": popularity decides
        ids, _ = self.search("the matrix", popularity={2: 50, 1: 10})
        self.assertEqual(ids[:2], [2, 1])
        self.assertFalse(fuzzy)

    def test_misspelling_falls_back_to_close_titles(self):
        ids, fuzzy = self.search("the dark knigt")
        self.assertTrue(fuzzy)
        self.assertIn(9, ids)

    def test_empty_query_returns_everything(self):
        self.assertEqual(sorted(self.search("  ")[0]), sorted(TITLES))


if __name__ == "__main__":
    unittest.main()

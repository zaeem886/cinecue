"""Readable MovieLens titles and a forgiving title search for the website."""

from difflib import get_close_matches
import re
import unicodedata

ARTICLE = re.compile(r"^(?P<name>.*), (?P<article>The|A|An|Les|La|Le|Il|L'|Das|Die|El)(?P<rest> \(.*\))?$")
YEAR = re.compile(r"^(?P<base>.*?)\s*\((?P<year>\d{4})\)\s*$")


def display_title(title):
    """'Matrix, The (1999)' -> ('The Matrix', '1999'): readable names for MovieLens titles."""
    match = YEAR.match(str(title))
    base, year = (match["base"], match["year"]) if match else (str(title), "")
    moved = ARTICLE.match(base)
    if moved:
        base = f"{moved['article']} {moved['name']}{moved['rest'] or ''}"
    return base, year


def label(title):
    name, year = display_title(title)
    return f"{name} ({year})" if year else name


def normalize(text):
    """Lower-case, accent-free words: "Amélie", "Spider-Man", "Sorcerer's" -> "amelie", "spider man", "sorcerers"."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    text = re.sub(r"['’`]", "", text)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text).split())


class TitleSearch:
    """Every query word must appear in the title (any order); titles that start with the query rank first.

    Each movie is indexed under its MovieLens title and its readable label, so "matrix",
    "the matrix", "matrix the" and "matrix 1999" all find "Matrix, The (1999)".
    """

    def __init__(self, titles):
        self.ids = list(titles)
        self.labels = [normalize(label(titles[m])) for m in self.ids]
        self.text = [f"{self.labels[k]} {normalize(titles[m])}" for k, m in enumerate(self.ids)]

    def __call__(self, query, popularity=None):
        """Matching ids, best first, and whether they are close spellings rather than matches."""
        words = normalize(query).split()
        if not words:
            return list(self.ids), False
        hits = [k for k, text in enumerate(self.text) if all(word in text for word in words)]
        fuzzy = False
        if not hits:
            close = set(get_close_matches(" ".join(words), self.labels, n=12, cutoff=0.6))
            hits, fuzzy = [k for k, text in enumerate(self.labels) if text in close], True
        phrase = " ".join(words)
        weight = popularity or {}
        hits.sort(key=lambda k: (not self.labels[k].startswith(phrase), -weight.get(self.ids[k], 0)))
        return [self.ids[k] for k in hits], fuzzy

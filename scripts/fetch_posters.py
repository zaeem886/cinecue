"""Find a poster image URL for each catalog movie on English Wikipedia.

MovieLens links.csv gives each movie's IMDb ID. Wikidata maps IMDb IDs (P345) to
English Wikipedia articles, and the Wikipedia API returns each article's lead image,
which for films is normally the poster. Only image URLs are stored, in
results/posters.csv; the website loads the images from Wikimedia when shown.
Movies without a match keep the website's generated genre tile.

Usage:
    python scripts/fetch_posters.py --archive data/raw/ml-32m/ml-32m.zip
Movie IDs come from the website models in artifacts/ and the MovieLens 1M catalog.
"""

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.data import table_path  # noqa: E402

HEADERS = {"User-Agent": "CineCue/1.0 (university recommender-system project; poster lookup)",
           "Accept": "application/json"}
WIKIDATA = "https://query.wikidata.org/sparql"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
OUT = ROOT / "results/posters.csv"
HOSTS = ("upload.wikimedia.org", "thumb.wikimedia.org")


def clean_url(url):
    """Keep Wikimedia image URLs only, without tracking query strings; None otherwise."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc not in HOSTS:
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def get_json(url, data=None, attempts=5):
    for attempt in range(attempts):
        try:
            request = Request(url, data=data, headers=HEADERS)
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        except URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(5 * (attempt + 1))


def top_movie_ids(archive, n):
    """The n most-rated movies in the archive: the catalog build_app_catalog.py keeps."""
    with ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith("ratings.csv"))
        with bundle.open(name) as source:
            movies = pd.read_csv(source, usecols=["movieId"], dtype={"movieId": "int32"}).movieId
    return set(movies.value_counts().index[:n].astype(int))


def catalog_movie_ids():
    ids = set()
    for path in sorted((ROOT / "artifacts").glob("recommender*.pkl")):
        with path.open("rb") as source:  # this project's own locally built artifacts
            ids |= set(pickle.load(source)["bundle"][1].index.astype(int))
    for folder in [ROOT / "results", ROOT / "data/processed"]:
        movies = table_path(folder, "movies_clean.csv")
        if movies.is_file():
            ids |= set(pd.read_csv(movies, usecols=["movie_id"]).movie_id.astype(int))
            break
    return ids


def imdb_links(archive):
    with ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith("links.csv"))
        with bundle.open(name) as source:
            links = pd.read_csv(source, dtype={"imdbId": str})
    links["imdb"] = "tt" + links.imdbId.str.zfill(7)
    return dict(zip(links.movieId.astype(int), links.imdb))


def wikipedia_titles(imdb_ids, batch=250):
    titles = {}
    ids = sorted(imdb_ids)
    for start in range(0, len(ids), batch):
        values = " ".join(f'"{i}"' for i in ids[start:start + batch])
        query = ("SELECT ?imdb ?article WHERE { VALUES ?imdb { " + values + " } ?item wdt:P345 ?imdb . "
                 "?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }")
        result = get_json(WIKIDATA, urlencode({"query": query, "format": "json"}).encode())
        for row in result["results"]["bindings"]:
            article = row["article"]["value"].rsplit("/wiki/", 1)[1]
            titles.setdefault(row["imdb"]["value"], unquote(article).replace("_", " "))
        print(f"  Wikidata: {min(start + batch, len(ids)):,}/{len(ids):,} IDs, {len(titles):,} articles", flush=True)
        time.sleep(1)
    return titles


def poster_urls(titles, batch=50, size=342):
    urls = {}
    names = sorted(set(titles))
    for start in range(0, len(names), batch):
        chunk = names[start:start + batch]
        result = get_json(WIKIPEDIA + "?" + urlencode({
            "action": "query", "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": size,
            "pilicense": "any", "redirects": 1, "format": "json", "formatversion": 2, "titles": "|".join(chunk)}))
        query = result.get("query", {})
        alias = {item["from"]: item["to"] for key in ("normalized", "redirects") for item in query.get(key, [])}
        thumbs = {page["title"]: clean_url(page["thumbnail"]["source"]) for page in query.get("pages", [])
                  if "thumbnail" in page}
        thumbs = {title: url for title, url in thumbs.items() if url}
        for name in chunk:
            normalized = alias.get(name, name)  # follow normalization, then any redirect
            target = alias.get(normalized, normalized)
            if target in thumbs:
                urls[name] = thumbs[target]
        if start // batch % 20 == 0:
            print(f"  Wikipedia: {min(start + batch, len(names)):,}/{len(names):,} articles, {len(urls):,} posters",
                  flush=True)
        time.sleep(0.5)
    return urls


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/ml-32m/ml-32m.zip")
    parser.add_argument("--top-movies", type=int, default=0,
                        help="also include the N most-rated movies (use before the catalog model is built)")
    args = parser.parse_args()
    movies = catalog_movie_ids()
    if args.top_movies:
        movies |= top_movie_ids(args.archive, args.top_movies)
    links = imdb_links(args.archive)
    imdb = {movie: links[movie] for movie in movies if movie in links}
    print(f"{len(movies):,} catalog movies, {len(imdb):,} with an IMDb ID", flush=True)
    titles = wikipedia_titles(set(imdb.values()))
    urls = poster_urls(set(titles.values()))
    rows = []
    for movie, code in sorted(imdb.items()):
        title = titles.get(code)
        if title and title in urls:
            rows.append({"movie_id": movie, "imdb_id": code, "wikipedia_title": title, "poster_url": urls[title]})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["movie_id", "imdb_id", "wikipedia_title", "poster_url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows):,} poster URLs ({len(rows) / max(len(movies), 1):.1%} of the catalog) to "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

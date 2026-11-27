"""Load the official MovieLens archive for a checkout without notebook outputs."""

import hashlib
from pathlib import Path
import ssl
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import certifi
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def official_tables(raw):
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "ml-1m.zip"
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        with urlopen(URL, context=context, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("The MovieLens download was incomplete. Reload to retry.")
        with NamedTemporaryFile(dir=raw, delete=False) as temporary:
            temporary.write(payload)
        Path(temporary.name).replace(archive)
    with ZipFile(archive) as bundle:
        with bundle.open("ml-1m/ratings.dat") as source:
            history = pd.read_csv(source, sep="::", engine="python",
                                  names=["user_id", "movie_id", "rating", "timestamp"])
        with bundle.open("ml-1m/movies.dat") as source:
            catalog = pd.read_csv(source, sep="::", engine="python", encoding="latin1",
                                  quoting=3, names=["movie_id", "title", "genres"])
    history = history.loc[history.timestamp <= 978133367].sort_values("timestamp", kind="stable")
    return history.drop(columns="timestamp"), catalog

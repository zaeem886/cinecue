"""Read the Phase 1 tables with the types the README prescribes."""

import pandas as pd

SPLITS = ("train", "validation", "test")


def table_path(folder, name):
    path = folder / name
    return path if path.is_file() else folder / (name + ".gz")


def read_interactions(path):
    """Read a split or interaction table: UTC times and a nullable release year."""
    frame = pd.read_csv(path, dtype={"release_year": "Int64"})
    frame["rated_at"] = pd.to_datetime(frame["rated_at"], utc=True)
    # Four six-hour UTC blocks; a coarse submission context, not a local time of day.
    frame["daypart_utc"] = (frame["hour_utc"] // 6).astype("int8")
    return frame


def load_splits(processed):
    return {name: read_interactions(processed / f"{name}.csv") for name in SPLITS}


def load_movies(processed):
    """Catalog titles joined to the fixed genre flags, one row per movie."""
    movies = pd.read_csv(table_path(processed, "movies_clean.csv"))
    features = pd.read_csv(table_path(processed, "movie_features.csv"), dtype={"release_year": "Int64"})
    catalog = movies[["movie_id", "title", "genres"]].merge(features, on="movie_id", how="inner",
                                                           validate="one_to_one")
    assert len(catalog) == len(movies) == len(features)
    return catalog


def genre_columns(frame):
    return [column for column in frame.columns if column.startswith("genre_")]

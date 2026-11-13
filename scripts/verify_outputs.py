"""Check saved Phase 1 tables and notebook outputs. Run after all three notebooks."""

from pathlib import Path
import json

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def main():
    clean = pd.read_csv(DATA / "ratings_clean.csv")
    movies = pd.read_csv(DATA / "movies_clean.csv")
    movie_features = pd.read_csv(DATA / "movie_features.csv")
    features = pd.read_csv(DATA / "interaction_features.csv")
    splits = {name: pd.read_csv(DATA / f"{name}.csv") for name in ["train", "validation", "test"]}
    cleaning = json.loads((DATA / "cleaning_summary.json").read_text(encoding="utf-8"))
    eda = json.loads((DATA / "eda_summary.json").read_text(encoding="utf-8"))
    splitting = json.loads((DATA / "split_summary.json").read_text(encoding="utf-8"))
    assert len(clean) == len(features) == cleaning["clean_ratings"] == eda["ratings"] == splitting["total_rows"]
    assert movies.movie_id.is_unique and movie_features.movie_id.is_unique
    assert set(movies.movie_id) == set(movie_features.movie_id)
    assert clean.movie_id.isin(movies.movie_id).all()
    assert clean.user_id.gt(0).all() and clean.movie_id.gt(0).all()
    assert clean.rating.between(1, 5).all() and clean.rating.mod(1).eq(0).all()
    assert clean.user_id.nunique() == cleaning["users"] == eda["users"]
    assert clean.movie_id.nunique() == cleaning["rated_movies"] == eda["rated_movies"]
    assert len(movies) == cleaning["catalog_movies"] == eda["catalog_movies"]
    assert len(clean[["user_id", "movie_id"]].drop_duplicates()) == eda["unique_pairs"]
    assert sum(value["count"] for value in eda["rating_frequency"].values()) == len(clean)
    assert abs(sum(value["percentage"] for value in eda["rating_frequency"].values()) - 100) < 1e-8

    combined = pd.concat(splits.values(), ignore_index=True)
    assert len(combined) == len(clean) and combined.interaction_id.is_unique
    pd.testing.assert_frame_equal(combined.sort_values("interaction_id").reset_index(drop=True),
                                  features.sort_values("interaction_id").reset_index(drop=True))
    pd.testing.assert_frame_equal(features[clean.columns].sort_values("interaction_id").reset_index(drop=True),
                                  clean.sort_values("interaction_id").reset_index(drop=True))
    assert splits["train"].timestamp.max() < splits["validation"].timestamp.min()
    assert splits["validation"].timestamp.max() < splits["test"].timestamp.min()
    assert splits["train"].timestamp.max() == splitting["train_cutoff_unix"]
    assert splits["validation"].timestamp.max() == splitting["validation_cutoff_unix"]
    for entry in splitting["splits"]:
        assert entry["rows"] == len(splits[entry["split"]])
    for entry in splitting["cold_start"]:
        later = splits[entry["split"]]
        reference = splits["train"] if entry["reference"] == "train" else pd.concat([splits["train"], splits["validation"]])
        unknown_users = ~later.user_id.isin(reference.user_id)
        unknown_movies = ~later.movie_id.isin(reference.movie_id)
        assert entry["unseen_users"] == later.loc[unknown_users, "user_id"].nunique()
        assert entry["unseen_movies"] == later.loc[unknown_movies, "movie_id"].nunique()
        assert entry["rows_either_unseen"] == (unknown_users | unknown_movies).sum()

    time = pd.to_datetime(features.rated_at, utc=True)
    pd.testing.assert_series_equal(pd.to_datetime(features.timestamp, unit="s", utc=True), time, check_names=False)
    assert time.dt.hour.eq(features.hour_utc).all()
    assert time.dt.dayofweek.eq(features.weekday_utc).all()
    assert features.weekday_utc.ge(5).eq(features.is_weekend_utc.astype(bool)).all()
    assert features[splitting["genre_columns"]].isin([0, 1]).all().all()
    assert features.release_year.isna().eq(features.release_year_missing.astype(bool)).all()
    for filename in ["invalid_ratings.csv", "same_time_conflicts.csv"]:
        inspection = pd.read_csv(DATA / filename)
        expected = cleaning["checks"]["invalid_rating_rows_total" if filename.startswith("invalid") else "same_time_conflict_rows"]
        assert len(inspection) == expected

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells), path.name
        assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs), path.name
        plots = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
        if path.name.startswith("02"):
            assert plots == 5, f"Expected five inline plot figures, found {plots}"
        print(f"{path.name}: {len(code_cells)} executed code cells, {plots} inline figures, no errors")
    print(f"Verified {len(clean):,} ratings, feature tables, chronological splits, and summary counts.")


if __name__ == "__main__":
    main()

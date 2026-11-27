"""Fit the selected models once and save them for future website launches."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.artifacts import save_bundle
from cinecue.demo import fit_demo
from cinecue.data import table_path


if __name__ == "__main__":
    required = ["train.csv", "validation.csv", "movies_clean.csv", "movie_features.csv",
                "rating_summary.json", "ranking_summary.json"]
    processed = next((p for p in [ROOT / "results", ROOT / "data/processed"]
                      if all(table_path(p, name).is_file() for name in required)), None)
    if processed is None:
        raise SystemExit("Notebook outputs are required in results/ or data/processed/.")
    print(f"Fitting the validation-selected models from {processed}...", flush=True)
    bundle = fit_demo(processed)
    path = ROOT / "artifacts/recommender.pkl"
    save_bundle(path, bundle)
    print(f"Saved {path}. Future website launches load these fitted models.", flush=True)

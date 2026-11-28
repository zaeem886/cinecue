"""Save the project's result charts as image files in figures/.

1. Every chart from the executed notebooks is copied out as a PNG, so the files always
   match what the notebooks show. Re-run this after re-running a notebook.
2. Three summary charts are drawn from the saved results (results/*.json): the rating
   models, the top-10 ranking models, and the explanation removal test.

Usage: python scripts/export_figures.py
"""

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import nbformat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
# Descriptive names for each notebook's charts, in the order they appear.
NOTEBOOK_FIGURES = {
    "02_eda": ["phase1_rating_distribution", "phase1_ratings_per_user_and_movie", "phase1_popularity_concentration",
               "phase1_genres", "phase1_utc_activity"],
    "05_matrix_factorization": ["phase2_mf_tuning"],
    "06_ranking": ["phase2_rerank_tradeoff", "phase2_recommended_popularity"],
    "08_context": ["phase2_context_variants", "phase2_context_weights", "phase2_weekday_genre_lean"],
}
ACCENT, GREY = "#c8202a", "#9aa7b0"


def results(name):
    for folder in (ROOT / "results", ROOT / "data/processed"):
        if (folder / name).is_file():
            return json.loads((folder / name).read_text(encoding="utf-8"))
    raise FileNotFoundError(name)


def export_notebook_figures():
    written = []
    for stem, names in NOTEBOOK_FIGURES.items():
        notebook = nbformat.read(ROOT / "notebooks" / f"{stem}.ipynb", as_version=4)
        images = [output["data"]["image/png"] for cell in notebook.cells if cell.cell_type == "code"
                  for output in cell.outputs if "image/png" in output.get("data", {})]
        if len(images) != len(names):
            raise SystemExit(f"{stem}: expected {len(names)} charts, found {len(images)}. Update NOTEBOOK_FIGURES.")
        for name, data in zip(names, images):
            path = OUT / f"{name}.png"
            path.write_bytes(base64.b64decode(data))
            written.append(path)
    return written


def bar_chart(path, labels, values, highlight, xlabel, title, fmt, note=None):
    fig, axis = plt.subplots(figsize=(7.5, 0.48 * len(labels) + 1.3))
    colours = [ACCENT if label == highlight else GREY for label in labels]
    bars = axis.barh(labels, values, color=colours)
    axis.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, fontsize=9)
    axis.set(xlabel=xlabel, title=title, xlim=(0, max(values) * 1.18))
    if "%" in fmt:
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
    axis.invert_yaxis()
    axis.grid(alpha=0.3, axis="x")
    axis.spines[["top", "right"]].set_visible(False)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_charts():
    rating, ranking, explained = results("rating_summary.json"), results("ranking_summary.json"), \
        results("explanation_summary.json")
    test = rating["test"]
    order = ["global mean", "bias (frozen)", "bias (online)", "MF (frozen)", "MF (online)", "MF + session (online)"]
    order = [name for name in order if name in test]
    rmse = bar_chart(OUT / "phase2_rating_models_rmse.png", order, [test[n]["RMSE"] for n in order], order[-1],
                     "test RMSE in stars (lower is better)", "Rating prediction on the test split", "{:.3f}",
                     "MovieLens 1M, 100,021 test ratings. Online = profile updated with each user's earlier ratings.")
    known = ranking["test_known_users"]
    names = {"popularity": "Most popular", "highest rated": "Highest rated", "MF": "Predicted stars (MF)",
             "EASE": "EASE", "blend": "CineCue blend (final)", "blend + re-rank": "Blend + discovery re-rank"}
    keys = [key for key in names if key in known]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for axis, metric, label, fmt in [(left, "ndcg", "NDCG@10 (higher is better)", "{:.3f}"),
                                     (right, "coverage", "catalog coverage", "{:.1%}")]:
        values = [known[key][metric] for key in keys]
        bars = axis.barh([names[key] for key in keys], values,
                         color=[ACCENT if key == "blend" else GREY for key in keys])
        axis.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, fontsize=9)
        axis.set(xlabel=label, xlim=(0, max(values) * 1.22))
        if "%" in fmt:
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
        axis.grid(alpha=0.3, axis="x")
        axis.spines[["top", "right"]].set_visible(False)
    left.invert_yaxis()
    fig.suptitle(f"Top-10 recommendation on the test split ({int(known['blend']['users']):,} users known at the cutoff)")
    fig.tight_layout()
    ranking_path = OUT / "phase2_ranking_models.png"
    fig.savefig(ranking_path, dpi=150)
    plt.close(fig)
    deletion = {row["condition"]: row for row in explained["deletion"]}
    removal = bar_chart(OUT / "phase2_explanation_removal_test.png",
                        ["named movie", "random rated movie", "most negative contributor"],
                        [deletion["named movie"]["share_left_top10"], deletion["random"]["share_left_top10"],
                         deletion["most negative"]["share_left_top10"]],
                        "named movie", "share of recommendations that leave the top 10",
                        "Remove one rated movie: does the recommendation drop out?", "{:.1%}",
                        f"{explained['recommendations']:,} recommendations for {explained['users']} test users (notebook 7).")
    return [rmse, ranking_path, removal]


def main():
    OUT.mkdir(exist_ok=True)
    written = export_notebook_figures() + summary_charts()
    for path in written:
        print(f"  {path.relative_to(ROOT)}")
    print(f"Wrote {len(written)} figures to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

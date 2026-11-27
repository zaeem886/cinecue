"""Check saved Phase 2 summaries and notebook outputs. Run after notebooks 4-7."""

from pathlib import Path
import json

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
NOTEBOOKS = {"04_baselines.ipynb": 0, "05_matrix_factorization.ipynb": 1,
             "06_ranking.ipynb": 2, "07_explanations.ipynb": 0}


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def check_ratings(baseline, ratings, splitting):
    validation_rows = next(entry["rows"] for entry in splitting["splits"] if entry["split"] == "validation")
    assert baseline["validation_rows"] == validation_rows
    assert sum(row["rows"] for row in baseline["by_user_history"]) == validation_rows
    selected = min(baseline["grid"], key=lambda row: row["online RMSE"])
    assert {key: selected[key] for key in ["user_reg", "item_reg"]} == baseline["selected"]
    best_mf = min(ratings["mf_grid"], key=lambda row: row["online RMSE"])
    assert (best_mf["factors"], best_mf["reg"]) == (ratings["mf"]["factors"], ratings["mf"]["reg"])
    assert ratings["context"]["kept"] == (ratings["context"]["validation_online_gain"] >= 0.002)
    for split in ["validation", "test"]:
        rmse = {name: row["RMSE"] for name, row in ratings[split].items()}
        assert rmse["MF (online)"] < rmse["bias (online)"] < rmse["global mean"], (split, rmse)
        assert rmse["bias (online)"] < rmse["bias (frozen)"] and rmse["MF (online)"] < rmse["MF (frozen)"]
    test_rows = next(entry["rows"] for entry in splitting["splits"] if entry["split"] == "test")
    assert sum(row["rows"] for row in ratings["test_by_user_history"]) == test_rows
    assert sum(row["rows"] for row in ratings["test_by_movie_history"]) == test_rows


def check_ranking(ranking):
    selected = ranking["selected"]
    best_blend = max(ranking["blend_grid"], key=lambda row: row["ndcg"])
    assert (best_blend["ease_reg"], best_blend["alpha"]) == (selected["ease_reg"], selected["alpha"])
    grid = {row["weight"]: row for row in ranking["rerank_grid"]}
    floor = 0.95 * grid[0]["ndcg"]
    assert selected["rerank_weight"] == max(weight for weight, row in grid.items() if row["ndcg"] >= floor)
    for table in ["validation_known_users", "test_known_users", "test_all_users"]:
        for name, row in ranking[table].items():
            for metric in ["precision", "recall", "ndcg", "hit", "coverage", "long_tail_share", "genre_diversity"]:
                assert 0.0 <= row[metric] <= 1.0, (table, name, metric)
    known, everyone = ranking["test_known_users"], ranking["test_all_users"]
    new_users = ranking["test_new_users_popularity"]["users"]
    assert all(everyone[name]["users"] == known[name]["users"] + new_users for name in known)
    rerank, plain = known["blend + re-rank"], known["blend"]
    assert rerank["long_tail_share"] >= plain["long_tail_share"] and rerank["coverage"] >= plain["coverage"]


def check_explanations(explanations):
    assert explanations["max_reconstruction_error"] < 1e-8
    assert explanations["recommendations"] == explanations["users"] * 10
    deletion = {row["condition"]: row for row in explanations["deletion"]}
    assert deletion["named movie"]["mean_score_change"] < deletion["random"]["mean_score_change"]
    assert deletion["named movie"]["share_left_top10"] > deletion["random"]["share_left_top10"]
    assert deletion["most negative"]["mean_score_change"] > 0


def check_notebooks():
    for name, figures in NOTEBOOKS.items():
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells), name
        assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs), name
        plots = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
        assert plots == figures, f"{name}: expected {figures} inline figures, found {plots}"
        print(f"{name}: {len(code_cells)} executed code cells, {plots} inline figures, no errors")


def main():
    splitting = load("split_summary.json")
    check_ratings(load("baseline_summary.json"), load("rating_summary.json"), splitting)
    check_ranking(load("ranking_summary.json"))
    check_explanations(load("explanation_summary.json"))
    check_notebooks()
    print("Verified Phase 2 model selection, test results, re-ranking rule, and explanation checks.")


if __name__ == "__main__":
    main()

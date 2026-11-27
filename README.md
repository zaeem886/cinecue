# CineCue

## Interactive demo

**Windows quick start:** install Python 3.11–3.14, clone the repository, and
double-click `start-cinecue.cmd`. It prepares the virtual environment and opens
Streamlit. The first run needs internet to install dependencies and fetch the
official MovieLens 1M archive. CineCue verifies the archive, caches it locally,
and fits the models using ratings through the validation cutoff. No notebook
runs or manual file transfers are needed.

A local Streamlit interface is available in `app.py`. It uses the notebook outputs
in `results/`, with `data/processed/` as a fallback.
From this directory, use a project virtual environment and run:

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Initial fitting may take several minutes. Choose favorites under **My taste**
or select a MovieLens user. Adjust the list length or enable **Discover less popular movies**.
Each card displays a title, genres, predicted rating, and an explanation. Toggle
**Show explanations** to compare the same list with and without reasons. This
supports a survey demonstration but does not collect survey responses.

The models are cached in memory across UI interactions. They fit train plus
validation using settings in `rating_summary.json` and `ranking_summary.json`;
the demo never reads test feedback. Refreshing data files invalidates the cache.
The UI uses the final ranker, without UTC or recency ranking adjustments, and
does not claim that those experiments improved recommendations. Streamlit is an
optional dependency; `requirements-app.txt` installs only the app's NumPy, pandas,
and Streamlit dependencies. Install `requirements.txt` separately when running
notebooks. The notebook requirements and existing model code are unchanged.
Run `python -m unittest discover -s tests` to check the models and demo adapter.

For a UI smoke test, launch the app, select your favorite movies, switch users,
toggle discovery, and hide/show explanations. The title list should stay identical
when only the explanation toggle changes, and contain no movies in that user's
train/validation history.

CineCue is a context-aware and explainable movie recommendation project. Phase 1
prepares MovieLens 1M through cleaning, exploratory data analysis (EDA), and
preprocessing. Phase 2 trains and evaluates rating and ranking models, re-ranks
lists to reduce popularity bias, tests the available context, and explains each
recommendation. The code and notes are written to be read one notebook at a time.

The problem is to recommend relevant movies while accounting for limited context
and explaining suggestions. Phase 1 establishes rating quality, sparse histories,
and popularity bias; [Phase 2](#phase-2-models-ranking-and-explanations) builds on
those findings.

## Dataset

Use the official [MovieLens 1M dataset from GroupLens](https://grouplens.org/datasets/movielens/1m/).
Read the [dataset README and usage terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
before use. Research use requires acknowledgement; redistribution and commercial
use require separate permission. This project does not redistribute the archive,
raw files, or generated row-level data.

Dataset citation: F. Maxwell Harper and Joseph A. Konstan (2015),
*The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

We use `ratings.dat` (`user_id`, `movie_id`, `rating`, `timestamp`) and `movies.dat`
(`movie_id`, `title`, `genres`). The delimiter is `::`, and the files are read with
Latin-1 encoding to preserve title characters. The target is the original integer
rating from 1 to 5. We do not use the demographic file `users.dat`.

## Setup and download

Run every command from the repository root, the directory containing this README.
Use Python 3.11–3.14. Direct dependency versions, including notebook execution tools,
are pinned in [requirements.txt](requirements.txt).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in all subsequent commands. Activation is only a convenience.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
```

For offline work, copy both `ratings.dat` and `movies.dat` into `data/raw/`.
The download script also accepts an extracted `data/raw/ml-1m/` folder or
`data/raw/ml-1m.zip`. It reuses existing files and does not contact the network
when both data files are present:

```bash
python scripts/download_data.py --offline
```

If Python cannot verify the local HTTPS certificate chain, download the
[official ZIP](https://files.grouplens.org/datasets/movielens/ml-1m.zip) in a browser
and place it in `data/raw/`. On Windows, this equivalent download uses the Windows
certificate store:

```powershell
Invoke-WebRequest -Uri 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -OutFile 'data/raw/ml-1m.zip' -UseBasicParsing
python scripts/download_data.py --offline
```

Google Colab: clone the repository, change to the cloned directory with `%cd`, and
run `!python -m pip install -r requirements.txt` followed by
`!python scripts/download_data.py`. Run the execution commands below with `!` at
the start of each line, or open the notebooks individually after cloning. Install
dependencies before importing them; restart the Colab runtime if older versions
were already imported. Paths use `pathlib`, so no personal drive paths are required.

## Run Phase 1

Execute the notebooks in order. These commands save the real output cells and inline
plots back into the notebooks so they can be viewed on GitHub:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/01_cleaning.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/02_eda.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/03_preprocessing.ipynb
python scripts/verify_outputs.py
```

For interactive use, run `python -m jupyterlab`, select a kernel from the installed
environment, and use **Restart Kernel and Run All Cells** in the same notebook order.
The notebooks support working directories at either the repository root or `notebooks/`.

## Cleaning decisions

- Profile shapes, schema, raw dtypes, nulls, blank strings, exact duplicates,
  repeated user/movie pairs, and duplicate movie keys before changing data.
- Validate positive integer IDs, whole-star ratings, and nonnegative whole Unix
  seconds before March 2003, consistent with this dataset's February 2003 release.
  Parse times in UTC. Save invalid rating rows locally with source row numbers.
- Keep valid extreme ratings and high-activity users and movies. Statistical
  outlier removal would erase useful recommendation behavior.
- Keep original titles; trim and collapse whitespace in the clean title. Normalize
  genre tokens against the 18 official genres. Stop for invalid metadata or
  conflicting movie keys rather than arbitrarily selecting movie details.
- Remove exact interaction duplicates. Retain valid repeat events at distinct
  timestamps. For conflicting ratings at the same user/movie/timestamp, retain
  the first source row and save all conflict records for inspection. Never replace
  an earlier event with a later interaction.
- Validate a many-to-one left join and stop if any rating lacks movie metadata.
  Account for all excluded rows. Every check reports its affected count, including zero.

On the official data, all 1,000,209 ratings were retained. Invalid IDs, ratings,
timestamps, and metadata each affected **0 rows**. Exact rating duplicates,
repeated user/movie pairs, same-time conflicts, duplicate movie keys, and ratings
without movie metadata were also **0**. All 3,883 movie records were retained.

## Measured findings

The downloaded MovieLens files contain **1,000,209 ratings**, **6,040 active users**,
and **3,706 rated movies** out of **3,883 catalog movies**. The other 177 catalog
movies have no ratings. Rating activity spans 2000-04-25 23:05:32 UTC to
2003-02-28 17:49:50 UTC. The notebooks calculate these findings from the data:

| Stars | Ratings | Share |
| --- | ---: | ---: |
| 1 | 56,174 | 5.62% |
| 2 | 107,557 | 10.75% |
| 3 | 261,197 | 26.11% |
| 4 | 348,971 | 34.89% |
| 5 | 226,310 | 22.63% |

- **Positive skew:** 57.52% of ratings are 4 or 5; the mean is 3.582/5. Phase 2
  should compare against a simple rating baseline. Unrated movies are not negative examples.
- **Uneven activity:** the median user has 96 ratings; the median rated movie has
  123.5 ratings. Report future performance for sparse histories and infrequent movies.
- **Sparse feedback:** 95.53% of cells are empty in the 6,040 × 3,706 active
  user/movie matrix, counting unique pairs rather than repeat events. Including all
  3,883 catalog movies increases sparsity to 95.74%.
- **Popularity concentration:** the top 371 movies (rounded-up top 10% of rated
  movies) receive 44.45% of rating events. Phase 2 should examine popularity bias
  and catalog coverage.
- **Overlapping genres:** Drama is the largest catalog genre, with 1,603 movies.
  Genre mean ratings range from 3.215 for Horror (76,386 rating events) to 4.075
  for Film-Noir (18,261). One multigenre interaction can enter several groups;
  full-data genre means must not become Phase 2 features.
- **Uneven UTC activity:** 21:00–21:59 UTC is the busiest hour, with 65,491 ratings;
  Monday has the most submissions, with 173,931. Weekend activity accounts for
  25.36%. These describe submission behavior and cannot establish viewing habits.

## Features and target

| Field | Role |
| --- | --- |
| `user_id`, `movie_id` | Categorical identities; do not scale as continuous numbers. |
| `rating` | Target: original integer 1–5, unchanged. |
| `release_year` | Final parenthesized year in the title, restricted to 1800–2003; nullable. |
| `release_year_missing` | Binary indicator for an unsupported or missing release year. |
| 18 `genre_*` columns | Multihot genre flags, 0 or 1; no scaling needed. |
| `hour_utc`, `weekday_utc` | Categorical rating-submission context; hour 0–23, Monday=0 through Sunday=6. |
| `is_weekend_utc` | Binary Saturday/Sunday flag; no scaling needed. |
| `interaction_id`, `source_row` | Audit fields only, never predictive features. |
| `timestamp`, `rated_at` | Audit and time ordering; Unix seconds and UTC dates. |

There are no session features. Any future sessions must be labelled rating-session
proxies. No continuous transformer is needed in Phase 1. If a future model needs
release-year imputation or scaling, fit it on training rows only and apply it unchanged
to validation and test. Fit learned categorical encoders on training rows with an
explicit unknown-category policy. CSV readers should parse `rated_at` in UTC and
`release_year` with pandas nullable `Int64`.

The split is global and chronological, aiming for 80/10/10. Cutoffs use the nearest
cumulative timestamp-group size to 80% and 90%, with ties resolved toward the earlier
group. Equal timestamps always stay in the same split. Unseen users and movies stay
in later splits. Full-data EDA aggregates are never used as model inputs. Any future
history or popularity feature must use events strictly before the predicted interaction.

The realized split is:

| Split | Rows | Share | First rating (UTC) | Last rating (UTC) |
| --- | ---: | ---: | --- | --- |
| Train | 800,168 | 80.00008% | 2000-04-25 23:05:32 | 2000-12-02 14:52:18 |
| Validation | 100,020 | 9.99991% | 2000-12-02 14:52:28 | 2000-12-29 23:42:47 |
| Test | 100,021 | 10.00001% | 2000-12-29 23:43:34 | 2003-02-28 17:49:50 |

The inclusive Unix-second cutoffs are `975768738` for training and `978133367`
for validation. Equal timestamps do not cross boundaries. All 1,000,209 interactions
are present exactly once across the splits.

| Later split | Reference data | Unseen users | Unseen movies | Rows with either unseen |
| --- | --- | ---: | ---: | ---: |
| Validation | Train | 611 | 16 | 76,180 |
| Test | Train | 200 | 33 | 19,413 |
| Test | Train + validation | 29 | 28 | 4,298 |

These users and movies remain in the files. Phase 2 will need cold-start fallbacks,
especially for validation, where many users arrive after the training cutoff.
All release years were parseable in this dataset, so both release-year missing
counts are zero; the nullable field and missing indicator still support other inputs.

## Limitations

These are voluntary ratings, not a random sample of all viewers or all watched movies.
The dataset is old, and the catalog and users may differ from a current service.
Missing feedback does not mean dislike. Genre groups overlap, and genre averages
are associations rather than causal effects. We omit correlations of numeric IDs,
Unix seconds, and cyclic time categories because these would be misleading.

Timestamps describe **rating activity**, not watch times. Local time zones, moods,
viewing sessions, and historical catalog availability are not available. Release year
does not establish when a title entered the service. UTC context should therefore
be treated as a limited signal in Phase 2.

## Verification

All three notebooks were executed in order on the official data using Windows and
Python 3.14.8 with the pinned dependencies. All 17 code cells finished without error
outputs, and the EDA notebook contains five inline figures. The saved outputs are
included in the notebooks for GitHub viewing.

`scripts/verify_outputs.py` checks saved table schemas and counts, unchanged targets,
feature values, source-to-feature row preservation, split row conservation, strict
time boundaries, cold-start counts, summary consistency, and executed notebook
outputs. Additional small input checks covered invalid values, repeated events,
conflicts, accented text, missing metadata, malformed records, missing years, and
timestamp ties. Local-file, extracted-folder, and ZIP offline download paths passed.
Installed dependency versions match `requirements.txt`, and `pip check` passed.
The official source, terms, and download links were checked; the DOI is the citation
provided by GroupLens. Git ignore checks exclude the archive and generated data.

On this Windows setup, Jupyter prints a Proactor event-loop warning and starts an
extra selector thread; the notebooks still complete and pass output verification.
There are no outstanding Phase 1 blockers. Linux and Colab commands use portable
paths but were not executed in this verification environment.

## Phase 2: models, ranking, and explanations

Run Phase 1 first, then execute the Phase 2 notebooks in order from the repository
root. Notebook 5 takes about seven minutes; the others take one to two minutes each.

```bash
python -m unittest discover -s tests
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 notebooks/04_baselines.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 notebooks/05_matrix_factorization.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 notebooks/06_ranking.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 notebooks/07_explanations.ipynb
python scripts/verify_phase2.py
```

Shared model code lives in `src/cinecue/`. Notebooks add `src/` to the import path, so
no installation step is needed and `requirements.txt` is unchanged. Everything uses
NumPy and pandas only.

### Protocol

- Every model is fitted on `train.csv` and tuned on `validation.csv`. Settings are then
  fixed, models are refitted on train + validation, and `test.csv` is scored. Test
  differences carry 95% paired bootstrap intervals over users (2,000 draws).
- Rating models are scored two ways. **Frozen** profiles stop at the training cutoff.
  **Online** profiles also absorb the same user's ratings at strictly earlier timestamps
  in the scored split, while movie parameters stay fixed; this mirrors a live service.
  Ratings that share a timestamp never inform one another. Online RMSE selects settings.
- Top-10 lists are built at the cutoff from movies the user has not rated. A hit is a
  movie the user rates 4–5 stars in the next period. Movies with fewer than 20 history
  ratings are never recommended. Users new at the cutoff get the popularity list and are
  reported separately.
- Decision rules were set before results were seen: a context feature is kept only if it
  lowers online validation RMSE by ≥ 0.002; the re-ranking weight is the largest that
  keeps ≥ 95% of validation NDCG@10.
- **Revision:** session context and recency weighting were added in a second round,
  selected on validation, and the test split was scored again. Recency was then left out
  of the final ranker because it lowered test NDCG@10 reliably. That one decision used
  the test split; the final ranker's settings and test result are the same as in the
  first round.

### Rating prediction (notebooks 4 and 5)

| Model (test split, 100,021 ratings) | Frozen RMSE | Online RMSE | Online MAE |
| --- | ---: | ---: | ---: |
| Global mean | 1.089 | 1.089 | 0.903 |
| Bias model (user + movie biases) | 0.920 | 0.894 | 0.701 |
| Matrix factorization (64 factors, ALS) | 0.893 | 0.851 | 0.667 |
| **MF + rating-session context** | – | **0.841** | **0.659** |

- MF beats the bias model on both splits (test: −0.043 RMSE, 95% interval −0.047 to
  −0.039). The online protocol matters most for users who were new at the cutoff (76%
  of validation ratings); for them, MF error falls by 0.10 stars on test.
- **Rating-session context (kept).** A rating session is a rating-session proxy:
  consecutive ratings by one user with no gap over one hour (validation has 2,115, with
  a median of 15 ratings). Each rating gets the shrunken mean error of the user's
  strictly earlier ratings in the same session, so a user rating above their profile
  today is predicted higher. It lowers online validation RMSE by 0.0036 (0.013 for users
  with a training history) and test RMSE by 0.010 (−0.013 to −0.008).
- **UTC context (not kept).** User × weekend and movie × six-hour UTC offsets lower
  online validation RMSE by only 0.0003, below the threshold. UTC submission time is a
  weak proxy for a user's situation.

### Top-10 recommendation (notebook 6)

Test split, 1,158 users known at the cutoff:

| Model | Precision@10 | NDCG@10 | Hit rate | Coverage | Long-tail share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.186 | 0.201 | 0.624 | 4.0% | 0.0% |
| Highest rated (movie bias) | 0.075 | 0.065 | 0.390 | 1.4% | 65.0% |
| MF predicted stars | 0.112 | 0.117 | 0.497 | 16.0% | 60.8% |
| EASE | 0.202 | 0.223 | 0.652 | 8.1% | 0.2% |
| **Blend** (0.25 · EASE + 0.75 · MF, z-scored) | **0.233** | **0.258** | **0.687** | 8.3% | 3.1% |
| Blend + popularity re-rank (w = 1.5) | 0.212 | 0.235 | 0.667 | 12.9% | 18.1% |

- Ranking by predicted stars performs poorly: rating accuracy does not translate into
  good top-N lists. EASE (Steck, 2019) learns from which movies are rated together and
  edges out popularity; blending it with MF beats both parts. Blend − popularity:
  +0.057 NDCG@10 (+0.047 to +0.066); blend − EASE: +0.035 (+0.026 to +0.043).
- Re-ranking the blend's top 100 by score − w · popularity raises long-tail exposure
  from 3% to 18% and coverage from 8% to 13%, at a reliable cost of −0.022 NDCG@10
  (−0.026 to −0.018). It kept 96.5% of NDCG@10 on validation but 91.4% on test.
- **Recency context (tested, not used).** Weighting each rated movie by
  0.5^(age / 3 days) in EASE raised validation NDCG@10 from 0.150 to 0.162, but lowered
  test NDCG@10 by 0.015 (−0.023 to −0.006). Scoring the same lists on the first 7, 28,
  90 and 365 test days shows +0.020 at 7 days (borderline) shrinking to a loss over the
  full two years: recency predicts what a user rates next, not over a long horizon.
- The 29 test users new at the cutoff receive popularity lists (NDCG@10 0.459).

### Explanations (notebook 7)

The blended score is linear in the user's history: an exact constant + a general-appeal
term + one term per rated movie (`src/cinecue/explain.py`). Each explanation names the
liked (4–5★) movies with the largest terms, the shared genres, MF's predicted rating, and
whether re-ranking moved the movie up. On 2,000 test recommendations for 200 users:

- The terms rebuild every score to within 2.2 × 10⁻¹⁵.
- Removing the named movie and refitting the profile lowers the score by 0.087 on
  average and drops the movie from the top 10 in 15.6% of cases. Removing a random rated
  movie drops it in 3.1%. Removing the most negative contributor raises the score.
- Influence is spread out: the named movie supplies a median 4% of the positive push,
  so explanations name the strongest contributors, not a sole cause.

### Phase 2 limitations

- Relevance is inferred from later ratings. Unrated movies are unknown, not disliked,
  so ranking metrics are conservative and favor movies people tend to rate.
- One chronological split is used. Bootstrap intervals cover sampling noise over users,
  not variation across different cutoffs. The test period spans two years, so absolute
  values are not comparable with random splits.
- The validation window (four weeks) is much shorter than the test window (two years),
  which is why recency looked better on validation than it was on test.
- Explanations describe the model's computation, not users' motives. Their usefulness
  to people would need a user study.
- `scripts/verify_phase2.py` checks model selection against the saved grids, the
  pre-set decision rules, test-result orderings, explanation exactness and deletion
  results, the bootstrap intervals, and executed notebook outputs. `tests/` holds 23
  synthetic unit tests for the models, online updates, session offsets, recency weights,
  the bootstrap, ranking metrics, and explanations.

## Repository layout

```text
cinecue/
  README.md
  requirements.txt
  .gitignore
  scripts/
    download_data.py
    verify_outputs.py
    verify_phase2.py
  src/cinecue/              # Phase 2 models, ranking, explanations
    data.py
    models.py
    ranking.py
    evaluation.py
    explain.py
  tests/
    test_phase2.py
  notebooks/
    01_cleaning.ipynb
    02_eda.ipynb
    03_preprocessing.ipynb
    04_baselines.ipynb
    05_matrix_factorization.ipynb
    06_ranking.ipynb
    07_explanations.ipynb
  data/
    raw/                    # local download or existing files
    processed/              # local generated tables and summaries
```

The processed directory contains `ratings_clean.csv`, `movies_clean.csv`,
`invalid_ratings.csv`, `same_time_conflicts.csv`, `movie_features.csv`,
`interaction_features.csv`, `train.csv`, `validation.csv`, `test.csv`, and the
aggregate `cleaning_summary.json`, `eda_summary.json`, and `split_summary.json`.
Phase 2 adds the aggregate `baseline_summary.json`, `rating_summary.json`,
`ranking_summary.json`, and `explanation_summary.json`.
Only directory placeholders are tracked under `data/`. Figures remain inline.

Suggested group milestones:

1. `Add MovieLens download setup and cleaning notebook`
2. `Explore ratings, popularity, genres, and UTC activity`
3. `Prepare context features and chronological data splits`

# Temporal Serial Dictator: Code Supplement

> This codebase was generated with Claude Code in manual mode.

This repository implements and evaluates the **serial dictator rule** for
**temporal (perpetual) approval voting** — a setting where the same group of
voters faces a *sequence* of decisions over time (rounds), rather than a
single one-shot election. In each round, one voter (the "dictator" for that
round) is picked from a fixed rotation and the winning alternative is chosen
from among their approved alternatives.

The main question this code answers empirically is whether decision
sequences produced by the serial dictator rule satisfy **Proportional
Justified Representation (PJR)** — a proportionality guarantee from the
committee-voting literature, adapted here to the temporal setting: any
sufficiently large, sufficiently cohesive group of voters should be
"satisfied" (see at least one of their approved alternatives win) often
enough, in proportion to their size and how often they agree.

The repository generates **synthetic** temporal voting instances (spatial /
Euclidean preference models) and also runs the same experiments on
**real-world data** (Eurovision Song Contest voting history, iPhone App
Store rankings, Spotify charts), then brute-force verifies PJR on the
resulting decision sequences.

---

## Repository layout

```
src/
  synthetic_data_tools/   Synthetic instance generation + the paper's synthetic experiment grid
  real_data_tools/        Real-dataset loading + the paper's real-data experiment grid
  data_transformation/    Converts raw .tsoi datasets to JSONL; downloads the raw data
  voting_rules/           The SerialDictator rule itself
  verification/           Brute-force PJR verification (sequential + multiprocess)
  encoding/                Saving/loading approval profiles and decisions as JSON
  eda/                    Jupyter notebooks exploring the experiment results
  utils/, _typing.py      Small shared helpers and type definitions
tests/                    Automated test suite (pytest)
aaai_experiments/         Pre-computed result archives (checked into git, ready to explore)
experiments/              Where freshly-run experiments are saved (not checked in)
real_data/                Real-world datasets, raw and converted (not checked in; see below)
lackner_perpetual_voting/ Vendored reference copy of Martin Lackner's original codebase (not used by src/)
```

Every script under `src/` is also runnable directly with `python -m`, and
every one of the common workflows has a corresponding one-line `task`
command (see the [task reference](#task-reference) table below).

---

## Getting started

These steps take you from "just downloaded this" to "exploring the paper's
results in a notebook." They assume no prior familiarity with Python
packaging tools.

If you downloaded this as a `.zip`/`.tar.gz` archive rather than via
`git clone`, that's completely fine — everything below works the same way.
The only difference is you won't have a `.git` folder, so anything
git-related (e.g. the optional `task hooks` step for contributors) doesn't
apply to you; everything needed to run and explore the experiments works
without git.

### 1. Check your Python version

You need **Python 3.11 or newer**. Check with:

```
python3 --version
```

If you need to install/upgrade Python, see [python.org/downloads](https://www.python.org/downloads/).

### 2. Install Task (recommended, but optional)

This project uses [Task](https://taskfile.dev/) — a small command runner —
so that every workflow is one short command (e.g. `task install`) instead of
a long one you'd have to remember or copy from documentation.

Install it by following the instructions for your OS at
**[taskfile.dev/installation](https://taskfile.dev/installation/)**, then
confirm it worked:

```
task --list
```

**If you'd rather not install anything new**, every `task` command in this
README has a manual, plain-Python equivalent — see the
[task reference](#task-reference) table.

### 3. Set up the Python environment and install dependencies

With Task:

```
task install
```

Without Task:

```
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

This creates a virtual environment in `.venv` and installs the project plus
its development tools (test runner, linter). If you used Task, you don't
need to activate the virtual environment yourself — every `task` command
does that internally.

### 4. Install notebook dependencies (needed for step 6)

The result-exploration notebooks need a few extra packages (pandas,
matplotlib, Jupyter) that aren't installed by step 3, to keep the base
install light for people who only want to run experiments.

```
.venv/bin/pip install -e '.[eda]'
```

### 5. Get the data

You have two options, and **you can start with the first one and do
nothing else** if you just want to explore the paper's actual results:

- **Already included, no download needed:** `aaai_experiments/` contains two
  pre-computed result archives (~40MB total) — the full synthetic
  experiment grid and the full real-data experiment grid, exactly as
  reported in the paper. The notebooks in step 6 read directly from these
  archives.
- **Full real-world raw data (~270MB download, ~630MB once converted):**
  needed only if you want to regenerate/extend the real-data experiments
  yourself, or run the Spotify-specific data-characteristics notebook. Get
  it with:

  ```
  task download-and-convert-data
  ```

  or manually:

  ```
  .venv/bin/python -m src.data_transformation.download_and_convert_data
  ```

  This downloads the real-world dataset collection, extracts it into
  `real_data/`, and converts it to the JSONL format the experiment scripts
  use. It's safe to re-run — it skips the download if the data is already
  present.

### 6. Explore the results

Start Jupyter and open one of the notebooks in `src/eda/`:

```
.venv/bin/jupyter lab
```

- **`pjr_eda.ipynb`** — synthetic experiment grid: violation rates by
  configuration, distributions, worst cases.
- **`real_data_eda.ipynb`** — the same analysis for the real-data grid,
  plus a per-dataset breakdown and a direct comparison against the
  synthetic grid's findings.
- **`spotify_eda.ipynb`** — a deeper look at *why* PJR verification is slow
  on the Spotify datasets specifically (candidate counts, approval set
  sizes, backfilled/indifferent voters). Requires the full data from
  step 5's second option.

### 7. (Optional) Run experiments yourself

See the [task reference](#task-reference) below for the full list. The
short version:

```
task aaai-experiment NUM_EXPERIMENTS=1000        # synthetic experiment grid
task natural-experiment NUM_EXPERIMENTS=100      # real-data experiment grid
task multi-verify-pjr DIR=<experiment directory>  # verify PJR on the results
```

### 8. (Optional, for contributors) Run the checks

```
task check     # lint + typecheck + tests
```

---

## Task reference

Every task assumes you've already run `task install` (or the manual
equivalent from step 3). Variables like `T=20` are optional — each has a
sensible default shown in `task --list`.

| Task | What it does | Manual equivalent |
|---|---|---|
| `task install` | Create `.venv` and install dependencies | `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'` |
| `task download-and-convert-data` | Download real-world datasets and convert to JSONL | `.venv/bin/python -m src.data_transformation.download_and_convert_data` |
| `task synthetic-experiment T=20 N=20 M=5 THRESHOLD=1.5 NUM_EXPERIMENTS=100` | Run one synthetic configuration | `.venv/bin/python -m src.synthetic_data_tools.run_synthetic_experiment 20 --n 20 --m 5 --approval-threshold 1.5 --num-experiments 100` |
| `task aaai-experiment NUM_EXPERIMENTS=1000` | Run the full synthetic experiment grid (paper) | `.venv/bin/python -m src.synthetic_data_tools.aaai_experiment --num-experiments 1000` |
| `task natural-experiment NUM_EXPERIMENTS=100` | Run the full real-data experiment grid (paper) | `.venv/bin/python -m src.real_data_tools.real_aaai_grid_experiment --num-experiments 100` |
| `task verify-pjr DIR=<name>` | Verify PJR for a saved experiment (sequential) | `.venv/bin/python -m src.verification.verify_pjr experiments/<name>` |
| `task multi-verify-pjr DIR=<name> MAX_WORKERS=8` | Verify PJR using multiple CPU cores | `.venv/bin/python -m src.verification.multiprocessing_verify_pjr experiments/<name> --max-workers 8` |
| `task clean-experiments` | Delete everything under `experiments/` | `rm -rf experiments/*` |
| `task clean-json-datasets` | Delete the converted real-world JSONL data | `rm -rf real_data/json_datasets` |
| `task lint` / `task typecheck` / `task test` / `task check` | Code-quality checks (for contributors) | `.venv/bin/ruff check .` / `.venv/bin/mypy .` / `.venv/bin/pytest` |

`DIR` for the verification tasks defaults to the most recently created
experiment directory if omitted.

---

## Reproducibility notes

- **Synthetic experiments are fully seeded.** Every run's instance
  generation and every serial dictator's sequence of choices are derived
  from a single recorded seed (stored in each run's `metadata.json`), so
  re-running with the same seed reproduces the exact same output — verified
  byte-for-byte as part of this project's own test suite.
- **Real-data experiments are not seeded.** Which voters and which
  contiguous window of rounds get sampled is randomized fresh each run
  (except each configuration's designated "first rounds" run, which always
  uses the dataset's actual first rounds and is flagged `is_first_rounds` in
  its metadata). The results in `aaai_experiments/` are a fixed, static
  snapshot and are re-verifiable from that snapshot, but re-running
  `real_aaai_grid_experiment.py` from scratch will sample a different
  random subset than what's archived.
- **One real-world dataset is silently incomplete:** `i_phone-news-paid_news_tsoi`'s
  raw source files are corrupted (upstream, not something introduced by
  this codebase — verified against a fresh download of the original data).
  The conversion step reports and skips it; every other dataset converts
  and runs normally.

---

## License and attribution

This project is MIT-licensed (see `LICENSE`). The core synthetic-instance
generation logic in `src/synthetic_data_tools/` is ported from Martin
Lackner's perpetual voting codebase (also MIT-licensed; see
[martin.lackner.xyz](http://martin.lackner.xyz/) and the accompanying
paper, *Perpetual Voting: Fairness in Long-Term Decision Making*, AAAI
2020); a reference copy of that original codebase is kept in
`lackner_perpetual_voting/` for comparison but is not imported by anything
in `src/`. The real-world `.tsoi` datasets are the same collection that
codebase uses, hosted at TU Wien
([dbai.tuwien.ac.at/proj/sudema](https://www.dbai.tuwien.ac.at/proj/sudema/data/data.zip)).

# Approach A: context-aware trust scoring for retrieved passages

This repository contains the source for Approach A, a lightweight defence against indirect prompt injection in retrieved evidence. It scores sentence or line spans using their text and neighbouring context, then either removes flagged spans or rejects the containing passage before answer generation.

This is a **source-only copy** of the existing implementation. Original Python files, dependency files and the experiment protocol are preserved unchanged. Datasets, fitted detectors, generated answers, caches, evaluation outputs and research reports are not included. Approach B and the shared OOD comparison are also excluded.

## What is included

| Location | Purpose |
| --- | --- |
| `rag_study/` | Approach A detector, training, operating-point selection, synthetic case construction, evaluation and experiment runner. |
| `rag_screen/` | Shared retrieval, token budgeting, generation, editing and metric utilities imported by Approach A, together with their existing compatibility interfaces. |
| `scripts/` | Public asset download, run analysis and integrity-audit source. These scripts are code, not precomputed results. |
| `tests/` | Existing checks for span edits, data integrity, scoring, grouping and runner behaviour. |
| `FOLLOWUP_PROTOCOL.md` | Original experiment plan retained as an input to source fingerprinting. It also documents historical baseline dependencies. |
| `requirements.txt` | Supported dependency ranges. |
| `requirements.lock.txt` | Exact dependency versions captured from the original environment. |
| `pyproject.toml` | Original package metadata and test configuration. |
| `docs/SETUP.md` | Asset prerequisites, core API use and limitations of replaying the complete experiment. |

Historical module names, paths and baseline adapters remain in the source so that this copy does not silently alter the experimental method. `rag_screen` is a dependency of Approach A; it is not Approach B. The package metadata retains the original name `rag-screen-experiment`. The primary Approach A command interface is `python -m rag_study`.

## Install

Use Python 3.11 or newer. Run commands from this repository's root, in a new environment rather than the original experiment directory.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For an environment matching the original dependency snapshot, use `requirements.lock.txt` instead of `requirements.txt`. This is an exact version list, not a lockfile containing distribution hashes or a guarantee that every version is available on every platform. No editable installation is required when commands run from the repository root. If desired, install the package after its dependencies with `python -m pip install --no-deps -e .` using the environment's Python.

The remaining examples use `python` to mean this environment's interpreter. Activate the environment or replace it with the full executable path above.

## Check the code without running an experiment

```text
python -m rag_study --help
python -m rag_screen --help
```

Some existing tests use the Qwen tokenizer. Obtain only that public asset using the short command in [SETUP.md](docs/SETUP.md#tokenizer-for-local-tests), then run:

```text
python -m pytest -q -p no:cacheprovider
```

The tests use a fake generator where generation is needed; Ollama is not required for these checks. Two checks intentionally skip when the excluded prepared research dataset is absent. Passing these checks validates code behaviour, not the effectiveness of the defence on a research benchmark.

## Using the defence

The main learned components are `ContextDetector` and `local_repair` in `rag_study.models`. The detector combines word and character TF-IDF features, logistic classification and a sigmoid calibration stage. Its passage risk is the maximum span score, rather than a separately calibrated probability that the entire document is malicious.

The same detector and selected threshold support two actions:

- `local_repair(..., whole=False)`: remove detected spans, rescan, and reject a passage if risk remains after two deletion passes.
- `local_repair(..., whole=True)`: reject a passage containing a detected span.

A trained checkpoint and its matching policy are required to use the learned detector. They are excluded from this upload. [SETUP.md](docs/SETUP.md) shows the exact API and how to train from privately supplied, correctly prepared inputs. The keyword baseline can be used without a checkpoint, but it is not the learned defence.

Ollama is needed only for real answer generation. The study configuration uses `qwen2.5:7b` at `http://127.0.0.1:11434`. Obtain the model locally with `ollama pull qwen2.5:7b` and start the service if it is not already running. Model weights are not part of this repository.

## Complete experiment prerequisites

**Installing dependencies and downloading public data does not by itself reproduce the complete archived study.** The unchanged preparation command uses external historical source inventories to exclude previously used material. The unchanged full runner also loads historical baseline checkpoints and source outside this repository. Those assets are not Approach B, but are not bundled in this source-only copy.

Read [the full setup notes](docs/SETUP.md#complete-experiment-prerequisites) before using `prepare`, `development`, `test` or `contrast`. The core detector and repair API do not require those historical baselines. Run any new work in this clone with separate outputs; do not use it to resume or overwrite the original experiment.

## Upload to GitHub

Upload this folder's contents as the root of a new repository, including the dotfile `.gitignore`. Do not add the surrounding workspace or `packaging_work` directory. The ignore rules exclude local data, checkpoints, generated outputs, reports, environments and caches. Before any later commit, inspect the staged files so that only intended source and documentation are published.

No software licence has been added to this copy. Repository owners can choose an appropriate licence before publishing; public datasets and model assets retain their own upstream terms.

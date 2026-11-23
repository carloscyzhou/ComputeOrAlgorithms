# Data and AI disclosure statement

## Data

All data comes from Epoch AI and is used under the Creative Commons Attribution 4.0 licence (CC BY 4.0):

- *Data on Notable AI Models*, https://epoch.ai/data/notable_ai_models.csv, downloaded 20 November 2026.
- *Capabilities & Benchmarking* (`mmlu_external.csv` and `epoch_capabilities_index/eci_scores.csv` inside https://epoch.ai/data/benchmark_data.zip), downloaded 23 October 2026.

The raw files are stored unmodified in `data/raw/`. Every figure on screen is computed from them by `make_data.py`, `analysis.py` and `app.py`, as documented in `DATA_DICTIONARY.md`. `verify.py` regenerates the processed files and recomputes key figures independently. No data values were typed in by hand or supplied by an AI tool.

## AI use

The group used generative AI tools as drafting assistants. Each substantive use is logged, with its verification, in `PROVENANCE.md`.

By member:

- **Chenyu Zhou** — *Claude (Anthropic, claude.ai chat)*: finding the Epoch AI benchmark datasets; drafting the Python for cleaning them, the compute trend and two-era search, the capability model, bootstrap intervals, the iso-capability map, replay and waterfall charts. *Claude Code (Anthropic)*: integrating the four Part 1 prototypes into this repository with every member's charts; drafting the README, data dictionary, `verify.py`, the provenance matrix and this statement; running the git commits and push to GitHub and filling in the Streamlit Community Cloud deployment form, after Chenyu Zhou signed in to both accounts and authorised them. No other AI tools.
- **Aksha Mansuri** — *Claude (Anthropic)*: helping build the Part 1 Streamlit dashboard (charts, filters, tiles, on-screen answer and limits text), the data pipeline and its independent check, fixing errors, drafting documentation, and an early brainstorm of candidate questions that was not used. The persona (an Australian Government funding adviser) and the design perspective — which charts to use and which to reject, with the reasons — were Aksha's own ideas; Claude did not suggest those rejections.
- **Citra Yulianti** — *Claude (Anthropic)*: helping write the Part 1 Python pipeline (`src/data.py`), the analysis functions (`src/analysis.py`) and the Streamlit code, finding a suitable dataset, and suggesting persona options. Citra chose a BRIN persona for her Part 1 prototype; the group later agreed on the Australian Government funding adviser.
- **Yanzhen Huang** — 

AI output was treated as a draft. Group members ran the code on the real Epoch files, checked on-screen numbers against the raw CSVs (see the "How verified" column of `PROVENANCE.md`), and edited the text before committing it.

## What AI did and did not do

AI did not supply, alter or invent any data values: every raw file was downloaded from Epoch AI and is stored unmodified, and every number on screen is computed from it. AI did not submit the assignment. Claude Code did carry out the GitHub push and the Streamlit deployment steps on Chenyu Zhou's instruction; Chenyu Zhou signed in to both accounts, approved them and checked the result. The final question and persona come from Aksha Mansuri's Part 1 prototype and were agreed by the group; AI-suggested questions and persona options were not adopted.

## Limits

Epoch's list covers notable models, not all models, and compute values are estimates. Many recent models have no compute estimate. Benchmark scores cover 2021–2026 only, and release year in the capability model stands in for all non-compute improvements (algorithms, data, post-training). The split between compute and algorithms is therefore an estimate with a wide interval, not a measurement.

# Data dictionary

Three raw source samples (Epoch AI, CC BY 4.0) and six processed files. The app reads only `data/processed/`. `python make_data.py` rebuilds every processed file from `data/raw/`; `python verify.py` checks that the rebuild matches the committed files byte-for-byte (meta.csv without its run date) and recomputes key figures from the raw files independently.

Licence for every dataset: **Creative Commons Attribution 4.0 (CC BY 4.0)**, credit Epoch AI (see `data/raw/EPOCH_LICENSE.md`). Processed files are adaptations by Group 4 under the same licence.

---

## Raw source samples (`data/raw/`, stored unmodified)

### R1. `notable_ai_models.csv`
| Field | Detail |
|---|---|
| Source | Epoch AI, *Data on Notable AI Models* |
| URL | https://epoch.ai/data/notable_ai_models.csv (documentation: https://epoch.ai/data/ai-models-documentation) |
| Licence | CC BY 4.0 |
| Downloaded | 20 November 2026 |
| Size | 1,102 rows × 47 columns |
| Columns used (units) | `Model` (text); `Organization` (text, may list several, comma-separated); `Country (of organization)` (text, may list several); `Domain` (text, may list several); `Publication date` (YYYY-MM-DD); `Training compute (FLOP)` (floating-point operations, Epoch estimate; blank for 556 rows); `Confidence` (Confident / Likely / Speculative / …); `Frontier model` (True/False: in the top 10 by training compute when released). All other columns are unused. |
| Primary key | `Model` (all 1,102 values unique) |
| Time coverage | 1950-07-02 to 2026-11-10 |
| Latest observation | 2026-11-10 |
| Transform | None (saved as downloaded) |

### R2. `mmlu_external.csv`
| Field | Detail |
|---|---|
| Source | Epoch AI, *Capabilities & Benchmarking* — externally reported MMLU results |
| URL | https://epoch.ai/data/benchmark_data.zip (file `mmlu_external.csv` inside the zip); page: https://epoch.ai/benchmarks |
| Licence | CC BY 4.0 |
| Downloaded | 23 October 2026 |
| Size | 249 records × 13 columns (some text fields contain line breaks) |
| Columns used (units) | `Model version` (text); `EM` (exact-match accuracy, 0–1); `Shots` (number of in-context examples: 0, 2, 5 or blank); `Release date` (YYYY-MM-DD); `Training compute (FLOP)` (FLOP); `Organization` (text) |
| Primary key | `id` (unique per reported result; one model version can have several reports) |
| Time coverage | 2021-08-05 to 2024-12-26 (release dates) |
| Latest observation | 2024-12-26 |
| Transform | None (extracted from the zip unchanged) |

### R3. `eci_scores.csv`
| Field | Detail |
|---|---|
| Source | Epoch AI, *Epoch Capabilities Index* |
| URL | https://epoch.ai/data/benchmark_data.zip (file `epoch_capabilities_index/eci_scores.csv`); page: https://epoch.ai/benchmarks/eci |
| Licence | CC BY 4.0 |
| Downloaded | 23 October 2026 |
| Size | 270 rows × 11 columns |
| Columns used (units) | `Model` (text, same naming as R1); `Display name` (text); `eci` (index points, no natural unit); `eci_ci_low`, `eci_ci_high` (index points, lower and upper bounds of the confidence interval reported by Epoch); `date` (release date, YYYY-MM-DD); `Organization` (text) |
| Primary key | `Model` (all 270 values unique) |
| Time coverage | 2023-02-24 to 2026-10-15 |
| Latest observation | 2026-10-15 |
| Transform | None (extracted from the zip unchanged) |

---

## Processed files (`data/processed/`, read by `app.py`)

### P1. `models_clean.csv` — notable models with a compute estimate
| Field | Detail |
|---|---|
| Built from | R1 by `make_data.py` → `analysis.clean_models` |
| Size | 546 rows × 11 columns |
| Columns (units) | `model` (text); `organization`, `country`, `domain` (text, **first listed** value only, blank → "Unspecified"); `publication_date` (YYYY-MM-DD); `year` (calendar year); `decimal_year` (year + (day of year − 1)/365.25); `training_compute_flop` (FLOP); `log10_flop` (log10 of FLOP, no unit); `confidence` (text, blank → "Not stated"); `frontier_model` (True/False) |
| Primary key | `model` |
| Time coverage | 1950-07-02 to 2026-11-05 |
| Latest observation | 2026-11-05 |
| Transform | (a) parse `Publication date`, drop rows that fail (0 dropped); (b) coerce `Training compute (FLOP)` to a number and keep values > 0 (1,102 → 546); (c) keep first-listed organisation, country and domain; (d) add `year`, `decimal_year`, `log10_flop`; (e) sort by date then model |

### P2. `cleaning_log.csv`
| Field | Detail |
|---|---|
| Built from | R1, by the same run as P1 |
| Columns (units) | `step` (text); `rows` (count remaining after the step) |
| Primary key | `step` |
| Time coverage / latest observation | Describes all of R1 (1950-07-02 to 2026-11-10) |
| Transform | One row per cleaning step of P1: 1,102 → 1,102 → 546 |

### P3. `coverage_by_year.csv`
| Field | Detail |
|---|---|
| Built from | R1 |
| Columns (units) | `year` (calendar year); `n_models` (count of notable models); `n_with_compute` (count with compute > 0); `share_with_compute` (0–1, 4 d.p.) |
| Primary key | `year` (62 years with at least one model) |
| Time coverage | 1950 to 2026 (2026 partial, to 10 Nov) |
| Latest observation | 2026 |
| Transform | Year of `Publication date`; count all rows and rows with positive compute per year; divide |

### P4. `mmlu_vs_compute.csv`
| Field | Detail |
|---|---|
| Built from | R2 |
| Size | 72 rows × 8 columns |
| Columns (units) | `Model version` (text); `score` (MMLU accuracy, 0–1, median of reports); `reports` (count of reports combined); `compute` (FLOP); `release` (YYYY-MM-DD); `organization` (text); `score_pct` (score × 100, %, 1 d.p.); `year` (release year) |
| Primary key | `Model version` |
| Time coverage | 2021-08-05 to 2024-12-26 |
| Latest observation | 2024-12-26 |
| Transform | (a) keep `Shots` = 5 so every score uses the same setting; (b) drop rows without compute, score or release date; (c) group by `Model version`: median `EM`, count of reports, first compute / release / organisation; (d) add `score_pct`, `year`; sort by compute |

### P5. `eci_vs_compute.csv`
| Field | Detail |
|---|---|
| Built from | R3 joined to R1 |
| Size | 46 rows × 8 columns |
| Columns (units) | `Model version` (text, from `Display name`); `score` (ECI, index points); `eci_ci_low`, `eci_ci_high` (index points); `compute` (FLOP, from R1); `release` (YYYY-MM-DD); `organization` (text); `year` (release year) |
| Primary key | `Model version` |
| Time coverage | 2023-02-24 to 2026-10-06 |
| Latest observation | 2026-10-06 |
| Transform | (a) inner join R3 to R1 on `Model` to attach `Training compute (FLOP)`; (b) keep compute > 0 and non-missing score and date; (c) rename columns; add `year`; sort by compute |

### P6. `meta.csv`
| Field | Detail |
|---|---|
| Built from | R1–R3 and P1 |
| Columns (units) | `raw_file` (text); `raw_sha256` (SHA-256 hash of the raw file); `raw_rows` (count); `models_clean_rows` (count); `earliest_observation`, `latest_observation` (YYYY-MM-DD, from P1); `processed_on` (YYYY-MM-DD, date `make_data.py` ran) |
| Primary key | `raw_file` (3 rows) |
| Time coverage / latest observation | As P1 (1950-07-02 to 2026-11-05) |
| Transform | Hash each raw file, count rows, copy P1's date range, record the run date. Shown in the app as "last refreshed". `processed_on` is the only value that changes between regenerations. |

---

## Values derived on screen (computed in `app.py` / `analysis.py`)

| Value | How it is computed |
|---|---|
| Compute growth (× / year), doubling time (months) | Least-squares fit of `log10_flop` on `decimal_year` (P1); growth = 10^slope; doubling = 12·log10(2)/slope. Headline: all P1 rows with `year` ≥ 2012 (474 models). Section 1: rows in the chosen years, domains and frontier filter. |
| 95% intervals | 2.5th and 97.5th percentiles of 1,000 bootstrap resamples (seed 0) |
| Methods exchange rate (× compute / year) | P4 (or P5): fit `link(score) = b0 + bC·log10(compute) + bT·(decimal release year − base)`; MMLU link = logit((score − 0.25)/0.75), base 2021; ECI link = identity, base 2023; rate = 10^(bT/bC) |
| R² (compute only / compute + year) | 1 − SSE/SST of the fit without and with the year term |
| Algorithm share | ln(rate) / (ln(rate) + ln(compute growth)), with the headline compute growth |
| 80% MMLU path and "× cheaper" | P4 in release order: each model with `score` ≥ 0.80 and less compute than every earlier such model; ratio of first to last compute |
| Waterfall split | Fitted score moved from (A's compute, A's date) → (B's compute, A's date) → (B's compute, B's date); residual = actual change − both parts |
| Share with a compute estimate | `share_with_compute` from P3 |

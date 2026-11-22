# Compute or algorithms? A data narrative on the rise of AI

**Question:** Since 2012, has frontier AI capability grown mainly because training compute was scaled up, or because algorithms improved?

- **Deployed app:** https://computeoralgorithms.streamlit.app
- **Repository:** https://github.com/carloscyzhou/ComputeOrAlgorithms
- **Assessed version:** release tag `v1.0` (its commit hash is given in the submission PDF)
- **Group 4:** Aksha Mansuri, Chenyu Zhou, Citra Yulianti, Yanzhen Huang

## Why the question is arguable

The same Epoch AI data supports two answers. Training compute grew far faster per year than the compute saved by better methods, which supports "mainly compute". But at fixed compute, newer models score much higher, the compute needed for 80% on MMLU fell sharply within two years, and on the newer Epoch Capabilities Index release date explains more than compute, which supports "algorithms matter as much or more". The app shows both readings and states which one the group thinks the evidence favours.

## Persona

An **Australian Government funding adviser** who must recommend whether public AI money should go to **compute infrastructure** (for example national GPU capacity) or to **algorithmic research**. The app serves that decision: how fast is the compute frontier moving, how much progress comes from methods rather than scale, and can each figure be defended in a funding brief.

## User stories

1. **As an** Australian Government funding adviser, **I want** to see how fast the training compute of notable AI models has grown since 2012, and whether that holds by domain, era and for frontier models only, **so that** I can judge how quickly the compute requirement at the frontier is rising.
   *Walk-through:* opening headline and tiles → section 1 (compute over time with one-trend / two-era / before-and-after-2010 lines, median and largest run per year, models per year, coverage, distribution, size vs compute) → change the sidebar year slider, domain selector and "Frontier models only" toggle → section 2 (largest runs, top organisations and countries, organisation-type share).
2. **As an** Australian Government funding adviser, **I want** to compare benchmark scores of models trained with the same compute but released in different years, **so that** I can estimate how much yearly progress comes from better algorithms rather than more compute.
   *Walk-through:* section 3 (switch between MMLU and the Epoch Capabilities Index; the Replay tab), the R² bars, the "one year of methods ≈ compute" figure, the iso-capability map, the 80%-MMLU chart, and "Our answer".
3. **As an** Australian Government funding adviser, **I want** to split the score gain between two specific models into compute, methods and model-specific parts and trace each figure to the source rows, **so that** I can cite a concrete, defensible example in my brief.
   *Walk-through:* section 4 (pick "From" and "To" models) → "Trace the numbers back to the source data" panel (cleaning log, formulas, row lookup, CSV download).

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python make_data.py      # rebuilds data/processed/ from data/raw/
.venv/bin/python verify.py         # regeneration + independent checks; exits non-zero on failure
.venv/bin/streamlit run app.py
```

## Repository contents

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app. Reads only `data/processed/`. |
| `analysis.py` | Cleaning, trend fit, capability model, decomposition. |
| `make_data.py` | Regenerates every processed file from `data/raw/`. |
| `verify.py` | Regenerates the processed files in a temp copy, compares them with the committed ones, and recomputes key numbers from the raw files independently. |
| `data/raw/` | Source samples from Epoch AI, unmodified, plus the Epoch licence note. |
| `data/processed/` | The seven files the app reads. |
| `DATA_DICTIONARY.md` | Source, URL, licence, columns and units, primary key, coverage, latest observation and transform for every dataset. |
| `PROVENANCE.md` | AI provenance matrix (one or more rows per member). |
| `DISCLOSURE.md` | Data and AI disclosure statement. |

## Method in one paragraph

Compute growth is a least-squares fit of log10(training compute) on decimal publication year; doubling time = 12·log10(2)/slope, with 95% intervals from 1,000 bootstrap resamples. Capability is modelled as `score = b0 + bC·log10(compute) + bT·(year − base)` (MMLU on the logit of the above-chance score). One year of better methods is worth 10^(bT/bC) times more compute; the algorithm share of yearly progress is ln(rate) / (ln(rate) + ln(compute growth)).

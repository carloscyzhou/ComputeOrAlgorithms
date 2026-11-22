"""Regenerate every processed file from the raw samples.   Run: python make_data.py

Inputs (Epoch AI, CC BY 4.0; see DATA_DICTIONARY.md for URLs and download dates)
  data/raw/notable_ai_models.csv
  data/raw/mmlu_external.csv
  data/raw/eci_scores.csv
Outputs (data/processed/, the only files app.py reads)
  models_clean.csv, models_all.csv, cleaning_log.csv, coverage_by_year.csv,
  mmlu_vs_compute.csv, eci_vs_compute.csv, meta.csv
"""
import hashlib
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from analysis import DATE_COL, FLOP_COL, all_models, clean_models

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def main():
    for f in ("notable_ai_models.csv", "mmlu_external.csv", "eci_scores.csv"):
        if not (RAW / f).exists():
            sys.exit(f"data/raw/{f} not found. See DATA_DICTIONARY.md for the download URL.")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. notable models with a compute estimate (+ cleaning log) ----
    raw = pd.read_csv(RAW / "notable_ai_models.csv", low_memory=False)
    models, log = clean_models(raw)
    models.to_csv(OUT / "models_clean.csv", index=False)
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(OUT / "cleaning_log.csv", index=False)
    all_models(raw).to_csv(OUT / "models_all.csv", index=False)   # incl. models without compute

    # ---- 2. coverage: share of notable models per year that have a compute estimate ----
    r = pd.DataFrame({"year": pd.to_datetime(raw[DATE_COL], errors="coerce").dt.year,
                      "has": pd.to_numeric(raw[FLOP_COL], errors="coerce") > 0}).dropna(subset=["year"])
    cov = r.groupby("year").agg(n_models=("has", "size"), n_with_compute=("has", "sum")).reset_index()
    cov["year"] = cov["year"].astype(int)
    cov["share_with_compute"] = (cov["n_with_compute"] / cov["n_models"]).round(4)
    cov.to_csv(OUT / "coverage_by_year.csv", index=False)

    # ---- 3. MMLU (5-shot) score vs training compute ----
    mmlu = pd.read_csv(RAW / "mmlu_external.csv")
    mmlu = mmlu[mmlu["Shots"] == 5]                          # same test setting for every score
    mmlu = mmlu.dropna(subset=["Training compute (FLOP)", "EM", "Release date"])
    mmlu = (mmlu.groupby("Model version")                    # several reports per model -> median score
                .agg(score=("EM", "median"), reports=("EM", "size"),
                     compute=("Training compute (FLOP)", "first"), release=("Release date", "first"),
                     organization=("Organization", "first"))
                .reset_index())
    mmlu["score_pct"] = (mmlu["score"] * 100).round(1)
    mmlu["year"] = pd.to_datetime(mmlu["release"]).dt.year
    mmlu.sort_values(["compute", "Model version"]).to_csv(OUT / "mmlu_vs_compute.csv", index=False)

    # ---- 4. Epoch Capabilities Index vs training compute (compute joined from file 1 on Model) ----
    eci = pd.read_csv(RAW / "eci_scores.csv")
    comp = raw[["Model", FLOP_COL]].assign(**{FLOP_COL: pd.to_numeric(raw[FLOP_COL], errors="coerce")})
    eci = eci.merge(comp, on="Model", how="inner")
    eci = eci[eci[FLOP_COL] > 0].dropna(subset=["eci", "date"])
    eci = eci.rename(columns={FLOP_COL: "compute", "date": "release", "Organization": "organization",
                              "Display name": "Model version", "eci": "score"})
    eci["year"] = pd.to_datetime(eci["release"]).dt.year
    eci = eci[["Model version", "score", "eci_ci_low", "eci_ci_high", "compute", "release",
               "organization", "year"]]
    eci.sort_values(["compute", "Model version"]).to_csv(OUT / "eci_vs_compute.csv", index=False)

    # ---- 5. meta (footer: hashes, coverage, refresh date) ----
    rows = []
    for f, n in (("notable_ai_models.csv", len(raw)),
                 ("mmlu_external.csv", len(pd.read_csv(RAW / "mmlu_external.csv"))),
                 ("eci_scores.csv", len(pd.read_csv(RAW / "eci_scores.csv")))):
        rows.append({"raw_file": f, "raw_sha256": hashlib.sha256((RAW / f).read_bytes()).hexdigest(),
                     "raw_rows": n})
    meta = pd.DataFrame(rows)
    meta["models_clean_rows"] = len(models)
    meta["earliest_observation"] = models["publication_date"].min()
    meta["latest_observation"] = models["publication_date"].max()
    meta["processed_on"] = date.today().isoformat()
    meta.to_csv(OUT / "meta.csv", index=False)

    print(f"models_clean.csv    {len(raw)} raw -> {len(models)} rows, latest {models['publication_date'].max()}")
    print(f"mmlu_vs_compute.csv {len(mmlu)} models")
    print(f"eci_vs_compute.csv  {len(eci)} models")


if __name__ == "__main__":
    main()

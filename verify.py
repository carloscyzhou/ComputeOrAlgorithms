"""Independent checks on the data package.   Run: python verify.py
Every check can fail; the script exits non-zero if any does.

1. Regenerates data/processed/ in a temporary copy with make_data.py and compares each file
   byte-for-byte with the committed one (meta.csv is compared without its processed_on date).
2. Recomputes key on-screen numbers straight from data/raw/ with plain pandas/numpy
   (not analysis.py) and compares them with the processed files.
"""
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
P = ROOT / "data" / "processed"
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


# 1. regeneration -------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp) / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
    for f in (work / "data" / "processed").glob("*.csv"):
        f.unlink()
    subprocess.run([sys.executable, "make_data.py"], cwd=work, check=True, capture_output=True)
    for f in sorted(P.glob("*.csv")):
        new = work / "data" / "processed" / f.name
        if f.name == "meta.csv":
            same = pd.read_csv(new).drop(columns="processed_on").equals(pd.read_csv(f).drop(columns="processed_on"))
        else:
            same = new.exists() and filecmp.cmp(new, f, shallow=False)
        check(f"regenerated {f.name} identical to committed", same)

# 2. independent recomputation from raw --------------------------------------
raw = pd.read_csv(ROOT / "data/raw/notable_ai_models.csv", low_memory=False)
d = pd.to_datetime(raw["Publication date"], errors="coerce")
f = pd.to_numeric(raw["Training compute (FLOP)"], errors="coerce")
ok = d.notna() & (f > 0)
proc = pd.read_csv(P / "models_clean.csv")
check("raw row count matches cleaning log", len(raw) == int(pd.read_csv(P / "cleaning_log.csv")["rows"].iloc[0]),
      f"{len(raw)}")
check("usable rows in raw equal rows in models_clean.csv", int(ok.sum()) == len(proc), f"{int(ok.sum())}")

yr = d.dt.year + (d.dt.day_of_year - 1) / 365.25
sel = ok & (d.dt.year >= 2012)
slope_raw = np.polyfit(yr[sel], np.log10(f[sel]), 1)[0]
p = proc[proc["year"] >= 2012]
slope_proc = np.polyfit(p["decimal_year"], p["log10_flop"], 1)[0]
check("since-2012 slope from raw equals slope from processed file", abs(slope_raw - slope_proc) < 1e-4,
      f"doubling {12 * np.log10(2) / slope_raw:.2f} months, {10 ** slope_raw:.2f}x/yr, n={int(sel.sum())}")

mm_raw = pd.read_csv(ROOT / "data/raw/mmlu_external.csv")
mm = pd.read_csv(P / "mmlu_vs_compute.csv")
phi = mm_raw[(mm_raw["Model version"] == "phi-4") & (mm_raw["Shots"] == 5)]["EM"].median()
check("phi-4 MMLU score in processed file equals raw (5-shot median)",
      abs(mm.set_index("Model version").loc["phi-4", "score"] - phi) < 1e-9, f"{phi * 100:.1f}%")
n5 = mm_raw[(mm_raw["Shots"] == 5) & mm_raw["Training compute (FLOP)"].notna()]["Model version"].nunique()
check("MMLU model count equals distinct 5-shot models with compute in raw", n5 == len(mm), f"{n5}")

# cheapest model to reach 80%, recomputed with a plain loop over the raw file
r5 = (mm_raw[(mm_raw["Shots"] == 5) & mm_raw["Training compute (FLOP)"].notna()]
      .groupby("Model version").agg(s=("EM", "median"), c=("Training compute (FLOP)", "first"),
                                     t=("Release date", "first")).sort_values("t"))
best, path = np.inf, []
for name, row in r5.iterrows():
    if row["s"] >= 0.8 and row["c"] < best:
        best = row["c"]
        path.append((name, row["c"]))
print(f"INFO 80% MMLU path: {path[0][0]} -> {path[-1][0]}, {path[0][1] / path[-1][1]:.1f}x less compute")

sys.exit(1 if fails else 0)

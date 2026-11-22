"""Cleaning and statistics shared by make_data.py, app.py and verify.py.

Compute trend:
    log10(training compute) = a + slope * year      doubling (months) = 12 * log10(2) / slope

Capability model (fitted on models that have both a benchmark score and a compute estimate):
    capability = b0 + bC * log10(training compute) + bT * (release year - base year)
bC = gain from 10x more compute; bT = gain from one more year of better methods at the SAME compute.
Exchange rate: one year of algorithmic progress == 10 ** (bT / bC) times more compute.
MMLU is fitted on the logit of the above-chance score (25% = guessing, 100% = ceiling).
"""
import numpy as np
import pandas as pd

DATE_COL = "Publication date"
FLOP_COL = "Training compute (FLOP)"
CHANCE = 0.25
N_BOOT = 1000


def frac_year(dates):
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year + (d.dt.day_of_year - 1) / 365.25).to_numpy()


def first_listed(series, fill):
    return series.fillna(fill).astype(str).str.split(",").str[0].str.strip()


# ---------------------------------------------------------------- cleaning
def clean_models(raw: pd.DataFrame):
    """Return (clean_df, log). log is a list of (step, rows_remaining)."""
    log = [("Rows in raw CSV", len(raw))]
    df = raw.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    log.append(("After dropping rows with no usable publication date", len(df)))
    df[FLOP_COL] = pd.to_numeric(df[FLOP_COL], errors="coerce")
    df = df[df[FLOP_COL] > 0]
    log.append(("After dropping rows with no positive training compute value", len(df)))
    out = pd.DataFrame({
        "model": df["Model"].astype(str).str.strip(),
        "organization": first_listed(df["Organization"], "Unspecified"),
        "country": first_listed(df["Country (of organization)"], "Unspecified"),
        "domain": first_listed(df["Domain"], "Unspecified"),
        "publication_date": df[DATE_COL].dt.strftime("%Y-%m-%d"),
        "year": df[DATE_COL].dt.year.astype(int),
        "decimal_year": frac_year(df[DATE_COL]).round(6),
        "training_compute_flop": df[FLOP_COL],
        "log10_flop": np.log10(df[FLOP_COL]).round(6),
        "confidence": df["Confidence"].fillna("Not stated").astype(str),
        "frontier_model": df["Frontier model"].eq(True),
    })
    return out.sort_values(["publication_date", "model"]).reset_index(drop=True), log


# ---------------------------------------------------------------- compute trend
def fit_trend(years, logc, n_boot=N_BOOT, seed=0):
    """Least-squares line on log10(compute) vs decimal year, with a bootstrap 95% CI. None if unusable."""
    years, logc = np.asarray(years, float), np.asarray(logc, float)
    if len(years) < 5 or np.unique(years).size < 2:
        return None
    slope, intercept = np.polyfit(years, logc, 1)
    if slope <= 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([np.polyfit(years[i], logc[i], 1)
                      for i in (rng.integers(0, len(years), len(years)) for _ in range(n_boot))])
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    return dict(n=len(years), slope=slope, intercept=intercept, growth=10 ** slope,
                growth_ci=(10 ** lo, 10 ** hi), doubling=12 * np.log10(2) / slope,
                doubling_ci=(12 * np.log10(2) / hi, 12 * np.log10(2) / lo) if lo > 0 else (np.nan, np.nan))


# ---------------------------------------------------------------- capability model
def to_link(score, kind):
    if kind == "mmlu":
        p = np.clip((np.asarray(score, float) - CHANCE) / (1 - CHANCE), 0.01, 0.99)
        return np.log(p / (1 - p))
    return np.asarray(score, float)


def from_link(z, kind):
    if kind == "mmlu":
        return CHANCE + (1 - CHANCE) / (1 + np.exp(-np.asarray(z, float)))
    return np.asarray(z, float)


def fit_capability(score, compute, years, kind, base_year, n_boot=N_BOOT, seed=0):
    y = to_link(score, kind)
    X = np.column_stack([np.ones(len(y)), np.log10(compute), np.asarray(years) - base_year])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - X @ beta) ** 2).sum() / sst
    b_c = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]
    r2_compute = 1 - ((y - X[:, :2] @ b_c) ** 2).sum() / sst
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        b = np.linalg.lstsq(X[i], y[i], rcond=None)[0]
        rates.append(b[2] / b[1])
    return dict(beta=beta, kind=kind, base_year=base_year, n=len(y), r2=r2, r2_compute=r2_compute,
                rate=10 ** (beta[2] / beta[1]), rate_ci=10 ** np.percentile(rates, [2.5, 97.5]))


def predict(fit, compute, years):
    b0, bc, bt = fit["beta"]
    return from_link(b0 + bc * np.log10(compute) + bt * (np.asarray(years) - fit["base_year"]), fit["kind"])


def algo_share(rate, compute_growth):
    """Share of yearly progress (in log-compute-equivalent units) that comes from algorithms."""
    return np.log(rate) / (np.log(rate) + np.log(compute_growth))


def decompose(fit, c_a, t_a, c_b, t_b, s_a, s_b):
    """Split the score change A -> B into compute, methods and model-specific parts."""
    p_a = predict(fit, c_a, t_a)
    p_mid = predict(fit, c_b, t_a)        # B's compute at A's date
    p_b = predict(fit, c_b, t_b)
    compute_part, algo_part = float(p_mid - p_a), float(p_b - p_mid)
    return compute_part, algo_part, float((s_b - s_a) - (compute_part + algo_part))


def cheapest_path(df, threshold):
    """Each model that reached the threshold score with less compute than any earlier model."""
    rows, best = [], np.inf
    for _, r in df.sort_values("release").iterrows():
        if r["score"] >= threshold and r["compute"] < best:
            best = r["compute"]
            rows.append(r)
    return pd.DataFrame(rows)

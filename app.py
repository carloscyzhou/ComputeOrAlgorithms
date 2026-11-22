"""The Rise of AI: compute or algorithms?   Run: streamlit run app.py

Reads only data/processed/*.csv, which make_data.py builds from data/raw/ (Epoch AI, CC BY 4.0).
Statistics live in analysis.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A

st.set_page_config(page_title="Compute or algorithms?", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding-top:1.6rem;max-width:1280px;}</style>", unsafe_allow_html=True)

PROC = Path(__file__).resolve().parent / "data" / "processed"
QUESTION = ("Since 2012, has frontier AI capability grown mainly because training compute was scaled up, "
            "or because algorithms improved?")
PERSONA = ("An Australian Government funding adviser deciding whether public AI money should go to compute "
           "infrastructure (national GPU capacity) or to algorithmic research.")
# Okabe-Ito: distinguishable under the common colour vision deficiencies
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
OTHER, INK = "#999999", "#222222"
YEAR_RAMP = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]          # older -> newer (lightness differs)
START = 2012
EMPTY = "No models match these filters. Widen the year range, add a domain or switch off 'Frontier models only'."
st.title(QUESTION)          # drawn before any calculation so the question shows first


# ================================================================== data
@st.cache_data
def load():
    m = pd.read_csv(PROC / "models_clean.csv", parse_dates=["publication_date"])
    log = pd.read_csv(PROC / "cleaning_log.csv")
    cov = pd.read_csv(PROC / "coverage_by_year.csv")
    meta = pd.read_csv(PROC / "meta.csv")
    mmlu = pd.read_csv(PROC / "mmlu_vs_compute.csv")
    eci = pd.read_csv(PROC / "eci_vs_compute.csv")
    for d in (mmlu, eci):
        d["t"] = A.frac_year(d["release"])
    return m, log, cov, meta, mmlu, eci


@st.cache_data
def trend_for(years, logc):
    return A.fit_trend(np.asarray(years), np.asarray(logc))


@st.cache_data
def capability_fit(kind):
    d = load()[4] if kind == "mmlu" else load()[5]
    return A.fit_capability(d["score"], d["compute"], d["t"], kind, 2021 if kind == "mmlu" else 2023)


try:
    models, log, cov, meta, mmlu, eci = load()
except FileNotFoundError:
    st.error("Processed data not found. Run `python make_data.py` first (see README).")
    st.stop()

meta0 = meta.iloc[0]
SOURCES = ("Epoch AI, *Data on Notable AI Models* (https://epoch.ai/data/notable_ai_models.csv) and "
           "*Capabilities & Benchmarking* (MMLU and Epoch Capabilities Index, https://epoch.ai/data/benchmark_data.zip), "
           "CC BY 4.0.")


def fmt_flop(x):
    return f"{x:.1e} FLOP"


def base_layout(fig, title, x_title, y_title, height=420, legend=None):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=20, t=60, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=INK, size=12), showlegend=legend is not None,
                      legend=dict(orientation="h", y=-0.22, x=0, title_text=legend or ""))
    fig.update_xaxes(title=x_title, gridcolor="#ececec", linecolor="#cccccc", zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor="#ececec", linecolor="#cccccc", zeroline=False)
    return fig


# ================================================================== whole-dataset figures (opening screen)
since = models[models["year"] >= START]
headline = trend_for(since["decimal_year"].to_numpy(), since["log10_flop"].to_numpy())
fit_m = capability_fit("mmlu")
share = A.algo_share(fit_m["rate"], headline["growth"])
share_lo, share_hi = (A.algo_share(r, headline["growth"]) for r in fit_m["rate_ci"])
path80 = A.cheapest_path(mmlu, 0.80)
drop80 = path80["compute"].iloc[0] / path80["compute"].iloc[-1]
months80 = (pd.to_datetime(path80["release"].iloc[-1]) - pd.to_datetime(path80["release"].iloc[0])).days / 30.44
raw_rows = int(log["rows"].iloc[0])

st.caption(f"**Who this is for:** {PERSONA}  \n**Data:** Epoch AI, Notable AI Models and Capabilities & "
           f"Benchmarking (CC BY 4.0) · **last refreshed** {meta0['processed_on']} · latest model "
           f"{meta0['latest_observation']} · full sources at the foot of the page")
st.success(
    f"**Across all {len(since):,} notable AI models published since {START} that have a compute estimate, "
    f"training compute doubled about every {headline['doubling']:.1f} months ({headline['growth']:.1f}× per year). "
    f"On the {fit_m['n']} models with an MMLU score (released 2021–2024), one year of better methods was worth "
    f"about {fit_m['rate']:.1f}× more compute.** "
    f"Source rows: {raw_rows:,} notable models in Epoch AI's list ({meta0['earliest_observation'][:4]}–"
    f"{meta0['latest_observation'][:4]}), of which {len(models):,} have a training compute estimate."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Compute growth", f"{headline['growth']:.1f}× / year")
k1.caption(f"Training compute since {START}, all domains, n = {headline['n']}; "
           f"95% CI {headline['growth_ci'][0]:.1f}–{headline['growth_ci'][1]:.1f}×")
k2.metric("Methods, as compute", f"{fit_m['rate']:.1f}× / year")
k2.caption(f"Compute saved per year by better methods; MMLU 5-shot 2021–2024, n = {fit_m['n']}; "
           f"95% CI {fit_m['rate_ci'][0]:.1f}–{fit_m['rate_ci'][1]:.1f}×")
k3.metric("Algorithm share", f"≈ {share:.0%}")
k3.caption(f"Of yearly progress, in compute-equivalent terms; range {share_lo:.0%}–{share_hi:.0%}; "
           "the rest is compute scale-up")
k4.metric("80% MMLU: cheaper", f"{drop80:.0f}×")
k4.caption(f"Less training compute: {path80['Model version'].iloc[0]} → {path80['Model version'].iloc[-1]}, "
           f"{months80:.0f} months apart")

st.markdown("### Our answer")
st.markdown(
    f"**Mainly compute, but not only compute.** Measured in compute-equivalent terms, scaling up training "
    f"compute ({headline['growth']:.1f}× a year since {START}) contributed more than better methods "
    f"({fit_m['rate']:.1f}× a year of compute saved, measured on MMLU 2021–2024), so roughly "
    f"**{1 - share:.0%} of yearly progress came from compute and {share:.0%} from algorithms**. "
    f"The algorithm share is uncertain (range {share_lo:.0%}–{share_hi:.0%}), and a reader who weights the "
    f"falling cost of a fixed score (80% MMLU became {drop80:.0f}× cheaper) could argue the other way. "
    "For the adviser: a compute-only budget would leave the algorithmic share unfunded."
)

# ================================================================== sidebar filters
st.sidebar.header("Filters for section 1")
y_lo, y_hi = int(models["year"].min()), int(models["year"].max())
years = st.sidebar.slider("Publication years", y_lo, y_hi, (START, y_hi))
dom_rank = models.loc[models["year"] >= START, "domain"].value_counts().index.tolist()
dom_rank += [d for d in models["domain"].value_counts().index if d not in dom_rank]
domains = st.sidebar.multiselect("Domain (first listed)", dom_rank, default=dom_rank[:4],
                                 help="Defaults to the four domains with the most models since 2012.")
frontier_only = st.sidebar.toggle("Frontier models only", value=False,
                                  help="Epoch AI flag: among the top 10 by training compute when released.")
st.sidebar.caption("Sections 2 and 3 use the benchmark tables and have their own controls.")
colour = {d: (OKABE[i] if i < len(OKABE) else OTHER) for i, d in enumerate(dom_rank)}

view = models[models["year"].between(*years) & models["domain"].isin(domains)]
if frontier_only:
    view = view[view["frontier_model"]]

# ================================================================== 1. compute
st.markdown("---")
st.header("1 · How fast has training compute grown?")
if view.empty:
    st.info(EMPTY)
else:
    fit = trend_for(view["decimal_year"].to_numpy(), view["log10_flop"].to_numpy())
    t1, t2, t3 = st.columns(3)
    t1.metric("Models shown", f"{len(view):,}")
    t2.metric("Doubling time", f"{fit['doubling']:.1f} months" if fit else "n/a")
    t3.metric("Growth per year", f"{fit['growth']:.1f}×" if fit else "n/a")
    if fit is None:
        st.info("Too few models, or no upward trend, in this selection to fit a line.")

    fig = go.Figure()
    for dom in [d for d in dom_rank if d in set(view["domain"])]:
        g = view[view["domain"] == dom]
        fig.add_trace(go.Scatter(
            x=g["publication_date"], y=g["training_compute_flop"], mode="markers", name=dom,
            marker=dict(size=7, opacity=0.8, color=colour[dom], line=dict(width=0.5, color="white")),
            customdata=np.column_stack([g["model"], g["organization"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x|%b %Y}<br>%{y:.2e} FLOP<extra></extra>"))
    if fit:
        xs = np.linspace(view["decimal_year"].min(), view["decimal_year"].max(), 50)
        fig.add_trace(go.Scatter(x=pd.to_datetime((xs - 1970) * 365.25, unit="D"),
                                 y=10 ** (fit["intercept"] + fit["slope"] * xs), mode="lines",
                                 name=f"Fitted trend: doubles every {fit['doubling']:.1f} months",
                                 line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
    top = view.nlargest(5, "training_compute_flop")
    fig.add_trace(go.Scatter(x=top["publication_date"], y=top["training_compute_flop"], mode="markers+text",
                             name="Five largest (labelled)", textposition="middle left",
                             text=[f"{m}: {v:.1e}" for m, v in zip(top["model"], top["training_compute_flop"])],
                             textfont=dict(size=11), hoverinfo="skip",
                             marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(width=1.5, color=INK))))
    fig.update_yaxes(type="log", exponentformat="power")
    st.plotly_chart(base_layout(fig, f"Training compute of notable AI models, {years[0]}–{years[1]} "
                                     "(one dot = one model)",
                                "Publication date", "Training compute (FLOP, log scale)", 470, "Domain"),
                    width="stretch")
    st.caption(f"Baseline: {len(view):,} models in the selected years and domains, "
               f"{view['publication_date'].min():%b %Y} to {view['publication_date'].max():%b %Y}. "
               "Log axis: each gridline is 100× more compute; a straight line means constant exponential "
               "growth. Labels give the five largest runs in FLOP.")

    c = cov[cov["year"].between(*years)]
    fig2 = go.Figure(go.Bar(x=c["year"], y=c["share_with_compute"] * 100, marker_color=OKABE[0],
                            text=[f"{v:.0%}" for v in c["share_with_compute"]], textposition="outside",
                            textfont=dict(size=10), hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
    fig2.update_yaxes(range=[0, 115], ticksuffix="%")
    fig2.update_xaxes(dtick=1 if years[1] - years[0] <= 16 else 5)
    st.plotly_chart(base_layout(fig2, "Share of notable models each year that have a training compute estimate",
                                "Publication year", "Models with a compute estimate (% of notable models)", 320),
                    width="stretch")
    st.caption("All domains (this chart ignores the domain filter). Models without an estimate cannot appear "
               "in the chart above, so years with a low share are under-represented.")

# ================================================================== 2. capability at fixed compute
st.markdown("---")
st.header("2 · At the same compute, do newer models score higher?")
bench = st.radio("Benchmark", ["MMLU, 5-shot (2021–2024)", "Epoch Capabilities Index (2023–2026)"],
                 horizontal=True)
kind = "mmlu" if bench.startswith("MMLU") else "eci"
df = mmlu if kind == "mmlu" else eci
fit_b = capability_fit(kind)
scale = 100 if kind == "mmlu" else 1
y_title = "MMLU score (% correct, 5-shot)" if kind == "mmlu" else "Epoch Capabilities Index (index points)"
yrs = sorted(df["year"].unique())
ramp = dict(zip(yrs, YEAR_RAMP[-len(yrs):] if len(yrs) <= len(YEAR_RAMP) else
                 [YEAR_RAMP[min(i * len(YEAR_RAMP) // len(yrs), 3)] for i in range(len(yrs))]))
symbols = ["circle", "square", "diamond", "triangle-up", "x", "cross"]

left, right = st.columns([2.4, 1])
with left:
    fig3 = go.Figure()
    xs = np.logspace(np.log10(df["compute"].min()), np.log10(df["compute"].max()), 60)
    for i, yr in enumerate(yrs):
        part = df[df["year"] == yr]
        fig3.add_trace(go.Scatter(x=xs, y=A.predict(fit_b, xs, yr + 0.5) * scale, mode="lines",
                                  line=dict(color=ramp[yr], width=2), name=f"Fitted, {yr}", showlegend=False,
                                  hoverinfo="skip"))
        fig3.add_trace(go.Scatter(
            x=part["compute"], y=part["score"] * scale, mode="markers", name=str(yr),
            marker=dict(size=9, color=ramp[yr], symbol=symbols[i % len(symbols)], line=dict(width=1, color=INK)),
            text=part["Model version"],
            hovertemplate="<b>%{text}</b><br>%{x:.2e} FLOP<br>score %{y:.1f}<extra></extra>"))
    best = df.loc[df.groupby("year")["score"].idxmax()]
    for _, r in best.iterrows():
        fig3.add_annotation(x=np.log10(r["compute"]), y=r["score"] * scale, showarrow=False, yshift=14,
                            text=f"{r['Model version']}: {r['score'] * scale:.1f}", font=dict(size=10),
                            bgcolor="rgba(255,255,255,.85)")
    if kind == "mmlu":
        fig3.add_hline(y=25, line=dict(color=OTHER, dash="dot"), annotation_text="Random guessing (25%)",
                       annotation_position="bottom right")
    fig3.update_xaxes(type="log", exponentformat="power", dtick=1)
    st.plotly_chart(base_layout(fig3, "Benchmark score against training compute, by release year "
                                      "(lines: fitted score for each year)",
                                "Training compute (FLOP, log scale)", y_title, 470, "Release year"),
                    width="stretch")
    st.caption(f"{fit_b['n']} models with both a score and a compute estimate. A higher line at the same "
               "compute means newer models score more without more compute. Labels: best score each year. "
               "All values are listed in the table below.")
with right:
    fig4 = go.Figure(go.Bar(
        x=[fit_b["r2_compute"], fit_b["r2"]], y=["Compute only", "Compute + release year"], orientation="h",
        marker_color=[OTHER, OKABE[0]], text=[f"{fit_b['r2_compute']:.2f}", f"{fit_b['r2']:.2f}"],
        textposition="outside", hovertemplate="%{y}: R² = %{x:.2f}<extra></extra>"))
    fig4.update_xaxes(range=[0, 1.15])
    st.plotly_chart(base_layout(fig4, "Share of score variance explained (R²)", "R² (0 to 1, no unit)", "", 230),
                    width="stretch")
    rate_txt = f"{fit_b['rate']:.1f}×" if fit_b["rate"] < 100 else f"> {fit_b['rate_ci'][0]:.0f}×"
    st.metric("One year of better methods ≈", f"{rate_txt} compute")
    st.caption(f"95% CI {fit_b['rate_ci'][0]:.1f}× to {fit_b['rate_ci'][1]:,.1f}× (1,000 bootstrap resamples). "
               "Exchange rate = 10^(bT / bC) from the fitted model.")
    if kind == "eci":
        st.info("Release year explains much more than training compute on this index. Post-training and "
                "reasoning compute are not counted as training compute, so the interval is very wide.")
with st.expander(f"Values behind this chart ({len(df)} models)"):
    tbl = df[["Model version", "organization", "release", "compute", "score"]].copy()
    tbl["score"] = (tbl["score"] * scale).round(1)
    tbl["compute"] = tbl["compute"].map("{:.2e}".format)
    st.dataframe(tbl.rename(columns={"compute": "Training compute (FLOP)", "score": y_title,
                                     "organization": "Organisation", "release": "Released"}),
                 hide_index=True, width="stretch", height=260)

# cheapest model to reach 80% MMLU
fig5 = go.Figure(go.Bar(
    y=[f"{r['Model version']} ({pd.to_datetime(r['release']):%b %Y}, {r['score'] * 100:.1f}%)"
       for _, r in path80.iterrows()],
    x=path80["compute"], orientation="h", marker_color=OKABE[2],
    text=[fmt_flop(v) for v in path80["compute"]], textposition="outside",
    hovertemplate="%{y}<br>%{x:.2e} FLOP<extra></extra>"))
fig5.update_xaxes(type="log", exponentformat="power", range=[np.log10(path80["compute"].min()) - 0.5,
                                                             np.log10(path80["compute"].max()) + 0.9])
fig5.update_yaxes(autorange="reversed", automargin=True)
st.plotly_chart(base_layout(fig5, "Training compute of each model that reached 80% on MMLU with less compute "
                                  "than any earlier model", "Training compute (FLOP, log scale)", "", 300),
                width="stretch")
st.caption(f"Read top to bottom in release order: the compute needed for an 80% MMLU score fell {drop80:.0f}× in "
           f"{months80:.0f} months. MMLU 5-shot table only ({len(mmlu)} models).")

# ================================================================== 3. decomposition
st.markdown("---")
st.header("3 · Split one model-to-model gain into compute and methods")
names = mmlu.sort_values("t")["Model version"].tolist()
p1, p2 = st.columns(2)
a = p1.selectbox("From model", names, index=names.index("Gopher (280B)") if "Gopher (280B)" in names else 0)
b = p2.selectbox("To model", names, index=names.index("phi-4") if "phi-4" in names else len(names) - 1)
if a == b:
    st.info("Choose two different models to split the score change.")
else:
    ra, rb = mmlu.set_index("Model version").loc[a], mmlu.set_index("Model version").loc[b]
    c_part, a_part, o_part = A.decompose(fit_m, ra["compute"], ra["t"], rb["compute"], rb["t"],
                                         ra["score"], rb["score"])
    steps = [c_part * 100, a_part * 100, o_part * 100]
    fig6 = go.Figure(go.Waterfall(
        x=[a, "More compute", "Better methods", "Model-specific", b],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[ra["score"] * 100, *steps, 0],
        text=[f"{ra['score'] * 100:.1f}%", *[f"{v:+.1f} pts" for v in steps], f"{rb['score'] * 100:.1f}%"],
        textposition="outside", connector=dict(line=dict(color=OTHER, dash="dot")),
        increasing=dict(marker_color=OKABE[0]), decreasing=dict(marker_color=OKABE[5]),
        totals=dict(marker_color=INK)))
    lo = min(ra["score"], rb["score"]) * 100
    fig6.update_yaxes(range=[max(0, lo - 25), 105])
    st.plotly_chart(base_layout(fig6, f"MMLU score change from {a} to {b}, split by the fitted model",
                                "", "MMLU score (% correct, 5-shot)", 400), width="stretch")
    st.caption(f"{a}: {fmt_flop(ra['compute'])}, released {ra['release']}. {b}: {fmt_flop(rb['compute'])}, "
               f"released {rb['release']} ({rb['compute'] / ra['compute']:.2f}× the compute, "
               f"{rb['t'] - ra['t']:.1f} years later). 'More compute' moves A to B's compute at A's date; "
               "'Better methods' then moves to B's date; 'Model-specific' is what the fit does not explain. "
               "Blue = increase, vermilion = decrease.")

# ================================================================== readings, limits, trace, sources
st.markdown("---")
st.header("Two readings of the same data")
l, r = st.columns(2)
l.markdown(f"**The case for compute.** Training compute has grown {headline['growth']:.1f}× a year since "
           f"{START}, faster than the {fit_m['rate']:.1f}× a year saved by methods. On MMLU, compute alone "
           f"explains R² = {fit_m['r2_compute']:.2f} of score variance.")
r.markdown(f"**The case for algorithms.** Adding release year lifts R² to {fit_m['r2']:.2f}; an 80% MMLU score "
           f"became {drop80:.0f}× cheaper in {months80:.0f} months; and on the newer Epoch Capabilities Index, "
           "release year explains far more than compute.")

c21 = cov.loc[cov["year"] == 2021, "share_with_compute"]
c_last = cov.iloc[-1]
st.warning(
    "**What this data cannot show.** "
    "(1) Causation: release year stands in for 'better methods' but also picks up better data, longer "
    "training and post-training, which the data does not separate. "
    "(2) Capability before 2021: benchmark scores cover 2021–2026 only, so the split cannot be checked for "
    "2012–2020. "
    f"(3) Missing compute: {float(c21.iloc[0]) if len(c21) else float('nan'):.0%} of notable models from 2021 have "
    f"a compute estimate against {c_last['share_with_compute']:.0%} in {int(c_last['year'])}, so recent "
    "models are under-represented. "
    "(4) Compute values are Epoch AI estimates, some rated speculative. "
    f"(5) The {len(mmlu)} MMLU models are mostly 2023–2024 open models, not a random sample, and MMLU may "
    "be inflated by test questions leaking into training data, which would overstate the methods effect. "
    "(6) 'Notable' models are Epoch AI's selection, not every AI model."
)

with st.expander("Trace the numbers back to the source data"):
    st.markdown("**Cleaning steps for the notable-models file (row counts):**")
    steps_df = log.rename(columns={"step": "Step", "rows": "Rows"})
    steps_df.loc[len(steps_df)] = ["After the section 1 filters (what is plotted)", len(view)]
    st.dataframe(steps_df, hide_index=True, width="stretch")
    st.markdown(
        f"**Doubling time:** least-squares fit of log10(FLOP) on decimal year. Headline: slope = "
        f"{headline['slope']:.4f} orders of magnitude per year, so doubling = 12 × log10(2) / slope = "
        f"12 × 0.3010 / {headline['slope']:.4f} = **{headline['doubling']:.1f} months**.  \n"
        f"**Methods exchange rate:** MMLU fitted as logit((score − 0.25)/0.75) = b0 + bC·log10(compute) + "
        f"bT·(year − 2021); bC = {fit_m['beta'][1]:.3f}, bT = {fit_m['beta'][2]:.3f}, "
        f"rate = 10^(bT/bC) = **{fit_m['rate']:.2f}×**.  \n"
        f"**Algorithm share:** ln({fit_m['rate']:.2f}) / (ln({fit_m['rate']:.2f}) + ln({headline['growth']:.2f})) "
        f"= **{share:.0%}**.")
    pick = st.selectbox("Look up one model's processed row", models.sort_values("publication_date",
                        ascending=False)["model"].tolist())
    row = models[models["model"] == pick].copy()
    row["training_compute_flop"] = row["training_compute_flop"].map("{:.3e}".format)
    row["publication_date"] = row["publication_date"].dt.strftime("%Y-%m-%d")
    row = row.astype(str)
    st.dataframe(row.T.rename(columns={row.index[0]: "Value in data/processed/models_clean.csv"}),
                 width="stretch")
    st.download_button("Download the rows plotted in section 1 (CSV)", view.to_csv(index=False),
                       "plotted_rows.csv", "text/csv")

st.caption(f"**Sources:** {SOURCES} Raw files downloaded 23 Oct 2026 (benchmarks) and 20 Nov 2026 (notable "
           f"models). **Data last refreshed:** {meta0['processed_on']}; latest model with a compute estimate: "
           f"{meta0['latest_observation']}.")

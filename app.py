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
ORG_COLOUR = dict(zip(A.ORG_TYPES, ["#0072B2", "#009E73", "#E69F00", OTHER]))
YEAR_RAMP = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]          # older -> newer (lightness differs)
SEQ = [[0, "#f7fbff"], [0.35, "#deebf7"], [0.7, "#9ecae1"], [1, "#6baed6"]]   # light, so dark labels stay legible
START = 2012
EMPTY = "No models match these filters. Widen the year range, add a domain or press 'Reset filters' in the sidebar."
st.title(QUESTION)          # drawn before any calculation so the question shows first


# ================================================================== data
@st.cache_data
def load():
    m = pd.read_csv(PROC / "models_clean.csv", parse_dates=["publication_date"])
    allm = pd.read_csv(PROC / "models_all.csv")
    log = pd.read_csv(PROC / "cleaning_log.csv")
    cov = pd.read_csv(PROC / "coverage_by_year.csv")
    meta = pd.read_csv(PROC / "meta.csv")
    mmlu = pd.read_csv(PROC / "mmlu_vs_compute.csv")
    eci = pd.read_csv(PROC / "eci_vs_compute.csv")
    for d in (mmlu, eci):
        d["t"] = A.frac_year(d["release"])
    return m, allm, log, cov, meta, mmlu, eci


CACHE_VERSION = 2   # bump when analysis.py output changes, so a deployed cache is not reused


@st.cache_data
def trend_for(years, logc, version=CACHE_VERSION):
    return A.fit_trend(np.asarray(years), np.asarray(logc))


@st.cache_data
def eras_for(years, logc, version=CACHE_VERSION):
    return A.find_breakpoint(np.asarray(years), np.asarray(logc)) if len(years) >= 20 else None


@st.cache_data
def capability_fit(kind, version=CACHE_VERSION):
    d = load()[5] if kind == "mmlu" else load()[6]
    return A.fit_capability(d["score"], d["compute"], d["t"], kind, 2021 if kind == "mmlu" else 2023)


try:
    models, allm, log, cov, meta, mmlu, eci = load()
except FileNotFoundError:
    st.error("Processed data not found. Run `python make_data.py` first (see README).")
    st.stop()

meta0 = meta.iloc[0]
SOURCES = ("Epoch AI, *Data on Notable AI Models* (https://epoch.ai/data/notable_ai_models.csv) and "
           "*Capabilities & Benchmarking* (MMLU and Epoch Capabilities Index, https://epoch.ai/data/benchmark_data.zip), "
           "CC BY 4.0.")


def fmt_flop(x):
    return f"{x:.1e} FLOP"


def to_date(t):
    return pd.to_datetime((np.asarray(t, float) - 1970) * 365.25, unit="D")


def base_layout(fig, title, x_title, y_title, height=420, legend=None):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=20, t=60, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color=INK, size=12), showlegend=legend is not None,
                      legend=dict(orientation="h", y=-0.22, x=0, title_text=legend or ""))
    fig.update_xaxes(title=x_title, gridcolor="#ececec", linecolor="#cccccc", zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor="#ececec", linecolor="#cccccc", zeroline=False)
    return fig


def hbar(series, title, x_title, y_title, colour, height=380):
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", text=s.values, textposition="outside",
                           marker_color=colour, hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_xaxes(rangemode="tozero", range=[0, s.max() * 1.18])
    fig.update_yaxes(automargin=True)
    return base_layout(fig, title, x_title, y_title, height)


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

# ================================================================== sidebar filters (sections 1 and 2)
y_lo, y_hi = int(models["year"].min()), int(models["year"].max())
dom_rank = models.loc[models["year"] >= START, "domain"].value_counts().index.tolist()
dom_rank += [d for d in models["domain"].value_counts().index if d not in dom_rank]
DEFAULTS = {"f_years": (START, y_hi), "f_domains": dom_rank[:4], "f_frontier": False}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.sidebar.header("Filters for sections 1 and 2")
years = st.sidebar.slider("Publication years", y_lo, y_hi, key="f_years")
domains = st.sidebar.multiselect("Domain (first listed)", dom_rank, key="f_domains",
                                 help="Defaults to the four domains with the most models since 2012.")
frontier_only = st.sidebar.toggle("Frontier models only", key="f_frontier",
                                  help="Epoch AI flag: among the top 10 by training compute when released.")
st.sidebar.button("Reset filters", on_click=reset_filters)
st.sidebar.caption("Sections 3 and 4 use the benchmark tables and have their own controls.")
colour = {d: (OKABE[i] if i < len(OKABE) else OTHER) for i, d in enumerate(dom_rank)}

view = models[models["year"].between(*years) & models["domain"].isin(domains)]
if frontier_only:
    view = view[view["frontier_model"]]
view_all = allm[allm["year"].between(*years) & allm["domain"].isin(domains)]   # incl. models without compute

# ================================================================== 1. compute
st.markdown("---")
st.header("1 · How fast has training compute grown?")
o1, o2, o3 = st.columns([1.2, 2, 1])
colour_by = o1.radio("Colour dots by", ["Domain", "Organisation type"], horizontal=True)
trend_mode = o2.radio("Trend line", ["One trend (95% band)", "Two eras (fitted break)", "Before / after 2010"],
                      horizontal=True)
log_axis = o3.radio("Compute axis", ["Log", "Linear"], horizontal=True) == "Log"

if view.empty:
    st.info(EMPTY)
else:
    xs_all, ys_all = view["decimal_year"].to_numpy(), view["log10_flop"].to_numpy()
    fit = trend_for(xs_all, ys_all)
    segments = []                                   # (start, end, coef, doubling, label)
    if trend_mode.startswith("One") and fit:
        segments = [(xs_all.min(), xs_all.max(), (fit["slope"], fit["intercept"]), fit["doubling"], "All shown")]
    elif trend_mode.startswith("Two"):
        er = eras_for(xs_all, ys_all)
        if er:
            segments = [(xs_all.min(), er["year"], er["before"], er["doubling_before"], f"Before {er['year']:.1f}"),
                        (er["year"], xs_all.max(), er["after"], er["doubling_after"], f"From {er['year']:.1f}")]
    else:
        for lo, hi, lab in [(xs_all.min(), 2010, "Before 2010"), (2010, xs_all.max() + 1e-9, "From 2010")]:
            m = (xs_all >= lo) & (xs_all < hi)
            f = trend_for(xs_all[m], ys_all[m])
            if f:
                segments.append((xs_all[m].min(), xs_all[m].max(), (f["slope"], f["intercept"]), f["doubling"], lab))

    t1, t2, t3 = st.columns(3)
    t1.metric("Models shown", f"{len(view):,}")
    if segments:
        t2.metric(f"Doubling time ({segments[-1][4].lower()})", f"{segments[-1][3]:.1f} months")
        t3.metric("Growth per year" if len(segments) == 1 else f"Doubling time ({segments[0][4].lower()})",
                  f"{10 ** segments[-1][2][0]:.1f}×" if len(segments) == 1 else f"{segments[0][3]:.1f} months")
    else:
        st.info("Too few models, or no upward trend, in this selection to fit a line.")

    fig = go.Figure()
    if trend_mode.startswith("One") and fit and log_axis and "boots" in fit:
        xs = np.linspace(xs_all.min(), xs_all.max(), 60)
        band = fit["boots"][:, [1]] + fit["boots"][:, [0]] * xs
        lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
        fig.add_trace(go.Scatter(x=np.r_[to_date(xs), to_date(xs[::-1])], y=10 ** np.r_[hi, lo[::-1]],
                                 fill="toself", fillcolor="rgba(0,0,0,0.08)", line=dict(width=0),
                                 hoverinfo="skip", name="95% band of the trend"))
    groups = (dom_rank if colour_by == "Domain" else A.ORG_TYPES)
    key, cmap = ("domain", colour) if colour_by == "Domain" else ("org_type", ORG_COLOUR)
    for gname in [g for g in groups if g in set(view[key])]:
        g = view[view[key] == gname]
        fig.add_trace(go.Scatter(
            x=g["publication_date"], y=g["training_compute_flop"], mode="markers", name=gname,
            marker=dict(size=7, opacity=0.8, color=cmap[gname], line=dict(width=0.5, color="white")),
            customdata=np.column_stack([g["model"], g["organization"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x|%b %Y}<br>%{y:.2e} FLOP<extra></extra>"))
    env = view.sort_values("publication_date").assign(top=lambda d: d["training_compute_flop"].cummax())
    fig.add_trace(go.Scatter(x=env["publication_date"], y=env["top"], mode="lines", line_shape="hv",
                             line=dict(color=INK, width=1), name="Largest model so far", hoverinfo="skip"))
    for a_, b_, (s_, i_), dbl, lab in segments:
        xs = np.linspace(a_, b_, 40)
        fig.add_trace(go.Scatter(x=to_date(xs), y=10 ** (i_ + s_ * xs), mode="lines",
                                 line=dict(color="#D55E00", width=2.5, dash="dash"),
                                 name=f"Trend {lab.lower()}: doubles every {dbl:.1f} months", hoverinfo="skip"))
    top = view.nlargest(5, "training_compute_flop")
    fig.add_trace(go.Scatter(x=top["publication_date"], y=top["training_compute_flop"], mode="markers+text",
                             name="Five largest (labelled)", textposition="middle left",
                             text=[f"{m}: {v:.1e}" for m, v in zip(top["model"], top["training_compute_flop"])],
                             textfont=dict(size=11), hoverinfo="skip",
                             marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(width=1.5, color=INK))))
    fig.update_yaxes(type="log" if log_axis else "linear", exponentformat="power")
    st.plotly_chart(base_layout(fig, f"Training compute of notable AI models, {years[0]}–{years[1]} "
                                     "(one dot = one model)", "Publication date",
                                f"Training compute (FLOP, {'log' if log_axis else 'linear'} scale)", 500,
                                colour_by), width="stretch")
    st.caption(f"Baseline: {len(view):,} models in the selected years and domains, "
               f"{view['publication_date'].min():%b %Y} to {view['publication_date'].max():%b %Y}. "
               + ("Log axis: a straight line means constant exponential growth. " if log_axis else
                  "Linear axis: almost every model sits on the zero line because the largest runs are many "
                  "orders of magnitude bigger; switch back to log to compare them. ")
               + "Black step line: the largest training run published up to each date. Labels give the five "
               "largest runs in FLOP. Two-era mode searches 2012–2022 quarterly for the split with the least error.")

    # ---- median and maximum per year
    yr = view.groupby("year")["training_compute_flop"].agg(["median", "max", "size"]).reset_index()
    fig2 = go.Figure()
    for col, name, c, pos in [("max", "Largest run that year", "#E69F00", "top center"),
                              ("median", "Median model that year", "#0072B2", "bottom center")]:
        fig2.add_trace(go.Scatter(x=yr["year"], y=yr[col], mode="lines+markers+text", name=name,
                                  text=[f"{v:.0e}" for v in yr[col]], textposition=pos, textfont=dict(size=9),
                                  line=dict(color=c, width=2.5, dash="dot" if col == "max" else "solid"),
                                  hovertemplate="%{x}: %{y:.2e} FLOP<extra></extra>"))
    fig2.update_yaxes(type="log", exponentformat="power")
    fig2.update_xaxes(dtick=1 if years[1] - years[0] <= 16 else 5)
    st.plotly_chart(base_layout(fig2, "Median and largest training compute of the models published each year",
                                "Publication year", "Training compute (FLOP, log scale)", 400, "Series"),
                    width="stretch")
    st.caption("Same selection as above. Years with few models are noisy; the next chart gives the counts.")

    c1, c2 = st.columns(2)
    with c1:
        per_year = view.groupby("year").size()
        fig3 = go.Figure(go.Bar(x=per_year.index, y=per_year.values, text=per_year.values, textposition="outside",
                                marker_color="#0072B2", hovertemplate="%{x}: %{y} models<extra></extra>"))
        fig3.update_yaxes(rangemode="tozero", range=[0, per_year.max() * 1.18])
        fig3.update_xaxes(dtick=1 if years[1] - years[0] <= 16 else 5)
        st.plotly_chart(base_layout(fig3, "Notable models with a compute estimate, per year", "Publication year",
                                    "Number of models", 360), width="stretch")
        st.caption("Counts models in Epoch's notable list for the selection, not all AI models.")
    with c2:
        c = cov[cov["year"].between(*years)]
        fig4 = go.Figure(go.Bar(x=c["year"], y=c["share_with_compute"] * 100, marker_color="#56B4E9",
                                text=[f"{v:.0%}" for v in c["share_with_compute"]], textposition="outside",
                                textfont=dict(size=10), hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
        fig4.update_yaxes(range=[0, 118], ticksuffix="%")
        fig4.update_xaxes(dtick=1 if years[1] - years[0] <= 16 else 5)
        st.plotly_chart(base_layout(fig4, "Share of notable models that have a compute estimate", "Publication year",
                                    "Models with an estimate (% of notable models)", 360), width="stretch")
        st.caption("All domains (ignores the domain filter). Years with a low share are under-represented above.")

    c3, c4 = st.columns(2)
    with c3:
        mag = np.floor(view["log10_flop"]).astype(int).value_counts().sort_index()
        fig5 = go.Figure(go.Bar(x=[f"10^{k}" for k in mag.index], y=mag.values, text=mag.values,
                                textposition="outside", marker_color="#009E73",
                                hovertemplate="%{x}: %{y} models<extra></extra>"))
        fig5.update_yaxes(rangemode="tozero", range=[0, mag.max() * 1.18])
        st.plotly_chart(base_layout(fig5, "Models by order of magnitude of training compute",
                                    "Training compute (FLOP, lower bound of each power of ten)",
                                    "Number of models", 380), width="stretch")
        st.caption("Distribution of the selection: each bar spans a tenfold range of compute.")
    with c4:
        pv = view.dropna(subset=["parameters"])
        if len(pv) >= 3:
            fig6 = go.Figure()
            for gname in [g for g in groups if g in set(pv[key])]:
                g = pv[pv[key] == gname]
                fig6.add_trace(go.Scatter(x=g["parameters"], y=g["training_compute_flop"], mode="markers", name=gname,
                                          marker=dict(size=6, opacity=0.8, color=cmap[gname]), text=g["model"],
                                          hovertemplate="<b>%{text}</b><br>%{x:.2e} parameters<br>%{y:.2e} FLOP<extra></extra>"))
            big = pv.nlargest(3, "training_compute_flop")
            fig6.add_trace(go.Scatter(x=big["parameters"], y=big["training_compute_flop"], mode="markers+text",
                                      name="Three largest runs", text=big["model"], textposition="middle left",
                                      textfont=dict(size=10), hoverinfo="skip",
                                      marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=1.5, color=INK))))
            fig6.update_xaxes(type="log", exponentformat="power")
            fig6.update_yaxes(type="log", exponentformat="power")
            r = np.corrcoef(np.log10(pv["parameters"]), pv["log10_flop"])[0, 1]
            st.plotly_chart(base_layout(fig6, "Model size against training compute (one dot = one model)",
                                        "Parameters (count, log scale)", "Training compute (FLOP, log scale)", 380,
                                        colour_by), width="stretch")
            st.caption(f"{len(pv):,} models with both values; correlation of the log values r = {r:.2f}. Compute "
                       "estimates are partly derived from parameter counts, so the two are not independent.")
        else:
            st.info("Not enough models with a parameter count in this selection.")

# ================================================================== 2. who trains the largest models
st.markdown("---")
st.header("2 · Who trains the largest models?")
if view.empty:
    st.info(EMPTY)
else:
    runs = view.nlargest(10, "training_compute_flop").iloc[::-1]
    labels = [f"{m} ({o}, {d:%Y})" for m, o, d in zip(runs["model"], runs["organization"], runs["publication_date"])]
    fig7 = go.Figure()
    for t in [t for t in A.ORG_TYPES if t in set(runs["org_type"])]:
        msk = (runs["org_type"] == t).to_numpy()
        fig7.add_trace(go.Scatter(x=runs["training_compute_flop"][msk], y=np.array(labels)[msk], mode="markers+text",
                                  name=t, marker=dict(size=12, color=ORG_COLOUR[t]), textposition="middle right",
                                  text=[fmt_flop(v) for v in runs["training_compute_flop"][msk]],
                                  hovertemplate="%{y}<br>%{x:.2e} FLOP<extra></extra>"))
    lx = np.log10(runs["training_compute_flop"])
    fig7.update_xaxes(type="log", exponentformat="power", range=[lx.min() - 0.3, lx.max() + 0.8])
    fig7.update_yaxes(categoryorder="array", categoryarray=labels, automargin=True)
    st.plotly_chart(base_layout(fig7, "The ten largest training runs in the selection",
                                "Training compute (FLOP, log scale)", "", 420, "Organisation type"), width="stretch")
    st.caption("Several of these estimates are rated 'Likely' or 'Speculative' by Epoch AI (see the trace panel).")

    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(hbar(view["organization"].value_counts().head(10),
                             "Top 10 organisations by number of notable models", "Number of models",
                             "Lead organisation", "#E69F00"), width="stretch")
        st.caption("Lead organisation = first one listed when a model has several. Models with a compute "
                   "estimate in the selection.")
    with d2:
        st.plotly_chart(hbar(view["country"].value_counts().head(8),
                             "Top 8 countries by number of notable models", "Number of models",
                             "Country of lead organisation", "#D55E00"), width="stretch")
        au = int(view["countries_all"].str.contains("Australia").sum())
        st.caption(f"**Models with an Australian organisation in this selection: {au}** (Australia listed among "
                   "any of the model's organisation countries). Only models with a compute estimate are counted.")

    sh = view_all[view_all["year"] >= max(years[0], 2010)]
    if sh.empty:
        st.info("No notable models from 2010 onwards match these filters.")
    else:
        counts = sh.groupby(["year", "org_type"]).size().unstack(fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        fig8 = go.Figure()
        for t in [t for t in A.ORG_TYPES if t in pct.columns]:
            fig8.add_trace(go.Bar(x=pct.index, y=pct[t], name=t, marker_color=ORG_COLOUR[t],
                                  text=[f"{v:.0f}%" if v >= 8 else "" for v in pct[t]], textposition="inside",
                                  insidetextanchor="middle", customdata=counts[t],
                                  hovertemplate="%{x} · " + t + ": %{y:.0f}% (%{customdata} models)<extra></extra>"))
        fig8.update_layout(barmode="stack")
        fig8.update_yaxes(range=[0, 100], ticksuffix="%")
        fig8.update_xaxes(dtick=1)
        base_layout(fig8, "Share of notable models by organisation type, per year (from 2010)",
                    "Publication year", "Share of notable models (%)", 400, "Organisation type")
        fig8.update_layout(legend_traceorder="normal")
        st.plotly_chart(fig8, width="stretch")
        st.caption("All notable models in the selected years and domains, with or without a compute estimate "
                   f"({len(sh):,} models). Segments under 8% are not labelled; their counts are in the table below.")
        with st.expander("Counts behind this chart"):
            st.dataframe(counts.reset_index().rename(columns={"year": "Year"}), hide_index=True, width="stretch")

# ================================================================== 3. capability at fixed compute
st.markdown("---")
st.header("3 · At the same compute, do newer models score higher?")
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

tab_explore, tab_replay = st.tabs(["Explore", "Replay: MMLU 2021 → 2024"])
with tab_explore:
    left, right = st.columns([2.4, 1])
    with left:
        fig9 = go.Figure()
        xs = np.logspace(np.log10(df["compute"].min()), np.log10(df["compute"].max()), 60)
        for i, yr_ in enumerate(yrs):
            part = df[df["year"] == yr_]
            fig9.add_trace(go.Scatter(x=xs, y=A.predict(fit_b, xs, yr_ + 0.5) * scale, mode="lines",
                                      line=dict(color=ramp[yr_], width=2), showlegend=False, hoverinfo="skip"))
            fig9.add_trace(go.Scatter(
                x=part["compute"], y=part["score"] * scale, mode="markers", name=str(yr_),
                marker=dict(size=9, color=ramp[yr_], symbol=symbols[i % len(symbols)], line=dict(width=1, color=INK)),
                text=part["Model version"],
                hovertemplate="<b>%{text}</b><br>%{x:.2e} FLOP<br>score %{y:.1f}<extra></extra>"))
        best = df.loc[df.groupby("year")["score"].idxmax()]
        for _, r in best.iterrows():
            fig9.add_annotation(x=np.log10(r["compute"]), y=r["score"] * scale, showarrow=False, yshift=14,
                                text=f"{r['Model version']}: {r['score'] * scale:.1f}", font=dict(size=10),
                                bgcolor="rgba(255,255,255,.85)")
        if kind == "mmlu":
            fig9.add_hline(y=25, line=dict(color=OTHER, dash="dot"), annotation_text="Random guessing (25%)",
                           annotation_position="bottom right")
        fig9.update_xaxes(type="log", exponentformat="power", dtick=1)
        st.plotly_chart(base_layout(fig9, "Benchmark score against training compute, by release year "
                                          "(lines: fitted score for each year)",
                                    "Training compute (FLOP, log scale)", y_title, 470, "Release year"),
                        width="stretch")
        st.caption(f"{fit_b['n']} models with both a score and a compute estimate. A higher line at the same "
                   "compute means newer models score more without more compute. Labels: best score each year. "
                   "All values are in the table below.")
    with right:
        fig10 = go.Figure(go.Bar(
            x=[fit_b["r2_compute"], fit_b["r2"]], y=["Compute only", "Compute + release year"], orientation="h",
            marker_color=[OTHER, "#0072B2"], text=[f"{fit_b['r2_compute']:.2f}", f"{fit_b['r2']:.2f}"],
            textposition="outside", hovertemplate="%{y}: R² = %{x:.2f}<extra></extra>"))
        fig10.update_xaxes(range=[0, 1.15])
        st.plotly_chart(base_layout(fig10, "Share of score variance explained (R²)", "R² (0 to 1, no unit)", "", 230),
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

with tab_replay:
    xs = np.logspace(22, 25.7, 60)
    steps = np.arange(2021.75, 2025.01, 0.25)
    colour_of = dict(zip(sorted(mmlu["year"].unique()), YEAR_RAMP))

    def frame_data(t):
        seen = mmlu[mmlu["t"] <= t]
        return [go.Scatter(x=xs, y=A.predict(fit_m, xs, t) * 100, mode="lines", line=dict(color="#D55E00", width=3),
                           name="Fitted score at this date", hoverinfo="skip"),
                go.Scatter(x=seen["compute"], y=seen["score"] * 100, mode="markers", name="Models released so far",
                           marker=dict(size=10, color=seen["year"].map(colour_of), line=dict(width=1, color=INK)),
                           text=seen["Model version"], hovertemplate="<b>%{text}</b><br>%{y:.1f}%<extra></extra>")]

    def qlabel(t):
        return f"{int(t)} Q{int(round((t % 1) * 4)) + 1}"

    rep = go.Figure(data=frame_data(steps[0]), frames=[go.Frame(data=frame_data(t), name=qlabel(t)) for t in steps])
    rep.update_layout(
        updatemenus=[dict(type="buttons", x=0, y=1.2, xanchor="left", direction="left", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=600, redraw=True),
                                                         transition=dict(duration=300), fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=1.24, currentvalue=dict(prefix="Up to "),
                      steps=[dict(label=qlabel(t), method="animate",
                                  args=[[qlabel(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in steps])])
    rep.add_hline(y=25, line=dict(color=OTHER, dash="dot"), annotation_text="Random guessing (25%)",
                  annotation_position="bottom right")
    rep.update_xaxes(type="log", exponentformat="power", dtick=1, range=[22, 25.7])
    rep.update_yaxes(range=[15, 95])
    base_layout(rep, "How the fitted MMLU curve rose, quarter by quarter", "Training compute (FLOP, log scale)",
                "MMLU score (% correct, 5-shot)", 560, "Series")
    rep.update_layout(margin=dict(t=130))
    st.plotly_chart(rep, width="stretch")
    st.caption("Press Play or drag the slider. Orange line: fitted MMLU score at that date; it lifts at every "
               "compute level as time passes. Dots: models released up to that date, shaded by release year.")

# ---- iso-capability map (Chenyu) and the 80% path
st.subheader("Iso-capability map: the compute needed for a given score falls every year")
gx, gy = np.linspace(2021.5, 2025.0, 120), np.linspace(22.0, 25.7, 120)
GX, GY = np.meshgrid(gx, gy)
Z = A.predict(fit_m, 10 ** GY, GX) * 100
iso = go.Figure(go.Contour(
    x=gx, y=gy, z=Z, colorscale=SEQ, zmin=25, zmax=95,
    contours=dict(start=40, end=90, size=10, showlabels=True, labelfont=dict(size=12, color=INK)),
    line=dict(color="rgba(255,255,255,.8)", width=1.2),
    colorbar=dict(title=dict(text="Fitted MMLU (%)", side="right"), thickness=12, len=0.8),
    hovertemplate="%{x:.1f}<br>10^%{y:.1f} FLOP<br>fitted MMLU %{z:.1f}%<extra></extra>"))
iso.add_trace(go.Scatter(x=mmlu["t"], y=np.log10(mmlu["compute"]), mode="markers", name="Models (actual)",
                         marker=dict(size=7, color=INK, opacity=0.7, line=dict(width=1, color="white")),
                         text=mmlu["Model version"], customdata=mmlu["score"] * 100,
                         hovertemplate="<b>%{text}</b><br>actual MMLU %{customdata:.1f}%<extra></extra>"))
iso.add_trace(go.Scatter(x=path80["t"], y=np.log10(path80["compute"]), mode="lines+markers+text",
                         line=dict(color="#E69F00", width=3), name="Cheapest model ≥ 80% so far",
                         marker=dict(size=12, color="#E69F00", line=dict(width=2, color="white")),
                         text=path80["Model version"], textfont=dict(color=INK, size=12),
                         textposition=["top right"] * (len(path80) - 1) + ["top left"]))
iso.update_yaxes(tickvals=list(range(22, 26)), ticktext=[f"10^{e}" for e in range(22, 26)], range=[22, 25.7])
iso.update_xaxes(tickvals=[2022, 2023, 2024, 2025], ticktext=["2022", "2023", "2024", "2025"], range=[2021.5, 2025.0])
base_layout(iso, "Fitted MMLU score for any release date and compute budget (contour labels = score %)",
            "Release date", "Training compute (FLOP, log scale)", 500, "Series")
st.plotly_chart(iso, width="stretch")
st.caption(f"Each contour is one fitted score level (40%–90%). Contours slope down: the same score needs about "
           f"{fit_m['rate']:.1f}× less compute each year. Fitted on the {len(mmlu)} MMLU models (2021–2024); the "
           "map shows the model's fit, not extra data.")

fig11 = go.Figure(go.Bar(
    y=[f"{r['Model version']} ({pd.to_datetime(r['release']):%b %Y}, {r['score'] * 100:.1f}%)"
       for _, r in path80.iterrows()],
    x=path80["compute"], orientation="h", marker_color="#E69F00",
    text=[fmt_flop(v) for v in path80["compute"]], textposition="outside",
    hovertemplate="%{y}<br>%{x:.2e} FLOP<extra></extra>"))
fig11.update_xaxes(type="log", exponentformat="power", range=[np.log10(path80["compute"].min()) - 0.5,
                                                              np.log10(path80["compute"].max()) + 0.9])
fig11.update_yaxes(autorange="reversed", automargin=True)
st.plotly_chart(base_layout(fig11, "Training compute of each model that reached 80% on MMLU with less compute "
                                   "than any earlier model", "Training compute (FLOP, log scale)", "", 300),
                width="stretch")
st.caption(f"The orange path on the map, read top to bottom in release order: the compute needed for an 80% MMLU "
           f"score fell {drop80:.0f}× in {months80:.0f} months.")

# ================================================================== 4. decomposition
st.markdown("---")
st.header("4 · Split one model-to-model gain into compute and methods")
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
    steps_ = [c_part * 100, a_part * 100, o_part * 100]
    fig12 = go.Figure(go.Waterfall(
        x=[a, "More compute", "Better methods", "Model-specific", b],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[ra["score"] * 100, *steps_, 0],
        text=[f"{ra['score'] * 100:.1f}%", *[f"{v:+.1f} pts" for v in steps_], f"{rb['score'] * 100:.1f}%"],
        textposition="outside", connector=dict(line=dict(color=OTHER, dash="dot")),
        increasing=dict(marker_color="#0072B2"), decreasing=dict(marker_color="#D55E00"),
        totals=dict(marker_color=INK)))
    lo = min(ra["score"], rb["score"]) * 100
    fig12.update_yaxes(range=[max(0, lo - 25), 105])
    st.plotly_chart(base_layout(fig12, f"MMLU score change from {a} to {b}, split by the fitted model",
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
           f"explains R² = {fit_m['r2_compute']:.2f} of score variance, and the largest runs come from a few "
           "industry labs.")
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

with st.expander("Charts we chose not to draw, and why"):
    st.markdown(
        "**Stacked \"share of gains\" area (compute vs algorithms).** Rejected because it implies that compute "
        "and algorithms add up to a known total, each with a measurable share. The evidence cannot support that "
        "precision: the algorithm share above is an estimate with a wide interval, not a measured part of a total.\n\n"
        "**Dual-axis line chart (compute on one axis, capability on the other).** Rejected because where the two "
        "lines cross depends on how each axis is scaled, so the chart could manufacture the conclusion. Every "
        "chart on this page uses one labelled scale per axis.")

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
    st.download_button("Download the rows plotted in sections 1 and 2 (CSV)", view.to_csv(index=False),
                       "plotted_rows.csv", "text/csv")

st.caption(f"**Sources:** {SOURCES} Raw files downloaded 23 Oct 2026 (benchmarks) and 20 Nov 2026 (notable "
           f"models). **Data last refreshed:** {meta0['processed_on']}; latest model with a compute estimate: "
           f"{meta0['latest_observation']}.")

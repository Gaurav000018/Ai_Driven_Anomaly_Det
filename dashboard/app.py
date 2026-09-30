"""SENTINEL-BI QA inspector dashboard.

    streamlit run dashboard/app.py

Built around what an inspector actually does: look at a lot, find the parts the
system flagged, understand *why*, and decide. The trajectory plot is the centre
of it - a flagged part drawn against its lot-mates makes the case in a way no
table of scores does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402

from src.api.service import ScoringService  # noqa: E402
from src.knowledge.library import MechanismLibrary  # noqa: E402
from src.fusion.adaptive_burnin import extend_burnin_whatif  # noqa: E402
from src.ingest.csv_reader import read_csv  # noqa: E402

st.set_page_config(page_title="SENTINEL-BI", layout="wide")

BAND_COLOUR = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "REJECT": "#c62828"}


@st.cache_resource
def load_service() -> ScoringService:
    return ScoringService()


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return read_csv(path, library=MechanismLibrary.load())


@st.cache_data
def score(path: str, lot_id: str):
    svc = load_service()
    df = load_data(path)
    lot_df = df[df["lot_id"].astype(str) == lot_id]
    result = svc.score_lot(lot_df)
    return result.decisions, result.attributions, result.features, result.scores


def trajectory_figure(df: pd.DataFrame, parameter: str, highlight: list[str], limit: float) -> go.Figure:
    sub = df[df["param_name"] == parameter]
    wide = sub.pivot_table(index="part_id", columns="hours", values="value")
    hours = list(wide.columns)

    fig = go.Figure()
    # Draw the population as a single faint band rather than thousands of
    # traces; the browser cannot render 500 individual lines usefully and the
    # inspector cannot read them either.
    q = wide.quantile([0.05, 0.5, 0.95])
    fig.add_trace(go.Scatter(x=hours + hours[::-1],
                             y=list(q.loc[0.95]) + list(q.loc[0.05])[::-1],
                             fill="toself", fillcolor="rgba(120,120,120,0.15)",
                             line=dict(width=0), name="lot 5-95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hours, y=list(q.loc[0.5]), mode="lines",
                             line=dict(color="#555", width=2, dash="dash"), name="lot median"))

    for pid in highlight[:15]:
        if pid in wide.index:
            fig.add_trace(go.Scatter(x=hours, y=list(wide.loc[pid]), mode="lines+markers",
                                     line=dict(width=2), name=str(pid)))

    fig.add_hline(y=limit, line=dict(color="#c62828", dash="dot"),
                  annotation_text="derated limit", annotation_position="top left")
    fig.update_layout(height=460, xaxis_title="burn-in hours", yaxis_title=parameter,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=10)))
    return fig


def wafer_map(features: pd.DataFrame, decisions: pd.DataFrame) -> go.Figure:
    df = features.join(decisions[["decision"]])
    if "x" not in df.columns or df["x"].isna().all():
        return go.Figure()
    fig = go.Figure()
    for band, colour in BAND_COLOUR.items():
        sel = df[df["decision"] == band]
        if sel.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sel["x"], y=sel["y"], mode="markers", name=band,
            marker=dict(size=9, color=colour, opacity=0.85 if band != "ACCEPT" else 0.35),
            text=sel.index, hovertemplate="%{text}<br>(%{x}, %{y})<extra></extra>",
        ))
    fig.update_layout(height=430, xaxis_title="die x", yaxis_title="die y",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def main() -> None:
    st.title("SENTINEL-BI")
    st.caption("Latent defect screening for component burn-in - SIH26170")

    default = ROOT / "data" / "synthetic" / "burnin.csv"
    path = st.sidebar.text_input("measurement CSV", str(default))
    if not Path(path).exists():
        st.error(f"not found: {path}. Run `python scripts/generate_data.py` first.")
        return

    try:
        svc = load_service()
    except Exception as exc:
        st.error(f"model unavailable: {exc}")
        st.info("Run `python scripts/train_fusion.py` to fit the model.")
        return

    df = load_data(path)
    lots = sorted(df["lot_id"].astype(str).unique())
    lot_id = st.sidebar.selectbox("lot", lots)
    parameter = st.sidebar.selectbox("parameter", svc.library.parameter_names())

    with st.spinner(f"scoring {lot_id} ..."):
        decisions, attributions, features, scores = score(path, lot_id)

    lot_df = df[df["lot_id"].astype(str) == lot_id]
    counts = decisions["decision"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("parts", f"{len(decisions):,}")
    c2.metric("accept", f"{int(counts.get('ACCEPT', 0)):,}")
    c3.metric("review", f"{int(counts.get('REVIEW', 0)):,}")
    c4.metric("reject", f"{int(counts.get('REJECT', 0)):,}")
    c5.metric("unknown mechanism", f"{int(attributions['is_unknown'].sum()):,}")

    if "is_defect" in features.columns:
        y = features["is_defect"].astype(bool)
        escaped = int((y & (decisions["decision"] == "ACCEPT")).sum())
        st.caption(
            f"ground truth available: {int(y.sum())} injected defects, "
            f"{escaped} accepted (escapes), "
            f"{int((y & (decisions['decision'] != 'ACCEPT')).sum())} flagged"
        )

    flagged = decisions[decisions["decision"] != "ACCEPT"].sort_values("escape_risk", ascending=False)

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{parameter} trajectories")
        st.plotly_chart(
            trajectory_figure(lot_df, parameter, list(flagged.index),
                              svc.library.parameter(parameter).derated_limit),
            use_container_width=True,
        )
    with right:
        st.subheader("wafer map")
        st.plotly_chart(wafer_map(features, decisions), use_container_width=True)

    st.subheader(f"flagged parts ({len(flagged):,})")
    table = flagged.join(attributions[["mechanism_name", "distance", "is_unknown", "severity"]])
    st.dataframe(
        table[["escape_risk", "decision", "mechanism_name", "distance", "severity", "is_unknown"]]
        .rename(columns={"distance": "fingerprint_widths"}),
        use_container_width=True, height=260,
    )

    st.subheader("certificate")
    if len(flagged):
        pid = st.selectbox("part", list(flagged.index))
        from src.api.service import LotScore

        lot_score = LotScore(decisions=decisions, attributions=attributions,
                             features=features, scores=scores)
        st.code(svc.certificate(lot_score, pid)["text"], language="text")
    else:
        st.info("no parts flagged in this lot")

    st.subheader("what-if: extend the soak")
    extra = st.slider("additional burn-in hours", 0, 168, 72, step=24)
    if extra:
        wi = extend_burnin_whatif(
            decisions["escape_risk"],
            features["is_defect"].astype(int) if "is_defect" in features else pd.Series(0, index=features.index),
            accept_below=float(svc.decider_.accept_below_ or 1.0),
            extra_hours=float(extra),
        )
        a, b, c = st.columns(3)
        a.metric("borderline parts", f"{wi['borderline_parts']:,}")
        b.metric("expected extra defects caught", f"{wi['expected_extra_defects_caught']:,}")
        c.metric("added cost", f"{wi['added_cost']:,.0f}")
        st.caption(f"Note: {wi['caveat']}.")


if __name__ == "__main__":
    main()

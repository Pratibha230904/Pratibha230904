import io
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud


@dataclass(frozen=True)
class Thresholds:
    positive_threshold: float
    negative_threshold: float


SENTIMENT_ORDER = ["positive", "neutral", "negative"]


def set_page() -> None:
    st.set_page_config(
        page_title="Twitter Sentiment Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_csv(file, max_rows: int) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, engine="python", on_bad_lines="skip")
    if max_rows is not None and max_rows > 0:
        return df.head(max_rows).copy()
    return df


def guess_text_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    # Prefer columns explicitly named like tweet text
    candidate_names = [
        "text",
        "tweet",
        "tweet_text",
        "content",
        "message",
        "body",
        "full_text",
        "review",
        "comment",
        "sentence",
    ]
    lower_name_to_col = {c.lower(): c for c in df.columns}
    for name in candidate_names:
        if name in lower_name_to_col:
            return lower_name_to_col[name]
    # Fallback: first object/string-like column with average length > 10
    object_like_cols = [c for c in df.columns if df[c].dtype == "object"]
    if object_like_cols:
        scored: List[Tuple[str, float]] = []
        for c in object_like_cols:
            series = df[c].dropna().astype(str)
            if series.empty:
                continue
            avg_len = series.str.len().mean()
            scored.append((c, avg_len))
        if scored:
            scored.sort(key=lambda p: p[1], reverse=True)
            return scored[0][0]
    return None


@st.cache_resource(show_spinner=False)
def get_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def clean_text_for_wordcloud(text: str) -> str:
    text = text or ""
    # remove URLs, mentions, hashtags, and extra spaces
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def apply_sentiment(
    df: pd.DataFrame,
    text_col: str,
    thresholds: Thresholds,
) -> pd.DataFrame:
    analyzer = get_analyzer()

    def score_row(text: str) -> Tuple[float, float, float, float, str]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0.0, 0.0, 0.0, "neutral"
        scores = analyzer.polarity_scores(text)
        compound = float(scores.get("compound", 0.0))
        pos = float(scores.get("pos", 0.0))
        neu = float(scores.get("neu", 0.0))
        neg = float(scores.get("neg", 0.0))
        if compound >= thresholds.positive_threshold:
            label = "positive"
        elif compound <= thresholds.negative_threshold:
            label = "negative"
        else:
            label = "neutral"
        return compound, pos, neu, neg, label

    text_series = df[text_col].astype(str)
    results = text_series.apply(score_row)
    df_out = df.copy()
    (
        df_out["compound"],
        df_out["pos"],
        df_out["neu"],
        df_out["neg"],
        df_out["sentiment"],
    ) = zip(*results)
    return df_out


def filter_dataframe(
    df: pd.DataFrame,
    text_col: str,
    include_keyword: str,
    exclude_keyword: str,
    include_labels: List[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df
    if include_labels:
        filtered = filtered[filtered["sentiment"].isin(include_labels)]
    if include_keyword:
        kw = include_keyword.strip().lower()
        filtered = filtered[filtered[text_col].str.lower().str.contains(re.escape(kw), na=False)]
    if exclude_keyword:
        kw = exclude_keyword.strip().lower()
        filtered = filtered[~filtered[text_col].str.lower().str.contains(re.escape(kw), na=False)]
    return filtered


def build_sidebar_controls() -> Tuple[Thresholds, str, str, List[str], int]:
    st.sidebar.caption("Controls")

    pos_th = st.sidebar.slider(
        "positive threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
    )
    neg_th = st.sidebar.slider(
        "negative threshold",
        min_value=-0.5,
        max_value=0.0,
        value=-0.05,
        step=0.01,
    )

    include_kw = st.sidebar.text_input("include keyword (optional)")
    exclude_kw = st.sidebar.text_input("exclude keyword (optional)")

    st.sidebar.markdown("Filter labels")
    label_cols = st.sidebar.columns([1, 1, 1])
    with label_cols[0]:
        pos_on = st.checkbox("positive", value=True)
    with label_cols[1]:
        neu_on = st.checkbox("neutral", value=True)
    with label_cols[2]:
        neg_on = st.checkbox("negative", value=True)

    top_examples = st.sidebar.slider("Top examples", min_value=3, max_value=20, value=5)

    include_labels = [
        label
        for label, flag in zip(SENTIMENT_ORDER, [pos_on, neu_on, neg_on])
        if flag
    ]

    return Thresholds(pos_th, neg_th), include_kw, exclude_kw, include_labels, top_examples


def header() -> None:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("## 🐦 Twitter Sentiment Analytics")
        st.write(
            "Upload your tweets dataset and get comprehensive insights with interactive visualizations, statistical analysis, and exportable reports."
        )
        st.markdown(
            "- **Interactive Charts**\n"
            "- **Trend/Distribution Analysis**\n"
            "- **Word Clouds & Top Examples**\n"
            "- **Export CSV / Excel**"
        )
    with right:
        st.markdown("### Upload Your Dataset")
        st.caption("Choose CSV file")
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["csv"],
            label_visibility="collapsed",
            accept_multiple_files=False,
        )
        max_rows = st.slider("Maximum rows", min_value=50, max_value=50000, value=2000, step=50)
        run = st.button("Run Sample Analysis", type="primary")
    st.divider()
    st.session_state.setdefault("uploaded_file", None)
    st.session_state.setdefault("max_rows", 2000)
    st.session_state.setdefault("run_clicked", False)

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["max_rows"] = max_rows
    if run:
        st.session_state["run_clicked"] = True


def render():
    set_page()
    thresholds, include_kw, exclude_kw, include_labels, top_examples = build_sidebar_controls()

    header()

    uploaded_file = st.session_state.get("uploaded_file")
    max_rows = st.session_state.get("max_rows", 2000)
    run_clicked = st.session_state.get("run_clicked", False)

    if uploaded_file is None:
        st.info("Upload a CSV to get started.")
        return

    df = load_csv(uploaded_file, max_rows=max_rows)
    if df.empty:
        st.warning("The uploaded file appears to be empty or unreadable.")
        return

    text_col = guess_text_column(df)
    if text_col is None:
        st.error("Could not detect a text column. Please ensure your CSV has a text-like column (e.g., 'text').")
        st.dataframe(df.head(10))
        return

    st.caption(f"Detected text column: `{text_col}`")

    if not run_clicked:
        st.stop()

    with st.spinner("Analyzing sentiments..."):
        df_scored = apply_sentiment(df, text_col, thresholds)

    tab_overview, tab_dist, tab_examples, tab_wc, tab_export = st.tabs(
        ["Overview", "Distribution", "Top Examples", "Word Clouds", "Export"]
    )

    with tab_overview:
        counts = df_scored["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        total = int(counts.sum())
        pos_count = int(counts.get("positive", 0))
        neu_count = int(counts.get("neutral", 0))
        neg_count = int(counts.get("negative", 0))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", f"{total:,}")
        m2.metric("Positive", f"{pos_count:,}")
        m3.metric("Neutral", f"{neu_count:,}")
        m4.metric("Negative", f"{neg_count:,}")

        st.markdown("#### Label distribution")
        bar_df = counts.reset_index()
        bar_df.columns = ["sentiment", "count"]
        bar_fig = px.bar(
            bar_df,
            x="sentiment",
            y="count",
            color="sentiment",
            category_orders={"sentiment": SENTIMENT_ORDER},
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#64748b",
                "negative": "#ef4444",
            },
        )
        bar_fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(bar_fig, use_container_width=True)

    with tab_dist:
        st.markdown("#### Compound score distribution")
        hist_fig = px.histogram(
            df_scored,
            x="compound",
            nbins=50,
            color="sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#64748b",
                "negative": "#ef4444",
            },
        )
        hist_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(hist_fig, use_container_width=True)

    with tab_examples:
        st.markdown("#### Top examples")
        filtered = filter_dataframe(
            df_scored,
            text_col=text_col,
            include_keyword=include_kw,
            exclude_keyword=exclude_kw,
            include_labels=include_labels,
        )
        examples = (
            filtered.sort_values("compound", ascending=False)
            if include_labels == ["positive"]
            else (
                filtered.sort_values("compound", ascending=True)
                if include_labels == ["negative"]
                else filtered.sort_values("compound", ascending=False)
            )
        )
        cols = st.columns(3)
        for idx, label in enumerate(SENTIMENT_ORDER):
            with cols[idx]:
                sub = examples[examples["sentiment"] == label].head(top_examples)[
                    [text_col, "compound"]
                ]
                st.markdown(f"**{label.capitalize()}**")
                st.dataframe(sub, use_container_width=True, hide_index=True)

    with tab_wc:
        st.markdown("#### Word Clouds")
        wc_cols = st.columns(3)
        for idx, label in enumerate(SENTIMENT_ORDER):
            with wc_cols[idx]:
                subset = df_scored[df_scored["sentiment"] == label][text_col].astype(str)
                text_blob = " ".join(clean_text_for_wordcloud(t) for t in subset)
                if not text_blob.strip():
                    st.info(f"No text for {label}.")
                    continue
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color="#0E1117",
                    colormap="viridis",
                    stopwords=STOPWORDS,
                ).generate(text_blob)
                st.image(wc.to_array(), caption=label.capitalize(), use_column_width=True)

    with tab_export:
        st.markdown("#### Export Reports")
        results_csv = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", results_csv, file_name="sentiment_results.csv", mime="text/csv")

        pos_csv = df_scored[df_scored["sentiment"] == "positive"].to_csv(index=False).encode("utf-8")
        neu_csv = df_scored[df_scored["sentiment"] == "neutral"].to_csv(index=False).encode("utf-8")
        neg_csv = df_scored[df_scored["sentiment"] == "negative"].to_csv(index=False).encode("utf-8")

        cols = st.columns(3)
        with cols[0]:
            st.download_button("Download positive CSV", pos_csv, file_name="positive.csv", mime="text/csv")
        with cols[1]:
            st.download_button("Download neutral CSV", neu_csv, file_name="neutral.csv", mime="text/csv")
        with cols[2]:
            st.download_button("Download negative CSV", neg_csv, file_name="negative.csv", mime="text/csv")


if __name__ == "__main__":
    render()

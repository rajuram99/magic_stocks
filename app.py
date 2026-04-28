import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



# Page setup

st.set_page_config(
    page_title="Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHANNELS = ["tv", "digital", "social", "audio"]
CHANNEL_LABELS = {
    "tv": "TV",
    "digital": "Digital",
    "social": "Social",
    "audio": "Audio",
}
SPEND_COLS = [f"{c}_spend" for c in CHANNELS]
CONTRIB_COLS = [f"{c}_contribution" for c in CHANNELS]



# Data loading from root directory

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    game_path = base_dir / "game_data.csv"
    curve_path = base_dir / "response_curves.csv"

    if not game_path.exists() or not curve_path.exists():
        st.error(
            "Could not find `game_data.csv` and/or `response_curves.csv`. "
            "Place both files in the same folder as this Streamlit app."
        )
        st.stop()

    games = pd.read_csv(game_path)
    curves = pd.read_csv(curve_path)

    games["date"] = pd.to_datetime(games["date"])
    games["month"] = games["date"].dt.to_period("M").astype(str)
    games["week"] = games["date"].dt.to_period("W").apply(lambda r: r.start_time)

    
    games["total_contribution"] = games[CONTRIB_COLS].sum(axis=1)
    games["contribution_share"] = np.where(
        games["total_viewership"] > 0,
        games["total_contribution"] / games["total_viewership"],
        np.nan,
    )
    games["viewers_per_dollar"] = np.where(
        games["total_spend"] > 0,
        games["total_contribution"] / games["total_spend"],
        np.nan,
    )

    for channel in CHANNELS:
        spend_col = f"{channel}_spend"
        contrib_col = f"{channel}_contribution"
        games[f"{channel}_roi"] = np.where(
            games[spend_col] > 0,
            games[contrib_col] / games[spend_col],
            np.nan,
        )

    # Normalize column names.
    curves = curves.rename(columns={c: c.strip().lower() for c in curves.columns})
    if "channel" not in curves.columns:
        st.error("`response_curves.csv` must include a `channel` column.")
        st.stop()

    spend_candidates = ["spend", "marketing_spend", "media_spend", "cost"]
    contrib_candidates = [
        "contribution_millions",
        "contribution",
        "viewership_contribution",
        "viewers",
        "incremental_viewers",
    ]
    spend_col = next((c for c in spend_candidates if c in curves.columns), None)
    contrib_col = next((c for c in contrib_candidates if c in curves.columns), None)

    if spend_col is None or contrib_col is None:
        st.error(
            "`response_curves.csv` must include spend and contribution columns. "
            "Expected names like `spend` and `contribution`."
        )
        st.stop()

    curves = curves.rename(columns={spend_col: "spend", contrib_col: "contribution"})
    curves["channel"] = curves["channel"].astype(str).str.lower().str.strip()
    curves = curves[curves["channel"].isin(CHANNELS)].copy()

    return games, curves


games_raw, curves_raw = load_data()



# Sidebar filters

st.sidebar.title("Filters")

min_date = games_raw["date"].min().date()
max_date = games_raw["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

selected_slots = st.sidebar.multiselect(
    "Time slot",
    options=sorted(games_raw["time_slot"].dropna().unique()),
    default=sorted(games_raw["time_slot"].dropna().unique()),
)

marquee_filter = st.sidebar.radio(
    "Game type",
    options=["All", "Marquee only", "Regular only"],
    horizontal=False,
)

teams = sorted(set(games_raw["home_team"].dropna()).union(set(games_raw["away_team"].dropna())))
selected_teams = st.sidebar.multiselect("Team involved", options=teams, default=[])


games = games_raw[
    (games_raw["date"].dt.date >= start_date)
    & (games_raw["date"].dt.date <= end_date)
    & (games_raw["time_slot"].isin(selected_slots))
].copy()

if marquee_filter == "Marquee only":
    games = games[games["marquee_game"] == 1]
elif marquee_filter == "Regular only":
    games = games[games["marquee_game"] == 0]

if selected_teams:
    games = games[
        games["home_team"].isin(selected_teams) | games["away_team"].isin(selected_teams)
    ]

if games.empty:
    st.warning("No games match the selected filters.")
    st.stop()


# Helper functions

def fmt_millions(value: float) -> str:
    return f"{value:,.1f}M"


def fmt_dollars(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        spend = df[f"{channel}_spend"].sum()
        contribution = df[f"{channel}_contribution"].sum()
        roi = contribution / spend if spend else np.nan
        rows.append(
            {
                "Channel": CHANNEL_LABELS[channel],
                "channel_key": channel,
                "Spend": spend,
                "Contribution": contribution,
                "Viewers per $1K": roi * 1000 * 1_000_000 if pd.notna(roi) else np.nan,
                "Spend Share": spend / df["total_spend"].sum() if df["total_spend"].sum() else np.nan,
                "Contribution Share": contribution / df[CONTRIB_COLS].sum().sum() if df[CONTRIB_COLS].sum().sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_insights(df: pd.DataFrame) -> list[str]:
    ch = channel_summary(df)
    top_contrib = ch.sort_values("Contribution", ascending=False).iloc[0]
    top_eff = ch.sort_values("Viewers per $1K", ascending=False).iloc[0]
    bottom_eff = ch.sort_values("Viewers per $1K", ascending=True).iloc[0]

    marquee_avg = df.groupby("marquee_game")["total_viewership"].mean()
    if {0, 1}.issubset(set(marquee_avg.index)):
        marquee_lift = marquee_avg.loc[1] - marquee_avg.loc[0]
        marquee_text = (
            f"Marquee games average {marquee_lift:,.1f}M more viewers than regular games; "
            "use them as anchor inventory for the channels with the strongest response."
        )
    else:
        marquee_text = "The current filter contains only one game type, so marquee lift cannot be compared here."

    slot = df.groupby("time_slot")["total_viewership"].mean().sort_values(ascending=False)
    best_slot = slot.index[0]
    worst_slot = slot.index[-1]
    slot_gap = slot.iloc[0] - slot.iloc[-1]

    contribution_share = df["total_contribution"].sum() / df["total_viewership"].sum()

    return [
        f"Marketing-driven contribution accounts for {contribution_share:.0%} of total viewership in the selected games.",
        f"{top_contrib['Channel']} drives the most incremental viewers ({top_contrib['Contribution']:,.1f}M total contribution).",
        f"{top_eff['Channel']} is the most efficient channel at {top_eff['Viewers per $1K']:,.0f} viewers per $1K, while {bottom_eff['Channel']} is the least efficient.",
        marquee_text,
        f"{best_slot} games outperform {worst_slot} games by {slot_gap:,.1f}M viewers on average.",
    ]


def add_curve_markers(fig: go.Figure, df: pd.DataFrame, channel: str) -> go.Figure:
    spend_col = f"{channel}_spend"
    contrib_col = f"{channel}_contribution"

    fig.add_trace(
        go.Scatter(
            x=df[spend_col],
            y=df[contrib_col],
            mode="markers",
            name="Games",
            marker={"size": 7, "opacity": 0.45},
            hovertemplate="Spend: $%{x:,.0f}<br>Contribution: %{y:.2f}M<extra></extra>",
        )
    )

    avg_spend = df[spend_col].mean()
    avg_contrib = df[contrib_col].mean()
    fig.add_trace(
        go.Scatter(
            x=[avg_spend],
            y=[avg_contrib],
            mode="markers+text",
            name="Average game",
            marker={"size": 14, "symbol": "diamond"},
            text=["Avg"],
            textposition="top center",
            hovertemplate="Average spend: $%{x:,.0f}<br>Average contribution: %{y:.2f}M<extra></extra>",
        )
    )
    return fig


# Main app

st.title("Marketing Viewership Dashboard")
st.caption("Understand how marketing spend translates into incremental game viewership.")

page = st.sidebar.radio("Dashboard page", ["1. Overview", "2. Channel Deep Dive"])


if page == "1. Overview":
    st.header("Overview")

    total_games = len(games)
    total_viewership = games["total_viewership"].sum()
    avg_viewership = games["total_viewership"].mean()
    total_spend = games["total_spend"].sum()
    total_contribution = games["total_contribution"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", f"{total_games:,}")
    c2.metric("Total viewership", fmt_millions(total_viewership))
    c3.metric("Avg. viewership / game", fmt_millions(avg_viewership))
    c4.metric("Total spend", fmt_dollars(total_spend))
    c5.metric("Marketing contribution", fmt_millions(total_contribution))

    st.subheader("Executive insights")
    for insight in make_insights(games)[:5]:
        st.markdown(f"- {insight}")

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Viewership trend over time")
        trend = (
            games.groupby("date", as_index=False)
            .agg(
                total_viewership=("total_viewership", "sum"),
                base_viewership=("base_viewership", "sum"),
                total_contribution=("total_contribution", "sum"),
                total_spend=("total_spend", "sum"),
            )
            .sort_values("date")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend["date"],
                y=trend["total_viewership"],
                mode="lines+markers",
                name="Total viewership",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trend["date"],
                y=trend["base_viewership"],
                mode="lines",
                name="Base viewership",
            )
        )
        fig.update_layout(
            yaxis_title="Viewers (M)",
            xaxis_title=None,
            hovermode="x unified",
            legend_title=None,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("What drives total viewership?")
        waterfall = pd.DataFrame(
            {
                "Component": ["Base", "TV", "Digital", "Social", "Audio"],
                "Viewers": [
                    games["base_viewership"].sum(),
                    games["tv_contribution"].sum(),
                    games["digital_contribution"].sum(),
                    games["social_contribution"].sum(),
                    games["audio_contribution"].sum(),
                ],
            }
        )
        fig = px.bar(
            waterfall,
            x="Component",
            y="Viewers",
            text="Viewers",
            title=None,
        )
        fig.update_traces(texttemplate="%{text:.1f}M", textposition="outside")
        fig.update_layout(
            yaxis_title="Viewers (M)",
            xaxis_title=None,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel contribution vs. spend")
    ch = channel_summary(games)
    ch_display = ch.copy()
    ch_display["Spend Label"] = ch_display["Spend"].map(fmt_dollars)
    ch_display["Contribution Label"] = ch_display["Contribution"].map(fmt_millions)

    fig = px.scatter(
        ch,
        x="Spend",
        y="Contribution",
        size="Contribution Share",
        text="Channel",
        hover_data={
            "Spend": ":,.0f",
            "Contribution": ":.2f",
            "Viewers per $1K": ":.2f",
            "Spend Share": ":.0%",
            "Contribution Share": ":.0%",
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Total spend ($)",
        yaxis_title="Contribution (M viewers)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("By time slot")
        slot = (
            games.groupby("time_slot", as_index=False)
            .agg(avg_viewership=("total_viewership", "mean"), games=("game_id", "count"))
            .sort_values("avg_viewership", ascending=False)
        )
        fig = px.bar(slot, x="time_slot", y="avg_viewership", text="avg_viewership")
        fig.update_traces(texttemplate="%{text:.1f}M", textposition="outside")
        fig.update_layout(xaxis_title=None, yaxis_title="Avg viewers (M)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Marquee effect")
        marquee = (
            games.assign(game_type=np.where(games["marquee_game"] == 1, "Marquee", "Regular"))
            .groupby("game_type", as_index=False)
            .agg(avg_viewership=("total_viewership", "mean"), games=("game_id", "count"))
        )
        fig = px.bar(marquee, x="game_type", y="avg_viewership", text="avg_viewership")
        fig.update_traces(texttemplate="%{text:.1f}M", textposition="outside")
        fig.update_layout(xaxis_title=None, yaxis_title="Avg viewers (M)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.subheader("By day of week")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = (
            games.groupby("day_of_week", as_index=False)
            .agg(avg_viewership=("total_viewership", "mean"), games=("game_id", "count"))
        )
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=day_order, ordered=True)
        dow = dow.sort_values("day_of_week")
        fig = px.line(dow, x="day_of_week", y="avg_viewership", markers=True)
        fig.update_layout(xaxis_title=None, yaxis_title="Avg viewers (M)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View filtered game-level data"):
        table_cols = [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "time_slot",
            "marquee_game",
            "total_spend",
            "base_viewership",
            "total_contribution",
            "total_viewership",
        ]
        table = games[table_cols].sort_values("date").copy()
        st.dataframe(
            table.style.format(
                {
                    "date": lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else x,
                    "total_spend": "${:,.0f}",
                    "base_viewership": "{:,.2f}M",
                    "total_contribution": "{:,.2f}M",
                    "total_viewership": "{:,.2f}M",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


else:
    st.header("Channel Deep Dive")

    ch = channel_summary(games).sort_values("Viewers per $1K", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Most efficient", ch.iloc[0]["Channel"])
    c2.metric("Viewers per $1K", f"{ch.iloc[0]['Viewers per $1K']:,.0f}")
    c3.metric("Highest contribution", ch.sort_values("Contribution", ascending=False).iloc[0]["Channel"])
    c4.metric("Highest spend", ch.sort_values("Spend", ascending=False).iloc[0]["Channel"])

    st.subheader("Channel scorecard")
    scorecard = ch[["Channel", "Spend", "Contribution", "Viewers per $1K", "Spend Share", "Contribution Share"]].copy()
    st.dataframe(
        scorecard.style.format(
            {
                "Spend": "${:,.0f}",
                "Contribution": "{:,.1f}M",
                "Viewers per $1K": "{:,.0f}",
                "Spend Share": "{:.0%}",
                "Contribution Share": "{:.0%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Efficiency ranking")
        fig = px.bar(
            ch,
            x="Channel",
            y="Viewers per $1K",
            text="Viewers per $1K",
            hover_data={"Spend": ":,.0f", "Contribution": ":.2f"},
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Viewers per $1K spend",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Spend share vs. contribution share")
        share = ch.melt(
            id_vars="Channel",
            value_vars=["Spend Share", "Contribution Share"],
            var_name="Metric",
            value_name="Share",
        )
        fig = px.bar(share, x="Channel", y="Share", color="Metric", barmode="group", text="Share")
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig.update_layout(xaxis_title=None, yaxis_title="Share", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Response curves: spend saturation and current game positions")
    st.caption(
        "Each curve shows expected contribution as spend rises. Dots show actual games in the selected filter. "
        "Points far to the right on a flattening curve indicate possible saturation."
    )

    selected_channel = st.selectbox(
        "Select channel for detailed response curve",
        options=CHANNELS,
        format_func=lambda x: CHANNEL_LABELS[x],
    )

    curve = curves_raw[curves_raw["channel"] == selected_channel].sort_values("spend")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["spend"],
            y=curve["contribution"],
            mode="lines+markers",
            name="Response curve",
            hovertemplate="Spend: $%{x:,.0f}<br>Expected contribution: %{y:.2f}M<extra></extra>",
        )
    )
    fig = add_curve_markers(fig, games, selected_channel)
    fig.update_layout(
        xaxis_title="Spend ($)",
        yaxis_title="Contribution (M viewers)",
        legend_title=None,
        hovermode="closest",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All channel response curves")
    cols = st.columns(2)
    for i, channel in enumerate(CHANNELS):
        with cols[i % 2]:
            curve = curves_raw[curves_raw["channel"] == channel].sort_values("spend")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["spend"],
                    y=curve["contribution"],
                    mode="lines",
                    name="Response curve",
                )
            )
            fig = add_curve_markers(fig, games, channel)
            fig.update_layout(
                title=CHANNEL_LABELS[channel],
                xaxis_title="Spend ($)",
                yaxis_title="Contribution (M viewers)",
                legend_title=None,
                margin=dict(l=10, r=10, t=40, b=10),
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Potential over- and under-spend flags")
    st.caption(
        "This simple diagnostic compares each game’s channel spend to the response curve range. "
        "High-spend games on weaker efficiency should be reviewed for reallocation."
    )

    flag_rows = []
    for channel in CHANNELS:
        spend_col = f"{channel}_spend"
        contrib_col = f"{channel}_contribution"
        roi_col = f"{channel}_roi"
        spend_q75 = games[spend_col].quantile(0.75)
        roi_q25 = games[roi_col].quantile(0.25)
        spend_q25 = games[spend_col].quantile(0.25)
        roi_q75 = games[roi_col].quantile(0.75)

        for _, row in games.iterrows():
            if row[spend_col] >= spend_q75 and row[roi_col] <= roi_q25:
                flag = "Review: high spend / low efficiency"
            elif row[spend_col] <= spend_q25 and row[roi_col] >= roi_q75:
                flag = "Opportunity: low spend / high efficiency"
            else:
                continue

            flag_rows.append(
                {
                    "Game": row["game_id"],
                    "Date": row["date"].date(),
                    "Matchup": f"{row['away_team']} @ {row['home_team']}",
                    "Channel": CHANNEL_LABELS[channel],
                    "Spend": row[spend_col],
                    "Contribution": row[contrib_col],
                    "Viewers per $1K": row[roi_col] * 1000 * 1_000_000 if pd.notna(row[roi_col]) else np.nan,
                    "Flag": flag,
                }
            )

    flags = pd.DataFrame(flag_rows)
    if flags.empty:
        st.info("No over- or under-spend flags found for the current filters.")
    else:
        st.dataframe(
            flags.sort_values(["Flag", "Viewers per $1K"], ascending=[True, False]).style.format(
                {
                    "Spend": "${:,.0f}",
                    "Contribution": "{:,.2f}M",
                    "Viewers per $1K": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("How to read this page")
    st.markdown(
        """
        1. A channel is attractive when its contribution share is higher than its spend share.
        2. A flattening response curve means extra dollars are producing fewer incremental viewers.
        3. High-spend / low-efficiency games are candidates for budget reduction or reallocation.
        4 Low-spend / high-efficiency games may indicate room to scale, especially if the response curve has not flattened.
        """
    )

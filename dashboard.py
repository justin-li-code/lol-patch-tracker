"""Streamlit dashboard: python -m lolpatch dashboard

One page at a time (sidebar), each leading with a single takeaway and one main
view; methods, caveats and full tables sit behind expanders and sub-tabs.
"""
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lolpatch.analyze import build_transitions, load_changes, verdict_effects
from lolpatch.config import LEAGUES, PROCESSED, patch_key
from lolpatch.events import PRIOR_GAMES
from lolpatch.meta import ROLES, SUB_MIN_MOVE_A, SUB_MIN_MOVE_B, displacement, group_presence, rise_window
from lolpatch.ui import champ_icon, image_column, row_layout, team_logo, with_icons

st.set_page_config(page_title="Patch Impact · LoL pro 2026", layout="wide", initial_sidebar_state="expanded")

# Reference palette (dataviz skill): blue/red diverging poles, gray neutral, categorical order.
BUFF, NERF, NEUTRAL = "#2a78d6", "#e34948", "#898781"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
VERDICT_COLOR = {"buff": BUFF, "nerf": NERF, "adjust": NEUTRAL, "none": "#c3c2b7"}
PCT = dict(tickformat=".0%")
EVENT_SHORT = {"first_stand": "First Stand", "msi": "MSI", "ewc": "EWC", "demacia": "Demacia Cup", "worlds": "Worlds"}
TOP_N = 10  # rows shown before "show all"

# Calmer typography: fewer heavy headings, more air between blocks.
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  h1 { font-size: 2.0rem !important; margin-bottom: .2rem !important; }
  h2 { font-size: 1.45rem !important; margin-top: .4rem !important; }
  h3 { font-size: 1.15rem !important; }
  .lead { font-size: 1.08rem; color: var(--text-color); opacity: .85; margin: .1rem 0 1.1rem 0; }
  .takeaway { font-size: 1.02rem; line-height: 1.5; }
  .muted { opacity: .7; font-size: .9rem; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# --- data -----------------------------------------------------------------------------

@st.cache_data
def load():
    stats = pd.read_csv(PROCESSED / "champion_patch_stats.csv", dtype={"patch": str})
    changes = load_changes()
    lines = pd.read_csv(PROCESSED / "patch_changes.csv", dtype={"patch": str})
    out = dict(stats=stats, changes=changes, lines=lines)
    for name in ["predictions", "backtest_predictions", "backtest_metrics"]:
        path = PROCESSED / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path, dtype={"patch": str, "from_patch": str, "to_patch": str,
                                                 "base_patch": str, "target_patch": str})
    meta = PROCESSED / "model_meta.json"
    out["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    return out


@st.cache_data
def transitions_for(league: str) -> pd.DataFrame:
    d = load()
    return build_transitions(d["stats"][d["stats"].league == league], d["changes"])


@st.cache_data
def load_events():
    path = PROCESSED / "event_metrics.json"
    if not path.exists():
        return None
    out = dict(metrics=json.loads(path.read_text()),
               games=pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str}))
    for key in out["metrics"]:
        for version in ["frozen", "current"]:
            out[f"{key}_{version}"] = pd.read_csv(PROCESSED / f"event_{key}_{version}_comparison.csv")
    for name, file in [("worlds_live", "event_worlds_live.csv"), ("adjustment", "event_adjustment_test.csv")]:
        if (PROCESSED / file).exists():
            out[name] = pd.read_csv(PROCESSED / file)
    regions = PROCESSED / "team_regions.json"
    out["team_regions"] = json.loads(regions.read_text()) if regions.exists() else {}
    return out


@st.cache_data
def group_presence_cached(members: tuple) -> pd.Series:
    from lolpatch.analyze import load_games
    return group_presence(load_games(), list(members))


@st.cache_data
def load_draft():
    path = PROCESSED / "draft_meta.json"
    if not path.exists():
        return None
    return dict(meta=json.loads(path.read_text()), lanes=pd.read_csv(PROCESSED / "lane_answers.csv"),
                first=pd.read_csv(PROCESSED / "first_pick_answers.csv"),
                breakdown=pd.read_csv(PROCESSED / "answer_breakdown.csv"),
                events=pd.read_csv(PROCESSED / "draft_lane_events.csv", dtype={"patch": str}))


@st.cache_data
def load_matches():
    path = PROCESSED / "player_ratings.csv"
    if not path.exists():
        return None
    read = lambda name: pd.read_csv(PROCESSED / name)
    out = dict(games=read("match_games.csv"), teams=read("match_teams.csv"), ratings=read("player_ratings.csv"),
               timeline=read("match_timeline.csv"), gold=read("match_gold.csv"), expected=read("match_expected.csv"))
    ev_path = PROCESSED / "expected_eval.json"
    out["eval"] = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    return out


@st.cache_data
def load_ripple():
    path = PROCESSED / "substitutions.csv"
    if not path.exists():
        return None
    read = lambda name: pd.read_csv(PROCESSED / name, dtype={"patch": str, "from_patch": str, "to_patch": str})
    return dict(subs=read("substitutions.csv"), pairs=read("substitute_pairs.csv"), meta=read("champion_meta.csv"))


@st.cache_data
def load_regions():
    path = PROCESSED / "priority_effects.json"
    if not path.exists():
        return None
    return dict(effects=json.loads(path.read_text()),
                regions=pd.read_csv(PROCESSED / "region_champions.csv"),
                teams=pd.read_csv(PROCESSED / "team_champions.csv"),
                events=pd.read_csv(PROCESSED / "event_region_champions.csv"))


# --- small helpers --------------------------------------------------------------------

def style(fig, height=380, **layout):
    # Legend sits in the top margin, above the plot, so it never covers data.
    layout = dict(margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                  hovermode="x unified") | layout
    fig.update_layout(height=height, **layout)
    fig.update_xaxes(showgrid=False)
    return fig


def lead(text: str):
    """One-sentence takeaway under a page title."""
    st.markdown(f"<p class='lead'>{text}</p>", unsafe_allow_html=True)


def table(df: pd.DataFrame, fmt: dict | None = None, images: list[str] | None = None, n: int = TOP_N,
          more_label: str = "Show all"):
    """First n rows inline; the rest behind an expander."""
    cfg = {}
    for label in images or []:
        cfg |= image_column(label)
    styled = lambda x: x.style.format(fmt or {}, na_rep="—")
    st.dataframe(styled(df.head(n)), column_config=cfg, hide_index=True, use_container_width=True)
    if len(df) > n:
        with st.expander(f"{more_label} ({len(df)})"):
            st.dataframe(styled(df), column_config=cfg, hide_index=True, use_container_width=True,
                         height=min(600, 36 * len(df) + 40))


def go_to(page: str):
    st.session_state["page"] = page


def champion_header(name: str, sub: str = "", size: int = 64):
    c1, c2 = st.columns([1, 12], vertical_alignment="center")
    if champ_icon(name):
        c1.image(champ_icon(name), width=size)
    c2.markdown(f"### {name}" + (f"\n<span class='muted'>{sub}</span>" if sub else ""), unsafe_allow_html=True)


def pair_table(df: pd.DataFrame, opener_label: str = "Their pick", answer_label: str = "Answer",
               detail: bool = False) -> pd.DataFrame:
    """Opener -> answer table with both portraits. `detail` adds the statistics columns."""
    t = pd.DataFrame({opener_label: df.opener.values, answer_label: df.answer.values})
    if "role" in df:
        t.insert(0, "Role", df.role.str.title().values)
    t["How often"] = [f"{n} of {m} times" for n, m in zip(df.n, df.opener_n)]
    t["vs usual"] = [f"×{x:.1f}" for x in df.lift]
    if "where_from" in df:
        t["Led by"] = df.where_from.replace("Broad meta", "everyone").values
    if detail:
        t["Answer rate"] = df.answer_rate.values
        t["Usual rate"] = df.base_rate.values
        if "answer_win_rate" in df:
            t["Answer won"] = df.answer_win_rate.values
    t = with_icons(t, opener_label, label=" ")
    return with_icons(t, answer_label, label="  ")


PAIR_FMT = {"Answer rate": "{:.0%}", "Usual rate": "{:.0%}", "Answer won": "{:.0%}"}


def forecast_chart(cmp: pd.DataFrame, base_patch: str, games: int, n: int = 20):
    """Domestic -> forecast -> actual presence for the champions that mattered most."""
    relevance = cmp[["domestic_presence", "pred_presence", "actual_presence"]].max(axis=1)
    show = cmp.loc[relevance.nlargest(n).index].sort_values("actual_presence")
    fig = go.Figure()
    for _, r in show.iterrows():
        fig.add_trace(go.Scatter(x=[r.domestic_presence, r.pred_presence, r.actual_presence], y=[r.champion] * 3,
                                 mode="lines", line=dict(color="#c3c2b7", width=1.5), showlegend=False, hoverinfo="skip"))
    hover = "<b>%{y}</b><br>%{x:.0%}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=show.domestic_presence, y=show.champion, mode="markers", name=f"Before ({base_patch})",
                             marker=dict(color=NEUTRAL, size=8), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=show.pred_presence, y=show.champion, mode="markers", name="Forecast",
                             marker=dict(color="rgba(0,0,0,0)", size=11, line=dict(width=2, color=SERIES[1])),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=show.actual_presence, y=show.champion, mode="markers", showlegend=False,
                             marker=dict(color=[VERDICT_COLOR[v] for v in show.verdict], size=11,
                                         line=dict(width=1.5, color="white")),
                             customdata=show[["verdict", "change_patches"]],
                             hovertemplate="<b>%{y}</b> — %{customdata[0]} %{customdata[1]}<br>actual %{x:.0%}<extra></extra>"))
    # Legend entry for "actual" in neutral ink: the real dots are colored by change type.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Actual ({games} games)",
                             marker=dict(color="#52514e", size=11)))
    # Start the axis slightly below 0 so 0% markers aren't cut off at the edge.
    fig = style(fig, xaxis=dict(title="Presence", range=[-0.03, 1.03], **PCT), hovermode="closest")
    st.plotly_chart(row_layout(fig, show.champion, legend=True), use_container_width=True)
    st.caption("Gray = before the event · orange ring = forecast · filled = what happened "
               "(blue buffed, red nerfed, gray unchanged/adjusted).")


def scorecard(sc: dict):
    a, b, c = st.columns(3)
    a.metric("Forecast error", f"{sc['all']['mae_model'] * 100:.1f} pp",
             help="Average gap between forecast and actual presence, in percentage points")
    b.metric("“Nothing changes” error", f"{sc['all']['mae_baseline'] * 100:.1f} pp",
             help=f"The same, if you assumed every champion stayed at its domestic {sc['base_patch']} presence")
    c.metric("Top-10 picked correctly", f"{sc['top10_model']}/10")


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


RATING_HELP = ("Out of 5. Compares the player's stats with every player in the same role at this year's "
               "international events (weighted for what that role is for), plus the result relative to "
               "expectation: an upset win counts for more than an expected one. 3.0 is a typical international "
               "performance; 4.5+ is the top ~10%.")


def game_label(g) -> str:
    return f"{g.date} · {g.stage.title()} · {g.blue} vs {g.red}"


def worlds_games_tab(mx: dict):
    games = mx["games"][mx["games"].event == "worlds"].copy()
    rt = mx["ratings"]
    exp = mx["expected"]
    games = games.sort_values(["date", "game_uid"], ascending=False)
    mvp = rt[rt.mvp].drop_duplicates("game_uid").set_index("game_uid")
    win_p = exp.set_index(["game_uid", "team"]).p_win
    rows = []
    for g in games.itertuples():
        pw = win_p.get((g.game_uid, g.winner), np.nan)
        rows.append({
            "Date": g.date, "Stage": g.stage.title(), "Blue": team_logo(g.blue), "Blue team": g.blue,
            "Red": team_logo(g.red), "Red team": g.red, "Winner": g.winner,
            "Winner's pre-game odds": pw, "": "upset" if pd.notna(pw) and pw < 0.4 else "",
            "MVP": f"{mvp.player.get(g.game_uid, '—')} ({mvp.champion.get(g.game_uid, '')}) "
                   f"{mvp.rating.get(g.game_uid, np.nan):.1f}" if g.game_uid in mvp.index else "—",
            "Length": g.duration_str,
        })
    table_df = pd.DataFrame(rows)
    st.markdown("Click a game to open it.")
    picked = st.dataframe(
        table_df.style.format({"Winner's pre-game odds": "{:.0%}"}, na_rep="—"),
        column_config={"Blue": st.column_config.ImageColumn(" ", width="small"),
                       "Red": st.column_config.ImageColumn("  ", width="small")},
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
        key="worlds_game_table", height=min(420, 36 * len(table_df) + 40))
    rows_sel = (picked or {}).get("selection", {}).get("rows", [])
    if not rows_sel:
        return
    g = games.iloc[rows_sel[0]]
    st.divider()
    game_detail(g, mx)


def game_detail(g, mx: dict):
    teams = mx["teams"][mx["teams"].game_uid == g.game_uid].set_index("side")
    rt = mx["ratings"][mx["ratings"].game_uid == g.game_uid]
    exp = mx["expected"][mx["expected"].game_uid == g.game_uid].set_index("team")
    blue, red = teams.loc["blue"], teams.loc["red"]

    # Header: logos, names, result, expectation.
    c1, c2, c3 = st.columns([5, 2, 5], vertical_alignment="center")
    for col, tm, align in ((c1, blue, "left"), (c3, red, "right")):
        with col:
            lc, nc = st.columns([1, 4], vertical_alignment="center")
            if team_logo(tm.team):
                lc.image(team_logo(tm.team), width=56)
            p = exp.p_win.get(tm.team, np.nan)
            nc.markdown(f"### {tm.team}\n{'**Won**' if tm.result else 'Lost'} · "
                        f"{'expected ' + format(p, '.0%') if pd.notna(p) else ''}")
    c2.markdown(f"<div style='text-align:center'><span class='muted'>{g.stage.title()} · {g.date}</span><br>"
                f"<b style='font-size:1.4rem'>{g.duration_str}</b><br><span class='muted'>patch "
                f"{str(g.version).replace('16.', '26.')}</span></div>", unsafe_allow_html=True)
    winner_p = exp.p_win.get(g.winner, np.nan)
    if pd.notna(winner_p):
        verdict = ("an upset" if winner_p < 0.4 else "as expected" if winner_p > 0.6 else "a close call beforehand")
        edge = exp.draft_edge.get(blue.team, np.nan)
        draft_note = ""
        if pd.notna(edge) and abs(edge) >= 0.005:
            ahead = blue.team if edge > 0 else red.team
            draft_note = (f" On the draft, {possessive(ahead)} picks had the better recent win rates "
                          f"(+{abs(edge) * 100:.1f} pts on average), though drafts haven't proven predictive.")
        st.caption(f"Before the game the model gave {g.winner} a {winner_p:.0%} chance, so this was {verdict}."
                   + draft_note)

    t_over, t_players, t_draft, t_gold = st.tabs(["Overview", "Players", "Draft", "Gold & objectives"])
    with t_over:
        def fmt(v, kind=""):
            if pd.isna(v):
                return "—"
            return f"{v / 1000:.1f}k" if kind == "gold" else ("✓" if kind == "flag" and v else ("" if kind == "flag" else f"{int(v)}"))
        rows = [("Kills", "kills", ""), ("Gold", "gold", "gold"), ("Towers", "towers", ""), ("Dragons", "dragons", ""),
                ("Barons", "barons", ""), ("Rift Heralds", "heralds", ""), ("Voidgrubs", "voidgrubs", ""),
                ("Turret plates", "plates", ""), ("First blood", "first_blood", "flag"), ("First tower", "first_tower", "flag")]
        comp = pd.DataFrame({"": [r[0] for r in rows],
                             blue.team: [fmt(blue.get(r[1]), r[2]) for r in rows],
                             red.team: [fmt(red.get(r[1]), r[2]) for r in rows]})
        l, r = st.columns([3, 2], gap="large")
        l.dataframe(comp, hide_index=True, use_container_width=True, height=36 * len(comp) + 40)
        with r:
            for tm in (blue, red):
                if isinstance(tm.dragon_types, str) and tm.dragon_types:
                    st.markdown(f"**{tm.team} dragons:** {tm.dragon_types}")
            best = rt.sort_values("rating", ascending=False).iloc[0]
            with st.container(border=True):
                st.markdown("**Best performance**")
                bc1, bc2 = st.columns([1, 3], vertical_alignment="center")
                if champ_icon(best.champion):
                    bc1.image(champ_icon(best.champion), width=52)
                bc2.markdown(f"**{best.player}** ({best.team})  \n{best.champion} · "
                             f"{int(best.Kills)}/{int(best.Deaths)}/{int(best.Assists)} · rating **{best.rating:.1f}**")

    with t_players:
        for tm in (blue, red):
            st.markdown(f"**{tm.team}**" + (" · won" if tm.result else ""))
            p = rt[rt.team == tm.team].copy()
            p["order"] = p.role.map({r: i for i, r in enumerate(ROLES)})
            p = p.sort_values("order")
            t = pd.DataFrame({
                "Role": p.role.str.title(), "Player": p.player, "Champion": p.champion,
                "K / D / A": [f"{int(k)} / {int(d_)} / {int(a)}" for k, d_, a in zip(p.Kills, p.Deaths, p.Assists)],
                "CS/min": p.CSM, "Gold share": p["GOLD%"] / 100, "Damage share": p["DMG%"] / 100,
                "Vision/min": p.VSPM, "Gold diff @15": p["GD@15"], "Rating": p.rating,
            })
            t = with_icons(t, "Champion")
            st.dataframe(t.style.format({"CS/min": "{:.1f}", "Gold share": "{:.0%}", "Damage share": "{:.0%}",
                                         "Vision/min": "{:.1f}", "Gold diff @15": "{:+,.0f}", "Rating": "{:.1f}"})
                         .background_gradient(cmap="Blues", subset=["Rating"], vmin=1, vmax=5),
                         column_config=image_column() | {"Rating": st.column_config.NumberColumn("Rating", help=RATING_HELP)},
                         hide_index=True, use_container_width=True)

    with t_draft:
        ev_games = load_events()["games"]
        dg = ev_games[ev_games.game_uid == g.game_uid].set_index("team")
        for tm in (blue, red):
            if tm.team not in dg.index:
                continue
            row = dg.loc[tm.team]
            fp = row.get("first_pick")
            st.markdown(f"**{tm.team}**" + (" · first pick" if fp == 1 else ""))
            pc, bc = st.columns([5, 5], gap="large")
            picks = row.picks.split("|")
            bans = row.bans.split("|") if isinstance(row.bans, str) else []
            pc.caption("Picks, in draft order")
            pc.image([champ_icon(c) for c in picks if champ_icon(c)], width=48, caption=picks)
            bc.caption("Bans")
            bc.image([champ_icon(c) for c in bans if champ_icon(c)], width=40, caption=bans)

    with t_gold:
        gl = mx["gold"][mx["gold"].game_uid == g.game_uid]
        tl = mx["timeline"][mx["timeline"].game_uid == g.game_uid]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gl.minute, y=gl.blue_lead.clip(lower=0), fill="tozeroy", mode="none",
                                 fillcolor="rgba(42,120,214,0.25)", name=f"{blue.team} ahead", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=gl.minute, y=gl.blue_lead.clip(upper=0), fill="tozeroy", mode="none",
                                 fillcolor="rgba(227,73,72,0.25)", name=f"{red.team} ahead", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=gl.minute, y=gl.blue_lead, mode="lines", line=dict(color="#52514e", width=2),
                                 showlegend=False, hovertemplate="%{x} min: %{y:+,.0f} gold<extra></extra>"))
        for side, color in (("blue", BUFF), ("red", NERF)):
            ev_side = tl[tl.side == side]
            if len(ev_side):
                lead_at = np.interp(ev_side.minute, gl.minute, gl.blue_lead) if len(gl) else np.zeros(len(ev_side))
                fig.add_trace(go.Scatter(x=ev_side.minute, y=lead_at, mode="markers", showlegend=False,
                                         marker=dict(color=color, size=9, line=dict(width=1.5, color="white")),
                                         customdata=ev_side[["time", "objective", "team"]],
                                         hovertemplate="%{customdata[0]} · %{customdata[2]}: %{customdata[1]}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        st.plotly_chart(style(fig, 320, xaxis=dict(title="Minute"), yaxis=dict(title="Gold lead", tickformat="+,.0f"),
                              hovermode="closest"), use_container_width=True)
        st.caption("Above the line = blue side ahead in gold. Dots are objectives (hover for which).")


def worlds_players_tab(mx: dict):
    rt = mx["ratings"][mx["ratings"].event == "worlds"]
    role_p = st.radio("Role", ["all"] + ROLES, horizontal=True, key="worlds_player_role",
                      format_func=lambda r: r.title() if r != "all" else "All roles")
    view = rt if role_p == "all" else rt[rt.role == role_p]
    agg = view.groupby(["player", "team", "role"]).agg(
        games=("game_uid", "nunique"), rating=("rating", "mean"), best=("rating", "max"),
        wins=("result", "sum"), k=("Kills", "sum"), d=("Deaths", "sum"), a=("Assists", "sum"),
        champs=("champion", lambda s: ", ".join(s.value_counts().index[:3])),
    ).reset_index()
    agg = agg[agg.games >= 2].sort_values("rating", ascending=False)
    if agg.empty:
        st.info("No player has two Worlds games in this role yet.")
        return
    top = agg.iloc[0]
    st.markdown(f"<div class='takeaway'>Highest-rated so far: <b>{top.player}</b> ({top.team}, {top.role}), "
                f"averaging <b>{top.rating:.1f}</b> over {top.games} games.</div>", unsafe_allow_html=True)
    t = pd.DataFrame({
        "Logo": agg.team.map(team_logo), "Player": agg.player, "Team": agg.team, "Role": agg.role.str.title(),
        "Games": agg.games, "Record": [f"{int(w)}–{int(n - w)}" for w, n in zip(agg.wins, agg.games)],
        "Avg rating": agg.rating, "Best": agg.best,
        "KDA": [(k + a) / max(d, 1) for k, d, a in zip(agg.k, agg.d, agg.a)], "Champions": agg.champs,
    })
    table(t, {"Avg rating": "{:.1f}", "Best": "{:.1f}", "KDA": "{:.1f}"}, n=15, more_label="Show all players")
    st.caption("Players with 2+ Worlds games. " + RATING_HELP)

    with st.expander("Look at one player"):
        who = st.selectbox("Player", agg.player + " (" + agg.team + ")", key="worlds_player_pick")
        name = who.rsplit(" (", 1)[0]
        pg = rt[rt.player == name].sort_values("date")
        gt = pd.DataFrame({"Date": pg.date, "Opponent": pg.opponent, "Champion": pg.champion,
                           "Result": pg.result.map({1: "Won", 0: "Lost"}),
                           "K / D / A": [f"{int(k)} / {int(d_)} / {int(a)}" for k, d_, a in zip(pg.Kills, pg.Deaths, pg.Assists)],
                           "Rating": pg.rating})
        st.dataframe(with_icons(gt, "Champion").style.format({"Rating": "{:.1f}"}), column_config=image_column(),
                     hide_index=True, use_container_width=True)
        from lolpatch.ratings import ROLE_WEIGHTS, STAT_LABELS
        role = pg.role.iloc[0]
        prof = pd.Series({STAT_LABELS[s]: pg[f"z_{s}"].mean() for s in ROLE_WEIGHTS[role]}).sort_values()
        fig = go.Figure(go.Bar(x=prof.values, y=prof.index, orientation="h",
                               marker=dict(color=[BUFF if v >= 0 else NERF for v in prof.values], cornerradius=3),
                               hovertemplate="%{y}: %{x:+.1f} SD vs other %{customdata}s<extra></extra>",
                               customdata=[role] * len(prof)))
        fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
        st.markdown(f"**What drives {name}'s rating** (vs other {role}s at international events)")
        st.plotly_chart(row_layout(style(fig, xaxis=dict(title="Above / below the role average", range=[-2.5, 2.5]),
                                         hovermode="closest"), prof.index, right=24),
                        use_container_width=True)


d = load()
stats, changes, lines = d["stats"], d["changes"], d["lines"]
ev, rg, rp, dr, mx = load_events(), load_regions(), load_ripple(), load_draft(), load_matches()
all_stats = stats[stats.league == "ALL"]

# --- navigation -------------------------------------------------------------------------

PAGES = ["Overview", "Worlds 2026", "Champions & patches", "Meta shifts", "Draft answers", "Teams & regions",
         "Model accuracy", "Changelog"]
CHANGELOG = PROCESSED.parent.parent / "CHANGELOG.md"
changelog_text = CHANGELOG.read_text() if CHANGELOG.exists() else ""
version_tag = next((line[3:].split("·")[0].strip() for line in changelog_text.splitlines()
                    if line.startswith("## v")), "")
with st.sidebar:
    st.markdown("## Patch Impact")
    st.caption("How balance changes move pro picks and bans · 2026 season")
    page = st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption(f"{' · '.join(LEAGUES)} and international events. **Presence** = how often a champion is picked "
               "or banned, as a share of games.")
    if version_tag:
        st.caption(f"Version {version_tag}")


# --- Overview --------------------------------------------------------------------------

if page == "Overview":
    st.title("Patch Impact")
    lead("What Riot's balance changes actually did to pro drafts in 2026, and what's happening at Worlds right now.")

    cards = []
    if ev and "worlds" in ev["metrics"]:
        wm = ev["metrics"]["worlds"]
        top = ev["worlds_current"].nlargest(3, "actual_presence")
        names = ", ".join(f"**{r.champion}** ({r.actual_presence:.0%})" for r in top.itertuples())
        text = (f"{wm['current']['games']} games played on patch {wm['patch']}. "
                f"The most contested champions so far: {names}.")
        if mx is not None:
            wr_ = mx["ratings"][mx["ratings"].event == "worlds"]
            lb = wr_.groupby(["player", "team"]).agg(n=("game_uid", "nunique"), r=("rating", "mean")).reset_index()
            lb = lb[lb.n >= 2].sort_values("r", ascending=False)
            if len(lb):
                text += f" Top-rated player: **{lb.iloc[0].player}** ({lb.iloc[0].team}), {lb.iloc[0].r:.1f} / 5."
        cards.append(("Worlds 2026", text, "Worlds 2026"))
    tr = transitions_for("ALL")
    eff = verdict_effects(tr)
    if {"buff", "nerf"} <= set(eff.index):
        hi_nerf = tr[tr.changed & (tr.verdict == "nerf") & (tr.prev_presence > 0.6)]
        cards.append(("Do buffs and nerfs work?",
                      f"On average a buff adds **{eff.loc['buff', 'mean_delta_presence'] * 100:+.0f} points** of "
                      f"presence and a nerf takes **{eff.loc['nerf', 'mean_delta_presence'] * 100:+.0f}**. "
                      f"Nerfs hit hardest on top picks: **{hi_nerf.delta_presence.mean() * 100:+.0f} points** "
                      "for champions above 60%.", "Champions & patches"))
    if dr is not None:
        solid = dr["lanes"][dr["lanes"].solid].sort_values("n", ascending=False)
        ex = solid.iloc[0]
        cards.append(("Draft answers",
                      f"Teams have clear go-to responses: **{len(solid)} lane answers** stand out, like "
                      f"**{ex.answer} into {ex.opener}** ({ex.role}), taken {ex.lift:.1f}× as often as usual.",
                      "Draft answers"))
    if ev:
        wins = sum(m["frozen"]["all"]["mae_model"] < m["frozen"]["all"]["mae_baseline"] for m in ev["metrics"].values())
        cards.append(("Can patch notes predict the meta?",
                      f"A little. The forecast beat “nothing changes” at **{wins} of {len(ev['metrics'])}** "
                      "international events, mostly by small margins.", "Model accuracy"))

    for row in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for col, (title, text, target) in zip(cols, cards[row:row + 2]):
            with col.container(border=True):
                st.markdown(f"#### {title}")
                # Markdown isn't processed inside raw HTML, so **bold** would show its asterisks.
                html_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
                st.markdown(f"<div class='takeaway'>{html_text}</div>", unsafe_allow_html=True)
                st.button(f"Open {target} →", key=f"go_{target}", on_click=go_to, args=(target,))

    st.markdown("")
    st.caption("Also: **Meta shifts** shows who lost ground as a group of champions rose; **Teams & regions** shows "
               "what each team and region leans on, and whether that wins games.")


# --- Worlds 2026 -----------------------------------------------------------------------

elif page == "Worlds 2026":
    if not ev or "worlds" not in ev["metrics"]:
        st.info("Run `python -m lolpatch events` to load Worlds.")
        st.stop()
    wm = ev["metrics"]["worlds"]
    sc = wm["current"]
    stages = " · ".join(f"{k} ({v})" for k, v in wm["stages"].items())
    st.title("Worlds 2026")
    lead(f"{sc['games']} games so far, all on patch {wm['patch']}: {stages}.")

    if "worlds_live" in ev:
        lv = ev["worlds_live"].head(15).iloc[::-1]
        st.markdown("### Most contested champions")
        fig = go.Figure(go.Bar(
            x=lv.live_presence, y=lv.champion, orientation="h",
            marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in lv.verdict], cornerradius=4),
            error_x=dict(type="data", symmetric=False, array=lv.live_high - lv.live_presence,
                         arrayminus=lv.live_presence - lv.live_low, color=NEUTRAL, thickness=1.2, width=3),
            customdata=lv[["live_low", "live_high", "actual_presence", "picks", "bans", "games", "pred_presence"]],
            hovertemplate="<b>%{y}</b> %{x:.0%} (likely %{customdata[0]:.0%}–%{customdata[1]:.0%})"
                          "<br>so far %{customdata[2]:.0%}: %{customdata[3]} picks + %{customdata[4]} bans in "
                          "%{customdata[5]} games<br>pre-Worlds forecast %{customdata[6]:.0%}<extra></extra>"))
        fig = style(fig, xaxis=dict(title="Expected presence for the rest of Worlds", range=[0, 1], **PCT),
                    hovermode="closest", showlegend=False, bargap=0.35)
        st.plotly_chart(row_layout(fig, lv.champion), use_container_width=True)
        st.caption("Blue = buffed since the leagues ended, red = nerfed. Whiskers show the likely range.")
        with st.expander("How this estimate works"):
            st.markdown(
                f"It blends the pre-Worlds forecast, counted as {PRIOR_GAMES} games' worth of evidence, with the "
                "games played so far, so it starts from the patch notes and shifts toward what teams actually do "
                "as games accumulate. Whiskers are an 80% range.")

    t_games, t_players, t_teams, t_patch, t_fc, t_draft, t_region = st.tabs(
        ["Games", "Players", "Teams", f"Patch {wm['patch']} changes", "Forecast check", "Draft answers", "By region"])

    with t_games:
        if mx is None:
            st.info("Run `python -m lolpatch matches` to load game details.")
        else:
            worlds_games_tab(mx)
    with t_players:
        if mx is not None:
            worlds_players_tab(mx)

    wg = ev["games"][ev["games"].league == "worlds"]
    with t_teams:
        series = wg.groupby(["series_id", "team"]).result.sum().reset_index()
        series["won"] = series.groupby("series_id").result.transform(lambda s: (s == s.max()) & (s.max() > s.min()))
        picks = wg.assign(champion=wg.picks.str.split("|")).explode("champion")
        top3 = picks.groupby("team").champion.agg(lambda s: ", ".join(s.value_counts().head(3).index))
        rec = series.groupby("team").won.agg(["sum", "size"])
        g_rec = wg.groupby("team").result.agg(["sum", "size"])
        tt = pd.DataFrame({"Team": sorted(wg.team.unique())})
        tt["Logo"] = tt.Team.map(team_logo)
        tt["Region"] = tt.Team.map(ev["team_regions"]).fillna("—")
        tt["Series"] = [f"{int(rec.loc[t, 'sum'])}–{int(rec.loc[t, 'size'] - rec.loc[t, 'sum'])}" for t in tt.Team]
        tt["Games"] = [f"{int(g_rec.loc[t, 'sum'])}–{int(g_rec.loc[t, 'size'] - g_rec.loc[t, 'sum'])}" for t in tt.Team]
        tt["_order"] = [rec.loc[t, "sum"] * 2 - rec.loc[t, "size"] for t in tt.Team]
        tt["Go-to picks"] = tt.Team.map(top3)
        tt = tt.sort_values(["_order", "Region"], ascending=[False, True])
        st.dataframe(tt[["Logo", "Team", "Region", "Series", "Games", "Go-to picks"]],
                     column_config={"Logo": st.column_config.ImageColumn(" ", width="small")},
                     hide_index=True, use_container_width=True, height=36 * len(tt) + 40)

    bridge = [p for p in sorted(changes.patch.unique(), key=patch_key)
              if patch_key(sc["base_patch"]) < patch_key(p) <= patch_key(wm["patch"])]
    with t_patch:
        st.markdown(f"The leagues finished on **{sc['base_patch']}**, so Worlds is the first pro play on "
                    f"{', '.join(bridge)}. Here's what changed and how those champions are doing.")
        wcmp = ev["worlds_current"].set_index("champ_key")
        bp = st.radio("Patch", bridge[::-1], horizontal=True, key="worlds_bridge_patch",
                      format_func=lambda p: f"{p} (Worlds patch)" if p == wm["patch"] else p)
        pc = changes[changes.patch == bp].copy()
        pc["before"] = pc.champ_key.map(wcmp.domestic_presence).fillna(0)
        pc["worlds"] = pc.champ_key.map(wcmp.actual_presence).fillna(0)
        picks_ = pc.champ_key.map(wcmp.picks).fillna(0).astype(int)
        wins_ = pc.champ_key.map(wcmp.wins).fillna(0).astype(int)
        pc["record"] = [f"{w}–{p - w}" if p else "—" for w, p in zip(wins_, picks_)]
        pc = pc.sort_values("worlds", ascending=False)
        t = pc[["champion", "verdict", "before", "worlds", "record"]].rename(columns={
            "champion": "Champion", "verdict": "Change", "before": f"Leagues ({sc['base_patch']})",
            "worlds": "At Worlds", "record": "W–L when picked"})
        table(with_icons(t, "Champion"), {f"Leagues ({sc['base_patch']})": "{:.0%}", "At Worlds": "{:.0%}"},
              images=[" "], n=12)

    with t_fc:
        scf = wm["frozen"]
        st.markdown(f"Before Worlds, the model forecast {wm['patch']} from league play on {scf['base_patch']} plus "
                    f"the patch notes since. No Worlds games went into it.")
        scorecard(scf)
        forecast_chart(ev["worlds_frozen"], scf["base_patch"], scf["games"])

    with t_draft:
        if dr is not None:
            we = dr["events"]
            w = we[we.scope == "Worlds"]
            rest = we[we.scope != "Worlds"]
            rest_open = rest.groupby(["role", "opener"]).size()
            rest_pair = rest.groupby(["role", "opener", "answer"]).size()
            w_open = w.groupby(["role", "opener"]).size()
            wp = w.groupby(["role", "opener", "answer"]).agg(n=("game_uid", "size"), won=("answer_won", "sum")).reset_index()
            wp = wp[wp.n >= 2]
            wp["opener_n"] = [w_open[(r, o)] for r, o in zip(wp.role, wp.opener)]
            wp["rest_rate"] = [rest_pair.get((r, o, a), 0) / rest_open[(r, o)] if (r, o) in rest_open else np.nan
                               for r, o, a in zip(wp.role, wp.opener, wp.answer)]
            wp = wp.sort_values(["n", "opener_n"], ascending=False)
            st.markdown("Lane answers teams have repeated at Worlds, and how common they were before.")
            t = pd.DataFrame({"Role": wp.role.str.title(), "Their pick": wp.opener, "Answer": wp.answer,
                              "At Worlds": [f"{n} of {m}" for n, m in zip(wp.n, wp.opener_n)],
                              "Before Worlds": wp.rest_rate,
                              "Answer W–L": [f"{x}–{n - x}" for x, n in zip(wp.won, wp.n)]})
            table(with_icons(with_icons(t, "Their pick", label=" "), "Answer", label="  "),
                  {"Before Worlds": "{:.0%}"}, images=[" ", "  "])

    with t_region:
        if rg is not None:
            wr = rg["events"][rg["events"].event == "worlds"]
            reg_games = wr.drop_duplicates("region").set_index("region").region_games
            top = wr.groupby("champion").pick_ban_rate.max().nlargest(15).index
            z = wr[wr.champion.isin(top)].pivot(index="champion", columns="region", values="pick_ban_rate").reindex(top).fillna(0)
            z.columns = [f"{c} ({int(reg_games[c])})" for c in z.columns]
            reg_cols = list(z.columns)
            z = with_icons(z.rename_axis("Champion").reset_index(), "Champion")
            st.markdown("How often each region's teams pick or ban each champion (games in brackets).")
            st.dataframe(z.style.format("{:.0%}", subset=reg_cols)
                         .background_gradient(cmap="Blues", vmin=0, vmax=1, subset=reg_cols),
                         column_config=image_column(), hide_index=True, use_container_width=True)


# --- Champions & patches ----------------------------------------------------------------

elif page == "Champions & patches":
    st.title("Champions & patches")
    lead("Follow any champion through the season, see what each patch did, and check whether buffs and nerfs work.")
    league = st.selectbox("League", ["ALL"] + LEAGUES, format_func=lambda x: "All leagues" if x == "ALL" else x,
                          key="league")
    scope = stats[stats.league == league]
    trans = transitions_for(league)
    patches = sorted(scope.patch.unique(), key=patch_key)

    t_champ, t_patch, t_effect, t_pred = st.tabs(["Champion", "Patch", "Do buffs & nerfs work?", "Next patch"])

    with t_champ:
        champs = scope.groupby("champion").presence.max().sort_values(ascending=False).index.tolist()
        champ = st.selectbox("Champion", champs, index=0, key="timeline_champ")
        key = scope[scope.champion == champ].champ_key.iloc[0]
        role = rp["meta"].set_index("champ_key").primary_role.get(key) if rp else None
        champion_header(champ, str(role).title() if isinstance(role, str) else "")
        cs = scope[scope.champion == champ].set_index("patch").reindex(patches)
        cs[["presence", "pick_rate", "ban_rate"]] = cs[["presence", "pick_rate", "ban_rate"]].fillna(0)
        cc = changes[changes.champ_key == key]

        # International events as their own points on the patch axis, after that patch's league games.
        ev_pts = []
        if ev:
            for order, (k, m) in enumerate(ev["metrics"].items()):  # metrics are in event date order
                rowset = ev[f"{k}_current"].set_index("champ_key")
                r = rowset.loc[key] if key in rowset.index else None
                ev_pts.append(dict(label=f"{m['patch']} {EVENT_SHORT.get(k, k)}", patch=m["patch"], order=order,
                                   games=m["current"]["games"], presence=float(r.actual_presence) if r is not None else 0.0))
        axis = sorted([(patch_key(p), -1, p) for p in patches] + [(patch_key(e["patch"]), e["order"], e["label"]) for e in ev_pts])
        x_order = [x for *_, x in axis]
        x_patch = {p: p for p in patches} | {e["label"]: e["patch"] for e in ev_pts}

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=patches, y=cs.presence, name="Presence in league play", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2.5), marker=dict(size=8),
                                 hovertemplate="Presence %{y:.0%}<extra></extra>"))
        if ev_pts:
            ep = pd.DataFrame(ev_pts)
            fig.add_trace(go.Scatter(x=ep.label, y=ep.presence, name="International event", mode="markers",
                                     marker=dict(symbol="square", size=10, color="#52514e", line=dict(width=1.5, color="white")),
                                     customdata=ep[["games"]],
                                     hovertemplate="<b>%{x}</b><br>presence %{y:.0%} in %{customdata[0]} games<extra></extra>"))
        ymax = max(cs.presence.max(), max((e["presence"] for e in ev_pts), default=0), 0.1) * 1.15
        for _, ch in cc.iterrows():
            nxt = next((x for x in x_order if patch_key(x_patch[x]) >= patch_key(ch.patch)), None)
            if nxt is None:
                continue
            sym = {"buff": "triangle-up", "nerf": "triangle-down"}.get(ch.verdict, "diamond")
            fig.add_trace(go.Scatter(x=[nxt], y=[ymax], mode="markers", showlegend=False,
                                     marker=dict(symbol=sym, size=13, color=VERDICT_COLOR[ch.verdict],
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate=f"<b>{ch.patch}: {ch.verdict}</b><extra></extra>"))
            # By index: Plotly reads number-like categories ("26.17") as numbers when placing shapes.
            fig.add_vline(x=x_order.index(nxt), line=dict(color=VERDICT_COLOR[ch.verdict], width=1, dash="dot"))
        st.plotly_chart(style(fig, 380, yaxis=dict(title=None, range=[0, ymax * 1.08], **PCT),
                              xaxis=dict(type="category", categoryorder="array", categoryarray=x_order, title=None)),
                        use_container_width=True)
        st.caption("▲ buff · ▼ nerf · ◆ adjustment. Squares are international events, on the patch they were played on.")
        if len(cc):
            with st.expander(f"{champ}'s balance changes ({len(cc)})"):
                for _, ch in cc.sort_values("patch", key=lambda s: s.map(patch_key)).iterrows():
                    st.markdown(f"**{ch.patch}: {ch.verdict}**" + (" · aimed at pro play" if ch.pro_targeted else ""))
                    if isinstance(ch.context, str) and ch.context:
                        st.caption(ch.context)
                    ln = lines[(lines.patch == ch.patch) & (lines.champ_key == key) & lines.text.notna()]
                    st.dataframe(ln[["ability", "stat", "before", "after"]], hide_index=True, use_container_width=True)

    with t_patch:
        trans_patches = sorted(trans.to_patch.unique(), key=patch_key)
        tp = st.select_slider("Patch", trans_patches, value=trans_patches[-1], key="patch_slider")
        t = trans[trans.to_patch == tp]
        fp = t.from_patch.iloc[0]
        window = [p for p in sorted(changes.patch.unique(), key=patch_key) if patch_key(fp) < patch_key(p) <= patch_key(tp)]
        st.markdown(f"Champions changed in **{', '.join(window) or '—'}**, and how their presence moved from "
                    f"{fp} to {tp}.")
        ch = t[t.changed].sort_values("delta_presence")
        fig = go.Figure()
        for _, r in ch.iterrows():
            fig.add_trace(go.Scatter(x=[r.prev_presence, r.next_presence], y=[r.champion] * 2, mode="lines",
                                     line=dict(color=VERDICT_COLOR[r.verdict], width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ch.prev_presence, y=ch.champion, mode="markers", name=f"Before ({fp})",
                                 marker=dict(color=NEUTRAL, size=9), hovertemplate="%{y}: %{x:.0%}<extra>before</extra>"))
        fig.add_trace(go.Scatter(x=ch.next_presence, y=ch.champion, mode="markers", showlegend=False,
                                 marker=dict(color=[VERDICT_COLOR[v] for v in ch.verdict], size=11,
                                             line=dict(width=1.5, color="white")),
                                 hovertemplate="%{y}: %{x:.0%}<extra>after</extra>"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"After ({tp})", marker=dict(color="#52514e", size=11)))
        x_hi = max(ch.prev_presence.max(), ch.next_presence.max(), 0.1) + 0.03
        fig = style(fig, xaxis=dict(title="Presence", range=[-0.02, x_hi], **PCT), hovermode="closest")
        st.plotly_chart(row_layout(fig, ch.champion, legend=True), use_container_width=True)
        st.caption("Line color: blue buff, red nerf, gray adjustment.")
        with st.expander("Biggest movers this patch, changed or not"):
            movers = t.reindex(t.delta_presence.abs().sort_values(ascending=False).index).head(15)
            mt = pd.DataFrame({"Champion": movers.champion, "Change": movers.verdict.replace("none", "—"),
                               "Before": movers.prev_presence, "After": movers.next_presence,
                               "Move": movers.delta_presence})
            st.dataframe(with_icons(mt, "Champion").style.format(
                {"Before": "{:.0%}", "After": "{:.0%}", "Move": "{:+.0%}"}),
                column_config=image_column(), hide_index=True, use_container_width=True)

    with t_effect:
        eff = verdict_effects(trans).reindex(["buff", "nerf", "adjust", "none"]).dropna(how="all")
        st.markdown("Average change in presence on the next patch with pro games:")
        cols = st.columns(len(eff))
        for col, (v, row) in zip(cols, eff.iterrows()):
            label = {"buff": "After a buff", "nerf": "After a nerf", "adjust": "After an adjustment",
                     "none": "No change"}[v]
            col.metric(label, f"{row.mean_delta_presence * 100:+.1f} pts")
        changed = trans[trans.changed]
        hi_nerf = changed[(changed.verdict == "nerf") & (changed.prev_presence > 0.6)]
        lo_buff = changed[(changed.verdict == "buff") & (changed.prev_presence < 0.05)]
        if len(hi_nerf) and len(lo_buff):
            st.markdown(f"<div class='takeaway'>Nerfs bite hardest on top picks (above 60%: "
                        f"<b>{hi_nerf.delta_presence.mean() * 100:+.0f} pts</b>, {(hi_nerf.delta_presence < 0).mean():.0%} fell). "
                        f"Buffs rarely rescue unplayed champions (under 5%: "
                        f"<b>{lo_buff.delta_presence.mean() * 100:+.0f} pts</b>).</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for v in ["buff", "nerf", "adjust"]:
            sub = changed[changed.verdict == v]
            fig.add_trace(go.Scatter(
                x=sub.prev_presence, y=sub.delta_presence, mode="markers", name=v,
                marker=dict(color=VERDICT_COLOR[v], size=8, line=dict(width=1, color="white")),
                customdata=sub[["champion", "from_patch", "to_patch"]],
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}→%{customdata[2]}"
                              "<br>presence before %{x:.0%}, change %{y:+.0%}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, dash="dot"))
        st.plotly_chart(style(fig, 380, xaxis=dict(title="Presence before the change", **PCT),
                              yaxis=dict(title="Change after", tickformat="+.0%"), hovermode="closest"),
                        use_container_width=True)

    with t_pred:
        if "predictions" not in d:
            st.info("Run `python -m lolpatch predict` first.")
        else:
            pr, meta = d["predictions"], d["meta"]
            st.markdown(f"Forecast for **{meta.get('target_patch')}** from league play on {meta.get('base_patch')} "
                        "plus every balance change since. Champions changed, or expected to move 5+ points:")
            show = pr[(pr.n_changes > 0) | (pr.pred_delta_presence.abs() >= 0.05)].sort_values("pred_delta_presence")
            fig = go.Figure(go.Bar(
                x=show.pred_delta_presence, y=show.champion, orientation="h",
                marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in show.verdict], cornerradius=4),
                customdata=show[["prev_presence", "pred_presence", "verdict"]],
                hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>%{customdata[0]:.0%} → %{customdata[1]:.0%}<extra></extra>"))
            fig = style(fig, xaxis=dict(title="Predicted change in presence", tickformat="+.0%"),
                        hovermode="closest", bargap=0.3)
            st.plotly_chart(row_layout(fig, show.champion), use_container_width=True)
            st.caption("The model's track record is on the **Model accuracy** page.")


# --- Meta shifts -----------------------------------------------------------------------

elif page == "Meta shifts":
    st.title("Meta shifts")
    lead("When champions rise, someone else loses the slot. Pick a group and see who it pushed out.")
    if rp is None:
        st.info("Run `python -m lolpatch meta` first.")
        st.stop()
    t_group, t_slot = st.tabs(["Rise & fall", "Who took the slot"])

    with t_group:
        all_champs = sorted(all_stats.champion.unique())
        g_sel, g_win = st.columns([3, 2])
        members = g_sel.multiselect("Champions", all_champs, key="ripple_group",
                                    default=[c for c in ["Nocturne", "Qiyana"] if c in all_champs])
        if not members:
            st.info("Pick at least one champion.")
        else:
            gp = group_presence_cached(tuple(members))
            g_patches = list(gp.index)
            lo, hi = rise_window(gp)
            # Keyed by group: a new group resets the window to its own rise.
            win = g_win.select_slider("Window", options=g_patches, value=(lo, hi),
                                      key="ripple_window_" + "|".join(sorted(members)))
            disp = displacement(all_stats, gp, members, win[0], win[1], rp["meta"])
            label = " + ".join(members)
            if disp.empty:
                st.info("Pick a window of at least three patches.")
            else:
                pushed = disp[(disp.change <= -0.05) & (disp.r <= -0.3)]
                fell = disp[disp.change <= -0.05]
                headline = ", ".join(pushed.champion.head(3)) if len(pushed) else ", ".join(fell.champion.head(3))
                verb = "pushed out" if len(pushed) else "fell over the same stretch (not clearly pushed out)"
                st.markdown(f"<div class='takeaway'><b>{label}</b> went from <b>{gp[win[0]]:.0%}</b> to "
                            f"<b>{gp[win[1]]:.0%}</b> of games ({win[0]} → {win[1]}). Biggest losers: "
                            f"<b>{headline}</b>, {verb}.</div>", unsafe_allow_html=True)

                fig = go.Figure(go.Scatter(x=g_patches, y=gp.values, name=label, mode="lines+markers",
                                           line=dict(color="#0b0b0b", width=3), marker=dict(size=8),
                                           hovertemplate=f"{label} %{{y:.0%}}<extra></extra>"))
                pv = all_stats.pivot_table(index="champion", columns="patch", values="presence")
                for i, c in enumerate(fell.champion.head(4)):
                    y = pv.loc[c].reindex(g_patches).fillna(0) if c in pv.index else pd.Series(0, index=g_patches)
                    fig.add_trace(go.Scatter(x=g_patches, y=y.values, name=c, mode="lines",
                                             line=dict(color=SERIES[i], width=1.8),
                                             hovertemplate=f"{c} %{{y:.0%}}<extra></extra>"))
                fig.add_vrect(x0=g_patches.index(win[0]), x1=g_patches.index(win[1]), fillcolor="#2a78d6",
                              opacity=0.06, line_width=0)
                st.plotly_chart(style(fig, 340, xaxis=dict(type="category", title=None),
                                      yaxis=dict(title="Presence", **PCT)), use_container_width=True)

                def lead_table(df):
                    t = pd.DataFrame({
                        "Champion": df.champion, "Role": df.role.fillna("—").str.title(),
                        "Presence": [f"{b:.0%} → {a:.0%}" for b, a in zip(df.before, df.after)],
                        "Pushed out?": ["likely" if r <= -0.3 else "no, moves with them" if r >= 0.3 else "unclear"
                                        if pd.notna(r) else "—" for r in df.r],
                    })
                    return with_icons(t, "Champion")

                l, r = st.columns(2)
                with l:
                    st.markdown("**Fell**")
                    table(lead_table(fell), images=[" "], n=8)
                with r:
                    st.markdown("**Rose alongside**")
                    table(lead_table(disp[disp.change >= 0.05].sort_values("change", ascending=False)), images=[" "], n=8)
                with st.expander("How “pushed out” is decided"):
                    st.markdown(
                        "Presence at each end of the window is averaged over two patches. **Pushed out** means the "
                        "champion's patch-to-patch moves go opposite the group's: when the group rises, it drops. "
                        "Champions that fell over the window but move *with* the group from patch to patch were "
                        "declining for other reasons, like their own nerfs or a different meta.")

    with t_slot:
        st.markdown("After a big buff or nerf, which untouched champion in the same role moved the other way?")
        subs, pairs = rp["subs"], rp["pairs"]
        role_f = st.radio("Role", ["all"] + ROLES, horizontal=True, key="ripple_role",
                          format_func=lambda r: r.title() if r != "all" else "All roles")
        sv = subs if role_f == "all" else subs[subs.role == role_f]
        sv = sv.reindex(sv.changed_delta.abs().sort_values(ascending=False).index)
        t = pd.DataFrame({
            "Patch": sv.to_patch.values, "Role": sv.role.str.title().values, "Changed": sv.changed.values,
            "Its change": [f"{c}: {b:.0%} → {a:.0%}" for c, b, a in zip(sv.change, sv.changed_before, sv.changed_after)],
            "Took the slot": sv.other.values,
            "Their move": [f"{b:.0%} → {a:.0%}" for b, a in zip(sv.other_before, sv.other_after)],
        })
        t = with_icons(with_icons(t, "Changed", label=" "), "Took the slot", label="  ")
        table(t, images=[" ", "  "])
        rec = pairs[pairs.times >= 2]
        if len(rec):
            st.markdown("**Happened more than once:** " + "; ".join(
                f"{r.other} replacing {r.changed} ({r.role}, {r.patches})" for r in rec.itertuples()))
        st.caption(f"A changed champion moved {SUB_MIN_MOVE_A:.0%}+ and an untouched same-role champion moved "
                   f"{SUB_MIN_MOVE_B:.0%}+ the other way in the same patch window. Timing suggests, it doesn't prove.")


# --- Draft answers ---------------------------------------------------------------------

elif page == "Draft answers":
    st.title("Draft answers")
    lead("When a team locks a champion, what does the opponent answer with in the same role?")
    if dr is None:
        st.info("Run `python -m lolpatch draft` first.")
        st.stop()
    solid = dr["lanes"][dr["lanes"].solid].copy()
    f1, f2 = st.columns([3, 2])
    role_d = f1.radio("Role", ["all"] + ROLES, horizontal=True, key="draft_role",
                      format_func=lambda r: r.title() if r != "all" else "All roles")
    kinds = {"Everyone": None, "Mostly at Worlds": "Worlds", "One region": "(region)", "One team": "(team)",
             "One player": "(player)"}
    kind = f2.selectbox("Who does it", list(kinds), key="draft_kind")
    view = solid if role_d == "all" else solid[solid.role == role_d]
    if kinds[kind]:
        view = view[view.where_from.str.contains(kinds[kind], regex=False)]
    view = view.sort_values(["n", "lift"], ascending=False)
    if len(view):
        top = view.iloc[0]
        st.markdown(f"<div class='takeaway'>Most common: <b>{top.answer} into {top.opener}</b> ({top.role}), "
                    f"{top.n} times, {top.lift:.1f}× as often as {top.answer} is usually picked there.</div>",
                    unsafe_allow_html=True)
    table(pair_table(view), images=[" ", "  "], more_label="Show all patterns")

    if len(view):
        with st.expander("Who makes a given answer? (by event, region, team and player)"):
            opts = list(zip(view.role, view.opener, view.answer))
            pick = st.selectbox("Answer", opts, key="draft_pattern", format_func=lambda k: f"{k[2]} into {k[1]} ({k[0]})")
            bd = dr["breakdown"]
            bd = bd[(bd.role == pick[0]) & (bd.opener == pick[1]) & (bd.answer == pick[2])]
            cols = st.columns(2)
            for i, (dim, g) in enumerate(bd.groupby("dimension", sort=False)):
                g = g.sort_values(["answers", "rate"], ascending=False).head(6)
                cols[i % 2].markdown(f"**By {dim}**")
                cols[i % 2].dataframe(pd.DataFrame({
                    dim.capitalize(): g.group, "Answered": [f"{a} of {o}" for a, o in zip(g.answers, g.openers)],
                    "Rate": g.rate, "Everyone else": g.rest_rate,
                }).style.format({"Rate": "{:.0%}", "Everyone else": "{:.0%}"}, na_rep="—"),
                    hide_index=True, use_container_width=True)

    with st.expander("Answers to the very first pick of the draft"):
        fp = dr["first"]
        fp = fp[fp.solid].sort_values(["n", "lift"], ascending=False)
        st.caption("Any role. Often it's “you took one of the two best champions, we take the other.”")
        # Already inside an expander (they can't nest), so show every row in a scrolling table.
        st.dataframe(pair_table(fp, "First pick", "Answer"), column_config=image_column(" ") | image_column("  "),
                     hide_index=True, use_container_width=True, height=min(420, 36 * len(fp) + 40))

    with st.expander("How answers are found"):
        dm = dr["meta"]
        st.markdown(
            f"gol.gg's pick lists reveal the draft order: the first-picking team takes picks 1, 4–5 and 8–9, the "
            f"other team 2–3, 6–7 and 10. That's known for **{dm['games']:,} games** "
            f"({dm['lanes']:,} lane match-ups). An answer is the later of the two picks in the same role.\n\n"
            f"A pattern needs at least 4 occurrences and to happen at least 1.5× as often as the answering champion "
            "is usually picked in that role **on the same patches** (so two champions that were simply popular at "
            "the same time don't count), and to survive a statistical correction for testing "
            f"{dm['lane_pairs_tested']:,} pairs.\n\n"
            "**Who does it** names an event, region, team or player when they make the answer at least twice as "
            "often as everyone else. It means *led by*, not *only by*.")


# --- Teams & regions ---------------------------------------------------------------------

elif page == "Teams & regions":
    st.title("Teams & regions")
    lead("What each team and region leans on, and whether leaning on it wins games.")
    if rg is None:
        st.info("Run `python -m lolpatch regions` first.")
        st.stop()
    t_team, t_region, t_wins = st.tabs(["Team explorer", "Region preferences", "Does priority win?"])

    with t_team:
        tc = rg["teams"]
        teams = tc.drop_duplicates("team").sort_values(["league", "team"])
        team = st.selectbox("Team", teams.team, key="team_pick",
                            format_func=lambda t: f"{t} ({teams.set_index('team').league[t]})")
        tt = tc[tc.team == team]
        c1, c2 = st.columns([1, 12], vertical_alignment="center")
        if team_logo(team):
            c1.image(team_logo(team), width=56)
        c2.markdown(f"### {team}\n<span class='muted'>{int(tt.team_games.iloc[0])} games in {tt.league.iloc[0]}</span>",
                    unsafe_allow_html=True)
        sig = tt[tt.picks >= 4].nlargest(10, "pick_lift").iloc[::-1]
        l, r = st.columns([3, 2], gap="large")
        with l:
            st.markdown("**Signature picks**: played more than the rest of their league")
            fig = go.Figure(go.Bar(
                x=sig.pick_lift, y=sig.champion, orientation="h", marker=dict(color=BUFF, cornerradius=4),
                customdata=sig[["picks", "pick_share", "league_pick_share", "win_rate"]],
                text=[f"{w:.0%} WR" for w in sig.win_rate], textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b>: in %{customdata[1]:.0%} of games vs league %{customdata[2]:.0%}"
                              "<br>%{customdata[0]} picks · %{customdata[3]:.0%} win rate<extra></extra>"))
            fig = style(fig, xaxis=dict(title="Pick share above league average",
                                        range=[0, sig.pick_lift.max() * 1.25], **PCT), hovermode="closest")
            st.plotly_chart(row_layout(fig, sig.champion, right=64), use_container_width=True)
        with r:
            st.markdown("**Most banned against them**")
            banned = tt.nlargest(8, "bans_against")
            bt = pd.DataFrame({"Champion": banned.champion, "Of their games": banned.against_rate})
            st.dataframe(with_icons(bt, "Champion").style.format({"Of their games": "{:.0%}"}),
                         column_config=image_column(), hide_index=True, use_container_width=True)

    with t_region:
        metric = st.radio("Measure", ["presence_lift", "early_lift"], horizontal=True, key="region_metric",
                          format_func=lambda m: {"presence_lift": "Picked or banned",
                                                 "early_lift": "Taken early (first 2 picks / 3 bans)"}[m])
        rc = rg["regions"]
        champs = rc.groupby("champion").presence.mean().nlargest(25).index
        z = rc[rc.champion.isin(champs)].pivot(index="champion", columns="league", values=metric).reindex(champs)
        base = "presence" if metric == "presence_lift" else "early_priority"
        exp = "expected_presence" if metric == "presence_lift" else "expected_early"
        hv = rc[rc.champion.isin(champs)].assign(
            txt=lambda x: [f"{o:.0%} vs {e:.0%} league average" for o, e in zip(x[base], x[exp])]
        ).pivot(index="champion", columns="league", values="txt").reindex(champs)
        lim = max(0.05, float(np.nanmax(np.abs(z.values))))
        fig = go.Figure(go.Heatmap(
            z=z.values, x=z.columns, y=z.index, zmid=0, zmin=-lim, zmax=lim,
            colorscale=[[0, NERF], [0.5, "#f0efec"], [1, BUFF]], xgap=2, ygap=2,
            customdata=hv.values, hovertemplate="<b>%{y}</b> · %{x}<br>%{customdata}<extra></extra>",
            colorbar=dict(title="vs avg", tickformat="+.0%", thickness=10)))
        fig = style(fig, 400, hovermode="closest", xaxis=dict(side="top"))
        st.plotly_chart(row_layout(fig, z.index, x_top=True, right=90, reverse=True), use_container_width=True)
        st.caption("Blue = the region plays it more than the average league on the same patches; red = less. "
                   "The 25 most-contested champions of the season.")

    with t_wins:
        eff = rg["effects"]
        md, cf, fv = eff["meta_draft"], eff["comfort"], eff["region_favorites"]
        st.markdown("<div class='takeaway'>Mostly <b>no</b>. Once you account for how good the teams are, priority "
                    "itself doesn't show a reliable edge.</div>", unsafe_allow_html=True)
        st.markdown("")
        rows = [
            ("Drafting the most-contested champions", f"{md['adjusted_win_prob_plus10pp']:.0%} win rate",
             "No edge: team strength explains the wins."),
            ("Sticking to champions a team has played a lot", f"{cf['within_team_diff'] * 100:+.0f} pts",
             "No edge; if anything slightly worse."),
        ]
        if fv["regions"]:
            rows.append(("Regions playing their home favorites at internationals",
                         f"{fv['within_region_diff'] * 100:+.0f} pts", "Too few games to tell."))
        for title, value, note in rows:
            with st.container(border=True):
                a, b = st.columns([3, 1], vertical_alignment="center")
                a.markdown(f"**{title}**  \n<span class='muted'>{note}</span>", unsafe_allow_html=True)
                b.markdown(f"### {value}")
        with st.expander("How these were tested"):
            st.markdown(
                f"- **Contested champions:** {md['games']:,} league games. Compares teams whose picks had higher "
                "presence on the *previous* patch (what teams knew at draft time), controlling for each team's "
                f"season win rate. 95% range {md['adjusted_ci'][0]:.0%}–{md['adjusted_ci'][1]:.0%}.\n"
                f"- **Comfort picks:** within each of {cf['teams']} teams, win rate on champions it had already played "
                f"3+ times *before that game* vs its other picks. 95% range {cf['ci'][0] * 100:+.0f} to "
                f"{cf['ci'][1] * 100:+.0f} pts.\n"
                + (f"- **Home favorites:** {fv['favorite_picks']} picks at international events against other "
                   f"regions. 95% range {fv['ci'][0] * 100:+.0f} to {fv['ci'][1] * 100:+.0f} pts." if fv["regions"] else ""))


# --- Model accuracy ----------------------------------------------------------------------

elif page == "Model accuracy":
    st.title("Model accuracy")
    lead("Can patch notes predict the meta? Each international event is a test the model never saw.")
    if not ev:
        st.info("Run `python -m lolpatch events` first.")
        st.stop()
    version = st.radio("Forecast", ["frozen", "current"], horizontal=True, key="fc_version",
                       format_func=lambda v: {"frozen": "Saved before the event (blind)",
                                              "current": "Rebuilt with today's model"}[v])
    rows = []
    for key, m in ev["metrics"].items():
        sc = m[version]
        rows.append({"Event": m["title"], "Patch": m["patch"], "Games": sc["games"],
                     "Forecast error": sc["all"]["mae_model"], "“Nothing changes” error": sc["all"]["mae_baseline"],
                     "Better?": "✓" if sc["all"]["mae_model"] < sc["all"]["mae_baseline"] else "—"})
    st.dataframe(pd.DataFrame(rows).style.format({"Forecast error": lambda v: f"{v * 100:.1f} pts",
                                                  "“Nothing changes” error": lambda v: f"{v * 100:.1f} pts"}),
                 hide_index=True, use_container_width=True)
    st.caption("Error = average gap between forecast and actual presence; lower is better.")

    keys = list(ev["metrics"])
    pick = st.selectbox("Look at one event", keys, index=keys.index("worlds") if "worlds" in keys else 0,
                        format_func=lambda k: ev["metrics"][k]["title"], key="fc_event")
    sc = ev["metrics"][pick][version]
    scorecard(sc)
    forecast_chart(ev[f"{pick}_{version}"], sc["base_patch"], sc["games"])

    with st.expander("How the forecast works, and what was tried"):
        st.markdown(
            "The model learns from the leagues how presence moves after buffs, nerfs and adjustments, and how "
            "champions drift on their own. For each event it starts from league play on the last patch before it "
            "and applies every balance change since. Forecasts saved before an event are never edited. "
            "(First Stand, MSI and EWC forecasts were built later with the current model, so only Worlds and the "
            "Demacia Cup are true blind tests.)")
        if "adjustment" in ev:
            st.markdown(
                "**Tried and dropped:** an “international adjustment” assuming top picks hold their priority at "
                "events. On the earlier events it made forecasts worse, because top picks actually tend to fall "
                "*more* at international events than in the leagues.")
        if "backtest_metrics" in d:
            row = d["backtest_metrics"].set_index(d["backtest_metrics"].columns[0]).loc["changed champions"]
            st.markdown(f"**League backtest** (each patch predicted from earlier patches only): for changed "
                        f"champions the error was {row.mae_model * 100:.1f} pts vs {row.mae_persistence * 100:.1f} "
                        f"for “nothing changes”, with the direction right {row.direction_acc_model:.0%} of the time.")


# --- Changelog ---------------------------------------------------------------------------

elif page == "Changelog":
    st.title("Changelog")
    lead("What changed in each version of this project, newest first.")
    # Skip the file's own title; the page has one.
    st.markdown(changelog_text.split("\n", 1)[1] if changelog_text.startswith("# ") else changelog_text)


# --- footer ------------------------------------------------------------------------------
st.divider()
st.caption(
    "Data: [Games of Legends (gol.gg)](https://gol.gg) · Riot patch notes · Riot Data Dragon · Meraki Analytics. "
    "A fan project, not endorsed by Riot Games and not reflecting the views of Riot Games or anyone officially "
    "involved in producing or managing Riot Games properties. Riot Games and all associated properties are "
    "trademarks or registered trademarks of Riot Games, Inc.")

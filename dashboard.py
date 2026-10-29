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
from lolpatch.ui import TEAM_SHORT, CHART_FONT, champ_icon, image_column, nice, row_layout, team_logo, with_icons

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
  @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');
  /* Inter for reading text; Chakra Petch only for headings. Icon spans set their own font. */
  html, body, .stApp, p, li, label, input, textarea, select, button, td, th,
  div[data-testid="stMarkdownContainer"], div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    font-family: 'Inter', 'Source Sans Pro', system-ui, sans-serif;
    font-feature-settings: 'tnum' 1;
  }
  h1, h2, h3, h1 *, h2 *, h3 * { font-family: 'Chakra Petch', 'Inter', system-ui, sans-serif !important; letter-spacing: .01em; }
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

@st.cache_resource
def _cache_state() -> dict:
    return {}


def _files_version() -> float:
    """Latest change time across the data and code the cached loaders depend on."""
    root = PROCESSED.parent.parent
    files = [*PROCESSED.glob("*"), *(root / "data" / "forecasts").glob("*"), *(root / "lolpatch").glob("*.py")]
    return max((p.stat().st_mtime for p in files if p.is_file()), default=0.0)


# Streamlit keeps cached data across redeploys unless a loader's own code changes, so new data
# (e.g. from the scheduled refresh) or changed columns would otherwise be served stale.
_state, _version = _cache_state(), _files_version()
if _state.get("version") != _version:
    st.cache_data.clear()
    _state["version"] = _version


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
    # event/league are empty for domestic games, so read them as text to avoid mixed-type columns.
    read = lambda name: pd.read_csv(PROCESSED / name, dtype={"event": str, "league": str, "series_id": str})
    out = dict(games=read("match_games.csv"), teams=read("match_teams.csv"), ratings=read("player_ratings.csv"),
               timeline=read("match_timeline.csv"), gold=read("match_gold.csv"), expected=read("match_expected.csv"))
    ev_path = PROCESSED / "expected_eval.json"
    out["eval"] = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    return out


@st.cache_data
def load_upcoming():
    path = PROCESSED / "team_strength.csv"
    if not path.exists():
        return None
    read = lambda name: pd.read_csv(PROCESSED / name)
    return dict(strength=read("team_strength.csv").set_index("team").strength, pools=read("team_pools.csv"),
                bans=read("team_bans.csv"), meta=read("worlds_role_meta.csv"), standings=read("worlds_standings.csv"),
                schedule=read("upcoming.csv"))


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
    layout = dict(margin=dict(l=10, r=10, t=40, b=10), font=dict(family=CHART_FONT),
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
    t["Compared with usual"] = [f"{x:.1f}× more often" for x in df.lift]
    if "where_from" in df:
        t["Who does it most"] = df.where_from.replace("Broad meta", "everyone").values
    if detail:
        t["Answer rate"] = df.answer_rate.values
        t["Usual rate"] = df.base_rate.values
        if "answer_win_rate" in df:
            t["Answer won"] = df.answer_win_rate.values
    t = with_icons(t, opener_label, label=" ")
    return with_icons(t, answer_label, label="  ")


PAIR_FMT = {"Answer rate": "{:.0%}", "Usual rate": "{:.0%}", "Answer won": "{:.0%}"}


def forecast_chart(cmp: pd.DataFrame, base_patch: str, games: int, n: int = 30):
    """Domestic -> forecast -> actual presence for the champions that mattered most."""
    relevance = cmp[["domestic_presence", "pred_presence", "actual_presence"]].max(axis=1)
    show = cmp.loc[relevance.nlargest(n).index].sort_values("actual_presence")
    fig = go.Figure()
    for _, r in show.iterrows():
        fig.add_trace(go.Scatter(x=[r.domestic_presence, r.pred_presence, r.actual_presence], y=[r.champion] * 3,
                                 mode="lines", line=dict(color="#c3c2b7", width=1.5), showlegend=False, hoverinfo="skip"))
    hover = "<b>%{y}</b><br>%{x:.0%}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=show.domestic_presence, y=show.champion, mode="markers", name=f"In the leagues ({base_patch})",
                             marker=dict(color=NEUTRAL, size=8), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=show.pred_presence, y=show.champion, mode="markers", name="Our prediction",
                             marker=dict(color="rgba(0,0,0,0)", size=11, line=dict(width=2, color=SERIES[1])),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=show.actual_presence, y=show.champion, mode="markers", showlegend=False,
                             marker=dict(color=[VERDICT_COLOR[v] for v in show.verdict], size=11,
                                         line=dict(width=1.5, color="white")),
                             customdata=show[["verdict", "change_patches"]],
                             hovertemplate="<b>%{y}</b> — %{customdata[0]} %{customdata[1]}<br>actual %{x:.0%}<extra></extra>"))
    # Legend entry for "actual" in neutral ink: the real dots are colored by change type.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"What happened ({games} games)",
                             marker=dict(color="#52514e", size=11)))
    # Start the axis slightly below 0 so 0% markers aren't cut off at the edge.
    fig = style(fig, xaxis=dict(title="Pick/ban rate (share of games where it was picked or banned)",
                                range=[-0.03, 1.03], **PCT), hovermode="closest")
    st.plotly_chart(row_layout(fig, show.champion, legend=True), use_container_width=True)
    st.caption("Gray dot = how often it was picked or banned in the leagues · orange ring = what we predicted · "
               "colored dot = what actually happened (blue = buffed since, red = nerfed, gray = no change). "
               f"The {n} champions that mattered most.")


def scorecard(sc: dict):
    a, b, c = st.columns(3)
    a.metric("Our prediction was off by", f"{sc['all']['mae_model'] * 100:.1f} points",
             help="On average, how far our predicted pick/ban rate was from the real one. "
                  "Points = percentage points: predicting 40% when it was 44% is 4 points off.")
    b.metric("Guessing “no change” was off by", f"{sc['all']['mae_baseline'] * 100:.1f} points",
             help=f"The same, if you simply assumed every champion would be picked or banned exactly as often as in "
                  f"the leagues on {sc['base_patch']}. Lower is better, so our prediction should beat this.")
    c.metric("Top 10 champions guessed right", f"{sc['top10_model']} of 10")


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


RATING_HELP = ("A score out of 5 for this game. It compares the player's damage, gold, kill participation, deaths, "
               "vision and more with every other player in the same role in 2026 pro play, including how they did "
               "against their lane opponent and how much of their team's and the game's damage and gold they had. "
               "Winning when your team wasn't expected to earns extra credit. 3.0 is an average pro game; 4.5 or "
               "more is one of the best 10%.")


def stage_label(tournament: str, stage: str) -> str:
    """gol.gg stage codes to readable names: ROUND2 at Worlds is 'Swiss Round 2' (or 'Play-In Round 2')."""
    m = re.fullmatch(r"ROUND(\d+)", str(stage).upper())
    if m:
        return f"{'Play-In' if 'Play-In' in str(tournament) else 'Swiss'} Round {m.group(1)}"
    return str(stage).replace("3RD.PALCE", "3rd Place").title()


def worlds_series(mx: dict) -> pd.DataFrame:
    """One row per Worlds series: teams, score, format, date, stage."""
    games = mx["games"][mx["games"].event == "worlds"]
    rows = []
    for sid, g in games.groupby("series_id"):
        g = g.sort_values("game_no")
        wins = g.winner.value_counts()
        teams = sorted(set(g.blue) | set(g.red), key=lambda t: -wins.get(t, 0))
        a, b = teams[0], teams[1] if len(teams) > 1 else ""
        best_of = {1: 1, 2: 3, 3: 5}.get(int(wins.max()), len(g))
        rows.append(dict(series_id=sid, date=g.date.iloc[0], stage=stage_label(g.tournament.iloc[0], g.stage.iloc[0]),
                         team_a=a, team_b=b,
                         wins_a=int(wins.get(a, 0)), wins_b=int(wins.get(b, 0)), games=len(g), best_of=best_of))
    return pd.DataFrame(rows).sort_values(["date", "series_id"], ascending=False).reset_index(drop=True)


def worlds_series_tab(mx: dict):
    series = worlds_series(mx)
    rt = mx["ratings"]
    exp = mx["expected"].set_index(["game_uid", "team"]).p_win
    games = mx["games"]
    rows = []
    for s in series.itertuples():
        sg = games[games.series_id == s.series_id]
        first = sg.sort_values("game_no").iloc[0]
        pre = exp.get((first.game_uid, s.team_a), np.nan)
        sr = rt[rt.series_id == s.series_id].groupby(["player", "team"]).rating.mean()
        mvp = sr.idxmax() if len(sr) else ("—", "")
        rows.append({
            "Date": s.date, "Stage": s.stage, "Format": f"Bo{s.best_of}",
            " ": team_logo(s.team_a), "Winner": s.team_a, "Score": f"{s.wins_a}–{s.wins_b}",
            "Loser": s.team_b, "  ": team_logo(s.team_b),
            "Winner's chance before game 1": pre, "Series MVP": f"{mvp[0]} ({mvp[1]})" if len(sr) else "—",
        })
    st.markdown(f"{len(series)} series so far. Click one to open its games.")
    picked = st.dataframe(
        pd.DataFrame(rows).style.format({"Winner's chance before game 1": "{:.0%}"}, na_rep="—"),
        column_config={" ": st.column_config.ImageColumn(" ", width="small"),
                       "  ": st.column_config.ImageColumn("  ", width="small")},
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
        key="worlds_series_table", height=min(440, 36 * len(rows) + 40))
    rows_sel = (picked or {}).get("selection", {}).get("rows", [])
    if not rows_sel:
        return
    s = series.iloc[rows_sel[0]]
    sg = games[games.series_id == s.series_id].sort_values("game_no")
    st.divider()
    st.markdown(f"### {s.team_a} {s.wins_a}–{s.wins_b} {s.team_b}")
    st.caption(f"{s.stage} · {s.date} · best of {s.best_of}")
    if len(sg) > 1:
        labels = {r.game_no: f"Game {r.game_no} · {r.winner} Won" for r in sg.itertuples()}
        game_no = st.radio("Game", list(labels), horizontal=True, format_func=labels.get,
                           key=f"series_game_{s.series_id}")
    else:
        game_no = sg.game_no.iloc[0]
    g = sg[sg.game_no == game_no].iloc[0]
    if len(sg) > 1:
        st.caption(f"Game {game_no} of {len(sg)}")
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
                        f"{format(p, '.0%') + ' chance to win' if pd.notna(p) else ''}")
    c2.markdown(f"<div style='text-align:center'><span class='muted'>{stage_label(g.tournament, g.stage)} · {g.date}</span><br>"
                f"<b style='font-size:1.4rem'>{g.duration_str}</b><br><span class='muted'>patch "
                f"{str(g.version).replace('16.', '26.')}</span></div>", unsafe_allow_html=True)
    winner_p = exp.p_win.get(g.winner, np.nan)
    if pd.notna(winner_p):
        verdict = ("an upset" if winner_p < 0.4 else "the expected result" if winner_p > 0.6 else "close to a coin flip beforehand")
        edge = exp.draft_edge.get(blue.team, np.nan)
        draft_note = ""
        if pd.notna(edge) and abs(edge) >= 0.005:
            ahead = blue.team if edge > 0 else red.team
            draft_note = (f" {possessive(ahead)} champions had been winning slightly more lately, but this year "
                          "drafts haven't turned out to predict who wins.")
        st.caption(f"Before the game, based on how both teams have played this year, {g.winner} had a "
                   f"{winner_p:.0%} chance to win, so this was {verdict}." + draft_note)

    # Players first, no click needed; # ranks all ten players in the game by rating.
    rt = rt.assign(game_rank=rt.rating.rank(ascending=False, method="min").astype(int))
    # Each player's record on their champion before this game (earlier days, like "# of Games").
    hist = mx["ratings"]
    hist = hist[hist.date < g.date].groupby(["player", "champion"]).result.agg(["sum", "count"])
    rt["champ_record"] = [f"{int(hist.loc[(pl, ch), 'sum'])}–{int(hist.loc[(pl, ch), 'count'] - hist.loc[(pl, ch), 'sum'])}"
                          if (pl, ch) in hist.index else "—" for pl, ch in zip(rt.player, rt.champion)]
    for tm in (blue, red):
        st.markdown(f"**{tm.team}**" + (" · won" if tm.result else ""))
        p = rt[rt.team == tm.team].copy()
        p["order"] = p.role.map({r: i for i, r in enumerate(ROLES)})
        p = p.sort_values("order")
        t = pd.DataFrame({
            "#": p.game_rank, "Role": p.role.str.title(), "Player": p.player, "Champion": p.champion,
            "# of Games": p.champ_games, "Record on Champ": p.champ_record,
            "K / D / A": [f"{int(k)} / {int(d_)} / {int(a)}" for k, d_, a in zip(p.Kills, p.Deaths, p.Assists)],
            "Kill participation": p["KP%"] / 100,
            "Damage": p["Total damage to Champion"], "% of team damage": p["DMG%"] / 100,
            "Gold": p.Golds, "% of team gold": p["GOLD%"] / 100,
            "Expected": p.expected_rating, "Rating": p.rating,
        })
        t = with_icons(t, "Champion")
        st.dataframe(t.style.format({"Kill participation": "{:.0%}", "Damage": "{:,.0f}", "% of team damage": "{:.0%}",
                                     "Gold": "{:,.0f}", "% of team gold": "{:.0%}", "Expected": "{:.1f}",
                                     "Rating": "{:.1f}"})
                     .background_gradient(cmap="Blues", subset=["Rating"], vmin=1, vmax=5),
                     column_config=image_column() | {
                         "Rating": st.column_config.NumberColumn("Rating", help=RATING_HELP),
                         "Expected": st.column_config.NumberColumn(
                             "Expected", help="The rating we'd have expected before the game, from how the player "
                                              "has been playing, how they do on this champion, and their team's "
                                              "chance to win."),
                         "Kill participation": st.column_config.NumberColumn(
                             "Kill participation", help="Share of their team's kills they got or assisted."),
                         "# of Games": st.column_config.NumberColumn(
                             "# of Games", help="Pro games on this champion this season before this game."),
                         "Record on Champ": st.column_config.TextColumn(
                             "Record on Champ", help="Wins–losses on this champion this season before this game.")},
                     hide_index=True, use_container_width=True)
    st.caption("# = rank among all ten players in this game, by rating.")
    with st.expander("More stats: compared with the whole game and with the lane opponent"):
        more = rt.copy()
        more["order"] = more.role.map({x: i for i, x in enumerate(ROLES)})
        more = more.sort_values(["team", "order"])
        mt = pd.DataFrame({
            "Team": more.team, "Role": more.role.str.title(), "Player": more.player, "Champion": more.champion,
            "% of all damage in the game": more.dmg_game / 100, "% of all gold in the game": more.gold_game / 100,
            "Damage vs lane opponent": [f"{a:,.0f} vs {b:,.0f}" for a, b in
                                        zip(more["Total damage to Champion"], more.opp_dmg)],
            "Gold vs lane opponent": [f"{a - b:+,.0f}" for a, b in zip(more.Golds, more.opp_gold)],
            "Damage per 1k gold": more.dmg_per_gold * 1000,
            "CS per min": more.CSM, "Vision per min": more.VSPM, "Gold lead at 15 min": more["GD@15"],
        })
        st.dataframe(with_icons(mt, "Champion").style.format(
            {"% of all damage in the game": "{:.0%}", "% of all gold in the game": "{:.0%}",
             "Damage per 1k gold": "{:,.0f}", "CS per min": "{:.1f}", "Vision per min": "{:.1f}",
             "Gold lead at 15 min": "{:+,.0f}"}),
            column_config=image_column(), hide_index=True, use_container_width=True)
        st.caption("All of these feed into the rating, weighted by role: carries are judged more on damage and "
                   "gold, supports more on vision, kill participation and protecting teammates.")

    t_over, t_draft, t_gold = st.tabs(["Team Stats", "Draft", "Gold & Objectives"])
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
            pc.image([champ_icon(c) for c in picks if champ_icon(c)], width=48, caption=[nice(c) for c in picks])
            bc.caption("Bans")
            bc.image([champ_icon(c) for c in bans if champ_icon(c)], width=40, caption=[nice(c) for c in bans])

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


def worlds_players_page(mx: dict):
    rt = mx["ratings"]
    w = rt[rt.event == "worlds"]
    domestic = rt[rt.event.isna()].groupby("player").rating.mean()
    min_games = st.slider("Minimum Games at Worlds", 1, 6, 3, key="wp_min_games")
    agg = w.groupby(["player", "team", "role"]).agg(
        games=("game_uid", "nunique"), rating=("rating", "mean"), expected=("expected_rating", "mean"),
        wins=("result", "sum"), champs=("champion", lambda s: ", ".join(map(nice, s.value_counts().index[:3]))),
    ).reset_index()
    agg["vs_expected"] = agg.rating - agg.expected
    agg["domestic"] = agg.player.map(domestic)
    agg = agg[agg.games >= min_games]
    if agg.empty:
        st.info("No players with that many Worlds games yet.")
        return

    # Overall and within-role ranks by average rating, among players shown at this minimum.
    agg = agg.sort_values("rating", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["role_rank"] = agg.groupby("role").cumcount() + 1

    def board(df, value_col, value_label, fmt, ranks=None, highlight=None):
        cols = {"#": range(1, len(df) + 1) if ranks is None else ranks, " ": df.team.map(team_logo), "Player": df.player, "Team": df.team,
                "Role": df.role.str.title(), "Games": df.games,
                "Record": [f"{int(w_)}–{int(n - w_)}" for w_, n in zip(df.wins, df.games)]}
        if value_col != "rating":  # the ranking value first, then the rating it came from
            cols[value_label] = df[value_col]
        cols |= {"Average rating": df.rating, "Expected rating": df.expected, "Rating in their league": df.domestic,
                 "Champions": df.champs}
        t = pd.DataFrame(cols)
        styled = t.style.format({value_label: fmt, "Average rating": "{:.1f}", "Expected rating": "{:.1f}",
                                 "Rating in their league": "{:.1f}"}, na_rep="—")
        if highlight is not None:
            styled = styled.apply(lambda row: ["background-color: rgba(42,120,214,.14); font-weight: 600"
                                               if row.Player == highlight else "" for _ in row], axis=1)
        st.dataframe(styled,
                     column_config={" ": st.column_config.ImageColumn(" ", width="small")},
                     hide_index=True, use_container_width=True, height=36 * min(len(t), 10) + 40)

    t_best, t_roles, t_low, t_up, t_down, t_look = st.tabs(
        ["Best", "Top 10 by Role", "Lowest Rated", "Above Expectations", "Below Expectations", "Player Lookup"])
    with t_best:
        top = agg.sort_values("rating", ascending=False).head(10)
        st.markdown(f"<div class='takeaway'>Best so far: <b>{top.iloc[0].player}</b> ({top.iloc[0].team}), "
                    f"averaging <b>{top.iloc[0].rating:.1f}</b> over {top.iloc[0].games} games.</div>",
                    unsafe_allow_html=True)
        board(top, "rating", "Average rating", "{:.1f}")
    with t_roles:
        st.markdown("The ten best-rated players in each position. Ratings compare each player with others "
                    "in the same position, so the positions are on the same scale.")
        role_order = [r for r in ["top", "jungle", "mid", "bot", "support"] if r in set(agg.role)]
        for role, tab in zip(role_order, st.tabs([r.title() for r in role_order])):
            with tab:
                board(agg[agg.role == role].head(10), "rating", "Average rating", "{:.1f}")
    with t_low:
        low = agg.sort_values("rating").head(10)
        st.markdown("The lowest averages so far. Often a mix of tough opponents and losing games, "
                    "since the result is part of the rating.")
        board(low, "rating", "Average rating", "{:.1f}")
    with t_up:
        up = agg.sort_values("vs_expected", ascending=False).head(10)
        st.markdown(f"<div class='takeaway'>Playing furthest above what their form, champions and opponents "
                    f"predicted: <b>{up.iloc[0].player}</b> ({up.iloc[0].team}), "
                    f"<b>{up.iloc[0].vs_expected:+.1f}</b> rating points per game above what we expected.</div>",
                    unsafe_allow_html=True)
        board(up, "vs_expected", "Above expected (per game)", "{:+.1f}")
    with t_down:
        down = agg.sort_values("vs_expected").head(10)
        st.markdown("Playing furthest below what their form, champions and opponents predicted.")
        board(down, "vs_expected", "Below expected (per game)", "{:+.1f}")
    with t_look:
        who = st.selectbox("Player", sorted(agg.player + " (" + agg.team + ")", key=str.lower),
                           key="worlds_player_pick")
        name = who.rsplit(" (", 1)[0]
        me = agg[agg.player == name].iloc[0]
        n_role = int((agg.role == me.role).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Rank", f"#{me['rank']} of {len(agg)}")
        c2.metric(f"Rank Among {me.role.title()} Players", f"#{me.role_rank} of {n_role}")
        c3.metric("Average Rating", f"{me.rating:.1f} / 5",
                  f"{me.vs_expected:+.1f} vs. expected", delta_color="normal")
        near = agg.iloc[max(0, me["rank"] - 2):me["rank"] + 1]
        st.markdown(f"**Next to {name} in the overall ranking** (the player just above and just below)")
        board(near, "rating", "Average rating", "{:.1f}", ranks=near["rank"].tolist(), highlight=name)
        game_rank = w.groupby("game_uid").rating.rank(ascending=False, method="min").astype(int)
        pg = w[w.player == name].assign(game_rank=game_rank).sort_values("date")
        gt = pd.DataFrame({"Date": pg.date, "Opponent": pg.opponent, "Champion": pg.champion,
                           "Result": pg.result.map({1: "Won", 0: "Lost"}),
                           "K / D / A": [f"{int(k)} / {int(d_)} / {int(a)}" for k, d_, a in zip(pg.Kills, pg.Deaths, pg.Assists)],
                           "Expected": pg.expected_rating, "Rating": pg.rating,
                           "Rank in game": [f"{r} of 10" for r in pg.game_rank]})
        st.dataframe(with_icons(gt, "Champion").style.format({"Rating": "{:.1f}", "Expected": "{:.1f}"}),
                     column_config=image_column(), hide_index=True, use_container_width=True)
        hist = rt[rt.player == name].groupby("champion").agg(games=("game_uid", "nunique"), rating=("rating", "mean"),
                                                             wins=("result", "sum")).reset_index()
        hist = hist.sort_values("games", ascending=False).head(8)
        l, r = st.columns(2, gap="large")
        with l:
            st.markdown(f"**{name}'s most-played champions this season**")
            ht = pd.DataFrame({"Champion": hist.champion, "Games": hist.games,
                               "Record": [f"{int(w_)}–{int(n - w_)}" for w_, n in zip(hist.wins, hist.games)],
                               "Average rating": hist.rating})
            st.dataframe(with_icons(ht, "Champion").style.format({"Average rating": "{:.1f}"}),
                         column_config=image_column(), hide_index=True, use_container_width=True)
        with r:
            from lolpatch.ratings import ROLE_WEIGHTS, STAT_LABELS
            role = pg.role.iloc[0]
            prof = pd.Series({STAT_LABELS[s]: pg[f"z_{s}"].mean() for s in ROLE_WEIGHTS[role]
                              if f"z_{s}" in pg}).sort_values()
            st.markdown(f"**Strengths and weaknesses at Worlds**, compared with a typical pro {role}")
            fig = go.Figure(go.Bar(x=prof.values, y=prof.index, orientation="h",
                                   marker=dict(color=[BUFF if v >= 0 else NERF for v in prof.values], cornerradius=3),
                                   hovertemplate="%{y}: %{customdata}<extra></extra>",
                                   customdata=["much better than most" if v >= 1 else "better than most" if v >= 0.3
                                               else "about average" if v > -0.3 else "worse than most" if v > -1
                                               else "much worse than most" for v in prof.values]))
            fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
            st.plotly_chart(row_layout(style(fig, xaxis=dict(title="← worse than a typical pro · better →", range=[-2.5, 2.5],
                                                             showticklabels=False),
                                             hovermode="closest"), prof.index, right=24), use_container_width=True)

    with st.expander("How ratings and expectations work"):
        st.markdown(
            "**Rating (out of 5):** how well a player did in a game, compared with every other pro in the same "
            "role this year. It looks at damage, gold, kill participation, deaths, vision and more, including "
            "how much of their team's and the whole game's damage and gold they had, and how they did against "
            "their lane opponent. Supports are judged more on vision and helping teammates, carries more on "
            "damage. Winning when your team wasn't expected to earns extra credit. 3.0 is an average pro game.\n\n"
            "**Expected rating:** what we'd have guessed before the game, from how the player has been playing, "
            "how they usually do on that champion, how likely their team was to win, and (at international "
            "events) how strong their region has been against other regions.\n\n"
            "**Rating in their league:** their average in league games this year. Players in weaker leagues face "
            "weaker opponents, so a big league number doesn't always carry over; the expected rating accounts "
            "for that.")


def likely_bans(up: dict, team: str, opponent: str) -> pd.Series:
    """Chance `team` bans each champion: half its own habits, half what opponents ban against `opponent`."""
    b = up["bans"]
    own = b[b.team == team].set_index("champion").own_rate
    vs = b[b.team == opponent].set_index("champion").against_rate
    p = (0.5 * own.reindex(own.index.union(vs.index)).fillna(0)
         + 0.5 * vs.reindex(own.index.union(vs.index)).fillna(0)).clip(upper=0.95)
    return p.sort_values(ascending=False)


def likely_picks(up: dict, team: str, banned: pd.Series, n: int = 2) -> dict:
    """Per role: the team's most likely game-1 picks, from its recent pool and the Worlds meta,
    minus champions likely to be banned."""
    pools, meta = up["pools"], up["meta"]
    out = {}
    for role in ROLES:
        own = pools[(pools.team == team) & (pools.role == role)].set_index("champion").share
        m = meta[meta.role == role].set_index("champion").share
        idx = own.index.union(m.index)
        score = 0.8 * own.reindex(idx).fillna(0) + 0.2 * m.reindex(idx).fillna(0)
        score = score * (1 - banned.reindex(idx).fillna(0))
        score = score / score.sum() if score.sum() else score
        out[role] = score.sort_values(ascending=False).head(n)
    return out


def worlds_form(mx: dict) -> pd.DataFrame:
    """Per team at Worlds: games, wins, wins the model expected before each game, average gold lead."""
    e = mx["expected"].merge(mx["games"][["game_uid", "event"]], on="game_uid")
    e = e[e.event == "worlds"]
    sides = mx["teams"][["game_uid", "team", "side"]]
    lead = mx["gold"].groupby("game_uid").blue_lead.mean()
    e = e.merge(sides, on=["game_uid", "team"], how="left")
    e["lead"] = e.game_uid.map(lead) * np.where(e.side == "blue", 1, -1)
    f = e.groupby("team").agg(games=("won", "size"), wins=("won", "sum"), expected=("p_win", "sum"),
                              lead=("lead", "mean"))
    f["vs_expected"] = f.wins - f.expected
    return f


def form_label(gap: float) -> str:
    """Wins above or below expectation, in words."""
    if abs(gap) < 0.3:
        return "As expected"
    return f"{'Above' if gap > 0 else 'Below'} ({gap:+.1f} wins)"


def form_text(f: pd.DataFrame, team: str) -> str:
    if team not in f.index:
        return f"<b>{team}</b>: no Worlds games yet"
    r = f.loc[team]
    gap = r.vs_expected
    vs = "about as expected" if abs(gap) < 0.3 else f"{abs(gap):.1f} {'more' if gap > 0 else 'fewer'} wins than expected"
    lead = "" if pd.isna(r.lead) else f", average gold lead {r.lead:+,.0f}"
    return f"<b>{team}</b>: {int(r.wins)}–{int(r.games - r.wins)} in games, {vs}{lead}"


def match_preview(up: dict, a: str, b: str, best_of: int):
    p_game = 1 / (1 + np.exp(-(up["strength"].get(a, 0) - up["strength"].get(b, 0))))
    from lolpatch.upcoming import series_prob
    p_series = series_prob(p_game, best_of) if best_of > 1 else p_game
    fav, fav_p = (a, p_series) if p_series >= 0.5 else (b, 1 - p_series)

    c1, c2, c3 = st.columns([5, 3, 5], vertical_alignment="center")
    for col, tm, p in ((c1, a, p_series), (c3, b, 1 - p_series)):
        with col:
            lc, nc = st.columns([1, 4], vertical_alignment="center")
            if team_logo(tm):
                lc.image(team_logo(tm), width=56)
            nc.markdown(f"### {tm}\n**{p:.0%}** chance to win")
    c2.markdown(f"<div style='text-align:center'><span class='muted'>Best of {best_of}</span><br>"
                f"<b>{fav}</b> favored</div>", unsafe_allow_html=True)
    st.progress(float(p_series), text=f"{a} {p_series:.0%} · {b} {1 - p_series:.0%}")
    st.caption(f"Based on how both teams have played all year, Worlds included, and how decisively they won or "
               f"lost, {fav} would win a single game about {max(p_game, 1 - p_game):.0%} of the time, which makes "
               f"them {fav_p:.0%} to win a best of {best_of}.")
    if mx is not None:
        f = worlds_form(mx)
        st.markdown(f"<div class='muted'>Form at Worlds: {form_text(f, a)} · {form_text(f, b)}</div>",
                    unsafe_allow_html=True)

    bans_a, bans_b = likely_bans(up, a, b), likely_bans(up, b, a)
    banned = 1 - (1 - bans_a.reindex(bans_a.index.union(bans_b.index)).fillna(0)) * \
        (1 - bans_b.reindex(bans_a.index.union(bans_b.index)).fillna(0))
    l, r = st.columns(2, gap="large")
    for col, tm, bans in ((l, a, bans_a), (r, b, bans_b)):
        with col:
            st.markdown(f"**{tm}: likely bans**")
            top = bans.head(5)
            st.image([champ_icon(c) for c in top.index if champ_icon(c)], width=44,
                     caption=[f"{nice(c)} {p:.0%}" for c, p in top.items() if champ_icon(c)])
            st.markdown(f"**{tm}: likely picks** (game 1)")
            picks = likely_picks(up, tm, banned)
            rows = [{"Role": role.title(), "Most likely": nice(ch.index[0]) if len(ch) else "—",
                     "Chance": ch.iloc[0] if len(ch) else np.nan,
                     "Or": nice(ch.index[1]) if len(ch) > 1 else "—",
                     "Chance ": ch.iloc[1] if len(ch) > 1 else np.nan} for role, ch in picks.items()]
            t = pd.DataFrame(rows)
            t.insert(1, " ", [champ_icon(x) if x != "—" else None for x in t["Most likely"]])
            st.dataframe(t.style.format({"Chance": "{:.0%}", "Chance ": "{:.0%}"}, na_rep="—"),
                         column_config=image_column(" "), hide_index=True, use_container_width=True)

    if mx is not None:
        rt = mx["ratings"]
        st.markdown("**Players to watch**")
        l, r = st.columns(2, gap="large")
        for col, tm in ((l, a), (r, b)):
            tr = rt[rt.team == tm]
            if tr.empty:
                col.caption(f"No game data for {tm}.")
                continue
            last = tr[tr.date == tr.date.max()].drop_duplicates("role")
            roster = last.assign(order=last.role.map({x: i for i, x in enumerate(ROLES)})).sort_values("order")
            w = rt[(rt.event == "worlds")].groupby("player").rating.mean()
            season = rt[rt.event.isna()].groupby("player").rating.mean()
            col.dataframe(pd.DataFrame({
                "Role": roster.role.str.title(), "Player": roster.player,
                "Rating at Worlds": roster.player.map(w), "Rating in their league": roster.player.map(season),
            }).style.format({"Rating at Worlds": "{:.1f}", "Rating in their league": "{:.1f}"}, na_rep="—"),
                hide_index=True, use_container_width=True)
        g = mx["games"]
        h2h = g[((g.blue == a) & (g.red == b)) | ((g.blue == b) & (g.red == a))]
        if len(h2h):
            wins_a = int((h2h.winner == a).sum())
            last = h2h.sort_values("date").iloc[-1]
            st.markdown(f"**Head to head this year:** {a} {wins_a}–{len(h2h) - wins_a} {b} in games "
                        f"(last met {last.date}, {last.winner} won).")
        else:
            st.markdown("**Head to head this year:** they haven't played each other yet.")
    st.caption("Worlds uses Fearless Draft: a champion used in one game of a series can't be picked again in "
               "that series, so the picks above are for game 1. Bans: half what the team usually bans, half what "
               "teams usually ban against its opponent. Picks: mostly the team's own recent champions, nudged by "
               "what's popular at Worlds, minus likely bans.")


def swiss_bracket_view(games: pd.DataFrame, sched: pd.DataFrame):
    """The Swiss stage as a bracket: one column per round, one box per record group."""
    from lolpatch.upcoming import swiss_bracket
    b = swiss_bracket(games, sched)
    if b.empty:
        return
    records = b.attrs["records"]
    teams = sorted(set(b.team_a) | set(b.team_b))
    short = lambda t: TEAM_SHORT.get(t, t)
    full = lambda t: t.replace("Anyone s", "Anyone's")
    # Each logo is embedded once as a CSS class, not once per appearance.
    logo_css = "".join(f".sw-l{i}{{background-image:url({team_logo(t, 40)})}}" for i, t in enumerate(teams)
                       if team_logo(t, 40))
    cls = {t: f"sw-l{i}" for i, t in enumerate(teams)}

    def team_line(t, score, state):
        sc = "" if score is None or pd.isna(score) else f"<span class='sw-sc'>{int(score)}</span>"
        return (f"<div class='sw-t {state}' title='{full(t)}'><span class='sw-logo {cls[t]}'></span>"
                f"<span class='sw-n'>{short(t)}</span>{sc}</div>")

    def match_html(m):
        if m.pending:
            when = pd.to_datetime(m.date).strftime("%b %-d")
            return (f"<div class='sw-m pend'>{team_line(m.team_a, None, '')}{team_line(m.team_b, None, '')}"
                    f"<div class='sw-when'>{when}</div></div>")
        st_a = "win" if m.winner == m.team_a else "loss"
        st_b = "win" if m.winner == m.team_b else "loss"
        return f"<div class='sw-m'>{team_line(m.team_a, m.score_a, st_a)}{team_line(m.team_b, m.score_b, st_b)}</div>"

    def pool_box(rnd, pool):
        ms = b[(b["round"] == rnd) & (b.pool == pool)]
        if ms.empty:
            return ""
        return f"<div class='sw-pool'><div class='sw-h'>{pool}</div>{''.join(match_html(m) for m in ms.itertuples())}</div>"

    def final_box(w, l):
        ts = [t for t, r in records.items() if r == (w, l)]
        if not ts:
            return ""
        kind, label = ("adv", "Advanced") if w == 3 else ("out", "Eliminated")
        lines = "".join(team_line(t, None, "") for t in ts)
        return f"<div class='sw-pool {kind}'><div class='sw-h'>{w}–{l} · {label}</div>{lines}</div>"

    cols = [
        ("Round 1", [pool_box(1, "0–0")]),
        ("Round 2", [pool_box(2, "1–0"), pool_box(2, "0–1")]),
        ("Round 3", [pool_box(3, "2–0"), pool_box(3, "1–1"), pool_box(3, "0–2")]),
        ("Round 4", [final_box(3, 0), pool_box(4, "2–1"), pool_box(4, "1–2"), final_box(0, 3)]),
        ("Round 5", [final_box(3, 1), pool_box(5, "2–2"), final_box(1, 3)]),
        ("Final records", [final_box(3, 2), final_box(2, 3)]),
    ]
    body = "".join(f"<div class='sw-col'><div class='sw-r'>{name}</div>{''.join(boxes) or '<div class=sw-tbd>To come</div>'}</div>"
                   for name, boxes in cols)
    st.markdown(f"""<style>{logo_css}
.sw-wrap{{overflow-x:auto;padding-bottom:.4rem}}
.sw-grid{{display:grid;grid-template-columns:repeat(6,minmax(128px,1fr));gap:12px;min-width:820px}}
.sw-col{{display:flex;flex-direction:column;justify-content:center;gap:10px}}
.sw-r{{font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;opacity:.6;text-align:center}}
.sw-pool{{border:1px solid rgba(128,128,128,.28);border-radius:8px;padding:6px 6px 4px}}
.sw-pool.adv{{border-color:#1baf7a;background:rgba(27,175,122,.08)}}
.sw-pool.out{{border-color:#e34948;background:rgba(227,73,72,.07)}}
.sw-h{{font-size:.75rem;font-weight:600;opacity:.7;margin:0 0 4px 2px}}
.sw-m{{border-radius:5px;background:rgba(128,128,128,.08);margin-bottom:5px;padding:2px 0}}
.sw-m.pend{{border:1px dashed rgba(128,128,128,.45);background:none}}
.sw-t{{display:flex;align-items:center;gap:6px;padding:2px 6px;font-size:.85rem;line-height:1.35}}
.sw-t.win{{font-weight:700}}
.sw-t.loss{{opacity:.5}}
.sw-logo{{width:16px;height:16px;flex:none;background-size:contain;background-repeat:no-repeat;background-position:center}}
.sw-n{{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.sw-sc{{font-variant-numeric:tabular-nums}}
.sw-when{{font-size:.7rem;opacity:.6;padding:0 6px 2px}}
.sw-tbd{{font-size:.8rem;opacity:.45;text-align:center;border:1px dashed rgba(128,128,128,.3);border-radius:8px;padding:14px 4px}}
</style><div class='sw-wrap'><div class='sw-grid'>{body}</div></div>""", unsafe_allow_html=True)
    st.caption("Teams with the same record play each other each round. Three wins sends a team to the knockout "
               "stage; three losses sends it home. Bold = series winner. Dashed boxes are scheduled matches "
               "(dates in UTC). Hover a team for its full name.")


def worlds_upcoming_tab(up: dict):
    sched = up["schedule"]
    from lolpatch.upcoming import series_prob
    if len(sched):
        st.markdown("### Next matches")
        rows = []
        for m in sched.itertuples():
            p = 1 / (1 + np.exp(-(up["strength"].get(m.team_a, 0) - up["strength"].get(m.team_b, 0))))
            ps = series_prob(p, int(m.best_of))
            fav, fp = (m.team_a, ps) if ps >= 0.5 else (m.team_b, 1 - ps)
            rows.append({"When (UTC)": m.date, "Stage": m.stage, " ": team_logo(m.team_a), "Team": m.team_a,
                         "vs": "vs", "  ": team_logo(m.team_b), "Opponent": m.team_b, "Format": f"Bo{int(m.best_of)}",
                         "Favorite": fav, "Chance": fp})
        picked = st.dataframe(pd.DataFrame(rows).style.format({"Chance": "{:.0%}"}),
                              column_config={" ": st.column_config.ImageColumn(" ", width="small"),
                                             "  ": st.column_config.ImageColumn("  ", width="small")},
                              hide_index=True, use_container_width=True, on_select="rerun",
                              selection_mode="single-row", key="upcoming_table")
        sel = (picked or {}).get("selection", {}).get("rows", [])
        st.caption("Click a match for the full preview: likely bans and picks, players to watch, head to head.")
        if sel:
            m = sched.iloc[sel[0]]
            st.divider()
            match_preview(up, m.team_a, m.team_b, int(m.best_of))

    stn = up["standings"].copy()
    if "p_title" in stn:
        st.markdown("### Who will win Worlds?")
        odds = stn[stn.p_title >= 0.005].sort_values("p_title")
        fig = go.Figure(go.Bar(x=odds.p_title, y=odds.team, orientation="h",
                               marker=dict(color=BUFF, cornerradius=4),
                               text=[f"{p:.0%}" if p >= 0.01 else "<1%" for p in odds.p_title], textposition="outside",
                               cliponaxis=False, hovertemplate="<b>%{y}</b>: %{x:.1%} chance to win Worlds<extra></extra>"))
        fig = style(fig, xaxis=dict(title="Chance to win Worlds", range=[0, odds.p_title.max() * 1.2], **PCT),
                    hovermode="closest", showlegend=False, bargap=0.35)
        st.plotly_chart(row_layout(fig, odds.team, right=48), use_container_width=True)
        st.caption("From playing out the rest of Worlds 20,000 times with each team's win chances: the remaining "
                   "Swiss rounds, then best-of-five quarterfinals, semifinals and final. The real quarterfinal "
                   "draw isn't known yet, so it's simulated with the usual rules (best records face the worst, "
                   "teams from the same region kept apart when possible).")

    st.markdown("### Swiss standings")
    stn["Logo"] = stn.team.map(team_logo)
    stn["Record"] = stn.wins.astype(str) + "–" + stn.losses.astype(str)
    stn["Reach knockouts"] = np.where(stn.status == "Advanced", 1.0,
                                      np.where(stn.status == "Eliminated", 0.0, stn.p_advance))
    # Progress bars label with printf formats, so show these as 0–100 rather than 0–1.
    pct = lambda label: st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=100)
    cols = ["Logo", "team", "Record", "status"]
    cfg = {"Logo": st.column_config.ImageColumn(" ", width="small")}
    if mx is not None:
        f = worlds_form(mx)
        stn["Form"] = [form_label(f.vs_expected.get(t, 0.0)) for t in stn.team]
        stn["Avg gold lead"] = stn.team.map(f.lead)
        cols += ["Form", "Avg gold lead"]
        cfg |= {"Avg gold lead": st.column_config.NumberColumn("Avg gold lead", format="%+.0f")}
    cols += ["Reach knockouts"]
    cfg |= {"Reach knockouts": pct("Reach knockouts")}
    if "p_title" in stn:
        stn = stn.rename(columns={"p_semis": "Reach semifinals", "p_final": "Reach final", "p_title": "Win Worlds"})
        cols += ["Reach semifinals", "Reach final", "Win Worlds"]
        cfg |= {c: pct(c) for c in ["Reach semifinals", "Reach final", "Win Worlds"]}
    for c in cols[cols.index("Reach knockouts"):]:
        stn[c] = stn[c] * 100
    st.dataframe(stn[cols].rename(columns={"team": "Team", "status": "Status"}), column_config=cfg,
                 hide_index=True, use_container_width=True, height=36 * len(stn) + 40)
    st.caption("Teams advance at 3 wins and go home at 3 losses. Form compares each team's game wins at Worlds with "
               "what we expected before each game; average gold lead shows how far ahead or behind they've been, win "
               "or lose. The bars show each team's chances from the same 20,000 simulated tournaments.")

    with st.expander("Preview any matchup"):
        alive = up["standings"].team.tolist()
        c1, c2, c3 = st.columns([3, 3, 2])
        a = c1.selectbox("Team", alive, index=0, key="pv_a")
        b = c2.selectbox("Opponent", [t for t in alive if t != a], index=0, key="pv_b")
        bo = c3.selectbox("Format", [1, 3, 5], index=1, format_func=lambda x: f"Best of {x}", key="pv_bo")
        match_preview(up, a, b, bo)

    if mx is not None:
        e = mx["expected"].merge(mx["games"][["game_uid", "event"]], on="game_uid")
        w = e[(e.event == "worlds") & (e.p_win > 0.5)]
        if len(w):
            st.caption(f"Track record: at Worlds so far, the team we favored before the game won "
                       f"{int(w.won.sum())} of {len(w)} games ({w.won.mean():.0%}).")


d = load()
stats, changes, lines = d["stats"], d["changes"], d["lines"]
ev, rg, rp, dr, mx, up = load_events(), load_regions(), load_ripple(), load_draft(), load_matches(), load_upcoming()
all_stats = stats[stats.league == "ALL"]

# --- navigation -------------------------------------------------------------------------

PAGES = ["Overview", "Worlds 2026", "Worlds Players", "Champions & Patches", "Meta Shifts", "Draft Answers", "Teams & Regions",
         "Model Accuracy"]
# The changelog itself lives in the repository (CHANGELOG.md); the site only shows the version.
CHANGELOG = PROCESSED.parent.parent / "CHANGELOG.md"
changelog_text = CHANGELOG.read_text() if CHANGELOG.exists() else ""
version_tag = next((line[3:].split("·")[0].strip() for line in changelog_text.splitlines()
                    if line.startswith("## v")), "")
with st.sidebar:
    st.markdown("## Patch Impact")
    st.caption("How balance changes move pro picks and bans · 2026 season")
    page = st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.caption(f"{' · '.join(LEAGUES)} and international events.\n\n**Pick/ban rate** = the share of games where a "
               "champion was picked or banned. Changes are shown in **points**: going from 40% to 44% is +4 points.")
    if version_tag:
        st.caption(f"Version {version_tag}")


# --- Overview --------------------------------------------------------------------------

if page == "Overview":
    st.title("Patch Impact")
    lead("What Riot's buffs and nerfs actually did to pro drafts in 2026, and what's happening at Worlds right now.")

    cards = []
    if ev and "worlds" in ev["metrics"]:
        wm = ev["metrics"]["worlds"]
        top = ev["worlds_current"].nlargest(5, "actual_presence")
        names = ", ".join(f"**{nice(r.champion)}** {r.actual_presence:.0%}" for r in top.itertuples())
        text = (f"{wm['current']['games']} games played so far, all on patch {wm['patch']}. "
                f"Picked or banned the most (share of games): {names}.")
        if mx is not None:
            wr_ = mx["ratings"][mx["ratings"].event == "worlds"]
            lb = wr_.groupby(["player", "team"]).agg(n=("game_uid", "nunique"), r=("rating", "mean")).reset_index()
            lb = lb[lb.n >= 2].sort_values("r", ascending=False)
            if len(lb):
                text += f" Best player so far: **{lb.iloc[0].player}** ({lb.iloc[0].team}), {lb.iloc[0].r:.1f} out of 5."
        cards.append(("Worlds 2026", text, "Worlds 2026"))
    tr = transitions_for("ALL")
    eff = verdict_effects(tr)
    if {"buff", "nerf"} <= set(eff.index):
        hi_nerf = tr[tr.changed & (tr.verdict == "nerf") & (tr.prev_presence > 0.6)]
        cards.append(("Do Buffs and Nerfs Work?",
                      f"Yes, but modestly. After a buff, pros pick or ban a champion about "
                      f"**{eff.loc['buff', 'mean_delta_presence'] * 100:.0f} points more often**; after a nerf, about "
                      f"**{-eff.loc['nerf', 'mean_delta_presence'] * 100:.0f} points less**. Nerfs hit the most popular "
                      f"champions hardest: ones picked or banned in over 60% of games dropped "
                      f"**{-hi_nerf.delta_presence.mean() * 100:.0f} points** on average.", "Champions & Patches"))
    if dr is not None:
        solid = dr["lanes"][dr["lanes"].solid].sort_values("n", ascending=False)
        ex = solid.iloc[0]
        cards.append(("Draft Answers",
                      f"Teams have go-to responses. When one team locks in **{nice(ex.opener)}** ({ex.role}), the "
                      f"other team answers with **{nice(ex.answer)}** {ex.lift:.1f}× more often than usual. We found "
                      f"**{len(solid)}** answers like this.",
                      "Draft Answers"))
    if ev:
        won = [m["title"].replace(" 2026", "") for m in ev["metrics"].values()
               if m["frozen"]["all"]["mae_model"] < m["frozen"]["all"]["mae_baseline"]]
        lost = [m["title"].replace(" 2026", "") for m in ev["metrics"].values()
                if m["frozen"]["all"]["mae_model"] >= m["frozen"]["all"]["mae_baseline"]]
        text = ("Before each international event, we predicted how often every champion would be picked or banned, "
                "using league games plus the patch notes since, then checked what really happened. "
                f"Reading the patch notes helped a little: our prediction was closer than simply assuming "
                f"“nothing changes” at **{len(won)} of {len(ev['metrics'])}** events ({', '.join(won)}), "
                "but only by small margins.")
        if lost:
            text += f" It did worse at {' and '.join(lost)}"
            wf = ev.get("worlds_frozen")
            if "Worlds" in lost and wf is not None:
                stubborn = wf[wf.verdict == "nerf"].nlargest(1, "actual_presence")
                if len(stubborn):
                    text += (f", where pros kept picking champions that had just been nerfed, like "
                             f"**{nice(stubborn.iloc[0].champion)}** ({stubborn.iloc[0].actual_presence:.0%} of games)")
            text += "."
        cards.append(("Can Patch Notes Predict the Meta?", text, "Model Accuracy"))

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
    st.caption("Also: **Meta Shifts** shows who lost ground as a group of champions rose; **Teams & Regions** shows "
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

    if mx is not None and up is not None:
        st.markdown("### Swiss stage")
        swiss_bracket_view(mx["games"], up["schedule"])

    if "worlds_live" in ev:
        with st.expander("Most Picked or Banned Champions"):
            t_top, t_all, t_how = st.tabs(["Top 25", "All Champions", "How This Works"])
            with t_top:
                lv = ev["worlds_live"].head(25).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=lv.live_presence, y=lv.champion, orientation="h",
                    marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in lv.verdict], cornerradius=4),
                    error_x=dict(type="data", symmetric=False, array=lv.live_high - lv.live_presence,
                                 arrayminus=lv.live_presence - lv.live_low, color=NEUTRAL, thickness=1.2, width=3),
                    customdata=lv[["live_low", "live_high", "actual_presence", "picks", "bans", "games", "pred_presence"]],
                    hovertemplate="<b>%{y}</b>: expected in %{x:.0%} of the remaining games "
                                  "(likely %{customdata[0]:.0%}–%{customdata[1]:.0%})"
                                  "<br>so far: %{customdata[3]} picks + %{customdata[4]} bans in %{customdata[5]} games"
                                  "<extra></extra>"))
                fig = style(fig, xaxis=dict(title="How often we expect it to be picked or banned for the rest of Worlds",
                                            range=[0, 1], **PCT),
                            hovermode="closest", showlegend=False, bargap=0.35)
                st.plotly_chart(row_layout(fig, lv.champion), use_container_width=True)
                st.caption("The top 25. Blue = buffed since the leagues ended, red = nerfed, gray = unchanged. "
                           "The thin line shows the likely range.")
            with t_all:
                allc = ev["worlds_live"]
                allc = allc[(allc.picks + allc.bans) > 0].sort_values("actual_presence", ascending=False)
                wins_ = ev["worlds_current"].set_index("champ_key").wins
                at = pd.DataFrame({
                    "Champion": allc.champion, "Picked or banned": allc.actual_presence, "Picks": allc.picks,
                    "Bans": allc.bans,
                    "Won–lost when picked": [f"{int(wins_.get(k, 0))}–{int(p - wins_.get(k, 0))}" if p else "—"
                                             for k, p in zip(allc.champ_key, allc.picks)],
                })
                st.dataframe(with_icons(at, "Champion").style.format({"Picked or banned": "{:.0%}"}),
                             column_config=image_column(), hide_index=True, use_container_width=True,
                             height=min(600, 36 * len(at) + 40))
            with t_how:
                st.markdown(
                    "We start from our pre-Worlds prediction (based on league games and patch notes) and shift it toward "
                    "what teams are actually doing as games are played. Early on, the prediction matters more; the more "
                    f"games are played, the more the real results take over (the prediction counts like {PRIOR_GAMES} "
                    "games of evidence). The thin line is the range we're 80% confident in.")

    t_up, t_games, t_teams, t_patch, t_fc, t_draft, t_region = st.tabs(
        ["Upcoming", "Series & Games", "Teams", f"Patch {wm['patch']} Changes", "Forecast Check", "Draft Answers",
         "By Region"])

    with t_up:
        if up is None:
            st.info("Run `python -m lolpatch upcoming` to load upcoming matches.")
        else:
            worlds_upcoming_tab(up)

    with t_games:
        if mx is None:
            st.info("Run `python -m lolpatch matches` to load game details.")
        else:
            worlds_series_tab(mx)

    wg = ev["games"][ev["games"].league == "worlds"]
    with t_teams:
        series = wg.groupby(["series_id", "team"]).result.sum().reset_index()
        series["won"] = series.groupby("series_id").result.transform(lambda s: (s == s.max()) & (s.max() > s.min()))
        picks = wg.assign(champion=wg.picks.str.split("|")).explode("champion")
        top3 = picks.groupby("team").champion.agg(lambda s: ", ".join(map(nice, s.value_counts().head(3).index)))
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
                      format_func=lambda p: f"{p} (Worlds Patch)" if p == wm["patch"] else p)
        pc = changes[changes.patch == bp].copy()
        pc["before"] = pc.champ_key.map(wcmp.domestic_presence).fillna(0)
        pc["worlds"] = pc.champ_key.map(wcmp.actual_presence).fillna(0)
        picks_ = pc.champ_key.map(wcmp.picks).fillna(0).astype(int)
        wins_ = pc.champ_key.map(wcmp.wins).fillna(0).astype(int)
        pc["record"] = [f"{w}–{p - w}" if p else "—" for w, p in zip(wins_, picks_)]
        pc = pc.sort_values("worlds", ascending=False)
        t = pc[["champion", "verdict", "before", "worlds", "record"]].rename(columns={
            "champion": "Champion", "verdict": "Change",
            "before": f"Picked or banned in the leagues ({sc['base_patch']})",
            "worlds": "Picked or banned at Worlds", "record": "Won–lost when picked"})
        table(with_icons(t, "Champion"), {f"Picked or banned in the leagues ({sc['base_patch']})": "{:.0%}",
                                          "Picked or banned at Worlds": "{:.0%}"}, images=[" "], n=20)

    with t_fc:
        scf = wm["frozen"]
        st.markdown(f"Before Worlds started, we predicted how often each champion would be picked or banned on "
                    f"{wm['patch']}, using league games on {scf['base_patch']} plus the patch notes since. "
                    "Here's how that compares with what's actually happened.")
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
            st.markdown("When one team picks a champion, what the other team answers with in the same lane, "
                        "for answers seen at least twice at Worlds.")
            t = pd.DataFrame({"Role": wp.role.str.title(), "Their pick": wp.opener, "Answer": wp.answer,
                              "At Worlds": [f"{n} of {m} times" for n, m in zip(wp.n, wp.opener_n)],
                              "Before Worlds": wp.rest_rate,
                              "Answer won–lost": [f"{x}–{n - x}" for x, n in zip(wp.won, wp.n)]})
            table(with_icons(with_icons(t, "Their pick", label=" "), "Answer", label="  "),
                  {"Before Worlds": "{:.0%}"}, images=[" ", "  "])

    with t_region:
        if rg is not None:
            wr = rg["events"][rg["events"].event == "worlds"]
            reg_games = wr.drop_duplicates("region").set_index("region").region_games
            top = wr.groupby("champion").pick_ban_rate.max().nlargest(30).index
            z = wr[wr.champion.isin(top)].pivot(index="champion", columns="region", values="pick_ban_rate").reindex(top).fillna(0)
            z.columns = [f"{c} ({int(reg_games[c])})" for c in z.columns]
            reg_cols = list(z.columns)
            z = with_icons(z.rename_axis("Champion").reset_index(), "Champion")
            st.markdown("How often each region's teams pick or ban each champion (games in brackets).")
            st.dataframe(z.style.format("{:.0%}", subset=reg_cols)
                         .background_gradient(cmap="Blues", vmin=0, vmax=1, subset=reg_cols),
                         column_config=image_column(), hide_index=True, use_container_width=True)


# --- Worlds Players ----------------------------------------------------------------------

elif page == "Worlds Players":
    st.title("Worlds Players")
    lead("Who's playing best at Worlds, and who's beating or missing what their form predicted.")
    if mx is None:
        st.info("Run `python -m lolpatch matches` first.")
        st.stop()
    worlds_players_page(mx)


# --- Champions & patches ----------------------------------------------------------------

elif page == "Champions & Patches":
    st.title("Champions & Patches")
    lead("Follow any champion through the season, see what each patch did, and check whether buffs and nerfs work.")
    league = st.selectbox("League", ["ALL"] + LEAGUES, format_func=lambda x: "All Leagues" if x == "ALL" else x,
                          key="league")
    scope = stats[stats.league == league]
    trans = transitions_for(league)
    patches = sorted(scope.patch.unique(), key=patch_key)

    t_champ, t_patch, t_effect, t_pred = st.tabs(["Champion", "Patch", "Do Buffs & Nerfs Work?", "Next Patch"])

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
        fig.add_trace(go.Scatter(x=patches, y=cs.presence, name="Pick/ban rate in league games", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2.5), marker=dict(size=8),
                                 hovertemplate="Picked or banned in %{y:.0%} of games<extra></extra>"))
        if ev_pts:
            ep = pd.DataFrame(ev_pts)
            fig.add_trace(go.Scatter(x=ep.label, y=ep.presence, name="International event", mode="markers",
                                     marker=dict(symbol="square", size=10, color="#52514e", line=dict(width=1.5, color="white")),
                                     customdata=ep[["games"]],
                                     hovertemplate="<b>%{x}</b><br>picked or banned in %{y:.0%} of %{customdata[0]} games<extra></extra>"))
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
        st.caption("How often the champion was picked or banned, patch by patch. ▲ buff · ▼ nerf · ◆ adjustment. "
                   "Squares are international events, placed on the patch they were played on.")
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
        st.markdown(f"Champions changed in **{', '.join(window) or '—'}**, and how often they were picked or banned "
                    f"before ({fp}) and after ({tp}).")
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
        fig = style(fig, xaxis=dict(title="Pick/ban rate", range=[-0.02, x_hi], **PCT), hovermode="closest")
        st.plotly_chart(row_layout(fig, ch.champion, legend=True), use_container_width=True)
        st.caption("Line color: blue = buffed, red = nerfed, gray = adjusted.")
        with st.expander("Biggest movers this patch, changed or not"):
            movers = t.reindex(t.delta_presence.abs().sort_values(ascending=False).index).head(25)
            mt = pd.DataFrame({"Champion": movers.champion, "Balance change": movers.verdict.replace("none", "—"),
                               "Pick/ban rate before": movers.prev_presence, "After": movers.next_presence,
                               "Change (points)": movers.delta_presence * 100})
            st.dataframe(with_icons(mt, "Champion").style.format(
                {"Pick/ban rate before": "{:.0%}", "After": "{:.0%}", "Change (points)": "{:+.0f}"}),
                column_config=image_column(), hide_index=True, use_container_width=True)

    with t_effect:
        eff = verdict_effects(trans).reindex(["buff", "nerf", "adjust", "none"]).dropna(how="all")
        st.markdown("On average, how much more or less often a champion was picked or banned on the next patch:")
        cols = st.columns(len(eff))
        for col, (v, row) in zip(cols, eff.iterrows()):
            label = {"buff": "After a buff", "nerf": "After a nerf", "adjust": "After an adjustment",
                     "none": "No change"}[v]
            col.metric(label, f"{row.mean_delta_presence * 100:+.1f} points")
        changed = trans[trans.changed]
        hi_nerf = changed[(changed.verdict == "nerf") & (changed.prev_presence > 0.6)]
        lo_buff = changed[(changed.verdict == "buff") & (changed.prev_presence < 0.05)]
        if len(hi_nerf) and len(lo_buff):
            st.markdown(f"<div class='takeaway'>Nerfs hit the most popular champions hardest: ones picked or banned "
                        f"in over 60% of games dropped <b>{-hi_nerf.delta_presence.mean() * 100:.0f} points</b> on "
                        f"average, and {(hi_nerf.delta_presence < 0).mean():.0%} of them fell. Buffs rarely rescue "
                        f"champions nobody plays: ones under 5% only gained "
                        f"<b>{lo_buff.delta_presence.mean() * 100:.0f} points</b>.</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for v in ["buff", "nerf", "adjust"]:
            sub = changed[changed.verdict == v]
            fig.add_trace(go.Scatter(
                x=sub.prev_presence, y=sub.delta_presence, mode="markers", name=v,
                marker=dict(color=VERDICT_COLOR[v], size=8, line=dict(width=1, color="white")),
                customdata=sub[["champion", "from_patch", "to_patch"]],
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}→%{customdata[2]}"
                              "<br>picked or banned in %{x:.0%} of games before, then %{y:+.0%} points<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, dash="dot"))
        st.plotly_chart(style(fig, 380, xaxis=dict(title="Pick/ban rate before the change", **PCT),
                              yaxis=dict(title="Change on the next patch (points)", tickformat="+.0%"),
                              hovermode="closest"),
                        use_container_width=True)
        st.caption("Each dot is one champion after one balance change. Above the line = picked or banned more often "
                   "afterwards, below = less.")

    with t_pred:
        if "predictions" not in d:
            st.info("Run `python -m lolpatch predict` first.")
        else:
            pr, meta = d["predictions"], d["meta"]
            st.markdown(f"Our prediction for **{meta.get('target_patch')}**: starting from league games on "
                        f"{meta.get('base_patch')} and applying every buff and nerf since. Champions that were changed, "
                        "or that we expect to move by 5+ points:")
            show = pr[(pr.n_changes > 0) | (pr.pred_delta_presence.abs() >= 0.05)].sort_values("pred_delta_presence")
            fig = go.Figure(go.Bar(
                x=show.pred_delta_presence, y=show.champion, orientation="h",
                marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in show.verdict], cornerradius=4),
                customdata=show[["prev_presence", "pred_presence", "verdict"]],
                hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>%{customdata[0]:.0%} → %{customdata[1]:.0%}<extra></extra>"))
            fig = style(fig, xaxis=dict(title="Predicted change in pick/ban rate (points)", tickformat="+.0%"),
                        hovermode="closest", bargap=0.3)
            st.plotly_chart(row_layout(fig, show.champion), use_container_width=True)
            st.caption("How good have these predictions been? See the **Model Accuracy** page.")


# --- Meta shifts -----------------------------------------------------------------------

elif page == "Meta Shifts":
    st.title("Meta Shifts")
    lead("When champions rise, someone else loses the slot. Pick a group and see who it pushed out.")
    if rp is None:
        st.info("Run `python -m lolpatch meta` first.")
        st.stop()
    t_group, t_slot = st.tabs(["Rise & Fall", "Who Took the Slot"])

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
                headline = ", ".join(map(nice, (pushed if len(pushed) else fell).champion.head(3)))
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
                                      yaxis=dict(title="Pick/ban rate", **PCT)), use_container_width=True)

                def lead_table(df):
                    t = pd.DataFrame({
                        "Champion": df.champion, "Role": df.role.fillna("—").str.title(),
                        "Picked or banned": [f"{b:.0%} → {a:.0%}" for b, a in zip(df.before, df.after)],
                        "Pushed out?": ["likely" if r <= -0.3 else "no, moves with them" if r >= 0.3 else "unclear"
                                        if pd.notna(r) else "—" for r in df.r],
                    })
                    return with_icons(t, "Champion")

                l, r = st.columns(2)
                with l:
                    st.markdown("**Fell**")
                    table(lead_table(fell), images=[" "], n=12)
                with r:
                    st.markdown("**Rose alongside**")
                    table(lead_table(disp[disp.change >= 0.05].sort_values("change", ascending=False)), images=[" "], n=12)
                with st.expander("How “pushed out” is decided"):
                    st.markdown(
                        "**Pushed out** means that, patch after patch, this champion went down whenever the group "
                        "went up. Champions that fell overall but usually moved *in the same direction* as the "
                        "group were dropping for other reasons, like their own nerfs. (Start and end values are "
                        "averaged over two patches so one odd patch doesn't skew them.)")

    with t_slot:
        st.markdown("After a big buff or nerf, which untouched champion in the same role moved the other way?")
        subs, pairs = rp["subs"], rp["pairs"]
        role_f = st.radio("Role", ["all"] + ROLES, horizontal=True, key="ripple_role",
                          format_func=lambda r: r.title() if r != "all" else "All Roles")
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
        st.caption(f"Shown when a changed champion's pick/ban rate moved by {SUB_MIN_MOVE_A * 100:.0f}+ points and "
                   f"an unchanged champion in the same role moved {SUB_MIN_MOVE_B * 100:.0f}+ points the other way "
                   "at the same time. The timing points to a swap, but doesn't prove it.")


# --- Draft answers ---------------------------------------------------------------------

elif page == "Draft Answers":
    st.title("Draft Answers")
    lead("When a team locks a champion, what does the opponent answer with in the same role?")
    if dr is None:
        st.info("Run `python -m lolpatch draft` first.")
        st.stop()
    solid = dr["lanes"][dr["lanes"].solid].copy()
    f1, f2 = st.columns([3, 2])
    role_d = f1.radio("Role", ["all"] + ROLES, horizontal=True, key="draft_role",
                      format_func=lambda r: r.title() if r != "all" else "All Roles")
    kinds = {"Everyone": None, "Mostly at Worlds": "Worlds", "One Region": "(region)", "One Team": "(team)",
             "One Player": "(player)"}
    kind = f2.selectbox("Who Does It", list(kinds), key="draft_kind")
    view = solid if role_d == "all" else solid[solid.role == role_d]
    if kinds[kind]:
        view = view[view.where_from.str.contains(kinds[kind], regex=False)]
    view = view.sort_values(["n", "lift"], ascending=False)
    if len(view):
        top = view.iloc[0]
        st.markdown(f"<div class='takeaway'>Most common: when a team picks <b>{nice(top.opener)}</b> ({top.role}), "
                    f"the other team answers with <b>{nice(top.answer)}</b>. That happened {top.n} times, "
                    f"{top.lift:.1f}× more often than {nice(top.answer)} usually gets picked.</div>",
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
                    dim.capitalize(): g.group, "Answered": [f"{a} of {o} times" for a, o in zip(g.answers, g.openers)],
                    "How often": g.rate, "Everyone else": g.rest_rate,
                }).style.format({"How often": "{:.0%}", "Everyone else": "{:.0%}"}, na_rep="—"),
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
            f"We know the order teams picked in for **{dm['games']:,} games**. In every lane, whichever champion "
            "was picked second was picked *knowing* what the opponent had, so we call it the answer.\n\n"
            "An answer is listed when it happened at least 4 times and at least 1.5× more often than that champion "
            "usually gets picked in that lane **on the same patches**. That way, two champions that were simply "
            "popular at the same time don't count. We also checked that it isn't a fluke, since with "
            f"{dm['lane_pairs_tested']:,} pairs tested, some would look special by luck.\n\n"
            "**Who Does It** shows an event, region, team or player when they make this answer at least twice as "
            "often as everyone else. It means they lead it, not that only they do it.")


# --- Teams & regions ---------------------------------------------------------------------

elif page == "Teams & Regions":
    st.title("Teams & Regions")
    lead("What each team and region leans on, and whether leaning on it wins games.")
    if rg is None:
        st.info("Run `python -m lolpatch regions` first.")
        st.stop()
    t_team, t_region, t_wins = st.tabs(["Team Explorer", "Region Preferences", "Does Priority Win?"])

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
            fig = style(fig, xaxis=dict(title="How much more often they pick it than other teams in their league",
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
                          format_func=lambda m: {"presence_lift": "Picked or Banned",
                                                 "early_lift": "Taken Early (First 2 Picks / 3 Bans)"}[m])
        rc = rg["regions"]
        champs = rc.groupby("champion").presence.mean().nlargest(40).index
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
        st.caption("Blue = this region picks or bans it more than the average league did on the same patches; "
                   "red = less. The 40 champions picked or banned most this season.")

    with t_wins:
        eff = rg["effects"]
        md, cf, fv = eff["meta_draft"], eff["comfort"], eff["region_favorites"]
        st.markdown("<div class='takeaway'>Mostly <b>no</b>. Better teams win more, of course. But once you take "
                    "that into account, picking the popular champions or a team's favorites doesn't win extra "
                    "games. Numbers below are win-rate differences in points.</div>", unsafe_allow_html=True)
        st.markdown("")
        rows = [
            ("Drafting the most-contested champions", f"{md['adjusted_win_prob_plus10pp']:.0%} win rate",
             "No edge: team strength explains the wins."),
            ("Sticking to champions a team has played a lot", f"{cf['within_team_diff'] * 100:+.0f} points",
             "No edge; if anything slightly worse."),
        ]
        if fv["regions"]:
            rows.append(("Regions playing their home favorites at internationals",
                         f"{fv['within_region_diff'] * 100:+.0f} points", "Too few games to tell."))
        for title, value, note in rows:
            with st.container(border=True):
                a, b = st.columns([3, 1], vertical_alignment="center")
                a.markdown(f"**{title}**  \n<span class='muted'>{note}</span>", unsafe_allow_html=True)
                b.markdown(f"### {value}")
        with st.expander("How these were tested"):
            st.markdown(
                f"- **Popular champions:** across {md['games']:,} league games, we compared teams that drafted "
                "champions that had been popular on the previous patch (what teams knew when drafting) with teams "
                "that didn't, after accounting for how strong each team is. The true effect is most likely between "
                f"{md['adjusted_ci'][0]:.0%} and {md['adjusted_ci'][1]:.0%} win rate, so about even.\n"
                f"- **Team favorites:** for each of {cf['teams']} teams, its win rate on champions it had already "
                "played 3+ times before that game, compared with its other picks. Most likely between "
                f"{cf['ci'][0] * 100:+.0f} and {cf['ci'][1] * 100:+.0f} points.\n"
                + (f"- **Home favorites:** {fv['favorite_picks']} picks at international events against other "
                   f"regions. Most likely between {fv['ci'][0] * 100:+.0f} and {fv['ci'][1] * 100:+.0f} points, "
                   "too wide to say." if fv["regions"] else ""))


# --- Model accuracy ----------------------------------------------------------------------

elif page == "Model Accuracy":
    st.title("Model Accuracy")
    lead("Before each international event, we predicted how often every champion would be picked or banned. "
         "Here's how close we got.")
    if not ev:
        st.info("Run `python -m lolpatch events` first.")
        st.stop()
    version = st.radio("Forecast", ["frozen", "current"], horizontal=True, key="fc_version",
                       format_func=lambda v: {"frozen": "Saved Before the Event",
                                              "current": "Rebuilt With Today's Model"}[v])
    rows = []
    for key, m in ev["metrics"].items():
        sc = m[version]
        rows.append({"Event": m["title"], "Patch": m["patch"], "Games": sc["games"],
                     "Our prediction was off by": sc["all"]["mae_model"],
                     "Guessing “no change” was off by": sc["all"]["mae_baseline"],
                     "Did we beat “no change”?": "Yes" if sc["all"]["mae_model"] < sc["all"]["mae_baseline"] else "No"})
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Our prediction was off by": lambda v: f"{v * 100:.1f} points",
         "Guessing “no change” was off by": lambda v: f"{v * 100:.1f} points"}),
                 hide_index=True, use_container_width=True)
    st.caption("“Off by” = on average, how far the predicted pick/ban rate was from the real one (lower is better). "
               "“No change” means assuming every champion stays exactly as popular as in the leagues; if reading "
               "the patch notes is worth anything, our prediction should beat it.")

    keys = list(ev["metrics"])
    pick = st.selectbox("Look at One Event", keys, index=keys.index("worlds") if "worlds" in keys else 0,
                        format_func=lambda k: ev["metrics"][k]["title"], key="fc_event")
    sc = ev["metrics"][pick][version]
    scorecard(sc)
    forecast_chart(ev[f"{pick}_{version}"], sc["base_patch"], sc["games"])

    with st.expander("How the forecast works, and what was tried"):
        st.markdown(
            "From a whole season of league games, the model learned how much more or less often pros pick or ban "
            "a champion after a buff, a nerf or an adjustment, and how very popular champions tend to cool off on "
            "their own. For each event, it starts from the last league patch before it and applies every balance "
            "change since.\n\n"
            "Predictions **saved before the event** were never edited afterwards, so they're a fair test. (For "
            "First Stand, MSI and EWC we built the prediction later, using only data from before each event; only "
            "Worlds and the Demacia Cup were saved in advance.)")
        if "adjustment" in ev:
            st.markdown(
                "**Tried and dropped:** assuming the most popular champions stay just as popular at international "
                "events. On the earlier events that made predictions worse: top picks usually lose *more* "
                "popularity at internationals than in the leagues.")
        if "backtest_metrics" in d:
            row = d["backtest_metrics"].set_index(d["backtest_metrics"].columns[0]).loc["changed champions"]
            st.markdown(f"**In the leagues** (predicting each patch using only earlier patches): for buffed or "
                        f"nerfed champions we were off by {row.mae_model * 100:.1f} points on average, vs "
                        f"{row.mae_persistence * 100:.1f} for “no change”, and we called whether a champion would "
                        f"rise or fall correctly {row.direction_acc_model:.0%} of the time.")


# --- footer ------------------------------------------------------------------------------
st.divider()
st.caption(
    "Data: [Games of Legends (gol.gg)](https://gol.gg) · Riot patch notes · Riot Data Dragon · Meraki Analytics. "
    "A fan project, not endorsed by Riot Games and not reflecting the views of Riot Games or anyone officially "
    "involved in producing or managing Riot Games properties. Riot Games and all associated properties are "
    "trademarks or registered trademarks of Riot Games, Inc.")

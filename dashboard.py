"""Streamlit dashboard: python -m lolpatch dashboard"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lolpatch.analyze import build_transitions, load_changes, verdict_effects
from lolpatch.config import LEAGUES, PROCESSED, patch_key
from lolpatch.events import PRIOR_GAMES
from lolpatch.meta import ROLES, SUB_MIN_MOVE_A, SUB_MIN_MOVE_B, displacement, group_presence, rise_window
from lolpatch.ui import champ_icon, image_column, row_layout, team_logo, with_icons

st.set_page_config(page_title="Patch Impact — LoL Pro 2026", layout="wide")

# Reference palette (dataviz skill): blue/red diverging poles, gray neutral, categorical order.
BUFF, NERF, NEUTRAL = "#2a78d6", "#e34948", "#898781"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
VERDICT_COLOR = {"buff": BUFF, "nerf": NERF, "adjust": NEUTRAL, "none": "#c3c2b7"}
PCT = dict(tickformat=".0%")


@st.cache_data
def load():
    stats = pd.read_csv(PROCESSED / "champion_patch_stats.csv", dtype={"patch": str})
    changes = load_changes()
    lines = pd.read_csv(PROCESSED / "patch_changes.csv", dtype={"patch": str})
    out = dict(stats=stats, changes=changes, lines=lines)
    for name in ["predictions", "backtest_predictions", "backtest_metrics", "feature_effects"]:
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


def style(fig, height=380, **layout):
    # Legend sits in the top margin, above the plot, so it never covers data.
    layout = dict(margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                  hovermode="x unified") | layout
    fig.update_layout(height=height, **layout)
    fig.update_xaxes(showgrid=False)
    return fig


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
    live = PROCESSED / "event_worlds_live.csv"
    if live.exists():
        out["worlds_live"] = pd.read_csv(live)
    adj = PROCESSED / "event_adjustment_test.csv"
    if adj.exists():
        out["adjustment"] = pd.read_csv(adj)
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


def pair_table(df: pd.DataFrame, opener_label: str = "Their pick", answer_label: str = "Answer") -> pd.DataFrame:
    """Opener -> answer table with both portraits."""
    t = pd.DataFrame({
        opener_label: df.opener.values, answer_label: df.answer.values,
    })
    if "role" in df:
        t.insert(0, "Role", df.role.str.title().values)
    t["Times"] = [f"{n} of {m}" for n, m in zip(df.n, df.opener_n)]
    t["Answer rate"] = df.answer_rate.values
    t["Usual rate (same patches)"] = df.base_rate.values
    t["Lift"] = [f"×{x:.1f}" for x in df.lift]
    if "answer_win_rate" in df:
        t["Answer won"] = df.answer_win_rate.values
        t["Its usual win rate"] = df.answer_usual_win_rate.values
    if "where_from" in df:
        t["Where it comes from"] = df.where_from.values
    t = with_icons(t, opener_label, label=" ")
    return with_icons(t, answer_label, label="  ")


PAIR_FMT = {"Answer rate": "{:.0%}", "Usual rate (same patches)": "{:.0%}", "Answer won": "{:.0%}",
            "Its usual win rate": "{:.0%}"}


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


def diverging_heatmap(z: pd.DataFrame, hover: pd.DataFrame, title: str, height: int):
    """Champions x groups, blue = prioritised more than average, red = less, gray = average."""
    lim = max(0.05, float(np.nanmax(np.abs(z.values))))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns, y=z.index, zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, NERF], [0.5, "#f0efec"], [1, BUFF]], xgap=2, ygap=2,
        customdata=hover.values, hovertemplate="<b>%{y}</b> · %{x}<br>%{customdata}<extra></extra>",
        colorbar=dict(title=title, tickformat="+.0%", thickness=12)))
    return style(fig, height, hovermode="closest", yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))


def forecast_test(ev: dict, key: str, version: str):
    """Scorecard + predicted-vs-actual dumbbell for one event."""
    m = ev["metrics"][key]
    sc = m[version]
    cmp = ev[f"{key}_{version}"]
    a, b, c, e = st.columns(4)
    a.metric("Presence error: patch model", f"{sc['all']['mae_model'] * 100:.1f} pp",
             help="Mean absolute error in presence across every champion seen in the forecast or at the event")
    b.metric("Baseline: “nothing changes”", f"{sc['all']['mae_baseline'] * 100:.1f} pp",
             help=f"Assume event presence = domestic presence on {sc['base_patch']}")
    dir_acc = sc["changed"]["direction_acc"]
    c.metric("Direction right (changed champs)", "—" if dir_acc is None or pd.isna(dir_acc) else f"{dir_acc:.0%}",
             f"of {sc['changed']['n_moved']} that moved ≥5pp", delta_color="off")
    e.metric("Top-10 overlap", f"{sc['top10_model']}/10", f"baseline {sc['top10_baseline']}/10", delta_color="off")

    # The 30 champions that mattered most on either side of the forecast.
    relevance = cmp[["domestic_presence", "pred_presence", "actual_presence"]].max(axis=1)
    show = cmp.loc[relevance.nlargest(30).index].sort_values("actual_presence")
    fig = go.Figure()
    for _, r in show.iterrows():
        fig.add_trace(go.Scatter(x=[r.domestic_presence, r.pred_presence, r.actual_presence], y=[r.champion] * 3,
                                 mode="lines", line=dict(color="#c3c2b7", width=1.5), showlegend=False, hoverinfo="skip"))
    hover = "<b>%{y}</b><br>%{x:.0%}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=show.domestic_presence, y=show.champion, mode="markers", name=f"Domestic {sc['base_patch']}",
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
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Actual ({sc['games']} games)",
                             marker=dict(color="#52514e", size=11)))
    st.markdown("##### Domestic → forecast → actual presence")
    # Start the axis slightly below 0 so 0% markers aren't cut off at the edge.
    fig = style(fig, xaxis=dict(title="Presence", range=[-0.03, 1.03], **PCT), hovermode="closest")
    st.plotly_chart(row_layout(fig, show.champion, legend=True), use_container_width=True)
    st.caption("Gray dot = domestic presence before the event · orange ring = forecast · filled dot = what happened, "
               "colored by the champion's change since then (blue buff, red nerf, gray adjust/none).")

    cmp = cmp.assign(gain=cmp.error_baseline.abs() - cmp.error_model.abs())
    cols = {"champion": "Champion", "verdict": "Change", "change_patches": "Patches", "domestic_presence": "Domestic",
            "pred_presence": "Forecast", "actual_presence": "Actual"}
    fmt = {c: "{:.0%}" for c in ["Domestic", "Forecast", "Actual"]}
    l, r = st.columns(2)
    l.markdown("**Where the patch notes helped**")
    l.dataframe(with_icons(cmp[cmp.gain > 0].nlargest(8, "gain")[list(cols)].rename(columns=cols), "Champion")
                .style.format(fmt), column_config=image_column(), hide_index=True, use_container_width=True)
    r.markdown("**Where they misled**")
    r.dataframe(with_icons(cmp[cmp.gain < 0].nsmallest(8, "gain")[list(cols)].rename(columns=cols), "Champion")
                .style.format(fmt), column_config=image_column(), hide_index=True, use_container_width=True)


d = load()
stats, changes, lines = d["stats"], d["changes"], d["lines"]

st.title("Patch Impact: how balance changes move pro picks & bans")
st.caption(f"{' · '.join(LEAGUES)} 2026 · pro data from gol.gg, changes from Riot patch notes. "
           "**Presence** = (picks + bans) ÷ games on that patch, the standard measure of draft priority.")

ev = load_events()

f1, f2, f3 = st.columns([1, 1, 2])
version = f1.radio("Event forecast", ["frozen", "current"], horizontal=True,
                   format_func=lambda v: {"frozen": "Frozen (blind test)", "current": "Current model"}[v],
                   help="Frozen forecasts were saved before scoring and are never overwritten. "
                        "'Current' is rebuilt with later fixes, so it isn't a blind test.")
league = f2.selectbox("League (season tabs)", ["ALL"] + LEAGUES,
                      format_func=lambda x: "All leagues" if x == "ALL" else x)
scope = stats[stats.league == league]
trans = transitions_for(league)
patches = sorted(scope.patch.unique(), key=patch_key)
f3.markdown(f"<br>**{int(scope.drop_duplicates('patch').patch_games.sum()):,} domestic games** across pro patches "
            f"{patches[0]}–{patches[-1]} (patches with fewer than 20 games in this league are skipped)",
            unsafe_allow_html=True)

rg = load_regions()
rp = load_ripple()
dr = load_draft()

tab_worlds, tab_events, tab_draft, tab_ripple, tab_regions, tab_effect, tab_champ, tab_patch, tab_pred = st.tabs(
    ["Worlds 2026", "Event tests", "Draft answers", "Ripple effects", "Regions & teams", "Do buffs & nerfs work?",
     "Champion timeline", "Patch explorer", "Predictions"])


# --- Draft answers ---------------------------------------------------------------
with tab_draft:
    if dr is None:
        st.info("Run `python -m lolpatch draft` first.")
    else:
        dm = dr["meta"]
        st.subheader("Draft answers: when a team picks X, what does the opponent answer with?")
        st.markdown(
            "gol.gg groups each team's picks by draft turn, which reveals the full pick order: the team that picks "
            "first takes picks 1, 4-5 and 8-9, the other team 2-3, 6-7 and 10. A **lane answer** is when one team "
            "fills a role and the opponent later fills the same role, so the second pick was made knowing the first. "
            "A pattern counts when the answer comes far more often than that champion is usually taken in that "
            "spot **on the same patches**, so two champions that were just popular at the same time don't count.")
        a, b, c = st.columns(3)
        a.metric("Games with known draft order", f"{dm['games']:,}")
        b.metric("Lane match-ups", f"{dm['lanes']:,}", f"roles {dm['role_source']}", delta_color="off")
        c.metric("Answer patterns found", f"{dm['lane_solid']}", f"of {dm['lane_pairs_tested']:,} pairs tested",
                 delta_color="off", help="At least 4 times, at least 1.5x the usual rate, and significant after "
                                         "correcting for testing many pairs (false discovery rate 10%).")

        lanes = dr["lanes"]
        solid = lanes[lanes.solid].copy()
        f1, f2 = st.columns([3, 2])
        role_d = f1.radio("Role", ["all"] + ROLES, horizontal=True, key="draft_role",
                          format_func=lambda r: r.title() if r != "all" else "All roles")
        kinds = {"All": None, "Broad meta": "Broad meta", "Stronger at Worlds": "Worlds", "Led by one region": "(region)",
                 "Led by one team": "(team)", "Led by one player": "(player)"}
        kind = f2.selectbox("Where it comes from", list(kinds), key="draft_kind")
        view = solid if role_d == "all" else solid[solid.role == role_d]
        if kinds[kind]:
            view = view[view.where_from == "Broad meta"] if kinds[kind] == "Broad meta" \
                else view[view.where_from.str.contains(kinds[kind], regex=False)]
        view = view.sort_values(["n", "lift"], ascending=False)
        st.dataframe(pair_table(view).style.format(PAIR_FMT, na_rep="—"),
                     column_config=image_column(" ") | image_column("  "), hide_index=True,
                     use_container_width=True, height=min(520, 36 * len(view) + 40))
        st.caption("Times = how often the answer came, out of the games where the opponent picked that champion in "
                   "that role first. 'Where it comes from' names an event, region, team or player when they make the "
                   "answer at least twice as often as everyone else (3+ times; for a team or player, half or more of "
                   "all occurrences). It means *led by*, not *only by*. Otherwise the pattern is broad meta.")

        if len(view):
            st.markdown("##### Break a pattern down")
            opts = list(zip(view.role, view.opener, view.answer))
            pick = st.selectbox("Pattern", opts, key="draft_pattern",
                                format_func=lambda k: f"{k[2]} into {k[1]} ({k[0]})")
            bd = dr["breakdown"]
            bd = bd[(bd.role == pick[0]) & (bd.opener == pick[1]) & (bd.answer == pick[2])]
            cols = st.columns(2)
            for i, (dim, g) in enumerate(bd.groupby("dimension", sort=False)):
                g = g.sort_values(["answers", "rate"], ascending=False).head(8)
                cols[i % 2].markdown(f"**By {dim}**")
                cols[i % 2].dataframe(pd.DataFrame({
                    dim.capitalize(): g.group, "Answered": [f"{a} of {o}" for a, o in zip(g.answers, g.openers)],
                    "Rate": g.rate, "Everyone else": g.rest_rate,
                }).style.format({"Rate": "{:.0%}", "Everyone else": "{:.0%}"}, na_rep="—"),
                    hide_index=True, use_container_width=True)

        st.markdown("##### Answers to the very first pick")
        fp = dr["first"]
        fp = fp[fp.solid].sort_values(["n", "lift"], ascending=False)
        st.dataframe(pair_table(fp, "First pick", "Answer with picks 2-3").style.format(PAIR_FMT, na_rep="—"),
                     column_config=image_column(" ") | image_column("  "), hide_index=True, use_container_width=True)
        st.caption("Any role. Often this is 'you took one of the two best champions, we take the other', "
                   "rather than a lane counter.")


# --- Ripple effects --------------------------------------------------------------
with tab_ripple:
    if rp is None:
        st.info("Run `python -m lolpatch meta` first.")
    else:
        st.subheader("Rise and fall: who lost ground as a group of champions took over?")
        st.markdown(
            "Pick one or more champions. Their **group presence** is the share of games where at least one of them "
            "was picked or banned. Over the window you choose, the defaults being their low point and their peak, "
            "every other champion's move is listed by name.")
        all_champs = sorted(stats[stats.league == "ALL"].champion.unique())
        g_sel, g_win = st.columns([3, 2])
        members = g_sel.multiselect("Champion group", all_champs,
                                    default=[c for c in ["Nocturne", "Qiyana"] if c in all_champs], key="ripple_group")
        if members:
            icons = [champ_icon(m) for m in members if champ_icon(m)]
            if icons:
                st.image(icons, width=48, caption=members[:len(icons)])
            gp = group_presence_cached(tuple(members))
            g_patches = list(gp.index)
            lo, hi = rise_window(gp)
            # Keyed by group: a new group resets the window to its own rise.
            win = g_win.select_slider("Window", options=g_patches, value=(lo, hi),
                                      key="ripple_window_" + "|".join(sorted(members)))
            disp = displacement(stats[stats.league == "ALL"], gp, members, win[0], win[1], rp["meta"])
            label = " + ".join(members)
            if disp.empty:
                st.info("Pick a window of at least three patches.")
            else:
                g_before, g_after = gp[win[0]], gp[win[1]]
                st.markdown(f"**{label}**: group presence **{g_before:.0%} → {g_after:.0%}** from {win[0]} to {win[1]}.")
                fell = disp[disp.change <= -0.05].head(12)
                rose = disp[disp.change >= 0.05].sort_values("change", ascending=False).head(8)

                # Chart: the group vs the five biggest fallers.
                fig = go.Figure(go.Scatter(x=g_patches, y=gp.values, name=label + " (group)", mode="lines+markers",
                                           line=dict(color="#0b0b0b", width=3), marker=dict(size=8),
                                           hovertemplate=f"{label} %{{y:.0%}}<extra></extra>"))
                pv = stats[stats.league == "ALL"].pivot_table(index="champion", columns="patch", values="presence")
                for i, c in enumerate(fell.champion.head(5)):
                    y = pv.loc[c].reindex(g_patches).fillna(0) if c in pv.index else pd.Series(0, index=g_patches)
                    fig.add_trace(go.Scatter(x=g_patches, y=y.values, name=c, mode="lines+markers",
                                             line=dict(color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][i], width=1.8),
                                             marker=dict(size=5), hovertemplate=f"{c} %{{y:.0%}}<extra></extra>"))
                fig.add_vrect(x0=g_patches.index(win[0]), x1=g_patches.index(win[1]), fillcolor="#2a78d6", opacity=0.06,
                              line_width=0)
                st.plotly_chart(style(fig, 380, xaxis=dict(type="category", title="Patch"),
                                      yaxis=dict(title="Presence", **PCT)), use_container_width=True)

                def lead(df):
                    t = pd.DataFrame({
                        "Champion": df.champion, "Role": df.role.fillna("—").str.title(),
                        "Same role as group": df.same_role.map({True: "yes", False: ""}),
                        "Presence": [f"{b:.0%} → {a:.0%}" for b, a in zip(df.before, df.after)],
                        "Change": df.change,
                        "In step?": [("moves opposite the group" if r <= -0.3 else
                                      "moves with the group" if r >= 0.3 else "unrelated moves") if pd.notna(r) else "—"
                                     for r in df.r],
                        "r": df.r,
                    })
                    return with_icons(t, "Champion")

                l, r = st.columns(2)
                l.markdown(f"**Fell while {label} rose**")
                l.dataframe(lead(fell).style.format({"Change": "{:+.0%}", "r": "{:+.2f}"}, na_rep="—"),
                            column_config=image_column(), hide_index=True, use_container_width=True)
                r.markdown("**Rose alongside**")
                r.dataframe(lead(rose).style.format({"Change": "{:+.0%}", "r": "{:+.2f}"}, na_rep="—"),
                            column_config=image_column(), hide_index=True, use_container_width=True)
                st.caption(
                    "Presence at each end is averaged over two patches. **In step?** compares patch-to-patch moves "
                    "(r): champions that *move opposite* the group were likely pushed out by it. Champions that fell "
                    "over the window but *move with* the group patch to patch were declining for other reasons "
                    "(their own nerfs, other metas).")

        st.markdown("##### Who took the slot after a big change?")
        subs, pairs = rp["subs"], rp["pairs"]
        role_f = st.radio("Role", ["all"] + ROLES, horizontal=True, key="ripple_role",
                          format_func=lambda r: r.title() if r != "all" else "All roles")
        sv = subs if role_f == "all" else subs[subs.role == role_f]
        sv = sv.reindex(sv.changed_delta.abs().sort_values(ascending=False).index)
        tbl = pd.DataFrame({
            "Patches": sv.from_patch + " → " + sv.to_patch, "Role": sv.role.str.title(),
            "Changed champion": sv.changed, "Change": sv.change + " (" + sv.change_patches.fillna("") + ")",
            "Its presence": [f"{b:.0%} → {a:.0%}" for b, a in zip(sv.changed_before, sv.changed_after)],
            "Untouched champion that moved the other way": sv.other,
            "Its presence ": [f"{b:.0%} → {a:.0%}" for b, a in zip(sv.other_before, sv.other_after)],
        })
        tbl = with_icons(tbl, "Changed champion", label=" ")
        tbl = with_icons(tbl, "Untouched champion that moved the other way", label="  ")
        st.dataframe(tbl, column_config=image_column(" ") | image_column("  "), hide_index=True,
                     use_container_width=True, height=min(420, 36 * len(tbl) + 40))
        st.caption(f"A changed champion moved at least {SUB_MIN_MOVE_A:.0%} in presence; listed alongside are untouched "
                   f"champions in the same pro role that moved at least {SUB_MIN_MOVE_B:.0%} the opposite way in the "
                   "same patch window. Timing makes these candidates, not proof.")
        rec = pairs[pairs.times >= 2]
        if len(rec):
            st.markdown("**Happened more than once**")
            rtbl = with_icons(with_icons(rec.rename(columns={"changed": "Changed", "other": "Moved the other way",
                                                             "role": "Role", "times": "Times", "patches": "Patches"})
                                         [["Changed", "Moved the other way", "Role", "Times", "Patches"]],
                                         "Changed", label=" "), "Moved the other way", label="  ")
            st.dataframe(rtbl, column_config=image_column(" ") | image_column("  "), hide_index=True,
                         use_container_width=True)

# --- Worlds ------------------------------------------------------------------
with tab_worlds:
    if not ev or "worlds" not in ev["metrics"]:
        st.info("Run `python -m lolpatch events` to load Worlds.")
    else:
        wm = ev["metrics"]["worlds"]
        sc = wm[version]
        stages = " · ".join(f"{k} {v}" for k, v in wm["stages"].items())
        bridge = [p for p in sorted(changes.patch.unique(), key=patch_key)
                  if patch_key(sc["base_patch"]) < patch_key(p) <= patch_key(wm["patch"])]
        st.subheader(f"Worlds 2026 · patch {wm['patch']}")
        st.info(
            f"**Every Worlds game so far was played on patch {wm['patch']}** ({sc['games']} games: {stages}). "
            f"The domestic leagues' last games were on **{sc['base_patch']}**, so no league games exist on "
            f"{', '.join(bridge)}. Worlds is the first pro data on {wm['patch']}.\n\n"
            f"The forecast starts from domestic {sc['base_patch']} play and applies the balance changes from "
            f"{', '.join(bridge)} to predict {wm['patch']}. No Worlds data went into it "
            f"(saved {sc['forecast_created']}).")

        wcmp = ev[f"worlds_{version}"].set_index("champ_key")
        st.markdown(f"##### What changed on the way to {wm['patch']}, and how those champions are doing at Worlds")
        bp = st.radio("Patch", bridge[::-1], horizontal=True, key="worlds_bridge_patch",
                      format_func=lambda p: f"{p} (Worlds patch)" if p == wm["patch"] else p)
        pc = changes[changes.patch == bp].copy()
        pc["domestic"] = pc.champ_key.map(wcmp.domestic_presence).fillna(0)
        pc["worlds"] = pc.champ_key.map(wcmp.actual_presence).fillna(0)
        pc["picks"] = pc.champ_key.map(wcmp.picks).fillna(0).astype(int)
        pc["bans"] = pc.champ_key.map(wcmp.bans).fillna(0).astype(int)
        pc["wins"] = pc.champ_key.map(wcmp.wins).fillna(0).astype(int)
        pc["record"] = [f"{w}–{p - w}" if p else "—" for w, p in zip(pc.wins, pc.picks)]
        pc = pc.sort_values("worlds", ascending=False)
        tbl = pc[["champion", "verdict", "score", "domestic", "worlds", "picks", "bans", "record"]].rename(columns={
            "champion": "Champion", "verdict": "Change", "score": "Score", "domestic": f"Domestic {sc['base_patch']}",
            "worlds": f"Worlds {wm['patch']}", "picks": "Picks", "bans": "Bans", "record": "W–L when picked"})
        st.dataframe(with_icons(tbl, "Champion").style.format(
            {"Score": "{:+.2f}", f"Domestic {sc['base_patch']}": "{:.0%}", f"Worlds {wm['patch']}": "{:.0%}"}),
            column_config=image_column(), hide_index=True, use_container_width=True)
        st.caption(f"Presence = picks + bans ÷ games. Score runs from −1 (clear nerf) to +1 (clear buff). "
                   f"Changes from {', '.join(bridge)} all stack into the Worlds forecast below.")

        st.markdown(f"##### Did the patch notes call it? Forecast for {wm['patch']} vs what happened")
        forecast_test(ev, "worlds", version)

        if "worlds_live" in ev:
            lv = ev["worlds_live"].head(25).iloc[::-1]
            st.markdown("##### Live estimate for the rest of Worlds")
            st.caption(f"The forecast counts as {PRIOR_GAMES} games of evidence, and every Worlds game played pulls the estimate "
                       "toward what teams are actually doing. Bars show the 80% range. Re-run "
                       "`python -m lolpatch events` after each Swiss round to update.")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=lv.live_presence, y=lv.champion, orientation="h", name="Live estimate",
                marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in lv.verdict], cornerradius=4),
                error_x=dict(type="data", symmetric=False, array=lv.live_high - lv.live_presence,
                             arrayminus=lv.live_presence - lv.live_low, color=NEUTRAL, thickness=1.5, width=3),
                customdata=lv[["live_low", "live_high", "actual_presence", "picks", "bans", "games", "pred_presence"]],
                hovertemplate="<b>%{y}</b> %{x:.0%} (80%: %{customdata[0]:.0%}–%{customdata[1]:.0%})"
                              "<br>so far %{customdata[2]:.0%} = %{customdata[3]} picks + %{customdata[4]} bans in "
                              "%{customdata[5]} games<br>pre-event forecast %{customdata[6]:.0%}<extra></extra>"))
            fig = style(fig, xaxis=dict(title="Expected presence", range=[0, 1], **PCT),
                        hovermode="closest", showlegend=False, bargap=0.3)
            st.plotly_chart(row_layout(fig, lv.champion), use_container_width=True)

        wg = ev["games"][ev["games"].league == "worlds"]

        # Teams: series record, games record, and what they lean on.
        st.markdown("##### Teams at Worlds")
        regions_map = json.loads((PROCESSED / "team_regions.json").read_text()) \
            if (PROCESSED / "team_regions.json").exists() else {}
        series = wg.groupby(["series_id", "team"]).result.sum().reset_index()
        series["won"] = series.groupby("series_id").result.transform(lambda s: (s == s.max()) & (s.max() > s.min()))
        picks = wg.assign(champion=wg.picks.str.split("|")).explode("champion")
        top3 = picks.groupby("team").champion.agg(lambda s: ", ".join(s.value_counts().head(3).index))
        bans_vs = wg.merge(wg[["game_uid", "team", "bans"]], on="game_uid", suffixes=("", "_opp"))
        bans_vs = bans_vs[bans_vs.team != bans_vs.team_opp].assign(champion=lambda x: x.bans_opp.fillna("").str.split("|"))
        bans_vs = bans_vs.explode("champion").query("champion != ''")
        feared = bans_vs.groupby("team").champion.agg(lambda s: ", ".join(s.value_counts().head(3).index))
        teams_tbl = pd.DataFrame({
            "Team": sorted(wg.team.unique()),
        })
        teams_tbl["Logo"] = teams_tbl.Team.map(team_logo)
        teams_tbl["Region"] = teams_tbl.Team.map(regions_map).fillna("—")
        rec = series.groupby("team").won.agg(["sum", "size"])
        teams_tbl["Series"] = [f"{int(rec.loc[t, 'sum'])}–{int(rec.loc[t, 'size'] - rec.loc[t, 'sum'])}" for t in teams_tbl.Team]
        g_rec = wg.groupby("team").result.agg(["sum", "size"])
        teams_tbl["Games"] = [f"{int(g_rec.loc[t, 'sum'])}–{int(g_rec.loc[t, 'size'] - g_rec.loc[t, 'sum'])}" for t in teams_tbl.Team]
        teams_tbl["_order"] = [rec.loc[t, "sum"] - (rec.loc[t, "size"] - rec.loc[t, "sum"]) for t in teams_tbl.Team]
        teams_tbl["Most picked"] = teams_tbl.Team.map(top3)
        teams_tbl["Most banned against them"] = teams_tbl.Team.map(feared)
        teams_tbl = teams_tbl.sort_values(["_order", "Region"], ascending=[False, True]).drop(columns="_order")
        st.dataframe(teams_tbl[["Logo", "Team", "Region", "Series", "Games", "Most picked", "Most banned against them"]],
                     column_config={"Logo": st.column_config.ImageColumn(" ", width="small")},
                     hide_index=True, use_container_width=True)
        st.caption("Records cover the play-in and Swiss games played so far. Play-in series were best-of-three or five; "
                   "Swiss round 1 was best-of-one.")
        rows = []
        for stage, g in wg.groupby("stage_group"):
            n = g.game_uid.nunique()
            pb = pd.concat([g.picks.str.split("|").explode(), g.bans.fillna("").str.split("|").explode()])
            rows.append((pb[pb != ""].value_counts() / n).rename(f"{stage} ({n})"))
        if rows:
            st.markdown("##### Worlds meta by stage (presence)")
            by_stage = pd.concat(rows, axis=1).fillna(0)
            by_stage = by_stage.loc[by_stage.max(axis=1).sort_values(ascending=False).index].head(25)
            stage_cols = list(by_stage.columns)
            by_stage = with_icons(by_stage.rename_axis("Champion").reset_index(), "Champion")
            st.dataframe(by_stage.style.format("{:.0%}", subset=stage_cols)
                         .background_gradient(cmap="Blues", vmin=0, vmax=1, subset=stage_cols),
                         column_config=image_column(), hide_index=True, use_container_width=True)

        if dr is not None:
            we = dr["events"]
            w = we[we.scope == "Worlds"]
            if len(w):
                st.markdown("##### Draft answers at Worlds")
                rest = we[we.scope != "Worlds"]
                rest_open = rest.groupby(["role", "opener"]).size()
                rest_pair = rest.groupby(["role", "opener", "answer"]).size()
                w_open = w.groupby(["role", "opener"]).size()
                wp = w.groupby(["role", "opener", "answer"]).agg(n=("game_uid", "size"), won=("answer_won", "sum")).reset_index()
                wp = wp[wp.n >= 2]
                wp["opener_n"] = [w_open[(r, o)] for r, o in zip(wp.role, wp.opener)]
                wp["worlds_rate"] = wp.n / wp.opener_n
                wp["rest_rate"] = [rest_pair.get((r, o, a), 0) / rest_open[(r, o)] if (r, o) in rest_open else np.nan
                                   for r, o, a in zip(wp.role, wp.opener, wp.answer)]
                wp = wp.sort_values(["n", "worlds_rate"], ascending=False)
                t = pd.DataFrame({"Role": wp.role.str.title(), "Their pick": wp.opener, "Answer": wp.answer,
                                  "At Worlds": [f"{n} of {m}" for n, m in zip(wp.n, wp.opener_n)],
                                  "Worlds rate": wp.worlds_rate, "Rate before Worlds": wp.rest_rate,
                                  "Answer W-L": [f"{x}-{n - x}" for x, n in zip(wp.won, wp.n)]})
                t = with_icons(with_icons(t, "Their pick", label=" "), "Answer", label="  ")
                st.dataframe(t.style.format({"Worlds rate": "{:.0%}", "Rate before Worlds": "{:.0%}"}, na_rep="—"),
                             column_config=image_column(" ") | image_column("  "), hide_index=True,
                             use_container_width=True)
                st.caption(f"Lane answers seen at least twice at Worlds so far ({w.game_uid.nunique()} games), next to "
                           "how often the same answer came in every earlier game. A big gap means it's a Worlds "
                           "development; small samples, so read it as a trend to watch.")

        if rg is not None:
            wr = rg["events"][rg["events"].event == "worlds"]
            if len(wr):
                st.markdown("##### What each region is playing at Worlds")
                reg_games = wr.drop_duplicates("region").set_index("region").region_games
                top = wr.groupby("champion").pick_ban_rate.max().nlargest(20).index
                z = wr[wr.champion.isin(top)].pivot(index="champion", columns="region", values="pick_ban_rate").reindex(top).fillna(0)
                z.columns = [f"{c} ({int(reg_games[c])})" for c in z.columns]
                reg_cols = list(z.columns)
                z = with_icons(z.rename_axis("Champion").reset_index(), "Champion")
                st.dataframe(z.style.format("{:.0%}", subset=reg_cols)
                             .background_gradient(cmap="Blues", vmin=0, vmax=1, subset=reg_cols),
                             column_config=image_column(), hide_index=True, use_container_width=True)
                st.caption("Share of each region's Worlds games in which one of its teams picked or banned the champion "
                           "(region game counts in brackets). Small samples: read big gaps only.")

# --- Event tests ---------------------------------------------------------------
with tab_events:
    if not ev:
        st.info("Run `python -m lolpatch events` first.")
    else:
        st.subheader("Every international event as an out-of-sample test")
        rows = []
        for key, m in ev["metrics"].items():
            sc = m[version]
            rows.append(dict(Event=m["title"], Patch=m["patch"], Games=sc["games"], Forecast_from=sc["base_patch"],
                             Model=sc["all"]["mae_model"], Nothing_changes=sc["all"]["mae_baseline"],
                             Model_changed=sc["changed"]["mae_model"], Baseline_changed=sc["changed"]["mae_baseline"],
                             Top10_model=sc["top10_model"], Top10_baseline=sc["top10_baseline"]))
        tbl = pd.DataFrame(rows)
        tbl["Model wins?"] = np.where(tbl.Model < tbl.Nothing_changes, "yes", "no")
        st.dataframe(tbl.rename(columns={"Forecast_from": "Forecast from", "Nothing_changes": "“Nothing changes”",
                                         "Model_changed": "Model (changed champs)",
                                         "Baseline_changed": "Baseline (changed champs)",
                                         "Top10_model": "Top-10 model", "Top10_baseline": "Top-10 baseline"})
                     .style.format({c: lambda v: f"{v * 100:.1f} pp" for c in
                                    ["Model", "“Nothing changes”", "Model (changed champs)", "Baseline (changed champs)"]}),
                     hide_index=True, use_container_width=True)
        st.caption("Presence error in percentage points (lower is better). Each forecast uses only domestic games from "
                   "patches before the event, plus the patch notes since. First Stand, MSI and EWC forecasts were "
                   "created after the fact with the current classifier, so only Worlds and Demacia have truly frozen "
                   "blind forecasts.")

        if "adjustment" in ev:
            adj = ev["adjustment"]
            st.markdown("##### Tested and rejected: an “international adjustment”")
            st.markdown(
                "After Worlds, the theory was that the top domestic picks hold their priority at internationals, so the "
                "model's expected cooling-off should be dropped. Checked on the three *earlier* events it was worse, "
                "not better. In fact, top picks usually fall **more** than the model expects at events, Demacia Cup "
                "excepted. The model stays as is.")
            st.dataframe(adj.rename(columns={
                "event": "Event", "role": "Used for", "games": "Games", "mae_nothing_changes": "“Nothing changes”",
                "mae_model": "Model", "mae_model_no_drift": "Model without drift", "n_top": "Top picks (>60%)",
                "top_actual_change": "Top picks: actual change", "top_model_drift": "Top picks: model drift"})
                .assign(**{"Used for": lambda x: x["Used for"].map({"train": "deciding", "test": "checking"})})
                .style.format({"“Nothing changes”": "{:.1%}", "Model": "{:.1%}", "Model without drift": "{:.1%}",
                               "Top picks: actual change": "{:+.0%}", "Top picks: model drift": "{:+.0%}"}),
                hide_index=True, use_container_width=True)

        st.markdown("##### Drill into one event")
        keys = list(ev["metrics"])
        pick = st.selectbox("Event", keys, index=keys.index("demacia") if "demacia" in keys else 0,
                            format_func=lambda k: ev["metrics"][k]["title"])
        forecast_test(ev, pick, version)

# --- Regions & teams -----------------------------------------------------------
with tab_regions:
    if rg is None:
        st.info("Run `python -m lolpatch regions` first.")
    else:
        eff = rg["effects"]
        st.subheader("Does priority actually make a difference?")
        a, b, c = st.columns(3)
        md, cf, fv = eff["meta_draft"], eff["comfort"], eff["region_favorites"]
        a.metric("Drafting the more contested champions", f"{md['adjusted_win_prob_plus10pp']:.1%} win",
                 f"95% CI {md['adjusted_ci'][0]:.0%}–{md['adjusted_ci'][1]:.0%}", delta_color="off",
                 help="Win probability for evenly matched teams when one drafts champions with 10pp higher average "
                      "presence on the previous patch. Logistic regression controlling for team strength.")
        a.caption(f"No edge. Team strength explains winning; the meta-ness of the draft doesn't add anything "
                  f"({md['games']:,} domestic games, p = {md['p_value']:.2f}).")
        b.metric("Comfort picks vs other picks", f"{cf['within_team_diff'] * 100:+.1f} pp",
                 f"95% CI {cf['ci'][0] * 100:+.1f} to {cf['ci'][1] * 100:+.1f}", delta_color="off",
                 help="Within each team: win rate on champions it had already played 3+ times before that game, minus "
                      "its win rate on other champions.")
        b.caption(f"No measurable edge from sticking to proven champions ({cf['teams']} teams).")
        if fv["regions"]:
            c.metric("Regional favorites at internationals", f"{fv['within_region_diff'] * 100:+.1f} pp",
                     f"95% CI {fv['ci'][0] * 100:+.0f} to {fv['ci'][1] * 100:+.0f}", delta_color="off",
                     help=f"Within each region, at international events, against teams from other regions: win rate "
                          f"on champions it prioritised at home ({fv['favorite_lift'] * 100:.0f}pp+ above the all-league "
                          "average) minus its win rate on other champions.")
            by_region = ", ".join(f"{r['region']} {r['diff'] * 100:+.0f}" for r in fv["regions"])
            c.caption(f"Too few games to call: {fv['favorite_picks']} favorite picks ({by_region} pp).")

        with st.expander("Details of the three tests"):
            st.markdown("**Win rate by how contested a team's draft was vs its opponent** "
                        "(average previous-patch presence of its five picks)")
            bk = pd.DataFrame(md["by_bucket"])
            fig = go.Figure(go.Bar(x=bk.bucket, y=bk.win_rate, marker=dict(color=SERIES[0], cornerradius=4),
                                   customdata=bk.n, hovertemplate="%{x}: %{y:.1%} of %{customdata} games<extra></extra>"))
            fig.add_hline(y=0.5, line=dict(color=NEUTRAL, dash="dot"))
            st.plotly_chart(style(fig, 280, yaxis=dict(range=[0.4, 0.6], **PCT), hovermode="closest"),
                            use_container_width=True)
            if fv["regions"]:
                st.markdown("**Regional favorites at international events, by region**")
                st.dataframe(pd.DataFrame(fv["regions"]).style.format(
                    {"win_rate_favorite": "{:.0%}", "win_rate_other": "{:.0%}", "diff": "{:+.0%}"}),
                    hide_index=True, use_container_width=True)
            st.caption("gol.gg lists picks and bans in draft order (checked: average presence falls from pick 1 to "
                       "pick 5). 'Early' = a team's first two picks or first three bans.")

        st.subheader("What each region prioritizes")
        metric = st.radio("Measure", ["presence_lift", "early_lift"], horizontal=True,
                          format_func=lambda m: {"presence_lift": "Presence vs all-league average",
                                                 "early_lift": "Early priority (first 2 picks / first 3 bans) vs average"}[m])
        rc = rg["regions"]
        champs = (rc.groupby("champion").presence.mean().nlargest(40).index)
        z = rc[rc.champion.isin(champs)].pivot(index="champion", columns="league", values=metric).reindex(champs)
        base = "presence" if metric == "presence_lift" else "early_priority"
        exp = "expected_presence" if metric == "presence_lift" else "expected_early"
        hv = rc[rc.champion.isin(champs)].assign(
            txt=lambda x: [f"{o:.0%} vs {e:.0%} expected ({l * 100:+.0f}pp)" for o, e, l in zip(x[base], x[exp], x[metric])]
        ).pivot(index="champion", columns="league", values="txt").reindex(champs)
        st.plotly_chart(row_layout(diverging_heatmap(z, hv, "vs avg", 400), z.index, x_top=True, right=90, reverse=True),
                        use_container_width=True)
        st.caption("Blue = the region prioritises the champion more than the average of all leagues on the same "
                   "patches; red = less. Rows are the 40 most-contested champions of the season.")

        st.subheader("Team explorer")
        tc = rg["teams"]
        teams = tc.drop_duplicates("team").sort_values(["league", "team"])
        t1, t_logo, t2 = st.columns([2, 1, 5], vertical_alignment="center")
        team = t1.selectbox("Team", teams.team, format_func=lambda t: f"{t} ({teams.set_index('team').league[t]})")
        tt = tc[tc.team == team]
        n_games = int(tt.team_games.iloc[0])
        if team_logo(team):
            t_logo.image(team_logo(team), width=64)
        t2.markdown(f"### {team}\n**{n_games} games** in {tt.league.iloc[0]} this season")
        sig = tt[tt.picks >= 4].nlargest(12, "pick_lift").iloc[::-1]
        fig = go.Figure(go.Bar(
            x=sig.pick_lift, y=sig.champion, orientation="h", marker=dict(color=BUFF, cornerradius=4),
            customdata=sig[["picks", "pick_share", "league_pick_share", "win_rate", "early_picks"]],
            text=[f"{w:.0%} WR" for w in sig.win_rate], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b>: picked in %{customdata[1]:.0%} of games vs league %{customdata[2]:.0%}"
                          "<br>%{customdata[0]} picks (%{customdata[4]} early) · %{customdata[3]:.0%} win rate<extra></extra>"))
        l, r = st.columns([3, 2])
        l.markdown("**Signature picks** (picked more than the league average, 4+ picks)")
        # Leave room past the longest bar for its win-rate label.
        fig = style(fig, xaxis=dict(title="Pick share above league average", range=[0, sig.pick_lift.max() * 1.22], **PCT),
                    hovermode="closest")
        l.plotly_chart(row_layout(fig, sig.champion, right=64), use_container_width=True)
        r.markdown("**What opponents ban against them**")
        banned = tt.nlargest(10, "bans_against")[["champion", "bans_against", "against_rate", "picks", "win_rate"]] \
            .rename(columns={"champion": "Champion", "bans_against": "Bans vs", "against_rate": "Share of games",
                             "picks": "Their picks", "win_rate": "Their WR"})
        r.dataframe(with_icons(banned, "Champion").style.format({"Share of games": "{:.0%}", "Their WR": "{:.0%}"}, na_rep="—"),
                    column_config=image_column(), hide_index=True, use_container_width=True)
        r.caption("Win rates on a handful of picks are noisy; a 4-pick 100% is luck as often as skill.")

# --- Effects -----------------------------------------------------------------
with tab_effect:
    eff = verdict_effects(trans).reindex(["buff", "adjust", "none", "nerf"]).dropna(how="all")
    c = st.columns(len(eff))
    for col, (v, row) in zip(c, eff.iterrows()):
        label = {"none": "No change"}.get(v, v.capitalize() + "ed" if v != "adjust" else "Adjusted")
        col.metric(label, f"{row.mean_delta_presence * 100:+.1f} pp", f"{row.share_presence_up:.0%} rose · n={int(row.n)}",
                   delta_color="off", help="Mean change in presence to the next pro patch")
    st.markdown("##### Distribution of presence change after each kind of change")
    fig = go.Figure()
    for v in eff.index:
        sub = trans[trans.verdict == v]
        fig.add_trace(go.Box(y=sub.delta_presence, name=v, marker_color=VERDICT_COLOR[v], boxmean=True,
                             line_width=2, customdata=sub[["champion", "from_patch", "to_patch"]],
                             hovertemplate="%{customdata[0]} %{customdata[1]}→%{customdata[2]}: %{y:+.1%}<extra></extra>"))
    st.plotly_chart(style(fig, yaxis=dict(title="Δ presence", **PCT), showlegend=False, hovermode="closest"),
                    use_container_width=True)

    st.markdown("##### Does the size of the change matter? Starting presence vs change, by verdict")
    changed = trans[trans.changed]
    fig = go.Figure()
    for v in ["buff", "nerf", "adjust"]:
        sub = changed[changed.verdict == v]
        fig.add_trace(go.Scatter(
            x=sub.prev_presence, y=sub.delta_presence, mode="markers", name=v,
            marker=dict(color=VERDICT_COLOR[v], size=9, line=dict(width=1.5, color="white")),
            customdata=sub[["champion", "from_patch", "to_patch", "change_patches"]],
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[3]})<br>%{customdata[1]}→%{customdata[2]}"
                          "<br>presence before %{x:.0%}, change %{y:+.1%}<extra></extra>"))
    st.plotly_chart(style(fig, 420, xaxis=dict(title="Presence before change", **PCT),
                          yaxis=dict(title="Δ presence", **PCT), hovermode="closest"), use_container_width=True)
    hi_nerf = changed[(changed.verdict == "nerf") & (changed.prev_presence > 0.6)]
    lo_buff = changed[(changed.verdict == "buff") & (changed.prev_presence < 0.05)]
    notes = []
    if len(hi_nerf):
        notes.append(f"Nerfs to champions above 60% presence moved them {hi_nerf.delta_presence.mean() * 100:+.0f} pp on average "
                     f"({(hi_nerf.delta_presence < 0).mean():.0%} fell, n={len(hi_nerf)}).")
    if len(lo_buff):
        notes.append(f"Buffs to fringe champions (<5% presence) moved them {lo_buff.delta_presence.mean() * 100:+.0f} pp on average "
                     f"({(lo_buff.delta_presence > 0).mean():.0%} rose, n={len(lo_buff)}).")
    st.caption(" ".join(notes))
    if "feature_effects" in d:
        fe = d["feature_effects"].rename(columns={"Unnamed: 0": "feature"})
        with st.expander("What predicts a presence change? (standardised ridge coefficients)"):
            st.dataframe(fe.set_index(fe.columns[0]).style.format("{:+.4f}"), use_container_width=True)

# --- Champion timeline -------------------------------------------------------
with tab_champ:
    champs = scope.groupby("champion").presence.max().sort_values(ascending=False).index.tolist()
    c_sel, c_img, c_name = st.columns([2, 1, 5], vertical_alignment="center")
    champ = c_sel.selectbox("Champion", champs, index=0)
    if champ_icon(champ):
        c_img.image(champ_icon(champ), width=72)
    c_name.markdown(f"### {champ}")
    cs = scope[scope.champion == champ].set_index("patch").reindex(patches)
    cs[["presence", "pick_rate", "ban_rate"]] = cs[["presence", "pick_rate", "ban_rate"]].fillna(0)
    key = scope[scope.champion == champ].champ_key.iloc[0]
    cc = changes[changes.champ_key == key]

    # International events as their own points on the patch axis (e.g. "26.20 Worlds"),
    # placed after the domestic games of the same patch.
    EVENT_SHORT = {"first_stand": "First Stand", "msi": "MSI", "ewc": "EWC", "demacia": "Demacia Cup", "worlds": "Worlds"}
    ev_pts = []
    if ev:
        for order, (k, m) in enumerate(ev["metrics"].items()):  # metrics are in event date order
            row = ev[f"{k}_current"].set_index("champ_key")
            r = row.loc[key] if key in row.index else None
            games = m["current"]["games"]
            ev_pts.append(dict(label=f"{m['patch']} {EVENT_SHORT.get(k, k)}", patch=m["patch"], games=games, order=order,
                               presence=float(r.actual_presence) if r is not None else 0.0,
                               picks=int(r.picks) if r is not None else 0, bans=int(r.bans) if r is not None else 0))
    axis = sorted([(patch_key(p), -1, p) for p in patches] + [(patch_key(e["patch"]), e["order"], e["label"]) for e in ev_pts])
    x_order = [x for *_, x in axis]
    x_patch = {p: p for p in patches} | {e["label"]: e["patch"] for e in ev_pts}

    fig = go.Figure()
    for i, (m, name) in enumerate([("presence", "Presence"), ("pick_rate", "Pick rate"), ("ban_rate", "Ban rate")]):
        fig.add_trace(go.Scatter(x=patches, y=cs[m], name=f"{name} (domestic)" if i == 0 else name, mode="lines+markers",
                                 line=dict(color=SERIES[i], width=2 if i == 0 else 1.5, dash=None if i == 0 else "dot"),
                                 marker=dict(size=8 if i == 0 else 5), hovertemplate=f"{name} %{{y:.0%}}<extra></extra>"))
    if ev_pts:
        ep = pd.DataFrame(ev_pts)
        fig.add_trace(go.Scatter(
            x=ep.label, y=ep.presence, name="Presence (international event)", mode="markers",
            marker=dict(symbol="square", size=10, color="#52514e", line=dict(width=1.5, color="white")),
            customdata=ep[["picks", "bans", "games"]],
            hovertemplate="<b>%{x}</b><br>presence %{y:.0%} (%{customdata[0]} picks + %{customdata[1]} bans "
                          "in %{customdata[2]} games)<extra></extra>"))
    # Changes are drawn on the first point (domestic patch or event) where they were live.
    ymax = max(cs.presence.max(), max((e["presence"] for e in ev_pts), default=0), 0.1) * 1.15
    for _, ch in cc.iterrows():
        nxt = next((x for x in x_order if patch_key(x_patch[x]) >= patch_key(ch.patch)), None)
        if nxt is None:
            continue
        sym = {"buff": "triangle-up", "nerf": "triangle-down"}.get(ch.verdict, "diamond")
        fig.add_trace(go.Scatter(
            x=[nxt], y=[ymax], mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=14, color=VERDICT_COLOR[ch.verdict], line=dict(width=1.5, color="white")),
            hovertemplate=f"<b>{ch.patch} {ch.verdict}</b> (score {ch.score:+.2f})<extra></extra>"))
        # By index: Plotly reads number-like categories ("26.17") as numbers when placing shapes.
        fig.add_vline(x=x_order.index(nxt), line=dict(color=VERDICT_COLOR[ch.verdict], width=1, dash="dot"))
    st.plotly_chart(style(fig, 420, yaxis=dict(title=None, range=[0, ymax * 1.08], **PCT),
                          xaxis=dict(type="category", categoryorder="array", categoryarray=x_order, title="Patch")),
                    use_container_width=True)
    if ev_pts:
        st.caption(f"Lines = domestic league play (through {patches[-1]}). Gray squares = international events, "
                   "each on the patch it was played on (Worlds is on "
                   f"{next((e['patch'] for e in ev_pts if e['label'].endswith('Worlds')), '?')}).")
    st.caption("▲ buff · ▼ nerf · ◆ adjustment, marked on the first pro patch where the change was live.")
    if len(cc):
        st.markdown("##### Balance changes")
        for _, ch in cc.sort_values("patch", key=lambda s: s.map(patch_key)).iterrows():
            with st.expander(f"{ch.patch} — {ch.verdict.upper()} (score {ch.score:+.2f})"
                             f"{' · mentions pro play' if ch.pro_targeted else ''}"):
                st.write(ch.context if isinstance(ch.context, str) else "")
                ln = lines[(lines.patch == ch.patch) & (lines.champ_key == key) & lines.text.notna()]
                st.dataframe(ln[["ability", "stat", "before", "after", "direction"]], hide_index=True, use_container_width=True)

# --- Patch explorer ----------------------------------------------------------
with tab_patch:
    trans_patches = sorted(trans.to_patch.unique(), key=patch_key)
    tp = st.select_slider("Pro patch", trans_patches, value=trans_patches[-1])
    t = trans[trans.to_patch == tp]
    fp = t.from_patch.iloc[0]
    window = [p for p in sorted(changes.patch.unique(), key=patch_key) if patch_key(fp) < patch_key(p) <= patch_key(tp)]
    st.markdown(f"Comparing **{fp} → {tp}**. Balance changes included: **{', '.join(window) or 'none'}**")

    ch = t[t.changed].sort_values("delta_presence")
    fig = go.Figure()
    for _, r in ch.iterrows():
        fig.add_trace(go.Scatter(x=[r.prev_presence, r.next_presence], y=[r.champion] * 2, mode="lines",
                                 line=dict(color=VERDICT_COLOR[r.verdict], width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ch.prev_presence, y=ch.champion, mode="markers", name=f"Before ({fp})",
                             marker=dict(color=NEUTRAL, size=9), hovertemplate="%{y}: %{x:.0%}<extra>before</extra>"))
    fig.add_trace(go.Scatter(x=ch.next_presence, y=ch.champion, mode="markers", showlegend=False,
                             marker=dict(color=[VERDICT_COLOR[v] for v in ch.verdict], size=11, line=dict(width=1.5, color="white")),
                             hovertemplate="%{y}: %{x:.0%}<extra>after</extra>"))
    # Legend entry in neutral ink: the real dots are colored by change type.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"After ({tp})",
                             marker=dict(color="#52514e", size=11)))
    st.markdown("##### Changed champions: presence before → after (line color = buff blue / nerf red / adjust gray)")
    x_hi = max(ch.prev_presence.max(), ch.next_presence.max(), 0.1) + 0.03
    # Start slightly below 0 so 0% markers aren't cut off at the edge.
    fig = style(fig, xaxis=dict(title="Presence", range=[-0.02, x_hi], **PCT), hovermode="closest")
    st.plotly_chart(row_layout(fig, ch.champion, legend=True), use_container_width=True)

    st.markdown("##### Biggest movers on this patch (changed or not)")
    movers = t.reindex(t.delta_presence.abs().sort_values(ascending=False).index).head(20)
    st.dataframe(with_icons(movers[["champion", "verdict", "change_patches", "pro_targeted", "prev_presence", "next_presence",
                                    "delta_presence", "prev_win_rate", "next_win_rate"]])
                 .style.format({c: "{:.0%}" for c in ["prev_presence", "next_presence", "prev_win_rate", "next_win_rate"]}
                               | {"delta_presence": "{:+.0%}"}),
                 column_config=image_column(), hide_index=True, use_container_width=True)

# --- Predictions -------------------------------------------------------------
with tab_pred:
    if "predictions" not in d:
        st.info("Run `python -m lolpatch predict` first.")
    else:
        pr = d["predictions"]
        meta = d["meta"]
        st.markdown(f"Predicting pro presence on **{meta.get('target_patch')}** from the last pro patch "
                    f"(**{meta.get('base_patch')}**) plus every balance change since then. "
                    f"The model is trained on {meta.get('n_transitions')} observed patch transitions (all leagues).")
        if "backtest_metrics" in d:
            bm = d["backtest_metrics"].set_index(d["backtest_metrics"].columns[0])
            a, b, c3 = st.columns(3)
            row = bm.loc["changed champions"]
            a.metric("Backtest error — model", f"{row.mae_model * 100:.1f} pp", help="Mean absolute error in presence for changed champions, walk-forward")
            b.metric("Baseline: “nothing changes”", f"{row.mae_persistence * 100:.1f} pp")
            c3.metric("Direction right (changed, moved ≥5pp)", f"{row.direction_acc_model:.0%}")
            with st.expander("Full backtest table"):
                st.dataframe(bm.style.format("{:.3f}"), use_container_width=True)

        show = pr[(pr.n_changes > 0) | (pr.pred_delta_presence.abs() >= 0.05)].sort_values("pred_delta_presence")
        fig = go.Figure(go.Bar(
            x=show.pred_delta_presence, y=show.champion, orientation="h",
            marker=dict(color=[VERDICT_COLOR.get(v, NEUTRAL) for v in show.verdict], cornerradius=4),
            customdata=show[["prev_presence", "pred_presence", "verdict", "change_patches"]],
            hovertemplate="<b>%{y}</b> %{customdata[2]} %{customdata[3]}<br>%{customdata[0]:.0%} → %{customdata[1]:.0%}"
                          " (%{x:+.1%})<extra></extra>"))
        st.markdown("##### Predicted change in presence (champions with changes, or moving ≥5pp)")
        fig = style(fig, xaxis=dict(title="Predicted Δ presence", **PCT),
                    hovermode="closest", bargap=0.25)
        st.plotly_chart(row_layout(fig, show.champion), use_container_width=True)
        st.dataframe(with_icons(pr[["champion", "verdict", "change_patches", "prev_presence", "pred_presence",
                                    "pred_delta_presence"]])
                     .style.format({"prev_presence": "{:.0%}", "pred_presence": "{:.0%}", "pred_delta_presence": "{:+.1%}"}),
                     column_config=image_column(), hide_index=True, use_container_width=True)

        if "backtest_predictions" in d:
            bp = d["backtest_predictions"]
            bp = bp[bp.changed]
            st.markdown("##### Backtest: predicted vs actual for changed champions")
            fig = go.Figure()
            for v in ["buff", "nerf", "adjust"]:
                sub = bp[bp.verdict == v]
                fig.add_trace(go.Scatter(x=sub.pred_model, y=sub.delta_presence, mode="markers", name=v,
                                         marker=dict(color=VERDICT_COLOR[v], size=8, line=dict(width=1, color="white")),
                                         customdata=sub[["champion", "to_patch"]],
                                         hovertemplate="%{customdata[0]} %{customdata[1]}<br>pred %{x:+.1%} · actual %{y:+.1%}<extra></extra>"))
            lim = [min(bp.pred_model.min(), bp.delta_presence.min()), max(bp.pred_model.max(), bp.delta_presence.max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1), showlegend=False, hoverinfo="skip"))
            st.plotly_chart(style(fig, 420, xaxis=dict(title="Predicted Δ presence", **PCT),
                                  yaxis=dict(title="Actual Δ presence", **PCT), hovermode="closest"), use_container_width=True)

# --- Footer ------------------------------------------------------------------------
st.divider()
st.caption(
    "Pro match data from [Games of Legends (gol.gg)](https://gol.gg). Balance changes from Riot's official patch "
    "notes. Champion portraits from Riot's Data Dragon; champion metadata from Meraki Analytics.\n\n"
    "This is a fan project. It isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games "
    "or anyone officially involved in producing or managing Riot Games properties. Riot Games and all associated "
    "properties are trademarks or registered trademarks of Riot Games, Inc.")

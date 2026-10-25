"""Per-patch champion stats and patch-to-patch deltas joined with balance changes."""
import pandas as pd

from .champions import champ_key
from .config import MIN_GAMES_PER_PATCH, PROCESSED, patch_key

WINRATE_PRIOR_GAMES = 10  # shrink small-sample win rates toward 50%


def load_games() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "games.csv", dtype={"patch": str, "version": str, "series_id": str})


def load_changes() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "champion_changes.csv", dtype={"patch": str})


def champion_stats(games: pd.DataFrame, league: str = "ALL", min_games: int = MIN_GAMES_PER_PATCH) -> pd.DataFrame:
    """One row per (patch, champion): games on patch, picks, bans, rates."""
    if league != "ALL":
        games = games[games.league == league]
    n_games = games.groupby("patch").game_uid.nunique().rename("patch_games")

    picks = games.assign(champion=games.picks.str.split("|")).explode("champion")
    picks = picks.groupby(["patch", "champion"]).agg(picks=("game_uid", "size"), wins=("result", "sum"))
    bans = games.assign(champion=games.bans.fillna("").str.split("|")).explode("champion")
    bans = bans[bans.champion != ""].groupby(["patch", "champion"]).size().rename("bans")

    stats = picks.join(bans, how="outer").fillna(0).reset_index().join(n_games, on="patch")
    stats = stats[stats.patch_games >= min_games].copy()
    stats["champ_key"] = stats.champion.map(champ_key)
    stats["pick_rate"] = stats.picks / stats.patch_games
    stats["ban_rate"] = stats.bans / stats.patch_games
    stats["presence"] = (stats.picks + stats.bans) / stats.patch_games
    stats["win_rate"] = (stats.wins + WINRATE_PRIOR_GAMES / 2) / (stats.picks + WINRATE_PRIOR_GAMES)
    stats["league"] = league
    return stats.sort_values(["patch", "presence"], ascending=[True, False],
                             key=lambda s: s.map(patch_key) if s.name == "patch" else s)


def pro_patches(stats: pd.DataFrame) -> list[str]:
    return sorted(stats.patch.unique(), key=patch_key)


def changes_between(changes: pd.DataFrame, after: str | None, upto: str) -> pd.DataFrame:
    """Sum champion changes for Riot patches in (after, upto]."""
    lo = patch_key(after) if after else (0, 0)
    hi = patch_key(upto)
    window = changes[changes.patch.map(lambda p: lo < patch_key(p) <= hi)]
    if window.empty:
        return pd.DataFrame(columns=["champ_key", "change_score", "n_changes", "pro_targeted",
                                     "base_stats", "n_lines", "change_patches", "verdict"])
    agg = window.groupby("champ_key").agg(
        change_score=("score", "sum"), n_changes=("score", "size"),
        pro_targeted=("pro_targeted", "max"), base_stats=("base_stats", "max"),
        n_lines=("n_lines", "sum"),
        change_patches=("patch", lambda s: ", ".join(sorted(set(s), key=patch_key))),
    ).reset_index()
    agg["verdict"] = pd.cut(agg.change_score, [-99, -0.25, 0.25, 99], labels=["nerf", "adjust", "buff"]).astype(str)
    return agg


def _frame(stats: pd.DataFrame, patch: str, keys: pd.Index) -> pd.DataFrame:
    cols = ["presence", "pick_rate", "ban_rate", "win_rate", "picks"]
    f = stats[stats.patch == patch].set_index("champ_key")[cols].reindex(keys)
    f[["presence", "pick_rate", "ban_rate", "picks"]] = f[["presence", "pick_rate", "ban_rate", "picks"]].fillna(0)
    f["win_rate"] = f.win_rate.fillna(0.5)
    return f


def build_transitions(stats: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """One row per champion per consecutive pair of pro patches.

    The 'prev_*' columns describe the champion before the change window and
    are what's available at prediction time; 'delta_*' are the outcomes.
    """
    patches = pro_patches(stats)
    names = stats.drop_duplicates("champ_key").set_index("champ_key").champion
    names = pd.concat([names, changes.drop_duplicates("champ_key").set_index("champ_key").champion])
    names = names[~names.index.duplicated()]
    keys = pd.Index(sorted(set(stats.champ_key) | set(changes.champ_key)), name="champ_key")

    rows = []
    for i in range(1, len(patches)):
        prev, cur = patches[i - 1], patches[i]
        before = _frame(stats, prev, keys).add_prefix("prev_")
        after = _frame(stats, cur, keys).add_prefix("next_")
        ch = changes_between(changes, prev, cur).set_index("champ_key").reindex(keys)
        # Momentum / lagged effects from the window before this one.
        if i >= 2:
            older = _frame(stats, patches[i - 2], keys)
            before["prev_delta_presence"] = before.prev_presence - older.presence
            lag = changes_between(changes, patches[i - 2], prev).set_index("champ_key").reindex(keys)
            before["lag_change_score"] = lag.change_score.fillna(0)
        else:
            before["prev_delta_presence"] = 0.0
            before["lag_change_score"] = 0.0
        t = before.join(after).join(ch)
        t["change_score"] = t.change_score.fillna(0.0)
        t["n_changes"] = t.n_changes.fillna(0)
        t["n_lines"] = t.n_lines.fillna(0)
        t["pro_targeted"] = t.pro_targeted.eq(True)
        t["base_stats"] = t.base_stats.eq(True)
        t["verdict"] = t.verdict.fillna("none")
        t["changed"] = t.n_changes > 0
        t["from_patch"], t["to_patch"] = prev, cur
        t["champion"] = names.reindex(keys).values
        for m in ["presence", "pick_rate", "ban_rate", "win_rate"]:
            t[f"delta_{m}"] = t[f"next_{m}"] - t[f"prev_{m}"]
        # Skip champions that were irrelevant on both sides and untouched.
        t = t[(t.prev_presence > 0) | (t.next_presence > 0) | t.changed]
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def verdict_effects(transitions: pd.DataFrame) -> pd.DataFrame:
    """Average outcome by type of change (including 'none' as a control group)."""
    g = transitions.groupby("verdict")
    return pd.DataFrame({
        "n": g.size(),
        "mean_delta_presence": g.delta_presence.mean(),
        "median_delta_presence": g.delta_presence.median(),
        "share_presence_up": g.delta_presence.apply(lambda s: (s > 0).mean()),
        "mean_delta_win_rate": g.delta_win_rate.mean(),
    }).sort_values("mean_delta_presence")


def build_all(log=print) -> dict[str, pd.DataFrame]:
    games, changes = load_games(), load_changes()
    leagues = ["ALL"] + sorted(games.league.unique())
    stats = pd.concat([champion_stats(games, lg) for lg in leagues], ignore_index=True)
    stats.to_csv(PROCESSED / "champion_patch_stats.csv", index=False)
    overall = stats[stats.league == "ALL"]
    transitions = build_transitions(overall, changes)
    transitions.to_csv(PROCESSED / "transitions.csv", index=False)
    effects = verdict_effects(transitions)
    effects.to_csv(PROCESSED / "verdict_effects.csv")
    log(f"Pro patches: {', '.join(pro_patches(overall))}")
    log(effects.round(3).to_string())
    return dict(stats=stats, transitions=transitions, effects=effects)

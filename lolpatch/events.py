"""International events as out-of-sample tests of the patch model.

The model only ever trains on domestic league games (games.csv). For each
event we forecast presence on the event's patch from the last domestic patch
plus the balance changes in between, freeze that forecast to disk, and score
it against what actually happened at the event.

For Worlds, which is still in progress, we also keep a live estimate for the
rest of the tournament: the frozen forecast is treated as a prior worth
PRIOR_GAMES games of evidence and updated with the games played so far.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr

from .analyze import build_transitions, champion_stats, load_changes, load_games
from .config import DATA, PROCESSED, patch_key
from .model import predict_future
from .scrape_matches import scrape_tournaments

FORECASTS = DATA / "forecasts"
PRIOR_GAMES = 20
MOVE_THRESHOLD = 0.05

# In date order. "train" events calibrate the international adjustment;
# "test" events are only ever scored.
EVENTS = {
    "first_stand": dict(title="First Stand 2026", tournaments=["2026 First Stand"], live=False, role="train"),
    "msi": dict(title="MSI 2026", tournaments=["MSI 2026"], live=False, role="train"),
    "ewc": dict(title="Esports World Cup 2026", tournaments=["Esports World Cup 2026"], live=False, role="train"),
    "demacia": dict(title="Demacia Cup 2026", tournaments=["Demacia Cup Global Invitational 2026"],
                    live=False, role="test"),
    "worlds": dict(title="Worlds 2026", tournaments=["Worlds 2026 Play-In", "Worlds 2026 Main Event"],
                   live=True, role="test"),
}


def _stage_group(row) -> str:
    stage = row.stage.upper()
    if row.tournament.endswith("Play-In") or stage.startswith("PLAY-IN"):
        return "Play-In"
    if stage.startswith("GROUP"):
        return "Groups"
    if stage.startswith("ROUND"):
        return f"Swiss {stage.replace('ROUND', 'R')}" if "Worlds" in row.tournament or "MSI" in row.tournament \
            else f"Round {stage.replace('ROUND', '')}"
    if "PALCE" in stage or "PLACE" in stage:  # gol.gg spells it "3RD.PALCE"
        return "3rd Place"
    return row.stage.title()


def scrape(refresh: bool = True, log=print) -> pd.DataFrame:
    frames = []
    for key, ev in EVENTS.items():
        log(f"{ev['title']}:")
        df = scrape_tournaments(ev["tournaments"], league_fn=lambda t, k=key: k, refresh=refresh,
                                refetch_recent_days=2 if ev["live"] else 0, log=log)
        if len(df):
            df["stage_group"] = df.apply(_stage_group, axis=1)
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games.to_csv(PROCESSED / "event_games.csv", index=False)
    log(f"Saved {games.game_uid.nunique()} event games")
    return games


def load_event_games() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str, "version": str, "series_id": str})


def event_patch(games: pd.DataFrame) -> str:
    return games.patch.mode().iloc[0]


def forecast(key: str, patch: str, version: str = "frozen", log=print) -> pd.DataFrame:
    """Event forecast built only from domestic games and patch notes.

    "frozen" is created once and never overwritten: it's the blind test.
    "current" is rebuilt every run with the latest classifier/model, so it
    benefits from fixes made after seeing results and is not a blind test.
    """
    path = FORECASTS / f"{key}_{patch}_{version}.csv"
    if path.exists() and version == "frozen":
        return pd.read_csv(path, dtype={"base_patch": str, "target_patch": str})
    games, changes = load_games(), load_changes()
    # As of the event: only domestic games from patches before it.
    games = games[games.patch.map(patch_key) < patch_key(patch)]
    stats = champion_stats(games, "ALL")
    transitions = build_transitions(stats, changes)
    f = predict_future(stats, changes, transitions, patch)
    f["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    FORECASTS.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)
    if version == "frozen":
        log(f"  froze forecast for {key} on {patch} -> {path.name}")
    return f


def actual_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Presence etc. at the event (all its games pooled), regardless of sample size."""
    s = champion_stats(games.assign(patch=event_patch(games)), "ALL", min_games=1)
    return s.drop(columns=["patch", "league"])


def compare(fc: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """One row per champion seen in either the forecast or the event."""
    a = act.set_index("champ_key")[["champion", "picks", "bans", "wins", "patch_games", "presence",
                                    "pick_rate", "ban_rate", "win_rate"]]
    f = fc.set_index("champ_key")
    keys = f.index.union(a.index)
    out = pd.DataFrame(index=keys)
    out["champion"] = f.champion.reindex(keys).fillna(a.champion.reindex(keys))
    out["verdict"] = f.verdict.reindex(keys).fillna("none")
    out["change_patches"] = f.change_patches.reindex(keys).fillna("")
    out["domestic_presence"] = f.prev_presence.reindex(keys).fillna(0)
    out["pred_presence"] = f.pred_presence.reindex(keys).fillna(0)
    out["actual_presence"] = a.presence.reindex(keys).fillna(0)
    for c in ["picks", "bans", "wins"]:
        out[c] = a[c].reindex(keys).fillna(0).astype(int)
    out["games"] = int(a.patch_games.max()) if len(a) else 0
    out["actual_win_rate"] = a.win_rate.reindex(keys)
    out["changed"] = out.verdict != "none"
    out["error_model"] = out.pred_presence - out.actual_presence
    out["error_baseline"] = out.domestic_presence - out.actual_presence
    return out.reset_index().sort_values("actual_presence", ascending=False)


def score(cmp: pd.DataFrame) -> dict:
    def block(df):
        moved = df[(df.actual_presence - df.domestic_presence).abs() >= MOVE_THRESHOLD]
        pred_dir = np.sign(moved.pred_presence - moved.domestic_presence)
        true_dir = np.sign(moved.actual_presence - moved.domestic_presence)
        return dict(
            n=len(df),
            mae_model=df.error_model.abs().mean(),
            mae_baseline=df.error_baseline.abs().mean(),
            direction_acc=(pred_dir == true_dir).mean() if len(moved) else np.nan,
            n_moved=len(moved),
        )

    top = 10
    pred_top = set(cmp.nlargest(top, "pred_presence").champ_key)
    base_top = set(cmp.nlargest(top, "domestic_presence").champ_key)
    act_top = set(cmp.nlargest(top, "actual_presence").champ_key)
    return dict(
        games=int(cmp.games.max()),
        all=block(cmp),
        changed=block(cmp[cmp.changed]),
        spearman_model=spearmanr(cmp.pred_presence, cmp.actual_presence).statistic,
        spearman_baseline=spearmanr(cmp.domestic_presence, cmp.actual_presence).statistic,
        top10_model=len(pred_top & act_top),
        top10_baseline=len(base_top & act_top),
    )


def live_estimate(cmp: pd.DataFrame, prior_games: int = PRIOR_GAMES) -> pd.DataFrame:
    """Blend the forecast (as a prior) with observed event games: Beta-binomial on presence."""
    n = cmp.games.max()
    hits = cmp.picks + cmp.bans
    prior = cmp.pred_presence.clip(0.005, 0.995)
    a = hits + prior_games * prior
    b = (n - hits).clip(lower=0) + prior_games * (1 - prior)
    out = cmp[["champ_key", "champion", "verdict", "change_patches", "domestic_presence",
               "pred_presence", "actual_presence", "picks", "bans", "games"]].copy()
    out["live_presence"] = a / (a + b)
    out["live_low"] = beta.ppf(0.1, a, b)
    out["live_high"] = beta.ppf(0.9, a, b)
    return out.sort_values("live_presence", ascending=False)


def adjustment_test(games: pd.DataFrame) -> pd.DataFrame:
    """Does an 'international' tweak to the domestic model help?

    Candidate: drop the model's drift (mean reversion/momentum) and keep only
    the patch-note effect, on the theory that the meta core holds at events.
    Scored on every event; the decision is made on the train events only.
    """
    rows = []
    for key, ev in EVENTS.items():
        eg = games[games.league == key]
        if eg.empty:
            continue
        patch = event_patch(eg)
        fc = forecast(key, patch, "current", log=lambda *a: None).set_index("champ_key")
        act = actual_stats(eg).set_index("champ_key")
        keys = fc.index.union(act.index)
        dom = fc.prev_presence.reindex(keys).fillna(0)
        drift = fc.pred_drift.reindex(keys).fillna(0)
        effect = fc.pred_change_effect.reindex(keys).fillna(0)
        y = act.presence.reindex(keys).fillna(0)
        top = dom > 0.6
        rows.append(dict(
            event=ev["title"], role=ev["role"], games=int(eg.game_uid.nunique()),
            mae_nothing_changes=(dom - y).abs().mean(),
            mae_model=((dom + drift + effect).clip(0, 1) - y).abs().mean(),
            mae_model_no_drift=((dom + effect).clip(0, 1) - y).abs().mean(),
            n_top=int(top.sum()),
            top_actual_change=(y - dom)[top].mean(),
            top_model_drift=drift[top].mean(),
        ))
    return pd.DataFrame(rows)


def run(refresh: bool = True, log=print) -> dict:
    games = scrape(refresh=refresh, log=log)
    results = {}
    for key, ev in EVENTS.items():
        eg = games[games.league == key]
        if eg.empty:
            log(f"{ev['title']}: no games yet")
            continue
        patch = event_patch(eg)
        act = actual_stats(eg)
        results[key] = dict(title=ev["title"], patch=patch, live=ev["live"],
                            stages=eg.groupby("stage_group").game_uid.nunique().to_dict())
        for version in ["frozen", "current"]:
            fc = forecast(key, patch, version, log=log)
            cmp = compare(fc, act)
            cmp.to_csv(PROCESSED / f"event_{key}_{version}_comparison.csv", index=False)
            sc = score(cmp)
            sc.update(base_patch=fc.base_patch.iloc[0], forecast_created=fc.created_at.iloc[0])
            results[key][version] = sc
            if ev["live"] and version == "current":
                live_estimate(cmp).to_csv(PROCESSED / f"event_{key}_live.csv", index=False)
            log(f"\n{ev['title']} [{version} forecast] (patch {patch}, {sc['games']} games; "
                f"from domestic {sc['base_patch']}, made {sc['forecast_created']}):")
            log(f"  presence error: model {sc['all']['mae_model']:.1%} vs 'nothing changes' {sc['all']['mae_baseline']:.1%}"
                f"  | changed champions: {sc['changed']['mae_model']:.1%} vs {sc['changed']['mae_baseline']:.1%}")
            log(f"  direction right (changed, moved >=5pp): {sc['changed']['direction_acc']:.0%} of {sc['changed']['n_moved']}")
            log(f"  rank correlation: model {sc['spearman_model']:.2f} vs baseline {sc['spearman_baseline']:.2f}"
                f"  | top-10 overlap: model {sc['top10_model']}/10 vs baseline {sc['top10_baseline']}/10")
    (PROCESSED / "event_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    adj = adjustment_test(games)
    adj.to_csv(PROCESSED / "event_adjustment_test.csv", index=False)
    log("\nInternational adjustment test (drop drift, keep patch-note effect):")
    log(adj.round(3).to_string(index=False))
    return results

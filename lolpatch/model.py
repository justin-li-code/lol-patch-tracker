"""Predict next-patch pro presence from balance changes.

Target: change in presence (picks + bans per game) from the last pro patch
before a change window to the first pro patch after it.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .analyze import build_transitions, changes_between, champion_stats, load_changes, load_games, pro_patches
from .config import PROCESSED, patch_key

FEATURES = [
    "change_score", "n_changes", "n_lines", "pro_targeted", "base_stats",
    "prev_presence", "prev_pick_rate", "prev_ban_rate", "prev_win_rate",
    "prev_delta_presence", "lag_change_score",
]
TARGET = "delta_presence"
MIN_TRAIN_TRANSITIONS = 4


def _X(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(float).copy()
    # A nerf hurts a 90%-presence champion more than a 5% one; a buff has more room to work at 5%.
    buff, nerf = X.change_score.clip(lower=0), X.change_score.clip(upper=0)
    X["buff_x_presence"] = buff * X.prev_presence
    X["nerf_x_presence"] = nerf * X.prev_presence
    X["change_x_room"] = X.change_score * (1 - X.prev_presence)
    X["change_x_pro"] = X.change_score * X.pro_targeted
    return X.values


def make_model():
    # Deliberately small: one season is ~1,500 rows, and bigger models lost to
    # the "nothing changes" baseline in the walk-forward backtest.
    return HistGradientBoostingRegressor(
        max_iter=80, learning_rate=0.05, max_depth=2, min_samples_leaf=40,
        l2_regularization=5.0, random_state=0,
    )


def walk_forward(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on transitions before each target patch, predict it. No look-ahead."""
    order = sorted(transitions.to_patch.unique(), key=patch_key)
    preds = []
    for i, target in enumerate(order):
        if i < MIN_TRAIN_TRANSITIONS:
            continue
        train = transitions[transitions.to_patch.isin(order[:i])]
        test = transitions[transitions.to_patch == target].copy()
        model = make_model().fit(_X(train), train[TARGET])
        # Mean-reversion-only baseline: ignores the patch notes entirely.
        mr = Ridge(alpha=1.0).fit(train[["prev_presence", "prev_delta_presence"]], train[TARGET])
        test["pred_model"] = model.predict(_X(test))
        test["pred_reversion"] = mr.predict(test[["prev_presence", "prev_delta_presence"]])
        test["pred_persistence"] = 0.0
        preds.append(test)
    preds = pd.concat(preds, ignore_index=True)

    def score(df):
        out = {}
        for name in ["model", "reversion", "persistence"]:
            err = df[f"pred_{name}"] - df[TARGET]
            out[f"mae_{name}"] = err.abs().mean()
            moved = df[df[TARGET].abs() >= 0.05]
            out[f"direction_acc_{name}"] = (np.sign(moved[f"pred_{name}"]) == np.sign(moved[TARGET])).mean() \
                if name != "persistence" else np.nan
        out["n"] = len(df)
        return pd.Series(out)

    metrics = pd.DataFrame({
        "all champions": score(preds),
        "changed champions": score(preds[preds.changed]),
        "buffed": score(preds[preds.verdict == "buff"]),
        "nerfed": score(preds[preds.verdict == "nerf"]),
    }).T
    return preds, metrics


def predict_future(stats: pd.DataFrame, changes: pd.DataFrame, transitions: pd.DataFrame,
                   target_patch: str, extra_changes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predict presence on `target_patch` from the latest pro patch plus all
    known balance changes since then (scraped notes and/or a user CSV)."""
    patches = pro_patches(stats)
    base = patches[-1]
    if patch_key(target_patch) <= patch_key(base):
        raise ValueError(f"Target {target_patch} must be after the latest pro patch {base}")
    if extra_changes is not None and len(extra_changes):
        changes = pd.concat([changes[~changes.patch.isin(extra_changes.patch)], extra_changes], ignore_index=True)

    # Same feature construction as build_transitions, with the future patch as "next".
    keys = pd.Index(sorted(set(stats.champ_key) | set(changes.champ_key)), name="champ_key")
    cur = stats[stats.patch == base].set_index("champ_key").reindex(keys)
    older = stats[stats.patch == patches[-2]].set_index("champ_key").reindex(keys)
    X = pd.DataFrame(index=keys)
    X["prev_presence"] = cur.presence.fillna(0)
    X["prev_pick_rate"] = cur.pick_rate.fillna(0)
    X["prev_ban_rate"] = cur.ban_rate.fillna(0)
    X["prev_win_rate"] = cur.win_rate.fillna(0.5)
    X["prev_delta_presence"] = X.prev_presence - older.presence.fillna(0)
    lag = changes_between(changes, patches[-2], base).set_index("champ_key").reindex(keys)
    X["lag_change_score"] = lag.change_score.fillna(0)
    ch = changes_between(changes, base, target_patch).set_index("champ_key").reindex(keys)
    X["change_score"] = ch.change_score.fillna(0)
    X["n_changes"] = ch.n_changes.fillna(0)
    X["n_lines"] = ch.n_lines.fillna(0)
    X["pro_targeted"] = ch.pro_targeted.eq(True)
    X["base_stats"] = ch.base_stats.eq(True)
    X["verdict"] = ch.verdict.fillna("none")
    X["change_patches"] = ch.change_patches.fillna("")

    model = make_model().fit(_X(transitions), transitions[TARGET])
    X["pred_delta_presence"] = model.predict(_X(X))
    # Same champion with no balance changes: isolates the patch-note effect from
    # the drift (mean reversion, momentum) the model predicts anyway.
    untouched = X.assign(change_score=0.0, n_changes=0.0, n_lines=0.0, pro_targeted=False, base_stats=False)
    X["pred_drift"] = model.predict(_X(untouched))
    X["pred_change_effect"] = X.pred_delta_presence - X.pred_drift
    X["pred_presence"] = (X.prev_presence + X.pred_delta_presence).clip(0, 1)
    names = pd.concat([stats.drop_duplicates("champ_key").set_index("champ_key").champion,
                       changes.drop_duplicates("champ_key").set_index("champ_key").champion])
    X["champion"] = names[~names.index.duplicated()].reindex(keys).values
    X["base_patch"], X["target_patch"] = base, target_patch
    X = X[(X.prev_presence > 0) | (X.n_changes > 0)]
    return X.reset_index().sort_values("pred_delta_presence", ascending=False)


def run(target_patch: str | None = None, extra_changes: pd.DataFrame | None = None, log=print):
    games, changes = load_games(), load_changes()
    stats = champion_stats(games, "ALL")
    transitions = build_transitions(stats, changes)
    preds, metrics = walk_forward(transitions)
    preds.to_csv(PROCESSED / "backtest_predictions.csv", index=False)
    metrics.to_csv(PROCESSED / "backtest_metrics.csv")
    log("Walk-forward backtest (presence = picks+bans per game, 0-1 scale):")
    log(metrics.round(3).to_string())

    # Interpretable view of what drives presence changes: ridge on standardised features.
    Xs = transitions[FEATURES].astype(float)
    Xs = (Xs - Xs.mean()) / Xs.std().replace(0, 1)
    coefs = pd.Series(Ridge(alpha=1.0).fit(Xs, transitions[TARGET]).coef_, index=FEATURES).sort_values()
    coefs.to_csv(PROCESSED / "feature_effects.csv", header=["std_coef"])

    if target_patch is None:
        latest_notes = max(changes.patch.unique(), key=patch_key)
        target_patch = latest_notes
    future = predict_future(stats, changes, transitions, target_patch, extra_changes)
    future.to_csv(PROCESSED / "predictions.csv", index=False)
    log(f"\nPredictions for {target_patch} (from pro baseline {future.base_patch.iloc[0]}):")
    cols = ["champion", "verdict", "change_patches", "prev_presence", "pred_presence", "pred_delta_presence"]
    log("Biggest predicted risers:\n" + future.head(10)[cols].round(3).to_string(index=False))
    log("Biggest predicted fallers:\n" + future.tail(10)[cols].round(3).iloc[::-1].to_string(index=False))
    (PROCESSED / "model_meta.json").write_text(json.dumps(dict(
        target_patch=target_patch, base_patch=future.base_patch.iloc[0],
        n_transitions=int(transitions.to_patch.nunique()), n_rows=len(transitions),
    ), indent=2))
    return dict(preds=preds, metrics=metrics, future=future, coefs=coefs)

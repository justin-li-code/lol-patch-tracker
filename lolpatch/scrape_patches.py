"""Scrape Riot patch notes and turn champion changes into buff/nerf scores.

Outputs:
    data/processed/patch_changes.csv  one row per individual change line
    data/processed/champion_changes.csv  one row per champion per patch, with
        verdict (buff / nerf / adjust), score in [-1, 1] and flags.

Riot does not label changes as buffs or nerfs, so we infer it from the
numbers (X => Y, with "lower is better" stats such as cooldowns flipped)
and from the designer context blurb. Anything wrong can be corrected in
data/overrides/champion_overrides.csv, which always wins.
"""
import re

import pandas as pd
from bs4 import BeautifulSoup

from . import http
from .champions import champ_key
from .config import OVERRIDES, PATCH_BRAND_MAJOR, PROCESSED

URL_PATTERNS = [
    "https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-{major}-{minor}-notes/",
    "https://www.leagueoflegends.com/en-us/news/game-updates/patch-{major}-{minor}-notes/",
]
ARROW = "⇒"

# Stats where a smaller number is better for the champion.
LOWER_IS_BETTER = re.compile(
    # Only resource *costs*: base mana / mana regen are higher-is-better.
    r"cooldown|\bcost\b|cast time|windup|wind-up|channel|delay|"
    r"recharge|lockout|charge time|damage taken|self[- ]damage|respawn|attack timer|"
    r"time to|ammo recharge|refund threshold|missile speed reduction",
    re.I,
)
NERF_CUES = re.compile(
    r"too strong|too oppressive|oppressive|overperform|dominan|dominat|toning (it |him |her |them )?down|"
    r"ta[mp]p? down|work (a little |a bit )?harder|"
    r"tone (it |him |her |them )?down|pulling back|pull back|reining|rein in|"
    r"frustrat|too safe|too reliable|too good|\bnerf|outlier|bring (him|her|them|it) down|"
    r"high win ?rate|problematic|reduce (his|her|their|its) power|power down|ease off|"
    r"overshot|overtuned|over-tuned|dial(ing)? back|in line|in check|too high",
    re.I,
)
BUFF_CUES = re.compile(
    # Not bare "weak": "weaker for average players" is how Riot justifies pro-targeted nerfs.
    r"struggl|underperform|\bweak\b|too weak|weaker than|feels? weak|lacklustre|lackluster|\bbuff|help (him|her|them|it)|boost|"
    r"falling behind|low win ?rate|not performing|more power|extra power|give (him|her|them|it) (some|a bit|more)|"
    r"back up|bump|in a rough spot|rough place|needs? (some )?help|left behind|underwhelming",
    re.I,
)
PRO_CUES = re.compile(r"\bpro\b|pro play|pro-play|competitive|esports|professional|organized play|\bmsi\b|worlds", re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def patch_url(patch: str) -> str | None:
    major, minor = patch.split(".")
    for pattern in URL_PATTERNS:
        url = pattern.format(major=major, minor=minor)
        if http.get(url):
            return url
    return None


def _numbers(text: str) -> list[float]:
    # Drop parenthetical scaling notes like "(based on level)" but keep "(+ 50% AP)".
    text = re.sub(r"\((?:based on|at|levels?|ranks?)[^)]*\)", "", text, flags=re.I)
    return [float(x) for x in NUM.findall(text)]


def score_line(stat: str, before: str, after: str, ability: str = "") -> tuple[float, float]:
    """Return (direction in {-1, 0, 1}, relative magnitude) for one 'X => Y' line."""
    a, b = _numbers(before), _numbers(after)
    if not a or not b:
        return 0.0, 0.0
    if len(a) == len(b):
        rel = [(y - x) / abs(x) for x, y in zip(a, b) if x != 0]
        if not rel:
            rel = [y - x for x, y in zip(a, b)]
        delta = sum(rel) / len(rel)
    else:
        mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
        delta = (mean_b - mean_a) / abs(mean_a) if mean_a else mean_b - mean_a
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    sign = 1.0 if delta > 0 else -1.0
    # Under an ability heading, a bare "Mana"/"Energy"/"Health" line is that ability's cost.
    is_cost = stat.strip().lower() in ("mana", "energy", "health") and "base stats" not in ability.lower()
    if LOWER_IS_BETTER.search(stat) or is_cost:
        sign = -sign
    return sign, min(abs(delta), 1.0)


def _section_blocks(soup: BeautifulSoup):
    """Yield champion change blocks from the Champions section(s) only."""
    for h2 in soup.find_all("h2"):
        hid = (h2.get("id") or "").lower()
        title = h2.get_text(" ", strip=True).lower()
        if not (hid == "patch-champions" or title in ("champions", "champion changes")):
            continue
        header = h2.find_parent("header") or h2
        for sib in header.find_next_siblings():
            if sib.name == "header" or sib.find("h2"):
                break
            for block in sib.select(".patch-change-block"):
                yield block


def parse_patch(patch: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for block in _section_blocks(soup):
        title = block.find("h3", class_="change-title")
        if not title:
            continue
        champion = title.get_text(" ", strip=True)
        context = " ".join(q.get_text(" ", strip=True) for q in block.find_all("blockquote"))
        ability = "General"
        for el in block.find_all(["h4", "li"]):
            if el.name == "h4":
                ability = el.get_text(" ", strip=True)
                continue
            text = el.get_text(" ", strip=True)
            stat, _, change = text.partition(":")
            if ARROW in change:
                before, _, after = change.partition(ARROW)
                direction, magnitude = score_line(stat, before, after, ability)
            else:
                stat, change, before, after = text, "", "", ""
                direction, magnitude = 0.0, 0.0
                low = text.lower()
                if low.startswith(("new", "added")):
                    direction = 0.0  # new effects are usually buffs but too ambiguous to auto-score
            rows.append(dict(patch=patch, champion=champion, champ_key=champ_key(champion),
                             ability=ability, stat=stat.strip(), before=before.strip(),
                             after=after.strip(), text=text, direction=direction,
                             magnitude=magnitude, context=context))
        if not block.find_all("li"):
            rows.append(dict(patch=patch, champion=champion, champ_key=champ_key(champion),
                             ability="General", stat="", before="", after="", text="",
                             direction=0.0, magnitude=0.0, context=context))
    return rows


def summarise(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse change lines into one verdict per champion per patch."""
    out = []
    for (patch, key), g in lines.groupby(["patch", "champ_key"], sort=False):
        context = g.context.iloc[0] or ""
        scored = g[g.direction != 0]
        numeric = scored.direction.mean() if len(scored) else 0.0
        weighted = (scored.direction * (0.5 + scored.magnitude)).sum()
        nerf_cue = bool(NERF_CUES.search(context))
        buff_cue = bool(BUFF_CUES.search(context))
        text = (1 if buff_cue else 0) - (1 if nerf_cue else 0)
        # Unanimous numbers decide the direction; the blurb (which often mentions past
        # buffs/nerfs) can only soften them. With mixed numbers the blurb breaks the tie.
        text_weight = 0.25 if abs(numeric) == 1 else 0.5
        score = max(-1.0, min(1.0, 0.75 * numeric + text_weight * text))
        verdict = "buff" if score > 0.25 else "nerf" if score < -0.25 else "adjust"
        out.append(dict(
            patch=patch, champion=g.champion.iloc[0], champ_key=key, verdict=verdict,
            score=round(score, 3), weighted=round(weighted, 3), n_lines=len(g),
            n_buff_lines=int((g.direction > 0).sum()), n_nerf_lines=int((g.direction < 0).sum()),
            pro_targeted=bool(PRO_CUES.search(context)),
            base_stats=bool(g.ability.str.contains("Base Stats", case=False).any()),
            context=context, source="auto",
        ))
    return pd.DataFrame(out)


def apply_overrides(changes: pd.DataFrame) -> pd.DataFrame:
    path = OVERRIDES / "champion_overrides.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("patch,champion,verdict,score,note\n")
        return changes
    ov = pd.read_csv(path, dtype={"patch": str})
    if ov.empty:
        return changes
    ov["champ_key"] = ov.champion.map(champ_key)
    changes = changes.copy()
    for _, o in ov.iterrows():
        mask = (changes.patch == o.patch) & (changes.champ_key == o.champ_key)
        default_score = {"buff": 0.75, "nerf": -0.75, "adjust": 0.0}[o.verdict]
        score = o.score if pd.notna(o.get("score")) else default_score
        if mask.any():
            changes.loc[mask, ["verdict", "score", "source"]] = [o.verdict, score, "override"]
        else:
            changes = pd.concat([changes, pd.DataFrame([dict(
                patch=o.patch, champion=o.champion, champ_key=o.champ_key, verdict=o.verdict,
                score=score, weighted=score, n_lines=0, n_buff_lines=0, n_nerf_lines=0,
                pro_targeted=False, base_stats=False, context=str(o.get("note") or ""),
                source="override")])], ignore_index=True)
    return changes


def scrape(first: int = 1, last: int = 24, refresh: bool = False, log=print) -> pd.DataFrame:
    lines = []
    for minor in range(first, last + 1):
        patch = f"{PATCH_BRAND_MAJOR}.{minor}"
        url = patch_url(patch)
        if not url:
            continue
        html = http.get(url, refresh=refresh)
        rows = parse_patch(patch, html)
        log(f"  {patch}: {len({r['champ_key'] for r in rows})} champions changed")
        lines.extend(rows)
    lines = pd.DataFrame(lines)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    lines.to_csv(PROCESSED / "patch_changes.csv", index=False)
    changes = apply_overrides(summarise(lines))
    changes.to_csv(PROCESSED / "champion_changes.csv", index=False)
    log(f"Saved {len(changes)} champion changes across {changes.patch.nunique()} patches")
    return changes


def changes_from_url(url: str, patch: str) -> pd.DataFrame:
    """Parse an arbitrary (e.g. newly released) patch notes page."""
    html = http.get(url, refresh=True)
    return summarise(pd.DataFrame(parse_patch(patch, html)))

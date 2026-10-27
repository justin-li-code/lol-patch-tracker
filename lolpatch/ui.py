"""Champion portraits and team logos for the Streamlit dashboard."""
import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from .assets import ASSETS, CHAMPIONS, LOGOS
from .champions import champ_key

ROW_PX = 28        # height of one category row in champion charts


@lru_cache(maxsize=1)
def _champion_files() -> dict:
    path = ASSETS / "champion_icons.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=1)
def _logo_files() -> dict:
    path = ASSETS / "team_logos.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _path(name: str, kind: str) -> Path | None:
    if not isinstance(name, str):
        return None
    if kind == "champion":
        fname = _champion_files().get(champ_key(name))
        return CHAMPIONS / fname if fname and (CHAMPIONS / fname).exists() else None
    fname = _logo_files().get(name)
    return LOGOS / fname if fname and (LOGOS / fname).exists() else None


@lru_cache(maxsize=1024)
def _render(path: Path, size: int, pad_right: int, rounded: bool) -> str:
    """PNG data URI: image fitted into size x size, optional rounded corners, transparent right padding."""
    img = Image.open(path).convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    square.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
    if rounded:
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 5, fill=255)
        square.putalpha(Image.composite(square.getchannel("A"), mask, mask))
    canvas = Image.new("RGBA", (size + pad_right, size), (0, 0, 0, 0))
    canvas.paste(square, (0, 0), square)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=1)
def _display_names() -> dict:
    path = ASSETS / "meraki_champions.json"
    if not path.exists():
        return {}
    return {champ_key(c["name"]): c["name"] for c in json.loads(path.read_text()).values()}


def nice(name):
    """Official spelling for display: gol.gg writes "KSante", "KhaZix"; players expect "K'Sante", "Kha'Zix"."""
    if not isinstance(name, str):
        return name
    return _display_names().get(champ_key(name), name)


def champ_icon(name: str, size: int = 64) -> str | None:
    p = _path(name, "champion")
    return _render(p, size, 0, True) if p else None


def team_logo(team: str, size: int = 64) -> str | None:
    p = _path(team, "team")
    return _render(p, size, 0, False) if p else None


def with_icons(df: pd.DataFrame, col: str = "champion", kind: str = "champion", label: str = " ") -> pd.DataFrame:
    """Insert an image column right before `col` (use with image_column)."""
    fn = champ_icon if kind == "champion" else team_logo
    out = df.copy()
    out.insert(out.columns.get_loc(col), label, out[col].map(fn))
    if kind == "champion":
        out[col] = out[col].map(nice)
    return out


def image_column(label: str = " "):
    import streamlit as st
    return {label: st.column_config.ImageColumn(label, width="small")}



def row_layout(fig, names, *, legend: bool = False, x_title: bool = True, x_top: bool = False,
               right: int = 16, reverse: bool = False):
    """Lay out a categorical-y chart (one champion per row) with even, fixed-height rows.

    Rows are ROW_PX tall and the top/bottom margins hold exactly the legend /
    x-axis title, so charts sit snugly under their headings. The left margin
    grows to fit the names (Plotly measures them in the browser's own font).
    No portraits here: browsers lay out tick labels differently enough that
    images beside them can overlap the names.
    """
    names = [nice(x) for x in names]
    # Rename the traces' row values the same way so they still match the category order.
    for trace in fig.data:
        if trace.y is not None and len(trace.y) and isinstance(trace.y[0], str):
            trace.y = [nice(v) for v in trace.y]
    n = max(len(names), 1)
    top = (34 if legend else 8) + (24 if x_top else 0)
    bottom = 8 if x_top else (46 if x_title else 24)
    fig.update_layout(
        height=n * ROW_PX + top + bottom,
        margin=dict(l=8, r=right, t=top, b=bottom),
        legend=dict(orientation="h", x=0, xanchor="left", y=1, yanchor="bottom", font=dict(size=12)),
    )
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=names,
                     range=[n - 0.5, -0.5] if reverse else [-0.5, n - 0.5], autorange=False,
                     ticklabelstandoff=8, ticks="", showgrid=False, automargin=True)
    return fig

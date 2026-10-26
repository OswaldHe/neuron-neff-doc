"""Small drawing helpers for the schematic figures (boxes, arrows, labels in data coordinates)."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import AXIS, INK, INK2, SURFACE, plt


def canvas(w, h, xmax=100, ymax=None):
    """A figure whose axes use data units 0..xmax x 0..ymax (aspect kept), no frame."""
    ymax = ymax if ymax is not None else xmax * h / w
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, text="", fc=SURFACE, ec=AXIS, color=INK, size=10, weight="normal",
        radius=0.8, lw=1.2, ha="center", va="center", style=None, alpha=1.0, z=2, family=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, linestyle=style or "-")
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else (x + 1.0 if ha == "left" else x + w - 1.0)
        ty = y + h / 2 if va == "center" else (y + h - 1.0 if va == "top" else y + 1.0)
        kw = {"family": family} if family else {}
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=size, color=color, weight=weight,
                zorder=z + 1, linespacing=1.25, **kw)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.5, style="-|>", ls="-", rad=0.0, z=3,
          mutation=14):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=mutation,
                        color=color, lw=lw, linestyle=ls, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def label(ax, x, y, text, size=10.5, color=INK2, ha="left", va="center", weight="normal",
          style="normal", z=4, **kw):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, weight=weight,
                   style=style, zorder=z, linespacing=1.3, **kw)

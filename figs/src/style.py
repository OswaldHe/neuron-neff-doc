"""Shared matplotlib style for the blog figures (light surface, reference palette).

Text is set in Inter, code-like labels in JetBrains Mono (figs/src/fonts/, SIL OFL), to match
the typography of the vLLM blog. Figures are sized for a ~900 px wide column: most are 9 in wide
at 200 dpi with 10-12 pt text, and save() trims the outer margin.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in sorted(os.listdir(FONTS)):
    if _f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, _f))
SANS, MONO = "Inter", "JetBrains Mono"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = "#cde2fb", "#fbd9c9", "#ecebe6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": [SANS, "DejaVu Sans"], "font.monospace": [MONO, "DejaVu Sans Mono"],
    "font.size": 11, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.labelsize": 11, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def trim(path, pad=24):
    """Crop the uniform SURFACE margin around everything drawn, keeping `pad` pixels."""
    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, SURFACE)).convert("L")
    box = diff.point(lambda v: 255 if v > 6 else 0).getbbox()
    if box:
        l, t, r, b = box
        im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width),
                 min(b + pad, im.height))).save(path, optimize=True)


def save_fig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    trim(path)

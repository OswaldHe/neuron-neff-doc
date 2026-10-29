"""Tensor Engine as a 4x4 systolic array: a general matmul (left) and the GDN outer product (right).

Each PE holds one stationary weight. Row k of the moving tile (partition k) feeds array row k;
sums run down each column, so array column m becomes output partition m.
Output: figs/fig18_systolic_array.png (400 dpi) and figs/fig18_systolic_array.pdf (vector)
"""
import os

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from style import AXIS, BLUE, INK, INK2, LIGHT_GRAY, MONO, ORANGE, VIOLET, plt, trim

plt.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3, in the PDF
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig18_systolic_array.png")
STEP = [BLUE, ORANGE, VIOLET]           # color of free-axis element n = 0, 1, 2
PW, PH = 1.5, 1.3                       # PE cell width, height
MW = 0.62                               # moving-matrix cell width
OH = 0.62                               # output-matrix row height
GAP = 1.0                               # gap between moving matrix and array


def arrow(ax, p, q, color=INK2, lw=1.3, ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def axis_label(ax, p, q, text, rot=0):
    arrow(ax, p, q, color=INK2, lw=1.1, ms=9)
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    off = (0, 0.22) if rot == 0 else (-0.25, 0)
    ax.text(mid[0] + off[0], mid[1] + off[1], text, ha="center", va="center", fontsize=10.5,
            color=INK2, rotation=rot)


def panel(ax, title, st, mv, active_rows):
    K, M = st.shape
    N = mv.shape[1]
    out = st.T @ mv                                    # [M, N]
    mx0 = -(GAP + N * MW)
    yb = -K * PH

    # input names and axes
    ax.text(mx0 + N * MW / 2, 1.55, "moving", ha="center", va="center", fontsize=12,
            weight="semibold", color=INK)
    ax.text(M * PW / 2, 1.55, "stationary", ha="center", va="center", fontsize=12,
            weight="semibold", color=INK)
    axis_label(ax, (mx0, 0.55), (mx0 + N * MW, 0.55), "free")
    axis_label(ax, (0, 0.55), (M * PW, 0.55), "free")
    axis_label(ax, (mx0 - 0.85, 0), (mx0 - 0.85, yb), "partition", rot=90)

    for k in range(K):
        on = k in active_rows
        y = -(k + 1) * PH
        for n in range(N):
            if on:
                ax.text(mx0 + (n + 0.5) * MW, y + PH / 2, str(mv[k, n]), ha="center",
                        va="center", fontsize=13, family=MONO, weight="bold", color=STEP[n])
        if on:
            arrow(ax, (-GAP + 0.15, y + PH / 2), (-0.1, y + PH / 2), color=AXIS)
        for m in range(M):
            ax.add_patch(FancyBboxPatch((m * PW + 0.05, y + 0.05), PW - 0.1, PH - 0.1,
                                        boxstyle="round,pad=0,rounding_size=0.1",
                                        fc="white" if on else LIGHT_GRAY,
                                        ec=AXIS if on else LIGHT_GRAY, lw=1.1))
            if on:
                ax.text((m + 0.5) * PW, y + PH / 2, str(st[k, m]), ha="center", va="center",
                        fontsize=15, family=MONO, weight="bold", color=INK)

    for m in range(M):
        arrow(ax, ((m + 0.5) * PW, yb - 0.02), ((m + 0.5) * PW, yb - 0.55))

    # output: columns are partitions, rows are the free axis
    oy = yb - 0.7
    for n in range(N):
        y = oy - (n + 1) * OH
        for m in range(M):
            ax.add_patch(Rectangle((m * PW, y), PW, OH, fc="white", ec=AXIS, lw=0.9))
            ax.text((m + 0.5) * PW, y + OH / 2, str(out[m, n]), ha="center", va="center",
                    fontsize=13, family=MONO, weight="bold", color=STEP[n])
    ax.text(mx0 + N * MW / 2, oy - N * OH / 2, "output", ha="center", va="center", fontsize=12,
            weight="semibold", color=INK)
    axis_label(ax, (-0.3, oy), (-0.3, oy - N * OH), "free", rot=90)
    axis_label(ax, (0, oy - N * OH - 0.3), (M * PW, oy - N * OH - 0.3), "")
    ax.text(M * PW / 2, oy - N * OH - 0.55, "partition", ha="center", va="center",
            fontsize=10.5, color=INK2)

    ax.set_title(title, loc="left", fontsize=13, pad=4)
    ax.set_xlim(mx0 - 1.3, M * PW + 0.2)
    ax.set_ylim(oy - N * OH - 0.9, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    st = np.array([[1, 2, 0, 1],
                   [0, 1, 3, 2],
                   [2, 0, 1, 1],
                   [1, 1, 0, 3]])
    mv = np.array([[1, 0, 2],
                   [2, 1, 0],
                   [0, 3, 1],
                   [1, 1, 1]])
    st_outer = np.zeros((4, 4), int)
    st_outer[0] = [2, 1, 3, 1]                         # k
    mv_outer = np.zeros((4, 3), int)
    mv_outer[0] = [1, 2, 3]                            # delta

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"wspace": 0.08})
    panel(a, "(a) General matmul", st, mv, range(4))
    panel(b, "(b) Outer product of k and δ", st_outer, mv_outer, [0])
    fig.savefig(OUT.replace(".png", ".pdf"))
    fig.savefig(OUT, dpi=400)
    plt.close(fig)
    trim(OUT, pad=48)


if __name__ == "__main__":
    main()

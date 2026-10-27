"""Data figures for the blog, drawn from figs/data/ (see export_data.py)."""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from style import (AQUA, BLUE, GRID, INK, INK2, LIGHT_GRAY, MUTED, ORANGE, VIOLET, YELLOW, plt,
                   save_fig)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..")

ROWS = [("Tensor", "Tensor Engine"), ("Vector", "Vector Engine"), ("Scalar", "Scalar Engine"),
        ("GpSimd", "GPSIMD Engine"), ("Sync", "Sync Engine")]
DMA_OPS = {"DMA_DIRECT2D", "WRITE"}
SYNC_OPS = {"EVENT_SEMAPHORE", "NOP", "DRAIN", "SET_ORDERING_MODE"}
CLASSES = [("compute", BLUE, "compute instruction"), ("dma", ORANGE, "DMA issue (DGE / trigger)"),
           ("table", YELLOW, "ACT_TABLE_LOAD"), ("sync", MUTED, "semaphore / control"),
           ("data", AQUA, "DMA data moving")]
STEP_LABEL = ["baseline", "1  DRAM round trips -> SBUF->SBUF", "2  drop redundant DMAs",
              "3  re-host DMAs, drop stale waits", "4  DMA -> compute, table loads",
              "5  fewer descriptors, table loads", "6  re-point one gate", "7  merge tile loads"]


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))


def cls(op):
    if op in DMA_OPS:
        return "dma"
    if op == "ACT_TABLE_LOAD":
        return "table"
    if op in SYNC_OPS:
        return "sync"
    return "compute"


def draw_timeline(ax, it, xmax, title=None, rows=ROWS, dma=True, h=0.72, labels=True, xmin=0):
    t = pd.read_csv(os.path.join(DATA, f"timeline_iter{it}.csv"))
    t["cls"] = t.opcode.map(cls)
    n = len(rows) + (1 if dma else 0)
    ticks = []
    for r, (eng, short) in enumerate(rows):
        y = n - 1 - r
        ticks.append((y, short))
        g = t[t.engine == eng]
        for c, color, _ in CLASSES[:4]:
            gg = g[g.cls == c]
            ax.broken_barh([(s, max(e - s, 0.05)) for s, e in zip(gg.s_us, gg.e_us)],
                           (y - h / 2, h), facecolors=color, edgecolors="none")
    if dma:
        d = pd.read_csv(os.path.join(DATA, f"dma_iter{it}.csv"))
        ax.broken_barh([(s, max(e - s, 0.05)) for s, e in zip(d.s_us, d.e_us)], (-h / 2, h),
                       facecolors=AQUA, edgecolors="none")
        ticks.append((0, "DMA"))
    end = t.e_us.max()
    ax.set_yticks([y for y, _ in ticks], [l for _, l in ticks] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(xmin, xmax)
    ax.spines["left"].set_visible(False)
    for y in range(n):
        ax.axhline(y, color=LIGHT_GRAY, lw=0.4, zorder=0)
    if title:
        ax.set_title(title)
    return end


def legend(fig, y=-0.04, classes=CLASSES, ncol=None):
    fig.legend(handles=[Patch(facecolor=c, label=l) for _, c, l in classes], loc="upper center",
               ncol=ncol or len(classes), bbox_to_anchor=(0.5, y), handlelength=1.2,
               columnspacing=1.2, handletextpad=0.5)


def lat():
    return json.load(open(os.path.join(DATA, "gdn_latency.json")))


def timeline_pair():
    L = lat()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    e0 = draw_timeline(axes[0], 0, 228, "Baseline NEFF, as compiled")
    e7 = draw_timeline(axes[1], 7, 228, "After 7 iterations of NEFF edits")
    for ax, e, k in ((axes[0], e0, "iter0"), (axes[1], e7, "iter7")):
        ax.axvline(e, color=INK2, lw=0.9, ls=(0, (2, 2)))
        ax.text(e + 2, 5.1, f"{L[k]['median_us']:.1f} µs", fontsize=11, color=INK, va="center",
                weight="semibold")
    axes[1].set_xlabel("time since the kernel's first instruction (µs); one profiled execution")
    fig.subplots_adjust(hspace=0.35)
    legend(fig, 0.035, ncol=3)
    save(fig, "gdn_timeline_detail.png")


def timelines_all():
    L = lat()
    fig, axes = plt.subplots(8, 1, figsize=(9, 11), sharex=True)
    for it, ax in enumerate(axes):
        e = draw_timeline(ax, it, 228, h=0.8)
        ax.set_yticks([5, 0], ["Tensor", "DMA"])
        ax.axvline(e, color=INK2, lw=0.9, ls=(0, (2, 2)))
        ax.text(e + 2, 2.5, f"{L[f'iter{it}']['median_us']:.1f} µs", fontsize=10.5, color=INK,
                va="center", weight="semibold")
        ax.set_title(f"iter {it}: {STEP_LABEL[it].split('  ', 1)[-1]}", fontsize=10.5, loc="left",
                     pad=3, weight="medium", color=INK2)
    axes[-1].set_xlabel("time (µs); rows top to bottom: Tensor, Vector, Scalar, GPSIMD, Sync Engine, DMA")
    fig.subplots_adjust(hspace=0.62)
    legend(fig, 0.068, ncol=3)
    save(fig, "gdn_timelines_all_detail.png")


def waterfall():
    L = lat()
    keys = [f"iter{i}" for i in range(8)]
    med = np.array([L[k]["median_us"] for k in keys])
    lo = med - np.array([L[k]["min_us"] for k in keys])
    hi = np.array([L[k]["max_us"] for k in keys]) - med
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(8)
    colors = [MUTED] + [BLUE] * 7
    ax.bar(x, med, width=0.62, color=colors, zorder=2)
    ax.errorbar(x, med, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=3, zorder=3)
    for i in range(8):
        txt = f"{med[i]:.1f}"
        if i:
            txt += f"\n{med[0] / med[i]:.2f}x"
        ax.text(i, med[i] + max(hi[i], 2) + 3, txt, ha="center", va="bottom", fontsize=10.5, color=INK)
    ticks = ["baseline", "1: Reduce\nSBUF-HBM\nround trips", "2: Remove\nredundant\nDMAs",
             "3: Balance\nDMAs from\nGPSIMD to\nScalar", "4: Fewer\nactivation-\ntable loads",
             "5: Pack data\nfor SBUF\naccess", "6: Change a\nsemaphore\nwait", "7: Merge\nweight\nDMAs and\nreorder"]
    ax.set_xticks(x, ticks, fontsize=9.5)
    ax.set_ylabel("total_exec_time (µs)")
    ax.set_ylim(0, 262)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig09_gdn_waterfall.png")


def dma_latency():
    d = pd.read_csv(os.path.join(DATA, "gdn_dge_dma.csv"))
    d = d[(d.iter == 0) & d.latency_us.notna()]

    def cat(r):
        if r.dst == "SBUF":
            return 0
        if r.elem >= 512:
            return 1
        return 2 if r.elem >= 8 else 3
    d = d.assign(c=d.apply(cat, axis=1))
    names = ["DRAM -> SBUF\n(any element size)", "SBUF -> DRAM\n512-byte elements",
             "SBUF -> DRAM\n8-byte elements", "SBUF -> DRAM\n2-byte elements"]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    rng = np.random.default_rng(0)
    for c in range(4):
        g = d[d.c == c]
        y = np.full(len(g), c) + rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(g.latency_us, y, s=34, color=[AQUA, BLUE, VIOLET, ORANGE][c], alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        if len(g):
            m = g.latency_us.median()
            ax.plot([m, m], [c - 0.3, c + 0.3], color=INK, lw=1.6, zorder=4)
            ax.text(max(g.latency_us.max(), m) + 0.8, c, f"median {m:.1f} µs  (n={len(g)})",
                    va="center", fontsize=10.5, color=INK2)
    ax.set_yticks(range(4), names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("issue -> completion semaphore reached (µs), baseline NEFF")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 48)
    ax.set_title("Narrow writes into DRAM report completion ~10x later than loads\n"
                 "(every DGE DMA of the baseline)")
    save(fig, "dma_latency_detail.png")


def issue_cost():
    d = pd.read_csv(os.path.join(DATA, "gdn_dge_dma.csv"))
    d = d.drop_duplicates(subset=["engine", "desc", "issue_us", "iter"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for eng, c in (("Pool", VIOLET), ("SP", ORANGE), ("Activation", BLUE)):  # NEFF stream names
        g = d[d.engine == eng]
        jit = np.random.default_rng(1).uniform(-12, 12, len(g))
        ax.scatter(g.desc + jit, g.issue_us, s=22, color=c, alpha=0.7, edgecolors="none",
                   label={"Pool": "GPSIMD Engine", "SP": "Sync Engine", "Activation": "Scalar Engine"}[eng],
                   zorder=3)
    x = d.desc.values.astype(float)
    b, a = np.polyfit(x, d.issue_us.values, 1)
    xs = np.array([0, 1100])
    ax.plot(xs, a + b * xs, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=4)
    ax.text(200, 1.45, f"fit: {a:.2f} µs + {b * 1000:.2f} ns x descriptors",
            fontsize=11, color=INK)
    ax.set_xlabel("descriptors in the transfer")
    ax.set_ylabel("engine time to issue (µs)")
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 1.8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="issuing engine", loc="lower right", title_fontsize=10.5)
    ax.set_title("A DGE DMA costs ~0.6 µs of engine time, almost\nregardless of size "
                 f"(all {len(d)} issues, 8 profiles)")
    save(fig, "dge_issue_cost_detail.png")
    return a, b


def speedups():
    k = pd.read_csv(os.path.join(DATA, "kernels.csv")).sort_values("gain")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(k))
    ax.barh(y, k.gain, height=0.62, color=[BLUE if f.startswith("gdn") else VIOLET for f in k.family],
            zorder=2)
    for i, r in enumerate(k.itertuples()):
        ax.text(r.gain + 0.05, i, f"{r.gain:.2f}x   {r.base_us:,.1f} -> {r.opt_us:,.1f} µs",
                va="center", fontsize=10, color=INK2)
    ax.set_yticks(y, k.config, fontsize=10, family="monospace")
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xlim(0.9, 7.0)
    ax.set_xlabel("speed-up from editing only the compiled NEFF (same inputs, outputs within tolerance)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Gated DeltaNet (nkilib experimental/gdn)"),
                       Patch(facecolor=VIOLET, label="Mamba scans (nkilib experimental/scan)")],
              loc="lower right")
    save(fig, "fig17_speedups.png")


def roofline(keys=("iter0",), out="fig07_roofline.png", ceilings=False):
    """Roofline for one LNC. keys: which NEFFs to plot; ceilings: add the per-engine busy-time caps."""
    r = json.load(open(os.path.join(DATA, "gdn_roofline.json")))
    bw, peak = r["peak"]["dma_bytes_per_s"], r["peak"]["tensor_flops_per_s"]
    lt = {k: v["median_us"] for k, v in lat().items()}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1e9, 3e14)
    x = np.logspace(-1, 3, 200)
    ax.plot(x, np.minimum(x * bw, peak), color=INK2, lw=2)
    fig.canvas.draw()
    (u0, v0), (u1, v1) = ax.transData.transform([(0.2, 0.2 * bw), (20, 20 * bw)])
    ang = np.degrees(np.arctan2(v1 - v0, u1 - u0))
    ax.text(2.2, 2.2 * bw * 1.6, f"memory roof: {bw / 1e9:.0f} GB/s", rotation=ang,
            rotation_mode="anchor", fontsize=10.5, color=INK2)
    ax.text(14, peak * 1.3, f"Tensor Engine peak: {peak / 1e12:.1f} TFLOP/s", fontsize=10.5,
            color=INK2)
    name = {"iter0": ("baseline", MUTED), "iter7": ("optimized", BLUE)}
    xy = {}
    for key in keys:
        ai = r[key]["tensor_flops"] / r[key]["hbm_bytes"]
        perf = r[key]["tensor_flops"] / (lt[key] * 1e-6)
        xy[key] = (ai, perf)
        ax.scatter([ai], [perf], s=90, color=name[key][1], zorder=5, edgecolor="white", linewidth=1.5)
        dy = 0.78 if (ceilings and key == "iter7") else 1.0
        ax.text(ai * 1.35, perf * dy, f"{name[key][0]}: {lt[key]:.1f} µs, {perf / 1e9:.1f} GFLOP/s",
                va="center", fontsize=10.5, color=INK)
    if len(keys) == 2:
        (x0, y0), (x1, y1) = xy[keys[0]], xy[keys[1]]
        ax.annotate("", xy=(x1, y1 * 0.88), xytext=(x0, y0 * 1.12),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    xt, yt = xy[keys[-1]]
    roof = xt * bw
    ax.plot([xt, xt], [yt * 1.15, roof * 0.87], color=INK2, lw=1.2, ls=(0, (3, 2)))
    if ceilings:
        ax.text(xt * 1.3, np.sqrt(yt * roof) * 1.15, f"{roof / yt:.0f}x below the roof", ha="left",
                va="center", fontsize=10.5, color=INK2)
    else:
        ax.text(xt * 0.8, yt * 2.6, f"{roof / yt:.0f}x below\nthe roof", ha="right", va="center",
                fontsize=10.5, color=INK2)
    ax.scatter([xt], [roof], s=60, facecolor="white", edgecolor=INK2, zorder=5, linewidth=1.5)
    ax.text(xt * 1.35, roof, f"memory roof at this intensity: {roof / 1e9:.0f} GFLOP/s", va="center",
            fontsize=10.5, color=INK2)
    if ceilings:  # the kernel cannot finish before its busiest engine: FLOPs / busy time
        busy = pd.read_csv(os.path.join(DATA, "gdn_engine_busy.csv")).set_index("iter").loc[7]
        flops = r["iter7"]["tensor_flops"]
        for col, label_, c, ls in (("ACT", "Scalar Engine", ORANGE, "-"),
                                   ("Pool", "Vector / GPSIMD Engine", VIOLET, (0, (4, 2)))):
            cap = flops / (busy[col] * 1e-6)
            ax.plot([0.1, 1000], [cap, cap], color=c, lw=1.8, ls=ls)
            ax.text(1000, cap * 1.12, f"{label_} busy {busy[col]:.1f} µs: caps at {cap / 1e9:.0f} GFLOP/s",
                    ha="right", va="bottom", fontsize=10, color=c)
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out)


if __name__ == "__main__":
    speedups()
    roofline()
    roofline(("iter0", "iter7"), "fig16_roofline_optimized.png", ceilings=True)
    waterfall()
    timeline_pair()
    timelines_all()
    dma_latency()
    print("issue cost fit", issue_cost())

"""Case-study figures: the GDN kernel, one illustration per optimization step, the NKI-vs-NEFF
attribution, and the control-surface overview."""
import csv
import json
import os
from collections import defaultdict

import numpy as np
from matplotlib.patches import Patch

from diagram import arrow, box, canvas, label
from fig_data import DATA, draw_timeline, legend
from style import (AQUA, AXIS, BLUE, GREEN, GRID, INK, INK2, LIGHT_BLUE, LIGHT_GRAY, LIGHT_ORANGE,
                   MAGENTA, MUTED, ORANGE, RED, SURFACE, VIOLET, YELLOW, plt, save_fig)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TINT = {BLUE: LIGHT_BLUE, ORANGE: LIGHT_ORANGE, AQUA: "#d3f1e6", YELLOW: "#fbecc7",
        VIOLET: "#e2def8", MAGENTA: "#f9e2eb", GREEN: "#d2ead2", RED: "#f8d9d9", MUTED: LIGHT_GRAY}
CAT = {"nki": ("expressible in NKI (by restructuring the source)", BLUE),
       "indirect": ("NKI only indirectly (a coarse knob; the compiler decides)", VIOLET),
       "neff": ("NEFF-only (no NKI construct)", ORANGE)}


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))


def gdn_kernel():
    fig, ax = canvas(10, 3.55, 118)
    y, h, w = 15, 11, 17
    st = [(1, "in_proj\n4 matmul groups\n(Tensor Engine)", "hidden x W_qkv, W_z,\nW_a, W_b", BLUE),
          (27, "conv1d decode\nwindow 4 + silu", "nested kernel\ngdn_conv1d_decode", VIOLET),
          (53, "split + GQA\nrepeat q, k, v, z", "per-v-head\nrow copies", BLUE),
          (79, "gdn_tkg\nrecurrence", "nested kernel: l2norm, gates,\n128x128 state update,\nRMSNormGated",
           VIOLET)]
    for x, t, sub, c in st:
        box(ax, x, y, w, h, t, fc=TINT[c], ec=c, size=10.5, weight="semibold")
        label(ax, x + w / 2, y - 1.4, sub, size=9.5, ha="center", va="top")
    for x, n in ((18, "mixed"), (44, "conv_out"), (70, "q, k,\nv, z")):
        arrow(ax, x + 0.3, y + 3.5, x + 8.7, y + 3.5, color=ORANGE, lw=1.6)
        box(ax, x + 0.75, y + 5.2, 7.5, 5.3, n, fc=TINT[ORANGE], ec=ORANGE, size=9.5, radius=0.5)
    box(ax, 101, y, 16, h, "outputs\ncore_out [1,1,512]\nconv_state', state'", fc=LIGHT_GRAY,
        ec=AXIS, size=9.5)
    arrow(ax, 96.3, y + h / 2, 100.7, y + h / 2)
    label(ax, 1, 33.4, "gdn_block_tkg: Qwen3.5-style Gated DeltaNet decode (one token)", size=11.5,
          color=INK, weight="semibold")
    label(ax, 1, 29.9, "hidden 128, 2 k-heads -> 4 v-heads, head dim 128, bf16 / fp32 state", size=10.5,
          color=INK2)
    box(ax, 1, 0.5, 116, 6.2, "", fc=SURFACE, ec=ORANGE, style=(0, (3, 2)))
    label(ax, 2.5, 3.6, "orange = HBM tensors handing data between the composed kernels (nl.shared_hbm). "
          "In the NEFF they become\nten small DRAM 'tmp-buf' variables: values go SBUF -> DRAM -> SBUF.",
          size=10, color=INK2)
    save(fig, "fig08_gdn_kernel.png")


def step1():
    fig, ax = canvas(9, 5.3, 100)
    sbuf = dict(fc=LIGHT_BLUE, ec=BLUE, size=10)
    label(ax, 1, 57.5, "Before: a per-partition copy routed through DRAM", size=11.5, weight="semibold",
          color=INK)
    box(ax, 1, 41, 15, 11.5, "SBUF X\n128 partitions\nx 2 B", **sbuf)
    box(ax, 42.5, 41, 15, 11.5, "DRAM\ntmp-buf\n256 B", fc=LIGHT_ORANGE, ec=ORANGE, size=10)
    box(ax, 84, 41, 15, 11.5, "SBUF Y\n128 partitions\nx 4 B", **sbuf)
    arrow(ax, 16.3, 46.75, 42.2, 46.75, color=ORANGE, lw=1.8)
    label(ax, 29.25, 49.6, "writer DMA, 2-B elements", size=10, ha="center")
    arrow(ax, 57.8, 46.75, 83.7, 46.75, color=AQUA, lw=1.8)
    label(ax, 70.75, 49.6, "reader DMA, bf16 -> fp32", size=10, ha="center")
    label(ax, 50, 38.3, "the writer completes ~17-22 µs after issue; only then may the reader start",
          size=10.5, ha="center", color=ORANGE)
    label(ax, 1, 33, "After: two fields swapped, same semaphores", size=11.5, weight="semibold",
          color=INK)
    box(ax, 1, 16.5, 15, 11.5, "SBUF X\n128 partitions\nx 2 B", **sbuf)
    box(ax, 84, 16.5, 15, 11.5, "SBUF Y\n128 partitions\nx 4 B", **sbuf)
    arrow(ax, 16.3, 24.5, 83.7, 24.5, color=BLUE, lw=1.8)
    label(ax, 50, 26.7, "writer: src X -> dst Y (cast on the way)", size=10, ha="center")
    arrow(ax, 16.3, 20, 83.7, 20, color=MUTED, lw=1.4, ls=(0, (3, 2)))
    label(ax, 50, 17.6, "reader: the same copy again (idempotent)", size=10, ha="center", color=MUTED)
    label(ax, 50, 13.8, "the writer's completion now means 'Y is ready': ~2 µs, no wait changed",
          size=10.5, ha="center", color=BLUE)
    label(ax, 1, 9.6, "NEFF edit: in each writer, copy the reader's destination address and pattern; in each "
          "reader, copy the\nwriter's source (PSEUDO_DMA_DIRECT2D bytes 16-63 / SP0.json descriptors). Every "
          "semaphore and wait stays\nas the compiler emitted it, so the change cannot break synchronisation. "
          "6 buffers, outputs bit-identical.", size=10, color=INK2, va="top")
    save(fig, "fig12b_step1_roundtrip.png")


def step2():
    fig, ax = canvas(9, 3.9, 100)
    mono = dict(family="monospace", size=10)
    label(ax, 1, 41.5, "A DMA whose bytes someone else already moves, replaced by a semaphore increment",
          size=11.5, weight="semibold", color=INK)
    box(ax, 1, 26, 98, 11.5, "", fc=LIGHT_ORANGE, ec=ORANGE)
    label(ax, 3, 34.6, "before: PSEUDO_DMA_DIRECT2D  (Sync Engine slot 13)", weight="bold", color=INK, **mono)
    label(ax, 3, 31.4, "wait S[72] >= 16      semaphore=62  +16 on completion", **mono)
    label(ax, 3, 28.3, "issue ~0.55 µs, completion lands 2-7 µs later", size=10, color=ORANGE)
    arrow(ax, 50, 25.7, 50, 22.3, lw=1.6)
    box(ax, 1, 10, 98, 12, "", fc=LIGHT_GRAY, ec=AXIS)
    label(ax, 3, 19.1, "after: EVENT_SEMAPHORE  (same slot, same 64 bytes)", weight="bold", color=INK, **mono)
    label(ax, 3, 15.9, "wait S[72] >= 16      SEM_ADD_IMM_COMPLETE  S[62] += 16", **mono)
    label(ax, 3, 12.8, "~0.03 µs; S[62] moves the instant the real producer is done", size=10, color=INK2)
    label(ax, 1, 7.6, "Every waiter on S[62] is untouched. Update mode 21 (SEM_ADD_IMM_COMPLETE) is in the ISA "
          "enum, but the compiler\nnever emits it; the compiled kernels only use '+1 on completion' (mode 19). "
          "Trap: the wait threshold and the\nadd amount share one 4-byte value field.", size=10, color=INK2,
          va="top")
    save(fig, "fig13b_step2_sem_add.png")


def step3():
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.3), sharex=True)
    rows = [("Scalar", "Scalar Engine"), ("GpSimd", "GPSIMD Engine"), ("Sync", "Sync Engine")]
    draw_timeline(axes[0], 2, 36, rows=rows, dma=False)
    draw_timeline(axes[1], 3, 36, rows=rows, dma=False)
    axes[0].set_title("Before (iter 2): the GPSIMD Engine issues 25 weight-load DMAs back to back,\n"
                      "while the Scalar Engine waits", fontsize=11)
    axes[1].set_title("After (iter 3): 8 of them issued from the Scalar Engine's 8 unused DGE queues;\n"
                      "the GPSIMD Engine keeps a cheap relay", fontsize=11)
    axes[1].set_xlabel("time since the kernel's first instruction (µs)")
    fig.subplots_adjust(hspace=0.55)
    legend(fig, 0.0, classes=[c for c in __import__("fig_data").CLASSES[:4]])
    save(fig, "fig14_step3_queues.png")


def step4():
    fig, ax = canvas(9, 3.75, 100)
    label(ax, 1, 40.5, "Activation-table loads in the per-head tail (1.283 µs each)", size=11.5,
          weight="semibold", color=INK)

    def ribbon(y, seq, title):
        label(ax, 1, y + 8, title, size=10.5, color=INK2)
        x = 1
        for kind in seq:
            if kind in ("L2", "L4"):
                box(ax, x, y, 4.5, 5.5, kind.replace("L", "load\n"), fc=YELLOW, ec=YELLOW, size=9,
                    radius=0.4, color=INK)
                x += 4.95
            else:
                c = {"s": GREEN, "r": VIOLET, "a": MUTED}[kind]
                box(ax, x, y, 2.2, 5.5, "", fc=c, ec=c, radius=0.4)
                x += 2.65
        return x
    before = ["L4", "r", "r", "a", "L2", "s", "L4", "r", "a", "a", "L2", "s", "s", "s", "a", "L4",
              "r", "a", "L2", "s", "L4", "r", "r", "a", "L2", "s", "L4", "r", "a"]
    after = ["L4", "r", "r", "a", "L2", "s", "s", "s", "s", "s", "s", "a", "a", "L4", "r", "a",
             "r", "r", "a", "r", "a", "a"]
    ribbon(27.5, before, "compiled order")
    ribbon(14.5, after, "after steps 4-5")
    for c, n, x in ((GREEN, "silu (needs table 2)", 1), (VIOLET, "rsqrt (needs table 4)", 28),
                    (MUTED, "table-agnostic (square, copy, CAST)", 56)):
        box(ax, x, 8.2, 2.2, 3.2, "", fc=c, ec=c, radius=0.3)
        label(ax, x + 3.2, 9.8, n, size=10)
    label(ax, 1, 5.6, "Illustrative sequence (not slot-exact). Tail loads went 4,2,4,2,4,2,4 -> 4,2,4; kernel "
          "total 13 -> 7.\nNo set holds both silu and rsqrt, so 5 loads is the floor.", size=10, color=INK2,
          va="top")
    save(fig, "fig15_step4_tables.png")


def step5():
    fig, ax = canvas(9, 6.0, 100)
    cols = (BLUE, VIOLET, GREEN, ORANGE)
    label(ax, 1, 64.5, "A partition -> row transpose costs one DMA descriptor per partition per column",
          size=11.5, weight="semibold", color=INK)

    def grid(x0, y0, w=3.3, h=1.2, ncol=4, rows=8, gap=0.35):
        for r in range(rows):
            for c in range(ncol):
                box(ax, x0 + c * w, y0 - r * h, w - gap, h - 0.25, "", fc=cols[c % 4], ec=cols[c % 4],
                    radius=0.15)
    label(ax, 1, 59.5, "before: 4 transfers x 128 descriptors (2 B each)", size=10.5, color=INK,
          weight="medium")
    grid(2, 55)
    label(ax, 2, 44.4, "SBUF: 128 partitions,\n4 columns 256 B apart", size=9.5, color=INK2, va="top")
    for k, c in enumerate(cols):
        arrow(ax, 16, 51.5 - k * 1.2, 26.5, 55.2 - k * 3.4, color=c, lw=1.3)
        box(ax, 27, 54 - k * 3.4, 32, 2.4, "", fc=c, ec=c, radius=0.2)
    label(ax, 62, 50, "4 rows in partition 0:\n512 descriptors,\n~5 µs of DGE time", size=10,
          color=ORANGE, va="center")
    label(ax, 1, 37, "after: pack by compute, then 1 transfer x 128 descriptors (8 B each)", size=10.5,
          color=INK, weight="medium")
    grid(2, 32.5)
    arrow(ax, 16, 28.5, 24.5, 28.5, lw=1.4)
    label(ax, 20.25, 30.6, "4 CASTs", size=9.5, ha="center")
    grid(25.5, 32.5, w=1.7, gap=0.3)
    label(ax, 25.5, 22, "8 contiguous bytes\nper partition", size=9.5, color=INK2, va="top")
    arrow(ax, 33, 28.5, 40.5, 28.5, lw=1.4)
    for k in range(28):
        c = cols[k % 4]
        box(ax, 41.5 + k * 1.15, 27.2, 0.95, 2.6, "", fc=c, ec=c, radius=0.1)
    label(ax, 41.5, 24.8, "one interleaved row; consumers read it with stride 4\n(incl. the Tensor Engine's "
          "reverse-stride LDWEIGHTS: every step x4)", size=9.5, color=INK2, va="top")
    label(ax, 1, 15.5, "Applied to both transpose groups: -1.4 µs and -2.8 µs. The DMA engine retires ~1,500 "
          "tiny transpose\ndescriptors serially (~8-15 ns each), so in that window only removing descriptors "
          "helped: three edits\nthat merely started transfers earlier were correct and lost time.", size=10,
          color=INK2, va="top")
    save(fig, "fig16_step5_pack.png")


def step6():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ev = [("Sync Engine slot 18: DGE transpose (the real data)", 35.80, 36.41, BLUE, 3, (35.80, 3.38, "bottom")),
          ("S[67] += 16: its completion", 37.80, 37.80, BLUE, 3, (37.98, 3.0, "center")),
          ("Sync Engine slot 31: trigger of a 2-byte dummy DMA", 39.00, 39.10, MUTED, 2,
           (35.80, 2.38, "bottom")),
          ("S[28] += 16: dummy completes", 40.39, 40.39, MUTED, 2, (40.58, 2.0, "center"))]
    for name, s, e, c, y, (tx, ty, va) in ev:
        if e > s + 0.05:
            ax.broken_barh([(s, e - s)], (y - 0.3, 0.6), facecolors=c)
        else:
            ax.plot([s], [y], marker="v", color=c, ms=10)
        ax.text(tx, ty, name, fontsize=10, color=INK2, va=va)
    ax.broken_barh([(37.85, 0.5)], (0.7, 0.6), facecolors=GREEN)
    ax.broken_barh([(40.44, 0.5)], (0.7, 0.6), facecolors=ORANGE)
    ax.text(38.45, 1.0, "Scalar Engine slot 55\nstarts (after the edit)", fontsize=10, color=GREEN,
            va="center")
    ax.text(41.05, 1.0, "Scalar Engine slot 55\nstarts (compiled gate)", fontsize=10, color=ORANGE,
            va="center")
    ax.annotate("", xy=(37.85, 1.52), xytext=(40.44, 1.52),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(39.3, 1.6, "2.6 µs", ha="left", va="bottom", fontsize=10.5, color=INK, weight="semibold")
    ax.set_yticks([3, 2, 1], ["Sync Engine / DMA", "dummy", "Scalar Engine"])
    ax.set_ylim(0.3, 3.95)
    ax.set_xlim(35.4, 43.6)
    ax.set_xlabel("time (µs), iteration-5 profile")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("One byte: Scalar Engine slot 54 'wait S[28] >= 16' -> 'wait S[67] >= 16'")
    save(fig, "fig17_step6_gate.png")


def step7():
    fig, ax = plt.subplots(figsize=(9, 3.1))
    for k in range(8):
        ax.broken_barh([(k * 0.69, 0.6)], (1.7, 0.6), facecolors=ORANGE if k % 2 == 0 else "#f3a07c")
    ax.text(5.7, 2.0, "8 DMAs x (0.6 µs fixed + 128 desc.) = 5.5 µs of issue", va="center",
            fontsize=10.5, color=INK2)
    ax.broken_barh([(0, 0.6 + 0.00071 * 1024)], (0.7, 0.6), facecolors=BLUE)
    ax.text(1.5, 1.0, "1 DMA x (0.6 µs + 1024 desc.) = 1.3 µs  (tile axis = dim 1 of the access pattern)",
            va="center", fontsize=10.5, color=INK2)
    ax.set_yticks([2, 1], ["8 tile loads", "merged"])
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0.3, 2.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("engine time to issue (µs), cost model fitted on the profiles")
    ax.set_title("Merging tile loads buys issue time; then split again along the consumer\n"
                 "boundary (4 + 4 tiles)")
    save(fig, "fig18_step7_merge.png")


def attribution():
    rows = list(csv.DictReader(open(os.path.join(DATA, "gdn_edit_attribution.csv"))))
    lat = json.load(open(os.path.join(DATA, "gdn_latency.json")))
    meas = {i: lat[f"iter{i - 1}"]["median_us"] - lat[f"iter{i}"]["median_us"] for i in range(1, 8)}
    per = defaultdict(lambda: defaultdict(float))
    for r in rows:
        per[int(r["iter"])][r["category"]] += float(r["delta_us"])
    for i in per:  # scale the logged per-edit deltas to the re-measured per-iteration delta
        f = meas[i] / sum(per[i].values())
        for k in per[i]:
            per[i][k] *= f
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 4.4), gridspec_kw={"width_ratios": [1, 2.2]})
    tot1 = {k: per[1].get(k, 0) for k in CAT}
    tot2 = {k: sum(per[i].get(k, 0) for i in range(2, 8)) for k in CAT}
    for x, tot in ((0, tot1), (1, tot2)):
        b = 0
        for k, (_, c) in CAT.items():
            v = tot[k]
            a0.bar(x, v, bottom=b, color=c, width=0.6, edgecolor=SURFACE, linewidth=1.5)
            if v > 4:
                a0.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=10.5,
                        weight="semibold")
            b += v
        a0.text(x, b + 2, f"{b:.0f} µs", ha="center", fontsize=10.5, color=INK)
    a0.set_xticks([0, 1], ["step 1", "steps 2-7"])
    a0.set_ylim(0, 140)
    a0.set_ylabel("latency removed (µs)")
    a0.yaxis.grid(True, color=GRID, lw=0.6)
    a0.set_axisbelow(True)
    a0.set_title("Where the 173 µs\ncame from")
    xs = np.arange(2, 8)
    b = np.zeros(6)
    for k, (_, c) in CAT.items():
        v = np.array([per[i].get(k, 0) for i in xs])
        a1.bar(xs, v, bottom=b, color=c, width=0.62, edgecolor=SURFACE, linewidth=1.5)
        b += v
    for x, t in zip(xs, b):
        a1.text(x, t + 0.3, f"{t:.1f}", ha="center", fontsize=10.5, color=INK)
    a1.set_xticks(xs, [f"step {i}" for i in xs])
    a1.yaxis.grid(True, color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    a1.set_title("Steps 2-7 by lever type: 22.5 of 51 µs\nneeded a non-NKI lever")
    a1.set_ylim(0, 25)
    a1.legend(handles=[Patch(facecolor=c, label=l) for l, c in CAT.values()], loc="upper right",
              fontsize=10, handlelength=1.2)
    save(fig, "fig22_attribution.png")
    return {k: (round(tot1[k], 1), round(tot2[k], 1)) for k in CAT}


def control_surface():
    fig, ax = canvas(9.5, 5.5, 100)
    cols = [(42, "NKI 0.6 source"), (61.5, "compiler (neuronx-cc)"), (81, "NEFF edit")]
    label(ax, 0.5, 56, "Who decides what ends up in the instruction stream", size=12, weight="semibold",
          color=INK)
    for x, t in cols:
        label(ax, x + 9.25, 51.6, t, size=10.5, weight="semibold", color=INK2, ha="center")
    rows = [("which ISA op, operands, dtypes, tile shapes", "full", "lowers", "full"),
            ("where a tensor lives (SBUF/PSUM/HBM), SBUF address", "full*", "allocates", "full"),
            ("which engine runs a compute op", "most ops", "default", "any capable engine"),
            ("which engine issues a DMA", "sync / scalar / gpsimd", "default", "+ position in stream"),
            ("which DMA queue + semaphore a transfer uses", "no", "decides", "full"),
            ("DMA descriptors, queued-block triggering", "access pattern", "decides", "full"),
            ("order of instructions on one engine", "program order", "reorders", "exact"),
            ("which semaphore each instruction waits on, threshold", "no", "derives", "full"),
            ("signal without moving data (SEM_ADD_IMM)", "no", "never emits", "full"),
            ("activation-table loads: count and placement", "no", "inserts", "full")]
    marks = {"no": RED, "never emits": RED}
    for k, (what, nki, cc, neff) in enumerate(rows):
        y = 47 - k * 4.6
        if k % 2 == 0:
            box(ax, 0, y - 2.25, 100, 4.5, "", fc="#f4f3ef", ec="#f4f3ef", radius=0.3, z=0)
        label(ax, 0.8, y, what, size=10, color=INK)
        for (x, _), val, fill in ((cols[0], nki, BLUE), (cols[1], cc, MUTED), (cols[2], neff, ORANGE)):
            c = marks.get(val, fill)
            box(ax, x, y - 1.75, 18.5, 3.5, val, fc=TINT.get(c, LIGHT_GRAY), ec=c, size=9.5, radius=0.4)
    label(ax, 0.8, 1.6, "* nkilib kernels are compiled with NKI's own allocation and scheduling; "
          "reused standalone kernels still take and return HBM tensors.", size=9.5, color=INK2)
    save(fig, "fig06_control_surface.png")


if __name__ == "__main__":
    gdn_kernel()
    step1()
    step2()
    step3()
    step4()
    step5()
    step6()
    step7()
    print(attribution())
    control_surface()

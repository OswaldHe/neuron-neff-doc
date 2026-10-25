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
                   MAGENTA, MUTED, ORANGE, RED, SURFACE, VIOLET, YELLOW, plt)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TINT = {BLUE: LIGHT_BLUE, ORANGE: LIGHT_ORANGE, AQUA: "#d3f1e6", YELLOW: "#fbecc7",
        VIOLET: "#e2def8", MAGENTA: "#f9e2eb", GREEN: "#d2ead2", RED: "#f8d9d9", MUTED: LIGHT_GRAY}
CAT = {"nki": ("expressible in NKI (by restructuring the source)", BLUE),
       "indirect": ("NKI only indirectly (a coarse knob; the compiler decides)", VIOLET),
       "neff": ("NEFF-only (no NKI construct)", ORANGE)}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def gdn_kernel():
    fig, ax = canvas(12, 3.9, 120)
    y = 18
    st = [(1, "in_proj\n4 matmul groups\n(Tensor Engine)", "hidden x W_qkv, W_z,\nW_a, W_b", BLUE),
          (27, "conv1d decode\nwindow 4 + silu", "nested kernel\ngdn_conv1d_decode", VIOLET),
          (53, "split + GQA\nrepeat q, k, v, z", "per-v-head\nrow copies", BLUE),
          (79, "gdn_tkg\nrecurrence", "nested kernel: l2norm, gates,\n128x128 state update,\nRMSNormGated", VIOLET)]
    for x, t, sub, c in st:
        box(ax, x, y, 18, 10, t, fc=TINT[c], ec=c, size=8.8, weight="bold")
        label(ax, x + 9, y - 1.2, sub, size=7.6, ha="center", va="top")
    hb = [(19.5, "mixed"), (45.5, "conv_out"), (71.5, "q, k, v, z\n(per head)")]
    for x, n in hb:
        box(ax, x, y + 2.5, 7, 5, n, fc=TINT[ORANGE], ec=ORANGE, size=7, radius=0.4)
        arrow(ax, x - 0.4, y + 5, x - 0.1, y + 5, color=ORANGE)
        arrow(ax, x + 7.1, y + 5, x + 7.5, y + 5, color=ORANGE)
    box(ax, 100, y, 19, 10, "outputs\ncore_out [1,1,512]\nconv_state', state'", fc=LIGHT_GRAY,
        ec=AXIS, size=8)
    arrow(ax, 97.2, y + 5, 99.8, y + 5)
    label(ax, 1, 34, "gdn_block_tkg (Qwen3.5-style Gated DeltaNet decode, one token): "
          "hidden 128, 2 k-heads -> 4 v-heads, head dim 128, bf16 / fp32 state",
          size=9.5, color=INK, weight="bold")
    box(ax, 19.5, 2, 59, 7, "", fc=SURFACE, ec=ORANGE, style=(0, (3, 2)))
    label(ax, 21, 5.5, "orange = HBM tensors handing data between the composed kernels "
          "(nl.shared_hbm).\nIn the NEFF they become ten small DRAM 'tmp-buf' variables: "
          "values go SBUF -> DRAM -> SBUF.", size=8, color=INK2)
    save(fig, "fig09_gdn_kernel.png")


def step1():
    fig, ax = canvas(12, 3.5, 120)
    ax.set_ylim(10, 45)
    label(ax, 1, 43, "Before: a per-partition copy routed through DRAM", size=10, weight="bold", color=INK)
    label(ax, 62, 43, "After: two fields swapped, same semaphores", size=10, weight="bold", color=INK)
    for x0, after in ((1, False), (62, True)):
        box(ax, x0, 24, 12, 13, "SBUF\nX\n128 part.\nx 2 B", fc=LIGHT_BLUE, ec=BLUE, size=8)
        box(ax, x0 + 44, 24, 12, 13, "SBUF\nY\n128 part.\nx 4 B", fc=LIGHT_BLUE, ec=BLUE, size=8)
        if not after:
            box(ax, x0 + 22, 24, 12, 13, "DRAM\ntmp-buf\n256 B", fc=LIGHT_ORANGE, ec=ORANGE, size=8)
            arrow(ax, x0 + 12.3, 32, x0 + 21.7, 32, color=ORANGE, lw=1.5)
            label(ax, x0 + 17, 35, "writer DMA\n2-B elements", size=7.3, ha="center")
            arrow(ax, x0 + 34.3, 32, x0 + 43.7, 32, color=AQUA, lw=1.5)
            label(ax, x0 + 39, 35, "reader DMA\nbf16->fp32", size=7.3, ha="center")
            label(ax, x0 + 22.5, 20.5, "writer completes ~17-22 µs after issue\n(only then may the reader start)",
                  size=8, ha="center", color=ORANGE)
        else:
            arrow(ax, x0 + 12.3, 33, x0 + 43.7, 33, color=BLUE, lw=1.5)
            arrow(ax, x0 + 12.3, 28, x0 + 43.7, 28, color=MUTED, lw=1.2, ls=(0, (3, 2)))
            label(ax, x0 + 28, 35, "writer: src X -> dst Y (cast on the way)", size=7.5, ha="center")
            label(ax, x0 + 28, 25.6, "reader: same copy again (idempotent)", size=7.5, ha="center",
                  color=MUTED)
            label(ax, x0 + 28, 20.5, "writer's completion now means 'Y is ready'\n~2 µs, no wait changed",
                  size=8, ha="center", color=BLUE)
    label(ax, 1, 16, "NEFF edit: in each writer, copy the reader's destination address and pattern; "
          "in each reader, copy the writer's source (PSEUDO_DMA_DIRECT2D bytes 16-63 / SP0.json "
          "descriptors).\nEvery semaphore and wait stays as the compiler emitted it, so the change "
          "cannot break synchronisation. 6 buffers, outputs bit-identical.", size=8.3,
          color=INK2, va="top")
    save(fig, "fig13b_step1_roundtrip.png")


def step2():
    fig, ax = canvas(12, 2.8, 120)
    ax.set_ylim(5, 33)
    label(ax, 1, 31.5, "A DMA whose bytes are already moved by someone else, replaced by a "
          "semaphore increment", size=10, weight="bold", color=INK)
    box(ax, 1, 13, 50, 12, "", fc=LIGHT_ORANGE, ec=ORANGE)
    label(ax, 3, 22.3, "before: PSEUDO_DMA_DIRECT2D  (Sync Engine slot 13)", size=8.6, weight="bold",
          color=INK, family="DejaVu Sans Mono")
    label(ax, 3, 18.8, "wait S[72] >= 16      semaphore=62  +16 on completion", size=8,
          family="DejaVu Sans Mono")
    label(ax, 3, 15.5, "issue ~0.55 µs, completion lands 2-7 µs later", size=8, color=ORANGE)
    arrow(ax, 52, 19, 59, 19, lw=1.5)
    box(ax, 60, 13, 59, 12, "", fc=LIGHT_GRAY, ec=AXIS)
    label(ax, 62, 22.3, "after: EVENT_SEMAPHORE  (same slot, same 64 bytes)", size=8.6,
          weight="bold", color=INK, family="DejaVu Sans Mono")
    label(ax, 62, 18.8, "wait S[72] >= 16   SEM_ADD_IMM_COMPLETE  S[62] += 16", size=8,
          family="DejaVu Sans Mono")
    label(ax, 62, 15.5, "~0.03 µs, S[62] moves the instant the real producer is done", size=8,
          color=INK2)
    label(ax, 1, 8.5, "Every waiter on S[62] is untouched. Update mode 21 (SEM_ADD_IMM_COMPLETE) is in "
          "the ISA enum but the compiler never emits it;\nthe compiled kernels only use '+1 on "
          "completion' (mode 19). Trap: wait threshold and add amount share one 4-byte value field.",
          size=8.2, color=INK2, va="top")
    save(fig, "fig14b_step2_sem_add.png")


def step3():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.6), sharex=True)
    rows = [("Scalar", "Scalar Engine"), ("GpSimd", "GPSIMD Engine"), ("Sync", "Sync Engine")]
    draw_timeline(axes[0], 2, 36, "Before (iteration 2): the GPSIMD Engine issues 25 weight-load DGE DMAs "
                  "back to back while the Scalar Engine waits", rows=rows, dma=False)
    draw_timeline(axes[1], 3, 36, "After (iteration 3): 8 of them issued from the Scalar Engine's 8 unused "
                  "DGE queues; the GPSIMD Engine keeps a cheap relay", rows=rows, dma=False)
    axes[1].set_xlabel("time since the kernel's first instruction (µs)")
    legend(fig, -0.1, classes=[c for c in __import__("fig_data").CLASSES[:4]])
    save(fig, "fig15_step3_queues.png")


def step4():
    fig, ax = canvas(12, 4.0, 120)
    label(ax, 1, 38, "Activation-table loads in the per-head tail (1.283 µs each)", size=10,
          weight="bold", color=INK)

    def ribbon(y, seq, title):
        label(ax, 1, y + 2, title, size=8.5, color=INK2)
        x = 22
        for kind in seq:
            if kind in ("L2", "L4"):
                box(ax, x, y, 4.2, 4, kind.replace("L", "load\n"), fc=YELLOW, ec=YELLOW, size=6.5,
                    radius=0.3, color=INK)
                x += 4.8
            else:
                c = {"s": GREEN, "r": VIOLET, "a": MUTED}[kind]
                box(ax, x, y, 2.4, 4, "", fc=c, ec=c, radius=0.3)
                x += 2.9
        return x
    before = ["L4", "r", "r", "a", "L2", "s", "L4", "r", "a", "a", "L2", "s", "s", "s", "a", "L4",
              "r", "a", "L2", "s", "L4", "r", "r", "a", "L2", "s", "L4", "r", "a"]
    after = ["L4", "r", "r", "a", "L2", "s", "s", "s", "s", "s", "s", "a", "a", "L4", "r", "a",
             "r", "r", "a", "r", "a", "a"]
    ribbon(25, before, "compiled order")
    ribbon(15, after, "after steps 4-5")
    for c, n, x in ((GREEN, "silu (needs table 2)", 22), (VIOLET, "rsqrt (needs table 4)", 50),
                    (MUTED, "table-agnostic (square, copy, CAST)", 78)):
        box(ax, x, 6.5, 2.4, 3, "", fc=c, ec=c, radius=0.3)
        label(ax, x + 3.5, 8, n, size=8)
    label(ax, 1, 2.5, "Illustrative sequence (not slot-exact). Tail loads went 4,2,4,2,4,2,4 -> 4,2,4; "
          "kernel total 13 -> 7. No set holds both silu and rsqrt, so 5 loads is the floor.",
          size=8, color=INK2)
    save(fig, "fig16_step4_tables.png")


def step5():
    fig, ax = canvas(12, 3.8, 120)
    ax.set_ylim(10, 48)
    label(ax, 1, 46, "A partition -> row transpose costs one DMA descriptor per partition per column",
          size=10, weight="bold", color=INK)

    def grid(x0, y0, cols, w=3.2, h=1.25, rows=8, colors=(BLUE, VIOLET, GREEN, ORANGE)):
        for r in range(rows):
            for c in range(cols):
                box(ax, x0 + c * w, y0 - r * h, w - 0.3, h - 0.25, "", fc=colors[c % 4], ec=colors[c % 4],
                    radius=0.15)
    label(ax, 1, 40, "before: 4 transfers x 128 descriptors (2 B each)", size=8.6, color=INK)
    grid(3, 37, 4, w=3.2)
    label(ax, 3, 26, "SBUF: 128 partitions,\n4 columns 256 B apart", size=7.5, color=INK2, va="top")
    for k, c in enumerate((BLUE, VIOLET, GREEN, ORANGE)):
        arrow(ax, 17, 35 - k * 2.2, 27, 38 - k * 4.2, color=c, lw=1.0)
        box(ax, 28, 36.6 - k * 4.2, 22, 2.6, "", fc=c, ec=c, radius=0.2)
    label(ax, 51, 31, "4 rows in\npartition 0:\n512 descriptors,\n~5 µs of DGE\ntime", size=7.8,
          color=ORANGE, va="center")
    label(ax, 62, 40, "after: pack by compute, then 1 transfer x 128 descriptors (8 B each)", size=8.6,
          color=INK)
    grid(62, 37, 4, w=3.2)
    arrow(ax, 76, 33, 80, 33, lw=1.2)
    label(ax, 76, 35.5, "4 CASTs", size=7.5)
    for r in range(8):
        for c in range(4):
            box(ax, 81 + c * 1.6, 37 - r * 1.25, 1.3, 1.0, "", fc=(BLUE, VIOLET, GREEN, ORANGE)[c],
                ec=(BLUE, VIOLET, GREEN, ORANGE)[c], radius=0.1)
    label(ax, 80, 26, "8 contiguous bytes\nper partition", size=7.5, color=INK2, va="top")
    arrow(ax, 88, 33, 93, 33, lw=1.2)
    for k in range(24):
        c = (BLUE, VIOLET, GREEN, ORANGE)[k % 4]
        box(ax, 94 + k * 1.05, 32, 0.9, 2.6, "", fc=c, ec=c, radius=0.1)
    label(ax, 94, 29.5, "one interleaved row; consumers read\nwith stride 4 (incl. the Tensor Engine's reverse-"
          "stride\nLDWEIGHTS: every step x4)", size=7.5, color=INK2, va="top")
    label(ax, 1, 16, "Applied to both transpose groups: -1.4 µs and -2.8 µs. The DMA engine retires "
          "~1,500 tiny transpose descriptors serially (~8-15 ns each),\nso in that window only "
          "removing descriptors helped - three edits that merely started transfers earlier were "
          "correct and lost time.", size=8.2, color=INK2, va="top")
    save(fig, "fig17_step5_pack.png")


def step6():
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ev = [("Sync Engine slot 18: DGE transpose (the real data)", 35.80, 36.41, BLUE, 3, (35.80, 3.38, "bottom")),
          ("S[67] += 16: its completion", 37.80, 37.80, BLUE, 3, (37.95, 3.0, "center")),
          ("Sync Engine slot 31: trigger of a 2-byte dummy DMA", 39.00, 39.10, MUTED, 2,
           (36.6, 2.38, "bottom")),
          ("S[28] += 16: dummy completes", 40.39, 40.39, MUTED, 2, (40.55, 2.0, "center"))]
    for name, s, e, c, y, (tx, ty, va) in ev:
        if e > s + 0.05:
            ax.broken_barh([(s, e - s)], (y - 0.3, 0.6), facecolors=c)
        else:
            ax.plot([s], [y], marker="v", color=c, ms=9)
        ax.text(tx, ty, name, fontsize=8, color=INK2, va=va)
    ax.broken_barh([(37.85, 0.5)], (0.7, 0.6), facecolors=GREEN)
    ax.broken_barh([(40.44, 0.5)], (0.7, 0.6), facecolors=ORANGE)
    ax.text(38.45, 1.0, "Scalar Engine slot 55\nstarts (after the edit)", fontsize=8, color=GREEN, va="center")
    ax.text(41.05, 1.0, "Scalar Engine slot 55\nstarts (compiled gate)", fontsize=8, color=ORANGE, va="center")
    ax.annotate("", xy=(37.85, 1.45), xytext=(40.44, 1.45),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(39.15, 1.6, "2.6 µs", ha="center", fontsize=8.5, color=INK, weight="bold")
    ax.set_yticks([3, 2, 1], ["Sync Engine / DMA", "dummy", "Scalar Engine"])
    ax.set_ylim(0.3, 3.9)
    ax.set_xlim(35.4, 43.2)
    ax.set_xlabel("time (µs), iteration-5 profile")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("One byte: Scalar Engine slot 54 'wait S[28] >= 16' -> 'wait S[67] >= 16'", fontsize=10)
    save(fig, "fig18_step6_gate.png")


def step7():
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    for k in range(8):
        ax.broken_barh([(k * 0.69, 0.6)], (1.7, 0.6), facecolors=ORANGE if k % 2 == 0 else "#f3a07c")
    ax.text(5.6, 2.0, "8 DMAs x (0.6 µs fixed + 128 desc.) = 5.5 µs of issue", va="center",
            fontsize=8.5, color=INK2)
    ax.broken_barh([(0, 0.6 + 0.00071 * 1024)], (0.7, 0.6), facecolors=BLUE)
    ax.text(1.5, 1.0, "1 DMA x (0.6 µs + 1024 desc.) = 1.3 µs  (tile axis = dim 1 of the access pattern)",
            va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([2, 1], ["8 tile loads", "merged"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0.3, 2.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("engine time to issue (µs), cost model fitted on the profiles")
    ax.set_title("Merging tile loads buys issue time; then split again along the consumer boundary "
                 "(4 + 4 tiles)", fontsize=10)
    save(fig, "fig19_step7_merge.png")


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
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 2.4]})
    tot1 = {k: per[1].get(k, 0) for k in CAT}
    tot2 = {k: sum(per[i].get(k, 0) for i in range(2, 8)) for k in CAT}
    for x, tot in ((0, tot1), (1, tot2)):
        b = 0
        for k, (_, c) in CAT.items():
            v = tot[k]
            a0.bar(x, v, bottom=b, color=c, width=0.6, edgecolor=SURFACE, linewidth=1.5)
            if v > 4:
                a0.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=8.5,
                        weight="bold")
            b += v
        a0.text(x, b + 2, f"{b:.0f} µs", ha="center", fontsize=8.5, color=INK)
    a0.set_xticks([0, 1], ["step 1", "steps 2-7"])
    a0.set_ylim(0, 138)
    a0.set_ylabel("latency removed (µs)")
    a0.yaxis.grid(True, color=GRID, lw=0.6)
    a0.set_axisbelow(True)
    a0.set_title("Where the 173 µs came from", fontsize=10)
    xs = np.arange(2, 8)
    b = np.zeros(6)
    for k, (_, c) in CAT.items():
        v = np.array([per[i].get(k, 0) for i in xs])
        a1.bar(xs, v, bottom=b, color=c, width=0.62, edgecolor=SURFACE, linewidth=1.5)
        b += v
    for x, t in zip(xs, b):
        a1.text(x, t + 0.3, f"{t:.1f}", ha="center", fontsize=8.5, color=INK)
    a1.set_xticks(xs, [f"step {i}" for i in xs])
    a1.yaxis.grid(True, color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    a1.set_title("Steps 2-7 by lever type: 22.5 of 51 µs needed a non-NKI lever", fontsize=10)
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in CAT.values()], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8.3)
    save(fig, "fig23_attribution.png")
    return {k: (round(tot1[k], 1), round(tot2[k], 1)) for k in CAT}


def control_surface():
    fig, ax = canvas(12, 5.6, 120)
    cols = [(38, "NKI 0.6 source"), (62, "compiler (neuronx-cc)"), (88, "NEFF edit")]
    label(ax, 1, 53.5, "Who decides what ends up in the instruction stream", size=10.5, weight="bold",
          color=INK)
    for x, t in cols:
        label(ax, x + 9, 49.5, t, size=9, weight="bold", color=INK2, ha="center")
    rows = [("which ISA op, operands, dtypes, tile shapes", "full", "lowers", "full"),
            ("where a tensor lives (SBUF / PSUM / HBM), SBUF address", "full*", "allocates", "full"),
            ("which engine runs a compute op", "most ops", "default", "any engine with the opcode"),
            ("which engine issues a DMA", "sync / scalar / gpsimd", "default", "+ position in stream"),
            ("which DMA queue + semaphore a transfer uses", "no", "decides", "full"),
            ("DMA descriptors, queued-block triggering", "access pattern", "decides", "full"),
            ("order of instructions on one engine", "program order", "reorders", "exact"),
            ("which semaphore each instruction waits on, threshold", "no", "derives", "full"),
            ("signal without moving data (SEM_ADD_IMM)", "no", "never emits", "full"),
            ("activation-table loads: count and placement", "no", "inserts", "full")]
    marks = {"no": RED, "never emits": RED}
    for k, (what, nki, cc, neff) in enumerate(rows):
        y = 45 - k * 4.4
        if k % 2 == 0:
            box(ax, 0.5, y - 2.1, 119, 4.2, "", fc="#f4f3ef", ec="#f4f3ef", radius=0.3, z=0)
        label(ax, 1.5, y, what, size=8.3, color=INK)
        for (x, _), val, fill in ((cols[0], nki, BLUE), (cols[1], cc, MUTED), (cols[2], neff, ORANGE)):
            c = marks.get(val, fill)
            box(ax, x, y - 1.6, 18, 3.2, val, fc=TINT.get(c, LIGHT_GRAY), ec=c, size=7.6, radius=0.4)
    label(ax, 1.5, 1.5, "* nkilib kernels are compiled with NKI's own allocation and scheduling; "
          "reused standalone kernels still take and return HBM tensors.", size=7.8, color=INK2)
    save(fig, "fig07_control_surface.png")


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

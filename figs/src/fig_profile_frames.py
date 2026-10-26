"""Frame each optimization on the neuron-explorer screenshots (figs/gdn_iter{0,7}_profile.png).

Times are the profile's raw timestamps (µs), which is what the neuron-explorer timeline axis shows;
they come from the median-capture profiles in experiments/.../blog_profiles/. Pixel geometry
(grid lines, row separators) was measured on the screenshots themselves.
"""
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import style  # noqa: E402,F401  (registers Inter / JetBrains Mono)

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STEP = {1: "#ff7a59", 2: "#ffd166", 3: "#4cc9f0", 4: "#c3a6ff", 5: "#7bd389", 6: "#ff6fb5",
        7: "#ffffff"}
BADGE = "①②③④⑤⑥⑦"

SHOTS = {
    "iter0": dict(png="gdn_iter0_profile.png", x0=267.5, px_per_us=9.1286,
                  rows={"Sync": (258, 325), "Tensor": (326, 433), "Vector": (542, 649),
                        "Scalar": (650, 757), "GpSimd": (758, 845)}),
    "iter7": dict(png="gdn_iter7_profile.png", x0=255.5, px_per_us=35.821,
                  rows={"Sync": (232, 299), "Tensor": (300, 427), "Vector": (616, 743),
                        "Scalar": (744, 851), "GpSimd": (852, 939)}),
}

# (steps, t0_us, t1_us, first_row, last_row, dashed, legend text)
FRAMES = {
    "iter0": [
        ((1,), 38.8, 203.8, "Sync", "GpSimd", True,
         "38-204 µs: engines mostly idle. Each SBUF->DRAM->SBUF hand-off writes 128 two-byte "
         "elements into DRAM (the DMA bursts on the Sync, Scalar and GPSIMD Engines) and its completion "
         "arrives 17-22 µs later."),
        ((2,), 59.2, 63.6, "Scalar", "Scalar", False,
         "The Scalar Engine issues 8 of the 2-byte DRAM writes itself; step 1 reroutes them into SBUF, step 2 "
         "turns them into compute CASTs (0.25 µs, no completion hop)."),
        ((2,), 130.2, 143.6, "Scalar", "Scalar", False, None),
        ((3, 7), 13.0, 34.5, "GpSimd", "GpSimd", False,
         "The GPSIMD Engine issues 32 input/weight-load DGE DMAs back to back (~0.66 µs each). Step 2 merges the "
         "8 conv_weight tiles, step 3 moves 8 loads to the Scalar Engine, step 7 merges the in_proj tiles."),
        ((4, 5), 204.0, 225.4, "Scalar", "Scalar", False,
         "Scalar Engine tail: 7 activation-table loads alternating silu / rsqrt (1.28 µs each)."),
        ((5,), 225.8, 229.5, "Sync", "Sync", False,
         "output_0 goes through DRAM: 9 queued DMAs for 1 KB."),
    ],
    "iter7": [
        ((3, 7), 12.4, 18.6, "Scalar", "Scalar", False,
         "The Scalar Engine issues the weight loads as 3 merged DGE DMAs (in_proj: 1,024 descriptors; table 2 "
         "in two halves); the first table load runs right after them."),
        ((2, 3), 13.0, 20.2, "GpSimd", "GpSimd", False,
         "GPSIMD Engine: the merged conv_weight load goes first; most of its old DMA slots are now "
         "0.03-µs semaphore relays."),
        ((4, 5), 26.4, 32.1, "GpSimd", "GpSimd", False,
         "GPSIMD Engine CASTs replace per-partition DMAs and pack 4 columns per partition for one "
         "transpose (step 7 hoisted this run above unrelated DMAs)."),
        ((1, 5), 31.7, 39.1, "Sync", "Sync", False,
         "Sync Engine: the former DRAM round trips are SBUF->SBUF, and with 4 columns packed per "
         "descriptor it issues 5 DGE DMAs (38, mostly through DRAM, in the baseline)."),
        ((6,), 34.2, 36.6, "Scalar", "Scalar", False,
         "One byte: the gate before the 4 scale ops waits on the real transfer, not on a "
         "dummy DMA triggered after it."),
        ((4, 5), 36.6, 45.2, "Scalar", "Scalar", False,
         "Tail: 2 activation-table loads (7 in the baseline); the silu ops sit in one table-2 "
         "region, hoisted above two barriers that only gate the Vector Engine (step 6)."),
        ((5,), 55.5, 56.4, "Sync", "Sync", False,
         "output_0: one 1 KB store from SBUF (was 9 queued DMAs through DRAM)."),
    ],
}


def badge(steps):
    return "".join(BADGE[s - 1] for s in steps)


def draw(key, out, title):
    cfg, frames = SHOTS[key], FRAMES[key]
    img = Image.open(os.path.join(FIGS, cfg["png"])).convert("RGB")
    W, H = img.size
    items = sorted([f for f in frames if f[6]], key=lambda f: f[0])
    wrapped = [textwrap.wrap(f[6], 110) for f in items]
    line_h, gap = 48, 26
    leg_h = 110 + sum(max(len(w) * line_h, 60) + gap for w in wrapped)
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, (H + leg_h) / dpi), dpi=dpi, facecolor="#1b1d22")
    ax = fig.add_axes([0, leg_h / (H + leg_h), 1, H / (H + leg_h)])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x = lambda t: cfg["x0"] + t * cfg["px_per_us"]  # noqa: E731
    for steps, t0, t1, r0, r1, dashed, _ in frames:
        c = STEP[steps[0]]
        y0, y1 = cfg["rows"][r0][0] + 3, cfg["rows"][r1][1] - 3
        ax.add_patch(Rectangle((x(t0), y0), x(t1) - x(t0), y1 - y0, fill=False, ec=c, lw=4,
                               ls=(0, (6, 4)) if dashed else "-", zorder=5))
        ax.text(x(t0) + 4, y0 + 4, badge(steps), color="#111111", fontsize=28, weight="bold",
                va="top", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=c, ec="none"))
    lg = fig.add_axes([0, 0, 1, leg_h / (H + leg_h)])
    lg.set_xlim(0, W)
    lg.set_ylim(leg_h, 0)
    lg.axis("off")
    lg.text(30, 26, title, color="#f0efec", fontsize=31, weight="semibold", va="top")
    y = 110
    for (steps, *_rest), lines in zip(items, wrapped):
        lg.text(30, y, badge(steps), color="#111111", fontsize=27, weight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc=STEP[steps[0]], ec="none"))
        lg.text(190, y + 2, "\n".join(lines), color="#e6e4dc", fontsize=26, va="top",
                linespacing=1.3)
        y += max(len(lines) * line_h, 60) + gap
    fig.savefig(os.path.join(FIGS, out), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    draw("iter0", "fig09_profile_baseline_framed.png",
         "Baseline (217.5 µs): what each step fixes - numbers are the optimization steps")
    draw("iter7", "fig19_profile_optimized_framed.png",
         "Optimized (44.4 µs): where each step shows up in the trace")


def utilization(out="fig20_profile_utilization.png"):
    """Your neuron-explorer DMA- and FLOPs-utilization panels, baseline | optimized."""
    from PIL import ImageDraw, ImageFont
    panels = [["gdn_iter0_dma.png", "gdn_iter7_dma.png"],
              ["gdn_iter0_util.png", "gdn_iter7_util.png"]]
    w, pad, head = 1160, 24, 92
    ims = [[Image.open(os.path.join(FIGS, p)).convert("RGB") for p in row] for row in panels]
    ims = [[im.resize((w, round(im.height * w / im.width))) for im in row] for row in ims]
    hs = [max(im.height for im in row) for row in ims]
    canvas = Image.new("RGB", (2 * w + 3 * pad, head + sum(hs) + 3 * pad), "#1b1d22")
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(os.path.join(style.FONTS, "Inter-SemiBold.ttf"), 52)
    for c, t in enumerate(["Baseline: 217.5 µs", "Optimized: 44.4 µs"]):
        d.text((pad + c * (w + pad) + 10, 20), t, fill="#f0efec", font=font)
    y = head
    for r, row in enumerate(ims):
        for c, im in enumerate(row):
            canvas.paste(im, (pad + c * (w + pad), y))
        y += hs[r] + pad
    canvas.save(os.path.join(FIGS, out))


if __name__ == "__main__":
    utilization()

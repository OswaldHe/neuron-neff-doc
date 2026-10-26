"""Background schematics: compile flow, NEFF anatomy, instruction slot layout, DMA paths,
semaphore model. Writes PNGs into figs/."""
import os

from diagram import arrow, box, canvas, label
from style import (AQUA, AXIS, BLUE, GREEN, INK, INK2, LIGHT_BLUE, LIGHT_GRAY, LIGHT_ORANGE,
                   MAGENTA, MUTED, ORANGE, SURFACE, VIOLET, YELLOW, plt, save_fig)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TINT = {BLUE: LIGHT_BLUE, ORANGE: LIGHT_ORANGE, AQUA: "#d3f1e6", YELLOW: "#fbecc7",
        VIOLET: "#e2def8", MAGENTA: "#f9e2eb", GREEN: "#d2ead2"}


def save(fig, name):
    save_fig(fig, os.path.join(OUT, name))


def compile_flow():
    fig, ax = canvas(12, 5.0, 120)
    y = 36
    stages = [
        (1, "NKI kernel\n(Python,\n@nki.jit)", "nki.isa /\nnki.language calls", BLUE),
        (18, "NKI IR\n(MLIR, nisa\ndialect)", "module.mlir:\n122 dma_copy,\n28 matmul, ...", BLUE),
        (35, "neuronx-cc\nfrontend", "HLOToTensorizer,\nTensorizer: mostly\npass-through for NKI", VIOLET),
        (52, "Walrus\nbackend\n(BIR)", "", VIOLET),
        (69, "NeffWrapper", "packs header +\ngzip/tar payload", VIOLET),
        (86, "NEFF", "engine streams,\nDMA descriptors,\nmemory map, tables", ORANGE),
        (103, "Neuron\nruntime +\nNeuronCore", "expands pseudo-\ninstructions, builds\nDMA rings, runs", GREEN),
    ]
    for x, title, sub, c in stages:
        box(ax, x, y, 14, 9.5, title, fc=TINT[c], ec=c, size=9, weight="bold")
        if sub:
            label(ax, x + 7, y - 1.2, sub, size=7.8, ha="center", va="top")
    for i in range(len(stages) - 1):
        arrow(ax, stages[i][0] + 14.2, y + 4.75, stages[i + 1][0] - 0.2, y + 4.75)
    groups = [
        ("translate + verify", "translate_nki_ast_to_bir, birverifier, unroll"),
        ("memory allocation", "coloring_allocator_psum / _sb / _dram, address_rotation_*"),
        ("dependencies + order", "build_fdeps, anti_dependency_analyzer, pre_sched, post_sched, dep_reduction"),
        ("DMA lowering", "alloc_queues, assign_dma_engine, lower_dynamic_dma, lower_dma, coalesce_dma_blocks"),
        ("synchronisation", "alloc_semaphores, lower_sync"),
        ("engine lowering", "lower_act, optimize_act_control (table loads), lower_dve, mem2reg"),
        ("emit", "debug_info_writer -> engine streams + JSON"),
    ]
    gy = 25
    box(ax, 1, 0.8, 83, gy + 0.2, "", fc=SURFACE, ec=VIOLET, style=(0, (3, 2)))
    label(ax, 2.5, gy - 0.6, "inside the Walrus backend: ~110 passes (names from log-neuron-cc.txt, in run order)",
          size=8.5, color=VIOLET, weight="bold", va="top")
    for k, (g, p) in enumerate(groups):
        yy = gy - 4.0 - k * 2.9
        label(ax, 3, yy, g, size=8.3, color=INK, weight="bold", va="center")
        label(ax, 21, yy, p, size=7.5, color=INK2, va="center", family="monospace")
    arrow(ax, 59, y - 0.2, 59, gy + 1.1, color=VIOLET, ls=(0, (3, 2)))
    box(ax, 86, 8, 14, 14, "this blog edits\nhere: the NEFF\nafter the\ncompiler, before\nthe runtime",
        fc=TINT[ORANGE], ec=ORANGE, size=8.2, color=INK)
    arrow(ax, 93, 22.2, 93, y - 9.5, color=ORANGE)
    save(fig, "compile_flow_detail.png")


def neff_anatomy():
    fig, ax = canvas(11, 6.6, 110)
    # file outline
    box(ax, 1, 1, 52, 63.5, "", fc=SURFACE, ec=AXIS)
    label(ax, 3, 62, "kernel.neff", size=11, weight="bold", color=INK)
    box(ax, 3, 52.5, 48, 7, "NEFF header - 1024 bytes\nversions, payload size, MD5 hash, UUID, feature bits",
        fc=LIGHT_GRAY, ec=AXIS, size=8.5)
    box(ax, 3, 3, 48, 48, "", fc=SURFACE, ec=AXIS, style=(0, (3, 2)))
    label(ax, 5, 48.8, "NEFF payload - gzip'd tar archive", size=9, weight="bold", color=INK2)
    top = [("neff.json", "graph: inputs, outputs, one kernel node"),
           ("kelf-0.json", "list of subgraphs, target 'cayman' (trn2)"),
           ("info.json", "compiler / NKI versions")]
    for k, (n, d) in enumerate(top):
        yy = 45 - k * 3.2
        label(ax, 6, yy, n, size=8.5, color=INK, family="monospace")
        label(ax, 19, yy, d, size=8, color=INK2)
    box(ax, 5, 5, 44, 30.5, "", fc=SURFACE, ec=AXIS)
    label(ax, 7, 33.6, "sg00/   one subgraph per physical NeuronCore", size=8.5, weight="bold",
          color=INK2)
    sub = [("def.json", "variables (SBUF, inputs, outputs, DRAM scratch),\nDMA queues + their semaphore sets", VIOLET),
           ("PE0.bin  DVE0.bin  Activation0.bin\nPool0.bin  SP0.bin", "engine streams: 64-byte instruction slots", BLUE),
           ("SP0.json (+ other *0.json)", "queued DMA transfers and their descriptors", ORANGE),
           ("*_bkt.bin / *_ctrl.bin", "activation function tables (+ Vector Engine microcode)", YELLOW),
           ("debug_info_*.dbg", "slot -> BIR instruction -> NKI source line", MUTED)]
    for k, (n, d, c) in enumerate(sub):
        yy = 29.4 - k * 5.15
        box(ax, 6.5, yy - 2.2, 1.2, 4.4, "", fc=c, ec=c, radius=0.3)
        label(ax, 9, yy + 0.9, n, size=8, color=INK, family="monospace", va="center")
        label(ax, 9, yy - 1.4, d, size=7.8, color=INK2, va="center")
    # right: what each part controls
    label(ax, 58, 62, "What controls performance, and where it lives", size=11, weight="bold",
          color=INK)
    rows = [("what each engine executes, in what order", "<Engine>0.bin", BLUE),
            ("operand addresses, shapes, dtypes, ops", "<Engine>0.bin fields", BLUE),
            ("who waits for whom (all synchronisation)", "events block of every slot", MAGENTA),
            ("DGE DMA: source, destination, pattern", "PSEUDO_DMA_DIRECT2D slots", ORANGE),
            ("queued DMA: descriptors", "SP0.json 'dma' list", ORANGE),
            ("which engine owns which DMA queue", "def.json 'dma_queue'", VIOLET),
            ("DRAM scratch buffers", "def.json 'var'", VIOLET),
            ("activation function tables", "ACT_TABLE_LOAD + table files", YELLOW)]
    for k, (what, where, c) in enumerate(rows):
        yy = 56 - k * 6.8
        box(ax, 58, yy - 2.6, 50, 5.4, "", fc=TINT.get(c, LIGHT_GRAY), ec=c)
        label(ax, 60, yy + 0.9, what, size=8.5, color=INK)
        label(ax, 60, yy - 1.2, where, size=8, color=INK2, family="monospace")
    save(fig, "neff_anatomy_detail.png")


def slot_layout():
    fig, ax = canvas(9, 3.15, 100)
    x0, cw = 2, 1.5  # 64 bytes span 96 units

    def strip(y, title, fields):
        label(ax, x0, y + 9.2, title, size=11, weight="semibold", color=INK)
        for a, b, name, c in fields:
            box(ax, x0 + a * cw, y, (b - a + 1) * cw, 7, name, fc=TINT.get(c, LIGHT_GRAY), ec=c,
                size=9, radius=0.3, lw=1.0)
        for byte in range(0, 65, 8):
            label(ax, x0 + byte * cw, y - 1.5, str(byte), size=9, color=MUTED, ha="center")

    strip(19, "ACTIVATE (Scalar Engine)", [
        (0, 3, "header", MUTED), (4, 11, "events:\nwait + update", MAGENTA),
        (12, 15, "imm,\nscale,\nbias src", AXIS), (16, 19, "src\naddr", BLUE),
        (20, 31, "src pattern\nstep[3] num[3]", BLUE), (32, 35, "dtypes,\nchan,\nfunc", AXIS),
        (36, 47, "scale | bias | imm\n(value or pointer)", AXIS), (48, 51, "dst\naddr", GREEN),
        (52, 63, "dst pattern\nstep[3] num[3]", GREEN)])
    strip(3, "DGE DMA (PSEUDO_DMA_DIRECT2D)", [
        (0, 3, "header", MUTED), (4, 11, "events:\nwait + update", MAGENTA),
        (12, 15, "cfg,\nsem,\n+16, op", ORANGE), (16, 23, "src addr\n(var/table+off)", BLUE),
        (24, 31, "src step[2]\n(bytes)", BLUE), (32, 39, "src num[2],\nelem size", BLUE),
        (40, 47, "dst addr\n(var/table+off)", GREEN), (48, 55, "dst step[2]\n(bytes)", GREEN),
        (56, 63, "dst num[2],\nelem, dtypes", GREEN)])
    label(ax, x0, 33.6, "Bytes 0-11 are the same for every opcode: the opcode, then ONE semaphore wait "
          "and ONE semaphore update", size=10.5, color=INK2)
    save(fig, "fig03_slot_layout.png")


def dma_paths():
    fig, ax = canvas(9, 5.4, 100)
    # queued
    box(ax, 1, 31, 98, 28, "", fc=SURFACE, ec=AXIS)
    label(ax, 3, 56.4, "Queued DMA: descriptors prepared at load time", size=11.5, weight="semibold",
          color=INK)
    box(ax, 3, 44.5, 24, 8.5, "SP0.json 'dma' list\nid, queue, sem, descriptors", fc=LIGHT_ORANGE,
        ec=ORANGE)
    box(ax, 3, 34.5, 24, 8.5, "Sync Engine stream\nPSEUDO_DMATRIGGER\n(queue, block id)", fc=LIGHT_BLUE,
        ec=BLUE)
    box(ax, 40, 39, 25, 10, "runtime builds 16-byte\nhardware descriptors\nin DMA rings", fc=LIGHT_GRAY,
        ec=AXIS)
    box(ax, 76, 39, 21, 10, "DMA engines\n(16 rings per queue)", fc="#d3f1e6", ec=AQUA)
    arrow(ax, 27.3, 48.7, 39.7, 46)
    arrow(ax, 27.3, 38.7, 39.7, 42, ls=(0, (3, 2)))
    label(ax, 33.5, 36.2, "trigger\n~0.1 µs", size=9.5, ha="center")
    arrow(ax, 65.3, 44, 75.7, 44)
    label(ax, 40, 35.2, "completion: S[queue sem] += 16; blocks of a queue run in order", size=10)
    # DGE
    box(ax, 1, 3, 98, 26, "", fc=SURFACE, ec=AXIS)
    label(ax, 3, 26.4, "DGE DMA: descriptors generated by hardware at run time", size=11.5,
          weight="semibold", color=INK)
    box(ax, 3, 12.5, 31, 11, "Sync / Scalar / GPSIMD Engine stream\nPSEUDO_DMA_DIRECT2D\n"
        "src, dst, 2-D pattern, elem size, sem", fc=LIGHT_BLUE, ec=BLUE)
    box(ax, 46, 13.5, 20, 9, "descriptor\ngenerator (DGE)", fc=LIGHT_GRAY, ec=AXIS)
    box(ax, 76, 13.5, 21, 9, "DMA engines", fc="#d3f1e6", ec=AQUA)
    arrow(ax, 34.3, 18, 45.7, 18)
    label(ax, 40, 20.2, "issue ~0.6 µs", size=9.5, ha="center")
    arrow(ax, 66.3, 18, 75.7, 18)
    label(ax, 3, 10.3, "one descriptor per contiguous run (e.g. 128 partitions x 2 B = 128 descriptors)\n"
          "queue = the instruction's semaphore, owned by the issuing engine\n"
          "completion: S[sem] += 16, reported ~1.3-4 µs after the data lands", size=10, va="top")
    save(fig, "fig04_dma_paths.png")


def semaphores():
    fig, ax = canvas(9, 5.25, 100)
    lanes = [("GPSIMD\nEngine", 44), ("Scalar\nEngine", 33), ("Vector\nEngine", 22), ("Sync\nEngine", 11)]
    for n, y in lanes:
        label(ax, 0.5, y, n, size=10.5, weight="semibold", color=INK)
        ax.plot([10, 99], [y, y], color=LIGHT_GRAY, lw=1, zorder=0)
    mono = dict(family="monospace", size=9.5)
    # GPSIMD Engine DGE DMA
    box(ax, 11, 41, 11, 6, "DGE DMA\nsem=13", fc=LIGHT_ORANGE, ec=ORANGE, **mono)
    box(ax, 22, 43.2, 14, 1.6, "", fc="#d3f1e6", ec=AQUA, radius=0.3)
    label(ax, 29, 48.6, "data moves, completion reported", size=9.5, ha="center")
    label(ax, 37, 44, "S[13] += 16", size=10, color=ORANGE, weight="bold", family="monospace")
    # Scalar Engine waits
    box(ax, 41, 30, 16, 6, "EVENT_SEMAPHORE\nwait S[13] >= 16", fc=LIGHT_GRAY, ec=AXIS, **mono)
    arrow(ax, 44, 42, 48, 36.2, color=ORANGE)
    box(ax, 58.5, 30, 12, 6, "ACTIVATE\nS[4]++", fc=LIGHT_BLUE, ec=BLUE, **mono)
    box(ax, 72, 30, 12, 6, "ACTIVATE\nS[4]++", fc=LIGHT_BLUE, ec=BLUE, **mono)
    # Vector Engine waits on S[4] >= 2
    box(ax, 82, 19, 16.5, 6, "TENSOR_TENSOR\nwait S[4] >= 2", fc=LIGHT_BLUE, ec=BLUE, **mono)
    arrow(ax, 80, 29.8, 86, 25.2, color=BLUE)
    label(ax, 79, 26.5, "the 2nd Scalar\nEngine completion", size=9.5, ha="right")
    # Sync Engine trigger waits on S[6]
    box(ax, 82, 8, 16.5, 6, "DMATRIGGER\nwait S[6] >= 1", fc=LIGHT_ORANGE, ec=ORANGE, **mono)
    arrow(ax, 90.25, 18.8, 90.25, 14.2, color=BLUE)
    label(ax, 89, 16.5, "S[6]++", size=9.5, ha="right", family="monospace")
    label(ax, 0.5, 57.5, "A wait counts producers since the start of the kernel, never a delta: 'S[4] >= 2' "
          "means\nthe 2nd Scalar Engine instruction that signals S[4] has finished.", size=10.5,
          color=INK2, va="top")
    label(ax, 0.5, 5.2, "S[3..6] = GPSIMD / Scalar / Tensor / Vector Engine completion counters (+1 each);\n"
          "DMA queue semaphores get +16 per finished transfer.", size=10, color=INK2, va="top")
    save(fig, "fig05_semaphores.png")


if __name__ == "__main__":
    compile_flow()
    neff_anatomy()
    slot_layout()
    dma_paths()
    semaphores()

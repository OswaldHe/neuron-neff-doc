#!/usr/bin/env python3
"""Build the instruction corpus: one row per NEFF instruction, joined to its debug info and to
the profiler's disassembly.

For every studied NEFF (artifacts/unpacked/<tag>) and its ingested rank-0 profile
(artifacts/profiles/<tag>/rank0), this writes artifacts/corpus/<tag>.parquet with:

    tag, sg, engine, idx            where the instruction lives: subgraph, engine stream, slot
    raw                             the 64 bytes of the slot
    op, op_name                     byte 0 and its name in neuron_isa's Opcode enum
    bir_id, bir_name                the BIR instruction the slot came from (asm debug info)
    kernel_file, kernel_line        the NKI source line of that BIR instruction
    bir_attrs                       all attrs of the BIR instruction, as JSON
    align_shift                     how far the profiler's compiler_pc label was off for this slot
    name_ok                         runtime opcode name agrees with the slot's opcode
    n_runtime                       how many runtime instructions the profiler saw for this slot
    p_opcode, p_operands            the profiler's disassembly of the first of them
    p_expansion                     JSON list of [pc, opcode, operands] for all of them
    p_compiler_opcode, p_bir_name   the profiler's own label for the row (its compiler_pc label
                                    can be off by the align_shift, so prefer bir_name)

Run with workspace/.venv (needs pandas + pyarrow):
    corpus.py [tag ...]
"""

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dbginfo  # noqa: E402

ART = os.path.join(HERE, "..", "artifacts")
ENGINES = [("Activation", "Activation0.bin", "Scalar"), ("DVE", "DVE0.bin", "Vector"),
           ("PE", "PE0.bin", "Tensor"), ("Pool", "Pool0.bin", "GpSimd"), ("SP", "SP0.bin", "Sync")]
SLOT = 64

OPCODE = {v: k for k, v in json.load(open(os.path.join(ART, "reference", "isa_enums.json")))["Opcode"].items()}
try:
    import isa as _isa
    VENDOR = _isa.OPCODES                   # vendor (neuron-dbg) opcode names, NeuronCore-v3
except Exception:                           # isa layouts not extracted yet
    VENDOR = {}

_RENAME = {"ACTIVATION_TABLE_LOAD": "ACT_TABLE_LOAD", "CROSS_LANE_REDUCE_ARITH": "CROSS_LANE_REDUCE",
           "CROSS_LANE_REDUCE_BITVEC": "CROSS_LANE_REDUCE"}


# Runtime opcodes each pseudo-instruction expands into, learned from the NEFFs whose profiler
# labels need no realignment (artifacts/reference/pseudo_expansions.json), plus three seen in
# the others. A pseudo opcode not listed here accepts any runtime opcode.
_EXP_PATH = os.path.join(ART, "reference", "pseudo_expansions.json")
EXPANSIONS = {k: set(v) for k, v in json.load(open(_EXP_PATH)).items()} if os.path.exists(_EXP_PATH) else {}
EXPANSIONS.setdefault("PSEUDO_GID_LOAD", {"MOVE"})
EXPANSIONS.setdefault("PSEUDO_TENSOR_STORE", {"TENSOR_STORE"})
EXPANSIONS.setdefault("PSEUDO_BRANCH_LABEL", set())


def compatible(op, runtime_name):
    """Can a runtime instruction named `runtime_name` come from a NEFF slot with opcode `op`?"""
    v = VENDOR.get(op, OPCODE.get(op, ""))
    if not v:
        return True
    if v.startswith("PSEUDO"):
        return runtime_name in EXPANSIONS[v] if v in EXPANSIONS else True
    v = _RENAME.get(v, v)
    for suf in ("_ARITH_OP", "_BITVEC_OP", "_ARITH", "_BITVEC"):
        if v.endswith(suf) and runtime_name == v[: -len(suf)]:
            return True
    return runtime_name == v or v.startswith(runtime_name)


def align(group_rows, slots):
    """Assign runtime rows (sorted by pc) to slot indices.

    The profiler's compiler_pc label runs ahead of the real slot index after some
    pseudo-instructions (seen: one extra per PSEUDO_GID_LOAD). Walk the rows keeping a running
    shift, and accept the label minus the shift when the opcode names agree."""
    out, k = [], 0
    for r in group_rows:
        c = int(r.compiler_pc)
        chosen = None
        for extra in range(0, 4):
            s = c - k - extra
            if 0 <= s < len(slots) and compatible(slots[s][0], r.opcode):
                chosen, k = s, k + extra
                break
        if chosen is None:
            s = c - k
            out.append((s, r, k, False))
        else:
            out.append((chosen, r, k, True))
    return out


def profile_rows(tag):
    p = os.path.join(ART, "profiles", tag, "rank0", "Instruction.parquet")
    if not os.path.exists(p):
        return None
    cols = ["engine", "pcore_idx", "pc", "compiler_pc", "opcode", "operands", "compiler_opcode",
            "bir_instruction_name", "nki_source_location", "start_ts"]
    df = pd.read_parquet(p, columns=cols)
    df = df[df.compiler_pc.notna()].copy()
    df["compiler_pc"] = df.compiler_pc.astype(int)
    # One row per distinct runtime pc; keep its first execution.
    df = df.sort_values("start_ts").drop_duplicates(["pcore_idx", "engine", "pc"])
    return df


def build(tag):
    unpacked = os.path.join(ART, "unpacked", tag, "payload")
    prof = profile_rows(tag)
    rows = []
    sgs = sorted(d for d in os.listdir(unpacked) if d.startswith("sg"))
    for sgi, sg in enumerate(sgs):
        d = os.path.join(unpacked, sg)
        for eng, binf, peng in ENGINES:
            raw = open(os.path.join(d, binf), "rb").read()
            asm = dbginfo.load(os.path.join(d, f"debug_info_asm_{eng}.dbg"))["instructions"]
            back = dbginfo.load(os.path.join(d, f"debug_info_backend_{eng}.dbg"))
            bmap = {i["id"]: i for i in back["instructions"]}
            files = back["kernel_filenames"]
            n = len(raw) // SLOT
            assert n == len(asm), (tag, sg, eng, n, len(asm))
            slots = [(raw[i * SLOT],) for i in range(n)]
            per_slot = {}
            if prof is not None:
                g = prof[(prof.pcore_idx == sgi) & (prof.engine == peng)].sort_values("pc")
                for s_idx, r, k, ok in align(list(g.itertuples()), slots):
                    per_slot.setdefault(s_idx, []).append((r, k, ok))
            for idx in range(n):
                slot = raw[idx * SLOT:(idx + 1) * SLOT]
                parents = asm[idx]["parent_ids"]
                b = bmap.get(parents[0]) if parents else None
                attrs = b["attrs"] if b else {}
                kf = attrs.get("kernel_file")
                got = per_slot.get(idx, [])
                exp = [[int(r.pc), r.opcode, r.operands] for r, _, _ in got]
                first = got[0][0] if got else None
                rows.append({
                    "tag": tag, "sg": sgi, "engine": eng, "idx": idx, "raw": slot,
                    "op": slot[0], "op_name": OPCODE.get(slot[0], "0x%02x" % slot[0]),
                    "bir_id": parents[0] if parents else None,
                    "bir_name": b["name"] if b else None,
                    "kernel_file": files[int(kf)] if (kf is not None and kf.isdigit() and int(kf) < len(files)) else kf,
                    "kernel_line": int(attrs["kernel_line"]) if attrs.get("kernel_line", "").isdigit() else None,
                    "bir_attrs": json.dumps(attrs) if attrs else None,
                    "align_shift": got[0][1] if got else None,
                    "name_ok": all(ok for _, _, ok in got) if got else None,
                    "n_runtime": len(exp),
                    "p_opcode": first.opcode if first is not None else None,
                    "p_operands": first.operands if first is not None else None,
                    "p_expansion": json.dumps(exp) if exp else None,
                    "p_compiler_opcode": first.compiler_opcode if first is not None else None,
                    "p_bir_name": first.bir_instruction_name if first is not None else None,
                })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ART, "corpus"), exist_ok=True)
    out.to_parquet(os.path.join(ART, "corpus", f"{tag}.parquet"), index=False)
    return out


def _summary(t):
    df = build(t)
    named = df.bir_name.notna().mean()
    seen = (df.n_runtime > 0).mean()
    bad = int((df.name_ok == False).sum())  # noqa: E712
    shifted = int((df.align_shift.fillna(0) > 0).sum())
    return (f"{t:26s} slots={len(df):8d} with_bir={named:6.1%} profiled={seen:6.1%} "
            f"name_mismatch={bad} realigned={shifted}")


def main(tags):
    from multiprocessing import Pool
    if not tags:
        tags = sorted(os.listdir(os.path.join(ART, "unpacked")))
    with Pool(min(7, len(tags))) as pool:
        for line in pool.imap_unordered(_summary, tags):
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

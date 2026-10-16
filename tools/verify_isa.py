#!/usr/bin/env python3
"""Check the decoder against the profiler, opcode by opcode, on the whole corpus.

For each instruction slot that the profiler disassembled, decode it with isa.py (struct from the
vendor dispatch table, artifacts/reference/opcode_dispatch_v3.json) and check that the decoded
fields reproduce what the profiler printed:

  wait    wait_mode GE/EQ etc. -> "S[<wait_idx>]" and "<op><semaphore_value>" in the text
  update  update_mode SEM_INC_COMPLETE -> "S[<update_idx>]" ... "++@complete"
  tensor  every access pattern -> "@0x<addr>[<steps>][<counts>]"
  chan    num_active_channels -> "channels=<n>"

Writes artifacts/reference/opcode_struct_v3.json (the decoder's opcode table, with the pass
rate of every check) and prints a summary. Opcodes that the dispatch table does not cover get
the best-scoring struct among all layouts, marked "inferred".

    verify_isa.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isa  # noqa: E402

ART = os.path.join(HERE, "..", "artifacts")
REF = os.path.join(ART, "reference")
PER_OPCODE = 400

WAIT_TXT = {1: "==", 2: "<", 3: "<=", 4: ">", 5: ">="}
KV = re.compile(r"(?<![\w\[])([a-z][a-z0-9_]*)=(-?0x[0-9a-fA-F]+|-?\d+(?:\.\d+)?)(?![\w\[@.])")


def tensor_groups(names):
    """Prefixes P such that P+'start_addr_addr_immediate', P+'step_elem', P+'num_elem' exist."""
    out = []
    for n in names:
        if n.endswith("start_addr_addr_immediate"):
            p = n[: -len("start_addr_addr_immediate")]
            if p + "step_elem" in names and p + "num_elem" in names:
                out.append(p)
    return out


def checks(ins, text):
    """List of (kind, passed) for one decoded instruction against its profiler text."""
    res = []
    names = {f["name"] for f in ins.fields()}
    if "events_wait_mode" in names:
        wm, wi, sv = ins["events_wait_mode"], ins["events_wait_idx"], ins["events_semaphore_value"]
        if wm in WAIT_TXT:
            res.append(("wait", f"S[{wi}]" in text and f"{WAIT_TXT[wm]}{sv}" in text))
        um, ui = ins["events_update_mode"], ins["events_update_idx"]
        if um == 19:
            res.append(("update", f"S[{ui}]" in text and "++@complete" in text))
    for p in tensor_groups(names):
        addr = ins[p + "start_addr_addr_immediate"]
        steps, nums = ins[p + "step_elem"], ins[p + "num_elem"]
        if not isinstance(steps, list) or not isinstance(nums, list):
            continue                        # 1-D formats print differently; not checked here
        if not any(nums):
            continue
        t = text.replace(" ", "")
        ok = False
        for pad in range(0, 3):            # the profiler pads 2-D patterns to 3 dims with 0
            st = steps + [0] * pad
            nu = nums + [0] * pad
            body = "0x%x[%s][%s]" % (addr, ",".join(map(str, st)), ",".join(map(str, nu)))
            if "@" + body in t or "=" + body in t:
                ok = True
                break
        res.append(("tensor", ok))
    # every numeric key=value the profiler prints, matched to a decoded field of that name
    vals = {}
    for f in ins.fields():
        v = ins[f["name"]]
        vals[f["name"]] = v
    for k, v in KV.findall(text):
        if k in ("channels",):
            continue
        cands = [n for n in names if n == k or n.endswith("_" + k)]
        if not cands:
            continue
        try:
            want = int(v, 16) if v.lower().startswith("0x") else (float(v) if "." in v else int(v))
        except ValueError:
            continue
        got = [vals[n] for n in cands]
        okk = any((isinstance(g, (int, float)) and abs(g - want) < 1e-6 * max(1, abs(want)))
                  for g in got)
        res.append(("named", okk))
    for n in names:
        if n.endswith("num_active_channels"):
            v = ins[n]
            if "channels=" in text:
                res.append(("chan", f"channels={v}" in text))
    return res


def score(struct_name, rows):
    tot = ok = 0
    per = defaultdict(lambda: [0, 0])
    for raw, text in rows:
        for kind, passed in checks(isa.decode(raw, struct_name), text):
            per[kind][0] += passed
            per[kind][1] += 1
            ok += passed
            tot += 1
    return (ok / tot if tot else None), {k: f"{a}/{b}" for k, (a, b) in per.items()}


def main():
    dispatch = json.load(open(os.path.join(REF, "opcode_dispatch_v3.json")))
    frames = []
    for p in sorted(glob.glob(os.path.join(ART, "corpus", "*.parquet"))):
        df = pd.read_parquet(p, columns=["tag", "engine", "raw", "op", "p_opcode", "p_operands",
                                         "n_runtime"])
        frames.append(df[df.n_runtime > 0])
    df = pd.concat(frames)
    all_structs = [f"{pkg}.{s}" for pkg in isa.PKG_ORDER for s in isa._LAYOUT.get(pkg, {})
                   if "header_opcode" in isa._LAYOUT[pkg][s]]
    table = {}
    for op, g in sorted(df.groupby("op"), key=lambda kv: kv[0]):
        name = isa.OPCODES.get(int(op), "0x%02x" % op)
        g = g.sample(min(PER_OPCODE, len(g)), random_state=0)
        rows = [(bytes(r.raw), r.p_operands or "") for r in g.itertuples()]
        entry = {"opcode": int(op), "engines": sorted(g.engine.unique().tolist()),
                 "profiler_names": sorted(g.p_opcode.dropna().unique().tolist())[:6],
                 "pseudo": name.startswith("PSEUDO")}
        st = isa.STRUCT_OF.get(int(op))
        if st:
            rate, detail = score(st, rows)
            entry.update(struct=st, source=isa.STRUCT_SOURCE.get(int(op)), pass_rate=rate,
                         checks=detail)
        else:
            best = None
            for cand in all_structs:
                rate, detail = score(cand, rows)
                if rate is not None and (best is None or rate > best[1]):
                    best = (cand, rate, detail)
            if best:
                entry.update(struct=best[0], source="UNRESOLVED: best guess from corpus",
                             pass_rate=best[1], checks=best[2])
            else:
                entry.update(struct=None, source="unknown", pass_rate=None, checks={})
        table[name] = entry
        pr = "  n/a" if entry["pass_rate"] is None else f"{entry['pass_rate']:5.3f}"
        print(f"{name:32s} {str(entry['struct']):55s} {pr} {entry['checks']} "
              f"[{entry['source'][:8]}]{' PSEUDO' if entry['pseudo'] else ''}")
    for k, v in dispatch.items():                # opcodes not in the corpus: dispatch only
        name = isa.OPCODES.get(int(k, 16), k)
        if name not in table:
            table[name] = {"opcode": int(k, 16), "struct": v["structs"][0] if v["structs"] else None,
                           "source": "neuron-dbg dispatch", "decoder": v["decoder"],
                           "pass_rate": None, "checks": {}, "note": "not in corpus"}
    json.dump(table, open(os.path.join(REF, "opcode_struct_v3.json"), "w"), indent=1,
              sort_keys=True)


if __name__ == "__main__":
    main()

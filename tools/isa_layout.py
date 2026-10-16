#!/usr/bin/env python3
"""Extract the wire layout of every Neuron ISA instruction field from neuron-dbg.

neuron-dbg links the vendor's C ISA accessors, compiled without optimisation: one function
`get_<pkg>_<STRUCT>_<field path>(const STRUCT *p [, int index])` per field, each a single load
from the packed instruction struct, optionally followed by a shift and a mask (bitfields), or an
indexed load (arrays). Reading those loads gives the byte offset, width, signedness, bit
position and array stride of every field, for every instruction format and ISA version.

Struct names come from the DWARF of the same binary (see dwarf_isa.py); a getter name is split
into <STRUCT> and <field path> by taking the longest known struct-name prefix.

Usage:
    isa_layout.py [neuron-dbg] [isa_dwarf.json] [out.json]
"""

import json
import re
import subprocess
import sys
from collections import defaultdict

BIN = "/opt/aws/neuron/bin/neuron-dbg"
PKGS = ("common", "v2", "v3")      # v3 = NeuronCore-v3 (Trainium2); v2 is what v3 builds on

# The struct pointer is in %rax (scalar fields, word arrays indexed by %rdx) or in %rdx (byte
# arrays indexed by %rax).
LOAD = re.compile(r"\s(mov[sz]?[bwlq]?[lwq]?|mov|movss|movsd)\s+(-?0x[0-9a-f]+|)\((%rax|%rdx)(?:,%r(?:dx|ax),(\d))?\),%(e?[a-z]+|r[a-z0-9]+|xmm\d)")
ADD_IDX = re.compile(r"\sadd\s+\$(0x[0-9a-f]+),%r(?:dx|ax)")
SHR = re.compile(r"\s(shr|sar)\s+\$(0x[0-9a-f]+),%(al|ax|eax|rax)")
AND = re.compile(r"\sand\s+\$(0x[0-9a-f]+),%(eax|al|ax|rax)")
LEA = re.compile(r"\slea\s+(-?0x[0-9a-f]+|)\(%rax\),%rax")

WIDTH = {"movzbl": (1, False), "movsbl": (1, True), "movzwl": (2, False), "movswl": (2, True),
         "movzbw": (1, False), "movsbw": (1, True), "movslq": (4, True), "movsbq": (1, True),
         "movswq": (2, True), "movzbq": (1, False), "movzwq": (2, False)}


def _width(mn, dst_reg):
    if mn == "movss":
        return (4, "float")
    if mn == "movsd":
        return (8, "float")
    if mn in WIDTH:
        return WIDTH[mn]
    if mn == "mov":
        if dst_reg.startswith("r") and not dst_reg.endswith("d"):
            return (8, False)
        if dst_reg.startswith("e"):
            return (4, False)
        if dst_reg in ("ax",):
            return (2, False)
        if dst_reg in ("al",):
            return (1, False)
    return (None, None)


def getter_symbols(binary):
    out = subprocess.run(["nm", "-n", binary], capture_output=True, text=True, check=True).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3:
            syms.append((int(parts[0], 16), parts[1], parts[2]))
    syms.sort()
    getters = []
    for i, (addr, kind, name) in enumerate(syms):
        m = re.match(r"get_(%s)_(NEURON_ISA_TPB_\w+)$" % "|".join(PKGS), name)
        if m and kind in "Tt":
            end = syms[i + 1][0] if i + 1 < len(syms) else addr + 128
            getters.append((addr, min(end, addr + 256), m.group(1), m.group(2)))
    return getters


def disassemble(binary, lo, hi):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={lo:#x}",
                          f"--stop-address={hi:#x}", binary], capture_output=True, text=True,
                         check=True).stdout
    lines = defaultdict(list)
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(.*)", line)
        if m:
            lines[int(m.group(1), 16)].append(m.group(2))
    return lines


def parse_getter(code):
    """code: list of instruction strings of one getter. Returns a field record or None."""
    text = "\n".join(" " + c for c in code)
    idx_base = 0
    m = ADD_IDX.search(text)
    if m:
        idx_base = int(m.group(1), 16)
    loads = [x for x in LOAD.finditer(text) if x.group(2) != "" or x.group(4)]
    # The accessor's own load from the struct is the one with a displacement off %rax that
    # follows "mov -0x8(%rbp),%rax"; -O0 code reloads %rax from the stack first, so skip
    # loads with negative displacement (stack slots use %rbp, not %rax, so they do not match).
    for x in LOAD.finditer(text):
        mn, disp, _, scale, dst = x.groups()
        off = int(disp, 16) if disp else 0
        w, signed = _width(mn, dst)
        if w is None:
            continue
        rec = {"offset": off, "width": w, "signed": signed is True}
        if signed == "float":
            rec["float"] = True
        if scale:
            rec["array_stride"] = int(scale)
            rec["offset"] = off + idx_base * int(scale)
        tail = text[x.end():]
        s = SHR.search(tail.split("\n ret")[0]) if "ret" in tail else SHR.search(tail)
        a = AND.search(tail.split(" ret")[0])
        if s:
            rec["shift"] = int(s.group(2), 16)
        if a:
            rec["mask"] = int(a.group(1), 16)
        return rec
    m = LEA.search(text)
    if m:
        return {"offset": int(m.group(1), 16) if m.group(1) else 0, "pointer": True}
    return None


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else BIN
    dwarf = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "isa_dwarf.json"))
    out_path = sys.argv[3] if len(sys.argv) > 3 else "isa_layout.json"
    struct_names = {k.split(".", 1)[1] for k in dwarf["structs"]}
    getters = getter_symbols(binary)
    lo = min(g[0] for g in getters)
    hi = max(g[1] for g in getters)
    code = disassemble(binary, lo, hi)
    addrs = sorted(code)
    layout = defaultdict(lambda: defaultdict(dict))
    unparsed = []
    for start, end, pkg, rest in getters:
        # STRUCT names are upper case, field paths lower case: split at the first lower-case
        # segment. (Some structs never appear in the Go DWARF, so the DWARF list cannot be
        # the only source of names.)
        m = re.match(r"(NEURON_ISA_TPB_[A-Z0-9_]+?)(?:_([a-z].*))?$", rest)
        st, path = m.group(1), m.group(2) or ""
        ins = [i for a in addrs if start <= a < end for i in code[a]]
        rec = parse_getter(ins)
        if rec is None:
            unparsed.append(rest)
            continue
        layout[pkg][st][path or "<self>"] = rec
    json.dump(layout, open(out_path, "w"), indent=1, sort_keys=True)
    n = sum(len(f) for p in layout.values() for f in p.values())
    print(f"{n} fields in {sum(len(p) for p in layout.values())} structs -> {out_path}; "
          f"{len(unparsed)} getters not parsed")
    if unparsed:
        print("  e.g.", unparsed[:8])


if __name__ == "__main__":
    main()

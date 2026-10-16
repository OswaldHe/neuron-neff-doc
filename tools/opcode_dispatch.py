#!/usr/bin/env python3
"""Recover neuron-dbg's opcode -> instruction-struct dispatch for NeuronCore-v3 (cayman).

neuron-dbg's disassembler (Go package tools/pkg/disassembler) chooses a decoder per opcode in
`caymanDisassembleInstruction`, which handles the v3-only opcodes and falls back to the shared
decoder `sundaDisassembleInstruction`. Both are Go `switch` statements compiled to compare chains:

    cmp $OPCODE,%rdx ; je CASE        (or: cmp $OPCODE,%rdx ; jne NEXT ; CASE...)

Each CASE calls a decoder function, which in turn calls
`tools/internal/isa/<pkg>.(*NEURON_ISA_TPB_<STRUCT>).ReadFromByteArray`. This script follows that
chain statically and writes artifacts/reference/opcode_dispatch_v3.json:

    {opcode: {"decoder": <function>, "structs": [<pkg>.<STRUCT>, ...]}}

Usage:
    opcode_dispatch.py [neuron-dbg] [out.json]
"""

import json
import re
import subprocess
import sys

BIN = "/opt/aws/neuron/bin/neuron-dbg"
ENTRY = ["tools/pkg/disassembler.caymanDisassembleInstruction",
         "tools/pkg/disassembler.sundaDisassembleInstruction"]

INS = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
CALL = re.compile(r"call\s+[0-9a-f]+ <([^>]+)>")
JMP = re.compile(r"^(j\w+)\s+([0-9a-f]+)")
CMP = re.compile(r"^\$0x([0-9a-f]+),%(r[a-z0-9]+|e[a-z]+|[a-z]+)$")
READ = re.compile(r"tools/internal/isa/(\w+)\.\(\*(NEURON_ISA_TPB_\w+)\)\.ReadFromByteArray$")


def symbols(binary):
    out = subprocess.run(["nm", "-n", binary], capture_output=True, text=True, check=True).stdout
    syms = []
    for l in out.splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and p[1] in "tT":
            syms.append((int(p[0], 16), p[2]))
    syms.sort()
    span = {}
    for i, (a, n) in enumerate(syms):
        span[n] = (a, syms[i + 1][0] if i + 1 < len(syms) else a + 4096)
    return span


def disasm(binary, lo, hi):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={lo:#x}",
                          f"--stop-address={hi:#x}", binary], capture_output=True, text=True,
                         check=True).stdout
    ins = []
    for l in out.splitlines():
        m = INS.match(l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return ins


def first_calls(ins, start, limit=60):
    """Calls reached from `start` by straight-line code, following unconditional jumps."""
    idx = {a: i for i, (a, _, _) in enumerate(ins)}
    calls, i, steps, seen = [], idx.get(start), 0, set()
    while i is not None and i < len(ins) and steps < limit and i not in seen:
        seen.add(i)
        a, mn, ops = ins[i]
        if mn == "call":
            m = CALL.search(mn + " " + ops)
            if m and not m.group(1).startswith("runtime."):
                calls.append(m.group(1))
                if "disassembler." in m.group(1) or "ReadFromByteArray" in m.group(1):
                    break
        elif mn == "jmp":
            m = JMP.match(mn + " " + ops)
            if m:
                i = idx.get(int(m.group(2), 16))
                steps += 1
                continue
            break
        elif mn == "ret":
            break
        i += 1
        steps += 1
    return calls


def _read_table(binary, addr, count):
    """`count` little-endian 8-byte code addresses at virtual address `addr`."""
    from elftools.elf.elffile import ELFFile
    with open(binary, "rb") as f:
        elf = ELFFile(f)
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD" and seg["p_vaddr"] <= addr < seg["p_vaddr"] + seg["p_filesz"]:
                f.seek(seg["p_offset"] + addr - seg["p_vaddr"])
                raw = f.read(8 * count)
                return [int.from_bytes(raw[i:i + 8], "little") for i in range(0, len(raw), 8)]
    return []


def jump_table_cases(binary, ins):
    """Go compiles dense switches to `lea -BASE(%rdx),%r9; cmp $MAX,%r9; ja DEFAULT;
    lea TABLE(%rip),%r10; jmp *(%r10,%r9,8)`. Returns opcode -> case address."""
    out = {}
    for k in range(len(ins)):
        a, mn, ops = ins[k]
        if mn != "jmp" or not ops.startswith("*("):
            continue
        base, top, table = 0, None, None
        for j in range(max(0, k - 8), k):
            _, m2, o2 = ins[j]
            mb = re.match(r"^-0x([0-9a-f]+)\(%rdx\),%r", o2)
            if m2 == "lea" and mb:
                base = int(mb.group(1), 16)
            mt = re.search(r"#\s*([0-9a-f]+) <[^>]*jump", o2)
            if m2 == "lea" and mt:
                table = int(mt.group(1), 16)
            mc = re.match(r"^\$0x([0-9a-f]+),%r", o2)
            if m2 == "cmp" and mc:
                top = int(mc.group(1), 16)
        if table is None or top is None:
            continue
        default = None
        for j in range(k - 1, max(0, k - 8), -1):
            if ins[j][1] == "ja":
                default = int(ins[j][2].split()[0], 16)
                break
        for i, target in enumerate(_read_table(binary, table, top + 1)):
            if target != default:
                out[i + base] = target
    return out


def cases(ins):
    """opcode -> case start address, from compare chains on the opcode register."""
    out = {}
    for k in range(len(ins) - 1):
        a, mn, ops = ins[k]
        if mn != "cmp":
            continue
        m = CMP.match(ops)
        if not m:
            continue
        val = int(m.group(1), 16)
        na, nmn, nops = ins[k + 1]
        j = JMP.match(nmn + " " + nops)
        if not j:
            continue
        if nmn == "je":
            out.setdefault(val, int(j.group(2), 16))
        elif nmn == "jne" and k + 2 < len(ins):
            out.setdefault(val, ins[k + 2][0])
    return out


def structs_read(binary, span, fn, depth=0, memo=None):
    memo = {} if memo is None else memo
    if fn in memo or depth > 3 or fn not in span:
        return memo.get(fn, [])
    memo[fn] = []
    lo, hi = span[fn]
    found = []
    for _, mn, ops in disasm(binary, lo, hi):
        if mn != "call":
            continue
        m = CALL.search("call " + ops)
        if not m:
            continue
        callee = m.group(1)
        r = READ.match(callee)
        if r:
            found.append(f"{r.group(1)}.{r.group(2)}")
        elif callee.startswith("tools/pkg/disassembler.") and callee != fn:
            found += structs_read(binary, span, callee, depth + 1, memo)
    # keep order, drop duplicates
    seen, uniq = set(), []
    for s in found:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    memo[fn] = uniq
    return uniq


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else BIN
    out_path = sys.argv[2] if len(sys.argv) > 2 else "opcode_dispatch_v3.json"
    span = symbols(binary)
    table = {}
    for entry in ENTRY:
        lo, hi = span[entry]
        ins = disasm(binary, lo, hi)
        found = cases(ins)
        found.update(jump_table_cases(binary, ins))
        for op, start in sorted(found.items()):
            if op > 0xff or op in table:
                continue                    # cayman entries win over the shared fallback
            calls = first_calls(ins, start)
            dec = next((c for c in calls if "disassembler." in c or "ReadFromByteArray" in c), None)
            if dec is None:
                continue
            r = READ.match(dec)
            st = [f"{r.group(1)}.{r.group(2)}"] if r else structs_read(binary, span, dec)
            table[op] = {"decoder": dec.split(".", 1)[-1] if not r else "ReadFromByteArray",
                         "structs": st, "from": entry.split(".")[-1]}
    json.dump({f"0x{k:02x}": v for k, v in sorted(table.items())}, open(out_path, "w"), indent=1)
    print(f"{len(table)} opcodes -> {out_path}")


if __name__ == "__main__":
    main()

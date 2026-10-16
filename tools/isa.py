#!/usr/bin/env python3
"""Decode and encode Trainium2 (NeuronCore-v3) instruction slots by vendor field name.

Layouts come from artifacts/reference/isa_layout.json (byte offset, width, bits, array stride of
every field, read out of the vendor's C accessors in neuron-dbg by isa_layout.py). Names, enum
values and array lengths come from artifacts/reference/isa_dwarf.json (dwarf_isa.py). Which
struct an opcode uses comes from the vendor disassembler's dispatch
(artifacts/reference/opcode_dispatch_v3.json, opcode_dispatch.py), corrected and extended by
artifacts/reference/isa_layout_supplement.json for the few opcodes neuron-dbg does not decode.
verify_isa.py checks the result against the profiler's disassembly of the whole corpus.

Library use:
    import isa
    ins = isa.decode(slot64)                 # -> Instruction (ordered fields, struct, opcode)
    ins["events_wait_idx"]                   # field values by vendor path
    ins.set("events_semaphore_value", 7)     # change a field
    new_slot = ins.encode()                  # 64 bytes, all other bits untouched

Command line:
    isa.py <Engine0.bin> [first [count]]     # print the decoded stream
"""

import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "artifacts", "reference")
SLOT = 64
PKG_ORDER = ("v3", "common", "v2")          # Trainium2 first; v3 builds on common and v2

_LAYOUT = json.load(open(os.path.join(REF, "isa_layout.json")))
_DWARF = json.load(open(os.path.join(REF, "isa_dwarf.json")))
# Which struct each opcode uses: the vendor disassembler's own dispatch (opcode_dispatch.py),
# then the documented corrections and additions in isa_layout_supplement.json.
_DISPATCH = json.load(open(os.path.join(REF, "opcode_dispatch_v3.json")))
_SUPP = json.load(open(os.path.join(REF, "isa_layout_supplement.json")))
_LAYOUT["manual"] = _SUPP["manual_layouts"]
EXTENSION_STRUCT = _SUPP["extension_struct"]  # previous opcode name -> struct of its 0xda slot

# ---------------------------------------------------------------- names and enums

def _short(const_name, type_short):
    n = const_name.split(".", 1)[1]
    pre = "NEURON_ISA_TPB_" + type_short + "_"
    if n.startswith(pre):
        return n[len(pre):]
    return n.replace("NEURON_ISA_TPB_", "")


ENUMS = {}                                  # full type name -> {value: short name}
for tname, consts in _DWARF["constants"].items():
    tshort = tname.split(".")[-1].replace("NEURON_ISA_TPB_", "")
    for cname, val in consts.items():
        pkg = cname.split(".")[0].split("/")[-1]
        # keep the v3 name when the same type has per-version constants
        d = ENUMS.setdefault(tname, {})
        if val not in d or pkg == "v3":
            d[val] = _short(cname, tshort)

OPCODES = {}                                # value -> name, NeuronCore-v3 only
for cname, val in _DWARF["constants"]["tools/internal/isa/common.NEURON_ISA_TPB_OPCODE"].items():
    if cname.startswith("tools/internal/isa/v3."):
        OPCODES[val] = cname.split("NEURON_ISA_TPB_OPCODE_", 1)[1]
OPCODE_VALUE = {v: k for k, v in OPCODES.items()}

STRUCT_OF = {}                              # opcode value -> "pkg.STRUCT"
for _k, _v in _DISPATCH.items():
    if _v["structs"]:
        STRUCT_OF[int(_k, 16)] = _v["structs"][0]
STRUCT_SOURCE = {k: "neuron-dbg dispatch" for k in STRUCT_OF}
for _k, _v in _SUPP["opcode_struct"].items():
    STRUCT_OF[int(_k, 16)] = _v["struct"]
    STRUCT_SOURCE[int(_k, 16)] = "supplement: " + _v["why"]

# ---------------------------------------------------------------- struct layouts

def find_struct(name):
    """(pkg, layout dict) for a struct name. A name may carry its package ("v2.NAME"); a bare
    name is looked up NeuronCore-v3 first."""
    if "." in name:
        pkg, bare = name.split(".", 1)
        return pkg, _LAYOUT[pkg][bare]
    for pkg in PKG_ORDER:
        if name in _LAYOUT.get(pkg, {}):
            return pkg, _LAYOUT[pkg][name]
    raise KeyError(name)


def _dwarf_struct(pkg, name):
    for p in (pkg,) + PKG_ORDER:
        s = _DWARF["structs"].get(f"tools/internal/isa/{p}.{name}")
        if s:
            return s
    return None


def _member_type(pkg, struct_name, path):
    """Walk DWARF members along an accessor path. Returns (type name, array length)."""
    s = _dwarf_struct(pkg, struct_name)
    rest = path
    while s is not None and rest:
        best = None
        for m in s["members"]:
            n = m["name"]
            if n and (rest == n or rest.startswith(n + "_")):
                if best is None or len(n) > len(best["name"]):
                    best = m
        if best is None:
            return None, None
        rest = rest[len(best["name"]) + 1:]
        t = best["type"] or ""
        if not rest:
            if t.startswith("["):
                return t[t.index("]") + 1:], int(t[1:t.index("]")])
            return t, None
        s = _DWARF["structs"].get(t)
    return None, None


_FIELDS_CACHE = {}


def fields(struct_name):
    """Leaf fields of a struct in wire order: list of dicts with name, offset, width, signed,
    shift, mask, count, stride, type."""
    if struct_name in _FIELDS_CACHE:
        return _FIELDS_CACHE[struct_name]
    pkg, lay = find_struct(struct_name)
    bare = struct_name.split(".", 1)[-1]
    names = [n for n in lay if n != "<self>"]
    leaves = []
    for n in names:
        r = lay[n]
        if r.get("pointer"):
            continue
        if any(o != n and o.startswith(n + "_") for o in names):
            continue                        # an aggregate; its leaves are listed instead
        if pkg == "manual":
            t, cnt = r.get("type"), r.get("count")
        else:
            t, cnt = _member_type(pkg, bare, n)
        # Signedness comes from the Go field type in DWARF (int vs uint): the -O0 C accessors load
        # many signed fields with zero-extending moves (an 0xffff step is printed -1 by the
        # profiler, a channel_multiplier of 0xffffffff is printed -1).
        signed = r.get("signed", False) or (t or "").lstrip("[]0123456789") in (
            "int", "int8", "int16", "int32", "int64")
        f = dict(name=n, offset=r["offset"], width=r["width"], signed=signed,
                 float=r.get("float", False), shift=r.get("shift", 0), mask=r.get("mask"),
                 stride=r.get("array_stride"), count=None, type=t)
        if f["stride"]:
            f["count"] = cnt
        leaves.append(f)
    # Arrays whose length DWARF does not give: run up to the next field.
    leaves.sort(key=lambda f: (f["offset"], f["shift"]))
    for i, f in enumerate(leaves):
        if f["stride"] and not f["count"]:
            nxt = next((g["offset"] for g in leaves[i + 1:] if g["offset"] > f["offset"]), SLOT)
            f["count"] = max(1, (nxt - f["offset"]) // f["stride"])
    _FIELDS_CACHE[struct_name] = leaves
    return leaves


def _read(raw, f, i=0):
    off = f["offset"] + i * (f["stride"] or 0)
    b = raw[off:off + f["width"]]
    if len(b) < f["width"]:
        return None
    if f["float"]:
        return struct.unpack("<f" if f["width"] == 4 else "<d", b)[0]
    v = int.from_bytes(b, "little", signed=False)
    if f["shift"]:
        v >>= f["shift"]
    if f["mask"] is not None:
        v &= f["mask"]
    if f["signed"] and not f["shift"] and f["mask"] is None:
        bits = 8 * f["width"]
        if v >> (bits - 1):
            v -= 1 << bits
    return v


def _write(buf, f, value, i=0):
    off = f["offset"] + i * (f["stride"] or 0)
    w = f["width"]
    if f["float"]:
        buf[off:off + w] = struct.pack("<f" if w == 4 else "<d", float(value))
        return
    cur = int.from_bytes(buf[off:off + w], "little")
    value = int(value)
    if f["shift"] or f["mask"] is not None:
        mask = (f["mask"] if f["mask"] is not None else (1 << (8 * w - f["shift"])) - 1)
        cur &= ~(mask << f["shift"]) & ((1 << (8 * w)) - 1)
        cur |= (value & mask) << f["shift"]
    else:
        cur = value & ((1 << (8 * w)) - 1)
    buf[off:off + w] = cur.to_bytes(w, "little")


# ---------------------------------------------------------------- instructions

class Instruction:
    def __init__(self, raw, struct_name):
        self.raw = bytearray(raw)
        self.struct = struct_name
        self.opcode = raw[0]
        self.name = OPCODES.get(raw[0], "0x%02x" % raw[0])

    def fields(self):
        return fields(self.struct) if self.struct else []

    def _field(self, name):
        for f in self.fields():
            if f["name"] == name:
                return f
        raise KeyError(f"{self.struct} has no field {name}")

    def __getitem__(self, name):
        f = self._field(name)
        if f["stride"]:
            return [_read(self.raw, f, i) for i in range(f["count"])]
        return _read(self.raw, f)

    def set(self, name, value):
        f = self._field(name)
        if isinstance(value, str):
            value = enum_value(f["type"], value)
        if f["stride"]:
            for i, v in enumerate(value):
                _write(self.raw, f, v, i)
        else:
            _write(self.raw, f, value)
        return self

    def encode(self):
        return bytes(self.raw)

    def as_dict(self, symbolic=True):
        out = {}
        for f in self.fields():
            v = self[f["name"]]
            if symbolic and f["type"] in ENUMS and not f["stride"]:
                v = ENUMS[f["type"]].get(v, v)
            out[f["name"]] = v
        return out

    def __repr__(self):
        return f"<{self.name} {self.struct}>"


def enum_value(type_name, symbol):
    for v, n in ENUMS.get(type_name, {}).items():
        if n == symbol:
            return v
    raise KeyError(f"{symbol} is not a value of {type_name}")


def decode(raw, struct_name=None, prev=None):
    """Decode one 64-byte slot. The struct is looked up from the opcode unless given. An
    extension slot (PSEUDO_EXTENSION) is decoded with the struct that belongs to the
    instruction before it, so pass that instruction's opcode name as `prev`."""
    if struct_name is None:
        if raw[0] == OPCODE_VALUE.get("PSEUDO_EXTENSION") and prev in EXTENSION_STRUCT:
            struct_name = EXTENSION_STRUCT[prev]
        else:
            struct_name = STRUCT_OF.get(raw[0])
    return Instruction(raw, struct_name)


def decode_stream(data):
    """Decode a whole engine stream (bytes). Yields (index, Instruction)."""
    prev = None
    for i in range(0, len(data), SLOT):
        ins = decode(data[i:i + SLOT], prev=prev)
        if ins.name != "PSEUDO_EXTENSION":
            prev = ins.name
        yield i // SLOT, ins


def iter_stream(path):
    yield from decode_stream(open(path, "rb").read())


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    first = int(argv[2]) if len(argv) > 2 else 0
    count = int(argv[3]) if len(argv) > 3 else 1 << 30
    for idx, ins in iter_stream(argv[1]):
        if idx < first:
            continue
        if idx >= first + count:
            break
        d = ins.as_dict()
        body = " ".join(f"{k}={v}" for k, v in d.items()
                        if not k.startswith("header_") and v not in (0, None, [0, 0, 0], [0, 0, 0, 0]))
        print(f"{idx:6d} {ins.name:28s} [{ins.struct}] {body}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

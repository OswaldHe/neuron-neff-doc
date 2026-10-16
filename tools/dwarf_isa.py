#!/usr/bin/env python3
"""Extract the Neuron ISA struct layouts that neuron-dbg carries in its DWARF debug info.

/opt/aws/neuron/bin/neuron-dbg is a Go program built with full DWARF. Its packages
tools/internal/isa/{common,v2,v3,v4} define one Go struct per ISA instruction format (generated
from the C ISA headers), plus typed constants for opcodes and enums. DWARF records every struct's
member names, byte offsets and types, so the layouts below come from the vendor's own definitions
rather than from guessing. Package v3 is NeuronCore-v3 (Trainium2, "cayman").

Usage (workspace/.venv has pyelftools):
    dwarf_isa.py [binary] [out.json]
"""

import json
import sys
from collections import defaultdict

from elftools.elf.elffile import ELFFile

BIN = "/opt/aws/neuron/bin/neuron-dbg"
PKG_PREFIX = "tools/internal/isa"


def _name(die):
    a = die.attributes.get("DW_AT_name")
    return a.value.decode() if a else None


def _type_die(die):
    a = die.attributes.get("DW_AT_type")
    return die.get_DIE_from_attribute("DW_AT_type") if a else None


def _type_name(die, depth=0):
    if die is None or depth > 8:
        return None
    n = _name(die)
    if n:
        return n
    t = _type_die(die)
    return _type_name(t, depth + 1)


def _size(die, depth=0):
    if die is None or depth > 8:
        return None
    a = die.attributes.get("DW_AT_byte_size")
    if a:
        return a.value
    return _size(_type_die(die), depth + 1)


def extract(path):
    structs, consts, typedefs = {}, defaultdict(dict), {}
    with open(path, "rb") as f:
        elf = ELFFile(f)
        dw = elf.get_dwarf_info()
        for cu in dw.iter_CUs():
            for die in cu.iter_DIEs():
                n = _name(die)
                if not n or not n.startswith(PKG_PREFIX):
                    continue
                if die.tag == "DW_TAG_structure_type":
                    members = []
                    for ch in die.iter_children():
                        if ch.tag != "DW_TAG_member":
                            continue
                        off = ch.attributes.get("DW_AT_data_member_location")
                        t = _type_die(ch)
                        members.append({"name": _name(ch), "offset": off.value if off else None,
                                        "type": _type_name(t), "size": _size(t)})
                    sz = die.attributes.get("DW_AT_byte_size")
                    structs[n] = {"size": sz.value if sz else None, "members": members}
                elif die.tag == "DW_TAG_constant":
                    v = die.attributes.get("DW_AT_const_value")
                    t = _type_name(_type_die(die))
                    if v is not None:
                        consts[t or "?"][n] = v.value
                elif die.tag == "DW_TAG_typedef":
                    typedefs[n] = _type_name(_type_die(die))
    return {"structs": structs, "constants": consts, "typedefs": typedefs}


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else BIN
    out = sys.argv[2] if len(sys.argv) > 2 else "isa_dwarf.json"
    data = extract(src)
    with open(out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print(f"{len(data['structs'])} structs, {sum(len(v) for v in data['constants'].values())} "
          f"constants, {len(data['typedefs'])} typedefs -> {out}")

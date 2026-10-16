#!/usr/bin/env python3
"""Read the NeuronCore ISA's hardware constants out of neuron-dbg.

The vendor ISA headers define limits and marker values as `static const` objects
(NEURON_ISA_TPB_STATE_BUF_NUM_PARTITIONS, NEURON_ISA_TPB_ADDR4_MARKER_ADDR_REG, ...). neuron-dbg
keeps them as read-only data symbols, so their values can be read straight from the ELF.

Usage (workspace/.venv):
    hw_constants.py [neuron-dbg] [out.json]
"""

import json
import subprocess
import sys

from elftools.elf.elffile import ELFFile

BIN = "/opt/aws/neuron/bin/neuron-dbg"


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else BIN
    out_path = sys.argv[2] if len(sys.argv) > 2 else "hw_constants_v3.json"
    nm = subprocess.run(["nm", "-S", binary], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in nm.splitlines():
        p = line.split()
        if len(p) == 4 and p[2] in "rR" and p[3].startswith("NEURON_ISA_TPB_") and int(p[1], 16) <= 8:
            syms.setdefault(p[3], (int(p[0], 16), int(p[1], 16)))
    vals = {}
    with open(binary, "rb") as f:
        elf = ELFFile(f)
        segs = [s for s in elf.iter_segments() if s["p_type"] == "PT_LOAD"]
        for name, (addr, size) in sorted(syms.items()):
            for s in segs:
                if s["p_vaddr"] <= addr < s["p_vaddr"] + s["p_filesz"]:
                    f.seek(s["p_offset"] + addr - s["p_vaddr"])
                    vals[name] = int.from_bytes(f.read(size), "little")
                    break
    json.dump(vals, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"{len(vals)} constants -> {out_path}")


if __name__ == "__main__":
    main()

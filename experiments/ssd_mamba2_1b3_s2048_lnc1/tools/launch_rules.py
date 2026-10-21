#!/usr/bin/env python3
"""The fairness constraint: the NEFF must keep the launch shape it is being compared against.

The baseline here is the nkilib `gdn_tkg` decode kernel as it ships for this case: **one logical
NeuronCore at LNC=2 (two physical cores, `sg00` and `sg01`), one rank, no collectives**. LNC=2 is
allowed because the kernel itself shards its flattened batch*head axis across the two cores --
that is how it is meant to run. Spreading the work across **ranks** is not: a 4-rank NEFF would
lower `total_exec_time` by using four times the hardware, which is not a better kernel. This
check allows the former and rejects the latter.

What it reads out of the NEFF (chapters 2-4 of neff_doc):

    kelf-0.json       exactly `lnc` subgraphs, from `bench/manifest.json` (`lnc: 2` -> `sg00`
                      and `sg01`; `lnc: 1` -> `sg00` only). More than that is more cores.
    sgNN/def.json     `cc_streams.num_streams == 0`, no `replica_groups`, `#rank_world_size == 0`.
                      A multi-rank collective NEFF has `num_streams: 1`,
                      `replica_groups: [[[0, 1, 2, 3]]]` and `#rank_world_size: 1`.
    info.json         `num_tpb == 1` -- one logical NeuronCore, whatever the LNC degree.
    neff.json         the same input tensor names in the same order as the manifest, and one
                      output head per manifest output per subgraph (the runtime merges the
                      per-core heads into the output tensors the verifier reads).

Usage:
    python tools/launch_rules.py            # exit 0 if the NEFF keeps the baseline's shape
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import neffio  # noqa: E402


def problems(neff_path: str, manifest: dict) -> list[str]:
    """Empty when the NEFF keeps the baseline's launch shape and interface."""
    out = []
    lnc = int(manifest.get("lnc", 1))
    members = neffio.members(neff_path)

    kelf = json.loads(members["kelf-0.json"])
    names = [g["name"] for g in kelf.get("graphs", [])]
    expected = [f"sg{i:02d}" for i in range(lnc)]
    if names != expected:
        out.append(f"expected exactly {lnc} subgraph(s) {expected}, found {names} -- the number "
                   f"of subgraphs is the number of physical cores, and this loop is measured "
                   f"against an LNC={lnc} launch")

    for sg in names:
        d = json.loads(members[f"{sg}/def.json"])
        streams = (d.get("cc_streams") or {}).get("num_streams", 0)
        if streams:
            out.append(f"{sg}/def.json has cc_streams.num_streams={streams} -- collectives are "
                       f"not allowed; the baseline runs on one rank")
        if d.get("replica_groups"):
            out.append(f"{sg}/def.json has replica_groups={d['replica_groups']} -- the NEFF would "
                       f"run across several ranks; the baseline runs on one rank")
        if d.get("#rank_world_size", 0) != 0:
            out.append(f"{sg}/def.json has #rank_world_size={d['#rank_world_size']}, expected 0")

    info = json.loads(members["info.json"])
    if info.get("num_tpb") != 1 or info.get("num_tpb_requested") != 1:
        out.append(f"info.json asks for num_tpb={info.get('num_tpb')} "
                   f"(requested {info.get('num_tpb_requested')}), expected 1")

    sig = neffio.io_signature(neff_path)
    want = [t["name"] for t in manifest["inputs"]]
    if sig["inputs"] != want:
        out.append(f"inputs are {sig['inputs']}, the benchmark feeds {want}")
    want_heads = len(manifest["outputs"]) * max(len(names), 1)
    if sig["n_outputs"] != want_heads:
        out.append(f"the NEFF has {sig['n_outputs']} output heads, expected {want_heads} "
                   f"({len(manifest['outputs'])} outputs x {len(names)} subgraph(s))")
    return out


def main():
    import device as D

    m = D.manifest()
    bad = problems(D.NEFF, m)
    sig = neffio.io_signature(D.NEFF)
    print(f"launch-shape check on kernel.neff ({os.path.getsize(D.NEFF)} bytes)")
    print(f"  subgraphs {sig['subgraphs']}  num_tpb {sig['num_tpb']}  "
          f"inputs {len(sig['inputs'])}  output heads {sig['n_outputs']}")
    for p in bad:
        print(f"  [FAIL] {p}")
    print(f"LNC={m.get('lnc', 1)}, one rank, interface unchanged" if not bad
          else "FAILED -- the NEFF must keep the baseline's launch shape")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

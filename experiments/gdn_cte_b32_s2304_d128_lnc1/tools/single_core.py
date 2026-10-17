#!/usr/bin/env python3
"""The fairness constraint: the NEFF must stay a single-core, single-rank, same-interface NEFF.

The baseline this loop is measured against is the nkilib GDN kernel as it ships: **one logical
NeuronCore (LNC=1), one rank, no collectives**. Spreading the same work over two physical cores
(LNC=2) or four ranks would lower `total_exec_time` without making the kernel better, so it is
not a win this loop is allowed to claim. This check rejects it.

What it reads out of the NEFF (chapters 2-4 of neff_doc):

    kelf-0.json       one subgraph (`sg00`). An LNC=2 NEFF lists `sg00` **and** `sg01`.
    sgNN/def.json     `cc_streams.num_streams == 0`, no `replica_groups`, `#rank_world_size == 0`.
                      A 4-rank collective NEFF has `num_streams: 1`,
                      `replica_groups: [[[0, 1, 2, 3]]]` and `#rank_world_size: 1`.
    info.json         `num_tpb == 1` (one NeuronCore requested).
    neff.json         the same input tensor names in the same order, and the same number of
                      outputs, as `bench/manifest.json` -- the kernel must still have the
                      interface the verifier feeds and the golden describes.

Usage:
    python tools/single_core.py                 # exit 0 if the NEFF obeys all of it
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import neffio  # noqa: E402


def problems(neff_path: str, manifest: dict) -> list[str]:
    """Empty when the NEFF is single-core, single-rank and keeps its interface."""
    out = []
    members = neffio.members(neff_path)

    kelf = json.loads(members["kelf-0.json"])
    names = [g["name"] for g in kelf.get("graphs", [])]
    if names != ["sg00"]:
        out.append(f"expected exactly one subgraph ['sg00'], found {names} -- more than one "
                   f"subgraph means LNC=2 (two physical cores), which this loop does not allow")

    for sg in names:
        d = json.loads(members[f"{sg}/def.json"])
        streams = (d.get("cc_streams") or {}).get("num_streams", 0)
        if streams:
            out.append(f"{sg}/def.json has cc_streams.num_streams={streams} -- collectives are "
                       f"not allowed; the baseline runs on one rank")
        if d.get("replica_groups"):
            out.append(f"{sg}/def.json has replica_groups={d['replica_groups']} -- the NEFF would "
                       f"run across several ranks; the baseline runs on one rank")
        if d.get("#rank_world_size", 0) not in (0,):
            out.append(f"{sg}/def.json has #rank_world_size={d['#rank_world_size']}, expected 0")

    info = json.loads(members["info.json"])
    if info.get("num_tpb") != 1 or info.get("num_tpb_requested") != 1:
        out.append(f"info.json asks for num_tpb={info.get('num_tpb')} "
                   f"(requested {info.get('num_tpb_requested')}), expected 1")

    sig = neffio.io_signature(neff_path)
    want = [t["name"] for t in manifest["inputs"]]
    if sig["inputs"] != want:
        out.append(f"inputs are {sig['inputs']}, the benchmark feeds {want}")
    if sig["n_outputs"] != len(manifest["outputs"]):
        out.append(f"the NEFF has {sig['n_outputs']} outputs, the benchmark expects "
                   f"{len(manifest['outputs'])}")
    return out


def main():
    import device as D

    m = D.manifest()
    bad = problems(D.NEFF, m)
    sig = neffio.io_signature(D.NEFF)
    print(f"single-core check on kernel.neff ({os.path.getsize(D.NEFF)} bytes)")
    print(f"  subgraphs {sig['subgraphs']}  num_tpb {sig['num_tpb']}  "
          f"inputs {sig['inputs']}  outputs {sig['n_outputs']}")
    for p in bad:
        print(f"  [FAIL] {p}")
    print("LNC=1, one rank, interface unchanged" if not bad
          else "FAILED -- the NEFF must stay LNC=1 and single-rank")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

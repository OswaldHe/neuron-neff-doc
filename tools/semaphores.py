#!/usr/bin/env python3
"""Semaphore accounting for one subgraph: who increments which semaphore, by how much, and who
waits for what. Used to check the synchronisation model and to re-balance waits after an edit.

Sources of increments, per semaphore:
  * any instruction whose events_update_mode is SEM_INC_* (+1) or SEM_ADD_IMM_* (+value);
  * PSEUDO_DMATRIGGER: the triggered DMA transfer (sgNN/<Engine>0.json "dma" entry with the
    same queue and id) increments that transfer's "semaphore" when it completes;
  * PSEUDO_DMA_DIRECT2D (DGE): increments its `semaphore` field by `sem_increment`.

Usage (workspace/.venv):
    semaphores.py <unpacked_neff_dir> [sgNN]       print the per-semaphore table
"""

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isa  # noqa: E402

ENGINES = ["Activation", "DVE", "PE", "Pool", "SP"]


def load_streams(sg_dir):
    out = {}
    for e in ENGINES:
        data = open(os.path.join(sg_dir, f"{e}0.bin"), "rb").read()
        out[e] = list(isa.decode_stream(data))
    return out


def dma_blocks(sg_dir):
    """(engine, queue, id) -> dma entry, from every engine description."""
    blocks = {}
    for e in ENGINES:
        j = json.load(open(os.path.join(sg_dir, f"{e}0.json")))
        for b in j.get("dma", []):
            # with the queue-set-instances feature an entry names a queue instance instead
            blocks[(e, b.get("queue") or b.get("instance_name"), b["id"])] = b
    return blocks


def queue_widths(sg_dir):
    d = json.load(open(os.path.join(sg_dir, "def.json")))
    out = {}
    for q, v in d.get("dma_queue", {}).items():
        out[q] = v["num_queues"]
        for inst in v.get("queue_instances", []):
            out[inst] = v["num_queues"]
    return out


def account(sg_dir):
    """Return {sem: {"inc": {source: amount}, "max_wait": v, "waiters": n}}."""
    streams = load_streams(sg_dir)
    blocks = dma_blocks(sg_dir)
    widths = queue_widths(sg_dir)
    sem = defaultdict(lambda: {"inc": defaultdict(int), "max_wait": 0, "waiters": 0})
    for eng, ins_list in streams.items():
        for idx, ins in ins_list:
            if not ins.struct or ins.name == "PSEUDO_EXTENSION":
                continue
            names = {f["name"] for f in ins.fields()}
            if "events_wait_mode" in names:
                wm = isa.ENUMS["tools/internal/isa/common.NEURON_ISA_TPB_WAIT_MODE"].get(ins["events_wait_mode"])
                if wm and wm.startswith("WAIT_FOR_SEM_") and wm.endswith("_IMM"):
                    s = sem[ins["events_wait_idx"]]
                    s["max_wait"] = max(s["max_wait"], ins["events_semaphore_value"])
                    s["waiters"] += 1
                um = isa.ENUMS["tools/internal/isa/common.NEURON_ISA_TPB_UPDATE_MODE"].get(ins["events_update_mode"])
                if um in ("SEM_INC_COMPLETE", "SEM_INC_READ"):
                    sem[ins["events_update_idx"]]["inc"][f"{eng}:{ins.name}"] += 1
                elif um in ("SEM_ADD_IMM_COMPLETE", "SEM_ADD_IMM_READ"):
                    sem[ins["events_update_idx"]]["inc"][f"{eng}:{ins.name}"] += ins["events_semaphore_value"]
            if ins.name == "PSEUDO_DMATRIGGER":
                q = bytes(ins["dma_queue_name"]).split(b"\0")[0].decode()
                b = blocks.get((eng, q, ins["count_union_block_id"]))
                if b is not None:
                    sem[b["semaphore"]]["inc"][f"dma:{q}"] += widths.get(q, 1)
            elif ins.name == "PSEUDO_DMA_DIRECT2D":
                sem[ins["semaphore"]]["inc"][f"dge:{eng}"] += ins["sem_increment"]
    return sem


def main():
    d = sys.argv[1]
    sg = sys.argv[2] if len(sys.argv) > 2 else "sg00"
    sem = account(os.path.join(d, "payload", sg))
    print(f"{'sem':>4} {'max_wait':>9} {'total_inc':>9} {'waiters':>7}  sources")
    for k in sorted(sem):
        s = sem[k]
        tot = sum(s["inc"].values())
        flag = "" if s["max_wait"] <= tot else "  <-- wait exceeds increments"
        print(f"{k:4d} {s['max_wait']:9d} {tot:9d} {s['waiters']:7d}  {dict(s['inc'])}{flag}")


if __name__ == "__main__":
    main()

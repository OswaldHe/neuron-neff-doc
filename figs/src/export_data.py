"""Export the small tables the data figures are drawn from into figs/data/.

Inputs (not in the repo): the per-iteration NEFFs unpacked by tools/neff.py, their ingested
neuron-explorer parquet profiles, and the measure.py results.json of the interleaved re-measurement.

usage: python export_data.py <unpacked_root> <parquet_root> [<results.json>]
       (<unpacked_root>/iterN/ and <parquet_root>/iterN/ for N = 0..7; without results.json
        the committed figs/data/gdn_latency.json is kept)
"""
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import isa  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
PROF = {"PE": "Tensor", "Activation": "Scalar", "Pool": "GpSimd", "DVE": "Vector", "SP": "Sync"}
unp, pq = sys.argv[1:3]
results = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(DATA, exist_ok=True)

# 1. latency per iteration (interleaved re-measurement)
if results:
    r = json.load(open(results))
    json.dump({k: {kk: v[kk] for kk in ("median_us", "min_us", "max_us", "samples")}
               for k, v in r.items()}, open(os.path.join(DATA, "gdn_latency.json"), "w"), indent=1)


def kernel_instructions(d):
    ins = pd.read_parquet(f"{d}/Instruction.parquet")
    k = ins[ins.compiler_pc.notna()].copy()
    t0 = k.start_ts.min()
    k["s_us"] = (k.start_ts - t0) / 1e3
    k["e_us"] = (k.end_ts - t0) / 1e3
    k["pc"] = k.compiler_pc.astype(int)
    return k, t0


busy = []
for it in range(8):
    d = f"{pq}/iter{it}"
    k, t0 = kernel_instructions(d)
    # 2. timeline: one row per executed instruction (runtime opcode)
    k[["engine", "pc", "opcode", "s_us", "e_us"]].round(4).sort_values(["engine", "s_us", "pc"]).to_csv(
        os.path.join(DATA, f"timeline_iter{it}.csv"), index=False)
    # DMA packets merged into busy intervals
    pk = pd.read_parquet(f"{d}/DmaPacket.parquet")
    pk = pk[pk.start_ts >= t0].sort_values("start_ts")
    iv = []
    for a, b in zip((pk.start_ts - t0) / 1e3, (pk.end_ts - t0) / 1e3):
        if iv and a <= iv[-1][1]:
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    pd.DataFrame(iv, columns=["s_us", "e_us"]).round(4).to_csv(
        os.path.join(DATA, f"dma_iter{it}.csv"), index=False)
    s = pd.read_parquet(f"{d}/Summary.parquet").iloc[0]
    busy.append({"iter": it, "total_exec_us": s.total_exec_time * 1e6,
                 **{e: s[f"{f}_active_time"] * 1e6 for e, f in
                    [("PE", "tensor_engine"), ("ACT", "scalar_engine"), ("DVE", "vector_engine"),
                     ("Pool", "gpsimd_engine"), ("SP", "sync_engine"), ("DMA", "dma")]},
                 "instructions": int(len(k)),
                 "dge_dma": int((k.opcode == "DMA_DIRECT2D").sum()),
                 "act_table_loads": int((k.opcode == "ACT_TABLE_LOAD").sum())})
pd.DataFrame(busy).round(3).to_csv(os.path.join(DATA, "gdn_engine_busy.csv"), index=False)

# 2b. roofline inputs: Tensor Engine FLOPs and HBM bytes per NEFF, per-NeuronCore peaks
roof = {}
for it in range(8):
    s = pd.read_parquet(f"{pq}/iter{it}/Summary.parquet").iloc[0]
    roof[f"iter{it}"] = {"tensor_flops": float(s.hardware_flops),
                         "hbm_bytes": float(s.hbm_read_bytes + s.hbm_write_bytes),
                         "io_bytes": float(s.inputs_outputs_weights_size_bytes)}
m = pd.read_parquet(f"{pq}/iter0/Metadata.parquet").iloc[0]
roof["peak"] = {"dma_bytes_per_s": float(m.dma_ddr_bandwidth),
                "tensor_flops_per_s": float(m.tensor_engine_num_rows * m.tensor_engine_num_cols * 2
                                            * m.tensor_engine_clock_freq * 1e9)}
json.dump(roof, open(os.path.join(DATA, "gdn_roofline.json"), "w"), indent=1)

# 3. every DGE DMA: issue cost; completion latency where the semaphore is DMA-only
rows = []
for it in range(8):
    root, d = f"{unp}/iter{it}", f"{pq}/iter{it}"
    k, t0 = kernel_instructions(d)
    agg = k.groupby(["engine", "pc"]).agg(start=("start_ts", "min"), end=("end_ts", "max"))
    sem = pd.read_parquet(f"{d}/SemaphoreUpdate.parquet")
    sem["sid"] = pd.to_numeric(sem["id"].str.extract(r"S\[(\d+)\]")[0], errors="coerce")
    sem = sem[sem.sid.notna()].sort_values("timestamp")
    dge, other = [], set()
    for eng in PROF:
        for i, x in isa.decode_stream(open(f"{root}/payload/sg00/{eng}0.bin", "rb").read()):
            a = x.as_dict()
            if a.get("events_update_mode") not in (None, 0, "NONE"):
                other.add(a["events_update_idx"])
            if x.name != "PSEUDO_DMA_DIRECT2D":
                continue

            def kind(p):
                if a[f"{p}_start_addr_addr_var_marker"] & 0x20:
                    return "DRAM"
                return "SBUF" if a[f"{p}_start_addr_addr_var_var_id_lo"] == 0 else "DRAM"
            n = a["src_num_elem"][0] * max(1, a["src_num_elem"][1])
            dge.append(dict(iter=it, engine=eng, slot=i, sem=a["semaphore"], src=kind("src"),
                            dst=kind("dst"), elem=a["dst_elem_size"] or 65536, desc=n))
    df = pd.DataFrame(dge)
    df["k"] = df.groupby("sem").cumcount() + 1
    for row in df.itertuples():
        key = (PROF[row.engine], row.slot)
        if key not in agg.index:
            continue
        st, en = agg.loc[key]
        lat = None
        if row.sem not in other:
            g = sem[(sem.sid == row.sem) & (sem.value >= 16 * row.k)]
            lat = (g.timestamp.iloc[0] - en) / 1e3 if len(g) else None
        rows.append({**{c: getattr(row, c) for c in df.columns},
                     "issue_start_us": round((st - t0) / 1e3, 4),
                     "issue_us": round((en - st) / 1e3, 4),
                     "latency_us": None if lat is None else round(lat, 4)})
pd.DataFrame(rows).to_csv(os.path.join(DATA, "gdn_dge_dma.csv"), index=False)
print("wrote", sorted(os.listdir(DATA)))

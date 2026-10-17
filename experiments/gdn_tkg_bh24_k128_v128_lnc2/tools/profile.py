#!/usr/bin/env python3
"""The metric: measure `kernel.neff` on the device and report `total_exec_time`.

Per repeat it runs the NEFF `--num-exec N` times and captures a hardware profile of the **last**
execution (so the measured one is warm), then reads `total_exec_time` out of
`neuron-explorer view --output-format summary-json`. That is one device execution of the NEFF as
timed by the device itself: no host, runtime or framework overhead in the number.

Prints, for the loop to pick up:

    ##autohelix[total_exec_time_us=<median over the repeats>]

and, for whoever is reading, the spread, the comparison against `baseline/profile_baseline.json`,
and the per-engine busy time from the median repeat -- which is where to look for the next edit.

Usage:
    python tools/profile.py                       # 3 repeats x 5 executions
    python tools/profile.py --repeats 5
    python tools/profile.py --parquet prof        # also ingest the profile for per-instruction work
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import device as D  # noqa: E402
import neffio  # noqa: E402

WORK = os.path.join(D.ROOT, ".profile")
BASELINE = os.path.join(D.ROOT, "baseline", "profile_baseline.json")

# Engine names as the profiler spells them, with the NEFF's own name for the same engine.
ENGINES = [
    ("tensor_engine_active_time", "PE / Tensor"),
    ("scalar_engine_active_time", "Activation / Scalar"),
    ("vector_engine_active_time", "DVE / Vector"),
    ("gpsimd_engine_active_time", "Pool / GpSimd"),
    ("sync_engine_active_time", "SP / Sync"),
    ("dma_active_time", "DMA"),
]
KEEP = ["total_active_time", "total_time", "neuroncore_cycle_count", "hardware_flops",
        "mfu_estimated_percent", "mbu_estimated_percent", "hfu_estimated_percent",
        "hbm_read_bytes", "hbm_write_bytes", "dma_transfer_total_bytes", "dma_transfer_count",
        "dma_queue_count", "spill_reload_bytes", "spill_save_bytes",
        "tensor_engine_instruction_count", "scalar_engine_instruction_count",
        "vector_engine_instruction_count", "gpsimd_engine_instruction_count",
        "sync_engine_instruction_count", "matmul_instruction_count", "nc_idx"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3, help="captures to take")
    ap.add_argument("--num-exec", type=int, default=5,
                    help="executions per capture; the last one is profiled, so it is warm")
    ap.add_argument("--timeout", type=int, default=900, help="per-capture timeout in seconds")
    ap.add_argument("--parquet", metavar="DIR",
                    help="also ingest the last profile into parquet tables for per-instruction "
                         "analysis (Instruction.parquet: engine, compiler_pc, opcode, "
                         "duration_ns)")
    ap.add_argument("--keep-ntff", action="store_true")
    a = ap.parse_args()

    m = D.manifest()
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)

    print(f"profiling kernel.neff on {m['case']}  {m['params']}")
    print(f"  sha256 {neffio.sha256(D.NEFF)}")
    print(f"  {a.repeats} repeats x {a.num_exec} executions, profiling the last of each")
    samples, summaries, last_ntff = [], [], None
    for r in range(a.repeats):
        stem = os.path.join(WORK, f"r{r}")
        res = D.capture(WORK, m, ntff=stem + ".ntff", num_exec=a.num_exec,
                        profile_nth=a.num_exec, timeout=a.timeout)
        if res["returncode"] != 0:
            D.save_json(os.path.join(WORK, f"r{r}_capture.json"), res)
            how = "timed out" if res["timed_out"] else f"exited {res['returncode']}"
            print(f"  capture {how}:\n{(res['stderr'] or res['stdout'])[-1200:]}")
            print("no measurement: fix the NEFF (run tools/verify.py) before measuring")
            return 1
        ntff = D.find_ntff(stem)
        if ntff is None:
            print(f"  capture wrote no NTFF: {res['stdout'][-800:]}")
            return 1
        s = D.view_summary(ntff)
        us = s["total_exec_time"] * 1e6
        samples.append(us)
        summaries.append(s)
        last_ntff = ntff
        print(f"    repeat {r}: total_exec_time {us:10.3f} us   (capture {res['wall_seconds']}s)")

    med = statistics.median(samples)
    mid = summaries[samples.index(min(samples, key=lambda x: abs(x - med)))]
    exec_us = mid["total_exec_time"] * 1e6
    report = {
        "case": m["case"], "params": m["params"],
        "neff_sha256": neffio.sha256(D.NEFF), "neff_bytes": os.path.getsize(D.NEFF),
        "repeats": a.repeats, "num_exec": a.num_exec,
        "total_exec_time_us": {
            "median": round(med, 3), "min": round(min(samples), 3), "max": round(max(samples), 3),
            "spread_percent": round(100 * (max(samples) - min(samples)) / med, 3),
            "samples": [round(x, 3) for x in samples],
        },
        "engine_busy_us": {label: round(mid.get(f, 0.0) * 1e6, 3) for f, label in ENGINES},
        # Worked out against one execution. The summary's own `*_active_time_percent` fields are
        # fractions of `total_time` (the whole capture), which is a different denominator.
        "engine_busy_percent_of_exec": {
            label: round(100 * mid.get(f, 0.0) * 1e6 / exec_us, 1) for f, label in ENGINES},
        "summary": {f: mid.get(f) for f in KEEP},
    }

    print(f"  median {med:.3f} us   min {min(samples):.3f}   max {max(samples):.3f}   "
          f"spread {report['total_exec_time_us']['spread_percent']:.2f}%")

    base = None
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            base = json.load(f)
        b = base["total_exec_time_us"]["median"]
        report["baseline_us"] = b
        report["speedup_vs_baseline"] = round(b / med, 4)
        report["change_percent_vs_baseline"] = round(100 * (med - b) / b, 2)
        print(f"  baseline {b:.3f} us  ->  {med:.3f} us   "
              f"{report['change_percent_vs_baseline']:+.2f}%  "
              f"({report['speedup_vs_baseline']:.4f}x)")

    print("  engine busy time, median repeat (an engine near 100% is the thing to attack):")
    for label, us in sorted(report["engine_busy_us"].items(), key=lambda kv: -kv[1]):
        pct = report["engine_busy_percent_of_exec"][label]
        bar = "#" * int(round(pct / 4))
        extra = ""
        if base:
            was = base["engine_busy_us"].get(label)
            if was:
                extra = f"  (baseline {was:9.3f} us, {100 * (us - was) / was:+.1f}%)"
        print(f"    {label:<22} {us:10.3f} us  {pct:5.1f}%  {bar:<25}{extra}")

    D.save_json(os.path.join(D.ROOT, "profile_latest.json"), report)

    if a.parquet:
        out = os.path.join(D.ROOT, a.parquet)
        shutil.rmtree(out, ignore_errors=True)
        D.view_parquet(last_ntff, out)
        print(f"  per-instruction tables in {a.parquet}/ (Instruction.parquet, ...)")
    if a.keep_ntff or a.parquet:
        print(f"  NTFFs kept in {os.path.relpath(WORK, D.ROOT)}/")
    else:
        for f in glob.glob(os.path.join(WORK, "*.ntff")):
            os.remove(f)

    # The loop reads this line. Median, not min: it is the number that has to come down.
    print(f"##autohelix[total_exec_time_us={med:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

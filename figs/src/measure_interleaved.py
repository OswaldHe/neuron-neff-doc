"""Profile several NEFFs of the gdn_block_tkg loop back-to-back, interleaved per round.

usage: python measure.py <out_dir> <repeats> <neff> [<neff> ...]

Each round captures every NEFF once (5 executions, profile the last), so slow device drift hits
all candidates equally. Writes <out_dir>/<stem>_r<k>.ntff and <out_dir>/results.json.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("GDN_LOOP_TOOLS", "/home/ubuntu/workspace/gdn-loops/gdn_block_tkg_b1_h128_vh4/tools"))
import device as D  # noqa: E402

out, repeats, neffs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
os.makedirs(out, exist_ok=True)
m = D.manifest()
res = {os.path.basename(n)[:-5]: [] for n in neffs}
for r in range(repeats):
    for n in neffs:
        stem = os.path.basename(n)[:-5]
        ntff_stem = os.path.join(out, f"{stem}_r{r}")
        c = D.capture(out, m, neff=n, ntff=ntff_stem + ".ntff", num_exec=5, profile_nth=5,
                      timeout=300)
        if c["returncode"] != 0:
            print(stem, r, "FAILED", c["stderr"][-500:], flush=True)
            continue
        ntff = D.find_ntff(ntff_stem)
        s = D.view_summary(ntff, neff=n)
        us = s["total_exec_time"] * 1e6
        res[stem].append({"r": r, "us": us, "ntff": ntff})
        print(f"round {r} {stem:8s} {us:9.3f} us", flush=True)
summary = {}
for k, v in res.items():
    xs = [x["us"] for x in v]
    med = statistics.median(xs)
    best = min(v, key=lambda x: abs(x["us"] - med))
    summary[k] = {"median_us": med, "min_us": min(xs), "max_us": max(xs), "samples": xs,
                  "median_ntff": best["ntff"]}
    print(f"{k:8s} median {med:9.3f}  min {min(xs):9.3f}  max {max(xs):9.3f}")
json.dump(summary, open(os.path.join(out, "results.json"), "w"), indent=2)

# gdn_cte_b32_s128_d128: nkilib.experimental.gdn.gdn_cte.gdn_cte, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_cte.gdn_cte` for batch_size 32, seq_len 128, head_dim 128, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **1351.404 µs** (`measurements/profile_baseline.json`) |
| Optimized | **1166.256 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.159x (-13.7%)** |
| Iterations | 7 (7h 15m 36s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.0011 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 1.097e-04, output_1 9.081e-04 |
| Error, optimized | output_0 1.097e-04, output_1 9.081e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 340.93 (25%) | 340.46 (29%) |
| Activation / Scalar | 626.67 (46%) | 520.95 (45%) |
| DVE / Vector | 534.25 (40%) | 548.66 (47%) |
| Pool / GpSimd | 254.79 (19%) | 234.26 (20%) |
| SP / Sync | 131.05 (10%) | 53.19 (5%) |
| DMA | 420.02 (31%) | 329.79 (28%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1    1313.320    1.027x   -2.7%   d6d348b   Fuse 32 SBUF->DRAM->SBUF staging round-trips into direct SBUF->SBUF queued transfers (rewrite the reloa
   2    1208.034    1.117x   -8.0%   f8380ab   Move all 160 DRAM staging round-trips onto a new queued-DMA queue (qSPSpillReload1, 16 rings on free se
   3    1204.834    1.120x   -0.3%   -------   (no change to kernel.neff in this iteration)
   4    1176.936    1.146x   -2.3%   a8b683b   Remove SP head-of-line blocking by moving 44 of iter-2's queued-DMA relay EVENT_SEMAPHOREs (which stall
   5    1172.353    1.151x   -0.4%   a04258d   Replaced iter-2's 160 SP relay EVENT_SEMAPHOREs by retargeting the queued-DMA queue to increment the co
   6    1170.257    1.153x   -0.2%   5585a33   Fused the two queued DMA hops on the W path into one for all 64 instances (Wg_grp --q2 cast--> W_all --
   7    1166.256    1.157x   -0.3%   fb6aac9   Deleted a ~1.4 us queued-DMA hop from the W-path's gating chain by retargeting the 32 Pool EVENT_SEMAPH
```

## Contents

```
baseline.neff / optimized.neff   the two NEFFs; diff them with
                                 /home/ubuntu/workspace/neff_doc/tools/neff_diff.py
nki/gdn_cte.py                   the kernel source as compiled
nki/gdn_cte_torch.py             torch reference, used for the golden
manifest.json                    IO names, shapes, dtypes, tolerances, max_abs_err_limit
tools/verify.py                  the constraint: run the NEFF on the device, compare to golden
tools/launch_rules.py            the constraint that kept the launch shape (subgraph count,
                                 no collectives, no replica groups, num_tpb 1, same interface)
tools/profile.py                 the metric: total_exec_time from a device profile
tools/device.py, neffio.py       helpers (capture/view, NEFF header and member reading)
measurements/                    baseline, optimized and per-iteration profiles
logs/                            improvement logs, per-iteration notes and reviews, history,
                                 the loop's console output, and the autohelix.yaml that drove it
```

## Re-verifying or re-measuring

The input and golden tensors are not copied here. They live in the loop repository, which is
where these tools expect them:

```bash
cd /home/ubuntu/workspace/gdn-loops/gdn_cte_b32_s128_d128
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_cte_b32_s128_d128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b32_s128_d128`.

# gdn_cte_b1_s256_d128: nkilib.experimental.gdn.gdn_cte.gdn_cte, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_cte.gdn_cte` for batch_size 1, seq_len 256, head_dim 128, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **67.525 µs** (`measurements/profile_baseline.json`) |
| Optimized | **53.247 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.268x (-21.1%)** |
| Iterations | 7 (8h 29m 57s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.00054 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 6.092e-05, output_1 4.494e-04 |
| Error, optimized | output_0 6.092e-05, output_1 4.494e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 17.21 (25%) | 17.75 (33%) |
| Activation / Scalar | 37.04 (55%) | 31.74 (60%) |
| DVE / Vector | 28.48 (42%) | 29.80 (56%) |
| Pool / GpSimd | 15.16 (22%) | 14.08 (26%) |
| SP / Sync | 9.65 (14%) | 11.93 (22%) |
| DMA | 25.27 (37%) | 20.38 (38%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1      61.939    1.092x   -8.4%   86e9bd9   Fused the five SBUF->DRAM->SBUF staging round-trips into direct SBUF->SBUF DMAs and moved five DGE stag
   2      61.245    1.104x   -1.1%   9d8415c   Moved two var9 reload DGEs off SP, where they sat behind a trigger waiting 2.9 us for an unrelated queu
   3      61.350    1.102x   +0.2%   -------   (no change to kernel.neff in this iteration)
   4      59.622    1.134x   -2.8%   ddb2b42   Edited only kernel.neff: moved the fence-critical var11 staging DGE to an earlier-issuing slot and spre
   5      57.899    1.168x   -2.9%   880982b   Built a backward critical-path walker over the profile and fixed the six nodes it pointed at: gave the 
   6      55.107    1.227x   -4.8%   a7d7590   Rebuilt the critical-path walker (now resolving queued-transfer semaphores) and took the six nodes it p
   7      53.247    1.270x   -3.4%   b928be2   Removed two provably dead DVE CASTs (each an exact re-execution of an earlier slot with no intervening 
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
cd /home/ubuntu/workspace/gdn-loops/gdn_cte_b1_s256_d128
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_cte_b1_s256_d128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b1_s256_d128`.

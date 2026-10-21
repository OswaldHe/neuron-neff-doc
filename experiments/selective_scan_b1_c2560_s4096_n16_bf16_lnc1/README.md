# selective_scan_b1_c2560_s4096_n16_bf16: nkilib.experimental.scan.selective_scan, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.scan.selective_scan` for batch 1, channels 2560, seq_len 4096, state_size 16, use_D True, use_initial_state False, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **7919.960 µs** (`measurements/profile_baseline.json`) |
| Optimized | **6603.962 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.199x (-16.6%)** |
| Iterations | 7 (7h 58m 26s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.15 (upstream atol/rtol 0.1/0.05) |
| Error, baseline | output_0 1.250e-01, output_1 1.812e-04 |
| Error, optimized | output_0 1.250e-01, output_1 1.812e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 2970.72 (38%) | 2969.83 (45%) |
| Activation / Scalar | 1902.25 (24%) | 1903.09 (29%) |
| DVE / Vector | 7860.18 (99%) | 6521.03 (99%) |
| Pool / GpSimd | 1214.15 (15%) | 5611.35 (85%) |
| SP / Sync | 609.03 (8%) | 609.06 (9%) |
| DMA | 1485.80 (19%) | 1442.87 (22%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1    6886.377    1.150x  -13.0%   fcf76a6   Move 2214 of the fp32 y_accum+=Cs TENSOR_TENSOR ADDs off the saturated DVE engine onto the 15%-idle Poo
   2    6818.831    1.161x   -1.0%   e7f3a7a   Move the last 351 y_accum+=Cs fp32 TENSOR_TENSOR ADDs that increment S[6] from the saturated DVE engine
   3    6712.060    1.180x   -1.6%   fa7f641   Merge 973 of the 1-element Pool carry COPYs pairwise into dim-1 strided copies (-300 us of Pool time) a
   4    6604.127    1.199x   -1.6%   51c6a8e   Stop the 98.8%-busy DVE engine blocking on waits it does not need yet: delete the 154 y_accum MEMSETs (
   5    6604.441    1.199x   +0.0%   -------   (not merged)
   6    6603.581    1.199x   -0.0%   -------   (not merged)
   7    6603.962    1.199x   +0.0%   -------   (not merged)
```

## Contents

```
baseline.neff / optimized.neff   the two NEFFs; diff them with
                                 /home/ubuntu/workspace/neff_doc/tools/neff_diff.py

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
cd /home/ubuntu/workspace/gdn-loops/selective_scan_b1_c2560_s4096_n16_bf16
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py selective_scan_b1_c2560_s4096_n16_bf16`.

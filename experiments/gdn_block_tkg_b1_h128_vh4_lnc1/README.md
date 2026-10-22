# gdn_block_tkg_b1_h128_vh4: nkilib.experimental.gdn.gdn_block_tkg.gdn_block_tkg, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_block_tkg.gdn_block_tkg` for batch 1, hidden 128, num_k_heads 2, num_v_heads 4, head_dim 128, kernel_width 4, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **218.172 µs** (`measurements/profile_baseline.json`) |
| Optimized | **44.384 µs** (`measurements/profile_optimized.json`) |
| Gain | **4.916x (-79.7%)** |
| Iterations | 7 (10h 8m 33s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.0036 (upstream atol/rtol 0.03/0.03) |
| Error, baseline | output_0 2.999e-03, output_1 1.928e-03, output_2 6.033e-04 |
| Error, optimized | output_0 2.999e-03, output_1 1.928e-03, output_2 6.033e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 7.57 (3%) | 6.63 (15%) |
| Activation / Scalar | 41.80 (19%) | 35.63 (80%) |
| DVE / Vector | 19.15 (9%) | 18.52 (42%) |
| Pool / GpSimd | 35.79 (16%) | 18.72 (42%) |
| SP / Sync | 27.11 (12%) | 6.32 (14%) |
| DMA | 52.31 (24%) | 17.62 (40%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1      95.217    2.287x  -56.3%   480fb81   Rerouted all six SBUF->DRAM->SBUF scratch round-trips (tmp-bufs 6_1, 52_47, 7_2, 111_106, 110_105, 109_
   2      83.176    2.618x  -12.6%   d596d33   Cut 29 DGE DMA instructions off the critical path (16 exact duplicates and 5 signal-only dummies left b
   3      74.921    2.907x   -9.9%   fa6ac0a   Cut 8.6 us (83.2 -> 74.6 us, 1.12x) by taking work off the two engines that gate the dependency chain a
   4      60.157    3.620x  -19.7%   e5fb209   Cut 14.6 us (74.6 -> 60.0 us, 1.24x) by moving six DMA transfers onto idle compute engines, deleting fo
   5      51.175    4.255x  -14.9%   dd179c9   Cut 9.0 us (60.0 -> 51.0 us, 1.18x) by converting both 512-descriptor DGE transpose groups into 128-des
   6      47.695    4.566x   -6.8%   0220238   Cut 3.3 us (51.1 -> ~47.8 us, 1.07x) by repointing one Activation gate off a dead 2-byte dummy DMA onto
   7      44.384    4.906x   -6.9%   dfe4a21   Cut 3.5 us (47.7 -> 44.2 us, 1.08x) by collapsing the 16 prologue weight-tile DMAs into two 8-tile DGE 
```

## Contents

```
baseline.neff / optimized.neff   the two NEFFs; diff them with
                                 /home/ubuntu/workspace/neff_doc/tools/neff_diff.py
nki/gdn_block_tkg.py             the kernel source as compiled
nki/gdn_block_tkg_torch.py       torch reference, used for the golden
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
cd /home/ubuntu/workspace/gdn-loops/gdn_block_tkg_b1_h128_vh4
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_block_tkg_b1_h128_vh4_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_block_tkg_b1_h128_vh4`.

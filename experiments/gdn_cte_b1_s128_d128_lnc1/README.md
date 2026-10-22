# gdn_cte_b1_s128_d128: nkilib.experimental.gdn.gdn_cte.gdn_cte, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_cte.gdn_cte` for batch_size 1, seq_len 128, head_dim 128, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **48.519 µs** (`measurements/profile_baseline.json`) |
| Optimized | **37.635 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.289x (-22.4%)** |
| Iterations | 7 (10h 2s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.00035 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 6.119e-05, output_1 2.864e-04 |
| Error, optimized | output_0 6.119e-05, output_1 2.864e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 12.15 (25%) | 11.99 (32%) |
| Activation / Scalar | 23.37 (48%) | 20.52 (55%) |
| DVE / Vector | 19.27 (40%) | 19.65 (52%) |
| Pool / GpSimd | 11.03 (23%) | 9.08 (24%) |
| SP / Sync | 6.51 (13%) | 8.50 (23%) |
| DMA | 17.92 (37%) | 15.75 (42%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1      43.427    1.121x  -10.8%   f8f9f89   Fused all five SBUF->DRAM->SBUF staging round-trips into direct SBUF->SBUF transfers, moved the gate-lo
   2      41.056    1.186x   -5.5%   ac545ca   Five cross-engine scheduling edits, no arithmetic touched: three Activation TENSOR_SCALARs now execute 
   3      40.094    1.215x   -2.3%   e833bbc   Four cross-engine scheduling edits, no arithmetic touched: the sem-43 DGE row moves off Activation onto
   4      38.417    1.268x   -4.2%   e6372de   Six cross-engine scheduling edits, no arithmetic touched: the sem-17 output-store DGE row now reads its
   5      38.170    1.276x   -0.6%   6d77c44   Four cross-engine scheduling edits on the blk20 branch of the tail, no arithmetic touched: ACT[76]/ACT[
   6      38.224    1.274x   +0.1%   2106d0b   Three scheduling edits in the tail, no arithmetic touched: the update-less ACT TENSOR_SCALAR (dst p0+0x
   7      37.635    1.294x   -1.5%   a344adb   Both tail output stores moved off their DGE rows onto new "type": "in" DMA queues added to def.json (th
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
cd /home/ubuntu/workspace/gdn-loops/gdn_cte_b1_s128_d128
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_cte_b1_s128_d128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b1_s128_d128`.

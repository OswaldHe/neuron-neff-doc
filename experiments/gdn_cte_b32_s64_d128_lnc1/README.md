# gdn_cte_b32_s64_d128: nkilib.experimental.gdn.gdn_cte.gdn_cte, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_cte.gdn_cte` for batch_size 32, seq_len 64, head_dim 128, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **1184.963 µs** (`measurements/profile_baseline.json`) |
| Optimized | **835.849 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.418x (-29.5%)** |
| Iterations | 3 (all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.00071 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 7.688e-05, output_1 5.842e-04 |
| Error, optimized | output_0 7.688e-05, output_1 5.842e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 263.47 (22%) | 262.65 (31%) |
| Activation / Scalar | 414.19 (35%) | 346.59 (41%) |
| DVE / Vector | 436.79 (37%) | 438.68 (52%) |
| Pool / GpSimd | 186.90 (16%) | 166.74 (20%) |
| SP / Sync | 479.69 (40%) | 34.20 (4%) |
| DMA | 308.04 (26%) | 200.34 (24%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1     874.474    1.357x  -26.3%   86fffa6   fused the 160 SBUF->DRAM->SBUF staging round-trips into single queued SB->SB transfers, replaced the 32
   2     836.402    1.419x   -4.4%   aafb305   Fused the A_full->Ball->transpose two-hop queued-DMA chain into one hop: all 161 qSPSpillReload2 transp
   3     835.849    1.420x   -0.1%   -------   (not merged)
```

## Contents

```
baseline.neff / optimized.neff   the two NEFFs; diff them with
                                 /home/ubuntu/workspace/neff_doc/tools/neff_diff.py
nki/gdn_cte.py                   original kernel source
nki/gdn_cte_torch.py             original kernel source
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
cd /home/ubuntu/workspace/gdn-loops/gdn_cte_b32_s64_d128
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_cte_b32_s64_d128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b32_s64_d128`.

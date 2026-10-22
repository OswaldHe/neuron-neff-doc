# gdn_tkg_bh24_k128_v128: nkilib.experimental.gdn.gdn_tkg.gdn_tkg, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_tkg.gdn_tkg` for bh 24, k_dim 128, v_dim 128, dtype bfloat16, num_v_heads 12, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **153.463 µs** (`measurements/profile_baseline.json`) |
| Optimized | **60.941 µs** (`measurements/profile_optimized.json`) |
| Gain | **2.518x (-60.3%)** |
| Iterations | 7 (9h 42m 47s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.028 (upstream atol/rtol 0.02/0.02) |
| Error, baseline | output_0 2.322e-02, output_1 1.633e-03 |
| Error, optimized | output_0 2.322e-02, output_1 1.633e-03 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 21.11 (14%) | 18.69 (31%) |
| Activation / Scalar | 143.47 (93%) | 39.13 (64%) |
| DVE / Vector | 34.24 (22%) | 46.82 (77%) |
| Pool / GpSimd | 9.38 (6%) | 15.49 (25%) |
| SP / Sync | 3.28 (2%) | 3.26 (5%) |
| DMA | 20.45 (13%) | 21.72 (36%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1      79.641    1.922x  -48.0%   86e367c   Edited kernel.neff only: collapsed the 24 per-head silu / k-scale / state-scale ops into one wide ACT i
   2      74.615    2.052x   -6.3%   504faee   Edited only kernel.neff: group-merged the 24 pre-outer S_bf casts, split the 3.35 us k_f*=k_scale DVE o
   3      72.684    2.106x   -2.6%   e8d4527   Edited only kernel.neff: split the 3.36 us DVE S*=exp_g into 4 chunks of 6 heads and the 1.5 MB state_i
   4      69.156    2.214x   -4.9%   66f6259   Edited only kernel.neff: deleted three artificial fences on the per-head pipeline (PE's outer-product g
   5      66.278    2.310x   -4.2%   b17922f   Edited only kernel.neff: folded the per-head "w = beta*delta" scale-multiply out of the Activation queu
   6      62.546    2.448x   -5.6%   8d63265   Edited only kernel.neff: moved 21 of the 24 per-head S->S_bf fp32->bf16 CASTs off the Activation engine
   7      60.941    2.512x   -2.6%   62c95b6   Edited only kernel.neff: deleted the 18 do-nothing DVE and 14 update-free Activation EVENT_SEMAPHORE sl
```

## Contents

```
baseline.neff / optimized.neff   the two NEFFs; diff them with
                                 /home/ubuntu/workspace/neff_doc/tools/neff_diff.py
nki/gdn_tkg.py                   the kernel source as compiled
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
cd /home/ubuntu/workspace/gdn-loops/gdn_tkg_bh24_k128_v128
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_tkg_bh24_k128_v128`.

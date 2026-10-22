# gdn_tkg_bh24_k128_v128_lnc2: nkilib.experimental.gdn.gdn_tkg.gdn_tkg, LNC=2

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_tkg.gdn_tkg` for bh 24, k_dim 128, v_dim 128, dtype bfloat16, num_v_heads 12, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **76.054 µs** (`measurements/profile_baseline.json`) |
| Optimized | **36.175 µs** (`measurements/profile_optimized.json`) |
| Gain | **2.102x (-52.4%)** |
| Iterations | 7 (7h 28m 33s, all merged) |
| Launch shape | LNC=2, 2 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.027 (upstream atol/rtol 0.02/0.02) |
| Error, baseline | output_0 2.322e-02, output_1 1.633e-03 |
| Error, optimized | output_0 2.322e-02, output_1 1.633e-03 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 22.17 (29%) | 19.49 (54%) |
| Activation / Scalar | 75.70 (100%) | 33.40 (92%) |
| DVE / Vector | 32.99 (43%) | 34.10 (94%) |
| Pool / GpSimd | 9.46 (12%) | 13.42 (37%) |
| SP / Sync | 5.33 (7%) | 6.06 (17%) |
| DMA | 18.05 (24%) | 21.45 (59%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1      55.601    1.368x  -26.9%   df60817   Rescheduled the Activation stream: hoisted all 12 silu activations into one wide ACTIVATE on private SB
   2      44.612    1.705x  -19.8%   400b689   Moved all 36 scale-pointer ACTIVATEs off the saturated ACT engine onto idle DVE as TENSOR_TENSORs with 
   3      40.740    1.867x   -8.7%   119bfcf   Deleted the 12 ACT PSUM->SBUF `o_sb` COPYs (both readers now read out_p straight from PSUM, bit-exact, 
   4      38.714    1.964x   -5.0%   7bc704d   Split the state-out state DGE DMA into 4 per-head-group transfers so the tail overlaps compute, pipelin
   5      38.396    1.980x   -0.8%   4118dcb   Tightened the 12 PE outer-product s4 waits onto the ACT `cast_delta` that actually produces each `delta
   6      36.714    2.071x   -4.4%   88ded90   Took the two wide head `CAST S_all -> S_bf` off the saturated ACT head chain by scheduling them against
   7      36.175    2.102x   -1.5%   582b1a7   Moved five of the twelve DVE `out[h] = out_norm * silu_z[h]` sinks forward to just before the DVE slots
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
cd /home/ubuntu/workspace/gdn-tkg-neff-opt
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_tkg_bh24_k128_v128_lnc2`.

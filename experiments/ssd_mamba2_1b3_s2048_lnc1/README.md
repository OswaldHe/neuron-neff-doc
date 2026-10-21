# ssd_mamba2_1b3_s2048: nkilib.experimental.scan.ssd, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.scan.ssd` for batch 1, nheads 64, seqlen 2048, headdim 64, dstate 128, chunk_size 128, use_D False, use_initial_state False, dtype float32, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **7154.946 µs** (`measurements/profile_baseline.json`) |
| Optimized | **4250.969 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.683x (-40.6%)** |
| Iterations | 7 (8h 15m 45s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 6.2e-07 (upstream atol/rtol 0.2/0.05) |
| Error, baseline | output_0 5.122e-07, output_1 3.166e-07 |
| Error, optimized | output_0 5.234e-07, output_1 3.334e-07 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 1966.84 (27%) | 1919.30 (45%) |
| Activation / Scalar | 6283.85 (88%) | 2549.75 (60%) |
| DVE / Vector | 2733.29 (38%) | 2132.45 (50%) |
| Pool / GpSimd | 2072.32 (29%) | 2071.73 (49%) |
| SP / Sync | 801.68 (11%) | 277.89 (7%) |
| DMA | 1878.87 (26%) | 1287.08 (30%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1    5129.004    1.395x  -28.3%   ef6b76e   Rewrote 1020 of 1024 ACT reciprocal(exp(x)) units as exp(-x) fed by one scaled PSUM->SBUF copy, which l
   2    5126.665    1.396x   -0.0%   038289d   Merged 303 adjacent pairs of identical-configuration ACT ACTIVATEs into single double-length instructio
   3    5106.762    1.401x   -0.4%   5b5d3e5   Replace the redundant second per-unit exp_cs_last broadcast (4 DVE STREAM_SHUFFLEs) with one 128-partit
   4    4855.986    1.473x   -4.9%   73236ba   Deleted the 1920 redundant per-chunk state_sb HBM round-trip DMAs by pinning state_sb to a fixed per-he
   5    4777.847    1.497x   -1.6%   86c8a41   Deleted 984 of the 1020 ACT cs_col PSUM->SBUF copies by pointing their two exp consumers straight at th
   6    4683.076    1.528x   -2.0%   c7383fe   Three bit-exact edits, all driven by the same new facts -- a 2-group 3-D access pattern works on ACTIVA
   7    4250.969    1.683x   -9.2%   383e6ac   Turned off the configured-but-unused `imm` pointer operand on all 6157 pointer-scale ACT ACTIVATEs (imm
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
cd /home/ubuntu/workspace/gdn-loops/ssd_mamba2_1b3_s2048
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/ssd_mamba2_1b3_s2048_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py ssd_mamba2_1b3_s2048`.

# selective_scan_b1_c768_s1024_n16_bf16: nkilib.experimental.scan.selective_scan, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.scan.selective_scan` for batch 1, channels 768, seq_len 1024, state_size 16, use_D True, use_initial_state False, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **602.637 µs** (`measurements/profile_baseline.json`) |
| Optimized | **497.880 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.210x (-17.4%)** |
| Iterations | 7 (6h 32m 34s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.075 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 6.245e-02, output_1 4.268e-05 |
| Error, optimized | output_0 6.245e-02, output_1 4.268e-05 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 219.14 (36%) | 218.57 (44%) |
| Activation / Scalar | 146.24 (24%) | 146.18 (29%) |
| DVE / Vector | 592.37 (98%) | 478.63 (96%) |
| Pool / GpSimd | 101.05 (17%) | 412.75 (83%) |
| SP / Sync | 48.32 (8%) | 48.30 (10%) |
| DMA | 108.41 (18%) | 104.08 (21%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1     516.791    1.166x  -14.3%   6e7af17   Move the 170 free y+=Cs fp32 TENSOR_TENSOR ADDs off the saturated DVE engine onto the 17%-idle Pool eng
   2     516.796    1.166x   +0.0%   5e1f980   Collapse the 12 per-group 512-element y-accumulator MEMSETs into one shared SBUF zero buffer (each grou
   3     510.633    1.180x   -1.2%   af2c8b3   Rewrite the 12 zero-buffer "first y writer" TENSOR_TENSOR ADDs as 2-stream COPYs (y = Cs instead of y =
   4     508.770    1.185x   -0.4%   7358c9d   Retarget the 10 Cs MULTs that feed each group's first y write straight into y (dropping the now-dead CO
   5     501.414    1.202x   -1.4%   74fb4cf   Measured that only DVE TENSOR_TENSORs with all three operands in SBUF lose bandwidth to Pool's 512-elem
   6     501.436    1.202x   +0.0%   -------   (not merged)
   7     497.880    1.211x   -0.7%   -------   (not merged)
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
cd /home/ubuntu/workspace/gdn-loops/selective_scan_b1_c768_s1024_n16_bf16
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/selective_scan_b1_c768_s1024_n16_bf16_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py selective_scan_b1_c768_s1024_n16_bf16`.

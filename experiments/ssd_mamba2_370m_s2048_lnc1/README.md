# ssd_mamba2_370m_s2048: nkilib.experimental.scan.ssd, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.scan.ssd` for batch 1, nheads 48, seqlen 2048, headdim 32, dstate 64, chunk_size 128, use_D False, use_initial_state False, dtype float32, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **5248.875 µs** (`measurements/profile_baseline.json`) |
| Optimized | **2905.247 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.807x (-44.7%)** |
| Iterations | 7 (8h 22m 33s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 2.3e-07 (upstream atol/rtol 0.2/0.05) |
| Error, baseline | output_0 1.881e-07, output_1 1.192e-07 |
| Error, optimized | output_0 2.058e-07, output_1 1.993e-07 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 1254.81 (24%) | 1267.87 (44%) |
| Activation / Scalar | 4721.67 (90%) | 1809.31 (62%) |
| DVE / Vector | 1688.75 (32%) | 1477.74 (51%) |
| Pool / GpSimd | 1558.81 (30%) | 1558.68 (54%) |
| SP / Sync | 575.13 (11%) | 584.23 (20%) |
| DMA | 1193.17 (23%) | 973.03 (33%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1    3624.206    1.448x  -30.9%   200d241   ACT-only edits: turned off the unused pointer `imm` operand on all 6144 ACT ACTIVATEs (bit-exact, -8.3%
   2    3290.283    1.595x   -9.2%   1ed7001   ACT-only + PE-wait edits: both exps of each unit now read the cs PSUM word directly (deleting iter-1's 
   3    3106.025    1.690x   -5.6%   1db4928   Rebuilt the state_sb spill path: 605 of 816 (head,chunk) bounces now move SBUF->SBUF in one SP transfer
   4    3071.035    1.709x   -1.1%   34a8f97   120 of the 166 leftover state_sb (head,chunk) bounces now go single-hop (SP's reload reads X[c-1] direc
   5    2987.616    1.757x   -2.7%   669f7b0   524 of the 768 dead `Y_off` accumulators are folded away: with num_dstate_tiles==1 the sequence `Y_off=
   6    2948.940    1.780x   -1.3%   73e7538   671 of 768 Y_intra units now read the matmul PSUM with the exp_cs_col scale applied in place of the sep
   7    2905.247    1.807x   -1.5%   86973e6   649 of 768 exp_cs_last_N broadcasts are gone: the exp_cs_last_Q column now lands in a private 4-byte SB
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
cd /home/ubuntu/workspace/gdn-loops/ssd_mamba2_370m_s2048
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/ssd_mamba2_370m_s2048_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py ssd_mamba2_370m_s2048`.

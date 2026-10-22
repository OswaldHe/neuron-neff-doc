# gdn_cte_b32_s2304_d128: nkilib.experimental.gdn.gdn_cte.gdn_cte, LNC=1

Compiled NEFF of the nkilib GDN kernel `nkilib.experimental.gdn.gdn_cte.gdn_cte` for batch_size 32, seq_len 2304, head_dim 128, dtype bfloat16, optimized by **editing
the compiled NEFF only** — no source change, no recompile — under a loop that ran the device
verifier every iteration.

| | |
|---|---|
| Baseline | **14778.976 µs** (`measurements/profile_baseline.json`) |
| Optimized | **14022.004 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.054x (-5.1%)** |
| Iterations | 6 (7h 1m 24s, all merged) |
| Launch shape | LNC=1, 1 subgraph(s), one rank — enforced by `tools/launch_rules.py` |
| Error limit | `max_abs_err_limit` 0.0008 (upstream atol/rtol 0.05/0.05) |
| Error, baseline | output_0 1.372e-04, output_1 6.841e-04 |
| Error, optimized | output_0 1.372e-04, output_1 6.841e-04 |

Engine busy time, baseline vs optimized (percent of one execution):

| engine | baseline | optimized |
|---|---|---|
| PE / Tensor | 3189.52 (22%) | 3192.43 (23%) |
| Activation / Scalar | 8943.25 (61%) | 8111.73 (58%) |
| DVE / Vector | 5674.79 (38%) | 5657.56 (40%) |
| Pool / GpSimd | 2751.26 (19%) | 2673.07 (19%) |
| SP / Sync | 1871.04 (13%) | 2738.14 (20%) |
| DMA | 5610.25 (38%) | 5195.11 (37%) |

## The edits

`logs/improvement_summary.log` has the table below in full width;
`logs/merged_edits.log` has the complete reasoning per edit, and `logs/notes/iter-N.md` the
findings behind it — including the levers that were refuted and the measurement that closed
each one.

```
   1   14760.853    1.002x   -0.2%   15aaef4   Move all 2784 DGE (PSEUDO_DMA_DIRECT2D) slots off the Activation stream onto Pool/SP via an S[72] hands
   2   14505.815    1.019x   -1.7%   d8d0cf8   Restore iter-1's 1392 Pool-hosted DGE slots to Activation (Pool0.bin back to baseline, S[72] renumbered
   3   14381.433    1.028x   -0.9%   065c8ca   Stage the 640 DGE-only "spill" buffers in free SBUF instead of DRAM (they are reshapes, not spills: sav
   4   14231.909    1.039x   -1.0%   e026018   Merge 993 affine runs of identical Activation/DVE instructions (mostly 1-element ACTIVATEs) into single
   5   14036.158    1.053x   -1.4%   0b7bb73   Fuse 188 DGE save/reload staging pairs (saves now write the reload's destination directly, the reload b
   6   14022.004    1.054x   -0.1%   602e09b   Retire 7 provably-dead SBUF-writing DGE transfers (replaced by the iter-5 EVENT_SEMAPHORE +16 pair, -1.
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
cd /home/ubuntu/workspace/gdn-neff-opt
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_cte_b32_s2304_d128_lnc1/optimized.neff kernel.neff
python tools/verify.py --runs 2
python tools/profile.py --repeats 5
```

Or regenerate the whole case from scratch, inputs and goldens included:
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b32_s2304_d128`.

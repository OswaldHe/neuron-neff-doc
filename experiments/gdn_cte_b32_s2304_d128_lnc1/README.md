# gdn_cte, batch 32 x seq 2304 x head_dim 128, bfloat16, LNC=1

The nkilib GDN **chunked prefill** kernel (`nkilib.experimental.gdn.gdn_cte`), compiled for one
logical NeuronCore (one subgraph, `sg00`), one rank, and then optimized by editing the compiled
NEFF only.

| | |
|---|---|
| Baseline | **14778.98 µs** (`measurements/profile_baseline.json`) |
| Optimized | **14022.00 µs** (`measurements/profile_optimized.json`) |
| Gain | **1.054x, −5.1%** |
| Iterations | 6, all merged, none rejected (7 h 1 m, $81.06) |
| Correctness | `max_abs_err` unchanged from the baseline: 1.372e-04 on `output_0`, 6.841e-04 on `output_1`, limit 8e-4 |
| Run-to-run spread | 0.01–0.11% on this kernel, so a win under ~0.5% was not claimed |

## Why only 5%

This kernel has **no saturated engine** (Activation 60%, DVE 38%, PE 22% of the execution) and
almost no idle: every engine is idle together for only ~1.3 ms of 14.8 ms, in 15183 gaps whose
largest is 3.9 µs. It is bound by a serial dependency chain — PE waits ~7 ms on the
`qSPSpillReload0` DMA queue, which carries 8285 transfers, 8125 of them hardware transposes
(`transpose_op: WZXY`, SBUF→SBUF). Scheduling levers therefore have small ceilings, and most of
them were measured and closed rather than guessed; `logs/notes/` records each refutation with the
number that closed it.

The one structural discovery worth carrying forward: the 800 `tmp-buf` DRAM variables are **not
spills**. For all 640 DGE-read ones the save's SBUF source and the reload's SBUF destination are
different regions, so DRAM was never holding data while SBUF was reused — it is a
partition/offset redistribution a 2-D DGE cannot do in one hop. Iteration 3 moved that staging
into a 45 KB-per-partition hole in SBUF that it found by reconstructing SBUF live ranges from the
literal access patterns.

## The edits, in order

See `logs/improvement_summary.log` for the table and `logs/merged_edits.log` for the full
reasoning per edit. In short:

1. All 2784 DGE DMA slots moved off Activation onto Pool+SP via a new `S[72]` handshake —
   Activation busy time −14.9%, latency −0.2%. (Occupancy is not the limit.)
2. Half of them moved back: Activation+SP beats both all-Activation and Pool+SP. −1.9%.
3. The 640 DGE-only staging buffers moved from DRAM into free SBUF; 1832 reloads read the
   original SBUF tile directly. 242 MB less DRAM traffic. −0.9%.
4. 993 affine runs of identical Activation/DVE instructions merged into single wider-pattern
   slots.
5. 188 DGE save/reload staging pairs fused (the save writes the reload's destination directly).
6. 7 provably dead SBUF-writing DGE transfers retired.

## Contents

```
baseline.neff / optimized.neff   the two NEFFs (diff them with ../../tools/neff_diff.py)
nki/gdn_cte.py                   the original NKI source; gdn_cte_torch.py is the reference
                                 used to compute the golden
manifest.json                    IO names, shapes, dtypes, atol/rtol, max_abs_err_limit
tools/verify.py                  the constraint: run the NEFF on the device, compare to golden
tools/single_core.py             the constraint that kept it LNC=1 and single-rank
tools/profile.py                 the metric: total_exec_time from a device profile
tools/device.py, neffio.py       helpers (capture/view, NEFF header and member reading)
measurements/                    baseline, optimized, and per-iteration profiles
logs/notes/                      the agent's findings and refutations per iteration
logs/reviews/                    the independent review per iteration (iter-4 onward: Sonnet 5.5,
                                 asked for a ranked list of candidate edits)
logs/autohelix.yaml              the goal, constraints, metric and reviewer prompt
```

The input and golden tensors (~100 MB) are not copied here; they are in
`/home/ubuntu/workspace/gdn-neff-opt/bench/`, or regenerate with
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_cte_b32_s2304_d128`.

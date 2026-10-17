# gdn_tkg, bh 24 x k_dim 128 x v_dim 128, bfloat16, LNC=2

The nkilib GDN **fused single-token decode** kernel (`nkilib.experimental.gdn.gdn_tkg`), compiled
for one logical NeuronCore at LNC=2 — two physical cores, `sg00` and `sg01`, which is how this
kernel ships, since it shards its flattened batch×head axis across them — one rank, then
optimized by editing the compiled NEFF only.

| | |
|---|---|
| Baseline | **76.05 µs** (`measurements/profile_baseline.json`) |
| Optimized | **36.18 µs** (`measurements/profile_optimized.json`) |
| Gain | **2.102x, −52.4%** |
| Iterations | 7, all merged, none rejected (7 h 29 m, $108.99) |
| Correctness | `max_abs_err` unchanged from the baseline: 2.322e-02 on `output_0`, 1.633e-03 on `output_1`, limit 2.7e-2 |
| Run-to-run spread | 0.5–1.2% at this size, so each win was repeated before it was claimed |

## Why 2x was available here

The baseline had the **Activation (Scalar) engine busy 99.5% of the execution** — a genuine
engine bottleneck, unlike the prefill case next door. Every lever that gives that engine less
work, or moves its work onto an idle engine, pays almost directly. After the seven edits the
engine picture is:

| engine | baseline busy | optimized busy |
|---|---|---|
| Activation / Scalar | 75.70 µs (99.5%) | 32.38 µs (89.8%) |
| DVE / Vector | 32.99 | 33.37 (92.5% — the new limit) |
| PE / Tensor | 22.17 | 17.96 |
| DMA | 18.05 | 17.44 |
| Pool / GpSimd | 9.47 | 13.06 |
| SP / Sync | 5.33 | 5.91 |

Activation lost 57% of its work; Pool and SP absorbed some of it; the kernel is now bounded by
DVE.

## The edits, in order

`logs/improvement_summary.log` has the table, `logs/merged_edits.log` the full reasoning. Every
one is described as arithmetic bit-identical — they change scheduling, placement and data
movement, never the maths.

| iter | latency | edit |
|---|---|---|
| 1 | 55.60 µs (−26.9%) | Rescheduled the Activation stream: 12 silu activations hoisted into one wide `ACTIVATE` on private SBUF buffers, `ACT_TABLE_LOAD` cut from 22 to 6 per core |
| 2 | 44.61 (−19.8%) | All 36 scale-pointer `ACTIVATE`s moved to idle DVE as `TENSOR_TENSOR` with a zero-stride broadcast src1 (two groups collapsing into one 1536-element instruction); the 12 `S->S_bf` casts merged into one wide cast |
| 3 | 40.74 (−8.7%) | The 12 Activation PSUM→SBUF `o_sb` copies deleted (both readers read `out_p` straight from PSUM; all 8 PSUM WAR constraints already implied by the happens-before graph), `z` made the first input DMA descriptor, 12 more casts moved to DVE |
| 4 | 38.71 (−5.0%) | State-out DGE DMA split into 4 per-head-group transfers so the tail overlaps compute; the `S *= exp_g` / `CAST` hand-off pipelined in two 6-head halves |
| 5 | 38.40 (−0.8%) | The 12 PE outer-product `s4` waits tightened onto the Activation cast that actually produces each tile — 4 of 12 casts never incremented `s4`, so PE waited up to 2.0 µs past its data being ready. A pure wait renumbering |
| 6 | 36.71 (−4.4%) | Two wide head casts scheduled against their real consumer deadlines instead of sitting on the Activation head chain; one slice moved to idle DVE |
| 7 | 36.18 (−1.5%) | Five DVE sinks moved forward past DVE's own stalls — not one byte changed, only the order of five slots |

## Contents

```
baseline.neff / optimized.neff   the two NEFFs (diff them with ../../tools/neff_diff.py;
                                 both subgraphs sg00 and sg01 differ)
nki/gdn_tkg.py                   the original NKI source
manifest.json                    IO names, shapes, dtypes, atol/rtol, max_abs_err_limit
tools/verify.py                  the constraint: run the NEFF on the device, compare to golden
tools/launch_rules.py            the constraint that allowed LNC=2 but rejected multi-rank:
                                 exactly `lnc` subgraphs, no collectives, no replica_groups,
                                 num_tpb 1, same interface
tools/profile.py                 the metric: total_exec_time from a device profile
tools/device.py, neffio.py       helpers (capture/view, NEFF header and member reading)
measurements/                    baseline, optimized, and per-iteration profiles
logs/notes/                      the agent's findings and refutations per iteration
logs/reviews/                    the independent review per iteration (Sonnet 5.5, asked for a
                                 ranked list of candidate edits)
logs/autohelix.yaml              the goal, constraints, metric and reviewer prompt
```

The input and golden tensors are not copied here; they are in
`/home/ubuntu/workspace/gdn-tkg-neff-opt/bench/`, or regenerate with
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py gdn_tkg_bh24_k128_v128_lnc2`.

Note for anyone editing this NEFF: `sg00` and `sg01` are separate instruction streams per engine
and are **not** identical, so an edit normally has to be applied to both, and each subgraph
carries its own semaphores (`semaphores.py work sg00`, then `sg01`). In the profile,
`pcore_idx` 0 is `sg00` and 1 is `sg01`.

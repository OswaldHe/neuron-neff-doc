# Experiments: optimizing a compiled NEFF, end to end

Two kernels were made faster by **editing the compiled NEFF only** — no source change, no
recompile — each under an automated loop (AutoHelix) that had to pass a device verifier every
iteration. One subfolder per configuration.

| Configuration | Kernel | Shape of the run | Baseline | Optimized | Gain | Iterations |
|---|---|---|---|---|---|---|
| [`gdn_cte_b32_s2304_d128_lnc1`](gdn_cte_b32_s2304_d128_lnc1/) | `nkilib.experimental.gdn.gdn_cte` (chunked prefill) | LNC=1, 1 rank | 14778.98 µs | 14022.00 µs | **1.054x (−5.1%)** | 6 |
| [`gdn_tkg_bh24_k128_v128_lnc2`](gdn_tkg_bh24_k128_v128_lnc2/) | `nkilib.experimental.gdn.gdn_tkg` (fused decode) | LNC=2, 1 rank | 76.05 µs | 36.18 µs | **2.102x (−52.4%)** | 7 |

Latency is `total_exec_time` from a device profile (`neuron-explorer view --output-format
summary-json`), median of several captures. In both cases the optimized NEFF's outputs are
within the same error of the torch golden as the unedited NEFF, to every printed digit — the
edits moved data and instructions, never the arithmetic.

## What is in each subfolder

```
baseline.neff          the NEFF as the compiler emitted it
optimized.neff         the NEFF after the loop, byte-comparable with the baseline
manifest.json          IO names, shapes, dtypes, tolerances, the error limit the verifier gates on
nki/                   the original NKI source of the kernel, and the torch reference
tools/                 the verifier and the profiler, exactly as the loop ran them
measurements/          profile + verify reports for baseline, optimized, and every iteration
logs/
  improvement_summary.log  one line per merged edit, oldest first - read this first
  merged_edits.log         the full message of every merged edit
  notes/iter-N.md          the agent's own findings per iteration, including refutations
  reviews/iter-N.md        the independent review of each iteration
  history.jsonl            metric per iteration, as the loop recorded it
  loop_run.log             the loop's own console output
  autohelix.yaml           the goal, constraints, metric and reviewer prompt that produced all this
```

## Reproducing a measurement

The tools need the input and golden tensors, which are not copied here (about 100 MB for the
prefill case). They live with the loop repositories:

```bash
# prefill: /home/ubuntu/workspace/gdn-neff-opt       (bench/inputs, bench/golden)
# decode:  /home/ubuntu/workspace/gdn-tkg-neff-opt
cd /home/ubuntu/workspace/gdn-tkg-neff-opt && source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2/optimized.neff kernel.neff
python tools/verify.py --runs 2      # outputs against the golden, on the device
python tools/profile.py --repeats 5  # total_exec_time
```

Those tensors can also be regenerated from scratch with
`/home/ubuntu/workspace/gdn-bench` (`compile_gdn.py <case>`), which compiles the NEFF, saves the
seeded inputs and computes the goldens.

## Reading the diff between the two NEFFs

```bash
python /home/ubuntu/workspace/neff_doc/tools/neff_diff.py baseline.neff optimized.neff
```

It reports which package members, which instruction slots and which decoded fields differ
(chapter 13.6). For the decode case the two NEFFs differ in both subgraphs, `sg00` and `sg01`.

## What the two cases show

The prefill kernel had **no saturated engine** — it was bound by a dependency chain through a
DMA queue, and six iterations of scheduling and placement work bought 5.1%. The decode kernel
had the Activation engine busy **99.5%** of the execution, and the same classes of edit — moving
work onto idle engines, deleting redundant copies, tightening semaphore waits onto the real
producer, reordering slots — bought **2.1x**. Same tools, same rules, two orders of difference
in payoff: what a NEFF edit is worth depends on whether the kernel is bounded by an engine or by
a dependency chain.

Both loops' `notes/` also record what did *not* work, with the measurement that closed it; those
refutations are as useful as the wins when planning the next edit.

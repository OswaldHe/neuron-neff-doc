# Experiments: optimizing compiled NEFFs, end to end

Twelve kernels were made faster by **editing the compiled NEFF only** — no source change, no
recompile — each under an automated loop that had to pass a device verifier every iteration and
was scored on a device profile. One subfolder per configuration.

Totals: **12 loops, 79 iterations, 0 rejected, 91 hours, $1,374.** Gains from **1.05x to 4.92x**.
Every optimized NEFF still matches its torch reference within the limit the loop was given, and
in 10 of 12 the error is unchanged from the baseline to every printed digit.

| configuration | kernel | baseline | optimized | gain | iters |
|---|---|---|---|---|---|
| [gdn_block_tkg_b1_h128_vh4_lnc1](gdn_block_tkg_b1_h128_vh4_lnc1/) | `gdn_block_tkg` | 218.2 µs | 44.4 µs | **4.92x** | 7 |
| [gdn_tkg_bh24_k128_v128_lnc1](gdn_tkg_bh24_k128_v128_lnc1/) | `gdn_tkg` | 153.5 | 60.9 | **2.52x** | 7 |
| [gdn_tkg_bh24_k128_v128_lnc2](gdn_tkg_bh24_k128_v128_lnc2/) | `gdn_tkg` (LNC=2) | 76.1 | 36.2 | **2.10x** | 7 |
| [ssd_mamba2_370m_s2048_lnc1](ssd_mamba2_370m_s2048_lnc1/) | `ssd` | 5248.9 | 2905.2 | **1.81x** | 7 |
| [ssd_mamba2_1b3_s2048_lnc1](ssd_mamba2_1b3_s2048_lnc1/) | `ssd` | 7154.9 | 4251.0 | **1.68x** | 7 |
| [gdn_cte_b32_s64_d128_lnc1](gdn_cte_b32_s64_d128_lnc1/) | `gdn_cte` | 1185.0 | 835.8 | **1.42x** | 3* |
| [gdn_cte_b1_s128_d128_lnc1](gdn_cte_b1_s128_d128_lnc1/) | `gdn_cte` | 48.5 | 37.6 | **1.29x** | 7 |
| [gdn_cte_b1_s256_d128_lnc1](gdn_cte_b1_s256_d128_lnc1/) | `gdn_cte` | 67.5 | 53.2 | **1.27x** | 7 |
| [selective_scan_b1_c768_s1024_n16_bf16_lnc1](selective_scan_b1_c768_s1024_n16_bf16_lnc1/) | `selective_scan` | 602.6 | 497.9 | **1.21x** | 7 |
| [selective_scan_b1_c2560_s4096_n16_bf16_lnc1](selective_scan_b1_c2560_s4096_n16_bf16_lnc1/) | `selective_scan` | 7920.0 | 6604.0 | **1.20x** | 7 |
| [gdn_cte_b32_s128_d128_lnc1](gdn_cte_b32_s128_d128_lnc1/) | `gdn_cte` | 1351.4 | 1166.3 | **1.16x** | 7 |
| [gdn_cte_b32_s2304_d128_lnc1](gdn_cte_b32_s2304_d128_lnc1/) | `gdn_cte` | 14779.0 | 14022.0 | **1.05x** | 6 |

\* `gdn_cte_b32_s64_d128` stopped early: the agent backend returned an API error on iteration 4
and the harness treated it as fatal. Its 1.42x came from 3 iterations and 4 remained unspent; the
loop repository is intact, so `autohelix run` would continue from iteration 4.

Latency is `total_exec_time` from a device profile (`neuron-explorer view --output-format
summary-json`), median of 3–7 captures; run-to-run spread was 0.00–0.03% on the large kernels and
up to 1.2% on the smallest.

## Where the kernels come from

| family | source | toolchain |
|---|---|---|
| `gdn_cte`, `gdn_tkg`, `gdn_block_tkg` | nkilib `experimental/gdn`, vendored in `../../gdn-bench/vendor/gdn/` (the installed wheel has no `gdn`) | NKI 0.6.0, neuronx-cc 2.27.5334 |
| `selective_scan`, `ssd` | nkilib `experimental/scan` at nki-library commit `92d11f6`, vendored in `../../gdn-bench/vendor/commit_92d11f6/`; byte-identical to the installed wheel's copies | same |

Inputs, torch references and tolerances for every case are taken from the kernels' own
integration tests, so a pass here means what a pass there means. Each baseline NEFF is what
`@nki.jit` produces at default settings — neuronx-cc `--optlevel 2` with the coloring allocator,
and NKI's own scheduling and allocation (the frontend disables the backend's scheduler and
allocator, as it does for every nkilib kernel).

## What these results say

**The biggest wins came from data movement, not from arithmetic.** The recurring finds, across
families: SBUF→DRAM→SBUF staging round-trips that are really layout reshapes and can be made
SBUF→SBUF or skipped; DGE DMA work parked on a busy engine that another engine can issue;
semaphore waits stricter than the data dependency; and whole instructions that re-execute an
earlier result. Nothing in the 79 iterations needed a numerical change to find speed.

**Engine saturation says where to look, not how much is there.** My early guess — a saturated
engine predicts a large win — does not survive twelve data points:

| baseline profile | gain |
|---|---|
| `selective_scan` 2560x4096: DVE **99%** busy | 1.20x |
| `selective_scan` 768x1024: DVE **98%** busy | 1.21x |
| `gdn_tkg` LNC=2: Activation **100%** busy | 2.10x |
| `gdn_tkg` LNC=1: Activation 93% | 2.52x |
| `ssd` 370m / 1.3b: Activation 90% / 88% | 1.81x / 1.68x |
| `gdn_block_tkg`: nothing above **24%** (DMA) | **4.92x** |
| `gdn_cte` b32 s2304: Activation 61% | 1.05x |

A saturated engine is only worth attacking if its work can be moved or deleted. On `gdn_tkg` it
could: activation-table loads, casts and scale-pointer ops moved onto idle DVE and Pool. On
`selective_scan` it could not: the 99%-busy DVE *is* the scan, so after four iterations of
shaving around it the loop ran out of levers and the last three iterations merged nothing. And an
idle-looking profile can hide the largest win of all — `gdn_block_tkg` had no engine above 24%
because it was waiting on DMA round-trips, and removing them compressed the execution 4.9x until
Activation became the new limit at 80%.

**A dependency-bound kernel can also be nearly irreducible.** `gdn_cte` at batch 32 × seq 2304
gave 5.1% over six iterations, and its notes close every ms-scale lever with a measurement: all
engines are idle together for only 1.3 ms of 14.8 ms, in 15183 gaps whose largest is 3.9 µs.

**The reviewer earned its place.** From iteration 4 of the first loop onward, a second model
(Sonnet 5.5) reviewed each iteration and handed the next agent a ranked list of candidate edits
with measured ceilings. It repeatedly refused to score sub-noise wins, and corrected the agent's
own conclusions with numbers — on the first loop it showed that the lever the agent wanted next
had a 112 µs ceiling rather than the ≥1 ms the agent assumed.

## What is in each subfolder

```
baseline.neff          the NEFF as the compiler emitted it
optimized.neff         the NEFF after the loop (diff them with ../../tools/neff_diff.py)
nki/                   the kernel's source as compiled, and the torch reference for the golden
tools/                 the verifier and the profiler, exactly as the loop ran them
manifest.json          IO names, shapes, dtypes, tolerances, the error limit the loop gated on
measurements/          profile + verify reports for baseline, optimized, and every iteration
logs/
  improvement_summary.log  one line per iteration: latency, cumulative speed-up, the edit
                           (a few iterations merged without changing the NEFF at all -- the
                            agent explored, kept nothing, and the metric moved only by noise)
  merged_edits.log         the full message of every merged edit
  notes/iter-N.md          the agent's findings, including refuted levers and what closed them
  reviews/iter-N.md        the independent review of each iteration
  history.jsonl            metric and merged commit per iteration, as the loop recorded it
  loop_run.log             the loop's own console output
  autohelix.yaml           the goal, constraints, metric and reviewer prompt behind it all
README.md              generated from the measurements above
```

## Rules every loop ran under

* **Only `kernel.neff` was editable.** No kernel source is present in a loop repository and
  recompiling was not the task; every other file is reverted before an iteration is judged.
* **Correctness was the gate, every iteration.** `tools/verify.py` runs the NEFF on the device
  with fixed inputs and requires every output within `max_abs_err_limit` of the torch golden —
  set at 1.2x the unedited NEFF's own error, so an edit that changes the arithmetic fails. It
  also checks the file is well formed, the device ran it without hanging, and no output is NaN.
* **The launch shape was pinned.** `tools/launch_rules.py` requires the baseline's subgraph count
  (LNC degree), `num_tpb == 1`, no collectives and no replica groups, and an unchanged interface:
  a win must come from the kernel, not from more cores or more ranks. Checked both ways — it
  accepts the LNC=2 NEFF and rejects a 4-rank collective one.
* **The metric was the device's own clock**, median of several captures, with a 2% regression
  gate.

Honest limits: the verifier uses **one** fixed input set and a single run per iteration, so a
data-dependent bug or a rare race could pass it (the agents often ran `--runs 2/3` themselves,
and the reviewer was asked to judge that risk); `ssd`'s two loops moved the error in the third
significant digit (1.88e-07 → 2.06e-07 against a 2.3e-07 limit), i.e. they did reassociate fp32
arithmetic slightly, while the other ten left it bit-identical; and all results are for one
device generation and one compiler version.

## Reproducing

The input and golden tensors are not copied here (about 100 MB for the largest case). Each
subfolder's README gives the exact commands; in outline:

```bash
cd /home/ubuntu/workspace/gdn-loops/<case>        # or gdn-neff-opt / gdn-tkg-neff-opt
source /home/ubuntu/workspace/.venv/bin/activate
cp /home/ubuntu/workspace/neff_doc/experiments/<config>/optimized.neff kernel.neff
python tools/verify.py --runs 2                   # outputs against the golden, on the device
python tools/profile.py --repeats 5               # total_exec_time
```

Or rebuild a case from scratch, inputs and goldens included, with
`/home/ubuntu/workspace/gdn-bench/tools/compile_gdn.py <case>`, and re-create its loop with
`prepare_loop.py <case>`.

## Not attempted, and why

The DeepSeek V3.2 components (`mla/deepseek`, `sparse_attention_indexer`, `deepseek_v32`) are
**Trainium3-only**: every one of their tests excludes TRN1/TRN2, and compiling one here fails in
the MX-fp8 matmul (`NCC_IBIR441`, PSUM alignment) while the same kernel compiles cleanly for
trn3. Without a device to run them on there is no verifier and no profile, so no loop. The
harness for them is built and waiting (`gdn-bench/tools/{exp_loader,ds32_kernels,mx_test_helpers}.py`).

`ring_attention_spmd_fwd` is TRN2-capable but context-parallel across 2–4 **ranks**, which needs
per-rank inputs in the verifier, per-rank NTFFs in the profiler, and a launch rule that pins
`replica_groups` instead of forbidding them. That extension is scoped but not written.

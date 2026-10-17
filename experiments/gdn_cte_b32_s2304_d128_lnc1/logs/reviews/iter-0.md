# Review — iter-0 (baseline)

## 1. What changed in the NEFF

**Nothing.** `kernel.neff` is byte-identical to `baseline/kernel.neff`:

```
neff_diff.py baseline/kernel.neff kernel.neff
  == header   identical
  == members: 34 in A, 34 in B; 34 identical, 0 changed
  payloads are identical
```

`md5sum` agrees (`8d6a7f55…` both), and `profile_latest.json` records the same
`neff_sha256 eb445f7e…` as `profile_baseline.json`. No slot, no field, no member was touched.

This is consistent with the notes — because there are no notes. `.autohelix/notes/` is empty,
`notes/` holds only the template `README.md`, and there is no `commit_summary.txt`. The run log
contains two lines, `baseline capture started` / `baseline: total_exec_time_us: 14783.3`.
iter-0 is the baseline-capture iteration; no optimisation was attempted. There is nothing to
reconcile, and nothing to credit.

Also note `verify_latest.json` does not exist — only `profile_latest.json` and `stdout.txt` were
captured in `.autohelix/observations/iter-0/`. The verifier was not re-run this iteration (the
baseline report in `baseline/verify_baseline.json` still stands).

## 2. Is the latency change real?

No — and there is no change to be real. It is noise on an unmodified binary.

| | median | min | max | spread |
|---|---|---|---|---|
| `baseline/profile_baseline.json` | 14778.976 | 14778.273 | 14779.613 | 0.009 % |
| `profile_latest.json` (iter-0) | 14783.312 | 14783.151 | 14789.182 | 0.041 % |

Delta of medians: **+4.34 µs = +0.029 %**, i.e. the same NEFF measured twice. Both captures sit
inside the stated ~0.1 % run-to-run band and far under the ~0.5 % significance threshold. My own
capture (`--repeats 1`) landed at 14786.458 µs (+0.05 %), bracketing the two. Every per-engine
number also matches within ±0.3 %.

So: the loop is not yet chasing noise, because it has not yet made a move. **The thing to guard
against from iter-1 on is exactly this**: the inter-capture offset on an *identical* binary is
+0.03 %, so any iteration reporting a gain below ~0.5 % (≈ 74 µs) has measured nothing. Require
3+ repeats and demand the median improve by more than the spread.

## 3. Is the edit safe beyond the verifier?

There is no edit, so nothing was loosened. For the record, the baseline is clean and I
established the reference points a future iteration will be judged against:

**Synchronisation.** `semaphores.py` on a fresh unpack of `kernel.neff` shows
`max_wait <= total_inc` for every semaphore, and equality for nearly all — no wait exceeds its
supply. Baseline supply:

| semaphore block | source | total |
|---|---|---|
| S[3] | Pool (`TENSOR_SCALAR_AFFINE_SELECT`) | 2306 |
| S[4] | Activation (`ACTIVATE` 5073, `TENSOR_SCALAR_ARITH_OP` 6282, `CAST` 2574) | 13929 |
| S[5] | PE (`MATMUL`) | 14884 |
| S[6] | DVE (`MEMSET`/`CAST`/`COPY`/`TENSOR_TENSOR`/`TENSOR_SCALAR`) | 21688 |
| S[8]–S[23] | `dge:Pool` → `qGpSimdDynamic` | 3136 |
| S[24]–S[39] | `dma:qSPSpillReload0` (queued) | 8288 |
| S[40]–S[55] | `dge:Activation` → `qScalarDynamicHW` | 2784 |
| S[56]–S[71] | `dge:SP` → `qSyncDynamicHW` | 2464 |

**Numeric headroom — the real hazard in this repo.** `output_1` is already at
`max_abs_err 6.841e-04`, **86 % of the 8e-4 limit** (`output_0` is at 17 %). There are only
1.2e-4 of slack. Any edit that changes arithmetic — chapter 14's lever 5 (cheaper dtypes), or
dropping a correction term — will very likely fail the constraint on `output_1`, and an edit that
*passes* at 7.9e-4 is one input away from failing. **Prefer edits that are bit-exact by
construction** (relocation, reordering, descriptor coalescing) over anything that touches a
dtype or an operation. Future iterations should run `verify.py --runs N` (bit-identical across
runs) rather than a single run, since a one-shot pass cannot distinguish a correct edit from a
race that happened to win.

## 4. Most promising next edit

**Move the 2784 `PSEUDO_DMA_DIRECT2D` slots off the Activation engine stream
(`payload/sg00/Activation0.bin`) onto the Pool and/or SP streams.**

This is chapter 14 **lever 3, "move work off the busy engine"**, applied not to arithmetic but to
spill-descriptor generation.

### The evidence

Activation is the busiest engine (8937 µs union-busy = 60.4 % of the 14801 µs span). I built the
real dependency DAG from `prof/Flow.parquet` (205k `FLOW_DEPENDENCE` + `trigger` + `ORDERED`
edges, plus same-engine program-order edges) and took the longest path: **14660 µs of the 14801 µs
span**, so the critical path is essentially the whole execution. Its composition:

| engine | opcode | µs on CP | % CP |
|---|---|---|---|
| Tensor | MATMUL | 3263.8 | 22.3 |
| Scalar | ACTIVATE | 2440.1 | 16.6 |
| Scalar | TENSOR_SCALAR | 1995.2 | 13.6 |
| **Scalar** | **DMA_DIRECT2D** | **1242.9** | **8.5** |
| Tensor | LDWEIGHTS | 1196.4 | 8.2 |
| Vector | EVENT_SEMAPHORE (pure wait) | 1047.2 | 7.1 |
| Vector | TENSOR_TENSOR | 709.3 | 4.8 |
| DMA | DMA_PACKET | 586.6 | 4.0 |
| Sync | DMA_DIRECT2D | 431.0 | 2.9 |
| GpSimd | DMA_DIRECT2D | 190.6 | 1.3 |

Those `DMA_DIRECT2D` on Activation are **not arithmetic**. They are the DGE instructions that
generate descriptors for the compiler's SBUF spill-saves: the per-instruction table attributes
exactly `160,432,128` bytes of `spill_save_bytes` to them — every spill-save byte in the
`summary` block. 1152 come from `gdn_cte.py:483` and 1152 from `:487`, the rest from `:317/:367/:368`.

Two measurements say this is on the critical path for the right reason:

1. **Exclusive-busy.** During the 1654 µs Activation spends on these, **622.8 µs (37.6 %) has no
   other engine busy at all** — the whole machine is stalled behind the Activation sequencer.
   Ranked by exclusive-busy time, `Scalar/DMA_DIRECT2D` is third overall (622.8 µs, 4.2 % of span)
   behind only `Scalar/ACTIVATE` (929 µs) and `Scalar/TENSOR_SCALAR` (770 µs) — both of which are
   the actual GDN math (SOFTPLUS gate, scale/bias).
2. **It blocks Activation, not the DMA.** Of the 1242.9 µs on the critical path, **1030.6 µs
   (83 %) is "mid-run"**: the path arrives from a Scalar instruction and leaves to a Scalar
   instruction. The path does *not* go through the triggered DMA. So the cost is the Activation
   sequencer being occupied while the next `ACTIVATE`/`TENSOR_SCALAR` waits behind it in program
   order — which is precisely the cost another engine can absorb. (Only ~144 µs leaves to
   Sync/GpSimd, where DMA completion is what matters.)

The per-descriptor cost is near-constant and size-independent (`:483` = 609 ns for 32 KB;
`:367` = 554 ns for 236 KB), so this is fixed issue overhead, not data movement.

### Why this edit is unusually safe

I decoded all 2784 slots in `Activation0.bin` with `isa.py`:

- **`events_update_mode=NONE` on all 2784** (`SEM_INC_COMPLETE`: 0, `events_update_idx`: 0).
  They increment **no** engine completion semaphore — confirmed independently by `semaphores.py`,
  which lists only `ACTIVATE`/`TENSOR_SCALAR_ARITH_OP`/`CAST` as S[4] sources. **Removing them
  from the Activation stream needs zero renumbering of any S[4] `events_semaphore_value`
  anywhere in the NEFF.** That eliminates the hard part of chapter 10 R2/R3. Symmetrically,
  inserting them into `Pool0.bin` does not perturb S[3] (whose only source is
  `TENSOR_SCALAR_AFFINE_SELECT`).
- **No register-sourced addresses**: `addr_reg_regnum` appears 0 times; all use
  `src_start_addr_addr_immediate` / `dst_start_addr_addr_immediate`, `src_shape_reg=0`,
  `dst_shape_reg=0`, `compute_op=NONE`. So the chapter 10.5 Pool `NRT_EXEC_OOB` failure mode
  (DGE reading address registers before `ALU_OP`/`TENSOR_LOAD` wrote them) **cannot apply** —
  there is no same-engine register state to carry along.
- **Consumers already synchronise cross-engine.** 2435 instructions wait on S[40]–S[55] from
  outside the DGE set: 622 `EVENT_SEMAPHORE` on Scalar, 577 on GpSimd, 577 on Sync, plus 476
  Sync `DMA_DIRECT2D` and 155 Sync `WRITE`. Nothing relies on Activation program order for
  these spills, so no consumer wait needs re-pointing (chapter 10 R7 is satisfied as-is).
- **The opcode already runs on the target engines in this very NEFF**: `Pool0.bin` has 3136
  `PSEUDO_DMA_DIRECT2D` slots and `SP0.bin` has 2464. This is a relocation between proven hosts,
  not a new capability.
- Identical bytes, addresses, `semaphore` and `sem_increment` ⇒ **bit-exact by construction**,
  which matters given the 86 %-of-limit headroom on `output_1`.

### Concretely

- **Engine / member**: `payload/sg00/Activation0.bin` (30789 slots) → `payload/sg00/Pool0.bin`
  (7584 slots) and/or `payload/sg00/SP0.bin` (17364 slots).
- **Slots**: the 2784 whose opcode is `PSEUDO_DMA_DIRECT2D`. They begin `78, 79, 80, 140, 141,
  143, 144, 145, …` and end `…30757, 30759, 30761, 30763, 30765`. Regenerate the full list with
  `isa.py payload/sg00/Activation0.bin | awk '/DMA_DIRECT2D/{print $1}'`.
  `compiler_pc` in `prof/Instruction.parquet` equals the slot index, so the profile maps straight
  onto the stream.
- **Fields to carry over unchanged**: `semaphore` (40–55), `sem_increment` (16),
  `events_wait_mode` / `events_wait_idx` / `events_semaphore_value`, `dge_op`,
  `src_start_addr_addr_immediate`, `src_step_elem`, `src_num_elem`, `src_elem_size`, and the
  `dst_*` equivalents, `in_dtype`, `out_dtype`. Nothing needs recomputing. The waits are on
  global semaphores (mostly `S[6]` Vector, some `S[4]` Scalar, some `S[40..55]` queue flow
  control) and are valid verbatim from any engine.
- **Two constraints on placement**: preserve the moved slots' *relative* order among themselves
  (they round-robin S[40]–S[55] with +16 and some wait on `S[4x] >= 16` as queue flow control),
  and insert them at points in the host stream where their own waits are already satisfied.

### Expected latency change

The hard bound is the 1030.6 µs of mid-run critical-path occupancy (7.0 % of span). Against that,
the host engines absorb new work: Pool 2899 → ~3875 µs (26 % of span) and SP 2773 → ~3607 µs
(24 %) if split evenly — both stay far below Activation's new ~7400 µs, so the receiving engines
do not become the new bottleneck. Some of the moved work will still land on the critical path, so
realistically expect **0.4–0.7 ms, i.e. 3–5 %** → **~14.1–14.4 ms**. That is 6–10× the 0.5 %
noise floor, so a real win will be unambiguous.

**De-risk it first.** Move only the 1152 slots from `gdn_cte.py:483` (the single largest group;
312 µs exclusive, ~350 µs on the CP) to `Pool0.bin`. Expect ~1.5–2.5 %, still above noise, at
40 % of the work. This also settles the one genuine open question cheaply: the three DGE queues
are per-engine hardware (`qScalarDynamicHW` idx 0 / `qGpSimdDynamic` idx 1 / `qSyncDynamicHW`
idx 2 in `DmaQueuesInfo`, all sharing `engine_idx 79`, `axi_port 0`, `dram_channel 2`), so it
must be confirmed on device that a Pool-issued DGE may signal a semaphore in Activation's block
S[40]–S[55]. The field is explicit per instruction and Sync DGEs already *wait* on
`qScalarDynamicHW`, so this should hold — but test it on one slot before doing 1152. If the
hardware rejects it, do **not** re-point 2435 consumer waits; switch to the fallback instead.

### Fallback, if the move is blocked

Chapter 14 **lever 4, "fewer, larger instructions"**. Lines `:483`/`:487` spend 2304 descriptors
to move 75.5 MB; lines `:367`/`:368` move the same 75.5 MB in 320. Since per-descriptor cost is
flat at ~600 ns, coalescing `:483`/`:487` to `:367` granularity would recover ~1.2 ms of
Activation time, and it also shortens the ~1.4 ms the critical path spends in Vector/Sync
`EVENT_SEMAPHORE` waiting on spill DMA completion. Structurally available: those slots use
`src_num_elem=[4,16]` / `dst_num_elem=[4,16]` — only 2 of the 4 pattern dimensions — so there is
room for a third. Riskier, though: the destination addresses are irregular (`dst` stride is 512
for 992 of 1151 consecutive pairs, but also 1110528/850432/815616), so merged groups must be
chosen carefully, and a wrong address gives `NRT_EXEC_OOB` or silently wrong data.

### What is *not* a lever here

`MATMUL` is the largest single item on the critical path (3263.8 µs, 22.3 %), but it is
irreducible at the NEFF level, exactly as chapter 14 states: all 17184 are
`performance_mode = NONE` (bf16), durations are tight and small (mean 290.6 ns, 206–318 ns), and
the `LDWEIGHTS:MATMUL` ratio is 0.853 — nearly one weight reload per matmul. That reflects the
compiler's tiling, which no NEFF field changes. Likewise the 1047 µs of Vector `EVENT_SEMAPHORE`
on the critical path is a *symptom*, not removable work: 552 of those 695 waits are on
`qSyncDynamicHW` DGE semaphores and 124 on `qSyncSpillReload0`, i.e. DVE blocked on spill traffic
— which is the same root cause the recommended edit attacks, and which only coalescing (not
relocation) will shrink further.

### The bigger picture

Spill machinery, not arithmetic, is the dominant removable cost: 302 MB of spill traffic
(`spill_save_bytes` 160.4 MB + `spill_reload_bytes` 141.6 MB) against 398.9 MB of total HBM
traffic, and ~3.2 ms of the 14.66 ms critical path (22 %) is spent generating spill descriptors
or waiting for spill DMAs. The sequence worth working through is: (1) move the Activation DGE off
the bottleneck engine, (2) coalesce `:483`/`:487`, (3) consider the same move for the 431 µs of
Sync DGE on the critical path. Only after that does the remaining time look genuinely
irreducible without recompiling.

# iter-2 — 95.3 → 83.2 us (1.15x on top of iter-1; 2.62x vs the recorded 218.17 baseline)

Shipped `kernel.neff` sha256 `ae5f1a04b12a28b53cff9509b85af07ca42bb84ee00f6294064dd5dcd64586ed`
(132170 bytes). `total_exec_time_us` of the shipped file, eight independent 5-repeat runs:
**83.35 / 82.74 / 83.30 / 81.83 / 83.43 / 82.83 / 83.11 / 83.61** → median **83.2 us**, and the
last run had spread 0.55%. My own re-measure of the *iter-1* file at the start of this session:
**95.27 / 95.62**. See §0: the machine drifts by ±3 us between captures, so **only trust
back-to-back A/B comparisons**, never two numbers taken minutes apart.

`verify.py` passes with `max_abs_err` **exactly equal to the baseline** on all three outputs
(2.999e-03 / 1.928e-03 / 6.033e-04) after every single edit: everything shipped is arithmetic
bit-identical. `launch_rules.py` passes. `semaphores.py` reports no wait exceeding its supply.

Changed members vs iter-1: `sg00/Activation0.bin` (15 slots), `sg00/Pool0.bin` (19),
`sg00/SP0.bin` (14). No slot inserted or deleted, no JSON changed. Engine busy:
Pool 35.8→26.7, SP 27.1→19.8, Activation 41.8→39.7, DMA 52.3→29.1 (vs the iter-0 baseline).

---

## 0. Measurement hygiene — read this first

**The device drifts on a scale of minutes, by ±3 us, in a way that looks like a bimodal
distribution.** Same NEFF, five consecutive `--repeats 5` runs gave medians 82.94, 85.96, 86.07,
85.92, 85.78. I nearly shipped a regression because of it (and nearly threw away iter-2's best
state). Rules that worked:

* A single 5-repeat median is worth ±2 us, no better. **Never** compare a number you took
  10 minutes ago against one you just took.
* To compare two candidates, alternate them back-to-back in one loop:
  `for r in 1 2; do for c in A B; do cp $c kernel.neff; profile --repeats 5; done; done`.
  That is how the V5 regression (§4) and the winner were decided.
* Keep every accepted state as a file (`/tmp/ckptN.neff` plus its unpacked `work/`) so you can
  re-measure it against a later candidate.

## 1. The two findings that carried the iteration

### Finding A — a DGE DMA instruction costs ~0.55 us of engine time no matter how small it is

iter-1 left behind four `PSEUDO_DMA_DIRECT2D` slots shrunk to `num_elem=[1,1]` (one 2-byte
transfer). **They still take 0.50–0.67 us on the engine**, the same as a 128-descriptor transfer
(0.60–0.68) and nearly the same as a 512-descriptor one (0.97). Fit over this NEFF:

> **engine time of a DGE DMA ≈ 0.49 us + 1.3 ns × descriptors.**

So the cost is ~90% fixed overhead. 89 DGE DMAs in the kernel ⇒ ~50 us of pure issue time, and
Pool/SP spend most of their busy time just *issuing* DMAs. **The only way to make an
issue-bound DGE chain shorter is to have fewer instructions in it**, not smaller ones.

### Finding B — `SEM_ADD_IMM_COMPLETE` works, and it makes a DMA removable for 0.03 us

`UPDATE_MODE` 21 (`SEM_ADD_IMM_COMPLETE`, `S[i] += v @complete`) is in the vendor enum but
**never used by the compiler** (chapter 6.4: "the compiled kernels use only update mode 19").
**It works on Trainium2, verified on the device, on Pool, SP and Activation.** That unlocks the
general move:

> **To delete a DMA whose data someone else already produces, overwrite its slot with an
> `EVENT_SEMAPHORE` that waits for the real producer and does
> `SEM_ADD_IMM_COMPLETE(own_semaphore, 16)`.** Every wait in the program keeps working untouched,
> the engine pays 0.03 us instead of 0.55, and the semaphore fires at the *instant* the wait
> passes instead of 2–7 us later when a DMA completion would have trickled in.

`semaphores.py` already understands mode 21 and counts it correctly (`'Pool:EVENT_SEMAPHORE': 16`
shows up as a source), so the accounting check still protects you.

**The one trap: there is only one value field** (chapter 6.4). An instruction that waits
`S[a] >= v` and also does an immediate update uses the *same* `v` for both. So:
* if the needed wait value is ≥ 16 you can add that value (adding 32 or 48 is harmless — every
  wait is `>=`, so overshooting a semaphore never breaks anything, and `semaphores.py` confirms
  `total_inc >= max_wait`);
* if the needed wait value is < 16 you cannot add 16. Either find a later-but-implied wait whose
  value is ≥ 16 (§3 edit D does exactly this), or re-point the consumer to the smaller value.
  I hit this twice and `semaphores.py` caught it both times ("wait exceeds increments").

## 2. How I found where to attack (reusable tooling)

The decisive tool was a **backwards critical-path walk with semaphore attribution**. For each
instruction: `gate = max(previous instruction's end on the same engine, time its wait value was
reached)`, then jump to whichever won, and recurse from the last instruction. It prints a chain
with ~0.02 us slack at every step, i.e. the real serial path.

Ingredients (all in `prof/*.parquet` from `profile.py --parquet prof`):
* `SemaphoreUpdate.parquet` — `id` is a string like `'S[59] (qSyncDynamicHW)'`, parse the number
  out. `value` is the running total, so "when did `S[i] >= v`" is
  `g[g.value>=v].iloc[0].timestamp`.
* `Instruction.parquet` — join to the stream by `(engine, compiler_pc)`; engine names are
  `Tensor/Scalar/GpSimd/Vector/Sync` for PE/Activation/Pool/DVE/SP.
* `DmaPacketAggregated.parquet` — one row per **queued** transfer with `block_id`,
  `semaphore_id`, `variable`, `source`, `dest` and, usefully, `trigger` = the triggering
  instruction's timestamp in ns. That is how you link a queued DMA back to its `PSEUDO_DMATRIGGER`.
* `Flow.parquet` exists but its edges do **not** include semaphore dependencies, so a walk built
  on it leaves 20–35 us gaps. Don't use it alone.

`def.json → dma_queue` is worth reading once: only **act (S[72..87]), pool (S[8..23]) and sp
(S[56..71] + spill S[24..39] + IO S[40..55])** have DGE queues. **PE and DVE have none**, so you
cannot move a DMA onto them. `qPoolDynamic` has
`dynamic_dma: {SB_scratch_partition_size: 16384}` — descriptors are built in 16 KB of SBUF
scratch; this is probably what the queue-depth waits protect (see §5).

## 3. The edits that shipped, in order, each measured

| # | what | median after |
|---|---|---|
| A | **16 duplicate DGE DMAs → ES + `ADD_IMM`.** iter-1's writer/reader rewrite left 16 pairs performing the *byte-identical* SB→SB transfer. Replaced the second of each pair. Pool 34,36,38,40,43,51,53,55 (twins SP 5,6,7,8,9,12,17,22) wait `s56/57/58/59/60/61/66/71 >= 16`; SP 13,15,24,26,44,46,52,55 (twins Activation 35,39,40,41,52,55,58,59) wait `s72..s79 >= 16` (for SP 24/26/44/46/52/55 the twin's wait already sits on the preceding ES, so those wait on their own `sem >= 16`). Each adds 16 to its own semaphore. | **89.20** |
| B | **5 signal-only DGE DMAs on SP → ES + `ADD_IMM`.** SP 62,64,66,68 and 75 are iter-1's `num_elem=[1,1]` dummies that exist purely to keep semaphore counts. Waits are `s57/58/59/60 >= 32` and `s57 >= 48`, so the add value is 32 / 48 — overshooting to 64 / 80 is fine. Plus: **dropped the wait on SP 50**, the trigger of dummy queued transfer 12 (`s32>=16` was a vestigial dependency on queued blk8; the queue is serial so ordering is kept by construction). | **86.74** |
| C | **8 Activation DGE DMAs → compute `CAST`.** Activation 35,39,40,41,52,55,58,59 are per-partition `SB[p][X] -> SB[p][Y]` bf16→bf16 copies. A DGE DMA needs 0.55 us to issue **and 2.9–7.1 us for its 16th completion increment to land**; a 128-channel × 1-element `CAST` does the same thing in 0.25 us and is visible the moment it completes. Cloned Activation slot 23 (`CAST`, `S4D4_TR`, `num_active_channels=128`, SBUF→SBUF) and patched addresses, `num_elem=[1,1,1,1]`, dtypes, and the events block to `SEM_ADD_IMM_COMPLETE(own_sem, wait_value)`. | **~85.9** (−0.8) |
| D | (part of C) Activation 35/39's waits were `s4>=13` / `s4>=15`, which would only add 13/15 while consumers need 16. **s4 producer #18 is Activation slot 34, which precedes both slots, so waiting `s4>=18` is implied by program order and costs nothing** — and lets the add be 18. (Producer→slot map: #13=slot 28, #15=slot 30, #16=slot 31, #18=slot 34.) | — |
| E | **Merged the 8-tile V3 (`conv_weight`) weight load into one DMA.** Pool slots 6,7,21,22,23,24,25,26 each loaded one 128×16 B tile (`src V3+2048*j -> dst SB+224768-768*j`). One DMA with a dim-1 pair does all eight: `src_step_elem=[16,-2048]`, `src_num_elem=[128,8]` from `V3+14336`; `dst_step_elem=[262144,768]`, `dst_num_elem=[128,8]` to `SB+219392`. The other seven slots became no-wait ES adding 16 to their own semaphore, and the ten consumers of those tiles (Activation 4,6,8,10,12,15,17 and Pool 44,46,48) were re-pointed to `s13>=16`, the merged transfer. | **83.0–83.4** |

**Verified positive and reusable: a DGE DMA accepts a negative dim-1 `src_step_elem` reading
DRAM** (`[16,-2048]`, `num=[128,8]`), i.e. source blocks walked backwards while the SBUF
destination walks forwards. That is how a reversed tile layout gets merged at all — the compiler
lays K-tiles out in *descending* SBUF order against ascending DRAM offsets in every group here.

## 4. Refuted — measured, reverted, do not redo

* **Bypassing the output_0 tail round trip made it *slower*.** `output_0` goes
  `SB(part 32, off 18184, 1024 B) → DRAM tmpbuf → 4×256 B back into SB → 4 IO DMAs → output_0`.
  I pointed the four `qSPIO0` descriptors straight at `SB+8406792+256k` (correct, verified
  bit-exact) and dropped the eight now-vestigial waits on SP 77–84: **86.74 → 88.76**. Then I also
  removed transfers 15–19 entirely (SP 76–80 → plain ES): **88.0, bimodal 83.9/88.8**. Both
  reverted. Reason, from the trace: the tail is **not** bounded by the output_0 path at all — it
  is bounded by Pool 66, the 256 KB `output_2` DRAM write that starts at ~90 us. Firing the
  output_0 stores ~3 us earlier only makes them contend with it. **The ~3 us output round trip is
  free slack; don't spend edits there.**
* **THE RULE FOR MERGING: it only pays when the tiles are small.** Three groups were merged,
  verified bit-exact, and then **measured as regressions**, all by back-to-back A/B:

  | group | tile size | merge | verifies | measured |
  |---|---|---|---|---|
  | V3 `conv_weight` | 128×16 B = 2 KB | 8 → 1 | yes | **−3 us (shipped)** |
  | V5 `W_in_z` | 128×256 B = 32 KB | 4 → 1 | yes | **+1.5 us** (84.3 vs 82.8, 4 runs) |
  | V0 `W_in_qkv` | 128×256 B = 32 KB | 8 → 4 (2-tile pairs A,C,D) | yes | **+0.9 us** (83.9 vs 83.0, 4 runs) |

  A merge trades ~0.55 us of engine issue per removed instruction for serialising the group's
  bytes onto **one** DGE queue instead of N queues in parallel. At 2 KB a tile the transfer time
  is negligible and you bank the issue time; at 32 KB the serialisation costs more than the issue
  it saves and PE's weights land later. **Check the tile size before merging anything.**
* **Merging V0 (`W_in_qkv`) or V2 (`conv_state`) 4 or 8 tiles at a time gives wrong numbers.**
  Bisected on the device: a **single** 2-tile V0 merge passes; **two** non-adjacent 2-tile merges
  pass; **three** (pairs at host slots 1, 11, 13) pass; adding the fourth (host slot 8, spare slot
  10) **fails** (output_1 11974% of the limit), as do the 4+4 split and the full 8-tile merge
  (output_0 300–10185%). The addressing is provably right — I re-derived every (src,dst) pair and
  the 2/3-pair subsets using the *same* generated code pass bit-exact — and `semaphores.py` is
  clean, so it is a race, not arithmetic. Since V0 is a regression even when it works, this only
  matters for **V2, whose tiles are 1 KB and which is therefore worth ~4.8 us if the corruption is
  understood.** See §5 step 1.
* **`ACTIVATION_TABLE_LOAD` is closed, confirming iter-1.** 7 loads × 1.28 us = 9 us in the tail,
  alternating `act_tbl_sel` 2 and 4. The instruction has exactly one meaningful field
  (`act_tbl_sel`) — no bank, no partial load. From `act_info.json` the NEFF ships 5 of the 21
  function sets, and `act_tbl_sel` 0..4 maps to
  natural_log_exp / softplus / **silu** / sigmoid / **reciprocal_sqrt_and_small**. **No set in the
  entire 21-entry catalogue contains both `silu` and `reciprocal_sqrt`**, so the alternation
  cannot be removed by field edits; `square` and `copy` are in every set.
* **Reordering the Pool prologue buys nothing.** The 32 prologue DMAs are already in an order that
  matches when their consumers need them (round-1 tiles 1–16 feed PE/DVE, round-2 tiles 17–26 feed
  Activation's `s13..s17` chain, 27–32 feed later PE). Any permutation that keeps each round-2
  DMA after its round-1 partner (required — the round-2 waits are `own sem >= 16`) just moves
  which consumer stalls. The prologue is bounded by its **total** issue time, so only removing
  instructions helps.

## 5. Where the remaining ~83 us goes, and the ranked next steps

Span 98 us traced, exec 83. Still latency-bound (no engine over 48% busy). Three blocks:

| window | us | what |
|---|---|---|
| 12 → 30 | 18 | **Pool prologue**: 25 serial DGE DMAs (was 32) loading the weights. Pure issue time. |
| 30 → 72 | 42 | a chain of ~15 DGE/queued hops: Pool prologue → Activation 1–34 → Activation's CASTs → SP 13–27 → SP 28–44 → queued blk5–8 → SP 50–60 → queued blk13/14 → Activation 60. |
| 72 → 94 | 22 | **Activation tail**, ~100% busy: 7 `ACT_TABLE_LOAD` (9 us, closed) + ~13 us of ACTIVATE/CAST/COPY at 0.2–0.5 us each. |
| 94 → 98 | 4 | output_0 stores + runtime epilogue, gated by Pool 66's `output_2` write. |

1. **Merge the V2 (`conv_state`) group — ~4.8 us of Pool issue time, the biggest single lever
   left — once the corruption from §4 is understood.** V2's tiles are 1 KB, i.e. in the regime
   where merging pays (§4 rule). The 8-tile V2 merge is written and verified-as-wrong; what is
   missing is why.
   * **Already refuted:** merge width alone (8-tile V3 passes, 2-tile V0 passes, 4-tile V5
     passes); descriptor count (V3 has the same 1024); byte count (V2 is 8 KB); the
     semaphore accounting (`semaphores.py` clean); vacuous DGE queue-depth waits (re-gating
     Pool 19/27/28/29/30 on the merged transfer did **not** fix the V0 8-tile case); mis-mapped
     consumer waits (I re-derived all of them, and 2- and 3-pair subsets of the identical
     generated code pass).
   * **Also refuted (last experiment of the iteration): it is not a count limit.** Merging V0 as
     2-tile pairs, the pass/fail pattern is per-pair, not per-count:
     `A(host 1,spare 3)` ok, `A+C(11,12)` ok, `A+C+D(13,14)` ok, but `A+C+B(8,10)` FAILS and
     `A+B+C+D` FAILS. So **pair B -- host Pool slot 8, spare Pool slot 10, i.e. the tiles at
     `V0+1024 -> SB+227840` and `V0+1280 -> SB+227584`, semaphores 15 and 17 -- is specifically
     un-mergeable**, in any combination, while the other three pairs merge freely. Every consumer
     wait was re-derived and is correct (`PE 6` on `s15>=16` = the merge, `PE 10` re-pointed from
     `s17>=16` to `s15>=16`), `semaphores.py` is clean, and the *identical* generated code passes
     for A/C/D.
   * **So the next step is to find what is special about Pool slots 8/10 (semaphores 15/17).**
     Candidates I did not get to: another reader of `SB+227584..228096` that I missed in the wait
     census (**rebuild the census from the current stream first** -- it was built from the iter-1
     stream and several waits moved during this iteration); `s15`'s ord-2 being Pool slot 24, a
     *no-wait* ES left by the V3 merge, which makes `s15` jump to 32 instantly; and `s17` having
     **all three** of its producers be ES after the V3 merge (slots 10, 26, 53). The last is the
     most suspicious and is specific to B -- for pairs A/C/D at least one producer of the affected
     semaphore is still a real DMA. Cheapest falsifier: redo pair B on top of the *iter-1* file
     (no V3 merge, so slot 26 is still a real DMA). Reproduce with the `an/e16.py` pattern: one
     `PAIRS` dict keyed A/B/C/D, pass the subset as argv.
2. **Finish the duplicate-removal sweep on Pool/SP with compute instructions (~3–5 us).**
   SP 5,6,7,8,9,12,17,22 are per-partition bf16→fp32 casts whose 2–4.6 us completions gate Pool
   33–55. Sync cannot run compute, **but their twins at Pool 34,36,38,40,43,51,53,55 are now
   cheap ES slots on an engine that can** — turn those ES back into compute `CAST`s that do the
   copy themselves (template Activation 23, `in_dtype=BFLOAT16, out_dtype=FP32`, dst elem 4) and
   neuter the SP halves. The blocker to work out first: the Pool ES currently waits on the SP
   twin's completion, so you must find what actually makes `SB+16384..18176` ready — SP 5–22 have
   **no wait at all** and rely on SP program order after SP 4's `s6>=25`, so `s6>=25` is the
   precondition, and the add value would be 25 (fine, ≥16).
3. **Activation tail, 13 us of non-table compute.** Activation is the busiest engine (39.7 us).
   Several tail `ACTIVATE`/`CAST` slots are 128-channel × small-element ops that DVE could do in
   parallel (DVE is 22% busy) — but moving work between engines needs a slot *inserted*, and
   **insertion looks feasible**: nothing records a stream's instruction count (`Pool0.json` is
   just `{"instr": "Pool0.bin", "name": ..., "dma": []}`), the length is the `.bin` size, and
   `neff.py pack` recomputes member sizes and both MD5s. Untested — worth one careful experiment.
4. **Don't bother with:** the output tail (§4, it is slack), `ACT_TABLE_LOAD` (closed twice), PE
   (7.6 us busy), DMA bandwidth (29 us busy over a 98 us span), any arithmetic change (output_0
   already uses 83% of the error budget).

## 6. Mechanics worth keeping

* Rewriting a slot: decode a known-good slot of the *same engine* as a template, `.set()` the
  fields, `.encode()`, splice the 64 bytes. Never hand-assemble a slot.
  `isa.py <Engine0.bin> <first> <count>` prints the result for eyeballing; it hides fields that
  are at their default, so a missing field in the dump usually means 0.
* `COPY`/`CAST` (`common.NEURON_ISA_TPB_S4D4_TR_STRUCT`) field names:
  `num_active_channels` is the **partition count**, `src/dst_mem_pattern_num_elem[4]` is the
  per-partition shape, `..._start_addr_addr_immediate` is the byte offset inside the partition
  (marker 0 = SBUF, 2 = PSUM). A per-partition copy is `num_active_channels=128`,
  `num_elem=[1,1,1,1]`, `step_elem=[1,1,1,1]`.
* `PSEUDO_DMA_DIRECT2D`: `src/dst_step_elem[2]` are **bytes** in *both* dimensions (a dim-1 step of
  -2048 really is -2048 bytes, not -2048 elements); dim0 pairs with dim0 and dim1 with dim1
  between source and destination, and `num_elem` must match in each dimension.
* Order of checks after an edit: `semaphores.py` (catches count errors instantly and for free) →
  `neff.py pack` → `verify.py` → `profile.py`. `semaphores.py` caught the only two real bugs I
  made before the device ever saw them.
* Keep the edit scripts idempotent (skip a slot that is already rewritten) — you will re-apply
  them against a reverted `work/` several times.

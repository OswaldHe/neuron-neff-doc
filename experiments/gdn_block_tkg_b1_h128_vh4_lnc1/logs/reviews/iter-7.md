# Review of iter-7 (kernel.neff, uncommitted vs HEAD 0220238)

Slot indices below are 0-based stream indices of the decoded streams. They equal `compiler_pc` in
`prof/Instruction.parquet`. They refer to the *current* kernel.neff unless marked "prev".
`notes/README.md` is still the empty template, so the agent wrote no notes for this iteration.
Everything below comes from diffing HEAD's NEFF against the working-tree NEFF.

## 1. What changed

I diffed against HEAD's kernel.neff (47.7 us), because the diff against baseline is just the sum
of seven iterations. DVE0.bin and PE0.bin are byte-identical. Three members changed, and 43 of the
46 package members are identical.

**Activation0.bin (119 -> 114 slots)**
- Slots 1-8 were 8 queued DGE loads, sems 80-87. They are now 3 loads.
  - Slot 1: the two 256 B row loads (sem 80) merge into one 2-D DMA. It is `[128,8]` with src step
    +256 and dst step -256.
  - Slot 2 (sem 86) and slot 3 (sem 82): 8-byte -> FP32 cast loads, each `[128,4]` with src step
    1024 and dst step -768.
    - Together they now cover all 8 columns. That includes the two columns Pool used to load itself
      (Pool slots 5, 6 in prev).
  - Sems 81, 83, 84, 85 and 87 are no longer used.
- Four Activation->DVE-style CASTs are inserted at slots 39-42 (bf16->bf16 copies). They write
  Pool-visible scratch via `SEM_ADD_IMM` on sems 17/18. They replace Pool's 3 DGE transposes (prev
  Pool 45-49) and Pool CAST 50-53. These were moved, not removed.
- The previous slots 47-49 (a CAST triple) were deleted.

**Pool0.bin (75 -> 75 slots, 29 differ)**
- 7 DGE DMAs (prev slots 4, 5, 6, 8, 9, 10, 11) became `EVENT_SEMAPHORE` (`SEM_ADD_IMM_COMPLETE`,
  +16).
  - Each is gated `WAIT_FOR_SEM_GE_IMM` on sems 80, 82 or 86, i.e. on the merged Activation loads.
- The Pool slots 2 and 3 field changes (semaphore 8->9, 9->16; src/dst addresses; element sizes
  256->2 and 2->8) are an alignment artefact of the diff, not edits. The slot order changed, so
  `neff_diff.py` pairs different DMAs by index.
- The three trailing output DMAs (prev 45, 47, 49) moved to 53, 55 and 57, behind the CASTs.
  - They are byte-identical apart from the position shift.

**SP0.bin (85 slots)**
- Two 2-byte -> FP32 transposes (prev 34 and 36, sems 61 and 62) became one `[128,2]` DMA with src
  step -768 (slot 34).
- Slot 36 is now an `EVENT_SEMAPHORE` that waits on sem 61 and adds to sem 62.

This matches the commit-message style of the previous iterations: DGE descriptor merging, DMA ->
semaphore-add, and ops moved between engines. It is a pure schedule edit and no ALU field changed.
The verifier agrees:
- All three outputs have max_abs_err 2.999e-3, 1.928e-3 and 6.033e-4.
- These are identical to the baseline's values to four digits, i.e. 83%, 54% and 17% of the limit.
  The error did not move.

## 2. Is the latency change real?

Yes.
- `profile_latest.json`: median **44.384 us**, min 43.964, max 44.490, 5 runs, spread 1.18%
  (~0.5 us).
- HEAD's own number is 47.695 us. The commit message says ~47.8.
- That is -3.3 us, about 6x the spread. The two distributions do not overlap.
- My re-run of `tools/profile.py` gave 44.437, which agrees.
- Versus baseline: 218.172 -> 44.384 (4.92x).

The loop is not chasing noise at this size. It will be from here on:
- Any candidate under ~0.6 us is inside the spread. Use `--repeats 10` or more before believing it.

Engine busy shows Activation at 35.6 us of 44.4 (80%) and DMA at 17.6 us. DVE, Pool and PE are at
15-19 us busy and SP at 6.3 us. Activation is the throughput-limiting engine and the only one near
saturation.

## 3. Safety beyond the verifier

**Semaphore accounting.** I diffed `semaphores.py` output between HEAD and the working tree.
- Every semaphore's `total_inc` still equals its `max_wait`, or exceeds it in the same way as
  before.
  - Sem 8, 9, 10, 11, 12, 17, 18 and 19: 64/64 or 48/48.
  - Sem 57: 64 incs, wait 48.
  - Sem 61: 80 incs, wait 32.
  - Sem 62: 32 incs, wait 32. It was 16 DGE + 16 event; it is now 32 event. This is consistent.
  - Sems 80, 82, 86: 16 incs each, wait 16, with 8, 4 and 4 waiters.
- Sems 81, 83, 84, 85 and 87 are now unreferenced.
- The "wait on a semaphore with too few incs passes early" failure from iter-5 does **not** recur.
  Totals match thresholds.

**Residual risks.**
1. The merge relies on the previous iterations' finding that a 2-D DGE merge is only legal in the
   *last* slot of an engine's queue.
   - Slot 1 is now a merged DMA at the *first* slot, and slots 2 and 3 follow.
   - That rule was refuted or confirmed for particular queues. It passed here, but it was run
     **once** (`"runs": 1` in verify_latest.json).
   - A race between the merged completion and Pool's seven waits (all `>=16` on one semaphore, so
     one early increment releases all of them) would not show up in one run.
   - Run `verify.py` at least 10 times, and the profile's 5 runs, and require identical outputs.
2. Seven Pool waiters now release on one ACT completion each. If the DGE completion of a
   multi-descriptor DMA fires per ring rather than per transfer, a waiter could release before the
   whole transfer has landed.
   - Totals do balance at 16, which suggests this is not the case. It is untested.
3. Error did not move, so there is no error-headroom concern from this iteration.

Verdict: acceptable. The change is plausible and correct on the evidence available. Re-verify with
more runs before committing it as a base for further merges.

## 4. Candidates for the next iteration

**What the profile shows.** (`prof/Instruction.parquet`, 1 run, times in us from the first
instruction. The kernel body runs from ~12.0 to ~55.4.)

- Activation spends 9.0 us of its 35.6 us busy time in **7 `ACTIVATION_TABLE_LOAD`s of 1.283 us
  each**. The table sets in sequence are 0 softplus, 1 exp, 2 silu, 3 sigmoid, 4 rsqrt, 2 silu,
  4 rsqrt.
- Activation runs back-to-back (gaps ~0) from 22.5 to 37.7, and again from 39 to 44.
- In the tail (44.9 -> 55.4) the four heads' chains alternate ACT -> DVE -> PE with 0.3-1.4 us
  hand-off gaps, and the engines are mostly idle.
- The first PE op is at 17.7, behind DMA completion.
  - Activation spends 4.7 us issuing 3 DGE DMAs at 12.2-17.15. The third alone takes 2.3 us, behind
    the other queue.
  - The Q0 queue moves 2064 packets (278 KB) in 13.7-19.0. About 1100 of them are 4-16 B packets
    taking 16-19.
- After the last DVE op at 55.4 there are ~3.5 us of Sync/Pool epilogue and a ~2.5 us output-DMA
  completion wait.

**Ranking** is by gain per unit risk.

| # | Edit | Ceiling | Risk |
|---|------|---------|------|
| 1 | Move the four serial ACT scaled-copies (Activation slots 50-53) to DVE | ~1.5 us | low |
| 2 | Move ACT slots 24-29 (four `func 128` scale ops + two CASTs) to DVE | up to ~1.3 us, unknown | low |
| 3 | `sigmoid` as `tanh` in set 2, deleting the set-3 table load | ~1.3-1.6 us | medium (numerics) |
| 4 | `softplus` as `ln(exp(x)+1)` in set 1, deleting one table load | <=1.0 us | medium (range) |
| 5 | Interleave the four head chains in the tail | 2-3 us, unverified | high |

### Candidate 1 — move ACT slots 50-53 (+54) to DVE

- **Where:** Activation0.bin, slots 50, 51, 52, 53. These are four `ACTIVATE func 128`
  (copy-with-scale), 1 channel x 128 bf16 elements at stride 4, ~0.64 us each. Slot 54 is a CAST
  that signals sem 76.
  - Slot 50 waits `sem 6 >= 38`.
  - The scale is a *pointer* to the FP32 rsqrt scalar written by slot 48 (dst 23920).
  - Insert the four replacement ops in DVE0.bin after DVE slot 90 or 91.
- **What:** build 4 DVE `SCALAR_TENSOR_TENSOR_ARITH` or tensor-scalar ops with `op0 = MULT`,
  `imm0_src = POINTER_IMMEDIATE`, `num_active_channels = 1` and `out_dtype = BF16`. DVE slot 86
  already uses this form with `op0 MULT`, `op1 ADD`, `imm0_src POINTER_IMMEDIATE`.
  - Make them replace the CAST at 54 as well.
  - Remove slots 50-54 from the Activation stream.
  - **Signalling:** slot 54 does `SEM_ADD_IMM +16` on sem 76. Sync slots 76-84 wait on it. The DVE
    op must signal the same semaphore with the same +16 contribution. Do not use `SEM_INC` on sem 6,
    because that renumbers every later DVE-sem-6 threshold.
  - Check the DVE stream's other update indices and thresholds around the insertion.
- **Mechanism:**
  - DVE pointer-scalar multiplies are used already (DVE 86-90).
  - Iter-6 moved eight Activation CASTs onto DVE, bit-identical.
  - "Inserting instruction slots works" was established in iter-1.
  - fp32 multiply followed by RNE to bf16 should be the same on both engines.
    That needs confirming: compare output hashes.
- **Ceiling:**
  - The ops are not on the critical path of the next stage. Their only consumer is Sync's final
    sem-76 wait (55.4).
  - Currently Activation runs 34.29 -> 36.13 (4 x 0.64 us) plus the CAST, then the table load
    (36.15-37.43), then `silu` at 37.54.
  - The `silu` inputs come from DVE slots 84-90 and are ready at 35.97, because slot 76 waits
    `sem 4 >= 20`, i.e. on the rsqrt in slot 48 (34.04).
  - After the edit the table load can start right after 34.04. `silu` can start at ~36.0.
    Gain = 37.54 - 35.97 = **1.57 us at most**.
  - DVE has ~2.3 us idle (35.97 -> 38.28) to absorb the four ops (~0.8 us).
- **Risk:** the verifier catches wrong arithmetic. It does not catch a wrong semaphore. A threshold
  that is too low can pass early and still verify. Use `semaphores.py`: sem 76 must still total 16
  from exactly one source, with waiters `>= 16`.
- **Cheapest falsifier (minutes):** change only slot 54's source and delete 50-53, i.e. check DVE
  cannot do `MULT` with a pointer scalar at `num_active_channels = 1` on a `[1,128]` stride-4
  bf16 pattern. The device run either hangs or passes. Then compare the outputs against the
  current NEFF's bits.

### Candidate 2 — same transformation on ACT slots 24-29

- **Where:** Activation0.bin slots 24, 25, 27, 29 (`func 128`, 128 channels, FP32->FP32, ~0.49 us
  each) and 26, 28 (CAST, 0.28 us). They run 25.7 -> 27.75, back-to-back with the table load at 30.
- **What:** same as candidate 1. Move them to DVE with the same pointer-scalar form (DVE 8-11 use
  it).
  - The new DVE op must signal the same semaphore. These ops update sem 4 with
    `SEM_INC_COMPLETE`, whose counts feed later waits. Renumbering is needed if their increments
    are removed.
  - Read each consumer's wait on sem 4 before touching anything.
- **Ceiling:** the 1.283 us table load at slot 30 could start at ~25.7 instead of 27.76. Unknown
  how much DVE (busy 25-28) can absorb. Treat 1.3 us as the upper bound, not the expectation.
- **Risk:** the sem-4 threshold renumbering is the dangerous part. Pass-once-race-later applies.
- **Falsifier:** check that DVE slot ordering is such that those ops' inputs are ready early (inputs
  arrive from Pool/DMA).

### Candidate 3 — `sigmoid` via `tanh`, deleting the set-3 table load

- **Where:** Activation slot 43 (`ACTIVATION_TABLE_LOAD`, `act_tbl_sel = 3`), slot 44
  (`ACTIVATE func 5`, FP32, 128 channels) and slot 45 (CAST).
- **What:** `sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5`.
  - Slot 44 becomes `tanh` with `scale = 0.5`. The `silu` set (set 2) contains `tanh` (size 4).
  - A DVE op does `x*0.5 + 0.5` and writes bf16, replacing the CAST at slot 45.
  - Delete slot 43. Table order becomes 0, 1, 2, 4, 2, 4.
- **Mechanism:** `act_info.json`'s set contents. Established from the NEFF's act_info: set 2 has
  `{silu, tanh, sin, square}`. The op ids are those from `docs/16`.
  - This is a numerical substitution, not a bit-identical one. Table-set contents are documented in
    `docs/03-package-files.md` (3.5).
  - Whether the tanh polynomial is accurate enough is untested.
- **Ceiling:** ACT is back-to-back in this region, so up to 1.28 us from the load plus ~0.28 us from
  the CAST, minus ~0.2 us for the extra DVE op.
- **Risk:**
  - The verifier catches a gross error. It will not catch a small drift, because err is at 83% of
    the limit on output 0.
  - Check how much the output moves. Reject if output_0 moves above ~90% of the limit.
- **Falsifier:** run once and read `verify_latest.json`. This takes minutes.

### Candidate 4 — `softplus` via `ln`/`exp`, deleting one table load

- **Where:** Activation slot 17 (`ACTIVATE func 9` = softplus, 4 elements) uses table set 0 loaded
  at slot 4. Slots 21-22 (`exp`) use set 1, loaded at slot 19.
- **What:** load set 1 at slot 4 and delete slot 19. Replace slot 17 by `exp` followed by `ln` with
  bias 1.0, both from set 1.
- **Ceiling:** at most 1.28 us minus the extra ACT ops (~0.4 us) = ~0.9 us.
- **Risk:** numerical and range-dependent. `exp(x)` overflows for x > 88, which the verifier's
  fixed inputs would not exercise.
  - It also changes the dtype-converting ops at slots 6-16, which read the active table.
- **Falsifier:** one device run, plus a check of the input range of `a + dt_bias`.

### Candidate 5 — interleave the four head chains in the tail (44.9 -> 55.4)

- **What:** the four v-heads are independent, but ACT/DVE/PE execute them head by head, with
  0.5-1.4 us hand-off gaps (ACT gaps at slots 83, 98, 100, 107, 110). Reorder the ACT, DVE and PE
  streams for software pipelining.
- **Ceiling:** unknown, plausibly 2-3 us. The gaps sum to ~4 us. Whether they are recoverable
  depends on the dependency graph, which I did not trace.
- **Risk:** high. It needs renumbering of the sem 4/5/6 thresholds across three engines, and wrong
  thresholds pass the verifier. Do this last, and only if candidates 1-3 are exhausted.

**Do not expect gains from the output store.** The epilogue after the last DVE op (~3.5 us) is a
fixed DGE completion latency (see Closed).

## Closed (already refuted; do not redo)

- **Relocating a DGE DMA to another engine's stream.** DGE queues are owned per engine with a fixed
  semaphore set (def.json `dma_queue`). Only a single *trailing* queue merges, and that legal merge
  regressed 1.8 us (iter-6).
- **Splitting the 256 KB output store.** Completion is ~2.5 us of fixed latency, not bandwidth
  (iter-6).
- **Letting SP overlap two DGE transfers.** It passes verify and then hangs the device (iter-6).
- **Moving a dtype-converting Activation op across an `ACTIVATION_TABLE_LOAD`.** Every
  dtype-converting op including `ACTIVATE func 128` reads the active table. Only a same-dtype CAST
  may cross one (iter-6).
- **Deleting `func 128` with a pointer scale as "just a copy".** It carries a memory-pointer scale
  (iter-6).
- **Replacing a DMA with a semaphore-add where a waiter reads an otherwise-untouched
  semaphore.** Sems 68-70 reached their 32-threshold early and hung the device while verify.py
  still passed (iter-5).
- **Merging tile loads into one 2-D DMA outside the last DGE slot of an engine** (iter-5). This
  iteration did it in the first slot and it passed, so treat the iter-5 rule as not fully
  characterised and re-verify.
- **The `dma_configs_dma_transpose_2d_tile` bit** is inert (iter-3).
- **A strided multi-element compute CAST does not gather what its step field says** (iter-5).
- **Single-run decisions on sub-0.6 us effects.** The measured spread is 1.2%.

**Closed by this review's tracing:**
- **Hoisting the second `silu` batch (Activation slots 55-62) up to join the first one**, to delete
  table loads 55 and 76. Its inputs come from DVE slots 76-90. Slot 76 waits `sem 4 >= 20` on the
  rsqrt in slot 48 (DVE start 34.08 right after slot 48's end 34.04). The batch is therefore
  downstream of the rsqrt, and the 2 -> 3 -> 4 -> 2 -> 4 order is data-forced. Only function
  substitution (candidates 3 and 4) can reduce the table-load count.

## Unknown (measuring these would change the ranking)

1. **Is candidate 1's DVE port bit-identical?** Settle it with one run and a hash comparison of the
   outputs against the current NEFF.
2. **What gates the first PE op at 17.7?** If it is the ACT slot-1 weight load (262 KB), halving
   the transfer saves ~1 us at the front. If it is the descriptor-queue limit (Q0 at ~20 ns per
   packet over 8 engines), it saves nothing. Measure by reading `DmaPacket.parquet` for the packets
   of sem 80 versus the PE wait on it.
3. **Whether the ~3.5 us epilogue is shortenable.** After the last DVE op at 55.4: Sync slots 76-84
   (0.6 us), Pool slots 71-74 (56.9-57.2, waiting on a 36 us event), and the Activation
   `EVENT_SEMAPHORE` of 2.67 us at 54.9. If the 2.67 us wait is the output-DMA completion, it is
   closed. If it is a fixed barrier, the last compute could overlap it. Settle it by comparing the
   timestamp of the output packets (`DmaPacket.parquet`, queue `qSyncIO0`, S[42], 56.4-56.6) with
   the instruction timeline.
4. **The ~4 us of hand-off gaps in the tail.** Whether the four heads are truly independent in the
   DVE/PE dependency graph, and how many sem-4/5/6 renumberings interleaving needs. Build the graph
   from `semaphores.py` waiters before committing to candidate 5.
5. **Run-to-run correctness of this iteration.** Run `verify.py` 10 times. One run is all that
   `verify_latest.json` holds.

If candidates 1-3 close with gains below the spread, what remains is 3.5 us of epilogue, ~5.7 us
of DMA-bound prologue and a serial tail with hand-off latencies. Those are structural, not
field-level edits, and the honest estimate of what is left is a few microseconds at best.

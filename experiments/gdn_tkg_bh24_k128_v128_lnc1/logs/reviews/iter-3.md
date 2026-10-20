# Review of iter-3 (kernel.neff, uncommitted vs HEAD 504faee)

Reference points: baseline 153.46 us; HEAD (accepted iter-2) 74.5 us as claimed in its commit message; this iteration 72.68 us (`profile_latest.json`; my re-run gave 72.64).
`notes/` contains only the README. The agent wrote no notes for this iteration and the commit message is for the previous one, so I cannot check "claimed vs done" for this iteration. I diffed HEAD's NEFF against the working tree.

## 1. What changed

`neff_diff.py HEAD kernel.neff`: 46 members, 41 identical, 5 changed. All five are `sg00` instruction streams. No DMA, descriptor or metadata members changed.

| member | slots | what |
|---|---|---|
| Activation0.bin | 170 -> 170, 124 differ | Mostly slot shifts from re-ordering. Net: 3 more `ACTIVATE -> CAST` (fp32->bf16 with the scale fused, same trick as iter-2), `ACTIVATION_TABLE_LOAD` moved, `events_semaphore_value` rewritten (slot 12: 16 -> 32), EVENT_SEMAPHOREs moved. |
| DVE0.bin | 166 -> 172 | +6 slots. DVE `TENSOR_TENSOR_ARITH_OP` on sem 6 goes 64 -> 67. STREAM_SHUFFLE / LOAD_MASK_SELECT / TT slots reordered. DVE now takes the work ACT dropped. |
| Pool0.bin | 14 -> 15 | +1 `PSEUDO_DMA_DIRECT2D`. The head DMA on sem 25 is split in two (Pool slots 2 and 3, +16 each). |
| PE0.bin | 163, 65 differ | Re-ordering and wait-value updates. Matmul count is unchanged at 74. |
| SP0.bin | 1 slot | One wait/update. |

In short, this is a continuation of iter-2: fuse more ACT work into casts, move work to DVE, and split the head DMA. This is a reasonable reading of the diff, but it is not confirmed by any notes.

## 2. Is the latency change real?

Yes. 74.5 -> 72.68 us is -2.4%. The reported spread is 0.19% (72.623 to 72.759 over 5 runs, about 0.14 us). The gain is about 13 times the spread, and my independent re-run gave 72.637. Against baseline it is 2.11x (153.46 us, spread 0.79%).

Caveat: the 74.5 us figure comes from a commit message, not from a stored profile, so the 1.8 us step has a single source. Even so, the 72.64 re-run against the baseline spread leaves no doubt the kernel is faster than baseline.

Not noise, but the lever is getting small. Each of the last two steps was only 2 to 5%.

## 3. Safety

- Verifier: pass. `output_0` max_abs_err is 2.322e-2 (83% of the 0.028 limit) and `output_1` is 1.633e-3. Both are **identical to the baseline NEFF to all digits**, so the numerics are unchanged. The 83% is inherent to bf16 against the golden, not something this edit moved.
- `semaphores.py` (unpacked copies): no "wait exceeds increments" flag. Totals balance on every semaphore, e.g. sem 4 120/120, sem 5 74/74, sem 6 103/103.
- **One real concern: sem 25 is now shared by two DMAs.**
  - Before: one DGE DMA (+16), one waiter on Act (`>=16`).
  - Now: Pool slots 2 and 3 each add +16 (total 32), with 5 waiters: Act 12 `>=32`, DVE 19 `>=16`, DVE 21 `>=16`, DVE 23 `>=32`, DVE 25 `>=32`.
  - A `>=16` wait on a counter fed by two DMAs passes when *either* DMA has finished. If DVE 19 or 21 reads data from the second DMA, it can run before that data lands. The reverse also holds.
  - This passes a fixed input set whenever the DMAs happen to finish in issue order, so the verifier cannot rule it out. I could not tell which tensor each waiter reads without decoding the DMA destinations, so it is **unverified, not shown wrong**.
  - The cheap check is in candidate 0 below.
- No semaphore was removed. Sem 6 and sem 4 totals went up or stayed flat.

## 4. Where the time is now

From `prof/Instruction.parquet` (one repeat):
- Engine busy over the 72.7 us exec: ACT 55.5 us (76%), DVE 40.6 (56%), PE 18.7 (26%), Pool 15.5 (21%), DMA 14.9, SP 3.4.
- **ACT is the critical engine.** In the body (about 30 to 80 us) it is busy 3.9 to 4.3 of every 5 us.
- ACT time by op:

| ACT op | count | total |
|---|---|---|
| ACTIVATE, `COPY` with scale | 24 | 11.8 us (491 ns each) |
| ACTIVATE, FUNC_29 / FUNC_30 and other | 53 | 14.7 us (~280 ns each) |
| CAST | 36 | 12.8 us (355 ns each) |
| ACT_TABLE_LOAD | 6 | 7.7 us (1.283 us each) |
| ACTIVATION_READ_ACCUMULATOR | 24 | 7.0 us |
| EVENT_SEMAPHORE | 30 | 8.4 us |

- The head (about 10 to 30 us) has 5 of the 6 ACT_TABLE_LOADs (at 10.2, 15.2, 17.4, 18.9, 21.8; the 6th is at 28.5). The 4 big Pool DGE DMAs run serially at 10.7 to 16.7 us, at 1.2 to 1.7 us each.
- The body has 8 state-out DMAs, issued from Pool every 4 to 8 us. Each takes 0.88 us to generate.
- Tail: the last DMA is issued at 79.9 to 80.8 us. Then all engines sit in EVENT_SEMAPHORE until about 83.6 us (Tensor 3.2, Act 2.4, DVE 1.7, Sync 1.4 us). This tail is the output-DMA completion wait.

## 5. Ranked candidates for the next iteration

**0. Prerequisite (no gain): make sem 25 safe.** Do this first, because every later candidate builds on this NEFF.
- Slots: Pool0 slots 2 and 3 (the two DGE DMAs), and Act 12 and DVE 19, 21, 23, 25 (the waits on sem 25).
- Check: decode each DMA's destination SBUF address. Match them to the tensors each waiter reads.
  - If DVE 19/21 read data from only the first DMA, `>=16` is correct only if the second DMA cannot finish first (it is issued after, but DGE queues are not ordered). Safer: wait `>=32` on both.
  - Alternative: put the second DMA on a different semaphore (as iter-2 did with sems 24 to 27).
- Cost of the `>=32` fix: probably small, since the DMAs are issued back to back.
- Falsify: change DVE 19 and 21 to `>=32`, re-profile. If the time moves more than 0.3 us, the early pass was real.

**1. Remove or merge redundant ACT_TABLE_LOADs.**
- Engine and member: ACT, `Activation0.bin`, the 6 `ACTIVATION_TABLE_LOAD` slots. They are at compiler pc 1, 3, 6, 8, 11 and 24, in the head, which is ACT-gated.
- Change: decode the table id in each slot. Re-order the head so ops that use the same table are adjacent, then delete (NOP) a load whose table equals the one already resident. Likely candidates: COPY, FUNC_29 and FUNC_30 are probably table-less or share a table.
- Mechanism: the iter-2 result. Cutting loads 46 -> 6 was a large part of 153 -> 80 us. A load costs 1.283 us fixed (profile).
- Ceiling: 5 loads x 1.283 = 6.4 us of ACT busy in the head. The real gain is lower, because the head also waits on DMA. I would expect 2 to 4 us. This is bounded above by 7.7 us (the 6-load total).
- Risk: if the table is wrong, the activation function outputs are wrong. The verifier would catch that on any input (it is bit-wise). Low risk.
- Cheapest falsifier: dump the table id of each of the 6 slots. If all 6 are distinct tables, ordering cannot remove any of them, and the lever dies in minutes.

**2. Shorten the tail: split the last state-out DMA.**
- Engine and member: Pool (DGE), `Pool0.bin`, the last `PSEUDO_DMA_DIRECT2D` (compiler pc 12, issued at about 79.9 us), plus its wait in the end `NOP`/`EVENT_SEMAPHORE` slots.
- Change: the last chunk covers the final 3 heads. Split it into per-head chunks (or 2 + 1) so that only the last head's data is on the critical path after its compute. Earlier chunks are issued as soon as ready.
- Mechanism: the same 8-way split iter-2 used for state-out.
- Ceiling: the tail wait of about 3 us (80.5 to 83.6) is partly fixed completion latency. I estimate 1 to 2 us, but it is **unknown**; settle it by measuring one DMA's issue-to-complete latency from `DmaPacket*` in the profile.
- Risk: moderate. It touches Pool and sem 27 (+16 each, wait `>=128`). The `128` is hard-wired. If the chunk count changes, sem 27's wait value must change with it, or the wait never fires, or fires early. The verifier cannot catch an early fire reliably.
- Falsifier: add a 9th DMA that is a copy of an existing one. If the total moves by less than the profile spread (0.15 us), the tail is latency-bound, not issue-bound, and the lever is dead.

**3. Move work off ACT onto Pool, which is idle.**
- ACT is 76% busy and sets the critical path. Pool is at 21% and DVE at 56%.
- Candidates: the 36 CASTs (12.8 us total) and the 24 READ_ACCUMULATORs.
- Change: re-emit the slot's S4D4 CAST struct in `Pool0.bin` (or the DVE stream) and re-wire the semaphore waits and updates. Sem 4 is currently incremented by ACT, so the consumer waits change.
- Mechanism: iter-2 moved ops from ACT to DVE and that worked (DVE TT count up). Pool is slower per element than DVE, so Pool may take longer than the 355 ns ACT cost. It might still win because it runs in parallel.
- Ceiling: each ACT op moved gives its duration back on the critical path, up to about 12.8 us for all 36 CASTs. Realistically 2 to 5 us, because the data dependencies between engines mean ACT may not be the true critical path in every window.
- Risk: **high**. Semaphore re-wiring is exactly what passes once and races later. Do it for one head and check `semaphores.py` totals.
- Falsifier: move one CAST from ACT to Pool. Check whether Pool can execute it at all (the opcode may not be valid in the Pool stream). If it faults, drop the lever.

**4. Parallelise the head DMA issue across engines.**
- Engine and member: `Pool0.bin` slots 1 to 4 (the 4 serial head DMAs at 10.7 to 16.7 us, about 6 us of serial DGE time).
- Change: re-emit 1 or 2 of them in `SP0.bin` (idle at 4.7%), with their own semaphores.
- Ceiling: **unknown**. ACT already starts at 12.4 us, before the last head DMA ends (16.7 us), so the head may not be DMA-gated. At most about 2 to 4 us.
- Risk: moderate. Whether SP can issue `PSEUDO_DMA_DIRECT2D` is unestablished here.
- Falsifier: copy one DMA slot to SP0, with a fresh semaphore. If it runs and the time moves, continue.

## 6. Closed (do not redo)

- 24 per-head silu / k-scale / state-scale ops collapsed into one wide ACT op and two DVE broadcasts: done in iter-2 (153.6 -> 79.6 us).
- ACT_TABLE_LOAD 46 -> 6: done. Profile shows exactly 6.
- The 24 PSUM->SBUF copies: dropped in iter-2.
- State-out DMA split 8 ways: done (8 chunks seen at 33.7 to 80 us).
- k_f*=k_scale split 8 ways: done.
- S_bf casts group-merged: done.
- Fusing `ACTIVATE -> CAST`: 16 of 24 in HEAD, 3 more in this iteration. Do not restart; see "Unknown" for what remains.
- Numerical error margin: the identical 2.322e-2 (83%) in baseline and current. It does not move with these edits, so the margin is not a constraint on timing-only edits.

## 7. Unknown (would change the ranking)

- **Table ids of the 6 ACT_TABLE_LOADs.** Decode the field in each slot. This decides whether candidate 1 is worth anything.
- **Issue-to-complete latency of one output DMA** (`DmaPacket*` in the profile). This decides candidate 2.
- **Which tensor each sem-25 waiter reads.** This decides whether the current NEFF is safe. Run candidate 0 first.
- Whether the first 10 us (TENSOR_LOADs and EVENT_SEMAPHOREs before the first real instruction at about 10 us) count in `total_exec_time_us`. The timeline runs to 84.9 us against 72.6 us reported. If it is included, it is a fixed cost no edit can remove.
- Why 24 COPY ACTIVATEs (491 ns) are still on ACT after "16 of 24 fused". The iter-2 count suggests I may be mixing them up with a different 24; verify by listing the COPY ACTIVATE slots and their dst/src before assuming 8 remain.
- Whether ACT or the PE->ACT dependency is the real critical path in the body (ACT is only 80% busy per 5 us window). A dependency-graph critical-path measurement would settle this and sets the ceiling for candidate 3.

## Verdict

The edit is real (about 2.4% faster, bit-identical outputs, well above the 0.19% spread) and acceptable if candidate 0 is cleared. The remaining headroom is likely 5 to 10 us at best (candidates 1 to 3 combined), about 7 to 14%. The next steps are 10% or less each, so a few more iterations at most; this is not an open-ended search.

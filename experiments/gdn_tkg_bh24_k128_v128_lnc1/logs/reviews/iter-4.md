# Review of iteration 4 (kernel.neff, uncommitted; HEAD = iter 3 at 72.68 us)

## 1. What changed
I diffed `HEAD:kernel.neff` against the working `kernel.neff`. Both are NEFFs for the same kernel, so the diff shows only this iteration's edit. 3 of 46 members changed: `Activation0.bin`, `DVE0.bin`, `PE0.bin`. Everything else is byte-identical, including the DMA json descriptors, `def.json` and the Pool and SP streams.

- **Activation0** (170 -> 170 slots, 31 differ):
  - Slots 1-6 and 20-24 are reordered. ACT_TABLE_LOAD, EVENT_SEMAPHORE and ACTIVATE swap places.
  - The wide ACTIVATE at slot 2/4 is resplit. A 3072-element fp32 op (func 36, src 0x834e00 -> dst 0x20000) moves to slot 4, and the 24-element func-9 op takes its old position.
  - The wait idx/value (sem 4/6/11/12) on slots 2, 4, 23 and the sem update on slots 2 and 23 are edited.
  - Slots 58/59 and 83/84 swap an f128 COPY-scale ACTIVATE with a fused fp32 `func 29` op. The `func 29` op carries immediate scale 0.0078125 (0x3c000000) and bias 0x358637bd, uses imm_src=INSTRUCTION_IMMEDIATE and has a 1-element dst.
  - The remaining diffs are small: sem values in slots 60 and 83/84, and wait idx 4/6.
- **DVE0** (172 -> 173 slots, 168 differ):
  - The slot-5..24 head is rotated. STREAM_SHUFFLE / LOAD_MASK_SELECT pairs shift by one slot.
  - One extra slot appears: a pure `EVENT_SEMAPHORE` with a sem-6 increment, +4 in total. The matching TENSOR_TENSOR_ARITH_OP count drops 67 -> 63.
  - The semaphore table shows the same total sem-6 increments (103) as before.
- **PE0** (163 slots, 17 differ): only `events_semaphore_value` on the sem-4 (ACT) waits is lowered. Examples: 15->14, 18->17, 23->19, 31->29, 31->30, 39->38.

**Consistency with the notes.** `notes/README.md` is still the empty template and no commit message exists yet, so there is no claim to check. **The agent must write a note** (slots touched, before/after, why) before this is committed. That note is also the next agent's only memory.

My reading of the diff is that this is a reordering of ACT and DVE head work (earlier ACT-table loads and wide ops) plus re-based semaphore thresholds. I could not tell from the diff alone which specific ops moved where.

## 2. Is the latency win real?
Yes.

| NEFF | median | min / max | spread |
|---|---|---|---|
| iter 3 (HEAD) | 72.68 us | - | - |
| iter 4 (`profile_latest.json`) | 69.156 us | 68.963 / 69.287 | 0.47% (about 0.3 us) |
| baseline | 153.46 us | 153.10 / 154.32 | 0.79% |

- The -3.5 us (-4.9%) step is about 10x the spread, so it is not noise.
- A second profile run during this review gave 69.059 us, which agrees.
- Across iterations the gains were 153.6 -> 79.6 -> 74.6 -> 72.7 -> 69.2. These are real, but the remaining gains per iteration are getting smaller, with each one a few us out of a 69 us exec.
- `total_exec_time_us` (69.1) is smaller than the instruction-trace span (80.8 us). The span includes about 4 us preamble and about 4 us postamble that the metric does not count.

## 3. Safety beyond the verifier
- `verify_latest.json` passes. `max_abs_err` is 2.322e-02 (83% of the 0.028 limit) on output_0 and 1.63e-03 on output_1. These match the baseline value exactly, so the output is still bit-identical and error has not moved.
- `semaphores.py` on both unpacked NEFFs shows no "wait exceeds increments" lines and total increments per semaphore unchanged. Sems 4, 5 and 6 balance at 120, 74 and 103 with max_wait equal to total_inc. The DMA/DGE sems are unchanged.
- The number of waiters changed: sem 4 went 165 -> 149 and sem 6 went 111 -> 110. So **16 ACT-sem waits were dropped or loosened**. This happens even though the PE waits show lower thresholds.
- The PE thresholds (e.g. 23->19, 31->29) are the risk. A lower cumulative count means PE can proceed after fewer ACT completions. That is correct only if the ACT op PE needs now sits earlier in the ACT stream. I checked the totals but **not** per-op producer/consumer ordering. The fixed-input verifier cannot catch a race that depends on timing. Because the output matches the baseline bit for bit across the run, the ordering is probably right, but this is not proven.
- **Request:** the agent should run `verify.py` with `--runs 5` or more (the verifier took 9.5 s for one run) and check each of the lowered PE waits against the ACT op that produces its data.
- I could not confirm that nothing else was removed, because the sem counters balance but they say nothing about WAR hazards on reused SBUF buffers. This is the usual way these edits go wrong, and the verifier cannot see it.

## 4. Ranked candidates for iteration 5
Profile facts used below (from `tools/profile.py --repeats 1`, in time since the first instruction, span 80.8 us):
- The ACT queue is the critical engine. It is busy 64.3 us of the span (about 81% of exec). Per-opcode busy: ACTIVATE 27.4, CAST 12.8, ACT_TABLE_LOAD 7.7 (6 x 1.283), EVENT_SEMAPHORE 7.1, ACTIVATION_READ_ACCUMULATOR 7.0 (24 x 0.293), COPY-scale ACTIVATE 11.8 (24 x 0.49).
- Body from about 24 us to 76 us runs about 2.15 us per head. ACT is busy about 3.2-3.5 us per 4 us window, which is 80-88%, so the body is ACT-bound.
- Head before PE steady state: first MATMUL at 17.1 us, ACT table loads at 9.6, 11.5, 15.5, 17.5, 20.4 and 25.9 us. DVE has a 3.08 us TENSOR_TENSOR at 15.6 us. Pool does the state-in DMA_DIRECT2D, 4 x about 1.5 us from 10.2 to 16.2 us, and then idles.
- Tail: last ACTIVATE ends at 76.8 us and the last state-out DMA is triggered at 75.76 us. Nothing computes after that, but the kernel does not end until about 79-80 us, so about 2.3 us after the last compute are drain, final-DMA completion and semaphore clears.

| # | Candidate | Where | Change | Mechanism | Gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Fuse the other 8 per-head `COPY(scale)` ACTIVATE f128 ops (490 ns each) into their following CAST, as iteration 2 did for 16 of 24 | Activation0.bin: the 8 remaining COPY ops and the CASTs that read them (slots near 58/59 and 83/84 show the pattern) | Set the CAST's scale/imm to the COPY's scale and drop the COPY. Use the same `func 29` / INSTRUCTION_IMMEDIATE form this iteration already used. Rebase the PE sem-4 waits down by 1 for each removed op. | Established in iterations 2 and 4 (bit-identical on device). | Up to 8 x 0.49 = 3.9 us of ACT busy. Realistic: 1-2 us, since the body is ACT-bound at 80-88%, not 100%. | Medium: one wrong sem-4 threshold passes once and races later. | Remove one op, run `verify.py` x5, and look at the per-head ACT duration in the profile. |
| 2 | Cut ACT_TABLE_LOAD from 6 to 4 or fewer, by reordering head ops so ops on the same table are adjacent | Activation0.bin slots 1-24, the 6 ACT_TABLE_LOAD slots (act_tbl_sel 1/4/...) | Reorder, then delete loads whose table is already resident. | ACT_TABLE_LOAD 46 -> 6 in iteration 1; each load costs 1.283 us. | At most 7.7 us on ACT. The head loads at 9.6-25.9 us are on the critical path before the first PE work. Realistic: 1.3-2.6 us. | Medium: deleting a load needed by a later op gives wrong values. The verifier catches that, deterministically. | Delete one candidate load, run `verify.py`. A wrong table fails the numbers at once. |
| 3 | Shorten the head: start PE earlier by moving the DVE 3.08 us op at 15.6 us (and similar) into chunks | DVE0.bin slots 5-24 and the first big TENSOR_TENSOR | Split further, as iteration 3 did for S*=exp_g (3.36 -> 4 x 6 heads). Move the pieces into DVE gaps. | Established on device in iterations 2-3. | Head is about 17 us before the first MATMUL. DVE idle in head gaps 4.4 us at 11.2 us. Ceiling: about 3 us. | Low-medium: chunks must wait on the right sems. | Check that the new first-MATMUL time moves earlier than 17.1 us in the profile. |
| 4 | Shrink the tail: split the last state-out DMA chunk(s) finer, or trigger earlier | Pool/SP DMA descriptors (`*.json` "dma" blocks) and the last DMA_DIRECT2D at 75.76 us | Split the last chunk 2-4 ways, as done for state-out (8 chunks) in iteration 1. | Same mechanism as iteration 1. | About 2.3 us between last compute (76.8 us) and end, part of it fixed drain. Ceiling is probably under 1 us. | Low: the DMA json also feeds sem accounting. | Profile again and check the end of the last DMA queue. |
| 5 | Move the 4 Pool state-in DMA_DIRECT2D (10.2-16.2 us, each 1.2-1.7 us) to issue in parallel on other queues | Pool0.bin DMA_DIRECT2D x4 and the queue/sem assignments (sems 24-27) | Issue the 4 at once instead of back-to-back, or spread over Pool and SP. | Iteration 3 split state_in in two, so a 4-way split is the same lever. | The 4 DMAs occupy 6 us of the head. Ceiling is unknown: it depends on whether PE/ACT waits on them. | Medium: sem-24..27 counts must still balance. | Move 1 DMA earlier and see whether the first MATMUL moves. |
| 6 | Fold ACTIVATION_READ_ACCUMULATOR (24 x 0.29 = 7.0 us) into the neighbouring ACTIVATE | Activation0.bin: the 24 READ_ACCUMULATOR slots | Fuse with the preceding ACTIVATE using the accumulate flag, if the ISA supports it. | Unknown: not established in the notes or docs I read. | Up to 7.0 us of ACT busy. | High: accumulate semantics must match exactly. | Check in `neff_doc/tools/isa.py` whether ACTIVATE has an accumulator-read field before trying anything. |

Rank is by expected gain per risk and effort: 1 > 2 > 3 > 4 > 5 > 6. 1 and 3 reuse mechanisms that are already proven bit-identical on this device.

**Closed** (measured, do not redo):
- Collapsing the 24 per-head silu / k-scale / state-scale ops into wide ACT/DVE ops: done, 153.6 -> 79.6 us.
- ACT_TABLE_LOAD 46 -> 6: done. Do not try to cut below what the table groups allow without checking candidate 2.
- state-out DMA split 8 ways: done.
- Splitting k_f*=k_scale 8 ways and S*=exp_g 4 ways: done (iterations 2 and 3).
- 16 of 24 ACTIVATE f128 + CAST fusions: done (iteration 2).
- The spread is 0.47%, so there is no need to re-run for noise.

**Unknown** (would change the ranking):
- Whether the 16 dropped sem-4 waits this iteration are all safe. Settle it with: x5 verify runs, plus an op-by-op check of each lowered PE threshold against the ACT producer.
- Whether the body is ACT-bound or DVE/PE-chain-bound. Settle it with: removing one ACT op and checking whether the per-head time (2.15 us) drops by the op's duration.
- Which ACT ops share an ACT table, which gates candidate 2. Settle it with: dumping `act_tbl_sel` per ACTIVATE.
- Whether the ACTIVATION_READ_ACCUMULATOR can be fused (candidate 6). Settle it with: reading the ISA fields.
- The fixed share of the 79-80 us tail. Settle it with: profiling an empty-body NEFF if one exists.

Overall, the remaining time is not irreducible. The ACT queue at about 64 us busy leaves room for candidates 1 and 2. Expect the achievable total to be roughly 3-6 us, not more.

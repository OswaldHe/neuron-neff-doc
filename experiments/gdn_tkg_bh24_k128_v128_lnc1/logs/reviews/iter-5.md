# Review: iter-5 (kernel.neff, 69.03 -> 66.28 us)

`notes/` holds only the README template, so the agent recorded nothing about this iteration.
The claims below come from the diff, the profile and the semaphore accounting, not from notes.
The committed baseline for this iteration is HEAD (66f6259, 69.03 us). The uncommitted `kernel.neff` is the candidate.

## 1. What changed (diff HEAD -> kernel.neff, 3 members, 54 slots)

**Activation0.bin: 25 slots.**
- 17 slots went from ACTIVATE (or one CAST) to EVENT_SEMAPHORE.
  - They keep the same wait idx, wait value and `SEM_INC` update. The work is removed but the semaphore count is preserved.
  - Slots: 22, 26, 29, 47, 59, 68, 74, 84, 86, 93, 98, 99, 109, 118, plus a few more.
- Slots 71, 96 and 121 are widened from `num_elem` [1,1,1] to [3,1,1], with src and dst moved to the base address.
  - These are the per-head recip-sqrt (FUNC_29) ops. They are now one 3-head ACTIVATE instead of three.
- Slot 60 (an ACT-side wait on another engine's queue) gets wait_idx 4 -> 6 and value 37 -> 43.

**DVE0.bin: about 20 slots.**
- The consumers of the batched rsqrt were `[128,3]` TENSOR_TENSOR ops. They are now `[128,1]`, applied per head (num_elem [128,3,1] -> [128,1,1]).
- Several TT / STT / LOAD_MASK_SELECT / STREAM_SHUFFLE ops are re-ordered in the queue, so the opcode changes at those slot positions.

**PE0.bin: 9 matmul gates.**
- Wait idx goes 4 -> 6, i.e. from the ACT semaphore to the DVE semaphore, with new counts (30 -> 38, 49 -> 54, 68 -> 65, and so on).

The change is "batch the per-head recip-sqrt into 3-head ACTIVATEs and let PE gate on DVE completion".
This is consistent with the previous commit's style, but nothing in notes confirms it.

## 2. Is the win real?

Yes.
- `profile_latest.json`: median 66.278 us, samples 66.278 / 66.270 / 66.278 / 66.420 / 66.291, spread 0.23%, so noise is about 0.15 us.
- My re-run gave 66.40 us.
- HEAD was 69.03 us, so the gain is 2.6 to 2.75 us. That is about 17 times the spread, and not noise.
- Against the unedited baseline (153.46 us, spread 0.79%) the speedup is 2.31x.

## 3. Safety

**Verifier.**
- `max_abs_err` is 2.322e-2 (83% of the 0.028 limit), identical to the baseline's 2.322e-2.
- output_1 is 1.633e-3, also identical to the baseline.
- That points to bit-identical outputs. No error has moved.

**Semaphores.**
- `semaphores.py` on the unpacked copy shows no "wait exceeds increments" line.
- Sem 4 (ACT) totals 120 increments with max wait 120. Sem 6 (DVE) totals 103 with max wait 103. Both balance.
- Moving ACTIVATE to EVENT_SEMAPHORE kept the increment counts, which is why this holds.

**The real risk: 9 PE gates retargeted from ACT sem 4 to DVE sem 6.**
- It is only safe if DVE's n-th completion really implies that the ACT write PE reads has landed.
- That is true only through a transitive chain: ACT write, then the DVE op waiting on ACT, then DVE completion.
- The count alone does not prove it. The count passed on one input. A race would show up as intermittent wrong values, not necessarily a fixed-input failure.
- Check: for each of the 9 PE slots (13, 19, 23, 40, 67, 77, 79, 93, 103), name the operand's last writer and confirm that the DVE op whose count is waited on has a wait on that writer's ACT count (or a later one).
- Also run the verifier on 3 or more random seeds.

The rest of the semaphore edits are neutral: ES placeholders with unchanged waits and updates are safe.

## 4. Ranked candidates for next iteration

Context from the profile (my re-run, trace `s` in us):
- The trace has about 10 us of fixed preamble.
- The ACT-bound head runs 10.5 to 28.
- The 24-head body runs 28 to 75.8, about 2.0 us per head.
- The output DMA starts at 76.5 and runs to 76.9, then about 2 us of fixed epilogue.
- Per-engine busy in the body is about 65-80% on ACT, 60-65% on DVE and 50-70% on PE. Nothing is saturated, so the body is dependency-latency bound.
- ACT per head is roughly FUNC_30 0.27 + READ_ACC 0.29 + FUNC_29 0.2 + 2 CAST (0.3 each) + COPY 0.49 every third head.

| # | Candidate | Engine / member / slots | Change | Mechanism and evidence | Ceiling | Risk | Cheapest falsifier |
|---|-----------|------------------------|--------|------------------------|---------|------|--------------------|
| 1 | Run silu(z) over many partitions instead of 1 | ACT, `Activation0.bin` slot 4 (the FUNC_36 / silu_32p ACTIVATE at 13.79); plus the z DMA descriptor and the consumers of the result at `fp32@0x20000` | Today `num_active_channels=1`, src and dst `[3072,1,1]` on one partition (2.762 us). Re-lay z as 24x128 or 128x24 and set channels to match, so the op takes about 0.2-0.3 us | ACT cost scales with elements per partition, so cutting 3072 to 24-128 per partition cuts the time. The ACT head is back-to-back (table loads and this op), so it is the critical resource there. | Up to 2.76 - 0.3, about 2.4 us if the head stays ACT-bound. Bound: the ACT head window 10.5 to 28 is 93% busy (notes from the sibling lnc2 experiment, same kernel; I did not re-verify the 93% here). The measured 2.762 us is this op's duration. | High. Needs a different SBUF layout for z, which means DGE descriptor edits (`qSyncIO` input DMA) and re-pointing the consumers. A wrong layout would likely fail the verifier, but a partially wrong stride could pass on a lucky input. | Find who writes `0x834e00` and who reads `0x20000`. If the readers need z as one row on partition 0 (broadcast), the candidate is dead. Check this in about 10 minutes from the DMA and consumer operands. |
| 2 | Widen the rsqrt batch from 3 to 6 or 8 heads | ACT slots 71, 96, 121 (the batched FUNC_29) and their DVE consumers | `num_elem` [3,1,1] -> [6,1,1] (or 8), delete the second ACTIVATE of each pair, and reorder the DVE consumers | Same mechanism as this iteration: one ACT op over several elements, with the per-op fixed cost paid once. This iteration got about 2.7 us from it. | ACT FUNC_29 is about 0.2 us x about 24 heads = 4.8 us total, so cutting 16 to 18 launches frees about 3 us of ACT time. Wall gain is probably 0.5 to 1.5 us, because the body is dependency-latency bound, not ACT-bound. Unknown beyond that. | Medium. Delaying the rsqrt for early heads lengthens their chain, and the PE and DVE gates need retargeting again. The semaphore counts must stay balanced; run `semaphores.py`. | Count the ACT FUNC_29 ops still unbatched, then batch just one group and look at the median. If the median is flat within 0.15 us, the body is not ACT-limited and the family is closed. |
| 3 | Remove the placeholder EVENT_SEMAPHOREs (now 17 on ACT) | ACT, the 17 slots listed in section 1 | Merge each one's wait / update into the neighbouring real op on the same queue, so the slot is not needed | Each costs about 40 ns plus a possible stall on its wait. Moving the update onto the previous op's event field avoids the extra instruction. | At most 17 x 0.04 = 0.7 us of ACT time, wall gain less. | Medium-high. Slots can't be deleted without shifting, so the work is merging fields. A wrong merge breaks a count (`semaphores.py` catches that, an ordering race it doesn't). | Check in the profile whether those ES ops sit on a wait: if their start-to-previous-end gap is about 0, they cost only 40 ns and the candidate is worth under 0.2 us. |
| 4 | Split the last state-out DMA unevenly | Pool, 8 `DMA_DIRECT2D`, last at 74.04 | Make the last chunk 1 head instead of 3 | The tail from the trigger at 74.04 to the first packet at 75.5 is 1.46 us, of which the Pool instruction itself is 0.88 us (it matches the chunk size). | At most 0.5 us, and probably less: the output DMA at 76.5 is the real end, and the 0.7 us between its trigger (75.79) and first packet is a fixed DMA latency. | Low. | Check whether `DMA_DIRECT2D` duration scales with bytes (the 8 chunks are equal at 0.88 us, so look at the 12-instruction set including the input pair at 1.73 and 1.25 us). If it doesn't scale, drop it. |
| 5 | Shrink the final output DMA | qSyncIO queue E65 (6168 B; the others are 24 B) | Move the E65 share onto other queues so all 16 queues finish together | E65 ends at 76.899, the others at 76.65 to 76.77 | About 0.15 to 0.25 us. | Low | Compare the per-queue end times on a second run; if E65 is not always last, drop it. |

Ranking is by gain per risk and effort.
- #2 is first in practice, because it is the same edit family that just worked and is the cheapest to test.
- #1 has the highest ceiling but may be structurally blocked, so spend only the 10-minute check on it first.
- #3 to #5 are small. Take them only if the earlier ones die.

## 5. Closed (do not redo)

- **Table loads: 6 is the floor.** The profile still shows exactly 6 `ACT_TABLE_LOAD` at 1.283 us each = 7.7 us. The sibling experiment's notes state that no vendor table set holds both the rsqrt and silu functions, so 6 is the structural floor. That claim is from `neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2`, not re-measured here.
- **Pool as a host for compute.** Closed by the sibling lnc2 notes ("Pool as a host for anything", iter-6). I did not verify it on this case.
- **Splitting the state DMAs.** The 8-way split of the 1.5 MB state-out is done. Per-queue transfer ends within about 0.2 us of each other, so there is nothing left.
- **Error margin.** 83% of the limit is a bf16 floor, identical to the baseline. Any numerics-changing edit (a different table, bf16 intermediates) risks it.

## 6. Unknown (would change the ranking)

- **Is the head ACT-bound or DMA-bound?** In the profile, FUNC_36 starts right after its table load (13.79 vs 13.78 us), and the DMA wait S[11]>=16 was already satisfied. If the head is limited by something other than ACT, #1's ceiling drops. Measure: delete FUNC_36 from the stream (use the placeholder trick) and see how much the median moves. If it drops by 2.4 us or more, #1 is real.
- **Is the body latency-bound?** Measure with the one-group experiment in #2. A gain under 0.15 us means ACT is not the limit and the whole ACT-trimming family is closed.
- **Is the DVE `TENSOR_TENSOR` of 3.08 us at 16.59 (head) splittable?** It is the longest DVE op. Its duration and consumer window are not examined here. Check whether it is on the head's critical path.
- **Is the PE gate retargeting transitively safe?** See section 3. The 9-slot writer audit would settle it, and should be done before building on this iteration.

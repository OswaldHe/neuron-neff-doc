# Review of iter-6 (kernel.neff 4777.8 -> ~4683 us; baseline 7154.9)

## 1. What changed, and does it match the notes?

`neff_diff.py HEAD:kernel.neff kernel.neff`: 35 of 37 members identical. Only `sg00/Activation0.bin` (14074 -> 14074 slots, 1568 differ) and `sg00/DVE0.bin` (24525 -> 24525, 3506 differ) changed. PE, SP, Pool, all DMA descriptors and all JSON are untouched. Slot counts are unchanged, so nothing was compacted.
Against the original baseline the same two plus PE and SP differ, but that is the earlier iterations.

The notes describe three edits:
- ACT: merge the two `*= exp_cs_col` ACTIVATEs into one at the later slot, with a 2-group 3-D pattern. The earlier slot becomes an EVENT_SEMAPHORE that keeps its wait.
- DVE: rewrite the `Y_off` ADD as a COPY from PSUM and shrink the MEMSET to 1 element.
- DVE: repoint the consumer of the second `exp_cs_last` broadcast at the first broadcast's buffer, and turn the dead STREAM_SHUFFLEs into no-update ES.

The slot-level diff is consistent with that:
- ACT now has 3602 wait-only ES, 967 ES with `SEM_INC_COMPLETE`, and 5 ES with no wait.
- DVE `COPY` went from 1431 to 2464 and `TENSOR_TENSOR_ARITH_OP` from 2950 to 1917 (+1033 / -1033, matching the 1040-site rewrite).
- DVE has 2829 ES with no wait and no update, which are the retired shuffles.

I did not decode each of the 708 + 1040 + 384 sites individually. The aggregate agrees with the notes.

## 2. Is the latency change real?

Yes.
- `profile_latest.json`: median 4683.076 us, samples 4681.4-4684.0, spread 0.055%.
- My independent 1-repeat profile gave 4681.4 us.
- The previous accepted commit measured 4777.85 us, so the gain is -94.8 us (-1.98%). That is about 40x the spread.
- The baseline is 7154.9 with 0.02% spread.
- Engine busy moved the way the notes say: ACT 3344 -> 3112, DVE 2427 -> 2176, the rest flat.
- The steps were reproduced on two profile runs each (notes' table), and the last two sub-steps are only ~1 us apart (4681.9 vs 4683.0).

This is not noise. The remaining wins are small, though: the last steps are 20-45 us each, against 2-4 us of spread.

## 3. Safety beyond the verifier

- **Verifier:** `max_abs_err` is 5.234e-07 against a limit of 6.2e-07 (84%). It is unchanged to the last digit from iter-5, as expected for bit-exact edits. No error headroom was consumed by this iteration. It is still tight for any edit that changes arithmetic.
- **Semaphores:** I ran `semaphores.py` on the HEAD and new unpacked NEFFs.
  - Per-semaphore max_wait, total_inc and waiter counts are identical.
  - The only difference is the source-opcode mix on sem 6 (COPY +1033, TENSOR_TENSOR_ARITH_OP -1033).
  - No increment was moved or removed, and no threshold was renumbered. The shrunk MEMSETs keep their compute-completion increment.
  - The 'sems 40-71 wait exceeds increments' alarm is the documented false positive.
- **Residual risk is in the WAR and ordering arguments, which the verifier covers only for one input set:**
  1. Repointing the consumer to `g1`'s buffer relies on a WAR check. The notes show that a missing check gave `7.9e-02` error, so the failure is loud when it occurs. The 384 shipped sites use a static ordering proof. The 24 timing-only sites were dropped.
  2. Merging at the later slot relies on a static "nothing touches the earlier tile in between" test. The test passed at 708 of 708 sites.
  3. The 95 merge sites whose two waits are on different semaphores depend on the placeholder ES keeping `si`'s wait. This is sound by in-order issue.
- Race-style bugs would typically show up as the large errors seen in the notes, not as silent pass-then-fail. Ordering was shown to be data-independent, since there is no data-dependent control flow. The edit was verified 3x, per the notes.
- **Overall:** low risk. A cheap extra check is to run `verify.py` with 2-3 different random seeds if the tool allows it, because the WAR guards are static.

## 4. Candidates for iter-7 (ranked)

State after this iteration (my profile, span ~4696 us): ACT busy 3694 (incl. DMA/ES), DVE 2654, PE busy ~2.2 ms net of overlap, Pool 2103 (almost all `DMA_DIRECT2D`, 3169 x 655 ns, 2.5 ms of >1 us gaps), SP 534.
Conversion rates measured in the notes: ACT element-time 69%, ACT instruction deletion 20%, DVE deletion 16-26%.

### C1. Compact out the DVE no-op ES slots (and the ACT dead-slot analogue)

| item | detail |
|---|---|
| Engine / member | DVE, `sg00/DVE0.bin`. 2829 `EVENT_SEMAPHORE` slots with wait NONE and update NONE (the retired shuffles from iter-6 §10, plus the originals). |
| Change | Delete the slots and shift the stream. |
| Mechanism | iter-1 already deleted 1020 dead ACT slots and shrank the member (985408 -> 900736 bytes). No stream has pc or branch fields (`iter-1.md` L116). A NONE/NONE ES cannot release or block anything, so no semaphore renumbering is needed. |
| Gain / ceiling | DVE ES in total is 190.6 us over 3716 slots (51 ns mean, my profile). The NONE/NONE share is at most 2829/3716 of that, ~145 us, and likely less (an ES with no wait is cheaper than the mean). At 16-26% DVE conversion, ceiling ~25-40 us, realistic ~10-20 us. |
| Risk | Low. The one catch is whether the runtime keys anything to slot index (the profiler maps `compiler_pc`, and DGE/DMA descriptors reference queues not PCs). The verifier catches a broken stream because it would hang or error. |
| Cheapest falsifier | Delete 100 of them, repack, run `verify.py` and a 1-repeat profile. Dies in ~3 minutes if the runtime rejects the shortened stream or the wall time does not move. Also: just measure the real ns of the no-wait ES by joining slot index to the parquet. |

### C2. Software-pipeline the serial head chain / fill the Y-multiply bubble

| item | detail |
|---|---|
| Engine / member | ACT + DVE + PE streams. |
| Change | Move instructions across units so ACT works while it waits on sem 6/5 (needs a sem-4 renumbering pass). |
| Mechanism | Notes iter-6 §5, §11.3. sem 4 is ACT-only, all waits are `GE_IMM`, so deleting an increment at ACT slot m allows the exact remap `T -> T-1 if T > m` (~9700 waits). |
| Gain / ceiling | ACT idle is 1314 us, of which ~1129 us is 1284 stalls. 592 us of that is the merged-Y site. The real ceiling is the gap between the data dependencies and the schedule, which is unknown. Cross-unit hoisting was not measured. |
| Risk | High. Wrong moves hang or race. Mild effort cost: the renumber pass has not been validated on a single site yet. |
| Cheapest falsifier | Build the renumber pass on one site (delete one ACT increment, apply the remap to all 9700 waits, check `verify.py` and the profile). If it hangs or errors, the primitive is dead and the whole line is closed. |

### C3. Remaining ACT merges: 15 odd units + 25 extra generic pairs

| item | detail |
|---|---|
| Engine / member | ACT, `iter-6-merge-generic.py`. |
| Change | Same recipe as §0, applied to the leftovers, and drop the `src_delta == dst_delta` restriction. |
| Gain / ceiling | ~40 x 325 ns of ACT busy at 20% conversion, ~2 us. This is under the noise floor; take it only if combined with another edit. |
| Risk | Low. |
| Falsifier | Run it once; if less than ~3 us, stop. |

### C4. Retire the iter-3 dedup `COPY` on the 545 units

| item | detail |
|---|---|
| Engine / member | DVE, those units' `exp_cs_last` dedup COPY. |
| Change | Repoint the consumer like iter-6 §10 and shrink the COPY. |
| Gain / ceiling | Only 73 of 545 copies have no sem-6 update, ~2 us (iter-6 §10a). |
| Risk | Low to medium (WAR check is required). |
| Falsifier | Same as C3. |

### C5. Pairwise merge of the two DVE PSUM->SBUF copies

Notes (iter-6 §11.5) price this at ~3 us (362 of 1004 sites fit the one-wait constraint). It would be bigger (~30 us) only if the sem-6 problem is solved generally, which is a prerequisite shared with C2.

I found no PE lever: all 6209 MATMUL are `FP32 LOW_HIGH` and the LDWEIGHTS come in LOW/HIGH pairs. Changing `fp32_mode` or `perf_opt` changes numerics, and the error is at 84% of the limit. The 3558 "repeated" LDWEIGHTS addresses are the LOW/HIGH passes, not redundant loads.

## Closed (numbers from notes)

- `exp_cs_last` recomputation: pairing solved, but the probe hangs the device (NRT_TIMEOUT). iter-5's -194.8 us is unreachable.
- Nulling redundant ACT waits: -2 us for 201 sites; ceiling ~35 us even at 100%. Handoff latency is worth only 208 us in total.
- DVE 3-D-pattern merging: only 136 pairs, ~2-7 us.
- `Y_off` MEMSET + add replaced by an ES (+28.9 us regression) or by renumbering (hang): the shrink form is the one that works.
- Moving multiplies ACT -> DVE costs 1 ulp; ACT may not drain a LOW_HIGH matmul's PSUM (PSUM_COLLISION).
- DGE threshold remaps (iter-4 §2).
- Merging at the earlier slot (iter-4 §5 failed on device).
- Cross-unit ACT pair merges (2863 rejected; they reuse addresses with different values).

## Unknown

- **Real cost of the DVE NONE/NONE ES.** It decides whether C1 is ~5 us or ~30 us. Measure by joining slot index to `Instruction.parquet` durations.
- **Whether the Pool queue is on any critical path.** Pool has 2.5 ms of >1 us gaps and 3169 DMAs at 655 ns each, so it appears idle. A probe is to set 10% of the DMAs' sizes to 1 element (destructive) and see whether the wall moves. If not, DMA is irrelevant, which would be good to know.
- **The sem-4 renumber primitive.** Single-site test, as in C2.
- **Error headroom.** With 84% used, nothing that changes arithmetic is viable. Whether the limit was set on a tolerance that could be exploited is outside the NEFF.

If C1 is dead and the renumbering primitive fails, the remaining time is mostly data-dependency latency (ACT waits 727 us on DVE and 257 us on PE) and is not reducible by local edits.

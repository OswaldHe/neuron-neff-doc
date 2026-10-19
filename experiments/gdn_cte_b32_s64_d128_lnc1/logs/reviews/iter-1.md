# Review of iter-1 (gdn_cte_b32_s64_d128, LNC=1)

## 1. What changed
`neff_diff.py` reports 5 changed members (29 identical):

- `sg00/def.json`: two new DMA queues, `qSPSpillReload1` and `qSPSpillReload2`, each with 16 semaphores (72-87 and 88-103). Only `qSPSpillReload0` existed before.
- `sg00/SP0.json`: the `dma` list grows from 222 to 543 entries. 222 stay on `qSPSpillReload0`, 160 are on `qSPSpillReload1` and 161 on `qSPSpillReload2`. SP0.json grows from 96 KB to 215 KB. The entries are pre-expanded transfers, for example SB→SB fp32 transposes (`transpose_op WZXY`).
- `sg00/Activation0.bin` (160 of 1823 slots differ), `Pool0.bin` (459→1121 slots) and `SP0.bin` (1366→1329 slots):
  - `PSEUDO_DMA_DIRECT2D` (software-DGE) slots become `PSEUDO_DMATRIGGER` of the pre-built queued transfers (13 slots) or `EVENT_SEMAPHORE` (33 on Activation, and others on Pool).
  - The `EVENT_SEMAPHORE` slots use `SEM_ADD_IMM_COMPLETE` (`events_update_mode`, +16, i.e. 0x10) to repay the DGE semaphore increment, plus an `events_extended_wait` on the new queue semaphores.
  - Other Pool slots are rewritten to carry waits and updates.

So the DGE descriptor generation that used to run on the engine streams (SP, Pool and Activation hosts) became pre-built queued DMA rings.

The profile agrees:

| | baseline | iter-1 |
|---|---|---|
| SP busy | 479.7 us | 34.4 us |
| Activation busy | 414 us | 346 us |
| Pool busy | 187 us | 166 us |
| DMA transfer count | 222 | 543 |
| "spill bytes" (profiler classification) | 3.9/4.5 MB | 0 |

HBM read/write bytes also drop, but I take that as a classification change, since `dma_transfer_total_bytes` rose from 1.87 MB to 5.96 MB.

**`notes/` contains only the template README.** The agent wrote no notes, so there is no claim to check against, and the next agent has no record of the method. This is a process failure. The commit message and notes need to say "DGE→queued DMA conversion".

## 2. Is the latency change real?
Yes.

- Baseline: 1184.96 us median, spread 0.145%.
- iter-1: 874.47 us median, samples 872.7-876.9, spread 0.49%.
- Gain: 310 us (-26.2%). The noise is about 4 us, so the gain is about 75 times the spread.
- My own re-run gave 871.6 us.

The loop is not chasing noise at this size. Wins below roughly 1% (about 9 us) are within about 2x of the spread, so the next agent should use `--repeats 5` or more and compare against this iter-1 NEFF, not the baseline.

## 3. Safety
- `semaphores.py` on the unpacked iter-1 NEFF shows no semaphore with `max_wait > total_inc`. That holds for the new 72-103 and the old 8-23. Semaphores 21 and 22 have max_wait 240 against supply 256, which is fine. This check does not catch sync that is merely too weak.
- `verify_latest.json`: all checks pass.
  - output_0 max error is 7.7e-5 (11% of the limit), identical to baseline.
  - output_1 max error is 5.84e-4 (**82% of the limit**), also identical to baseline to every printed digit. The arithmetic is untouched, but there is very little headroom, so any precision change (fp32→bf16 on a matmul operand, say) would likely fail.
- Residual risks:
  - Only one verify run was recorded. A queued ring has no engine-side ordering with the consumer, so correctness depends entirely on semaphore waits. A wait loosened here can pass on one run and race later. Run `verify.py --runs 5` or more before trusting it.
  - Queued DMAs now run in ring order on 3 queues. The old DGE could issue out of order, so this is stricter, not looser.
  - I found no waits dropped without a replacement. The Pool slots 26 and 31 moved their wait into `events_extended_wait` on semaphores 89 and 90, and the counts are conserved.

## 4. Ranked candidates for the next iteration
The profile after iter-1 is a PE↔DVE/ACT ping-pong chain, not a throughput limit.

- Real work per engine, with semaphore stalls excluded: PE MATMUL 455 us + LDWEIGHTS 171 us = 626 us (72% of 874), DVE about 455 us, ACT about 380 us.
- Gaps over 200 ns in the instruction table:
  - PE stalls 582 us, mostly waiting on DVE (sem 6, 236 us) and ACT (sem 4, 72 us).
  - DVE stalls 288 us waiting on PE (sem 5).
  - ACT stalls 268 us waiting on PE and 163 us waiting on DVE.
- Load is uniform over time, about 75-80 us of PE activity per 100 us bucket. There is no phase with a single dominant stall.
- The 32 batch items are independent, but the in-order streams serialise each item's chain.

| # | Edit | Mechanism and evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Delete or merge dominated EVENT_SEMAPHOREs on the critical engines.** ACT stream has 356 (67 us busy), PE 347 (32 us), DVE 173 (13.5 us). Candidates: waits already implied by an earlier wait on the same semaphore with a higher count, and semaphore-only slots whose wait and update can be folded into the neighbouring slot's `events_*` fields. | The other GDN experiment merged them in iters 4-5 for -1.1% and -1.4%. `SEM_ADD_IMM_COMPLETE` is device-verified there. | About 110 us of engine busy time at most. Only the part on the PE↔DVE chain counts, so I expect 1-3%, about 10-25 us. | Medium. Deleting a non-dominated wait passes once and races later. Per-semaphore counts must stay unchanged and `semaphores.py` must pass. | Build the dominance map for ACT's 356 slots. If fewer than about 50 are provably dominated, drop it. |
| 2 | **Merge affine runs of identical instructions into wider-pattern slots, then delete the followers.** Candidates: the 1-element ACTIVATE/CAST runs on ACT (CAST is 896 slots at 160-300 ns) and the DVE COPY/CAST (544 + 518 slots, about 200-250 ns). | Per-instruction fixed overhead. The other experiment got -1.13% from this in its iter 4. | Unknown. Ceiling is bounded by the DVE+ACT CAST/COPY total (about 330 us) but only the fixed-overhead share is recoverable. Measure with a two-instruction merge. | Medium. The merged AP must exactly cover the same elements and the semaphore increments must be preserved. | Find one run of N identical slots, merge it, and measure the per-slot saving. If it is below 20 ns per removed slot, drop it. |
| 3 | **Pull DMA waits earlier / reorder the reload triggers.** ACT CASTs still wait on `qSyncDynamicHW`/`qSyncSpillReload0` (sem 24-35, 49-70; about 50 us of ACT stall, PE stalls of 16-18 us per sem on `qSyncDynamicHW` 60-70). | The PSEUDO_DMATRIGGER for each reload can issue before earlier compute finishes if its source is already written. Check the `wait_idx` it carries. | About 50-80 us if the waits are on the critical path, otherwise near 0. The measured stall sum bounds it. | High. A moved trigger can read data before it is written and pass the verifier on one input. | For the 5 largest PE stalls on `qSyncDynamicHW`, check whether the trigger's own wait could be satisfied earlier. If the trigger is already issued before the consumer arrives, the lever is dead. |
| 4 | **Software-pipeline two batch items across the engine streams** (interleave item i+1's PE slots into item i's DVE/ACT stall gaps). | The 580 PE stalls over 200 ns total 582 us and the chain is dependency-bound. | Biggest: PE real work is 626 us, so the floor is about 650-700 us, about 20% gain. | Very high. It is a full instruction reschedule and SBUF/PSUM allocations may collide. Only a multi-input verifier would catch races. | Check whether PSUM tiles (0x2000000 and up) are reused between consecutive items. If each item reuses the same PSUM address, interleaving needs allocation changes and is not cheap. |
| 5 | **LDWEIGHTS reuse.** Skip a LDWEIGHTS when the next MATMUL's stationary operand is identical to the loaded one. | 1742 LDWEIGHTS cost 171 us, mostly 80 ns (1059) and 130 ns (460). | At most 171 us on PE. Matmul and LDWEIGHTS counts are nearly 1:1, which suggests few repeats. | Low if the operands are provably identical. | Count consecutive LDWEIGHTS with identical source address and shape. If it is under about 100, drop it. |

The lever I would not pursue is moving the fp32 matmul operands (src `fp32@...`, 260-320 ns versus about 170 ns for bf16) to bf16. It would give roughly 100 us on PE, but output_1 is at 82% of its limit, so it would very likely fail.

### Closed
- **DGE on SP/Pool/ACT streams.** This was the baseline's dominant cost: SP 479.7 us busy. It is now 34 us, a 310 us gain, and cannot be redone.
- **Hosting DGE on Pool.** In the s2304 experiment, Pool was the worst DGE host, and it hit 14771 versus 14501 for ACT+SP. That experiment's log no longer applies here because the DGE is gone.
- **Spill staging fusion** (s2304 iters 3-6). The DMA transposes are mandatory (`nc_matmul` already transposes the stationary operand; `transpose_mode` is `nc_transpose`). This is the same kernel family, so I would not retry it.
- **Reduced precision of the matmul operands.** Closed by the 82% output_1 error margin.

### Unknown
- Which of the 834 Pool and 789 SP `EVENT_SEMAPHORE`s are on a chain. Measure with `semaphores.py` and the dominance analysis from candidate 1.
- Whether PSUM tile addresses repeat between items, which decides candidate 4. Read the PSUM destination of the 2048 MATMULs (`dst=0x2000000...` in `operands`).
- Per-slot saving from merging two identical slots (candidate 2). One merge and 5 profile repeats.
- Whether `verify.py --runs 5` is stable on this NEFF. Do this now, before building on it.

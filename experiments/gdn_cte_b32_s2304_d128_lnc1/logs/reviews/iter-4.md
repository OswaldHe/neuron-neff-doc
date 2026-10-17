# Review of iter-4

Note: `baseline/kernel.neff` is the iter-0 NEFF. HEAD already holds iters 1-3, so I diffed against `git show HEAD:kernel.neff` as well. Against the baseline, 4 members differ (the cumulative iter 1-4 work). Against HEAD, only 2 differ.

## 1. What changed vs. what the notes claim

HEAD -> working tree (`neff_diff`): 2 of 34 members changed, as `iter-4.md` claims.
- `sg00/Activation0.bin`: 30789 -> 30069 slots (-720).
- `sg00/DVE0.bin`: 29896 -> 29623 slots (-273).
- `Pool0`, `PE0`, `SP0.bin`, `SP0.json` and `def.json` are identical to HEAD.

The diff reports about 30k "differing" slots. That is the index shift after deleting slots, not 30k edits.

The first slots show the intended mechanism. A single-element ACTIVATE gets `src num_elem [128,1,..] -> [128,2,..]` and `dst step_elem[1] = 2560`. The followers are removed. That is 993 deleted slots (720 Activation, 273 DVE), matching the notes.

Consistent with the notes: the edit is "widen the head's access pattern, delete the followers". Nothing else was touched.

## 2. Is the latency change real?

| NEFF | total_exec_time_us |
|---|---|
| iter-0 baseline | 14779.0 (spread 0.009%) |
| iter-3 (HEAD) | about 14393 (notes: 14387 / 14393 / 14510) |
| iter-4 `profile_latest.json` | 14231.9 (14227.8 / 14232.1 / 14231.9, spread 0.03%) |
| my re-profile just now | 14233.0 |

- Iter-4 vs. HEAD is -1.1% (about 160 us). That is more than twice the 0.5% noise bar. The notes report ten of eleven captures at 14214..14240, all below every HEAD capture. It is real, and I reproduced 14233.
- Cumulative vs. baseline is -3.7%.
- The HEAD figure is the agent's own measurement. I did not re-profile HEAD.
- Instruction counts: Activation 30812 -> 30092 and DVE 29920 -> 29646 against the baseline profile, matching the deletions.

The loop is not chasing noise here. The remaining levers listed below are, with one exception, each under 1%. Expect the loop to start chasing noise soon unless it takes the transpose/fusion route.

## 3. Safety

- `semaphores.py` output (HEAD vs. new) is byte-identical across all 70 lines. No wait or increment count moved. Followers carried no waits and no updates, so no ordering point was crossed. I checked this myself on fresh unpacks.
- Verify passed.
  - output_0 max_abs_err is 1.37e-4 (17% of the limit).
  - output_1 is 6.84e-4 (86% of the limit).
  - The notes say both are identical to the baseline to every printed digit, so the merge did not move error. Output_1 at 86% is already a baseline property, so there is little headroom for any numerics-changing edit.
- The residual risk is small. A merged run reads or writes all its elements at the head's time instead of spread over a few hundred ns. This is only wrong if another engine writes the follower's source in that window without synchronisation. That would be an unsynchronised race in the original, and it is not a legitimate dependence. The verifier is a single fixed input, but the edit is data-independent.
- The margin concern is on the output_1 budget, not on this edit.

## 4. Candidates for iter-5 (ranked by gain per risk and effort)

Profile facts that bound everything below (my profile, `Instruction.parquet`, deduped by slot):
- Span is 14242 us.
- Per-engine busy (sum of `duration_ns`):
  - Scalar 8986 us (TENSOR_SCALAR 3308, ACTIVATE 3223, CAST 1191, DMA_DIRECT2D 887).
  - Tensor 7067 us.
  - Vector 7054 us.
  - Sync 3451 us (DMA_DIRECT2D 2361).
  - GpSimd 2894 us (DMA_DIRECT2D 2198).
- The all-engines-idle time is 1321 us across 14557 gaps (mean 90 ns, max 4.3 us). That is sequencer latency and handshakes, not one big stall.
- No engine is occupancy-bound. The span is a dependency chain through the spill/reload DMAs plus per-instruction sequencer cost.

| # | Candidate | Where | Change | Mechanism | Ceiling | Risk / verifier | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Finish the ACTIVATE/COPY merge across non-consecutive windows, gated by a static alias checker | `Activation0.bin` and `DVE0.bin`. Candidates are the 514 slots (346 Act + 168 DVE) from the `iter-4-merge.py` windows that cross a semaphore increment. | Move a member next to its run head, then widen the head as in iter-4 and delete the follower. | iter-4 merge conversion was about 78% (162 us gain for 208 us of issue). The check: for every writer (compute dst, DGE dst, queued `to`), the profile's `(start_ts, end_ts)` must not overlap the moved range in partitions, bytes and the time between the original and new position. | About 104 us (0.7%) from the notes' 78% conversion. It is derived from the merge result, not measured for these windows. 106 slots are already safe without the checker, about 20 us. | HIGH. Pass-one-input, wrong-later is possible. The verifier will not catch an alias mistake. Needs the DGE address decode (`dge2.py`-style VAR/TBL markers) or the checker is incomplete. | Run `iter-4-merge.py` in dry-run mode for the 106 provably-safe slots only: if the gain is under 15 us, the checker is not worth building. |
| 2 | Fold consecutive Activation `EVENT_SEMAPHORE` pairs into one via `events_extended_wait_*` | `Activation0.bin`, the 343 qualifying pairs | One EVENT_SEMAPHORE carrying both waits, delete the second. | A conjunction of two sequential waits is equivalent. Documented in the ISA (`events_extended_wait_*`). | About 30 us (0.2%), the notes' estimate. This is inside the noise band on its own. | LOW. Counts are unchanged, so `semaphores.py` must stay byte-identical. | Do 20 pairs. If the semaphore table changes or the device hangs, kill it. |
| 3 | Save/reload DGE fusion (iter-3's unfinished work) | `SP0.bin` / `Pool0.bin` / `Activation0.bin` DGE slots. 319 groups are already analysed. | Retire a reload slot by deleting it and re-adding its +16 through an `EVENT_SEMAPHORE`'s `events_extended_sem_update_value`. | Shrinking a DGE slot hangs (iter-3). `events_extended_sem_update_value` is used nowhere in this NEFF and is unverified on the device. | Unknown. The bounds: SP DGE total simulated at -423 us (upper, all slots), 4.8 ms of DGE sequencer time. Realistic is a fraction of that. | MEDIUM. Semaphore arithmetic errors hang or race. A hang is caught by the verifier. A race is not. | Probe `events_extended_sem_update_value` on 8 slots. If the +16 does not land (S[n] count short in `SemaphoreUpdate.parquet`), the lever is dead. |
| 4 | PE operand-role swap to delete the DMA transposes that feed only a stationary operand | `SP0.json` and the PE stream. 689 transposes, (starting at `SP0.json` dma[0]: SB 227072 -> SB 16896, sem 24). | Point LDWEIGHTS at the untransposed source, swap the moving operand, delete the descriptor and lower every later wait on that semaphore by 16. | PE contracts the partition axis of both operands (hints). The result comes out transposed, so the consumer must change too. | The hint's 22% is the upper bound for all 8125. The iter-3 review scoped the reachable part to 689 transposes (866 us of queue time, a few hundred us of PE time). Realistic ceiling is a few hundred us. | HIGH. A whole iteration. Bit-exact is the bar. Aliasing can pass on one input. | Single instance on dma[0]: if the PSUM consumer chain does not close inside the chunk, stop and record it. About 2 minutes of device time. |
| 5 | Merge the 160 signalling followers by replacing them with EVENT_SEMAPHOREs | Activation / DVE | Replace instead of delete. | Same mechanism as iter-4, keeping the increment. | About 30 us (notes). | LOW | Only worth batching with candidate 2. |

Rank rationale: 1 and 4 have the highest ceilings but are high risk. If I were the next agent, I would run candidate 4's single-instance test first, because it is two minutes of device time, it has the only multi-hundred-us ceiling, and it also closes or opens the biggest remaining question. Candidate 3's probe is the second quick test. Candidates 2 and 5 are low risk but are each at or below noise, so bundle them with another edit and do not spend a measurement cycle on them alone.

## Closed (do not redo)

- Move Activation CAST/TENSOR_SCALAR to Pool: the simulator gave -21 us idealised (+26 us realistic). On device Pool is 14.1 ns/elem for TENSOR_SCALAR against 0.83 on Activation, and ACTIVATE hangs on Pool (status 1004).
- Move compute to DVE: -50 us simulated, below the noise floor.
- ACTIVATE -> COPY/CAST downgrade: `activation_func=7` is exp, and 128 is affine copy with scale. No cheaper opcode has a scale field.
- DGE slot merging: of 8383 adjacent DGE pairs, 0 are mergeable (they round-robin over 16 semaphores). Same-semaphore slots are 40-200 apart and every DGE increments its semaphore.
- DGE host ratio (iter-2), `def.json` DMA queues, DMA priority (iter-3), trigger/wait tightness (iter-1), `dge_op=DMA_TRANSPOSE` (iter-1), queued-path SBUF sources (+199 us).
- Free SBUF: the 45,696-byte run is gone (iter-3 spent it). Re-derive the map from the current streams. Do not trust the old note.

## Unknown (would change the ranking)

- Whether `events_extended_sem_update_value` works on the device. Settles candidate 3. Measure: 8-slot probe.
- Whether the transpose role swap closes within a chunk. Settles candidate 4. Measure: the dma[0] single-instance test.
- The real conversion rate for the 514 blocked-window slots. Settles candidate 1. Measure: the 106-slot subset first.
- How much of the 1321 us all-idle time is handshake latency that a smaller semaphore-ring count could shave. No measurement exists. I do not recommend chasing it blind.

Overall: after iter-4, the cheap sequencer-cost levers are nearly spent (candidates 1, 2 and 5 total about 150 us, 1%). Anything bigger needs candidate 3 or 4 to work.

# Review of iter-5 (working-tree kernel.neff vs HEAD a8b683b)

`notes/` is empty (README only), so there is no agent account to check against. The only claim to compare with is the HEAD commit message, which describes iter-4 and not this edit.

## 1. What changed
Compared with the previous NEFF (HEAD), four members differ: `def.json`, `Pool0.bin`, `SP0.bin` and `SP0.json`. `Activation0.bin` is unchanged against HEAD.
- **Relay removal.** 160 of the 192 queued-DMA reloads on `qSPSpillReload1` no longer go through an SP relay EVENT_SEMAPHORE.
  - The ring moved from semaphores 72-87 to 56-71. Those were the relays' target semaphores, and the rewritten slots show no wait mode and no update mode.
  - The ring's DMA completions now increment 56-71 directly, +16 each, 160 per semaphore. This is the same total the relays used to add.
  - Waiters on 56-71 are untouched.
  - In `SP0.bin` the relay slots become NOPs (`events_update_mode` SEM_ADD_IMM_COMPLETE -> NONE and `events_extended_wait_mode` GE_IMM -> NONE). This is about 160 of the 348 differing slots.
  - `SP0.json` has 13989 differing lines, mostly the `semaphore` field of the moved transfers.
- **Second ring.** The other 32 transfers move to a new queue `qSPSpillReload2` (`def.json`, 16 rings on semaphores 72-87).
  - Their consumers on Pool wait directly on those semaphores. 32 `Pool0.bin` slots change `events_extended_wait_idx` (76->72, 82->73, ...) and sometimes `sem_wait_value` (32->16).
  - Each of 72-87 now totals 32 increments, down from 192.
- **Consistency.** I found no mismatch with the commit history's design (iter-2's relay removed, DMA completion repaying the semaphore directly). `semaphores.py` shows identical `max_wait` and waiter counts on every semaphore other than 56-87, and no warnings.

## 2. Is the gain real? No, it is noise-level
- `profile_latest.json`: median 1172.35 us (min 1167.9, max 1173.1, spread 0.44%).
- I re-measured both NEFFs back to back, 5 repeats each:

| NEFF | median | min | max |
|---|---|---|---|
| HEAD | 1176.83 us | 1176.09 | 1177.75 |
| iter-5 | 1174.96 us | 1170.69 | 1177.61 |

- The difference is -1.9 us (-0.16%), and the ranges overlap. The spread is 0.14-0.59% (1.6-7 us), so the win is at most about 0.3% and I would not call it real.
- SP busy fell from about 53 us to 51.7 us (profile 52-53 vs 51.7 on my run), which is consistent with removing 160 cheap 20 ns relays.
- This is a cleanup with a plausible but unmeasurable gain, not progress. The loop is near the noise floor for this class of edit.
- Total gain vs baseline is still 1351.4 -> ~1175 (-13%).

## 3. Safety
- **Semaphore balance.** Totals per semaphore are unchanged and no wait can go unmet: 56-71 receive 160 x +16 from DMA completions, the same as from relays. 72-87 receive 32 x +16.
- **Looser sync.** A relay used to delay the increment until after the DMA completion plus SP's turn. Waiters now see the increment at DMA completion, which is earlier but never before the data lands. That is not a loosening.
- **Reload2 ring.** Each ring has two transfers (2 x 16 = 32). The Pool waiters use value 16 for the first and 32 for the second, so the first wait is a "GE 16" that stays satisfied afterwards. If a ring's second DMA could complete before its first, the first consumer could see 16 from the wrong transfer. Rings are in-order queues, so this should be safe. Nothing in the verifier tests ordering, though, so an ordering race is not excluded by inputs.
- **Error.** max_abs_err 1.097e-04 against a limit of 1.1e-3, about 10% of the limit, with cosine 0.999994. The data and arithmetic were not touched.

## 4. Candidates for the next iteration
No engine is near saturation. Busy time (excluding semaphore waits) is about 50% for ACT, DVE and PE per 100 us window. The remaining time is a cross-engine dependency chain of about 36 us x 32 batches. The levers are therefore small, and several have no measured ceiling.

| # | Candidate | Engine / slots | Change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Reorder the remaining SP head-of-line stalls | SP0.bin. In `Instruction.parquet`, Sync EVENT_SEMAPHORE with duration > 300 ns: 35 instructions, 38.9 us, of which 5.3 us is the final drain. About 33 us are real blocks. Take their `compiler_pc` from the table. | Move each stalling relay or wait slot past the already-unblocked DMA triggers behind it (permute slots only). | Same technique as iter-4 (a8b683b: 130 us of SP stalls moved, 44 slots, -31 us). | At most 33 us of SP stall. The realised fraction in iter-4 was ~24%, so expect about 5-8 us. Only counts if a trigger behind it feeds a critical consumer. | Low. A pure permutation with semaphore totals unchanged. | For each candidate stall, check whether the first consumer of a trigger behind it starts later than the trigger's completion. If no consumer is waiting, drop it. |
| 2 | Prologue: Pool DGE -> queued DMA | Pool0.bin, the ~192 DMA_DIRECT2D at gdn_cte.py:169/171/173/191/207 (0.7 us each, 138 us of Pool time). The first ~130 us hold the early instances. | Replace DGE with a queued DMA plus a trigger (as iters 1-2 did on ACT: 224 DGE -> 40 ns EVENT_SEMAPHORE, -8.2%). | `PSEUDO_DMATRIGGER` on a new data queue and ring. DGE descriptor generation is moved off the issuing engine. | Window-0 PE busy is 51 us vs 65 us in steady state, so ~14 us at most, realistically 3-8 us. Pool is only 20% busy in steady state, so later instances give about 0. | Medium. This needs a new queue entry in `def.json`, `SP0.json` blocks and exact descriptor translation (the `bf16->fp32` cast in line 191 needs the `op:"cast"` composite). A wrong descriptor would fail verification. | Compare each DGE's completion time against the first matmul that consumes the data in the first 130 us. If the consumer is not waiting for it, there is nothing to win. |
| 3 | Output stores at the tail | Pool0.bin, the last DGEs at gdn_cte.py:623/628 (t=1184.9, 1185.5) | Queued store instead of DGE | As in candidate 2 | ~1-2 us (the stores end at 1186.2 and the drain finishes at 1187.8) | Low-medium | The last compute ends at 1184.8. The stores run from 1184.9 to 1186.2, so if nothing waits on the Pool DGE the gain is zero. |
| 4 | Pool EVENT_SEMAPHORE stalls | Pool0.bin, 35 instructions > 300 ns, 55 us of waiting | Check whether Pool is delaying a DGE issue (it blocks in order) | Same reasoning as 1 | Unknown | Low | Time each long wait against the next DGE it delays. |

Do not run candidate 1, 2 or 4 blind. Each must first show that the stalled engine is on the critical path.

**Irreducibility check.** The tail is about 5 us from the last compute to the final DMA completion. The per-instruction durations are dominated by MATMUL (540 us across 2272 instructions) and the ACT/DVE casts. I see no NEFF-level lever with a measured gain above the ~0.5% noise floor, so the honest expectation for the next iteration is 0-1%.

## Closed
- DRAM staging round-trips: spill bytes went from 7.86 MB to 0 (iter-1/2).
- ACT DGE saves: 224 -> 0. Scalar has no DMA opcodes left in the profile.
- SP relay head-of-line, first 44 slots (iter-4): 1207.6 -> 1176 us.
- The 160 relays removed in this iteration: -1.9 us, within noise.

## Unknown
- Which of the 35 long SP and 35 long Pool semaphore stalls sit on the critical path. A per-batch dependency trace (join `SemaphoreUpdate.parquet` on waiter and setter) would settle it.
- Whether Pool DGE issue time gates the prologue. Check the gap between DGE completion and the first consuming matmul in the first 130 us.
- Run-to-run noise of `profile.py` on this NEFF. Take 3 x 5 repeats of the same file to get a floor before accepting any sub-0.5% change.

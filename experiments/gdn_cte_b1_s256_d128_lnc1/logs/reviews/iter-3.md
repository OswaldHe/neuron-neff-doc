# Review of iter-3 (HEAD 9d8415c, kernel.neff vs HEAD~1)

## 1. What changed
`neff_diff` (HEAD~1 -> HEAD): 34 members, 32 identical, 2 changed: `sg00/Pool0.bin` (35 -> 37 slots, 19 slots shifted/differ) and `sg00/SP0.bin` (64 -> 66 slots, 4 differ).
- SP0 slots 14, 15: `PSEUDO_DMA_DIRECT2D` -> `EVENT_SEMAPHORE`. The two var9 reload DGEs left SP, and SP's DMA queue entries sit behind the replacement semaphore ops. Two new tail slots, 64 and 65, were added.
- Pool0: two `PSEUDO_DMA_DIRECT2D` were inserted (slots 18-31 re-ordered around the affine_selects), plus an extra `EVENT_SEMAPHORE` (sem 16, +1, no waiter), and the trailing `NOP` became an `EVENT_SEMAPHORE`.
- Semaphore table: sem 20/21 and 62/63 changed owner (`dge:Pool` <-> `dge:SP`). Every semaphore still has max_wait == total_inc (see section 3).

This matches the commit message ("two var9 reload DGEs SP -> Pool, guard semaphore made explicit"). `notes/` is empty apart from the README, so the agent left no record of what was tried. Iterations 1-2 had none either.

## 2. Is the gain real?
Yes, but small. It is about 1.0 us (1.6%). It is not the 60.930 in the commit message; that number was a lucky draw.

| NEFF | run 1 median (min-max) | run 2 median (min-max) |
|---|---|---|
| HEAD~1 | 62.035 (61.80-62.33) | 62.055 (61.79-62.32) |
| HEAD | 61.015 (60.73-61.46) | 61.106 (60.94-62.13) |

(Interleaved A/B, 5 repeats each. The loop's own `profile_latest.json` reads 61.35, spread 1.99%.)

- Run-to-run drift of the same NEFF is about 0.5 us. This iteration's gain is about 2x that, and the medians separate consistently across both pairs.
- The iteration-2 commit message claims +1.1% for what was a 0.7 us step (61.939 -> 61.245). That step was within drift. Treat any step under about 1 us as unproven until it has been A/B'd twice.
- Cumulative: 67.525 -> about 61.0 us (-9.7%). Nearly all of it (-8.3%) came from the iteration-1 fusion of the SBUF->DRAM->SBUF round-trips. Iterations 2 and 3 added about 1.7% together.

## 3. Safety
- `semaphores.py` on the unpacked NEFF: all waited semaphores have max_wait == total_inc (3, 4, 5, 6, 8-19, 21, 24-39, 43-64), so no wait was reduced. HEAD vs HEAD~1 differs only in engine ownership of 20/21 and 62/63, and the Act waiter count on sem 4 (71 -> 73).
- Semaphores 16, 20, 45, 49, 58 and 62 have increments but zero waiters. The same holds in the baseline for 20, 45, 49 and 58. So the reload DGEs that write through those are ordered only by program order and queue FIFO, not by an explicit wait. Moving them between engines (this iteration) changes which queue they sit on, so a race is possible if timing shifts. The verifier has one fixed input set, so it would not catch such a race.
- Error: output_0 max_abs_err is 6.1e-5 (11% of limit). output_1 is 4.49e-4 (83% of limit). Both are identical to the baseline, so the edits are bit-exact and no error has moved. The 83% on output_1 is the original kernel's, and it leaves no room for any change that touches precision (see candidate 5).
- Spill bytes went 491520/557056 -> 0 and HBM traffic fell sharply. That is consistent with iteration 1.

## 4. Candidate edits for the next iteration (ranked)

Timeline facts used (`prof/Instruction.parquet`, t=0 is the profile start). The first ~9.5 us and the last ~4.5 us are runtime pre/postamble, so the kernel body is about 10 -> 71 us.
- Pool issues 8 DGEs back to back from 12.68 to 19.20 us. Each is 0.65-0.8 us on the issuing engine (lines 169/171/173/191/207).
- Sync is almost idle 11.9-16.5 us. Pool is near-idle after 20 us.
- Act (busy 32.6 us, 53%) issues DGEs at 49.85, 51.55 and 52.88.
- Tensor is idle 49.5 -> 56.2 us (about 6.7 us) while a chain of 5-6 DGEs (Act/Sync/Pool) plus about 1.9 us DMA completion waits (Sync 52.82-54.75, Pool 53.26-54.58, Sync 57.73-59.20) feeds it.
- The tail DGE is at 72.08 (0.88 us), with the finish at about 74.6.

| # | Engine / member / slots | Change | Mechanism | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Pool0 + SP0: the 8 DGEs at compiler_pc 1, 6, 8, 9, 10, 11, 12, 13 (lines 169-207), first Pool window | Move the alternate 4 DGEs (say pc 8, 10, 12, 6) to the idle Sync engine. Keep each DGE's semaphore number and increment. Make sure each consumer's wait is unchanged. | Same DGE-engine swap that worked in iter-3 (SP -> Pool) and iter-2 (Act -> Pool/Sync): each DGE costs about 0.65-0.8 us of issue time on its engine, so the chain is issue-bound. | Up to about 3 us if this chain gates the first ACTIVATE (16.49 us) and the later ones. Ceiling: Pool chain 12.68 -> 19.20 = 6.5 us; halving gives at most about 3.2 us. Could be 0 if the consumers wait on DMA data latency. | Medium. Sync queue ordering and unwaited semaphores (see section 3) could race. A fixed-input verifier would not catch a race. | Move just ONE DGE (pc 13) to Sync. If Act's ACTIVATE at 17.87 / 18.13 / 19.3 does not move earlier, the chain is not DGE-issue-bound; stop. A/B twice. |
| 2 | Activation0: DGEs at pc 82, 88, 94 (lines 483, 487) | Move these 3 to Sync or Pool (Sync is idle 49.8-51 us), keeping the guard semaphores. | Same engine swap. Act is the busiest engine; each of these DGEs costs about 0.65 us of Act time inside the 49.8-53.5 us window where Tensor starves. | Up to about 2 us, ceiling = three 0.65 us DGE issues on Act. Realistic about 0.5-1 us. | Medium. The ordering vs the semaphores 43/47/50 wait (Act-sourced, 1 waiter each) must be preserved. | Move only pc 82. If the 6.7 us Tensor gap does not shrink, the gap is DMA latency (about 1.9 us per hop), not issue time. |
| 3 | Sync/Pool chain in 50-56 us (DGEs at pc 30, 34 on Sync; pc 21, 22, 23 on Pool) | Merge two adjacent same-queue DGEs with contiguous src/dst into one larger descriptor (fewer 0.65 us issues and fewer 1.9 us completion hops). Edit the PSEUDO_DMA_DIRECT2D `elem_size` / step fields and drop one slot. | One 2-D DGE can cover a contiguous pair; the DMA fabric cost is per descriptor. | Unknown. At best one 0.65 us issue plus part of a 1.9 us hop per merge. | High. Address and step arithmetic errors give wrong data. Layout must be checked on the dst var offsets. | Check src/dst contiguity for one pair in the decode first; if not contiguous, drop it. |
| 4 | Pool0/SP0 tail: DGE at pc 32 (67.48) and Sync pc 64/65 (71.38, 72.08) | Split the final output DGE across Sync and Pool (two half-width stores in parallel), or issue it right after the last producer instead of at 72.08. | The store is 0.88 us issue plus about 1.7 us completion. Parallel queues overlap. | Up to about 0.5-0.9 us. Ceiling: tail from 72.08 to 74.6. | Low-medium: output bytes must cover the whole tensor exactly; the verifier would catch a missing half. | Time it from the parquet: if the DMA time is mostly queue latency, not bytes, splitting does nothing. |
| 5 | Activation0 TENSOR_SCALAR_ARITH_OP (25 instr, about 0.26-0.39 us each) | Re-encode a few fp32 elementwise ops to Pool. | Pool is 26% busy, Act 53%. | Unknown. Act is not the bottleneck in the 35-50 us window (PE 5.8-7.9 us of every 5 us), so likely under 1 us. | High effort and high error risk: output_1 is at 83% of its error limit, so any fp32 -> bf16 or ordering change could break it. Needs semaphore re-plumbing across engines. | Estimate first: does Act idle after the move? Only start it if 1-4 are exhausted. |

Rank logic: 1 and 2 reuse the exact mechanism proven in iterations 2 and 3 with a measured issue cost (0.65-0.8 us per DGE) and an idle engine to take them. 4 is cheap but small. 3 and 5 are the least certain.

### Closed
- **ACT_TABLE_LOAD** (1.28 us, Act slot 1): not worth chasing. It runs at 12.13-13.41 us, while the first ACTIVATE does not issue until 16.49 us, so it is hidden behind the input DMA.
- **Spill/reload round-trips**: closed by iteration 1 (spill bytes 491520/557056 -> 0, 67.5 -> 61.9 us).
- **Moving the var9 reload DGEs SP -> Pool**: done in iteration 3 (about -1.0 us, from my two interleaved A/Bs).
- **Preamble/postamble** (about 9.5 us and about 4.5 us of the 76 us timeline): runtime-owned and not part of the 61 us measurement.

### Unknown (would change the ranking)
- Whether the 8-DGE Pool chain at 12.7-19.2 us is on the critical path, or the Act/Vector consumers wait on DMA data latency. Settle it with the one-DGE move in candidate 1.
- Whether the 6.7 us Tensor gap at 49.5-56.2 us is issue-bound or DMA-latency-bound (about 1.9 us per hop). Settle it with the single-DGE move in candidate 2.
- The real run-to-run noise of this kernel. Run `profile.py --repeats 7` on one NEFF three times in a row. I saw 0.5 us drift (61.0 / 61.1 / 61.35 / 61.8 for the same NEFF across tools and runs), so a candidate needs at least 1 us before it can be believed.
- Whether DMA latency can be cut by issuing the DGEs from the engine whose queue is least loaded, rather than the least busy engine.

Process note: write the notes. The notes directory is empty after three iterations.

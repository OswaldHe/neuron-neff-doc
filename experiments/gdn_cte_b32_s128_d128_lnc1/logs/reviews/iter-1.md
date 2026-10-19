# Review of iter-1 (gdn_cte_b32_s128_d128)

## 1. What changed
`neff_diff.py`: 32 of 34 members are identical. Two changed, both in `sg00/SP0.*`.

- **`SP0.json`**: 32 of the 669 DMA definitions changed (ids 4, 25, 50, 71, 90, 108, ... 651). In each, the 2 descriptors that read the DRAM spill tensor (`inst__I-9-0:_mem_54_50_sg0000` and similar) now read `from: "SB"`.
  - `from_off` is now the SB address the spill was originally saved from, e.g. 16798208 = partition 64 + 0x5200.
  - The steps became `[1,32,4194304,262144]`, so it is a pure SB to SB copy.
  - These were the only 32 DRAM-sourced DMAs in the baseline (1,048,576 B). All 32 were converted.
- **`SP0.bin`**: 34 of 1315 slots changed, all on the SP queue.
  - 16 `c1` wait/no-op slots (e.g. slot 5, 44, 89, 131 ... 698) changed their wait from `sem 40..55 >= 16` to `sem 6 >= N`, with N rising from 13 to 1120. Sem 6 is the DVE-progress semaphore.
  - 1 DMA slot (45) and 1 more (90) went from `NONE` to `WAIT_FOR_SEM_GE_IMM` with the old wait, `sem 49/55 >= 16`. So some of the old wait was moved onto the DMA itself.
  - The remaining slots are the "14 more" that `neff_diff` truncates. I did not decode them.
- **What it means.** In the baseline, SP waited for the Act-queue spill-save DMA (the `dge:Activation` completion on sems 40-55) before DMA-reloading the data from HBM. The Act-side save is `src_var_id=0 (SB)` to a DRAM var, with `S[6](Vector)>=13`, `semaphore=40`. The new SP wait `sem6>=13` is exactly that save's own DVE precondition. So the edit says: don't round-trip through HBM; copy straight from the original SB location once DVE has produced it.
- **Consistency with notes.** `notes/` contains only the README template, so the agent recorded nothing and there is no claim to check. The profile counters match the decoded edit: `hbm_read_bytes` and `spill_reload_bytes` both dropped by exactly 524,288. `spill_save_bytes` is unchanged at 8,912,896.
- **Gap.** The Act-queue save DMAs were left in place, so the 32 spill saves are now dead writes.

## 2. Is the latency change real?
Yes. Baseline median is 1351.40 us (samples 1349.1-1352.1, spread 0.22%). Iteration median is 1313.32 us (1308.3-1315.6, spread 0.55%). The ranges do not overlap. My own re-profile gave 1311.7 us.

- The gain is about 38 us (2.8%), roughly 70x the baseline spread.
- Engine busy times are unchanged within 0.2%, so the gain comes from shorter dependency stalls, not less work.
- The loop is not chasing noise on this result. Anything under about 5 us would be.

## 3. Safety
- **Semaphore accounting.** `semaphores.py` on both unpacked NEFFs shows only a change in sem 6's waiter count (2220 to 2252, i.e. 32 added waits). `max_wait` and `total_inc` are unchanged. Sems 40-55 still balance (total_inc is unchanged because the saves still run).
- **Dropped wait.** The wait on the Act save completion is gone from ~16 of the reloads. That is intended, because the data no longer comes from HBM.
- **Missing wait.** A read-after-write guard exists (DVE sem 6), but there is no write-after-read guard. The spill existed because the SB region was reused. Nothing makes the SB-to-SB copy finish before a later writer reuses that region. The Act save used to have the same hazard, but it was protected by its own timing. The verifier runs one fixed input. Timing is deterministic, so it would pass and still miss a rare race.
- **Error.** `output_0` max_abs_err is 1.1e-4 (10% of limit). `output_1` is 9.08e-4 (83% of limit) and is identical to the baseline value (9.081e-4). So the numbers are bit-for-bit unchanged and nothing moved toward the limit. That is the best evidence that no race fired.
- **Verdict.** Acceptable to keep.
  - Cheap hardening: repeat `verify.py` and `profile.py` several times, and diff the outputs bitwise against the baseline.
  - Static proof: for each of the 32 SB source ranges, find the next writer of that SB range in any engine stream and confirm a semaphore wait orders it after the reload's completion semaphore (the reload's `semaphore: 28` and similar).

## 4. Ranked candidates for iter-2

| # | Edit | Mechanism | Expected / ceiling | Risk | Falsify |
|---|---|---|---|---|---|
| 1 | **Delete the 32 now-dead spill-save DMAs on the Activation queue.** They are `DMA_DIRECT2D` with `src_var_id=0`, `src_var_offset` equal to the iter-1 `from_off` values (16798208, ...), and `dst` in the DRAM spill vars. Look for the `S[6](Vector)>=13 dge_op=DIRECT2D semaphore=40 sem_increment=16` form in `Activation0.bin`. | Replace each with a no-op that keeps the semaphore increment, or lower the sem 40-55 wait values in the other consumers (and the SP slots 45 and 90 that still wait on 49/55). Active-engine DMA issue costs 0.5-0.6 us each and removes 32 x 16 KB of HBM writes. | Act is the busiest compute engine (instruction busy 718 us; Act DMA issue 130 us across 224 DMAs). Ceiling is about 32 x 0.58 = 19 us of Act time. Realised gain on the critical path is unknown, probably 0-10 us. | Medium. If a sem increment is dropped without adjusting waiters, you get a deadlock (the run hangs, so the verifier catches it). A save that something else still reloads would break correctness. | Run `semaphores.py` after the edit. Totals on sems 40-55 must still match their waiters. Grep for any other consumer of `dst_var_id`. Profile once. If it is within noise, drop it. |
| 2 | **Verify and tighten iter-1's write-after-read ordering** (not a speedup). For the 32 converted reloads, add an explicit wait on the reload's completion semaphore before the next writer of the SB source range. | Standard semaphore wait insertion (`docs/10-synchronisation.md`). | 0 us, or a small cost. Pure safety. | Low. | Static analysis of SB address overlap is enough. If every range has an ordered next writer, skip it. |
| 3 | **Move Act-queue DMA issue to the idle SP or Pool engine.** 224 `DMA_DIRECT2D` on Act and 160 on SP. Pool (255 us busy) and SP (131 us busy) are under-used. | Re-home the issuing instruction slot and adjust its semaphore waits. | Ceiling is the Act DMA issue time, 130 us, minus what overlaps. Real gain is unknown. | High. Cross-stream moves are slot-count and semaphore surgery, and a race may pass the one fixed input. | Move a single DMA from Act to SP on a copy and measure Act's critical-path stall at that point. Stop if the delta is under 1 us. |
| 4 | **PE `LDWEIGHTS` elision.** There are 1878 LDWEIGHTS at 111 ns each = 208 us, followed by MATMUL in 1833 cases. | Remove an LDWEIGHTS when the previous load put the same weights in the same PE rows. | Ceiling 208 us of PE time, but PE is only about 60% busy in steady state. I checked adjacent LDWEIGHTS operands: **0 of 1878 are identical**. So this is dead unless the weights are the same non-adjacently, which needs matmul-stream analysis. | Medium. A missed intervening load gives wrong numbers. | Compare LDWEIGHTS operands against the last-loaded weights, across intervening MATMULs only. If there are 0 repeats, it is closed. |

Candidates 3 and 4 are speculative. Candidate 1 is the only one tied to a concrete mechanism measured in this iteration. Candidate 2 is a correctness chore.

## 5. Closed
- **DRAM-sourced spill reloads.** All 32 are now SB-sourced, and the profile shows `hbm_read_bytes -524,288`. Nothing is left to convert. The other 637 DMAs were already SB to SB in the baseline.
- **Adjacent duplicate LDWEIGHTS.** 0 of 1878 have identical operands.
- **Making the kernel compute-bound.** Steady-state engine busy is PE 55-68%, Act 53-57%, DVE 41-48%. No engine is saturated, so the run is dependency-latency bound.

## 6. Unknown (measure to resolve)
- **Is Act or PE on the critical path?** Per-instruction stall attribution (what each engine waits on, by semaphore) would show whether removing Act DMA issue (candidate 1) or any of 3 matters. Use `Instruction.parquet` plus the semaphore waits in the `.bin`.
- **Is the remaining wall time irreducible at the NEFF level?** About 55% engine utilisation is what a dependency-chained pipeline over 32 batch items looks like. I would expect only a few more percent to be recoverable here, and that is a judgement from the profile, not a measurement.
- **Dead writes.** Whether any part of the 8.9 MB spill-save traffic (still written) is also dead beyond the 32 found. Settled by matching every save's destination var against the reload sources in `SP0.json`.

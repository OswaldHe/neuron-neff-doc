# Review of iter-2 (kernel.neff, working tree vs HEAD f8f9f89)

## 1. What changed

`neff_diff.py` against HEAD (iter-1's result) shows 4 members changed, and 30 of 34 are identical. Against the unedited baseline the diff is the cumulative iter-1 + iter-2 one.

| member | slots | what |
|---|---|---|
| `Activation0.bin` | 9 of 77 differ, same size | 3 `TENSOR_SCALAR_ARITH_OP` replaced by `EVENT_SEMAPHORE` (slots 11, 13, 16). 3 `PSEUDO_DMA_DIRECT2D` replaced by `EVENT_SEMAPHORE` (slots 14, 20, 21, the sem 44/45/46 DGEs). A CAST/ACTIVATE pair swapped (slots 26-28). |
| `DVE0.bin` | 87 of 100 differ, grew 94 to 100 slots | The 3 Act TENSOR_SCALAR ops reappear as DVE `TENSOR_SCALAR_ARITH_OP` (new sems 47-49, 1 increment each). The rest is a re-ordered, re-timed DVE stream: operand addresses and wait idx/values changed, one ADD/BF16 op became MULT/FP32 with 64 channels, and a COPY became a CAST with 64 elements. |
| `SP0.bin` | 36 of 42 differ, grew 37 to 42 slots | The 3 Act DGEs (sems 44-46) now issue from SP. `PSEUDO_DMATRIGGER` ops and `EVENT_SEMAPHORE` waits re-ordered. Wait idx/value changed on several DGE/trigger slots. DGE count fields changed (5 to 4, 6 to 5, 7 to 6). |
| `Pool0.bin` | 1 of 23 | A single slot. |

In plain terms, three small elementwise ops moved from ACT to DVE and three DGE launches moved from ACT to SP. DVE and SP were re-scheduled around them.

`notes/README.md` is still the empty template, so there is no claim to check the diff against. The only description is the iter-1 commit message, which does not cover this iteration. The diff is internally coherent: every moved op has a matching placeholder or new semaphore. The next agent should write notes.

## 2. Is the latency change real?

| run | median (us) | min-max (us) | spread |
|---|---|---|---|
| baseline | 48.519 | 48.21-49.16 | 1.9% |
| iter-1 (from commit message) | 43.43-43.50 | not recorded | not recorded |
| iter-2, `profile_latest.json` | 41.056 | 40.84-41.16 | 0.77% |
| iter-2, my re-run (7 repeats) | 40.819 | 40.73-41.24 | 1.25% |

The two iter-2 runs agree within 0.25 us. The improvement over iter-1 is about 2.4-2.7 us (about 6%), roughly 2-3 times the spread. I did not re-measure iter-1 myself. Its 43.4-43.5 number is from the commit message, but the baseline reproduces (48.5), so it is plausible. The win is real, not noise.

Engine busy time moved as the diff predicts:
- ACT: 23.4 to 20.1 us.
- Pool: 11.0 to 9.1 us.
- DMA: 17.9 to 15.3 us.
- SP: 6.5 to 9.5 us.
- DVE: 19.3 to 20.2 us.

## 3. Safety

Semaphores, checked with `semaphores.py` on both unpacked NEFFs:
- Every semaphore has `total_inc == max_wait` before and after, so no wait was weakened.
- The waiter count rose for sems 4, 6, 9, 10, 11 and 12, so syncs were added or re-pointed, not dropped.
- Sems 44-46 are now produced by SP DGEs (was ACT). Sems 47-49 are new single-increment DVE to waiter pairs.
- Sem 16 has 16 increments and 0 waiters (dge:Pool), and it has no waiter in the HEAD NEFF either. This is pre-existing, not new.

Remaining race risk is in the reordered DVE stream, not in the semaphore totals. The verifier checks one fixed input set, and "total_inc equals max_wait" does not prove that every consumer waits on the right producer. A WAR/RAW hazard on a reused SBUF tile would only show up as a rare mismatch.

Accuracy:
- output_0 max_abs_err is 6.1e-5 (17% of the limit) and output_1 is 2.86e-4 (82%).
- Both match the baseline values to the digit shown, so the output is unchanged.
- output_1 was already at 82% of the limit at baseline, so any numerics edit (dtype change, accumulation order) has about 18% headroom. Pure scheduling edits are safe on this front.

Recommendation: accept. Before accepting, run `verify.py` 3-5 times on the final NEFF to catch a flaky race. The notes file needs filling in.

## 4. Ranked candidates for iter-3

Measured from `Instruction.parquet` (one capture, 7 repeats; times are trace-relative, the kernel body starts at about 12.5 us). The exec window is about 41 us: roughly 3 us of prologue, then about 38 us of body, ending at about 54 us.

| # | Edit | Mechanism / evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Pool input-load DGEs start too late and are serial.** Pool0 (`GpSimd` track): move the 5 input-load `PSEUDO_DMA_DIRECT2D` slots (profiler 13.75-17.28 us) ahead of the 3 `TENSOR_SCALAR_AFFINE_SELECT` mask builds. Then move 2 of the 5 onto SP0 (idle 13.3-19.9 us), re-pointing their wait/sem fields. | DGEs cost 0.65-0.75 us each on the issuing engine and run back to back. They start at 13.75 us although they wait only about 22 ns, because the affine selects (12.85-13.80 us) sit in front. The first MATMUL starts at 15.96 us, right after the third DGE lands. Moving DGEs across engines is already proven in iter-1 and iter-2 (ACT to SP). | At most about 1.0 us from the reorder alone, and at most about 2 us with the SP split. The serial issue span is 3.5 us. DMA completion latency is unknown. | Low-medium. The data dependency is checked by the verifier. The mask tile is consumed later (17.3 us), so the reorder is safe if the AFFINE_SELECT outputs are not read by the DGE sources. | Swap two Pool slots only (first DGE ahead of the first affine select) and re-profile. If the first MATMUL start (15.96 us) does not move, the DMA latency, not the issue, is the bound, and the idea is dead. |
| 2 | **Last ACT-issued DGE (sem 43, ACT track 38.56-39.22 us).** Move this single slot to Pool0, which is idle at that time, using the same op-move recipe as iter-2 sems 44-46. | A 0.66 us DGE issue stalls ACT in the busiest part of the kernel (ACT 53% busy), and Pool does nothing from 38 to 45 us. | At most 0.66 us if ACT is on the critical path there. ACT is idle 18 us in total, so it may not be critical. | Low. Same fields as iter-2's moves. | Move it, re-profile. If the median moves under 0.2 us, ACT is not critical in that window. |
| 3 | **Tail: split the final output store.** The last compute ends at 51.25 us. The store DGE issues at 51.61-52.27 us from SP, then the completion wait runs to about 54.0 us (2.8 us from compute end to the end of the body). Split it into 2 half-size DGEs on SP and Pool/ACT, so the DMA overlaps and each half is shorter. | The store is 65664 bytes per output. A single queue sets the 1.24 us transfer wait. | Likely 0.3-0.6 us. Unknown: whether DMA time scales with bytes per queue. | Medium. This changes descriptors and counts (the DGE count field was edited in iter-2), so a wrong element count can truncate the output. The verifier will see that. | Check the DMA bytes and queue count in `Instruction`/`DmaPacket` for the final store. If the transfer is already spread over many queues, the split gains nothing. |
| 4 | **Keep rebalancing compute off the busiest engine.** DVE (20.2 us, 49.5%) and ACT (20.1 us, 49.3%) are now balanced. Pool is at 9.1 us, 22%. Move SBUF-sourced DVE `TENSOR_TENSOR`/`COPY` ops to Pool. | ACT to DVE moves already paid off (DVE +0.9 us, ACT -3.2 us). Pool has about 30 us of idle time, and GpSimd gaps total 33 us. | Unknown. Per-engine busy time is not the bound here (no engine is above 50%); the dependency chain is. Expect under 0.5 us. | Medium: Pool is slower per element than DVE, and PSUM access from Pool must be checked in `neff_doc` first. | Pick one SBUF-only DVE op off the critical chain, move it, and compare median. If it does not move, drop the idea. |
| 5 | **Fold standalone `EVENT_SEMAPHORE` into the next instruction's wait field.** 53 short (under 120 ns) EVENT_SEMAPHORE ops total about 2.0 us across engines. | Each instruction carries `events_wait_*` fields; a standalone wait can ride on the next op. | Small: the ops are about 40 ns each and mostly hidden, so under 0.3 us wall time. | Medium. Lost or merged waits are exactly the race pattern the verifier can miss. | Merge a single pair on an engine on the critical path (Tensor or DVE around 36-40 us) and see whether the median moves. |

Ranking is by expected gain per risk and effort. The strongest case is candidate 1 because the whole startup chain is gated by it. 2 is cheap and almost free. 3 has a clear bound but needs a look at the DMA first. 4 and 5 are weak and are listed for completeness.

### Closed (do not redo)
- **Staging round-trips** (SBUF to DRAM to SBUF): all five already fused in iter-1 (48.5 to 43.5 us).
- **ACT_TABLE_LOAD** (1.29 us, ACT track at 12.68-13.97 us): finishes at 13.97 us, before the first ACTIVATE at 16.37 us. It is hidden behind the input DMAs, so moving it gains nothing.
- **ACT/Pool DGEs for outputs 44-46:** done in iter-2.
- **Tail runtime teardown** (about 54.1-55.4 us: DRAIN, RANGE_CLEAR, NOTIFY): runtime sequencing, not editable and mostly outside the timed window.

### Unknown (measure before ranking)
- **DMA completion latency for the input loads.** This decides how much candidate 1 can gain. Measure it by reading `DmaPacket`/`DmaPacketAggregated` timestamps for the first Pool DGE against the first MATMUL.
- **Critical path through the middle (24-36 us).** I did not trace it. Tensor shows over 100% utilisation across queues in this window, so a PE/DVE dependency chain probably sets the floor. A per-op critical-path pass over `Instruction.parquet` with sem edges from `semaphores.py` would settle it.
- **Whether the 53 short EVENT_SEMAPHOREs are actually on the critical path.**
- **Sem 16:** 16 increments, 0 waiters. Check whether it protects a hazard that the verifier does not exercise.
- **Remaining headroom.** The body is dominated by a dependency chain of small ops (0.2-0.4 us each) with an irreducible DMA and notify prologue and epilogue (about 14 us of the trace is outside the body). The remaining NEFF-level gain is probably 1-3 us, not another 6 us.

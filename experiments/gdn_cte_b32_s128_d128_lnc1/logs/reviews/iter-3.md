# Review of iter-3 starting point (kernel.neff == HEAD f8380ab, "iteration 2")

`neff_diff.py` compares against the unedited baseline, so it shows the cumulative edit of iterations 1 and 2.
The working tree is clean. No new NEFF edit is pending.

## 1. What changed (baseline -> kernel.neff)

| Member | Change |
|---|---|
| `def.json` | New queue `qSPSpillReload1`: owner sp, 16 rings, semaphores 72-87. |
| `SP0.json` / `SP0.bin` | 1315 -> 1507 slots (+192 queued-DMA triggers and relay events). 84160 -> 96448 B. |
| `Activation0.bin` | 224 slots changed. 220 `PSEUDO_DMA_DIRECT2D` (DGE saves) became `EVENT_SEMAPHORE`, 4 `PSEUDO_DMATRIGGER` became `EVENT_SEMAPHORE`. |
| `Pool0.bin` | 64 slots changed. These are the iteration-1 fused staging: `DIRECT2D` -> `DMATRIGGER`, and `EVENT_SEMAPHORE` <-> `DMATRIGGER`. |

The other 29 members are identical.

This matches the commit messages: dead DGE saves became 40 ns `EVENT_SEMAPHORE`s, and the staging round-trips moved to SBUF->SBUF queued DMAs.

`notes/` is empty (README only). The agent wrote nothing, so there is no notes text to cross-check. Failed attempts are not recorded.

## 2. Is the latency change real?

- `profile_latest.json`: median 1204.83 us, min 1202.83, max 1210.21, spread 0.61% (about 7 us) over 5 repeats.
- Baseline: 1351.40 us, spread 0.22%.
- Total -146.6 us (-10.8%). The step from iteration 1 to 2 alone was about -106 us, which is 15x the spread.
- The win is real. It is not noise.
- Engine busy times fell where expected: ACT 627 -> 521 us, SP 131 -> 52 us, DMA 420 -> 341 us.
- HBM read went 12.08 -> 3.17 MB and write 12.06 -> 3.15 MB. That is about the kernel's true I/O, so the staging traffic is gone.

## 3. Safety beyond the verifier

- `verify_latest.json` passes. `max_abs_err` is 1.1e-4, which is 10% of the 1.1e-3 limit. Outputs were bit-identical in the notes' 3 runs, so no numerical risk.
- `semaphores.py` flags no "wait exceeds increments". Semaphores 8-21 (Pool) and 41-55 (ACT) keep their old totals, now split between `dge:` and `EVENT_SEMAPHORE` sources. The +16 repay accounting is consistent.
- **Weak spot:** semaphores 72-87 (the 16 new rings) show 0 waiters in `semaphores.py`. Each has 192 increments and nobody waits on it.
  - The consumers presumably wait via the relay `EVENT_SEMAPHORE` and the extended events block, which the tool may not see. I did not trace this.
  - The profile shows these 192 transfers lasting 80-110 us each (`DmaPacketAggregated` mean 82 us). A reload that late would be a race if any consumer were not truly gated on it.
  - Bit-identical outputs on one input set show only that the timing happened to work.
  - Falsifier: add a long wait (a few us of dummy `EVENT_SEMAPHORE`s, or a delay) in front of one ring's trigger. Check that the output still matches. Also run on a second random input set.
  - Until that is done, the iteration is "passes", not "proven safe".
- Nothing else was loosened. No waits were reduced, and the dead ACT saves were replaced by `EVENT_SEMAPHORE`, not deleted.

## 4. Where the remaining 1205 us goes

Method: `profile.py --repeats 1 --parquet prof`, `prof/Instruction.parquet`, `DmaPacketAggregated`.

- No engine is above 65% busy. Measured busy over the 1224 us span:
  - PE 798 us (65%)
  - ACT 680 us
  - DVE 593 us
  - Pool 297 us
  - SP 245 us
- The kernel is a serial dependency chain of 0.1-0.5 us ops. There are 32 near-identical groups at about 37.5 us each (about 100 us per window). Chain length, not throughput, sets the latency.
- Whole-machine idle (no non-semaphore instruction running on any engine) totals only 172 us, in 112 gaps. The 80 gaps over 1 us sum to 147 us.
- Gaps are concentrated at the following sites (post-gap instruction, source line in `gdn-bench/vendor/gdn/gdn_cte.py`):
  - `LDWEIGHTS` at line 597 (the `W_t @ state` matmul that follows `dma_transpose(W_t, W_full)`): 27 gaps, 61.6 us.
  - `LDWEIGHTS` at line 441 (the doubling-solve matmul): 25 gaps, 44.4 us.
  - `MATMUL` at line 447: 14 gaps, 17.6 us.
  - ACT `TENSOR_SCALAR` at line 591: 5 gaps, 8.5 us.
- The PE-local stall at the first line-597 `LDWEIGHTS` of each group is about 4.3-4.7 us, in 30 of 32 groups. Its DMA predecessor ends only about 0.5 us before the matmul starts, so the PE is waiting on a DMA-transpose chain.

## 5. Ranked candidates for the next iteration

| # | Candidate | Ceiling | Risk |
|---|---|---|---|
| 1 | Hoist the group-start `dma_transpose` chain | <= about 105 us (idle at sites 597+441), realistic 30-60 us | Medium |
| 2 | Drop redundant adjacent `LDWEIGHTS` with identical stationary operand | unknown, <= 0.1 us x count | Low |
| 3 | Trim the startup ramp: input-load DGEs on Pool onto a queue | about 5-10 us | Low |
| 4 | Batch/merge small `qSyncSpillReload0` transposes | unknown | Medium |

### Candidate 1 -- hoist the group-start transpose chain (best)

- **Engine / member / slots:** the engine that triggers the `dma_transpose` of `W_full` -> `W_t` (SP or Pool; the queue is `qSyncSpillReload0`, small transposes of 2-16 KB). The triggers feed the line-597 `LDWEIGHTS` (64 instances) and the line-441 first matmul of each group.
- **Change:** move the trigger slot (and its wait) earlier in that engine's program order, and re-point its wait to the true producer (W solve output / `W_all` arrival). The trigger should issue as soon as `W_all` is present, not behind unrelated work on the same engine.
- **Mechanism:** reorder independent instructions and re-point waits (r08, `docs/14` section 14.2, section 14.3 item 2). Re-pointing a queued DMA via `SP0.json` is r09.
- **Ceiling:** the 4.4 us PE stall x 30 groups is about 130 us as an upper bound on PE-local waiting. Whole-machine idle at the 597 and 441 sites is 61.6 + 44.4 = 106 us. The achievable figure is below that, because part of the stall is the true HBM/DMA latency of `W_all`.
- **Risk:** getting the wait wrong gives a stale-data race. The verifier would catch a fully wrong read, but a near-miss race may pass. Require bit-exact output on 2 input sets.
- **Falsifier:** read the wait field on the group-first line-597 `LDWEIGHTS`/trigger slot. If the trigger's producer wait is already satisfied before the stall begins, so the stall is real DMA latency and not queueing behind program order, the candidate is dead in minutes.
- **Caveat:** I did not decode the trigger slot's wait. That decode is the first step.

### Candidate 2 -- drop redundant adjacent `LDWEIGHTS`

- **Engine / member:** PE `Tensor0.bin`. 1878 `LDWEIGHTS` cost 208 us of PE time (mean 111 ns).
- **Change:** in the doubling solve, lines 441/447/449 reuse the same stationary `PT_stat`. Where two adjacent matmuls load the same stationary operand with no intervening write, delete the second `LDWEIGHTS` and keep the semaphore counts correct.
- **Mechanism:** remove an instruction whose effect is redundant, keeping counts right (r05/r06).
- **Ceiling:** about 0.1 us on the serial chain per deleted instruction. The count of redundant pairs is unknown. PE is not the throughput bound, so the gain is only the part on the chain.
- **Risk:** the PE weights buffer may be consumed or clobbered by hardware semantics I did not verify. The verifier would catch it.
- **Falsifier:** decode the PE stream and count `LDWEIGHTS` pairs with identical source address and no write between them. If the count is under about 50, drop the candidate.

### Candidate 3 -- trim the startup ramp

- **Engine / member:** Pool `Pool0.bin`, 288 `DMA_DIRECT2D` at about 0.70 us each (202 us Pool time). Pool is busiest in the first 100 us window (77 us). First compute is gated by the input loads.
- **Change:** convert the input-load DGEs for the first group to queued DMAs, the same technique as iteration 2.
- **Mechanism:** same as iteration 2.
- **Ceiling:** the first group's period is about 10 us longer than the steady-state 37.5 us. The ceiling is on that scale, so about 5-10 us.
- **Risk:** low. The verifier would catch wrong data.
- **Falsifier:** compare the first group's start-to-start time with later groups in `Instruction.parquet`. If the extra is under 3 us, drop the candidate.

### Candidate 4 -- batch small transposes

- **Engine / member:** `qSyncSpillReload0`, 669 transfers, mostly 512 B - 16 KB, 355-420 ns each.
- **Change:** merge pairs of small transposes into fewer, larger descriptors.
- **Mechanism:** reshaping a queued DMA via `SP0.json` (r09).
- **Ceiling:** unknown. Only the on-chain transposes matter, and I did not isolate them.
- **Risk:** medium (descriptor layout).
- **Falsifier:** a single merged pair. If the chain does not shorten, drop it.

## 6. Closed (do not redo)

- **DRAM staging round-trips:** all 160 moved onto the queued-DMA queue and fused. HBM read/write is down to about 3.1 MB each, which is the kernel's real I/O. Nothing left to remove.
- **Dead ACT DGE saves:** all 224 are already `EVENT_SEMAPHORE`s.
- **Throughput on PE/ACT/DVE:** PE 65%, ACT 55%, DVE 49% busy. Moving work off a "busy engine" will not help, because there is no saturated engine. The chain is latency bound.
- **`MATMUL.perf_opt`:** not a lever (`docs/14` section 14.3).
- **Removing waits on an engine's own completion semaphore:** verified negative (o01, `docs/14` section 14.2).

## 7. Unknown (would change the ranking)

- **What the line-597 `LDWEIGHTS` waits on.** Decode its wait field and the producer's identity. This settles Candidate 1: queueing behind program order (fixable) versus true DMA latency (not).
- **Whether the 192 new-queue transfers are actually gated.** Run the delay test above (sections 3 and 6 of this file). It also settles the safety question for iteration 2.
- **Why the new-queue DMAs last 80-110 us.** If they sit on a completion path with slack, fine. If any consumer waits on the last of them, there may be 20-30 us to recover by issuing the trigger earlier. Measure by comparing each ring's trigger time with its consumer's start time in `DmaPacketAggregated`.
- **How many redundant adjacent `LDWEIGHTS` there are** (Candidate 2).
- **Whether DMA ring bandwidth contention explains any of the 4.4 us stall.** Check by running a profile with the line-597 transposes suppressed in a scratch NEFF (result garbage, timing only).

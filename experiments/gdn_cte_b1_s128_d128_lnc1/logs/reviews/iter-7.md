# Review of iter-7 (kernel.neff, working tree vs HEAD 2106d0b)

## 1. What changed
The iter-7 change is the working tree against HEAD; against `baseline/` the diff also contains iters 1-6.
Diffed with `neff_diff.py` (HEAD vs working tree), then unpacked and decoded:

| member | change |
|---|---|
| `sg00/Activation0.bin` slot 81 | the sem-16 `PSEUDO_DMA_DIRECT2D` row (waits S[6]>=65) becomes an update-less `EVENT_SEMAPHORE`, a stand-in that keeps the slot count |
| `sg00/SP0.bin` | slot 40: the sem-17 DGE row (waited S[6]>=66) becomes a `PSEUDO_DMATRIGGER` on `qSPIO0` that waits S[61]>=1. Slots 41 and 42 are new triggers: slot 41 on `qSPIO1` waits S[6]>=65, slot 42 on `qSPIO0` waits S[6]>=66 |
| `sg00/SP0.json` | three static queued-DMA blocks: `iter7-output_1-store` (SB->output_1, fp32 512x128, sem 18, qSPIO0 id0), `iter7-descriptor-fetch-warmup` (a 4-element SB->SB self-copy at SB 227584, sem 19, qSPIO0 id1), `iter7-output_0-hi-store` (SB->output_0 fp32->bf16 cast, sem 20, qSPIO1 id0) |
| `sg00/def.json` | two new queues, `qSPIO0` and `qSPIO1` (16 queues each, owner sp) |
| `sg00/Pool0.bin` slot 22 | the end fence's wait indices change: events_wait_idx 17->18 and extended wait 16->20, so it now waits on S[18] and S[20] (the new store semaphores) instead of S[17] and S[16] |

- **What it does:** the two output-store DGE rows become pre-posted queued DMAs. A queued trigger starts moving data about 0.7 us after the trigger. A DGE row needs about 1.6 us from issue to first packet.
- **Evidence:** in `/tmp/rv/prof`, the Sync WRITE at 49.14 leads to qSyncIO0 packets at 49.89. Sync[24] DGE at 38.83 leads to the first Q0 packets at 40.42.
- **Consistency with the notes:** `notes/` contains only the README. There is no iter-7 note and no iter-6 note, although the HEAD commit cites `notes/iter-6.md`. The change itself matches the three `iter7-` comments in the JSON, but nothing says what was tried and refuted. The next agent should write the notes.
- **Warmup copy:** SB 227584 is inside the range other DMAs already use. The maximum intra-partition offset in the existing DMAs is 228864. The copy is a same-address self-copy, so it is harmless.

## 2. Is the speed-up real?
Yes. I ran HEAD and the working tree interleaved, three pairs of 7 captures each:

| | median of each run (us) |
|---|---|
| HEAD | 38.162, 38.046, 38.146 |
| iter-7 | 37.625, 37.649, 37.627 |

- **Size of the win:** about -0.5 us (-1.3%). The three runs are non-overlapping in median, and the iter-7 runs have a tight spread of about 0.3 us.
- **Agreement:** `profile_latest.json` shows 37.635 (baseline 48.519, -22.4% cumulative).
- **Caveat:** the HEAD commit message quotes HEAD at 38.11. I measured 38.05-38.16, so the two agree.
- **Verdict:** this is not noise.

## 3. Safety
- **Gating is preserved exactly.** The old sem-16 row waited S[6]>=65, and the new qSPIO1 trigger waits S[6]>=65. The old sem-17 row waited S[6]>=66, and the new qSPIO0 trigger waits S[6]>=66. Nothing was loosened.
- **Semaphores** (`semaphores.py` on the unpacked copy): S18, S19 and S20 each show total_inc 16, all from the DMA queues.
  - The end fence (Pool slot 22) waits on S18 and S20, so both output stores are fenced before the kernel ends.
  - S19 (warmup) and S20 have no `WAIT_FOR_SEM_GE` waiter in the table. S20 is covered only through the extended wait on Pool slot 22. S19 is covered by nothing, and does not need to be: it is a 4-byte same-address self-copy.
- **Hazard to remember:** anything that is queued and unfenced can race, so keep adding the fence for each new store sem.
- **Error:** identical to baseline. output_0 is 6.1e-5, which is 17% of the limit. output_1 is 2.864e-4, which is **82% of the 3.5e-4 limit**. That is the baseline value, unchanged by every iteration. The headroom is thin but untouched, and there is no arithmetic edit in the diff.
- **Verdict:** it is safe beyond the verifier, with one caution. The output stores are now static descriptors tied to fixed SB addresses. They stay correct only if no later edit moves the SB buffers.

## 4. Ranked candidates for iter-8
The profile was taken from `tools/profile.py --repeats 1 --parquet`. Kernel body is about 12.4 to 50.5 us in that clock. The 0-12.4 us preamble and the epilogue after 50.5 us are outside the metric.

Ranked by gain per risk and effort:

| # | edit | mechanism | ceiling | risk | cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Move the prologue Pool DGE rows to idle engines.** GpSimd compiler_pc 7-11 are five back-to-back `DMA_DIRECT2D` rows, 13.63 to 17.18 (about 0.65-0.73 us each). Sync is idle from 13.2 to 16.2 after its pc1 row. Scalar is idle from 13.84 after `ACT_TABLE_LOAD`. Rehost rows 9-11 on Sync and Scalar, keeping their sems 9-13 and 15, and give a stand-in `EVENT_SEMAPHORE` on Pool to keep program order | the same rehosting precedent as commits ac545ca and e833bbc (DGE rows moved to idle engines) | First PE MATMUL is at 15.85. The next waves are at 16.8 and 18.4, probably gated by pc10 and pc11 completing at 16.44 and 17.18 plus DMA latency. Realistic gain is **0.5-1.5 us**. The bound is the Pool issue chain itself, 3.6 us. | Moving a row changes which engine's queue runs it. The verifier catches data errors but not a missed wait, so check each waiter still waits the same sem value. | Move only the last row (pc11) to Sync before its slot-2 WRITE. If the first compute at 18.4 does not move earlier, drop the idea. |
| 2 | **Convert the prologue input-load DGE rows to queued triggers.** Same mechanism as iter-7: a static block in `GpSimd0.json` or `SP0.json` plus a new queue in `def.json`, with `PSEUDO_DMATRIGGER` slots at t=12.4 | the trigger->packet latency is about 0.7 us against about 1.6 us for DGE, measured in the tail. A trigger also costs about 0.05 us of engine time against about 0.7 us per DGE row | up to the sum of #1 plus about 0.9 us of latency. Upper bound unknown: it depends on whether the loads are static contiguous or strided. If the input DGEs transpose or gather with a 2-byte element, a static block may be much slower. | the descriptor count and shape must be reproduced exactly. The verifier will catch a wrong layout, not a wrong queue count. | Build one load first. Compare its packet-completion time with the existing DGE packets in `DmaPacket.parquet`. |
| 3 | **Convert the mid-kernel round-trip DGE rows** (Pool[14] at 38.61, Sync[24] at 38.83, Scalar[66] at 38.91, Sync[25] at 39.53) to queued triggers | as #2 | **PE is idle 38.3 to 42.6, about 4.3 us**. The packets run 40.42 to 40.84, so DGE issue to first packet is 1.6 us. A queued trigger would start the same data about 0.8 us sooner. I estimate 0.5-0.8 us if the rows are static; I did not isolate the critical path. | the same as #2, plus the S[4]/S[45] ordering the earlier iterations balanced | Convert one of the four rows. Check whether `PE MATMUL pc129` (42.61) moves. |
| 4 | **Final store tail straggler.** The last output_1 queue (E78) starts 50.298, 0.2 us after the other 15 (50.02-50.09) | a 17th or 2nd-queue split of that queue's descriptor, or issuing the qSPIO0 trigger earlier by 0.2 us | **<=0.2 us**, set by the E78 start minus the median start | very low | In `DmaPacket.parquet`, compare E78's start with the others after the change. |
| 5 | **Find what Scalar pc69 waits on** (TENSOR_SCALAR at 42.31, evt_wait 2306 ns) and what PE pc128/129 wait on (evt_wait 4979 ns) | not an edit. This is the unresolved serialisation after the 38.3-42.6 stall | unknown | none | Decode the wait indices of Scalar slot 69 and PE slots 128-130, then map them to the DMA completions in `semaphores.py`. |

I ranked #1 first because it is a re-hosting of the same kind as the prior wins. #2 and #3 are the larger levers, but each needs a static-descriptor block and carries a layout risk.

**Closed** (from the commit messages of this branch; the iter-6 notes are not in this worktree):
- DGE `sem_increment` tuning, no effect.
- Moving the Vector CAST to Activation or to Pool, no gain.
- Rotating the wait-less Vector TT, no gain.
- The Q1/Q2 store-channel contention theory, refuted on device.
- `ACT_TABLE_LOAD` at 12.56-13.84 is not critical: first input data arrives about 15 us.
- The tail's 2.6 us of fixed DMA latency and the two-branch diamond: 0.16 us margin and 0.08 us of schedulable slack, per commit 2106d0b.

**Unknown** (measurements that would change the ranking):
- Which compute op first needs which prologue load. Decode the Pool pc7-11 descriptors against the first consumers; this decides #1 versus #2.
- Whether the mid-kernel DGE rows are static. Check their `DGETable0` index and descriptors.
- The size of the iter-7 warmup copy's own contribution. Remove just the warmup and measure; if it contributes 0, drop it from `SP0.json`.
- The remaining head-room is about 9 us from PE, Scalar and Vector being only 52-55% busy, but most of it is the DMA round-trips at 38-42 us and the 2.6 us tail latency. These are not obviously reducible at NEFF level without changing data movement.

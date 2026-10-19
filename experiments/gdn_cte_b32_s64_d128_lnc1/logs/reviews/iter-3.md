# Review of iter-3 (commit aafb305)

Scope: `kernel.neff` at HEAD against HEAD~1 (this iteration) and against `baseline/`. `notes/README.md` is the empty template, so there are no agent notes. The only description of the edit is the commit message.

## 1. What changed

Diff HEAD~1 -> HEAD: 2 members changed, 32 identical.

- `sg00/SP0.bin`: 1329 slots, 438 differ. This is only a reordering and retyping of SP slots:
  - 10 slots go EVENT_SEMAPHORE -> PSEUDO_DMATRIGGER.
  - 4 slots go PSEUDO_DMATRIGGER -> EVENT_SEMAPHORE.
  - The rest are moved or renumbered slots, with the trigger and wait fields shifted.
- `sg00/SP0.json`: the 161 transpose descriptors change.
  - `from_off` goes from a Ball offset (114816) to an A_full offset (20988672).
  - `from_sizes` and `from_steps` are composed, for example [32,16,2,1] / [1,262144,128,...] -> [..., 4194336].
  - This matches the claim of reading A_full directly and removing the A_full -> Ball hop.

Against the baseline, `def.json` gains the `qSPSpillReload1` and `qSPSpillReload2` queues (16 rings each, semaphores 72-103). `Activation0.bin` has 160 DMA_DIRECT2D slots turned into EVENT_SEMAPHORE. Those two changes come from the earlier iterations (86fffa6).

The commit message is consistent with the diff. The "cell-set verified" part (checking the composed access pattern covers the same cells) is not in the repo, so I did not re-check it.

## 2. Is the gain real?

Yes.

- HEAD~1 is 873.6 us (per the commit log).
- Now: median 835.85 us, samples 834.2–838.9, spread 0.56%.
- Baseline: 1184.96 us, spread 0.15%.
- The iteration gain is -37.7 us (-4.3%) against a 4.7 us sample range. That is about 8x the noise. It is not noise.
- Overall gain vs baseline is 1.418x.

## 3. Safety

- **Output error is unchanged.**
  - output_0 max_abs_err is 7.7e-5 (11% of the limit).
  - output_1 max_abs_err is 5.84e-4 (82% of the limit).
  - Both match the baseline value, so the numbers are bit-identical. The edit does not touch arithmetic. output_1 was already at 82% in the baseline.
- **Semaphores.** `semaphores.py` on the unpacked HEAD shows `max_wait <= total_inc` for every semaphore. Examples: sem 5 is 1841/1841, sems 24–39 are 224/224, and the qSPSpillReload1/2 rings are 160/160 with a few at 176. There are no waits that exceed their supply, so no hang risk and no under-supplied waits. Wait counts were not loosened: the 160/160 rings are exact.
- **Residual risk: the moved SP triggers.**
  - The transposes now read A_full directly, and their triggers were hoisted to just after the Ball copy's trigger.
  - That relies on the trigger carrying an "A_full ready" wait.
  - The wait is present, and the verifier passes. The check is dynamic, though: one fixed input set cannot expose a race between a hoisted trigger and the A_full writer if the wait is weaker than the true dependency.
  - Sanity check: run `verify.py --runs 3` and compare the output bytes (sha) across runs.

## 4. Ranked candidates for the next iteration

### What the profile says

Source: `prof/Instruction.parquet`, 852 us span.

- PE is the busiest engine by instruction time: 660 us (77%).
  - MATMUL 455 us over 2048 instructions: 1792 are 16x16 at median 213 ns, and the rest are 128x64, 64x64 and 64x128.
  - LDWEIGHTS 170 us.
  - EVENT_SEMAPHORE 32 us.
- DVE is about 55% busy, Scalar about 52%, Pool about 35%, SP about 25%.
- PE has 570 us of gaps in total. 78 gaps over 1.5 us sum to 255 us (the table below counts gaps over 1 us). They break down by the semaphore the next PE instruction waits on:

| PE waits on | gaps >1 us | total gap |
|---|---|---|
| qSyncDynamicHW (sems 56–71) | 33 | 143.7 us |
| qSyncSpillReload0 (sems 24–39) | 34 | 70.7 us |
| Vector | 45 | 58.8 us |
| Scalar | 24 | 53.1 us |

- The qSyncDynamicHW stalls come every ~26 us (about once per batch group). Each is 4–7.5 us, for example t=80.5 us (7.5 us) and t=231 us (7.4 us).
- Those two DMA-queue waits are the largest removable slack: 214 us of PE gaps wait on DMA completion. That is not all recoverable.

### Candidates

| # | Engine / member / slots | Change | Mechanism | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | SP (`SP0.bin`/`SP0.json`) and PE wait slots on sems 56–71 (qSyncDynamicHW). The first one in the stream is the LDWEIGHTS at t≈24 us. | Find which descriptors increment sems 56–71. These are DGE DMAs from SP or `qSyncDynamicHW`. Hoist their triggers to the earliest point their source is ready, as iter-3 did with the transposes. If the source is a DRAM round trip, fuse it into an SB->SB hop. | Same mechanism as iter-1/2/3 here: queued DMA has a fixed ~2 us latency per hop, and a trigger issued after the PE already needs the data shows up as a PE stall. Established on-device in this loop (873.6 -> 835.6 us) and in the sibling run `neff_doc/experiments/gdn_cte_b32_s128_d128_lnc1` (iters 1–6). | Ceiling is 143.7 us (sum of PE gaps >1 us waiting on those semaphores). Realistic: 30–60 us (4–7%). | Low-medium. A wrong hoist races, and a fixed-input verifier can still pass. Keep every producer wait. | For each of the 33 gaps, print the producing DMA's trigger time from `Instruction.parquet` and compare it to the time its source was written. If the trigger is already early and the gap is raw transfer time, the lever is dead. |
| 2 | Pool/SP relay for qSyncSpillReload0 (sems 24–39, 224 incs each) | Same analysis: the remaining A_full / Ball chain that feeds PE. Check whether qSPSpillReload0 still has a two-hop chain, and fuse or hoist it. | As candidate 1. | Ceiling is 70.7 us (34 PE gaps). Realistic: 15–30 us. | Same as above. | Same trace: trigger time vs source-ready time for the 34 gaps. |
| 3 | Pool `EVENT_SEMAPHORE` relays (834 instructions, 158 us busy on the Pool queue) | Retarget the relay so the DMA completion increments the consumer's semaphore directly, which deletes the relay hop. The sibling run did this in iter-4/6 ("retargeting the queued-DMA queue to increment the consumer's semaphore"). | Documented in the sibling run's logs, not yet checked on this NEFF. | Unknown. Bound: at most the Pool relay latency times the number of relays on the critical chain. | Medium. A dropped relay can leave the consumer's wait count wrong. `semaphores.py` catches count errors. | Measure the Pool EVENT_SEMAPHORE start-to-DMA-complete delay on the PE critical chain. If it is under 200 ns, it is dead. |
| 4 | DVE/ACT hand-offs to PE: the 45 Vector gaps (58.8 us) and 24 Scalar gaps (53.1 us) | Rebalance engine assignment: move a CAST or COPY from the busier engine to the idle one (Pool is about 35% busy). | Engine-balancing as in the sibling run (it moved work off Scalar). | Ceiling is about 110 us in gaps; realistic: under 20 us. The tail (t>700 us) has 41.8 us of Vector waits. | Medium-high. Changing the engine changes instruction encoding and rounding behaviour. Pool CAST and DVE CAST may round differently, which would move output_1 (already at 82% of the limit). | Check bit-identity of one moved instruction type in isolation first. |
| 5 | PE LDWEIGHTS (1742 instructions, 170 us) | Drop an LDWEIGHTS where the next matmul reuses the same weights. | No reuse mechanism is documented for this NEFF. | Ceiling 170 us, realistic about 0–10 us. | High. A wrong weight load is silent. | Check whether the 306 matmuls without a preceding LDWEIGHTS already show reuse. If the count of unique weight addresses equals the LDWEIGHTS count, it is dead. |

I would try 1 then 2 in one pass, since both are the same trigger-vs-ready analysis. 3 is the next best.

### Closed

- The two-hop A_full -> Ball -> transpose chain is closed by this iteration: 876.0 -> 835.6 us.
- SP DGE 16x16 DMA_TRANSPOSE slots and the SBUF->DRAM->SBUF staging round-trips are closed by iter-2: 1184.6 -> 873.6 us.
- SP head-of-line blocking is closed for this kernel: SP is down to 34 us busy (4%), from 480 us in the baseline.
- I did not find any earlier refutation for this kernel; there are no notes.

### Unknown

- **Which descriptors sit behind sems 56–71.** Measurement: group `SP0.json` entries by their `semaphore` field, and list the trigger slot index and time. That settles candidates 1 and 2.
- **Is PE the real floor?** PE instruction time is 660 us (77%) against an 836 us span. If gaps were removed completely the floor would be near 700 us. A cut to the 16x16 matmul count would change that, but I found no NEFF-level way to do it without changing arithmetic.
- **The cause of the Vector tail.** 41.8 us of PE waits on Vector in the last 150 us. Measurement: list the DVE instructions around t>700 us and see whether they wait on something avoidable.
- I did not run the candidates, so the gains above are bounds from the profile, not measurements.

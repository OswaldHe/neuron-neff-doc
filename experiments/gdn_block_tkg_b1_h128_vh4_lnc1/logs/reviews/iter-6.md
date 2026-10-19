# Review of iter-6 (kernel.neff, HEAD dd179c9 -> working tree)

Note: `baseline/` is the unedited NEFF (218 us). The iteration's real parent is HEAD (51.0 us), so I diffed against `git show HEAD:kernel.neff`. `notes/` holds only the README template, so the iteration left no notes and I could not check the change against what the agent says it did.

## 1. What changed

Only two package members changed: `sg00/Activation0.bin` (127 -> 119 slots, 64 slots differ) and `sg00/DVE0.bin` (119 -> 127 slots, 36 slots differ). Pool, SP and PE are byte-identical.

- **Eight CASTs moved from Activation to DVE.** Activation CAST count went 32 -> 24 and DVE CAST count 17 -> 25. Activation's eight untriggered CAST/ACTIVATE slots were deleted: four 128-channel bf16->bf16 CASTs (old slots 68-71) and four 1-channel bf16->fp32 CASTs (old slots 74-77).
- **DVE's new CASTs replace slots that held other work.** DVE slots 91-98 are now CAST. Slots 100-110 shifted and now hold TENSOR_TENSOR_ARITH / LOAD_MASK_SELECT / STREAM_SHUFFLE where the old stream had different ops. I did not decode these one by one.
- **Handshake with Activation.** The last of those CASTs (DVE slot 98) bumps semaphore 48 by 16 (SEM_ADD_IMM_COMPLETE); a second DVE CAST bumps semaphore 47 by 16. Activation waits on 47 or 48 at its ACTIVATE func 36 (slot 71) and ACTIVATE func 30 (slot 84).
- **One silu on Activation became four.** The single func-36 ACTIVATE (old slot 82) is now four back-to-back func-36 ACTIVATEs (slots 71-74).
- **Table-load placement.** The ACTIVATION_TABLE_LOAD count is still 7. One load swapped places with a CAST (old slots 59/60) and an ACTIVATE/CAST pair swapped order (old slots 61/62).
- **Two waits rewired.**
  - Activation slot 54: wait on sem 28 (a qSPSpillReload0 DMA completion) became a wait on sem 67 with the same value, 16.
  - Activation old slot 82: wait on sem 38 (Activation's own EVENT_SEMAPHORE post) became a wait on sem 47.
- **One wait dropped.** Old Activation slot 92 waited on sem 37 (Activation's own post); it is gone.
- **Orphaned producers.** Sems 28 and 38 are still incremented but nobody waits on them now.

In short, this is "rebalance engine load plus rewire waits": it moves cheap work off Scalar, the 79.5%-busy engine. It does not delete any instruction, and I believe it keeps the arithmetic identical (the "bit-identical" part is not checked here; see section 3). I cannot confirm this matches what the agent intended, since there are no notes.

## 2. Is the latency change real?

Yes. `profile_latest.json` has median 47.695 us over 5 samples, range 47.655-47.780, spread 0.26%. My re-run of 3 repeats gave 47.625 / 47.700 / 47.634, median 47.634, spread 0.16%.

The parent was recorded at 51.0 us in its commit message, so the gain is about 3.4 us. That is more than 50x the spread (about 0.06-0.12 us), so the win is not noise. The baseline's own spread was 1.3%, but that does not matter at this size. The remaining caveat is that the 51.0 figure comes from a commit message and I did not re-measure HEAD.

Engine busy time: Activation 37.85 us (79.5%), Pool 23.9 us, DMA 19.5 us, DVE 18.9 us, PE 7.3 us, SP 6.6 us.

## 3. Safety beyond the verifier

`semaphores.py` finds no wait that exceeds its increments, either before or after the edit. I checked the three semaphore changes by hand.

- **Sem 67 (Activation slot 54).**
  - Sem 67 is 16 from SP's DGE (SP slot 18) plus 16 from `SP:EVENT_SEMAPHORE` (SP slot 56). The bare count 16 could in principle come from the event semaphore alone, which is the hazard the previous commit documented.
  - Here it cannot: SP slots 19-21 wait for 16 on 67 before slot 56 posts. So 16 on 67 always means "SP's DGE at slot 18 has completed".
  - **Unverified:** whether that DMA's destination is what Activation slot 55 reads. The old wait was on an unrelated qSPSpillReload0 completion. Check SP slot 18's destination against slot 55's source.
- **Sems 47 and 48 (new).** Each has exactly one producer (a DVE CAST adding 16) and one waiter at 16, so the wait is exact. It is correct only if the DVE CAST really is the last writer of the data that ACTIVATEs 71 and 84 read. The `COMPLETE` update modes post after the write, which is right.
- **Sems 37 and 38 (self-posted by Activation).** Activation waits on its own earlier post; the dropped wait is in-order on one engine, so the risk is at most a few ns of write-to-read latency. Low risk but not zero.
- **Not a regression, but worth knowing.** The verifier uses one fixed input set; missing-wait races pass silently.
- **Error margin.** output_0 max_abs_err is 2.999e-3 against a limit of 3.6e-3, 83% of the limit. output_1 is 54% and output_2 is 17%. I have no number for how much of that was already present in the parent. The compute is claimed bit-identical, so this should be the same as at HEAD. Compare outputs of HEAD and the working tree bit-for-bit on a few random inputs; a one-liner with `tools/device.py` would settle it.
- **Wait-time check.** Because only dependencies moved (not data), repeating the run 10 times and getting identical outputs would be good evidence. Only 3 runs were done here.

## 4. Where the remaining time is

Timeline from `Instruction.parquet`:

- The kernel's work runs from about 12 us (after the runtime preamble) to about 58.5 us (the last DVE op).
- The tail through about 62 us is the output store and runtime postamble.
- The first ~12 us is preamble: TENSOR_LOADs, DRAINs and semaphore resets. It is not our code and I treat it as fixed.

Findings that drive the ranking:

- **DMA issue is serial.** Activation issues 8 weight DMAs back to back from 12.1 to 19.5 us (about 0.88 us each). Pool issues 14 DMAs back to back from 12.7 to 26.6 us (about 0.7 us each), then 12 more at 32-44 us. SP, 13.9% busy, has only 6 DGE DMAs in the whole kernel.
- **Scalar data-wait gaps in 21-37 us.** Activation (Scalar) idles waiting for data: 1.50 us at 21.5-23.0, 0.53 us at 24.2-24.8, 1.00 us at 27.2-28.2. These are waits on DMA arrival, not on its own work. Total about 3.0 us.
- **Table loads.** Seven loads at 1.283 us each = 9.0 us of Scalar's 37.8 us. Between 30.5 and 36.1 us there are three consecutive loads around a single 0.2 us GELU ACTIVATE (func 5), then Scalar idles 1.13 us at 36.8-37.9.
- **Serial tail, 43.9-58.5 us.** This is a Scalar -> PE -> Vector -> Scalar recurrence. Scalar ops of 0.2-0.5 us have gaps of 0.3-1.4 us between them, totalling about 4.8 us of Scalar idle. Pool's last DMA (pc 128, the 1024-byte store) is issued at 56.55 and the engines wait for its completion at about 60.5 us.
- **Last compute to store completion.** The last compute ends at 58.52 us, the store completes at about 60.5 us, so about 2 us from last compute to the output landing. Sync's 5 EVENT_SEMAPHOREs (58.5-58.7) and 2 WRITEs (58.8-58.9) sit on that path.

## 5. Ranked candidates for iter-7

| # | Edit | Mechanism / source | Gain and ceiling | Risk | Cheapest falsifier |
|---|------|--------------------|------------------|------|---------------------|
| 1 | **Spread the Pool DMA run onto SP's idle DGE queues.** Pool pc 85-90 (the last 6 of the 14 DMAs at 22.4-26.6) and pc 103-107 (32.4-34.6) go to unused DGE slots on SP. | iter-2 moved 8 weight loads to Activation's unused DGE queues with their own semaphores 80-87 (fa6ac0a). SP is 13.9% busy. Inserting slots into a stream is established to work. | Ceiling about 3.0 us: that is the sum of the Scalar data-wait gaps in 21-37 us. Realistically 1-2 us, since not every wait is on those DMAs. | Medium. A missed semaphore count passes the fixed-input verifier but can race, and DGE ordering hangs have happened (see the closed list). Verifier would catch wrong data only if timing is hit. | Move just the last 3 (pc 88-90) to SP and re-profile. If Scalar's evt_wait at pc 78/90 (3.2 us, 3.4 us) does not shrink, drop it. About 10 minutes. |
| 2 | **Move Activation's first ACTIVATION_TABLE_LOAD (pc 73, 19.56-20.84 us) to just after pc 76 (the second COPY) or into the 21.5-23.0 us idle gap.** Only pc 85 (func 9, the first func that needs a table) actually needs it; pc 74-84 are COPY (func 128/CAST/COPY, table-agnostic per e5fb209). | Table map from e5fb209 and `act_info.json`: 30/128/CAST/COPY agnostic. The load is currently ahead of two COPYs that wait 18 ns and 1 ns for data, so the load itself gates them. | Ceiling 1.28 us on the pc 74/76 results, but only if their consumers gate something; the next op (pc 78) is data-gated until 23.0 us anyway, so this could be 0. Unknown. | Low. Same bytes, only slot order; verifier catches a wrong table. | Check who waits on pc 74/76's semaphores. If the earliest consumer starts after 23.0 us, ceiling is 0. Takes 5 minutes with the decoded streams. |
| 3 | **Software-pipeline the 4-head tail (43.9-58.5 us).** The four head iterations of COPY -> ACTIVATE(29/30) -> ACTIVATION_READ_ACCUMULATOR -> DVE STT -> PE MATMUL run one after another with cross-engine waits. Start iteration k+1's Scalar ops while iteration k is on PE/DVE. | Heads are independent in the math; the only ordering is semaphore waits. Evidence: Scalar idle gaps 0.4, 1.4, 0.6, 0.6, 0.3, 0.5 us (about 4.8 us). | Ceiling about 4.8 us (Scalar idle in the tail); realistically 1-3 us. Unknown until the dependency graph is traced. | High. This rewires waits across 3 engines. A race passes verify on one input. Do last. | Trace one iteration's producer/consumer pairs (pc 151-159 vs 168-181); if iteration k+1's first Scalar op already waits only on a late external input rather than on iteration k, there is nothing to gain. |
| 4 | **Shorten the post-compute path to the store.** Sync does 5 EVENT_SEMAPHOREs + 2 WRITEs (58.55-58.94 us) between the last DVE op (58.52) and the doorbell, then the store lands at about 60.5. Remove the redundant EVENT_SEMAPHOREs (158-162, 165-167) if they only mirror waits other engines already hold. | Sync EVENT_SEMAPHOREs cost 0.018-0.04 us each; they precede the WRITEs. | Ceiling about 0.3-0.6 us (the Sync chain). The store completion itself (about 1.6 us after the doorbell) looks fixed. | Medium: these may be the only wait for the last DVE write. A removed wait can pass and race. | Delete one wait, run verify 10 times; also check that the semaphore it waits on has another waiter. |
| 5 | **Remove one of the two table loads around GELU (func 5).** Loads pc 108 and pc 114 bracket one 0.2 us ACTIVATE (pc 109). | If func 5's table matches silu's (36) or the 29/30 set, one load disappears. `act_info.json` says gelu is in `gelu_and_others` only, which is neither. | Ceiling about 1.1 us: Scalar idles 1.13 us at 36.8-37.9 afterwards. Likely 0, since GELU needs its own table. | Low if it does not work (output wrong; verifier catches it). | Replace the pc 108 table load with a no-op and see whether verify still passes. If it fails, drop it. 2 minutes. |

Ranking is by gain per risk and effort. #1 has the biggest evidenced ceiling at moderate effort. #2 and #5 are cheap to test. #3 has the biggest ceiling but needs the dependency trace and has the highest risk.

## 6. Closed

These are taken from the commit messages and this session's measurements, not re-run here.

- DGE is one serial resource: about 15 ns per 2-byte transpose descriptor, ~1536 of them between 34 and 50 us (e5fb209). Edits that only start a transfer earlier lost time.
- `dma_configs_dma_transpose_2d_tile` is inert (e5fb209).
- Replacing a DMA with a semaphore add can let a wait on an untouched semaphore pass early; sems 68-70 hit their threshold of 32 from two event semaphores alone and hung the device while verify.py passed (dd179c9).
- Merging tile loads into one 2-D DMA is only correct in the last DGE slot of an engine (dd179c9).
- A strided multi-element compute CAST does not gather what its step field says (dd179c9).
- Skipping middle queue blocks hangs the device; trailing untriggered ones are fine (dd179c9).
- fp32 [128,1] results are 512 bytes and alias 256-byte bf16 matmul operands that PE reads without a semaphore; give such ops a private in-place destination (dd179c9).
- Narrow SBUF->DRAM->SBUF round trips cost about 135 ns per element; do not reintroduce them (480fb81).
- Per-partition SBUF->SBUF DMAs lose to compute CASTs: 0.25 us versus 0.55 us issue plus a 2-5 us completion hop (fa6ac0a).
- Table switches: 36=silu, 29=rsqrt, 30/128/CAST/COPY are table-agnostic (e5fb209).

## 7. Unknown

| Question | Measurement that settles it |
|----------|------------------------------|
| Does the 8-CAST move to DVE preserve bits at HEAD? | Run HEAD and the working-tree NEFF on 10 random input sets and diff the outputs byte-for-byte. |
| How much of output_0's 83% of limit is new? | Same comparison; also run baseline/kernel.neff for a floor. |
| Does SP slot 18's destination match what Activation slot 55 reads (the rewired sem-28 -> sem-67 wait)? | Decode the DGE descriptor at SP slot 18 and the source of ACTIVATE slot 55. |
| Which engine is waiting on which DMA at 21.5-28.2 us? | Per-engine start_ts versus DmaPacket end_ts on `prof/DmaPacket.parquet`, for the Pool pc 85-90 transfers. |
| Is the 2 us from last compute to store completion fixed? | Compare the DmaPacket timestamp of the 1028-byte output write with the Sync WRITE doorbell time across 3 runs. |
| What is DMA 128 (Pool, 56.55 us) storing? | Decode its descriptors; if it is not the output store, there may be a second store that can be issued earlier. |

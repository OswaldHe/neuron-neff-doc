# Review of iter-2 (kernel.neff vs HEAD 86e9bd9 = iter-1 result; baseline = unedited)

`notes/` is empty (README template only), so there is nothing to compare the diff against. Every claim below comes from the NEFF.

## 1. What changed
Against the baseline the NEFF has five changed members. Most of that is iter-1's work, which is already committed. This iteration's own diff (HEAD -> working tree) touches only `sg00/Pool0.bin` (35 -> 37 slots) and `sg00/SP0.bin` (64 -> 66 slots). It is a swap of which queue issues four DGE (`PSEUDO_DMA_DIRECT2D`) DMAs, plus the instructions that move with them:
- Two DMAs were moved from the SP stream to the Pool stream. They write sem 62/63 and are the old SP slots 14/15, now `EVENT_SEMAPHORE` stubs.
- Two DMAs were moved from the Pool stream to the SP stream. They write sem 20/21 and appear as SP slots 64/65. Their sources are the DRAM scratch addresses 0x4b00/0x4f80 -> SBUF.
- Pool slots 18-32 are shifted and reordered, with new `semaphore`, `events_semaphore_value` and src/dst address fields. Pool slot 33's double wait (sem 21>=16 and sem 20>=16) is deleted and re-emitted at slot 35. Slot 34 (NOP) and slot 36 are also re-emitted.
- One new Pool `EVENT_SEMAPHORE` with update idx 16 has no waiter. One new SP `EVENT_SEMAPHORE` on sem 64 is balanced (1 increment, 1 wait).

The commit message for this iteration is not available, so there is no stated claim to check. Judging by the diff, the intent was to rebalance DGE issue between Pool and SP. The diff is consistent with that.

## 2. Is the latency change real?
Yes, but it is small. All three runs below used 7 repeats and the same tool.

| NEFF | median of 7 | min-max |
|---|---|---|
| HEAD (iter-1), run 1 | 62.134 | 61.909-62.188 |
| HEAD (iter-1), run 2 | 62.106 | 61.779-62.391 |
| working tree, profile_latest.json | 61.245 | 60.844-61.408 |
| working tree, rerun 1 | 60.912 | 60.698-61.307 |
| working tree, rerun 2 | 61.111 | 60.803-61.777 |

- The gain over HEAD is about 1.0 us (-1.6%). That is roughly 2x the within-run spread (0.45-1.6%), and the HEAD and current ranges do not overlap.
- HEAD re-measured at 62.1 us here, versus the 61.905 in its commit message. Machine drift of about 0.2 us is real, so gains under about 0.5 us need A/B interleaving.
- The cumulative gain over baseline is 67.5 -> 61.1 us (-9.5%).

## 3. Safety
- **Semaphores:** `semaphores.py` on the unpacked HEAD and working-tree NEFFs gives no "wait exceeds increments" flag. Every semaphore that has a waiter still has total increments >= max wait. The sem 20/21 and 62/63 producers swapped engine (Pool <-> SP) and kept their counts of 16.
- **Waits moved, not dropped:** the removed Pool wait pair is re-emitted two slots later, so the pair is not lost.
- **Leftover:** sem 16 now has an increment and no waiter. That is harmless.
- **What to watch:** this is a pure reorder of DMA issue between queues. It can pass the verifier's one fixed input and still race if a DMA now issues before its WAR/RAW producer. No wait was reduced, which is the good sign.
- **Error:** unchanged. output_0 max_abs_err is 6.09e-5 (11% of the 5.4e-4 limit). output_1 is 4.49e-4 (83% of the limit). Both numbers are identical to the baseline, so nothing moved.
- **Standing hazard:** output_1 sits at 83% of the limit already. Any edit that changes dtype or rounding there is the dangerous one; reordering is not.

## 4. Ranked candidates for iter-3

Profile facts that bound everything below (`prof` run, instruction span 76 us):
- The first ~12 us is runtime preamble (`TENSOR_LOAD`, `COMPARE_BRANCH`, semaphore setup). It is not kernel code.
- Each DGE `PSEUDO_DMA_DIRECT2D` occupies its issuing engine for 0.6-0.9 us. There are 29 of them.
- In the 12-20 us window Tensor is busy only 1.8 us. Pool issues 8 DMAs back to back (12.8 -> 19.4 us, `gdn_cte.py:169-207`), while Sync is idle until 25.4 us.
- The tail from 71 to 74.6 us has Tensor, Scalar and Vector parked on a semaphore for 3.1-3.8 us, waiting for the final output DMA to land.

| # | Edit | Engine / member / slots | Mechanism | Expected gain and ceiling | Risk and what catches it | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Spread the 8 prologue load DMAs over Pool and SP (and Act after its table load, 13.5 us) | `Pool0.bin` DMA slots feeding `gdn_cte.py:169-207`. Re-emit alternate ones in `SP0.bin` or `Activation0.bin` with the matching `semaphore` / `events_*` fields. | Same mechanism as this iteration and iter-1: DGE DMAs can be re-emitted on another engine with its own sem pair, with waits and sem accounting preserved. | 1-2.5 us. Ceiling = Pool serial issue 12.8 -> 19.4 (5.6 us) cut to about half, bounded by the first matmul at 16.1 us and data arrival. Unknown whether the later loads are on the critical path. | Low-medium. A wrong semaphore index passes silently only if the producer still finishes first; `semaphores.py` flags the unbalanced cases. | Move only the last 4 loads to SP and measure. If the median is unchanged within 0.3 us, the loads are not critical; stop. |
| 2 | Move the remaining Act-issued DGE DMAs (3 left, near 49.8 / 51.5 / 52.9 us, `gdn_cte.py:483,487`) to Pool | `Activation0.bin` -> `Pool0.bin` (Pool has 0.2 us busy in 40-50) | Same as above. Act is the busiest engine (32.5 us busy, 53%). | 0.5-1.5 us. Ceiling = 3 x 0.65 us of Act time; Sync and Pool wait 1.9 us at 52.8-54.7 us on `gdn_cte.py:495`, which suggests this window is partly critical. | Low. Same semaphore accounting rule. | Move one of the three; if the median does not drop by about 0.3 us, drop the rest. |
| 3 | Split the final output store across two idle queues | `SP0.bin` slots for the two stores at 71.3 and 72.0 us (about 0.7 us each, serial on Sync): re-issue the second from Pool (idle after 68 us) | Same re-emit mechanism; the second store's issue is currently queued behind the first on Sync. | Up to 0.7 us. Ceiling = one DGE issue time. | Low. Output ordering is by semaphore; the verifier compares the final tensors. | Check that the second store's start timestamp moves earlier than 72.0 us in `Instruction.parquet`. |
| 4 | Re-balance DGE issue in 20-30 us (Sync and Pool alternate; Act is busy 8.1 us there) | `SP0.bin`/`Pool0.bin` slots feeding `gdn_cte.py:382` | Same. | Unknown; probably under 0.5 us. | Low. | If no instruction in the window sits waiting on a DMA semaphore, skip. |

No candidate here changes data values, dtypes or arithmetic, so none moves output_1 closer to its limit.

## Closed
- **Staging round-trips via DRAM:** fused to SBUF->SBUF in iter-1 (spill_reload 491520 -> 0 bytes). Done.
- **Act-issued DGE saves:** five moved to Pool/SP in iter-1 (Act busy 37.0 -> 32.5 us). Done.
- **Pool/SP swap of sem 20/21 and 62/63:** done this iteration (62.1 -> 61.1 us).
- **Preamble (first ~12 us) and the end-of-kernel semaphore-range-clear / DRAIN sequence (74.6 -> 76 us):** runtime-generated, not kernel code. Not editable.
- **`ACT_TABLE_LOAD` (1.28 us at 12.3 us):** off the critical path. It finishes at 13.6 us and Scalar's first `ACTIVATE` is at 16.6 us.

## Unknown
- **Which of the 8 prologue loads gate the first matmul?** Settle by bisecting as in candidate 1. The `DmaPacketAggregated` timestamps already show each load's completion.
- **Why total_exec_time (61 us) is about 15 us shorter than the instruction span (76 us).** Settle by reading `Summary.parquet` fields against `Instruction.parquet` start and end. This changes how much of the 12 us preamble is really in the score.
- **Whether the 3-3.8 us tail wait is DMA completion latency or a fixed sync.** Settle by timing the DMA packet end for the last stores in `DmaPacket.parquet`.
- **Whether Tensor is the true critical path in 30-50 us** (10-12 us of Tensor busy per 10 us window). If it is, DMA-issue edits stop paying and the remaining time is mostly irreducible at the NEFF level.

Overall, the remaining NEFF-level lever is DGE-issue placement, worth about 1.5-3 us in total at best. Beyond that the evidence points to an irreducible floor around 58-59 us.

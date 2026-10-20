# Review, iter-0 (gdn_cte_b1_s256_d128)

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports an identical header, 34 of 34 members identical and "payloads are identical".
- `notes/` holds only the README template, so there are no claims to compare.
- `profile_latest.json` and `verify_latest.json` are absent. The iteration produced no edit and no metric.

## 2. Is the latency change real?
There is no change to measure. I re-profiled the unedited `kernel.neff` once: 67.652 us against a baseline median of 67.525 us (samples 67.30–67.94, spread 0.95%). That is +0.19%, which is noise. The unedited NEFF scatters by about ±0.3 us. A win under roughly 0.7 us (1%) is indistinguishable from that scatter in one 7-repeat run. Repeat any candidate at least 2×7 times, interleaved with the baseline.

## 3. Safety
Not applicable, since no edit was made. `semaphores.py` on the unpacked baseline reports every wait within its supply, so it is a valid starting point. Baseline error is 6.09e-05 against a limit of 5.4e-04 (11% of the limit). There is plenty of error headroom, so none of the candidates below touches arithmetic.

## 4. Profile read (1 repeat, `prof/Instruction.parquet`)
- **Span.** The trace spans 82 us. The first ~12 us is runtime preamble (TENSOR_LOAD, table, ordering setup) that precedes the kernel's first MEMSET at 12.27 us. The last ~1.5 us is the postamble. The kernel proper runs from 12.3 to about 79–80.6 us.
- **No engine is saturated.** Busy time is Act 36.8 us (54%), DVE 28 us, DMA 25.5 us, PE 17.4 us, Pool 15 us, SP 9.3 us. This is a dependency-chain kernel, the same regime as the earlier `gdn_cte` experiments, so only small gains are plausible.
- **Phases.**
  - 12.8–20 us: input loads. Eight `DMA_DIRECT2D` descriptors (gdn_cte.py:191, 169, 207×5, 171, 173) are issued back-to-back on the Pool DGE queue. Each takes 0.65–0.8 us of Pool time, and they are serialised on Pool. SP and Act are idle at that point.
  - 21–60 us: the chunk-0 and chunk-1 compute chain, with SB->SB `qSyncSpillReload0` DMA transposes of 0.25–0.6 us each.
  - ~60–79 us: chunks 2 and 3, about 4.4 us per chunk. The output stores (line 623) are issued at 64.7, 69.2, 73.3 and 78.4 us.
- **Tail.** The state store (line 628) is issued at 77.7 us and the last output store (line 623) at 78.4 us, directly after it on the same Pool queue. Then ~1.6 us of DMA-completion wait follows, plus the postamble.
- **Staging round-trips.** Earlier `gdn_cte` loops won big by fusing SBUF->DRAM->SBUF staging round-trips. This NEFF has none: `DmaPacketAggregated` shows 33 transfers, all SB->SB, except one TMPBUF->SB copy of 32 KB at 31.6–33.2 us. That lever is mostly already spent here.

## 5. Candidates, ranked

Gains are small. Total kernel time is ~68 us, and everything below is bounded by the 1% noise floor except possibly C1 and C3.

| # | Candidate | Engine / member | Edit | Mechanism | Expected gain, ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| C1 | Hoist the Pool input-load DGE issues ahead of the iota/mask work, and spread them over idle queues | `Pool0.bin`. Slots of the 8 `DMA_DIRECT2D` at pc lines 191/169/207×5/171/173, and the 3 `TENSOR_SCALAR_AFFINE_SELECT` (118/223/136) and `MOVE` in between. Optionally `SP0.bin` and `Activation0.bin` for 2–4 of the loads. | (a) Reorder in program order so all DMA issues come before the AFFINE_SELECTs (re-point the waits, ch.10.4 R4). The mask constants are first needed at ~22.8 us, but the DMAs sit at 12.8–19.4 us. (b) Move the 207×4 loads to SP's DGE. They need a new ring or semaphore pair, as in the s64/s128 `qSPSpillReload` retarget. | Ch.14.3 #2 "start long-latency work earlier"; ch.9. DGE retarget established in `experiments/gdn_cte_b32_s*/logs/notes` (iters 2–4 of s128). | The first matmul (16.2 us) and the line-234 matmuls (21.3 us) wait on these loads. The last load completes at ~19.4 us plus DMA latency, so the ceiling is the serial queue span of ~6.6 us minus about 2 us of useful overlap. Estimated 1–2 us (1.5–3%); a hard bound is unknown. | Wrong semaphore re-pointing may deadlock. `semaphores.py` catches that. A race (compute reading a tile before its DMA lands) may pass one input set, so run `verify.py --runs 3`. | Reorder only (a). If the last load's completion (DmaPacket) does not move earlier in the next profile, kill it. 10 minutes. |
| C2 | Order the final stores: put the last output store before the state store, or issue the state store from SP | `Pool0.bin`, the DMA at line 628 (77.7 us) and the last line-623 DMA (78.4 us). `SP0.bin` is the alternative. | Swap the two DGE slots, or move 628 to SP. Both need the right ring semaphores. The postamble drain must still wait for both. | Ch.14.3 #2. The tail is one serial Pool queue. | ≤0.65 us (one DGE issue time). It is the final-DMA critical path and no more, and it is at the noise floor. | Dropping the postamble wait for one of the two stores would lose data. Keep both waits. The verifier compares `result` and `state`, so it would catch it. | Check that the new end time (last DMA completion) drops by ~0.6 us in the parquet. If not, discard. |
| C3 | Remove or shorten SP EVENT_SEMAPHORE waits that sit in the per-chunk transpose chain | `SP0.bin`, the 19 EVENT_SEMAPHOREs waiting on sems 24–39, 48 and 50 (value 16). Profile: Sync EVENT_SEMAPHORE totals 14.7 us, with single waits of 1.5 us (63.8–65.3) and 0.83 us (66.8–67.6). | Retarget the DGE ring so the consumer engine's semaphore is incremented directly, instead of SP relaying it (s128 iter-4: "replaced SP relay EVENT_SEMAPHOREs by retargeting the queued-DMA queue"). | `experiments/gdn_cte_b32_s128_d128_lnc1` iters 3–4 (−0.3% and −2.3% there). | Unknown. I did not establish whether these SP waits are relays on the critical path or just SP-local drains. Bounded above by 3 chunks × maybe 0.5–1 us, ~1.5–3 us. | Loosening syncs risks a race that passes once. The verifier would likely not catch a missing wait. Re-run `verify.py --runs 5`, `semaphores.py`, and repeat the timing. | Decode the consumers of each of the 19 sems (`semaphores.py`) and confirm each has exactly one waiter on a different engine. If a consumer already waits on the DMA's own ring semaphore, the SP wait is not on the critical path and the candidate is dead. |
| C4 | Fuse the 4 per-chunk output stores' trigger latency | `Pool0.bin`, the 4 line-623 DMAs | Issue each store from the engine that produces `o_out` (DVE, or Act's DGE) rather than Pool, so the store does not wait on Pool-side AFFINE_SELECT and MOVE ahead of it in program order. Check the program order first. | Ch.14.3 #3 and #2. | ≤0.3–0.5 us, only on the last chunk. | Low. | Look at the Pool stream order around the 623 slots. If no Pool instruction sits between the data-ready semaphore and the DMA issue, drop it. |

I would try C1(a) first (pure reorder, no queue or ring changes), then C2, then investigate C3 by decoding before editing. Combined, the plausible total is 1–3 us (1.5–4.5%), and each item alone is near the noise floor. Make the keep decision on the median of ≥2×7 interleaved runs.

## Closed
No iteration has refuted anything for this kernel yet. From the earlier experiments:
- **Staging round-trip fusion.** This NEFF has no SBUF->DRAM->SBUF round-trips to fuse (all 33 DMAs are SB->SB except one 32 KB TMPBUF->SB), so the −26% win of s64 does not apply.
- **`MATMUL.perf_opt`.** Not a lever (ch.14.3).
- **Removing waits on an engine's own completion semaphore.** Ch.14.2 gives wrong output on DVE and an out-of-bounds error on Pool. Do not do it.

## Unknown
- **Which loads gate the first compute.** Decode the BIR dependencies, or compare the matmul start time at 16.2 and 21.3 us against each load's completion. This settles C1's ceiling.
- **Whether the SP semaphore waits are on the critical path.** Compare the start of the consuming engine instruction against the SP wait end. This settles C3.
- **Noise floor at this size.** Run the baseline 3×7 and compare the spreads. Single-run spread was 0.95%, about 0.64 us.
- **How much of the 12 us preamble and ~1.5 us postamble is counted in `total_exec_time_us`.** Total time is 67.5 us against an 82 us trace span, so about 14 us is excluded and is not editable. The scored window is 12.3–80 us and any gain must come from that.

Overall: the remaining time is mostly a dependent chain through four serial chunks with ~1–2 us DMA-latency hops. Realistically it is worth ≤4% at the NEFF level.

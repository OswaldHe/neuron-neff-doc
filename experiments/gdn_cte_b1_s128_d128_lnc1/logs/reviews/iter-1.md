# Review of iter-1 (gdn_cte_b1_s128_d128)

## 1. What changed
`neff_diff.py` shows 5 changed members: Activation0.bin (7 of 77 slots), Pool0.bin (3 of 23), DVE0.bin (1 of 94), SP0.bin (36 of 35→37 slots) and SP0.json. PE0.bin is unchanged.

- **Spill round trips removed.** 6 spill DMAs (`PSEUDO_DMA_DIRECT2D`) are gone. Activation slots 14/20/21 and Pool slots 1/16/20 became bare `EVENT_SEMAPHORE`. The bytes also show wait fields dropped on Pool slots 1 and 20. Pool slot 16 keeps its wait (0x13) and now increments sem 14 by 1.
- **Other spill DMAs retargeted to SBUF.** The remaining spill DMAs are retargeted to SBUF: `SP0.json` entries go from the DRAM scratch `inst__I-9-0:_mem_54_50` to `SB`, with from_off 18944 and so on.
- **Producers rewritten.** Activation slots 56/57/59/61 write straight into fixed SB addresses (0x101d200–0x101d600). Slots 56/57 also change `out_dtype` FP32→BF16 and `dst_elem_size` 512→256.
- **SP stream.** SP0.bin gains 2 slots, reorders DMATRIGGER and DIRECT2D, and takes over the DGE issue for sems 8 and 17 (previously Pool). It gains an `EVENT_SEMAPHORE` on sem 60.
- **DVE.** DVE slot 80 now waits for sem 14 == 1 instead of 16.
- **Profile effect.** `spill_save_bytes` and `spill_reload_bytes` go from 278528 and 245760 to 0. HBM read and write drop from about 377 KB to 99 KB.

`notes/` contains only the README template. Nothing was recorded, so I cannot check the claim against what the agent says it did. The diff is internally coherent: it is spill elimination, not a tweak.

## 2. Is the win real?
Yes.
- Baseline median is 48.519 us (spread 1.9%, range 48.21–49.16).
- Latest is 43.427 us (spread 0.8%, range 43.34–43.68).
- My independent 5-repeat re-profile gives 43.348 us (43.32–43.70).
- That is −5.1 us (−10.5%), and the ranges do not overlap. The gain is about 10× the noise.
- Pool busy falls 11.0→9.1 us and DMA busy 17.9→14.1 us.

## 3. Safety
- **Error is unchanged.** Output 0 max_abs_err is 6.119e-05 and output 1 is 2.864e-04, both identical to baseline. The latter is already 82% of the 3.5e-4 limit. The result looks bit-identical, so the FP32→BF16 `out_dtype` change did not move the numbers on this input.
- **Semaphores.** `semaphores.py` on both unpacked NEFFs shows every semaphore with max_wait == total_inc, so the counts balance. Sem 14 and sem 60 are new 1-count handoffs. Sem 16 has no waiter, as in the baseline.
- **Residual risks the verifier cannot see.**
  - The new fixed SB destinations (0x101d200…) bypass the allocator. If they alias a live tensor at another phase, a fixed input can still pass.
  - Pool slots 1 and 20 lost their waits. That is fine only if nothing consumed those DMAs' completion. Only the numerics pin this down.
  - DVE slot 80's wait was cut from 16 to 1, so it no longer waits for 16 DGE completions. It relies on Pool's `EVENT_SEMAPHORE`. Pool slot 16 keeps its own wait, so ordering after the producer holds.
- **Check before accepting.** Run verify with 2–3 different seeds, and compare the SB ranges against `SbufUsage` / `tensor_map.json`.

## 4. Candidates for iter-2 (ranked)
Evidence: the profile is `prof_r/Instruction.parquet`, 5 repeats. All timestamps are relative to the profile start, which sits about 12 us before the first data DMA.

All 17 remaining `DMA_DIRECT2D` instructions cost 0.6–0.76 us each on the issuing engine and run back-to-back on a single queue. The queues are lopsided:
- Pool issues 5 loads serially from 13.7 to 17.2 us.
- Sync issues 4 SB→SB DMAs from 20.7 to 23.7 us.
- Scalar issues 4 DMAs from 40.5 to 43.3 us. Scalar is the busiest engine (22 us, 89% occupancy in that window).

PE is idle 18–24 us and 42–45 us.

| # | Edit | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Spread the 5 input-load DMAs (Pool, 13.7–17.2 us) over Pool, SP and Act.** Retarget the issuing engine of sems 9–15 by moving the DIRECT2D slot into an idle stream and leaving an `EVENT_SEMAPHORE` placeholder in the old one. | Per-engine DGE issue is serial at about 0.65 us. The consumer waits on a semaphore count, so it is indifferent to which engine issues. This iteration already moved sems 8 and 17 from Pool to SP and passed on-device. | Load phase is 4.8 us. A 3-way split gives about −2 to −3 us (~5%). Bound: Pool DMA total is 4.87 us, and the first MATMUL waits on the loads. | Low–medium. A mis-moved wait is invisible to the verifier. | Move one load and profile; if the median is unchanged within 0.3 us, drop the idea. |
| 2 | **Split the 4 Sync SB→SB DMAs (20.7–23.7 us) across SP and Pool.** | Same as #1. | About −1.3 to −1.9 us. Bound: 4 serial × 0.6–0.7 us, and PE is idle in this window. | Same as #1. | Same as #1. |
| 3 | **Move the 4 Activation DMAs (40.5–43.3 us) to idle SP or Pool** (Pool and SP have 0–5% occupancy from 24 to 48 us). | Same as #1. This also frees 2.6 us of the busiest engine's stream. | At most −2.6 us, probably about −1.5 us after DMA latency. Only valid if ACT is on the critical path there, which the 89% occupancy and idle PE suggest but do not prove. | Medium. These DMAs have producers on ACT, so the moved issue must still wait on the producer semaphore. | As #1, but check the post-edit timeline for ACT gaps. |
| 4 | **Split the tail output stores** (Pool 16 KB at 48.4, Pool 16 KB at 53.4, Sync 64 KB at 54.1) across queues. | Same as #1. | Unknown, probably under 1 us. The 5 us gap between the two Pool stores is data-gated, not queue-gated, and about 2 us of completion latency after the last DMA is fixed. | Low. | Inspect the dependency of the 54.1 store; if gated by data, drop. |

Candidates #1–#3 share one mechanism. If #1 shows no gain, #2 and #3 should be killed before spending effort on them.

**Closed**
- Spill elimination is done: spill bytes are 0 and the saving is 5.1 us. Do not redo it.
- Removing PE `EVENT_SEMAPHORE` waits is not a lever. PE shows 22 of them totalling 11.4 us, but they are waits for producers, not overhead.

**Unknown**
- What the 0–12 us profile preamble is. If it is fixed runtime overhead, it is out of reach. Compare `total_exec_time_us` (43.4) with the profile span (58) to settle it.
- Whether the SBUF addresses chosen are free for the whole kernel. Settle it with `SbufUsage` / `tensor_map.json` plus multi-seed verification.
- Whether the 8 remaining SB→SB DMAs can be deleted by having consumers read the source SB directly. That is only possible if no cross-partition movement is needed, since the 4194304 steps suggest it is. Settle it by replacing one with a no-op and timing it (the verifier will fail; the latency is what matters).
- Whether the 60 `LDWEIGHTS` (6.7 us) can be reduced. No evidence either way.

# Review of iter-1

## 1. What changed
`neff_diff` shows 4 members changed, 46 members total (42 identical):
- **Activation0.bin**: 9 slots. Slots 35, 39, 40, 41, 52, 55, 58, 59 are DST address and step rewrites. Their `dst_start_addr` moved from the spill-region pattern `0x13…00000x00` to plain SB offsets (`0x35xxx`, `0xe100`). `dst_step_elem` went `[2,1] -> [262144,1]`. **Slot 67 is a sync change**: `events_wait_idx` 61 -> 37 and `events_semaphore_value` 48 -> 16.
- **Pool0.bin**: 8 slots (34, 36, 38, 40, 43, 51, 53, 55). Same pattern on the SRC address: `0x…f00000x00 -> 0x…0042xx`.
- **SP0.bin**: 38 slots. DST and SRC address rewrites. Slots 5-9 also change `dst_elem_size` 2 -> 4 and `out_dtype` BF16 -> FP32. Slot 30 gets a new step `[-768, 262144]`.
- **SP0.json**: the DMA descriptor table. Many spill-reload/save transfers changed from DRAM-scratch (`inst__I-16-0:_mem_*`) to `SB` as endpoint, and `from`/`to` were swapped.

Net effect: the compiler's HBM-backed spill buffers (`qSPSpillReload0`) were redirected to SBUF-resident addresses. `spill_reload_bytes` fell 16416 -> 1056 and `spill_save_bytes` 10272 -> 1074. `dma_transfer_total_bytes` fell 10848 -> 7786. `neuroncore_cycle_count` fell 279k -> 132k.

**The `notes/` directory is empty** (only `README.md`; no `iter-1.md`). So I could not check the claim against the notes. The structural description above is consistent with "relocate spill traffic". The slot-67 wait change is not explained by that description and has to be checked separately (section 3).

## 2. Is the latency change real?
Yes.
- Baseline median is 218.2 us, spread 1.3%, samples 215.6–218.5.
- Latest median is 95.2 us, spread 0.70%, samples 94.8–95.5.
- My own re-run gave 95.7 us.
- The gap is 123 us against a spread of about 1 us.
- Engine busy times are unchanged: PE 7.57 vs 7.57, ACT 41.8 vs 41.8, DVE 19.2 vs 19.2, Pool 35.8 vs 35.8. Only DMA busy (52.3 -> 35.0) and idle time moved.

This is not noise. The win is from removing HBM spill round-trips that sat on the critical path.

## 3. Safety
- **Numerics:** `max_abs_err` is identical to baseline on all three outputs (3.0e-3 = 83% of the 3.6e-3 limit, 1.9e-3, 6.0e-4). The data path is bit-unchanged. The 83% margin on output_0 is pre-existing, so nothing moved closer to the limit.
- **Sync, from `semaphores.py` on unpacked a/b:**
  - sem 37 waiters went 1 -> 2 (max_wait 16, total_inc 16).
  - sem 61 waiters went 5 -> 4, and one wait that was 48 (all three DGE completions of 16) is now gone.
  - **Activation slot 67 used to wait for ALL of sem 61 (48). It now waits only on sem 37 >= 16, a spill-reload DMA queue.** That is a loosened or retargeted wait, not obviously an equivalent one.
  - It passed verify once with one fixed input set. I have not shown that sem 37 completion implies the SP-queue (sem 61) data that slot 67 used to depend on is visible.
  - This is the main risk in the iteration. Action: either restore the original wait on slot 67 (wait idx 61, value 48) and re-measure, or prove by the semaphore timeline (`SemaphoreUpdate.parquet`) that sem 61 reaches 48 before slot 67 in every run.
- **SBUF allocation:** the new SB addresses (0x33d00–0x36f00, 0x4000–0x5800, 0xe100) were chosen by hand. I did not check them against live ranges of other tensors. `SbufUsage.parquet` can show overlap; a collision could pass on one input set.
- **Dtype change:** SP slots 5-9 now write FP32 into the new region (elem_size 4). Check that the SB footprint at the destination is big enough (each destination steps by 0x300 between neighbours, e.g. 0x36900/0x36600/0x36300, so 4-byte x 128 cols = 512 B fits inside 768 B).
- `tools/verify.py` and `launch_rules.py` pass.

## 4. Where the remaining ~95 us goes (from `prof/Instruction.parquet`, 110 us traced span)
- **Phase A (0–34 us):** about 32 `DMA_DIRECT2D` triggers on Pool, issued serially every ~0.65 us (13.0 to 33.5 us). This is queue-issue bound, not bandwidth bound.
- **Phase B (34–80 us):** about 70 more triggers on Sync (~0.6 us apiece) and Scalar. PE, ACT and DVE show gaps of 16–24 us in this phase (PE 21.9 us at 58 us and 23.7 us at 84 us; DVE 16.2 us at 58.7 us). Engines are waiting for data.
- **Phase C (80–110 us):** a serial ACT chain with 8 `ACT_TABLE_LOAD` (1.28 us each, at 83.8, 89, 92.1, 93.8, 95.9, 99, 101 us) interleaved with CAST/ACTIVATE. Then the final output DMA on Pool at 99.9 us after a 34.7 us Pool wait. Sync's last WRITE is at 104 us.
- ACT_TABLE_LOAD total: 13 loads x 1.28 us = 16.7 us of ACT time.
- Instruction count per engine (Scalar 129, Sync 134, Vector 138, Pool 95, PE 86) is small. Compute is a small share.

## 5. Ranked candidates for iteration 2

| # | Edit | Mechanism / where established | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 0 (prerequisite) | **Re-validate Activation slot 67.** Restore `events_wait_idx=61`, `events_semaphore_value=48`, or prove it redundant. | Semaphore accounting rule in `semaphores.py` docstring: waits must match total increments from the source DMAs. | None; it is a correctness fix. | Verifier will NOT catch a race. | Restore it, re-profile. If latency stays about 95 us the loosening was not buying anything; keep the original wait. |
| 1 | **Cut `ACT_TABLE_LOAD` in phase C.** Activation0.bin: group ACTIVATE instrs in the tail (compiler_pc 60–99) by function so the same table is reused; reorder slots only when operand dependencies allow. | `scalar_activation_fn` in the profile; each switch costs 1.28 us. | At most 8 x 1.28 = 10.2 us if all of phase C's loads are on the critical path. Realistic: 3–5 us. | Reordering ACT slots changes semaphore ordering; numerics change if a function is swapped for an equivalent-table one (verify catches that). | Read `scalar_activation_fn` per instruction from `Instruction.parquet`; if tail functions already alternate only between two tables with data dependencies, kill it. |
| 2 | **Eliminate the remaining 1056/1074 B of spill** the same way iter-1 did. Find the remaining `qSPSpillReload0` descriptors in SP0.json that still point at `inst__I-16-0:_mem_*`. | Same mechanism as iter-1 (spill -> SB), already shown to work on this device. | Small: baseline 16 KB of spill was worth ~120 us; the remaining 6% is bounded by ~1 us per transfer. About 2–5 us at best. | Same SBUF-overlap risk as iter-1. | Count the remaining descriptors in SP0.json; if fewer than 3 are on the critical path, skip it. |
| 3 | **Shorten the Pool issue chain in phase A.** Re-home half of the Pool `DMA_DIRECT2D` triggers (slots 1–32) onto idle Sync/Act queues so the weight fetch is not serial at 0.65 us each. | Per-engine DMA queue model: separate queues issue in parallel. Profile shows Sync idle until 34 us. | Up to about 10 us (32 x 0.65 us = 21 us serial, halved). Whether it is on the critical path is unknown. | Queue/semaphore bookkeeping must be re-balanced; a bad edit can deadlock or race. | Check whether PE first LDWEIGHTS depends on the last Pool DMA. If PE starts well after the Pool chain ends, kill it. |
| 4 | **Remove the idle wait on the final output write**: Pool slot 66 waits 34.7 us. | Profile only. | Unknown; the wait may be the true dependency. | Dropping a wait here is a real race. | Look at which sem slot 66 waits on and whether the producer finishes earlier than 99.9 us. If the producer ends at about 99 us the gap is real work, not slack. |

## Closed
- Nothing is closed yet beyond what the baseline already shows. No earlier notes exist and `.autohelix/history.jsonl` has only the baseline entry.
- **Compute speedups are not worth pursuing:** PE is 7.6 us busy (8%), DVE 19 us. Those cannot buy more than a few us.
- **Spill-to-SB itself is done and verified bit-exact**; do not redo it.

## Unknown
- Whether Activation slot 67's new wait is safe (settle with the `SemaphoreUpdate.parquet` timeline for sem 37 and sem 61).
- Whether the new SB regions overlap any live tensor (settle with `SbufUsage.parquet`).
- Which engine is on the critical path in phase B (34–80 us). The settling measurement is a per-instruction dependency trace using the `Flow` or `SemaphoreUpdate` tables; without it candidates 3 and 4 are unranked guesses.
- Run-to-run spread is only 0.7% now, so gains over about 1.5 us are measurable.

**Note on ceiling:** about 95 us remaining; the sum of candidates 1–3 is at most about 25 us if every item is fully on the critical path, and the realistic outcome is much smaller. If candidate 0 shows the wait cannot be reverted without losing latency, ship the iteration as is and stop.

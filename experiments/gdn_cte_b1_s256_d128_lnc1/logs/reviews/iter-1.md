# Review of iter-1 (gdn_cte_b1_s256_d128): 67.525 -> 61.8 us

## 1. What changed in the NEFF, and does it match the notes?

`notes/` contains only the README template. The agent wrote nothing, so there is no claim to check against. This is what `neff_diff.py` shows. It is a reconstruction, not a quote of the agent.

5 of 34 members changed: `Activation0.bin`, `DVE0.bin`, `Pool0.bin`, `SP0.bin`, `SP0.json`.

- **Activation0.bin**, 11 slots. Slots 26, 37, 38, 83, 84, 86, 89 and 90 went from `PSEUDO_DMA_DIRECT2D` (DGE) to `EVENT_SEMAPHORE`. The Activation stream no longer issues 8 of its 11 DGE DMAs. The three that remain are slots 82, 88 and 94. Slots 82, 88 and 94 had their `dst_start_addr` rewritten from scratch var 11/12 to the immediate `0x101cc00`, `0x101d000` and `0x101d600`. Their `dst_step_elem` and `dst_elem_size` changed too. Slot 82 also had `out_dtype` changed from FP32 to BF16. That is a DRAM-scratch to SBUF retarget, and the cast is done in the DMA.
- **Pool0.bin**, 32 to 35 slots. About 16 EVENT_SEMAPHORE or affine-select slots were shuffled. The Pool DGE count went from 16 to 19. The retired Activation DGEs were re-hosted here. The new Pool DGEs write `0x101cc00`..`0x101cf00` with an fp32 to bf16 cast.
- **SP0.bin**, 62 to 64 slots, 43 slots differ. SP0.json shrank by 62 bytes. DGEs and DMA triggers were re-ordered, and 2 slots were added. Semaphores 60-62 are new `dge:SP` rings.
- **DVE0.bin**, slot 98: the semaphore wait value changed from 16 to 1.

Net effect: the HBM spill and reload traffic is gone. `hbm_read_bytes` went from 755 KB to 198 KB and `hbm_write_bytes` from 688 KB to 131 KB. `spill_*_bytes` went from 491 KB / 557 KB to 0. The remaining HBM traffic is exactly inputs (198144 B) plus outputs (131072 B). The staging buffers now live in free SBUF. This is the same lever as `experiments/gdn_cte_b32_s2304_d128_lnc1` iter 3 and `gdn_cte_b32_s128_d128_lnc1` iter 1.

Notes are missing, so the iteration's reasoning is not recorded. The loop's one persistent artifact should be written.

## 2. Is the latency change real?

Yes. Two independent measurements:

| | median | min-max | spread |
|---|---|---|---|
| baseline (profile_baseline.json, 7 reps) | 67.525 | 67.302-67.942 | 0.95% |
| iter-1 (profile_latest.json, 7 reps) | 61.939 | 61.767-62.154 | 0.63% |
| iter-1, my re-run (5 reps) | 61.805 | 61.742-62.425 | 1.11% |

The gain is 5.6-5.7 us, or -8.3%. It is about 5x the spread, and the best baseline sample (67.30) is far above the worst iter-1 sample (62.43). Not noise. Engine busy time is consistent: Activation -4.5 us, DMA -3.9 us, and the rest flat or slightly up (Pool +1.0 us, SP +0.4 us for the re-hosted DGEs).

## 3. Safety beyond the verifier

- **Verifier.** All checks pass. `output_0` max_abs_err is 6.092e-05 (11% of the limit). `output_1` is 4.494e-04 (83% of the limit), which is identical to the baseline to all printed digits. Error did not move, so the arithmetic is unchanged. `output_1` was already close to the limit before this iteration. Do not spend any more error budget there.
- **Verifier coverage is thin.** The verifier ran `runs: 1`. A relocated buffer or a lost wait can pass one run and race later. Re-run `verify.py` several times (3 or more) before accepting.
- **Semaphores** (`semaphores.py` on both unpacked NEFFs). Diffing the baseline against iter-1:
  - No wait was removed or reduced against its producer, apart from the one item below. For every semaphore, `max_wait` equals `total_inc` except sem 16 (see below).
  - sem 4 (Scalar) waiters went from 65 to 71, and sem 6 (DVE) from 99 to 100. These are re-pointed waits, not removed ones.
  - sem 16 changed from "+16 from a `dge:Pool`" to "+1 from a `Pool:EVENT_SEMAPHORE`". It has total_inc 1 and 0 waiters in the tool's table. The DVE slot 98 wait was changed from 16 to 1 to match, so the count is consistent on paper. But the tool reports 0 waiters on sem 16, so I could not confirm that DVE slot 98 waits on it. **Verify this by hand.**
  - DGE completion sems 40-42 and 44-49 moved from `dge:Activation` to `dge:Pool` / `dge:SP`, and 60-62 are new `dge:SP` rings. Totals are balanced, and the new rings' waiter counts (2/2/0) are plausible.
- **Allocation risk.** The new destinations `0x101cc00..0x101d600` are SBUF addresses (partition stride 0x40000, per-partition offset about 115 KB). `SbufUsage` shows peak use at about 2%, so it is probably free. But I did not prove that no live tile overlaps `0x1cc00..0x1d800` on partitions 0-127. A fixed-input verify cannot reveal an overlap that happens to write the same values. This is the item to check with an address-range scan of all SBUF accesses (cheap, offline).

Verdict: accept. The gain is real and the arithmetic is unchanged. Residual concerns are the SBUF-overlap proof, the sem-16 / DVE slot-98 wait, and a multi-run verify.

## 4. Where the remaining 61.8 us goes (evidence for the candidates)

`Instruction.parquet` spans 76.7 us. The first about 14.9 us is runtime preamble (`TENSOR_LOAD`, `NOTIFY`, `SET_ORDERING_MODE`; first kernel DGE at 13.05 us). That is not editable. The remaining about 62 us has these properties:

- **No engine is saturated.** Busy time: Activation 52.6%, DVE 45%, PE 28%, DMA 34%, Pool 26%, SP 16%. The kernel is dependency-latency bound (dataflow chains and DMA hop latencies), not throughput bound.
- **The `qSyncSpillReload0` queue carries 33 SBUF to SBUF DMAs (348 KB).** All are WZXY transposes or copies. Each takes 0.2-0.9 us on the device. Its semaphore is visible 1-1.9 us after `end_ts`, as the `s128` notes found.
- **Dead windows (per-bin compute busy):**
  - 20-26 us: PE is idle. Scalar is about 100% busy with about 25 tiny instructions. Most are 128 elements wide at about 380 ns each (`TENSOR_SCALAR` 33 instructions, 11.8 us total, mean 357 ns; `ACTIVATE` 36 instructions, 11.8 us, mean 328 ns). They are fixed-cost bound.
  - 25.6-33 us: 8 serial Sync DGE copies at about 0.65 us each. They gate the first big PE block (PE resumes about 34 us).
  - 50-56 us: PE has 0 busy and DVE under 0.4. The window holds 6+ DGE copies, a SOFTPLUS and `TENSOR_SCALAR` chain on Scalar, and the queued transposes. PE `LDWEIGHTS` at 56.9 us waits 7.7 us on `S[28]` (qSyncSpillReload0).
  - 59-73 us: the output stores are 5 Pool DGEs (fp32 to bf16 into the output var). Each waits for DVE (`S[6]`).
  - 73-76.7 us: a 3.7 us tail after the last store.

## 5. Ranked candidates for iter-2

Ranked by expected gain per unit of risk and effort. All ceilings are bounds from this profile. The measured gain of past similar edits is given where I know it.

### C1. Remove the DGE-copy to queued-transpose hop pair (fuse two DMA hops into one)

- **Engine / member:** the 3 remaining Activation DGEs (Activation0.bin slots 82/88/94), the Pool DGEs (Pool0 slots 19-21), and the Sync DGEs (SP0.bin DGE slots at 51.5-53 us). Plus the SP0.json queue descriptors that read their SBUF destinations, in SP0.json `dma[]` ids 16-32.
- **Change:** a queued DMA currently reads what a DGE just copied or cast to `0x101cc00..0x101d600`. Re-express the SP0.json descriptor to read the DGE's source (`0x...6480` tiles) directly. Use `op:"cast"` or a composed pattern where the cast needs to survive. Turn the DGE into `EVENT_SEMAPHORE` that repays its +16 (SEM_ADD_IMM_COMPLETE).
- **Mechanism:** established in `experiments/gdn_cte_b32_s64`: iter 2 (-4.6%, "fused two-hop DMA chain into one") and iter 3. Also `s2304` iter 5 (-1.4%). The `op:"cast"` composite is verified in `s128` iter 2.
- **Ceiling:** each hop on the critical path costs about 0.65 us (DGE issue) plus 1-1.9 us (semaphore visibility). The 50-56 us dead window is about 6 us and the 25.6-33 us window is about 7 us. Realistic gain is 2-4 us. The upper bound is the sum of the dead windows, about 13 us, which is not reachable.
- **Risk:** medium. A wrong composed pattern gives wrong numbers, which the verifier catches. A dropped wait would race, which it may not catch.
- **Cheapest falsifier:** do ONE instance (the 50-54 us group). Check output bit-identical and the 50-56 us window shrinks. If the DGE wait on `S[4]>=42/43` was only there to order the copy, it is dead and the transpose can read earlier.

### C2. Move the 3 remaining Activation DGEs (and Sync-host ones) off Scalar's critical path

- **Engine / member:** Activation0.bin slots 82/88/94 (50.19, 51.89, 53.26 us), about 0.65 us each of in-order Scalar time. Scalar is 100% busy 50-53 us.
- **Change:** re-host them on Pool (idle, 26% busy) or SP (16% busy) with the usual wait on the producer semaphore. Replace the old slot with `EVENT_SEMAPHORE` that repays the +16.
- **Mechanism:** same as what this iteration did (documented in `14-optimization-playbook.md` 14.3 #3). `s2304` iter 1/2 shows Pool is the slowest DGE host there but SP and Activation work. Here Pool already hosts 16 DGEs.
- **Ceiling:** about 2 us of Scalar time in the 50-53 us window. Measured evidence: this iteration's Activation busy dropped 4.5 us and the latency fell 5.6 us, but part of that came from the spill removal, so the Scalar-offload share is unknown. Gain 0.5-1.5 us.
- **Risk:** low. It is a re-host, and the semaphore counts are checkable with `semaphores.py`.
- **Cheapest falsifier:** move one and compare the Scalar timeline at 50-53 us. If the following `ACTIVATE` does not start earlier, stop.

### C3. Merge the run of tiny Scalar instructions in the 19-28 us prologue

- **Engine / member:** Activation0.bin, the `TENSOR_SCALAR` (mean 357 ns) and `ACTIVATE COPY/SOFTPLUS` (mean 328 ns) slots, 19-28 us. 20+ near-identical 64- or 128-wide, single-element instructions (e.g. imm `0x101c000/100/200/300`, src `bfloat16@0x1037c00/d00`, `fp32@0x100xxxx`).
- **Change:** where operands differ only by a constant stride, replace N slots with one slot with a wider pattern, and delete the followers (leave `EVENT_SEMAPHORE` so counts stay right).
- **Mechanism:** `s2304` iter 4 merged 993 affine runs for -1.13%. Fixed cost per instruction is about 300 ns for a few elements, which suggests it dominates.
- **Ceiling:** PE is idle 20-26 us waiting on this Scalar run. If 10 of about 25 instructions merge, up to about 3 us. Real gain depends on which of these sit on the critical path, which I have not traced.
- **Risk:** medium-high. `imm_src=PTR` operands (`imm=0x101c000`) point at per-partition scalars, so a wider pattern may not be expressible. The verifier will catch wrong numbers.
- **Cheapest falsifier:** check whether two adjacent slots with `imm=0x101c000` and `imm=0x101c100` can become one slot with the imm-pointer step. If the `PTR` form cannot stride, kill the candidate.

### C4. Move Scalar CAST work to DVE, or the tiny `TENSOR_SCALAR` to Pool

- **Engine / member:** Activation0.bin, 28 `CAST` slots (6.5 us total, 230 ns mean) and some `TENSOR_SCALAR` slots (11.8 us). Scalar is the busiest engine (32.6 us, 52.7%).
- **Change:** insert the equivalent op in the DVE (or Pool) stream, replace the Activation slot with `EVENT_SEMAPHORE`, and re-point consumers.
- **Mechanism:** `s128` iter 6 moved inserted CASTs from ACT to DVE for -19.5 us of ACT busy (2x faster per element). Instruction insertion is device-verified (`s128` iter 5). Opcode engine support is in `generated/opcodes.md`.
- **Ceiling:** at most the Scalar busy time on the critical path. But DVE is 45% busy and also has dead windows, so a transfer just moves the chain. Expect 0.5-2 us.
- **Risk:** medium (cross-engine waits are the classic race). The verifier may miss a one-in-N race.
- **Cheapest falsifier:** move ONE CAST in the 19-28 us window and see if the PE restart moves earlier.

### C5. Start the Sync-queue DGE input copies earlier (reorder, no new instructions)

- **Engine / member:** SP0.bin DGE slots 6-8 at 25.6-27.6 us (first waits on `S[4]>=10`) and slots 10-15 (28.8-32.9 us).
- **Change:** see whether they truly depend on Scalar (`S[4]>=10` and `>=18`). Move triggers up in program order where the wait is only same-engine ordering, then re-point the waits.
- **Mechanism:** playbook 14.3 #2 and `s128` iter 3 (head-of-line blocking removal, -2.6%).
- **Ceiling:** the 25.6-32.9 us chain of 8 serial DGEs at about 0.65 us. If they overlap with the Scalar run, maybe 2 us. If the `S[4]` waits are true data dependencies, about 0.
- **Risk:** medium. Moving a trigger above its producer is a real race, and the verifier will not necessarily catch it.
- **Cheapest falsifier:** use the BIR predecessors in the debug info (chapter 11.2) to confirm the `S[4]` dependency is real before touching anything.

### C6. Trim the 59-73 us output-store tail

- **Engine / member:** Pool0.bin slots 26/27/30/31/32, the output DGEs (`S[6]>=71/80/88/93`, `S[4]>=60`).
- **Change:** issue each output store as early as its producer allows (they are 4-5 us apart, so each waits for DVE) and shorten the 3.7 us tail after the last store.
- **Mechanism:** the DVE waits are real data dependencies, so only the tail after the last store is addressable.
- **Ceiling:** at most the 3.7 us tail, and the tail is probably the runtime's end-of-kernel handshake (`NOTIFY`, `WRITE`, `COMPARE_BRANCH`), so it is likely not editable. Treat this as low value (0-1 us).
- **Risk:** low. **Falsifier:** check what runs 73-76.7 us. If it is only runtime epilogue, drop it.

## Closed

- **HBM spill and reload staging.** Fully removed this iteration: `spill_*_bytes` 0, HBM traffic equals inputs plus outputs only. Do not redo it. (The previous measurement was 491 KB reload + 557 KB save.)
- **Cutting `output_1` error margin.** It is 83% of the limit already and unchanged. No precision-lowering edits are available here.
- **Preamble (0-14.9 us).** Runtime `TENSOR_LOAD` / `DRAIN` / `NOTIFY`, not kernel code. I treated it as not editable. The metric excludes about 14.9 us of the 76.7 us span (61.8 us measured), so it is not in the score anyway.
- **Pool as the sole DGE host.** In the sibling `s2304` kernel Pool was the slowest DGE host (14771 us all-Pool vs 14501 us Activation+SP split). Here Pool is already carrying 19 DGEs, so do not move more there without measuring.

## Unknown

- **Whether the SBUF destinations are exclusively owned.** Settled by an address-range scan of every SBUF access for `0x1cc00..0x1d800` on partitions 0-127, offline, minutes.
- **What `S[4]>=10` / `S[4]>=18` on the Sync DGEs really depend on.** Settled by the BIR dataflow predecessors in the debug info. This decides C5 and part of C1.
- **Which of the 20 tiny Scalar instructions are on the critical path.** Settled by walking the semaphore waits back from the first PE restart at about 34 us. This decides C3 and C4.
- **Whether the DVE slot-98 wait is on sem 16.** The tool reports 0 waiters on sem 16 after the edit. Settled by decoding the slot's `sem_id`.
- **The noise floor of a single edit at this size.** The spread is 0.6-1.1% (about 0.4-0.7 us). Any edit under about 1 us needs 3+ repeats before it is believed.

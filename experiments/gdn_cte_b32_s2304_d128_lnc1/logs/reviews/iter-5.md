# Review of iter-5

I diffed against `git show HEAD:kernel.neff` (the iter-4 NEFF) as well as the baseline, because the baseline diff is cumulative over iterations 1-5.

## 1. What changed vs. what the notes claim

HEAD to working tree: 5 of 34 members changed (Activation0, DVE0, PE0, Pool0, SP0 `.bin`). `SP0.json` and `def.json` are identical.

The slot counts differ by deletions, so `neff_diff` reports about 30k "differing" slots. That is the index shift, not 30k edits.

| Member | Slots, HEAD to new | Real edits |
|---|---|---|
| Activation0 | 30069 to 29932 | Fused save slots: `dst_start_addr_addr_immediate` re-pointed (e.g. `0x...12e80` to `0x...101c900`), `dst_elem_size` 512 to 256, `out_dtype` FP32 to BFLOAT16. 18 `out_dtype` flips appear in the diff output. |
| DVE0 | 29896 to 29523 | Deleted slots, with `EVENT_SEMAPHORE` turning into or out of compute slots at the shifted indices. |
| PE0 | 35084 to 34853 | Dominated `EVENT_SEMAPHORE`s deleted. |
| Pool0 | grew 485376 to 490944 bytes | Each fused reload `PSEUDO_DMA_DIRECT2D` becomes two `EVENT_SEMAPHORE`s. |
| SP0 | shrank 1200384 to 1190848 bytes | Same reload to ES pair on SP (68 groups), plus deletions. |

This matches `.autohelix/notes/iter-5.md`:
- 188 save/reload groups fused. The saves write the reload's destination directly, and 120 of them fold the FP32 to BF16 cast.
- The reload becomes `ES_a` (original wait plus `S[sid] >= 16*(m-1)`) and `ES_b` (`SEM_ADD_IMM_COMPLETE +16`).
- 715 dominated `EVENT_SEMAPHORE`s are deleted (the notes say 707 plus the later dedupe).

I did not decode every slot. I checked that the changed fields are the ones the notes describe, that the member set is right, and that the semaphore totals below hold.

## 2. Is the latency change real?

| NEFF | total_exec_time_us |
|---|---|
| iter-0 baseline | 14779.0 (spread 0.009%) |
| iter-4 (HEAD) | 14231.9 (notes: 14240 in their own capture) |
| iter-5 `profile_latest.json` | median 14036.2, samples 14036.2 / 14031.3 / 14309.9 |
| my re-profile | 14036.7 |

- The change is real: -1.4% vs HEAD (about 195 us) and -5.0% vs baseline. The noise bar is 0.5%, about 70 us.
- The 1.99% "spread" in `profile_latest.json` is one high capture at 14309.9. The other two agree within 5 us. The notes' ten captures (14024 to 14045) all sit below every HEAD capture. This is the known "1 in 6 captures lands high" behaviour, not a failure of the result.
- The loop is not chasing noise this iteration. It will start to once the levers below are used up. See the ceilings in section 4.

## 3. Safety beyond the verifier

**Semaphore accounting is intact.**
- `semaphores.py` on fresh unpacks of the baseline and the new NEFF: `max_wait` and `total_inc` are identical for every semaphore S[3] to S[71].
- Only the source attribution moves, e.g. S[8] `dge:Pool 3136` becomes `dge:Pool 3040 + Pool:EVENT_SEMAPHORE 96`, and S[68] gets an `SP:EVENT_SEMAPHORE` share.
- S[72] is iter-1's scheme and is the same as before.
- There is no over-wait.

**Error is unchanged.** Output_0 is 1.372e-4 (17% of the limit), output_1 is 6.841e-4 (86%). Both are identical to the baseline to every printed digit. Output_1 has only 14% headroom, which is a baseline property. Any numerics-changing edit has little room.

**Residual risk, which the verifier cannot see.**
1. The fusion hazard check (check 1 in the notes) uses profiled `start_ts`/`end_ts` windows, not happens-before through semaphores.
   - A hazard that does not manifest under the observed timing, but is allowed by the semaphore graph, passes.
   - The addresses are static, so the single fixed input does not hide a data-dependent bug. A timing race is the only remaining failure mode.
   - The first attempt of this iteration produced 1.47M NaNs from exactly this class (queued descriptors missing from the hazard set), so the check is fallible.
   - Cheap hardening: run `verify.py` with several runs (it takes a `runs` count) and re-check each fused group against the semaphore graph rather than timestamps.
2. The `SEM_ADD_IMM_COMPLETE` (mode 21) path is probed on the device, but the profiler does not record S[72] and S[73]. Its correctness rests on the wait-based probe and on bit-exact outputs.
3. The Pool-hosted fused saves now cast FP32 to BF16 inside the DGE (`out_dtype` BFLOAT16). The output is bit-exact, so the rounding matches the original cast. That is one more reason to keep the bit-exact bar.

## 4. Candidates for iter-6 (ranked by gain per risk and effort)

Profile facts (my re-profile, deduped `Instruction.parquet`, span 14051 us):
- Busiest engines (sum of `duration_ns`):
  - Tensor: MATMUL 4997 + LDWEIGHTS 1891.
  - Scalar: TENSOR_SCALAR 3310 + ACTIVATE 3221 + CAST 1192 + DMA 904.
  - Vector: TENSOR_TENSOR 2945 + CAST 1420 + COPY 958.
  - Sync: DMA_DIRECT2D 2341. GpSimd: DMA_DIRECT2D 2127.
- Instructions run back-to-back at 14435 all-engine-idle gaps. They total 1216 us (8.7%), the largest is 3.3 us, and the mean is 84 ns. That is handshake and DMA latency, not one big stall.
- The notes' critical-path walk (99.3% accounted) puts SP at about 44% of the path:
  - 2874 us of queued `dma_latency` over 2631 triggers.
  - 1140 us of SP DGE exec.
  - 939 us of SP serial stall.
- No source line exceeds 180 us. The chain is flat and wide.

| # | Candidate | Where | Change | Mechanism and evidence | Ceiling | Risk / verifier | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Finish the save/reload fusion on the 132 unfused groups | `Activation0.bin`/`Pool0.bin`/`SP0.bin` DGE slots. Scripts `.autohelix/notes/iter-5-{grp2,fuse2,qacc,fuse3,prov,prov_half,filt,apply}.py`. | Same edit as iter-5. First refine `crange` in `iter-5-fuse2.py` to enumerate outer pattern dims instead of one bounding interval per partition, which clears about 21 compute-only hazard rejections. Then order the 23 save-overlap groups on one engine. | Device-verified this iteration: 188 groups gave -1.41%, about 1.07 us per group (14240 to 14039 us). | Theoretical 132 x 1.07 = about 140 us. Realistic is 21 to 44 groups, so 25 to 50 us. 82 of the 101 hazard rejections involve queued reads or DGE writes and are real, so the full 140 is not reachable. | Medium. Timestamp-based hazard check, see section 3. The verifier catches an overwritten window only if the window is read in the fixed run, which it is. | Apply the 21 refined groups only. Run `verify.py` (several runs) and `profile.py`. If the median does not drop 20 us or more, the list is below noise. |
| 2 | `transpose_mode` on the stationary-only transposes | `SP0.json` queued descriptors and PE `LDWEIGHTS`. The 689 transposes feeding only a stationary operand. Pair `dma[0]` (SB 227072 to SB 16896, S[24]) with the `LDWEIGHTS` that reads 16896. | Set `transpose_mode = ENABLED` on the LDWEIGHTS, point it at the untransposed tile, and delete the descriptor with every later wait on that semaphore lowered by 16. | 1152 of the 14656 LDWEIGHTS already use `TRANSPOSE_MODE_ENABLED` (iter-3). The blocker is that all 3960 queue-fed LDWEIGHTS read the tile with `src_mem_pattern_step_elem = -1`, reversing the free axis. Transpose mode cannot reverse partitions, so the reversal must be pushed upstream into the producer's matmul stationary columns. | Notes put the queue time at 866 us for the 689, a few hundred us for PE. The 2874 us of queued `dma_latency` is the whole population, of which about 785 ns per transfer is a fixed floor. Realistic: 100 to 300 us if the chain closes. | High. This is the only multi-hundred-us lever. A wrong reversal gives wrong numbers, which the verifier does catch. A reused scratch tile (SB 16896 has 2090 writers and 1782 readers) needs temporal pairing, and an aliasing mistake can pass one input. | One instance (`dma[0]`): does the producer's stationary order reverse cleanly? Budget about 30 minutes. If the chain relocates rather than closes, record that and stop. |
| 3 | Delete the now-dead staging saves | `Activation0.bin`/`Pool0.bin`/`SP0.bin` DGE saves whose window is read by nothing | Replace with the `ES_a`/`ES_b` pair (the 128 to 256 KB transfer goes, the semaphore increment stays) | Same mechanism as iter-5. `iter-5-fuse2.py` and `iter-5-qacc.py` already build the read/write map. Only the "no reader anywhere" query is missing. | Unknown. iter-3 found that the traffic alone bought 0 us. The slot and its completion are the prize. Per-slot value is at most about 0.5 us (the 1.07 us per group includes a removed reload too). | Medium. An unproven dead write corrupts a later reader on a different input. | Count the dead saves first. If fewer than about 100, drop it. |
| 4 | Remove `EVENT_SEMAPHORE`s on SP that only precede a DGE and add serial stall | `SP0.bin`: 6537 `EVENT_SEMAPHORE` (759 us busy) plus 8321 `WRITE` (394 us) | Fold the two-slot ES pairs created by this iteration (SP 68 groups, Pool 120) into one slot using `events_extended_sem_update_value`, if the device accepts it. | The notes say the extended update mode is used nowhere in this NEFF and is unprobed. The notes also explain why two slots were needed: one value field per slot. | About 188 x 130 ns = 25 us. | Low to medium. A bad update mode hangs. | 8-slot probe as in iter-5 (about 2 minutes). Below noise, so only worth it bundled with candidate 1. |

Rank rationale: candidate 2 has the only big ceiling but needs the most budget. Candidate 1 is the cheapest test with working tooling, but its realistic ceiling (25 to 50 us, about 0.2% to 0.4%) is at the noise floor on its own. The loop will mostly chase noise if it takes only candidates 1, 3 and 4. I recommend candidate 2's single-instance test first. If it fails, record the result and stop: that closes the last ms-scale NEFF-level lever, and the rest is under 1% in total.

## Closed (do not redo)

- Deleting dominated `EVENT_SEMAPHORE`s: 707 slots, 14244.3 vs 14240.3 us, which is nothing. `events_extended_wait_*` folding adds 0 deletions. (iter-5 section 3.1)
- Adjacent DGE slot merging: only 262 followers fit, 256 of them carry a wait; about 50 us on a Pool critical-path budget of 840 us. (iter-5 section 3.3; iter-4: 0 of 8383 pairs mergeable.)
- iter-4's consecutive-run instruction merge: exhausted. 13 runs of 2 remain, and 2-level nesting adds 0. (iter-5 section 3.4)
- PE `perf_opt` modes: the field is `NONE` on all 31840 slots, but flipping it alone changes the operand packing and so the results. DVE `performance_mode` is already `2x`. (iter-5 section 3.5)
- Replacing DGE with a compute instruction: only 169 of 8384 transfers are identity-partition SB to SB. (iter-5 section 3.2)
- SP/Act/Pool DGE host ratio: at the optimum, +-2 us; Pool is the worst host. (iter-2, iter-5 section 3.6)
- Moving Activation work to Pool or DVE: -21 us idealised (+26 us realistic) for Pool, -50 us simulated for DVE. ACTIVATE hangs on Pool. (iter-4 review)
- ACTIVATE to a cheaper opcode, `def.json` DMA queues, DMA priority, trigger/wait tightness, `dge_op = DMA_TRANSPOSE`, queued-path SBUF sources (+199 us). (iter-1 to iter-3)
- Free SBUF: the 45,696-byte run was spent in iter-3. Re-derive the free map from the current streams before using any.

## Unknown (would change the ranking)

- Whether the `transpose_mode` reversal chain closes (candidate 2). Measure with the `dma[0]` single-instance test.
- Whether `events_extended_sem_update_value` works on the device (candidate 4). Measure with an 8-slot probe like iter-5's.
- How many staging saves are actually dead (candidate 3). Measure by the "no reader anywhere" query on the existing access map.
- How much of the 1216 us all-idle time is handshake latency that fewer or longer semaphore rings could shave. No measurement exists; do not chase it blind.
- Whether the timestamp-windowed hazard check hides a latent race. Measure with the semaphore-graph happens-before check on the 188 fused groups plus several `verify.py` runs.

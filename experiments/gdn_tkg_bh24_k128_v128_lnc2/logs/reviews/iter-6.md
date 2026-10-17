# Review of iter-6 (kernel.neff vs HEAD 4118dcb; HEAD itself is 38.38 us, baseline 76.05 us)

## 1. What changed
`neff_diff` HEAD -> working tree: 10 members changed, **symmetrically in sg00 and sg01** (same slot counts and same number of differing slots in each: Activation 20, DVE 90 (123->124 slots), PE 28, Pool 4, SP 1). Both cores were edited, so no one-sided-edit problem. The PE0.bin md5 is identical across sg00/sg01, as is SP0.

Content:
- Activation0 (slots 15-54): the head of the ACT stream is re-ordered (CAST / ACTIVATE / ACTIVATION_TABLE_LOAD swapped around slots 15-22), and the semaphore-value operands of ~12 later casts are bumped by +1 (e.g. 26->27, 39->40, 47->48 ...).
- DVE0: one extra instruction (a CAST; sem-6 DVE:CAST 12->13, total sem-6 increments 79->80). That shifts the sem-6 wait values on later DVE-produced tiles.
- PE0 (28 slots) and Pool0/SP0: wait operands retargeted. Waiters on sem 4 (ACT) fell 72->68 and waiters on sem 6 (DVE) rose 60->64, i.e. 4 waits moved from ACT to DVE completion.
- Net: one cast moved ACT -> DVE and the ACT head was re-sequenced, with every downstream wait renumbered.

**The notes are empty** (`notes/README.md` is only the template), so I cannot compare against the agent's claim. The commit message is the only claim available, and it is not yet written for this iteration. The diff is consistent with "move a cast to DVE + re-order ACT head + renumber waits". Nothing suggests a one-core edit. The agent must write a note for this iteration.

## 2. Is the latency change real?
Yes. HEAD 38.38 us (the 38.396 of its commit message) -> median **36.714 us** in `profile_latest.json` (samples 36.64-37.19, spread 1.5%). My own single-repeat run read 36.82. The gain is -1.7 us (-4.4%), roughly 10x the spread, so not noise. Against baseline, 76.05 us -> 36.7 us (2.07x). Caveat: the earlier iteration's -0.5% (38.59 -> 38.38) was inside ~1.5% spread and was not proven. Only gains above about 0.6 us (about 1.5%) should be trusted from a 5-repeat median.

## 3. Safety
- `semaphores.py` on both subgraphs: identical output. Every semaphore has max_wait <= total_inc (sem4 45/45, sem5 38/38, sem6 80/80, DMA sems 16/16, 64/64). No "wait exceeds increments" flag. Each of sems 4/5/6 has a single producing engine (ACT / PE / DVE), so counts are in program order and deterministic. Renumbering is therefore sound *as long as* each new wait value equals the producer's running count at the intended tile. A wrong-by-one that is too large would deadlock (the verifier would see it). A wrong-by-one that is too **small** would race and could still pass one fixed input. I did not re-derive the 4 retargeted sem-4 -> sem-6 waits tile by tile. The agent should state it did.
- sem 8 has 16 DMA increments and 0 waiters (dt_bias queue). That is pre-existing and harmless.
- Error: output_0 max_abs_err 2.322e-2 (86% of 0.027), output_1 1.633e-3. **Both are identical to the baseline verify**, so the arithmetic is bit-identical. The 86% is inherent to the kernel, not drift from these edits. Within tol 100%.
- The cores see identical programs, so they cannot disagree on this edit.

## 4. Candidates for next iteration (ranked)

Evidence from `prof/Instruction.parquet` (sg00; sg01 is symmetric):
- The body runs 12.5 us -> 48.2 us (last output DMA). About 3 us after that (49.8-51.2) is a fixed runtime drain/notify tail, so treat it as irreducible.
- Inputs finish landing at ~15.0 us (`DmaPacketAggregated`: dt_bias..A_log done 13.5-15.0). State-in Pool DGE DMAs run 13.3-17.0.
- Busy time: DVE 32.3 us (88%), ACT 29.9 (81%), PE 13.5, Pool 12.0, DMA 20.4.
- **Head, 12.6-21.9 us:** the ACT stream is ACTIVATE (0.21 us) / ACT_TABLE_LOAD (1.28 us) alternating **5 times** (pc 1,3,5,8,10 at 12.7, 15.0, 16.6, 18.6, 20.1). That is 6.4 us of table loads in a 9 us window. DVE meanwhile shows 1.8-2.0 us idle gaps (14.6, 16.6, 18.6) waiting on these ACT results.
- **Tail, 38-48 us:** DVE TENSOR_TENSOR / SCALAR_TENSOR_TENSOR chain is back to back at ~0.28-0.34 us each with ACT dependent on it. ACT shows 1.2-1.9 us gaps (pc 63, 56, 46). The 12-head loop is DVE-bound.

| # | Candidate | Engine / member / slots | Change | Mechanism | Gain / ceiling | Risk | Cheapest falsifier |
|---|-----------|-------------------------|--------|-----------|----------------|-------|--------------------|
| 1 | Cut ACT table loads in the head (5 -> 2-3) | sg00+sg01 `Activation0.bin`, slots ~1-30 (ACT_TABLE_LOAD at pc 1,3,5,8,10 and the tiny 212 ns ACTIVATEs between them) | Group the head ACTIVATEs by `activation_func`/`act_tbl_sel` so each table is loaded once. Deps decide what is legal. Moving the table-independent ones (copy/identity-like) onto DVE/Pool is the alternative. Renumber `events_wait_idx`/semaphore values after. | Same technique already measured in this repo: df60817 cut loads 22->6 for -27%, 7bc704d hoisted the T3 load for -5.8%. ACT_TABLE_LOAD = 1.28 us each is directly in the profile. | Each avoided load is up to ~1.28 us on the head critical path. Realistic 1.5-3 us. **Ceiling: ~6.4 us of table loads in the head, minus the minimum number of distinct tables the head needs.** Must check the function dependency chain (softplus -> exp -> sigmoid?), which may force the alternation. | Wrong act_tbl_sel gives wrong numerics. The verifier catches that (bit-identical test). Wait renumbering can race silently. | Decode the 5 head ACTIVATEs (`activation_func`, `act_tbl_sel`) and the ACT_TABLE_LOAD operands. If consecutive ACTIVATEs already need different tables, the lever is dead in minutes. |
| 2 | Start the head gate math earlier / overlap with state DMA | sg00+sg01 `Activation0`/`DVE0` head, plus the first Pool DMA_DIRECT2D (13.3 us) | The gate-path inputs (a, b, A_log, dt_bias) land at 13.5-15.0 but the first DVE compute lands at 14.6 and the next at 16.6/18.6. Order the 4 small-input waits so ACT begins on the earliest-arrived tile (dt_bias at 13.5) rather than the slowest. Reorder the DMA descriptor order (first-needed first). | The previous iteration did the same trick (z first descriptor, 119bfcf, -9.5% jointly). DMA completion times are in `DmaPacketAggregated`. | ~0.5-1.5 us. Ceiling: arrival of the last gate input at ~14.97 us versus ACT start at 14.8, so only about 0-1 us remains there. The spacing loss is the table loads (#1). | Low. Wait values only. | Compare per-variable DMA end times with the first wait on each (`evt_wait_time_ns`). If the waits start after the data has arrived, no gain. |
| 3 | Shift more of the 12-head body's ACT-bound ops to Pool/idle engines | sg00+sg01 `Activation0`, `Pool0` (Pool busy only 12 us, 32%) | Move simple elementwise ops currently on ACT/DVE (e.g. casts, `* beta_K`) to GpSimd. Pool0 is under-used. | The 7bc704d / 119bfcf iterations moved casts between ACT and DVE with success. Pool is the one engine not tried (grep notes: no Pool compute). | Per-head DVE and ACT chain ~0.3 us per op. Offloading one op per head could save up to ~12 x 0.3 = 3.6 us only if that op is on the critical path (DVE 88% busy). **Ceiling unknown**: GpSimd throughput per elem is typically lower than DVE and may not support the op/dtype. Needs measurement. | Pool instruction set subset. Different rounding of casts if the dtype handling differs. The verifier would catch numerics but not a missing wait. | Re-encode one Pool-eligible op in a scratch NEFF and time it standalone with `profile.py`. If duration is >2x DVE, stop. |
| 4 | Remove the PE-wait-on-DVE slack in the tail | sg00+sg01 `PE0.bin` waits in slots ~56-75 | Same trick as 4118dcb: Tensor waits at 38.19, 39.5, 40.5, 41.3, 42.1, 43.5, 44.7 show `evt_wait_time_ns` ~0.75-1.35 us. Check each wait matches the producer that actually writes the tile. | 4118dcb found 4 of 12 waits pointing at a later producer. | Only gains if PE is on the critical path. Today it is not (PE busy 37%). **Ceiling ~0.3 us**, because the tail is DVE-bound. | Wait value renumber race. | Lower one PE wait value by one in a scratch NEFF. If time is unchanged, no gain. |
| 5 | Shrink the DVE tail chain (pc 92-123) | `DVE0.bin` slots ~100-124 | Merge the adjacent dependent TENSOR_TENSORs (~0.28 us each, 8+ in the final head) into SCALAR_TENSOR_TENSOR (fused mult-add) where the algebra permits. | 400b689 and 7bc704d used wide/fused ops. Each fusion removes one ~0.28 us. | 0.3-1 us. Ceiling: the DVE chain tail at 43-48 us is ~5 us for 8-12 ops. | Fusion changes rounding order, and the error is already 86% of the limit. A fused op keeps fp32 intermediates, which would be at least as accurate, but only the verifier can say. | Check that the verifier error does not move on a scratch build. |

(Only candidate 1 and 2 are cheap enough to test within minutes. 3 to 5 are speculative and should be attempted in that order only if 1 and 2 are dead.)

### Closed (do not redo)
- 36 scale-pointer ACTIVATEs on ACT: moved to DVE in 400b689 (55.6 -> 45.0 us).
- ACT PSUM->SBUF `o_sb` COPYs: deleted in 119bfcf (44.8 -> 40.5).
- State-out as a single DMA: split into 4 per-head-group DGE transfers in 7bc704d.
- ACT table loads in the *body*: 22 -> 6 per core in df60817, T3 load hoisted in 7bc704d. Only the head remains (candidate 1).
- 12 PE outer-product s4 waits: already tightened in 4118dcb. The remaining gain there is <=0.3 us.
- The fixed runtime preamble/postamble (before 12.5 us and after ~49 us) is outside the kernel stream and not editable.

### Unknown (would change the ranking)
- Which table each of the 5 head ACTIVATEs needs. Settle: decode `act_tbl_sel` and `activation_func` in slots 1-30 of `Activation0.bin`. Decides candidate 1.
- Whether GpSimd can run the casts / scalar-tensor ops at an acceptable rate. Settle: time one Pool instruction in a scratch NEFF. Decides candidate 3.
- Whether the sg00 vs sg01 profile are balanced (I profiled only pcore 0). Settle: repeat the gap analysis with `pcore_idx==1`. If sg01 is slower the exec time is set by it.
- Whether the iteration's 4 retargeted sem-4 -> sem-6 waits are tile-exact. Settle: have the agent print the producer slot index of each retargeted wait.

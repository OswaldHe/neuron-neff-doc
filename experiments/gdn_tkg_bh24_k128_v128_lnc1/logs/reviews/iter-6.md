# Review of iteration 6 (kernel.neff 66.41 -> 62.55 us)

Method: `neff_diff.py` against the committed HEAD NEFF (`git show HEAD:kernel.neff`), because the
diff against `baseline/` is the cumulative history. I also ran `semaphores.py` on both unpacked
NEFFs, read `profile_latest.json` / `verify_latest.json`, and re-profiled with
`tools/profile.py --repeats 1 --parquet` for the per-instruction table.
`notes/README.md` is still the empty template. The agent wrote no notes and there is no commit
message for this iteration, so "consistent with the notes" cannot be checked. The only claim
available is the HEAD commit message, which describes the previous iteration.

## 1. What changed

Members changed: `Activation0.bin`, `DVE0.bin`, `PE0.bin`, `Pool0.bin`. 42 of 46 members are identical.
The SP queue and the DMA descriptors are untouched.

| Member | Slots (HEAD -> now) | Slots differing | What the diff shows |
|---|---|---|---|
| Activation0 | 170 -> 170 | 28 | About 19 `ACTIVATE` / `CAST` slots became `EVENT_SEMAPHORE` placeholders. Slot count is unchanged. |
| DVE0 | 185 -> 213 | 153 | Net +28 slots. Four ops flipped `op` and dtype (`SUBTRACT`/`ADD`/`MULT`, FP32 <-> BF16 in/out), 16 `events_semaphore_value` edits, and many reorders between `TENSOR_TENSOR_ARITH_OP`, `SCALAR_TENSOR_TENSOR_ARITH` and `EVENT_SEMAPHORE`. |
| PE0 | 163 -> 182 | 146 | +19 slots, mostly new `EVENT_SEMAPHORE` and shuffled `LDWEIGHTS`/`MATMUL` order. |
| Pool0 | 15 -> 15 | 7 | Wait-index and value edits only. |

What the profile shows is a work move from ACT to DVE:

- ACT `CAST` incrementers of sem 4 dropped from 32 to 11. ACT `EVENT_SEMAPHORE` incrementers rose from 28 to 53.
- Each per-head `S -= outer` now appears twice on DVE, back to back. Both read the same fp32 S and the same PSUM operand. One writes fp32 (state) and one writes bf16, for example 32.58/32.77 and 36.15/36.35. This replaces the per-head ACT CAST.
- DVE TT incrementers fell from 51 to 44, STT rose from 37 to 44, and PE sem-5 waiters rose from 74 to 95.
- The ACT queue keeps its slot count and sem-4 total (120), so downstream wait values did not need renumbering. That is the correct way to remove an op.

This is a cross-engine move of the S->bf16 cast. It is not a pure fence deletion. Without notes I
cannot confirm the agent's intent. I can only say the NEFF is consistent with "casts moved ACT->DVE
and ACT slots turned into count-preserving EVENT_SEMAPHOREs".

## 2. Is the latency change real?

Yes.

- `profile_latest`: median 62.546 us, samples 62.526-62.584, spread 0.092%.
- My re-profile gave 62.620.
- The previous iteration's 66.41 us was measured with the same tool. The ~3.9 us gain is about 40 times the spread.
- Verifier: output_0 max_abs_err 2.322e-2 and output_1 1.633e-3, identical to `verify_baseline.json`. The outputs are bit-identical, as claimed.

Caveat: the parquet span is 75 us. It includes about 10 us of runtime preamble (first kernel op at
~10 us) and about 2-3.4 us of epilogue `EVENT_SEMAPHORE` waits on every engine (70.4-73.8 us). Both are fixed runtime overhead.

## 3. Safety beyond the verifier

- `semaphores.py` shows the max wait is at most the total increments on every semaphore.
  - Sem 4: 120/120. Sem 5: 74/74. Sem 6: 103/103. DMA sems 8-13, 24-27: unchanged.
  - Totals are identical to HEAD, so no count was lost.
- Waiter counts rose (sem 4: 149 -> 153, sem 5: 74 -> 95, sem 6: 122 -> 144). That means more consumers wait on the same producers, so synchronisation was tightened or rerouted, not loosened.
- I did not prove that every individual WAR/RAW edge is preserved. Specific worry: PSUM banks `0x2000000..0x2003800` are reused round-robin by the PE across heads. The DVE now reads each bank twice (the doubled subtract), so a PE overwrite of head h+4 must wait for the second read. In the profile the matmul for the next use of a bank always starts after both DVE reads, but one input set and one timing proves little.
  - A missing PE-after-DVE wait would give a stale-read race that can pass once.
  - Cheap check: for each PE `MATMUL` with `dst=0x200X000`, confirm its sem-6 wait value is at least the completion count of the second DVE read of that bank's previous use.
- Error is unchanged from baseline (83% of the limit, as before), so the margin did not move.

## 4. Ranked candidates for iteration 7

Profile facts used:
- DVE is the busiest engine: 46.6 us busy, 74.6%. ACT is 39.6 us (63%), PE 19.2 us, Pool 15.5 us (mostly DMA issue).
- DVE per-head fixed cost is about 147 ns. I fitted this from the 3080 ns op over 2816 elements versus the 280 ns op over 128 elements, which gives about 1.04 ns/element marginal.
- The per-head body runs at about 2 us/head and is DVE-paced.
- The head (10-26 us) is paced by one dependent chain:
  1. State DMA lands at 13.4 us.
  2. ACT `FUNC_36` over 3072 elements on one channel takes 2.76 us (13.42-16.19).
  3. One unsplit DVE multiply (`TENSOR_TENSOR`, [128,22,1], 3.08 us) runs 16.22-19.30 and waits on all of the `FUNC_36` result.
  4. A DVE STT op follows at 19.73.
  5. The second SOFTPLUS runs at 20.84.
  6. The first S chunk starts at 21.32.
  7. The ACT table loads (2, 0, 3) are hidden behind the DVE 3.08 us op.

| # | Candidate | Engine / member / slots | Change | Mechanism and where established | Expected gain / ceiling | Risk, and would the verifier catch it? | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Batch the remaining per-head `v *= beta` TT multiplies into one or two strided ops | DVE0.bin. The 12 `TENSOR_TENSOR` MULTIPLY ops, `channels=1`, 272-281 ns each, with src0 `0x34e00 + k*0x200`... (for example 26.50 `0x34e00*0x4120`, 27.16, 29.09, 30.12, 33.91, 34.74, 37.39, 38.62, 42.83, 44.74, 45.32, and the tail ones `0x37200*0x4168`...) and src1 `0x4120 + 4h` (the batched GELU(24) output, ready at 26.47) | Replace the 12 ops with one op: `num_elem=[128, 12]`, src0 step 0x200 (or 0x400 if the heads alternate), src1 inner step 0, outer step 1 word, in-place. Turn the other 11 slots into count-preserving `EVENT_SEMAPHORE`s. | Broadcast and strided patterns of this form are already used in the same stream (the 540 ns `[128,3,1]` op with src1 `[0,1,1]`). Fewer larger instructions is playbook 14.3 item 4. Fixed cost about 147 ns per DVE op was measured from the profile above. | 12 x 0.275 = 3.3 us -> about 1.7 us, so up to about 1.6 us of DVE busy. The real gain is lower if the tail ones wait on the per-head ACT FUNC_29 chain (their src1 is `0x4168...`, the per-head rsqrt, not the batched GELU). Batching only the heads whose scalar is the batched GELU gives about 1 us. | Low-medium. Data-parallel, so a wrong stride or shifted wait gives wrong numbers and the verifier catches it. A consumer waiting on the old per-head semaphore count can race if rebalanced wrongly, and a race may pass once. Run `semaphores.py` and compare per-head wait values. | Read the src1 addresses of the 12 ops. If they are not consecutive words from the batched GELU/silu output (`0x4120+4h`), drop the idea. Then do the 2-head merge of the first two (26.50 and 27.16) and measure the op duration. Expected about 400 ns versus 550 ns. |
| 2 | Pipeline the head's critical chain: split ACT `FUNC_36` (3072 elements) and the dependent DVE 3.08 us multiply into 3-4 element-range chunks | Activation0.bin, the `ACTIVATE FUNC_36` slot (~slot 4, `src 0x834e00`, `num_elem 3072` -> 4 x 768, dst `0x20000` + offset). DVE0.bin, the `[128,22,1]` `TENSOR_TENSOR` MULTIPLY at 16.22 and its sem waits. | Four ACT chunks, each incrementing sem 4. Four DVE multiply chunks, each waiting on the matching count. Dst and src offsets are in elements. | Same mechanism as the previous iterations' chunking (`e8d4527`, `504faee`, per commit messages). I did not read notes on it. Reordering and waits re-pointing are verified in doc 10.4 / 14.2. | Critical-path overlap is at most min(2.76, 3.08) x 3/4, so up to about 2 us, minus about 0.2 us extra ACT fixed cost per chunk. Real gain about 1-1.5 us. The ceiling is bounded by the chain above (ACT 2.76 + DVE 3.08 serial from 13.4 us). | Low risk, and the verifier catches indexing errors. The count renumbering of everything after the split is where it can go wrong, and ACT sem 4 appears in many waits. | Split only the ACT op in two with the DVE op unchanged and read `FUNC_36` total duration. If the two halves sum to more than 3.4 us, kill it. Also check that the `[128,22,1]` pattern is contiguous in the chunk axis. |
| 3 | Rebalance DVE vs ACT vs Pool on the per-head bf16 write (the ACT->DVE cast move this iteration made) | The doubled `S -= outer` pairs on DVE (24 heads x about 0.28 us = 6.7 us of DVE). ACT `EVENT_SEMAPHORE` placeholders that were `CAST`s. Pool0.bin is mostly empty after 17 us. | For some heads (try k = 4, 8), put the fp32->bf16 cast back on ACT, or on Pool (`CAST` is available on Pool per `generated/opcodes.md`), instead of the second DVE subtract. | DVE is the top engine (46.6 us versus ACT 39.6 us), so shifting work toward ACT and Pool until they are level is playbook 14.3 item 3. Pool per-op cost for a 128x128 cast is not measured. | Between 0 and about 1 us. If the DVE-ACT gap (about 7 us busy) closes by moving about 3 us, the per-head pace could drop a few hundred ns x 24 heads, but the tail is also paced by the PE->ACT->DVE latency per head, so the real ceiling is lower. Unknown until measured. | Medium. Each move changes producer/consumer sem identity for the bf16 consumer (PE matmul operand `LDWEIGHTS`/`MATMUL` src). The verifier catches wrong values but not a missing wait that happens to be satisfied once. | Move one head's cast to Pool and read its duration in the profile. If it exceeds about 0.5 us, kill the Pool route. For the ACT route, move one head and compare the median. A change smaller than 0.1 us means the engine is not the limiter. |
| 4 | Remove the useless head ACT table load (`table_sel=1` / `HARDSWISH`, 24 elements) | Activation0.bin slots 1-2 (the `ACTIVATION_TABLE_LOAD sel=1` and its 222 ns `ACTIVATE` dst `0x4000`) | Not a direct gain. The load is hidden behind the DMA wait (13.4 us), so deleting it alone gains nothing. The point is to use the free window (10.3-13.4 us) for useful work: put SOFTPLUS #1 (which waits only on sem 8, DMA) in front, and re-sequence tables 4 -> 0 -> 3 -> 2 so table 2 is loaded once. | Table order is in the instruction stream (`act_tbl_sel` field, diff slot 1). Table membership is not documented anywhere I checked (`neff_doc` open-questions 16). | Currently tables load 1, 4, 2, 0, 3, 2 = 6 x 1.28 us. Only the 2-0-3-2 loads after 16.2 us sit between FUNC_36 and the DVE chunks, and the DVE 3.08 us op already hides them. So the gain is at most 0 us until candidate 2 shortens the DVE op. After candidate 2 it could be up to about 1.3 us per removed load. | Medium-high. A function whose table is not resident gives silently wrong values, and the verifier catches that. | Check whether `FUNC_30`/`FUNC_29` and SOFTPLUS/GELU exist in the same table (a quick ACT-only test NEFF, outside the loop). If not, the idea is dead. |
| 5 | Fold count-preserving placeholder `EVENT_SEMAPHORE`s into adjacent real instructions | Activation0 (75 ES, about 40 ns each = 3 us), DVE0 (54, 24 ns), PE0 (41, 53 ns) | Move the increment into `events_update_*` of the neighbouring op and delete the ES slot. | Playbook 14.2, "remove an instruction ... keeping semaphore counts right", verified (r05, r06). | Under about 1.5 us total even if all sat on the critical path. In the tail they are on PE/ACT, which wait on DVE anyway. | Medium. This is the same kind of edit that creates the race risk in section 3, and it adds to the sync complexity. | Delete one ES on DVE that sits between two ops of the same head and read the run-to-run delta. If it is under 0.05 us, stop. |

Ranking rationale: 1 and 2 are pure data-parallel edits on the engine and chain that bound the
kernel. Their gain is measured from instruction durations in this profile, and the verifier
catches wrong data. 3 is an engine-balance edit with an unknown ceiling, so it needs one cheap
measurement first. 4 and 5 rely on mechanisms I could not confirm or have little headroom.

## Closed (measured, do not redo)

- Group-merge of the 24 pre-outer S_bf casts, 8-way split of `k_f*=k_scale`, fusing 16 of 24 `ACTIVATE f128` multiplies into `CAST`: 79.6 -> 74.5 us (`504faee`).
- 4-way split of `S*=exp_g` and 2-way split of the state DMA input, hoisting 3 wide casts into the 3.1 us ACT idle window: 74.4 -> 72.7 us (`e8d4527`).
- Deleting three artificial fences, moving the body table load, folding `norm_weight` into `silu(z)`: 72.60 -> 69.03 us (`66f6259`).
- Folding `w = beta*delta` into the DVE delta op for 12 heads, merging 3 rsqrt ACTIVATE pairs: 69.03 -> 66.41 us (`b17922f`).
- This iteration (casts to DVE): 66.41 -> 62.55 us.
- Removing a wait on an engine's own completion semaphore gave wrong output on DVE and an OOB error on Pool (doc 14.2, o01). Do not use it.
- The 2-3.4 us epilogue `EVENT_SEMAPHORE` on all engines (70.4-73.8 us) and the ~10 us preamble (`TENSOR_LOAD` + `EVENT_SEMAPHORE` on every engine) are fixed runtime overhead.
- The profile.py total (about 62.5 us) already excludes some of them. Parquet span is 75 us.

## Unknown (would change the ranking)

- **Pool cast speed.** Candidate 3 depends on it. Measure one fp32->bf16 128x128 `CAST` on Pool.
- **Table membership of FUNC_29/30/36, SOFTPLUS, GELU, HARDSWISH.** It decides whether candidate 4 can exist. Settle it with a tiny ACT-only test.
- **The producer of the 12 `v` rows (`0x34e00 + 0x200*h`).** If it is DMA and landed before 26 us, candidate 1 can also move out of the head into the idle DVE gap at 11.9-16.2 us (4.4 us idle), but only if beta is available. Read DVE0 slots before 26 us.
- **PE-bank WAR edges after the doubled subtract.** See section 3. This is a correctness unknown, not a speed one.
- **Prior iteration's profile numbers.** `profile_latest.json` was overwritten, so I could not compare per-engine busy before and after. DVE busy of 46.6 us is only the current value.

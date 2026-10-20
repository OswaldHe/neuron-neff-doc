# Review — iter-0 (gdn_tkg bh24 k128 v128, LNC=1)

## 1. What changed
**Nothing.** `neff_diff.py baseline/kernel.neff kernel.neff` reports an identical header, 46/46 identical members, "payloads are identical". `git status` is clean. `notes/` holds only the README stub. There is no `profile_latest.json` or `verify_latest.json`. The iteration produced no edit, so there is nothing for the notes to contradict.

## 2. Is the latency change real?
There is no change to measure. I re-profiled the unedited NEFF: 153.83 us single capture (baseline median 153.46, samples 153.10–154.32, spread 0.79%). So the noise floor is about ±0.6 us, and anything under ~0.7 us needs repeats before it is believed. The wins below are 10–60 us, far above that.

## 3. Safety
Nothing to check on this iteration. Baseline `max_abs_err` is 2.322e-02 against a 0.028 limit (83%). Any edit that changes arithmetic, even rounding order, will fail, so only schedule, placement and engine moves are viable. Run `semaphores.py` on the unpacked copy after every edit. That is necessary but not sufficient: a prior LNC2 run hung on Pool even though the semaphore graph was balanced.

## Where the time goes (this profile)
The body runs from ~10.2 us (kernel start; the runtime prologue before it has `compiler_pc` NaN) to ~161 us. Activation (ACT) is the bottleneck at 143.5 us busy (93.5%). DVE is 34 us, PE 21 us, DMA 20 us, Pool 9 us.

ACT busy time, from `prof/Instruction.parquet`:

| ACT item | count | total |
|---|---|---|
| `ACT_TABLE_LOAD` (1283 ns each, fixed) | 46 | **59.0 us (38% of the kernel)** |
| `ACTIVATE` copy-with-scale-pointer | 72 | 35.1 us |
| `ACTIVATE` other (silu, rsqrt, square, exp, ...) | ~80 | ~21 us |
| `CAST` | 72 | 20.9 us |
| `ACTIVATION_READ_ACCUMULATOR` | 24 | 7.0 us |
| `COPY` | 24 | 6.0 us |

This is the same nkilib kernel as `neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2`. That experiment went from 76.05 to 36.18 us by editing the NEFF alone. Its baseline had 22 table loads per core for 12 heads per core; this one has 46 for 24 heads on one core. The source is at `/home/ubuntu/workspace/gdn-bench/vendor/gdn/gdn_tkg.py`, and `nki_source_location` in the parquet points into it. The experiment's `logs/notes/iter-1.md` has the mechanics. Those claims were established on the LNC2 build, so re-check each on this one.

## 4. Ranked candidates for iter-1
The first four have the same structure as the LNC2 edits. The expected gains are extrapolations from a different NEFF, not measurements on this one.

| # | Edit | Ceiling | Risk |
|---|---|---|---|
| 1 | Reorder ACT and merge silu, to cut table loads 46 → ~6–8 | ≤ 51 us ACT busy | medium |
| 2 | Move the 72 scale-pointer `ACTIVATE`s to DVE | ≤ 35 us ACT busy | medium |
| 3 | Merge the `CAST S->S_bf` ops into wide casts | ≤ ~10 us ACT | medium |
| 4 | Delete the 24 `COPY out_p->o_sb` (PSUM→SBUF) | ≤ 6 us ACT | low |
| 5 | Fix PE and DVE stragglers by wait renumbering and slot reordering | ~1–2 us | low |

**1. Cut ACT table loads (best gain per risk).**
- **Where:** `Activation0.bin`, 46 `ACTIVATION_TABLE_LOAD` slots.
- **Change:**
  - The 24 silu `ACTIVATE`s (func 36) depend only on the `z` input DMA. Give each a private dst buffer, which removes the WAR waits.
  - Collapse the 24 into one `ACTIVATE` with `num_elem=[24*128,1,1]`.
  - Put it at the front of the ACT stream under one table load, then delete the dead loads.
  - Table-set selection is `act_tbl_sel` at byte 35 of the load slot.
  - Free SBUF is around 0x9078–0x1c000 and 0x1d800–0x35500 in the LNC2 build; re-derive it for this NEFF.
- **Mechanism:**
  - `ACTIVATION_TABLE_LOAD` touches no semaphores (`events_wait_mode=NONE`, `events_update_mode=NONE`).
  - Only the invariant "the loaded set contains the func at each ACTIVATE" matters. `ACTIVATE` has no table field.
  - Established in LNC2 notes iter-1.
  - No vendor table set holds both rsqrt (f29) and silu (f36), so the rsqrt/silu alternation cannot be removed, only grouped.
- **Gain and ceiling:**
  - The LNC2 floor was 6 loads (T4, T1, T2, T0, T3, T2), set by a forced dependency chain.
  - The ceiling here is (46−6)×1.283 ≈ 51 us of ACT.
  - The 24 rsqrts are each the serial tail of a head's chain, so expect the real floor to be above 6. The LNC2 experiment cut 22 → 6 and got −27% in total time.
  - Converting ACT time into total time is not 1:1 once ACT stops being the only bottleneck. The ACT trace here already shows idle gaps, e.g. ACT waiting 5.3 us at ~16.8 us and 42 us at ~22.5 us on a semaphore.
- **Risk:**
  - Cycles, which hang the device; build a cycle checker before editing.
  - Scalar operands passed by pointer in `scale_value_imm_ptr` and `bias_imm_ptr` are missed by dependency scans that only read `src/dst_mem_pattern_start_addr`.
  - The verifier would catch a hang or bad numbers. It would not catch a WAR race that happens to pass on the fixed input.
- **Cheapest falsifier:** do the step-1 reorder only, with no merge. Move the f36 ACTIVATEs next to each other, delete the dead loads, and run `semaphores.py` plus the cycle check. If the ACT stream shows no `ACT_TABLE_LOAD` count drop, or total time does not move by at least ~1 us, stop.

**2. Move scale-pointer ACTIVATEs to DVE (best second step).**
- **Where:** 72 `ACTIVATE` with func 128 (`COPY(scale/bias enabled)`), 490 ns each, 35.1 us. DVE has 34 us busy of ~150, so there is headroom.
- **Change:**
  - Re-encode as DVE `TENSOR_TENSOR` with a zero-stride broadcast `src1`, i.e. a real re-encode, not a verbatim copy.
  - The LNC2 run merged groups into one 1536-element instruction.
  - The semaphore index changes with the engine. Key producers by `id(instruction)` and remap waits.
- **Mechanism:** LNC2 iter-2 reported −19.8% from this plus a cast merge. The 3×24 grouping per head here needs checking against the source.
- **Gain and ceiling:** ≤ 35 us of ACT, less the DVE cost; DVE would become the limiter.
- **Risk:**
  - Mid-chain instructions can create cycles; the LNC2 attempt to move COPY to DVE failed that way.
  - Numerics: the broadcast multiply must match the ACT result bit-for-bit. The 83% error budget is tight, so run `verify.py --runs 5` and compare max_abs_err to the baseline's exactly.
- **Cheapest falsifier:** convert one instance, check its output bytes against the baseline for equality, and check that `cycle.py` is acyclic.

**3. Merge the `CAST S->S_bf` ops.**
- **Where:** 72 `CAST` ops at ~290 ns each (20.9 us). Instruction cost is mostly fixed overhead (128 elements ≈ 290 ns).
- **Change:** give the S_bf tiles a private contiguous strided region and re-point the matching PE matmul operands. A constant stride is enough for a merge, e.g. `num_elem=[128,12]` with a stride in elements.
- **Gain and ceiling:** ~2.4 us per 12 casts merged in the LNC2 run. Here that is up to ~5–10 us. Casts may also move to DVE.
- **Risk:**
  - The merged tile needs ordering before the DVE `S += outer` writes.
  - After large ACT reductions, removed ACT work often turns into idle time instead of saved time. Re-measure the idle gaps first.
- **Cheapest falsifier:** merge two heads' casts and check that ACT busy time drops by about 0.3 us.

**4. Delete the 24 `COPY out_p->o_sb`.**
- **Where:** 24 ACT `COPY`s, 6.0 us.
- **Change:** re-point both readers (the ACT square-reduce and the DVE `SCALAR_TENSOR_TENSOR`) to read PSUM directly, then drop the COPY.
- **Mechanism:** worked in LNC2 iter-3 when all 8 PSUM WAR constraints were already implied by the existing happens-before graph.
- **Gain and ceiling:** ≤ 6 us ACT.
- **Risk:** a PSUM WAR race may pass on one input set. Check the WAR constraints explicitly for each of the 24 heads.
- **Cheapest falsifier:** do it for one head and check the happens-before graph covers that head's PSUM buffer reuse.

**5. Tune the tail and small waits.**
- **Where:** PE outer-product `s4` waits (LNC2 iter-5 found 4 of 12 casts did not update s4, so PE waited up to 2 us late) and DVE sink ordering (LNC2 iter-7).
- **Gain and ceiling:** ~0.5–1.5 us each. These are only worth doing after the large wins; a 0.5 us gain is near the 0.6 us noise.

## Closed (from the LNC2 notes; refuted there, so re-check cheaply before reusing here)
- Merging the rsqrt and silu table sets into one: no vendor set contains both, and a hand-merged set would need ~2126 entries against a vendor maximum of 1447.
- Grouping the f36 without giving each a private dst buffer: blocked by the WAR waits that protect buffer reuse.
- Grouping the rsqrts (f29): they are the serial tail of each head's chain and share reused scalar buffers.
- Moving the q_scale rsqrt (`f29@7`) later: it hangs, as it creates a cycle.
- Moving the `o_sb` COPY to DVE: structurally impossible (a cycle through ACT, PE and DVE).
- Moving the `o_sb` COPY to Pool: hangs even with a clean semaphore graph. Inserting slots between `PSEUDO_DMA_*` instructions on Pool is the suspected cause.
- `ACT_TABLE_LOAD` has no knob that changes its 1283 ns.

## Unknown (what would change the ranking)
- The head count per table-load group and the real table-load floor for 24 heads. Settle it by decoding the ACT stream with `isa.py`, mapping funcs to table sets, and building the forced-order graph.
- How much of the ACT table-load cut turns into total time. The trace has 5.3 us and 42 us semaphore stalls on ACT near the head (~16.8 us and ~22.5 us) and an input-DMA-bound start. Settle it by profiling after step 1 alone and checking ACT busy against total time.
- Whether DVE (34 us) and PE (21 us) can absorb 35 us of moved ACT work without becoming the new limiter. Settle it by converting a few instances and measuring DVE busy.
- Whether the DMA tail (state-out DMA, 6 transfers) is on the critical path at the end. In LNC2 splitting it into 4 per-group transfers gave −5%. Settle it by looking at the last ~10 us of the DMA and ACT timelines.

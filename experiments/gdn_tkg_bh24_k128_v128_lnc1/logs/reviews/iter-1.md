# Review of iter-1 (kernel.neff vs baseline; 153.46 us -> 79.64 us)

## 1. What changed
`neff_diff baseline/kernel.neff kernel.neff`: 40 of 46 members identical. Changed: `sg00/Activation0.bin`, `DVE0.bin`, `PE0.bin`, `Pool0.bin`, `SP0.bin`, `SP0.json`. `notes/` is still the empty template, so there is no claim to check. Everything below comes from the diff.

- **Activation0.bin: 324 -> 191 slots, 317 differ.** This is the main edit.
  - Slot 2: the per-bh `ACTIVATE` with `activation_func` 7 -> 36 becomes one wide op. `src/dst num_elem` goes [24,1,1] -> [3072,1,1], dst 0x4060 -> 0x20000, and it gets `events_update_mode=SEM_INC_COMPLETE idx 4`. Same technique as lnc2 iter 1 (many per-bh activations hoisted into one wide `ACTIVATE`).
  - Table-load / `ACTIVATE` order is rearranged. `ACT_TABLE_LOAD` is now 6 (the profile shows 6).
  - Roughly 133 slots are gone: ACT instruction count 347 -> 214, `ACTIVATE` 93 -> 53, the opcode-`COPY` slots 24 -> 0. Waits and semaphore values downstream are renumbered (e.g. `events_semaphore_value` 5->7, 6->8, 12->14).
- **DVE0.bin: 156 -> 159 slots, 140 differ** (shifted and renumbered). `PE0.bin: 151 -> 152, 146 differ`. Both are mostly wait renumbering.
- **Pool0.bin: 7 -> 14 slots.** The state-out `PSEUDO_DMA_DIRECT2D` is split from one transfer (`src/dst_num_elem` [24,128]) into 8 transfers of 3 bh each ([3,128]). Each has its own wait on sem 6 (24, 33, 46, 53, 64, 71, 82, 90). The final waits are sem 13 >= 16 and sem 27 >= 128.
- **SP0.json / SP0.bin:** a new input DMA block is added in the json (`norm_weight`, `z`, ... reordered). The output doorbell wait on s6 goes 91 -> 93, which matches 2 extra DVE increments.

Consistency: this matches the lnc2 experiment's iterations 1 and 4 (wide ACT hoist plus state-out split). It is a schedule and placement edit with the maths unchanged. The verifier agrees: `max_abs_err` is 2.322e-02 and 1.633e-03, identical to the baseline's digits, so the output is almost certainly bit-identical.

## 2. Is the latency real?
Yes. This is not noise.
- `profile_latest.json`: median 79.641 us, samples 79.44-79.66, spread 0.28%.
- Baseline: median 153.463 us, spread 0.79%.
- My own re-profile gave 79.449 us.
- The gain is 74 us against a spread of well under 1 us. The Activation engine went from 143.5 us busy to 63.6 us, and the instruction counts agree.

Residual doubt: `verify_latest.json` has `runs: 1`, so the verifier saw one execution.

## 3. Safety beyond the verifier
- **Semaphores.** I ran `semaphores.py` on both unpacked NEFFs. Every semaphore has `max_wait == total_inc` in the new NEFF, so no deadlock or over-wait is visible.
  - sem4 203 -> 140 (the `COPY` ops are gone or recoded).
  - sem6 91 -> 93.
  - sem12 waiters 2 -> 3, sem25 waiters 1 -> 2.
  - sem27 16 -> 128 (the 8 DGE chunks x 16).
  - The tool checks counts, not which instruction a wait is really for. A wait pointed at the wrong producer would still balance. This is the same blind spot lnc2's iter 7 found with edit E, which gave error 4.4 and passed every tool except the verifier. The only evidence here is one verifier run, with output identical to the baseline's digits.
- **Error margin.** output_0 is at 83% of the limit (0.0232 of 0.028). That equals the baseline, so it is inherent to the kernel and not drift.
- **Hazards I did not audit.** I did not re-derive PSUM or SBUF WAR hazards for the removed `COPY` slots, and I did not check the rewritten SBUF destination (0x20000).
- **Cheap extra check:** run `verify.py` with `--runs 3`, plus one different random input if it supports that. A race would show as run-to-run variation.

## 4. Where the remaining time is (fresh profile, `tools/profile.py --repeats 1 --parquet`)
Timeline, in us from the start of the first instruction:

| span | what happens |
|---|---|
| 0-10 | Runtime preamble: `TENSOR_LOAD`, semaphore setup. Not editable. |
| 10-21 | Input DMAs. State-in DGE data runs 12.9 -> ~21.2 (1.5 MB, 3 Pool triggers of 1.73 us each at 10.3, 12.0, 13.8). |
| 9.7-22.2 | ACT runs a serial chain of tiny ops (gelu, hardswish, softplus and others) with 5 `ACT_TABLE_LOAD`s of 1.28 us each. |
| 20.8-27.5 | DVE `TENSOR_TENSOR` slot 18, then slot 20, 3.35 us each. Slot 18 is 1 channel x 3072 elements. Slot 20 is 128 channels x [128,24]. |
| 22.2-27.5 | **ACT is idle for 5.3 us**, waiting on its slot 13 (s6 >= 10). |
| 27.5-87.2 | Body: 24 bh at about 2.5 us each. |
| 87.2-91.2 | Tail: last state-out DMA completes at 88.3, output_0 DMA runs 88.86-89.26, then runtime drain. |

Engine totals, busy / wall: ACT 63.6 us (80%), DVE 40.2 (50%), PE 20.8 (26%), Pool 14.7, DMA 21.6.

ACT body ops (counts and totals from the instruction table):

| ops | count | total |
|---|---|---|
| `CAST` | 72 | 21.2 us |
| `ACTIVATE` func `COPY(scale/bias)` | 24 | 11.8 us (490 ns each) |
| `ACTIVATE` func `UNKNOWN` | 53 | 14.7 us (~278 ns each) |
| `ACTIVATION_READ_ACCUMULATOR` | 24 | 7.0 us |
| `ACT_TABLE_LOAD` | 6 | 7.7 us |

DVE is only half used, and PE is a quarter used.

## 5. Ranked candidates for iter-2

| # | Candidate | Where and what | Mechanism / evidence | Gain, ceiling and its source | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | **Release ACT's first CASTs as soon as DVE slot 20 is done, and optionally split slot 20 per bh group.** | `DVE0.bin` slots 18 and 20 (the two 3.3 us `TENSOR_TENSOR`s); `Activation0.bin` slots 12-14 (the s25 and s6 waits). (a) Reorder DVE so slot 20 comes first, then slot 18, and lower ACT's s6 wait from >= 10 to >= 9. This only works if ACT's first CASTs do not consume slot 18's output. (b) Split slot 20 [128,24] into 3 or 8 chunks of bh (num_elem [128,8]), each with its own s6 increment. Make the ACT casts wait chunk by chunk, and wait per DMA chunk (s25 value 16 is the 3-chunk state-in) instead of all of it. | The ACT stall 22.2 -> 27.5 comes from the two serial 3.3 us DVE ops, not from DMA. State-in is complete by ~21.2 and slot 20 depends only on ACT's softplus (done 21.07). Semaphores count increments, so lowering the wait value releases ACT earlier. Splitting by bh follows lnc2 iter 4 (state-out split / pipelined `S *= exp_g` hand-off, -5%). | (a) ~3 us (slot 20 can finish at ~24.4 instead of 27.5). (b) up to ~5 us (the ACT idle 5.3 us bounds it). Total budget is the 5.3 us ACT idle plus the DVE chain. | Medium. If a CAST or PE consumer needs slot 18's output (k/q scaling) the wait lowering is wrong. That passes one verifier input only by luck and would be a race. Reordering DVE slots changes which `events_update_idx=6` increment is which, so check every s6 waiter's value against the new order. | Profile after (a) alone. If wall is not lower by ~2 us, ACT is gated by something else, so stop. Then dump which ACT instructions read the slot-18 dst, and the dst of slot 20, from `Activation0.bin`. |
| 2 | **Delete the 24 PSUM->SBUF `COPY(scale/bias)` ACTIVATEs (11.8 us of ACT)** by pointing their consumers at PSUM. | `Activation0.bin` the `ACTIVATE func=COPY` slots (~49 for 24 bh); their readers in `DVE0.bin` and `PE0.bin`. Change the consumer's src address to the PSUM tile and drop the ACT slot. Renumber the s4 waits and `events_semaphore_value`s of everyone downstream. | Done in lnc2 iter 3: 12 ACT PSUM->SBUF `o_sb` copies deleted, both readers read `out_p` straight from PSUM, -8.7% (-4 us of 44). It required that all PSUM WAR constraints be already implied by the happens-before graph (8 PSUM banks). | ACT work -11.8 us of ~63. The body is ACT-bound, so a realistic wall gain is 4-8 us. DVE has 35 us of headroom to absorb extra read cost. | **High.** A wrong renumber or a skipped PSUM WAR is a silent race. The verifier covers one input. Per lnc2 notes, `semaphores.py`/`cycle.py` missed edit E. | Do 1 bh of 24 first. Check PSUM tile reuse distance with `hb.check_psum`, then measure. If the PE reuses that tile within the next bh, stop. |
| 3 | **Merge the 72 per-bh CASTs into wide casts** (3 bh or 6 bh per op). | `Activation0.bin` CAST slots (72, ~295 ns each, 21.2 us). Merge the `S->S_bf` style casts by bh group (num_elem x3 or x6 on distinct private tiles). | lnc2 iter 2 merged 12 casts into one wide cast. The per-op floor is ~200 ns of the 295 ns, so merging 3 saves ~2 fixed costs. Iter 1 here already did the wide hoist for `ACTIVATE`. | If a cast is ~200 ns fixed + ~95 ns of data, merging 3 saves ~0.4 us x 24 = ~9.6 us of ACT time. Real gain maybe 3-5 us. **Unknown whether wide cast scales** (lnc2 listed this as unknown). | Medium. Group fill delays the first bh in each group. Needs distinct tiles and PE `s4` re-deriving. | Time a single 3-bh wide CAST in a scratch NEFF. Keep it only if it is below 3 x 295 ns by >=30%. |
| 4 | **Move CASTs (or `UNKNOWN` ACTIVATEs) from ACT to DVE.** | The `CAST` slots that read PSUM and write a private tile; DVE has 35 us of headroom here. | lnc2 closed `fuse.py`/ACT<->DVE moves at 0.0 us, but that was after DVE reached 95%. Here DVE is 50% busy, the opposite regime, so it is not refuted for this kernel. Precedent: lnc2 iter 2 and iter 3 moved casts to DVE. | Each moved CAST frees 295 ns of ACT; DVE cost ~150 ns + 1 ns/elem. Ceiling: ACT is 80% busy, so ~10 us of work moved is worth ~5 us wall. | Medium. Pool is not a host (closed). DVE reads PSUM, so check that the dtype and the output cast match exactly (bit-identical output is the only goal). | Move 4 CASTs and measure. If wall does not drop ~1 us, ACT is not the binding engine for those. |
| 5 | **Split the last state-out chunk and shorten the tail.** | `Pool0.bin` slots 11-13 (the last 3-bh transfer plus waits); `SP0.bin` slot 6. Split the final chunk into 1-bh transfers. | lnc2 iter 4 split state-out for -5%. Here the last chunk is triggered at 86.28 (882 ns), completes ~88.3, and gates output_0 at 88.86. | Ceiling ~0.5-1.0 us (88.3 -> ~87.5). Everything after 87.2 is already mostly runtime drain. | Low-medium: wrong `semaphore_value` deadlocks (visible). | Check whether SP@6's trigger precedes or follows 88.3 in a fresh profile. If the doorbell is not gated by the state-out completion, stop. |
| 6 | **Drop one `ACT_TABLE_LOAD` (6 -> 5).** | `Activation0.bin` table-load slots between the gelu/hardswish/softplus ops (9.7-22.2). | lnc2 notes say 5 is the floor ("head table-load sequence (7 loads, 5 is the floor)"). | 1.28 us per load, but only if this serial chain is on the critical path. The chain ends at 22.2 while state-in ends ~21.2, so about 1 us. | Medium: a wrong table gives wrong activation values (verifier catches). | Check which functions actually share a table in the 5-load set. If no merge exists, drop it. |

Ranked by gain per unit of risk and effort: 1 is cheapest and best-evidenced. 2 has the largest ceiling but the highest risk, so do 1 first. 3 and 4 are gated by a one-op scratch measurement. 5 and 6 are small.

## Closed (do not redo)
These come from the lnc2 sibling experiment. None of these were re-measured on this kernel.
- Pool as host for CAST or per-head `TENSOR_TENSOR`: Pool CAST ~4x slower than ACT; Pool TT gave error 4.8.
- Further PE wait tightening: slack was 0.000.
- Input DMA order: no gain.
- Stride-across-partitions access patterns: `step_elem` has no partition axis.
- Folding `norm_weight*silu_z` into one wide op (`fold.py`): +0.37 us (worse) at lnc2.
- ~3 us runtime drain after the last compute slot: not editable.

## Unknown (would change the ranking)
- **Whether ACT's first CASTs depend on DVE slot 18's output.** Decide candidate 1(a): decode the producers and consumers of slot 18's dst.
- **Whether a wide CAST scales** (candidate 3's ceiling). Settle: one scratch NEFF with a 3-bh cast.
- **Whether the body is ACT-bound or chain-bound.** ACT has ~7 us of waits inside the body. If those waits are on DVE or PE, candidates 2-4 are worth less. Settle: sum ACT `evt_wait_time_ns` by waited semaphore in the body.
- **Why the three Pool state-in triggers are serial** (1.73 us each, at 10.3, 12.0, 13.8). If they could issue in parallel on other queues, state-in would end ~3 us earlier. Settle: check which queues the SP/ACT engines can drive and whether a second DGE issuer is legal.

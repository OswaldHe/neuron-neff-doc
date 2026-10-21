# Review: iter-2 (selective_scan b1 c2560 s4096 n16 bf16)

## 1. What changed
`kernel.neff` (uncommitted) is a continuation of the committed iter-1 edit (HEAD fcf76a6), not a fresh edit of baseline. Diffed with `neff_diff.py` against HEAD and baseline, and with `semaphores.py` on unpacked copies of all three.

Four members change. The other 30 are identical.

| member | baseline -> HEAD -> now (slots) | what |
|---|---|---|
| Pool0.bin | 3842 -> 6056 -> 6407 | HEAD added 2214 fp32 `TENSOR_TENSOR_ARITH_OP ADD` (512 elem) and their waits. Now 2565 of them: +351 this iteration. |
| DVE0.bin | 15929 -> 15929 -> 13886 | The DVE `y_accum += Cs` ADDs went 5779 -> 5428 (the DVE TT_ARITH count; this iteration moved 351). |
| Activation0.bin | n/a | About 2290 slots differ, only in `events_semaphore_value`. The values are re-based: some by +13 or +14, most by -2. |
| PE0.bin | n/a | 4961 slots differ, also only wait/semaphore values (not decoded in detail). |

The moved ADDs are the same op. Dtypes are FP32 in and out, `op=ADD`, 512 elements, 128 channels. Only the engine, the wait on sem 6 (DVE) and the increment on sem 3 (Pool) change.

This matches the HEAD commit message's mechanism, which is "move ADDs to Pool with semaphore hand-off". It is just more of them.

**`notes/` is empty.** It contains only the README stub, so nothing was recorded about which slots moved, the count, or why 351. The next agent must not assume anything beyond this review.

The semaphore re-basing on the other engines is expected. Sem 3 and sem 6 totals changed, so every downstream `events_semaphore_value` is shifted.

## 2. Is the latency change real?
| NEFF | median us | spread |
|---|---|---|
| baseline | 7919.96 | 0.024% |
| HEAD (committed claim) | 6886.3 | n/a |
| now | 6818.83 | 0.02% (6817.91 to 6819.29, 5 samples) |

- The spread is about 1.5 us. The step from HEAD is about 68 us (-1.0%), roughly 45x the spread, so it is real, not noise.
- Cumulative gain is -13.9% (-1101 us).
- Engine busy: DVE 7860 -> 6453 us (94.6% of exec), Pool 1214 -> 6068 us (89.0%), ACT 1903, PE 2969, DMA 1429, SP 609.
- The two engines are now nearly co-saturated, so the easy gain is almost exhausted (see candidate 1).
- 6818.8 is DVE busy plus 365 us. The wall is about 5% above the busiest engine.

## 3. Safety
- **Error is unchanged.** `output_0` max_abs_err is 0.125, which is 83% of the 0.15 limit. It is identical to the baseline's 1.250e-01, with cosine 0.999944. `output_1` is 1.8e-4. The arithmetic is bit-identical, as the ADDs are the same fp32 op. The 83% figure is the baseline's own error, not new error.
- **Synchronisation was tightened, not loosened.**
  - The total increments on sems 3 and 6 balance their max waits (sem 3: 5053 inc vs 5052 max wait; sem 6: 8174 vs 8174).
  - Sem 3 waiters rose from 2499 to 4339 and sem 6 waiters fell from 17683 to 14391, i.e. waits were added and re-pointed, not removed.
  - The DMA and DGE sems (8-39) are unchanged.
- **Residual risks the verifier cannot see.**
  - A Pool ADD reads and writes `y_accum` in place. If its WAR/RAW ordering against the DVE scan or PE consumer relies on the previous engine's in-order execution, a one-slot race would pass fixed inputs.
  - Pool ADD latency is bimodal (1.27 us vs 2.16 us, from the iter-1 profile). The slow mode is plausibly contention for the shared DVE/Pool SBUF port, so timing varies.
  - I did not do a per-slot hazard proof. I checked only the semaphore accounting.
  - Cheap extra check: re-run `verify.py` with the other `bench/inputs` seeds if any exist, and run the NEFF 20 or more times, comparing outputs bit-for-bit across runs. A race would show as run-to-run differences.

## 4. Ranked candidates for the next iteration

Per-op costs (iter-1 profile in `prof/`, which has 2214 Pool ADDs; the numbers are close enough for ranking):
- DVE TT fp32 512-elem: ~676 ns.
- Pool TT fp32 ADD: 1268 ns, or 2156 ns when contended; mean 1551.
- DVE remaining mix: 5120 fp32 MULT (512 elem), 2560 `TENSOR_TENSOR_SCAN` (~1291 ns), 155 ADD.
- Pool COPY: 2560 x ~688 ns = 1760 us. These are 1-element FP32 copies (`num_elem=[1,1,1,1]`, 128 partitions) into a staging row at dst 33024+4k. They are almost pure fixed overhead.
- DVE `EVENT_SEMAPHORE`: 7390 instructions, 722 us total. 191 of them are over 100 ns and account for 439 us.
- Idle capacity: ACT is 72% idle (~4900 us), PE 56% idle.

| # | Candidate | Engine / member | What changes | Mechanism | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Move ~150-190 more `y_accum+=Cs` ADDs DVE -> Pool | DVE0.bin and Pool0.bin, the remaining 155 DVE ADDs plus any with the same structure | Same edit as this iteration: change the op's home, re-point the wait to sem 6 and the increment to sem 3, re-base downstream sem values | Established on device (iter-1 and iter-2) | Balance point x where 6453 - 0.68x = 6068 + 1.4x (us), so x ~ 185 and DVE ~6330. Wall gain ~60-120 us (1-1.7%). The ceiling comes from the engine-busy numbers above. | Low (same as the accepted edit) | Move 100 and read DVE/Pool busy. If the wall does not move by more than ~1.5 us per 10 ops, stop. Run it only after candidate 2 or 3, because Pool is already at 89%. |
| 2 | Cut the 1-element Pool COPYs (carry/stage) | Pool0.bin, the 2560 COPY slots (opcode COPY, `src_mem_pattern_num_elem=[1,1,1,1]`) and their consumers | Decode each COPY's src (e.g. 53564, 24572, 41276) and dst (33024+), find the consumer reading the staging row, and re-point that consumer's src address to the COPY's src. Then delete the COPY and shift sem-3 values. | Not established. It is only valid if the staging copy exists for layout or address contiguity and not for WAR protection. This is the main unknown. | Pool frees up to ~1760 us. With candidate 1 continued, x ~ (6453-4368)/2.08 ~ 1000 ADDs moved, so DVE ~5770 and wall maybe ~6100. That is up to ~700 us (10%), and it is unmeasured. | High. A wrong source address, a hazard, or a stride-contiguity requirement gives silently wrong numbers. The verifier would catch most of it because outputs depend on every chunk, but not a race. | Decode 20 COPY/consumer pairs with `isa.py`. If any consumer reads a range of staging addresses (stride > 1 over the staged slots) and not a single source, kill it within 10 minutes. |
| 3 | Move the 1-element COPYs to ACT instead of deleting them | Activation0.bin (insert) and Pool0.bin (delete) | Insert an `ACTIVATE` identity/copy (fp32 -> fp32) in place of the Pool COPY, same src and dst, and re-point the sem-3 increment to sem 4 | Idle ACT has ~4900 us of slack, and ACT ACTIVATE costs ~720 ns. A table load for the function is needed (`ACT_TABLE_LOAD` seen once). | Same Pool relief as 2, ceiling the same ~700 us, minus ACT's added latency in the dependency chain (sem 4 already has 908 waiters). | Medium-high. Wrong ACT encoding or table; the chain may become latency-bound. | Hand-write one ACTIVATE copy slot, run once, and check that ACT executes it and the value is bit-exact. |
| 4 | Remove redundant DVE `EVENT_SEMAPHORE` slots | DVE0.bin | Delete standalone waits already implied by the next instruction's own wait or by an earlier wait on a monotone sem (sems are monotone) | Documented semaphore model (`semaphores.py`: sems only increase, waits are `>= value`) | Up to 439 us, but only if truly redundant. The realistic fraction is unknown, so I'd guess <150 us, hence ranked below the safe lever. | Highest. A removed wait can pass once and race later, and the verifier will not catch it. | Static check first: for each candidate, prove implication from an earlier wait on the same engine. Do nothing if the proof is not mechanical. |
| 5 | Pool ADD contention (bimodal 1268 vs 2156 ns) | Pool0.bin | Re-time or stagger so Pool ADDs avoid overlapping DVE port use | Hypothesis only | Up to ~890 ns x number of slow-mode ADDs (about half of 2214 in the iter-1 profile, ~1000 ns each saved). Unconfirmed. | Medium. It can lengthen the DVE critical path. | Read `Instruction.parquet`: do slow-mode Pool ADDs overlap in time with DVE TT ops, and fast-mode ones not? If not, drop it. |

Ranking by gain per risk and effort: 1 first (safe, small), then 2/3 (the only large lever), then 4 and 5.

**Closed.**
- Moving the `y_accum+=Cs` ADDs off DVE onto Pool is established, with no accuracy cost. DVE was 99.2% busy and Pool 15% before it. Both are now ~90%, so a wholesale further move is exhausted. (7919.6 -> 6886.3 -> 6818.8 us.)
- Noise is not a concern: the measured spread is 0.02% (~1.5 us), so a >5 us change is real.

**Unknown.**
- Why each Pool COPY exists. A decode of 20 COPY/consumer pairs settles candidate 2.
- Whether slow-mode (2156 ns) Pool ADDs coincide with DVE activity. An overlap analysis of `Instruction.parquet` start/end times settles candidate 5.
- A fresh per-instruction profile of the current NEFF. `prof/` is from the iter-1 NEFF (2214 Pool ADDs), so run `profile.py --repeats 1 --parquet prof` on this one before acting.
- Whether the ~365 us gap between wall and DVE busy is DVE wait stalls. Sum the `EVENT_SEMAPHORE` stall durations on DVE by blocking sem to find out.

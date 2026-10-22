# Review of iter-6 (kernel.neff vs HEAD 669f7b0 and vs baseline)

## 1. What changed
`neff_diff.py` against baseline shows 3 engine streams edited (Activation0, DVE0, plus the already-landed history). Against HEAD (the part this iteration adds, the uncommitted `M kernel.neff`):
- Only `Activation0.bin` (1099 slots differ) and `DVE0.bin` (332 slots differ) change; same size, same slot count, no other member.
- DVE: 332 `COPY` slots become `TENSOR_TENSOR_ARITH_OP` (a src1 operand with a per-partition broadcast is added; this is the same `step_elem=0` trick as the previous commit, now applied to the remaining Y_off units). 5 more `COPY` become `ACTIVATE`.
- ACT: ~148 `ACTIVATE`/`COPY` slots become `EVENT_SEMAPHORE`, and their neighbouring wait slots are rewritten (`events_wait_mode` / `events_extended_*` fields, e.g. slot 19 loses its `WAIT_FOR_SEM_GE_IMM 10` and extended wait).
- Semaphore accounting (`semaphores.py`, sg00), HEAD -> now: sem4 `ACT:COPY` 366->27, `ACT:ACTIVATE` 5127->5318, `ACT:EVENT_SEMAPHORE` 435->583; sem6 `DVE:COPY` 444->112, `DVE:TT_ARITH` 2298->2630. Total increments per semaphore are unchanged (sem4 5928, sem6 4846), but sem6 waiter count drops 6419->6093, so ~326 waits were removed or merged.
- Consistent with the HEAD commit's story (the continuation of the Y_off fold: the remaining ~244 of 768 units). **`notes/README.md` is still the 279-byte template**, so there is no written record of what was done, which slots were gated, or why. The claim cannot be checked against notes; the commit message is not written yet.

## 2. Is the speedup real?
Yes. `profile_latest.json`: median 2948.94 us, samples 2947.07 to 2949.26, spread 0.075%. A fresh single-repeat run gave 2950.19. HEAD recorded 2988.1, so this iteration is about -39 us (-1.3%), about 500x the spread. Versus baseline 5248.9 us it is -43.8%. Not noise. The gains are now small per step (-2.7%, -1.3%), though.

## 3. Safety
- verify_latest: all pass; output_0 max_abs_err 2.058e-07 (89% of the 2.3e-07 limit), output_1 1.993e-07 (87%). Earlier commits say error is unchanged vs baseline, so this is inherited, not new, but the margin is thin and any real numerical change (reordered fp32 accumulation) would fail. `runs: 1` only.
- Sync risk: this iteration adds 148 more `EVENT_SEMAPHORE` that inherit a sem4 increment from an ACTIVATE that used to write SBUF. The increment can now fire before the pipelined predecessor ACTIVATE finishes its write unless an extended wait covers it. The HEAD commit notes that without the gate "the state silently corrupted". Dropping 326 sem6 waits is the other loosening. Totals match, so no deadlock, but a race would be timing dependent and one run on a fixed input can pass it. Run verify at least 5-10 times and compare bit-exact with HEAD output, not just within the limit.
- I did not re-derive the liveness gate; I only confirmed the totals are conserved.

## 4. Next candidates (ranked)
Profile facts used: PE 43%, ACT 61%, DVE 56%, Pool 53%, SP 20% busy of 2949 us. No engine is saturated, so the kernel is dependency/latency bound (engine gaps 1.3-2.3 ms each). Instruction counts: ACT ACTIVATE 6836 (COPY 4540 x 328 ns, SOFTPLUS 2108 x 177 ns), PE MATMUL 5441 x 350 ns + LDWEIGHTS 5441 x 277 ns, Pool DMA_DIRECT2D 2385 x 655 ns, DVE STREAM_SHUFFLE 4608 x 148 ns, TENSOR_TENSOR_SCAN 768 x 470 ns, ACT_TABLE_LOAD 17 x 1283 ns.

| # | Edit | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|------|-----------|---------|------|------|
| 1 | **Finish the Y_off tail**: ACT `COPY` 27 and DVE `COPY` 119 (avg 212 ns) left; fold or turn them into EVENT_SEMAPHORE where the same gate admits | same `step_elem=0` broadcast + sem-4 gate as 669f7b0 | <= 119x212 + 27x206 = ~31 us busy, realised maybe 10-15 us (0.4%) | low, same gate | count remaining COPY after edit; profile median must drop >5 us |
| 2 | **Remove ACT table reloads**: 17 `ACT_TABLE_LOAD` remain (21.8 us total). Group each reload so SOFTPLUS and EXP/COPY sit adjacent, or convert the remaining recip(exp) units to exp(-x) | table swap happens only when func changes; established in 1ed7001/1db4928 (1283 ns each) | hard cap 21.8 us (0.74%) | low, ACT-only | list ACT_TABLE_LOAD pcs in Instruction.parquet; if the remaining loads are init-only, close it |
| 3 | **Drop PE LDWEIGHTS where the stationary operand repeats**: top weights (`0x487c` 64x128 x624, `0x549c` 128x128 x576, `0x457c` x588) are loaded per matmul | skip LDWEIGHTS when previous matmul used identical src/shape and nothing wrote that SBUF in between; need PE sem/wait fix-up | LDWEIGHTS busy 1506 us but pipelined with MATMUL; real gain unknown, PE is only 43% busy and not the binding engine | medium: stale weights give wrong numbers but verifier would catch it (deterministic) | consecutive LDWEIGHTS operands differ in all 5441 (checked: 0 identical neighbours), so first check non-adjacent repeats; if none, close |
| 4 | **Cut Pool DGE descriptors**: Pool issues 2385 DMA_DIRECT2D at 655 ns (1.56 ms, 53% of exec); merge adjacent (head,chunk) transfers into one 2D descriptor | larger `step_elem`/count in one PSEUDO_DMA_DIRECT2D, fewer sems (8-23, 2384 incs each) | bounded by Pool busy 1.56 ms but only if Pool is on the critical path: unknown | high: needs semaphore total rebalance (`semaphores.py`), races data-dependently | check whether Pool gaps precede PE/ACT stalls in Instruction.parquet (evt_wait); if consumers are not waiting on Pool DMAs, close |
| 5 | **SOFTPLUS (2108 x 177 ns = 373 us)**: check whether it can be batched across heads into longer free-size ACTIVATEs | fewer fixed-overhead ACTIVATEs; ACT cost is ~150 ns fixed + per-element | up to ~100-150 us busy on ACT; realised unknown | medium-high (AP shape and PSUM bank layout) | pick 2 adjacent SOFTPLUS slots, check if src/dst are contiguous; if not, close |

If candidates 1 and 2 give under 1% combined, the remaining 2.95 ms is mostly cross-engine dependency latency and not reducible by field edits; say so rather than inventing levers.

**Closed** (from commit history; notes are empty so these are the only records): imm pointer operand on ACTIVATE (done, -8.3%); 1531 -> 17 table reloads (done); HBM tmp spill bounces for state_sb (done, 6.68 MB -> 0.37 MB spill bytes); fp32 -> lower-precision matmul is not allowed (error already at 89% of limit).

**Unknown**
- True critical path: reconstruct the semaphore wait graph from `Instruction.parquet` (`evt_wait_time_ns`, start/end) and find which engine the longest chain sits on. Settles the ranking of 3, 4 and 5.
- Whether 148 gated EVENT_SEMAPHORE are race-free across runs: run verify 10 times.
- Run-to-run error stability (89% of limit): repeat verify and record the max.

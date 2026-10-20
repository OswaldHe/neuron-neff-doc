# Review of iteration 4 (working-tree `kernel.neff` vs HEAD 9d8415c and vs baseline)

## 1. What changed

`notes/` holds only the template README, and this iteration has no commit message yet. So "consistent with the notes" cannot be checked. The diff is explained below from the NEFF alone.

Against baseline, the 6 changed members are Activation0, DVE0, PE0, Pool0, SP0 and one more (header/json). Iterations 1-2 account for most of that: staging round-trips fused, DGE saves moved off Activation, two var9 reloads moved to Pool.

**This iteration** (HEAD -> working tree) changes 5 members and 16 slots:

- **Activation0 slot 88.** This DGE is swapped with Pool0 slot 22.
  - Activation now does the job Pool did: src 0x806480 -> dst 0x101ce00, sem 65.
  - Pool 22 now does the old Activation job: src 0x1006480 -> dst 0x101d000, sem 66.
  - Both gain a guard, `wait sem4 >= 42/43`. Before, they had no wait.
- **Pool 23 and SP 30/34.** Only their completion semaphores are renamed: 48 -> 67, 46 -> 22, 49 -> 23. Src and dst are unchanged.
- **SP 37, 38, 39 become zero-field no-ops.** These were relay waits on sems 50, 49, 48 and 47, plus a relay increment of sem 64 for DVE 98.
  - Consumers now wait directly on the DGE semaphores:
    - DVE98: sem66 >= 16
    - DVE107: ext sem67
    - DVE117: ext sem23
    - DVE128: sem50 >= 16
    - Pool24: sem22 and sem65
    - SP35: ext sem43
    - SP42: ext sem16
- **PE0 slot 134 (LDWEIGHTS), the one structural change.**
  - It used to read the DMA-transposed copy at SBUF 0x457e. It now reads partition 64, offset 0x1ccfe, which is the DGE output (Activation slot 82, dst 0x101cc00). The row step changed from 0x4000 to 0x8000.
  - Its wait changed from `sem28 >= 32` (both SpillReload transposes) to `sem43 >= 16` (the Activation82 DGE only).
  - The transpose DMA (SP trigger 36, block 20) is still triggered but nothing in the diff consumes it. It is probably dead; I did not grep for other readers of 0x4500-0x45ff.

## 2. Is the latency change real? Yes.

- `profile_latest.json`: median 59.622 us, min 58.759, max 59.861, spread 1.85% (about 1.1 us). Baseline is 67.525, spread 0.95%.
- I re-measured the working tree and HEAD interleaved, twice, 5 repeats each:

| Build | Medians |
|---|---|
| HEAD | 61.4 to 61.5 us (reported 60.93 at commit time, so about 0.5 us of drift between sessions) |
| Working tree | 59.71 and 59.36 us |

- The gain is about 1.9 us (3%). That is above the 1.1 us spread and reproduced in both rounds. It is not noise.
- Overall: 67.5 -> about 59.5 us, a 12% gain.
- The sub-1 us wins in earlier iterations (-0.9 us for iteration 2) were within about one spread and rest on a single A/B. Treat anything under about 1 us as unproven until it is repeated.

## 3. Safety

- **Verifier.** Both outputs have the same max_abs_err as baseline: 6.092e-05 and 4.494e-04. They are identical to 4 digits, consistent with bit-identical output.
- **Error headroom.** output_1 is at 83% of the 0.00054 limit. That is baseline behaviour, not something this iteration moved, but it leaves little room for any numerics-changing edit (lower-precision dtypes are out).
- **Semaphores.** `semaphores.py` is clean: no wait exceeds its increments. One tool gap: it counts only the primary `events_wait_*`, not the extended wait field. It reports "0 waiters" for sems that are in fact awaited through ext waits (23, 65, 67, 43 and others). Do not read 0 as dead.
- **Race review of the rewired waits, done by hand.**
  - Every consumer of a DGE-produced region now waits on that DGE's own semaphore at the right count.
  - Several waits are tighter than before. The two moved DGEs previously had no wait on their Activation-produced sources.
  - PE134 waits on sem43, the single Activation82 DGE, which is a correct guard for what it now reads.
  - The remaining risk is WAR: PE134 and the DGE destinations are reused later. The addresses are unchanged, so I found no new hazard.
- **Why the verifier might not catch a race.** It uses one fixed input set. A mis-ordered DGE completion would show up as wrong data on that input, so a pass is evidence. It is not proof for a late-arriving DMA that wins the race on every run but loses on slow runs.
- **Run-to-run check.** All 14 profile runs (7 repeats in `profile_latest.json` plus my 4 and the verify runs) executed without error. That is mild evidence against a race.

## 4. Candidates for the next iteration (ranked by gain per risk and effort)

Timeline facts below come from `prof/Instruction.parquet` of the current NEFF. Profile time zero is the start of the preamble, so kernel time is profile time minus about 12 us.

| # | Edit | Mechanism / evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Startup input DGEs: stop serialising on Pool.** Pool0 slots 1, 6, 8-13 (8 DRAM->SBUF input loads, no waits) run back to back at 12.73-19.25 us, 0.65-0.79 us each. SP is idle until 16.5 and Activation is idle after 13.5. Keep 1, 6 and about 2 others on Pool and move the rest to SP and Activation as DGEs. Keep the same semaphores (8-15), so consumers stay valid. | DGE retargeting across engines is the lever iterations 1-3 established (commits 86e9bd9, 9d8415c, and this diff). First chain: DVE CAST (pc 8) starts at 15.81 after a 2.07 us gap. That matches Pool6's DGE (ends 14.78) plus about 1 us DMA. DVE COPY (16.34) then gates SP1/Act2, and the transposes at 17.3. Pool6 sits behind two MOVEs and two AFFINE_SELECTs. | **About 1.5-2 us.** Bound: CAST could start at about 14.0 instead of 15.81. Later input DGEs (ending 19.25) may become the next gate, so it could be less. | Low. Same mechanism as before. Inserting a slot at the head of the SP or Activation stream means rotating slots; SP37-39 are now free no-op slots to absorb the shift. A wrong destination would corrupt the inputs, which the verifier would catch. | Swap Pool0 slot 6 ahead of slots 2-5. This is a pure within-engine reorder (neff_doc ch14 r08). If the DVE CAST start moves from 15.81 toward about 14.6, the hypothesis holds. |
| 2 | **Tail: issue the two output DGEs in parallel.** SP slot 64 (src 0x4f00 -> 0, `wait sem4 >= 60`) and slot 65 (0x4b00 -> 0xc000, `wait sem6 >= 93`) run serially on Sync: 68.70 + 0.64, then 69.40 + 0.88. Move slot 65 to Pool, which is idle after 64.8, with the same sem21 and the same wait. | Slot 65's wait is satisfied at about 68.35 (DVE131 ends). It starts at 69.40 only because it queues behind slot 64. Pool is free. | **About 0.7-0.9 us.** Slot 64 then binds, ending at about 69.35 plus the DMA. | Low. The output DGE destination is unchanged. The end-of-kernel wait must keep the sem21 `>= 16` wait. Check that no tail relay waits on sem20 or sem21 via an ext field (the tool does not show them). The verifier would catch a lost output. | Move the slot, measure. Expect about -0.8 us. If under 0.3 us, drop it. |
| 3 | **Shorten the block-5 copy.** SP trigger 9 (qSPSpillReload0 block 5, the I-92 copy, 32 KB) takes 2.0 us on the DMA ring (29.02-31.04) and gates PE21, trigger 13, and the Activation wait at 31.45. PE sits idle about 10 us (21.7 -> 31.68). Re-pattern the descriptor in `SP0.json` into larger contiguous runs, or split it into two halves on two queue ids. Also move SP DGEs 10-12 ahead of trigger 9: they wait only on sem4 >= 18 but sit behind trigger 9, which waits for DVE at 28.3. | r09 verified descriptor re-pointing (ch14.2). Evidence is the 2.0 us duration for 32 KB, against 0.2-0.4 us for the 8-16 KB transposes around it. | **About 1-1.5 us, uncertain.** The 31.45 Activation wait may be gated by DGEs 10-12 (ending about 30.4 plus latency), not only block 5. | Medium. A wrong element mapping gives garbage, which the verifier would catch on any input because it is data-independent. A queue split needs a free SpillReload queue id (check `def.json`). | Time block 5 after the descriptor change in the DMA parquet. If it is not under 1 us, close it. |
| 4 | **Offload Activation work that does not touch PSUM to Pool.** Activation is the busiest engine (32.5 us busy). It has 25 non-PSUM TENSOR_SCALAR (mean 374 ns, 9.3 us total) and 8 non-PSUM CAST (mean 296 ns), mostly in the dense 36-50 us stretch. Pool is idle in that stretch (a 17.3 us gap before Pool slot 20). | Playbook ch14.3 item 3: TENSOR_SCALAR_ARITH_OP and CAST exist on Pool. The recipe is to put a semaphore-preserving EVENT_SEMAPHORE on the old engine, insert the op on Pool with the producers' waits, and re-point the consumers' waits. | **Unknown.** Upper bound: the 9.3 + 2.4 us of non-PSUM Activation work, but only chains on the critical path count. Pool per-element speed is unmeasured (its AFFINE_SELECT costs 238-257 ns). | High effort and moderate risk. Many semaphore re-points; a wrong count can pass once and race. The verifier is a weak check on ordering. | Move one TENSOR_SCALAR in the 36-50 us stretch and time that op on Pool. If Pool takes 2x or more, the lever is dead. |
| 5 | **Cleanup: the dead transpose.** SP trigger 36 (block 20) and its SP35 ext wait look unused after this iteration. | Block 20 runs at 53.42-53.62 (200 ns). It is off the critical path. | **Under 0.1 us.** | Low, but only after confirming nothing reads 0x4500-0x45ff. | Grep all engines' address fields for that range. Do it only if bundled with another change. |

## 5. Closed (do not redo)

- **SBUF->DRAM->SBUF staging round-trips and DGE saves on Activation.** Closed by iteration 1: 67.5 -> 61.9 us.
- **Two var9 reload DGEs on SP behind a 2.9 us trigger.** Closed by iteration 2: -0.9 us.
- **Relay waits (SP37-39) and the DGE semaphore renames.** Closed by this iteration (about -1.9 us together with the PE134 retarget).
- **Removing waits on an engine's own completion semaphore** (neff_doc ch14.2 / ch10.5):
  - wrong output on DVE
  - runtime out-of-bounds error on Pool
  - no gain on Activation
- **Lower-precision dtypes.** output_1 is at 83% of the error limit, so there is no numeric headroom.

## 6. Unknown (would change the ranking)

- **What gates the Activation wait at 31.45 us:** block 5 (done 31.04) or DGEs 10-12 (done 30.4)? Settle it by reading the wait fields of Activation slots 39-40 and matching them against the DMA completions. It decides whether candidate 3 is about 0.4 us or about 1.5 us.
- **Whether Pool can read PSUM.** It decides whether PSUM-reading Activation casts (20 CAST, 8 TENSOR_SCALAR) can join candidate 4. Test with one op.
- **Pool's speed for TENSOR_SCALAR at the sizes Activation uses (about 128x128).** One timed op decides candidate 4.
- **Whether the 12 us preamble and 3-5 us tail are inside `total_exec_time`.** Span is about 73 us and exec is about 59.6 us. The fixed preamble and postamble (TENSOR_LOAD, end barriers) look runtime-owned and not editable. Unconfirmed, but nothing in the NEFF points to a lever there.
- **Remaining headroom.** Engine busy is 54% for Activation and 47% for DVE. The rest is dependency and DMA latency. Candidates 1-3 are the three visible latency chains; after them I would expect the remaining NEFF-level gains to be small.

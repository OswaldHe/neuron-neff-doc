# Review of iter-3 (uncommitted `kernel.neff` vs HEAD 400b689)

`notes/` is empty (only the README stub), so the claims below are checked against the commit messages and the NEFF itself.
Baseline is 76.05 us. HEAD is 45.0 us by its commit message; I re-measured it at 45.31 us.

## 1. What changed

`neff_diff.py` against `baseline/` shows 12 members changed (6 per subgraph, identical in `sg00` and `sg01`).
I re-ran it against HEAD, which isolates this iteration.

| member (each of sg00, sg01) | slots HEAD -> now | what changed |
|---|---|---|
| `Activation0.bin` | 93 -> 70 | ACT sheds 12 COPY and 11 CAST, down to 9 CAST. `semaphores.py` sem 4 increments go 63 -> 40. |
| `DVE0.bin` | 110 -> 122 | +12 `CAST` on DVE (sem 6 increments 66 -> 78). |
| `PE0.bin` | 80 | 26 slots differ, only `events_semaphore_value` and `events_wait_idx` (4 -> 6). Those matmuls now wait on DVE sem 6 instead of ACT sem 4. |
| `Pool0.bin` | 7 | Slot 4, the state-out DGE DMA: its wait value on DVE sem 6 goes 63 -> 74. |
| `SP0.bin` and `SP0.json` | 10 | One slot differs in `SP0.bin`. In `SP0.json` the input DMAs are re-ordered: `z` is now DMA id 1 and `A_log` is now last (id 7). The descriptors are otherwise the same. |

- **Summary:** the 12 remaining fp32-to-bf16 S->S_bf cast and copy pairs moved from ACT to DVE and were fused.
  Every consumer's semaphore wait was rewritten to match, and the input DMA order was shuffled.
- **Both subgraphs:** both were edited. The slot-level difference between `sg00` and `sg01` is the same as at HEAD (1 slot in each of Activation and DVE, 4 in Pool). Nothing is applied to only one core.
- **Notes:** nothing to compare against, because the notes are empty. This is a process gap; the next agent should write one file per line of attack.

## 2. Is the speedup real?

Yes. It is roughly -9.6%, about 4.3 us.

- **Stored profile:** `profile_latest.json` had median 40.74 us, samples 40.5 to 41.5, spread 2.4%.
- **My re-measure:** one capture gives 41.04 us.
- **HEAD re-measure:** 3 repeats give 45.31 us, spread 1.0% (45.19 to 45.66).
- **Margin:** the gap is about 10%, against a spread of 1 to 2.4%. This is not noise.
- **Side effect of my check:** my profile run overwrote `profile_latest.json` with a single-sample file. The numbers above are the originals.

Per-engine busy time (median repeat) is balanced now:

| engine | busy | share |
|---|---|---|
| DVE | 35.7 us | 87% |
| ACT | 35.5 us | 87% |
| DMA | 19.4 us | 47% |
| PE | 18.9 us | 46% |
| Pool | 10.3 us | 25% |
| SP | 6.2 us | 15% |

## 3. Safety beyond the verifier

- **Semaphore accounting:** `semaphores.py` on both unpacked subgraphs shows `max_wait <= total_inc` on every semaphore, so no wait exceeds its increments.
  - sem 4: 40 of 40. sem 5: 38 of 38. sem 6: 78 of 78.
  - The DMA sems are unchanged at 16 of 16.
  - `sg00` and `sg01` are identical. There is no loosened or reduced wait; the thresholds were renumbered.
  - The rewiring (PE waits moved from ACT to DVE) follows the moved producer.
- **Verify, 5 runs:** `tools/verify.py --runs 5` passes and the 5 outputs are bit-identical. That supports "no flaky race", though not a proof.
- **Error:** output_0 `max_abs_err` is 2.322e-2, 86% of the 0.027 limit. That is identical to the unedited baseline (2.322e-2), so no new error was introduced. The remaining margin is only 14%, and it is a bf16 floor. Any numerics-changing edit (a different activation table, bf16 intermediates) should be assumed to risk it.
  - output_1: 1.633e-3, also identical to baseline.
- **Cores disagreeing:** no. The two streams are structurally the same.
- **Not checked:** WAR (write-after-read) hazards on the private S/S_bf tiles.
  - A fixed input set would not expose a hazard that only shows under different timing.
  - The 5 bit-identical runs are weak evidence only.

## 4. Candidates for the next iteration (ranked)

Timeline facts that bound every candidate. All times are in us relative to the first MATMUL, core `sg00`:

- The head is ACT-bound. ACT runs back-to-back from 0.4 to 10.4 with 4 in-window `ACTIVATION_TABLE_LOAD`s of 1.28 us each, 5.1 us in total. PE is idle until 7.3.
- DVE is dense from 11.3 to 32.9 (TENSOR_TENSOR 52 x 0.33 us).
- Last compute ends at 32.9 (`sg00`) and 31.9 (`sg01`).
- The state-out DMA then runs until about 35.4 (roughly 790 KB per core in about 3.4 us). The kernel ends at about 37.7.
- Pool is idle apart from DMA issue.

### C1. Split the state-out DMA per head group (tail overlap) — rank 1

- **Where:** `sg00/Pool0.bin` and `sg01/Pool0.bin`, slot 4 (`PSEUDO_DMA_DIRECT2D`, sem 27).
- **Change:**
  - Replace it with 3 copies, each with `sem_increment` 16 and the same sem 27.
  - Copy A covers heads 0-5 and waits sem 6 >= 42. Copy B covers heads 6-9, >= 64. Copy C covers heads 10-11, >= 74 (the existing wait).
  - Each copy uses `src_num_elem[0]` = its head count.
  - `src_start` increases by 512 B per head in SBUF. The dst offset (table index 3) increases by 65536 B per head.
  - Raise the final `NOP` wait (slot 6, sem 27) from 16 to 48.
- **Basis:**
  - I decoded the head completion counts myself. The last DVE write to each head's state tile (0x1c000 + 512*h) lands at DVE sem 6 counts [21, 28, 30, 32, 39, 42, 48, 53, 62, 64, 70, 74]. It is identical in `sg00` and `sg01`.
  - Head 11 equals the current wait of 74, which confirms the mapping.
  - Nothing else writes those tiles after that point (Activation has no writes in the range; slot 24's write to head 0 is an intermediate write).
  - Mechanism: `semaphores.py` accounting; the DGE field layout is in the `isa.py` decode.
  - Not tried on the device before, so **inferred**.
- **Gain:** about 1 to 2 us.
- **Ceiling:** 2.6 us on `sg00` (32.9 to 35.5), larger on `sg01`. A residual of roughly 1 us stays for the last chunk's DMA latency. The Pool DGE issue itself costs about 1.25 us per DMA, and it is unknown whether that scales with size.
- **Risk:** low to medium. A too-early threshold reads stale state, and the verifier catches that because output state is compared at 1.6e-3.
  - A threshold that is too late is only slow.
  - Inserting slots changes the Pool member size; check the `Pool0.json` `dma` list is empty (it is) and that the other sems are untouched.
- **Cheapest falsifier (10 min):**
  - Edit only the `sg00` stream into 2 DMAs (heads 0-9 at >= 64, heads 10-11 at >= 74), verify, and profile.
  - If the `Q0` write packets still end at about 35.4 us, the tail is bandwidth- or issue-bound and C1 is dead.
  - Check with `DmaPacket.parquet`, queue `Q0-*`.

### C2. Cut in-window ACT table loads, bit-identical reorder — rank 2

- **Where:** `sg00/Activation0.bin` and `sg01/Activation0.bin`, head slots 1-19.
- **What the sequence is now.** Table sets are `0 exp`, `1 softplus`, `2 rsqrt`, `3 sigmoid`, `4 silu`. The loads are slot 1 (tbl1), 3 (tbl0), 5 (tbl2), 8 (tbl0), 10 (tbl4), 15 (tbl3), 18 (tbl2).
  - Function use: f9 softplus, f7 exp (slots 4 and 9), f30 and f29 (square and rsqrt, slots 6-7), f36 silu, f5 sigmoid, f29.
  - Slots 5 and 8 exist only because the two exp ops (4 and 9) are separated by the square/rsqrt pair (6-7).
- **Changes:**
  1. Move slots 6-7 (square, rsqrt, n=12) after slot 9, or down to the tbl2 block at slot 18-19. This removes the tbl2 load at slot 5 and the tbl0 reload at slot 8.
  2. Pre-issue the sigmoid pair (slots 15-16, tbl3) before the `EVENT_SEMAPHORE` at slot 12, so its load hides in the 4.7 to 5.8 us ACT idle gap.
- **Basis:** `Activation0.json` set order; the activation-id table in `docs/generated/activation_functions.md`. Table loads are in-window at 0.40, 1.92, 7.41 and 8.94 us (1.28 us each).
- **Gain:** 1.3 to 2.6 us. Ceiling 5.1 us, which is all four in-window loads and cannot be reached because 4 sets must still be loaded.
- **Risk:** medium. This is a dependency question. Slots 6-7 wait on sem 24 (the Pool DMA) and slot 9 on DVE; check what produces and consumes each. Reordering changes semaphore counts, so `semaphores.py` must still balance.
  - The verifier would catch a wrong dependency only if the stale data differs.
  - ACT ops keep the same arithmetic, so the output should stay bit-identical.
- **Falsifier:** after the reorder, `semaphores.py` balances and a profile shows ACT's first slot-19 instruction earlier than the current 10.23 us. If the time to the first matmul does not move, the head is bound by DVE (TT1 then CAST), not by ACT.

### C3. Shorten the head DVE then ACT chain by splitting the two 1536-wide ops — rank 3

- **Where:** `sg00/DVE0.bin` and `sg01/DVE0.bin`, the two wide `TENSOR_TENSOR`s. They run at 4.0-5.8 and 5.7-7.4 us, 1.76 us each, and are the HEAD 400b689 collapse of two groups into one 1536-element op. Also the wide ACT `CAST` (slot 13, 5.81-7.28 us).
- **Change:** split each wide op into halves running concurrently on different engines (DVE plus idle Pool for the TTs, ACT plus Pool for the CAST).
- **Basis:** PE first loop matmul starts at 7.33 us, right after that CAST. The chain TT1 -> CAST -> matmul gates the loop start.
- **Gain / ceiling:** unknown. At most about 0.9 us from TT1 and about 0.7 us from the CAST.
- **Risk:** medium to high. It is a cross-engine move, so a new semaphore must be allocated for the Pool half. DVE and Pool may share the SBUF port, so Pool halves may be slow.
- **Falsifier:** first check what TT1 waits on (`evt_wait_time_ns` in the `Instruction` parquet, plus the `SemaphoreUpdate` table). If TT1 is waiting for a DMA or ACT rather than running as the bottleneck, drop it.

### C4. Move the 12 DVE CASTs (2.6 us of DVE) to idle Pool — rank 4

- **Where:** `DVE0.bin` casts (opcode `CAST`, sem 6 sources, 12 per subgraph); the destination is `Pool0.bin`.
- **Why it might work:** DVE is the loop's critical engine (contiguous 11.3 to 32.9 us) and Pool is 25% busy.
- **Risk:** a stream move needs sem 4, 5 and 6 rewiring, because PE waits changed to sem 6 this iteration. Pool throughput on fp32 to bf16 is unknown, and the shared SBUF port may slow DVE.
- **Gain:** up to 2.6 us, only if DVE stays the bottleneck.
- **Falsifier:** retarget just one cast to Pool and profile it. If the Pool cast takes more than 2x a DVE cast (0.22 us), or DVE TT durations rise, drop it.

## Closed (already measured)

- **ACT table-load churn, bulk:** 22 -> 6 loads per core gave 76.0 -> 55.6 us (HEAD df60817). The remaining 7 loads and the in-window 4 are open only as C2.
- **Scale-pointer ACTIVATEs off ACT:** 36 moved to DVE TENSOR_TENSOR with zero-stride broadcast src1, plus 12 merged casts: 55.6 -> 45.0 us (HEAD 400b689). This iteration is the continuation: the casts left on ACT moved too, 45.3 -> 40.7 to 41.0 us.
- **ACT/DVE rebalance:** ACT is now 87% and DVE 87%, balanced. No further ACT-to-DVE or DVE-to-ACT moves are expected to help.

## Unknown (would change the ranking)

- **What gates the first ACTIVATE at -0.26 us.** Its inputs landed by -3.4 us, which leaves about 3 us. The `SemaphoreUpdate` table and `evt_wait_time_ns` on the first ACT and DVE instructions would settle it. If it is a fixed DMA latency, nothing at NEFF level helps; if it is a re-orderable wait, it is the biggest head lever.
- **Whether Pool DGE issue time (about 1.25 us per DMA) scales with transfer size.** This decides C1's real ceiling. Compare one small DGE against a large one.
- **WAR hazard coverage on the private S/S_bf tiles.** Needs a timing-perturbation test: insert a short delay (a NOP) on the consumer stream and re-verify with `--runs 5`.
- **Fixed overhead in the 41.0 us total.** The in-kernel timeline spans about 37.7 us (first matmul to kernel end); the remainder is preamble/postamble that is not editable here. This bounds what any instruction-level work can recover.

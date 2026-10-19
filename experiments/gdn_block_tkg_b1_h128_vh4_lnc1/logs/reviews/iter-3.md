# Review of iter-3 (kernel.neff, gdn_block_tkg_b1_h128_vh4)

Method: `neff_diff.py` of `git HEAD:kernel.neff` (the 83.2 us iter-2 commit) against the working tree, and of
`baseline/` against the working tree. I ran `semaphores.py` on an unpacked copy, read `verify_latest.json`, re-ran
`profile.py --repeats 3 --parquet` (independent measurement), and read `Instruction.parquet`. `.autohelix/notes/iter-3.md`
exists this time and is detailed.

## 1. What changed (HEAD -> working tree) and is it consistent with the notes?

Three members changed, `Activation0.bin` (105 -> 113 slots, the only length change), `Pool0.bin` (24 slots) and
`SP0.bin` (20 slots). No JSON changed.

| Engine | Slots | Change |
|---|---|---|
| Activation | 1-8 (inserted) | 8 `PSEUDO_DMA_DIRECT2D`s moved verbatim from Pool prologue slots 13-20. Only `semaphore` changed (80-87, the unused `qActDynamicHW` set). The queue-guard waits were removed. The rest of the stream shifts by 8. |
| Pool | 13-20 (8 slots) | DMA became `EVENT_SEMAPHORE`: wait on s80-87 >= 16, `SEM_ADD_IMM_COMPLETE` of 16 onto the original Pool semaphore. Consumers are untouched. |
| Pool | 34, 36, 38, 40, 43, 51, 53, 55 | iter-2's `EVENT_SEMAPHORE` became a bf16->fp32 `CAST` (SBUF->SBUF, 128 channels). Same wait, same add. |
| Pool | 50 and 52, plus the order of 1-6 | Vacuous pure-wait ESs became `CAST`s. The merged V3 DMA now issues first. |
| SP | 5-9, 12, 17, 22 (+14, 16) | The DMA became a bare ES `SEM_ADD_IMM_COMPLETE` 16. Pool `CAST`s now do the copy. |
| SP | 1-3 | Rotated. The `qSPIO0` block-0 trigger (no wait) is now first. |
| SP | 39-42, 47-49 | `PSEUDO_DMATRIGGER` waits `WAIT_FOR_SEM_GE_IMM` -> `NONE` (diff shows slots 39, 40 with the sem 58/60 waits removed). |

This matches the seven edits in the notes (e1-e5, e8, e10, e11). The only mismatch is minor: the notes say 13 SP
slots differ, but the diff shows 20 (rotation 3 + 8 ES + 2 `CAST` hosts + the dropped waits). Both Pool (24) and
Activation counts match.

## 2. Is the latency change real?

Yes.

- `profile_latest.json`: median 74.92 us, samples 74.30-75.95, spread 2.2%.
- My independent 3-repeat run gave **74.31 us, spread 0.17%** (74.19 / 74.31 / 74.33).
- The agent's three later runs were 74.61 / 74.54 / 74.91, and its back-to-back A/B ladder is in notes §6.
- The step from 83.2 to 74.6 is about 8.6 us. That is 4-10x the spread, so it is not noise.
- Baseline is 218.17 us (spread 1.3%), so the total is 2.93x.
- Caveat from iter-2: the device drifts by +-3 us over minutes, so only back-to-back A/B comparisons are trustworthy.
  The agent did use A/B comparisons. Per-edit gains of 0.4-0.8 us (e2, e5) are at the noise limit, but the
  summed gain is not.

## 3. Safety beyond the verifier

- **Error:** unchanged. `max_abs_err` is 3.00e-3 / 1.93e-3 / 6.03e-4 (83% / 54% / 17% of the 3.6e-3 limit),
  identical to baseline. Everything is bit-identical and nothing moved toward the limit.
- **`semaphores.py`:** no wait exceeds its supply. Supply above max wait is benign for `>=` waits.
  Sems 29-35 and 42-44 are queued-DMA completions with **zero waiters**.
- **Hang risk:** the agent found that `verify.py` with one run passes a deadlocking edit (e6) and `profile.py` catches
  it. The shipped file ran 5x5 and 3x5 executions here without a hang. That is good evidence, not proof.

**Soft spots (the verifier cannot see these):**

1. **SP 39-42 and 47-49 lost their waits** (e10/e11). The justification is that the queued blocks 5-8 are byte-identical
   duplicates of the SP 27/32/37/38 DGE transposes, and 9-11 write a dead tmp-buf. The queued transfers still run and
   still write `SB+218368..216064`. They now fire **without waiting for their sources to be final**. If a duplicate
   reads its source before the producer has written it, it stores stale data over the correct data after the real
   DMA. On the device SBUF holds the previous run's value, which for a fixed input is the *right* value, so a
   fixed-input verifier passes. Whether the order is safe rests on timing, not synchronisation.
   **Falsifier:** run `verify.py --runs 10` with a *different* random input per run (or alternate two inputs
   back to back). A stale-write race then produces the previous input's data.
2. **Activation 1-8 have no queue-guard waits** (e8). Safe by construction, because act queues 80-87 are used
   exactly once. Low risk.
3. **Pool `CAST`s on s56-s61, s66, s71 wait `>= 16` while SP also bumps those sems.** Already inspected in iter-2;
   consistent.
4. **Own-semaphore ESs on Activation** (sems 72-79 now receive 16-24): over-supply with `>=` waits is fine.

## 4. Where the remaining time is (profile rerun, exec 74.3 us)

Absolute trace times, from `Instruction.parquet`; the body is about 8 -> 82 us.

| Window | What | Bound |
|---|---|---|
| 8-47 | Prologue plus Activation/Vector/PE compute, Pool prologue (now 88% busy in the first 10 us) | mixed, already shortened |
| 49.8-60.4 | **Every engine is idle for ~3 us at 57.4-60.4.** Activation pc 68 waits `S[38]` (`qSPSpillReload0` block 14) for 10.8 us, Vector pc 87 waits `S[37]` (block 13) for ~10 us. Before that, SP issues transposes 53, 56, 58, 60 (49.5-53), with `S[56]>=48` landing at 56.1. | DGE-DMA completion chain |
| 60.4-81.7 | **Activation tail**, 83% busy: 7 `ACT_TABLE_LOAD` (9.0 us of 1.283), alternating `table_sel` 4 and 2 (silu / reciprocal_sqrt), plus ~10 us of `ACTIVATE`/`CAST`/`COPY` at 0.2-0.5 us each. | Activation |
| 81.7-82.5 | Output | fixed |

Busy totals: Activation 45.8 us (61.6%), DMA 24.9, Pool 23.2, DVE 17.6, SP 13.8, PE 7.3. Table loads are 13 x 1.283 =
16.7 us in total.

## Ranked candidates for the next iteration

Ranked by expected gain per unit of risk and effort.

| # | Candidate | Engine / slots | Change | Mechanism and source | Gain and ceiling | Risk | Falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Replace the two serial spill-reload blocks that gate the Activation tail with compute `CAST`s | SP0.json `dma` ids 13 and 14 on `qSPSpillReload0` (`S[37]`, `S[38]`); triggers SP 72-ish; waiters Vector 87 (`S[37]`) and Activation `ACTIVATE` pc 68 (`S[38]`). Host the `CAST`s in vacuous slots: Pool 33, 35, 37, 39, 42, 54 (pure-wait ESs, per notes) or an inserted Activation/Pool slot. | Block 14 = four `cast` descriptors, SB p0 +218368/217600/216832/216064 (256 B bf16) -> SB p32 +213760..215296 (512 B fp32). Do each as a compute `CAST` with immediate address. **The partition is encoded in the address: the profile shows `fp32@0x834500`, and 0x800000 = 32 x 262144, so address = partition x 262144 + byte offset.** That resolves the "unknown" from notes §5.2. Then turn the trigger into a bare ES and keep `S[37]`/`S[38]` at 16 via `SEM_ADD_IMM_COMPLETE`. Do the same for block 13. | Iter-2 Finding B (`SEM_ADD_IMM_COMPLETE`) and iter-3 e1/e4 (Pool `CAST` bf16->fp32 verified, 0.25 us, instant completion). Partition addressing from the trace operands. | Gate chain 57.05 -> 60.4 = **3.3 us**. Ceiling about 3-4 us. | Moderate. **Unverified whether a compute engine can write partition 32 from partition 0** (cross-partition shift). If not, it fails at compile/run time or gives wrong output, and the verifier would catch it. | Build one `CAST` p0 -> p32 on Pool in isolation and run `verify.py`. 5 minutes. If it fails, the candidate dies. |
| 2 | Move the SBUF partition->row transposes off DGE | SP 53, 56, 58, 60 and the SP 18-21 / 27, 32, 37, 38 groups (2-byte element, `dst_step [2,1/256]`, 512 descriptors each) | Replace with a PE identity-matmul or DVE `STREAM_SHUFFLE` transpose plus a PSUM->SBUF copy, then a bare ES for the old semaphore. | PE is 9.9% busy, DVE 23.7%. The 28 existing matmuls show PE works here. The identity tile and PSUM space do not exist yet. | Chain 50.4 -> 56.1 = **5.7 us**, of which DGE issue is about 2.2 us and completion latency about 3.5 us. Ceiling ~5 us. | High: new SBUF/PSUM allocation, which a fixed-input verifier can pass while still racing. Merging is closed (below), so this is the only route. | Spike one transposed column on PE, `verify --runs 10` with varied inputs, then `profile --repeats 3` as the hang detector. |
| 3 | Reorder the Activation tail so same-table work batches across the 4 v-heads | Activation0.bin pc 72-112. Loads at pc 72 (4), 83 (2), 90 (4), 93 (2), 97 (4), 104 (2), 107 (4). | Group the table-4 `ACTIVATE`s (FUNC_36) together and the table-2 ones (FUNC_29/30) together, then delete redundant loads (turn them into ESs). Re-point `S[4]>=n` own-semaphore thresholds and `S[5]`/`S[6]` waits to match the new order. | Notes §5: `COPY`, `CAST` and `READ_ACC` are table-agnostic. Closed levers: a single load cannot be deleted by a field edit. The reorder has never been attempted. | Ceiling **9.0 us** for the 7 tail loads, realistic 3-6 (each batching removes 1.28 us) if the 4 heads' chains are independent. | High: the Activation sequence counts on its *own* semaphore `S[4]`, and removing own-sem waits was refuted before (iter-2 o01). Every reorder shifts all thresholds. | Decode `FUNC_29/30/36` against `act_info.json` to find which table each belongs to, and check whether FUNC_29 following the table-4 load at pc 90-91 is a redundant load. If the pc 90 load is not needed, NOP it and run `verify`: a drop of ~1.3 us or it dies. |
| 4 | Hoist the free tail work earlier or off Activation | Activation `CAST`/`COPY` at pc 71, 73, 74, 76, 79, 85, 92, 95, 96, 98, 102, 105, 109 | Move immediate-addressed ones to Pool or DVE. Register-addressed ones (`regnum=8`) cannot move. | Slot insertion works (iter-3 §0), and Pool `CAST` is byte-copyable from Activation. | 13 ops x ~0.28 = **3.6 us** of Activation time, only the critical part counts. Depends on candidate 3: if loads are batched, this removes real time. | Moderate. | Count how many of the 13 have immediate addressing. If <6, drop it. |
| 5 | Close the stale-write hole from e10/e11 | SP 39-42, 47-49 | Put back `WAIT_FOR_SEM_GE_IMM` on the *producers'* semaphore (not the old chained one) so the duplicate queued writes cannot precede their sources. | iter-2 review "cheap hardening" pattern. | About 0 us if the sources are ready by then. Costs time if not. | Low. This is a correctness fix, not a speedup. | Varied-input `verify --runs 10` before and after. |

Ranking logic: 1 has the shortest, cheapest test and a well-measured 3.3 us gate. 2 has a bigger ceiling but needs new
state. 3 has the biggest ceiling but needs a large rewiring of own-semaphore waits. 4 is dependent. 5 is hygiene, and
should come before the next edit builds on e10/e11.

## Closed (do not redo)

- **Six DRAM scratch round-trips** (iter-1): 215.2 -> 95.3 us.
- **Duplicate DGE DMAs, signal-only dummies, and the Activation per-partition DMAs** (iter-2): 95.3 -> 83.2 us.
- **V3 `conv_weight` merge:** 8 tiles into 1 (2 KB tiles), done in iter-2: -3 us.
- **Merging 32 KB tiles** (V5 `W_in_z` +1.5 us, V0 `W_in_qkv` +0.9 us): regressions. Merging pays only for small tiles.
- **V0 / V2 4- and 8-tile merges:** give wrong numbers (pair B specifically).
- **Merging transposes that already sit on distinct queues** (SP 27, 32, 37, 38 -> 1): +2.3 us (80.4 vs 78.1). SP 18-21
  merge **hangs** (passes `verify`, fails `profile`).
- **Moving Pool 27-32 off Pool:** worth nothing, they are off the critical path (the critical path leaves Pool at slot 6).
- **Moving more than 8 prologue DMAs to Activation** (eA, only 4 moved): +0.1 us regression (78.4 vs 78.3), and the
  DMA block at Activation slot 4 (e9): +3 us regression.
- **`ACT_TABLE_LOAD` field edits:** no set holds both silu and rsqrt (21-entry catalogue). Only a reorder can help.
- **Output-0 tail bypass:** slower (86.7 -> 88.8). The ~3 us output round trip is slack (bound is Pool 66's 256 KB write).
- **Removing own-semaphore waits** on DVE/Pool/Activation: refuted in iter-2.
- **PE `perf_opt`, PE/DVE arithmetic:** not bottlenecks (PE 9.9%, DVE 23.7%).

## Unknown (would change the ranking)

1. **Can a compute `CAST` read partition 0 and write partition 32?** It sets whether candidate 1 exists at all.
   Measure with the 5-minute spike above.
2. **Is the e10/e11 ordering safe with varied inputs?** Run `verify.py` with a different input each run. It decides
   whether candidate 5 is urgent and whether the next edit may build on this file.
3. **Which table do FUNC_29, FUNC_30 and FUNC_36 belong to?** Decode with `act_info.json`. It sets the real ceiling of
   candidate 3, and may show some of the 7 tail loads are already redundant.
4. **How many of the 13 tail `CAST`/`COPY` slots have immediate addressing?** It sets candidate 4.
5. **The floor.** The runtime prologue/epilogue and the output store after the last `ACTIVATE` (81.7 -> 82.5) are not
   editable. The compute floor is not measured. The distance from 74.3 us to roughly 60 us looks reachable only through
   candidates 1 and 3 together, so I would stop after those two if neither moves the median by >2 us.

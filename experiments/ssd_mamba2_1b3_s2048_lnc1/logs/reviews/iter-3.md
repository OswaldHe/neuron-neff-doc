# Review of iter-3 (working-tree `kernel.neff` vs HEAD 038289d)

Baseline NEFF = 7154.9 us. HEAD = 5127.2 us (re-measured by me, 3 repeats). Working tree = 5106.8 us.
`notes/README.md` is still the empty template. Nothing was recorded for this iteration.

## 1. What changed

`neff_diff.py baseline/kernel.neff kernel.neff` shows the cumulative change since the baseline, mostly the earlier ACT work.
Diffing against the HEAD NEFF (`git show HEAD:kernel.neff`) shows this iteration's edit.

- One member changed: `sg00/DVE0.bin`. It still has 24525 slots, so nothing was inserted or deleted. 2180 slots changed.
- 1635 slots went `STREAM_SHUFFLE` -> `EVENT_SEMAPHORE` with no wait and no update. These are NOPs. All of them had `events_wait_mode` and `events_update_mode` NONE before.
- 545 slots went `STREAM_SHUFFLE` -> `COPY`.
  - The COPY is 128 channels and 1 element, from SBUF 0x9208 to 0x9918.
  - It waits `sem6 >= N`, with N = 11, 20, 27, 35, ... It updates sem6 with SEM_INC_COMPLETE.
  - The original sequence was 4 shuffles of 32 channels each. They wrote 0x9918 into partition quadrants 0, 32, 64 and 96. Only the 4th carried `SEM_INC(sem6)`.
  - The edit turns the first 3 into NOPs. It turns the 4th into one 128-partition COPY that carries the inc.
  - So it replaces each "quad-shuffle broadcast into 0x9918" with a plain copy of the 0x9208 tile.
- That is only valid if 0x9208 already holds the same value in every partition. DVE writes it earlier with its own `STREAM_SHUFFLE` (for example slot 103 -> dst 0x9208).
- The notes do not describe any of this.
- HEAD's commit message is about merging ACT ACTIVATEs and says to target semaphore-handoff latency. The actual edit is on DVE and is unrelated to that message. There is no claim to compare against.

## 2. Is -20 us real?

| NEFF | median us | min-max | spread |
|---|---|---|---|
| HEAD (my 3 repeats) | 5127.2 | | |
| working tree (`profile_latest.json`) | 5106.8 | 5101.7-5108.6 | 0.135% (~7 us) |
| working tree (my 1-repeat parquet run) | 5110.0 | | |

- The delta is -20.4 us (-0.40%). That is about 3 times the spread, so it is real, though small.
- Spread was 0.021% at the baseline and is 0.135% now. The ~7 us floor is the noise to beat.
- Whole-run effect: 7154.9 -> 5106.8 is -28.6%. Wall time follows the semaphore ring, not any one engine's busy time.
- DVE busy fell 2700 -> 2441 us (-260 us), but wall fell only 20 us. The conversion rate is about 0.08 us of wall per us of busy time removed.
- The previous iteration saw the same effect on ACT: -97 us busy gave -1.6 us wall.

## 3. Safety

- **Sync loosening: none.**
  - Total sem6 increments are unchanged at 6791 and `max_wait` is unchanged at 6791 (`semaphores.py` on both unpacked NEFFs).
  - Waiters on sem6 grew 8419 -> 8964 (+545, the new COPY waits). The new waits are tighter, not looser.
  - The 1635 NOPs never waited or updated, and 472 of the old shuffles that incremented sem6 became COPYs that increment it.
  - The `semaphores.py` table, which is the only count of sem6 increments, therefore does not change.
- **Data-dependent, not sync-dependent.**
  - The COPY reads 0x9208. Program order on DVE guarantees the earlier DVE write has completed.
  - Correctness rests on the claim that 0x9208 is identical in every partition to what the quad-shuffle produced, or that 0x9918 is only read at that point.
  - I confirmed only that the verifier passes.
  - The verifier uses one fixed input set. A value-dependent equality, such as a per-head scalar that happens to be broadcast, would pass here.
  - The shuffle -> COPY step is bit-exact, so it should not change the numbers. The error has not moved because of it.
- **Error headroom.** Output 0 is at 5.234e-07 of 6.2e-07 (84%), up from 5.122e-07 (83%) at the baseline. Output 1 is at 3.334e-07 (54%), up from 3.166e-07 (51%).
  - This iteration's op is exact, so the drift comes from earlier iterations (the `exp(-x)` rewrite).
  - Output 0 has about 16% headroom left. An arithmetic-changing candidate can use that headroom, but not much of it.

## 4. Ranked candidates for iter-4

Measured conversion rates, used for every ceiling below:
- Wall gain per us of DVE busy removed: 20 / 260 = 0.08.
- Wall gain per us of ACT busy removed: 1.6 / 97 = 0.016.

| # | Edit | Engine / member | Mechanism and evidence | Ceiling | Risk / verifier | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Convert the remaining quad-shuffle groups the same way. | DVE0.bin. 6012 `STREAM_SHUFFLE` slots remain, at 147 ns each (885 us busy per `Instruction.parquet`). They form ~1500 groups of 4 (32 channels, same src, dst quadrants 0/32/64/96). | Same transform as iter-3. Replace each group by 1 COPY of 128 channels, from the tile that already holds the broadcast, to the group's dst, carrying the sem6 inc. Established on this device by this iteration: verifier passes and wall falls. | If all are convertible, 885 us x 0.08 = ~70 us at most. Realistically 20-40 us, because the first pass probably took the easy groups (src already in every partition). | Value equality is the risk. The verifier uses one input and does not catch it for a broadcast that happens to be uniform. | Convert 50 groups whose src was itself written by a shuffle or broadcast. Compare their src and dst bytes on the golden run, and run `verify.py` with a different random seed. If a group's src is a partition-specific value, drop it. |
| 2 | Delete the 1635 NOP slots (compaction) instead of leaving them. | DVE0.bin. 1635 `EVENT_SEMAPHORE` slots with no wait or update. | An earlier iteration already deleted ACT slots (the 1020 dead slots). The NOP issues cost ~64 ns each on DVE. | ~100 us of DVE busy x 0.08 = ~8 us, about 1 spread. Marginal. | Compaction must keep any slot-index references valid (jumps, `PSEUDO_*`, DMA descriptor links). The verifier catches a wrong jump. | Count how many streams reference slot indices. If any branch lands inside a deleted range, drop this. Otherwise delete and profile. Needs 3+ repeats to see 8 us. |
| 3 | Move the 1088 DMA-trigger `DMA_DIRECT2D` instructions off the ACT queue. | ACT0.bin, then `dge:Activation` entries in Activation0.json. 615 us of ACT in-order time (565 ns each). SP is only 15.7% busy. | The triggers sit in the ACT stream between ACTIVATEs and delay the PE -> ACT -> PE ring directly. | Unknown. ACT throughput is a closed lever, but this removes 1088 serial stalls in the loop rather than shrinking busy time. | High. Changes DMA-queue ownership (JSON and semaphores 40-55). The sem accounting will catch a mismatch. The verifier may not. | Count whether the ACT triggers wait on a sem that ACT itself increments. If so, the move is not free. Do not start before checking the sem table. |
| 4 | Split the 445 ns PSUM-evacuation `ACTIVATE COPY(scale/bias)` between ACT and DVE. | ACT0.bin. 6865 slots, 3054 us busy. | Shortens the PE -> ACT latency of each evac. The heuristic attribution shows PE waiting on ACT for ~1.8 ms of gaps. | Unknown. ACT busy does not convert on its own. This only helps if the split cuts per-evac latency. | Medium. DVE has room (47.8% busy). A bad split is exact-math safe if only the engine changes. | Convert 10% of the evacs to DVE `TENSOR_SCALAR`. If wall is flat while ACT busy drops, the ring is not latency-bound on those instructions. Kill it. |

I would run #1 first. It is the only candidate with a measured gain per unit. #2 is a small cleanup on top of it. #3 and #4 are hypotheses that need a measurement before any edit.

## Closed (do not redo)

- ACT activation-table reloads: 2040 of 2046 removed in iter 1, 7154.9 -> 5130 us.
- ACT ACTIVATE merging: 303 pairs, -97 us ACT busy, -1.6 us wall. ACT throughput is not the bottleneck.
- More DVE busy reduction by shuffle removal converts at only ~0.08.

## Unknown (measure first)

- **Which shuffle sources hold the per-partition value.** Settle it by dumping the golden-run tile at 0x9208 and at each source, then comparing across partitions.
- **True critical path.** Settle it by tracing semaphore edges (who increments what, per 1024 units). My gap attribution is heuristic: PE waits mostly on ACT, ACT mostly on DVE, DVE on PE and ACT.
- **Per-unit chain latency.** Profile a single (head, chunk) unit and sum handoff gaps. Wall is ~5 us per unit; handoffs are the unknown share.
- **Noise floor of a 3-repeat profile.** It is ~7 us. Anything under ~10 us needs 5+ repeats on both NEFFs to accept.

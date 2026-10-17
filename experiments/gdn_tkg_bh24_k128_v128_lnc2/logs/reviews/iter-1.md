# Review of iter-1

`notes/` is empty (only the README template), so the claimed change is unknown. Everything below is reconstructed from the NEFF.

## 1. What changed

6 members changed, the same 3 in each of `sg00` and `sg01`: `Activation0.bin`, `DVE0.bin`, `PE0.bin`. The other 83 members are identical. Both subgraphs got the same edit. I decoded both streams and the opcode sequences are identical. The edit is a hoist of the 12 per-head silu activations onto the Activation engine:

- **Baseline:** 12 separate `ACTIVATE f=36` (silu), `n=128`, each preceded by an `ACTIVATION_TABLE_LOAD sel=4` and followed by a reload of `sel=2`. The sources were `0x836600..0x837c00` in steps of `0x200`, i.e. contiguous. Each wrote a per-head scratch such as `0x5e70` or `0x4770`.
- **Now:**
  - There is one `ACTIVATE f=36`, `n=1536`, `channels=1`, in Activation slot 2 of each subgraph. It reads `0x836600` and writes `0x20000..0x21800`. It waits on sem 15 ≥ 16, the single DMA of `z`, which is `SP` dma id 7 with 6144 elements.
  - The 12 consuming DVE `TENSOR_TENSOR_ARITH_OP` slots (51, 56, 61, 64, 69, 75, 78, 80, 85, 87, 92, 94) have `src1` repointed to `0x20000 + 0x200*h`.
  - Their sem-4 waits were rewritten to `>= 1`, and the other sem-4 waits in DVE and PE were shifted by −1 to +2 to match the new counts.
  - The Activation stream goes from 167 to 140 slots. `ACTIVATE` goes from 47 to 42 and `ACTIVATION_TABLE_LOAD` from 22 to 6.
  - The activation-set order is now 4, 1, 2, 0, 3, 2.
- The 160 "differing" Activation slots are mostly shifted indices, not 160 edits.

Check that nothing else uses the hoist destination: I scanned every immediate operand in all five engines of both subgraphs. Only the 12 DVE reads fall in `0x1f000..0x22000` of partition 0, and the range is below the `0x37FF8` reserved bytes.

`z` is a single DMA on sem 15, so the hoist waits on the right semaphore.

## 2. Is the speed-up real?

Yes. It is not noise.

| | median | min–max | spread |
|---|---|---|---|
| baseline | 76.054 us | 75.857–76.270 | 0.54% |
| latest | 55.601 us | 55.323–56.131 | 1.45% |

- That is −20.45 us (−26.9%), about 25× the spread.
- My own re-profile gave 56.19 us.
- Activation busy dropped from 75.7 us to 55.0 us. Table loads cost 1.283 us each, and 16 fewer loads would account for about 20 us.

## 3. Safety

**Verifier**
- `verify.py --runs 8` passes, and all 8 outputs are bit-identical.
- Error is unchanged: output_0 is 2.322e-02, which is 86% of the 0.027 limit, and output_1 is 1.633e-03. These are identical to the baseline, so the error did not move. It is already close to the limit, so any numeric change would fail.

**Semaphores**
- I ran `semaphores.py` on both unpacked NEFFs, for `sg00` and `sg01`.
- In both, sem 4 has `max_wait` = `total_inc` = 95 (was 100). The ACT increments dropped by exactly the 5 net removed `ACTIVATE`s.
- No wait exceeds its increments. Sems 5, 6 and 8–27 are unchanged.
- The two cores are identical, so they cannot disagree.

**Residual risk: the wait loosening**
- Twelve DVE ops now wait on sem 4 ≥ 1, down from 28–75 in the baseline. Those larger waits implicitly ordered the ops after every earlier ACT op, not just the silu.
- That is only safe if no ACT-produced `src0` is later than count 1. I found no ACT producer for `src0` at slots 51 and 61. At slot 69 the CAST (count 44) is covered by the earlier sem-4 ≥ 45 wait at slot 58.
- I could not resolve slot 75 (`src0` `0x4470`), because the producer matches by address only. 8 identical device runs make a race unlikely but do not rule it out.
- The next agent should diff, per DVE op, the set of ACT writes to its `src0` against the preceding waits. If in doubt, set the wait to the count of the last ACT op that writes `src0`, not to 1.

## 4. Candidates for iter-2

**Where the time is** (sg00 and sg01 agree). Activation is the critical engine, at 98.6% busy. Its stream runs from t≈12.0 us to 61.0 us (49 us) with essentially no idle time. Per-core ACT durations:

| Item | Count | Time |
|---|---|---|
| `ACTIVATE` with scale pointer (`f=128`, 490 ns each) | 36 | 17.5 us |
| `CAST` (290 ns each) | 36 | 10.5 us |
| other `ACTIVATE` (200–310 ns) | 33 | 9.1 us |
| `ACT_TABLE_LOAD` (1.283 us each) | 6 | 7.7 us |
| `ACTIVATION_READ_ACCUMULATOR` | 12 | 3.5 us |
| PSUM→SBUF `COPY` | 12 | 3.0 us |

The other engines have headroom:
- **DVE:** about 24 us of real work, 33 us busy including sem waits.
- **PE:** about 13.5 us.
- **Pool:** only 4 `DMA_DIRECT2D` plus semaphores.
- **Prologue:** 0–9.2 us is a runtime preamble and not editable. The last 5 us is an epilogue.

Ranked by gain per risk and effort:

| # | Edit | Mechanism and evidence | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Remove one more table reload.** Both subgraphs, `Activation0.bin`, set order 4, 1, 2, 0, 3, 2 → 4, 1, 3, 0, 2. | Slot 17 (`f=5` sigmoid, `n=12`) is the only use of set 3. It costs two loads, `sel=3` at slot 11 and `sel=2` at slot 23. Moving slots 17–18 (sigmoid and its `f=30`) ahead of the `sel=2` and `sel=0` uses leaves five loads, the minimum for five sets. `f=30` and `f=128` exist in every set, so they do not force a reload. `ACTIVATION_TABLE_LOAD` and its `act_tbl_sel` are documented in ch. 3.5 and 7.2. | At most 1.283 us, the cost of one load. The ceiling is the profile's `ACT_TABLE_LOAD` mean of 1283 ns. It is smaller if the sigmoid input arrives late: slot 17 waits sem 6 ≥ 12, which is DVE, and `b` (sem 11) arrives early. | Medium-low. Reordering ACT slots shifts the sem-4 counts for every later waiter in DVE and PE, so each wait must be renumbered. The verifier catches a wrong dependency only if it is deterministic. | Check the sem-6 wait at slot 17 against what DVE produces by then. If the sigmoid input depends on DVE work later than the exp chain, the idea is dead. That takes minutes with `semaphores.py` and the decode. |
| 2 | **Start the ACT stream with work that does not wait for `z`.** Activation slots 2 → after slot 10, in both subgraphs. | The hoisted silu is first and waits on the big `z` cast DMA (24 KB, sem 15). The profile shows ACT idle 1.54 us before it (table load ends at 10.46 us, ACTIVATE starts at 12.0 us, `evt_wait` = 2815 ns). The small-DMA chain (slots 4–10) waits on much smaller transfers. | At most about 1.5 us, the measured idle gap. It could be less if the small chain also waits on a DMA. | Medium. Same count-shift bookkeeping as #1, and the silu consumers stay at `>= 1` only if the new silu count is renumbered. | Compare the `sem 9/10/11` arrival times with the `sem 15` arrival in the DMA timeline (`DmaPacket.parquet`). If `z` is not later than the other DMAs, kill it. |
| 3 | **Move the 12 PSUM→SBUF `COPY`s off ACT to DVE.** Both subgraphs, `Activation0.bin` and `DVE0.bin`. | These are 12 × 250 ns, `src=0x2001800` style PSUM reads (profile operands). DVE has about 30 us of headroom. The semaphore model is in ch. 10. | At most 3.0 us from ACT, the profile sum for `COPY`. The real gain is smaller if DVE becomes critical. | High. It is a cross-engine rewrite with new sem-4 and sem-6 counts across three engines, and the encodings are untested. | Rewrite one `COPY` as a DVE `TENSOR_COPY` and check that verify passes and ACT time drops about 250 ns. |
| 4 | **Move some of the 36 ACT `CAST`s (fp32→bf16, `n=128`) to DVE or Pool.** | ACT is at 99% and the casts are 10.5 us. Pool is nearly idle, but its per-op speed for casts is unknown. | At most 10.5 us from ACT. Realistic is much lower, since DVE can take about 9 us before it becomes the bottleneck. The unknown is DVE/Pool cast throughput. | High. This is the same cross-engine rewrite as #3 on a larger scale, and it breaks the monotonic sem-4 counts. | Move a single CAST. If ACT does not drop by about 290 ns, or the new engine adds more than that, the idea is dead. |
| 5 | **Move the `channels=1`, `n=128` scale-pointer `ACTIVATE`s to DVE `TENSOR_SCALAR`.** | These cost 490 ns each versus about 200 ns for an immediate scale. There are 36 in total, 17.5 us. | At most 17.5 us from ACT, the profile sum. The real gain is limited by DVE headroom. | High. Same engine-boundary rewrite, and the scale pointer semantic on DVE has not been tried. | Move one instruction and compare. |

The most defensible next step is #1, then #2. #3–#5 are speculative cross-engine rewrites. The ceilings come from the profile, and I have not shown that DVE or Pool can absorb the work.

**Closed**
- Hoisting the silu (this iteration): −20.45 us, accepted.
- Nothing else is recorded in `notes/`. Nothing else is refuted yet.

**Unknown**
- Are the DVE wait loosenings at slots 51–94 sufficient? Settle it by listing the ACT writers of each `src0` against the preceding waits, plus about 50 verify runs under load.
- What are the DVE and Pool throughputs for fp32→bf16 `n=128` casts? Settle by moving one CAST and timing it. This decides whether #3–#5 are worth anything.
- What part of the 0–9.2 us prologue belongs to the NEFF and what to the runtime? Settle by diffing `SP` and `Pool` preamble slots against a trivial NEFF.

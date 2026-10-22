# Review of iter-5

## 1. What changed

Compared with the committed HEAD (34a8f97, 3071.8 us). Only `sg00/Activation0.bin` and `sg00/DVE0.bin` differ, 524 slots each. The other 35 members are identical. The slot count is unchanged: 10493 ACT slots and 15169 DVE slots, so nothing moved.

- **ACT:** 435 or more `ACTIVATE` slots (a COPY with scale, PSUM→SBUF) became `EVENT_SEMAPHORE`.
  - Each keeps its original wait word (wait mode, wait index, wait value).
  - Each now carries the `SEM_INC` on semaphore 4, so ACT's own increment is preserved.
  - Semaphore 4 stays at 5928 increments against a max wait of 5928: 5127 ACTIVATE + 366 COPY + 435 EVENT_SEMAPHORE.
- **DVE:** the paired `TENSOR_TENSOR` slots changed from `op ADD` to `op MULT`, with src0 repointed (for example 0x4218→0x8390, 0x8118→0x4688) and `src0 step_elem` 1→0.
  - The step-0 src0 is a per-partition scalar broadcast.
  - So the ACT scale-copy was folded into the DVE op that consumed it.
- **Notes:** `notes/README.md` is only the template, so there is nothing to compare the change against. The commit message is not available either. My reading is an ACT→DVE offload of the scale-copy, and the diff is consistent with that.
- **Not verified:** I did not prove that `ADD(x, s*y)` equals the new `MULT(s, ...)` numerically. The notes should say what identity was used.

## 2. Is the latency change real?

- Median 2987.6 us, five samples from 2986.2 to 2988.5, spread 0.076%. A separate run of `profile.py` gave 2987.3 us.
- Against 3071.8 us that is −84 us (−2.7%), more than 1000× the spread. It is real.
- Against baseline 5248.9 us, the total is now 1.757×.
- The loop is not chasing noise. A 0.5 us win would still be measurable.

## 3. Safety

- **Semaphores:** `semaphores.py` on both unpacked NEFFs shows no "wait exceeds increments" flag.
  - The only diff is the ACT source mix on semaphore 4; the totals are unchanged.
  - No wait was removed or reduced. The new `EVENT_SEMAPHORE` keeps the old wait word and the old increment.
  - **Residual race:** I did not check that DVE's multiply still waits on ACT's former producer and on the PSUM read. A DVE wait that ACT's output used to satisfy transitively could now be missing.
- **Error is creeping up.** `max_abs_err` limit is 2.3e-7.

  | Output | Baseline | Now |
  |---|---|---|
  | output_0 | 1.88e-7 (82%) | 2.06e-7 (89%) |
  | output_1 | 1.19e-7 (52%) | 1.99e-7 (87%) |

  output_1 gained 35 points over the whole loop. Another rounding-changing edit could fail the verifier. Treat error headroom as a budget of about 10%.
- The verifier uses one fixed input set, and the 524 edits are mechanical, so the diff is low risk for ordering. The numeric change comes from a different rounding order.

## 4. Candidates for the next iteration

Where the time goes, from `Instruction.parquet` (total 3001 us):

| Engine | Busy | Gaps |
|---|---|---|
| ACT | 2359 us | 1160 us |
| PE | 3492 us summed (overlaps; `profile.py` says 1270 us) | – |
| DVE | 1857 us | – |
| Pool | 1582 us | – |
| SP | 688 us | – |

- ACT is still the busiest engine and runs about 80% busy per 500 us window.
- ACT instruction counts: 7168 ACTIVATE (4852 `COPY(scale/bias)` at 330 ns average = 1.60 ms; 2296 SOFTPLUS = 0.41 ms), 2858 `EVENT_SEMAPHORE` (0.20 ms), 91 DMA_DIRECT2D, 17 table loads (22 us).
- Pool and DVE are 52–55% busy, so there is room to take ACT work.

| # | Candidate | Slots / change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Finish the ACT→DVE scale-copy fold | `Activation0` `ACTIVATE` COPY-with-scale (4852 − 435 ≈ 4400 left). Do the ones whose consumer is a DVE `TENSOR_TENSOR` ADD. Same recipe as this iteration: ACT slot → `EVENT_SEMAPHORE` keeping wait + inc, DVE `ADD`→`MULT` with scalar src0 step 0. | Established this iteration on-device: 435 units gave −84 us, about 0.19 us per unit. | The ACT COPY pool is 1.6 ms of 2.36 ms. DVE has about 1.1 ms of slack (3.0 ms − 1.86 ms) but is also fed by the fold. Realistic ceiling about 200–300 us if the critical path stays on ACT; unknown beyond that. | Rounding. output_1 is at 87% of the limit. Wrong DVE waits would race silently. | Apply to 50 units and check `max_abs_err` per output before scaling up. Check which consumers are actually ADD before touching the rest. |
| 2 | Move SOFTPLUS or COPY work to Pool (GpSimd) | Pool is 52% busy and idles 1575 us. Candidates are ACT COPYs with no PSUM source (SBUF→SBUF scale copies, the 366 `COPY` and the 3040 ACTIVATEs at 320 ns). | Pool can run elementwise ops with per-partition scalars; its table-free ALU avoids ACT's 1283 ns table reloads. | Same ACT pool as #1; shares the ceiling with it. Pool DMA descriptor generation is on the same queue (2385 DMA_DIRECT2D × 655 ns = 1.56 ms), so Pool may be near its real limit. **Unknown.** | Slower per instruction than ACT. Pool already runs the DGE work, so added instructions can delay DMA issue. | Move 20 units and look at Pool start/end gaps in the parquet. |
| 3 | Remove ACT `EVENT_SEMAPHORE`s that are now no-ops | 2858 ACT `EVENT_SEMAPHORE`s (0.20 ms). The 435 new ones are pure semaphore bumps standing in for deleted work. Where the next ACT instruction already carries the same increment, merge it into that instruction. | A semaphore bump on a neighbouring instruction replaces a standalone one; same trick as `SEM_ADD_IMM` used in commits 1db4928 and 34a8f97. | At most about 0.07 us × 435 ≈ 30 us. | Loosening a wait or splitting an increment can pass once and race later. | Check `semaphores.py` totals are unchanged and every waiter still sees its value. |
| 4 | Remaining state_sb bounces (46 of the 166 left) | `SP` `DMA_DIRECT2D`, 816 × 536 ns = 437 us total. | Continue commit 34a8f97's single-hop rewrite. | At most about 46 × 0.5 us = 25 us. SP is only 19.6% busy; this gains only if it sits on the critical path. | Liveness gate must still hold for writes in flight. | Gate-check each unit before editing; stop if any fail. |

Candidates 1–2 are the only ones whose ceilings are large. Candidates 3–4 are small clean-ups.

**Closed**

- ACT `imm` pointer operand: disabled on all 6144 ACTIVATEs (200d241, −8.3%).
- 1283 ns table reloads: 17 left, 22 us total. Nothing more to gain beyond about 20 us.
- HBM spill/reload: 368 KB now against 6.7 MB at baseline.
- ACT scratch copies and barriers: 621 deleted in 1ed7001.
- DMA count: 1149 transfers, unchanged and not the bottleneck (DMA busy 32%).

**Unknown**

- Which engine is on the critical path. Gap analysis shows every engine idle 1.2–2.3 ms. A dependency graph over `Flow.parquet` or `SemaphoreUpdate.parquet` would settle whether ACT work is actually serial.
- Whether Pool can take more work. Compare Pool timeline gaps against DGE descriptor issue times.
- How much rounding the DVE MULT introduces. Run the verifier with the 435 new units reverted in two halves to see which half moved output_1.

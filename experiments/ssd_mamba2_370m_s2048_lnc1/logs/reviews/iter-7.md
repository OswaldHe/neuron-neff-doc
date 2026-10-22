# Review of iter-7 (kernel.neff vs HEAD 73e7538, and vs baseline)

## 1. What changed

`neff_diff.py` baseline -> kernel.neff: 4 members changed (Activation0, DVE0, PE0, SP0), the same 4 as in earlier iterations. The step from HEAD (73e7538) to this iteration touches only two members:

| member | slots HEAD -> now | slots differing |
|---|---|---|
| `sg00/DVE0.bin` | 15169 -> 13498 (-1671) | 14493 (mostly a shift from deleting slots) |
| `sg00/Activation0.bin` | 10493 -> 10493 | 649 |

- **DVE:** the opcode census from `semaphores.py` for sem 6 (DVE's increment source) went from `STREAM_SHUFFLE` 785, `TENSOR_TENSOR_ARITH_OP` 2630, `MEMSET` 702 to `STREAM_SHUFFLE` 625, `TENSOR_TENSOR_ARITH_OP` 2782, `MEMSET` 709, `SCAN` 618 (was 617).
  - That is 160 shuffles removed, 152 TT_ARITH added, 7 MEMSET added and 1 SCAN added.
  - It reads as the 160-shuffle group being re-expressed as DVE arithmetic, with their sem-6 increments preserved. Sem 6 total_inc stays 4846.
- **Activation:** 649 slots differ in place. Total ACT increments on sem 4 are unchanged at 5928. Slot decodes show ACTIVATE <-> EVENT_SEMAPHORE and PSEUDO_DMA swaps, plus `dst_step_elem` and address edits.
- **PE and SP:** untouched this iteration.
- **Notes:** `notes/README.md` is only the template, so the agent wrote no notes. The change cannot be checked against a stated claim. The profile numbers say it is a ~1.5% DVE/ACT trim: 2950.5 -> ~2905 us.

## 2. Is the latency change real?

- `profile_latest.json` has median 2905.247 us. The samples span 2904.52 to 2906.45, a spread of 0.066%. The baseline is 5248.9 us with 0.022% spread.
- The step from the previous commit, 2950.5 -> 2905.2 (-45 us, -1.5%), is about 20 times the spread.
- I re-profiled and got 2904.02 us, which agrees.
- The win is real, not noise. Cumulatively it is -44.7% vs baseline.
- Engine busy vs baseline:

| engine | baseline | now |
|---|---|---|
| ACT | 4722 us | 1809 us |
| DVE | 1689 us | 1478 us |
| DMA | 1193 us | 973 us |
| PE | 1255 us | 1268 us |

## 3. Safety beyond the verifier

- **Error:** `max_abs_err` is 2.058e-07 on output_0 and 1.993e-07 on output_1, against a limit of 2.3e-07. That is 89% and 87% of the limit.
  - The headroom is only ~11%. The agent's earlier commits claim bit-exact outputs; this run is not obviously bit-exact. The error is close enough to the limit that any further non-bit-exact change (reassociating a sum, a different `exp` formulation) risks failing.
  - Check bit-exactness vs the previous commit before accepting it. `verify.py` uses a single fixed input set.
- **Semaphores:** I unpacked HEAD and now and diffed `semaphores.py`. The only differing line is sem 6's source census. For every semaphore, `max_wait == total_inc` is unchanged, and waiters are identical (6093 on sem 6, 7311 on sem 4). No wait was removed or lowered, and no increment total changed.
  - What `semaphores.py` cannot show is whether each of the 152 new TT_ARITH ops still orders correctly against its readers. Each one carries the shuffle's old semaphore increment, so ordering by count is kept.
  - The remaining risk is that an in-place rewrite changes the real latency or hazard window for a consumer that waits on the same count. The earlier iterations already hit this: the 669f7b0 message describes silent state corruption when an increment fired while the ACTIVATE was still writing.
- **Not verified, so treat as open:** a multi-input check (different seeds). `bench/` has one input set.

## 4. Where the time is, and ranked candidates

### What the profile says

Per-engine busy time (sum of instruction durations, from `/tmp/profrev/Instruction.parquet`, 1 repeat) against a ~2919 us span:

| engine | busy | idle gap |
|---|---|---|
| ACT | 2197 us | 1213 us |
| DVE | 1702 us | 1653 us |
| GpSimd | 1583 us (all `DMA_DIRECT2D`) | 1491 us |
| PE | 3493 us summed LDWEIGHTS+MATMUL (overlaps; `profile.py` reports 1268 busy) | 1582 us |

The kernel is a latency-bound ACT -> PE -> DVE -> ACT dependency ring per (head, chunk) unit, about 3.78 us per unit over 768 units.

Where each engine's idle gap goes, attributed to the first `S[n]>=` in the waiting instruction:
- ACT waits: DVE 440 us, PE 348 us, GpSimd DMA 188 us, its own sem 4 53 us.
- PE waits: ACT 1175 us, DVE 260 us, SP DMA 131 us.
- DVE waits: PE 787 us, ACT 499 us.
- GpSimd waits: DVE 991 us.

Nothing is saturated. Time is lost to cross-engine hand-offs, so only reducing the number or latency of hand-offs in the chain pays. Shaving instruction durations off a single engine does little unless that engine is on the chain.

The largest single instruction classes:
- ACT `COPY(scale/bias)`: 4520 x ~328 ns = 1.48 ms. Most are SBUF->SBUF with a per-partition scale.
- PE matmul pairs: 5441 matmuls, each fp32 and split into acc_flags=1/2 passes, with LDWEIGHTS ~277 ns plus MATMUL ~350 ns.
- GpSimd `DMA_DIRECT2D`: 2385 x 655 ns.
- DVE `STREAM_SHUFFLE`: 3597 x 147 ns = 530 us. Each is a single element to one of 4 partition quadrants, with `LOAD_MASK_SELECT` ahead of it (3948 of them, ~5 ns).
- ACT `SOFTPLUS`: 2296 x ~178 ns = 410 us.
- 17 ACT table loads at 1283 ns = 22 us.

### Ranked candidates

**None of the ceilings below has been measured by editing.** They are bounds from the profile, not predictions.

#### 1. Delete or collapse the remaining DVE shuffle quadruples

- **Engine and member:** DVE0.bin, `STREAM_SHUFFLE` (3597 left) plus their `LOAD_MASK_SELECT` (3948).
- **Pattern:** per unit, 4 shuffles with the same source and `dst = base + q*0x800000` for q=0..3, `channels=32`. They look like a 1-element broadcast of a scalar into each 32-partition quadrant.
- **Change:** replace each group by one DVE TT op or a `step_elem=0` broadcast read where the scalar's consumer can read it directly. The consumer may instead read the scalar from a partition-0 location through a broadcast.
- **Mechanism:** the `step_elem=0` per-partition broadcast on a DVE source operand was established on the device in commit 669f7b0. The 160-shuffle removal in this iteration is a precedent.
- **Ceiling:** the work is 530 us of DVE busy plus LMS. DVE is not saturated and waits on PE and ACT. Wall gain is somewhere between 0 and ~100 us. I'd guess 10 to 40 us. **Unknown until one group is removed and re-timed.**
- **Risk:** a wrong broadcast gives wrong state in only some heads or chunks. The verifier would catch gross errors, since it compares every output element, but not a hazard that appears only at other timings.
- **Cheapest falsifier:** convert one group for one (head, chunk), then check the output element tolerance against golden. Then convert all and compare the median against 2905 us. If it is not below ~2895, stop.

#### 2. Cut ACT->PE and PE->ACT hand-offs by removing ACT-side stand-in EVENT_SEMAPHOREs

- **Engine and member:** Activation0.bin, 3529 `EVENT_SEMAPHORE` at ~63 ns each = 222 us.
- **What they are:** stand-ins for deleted copies, left over from earlier iterations. Many carry real waits (`S[6]>=n`, `S[47]+=16@complete`), so they are not free to delete.
- **Change:** merge a stand-in's wait/increment into the adjacent surviving ACTIVATE's wait/update fields. This removes the slot and its issue overhead.
- **Mechanism:** an ACTIVATE already carries a wait and an update field (e.g. `S[6]>=2494 S[4]++@complete` on one instruction).
- **Ceiling:** 222 us of ACT busy, bounded above by that. ACT is not binding (1213 us idle), so the real gain is less. **Unknown, likely under 30 us.**
- **Risk:** merging waits changes when the update fires. Prior notes (669f7b0) show silent corruption from exactly this. The verifier may miss a race.
- **Cheapest falsifier:** fold one stand-in only where its wait is a subset of the next ACTIVATE's wait, and re-run `semaphores.py` for count parity.

#### 3. Move SOFTPLUS and small ACT work off the ring's critical edge

- **Engine and member:** Activation0.bin SOFTPLUS ACTIVATEs (2296 x ~178 ns).
- **Observation:** in the trace SOFTPLUS issues from ACT and its dependents (PE LDWEIGHTS `S[4]>=3034`, DVE `S[4]>=3032`) wait on it.
- **Change:** reorder the units so the dt-softplus for chunk k+1 runs before the matmul-feeding COPYs of chunk k. This is slot reorder with remapped sem thresholds.
- **Ceiling:** at most the ACT->PE gap, which is 1175 us total of PE idle. A better order might recover 5 to 10% of that. **Unknown. Nothing here has been measured.**
- **Risk:** high. Sem 4 is a monotone counter whose thresholds encode order. Reordering means renumbering every `S[4]>=n` for PE, DVE, SP and Pool waiters, and any slip silently deadlocks or races. The verifier would catch a hang but not a one-shot race.
- **Cheapest falsifier:** swap two adjacent independent ACT instructions in one unit, with thresholds adjusted, and check for a hang.

#### 4. Take GpSimd `DMA_DIRECT2D` descriptor generation off the critical path

- **Engine and member:** Pool0 (GpSimd), 2385 DMAs x 655 ns = 1.56 ms. They run in 16 queues with sems 8-23 (2384 increments each) and the ACT waits on them for 188 us.
- **Change:** reassign some of these to SP's hardware DGE (a `Sync DMA_DIRECT2D` takes ~535 ns), or batch two small ones into one.
- **Ceiling:** ACT waits on the GpSimd queues 188 us. That bounds the gain. **Real gain unknown.**
- **Risk:** medium-high. A queue and semaphore id change is needed, and the layouts must stay identical. A DMA that completes late would read stale SBUF. The verifier would catch that on this input but maybe not under another timing.
- **Cheapest falsifier:** confirm from a trace slice that ACT's `S[1x]>=n` waits actually stall (evt_wait >> 100 ns) at the units where the GpSimd DMA is the last arriver. If the DMA is usually already complete, the lever is dead.

#### 5. Remaining table loads

- **Engine and member:** Activation0.bin, 17 `ACT_TABLE_LOAD` at 1283 ns = 21.8 us.
- **Change:** delete or merge any that are still redundant (same table before and after).
- **Ceiling:** 21.8 us. **Bound is hard, from the profile.**
- **Risk:** low, but the lever is small and may already be exhausted. The agent has already removed many.
- **Cheapest falsifier:** list the 17 loads with the function before and after each. If every one switches table, close the lever.

### Closed

- ACT as the sole bottleneck: ACT busy fell 4722 -> 1809 us (baseline -> now) while total only fell 5249 -> 2905. ACT is now 62% busy, so more ACT cutting has diminishing returns.
- Per-instruction duration cuts on PE: PE busy is flat at ~1268 us vs 1255 baseline across all iterations.
- Reducing matmul precision (fp32 two-pass -> one-pass): error is at 87 to 89% of the 2.3e-07 limit, so no precision change is affordable.

### Unknown

- **Which engine is on the true critical path per unit.** Measurement: build the longest dependency chain through one mid-run unit from the sem waits (the `operands` strings in `Instruction.parquet` give `S[n]>=k` and `S[n]++@complete`) and sum durations plus handoff latencies. If chain length is ~3.7 us, only chain edits matter.
- **Whether sub-iteration noise exists in the 2905 baseline.** Spread is 0.066%, so differences above ~0.2% are credible.
- **Whether the new TT_ARITH ops are bit-exact vs the shuffles they replaced.** Measurement: dump both outputs and compare against HEAD's output bytes. I did not do this.
- **Whether GpSimd can read PSUM.** If it can, PSUM-evacuating COPYs could move from ACT to the idle part of Pool. I do not know and did not verify on the device.
- **Multi-input robustness.** Only one input set exists in `bench/`.

### Overall

The kernel is at 2905 us, 55% of baseline. The remaining time is mostly serialisation across engines, not any engine's throughput. I'd expect the next few percent to be small and increasingly hazardous: candidates 1 and 5 are the cheapest checks, and 3 and 4 carry the most synchronisation risk. If candidate 1's falsifier shows under 0.3% gain, treat the remaining NEFF-level headroom as small.

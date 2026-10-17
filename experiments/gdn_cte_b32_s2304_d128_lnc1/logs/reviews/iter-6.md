# Review of iter-6

Verdict: **the edit is sound but worth nothing. This iteration measured noise.** The cheap NEFF-level levers are used up. The list in section 4 is short on purpose.

## 1. What changed, and does it match the notes?

`neff_diff.py` against the iter-5 commit (HEAD), which is the right comparison. Against `baseline/` it shows the cumulative diff of all six iterations.

- Two members changed, both in `sg00`. All 32 others are byte-identical.
  - `Activation0.bin`: 29932 → 29935 slots, 439 slots differ.
  - `SP0.bin`: 18607 → 18611 slots, 432 slots differ.
- That matches the notes' "7 dead DGE transfers retired, Act 3 / SP 4".
  - Each retired `PSEUDO_DMA_DIRECT2D` became an `EVENT_SEMAPHORE` pair, so the slot count rose by 3 and 4. This is the iter-5 `ES_a`/`ES_b` recipe.
  - The `+16` increment on the transfer's semaphore is repaid with `SEM_ADD_IMM_COMPLETE`.
- Nearly all of the 439 and 432 differing slots are bookkeeping from the insertions:
  - Later cumulative wait values shift by 1, for example `S[4]` 14629 → 14628.
  - `events_update_mode`/`events_update_idx` flip between `NONE` and `SEM_INC_COMPLETE` on neighbouring slots.
  - Some wait indices change, for example 6 → 41, and some extended-wait indices change.
- Nothing outside these two streams moved. The change is consistent with the notes.

## 2. Is the latency change real? No.

| NEFF | `total_exec_time_us` |
|---|---|
| iter-5 commit | 14039 (`history.jsonl`: 14036) |
| `profile_latest.json` (3 repeats) | median 14022.0, min 14021.6, max 14032.2, spread 0.076% |

- The change is −17 us, which is **−0.12%**. The threshold is about 0.5% (about 70 us), so this is noise.
- The notes say the same thing: "the 16 us is below the noise floor".
- The loop is chasing noise if it keeps doing 0.1% edits. The notes' own item 4 ("confirm the current number rather than chase 0.1% edits") is correct.
- Cumulative gain against baseline (14779) is −5.1%. All of it came from iterations 2–5.
- **Measurement warning.**
  - My `profile.py --repeats 1 --parquet` run reported **13886 us**. That is about 1% below the 3-repeat median.
  - `Instruction.parquet` for that run spans 13.90 ms.
  - So the parquet or single-repeat mode reads about 1% low, or one capture can land outside the quoted 0.1% spread.
  - Never compare numbers across the two modes.

## 3. Is the edit safe beyond the verifier?

- **Error.** Output_0 is 1.372e-4 and output_1 is 6.841e-4 (86% of the 8e-4 limit). Both are identical to baseline to every digit, so the edit moved no error.
  - Output_1 already sits at 86% of the limit in the unedited NEFF. Any future edit that changes arithmetic has only about 1.2e-4 of headroom there.
  - Only data-movement edits are safe.
- **Semaphores.**
  - I ran `semaphores.py` on unpacked baseline and edited copies. The edited copy has **no `max_wait > total_inc`** over-waits.
  - 13 semaphores have `max_wait < total_inc` in both baseline and edited copies. That is not new and not a hang risk.
  - Per-semaphore totals match what the notes claim. Only the source labels moved, from `dge:*` to `*:EVENT_SEMAPHORE`.
- **Residual risk: the "dead" judgement.**
  - The 7 transfers were judged dead from the observed timeline: no reader whose `end_ts` reaches the write.
  - The notes' own strict test ("no reader anywhere in the program") returns **0** dead writes.
  - So deadness is timing-based. If a reader is not ordered after the write by a semaphore, it could be a latent race that one input set never exposes.
  - The same iteration found that iter-5's access map missed 42% of accesses.
  - The 7 are all in the last 340 us, and the verify output is bit-identical, so the risk is low. It is still not proven.
  - The gain is zero, so I would **revert it** and keep iter-5 unless the next edit builds on it. Zero gain does not justify any non-zero race risk.

## 4. Candidates for iter-7, best first

**The honest state:** no engine is above about 58–65% busy. The span is a flat latency chain.

- The iter-5 critical-path walk gave: SP queued-DMA latency 2874 us, Activation exec about 3100 us, SP DGE exec 1140 us, PE 1079 us, DVE about 2000 us.
- No single `gdn_cte.py` line holds more than 180 us of that path.
- My per-1 ms-window busy table is flat. Activation is 62–66% in every window and the rest are steady. There is no ramp, tail or startup to attack.
- All-engine idle time is 1.25 ms in total. 338 gaps are over 1 us, and only 50 are over 2 us (112 us in all). They are spread across `TENSOR_TENSOR`, `WRITE`, `DMA_DIRECT2D` and `EVENT_SEMAPHORE`. So there is no single stall to remove.
- Remaining levers are projects with unknown ceilings. I would not expect more than 1–2% from any of them.

### Ranked list

| # | Candidate | Ceiling | Risk |
|---|---|---|---|
| 1 | Simulate the Activation reordering before building it | unknown, plausibly 0.2–1 ms | high |
| 2 | bf16 `nc_transpose` probe on PE | unknown; hard upper bound 2.9 ms | high |
| 3 | Confirm the iter-5 number and ship iter-5 | 0 | none |

**1. Activation-stream reordering (iter-1's untried lever).** Run the iter-4 simulator before building anything.
- *Engine and member:* `sg00/Activation0.bin`, 29935 slots. Only pairs of `S[4]`-signalling slots that gate an SP `PSEUDO_DMATRIGGER`.
- *Change:* swap two adjacent Activation slots, then swap the wait values of their two waiters.
- *Mechanism:* ch. 10.3 r08 (device-verified).
- *Gain and ceiling:* The bound is the iter-6 measurement that 5,077 data-dependent triggers show 8,012 us of SP idle behind compute producers. That is an upper bound on slack, not on gain. Idle waiting is not removable if the producer simply is the critical path. A realistic estimate is 0.2–1 ms. The honest ceiling is **unknown**.
- *Risk:* a dependence mistake is a silent race. A wrong wait value hangs.
  - The verifier would catch a hang.
  - It might not catch a race. It runs one input set, though outputs are bit-exact and deterministic. Run with `--runs 3`.
  - Use `iter-6-acc3.py` for the access map. Do **not** use `iter-5-fuse2.py`.
- *Cheapest falsifier:* feed `iter-4-sim.py` (it predicts edits to about 1%) a version with only 20 swapped pairs chosen by the trigger's `start_ts − t_reach` slack. If the predicted gain is under 70 us, stop.

**2. bf16 `nc_transpose` on PE (replacing some of the 8,125 DMA transposes).**
- *Engine and member:* PE stream, `LDWEIGHTS` with `transpose_mode=ENABLED` plus `MATMUL` into PSUM. Then an Act or DVE `CAST` PSUM→SBUF, replacing the queued DMA transpose descriptors.
- *Mechanism:* The existing 1,152 ENABLED slots (`gdn_cte.py:223`) show the PE transpose path works for FP32 (`fp32_mode=LOW`). Nothing shows bf16.
- *Ceiling:* at most the 2.9 ms of queued transpose latency.
  - The real net is that minus the cost of 2 inserted PE slots, a PSUM bank and a CAST per transpose. Those are about 0.5 us of Act or DVE issue each.
  - The net is likely under 1 ms and **unknown**.
- *Risk:* PSUM bank allocation and slot insertion shift every cumulative semaphore count. The verifier would catch wrong data, not a race.
- *Cheapest falsifier:* one isolated probe. Pick one clean 64×64 bf16 transpose (987 qualify, from iter-6 §3) and add the PE pair by hand.
  - If bf16 gives NaN or wrong data, close the lever.
  - If it works, time the extra Act or DVE `CAST` at 213–266 ns against the removed 1,092 ns queued latency.

**3. Do not edit; confirm the number.**
- Ship iter-5 (or this NEFF, if you accept the unproven deadness) and re-measure with `--repeats 5`.
- The loop has no remaining sub-0.5% win worth the risk.

## Closed levers (with the number that closed them)

| Lever | Number |
|---|---|
| Swap matmul operands to delete DMA transposes | NaN on device, twice (iter-6 §3) |
| `transpose_mode` on a matmul operand | NaN, 8,192 and 32,768 elements (iter-6 §3) |
| Finish save/reload fusion | 0 of 132 groups fusable; exhausted at 188 of 320 (iter-6 §2b) |
| Delete dead staging saves | 7 found, 16 us (iter-6 §2a) |
| Fold FP32→BF16 `CAST` into producer | about 10 of 7,940 foldable (iter-6 §4) |
| Move `CAST` or `TENSOR_SCALAR` to Pool | +26 us and +15 us in simulation; Pool 17× slower per element; `ACTIVATE` hangs there (iter-4 §4.1) |
| Move to DVE | −50 us in simulation, below noise (iter-4) |
| Delete dominated `EVENT_SEMAPHORE`s | 707 slots, 0 us (iter-5 §3.1) |
| Queued-ring depth | 58 ms slack already; SP slots back-to-back (iter-6 §5) |
| `def.json` DMA queues | no editable depth; 3rd DGE host impossible (iter-2) |
| Merge consecutive identical instructions | exhausted; 13 runs of 2 remain (iter-5 §3.4) |
| DGE adjacency merge | about 50 us at most (iter-5 §3.3) |
| `ACTIVATE` → `COPY` downgrade | func 7 is `exp`, not identity (iter-4 §4.2) |
| PE `perf_opt` modes | changes operand packing (iter-5 §3.5) |
| `dge_op = DMA_TRANSPOSE` | iter-1 refutation 7 |
| DGE host ratio | Act/SP 50:50 within 2 us of the optimum (iter-2, iter-5) |

## Unknown (would change the ranking)

| Unknown | Measurement that settles it |
|---|---|
| Real run-to-run noise vs capture mode | Three `--repeats 3` runs of one NEFF, once with `--parquet` and once without. This explains the 13886 vs 14022 discrepancy. |
| Whether Activation reordering helps at all | Simulate 20 swaps in `iter-4-sim.py` (candidate 1). |
| Whether bf16 PE transpose works | One hand-built probe (candidate 2). |
| Whether the 7 retired DGEs are truly dead in all schedules | Check each reader's semaphore ordering against the write using `iter-6-acc3.py`, not timeline order. |
| Critical-path table on the current binary | `iter-5-cpath2.py` is broken: `KeyError: 'SP'` and an unattributable Pool wait. Fix it in about 20 min. Fixing it would re-rank candidates 1 and 2. |

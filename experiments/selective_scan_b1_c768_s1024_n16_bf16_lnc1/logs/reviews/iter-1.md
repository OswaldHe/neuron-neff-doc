# Review: iter-1 (selective_scan_b1_c768_s1024_n16_bf16)

## 1. What changed
`neff_diff.py` baseline vs iter-1: 3 of 34 members changed, the rest are byte-identical.

| member | slots | change |
|---|---|---|
| `sg00/Pool0.bin` | 296 -> 389 (+93), 382 differ | 170 `TENSOR_TENSOR` (ADD, fp32, 128ch x 512) now run on Pool. Previously they were DVE. Pool `COPY` fell from 181 to 112. Pool gained `WAIT_FOR_SEM_GE_IMM` on sem 3 and a DMA descriptor changed (src step `[1,3072]`, num_elem `[1,2]`, dst addr 0x2ca40 -> 0x2c640). |
| `sg00/DVE0.bin` | 1204 -> 1216 (+12), 1154 differ | The ADD ops were removed and waits were re-targeted. Sem 6 waiters rose from 1295 to 1388. |
| `sg00/Activation0.bin` | 17 slots differ | Only `events_semaphore_value`, for example 14->22, 43->68, 90->140. These are renumberings of the sem-4/sem-3 counters. |

In short, the iteration moved the cross-chunk accumulation (`acc += partial`) from DVE to GpSimd/Pool and renumbered the semaphores to match.

**Consistency with the notes:** `notes/` contains only the README template. The agent wrote no notes, so there is no claim to check the diff against, and the next agent has no record of what was tried. Notes need to be written.

## 2. Is the latency change real?
Yes.
- Baseline median is 602.64 us (min 602.52, max 602.83, spread 0.052%).
- Iter-1 median is 516.79 us (min 516.68, max 517.03, spread 0.068%).
- The delta is -85.8 us (-14.2%), which is more than 1000x the spread, so this is not noise.
- It is consistent with the engine busy numbers. DVE fell 592 -> 502 us and Pool rose 101 -> 415 us. PE (219 us), ACT (146 us) and DMA are unchanged. HBM bytes and DMA counts are identical.
- DVE is still 97% busy, so it is still the critical engine.

## 3. Safety
- **Synchronisation was not loosened.** In `semaphores.py` for both NEFFs, every semaphore has `max_wait == total_inc` (sems 3, 4, 5, 6, 24-39) and the DMA sems match the baseline. The only differences are:
  - sem 3: Pool increments went from 181 to 282.
  - sem 6: waiters went from 1295 to 1388 (more waits, not fewer).
- **Error moved little.**
  - output_0 max_abs_err is 0.06245, which is 83% of the 0.075 limit. The baseline reading in `verify_baseline.json` is the same figure, so the move to Pool did not change it. The headroom is thin.
  - output_1 error is 4e-5.
  - Any dtype or precision lever has almost no room.
- **Residual risk (unverified):** Pool ADD reads and writes the same fp32 accumulator (`0x4940` as both src0 and dst). The new sem-3 waits are the only guard on the read-after-write ordering against DVE. I did not prove statically that every `acc` reader waits on sem 3. The verifier uses one fixed input set, so a race that is rare in time could pass it. A cheap check: run verify and profile 5x and confirm that the output is bit-identical across runs.

## 4. Ranked candidates for iter-2

Common facts from the profile:
- DVE is 97% busy. `TENSOR_TENSOR` (430 ops, about 0.68 us each, 298 us) and `TENSOR_TENSOR_SCAN` (192 ops, 1.27 us each, 244 us) dominate it.
- Pool is 80% busy. Its 170 ADDs take 1.3-2.2 us each, about 3x the DVE time per op.
- PE and ACT have headroom (PE about 58% idle, ACT about 72% idle).

| # | Candidate | Detail |
|---|---|---|
| 1 | Shift a few more DVE `TENSOR_TENSOR` ops to Pool | **Where:** `DVE0.bin` fp32 `TENSOR_TENSOR` (ADD / MULTIPLY, no scan) -> `Pool0.bin`. |
| | | **Change:** retarget about 30 ops, copying the iter-1 pattern (the slots already moved; opcode field and sem wait/inc fields). |
| | | **Mechanism:** Pool can execute TENSOR_TENSOR with the same encoding. Established by iter-1. |
| | | **Ceiling:** balance point where DVE equals Pool is n = (502-415)/(0.68+2.0), about 32 ops, so about 20 us at most. In practice 5-10 us because of dependency latency. The bound comes from engine_busy (DVE 502, Pool 415). |
| | | **Risk:** low-medium. A semaphore renumbering mistake can race, and the verifier may not catch it. |
| | | **Kill check:** move 10 ops and profile. If total is not below 516 us, stop. |
| 2 | Fold standalone `EVENT_SEMAPHORE` waits on DVE into the next compute instruction's wait field | **Where:** `DVE0.bin`, 577 `EVENT_SEMAPHORE` instructions (55 ns mean, 32 us total busy). They sit immediately before `TENSOR_TENSOR` / `SCAN`, each waiting on one `S[x]>=n`. |
| | | **Change:** set `events_extended_wait_mode=WAIT_FOR_SEM_GE_IMM`, `wait_idx` and `sem_wait_value` on the next compute slot, then turn the standalone wait into a NOP or delete it. Deleting changes the slot count and needs jump / branch checks, so NOP in place first. |
| | | **Mechanism:** iter-1 itself added WAIT_FOR_SEM_GE_IMM to Pool slots (Pool0 bytes 32/33/36..39), so the field works on device. Whether it works on DVE compute slots is not shown. |
| | | **Ceiling:** at most 32 us (DVE EVENT_SEMAPHORE busy). Realistically about 15 us, because a slot has one wait field and only one wait can be folded per instruction. Also, the profile shows about 400 ns dispatch gaps after each EVENT_SEMAPHORE. If those go away, the gain could be larger. Unknown. |
| | | **Risk:** medium. A wrong fold loses a wait, and the verifier may pass anyway. Check with `semaphores.py` (waiters count down, `max_wait` unchanged). |
| | | **Kill check:** fold 20 waits and re-profile. If no change in total or DVE busy, the dispatch gap is not the cost, so abandon. |
| 3 | Reduce the Pool ADD cost | **Where:** `Pool0.bin`, the 170 ADDs. Durations vary 1.26 to 2.17 us. |
| | | **Change:** investigate and lower the high-duration ones. Candidate cause is an SBUF port conflict with DVE reads. Placing the operands on non-colliding offsets would mean address changes. |
| | | **Mechanism:** none established on the device. Do not start without a measurement. |
| | | **Ceiling:** up to about 100 us of Pool time, but Pool is not the critical engine, so end-to-end gain is only indirect, via enabling candidate 1. |
| | | **Risk:** medium (SBUF allocation overlap is not caught by one input set). |
| | | **Kill check:** correlate `duration_ns` with the concurrent DVE op in `Instruction.parquet`. No correlation means no lever. |
| 4 | Offload fp32 `TENSOR_TENSOR` MULTIPLY by a per-partition scalar to ACT | **Where:** DVE multiplies where `src1` is a broadcast, if any exist. |
| | | **Mechanism:** ACT `ACTIVATE` with `scale`. ACT cannot do tensor x tensor. |
| | | **Ceiling:** unknown. I did not verify that any DVE multiply has a per-partition scalar operand. The ones seen read full `[512]` tiles. |
| | | **Risk:** needs ACT table / func re-encoding, which is high effort. |
| | | **Kill check:** list the operand shapes of the 430 DVE TTs. If all are full tiles on both sides, drop this. |

A dtype change (fp32 -> bf16) in the scan or TT is **not recommended**. Error is already at 83% of the limit.

### Closed
- **Cross-chunk ADD on DVE:** moving the 170 ADDs to Pool took 602.6 -> 516.8 us (done in iter-1).
- **Fewer DMA / more bytes:** DMA is 18-20% busy and sems 24-39 are balanced. DMA is not on the critical path.
- **Precision reductions:** error is 6.245e-2 against a limit of 7.5e-2. No budget.

### Unknown
- Whether the 400 ns gap after each standalone DVE `EVENT_SEMAPHORE` is dispatch overhead or overlap in the profiler. Measure by deleting (NOP) 20 waits and diffing total.
- Whether Pool ADD duration (1.26-2.17 us) depends on the concurrent DVE op. Settle with a join on `Instruction.parquet`.
- The exact balance point between DVE and Pool. Settle with the 10-op experiment in candidate 1.
- Whether the new sem-3 waits guard every reader of the `0x4940` accumulator. Settle by checking that output is bit-identical across 5 runs and by reading the Pool/DVE slots that touch `0x4940`.

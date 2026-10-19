# Review of iter-0

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports 34/34 members identical, "payloads are
identical"; `cmp` agrees (byte-identical). `notes/` is empty, so there is no claim to contradict. There is
no `verify_latest.json`.

## 2. Latency
There is no real change. `observations/iter-0/profile_latest.json` has a median of 1349.279 us
(min 1347.1, max 1351.6, spread 0.334%). The baseline median is 1351.404 us (spread 0.218%). The -2.1 us
(-0.16%) difference is inside the combined sample range, and the NEFF is byte-identical, so it is pure noise.
My own `--repeats 1` re-profile gave 1350.457 us. Run-to-run noise is about +-2 us, so any win under about
5 us needs `--repeats 5` before it counts. Per-engine busy times also move by up to 0.8% with no edit.

## 3. Safety
There is nothing to check, since no semaphores changed. Baseline verify: max_abs_err 1.097e-04 against a
limit of 1.1e-3 (10%), so there is plenty of numeric headroom. Edits that loosen synchronisation are the
ones the verifier cannot catch, because it runs a fixed input set. Run `semaphores.py` on any such edit.

## 4. Candidates for the next iteration

Evidence is from one `Instruction.parquet` run. Summed instruction durations are PE 799 us, ACT 718,
DVE 591, Pool 266, Sync 260, against 1351 us wall. Busiest engine by engine-busy: ACT at 46%. The kernel
is therefore dependency/latency-bound, not engine-bound. It also spills heavily: 7.9 MB reload and 8.9 MB
save, against 4.9 MB of DMA transfer that the kernel needs. That spill is a compiler allocation choice, and
removing it in a binary patch means re-allocating SBUF and rewriting DMA descriptors.

| engine / opcode | n | sum us |
|---|---|---|
| PE MATMUL | 2272 | 540 |
| DVE TENSOR_TENSOR | 1152 | 253 |
| Pool DMA_DIRECT2D | 320 | 223 |
| PE LDWEIGHTS | 1878 | 208 |
| ACT TENSOR_SCALAR | 544 | 192 |
| ACT ACTIVATE | 576 | 189 |
| ACT CAST | 896 | 186 |
| DVE CAST | 672 | 143 |
| ACT DMA_DIRECT2D | 224 | 130 |
| Sync EVENT_SEMAPHORE | 489 | 129 |

I have not measured any of the levers below, and none is established on the device. Ceilings are "unknown"
unless stated. Treat this as a list of hypotheses, each with a cheap falsifier.

### Candidates, best first

**A. Find the critical path before editing anything (a measurement, no edit).**
- Build the wait graph from `SemaphoreUpdate.parquet` and `evt_wait_time_ns` in `Instruction.parquet`.
  Work out which engine's instruction chain gates the 1351 us.
- It is first because every other candidate depends on its answer. An edit off the critical path gains 0 us.
- Cost: minutes. Risk: none.

**B. Move off-critical-path ACT CASTs (896 instrs, 186 us) to Pool (19% busy).**
- Where: ACT (Scalar) member, CAST slots. Change the engine/queue and rewire the semaphore waits.
- Mechanism: engine load balancing. It is not established on this device.
- Ceiling: unknown. ACT busy is 627 us, but only the critical-path CASTs matter.
- Risk: high. It needs new semaphore edges, and a race can pass the fixed-input verifier once.
- Falsifier: if A shows the consumers of those CASTs are never waiting (`evt_wait_time_ns` ~0), drop it.

**C. Remove provably redundant semaphore waits (Sync 489 waits / 129 us; PE 501 waits / 48 us).**
- Where: Sync and PE members, EVENT_SEMAPHORE slots whose condition is already implied by an earlier wait on
  the same semaphore.
- Ceiling: at most 177 us summed, but most of that is real waiting, so expect a small fraction.
- Risk: high; an invisible race.
- Falsifier: `semaphores.py` finds no wait implied by an earlier one, so drop it.

**D. Dedupe repeated PE LDWEIGHTS (1878 instrs, 208 us).**
- Where: PE member. Drop an LDWEIGHTS whose source is identical to the one still loaded.
- Ceiling: at most 208 us of PE time, and PE is only 25% busy, so the wall gain is probably much smaller.
- Risk: medium. A weight tile may have been overwritten between the loads.
- Falsifier: compare operands of consecutive LDWEIGHTS per PE queue in the parquet. If none repeat, drop it.

I have not listed more because I can't support them from the profile. If A shows the critical path is a
serial recurrence, for example a chunked state update, then the remaining time is probably irreducible by a
descriptor-level edit, and the right outcome is to stop.

### Closed
None yet: `notes/` is empty. Baseline noise is the only number so far (0.2-0.33% spread, about +-2 us).

### Unknown (would change the ranking)
- Critical-path engine and chain: settled by A.
- Whether the spill DMAs (Pool/ACT/Sync DMA_DIRECT2D, about 450 us summed) overlap compute or sit on the
  critical path. Check whether compute instructions wait on them in `evt_wait_time_ns`. If they overlap, leave spill alone.
- Whether the 25% PE busy is PE waiting on DVE/ACT. Check the PE `EVENT_SEMAPHORE` wait durations per
  semaphore and who signals them.

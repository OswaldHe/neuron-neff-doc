# Review of iter-0

## 1. What changed
Nothing. `cmp baseline/kernel.neff kernel.neff` reports identical files. `neff_diff.py` reports 89/89 members identical and "payloads are identical". Neither sg00 nor sg01 was touched. `notes/README.md` is only the template, and `profile_latest.json` and `verify_latest.json` do not exist in the worktree, so there is no claim to check the diff against.

## 2. Is the latency change real?
There is no change. I profiled the unmodified `kernel.neff` once and got 75.875 us. The baseline median is 76.054 us with samples from 75.857 to 76.270 (spread 0.54%, about 0.4 us). 75.875 is inside that range, so any difference of 0.4 us or less is noise. A candidate needs a gain of at least 1 us, measured as a median of 5 or more repeats, before it counts.

## 3. Safety
There is nothing to assess because the NEFF is unchanged. Two points for later iterations:
- The baseline error on `output_0` is 2.32e-2 against a limit of 2.7e-2, which is 86% of the limit. Any edit that changes arithmetic or rounding is dangerous.
- The verifier uses one fixed input set and `--runs` only checks bit-identical repeats. A semaphore or allocation change can pass both and still race. Run `semaphores.py` on both sg00 and sg01 after any such edit.

## 4. Candidate edits for iter-1

### Main finding: ACT table-set thrashing
Activation is 99.5% busy (75.7 of 76 us). That is the critical engine.
- Per core, `ACT_TABLE_LOAD` takes **22 x 1283 ns = 28.2 us**, which is 37% of the kernel.
- The same sequence appears in sg00 and sg01.
- The ACTIVATE stream (`act_tbl_sel` / `activation_func`) from decoding `Activation0.bin` is:
  - Head: `T0 f7 | T1 f30 f9 | T2 f29 | T0 f7 | T3 f128 f128 f5 f30 f128..`
  - Tail: `T2 f29 .. | T4 f36 f30 | T2 f29 .. | T4 f36 .. | T2 .. | T4 ..` and so on.
  - The tail alternates T2 and T4 **17 times**, because it switches between rsqrt (f29, set `reciprocal_sqrt_and_small`) and silu (f36, set `silu_and_others`).
- The table sets are `natural_log_exp_and_others`, `softplus_and_others`, `reciprocal_sqrt_and_small`, `sigmoid_and_others`, `silu_and_others`.
- f30 (square) and f128 (copy) appear in several sets. They are free riders, so their position never forces a load.

| # | Candidate | Where | Change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Regroup tail ACTIVATEs so each table set is loaded once or twice | Activation0.bin, sg00 and sg01 separately; the T2/T4 region, about ACT slots 22-167 | Reorder independent ACTIVATE/CAST/COPY slots so all f29 run under one T2 load and all f36 under one T4 load. Delete the ACT_TABLE_LOAD slots that become redundant, keeping semaphore counts right. Re-point waits. | Reordering independent slots with re-pointed waits is **Verified** mechanics (doc 14.2 r08, 10.4 R4). Removing a slot while keeping semaphore counts is also Verified (r05/r06). That the saving is real is **not measured**. | Upper bound is 17 loads x 1.283 = about 21 us per core. The real figure is lower because the f29 and f36 chains probably depend on each other per head group. Unknown until the dataflow is read from `nki_source_location`. | Medium-high. Wrong re-pointing deadlocks or races. The verifier would likely catch a deadlock (timeout) but may miss a race. Each ACT slot carries wait and update fields, so edit sg00 and sg01 separately. | Group the Instruction.parquet ACT rows by `nki_source_location` and `bir_id`. Check that every f36 input is independent of the f29 output of the same stage. If each f36 consumes the preceding f29, the loop is serial and the candidate dies in minutes. |
| 2 | Cut the head's 5 table loads | Activation0.bin, slots 1, 3, 6, 8, 10 | Sequence is `T0 f7, T1 f30 f9, T2 f29, T0 f7, T3`. T0 is loaded twice. Check whether the second `f7` can run under the first T0 load, or whether the first `f7` is dead. | Same as candidate 1. | The head is about 9.8 us (12.2-22.0) for about 1 us of ACTIVATE work, so the ceiling is up to about 5 us. Part of it may be hidden behind DMA waits. | Low-medium. | Profile after deleting one `T0` load plus reordering. If total time does not move, the head is DMA-bound and this is closed. |
| 3 | Move COPY/CAST/copy-ACTIVATE work off Activation to Pool or DVE | Activation0.bin to Pool0.bin or DVE0.bin, both subgraphs | Per core: 36 CAST (about 10.5 us), 12 COPY (2.9 us), about 25 ACTIVATE f128 at about 490 ns (about 12 us). Move the CASTs and COPYs, then the f128 ones, to Pool (12% busy, 9.5 us) or DVE (43%). Replace each with an EVENT_SEMAPHORE on Act so its semaphore count is kept. | Doc 14.3 item 3. The instruction-mix claim is from `generated/opcodes.md`. Pool and DVE speed for the same op is **unmeasured**. | Up to about 25 us of Act time is movable. Real gain is lower because Act waits on other engines and Pool/DVE cost more per op. | High effort, medium-high risk. It needs new semaphore wiring on a second engine and the verifier cannot see a race. A DVE CAST to bf16 could round differently from the Act CAST. | Move one CAST. Check output bits are identical to baseline and that the DVE duration is under 290 ns. |
| 4 | Replace f29 rsqrt on Act with DVE pow(-0.5) | Activation to DVE, f29 slots in T2 | Removes T2 and probably leaves only T4 (silu, f30, f128) for the tail. This kills the thrash entirely, about 17 loads. | Whether DVE `TENSOR_SCALAR` supports a pow op is **unverified**. | Up to about 21 us. | **High.** The numerics change (40000-point table versus pow) and the error is already at 86% of the limit. High encoding effort. | Check the ISA reference for a pow/rsqrt ALU op on DVE. If absent, closed. If present, test a single slot's output error before any wider edit. |

Rank by gain per risk: 2, then 1, then 3, then 4. Candidate 1 has the largest plausible gain, so it is worth the dataflow check first. Candidate 2 is the cheapest to test.

### Closed
Nothing is closed by this iteration. No NEFF edit has been measured yet, and `notes/` is empty. The doc's Verified negatives (doc 14.2) still apply:
- Removing waits on an engine's own completion semaphore gave wrong output or errors and no gain.

### Unknown
- **Dataflow between f29 and f36 per head group** (decides candidate 1). Measure it from `nki_source_location` and `bir_id` in `Instruction.parquet`.
- **How much of the head, 12.2-22.0 us, is hidden behind DMA or semaphore waits** (decides candidate 2). Test it by moving or deleting one load and re-profiling.
- **Whether the two cores' ACT streams are exactly the same sequence.** They look identical in the function stream, but the number of semaphore waits differs (14 versus 13). Check the wait fields before copying an edit between subgraphs.
- **Whether `ACT_TABLE_LOAD` duration is fixed at 1283 ns.** I saw no field in the instruction that varies it.

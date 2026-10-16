# 14. Making a kernel faster by editing its NEFF

This chapter is about choosing edits that save time. It separates what has been measured from
what is only a candidate. All mechanics used here (field edits, deleting, inserting, reordering,
DMA edits) are **Verified** to run correctly (chapters 10 and 13); whether a given use of them
saves time depends on the kernel and has to be measured.

## 14.1 What sets the latency

The five engines and the DMA queues run in parallel; the program ends when the last of them is
done. Latency is therefore set by the busiest unit and by the waits on the critical path.

Find the busy unit from a profile (chapter 13.4): sum `duration_ns` per `engine` and `pcore_idx`
and compare with the span (`max(end_ts) - min(start_ts)`). Example, 40-MoE (4.577 ms span):

| Engine | Busy (core 0) | Busy (core 1) |
|---|---|---|
| Vector (DVE) | 4.30 ms | 4.42 ms |
| Scalar (Activation) | 2.29 ms | 2.11 ms |
| Sync (SP) | 1.64 ms | 2.15 ms |
| GpSimd (Pool) | 1.57 ms | 1.62 ms |
| Tensor (PE) | 0.19 ms | 0.16 ms |

40-MoE is DVE-bound (94% busy). Saving time there means taking work off DVE or making DVE
instructions cheaper; changes on other engines only help if they shorten DVE's waits.

Then find the expensive instructions on that engine: group `duration_ns` by `opcode` and by
`nki_source_location`. In 40-MoE three NKI lines take 3.2 ms of DVE time (one `TENSOR_TENSOR`
line at 1.32 ms over 1120 instructions, one at 1.19 ms over 224, one `TENSOR_SCALAR` line at
0.71 ms).

## 14.2 Levers that were tested

| Lever | Result | Status |
|---|---|---|
| Remove waits on an engine's own completion semaphore | Wrong output (DVE), runtime out-of-bounds error (Pool), no gain where it was harmless (Activation) | **Verified negative** (o01, chapter 10.5) |
| Remove an instruction whose result is not needed, keeping semaphore counts right | Runs correctly; the output is exactly what the remaining instructions compute | **Verified mechanics** (r05, r06) |
| Reorder independent instructions on one engine (re-point waits) | Runs correctly, output unchanged | **Verified mechanics** (r08) |
| Re-point or reshape a queued DMA through `SP0.json` | Runs correctly, output exactly as predicted | **Verified mechanics** (r09) |

## 14.3 Candidate levers (not yet measured)

Each of these uses only the verified mechanics above. None has been measured to save time.

1. **Remove duplicated work between the two cores.** In kernels that do not split work between
   the physical cores (HCCollapse, HCSublayerIn), both subgraphs run the same stream and write the
   same output. Removing redundant work from one subgraph shortens nothing by itself (the cores run
   in parallel), but it frees shared resources such as DMA and HBM bandwidth.
2. **Start long-latency work earlier.** If an SP DMA trigger or a PE matmul waits behind
   instructions it does not depend on (same engine, program order), move it earlier and re-point
   the waits (chapter 10.4 R4). The BIR dataflow predecessors (chapter 11.2) say what it really
   depends on.
3. **Move work off the busy engine.** `TENSOR_TENSOR_ARITH_OP` runs on DVE and Pool,
   `TENSOR_SCALAR_ARITH_OP` on DVE, Activation and Pool, `COPY`/`CAST` on DVE, Activation and Pool
   (engine counts in `generated/opcodes.md`).
   Moving an instruction to another engine's stream means: replace it on the old engine with an
   `EVENT_SEMAPHORE` that keeps the old engine's count; insert it on the new engine with waits for
   its inputs' producers; make its consumers wait on the new engine's semaphore at the new count.
   Engines differ in speed for the same opcode; measure.
4. **Fewer, larger instructions.** Each instruction has a fixed cost. Where the compiler split a
   tile into several instructions over contiguous memory, one instruction with a larger access
   pattern may be cheaper (keep `num_elem` within 16 bits and the pattern within 4 dimensions).
5. **Cheaper data types.** Lower-precision operands can be faster on DVE (the header advertises
   `v3-2x-4x-dve-perf-mode`), but this changes the numbers: the edit must still pass the module's
   tolerance. Which dtype combinations run faster has not been measured.

Not a lever in the studied kernels: `MATMUL.perf_opt`. The fp8 matmuls already use `DOUBLE_ROW`
and the bf16 ones use `NONE` (chapter 7.3).

## 14.4 A loop for an agent

1. Profile the unedited NEFF; find the busy engine and its most expensive NKI lines (14.1).
2. Pick one candidate edit; write down the expected change in time and why.
3. Apply it with the tools; re-balance semaphores; run `semaphores.py`.
4. Run with `run_neff.py --exec-timeout 60`; require `passed = 1` (and bit-exact output for edits
   that should not change numbers).
5. Repeat baseline and edit 3+ times; keep the edit only if the median improves by more than the
   spread (chapter 13.2 step 8).
6. Keep a log of every attempt and its result, including the ones that did not help.

# 10. Synchronisation: semaphores

The five engines run their streams independently. The only things that order work between them
(and between instructions and DMA) are **semaphore waits** and **semaphore updates** (chapter 6.4).
Every edit that adds, removes or moves an instruction or a DMA transfer has to keep the
semaphore accounting right. This chapter gives the model and the rules; all rules marked
**Verified** were tested on the device with bit-exact predicted output.

## 10.1 The model

1. A NeuronCore has 256 semaphores (`NUM_SEMAPHORES = 0x100`). S[0]..S[2] belong to the runtime
   (`runtime_semaphore_count = 3`); no NEFF instruction uses them. **Observed.**
2. Every wait value is a **count since the start of the program, or since the last
   `EVENT_SEMAPHORE_RANGE_CLEAR` that zeroed that semaphore** — never a delta. **Verified** for
   straight-line code (the model below predicts every wait in all 28 subgraphs; see 10.3).
   Loops (chapter 8.6) clear the semaphores their body uses before the loop and again just
   before the back-branch, so inside a loop body the counts restart every iteration.
   **Observed** (01-Engram-long8192; the attention kernels have no loops). The runtime epilogue
   zeroes the semaphores for the next run with `EVENT_SEMAPHORE_RANGE_CLEAR`, split over the
   engines (in the HCCollapse profile: Tensor 3-24 and 26-53, Scalar 54-104, GpSimd 105-155,
   Vector 156-206, Sync 207-255). **Observed.**
3. An instruction with `wait S[i] >= v` does not start until S[i] has reached v.
4. An instruction with `SEM_INC_COMPLETE` adds 1 to its `update_idx` semaphore when it finishes.
5. Within one engine, instructions **complete in program order**. The compiler relies on this:
   in HCCollapse the stage-1 result instruction does not signal at all, and the SP trigger that
   copies both stages out waits only for the *later* stage-0 instruction. **Verified** (that
   NEFF runs correctly as compiled).

## 10.2 Who increments what

| Semaphore | Incremented by | Amount | Status |
|---|---|---|---|
| S[3] | Pool engine instructions | +1 each | Observed (3,860) |
| S[4] | Activation engine instructions | +1 each | Observed (99,128) |
| S[5] | PE engine instructions (`MATMUL`) | +1 each | Observed (130,151) |
| S[6] | DVE engine instructions | +1 each | Observed (164,048) |
| A queue's `semaphore_set[k % 16]` | completion of queued DMA transfer *k* of that queue | + `num_queues` (16) | Verified by accounting; assignment rule verified on all 45,909 transfers |
| A DGE instruction's `semaphore` | completion of that `PSEUDO_DMA_DIRECT2D` | + `sem_increment` (16) | Verified by accounting |
| S[8] .. S[11] | `PSEUDO_TRIGGER_COLLECTIVE2` (its `update_idx`) | +1 per collective | Observed |
| S[24] | `PSEUDO_CORE_BARRIER` (its `update_idx`) | +1 per barrier | Observed |
| other (26-29 in some NEFFs) | register instructions (`MOVE`, `ALU_OP`, `PSEUDO_TENSOR_LOAD/STORE`) | +1 | Observed |

So "wait `S[6] >= 7`" means "wait until the DVE has finished its 7th signalling instruction", and
"wait `S[27] >= 16`" (queue `qSPIO0`, `semaphore_set[1] = 27`) means "wait until DMA transfer 1 of
`qSPIO0` is done". A second transfer on the same semaphore (id 17) would be waited for as `>= 32`.

`tools/semaphores.py <unpacked> sgNN` prints this table for a subgraph: for every semaphore, its
sources, the total it can reach and the largest value any instruction waits for.

## 10.3 Evidence for the model

For every semaphore of all 28 subgraphs, the largest wait value is at most the total the model
says the program can produce, and in most cases exactly equal (the last consumer waits for
everything). There is no wait the model cannot satisfy. **Verified** (`tools/semaphores.py`
run over all 28 subgraphs).

Edits that change counts behave exactly as the model predicts (experiments in
`artifacts/experiments/runtime_rules/`, all on HCCollapse, 4 ranks, bit-exact outputs):

| Experiment | Edit | Result |
|---|---|---|
| r05 neutralize | replace an instruction by an `EVENT_SEMAPHORE` with the same wait and update | runs; output is exactly "that term missing" |
| r06 delete | remove the slot, then lower by one every wait on S[6] with value >= 2 (in all engines, including the SP trigger) | runs; same output as r05 |
| r07 insert | insert a copy of an accumulate step; raise later waits on S[6] that count past it; make the consumer of its result wait for it | runs; output is exactly "that term twice" |
| r08 reorder | swap two independent instructions; re-point the two waits that referred to them by count | runs; output unchanged |
| r11 unreachable wait | wait for a value nobody produces | hang, then `NRT_TIMEOUT` (10.6) |

## 10.4 Rules for editing

**R1 — Think in producers, not numbers.** A wait `S[i] >= v` refers to "the v-th increment of
S[i]". Before an edit, write down for each wait *which* instruction or DMA transfer produces that
increment. After the edit, recompute v from the new positions.

**R2 — Removing an instruction.** Either
* replace it with an `EVENT_SEMAPHORE` that keeps its wait and its update (no other change
  needed; costs one cheap slot), or
* delete the slot and, for its engine's semaphore, lower by one every wait (in **all five**
  engines of the subgraph) whose value counted that increment.

**R3 — Adding an instruction.** If it increments a semaphore, raise by one every later wait that
counts past it, and make the instruction that consumes its result wait for it. A purely mechanical
"+1 for everything after" is not enough: the consumer that used to wait for the previous producer
now has to wait for the new one (r07).

**R4 — Reordering on one engine.** Swapping two instructions changes which one is the k-th
increment. Re-point every wait that referred to either of them.

**R5 — DMA.** A queued transfer signals `semaphore_set[id % 16]` with +16. Adding a transfer to a
queue changes the `id`s after it unless you append it at the end; appending is simplest
(new `id` = last + 1, semaphore = `semaphore_set[new_id % 16]`, and that semaphore's later waits
are unaffected). A DGE instruction signals its own `semaphore` field with `sem_increment`.

**R6 — Both subgraphs.** Each subgraph has its own semaphores. Re-balance each subgraph you edit.

**R7 — Cross-engine waits are the real dependencies.** A wait on another engine's semaphore or
on a DMA semaphore orders data between units that run in parallel. Never drop one unless the data
it protects is already safe by some other wait.

## 10.5 Waits on an engine's completion semaphore are real dependencies

Many instructions wait on their own engine's **completion semaphore** (for example a DVE
instruction waiting on S[6] for the DVE instruction that wrote its input). Engines start the next instruction before
the previous one has finished writing, so these waits are needed. **Verified** (experiment o01
on 40-MoE, both subgraphs, everything else unchanged):

| Waits removed | Count | Result |
|---|---|---|
| DVE waits on S[6] | 4,121 | runs, but the output is wrong (not bit-identical, validator fails); 4.576 ms vs 4.561 ms baseline |
| Pool waits on S[3] | 20 | `NRT_EXEC_OOB` (10.6): DGE DMAs read address registers before `ALU_OP`/`TENSOR_LOAD` had written them |
| Activation waits on S[4] | 25 | output bit-identical, 4.564 ms: no gain |
| PE waits on S[5] | 0 | PE had none |

Removing a single DVE self-wait in HCCollapse (r10) happened to give the right answer. That is
timing luck, not a rule: do not remove same-engine waits.

## 10.6 What a broken wait looks like

A wait that can never be satisfied hangs the engine. With `NEURON_RT_EXEC_TIMEOUT=30` the run
fails after 30 s with

```
ERROR NRT:nrt_infodump   Failure: NRT_TIMEOUT in tpb_xu_step
RuntimeError: ... [Timeout]: [NRT_TIMEOUT] status_code=5. Operation failed due to exceeding the
maximum allowed execution time limit.
```

The dump does not say which engine or instruction was stuck. The device recovers: the next run
works normally. **Verified** (r11, then the o01 baseline ran on the same device).

Always run edited NEFFs with a short `--exec-timeout` (`tools/run_neff.py`); the validator's own
default is 1800 s.

An instruction that runs before its address registers are ready can send a DGE DMA outside its
variable. The runtime stops the run with

```
ERROR TDRV:exec_process_custom_notification  ... scatter/gather (indirect memory copy via scalar DGE)
      out-of-bound access. ... neff instruction index = 5325
RuntimeError: ... [NRT_EXEC_OOB] status_code=1006. Execution failed due to one or more indirect
memory copies or embedding updates attempting to access memory outside the allocated bounds.
```

`neff instruction index` points at the failing slot. **Verified** (o01 Pool variant).

A wait that is satisfied too early (the opposite mistake) does not hang. It lets an instruction
read data before it is written, which gives wrong numbers that may change from run to run.
Compare outputs bit-for-bit with the unedited NEFF (chapter 13) to catch it.

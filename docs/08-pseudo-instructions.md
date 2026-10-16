# 8. Pseudo-instructions

A **pseudo-instruction** is an instruction slot the runtime rewrites when it loads the NEFF. Its
opcode name starts with `PSEUDO_`. The rewriting is needed because the NEFF does not know
run-time facts such as physical addresses, DMA ring registers, the rank id or the other core's
registers.

The table lists what each pseudo-instruction in the corpus becomes at run time. **Verified** from
the profiler: every runtime instruction is matched to the NEFF slot it came from
(`artifacts/reference/pseudo_expansions.json`).

| Pseudo-instruction | Engines | Becomes | Purpose |
|---|---|---|---|
| `PSEUDO_DMATRIGGER` | SP | `WRITE` to DMA ring registers | start queued DMA transfer `count_union_block_id` of `dma_queue_name` (chapter 9.3) |
| `PSEUDO_DMA_DIRECT2D` (+ `PSEUDO_EXTENSION`) | SP, Activation, Pool | `DMA_DIRECT2D`, `DMA_INDIRECT`, `MOVE` | DGE DMA (chapter 9.4) |
| `PSEUDO_TENSOR_LOAD` | all | `TENSOR_LOAD` | load a value from memory into a register |
| `PSEUDO_TENSOR_STORE` | all | `TENSOR_STORE` | store a register to memory |
| `PSEUDO_GID_LOAD` | all | `MOVE $R[n] = <global id>` | give the program the rank's id (8.4) |
| `PSEUDO_CORE_BARRIER` | Pool | `WRITE` + `EVENT_SEMAPHORE` | barrier between the two physical cores (8.3) |
| `PSEUDO_TRIGGER_COLLECTIVE2` (+ `PSEUDO_EXTENSION`) | Pool | `WRITE` | start a collective (8.5) |
| `PSEUDO_SYNC_BARRIER` | all | `DRAIN` + `EVENT_SEMAPHORE` on S[2] | barrier between the engines of one core (8.2) |
| `PSEUDO_DMASWAP_QUEUE_SET` | SP | `DRAIN`, `EVENT_SEMAPHORE`, `WRITE`, `TENSOR_STORE` | switch DMA queue instances (`queue-set-instances` feature) |
| `PSEUDO_BRANCH_LABEL` | all | nothing | a branch target (8.6) |
| `PSEUDO_BRANCH_PREFETCH_HINT` | all | `BRANCH_PREFETCH_HINT` | branch prediction hint (8.6) |
| `PSEUDO_EXTENSION` | after the two above | nothing on its own | extra operands of the previous slot (8.1) |

Field layouts: [`generated/structs.md`](generated/structs.md).

## 8.1 Extension slots

`PSEUDO_EXTENSION` (0xDA) carries extra operands for the slot just before it: the indirect
gather/scatter part of a `PSEUDO_DMA_DIRECT2D` (`v3...PSEUDO_DMA_EXT_STRUCT`) or the destination
part of a `PSEUDO_TRIGGER_COLLECTIVE2` (`v3...PSEUDO_TRIGGER_COLLECTIVE2_EXT_STRUCT`). It has no
events block and never runs on its own. **Verified** (it is never profiled; decoding it with the
previous slot's extension struct gives the fields the profiler prints for that slot).

Rules: keep an extension slot directly after its instruction; move and delete them together;
`tools/isa.py decode_stream()` picks the right struct from the previous slot.

## 8.2 `PSEUDO_SYNC_BARRIER`

No operands. The runtime turns it into a `DRAIN` (wait until this engine's earlier instructions
are done) followed by `EVENT_SEMAPHORE` steps on S[2], the runtime's own semaphore, through which
all five engines meet. The compiler puts one per engine at block boundaries (BIR names like
`Block1-barrier0-Activation0`). **Observed.** Removing one from only some engines would leave
the others waiting on S[2] forever.

## 8.3 `PSEUDO_CORE_BARRIER`

Fields: `id` (barrier number), `semaphore` (25 in the corpus), and the usual events (the update
is `S[24]++`). At run time it becomes a `WRITE` of 1 to a register of the other physical core
followed by `EVENT_SEMAPHORE ... S[25]>=1 S[25]--@complete` with `S[24]++@complete`: tell the
other core "I am here", wait until it has said the same, consume its signal, and count the
barrier on S[24]. **Observed** (profiler text). Use it in both subgraphs, the same number of
times.

## 8.4 `PSEUDO_GID_LOAD`

Becomes `MOVE dtype=uint32 $R[61]=<id>`; the destination register number is byte 23 of the slot
(61 in all six NEFFs that use it). On rank 0 the value is 0. Later code uses `$R[61]` to make one
NEFF behave differently per rank (register-based addresses, `COMPARE_BRANCH`). **Observed.** The
layout of this slot is not in neuron-dbg; `artifacts/reference/isa_layout_supplement.json` gives
the fitted fields.

## 8.5 `PSEUDO_TRIGGER_COLLECTIVE2`

Main slot (`common...PSEUDO_TRIGGER_COLLECTIVE2_STRUCT`): `op` (reduce operation, e.g. `ADD` or
`BYPASS`), `dtype`, `ctype` (`ALL_GATHER`, `ALL_REDUCE`, ... from `COLLECTIVE_TYPE`), `cc_dim`
(`DIM_0`), `src0`/`src1` (8-byte addresses, usually table form pointing at the collective
buffers). Extension slot: destination address and chaining. The update (`S[8]++`, `S[9]++`, ...)
fires when the collective is complete; consumers wait on it. At run time the slot becomes a
`WRITE` that starts the collective engine. **Observed.**

The collective buffers are the `virtual` (in `sg00`) / `remote` (in `sg01`) variables named
`cc_src`, `cc_dst`, and `list` variables group them (`CCGrp0`).

## 8.6 Branches and loops

Most studied kernels are fully unrolled. 01-Engram-long8192 has loops:

```
252  COMPARE_BRANCH  if $R[61] >= 4096 goto label 12      (skip the loop)
253  PSEUDO_BRANCH_LABEL  label 3                          (loop start)
254  PSEUDO_BRANCH_PREFETCH_HINT  branch_pc=331
 ...  loop body ...
329  EVENT_SEMAPHORE_RANGE_CLEAR  (bitmask: semaphores used in the body)
331  COMPARE_BRANCH  if $R[61] < 4096 goto label 3         (back-branch)
332  PSEUDO_BRANCH_LABEL  label 4
333  PSEUDO_BRANCH_LABEL  label 12
```

* `COMPARE_BRANCH`'s `br_immediate_int32[0]` is a **label id**, not a slot offset. The target is
  the `PSEUDO_BRANCH_LABEL` with that `label_id`. The runtime turns it into a relative byte
  offset (the profiler prints `label_id=5440`, i.e. +85 instructions, and `4294961984`, i.e.
  -83 instructions). **Observed** (every branch in the NEFF has a matching label).
* Labels produce no runtime instruction (0 of 60 labels profiled). **Verified.**
* `PSEUDO_BRANCH_PREFETCH_HINT`'s `branch_pc` is the **absolute slot index** of the branch it
  helps (331, 349). **Observed.**
* Loop counters live in registers (`$R[61]` here) and are updated by `ALU_OP` instructions.

**When editing a NEFF with loops:** inserting or deleting slots does not break `COMPARE_BRANCH`
(labels move with the code), but you must fix every `branch_pc` after the edit point, and keep
the semaphore clears at the loop ends (chapter 10.1).

## 8.7 What pseudo-instructions mean for editing

* You can edit their fields (queue name, transfer id, DGE addresses and shapes, collective type)
  like any other instruction; the runtime reads them at load time.
* The profiler shows the expanded form, so a pseudo-instruction's runtime pc and the slot index
  differ. After a `PSEUDO_GID_LOAD` the profiler's `compiler_pc` labels run one ahead of the real
  slot index (chapter 11.4).

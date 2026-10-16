# 11. Debug info

Each subgraph carries protobuf debug files. They are what connects an instruction slot to the
NKI source line that produced it. The runtime does not need them. **Verified** (r03: a NEFF with
all `debug_info_*.dbg` files and `*_dbg` keys removed loads, runs bit-exact and can still be
profiled).

## 11.1 Files

| File | Level | One record per |
|---|---|---|
| `debug_info_asm_<Engine>.dbg` | asm | instruction slot of `<Engine>0.bin`, in slot order |
| `debug_info_backend_<Engine>.dbg` | BIR | BIR instruction of that engine |
| `debug_info_backend_DMA.dbg` | BIR | queued DMA transfer |

The asm file's entry *i* has `parent_ids = [bir_id]` pointing to a record of the backend file.
Entry 0 (the `SET_ORDERING_MODE` the compiler puts first) has no parent. The backend files name
a parent file `debug_info_penguin.dbg` that is not shipped in the NEFF. **Observed.**

Schema: `neuronxcc/proto/ir_debug_info_pb2.py` (message `ir_debug_info`). `tools/dbginfo.py`
decodes it without protobuf; it matches the official decoder on 88 files and 53,690 records.
**Verified.**

## 11.2 A BIR record

| Field | Meaning |
|---|---|
| `id` | BIR instruction id (the asm level points here) |
| `name` | BIR name (11.3) |
| `parent_ids` | the NKI-level instruction it came from (in the missing penguin file) |
| `dataflow_predecessors` + `_types` | BIR ids this instruction depends on through data: `FLOW_DEPENDENCE` (reads what they wrote), `ANTI_DEPENDENCE` (overwrites what they read), `OUTPUT_DEPENDENCE`, `ORDERED` |
| `scheduling_predecessors` + `_types` | BIR ids it must wait for after scheduling (these become semaphore waits) |
| `instruction_type` | `REGULAR`, `TRANSPOSE` or `SPILL` |
| `attrs.kernel_file`, `attrs.kernel_line`, `attrs.kernel_name` | source position: indexes into the file's `kernel_filenames` / `kernel_functions`, and the line number |
| `attrs.dma_queue` | for DMA transfers: the queue name |

`stack_frame_index` is empty in all studied files; the source position is always in the attrs.
**Observed.**

The predecessor lists are useful when you edit: they say which instructions really produce an
instruction's inputs (`FLOW_DEPENDENCE`), which is exactly what a semaphore wait must protect
(chapter 10).

## 11.3 BIR names

The name tells what a slot is for. Patterns seen (with `N` for numbers):

| Pattern | Meaning |
|---|---|
| `I-N_inst__I-K-0` | NKI instruction `I-N` of kernel call `I-K` |
| `...-wait-N` | an `EVENT_SEMAPHORE` placed just before the instruction with the same base name. Because an instruction has only one wait field, extra dependencies need extra slots (**Inferred** from the slot layout) |
| `...-trig`, `...-trig-wait-N` | a queued DMA trigger and its extra waits |
| `...-block-N` | a queued DMA transfer (in `debug_info_backend_DMA.dbg` and in `SP0.json` `#comment`) |
| `...-act`, `...-PWP` | the activation-function part of an op, and its table load |
| `...-reg-mul-N`, `...-reg-add-N`, `...-tensor-load-N` | register arithmetic and loads for dynamic addresses |
| `CB-N_inst__...`, `cb_sg00_Block1_start` | core barriers |
| `AG-N_inst__...`, `AR-N_inst__...` | all-gather and all-reduce triggers |
| `BlockN-barrierN-<Engine>0`, `Block1-final-DMA-NOP` | engine barriers and the end-of-program DMA wait |
| `BlockN_LoopBody_N-...`, `scf.for-...` | loop control |

**Observed.**

## 11.4 Using the profiler with debug info

neuron-explorer's `Instruction` table has `compiler_pc` (the slot index), `bir_instruction_name`
and `nki_source_location` per executed instruction. Two things to know:

* Pseudo-instructions expand into several runtime instructions; they share the slot's
  `compiler_pc`. Extension slots and branch labels never appear.
* **Profiler quirk:** in NEFFs that start with `PSEUDO_GID_LOAD` (6 of 14), the profiler's
  `compiler_pc`, `compiler_opcode` and `bir_instruction_name` labels run one slot ahead of the
  real slot index from that instruction on. The runtime `opcode` column is correct. **Verified**
  (opcodes realigned against the NEFF slots: 0 mismatches after shifting by one; without the
  shift, about 70% of the field checks fail in those NEFFs). `tools/corpus.py` does the
  realignment.

## 11.5 Keeping debug info consistent after an edit

The runtime ignores it, so a mismatch only affects tools:

* If you add or remove slots and do not update `debug_info_asm_<Engine>.dbg`, the profiler maps
  later slots to the wrong source lines. Runs and timings are unaffected (r04).
* To keep it right, insert or delete the matching asm entry (copy a neighbour's entry for a new
  slot). The runtime-rule experiments do this with the official protobuf class
  (`artifacts/experiments/runtime_rules/rr_lib.py: asm_dbg, asm_dbg_write`).

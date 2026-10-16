# 6. Instruction slots: the common format

## 6.1 Where the layouts come from

The instruction layouts in these documents are the vendor's own, not guesses:

1. `neuron-dbg` (`/opt/aws/neuron/bin/neuron-dbg`) is a Go program built with full DWARF debug
   info. Its packages `tools/internal/isa/{common,v2,v3,v4}` define one struct per instruction
   format and typed constants for every enum. `tools/dwarf_isa.py` extracts them
   (`artifacts/reference/isa_dwarf.json`). Package `v3` is NeuronCore-v3 (Trainium2); v3
   builds on `common` and `v2`.
2. `neuron-dbg` also links the vendor's C accessors, compiled without optimisation: one tiny
   function `get_<pkg>_<STRUCT>_<field>` per field, each a single load from the packed
   instruction. `tools/isa_layout.py` reads the load instruction of every accessor and so gets
   each field's byte offset, width, bit position and array stride
   (`artifacts/reference/isa_layout.json`, 4355 fields in 192 structs).
3. `neuron-dbg`'s disassembler (`tools/pkg/disassembler.caymanDisassembleInstruction` and
   `sundaDisassembleInstruction`) chooses a struct per opcode. `tools/opcode_dispatch.py`
   follows its compare chains and jump table to recover that choice for 128 opcodes
   (`artifacts/reference/opcode_dispatch_v3.json`).
4. Five opcodes need corrections or are missing from `neuron-dbg`; they were fitted on the
   corpus and are listed with their evidence in `artifacts/reference/isa_layout_supplement.json`.
5. `tools/verify_isa.py` decodes the corpus with the result and checks every decoded wait,
   update, access pattern, channel count and named numeric field against the profiler's own
   disassembly. Every non-pseudo opcode in the corpus passes 100% (the two cases below 1.000
   are known checker limits: a register-addressed operand and a list-valued field).

`tools/isa.py` is the resulting decoder/encoder. The per-opcode table is
[`generated/opcodes.md`](generated/opcodes.md), the per-struct layouts are
[`generated/structs.md`](generated/structs.md) and the enum values are
[`generated/enums.md`](generated/enums.md).

## 6.2 Slot size

Every instruction slot is **64 bytes**. Slot *i* of an engine stream starts at byte `64 * i`.
**Verified** (all 2,051,467 slots of the corpus; every asm debug file has exactly one entry per
slot).

## 6.3 Header (bytes 0-3)

| Byte | Field | Value |
|---|---|---|
| 0 | `header_opcode` | The opcode. Selects the struct for bytes 4-63. |
| 1 | `header_inst_word_len` | 16 (the slot length in 4-byte words). In a `PSEUDO_EXTENSION` slot this byte holds flags instead. |
| 2 | `header_debug_cmd` | 0 |
| 3 | `header_debug_hint` | 0 |

**Observed**: bytes 1-3 are 16, 0, 0 in all 2,043,514 non-extension slots. Use these values for
new slots.

## 6.4 Events (bytes 4-11)

Every instruction except `PSEUDO_EXTENSION` has the same 8-byte events block. It holds the
instruction's **semaphore wait** and **semaphore update**. **Verified.**

| Byte | Field | Meaning |
|---|---|---|
| 4 | `events_wait_mode` | When the instruction may start (table below). 0 = no wait. |
| 5 | `events_wait_idx` | Semaphore (or event) number to wait on. |
| 6 | `events_update_mode` | What to do to a semaphore (or event) when the instruction reads its inputs or completes. 0 = nothing. |
| 7 | `events_update_idx` | Semaphore (or event) number to update. |
| 8-11 | `events_semaphore_value` | u32. The value the wait compares with. Also the operand of `ADD_IMM` / `SUB_IMM` / `WR_IMM` updates. |

There is only **one** value field. An instruction that waits for `S[a] >= v` and also does an
immediate update uses `v` for both. The common case, `SEM_INC_COMPLETE` (+1), needs no value.

Wait modes (`WAIT_MODE`):

| Value | Name | Profiler text |
|---|---|---|
| 0 | `NONE` | |
| 1 | `WAIT_FOR_SEM_EQ_IMM` | `S[i]==v` |
| 2 / 3 | `WAIT_FOR_SEM_LT_IMM` / `_LE_IMM` | `S[i]<v`, `S[i]<=v` |
| 4 / 5 | `WAIT_FOR_SEM_GT_IMM` / `_GE_IMM` | `S[i]>v`, `S[i]>=v` |
| 6 / 7 | `WAIT_FOR_EVT_SET` / `_SET_THEN_CLEAR` | (event waits; only the runtime prologue uses events) |
| 14 / 15 | `WAIT_FOR_EVT_CLEAR` / `_CLEAR_THEN_SET` | |
| 129-133 | `WAIT_FOR_SEM_*_REG` | compare with a register instead of the value |

Update modes (`UPDATE_MODE`):

| Value | Name | Profiler text |
|---|---|---|
| 0 | `NONE` | |
| 19 | `SEM_INC_COMPLETE` | `S[i]++@complete` |
| 20 | `SEM_DEC_COMPLETE` | `S[i]--@complete` |
| 21 / 23 / 25 | `SEM_ADD_IMM_COMPLETE` / `SEM_SUB_IMM_COMPLETE` / `SEM_WR_IMM_COMPLETE` | `S[i]+=v@complete`, ... |
| 3 / 4 / 5 / 7 / 9 | `SEM_INC_READ`, `SEM_DEC_READ`, `SEM_ADD_IMM_READ`, `SEM_SUB_IMM_READ`, `SEM_WR_IMM_READ` | same, when the inputs have been read |
| 17 / 18 | `EVT_SET_COMPLETE` / `EVT_CLR_COMPLETE` | |
| 1 / 2 | `EVT_SET_READ` / `EVT_CLR_READ` | |

The compiled kernels use **only** wait mode 5 (`>=`; 635,498 slots) and update mode 19
(`++@complete`; 397,715 slots); every other slot has mode 0. No NEFF instruction waits on or
updates an event; only the runtime prologue and epilogue do. **Observed** (all 2,043,514
non-extension slots). Chapter 10 explains how the values are chosen.

## 6.5 How the rest of the slot is named

Struct names say which operands the instruction has: `s<dims>` per source tensor, `d<dims>` per
destination tensor, then a kind. Examples:

| Struct | Reads as | Used by |
|---|---|---|
| `S3S3D3_TT` | two 3-D sources, one 3-D destination, tensor-tensor | `TENSOR_TENSOR_ARITH_OP`, `_BITVEC_OP`, `COPY_PREDICATED` |
| `S3D3_TS` | 3-D source, 3-D destination, tensor-scalar | `TENSOR_SCALAR_*`, `TENSOR_SCALAR_CACHE_*` |
| `S2S2D2_STT` | scalar-tensor-tensor | `SCALAR_TENSOR_TENSOR_*`, `SELECT_REDUCE`, `TENSOR_TENSOR_SCAN_ARITH` |
| `S4D4_TR` | 4-D source and destination, tensor-reduce | `TENSOR_REDUCE_*`, `COPY`, `CAST`, `RECIPROCAL`, `STREAM_SHUFFLE` |
| `S3D3_AC` | activation | `ACTIVATE` |
| `S3D3_MM` / `S3_LW` | matmul / load weights | `MATMUL` / `LDWEIGHTS` |
| `CTRL_*` | control (no tensors) | `EVENT_SEMAPHORE` (`ES`), `NOP`/`DRAIN`/`ACTIVATION_TABLE_LOAD` (`NO`), `WRITE` (`WR`), `ALU_OP` (`AL`), `COMPARE_BRANCH` (`BR`), `MOVE` (`MV`) |
| `PSEUDO_*` | pseudo-instructions | chapter 8 |

## 6.6 Building blocks

| Block | Wire size | Layout |
|---|---|---|
| `ADDR4` (start address) | 4 | u32 address; top byte is the marker (chapter 5.2) |
| `TENSOR2D` / `TENSOR3D` / `TENSOR4D` | 12 / 16 / 20 | `ADDR4`, then n x i16 `step_elem`, then n x u16 `num_elem` |
| `DTYPE_PAIR` | 1 | low nibble `dtype_lo`, high nibble `dtype_hi`. In `in0_in1_dtype` the low nibble is input 0 and the high nibble input 1 (**Observed**: HCCollapse's scalar-tensor-tensor has bf16 `src0` and fp32 `src1`, encoded `0xa6`). |
| `DTYPE` | 1 | dtype code (6.7) |
| `ALU_OP` | 1 | ALU operation code (6.8) |
| `num_active_channels` | 1 | number of partitions the instruction works on |
| `IMM_VAL_INST_FIELD` (`imm0`, `imm1`) | 4 | union: `imm_arith_fp32` (float32), `imm_bitvec_uint32` / `_int32` / `_uint16[2]`, `imm_ptr` (SBUF byte offset), `imm_reg` (register). The sibling `imm*_src` field says which member applies. |
| `ADDR8` / `PSEUDO_ADDR8` | 8 | used by DMA instructions; chapter 8 |

**Verified** (vendor layouts; checked on the corpus).

## 6.7 Data types (`DTYPE`)

| Code | Name | Code | Name |
|---|---|---|---|
| 1 | `UINT64` | 9 | `UINT32` |
| 2 | `INT8` | 10 | `FP32` |
| 3 | `UINT8` | 11 | `FP32R` |
| 4 | `INT16` | 12 | `INT64` |
| 5 | `UINT16` | 13 | `FP8_EXP3` |
| 6 | `BFLOAT16` | 14 | `FP8_EXP4` (printed `fp8_exp4`) |
| 7 | `FP16` | 15 | `FP8_EXP5` (printed `fp8_exp5`) |
| 8 | `INT32` | 16 | `FP4_EXP2` |

A 4-bit dtype field (in a `DTYPE_PAIR`) can only hold codes 0-15.

## 6.8 ALU operations (`ALU_OP`, selection)

| Code | Name | Code | Name |
|---|---|---|---|
| 0 | `BYPASS` | 10-12 | `BITWISE_AND/OR/XOR` |
| 1 | `BITWISE_NOT` | 13-15 | `LOGICAL_AND/OR/XOR` |
| 2 / 3 | `ARITH_SHIFT_LEFT/RIGHT` | 16 / 17 | `LOGICAL_SHIFT_LEFT/RIGHT` |
| 4 | `ADD` | 18-22, 24 | `IS_EQ`, `IS_GT`, `IS_GE`, `IS_LE`, `IS_LT`, `IS_NE` |
| 5 | `SUBTRACT` | 23 / 25 | `ABSOLUTE_DIFF` / `ABSOLUTE_VALUE` |
| 6 | `MULT` | 26 / 27 | `POW` / `MOD` |
| 7 | `DIVIDE` | 29 | `RSQRT` |
| 8 / 9 | `MAX` / `MIN` | 196+ | integer variants (`ADD_INT`, `MULT_INT`, ...) |

The full list (97 values) is in [`generated/enums.md`](generated/enums.md). The profiler prints
`MULT` as `MULTIPLY`.

## 6.9 Decoding and encoding with `tools/isa.py`

```python
import sys; sys.path.insert(0, "neff_doc/tools")
import isa
data = open("sg00/DVE0.bin", "rb").read()
for idx, ins in isa.decode_stream(data):          # handles PSEUDO_EXTENSION context
    print(idx, ins.name, ins.struct, ins.as_dict())
ins = isa.decode(data[64:128])                    # one slot
ins.set("events_semaphore_value", 3)              # numbers ...
ins.set("op0", "ADD")                             # ... or enum names
new_slot = ins.encode()                           # 64 bytes; untouched bits stay as they were
```

Fields are named by their vendor path, for example
`src_mem_pattern_start_addr_addr_immediate`, `src_mem_pattern_step_elem` (a list),
`num_active_channels`, `op0`, `imm0_imm_ptr`. `isa.fields(struct)` lists them with offsets.

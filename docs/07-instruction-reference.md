# 7. Instruction families

This chapter explains what the fields of the common instructions mean. Exact offsets are in
[`generated/structs.md`](generated/structs.md); which struct each opcode uses and how often it
appears are in [`generated/opcodes.md`](generated/opcodes.md). Every family listed here passes
the decoder check against the profiler at 100% (chapter 6.1).

All compute instructions share three things: the events block (chapter 6.4),
`num_active_channels` (how many partitions, starting at the partition in the address), and
tensor operands made of an address plus an access pattern (chapter 5).

## 7.1 Element-wise and reduce on DVE, Pool and Activation

| Opcode | Struct | Operation |
|---|---|---|
| `TENSOR_TENSOR_ARITH_OP` (0x41), `_BITVEC_OP` (0x51) | `S3S3D3_TT` | `dst = src0 op src1` |
| `TENSOR_SCALAR_ARITH_OP` (0x43), `_BITVEC_OP` (0x53) | `S3D3_TS` | `dst = (src op0 imm0) op1 imm1` |
| `SCALAR_TENSOR_TENSOR_ARITH` (0x9D), `_BITVEC` (0x9E) | `S2S2D2_STT` | `dst = (src0 op0 imm0) op1 src1` |
| `TENSOR_REDUCE_ARITH_OP` (0x42) | `S4D4_TR` | reduce `src` along `op_dim` with `op` |
| `COPY` (0x46), `CAST` (0x47), `RECIPROCAL` (0x48) | `S4D4_TR` | copy, dtype conversion, 1/x |
| `MEMSET` (0x49) | `D4_MR` | fill `dst` with `set_value` |
| `IOTA` (0x7E) | `D4_IOTA` | write an index pattern (`channel_multiplier`, base) |
| `SELECT_REDUCE` (0xEA) | `S2S2D2_STT` layout | select-and-reduce (layout established from the corpus) |

Key fields:

* `op`, `op0`, `op1` — `ALU_OP` codes (chapter 6.8). `BYPASS` (0) passes the value through, so
  `op1 = BYPASS` makes a one-op tensor-scalar.
* `imm0_src`, `imm1_src` — where the scalar comes from: `INSTRUCTION_IMMEDIATE` (the value is in
  `imm0_imm_arith_fp32` / `imm0_imm_bitvec_uint32`), `POINTER_IMMEDIATE` (`imm0_imm_ptr` is a
  per-partition SBUF byte offset, chapter 5.5), or a register.
* `reverse_operands` — swap the operands of op0, op1 or both (`NONE`, `FIRST`, `SECOND`, `BOTH`),
  for non-commutative ops such as `SUBTRACT` and `DIVIDE`.
* `in_dtype` / `in0_in1_dtype` / `out_dtype` — element types; the result is computed in fp32 and
  rounded to `out_dtype` (**Inferred** from e01, where bf16 inputs with fp32 scalars matched an
  fp32 computation rounded once to bf16, bit for bit).
* `accumulator_cmd` — `IDLE`, `ZERO`, `ACCUMULATE`, `ZERO_ACCUMULATE`, `LOAD_ACCUMULATE`; used by
  the reduce-while-computing forms, read back with `DVE_READ_ACCUMULATOR` /
  `ACTIVATION_READ_ACCUMULATOR`.

The same opcode runs on more than one engine (for example `TENSOR_TENSOR_ARITH_OP` on DVE and
Pool, `TENSOR_SCALAR_ARITH_OP` on DVE, Activation and Pool). The engine is given by which stream
the slot is in. **Observed** (engine counts per opcode in `generated/opcodes.md`).

## 7.2 `ACTIVATE` (Activation engine)

`dst = func(src * scale + bias)`, optionally accumulated. Struct `S3D3_AC`.

* `activation_func` — the function id from the activation table set (chapter 3.5,
  [`generated/activation_functions.md`](generated/activation_functions.md)): 1 identity, 2 relu,
  7 exp, 29 rsqrt, 30 square, 31 sign, 33 abs, 36 silu, 128 copy, ... The function must be in a
  set this subgraph loads, and the set must be active (`ACTIVATION_TABLE_LOAD`). **Verified**
  against the NKI source (exp: about 46,000 slots from `nl.exp` calls carry id 7).
  The profiler prints wrong names for several ids on Trainium2 (`SOFTPLUS` for 7, `GELU` for 5);
  use the table set, not the profiler.
* `scale_src` + `scale_value_imm_fp32`, `bias_src` + `bias_imm_fp32` (or a bias pointer),
  `imm_src` + `imm_imm_fp32` — the affine part, each an immediate or a pointer.
* `accumulator_cmd` — accumulate across calls; read back with `ACTIVATION_READ_ACCUMULATOR`.

`nisa.tensor_scalar(..., engine=scalar)` is often compiled to an `ACTIVATE` with the identity
or copy function and a scale/bias. **Observed.**

## 7.3 Matrix multiply (PE)

A matmul is two instructions: `LDWEIGHTS` (0x01, `v3...S3_LW`) loads the stationary operand into
the 128x128 PE array, then `MATMUL` (0x02, `v2...S3D3_MM`) streams the moving operand through it
and writes PSUM. **Observed** (one `LDWEIGHTS` per `MATMUL` in almost all cases: 642k / 670k).

| Field | Meaning |
|---|---|
| `src_mem_pattern` | the operand in SBUF (`LDWEIGHTS`: stationary; `MATMUL`: moving) |
| `dst_mem_pattern` | PSUM destination (`MATMUL` only; address `0x2000000 + offset`) |
| `num_active_rows`, `num_active_cols` | how much of the PE array is used (the profiler prints `128*5`) |
| `row_grp`, `col_grp` | which row / column group of the array (4 groups of 32 each) |
| `in_dtype` | operand type |
| `perf_opt` | `NONE`, `DOUBLE_ROW`, `DOUBLE_COLUMN`, `DOUBLE_PIXEL`. fp8 (`FP8_EXP4`) matmuls in the corpus use `DOUBLE_ROW`; bf16 uses `NONE`. |
| `fp32_mode` | `NONE`, `LOW`, `HIGH`, `LOW_HIGH`; fp32 matmuls use `LOW_HIGH` |
| `psum_accumulate_flags` | bit 0 = first matmul of an accumulation group (zero PSUM first), bit 1 = last; 3 = a group of one, 0 = middle |
| `psum_zero_region` | size of the PSUM region a "first" matmul zeroes |
| `transpose_mode` (`LDWEIGHTS`) | load the weights transposed |

**Observed** (value distributions over the corpus; flag meanings from the vendor enum
`MatmulPsumAccumulateFlags`: `BeginMatmulGroup = 1`, `EndMatmulGroup = 2`).

## 7.4 Control instructions

| Opcode | Struct | Use |
|---|---|---|
| `EVENT_SEMAPHORE` (0xA0) | `CTRL_ES` | wait and/or update a semaphore, nothing else. The building block for extra waits and for neutralising an instruction (chapter 10.4). **Verified** (r05). |
| `NOP` (0xA4) | `CTRL_NO` | no operation; also carries a wait (the compiler's end-of-program `NOP` waits for the last DMA) |
| `DRAIN` (0xA2) | `CTRL_NO` | wait until this engine's earlier work is finished |
| `ACTIVATION_TABLE_LOAD` (0x23) | `CTRL_NO` | make table set `act_tbl_sel` active (chapter 3.5) |
| `SET_ORDERING_MODE` (0xB1) | `CTRL_OM` | always slot 0 of every stream; the profiler prints `ordering_mode=relaxed` |
| `ALU_OP` (0xA8) | `CTRL_AL` | register arithmetic: `dst = src0 op src1` on registers or an immediate |
| `MOVE` (0xA7) | `CTRL_MV` | move an immediate or a register into registers |
| `COMPARE_BRANCH` (0xA9) | `CTRL_BR` | conditional branch to a label (chapter 8.6) |
| `EVENT_SEMAPHORE_RANGE_CLEAR` (0xB0) | `CTRL_ER` | zero a range or a bitmask of semaphores |
| `WRITE` (0xA5) | `CTRL_WR` | write to a device address; appears only after runtime expansion |

## 7.5 Building a new slot

1. Start from an existing slot of the same opcode (copy its 64 bytes), or from 64 zero bytes with
   `header_opcode`, `header_inst_word_len = 16` set.
2. Set fields with `Instruction.set()`. Leave fields you do not understand as copied.
3. Set the events block for the dependencies the instruction really has (chapter 10).
4. Check the result against the ISA constraints you know: start partition + channels <= 128,
   start partition a multiple of 32, SBUF offsets in 0x4000..0x37FFF, PSUM offsets < 0x4000,
   dtype codes valid for the opcode (copy them from a working instruction of the same engine).

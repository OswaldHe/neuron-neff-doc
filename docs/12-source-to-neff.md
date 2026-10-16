# 12. From NKI source to NEFF and back

## 12.1 The chain

```
NKI source line   --(backend debug attrs: kernel_file, kernel_line)-->   BIR instruction
BIR instruction   --(asm debug entry i: parent_ids)-->                    slot i of <Engine>0.bin
slot i bytes      --(tools/isa.py)-->                                     decoded fields
slot i            --(runtime: pseudo expansion, prologue offset)-->       runtime instruction(s)
runtime instr.    --(neuron-explorer: compiler_pc, opcode, operands)-->   profiler text and timing
```

Each link was checked on the corpus: the asm level has exactly one entry per slot; the BIR name
of every slot equals the profiler's `bir_instruction_name` for that slot (after the GID-load
realignment, chapter 11.4); and the decoded fields reproduce the profiler's operand text
(chapter 6.1). **Verified.**

`tools/corpus.py` builds this join for every slot of a NEFF
(`artifacts/corpus/<tag>.parquet`: raw bytes, opcode, BIR name, NKI file and line, runtime
expansion and profiler text).

## 12.2 Finding the slots of an NKI line

```python
import pandas as pd
df = pd.read_parquet("neff_doc/artifacts/corpus/HCCollapse.parquet")
df[(df.kernel_line == 150) & (df.sg == 0)][["engine", "idx", "op_name", "bir_name", "p_operands"]]
```

Without a profile, read the two debug files directly (`tools/dbginfo.py`): map each asm entry's
`parent_ids[0]` to a backend record and read its `kernel_line`.

## 12.3 Which instructions an NKI call becomes

[`generated/nki_to_isa.md`](generated/nki_to_isa.md) lists, for 53 NKI calls, the engine and
opcode of every slot they produced in the 14 kernels. The main patterns (**Observed**):

| NKI call | Becomes | Notes |
|---|---|---|
| `nisa.nc_matmul` | PE `LDWEIGHTS` + `MATMUL` | one weights load per matmul (642k / 670k slots) |
| `nisa.nc_transpose` | PE `LDWEIGHTS` + `MATMUL` | transpose by a matmul with an identity matrix; the identity is a `file` constant (20-Attention) or is built in SBUF with DVE `MEMSET` + Pool `TENSOR_SCALAR_AFFINE_SELECT` |
| `nisa.activation` | Activation `ACTIVATE`, `ACTIVATION_READ_ACCUMULATOR`, `ACTIVATION_TABLE_LOAD` | read-accumulator when a reduce is requested |
| `nisa.tensor_tensor`, `nl.multiply`/`nl.add` on two tiles | DVE (sometimes Pool) `TENSOR_TENSOR_*` | |
| `nisa.tensor_scalar` | DVE `TENSOR_SCALAR_*`, or Activation `ACTIVATE` / `TENSOR_SCALAR_ARITH_OP` when placed on the Activation engine | `engine=` decides |
| `nisa.scalar_tensor_tensor` | DVE `SCALAR_TENSOR_TENSOR_*` | |
| `nisa.tensor_copy`, `nl.copy` | `COPY` (same dtype) or `CAST` (dtype change) on DVE or Activation | |
| `nisa.tensor_reduce`, `nl.sum`, `nl.max` | DVE `TENSOR_REDUCE_ARITH_OP` (or `ACTIVATE` with accumulate) | |
| `nisa.select_reduce`, `nisa.tensor_scalar_reduce` | DVE `SELECT_REDUCE` / `TENSOR_SCALAR_CACHE_REDUCE` + `DVE_READ_ACCUMULATOR` | |
| `nisa.dma_copy`, `nl.load`, `nl.store` | queued DMA (`PSEUDO_DMATRIGGER` + `SP0.json` entry) or DGE DMA (`PSEUDO_DMA_DIRECT2D`) | DGE ones often come with Pool `ALU_OP` / `PSEUDO_TENSOR_LOAD` address arithmetic |
| `nisa.dma_transpose`, `nl.load_transpose2d` | queued DMA with `transpose_op: WZXY` descriptors | |
| `nisa.memset` | DVE `MEMSET` | |
| `nisa.topk` | Pool `TOP_K` | |
| `nisa.core_barrier` | Pool `PSEUDO_CORE_BARRIER` | |
| `ncc.all_gather`, `ncc.all_reduce` | Pool `PSEUDO_TRIGGER_COLLECTIVE2` + `PSEUDO_EXTENSION` | |
| `nisa.register_store` / rank-dependent code | `PSEUDO_GID_LOAD`, `PSEUDO_TENSOR_STORE` | |

Every call also produces `EVENT_SEMAPHORE` slots when the instruction has more dependencies than
its one wait field can hold (chapter 11.3).

## 12.4 How NKI arguments land in the fields

From the vendor's own NKI-to-ISA tables (`neuronxcc/include/isa/*_info`, dumped in
`artifacts/reference/isa_info_modules.txt`) and the corpus:

| NKI argument | ISA field |
|---|---|
| tile `dst=` / `data=` / `src=` | `dst_mem_pattern` / `src_mem_pattern` (address = partition * 0x40000 + offset; pattern in elements) |
| tile's partition count | `num_active_channels` |
| `op0=`, `op1=` | `op0`, `op1` (`ALU_OP`) |
| `operand0=` a Python constant | `imm0_src = INSTRUCTION_IMMEDIATE`, value in `imm0_imm_arith_fp32` (or the bitvec member) |
| `operand0=` a per-partition tile `[:, k:k+1]` | `imm0_src = POINTER_IMMEDIATE`, `imm0_imm_ptr` = byte offset of the scalar (chapter 5.5) |
| `reverse0/1=` | `reverse_operands` (`TENS_SCALAR_REV_OPS`) |
| `dtype=` of the result | `out_dtype` |
| `engine=` | which engine stream the slot is in |

**Verified** for `operand0` as a tile (e01) and for dtypes, ops and channels (corpus checks);
**Observed** for the rest.

## 12.5 Worked example

[`generated/walkthrough_hccollapse.md`](generated/walkthrough_hccollapse.md) lists every slot of
HCCollapse `sg00` with its decoded fields, BIR name and source line. Read it next to
`optimization-runs-HCCollapse/hc_collapse-full/source.py`:

* line 138 `nisa.dma_copy(dst=pm, ...)` → SP slot 2, a trigger of transfer 1 of `qSPIO0`
  (16 bytes per token to SBUF 0x37F00);
* line 145 (twice, `STAGES = (320, 960)`) → SP slots 1 and 3, transfers 0 and 2;
* lines 147-157 → DVE slots 1-8 (two `TENSOR_SCALAR` and six `SCALAR_TENSOR_TENSOR`, waiting on
  the DMA semaphores 27/28 and on S[6]);
* line 160 → SP slots 4-5 (`EVENT_SEMAPHORE` waiting for the core barrier, then the trigger of the
  `qSPSpillReload0` transfer `yg -> cc_src`, waiting S[6] >= 7);
* line 164 `ncc.all_gather` → Pool slots 2-5 (core barrier, `PSEUDO_TRIGGER_COLLECTIVE2` +
  extension, core barrier);
* line 172 (loop of 4) → SP slots 6-9 (transfers 3-6, the first waiting for the collective,
  S[8] >= 1) and slot 10 (the final `NOP` waiting for the last transfer).

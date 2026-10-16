# 16. Open questions and resolved conflicts

## 16.1 Conflicts found and how they were resolved

Each of these was an early reading that a later check contradicted. The documents now state the
checked version only.

| Topic | First reading | Evidence that settled it | Now |
|---|---|---|---|
| Which struct an opcode uses | Scoring every struct against the profiler text | Many ties and implausible winners (`EVENT_SEMAPHORE` -> a random-number struct). neuron-dbg's own disassembler dispatch, read from its machine code, gives a different, consistent answer that passes all checks. | Vendor dispatch + documented supplement (chapter 6.1) |
| `PSEUDO_DMA_DIRECT2D` struct | Dispatch said `PSEUDO_DMA_EXT_STRUCT` | That decoder handles both 0xD4 and its extension slot; field-by-field comparison with the runtime text matches `PSEUDO_DMA_DIRECT2D_STRUCT` | Supplement entry for 0xD4 |
| Profiler `compiler_pc` | Equal to the slot index | In 6 NEFFs every label after `PSEUDO_GID_LOAD` is one ahead (runtime opcodes disagree with the slot until shifted) | Profiler quirk; realigned (chapter 11.4) |
| sg00 vs sg01 | Same engine streams | Byte comparison: different in 12 of 14 NEFFs | Separate programs (chapter 1.5, 4.5) |
| Usable SBUF per partition | `STATE_BUF_PARTITION_ACTIVE_SIZE = 0x30000` (vendor constant) | Trainium2 NEFFs use offsets up to 0x37FD0 in thousands of instructions | Observed range 0x4000-0x37FFF (chapter 5.3) |
| PSUM banks | `PSUM_BUF_BANK_SIZE = 0x400`, 16 banks (vendor constant) | Usage clusters at 2 KiB boundaries | 2 KiB banks, 16 KiB per partition (chapter 5.3) |
| ACTIVATE function names | Profiler names (`SOFTPLUS`, `GELU`, ...) | NKI source: id 7 comes from `nl.exp` (about 46,000 slots), 30 from `nl.square`, 33 from `nl.abs`, ... matching the table-set `func_id` | Table-set ids (chapter 7.2) |
| Same-engine waits | One removal (r10) kept the output right | Removing all of them (o01): wrong output on DVE, `NRT_EXEC_OOB` on Pool | Required (chapter 10.5) |
| Pseudo opcode range | "193 and above are pseudo" | 229-240 (`TENSOR_TENSOR_SCAN_ARITH`, `INDIRECT_COPY`, `EXTENDED_INST`, ...) are not pseudo | Name starts with `PSEUDO_` |
| `TOP_K`, `SELECT_REDUCE` | Inferred structs that only matched the header | Fitted on raw bytes vs profiler text: `SELECT_REDUCE` = scalar-tensor-tensor layout (1,200/1,200 patterns), `TOP_K` = custom layout | Supplement entries |

## 16.2 Open questions

| Question | What is known |
|---|---|
| Header 0x1DC and 0x1E0 | Both 1 in every NEFF; neuron-explorer ignores them; not tested on the runtime |
| Header bits 23-27, 29-31 | Unnamed in this neuron-explorer; 0 in every NEFF |
| Does the runtime need the feature bits to match the program? | Not tested (chapter 2.2) |
| `num_tiling_dimensions` in DMA descriptors | 1 or 2; meaning not established |
| DMA transpose descriptors (`transpose_op: WZXY`) | Seen only in attention kernels; field semantics not tested |
| `PSEUDO_GID_LOAD` bytes 12-15 | Fitted layout marks them unverified |
| `TOP_K` output fields | Two (address, step, count) groups at +40 and +48; the profiler does not print them |
| DGE-table indexing | `table_index` is assumed to be a position in the `dge-table` variable's list; not tested |
| Which DVE dtype combinations run faster | Not measured (chapter 14.3) |
| Order of package members | Kept as compiled; whether the runtime cares was not tested |
| Engine speeds for the same opcode | Not measured; needed before moving work between engines |
| Event usage | NEFF instructions never use events; the runtime does. Rules for events not studied. |

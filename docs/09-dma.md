# 9. DMA

There are two kinds of DMA in a NEFF.

| Kind | Started by | Described in | Descriptors made |
|---|---|---|---|
| **Queued DMA** | `PSEUDO_DMATRIGGER` (SP only) | `sgNN/SP0.json` `dma` list | by the runtime, at load time, from the JSON |
| **DGE DMA** | `PSEUDO_DMA_DIRECT2D` (SP, Activation, Pool; plus an optional `PSEUDO_EXTENSION` slot) | the instruction itself | by the DGE hardware, at run time, from the instruction |

All 45,909 queued DMA transfers of the corpus are listed in `SP0.json`, and there are exactly
45,909 `PSEUDO_DMATRIGGER` slots, all on SP: one trigger per transfer. The other engine
descriptions have empty `dma` lists. DGE DMA instructions: 18,062 on SP, 16,216 on Activation,
14,811 on Pool. **Observed.**

## 9.1 Queued DMA transfers (`SP0.json` → `dma`)

```json
{"id": 1, "queue": "qSPIO0", "#comment": "I-0_inst__I-4-0-block-1", "semaphore": 27,
 "desc": [ { ...descriptor... } ]}
```

| Key | Meaning |
|---|---|
| `id` | Transfer number within its queue. `PSEUDO_DMATRIGGER`'s `count_union_block_id` names it. **Verified** (r09: editing transfer 3 and 4 changed exactly what the triggers with block ids 3 and 4 moved). |
| `queue` | The DMA queue (chapter 4.3). With the `queue-set-instances` feature the key is `instance_name` instead. |
| `#comment` | The BIR name of the transfer. It matches the `name` in `debug_info_backend_DMA.dbg`, which gives the NKI source line. |
| `semaphore` | The semaphore the transfer signals when it completes. Always `semaphore_set[id % 16]` of its queue. **Verified** (all 45,909 transfers). |
| `desc` | One or more descriptors, done in order. |

When a transfer completes, its semaphore grows by the queue's `num_queues` (16). Waiting for
`S[sem] >= 16 * k` therefore waits for the *k*-th transfer that uses `sem`. **Verified** by
semaphore accounting on all 28 subgraphs (chapter 10).

## 9.2 Descriptors

```json
{"op": "copy", "num_tiling_dimensions": 1,
 "from": "input1", "from_dtype": "bfloat16", "from_off": 640,
 "from_sizes": [1920, 4, 128], "from_steps": [1, 2560, 10240],
 "to": "SB", "to_dtype": "bfloat16", "to_off": 218880,
 "to_sizes": [1920, 4, 128], "to_steps": [1, 1920, 262144]}
```

| Key | Meaning |
|---|---|
| `op` | `copy`, `cast` (convert `from_dtype` to `to_dtype` on the way) or `add` (sum the sources in `from_arr`). Absent on descriptors that carry `transpose_*` keys. |
| `from` / `to` | A variable name from the definition file. `SB` is the state buffer; `SB{1}` is the state buffer of physical core 1 (seen as a source of an `add`). |
| `from_off` / `to_off` | Byte offset in that variable. For `SB` it is `partition * 262144 + offset`. |
| `from_sizes` / `to_sizes` | The access pattern. Entry 0 is a **byte count** (one contiguous run); the others are repeat counts. |
| `from_steps` / `to_steps` | Byte step of each dimension. Entry 0 is always 1. For `SB` the partition dimension has step 262144. |
| `from_dtype` / `to_dtype` | Element types. |
| `num_tiling_dimensions` | 1 or 2. **Observed**; meaning not established. |
| `from_arr` | For `op: add`: a list of sources (each with `from`, `from_dtype`, `from_off`, `from_sizes`, `from_steps`) summed into `to`. |
| `transpose_shape`, `transpose_op`, `transpose_element_size` | A DMA-engine transpose (`transpose_op` is always `WZXY`, element size 2, shapes `[1,1,128,128]`, `[1,1,256,128]`, `[1,1,512,128]`). Only in the attention kernels. |

**Verified**: byte units and the meaning of `from_off`, `from_sizes[0]` and the steps follow from
HCCollapse, where `input1` is `(1,128,4,1280)` bf16 (`2560 = 1280*2`, `10240 = 4*1280*2`) and the
two stage loads move `640 = 320*2` and `1920 = 960*2` bytes per row. Experiment r09 (swapping
two `from_off` values moved two output blocks exactly as predicted) verified that the runtime
reads these fields at load time.

Example: HCCollapse `I-0` loads the per-token scalars:
`from input0 off 0 sizes [16, 128] steps [1, 16]` → `to SB` (the state buffer) `off 229120 (0x37F00) sizes [16, 128]
steps [1, 262144]`: 16 bytes (4 fp32) per token into 128 partitions at offset 0x37F00.

## 9.3 Starting a queued transfer: `PSEUDO_DMATRIGGER`

| Field | Meaning |
|---|---|
| `dma_queue_name` | 32 bytes, NUL-padded ASCII queue name (or queue instance name) |
| `count_union_block_id` | The transfer `id` to start |
| `use_raw_count` | 0 in all 45,909 triggers |
| events | Wait: what must be ready before the transfer starts (the data to read). Update: none in all 45,909 triggers; completion is signalled by the transfer's own semaphore. |

The runtime turns each trigger into `WRITE` instructions to the queue's DMA ring registers
(the profiler shows `WRITE data=0x21,0x21 dst=0x8010020a2038,0x801002082038`). **Verified**
(profiler expansion of every trigger).

## 9.4 DGE DMA: `PSEUDO_DMA_DIRECT2D`

The instruction itself describes a 2-D transfer. The DGE hardware turns it into descriptors at
run time, so nothing is listed in the JSON.

| Field | Meaning |
|---|---|
| `semaphore`, `sem_increment` | The semaphore this transfer adds to when it completes, and by how much (16 in all 49,089 DGE instructions). Waiters wait for multiples of `sem_increment`. |
| `dge_op` | `DMA_DIRECT2D`, `DMA_INDIRECT1D`, ... (`DGE_OPCODE`) |
| `src_start_addr_*`, `dst_start_addr_*` | 8-byte address (`PSEUDO_ADDR8`), see 9.5 |
| `src_step_elem[2]`, `src_num_elem[2]`, `src_elem_size` | Source pattern: `src_num_elem[d]` rows of `src_elem_size` contiguous bytes, row *d* at `src_step_elem[d]` bytes (state-buffer partition stride 262144). `src_elem_size` is 16-bit, so a 65536-byte row is stored as **0**: 00-Attention-B's output writes have `elem_size = 0`, `step = [65536, 1]`, `num = [160, 1]` and move 160 x 65536 = 0xA00000 bytes (1024 rows of 5120 bf16). **Verified** (experiment a01: the data arrives, and swapping two such transfers' sources swaps exactly 1024-row blocks) |
| `dst_step_elem[2]`, `dst_num_elem[2]`, `dst_elem_size` | Destination pattern |
| `in_dtype`, `out_dtype` | Element types |
| `dma_configs_*` | Priority class, 2-D tile transpose, config register enable |
| `src/dst_bound_reg_bc_*` | Optional bound-check register |

The profiler prints the runtime form as
`DMA_DIRECT2D semaphore=62 sem_increment=16 src_elem_size=10240 dst_elem_size=10240
src_pattern=[262144,1][128,1] dst_pattern=[10240,1][128,1] src_addr_imm=0x80200000b800 ...`.
**Verified**: semaphore, increment, element sizes, both patterns and the table index match the
decoded slot in about 2,780 compared instructions.

The next slot may be a `PSEUDO_EXTENSION` (`v3...PSEUDO_DMA_EXT_STRUCT`) with gather/scatter
settings: index addresses (`src_idx_start_addr`, `dst_idx_start_addr`), `idx_num_active_channels`,
`compute_op` (`NONE`, `ADD`, ... for accumulate-on-write), `flags_*` (indirect mode, gather and
scatter dimension), tile sizes and bounds. The runtime consumes it with the instruction before.

## 9.5 Pseudo-instruction addresses (`PSEUDO_ADDR8`)

8 bytes. Byte 7 is the marker; its bits say which form the other bytes take
(`hw_constants_v3.json`: `PSEUDO_ADDR8_MARKER_*`). **Verified** (marker constants from the vendor;
forms checked on the corpus).

| Marker bit | Form | Fields |
|---|---|---|
| none (0x00) | immediate | `addr_immediate` (u64) |
| 0x10 `VAR_ID` | variable | `var_offset` (u32, bytes 0-3), `var_var_id_lo` (u16, bytes 4-5), `var_var_id_hi` (byte 6): `var_id = lo + (hi << 16)` |
| 0x20 `TBL` | DGE table | `table_offset` (u32), `table_index` (u16): entry `table_index` of the `dge-table` variable |
| 0x80 `REG` | register | `reg_lo`, `reg_hi` |
| 0x08 / 0x04 | offset from a register / wide offset | |

Example: `src = var[0] + 0xB800` is the state buffer at offset 0xB800 (partition 0);
`dst = var[25] + 0x140000` is byte 0x140000 of variable 25 (`inst__I-23-0:wqkv_sg0000`, DRAM).

## 9.6 Changing a DMA

* Move data elsewhere or reshape a queued transfer: edit its descriptor in `SP0.json`
  (variable, offsets, sizes, steps). Keep `sizes[0]` a byte count and stay inside the variable's
  `size`. **Verified** (r09).
* Change a DGE transfer: edit the `PSEUDO_DMA_DIRECT2D` fields with `tools/isa.py`. **Verified**
  (a01: exchanging the `src_start_addr_addr_var_offset` of two of 00-Attention-B's output writes
  exchanged exactly those two blocks of the output, bit for bit, with no other change).
* If you change how many transfers signal a semaphore, re-balance the waits (chapter 10).
* Do not leave a trigger pointing at a transfer id that does not exist in the JSON.

# 4. The definition file (`sgNN/def.json`)

One definition file per subgraph. It declares every memory object the subgraph's instructions
and DMA transfers refer to, the DMA queues, the collective setup and some runtime settings.

## 4.1 Top-level keys

| Key | Value seen | Meaning | Status |
|---|---|---|---|
| `name`, `version` | `definition`, `0.6` | Format name and version | Observed |
| `#rank_world_size` | 0 or 1 | Keys starting with `#` look like comments (DMA entries also carry `#comment`) | Observed |
| `runtime_event_count` | 12 | Events reserved for the runtime | Observed (meaning Inferred) |
| `runtime_semaphore_count` | 3 | Semaphores reserved for the runtime. No NEFF instruction uses S[0]..S[2]; the runtime prologue does. | Observed |
| `act`, `dve`, `pe`, `pool`, `sp` | `Activation0.json`, ... | Engine description file per engine | Observed |
| `act_instr`, ... `sp_instr` | `Activation0.bin`, ... | Engine stream file per engine | Observed |
| `*_dbg`, `*_asm_dbg`, `dma_dbg` | lists of `.dbg` names | Debug info files. **Optional**: removing them all (and the files) does not affect loading, results or profiling. | Verified (experiment r03) |
| `var` | object | Variables, see 4.2 | |
| `dma_queue` | object | DMA queues, see 4.3 | |
| `cc_streams` | `{"num_streams": 1}` | Collective streams | Observed |
| `replica_groups` | `[[[0,1,2,3]]]` | The ranks that take part in collectives | Observed |
| `runtime_statebuffer_reservation` | see 4.4 | State-buffer bytes the runtime uses | Observed |
| `fp8_conversion_config` | two variants | fp8 rounding and range settings (`fp8_range_extension`, `ocp_sat`, `fp8_sis`, `fp8_sns`, `emax_fp8_e5m2`, `emax_fp8_e4m3`) | Observed |
| `neff_features` | `neff_feature_vnc`, `neff_coalesced_ccops`, `neff_queue_set_instances` | Features this subgraph uses; they match header feature bits 10, 3 and 4 | Observed |
| `debug_info` | `{}` | Always empty | Observed |

## 4.2 Variables (`var`)

`var` maps a variable name to its declaration. Every declaration has a `type` and a `var_id`.
Instructions that address memory through a variable (DMA pseudo-instructions, chapter 8) use the
`var_id`, not the name. **Verified** (a DGE DMA's destination `var[25]` is the variable with
`var_id` 25, `inst__I-23-0:wqkv_sg0000`, and the runtime resolves it to a DRAM address).

| `type` | Extra keys | What it is |
|---|---|---|
| `state-buffer` | none | The whole state buffer. Always named `SB`, always `var_id` 0. |
| `input` | `size`, `ops` | A program input (`input0`, `input1`, ...). `size` in bytes. |
| `output` | `size`, `ops` | A program output (`output0`, ...). |
| `virtual` | `size`, `backing_variable_off`, `ops` | A DRAM scratch tensor. It lives at byte `backing_variable_off` of one scratch area the runtime allocates for the NEFF. Names look like `inst__I-23-0:wqkv_sg0000` (`I-23-0` is the NKI kernel call, `wqkv` the NKI tensor name, `sg0000` the subgraph). |
| `remote` | `size`, `backing_variable_off`, `ops` | A tensor owned by another physical core of the same Logical NeuronCore. Name ends in `{0}` (owned by core 0). Seen in `sg01` for the collective buffers that `sg00` declares as `virtual`. |
| `file` | `size`, `file_name`, `alignment`, `fabric_path`, `ops` | A constant stored in the payload as a `.npy` file. |
| `list` | `list` (var ids) | A group of variables, e.g. `CCGrp0` = the buffers of one coalesced collective. |
| `dge-table` | `list` (var ids) | The table that DGE DMA instructions index with `table_index` (chapter 8). Position *i* in `list` is table index *i* (**Inferred** from the names; not tested). |

The `ops` lists were empty in every studied NEFF. **Observed.**

How big the scratch area is: the largest `backing_variable_off + size` over the `virtual`
variables (this is what `neuronxcc/starfish/bin/analyze_neff_artifacts.py` reports as "TmpBuf").
**Inferred.**

## 4.3 DMA queues (`dma_queue`)

```json
"qSPIO0": {"type": "in", "num_queues": 16, "owner": "sp", "fabric_path": "main",
           "semaphore_set": [26, 27, ..., 41]}
```

| Key | Meaning | Status |
|---|---|---|
| `type` | `in` (program inputs and outputs), `data` (spill/reload and scratch traffic), `dynamic` (DGE: descriptors made at run time by instructions) | Observed |
| `num_queues` | Hardware DMA rings behind this queue. Always 16. A queued DMA transfer's completion adds `num_queues` to its semaphore. | Verified by semaphore accounting on all 28 subgraphs (chapter 10) |
| `owner` | The engine that starts transfers on this queue: `sp`, `act` or `pool` | Observed |
| `fabric_path` | Always `main` | Observed |
| `semaphore_set` | The semaphores this queue's transfers may signal. Each DMA transfer names one of them (`semaphore` in the engine description). | Observed |
| `dynamic_dma` | Present on `dynamic` queues | Observed |
| `queue_instances` | With the `queue-set-instances` feature: names of queue instances (`qSPSpillReload0_1_0`). DMA transfers then name an `instance_name` instead of a `queue`. | Observed |

Queue names seen: `qSPIO0`, `qSPSpillReload0`, `qActIO0`, `qSyncDynamicHW`, `qActDynamicHW`,
`qGpSimdDynamic`, ... The profiler renames `SP` to `Sync` (`qSPIO0` shows up as `qSyncIO0`).

## 4.4 Runtime state-buffer reservation

```json
"runtime_statebuffer_reservation": [
  {"base_partition": 0, "offset": 229368, "partitions": 128, "size": 8, "type": "evtaccel"}]
```

The runtime uses 8 bytes at offset 0x37FF8 of all 128 partitions. Do not place data there.
**Observed** (identical in all 28 subgraphs).

## 4.5 The two subgraphs of one NEFF

`sg00` and `sg01` are separate programs for the two physical cores. Their engine streams are
identical only when the kernel does not split work between the cores (2 of the 14 studied NEFFs).
Their definition files differ in the collective buffers: `sg00` declares them `virtual`, `sg01`
declares them `remote` with a `{0}` suffix. **Observed.** When you edit an instruction, find the
matching instruction in the other subgraph (by BIR name in the debug info, chapter 11) and decide
whether it needs the same edit; do not assume the same slot index.

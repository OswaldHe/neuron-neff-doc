# 3. Package members other than the subgraph programs

This chapter covers the small top-level files and the table files. The definition file is
[chapter 4](04-definition-file.md); the engine streams are chapters 6 to 8.

## 3.1 `neff.json` — the graph file

A TVM graph-runtime graph with one compute node. **Observed** in all 14 NEFFs.

| Key | Meaning |
|---|---|
| `nodes` | One `"op": "null"` node per program input (`input0`, `input1`, ...), then one `"op": "tvm_op"` node named `sg_coreV10` whose `attrs.func_name` is `__kelf` and `attrs.kelf` names the kelf file. `attrs.num_inputs` / `num_outputs` give the counts. |
| `arg_nodes` | Indexes of the input nodes. |
| `heads` | `[node, output_index, 0]` per program output. |
| `node_row_ptr` | TVM's offset table into the flattened output list. |
| `attrs.dltype` | Element type of every entry (inputs first, then outputs). |
| `attrs.shape` | Shape of every entry, **flattened to 2-D** (for example `(1,128,4,1280)` becomes `[128, 5120]`). |
| `attrs.storage_id` | Storage slot per entry (here always 0, 1, 2, ...). |
| `metadata.signatures.default.inputs` | Input name -> `{id, dtype, shape}`. |

The kelf node's `output_names` lists every program output **twice** (one output: `output0,
output0`; four outputs: `output3, output2, output1, output0, output3, ...`), and `num_outputs`
counts both copies. **Observed** in all 14 NEFFs (all are 4-rank collective kernels); the reason
is not known.

Inputs are numbered by the compiler, not by kernel parameter order. Use `tensor_map.json` to
find which input is which kernel parameter.

## 3.2 `kelf-0.json` — the subgraph list

```json
{"graphs":[{"definition":"sg00/def.json","name":"sg00"},
           {"definition":"sg01/def.json","name":"sg01"}],
 "version":"0.5","target":"cayman","revision":""}
```

`target` names the hardware generation. `cayman` is Trainium2 (NeuronCore-v3). **Observed.**
One entry per physical core of the Logical NeuronCore (two on this LNC=2 host).

## 3.3 `info.json`

| Key | Value seen | Meaning |
|---|---|---|
| `tool_version` | `2.27.5334.0+f702b353` | neuronx-cc version. neuron-explorer reports it as `compiler_version`. **Verified** (profile metadata). |
| `kelp_version` | `Dynamic` | **Observed**; meaning unknown. |
| `name` | compile work-dir path | Same string as the header's network name. **Observed.** |
| `num_tpb`, `num_tpb_requested` | 1 | Logical NeuronCores used by this NEFF. **Inferred** (matches `logical_neuroncore_count`). |
| `num_tpb_per_node` | `[1, 0, 0, ...]` (64 entries) | Per-node core counts. **Observed.** |
| `nki_version` | `0.6.0+...` | NKI version used. **Observed.** |

## 3.4 `sgNN/tensor_map.json`

One entry per input and output variable of the subgraph, plus (in 2 of the 14 NEFFs) entries
without a `kind` key for constants (`sg0000_identity_220_CRSM`, with `split_axis` keys) and
intermediate HLO values (`dot.1`, `custom_call.1`). Entries of inputs and outputs have:

| Key | Meaning |
|---|---|
| `kind` | `input` or `output` |
| `kernel_param_name` | The NKI kernel parameter this variable came from (inputs only) |
| `dtype`, `tf_shape`, `tf_format` | Element type and the shape as the framework passed it |
| `sim_shape`, `sim_format` | The same shape for the compiler's simulator |
| `tf_file` | File name used by the compiler's simulator (`value_input0.npy`); the file is not in the NEFF |
| `is_const` | `false` for all studied variables |

**Observed.** Use it to answer "which `inputN` is the weight tensor I care about".

## 3.5 Activation table sets

The Activation engine computes non-linear functions with piecewise-polynomial lookup tables.

* `sgNN/act_info.json` is a catalog of every table set the compiler knows. It is identical in
  all 28 subgraphs. **Observed.** For each set it gives the file names and, under `act`, the
  functions the set contains with a size per function (for example `exp: 400`, `tanh: 4`).
* `sgNN/Activation0.json` key `activation_function_sets` lists the sets this subgraph really
  uses, in order. **Observed.**
* Each used set is three files: `<set>.json` (per-function control: thresholds, symmetry, NaN
  and Inf results, polynomial base indexes), `<set>_bkt.bin` (bucket table: float32
  coefficients) and `<set>_ctrl.bin` (control table).

Every table file in the 14 NEFFs is a byte-for-byte copy of the file with the same name in
`neuronxcc/pwp/pwp_bin_trainium/`. **Verified** (compared all of them).

`ACTIVATION_TABLE_LOAD` instructions select which set is active. Its `act_tbl_sel` field
(printed `table_sel` by the profiler) takes the values 0 .. N-1 in a subgraph with N sets
(0-2 for three sets, 0-3 for four), so it is the position in `activation_function_sets`.
**Observed** (consistent in every NEFF; not tested by editing).

**When editing:** to use a function from a set the NEFF does not have, copy the three files from
`neuronxcc/pwp/pwp_bin_trainium/`, add the set name to `activation_function_sets`, and load it
with an `ACTIVATION_TABLE_LOAD`. This is **Inferred** and has not been tried.

## 3.6 DVE table set

`sgNN/default_opcode_table.bin` (1024 bytes), `default_datapath_table.bin` (16384),
`default_control_fast_table.bin` (4096) and `default_control_slow_table.bin` (4096) are the
DVE engine's microcode. `DVE0.json` names them under `dve_tables`.

All 28 copies are byte-identical to `neuronxcc/dve/dve_bin_gen3/default_*_table.bin`.
**Verified.** "gen3" is the Trainium2 DVE (gen2 and gen4 exist in the same package for other
generations). `neuronxcc/dve/dve_bin_gen3/dve_info.json` lists, per table set, which opcodes the
set implements (`default` implements 52 opcodes). Do not edit these files.

## 3.7 Constant files

A `file` variable in `def.json` (for example
`identity_220_sg0000 -> value_sg0000_identity_220_CRSM.npy`, 32 KiB, alignment 32768) is a
constant tensor stored in the payload as a `.npy` file. It appeared once in the corpus (an
identity matrix for a PE transpose). **Observed.**

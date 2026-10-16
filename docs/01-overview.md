# 1. Overview: what a NEFF is and how it runs

This chapter gives the big picture. The later chapters give the byte-level detail.
Words in **bold** are defined in [`../GLOSSARY.md`](../GLOSSARY.md).

Scope: NEFFs built by neuronx-cc 2.27.5334 (NKI 0.6.0) for Trainium2 (`trn2`, NeuronCore-v3,
code name "cayman"), run by Neuron runtime 2.34 on a trn2.3xlarge with LNC=2.

## 1.1 Status labels

Every fact in these documents has one of three labels.

| Label | Meaning |
|---|---|
| **Verified** | Confirmed by an experiment on the device, or by the vendor's own definitions (DWARF types and C accessors in `neuron-dbg`) and then checked on the whole corpus. |
| **Observed** | True in all 14 studied NEFFs, but not tested by editing. |
| **Inferred** | A reasoned conclusion that no experiment has confirmed yet. |

## 1.2 The layers of a NEFF

```
NEFF file
├── NEFF header (1024 bytes)              versions, sizes, payload hash, UUID, feature bits
└── NEFF payload (gzip + tar)
    ├── neff.json                          graph file: inputs, outputs, one kelf node
    ├── kelf-0.json                        lists the subgraphs and the target ("cayman")
    ├── info.json                          compiler version, core counts
    ├── sg00/                              subgraph for physical core 0
    │   ├── def.json                       definition file: variables, DMA queues, collectives
    │   ├── <Engine>0.bin                  engine stream: 64-byte instruction slots
    │   ├── <Engine>0.json                 engine description: DMA transfers, tables
    │   ├── tensor_map.json                variable -> NKI kernel parameter
    │   ├── debug_info_*.dbg               debug info (protobuf)
    │   ├── default_*_table.bin            DVE table set (microcode)
    │   └── <set>.json/_bkt.bin/_ctrl.bin  activation table sets
    └── sg01/                              subgraph for physical core 1 (same layout)
```

`<Engine>` is one of `Activation`, `DVE`, `PE`, `Pool`, `SP`. Every subgraph has all five engine
streams, even when a stream does almost nothing. **Observed.**

## 1.3 Engines

A NeuronCore has five **engines**. Each runs its own **engine stream** in order, in parallel with
the others.

| NEFF name | neuron-explorer name | Main work |
|---|---|---|
| `PE` | Tensor | Matrix multiply (`MATMUL`, `LDWEIGHTS`) into PSUM |
| `Activation` | Scalar | `ACTIVATE` (table-driven functions), copies, casts, some tensor-scalar ops |
| `DVE` | Vector | Element-wise and reduce ops (`TENSOR_TENSOR`, `TENSOR_SCALAR`, `TENSOR_REDUCE`, ...) |
| `Pool` | GpSimd | General ops, register arithmetic (`ALU_OP`), DGE DMA, collectives, `IOTA`, `TOP_K` |
| `SP` | Sync | Starts queued DMA transfers (`PSEUDO_DMATRIGGER`) and waits on semaphores |

**Verified** (name mapping): every profiled instruction of the 14 NEFFs maps to the NEFF engine
stream with the matching name.

## 1.4 How the runtime runs a NEFF

1. The runtime reads the NEFF header and unpacks the payload.
2. For each subgraph it loads the five engine streams into the engines' instruction buffers.
   It **expands every pseudo-instruction** into one or more real instructions (for example a
   `PSEUDO_DMATRIGGER` becomes register `WRITE`s to a DMA queue). See
   [chapter 8](08-pseudo-instructions.md).
3. It builds the DMA rings: it turns each **DMA descriptor** in the engine descriptions into
   16-byte **hardware descriptors**. See [chapter 9](09-dma.md). (**Inferred** from the
   vendor's `neuronxcc/starfish/bin/analyze_neff_artifacts.py`, which sizes DMA rings at 16
   bytes per descriptor; the NEFF itself contains no hardware descriptors.)
4. It adds a **runtime prologue** before and a **runtime epilogue** after each engine stream.
   Both synchronise the engines and send trace notifications; the epilogue also zeroes the
   semaphores for the next run (chapter 10.1).
5. It runs all engines. Engines order their work only through **semaphores** and events. See
   [chapter 10](10-synchronisation.md).

**Verified**: the profiler sees runtime-only instructions before index 0 of every engine
stream, and sees `WRITE`, `MOVE` and similar instructions where the NEFF has pseudo-instructions.

## 1.5 One NEFF, two physical cores, four ranks

On this host one Logical NeuronCore is two physical NeuronCores (LNC=2), so every NEFF has two
subgraphs, `sg00` (physical core 0) and `sg01` (physical core 1). Each subgraph is a complete
program for its core. When the NKI kernel splits work between the two cores, the two
subgraphs have different engine streams (12 of the 14 studied NEFFs: for example 00-Attention-B
has 110,098 PE slots in `sg00` and 112,159 in `sg01`). Only HCCollapse and HCSublayerIn, which do
not split work, have identical streams in both. Their definition files also differ: `sg01`
reaches buffers owned by `sg00` through `remote` variables. **Observed.**

The `-full` kernels run on 4 ranks (4 Logical NeuronCores). All four ranks run the **same NEFF**
(**Verified**: the validator loads one NEFF file on all four ranks). Kernels whose code depends on
the rank read it at run time with `PSEUDO_GID_LOAD` (6 of the 14 NEFFs; chapter 8.4); the others
never need it.

## 1.6 What you can change

| Change | Where | Chapter |
|---|---|---|
| An instruction's operands (addresses, shapes, ops, immediates, dtypes) | `<Engine>0.bin` | [6](06-instruction-format.md), [7](07-instruction-reference.md) |
| Add, remove, reorder instructions | `<Engine>0.bin` + semaphore waits | [10](10-synchronisation.md), [13](13-editing-guide.md) |
| A queued DMA transfer's source, destination or shape | `SP0.json` | [9](09-dma.md) |
| A DGE DMA's source, destination or shape | the `PSEUDO_DMA_DIRECT2D` slot | [9](09-dma.md) |
| Variables, queues | `def.json` | [4](04-definition-file.md) |
| Repack | whole NEFF | [2](02-container.md) |
| Run and measure | sandboxed validator | [13](13-editing-guide.md) |

The tools for each step are in [`../tools/`](../tools/) and described in
[chapter 15](15-tools.md).

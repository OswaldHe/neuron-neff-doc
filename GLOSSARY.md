# NEFF (Trainium2)

The words used across `neff_doc/` for the Neuron Executable File Format as built by neuronx-cc
2.27 for Trainium2, and for the runtime and hardware ideas a NEFF refers to. One idea, one word:
when a tool uses another word for the same thing, it is listed under _Avoid_ or, when you must
recognise it in tool output, under _Tool name_.

## File structure

**NEFF**:
One compiled program for one Logical NeuronCore: a NEFF header followed by a NEFF payload.
_Avoid_: model file, binary, executable

**NEFF header**:
The fixed 1024-byte block at the start of a NEFF with versions, sizes, the payload hash, the UUID, the name and the feature word.
_Avoid_: preamble, file header

**NEFF payload**:
The gzip-compressed tar archive after the NEFF header that holds every other file.
_Avoid_: data section, body

**Package member**:
One file inside the NEFF payload, named by its path in the archive (for example `sg00/def.json`).
_Avoid_: entry, blob

**Feature bit**:
One bit of the header's feature word, naming a runtime capability the program uses (for example `dge`, `coalesced-cc`).
_Avoid_: flag, capability bit

**Graph file**:
`neff.json`, the TVM-style graph listing the program's inputs and outputs and the one kelf node that runs them.
_Avoid_: TVM graph, network json

**Kelf**:
`kelf-N.json`, the list of subgraphs and the target hardware generation.
_Avoid_: kelp

**Subgraph**:
The program for one physical NeuronCore, stored in one `sgNN/` directory with its own definition file and engine streams.
_Avoid_: partition, shard

**Definition file**:
`sgNN/def.json`: a subgraph's variables, DMA queues, collective setup and runtime reservations.
_Avoid_: def, kelp definition

**Engine description**:
`sgNN/<Engine>0.json`: the files an engine uses and, for SP, the list of queued DMA transfers.

**Tensor map**:
`sgNN/tensor_map.json`, linking each input and output variable to the NKI kernel parameter it came from.

**Activation table set**:
A named group of piecewise-polynomial tables (`<name>.json`, `<name>_bkt.bin`, `<name>_ctrl.bin`) from which the Activation engine computes its functions.
_Avoid_: PWP file, LUT

**DVE table set**:
The four microcode tables that define how the DVE engine runs each of its opcodes.

**Debug info file**:
One `sgNN/debug_info_*.dbg` protobuf. The asm level has one record per instruction slot; the backend level describes BIR instructions and their NKI source line.

## Engines and instructions

**Engine**:
One of the five instruction processors of a NeuronCore: PE, Activation, DVE, Pool, SP.
_Tool name_: in neuron-explorer, PE = Tensor, Activation = Scalar, DVE = Vector, Pool = GpSimd, SP = Sync

**Engine stream**:
The ordered instructions one engine runs for one subgraph (`sgNN/<Engine>0.bin`).
_Avoid_: instruction buffer, program, bin

**Instruction slot**:
One 64-byte record of an engine stream.
_Avoid_: instruction word, line

**Opcode**:
Byte 0 of an instruction slot; it selects the instruction and its layout.
_Tool name_: neuron-explorer prints a short form, for example `TENSOR_SCALAR` for `TENSOR_SCALAR_ARITH_OP`

**Instruction struct**:
The vendor's name for the layout of one family of instructions, such as `S3D3_TS` (tensor-scalar).
_Avoid_: format, encoding

**Pseudo-instruction**:
An instruction the runtime replaces, at load time, with one or more real instructions. Its opcode name starts with `PSEUDO_`.
_Avoid_: macro, meta-instruction

**Runtime expansion**:
The real instructions a pseudo-instruction becomes at load time.

**Extension slot**:
A `PSEUDO_EXTENSION` slot holding extra operands of the pseudo-instruction just before it.

**Slot index**:
The position of an instruction slot within its engine stream, from 0.
_Tool name_: `compiler_pc`
_Avoid_: compiler PC

**Runtime PC**:
The position of an instruction in the engine's instruction buffer while it runs, after the runtime prologue and pseudo-instruction expansion.
_Tool name_: `pc`

**Runtime prologue**:
The instructions the runtime runs before a subgraph's first slot.

**Runtime epilogue**:
The instructions the runtime runs after a subgraph's last slot.

**Branch label**:
A `PSEUDO_BRANCH_LABEL` slot that marks a branch target by number; it produces no runtime instruction.

## Synchronisation

**Semaphore**:
A counter, numbered per NeuronCore, that instructions and DMA transfers add to and that instructions wait on.
_Tool name_: `S[n]`

**Semaphore wait**:
The condition in an instruction's events block (for example `S[27] >= 16`) that must hold before the instruction starts.

**Semaphore update**:
The change an instruction makes to a semaphore when it reads its inputs or completes (for example `S[6]++`).

**Completion semaphore**:
The semaphore an engine's ordinary instructions add one to when they finish: S[3] Pool, S[4] Activation, S[5] PE, S[6] DVE.
_Avoid_: own semaphore

**Event**:
A flag, numbered per NeuronCore, that an instruction can wait on; used by the runtime, not by NEFF instructions.
_Tool name_: `E[n]`

## Memory

**State buffer**:
The on-chip SRAM the engines read and write, in 128 partitions. Short form: SBUF. In DMA descriptors it is the variable `SB`.
_Avoid_: scratchpad

**PSUM**:
The memory the PE engine accumulates matmul results into.
_Avoid_: accumulator buffer

**Partition**:
One of the 128 rows of the state buffer or PSUM, with its own address window.
_Avoid_: lane

**Variable**:
A named memory object in a definition file's `var` section: an input, an output, the state buffer, a DRAM scratch tensor, a remote buffer, a constant file, a group, or a DGE table.
_Avoid_: tensor (when the declaration is meant), buffer

**Access pattern**:
Which elements an operand covers: a start address and, per dimension, a step and a count.
_Avoid_: AP, stride pattern

**Address marker**:
The top byte of an instruction address, saying whether the rest is an immediate address, a register, a variable reference or a table reference.

**Variable address**:
A pseudo-instruction address given as a variable id plus a byte offset, which the runtime turns into a physical address.

## Data movement

**DMA queue**:
A named group of 16 hardware DMA rings declared in a definition file, owned by one engine, with a set of 16 semaphores its transfers signal.
_Tool name_: `qSPIO0` also appears as `qSyncIO0`

**Queued DMA**:
A DMA transfer listed in advance in `SP0.json` and started by a `PSEUDO_DMATRIGGER` slot.

**DMA transfer**:
One entry of `SP0.json`'s `dma` list: an id, a queue, a semaphore and its DMA descriptors.
_Avoid_: DMA block, DMA op

**DMA descriptor**:
One element of a DMA transfer's `desc` list: source and destination variable, offsets, types and byte-based patterns.
_Avoid_: desc

**Hardware descriptor**:
A 16-byte record in a DMA ring that the runtime makes from DMA descriptors; not stored in the NEFF.

**DGE DMA**:
A DMA whose descriptors are made at run time by hardware from a `PSEUDO_DMA_DIRECT2D` instruction.
_Avoid_: dynamic DMA

## Hardware and runtime

**NeuronCore**:
One physical core with five engines, a state buffer and PSUM.
_Avoid_: TPB, NC (except when quoting tools)

**Logical NeuronCore**:
The group of physical NeuronCores the runtime treats as one core; two on this host (LNC=2), so a NEFF has two subgraphs.
_Avoid_: LNC (except in numbers like LNC=2), virtual core

**Rank**:
One Logical NeuronCore taking part in a collective; the `-full` kernels run the same NEFF on 4 ranks.

## Working with NEFFs

**Validator**:
A module's frozen `inference.py`, which runs a NEFF on 4 ranks, checks the output against the golden and measures latency.
_Avoid_: test, harness (for the frozen one)

**Sandbox**:
A copy of a validator whose compile cache holds an edited NEFF, so the validator runs the edit without touching the original folder.

**Corpus**:
The table of every instruction slot of the 14 studied NEFFs joined with its debug info and the profiler's disassembly (`artifacts/corpus/`).

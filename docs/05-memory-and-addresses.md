# 5. Memory and addresses

## 5.1 Three kinds of memory

| Memory | Who addresses it | How |
|---|---|---|
| State buffer (SBUF) | Compute instructions, DMA | 128 partitions; an instruction gives a start address and how many partitions (`num_active_channels`) |
| PSUM | PE writes, other engines read | 128 partitions; the PE engine accumulates matmul results here |
| DRAM (HBM) | DMA only | Through variables (`input`, `output`, `virtual`, `remote`, `file`) of the definition file |

Compute instructions never address DRAM. All DRAM traffic goes through DMA (chapter 9).
**Observed** (no compute-instruction address in the corpus falls outside the SBUF and PSUM
ranges below).

## 5.2 The 32-bit address of a compute operand

Every tensor operand of a compute instruction starts with a 4-byte `start_addr` (vendor type
`NEURON_ISA_TPB_ADDR4`). Its top byte is a **marker** that says how to read the other bits.
Marker values come from the vendor constants in `neuron-dbg`
(`artifacts/reference/hw_constants_v3.json`). **Verified.**

| Marker (byte 3) | Name | Meaning of the address |
|---|---|---|
| `0x00` | `ADDR4_MARKER_IMM` | Immediate address (5.3) |
| `0x80` | `ADDR4_MARKER_ADDR_REG` | Address taken from register `$R[byte 0]` at run time. The profiler prints `@$R[60]@`. |
| `0x40` | `ADDR4_MARKER_SHAPE_REG` | Shape from a shape register |
| `0xC0` | `ADDR4_MARKER_ADDR_SHAPE_REG` | Address and shape from registers |
| `0x20`, `0xA0` | `ADDR4_MARKER_INDIRECT_IMM/_REG` | Indirect addressing |

**Verified** for `0x00` and `0x80` on the corpus (a `COPY` with marker `0x80` and byte 0 = 60 is
printed by the profiler as `src=bfloat16@$R[60]@...`).

## 5.3 Immediate addresses

```
SBUF:  address = partition * 0x40000 + offset         (partition 0..127)
PSUM:  address = 0x2000000 + offset                   (offset < 0x4000)
```

**SBUF.** Each partition has a 256 KiB (0x40000) address window. This matches the vendor
constant `STATE_BUF_PARTITION_SIZE = 0x40000` and the DMA descriptors, which step from one
partition to the next by 262144 bytes. **Verified.**

* Offsets used in the corpus run from 0x4000 to 0x37FFF. None of 208,857 sampled SBUF
  references uses an offset below 0x4000. **Observed.**
* The runtime reserves 8 bytes at offset 0x37FF8 of every partition (chapter 4.4). **Observed.**
* Start partitions seen: 0, 32, 64, 96 — multiples of 32. Start partition plus
  `num_active_channels` never exceeds 128. **Observed.** Keep both rules when you edit.

Conflict note: `neuron-dbg`'s ISA headers also give `STATE_BUF_PARTITION_ACTIVE_SIZE = 0x30000`
(192 KiB). Trainium2 NEFFs use offsets up to 0x37FD0, so that constant does not describe the
usable size on this hardware; the observed range above does.

**PSUM.** All 172,473 sampled PSUM references are `0x2000000 + offset` with offset below 0x4000
(16 KiB per partition) and the partition field equal to 0. Use clusters at every 2 KiB
boundary, so PSUM is handed out in 2 KiB banks (8 per partition). **Observed.** (The vendor
constants list `PSUM_BUF_BANK_SIZE = 0x400` and 16 banks; the Trainium2 data shows 2 KiB
granularity. Use the data.)

A matmul's PSUM destination is printed by the profiler without a dtype (`dst=0x2000800[...]`).

## 5.4 Access patterns

After the address comes the access pattern: `step_elem[n]` (signed 16-bit) then `num_elem[n]`
(unsigned 16-bit), `n` = 2, 3 or 4 depending on the operand type (`TENSOR2D`, `TENSOR3D`,
`TENSOR4D`). **Verified** (vendor field names and accessor offsets, and every pattern in the
corpus reproduces the profiler's `[steps][counts]` text).

* Steps and counts are in **elements** of the operand's dtype (the fields are `step_elem` and
  `num_elem`; a contiguous bf16 row has step 1, not 2). Element 0 is at `start_addr`.
* Dimension 0 is the innermost: patterns such as `[1,32,1][32,8,1]` walk 32 contiguous
  elements, then step 32 elements to the next group. **Observed.**
* The partition axis is **not** part of the pattern: it is `num_active_channels` partitions
  starting at the address's partition.
* A step of 0 repeats an element (broadcast). A step of `-1` (0xFFFF) walks backwards.
  **Observed** (`LDWEIGHTS` uses `[-1,0,0]`).
* The profiler pads 2-D patterns with zeros to three dimensions (`[1,1,0][960,1,0]`).

Example (HCCollapse, DVE slot 1): `src = bf16 @ 0x35700 [1,1,1][960,1,1], channels=128` reads
960 contiguous bf16 elements at offset 0x35700 of partitions 0..127.

## 5.5 Scalar operands in memory (pointer immediates)

Some instructions take a per-partition scalar from SBUF instead of an immediate value (field
`imm0_src = POINTER_IMMEDIATE`, value in `imm0_imm_ptr`). The pointer is a byte offset in each
active partition: partition *p* uses the scalar at `p * 0x40000 + imm0_imm_ptr`.
**Verified** (experiment e01: moving `imm0_imm_ptr` from 0x37F0C to 0x37F08 made the kernel use
column 2 instead of column 3 of the per-token `pm` scalars, with bit-exact predicted output).

## 5.6 DRAM addresses

DRAM is only named by DMA, in two ways:

* In engine-description DMA descriptors: a variable **name** and a byte offset
  (`"from": "input1", "from_off": 640`). Chapter 9.
* In DMA pseudo-instructions: an 8-byte address whose marker says "variable": `var_id` plus a
  byte offset, or "table": a DGE-table index plus offset. The runtime replaces it by a physical
  address (the profiler shows `src_addr_imm=0x8020...`). Chapter 8.

**Verified** (the `var_id` of a DGE DMA destination names the variable of that id, and the runtime
text shows the resolved physical address).

# 15. Tools

All tools are in [`../tools/`](../tools/). Python tools run with `workspace/.venv`
(`source .venv/bin/activate`) unless noted; `neff.py` also runs with plain `python3`.

## 15.1 Everyday tools

| Tool | Does |
|---|---|
| `neff.py header/unpack/pack/verify` | Read the header; unpack to `header.json`, `header.bin`, `members.txt`, `payload/`; repack byte-exactly (recomputes size and MD5); check size and MD5 (chapter 2). |
| `isa.py <Engine0.bin> [first [count]]` | Decode an engine stream. As a library: `decode`, `decode_stream`, `Instruction.set/encode/as_dict`, `fields(struct)`, `OPCODES`, `ENUMS` (chapter 6.9). |
| `dbginfo.py <file.dbg>` | Decode a debug file to JSON without protobuf (chapter 11). |
| `neff_diff.py <a.neff> <b.neff>` | Diff two NEFFs at the level that matters: header fields, which package members changed, which instruction slots changed (raw bytes and decoded fields, union aliases grouped by byte range), and a JSON diff for JSON members. A byte diff of the files themselves is useless because the payload is gzipped. |
| `semaphores.py <unpacked> sgNN` | Per semaphore: who increments it and by how much, the most anyone waits for, and a warning when a wait can never be met (chapter 10). |
| `run_neff.py <full-dir> <edited.neff> [--sandbox D] [--exec-timeout S]` | Run an edited NEFF through the module's frozen validator in a sandbox; prints `passed`, errors, latency per rank (chapter 13). |
| `verify_edit.py <full-dir> <edited.neff> [--slots sgNN:Engine:a-b] [--baseline a.pt] [--expect b.pt]` | Run an edited NEFF and check the whole evidence chain link by link, printing PASS/FAIL per claim: the file is well formed; the NEFF the validator loaded and profiled is byte-identical to it; the runtime did not recompile; the profile's uuid/hash match the file; the profiler's disassembly of the named slots; and the output against a baseline or a prediction (chapter 13.8). |
| `ingest_profiles.sh` | For every NEFF in `artifacts/neff_list.txt`: capture a profile if there is none, then convert rank 0's profile to parquet in `artifacts/profiles/<tag>/rank0/`. |
| `corpus.py [tag ...]` | Join every slot with its debug info and the profiler's disassembly into `artifacts/corpus/<tag>.parquet` (chapter 12); realigns the profiler's `compiler_pc` labels. |

## 15.2 Tools that extract the vendor's definitions

Run these again only if the Neuron SDK changes (they read `/opt/aws/neuron/bin/neuron-dbg` and
write `artifacts/reference/`):

| Order | Tool | Writes |
|---|---|---|
| 1 | `dwarf_isa.py neuron-dbg isa_dwarf.json` | struct member lists, enum constants, typedefs from DWARF |
| 2 | `isa_layout.py neuron-dbg isa_dwarf.json isa_layout.json` | byte layout of every field, from the C accessors |
| 3 | `opcode_dispatch.py neuron-dbg opcode_dispatch_v3.json` | opcode -> struct, from the disassembler's compare chains and jump table |
| 4 | `hw_constants.py neuron-dbg hw_constants_v3.json` | hardware constants and address marker values |

`isa_layout_supplement.json` (hand-made, with evidence per entry) and
`pseudo_expansions.json` (made from the corpus) complete them.

## 15.3 Tools that check and document

| Tool | Does |
|---|---|
| `verify_isa.py` | Decode the whole corpus and check every wait, update, pattern, channel count and named field against the profiler text; writes `artifacts/reference/opcode_struct_v3.json` and prints a summary (saved as `artifacts/reference/verify_isa.txt`). |
| `gen_reference.py` | Writes `docs/generated/opcodes.md`, `structs.md`, `enums.md`. |
| `nki_map.py` | Writes `docs/generated/nki_to_isa.md` (NKI call -> instructions). |

## 15.4 Making the NEFFs

| Tool | Does |
|---|---|
| `gen_full_neffs.sh <full-dir> ...` | Run each `-full` folder's frozen validator and copy the NEFF compiled for the current `source.py` to `optimization-runs-*/<full-dir>.neff`. |
| `compile_hc_attn_in_full.py` | Compile-only 4-rank harness for `optimization-runs-HCSublayerIn/layers-0-hc_attn_in-full`, which has no validator. |

## 15.5 Full rebuild order

```bash
source .venv/bin/activate
# (1) NEFFs:            neff_doc/tools/gen_full_neffs.sh optimization-runs-*/<module>-full
# (2) unpack:           python3 neff_doc/tools/neff.py unpack <x.neff> neff_doc/artifacts/unpacked/<tag>
# (3) vendor data:      15.2, steps 1-4 (only after an SDK change)
# (4) profiles:         neff_doc/tools/ingest_profiles.sh
# (5) corpus:           python neff_doc/tools/corpus.py
# (6) checks and docs:  python neff_doc/tools/verify_isa.py && python neff_doc/tools/gen_reference.py \
#                       && python neff_doc/tools/nki_map.py
```

## 15.6 Other Neuron tools

| Tool | Useful for | Limits |
|---|---|---|
| `neuron-explorer view` | per-instruction disassembly, timing, source lines (needs a profile) | only executed instructions; pseudo-instructions in expanded form; wrong ACTIVATE function names on Trainium2; `compiler_pc` shifted after `PSEUDO_GID_LOAD` |
| `neuron-explorer capture` | run a NEFF and record a profile | no output check |
| `neuron-dbg disasm`, `descriptors` | disassemble a live core's instruction buffer and DMA rings | needs the NEFF loaded and halted on the device |
| `neuronxcc.isa_tpb.sunda.neuron_isa` | official enums; a full codec for `TENSOR_TENSOR` only (`s3s3d3_tt_struct`) | other formats not exposed |
| `neuronxcc/starfish/bin/analyze_neff_artifacts.py` | DMA and scratch-memory summary of an unpacked subgraph | not a disassembler |

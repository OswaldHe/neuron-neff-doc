# NEFF documentation for Trainium2

How a Neuron Executable File Format (NEFF) file is built, what every part means, and how to
**modify a compiled NEFF, run it, check it and measure it**. Written from 14 NEFFs compiled by
neuronx-cc 2.27.5334 (NKI 0.6.0) for Trainium2 and run on a trn2.3xlarge (LNC=2, 4 ranks).

Every statement is labelled **Verified** (device experiment, or vendor definitions checked on the
whole corpus), **Observed** (true in all 14 NEFFs, not tested by editing) or **Inferred**.
Words in bold in the chapters are defined in [GLOSSARY.md](GLOSSARY.md).

## Quick start: edit, run, measure

```bash
cd /home/ubuntu/workspace && source .venv/bin/activate
python3 neff_doc/tools/neff.py unpack optimization-runs-HCCollapse/hc_collapse-full.neff work/
python  neff_doc/tools/isa.py work/payload/sg00/DVE0.bin                 # decoded slots
# ... edit with neff_doc/tools/isa.py (chapter 13), re-balance semaphores (chapter 10) ...
python3 neff_doc/tools/neff.py pack work/ edited.neff
python  neff_doc/tools/run_neff.py optimization-runs-HCCollapse/hc_collapse-full edited.neff --exec-timeout 60
```

## Chapters

| # | Chapter | Read it for |
|---|---|---|
| 1 | [Overview](docs/01-overview.md) | the layers of a NEFF, engines, how the runtime runs it |
| 2 | [Container](docs/02-container.md) | the 1024-byte header, feature bits, byte-exact repacking |
| 3 | [Package files](docs/03-package-files.md) | `neff.json`, `kelf`, `info.json`, tensor map, activation and DVE tables |
| 4 | [Definition file](docs/04-definition-file.md) | `def.json`: variables, DMA queues, the two subgraphs |
| 5 | [Memory and addresses](docs/05-memory-and-addresses.md) | SBUF/PSUM address encoding, access patterns, address markers |
| 6 | [Instruction format](docs/06-instruction-format.md) | the 64-byte slot, header, events block, building blocks, dtypes, ALU ops |
| 7 | [Instruction families](docs/07-instruction-reference.md) | what the fields of the common instructions mean |
| 8 | [Pseudo-instructions](docs/08-pseudo-instructions.md) | what the runtime rewrites, barriers, collectives, branches and loops |
| 9 | [DMA](docs/09-dma.md) | queued DMA (JSON) and DGE DMA (instructions) |
| 10 | [Synchronisation](docs/10-synchronisation.md) | the semaphore model and the rules for editing |
| 11 | [Debug info](docs/11-debug-info.md) | slot -> BIR -> NKI line; profiler quirks |
| 12 | [Source to NEFF](docs/12-source-to-neff.md) | which NKI call becomes which instructions |
| 13 | [Editing guide](docs/13-editing-guide.md) | the step-by-step modify / run / check / measure procedure, a verified worked example, and which edits prove best that a change took effect |
| 14 | [Optimization playbook](docs/14-optimization-playbook.md) | finding where time goes; tested and candidate levers |
| 15 | [Tools](docs/15-tools.md) | every tool and the rebuild order |
| 16 | [Open questions and conflicts](docs/16-open-questions-and-conflicts.md) | what is still unknown; earlier readings that were corrected |

Generated references (from the vendor data and the corpus):
[opcodes](docs/generated/opcodes.md), [struct layouts](docs/generated/structs.md),
[enums](docs/generated/enums.md), [NKI call -> instructions](docs/generated/nki_to_isa.md),
[activation function ids](docs/generated/activation_functions.md),
[HCCollapse slot-by-slot walkthrough](docs/generated/walkthrough_hccollapse.md).

## Where the facts come from

1. **The NEFFs.** All 14 studied NEFFs unpacked (`artifacts/unpacked/`); 2,051,467 instruction slots.
2. **The vendor's own definitions.** `neuron-dbg` carries the ISA as Go types with DWARF, the C
   field accessors and its disassembler's opcode dispatch; the tools read the layouts, enums,
   constants and opcode table out of the binary (chapter 6.1).
3. **The profiler.** One profile per NEFF, joined slot by slot with the NEFF (`artifacts/corpus/`):
   every decoded field is checked against the profiler's disassembly.
4. **The NKI source.** Debug info links every slot to its NKI line (chapter 12).
5. **Experiments on the device** (`artifacts/experiments/`): header probes, a field edit with an
   exactly predicted output (e01), eleven runtime-rule cases (r01-r11), a self-wait test (o01),
   and a full edit/run/verify pass on the 7.8 MB 00-Attention-B NEFF (a01, chapter 13.6).

## Directory layout

```
neff_doc/
  README.md, GLOSSARY.md
  docs/                    chapters 1-16, docs/generated/ (generated references)
  tools/                   all tools (chapter 15)
  artifacts/
    neff_list.txt          the 14 NEFFs
    unpacked/<tag>/        header.json, header.bin, members.txt, payload/
    profiles/<tag>/rank0/  neuron-explorer parquet tables
    corpus/<tag>.parquet   slot-level join (bytes, debug info, profiler text)
    reference/             vendor data extracted from neuron-dbg, supplement, checks
    experiments/           e01, header_bits, runtime_rules (r01-r11), o01
    neff_generation_logs/  logs of the NEFF builds
```

## The 14 NEFFs

`optimization-runs-<X>/<module>-full.neff`, built from each folder's current `source.py`:
00-Attention-B, 00-Attention-C, 01-Engram, 01-Engram-long8192, 01-MoE-long8192, 02-Attention,
20-Attention, 24-Attention, 40-DSparkAttention, 40-MoE, 42-DSparkConfidenceHead,
42-DSparkMarkovHead, HCCollapse, HCSublayerIn. (HCExpand, HCSublayerIn-long8192 and HCSublayerOut
have no kernel in their `-full` folder.)

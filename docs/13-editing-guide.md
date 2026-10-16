# 13. Editing guide: modify a NEFF, run it, measure it

This is the procedure for the target flow: take a compiled NEFF, change it, run it without
errors, check the result, and measure the speed. Every step here was done end to end on the
device (experiments e01, r01-r11, o01).

## 13.1 Setup

```bash
cd /home/ubuntu/workspace
source .venv/bin/activate          # torch_xla, neuronx-cc 2.27.5334, pandas, pyarrow, protobuf, pyelftools
export T=neff_doc/tools
```

The tools need the extracted vendor data in `neff_doc/artifacts/reference/` (already there). If
the Neuron SDK changes, regenerate it (chapter 15.2).

## 13.2 Step by step

**1. Pick the NEFF and its validator.** Each studied NEFF is
`optimization-runs-<X>/<module>-full.neff`, and its frozen validator is
`optimization-runs-<X>/<module>-full/inference.py`. The validator's compile cache must hold the
original NEFF (`build/cache/<sha256(source.py)[:16]>/.../model.neff`); it does after one normal
validator run.

**2. Unpack.**

```bash
python3 $T/neff.py unpack optimization-runs-HCCollapse/hc_collapse-full.neff work/
```

**3. Find what to change.** Decode a stream, or use the corpus (slot + source line + profiler
text and time):

```bash
python $T/isa.py work/payload/sg00/DVE0.bin                  # every slot, decoded
python $T/semaphores.py work sg00                            # who signals what (chapter 10)
```

```python
import pandas as pd
df = pd.read_parquet("neff_doc/artifacts/corpus/HCCollapse.parquet")    # chapter 12
```

For a profile of an edited NEFF, see 13.4.

**4. Edit.** In Python:

```python
import sys; sys.path.insert(0, "neff_doc/tools")
import isa
p = "work/payload/sg00/DVE0.bin"
data = bytearray(open(p, "rb").read())
ins = isa.decode(bytes(data[6*64:7*64]))           # slot 6
ins.set("imm0_imm_ptr", 0x37F08)                    # change a field by vendor name
data[6*64:7*64] = ins.encode()
open(p, "wb").write(data)
```

* Fields: chapter 6 and [`generated/structs.md`](generated/structs.md).
* Removing, adding or moving slots: re-balance semaphores (chapter 10.4). Keep
  `PSEUDO_EXTENSION` slots with their instruction, fix `branch_pc` in prefetch hints (chapter
  8.6), keep slot 0 (`SET_ORDERING_MODE`).
* DMA: edit `work/payload/sgNN/SP0.json` (chapter 9).
* Both subgraphs: decide for each edit whether `sg01` needs it too (chapter 4.5).
* Debug info: optional (chapter 11.5).

**5. Check the semaphores.** `python $T/semaphores.py work sg00` must show no
`wait exceeds increments` line, for every subgraph you touched. This catches waits that can
never be met (hangs). It does not catch a wait that is too *low* (a race); only the output
comparison in step 7 does. The tool counts each stream as straight-line code; in a NEFF with
loops (chapter 8.6) the counts restart at every `EVENT_SEMAPHORE_RANGE_CLEAR`, so there its
totals are only an upper bound per loop body.

**6. Repack and run.**

```bash
python3 $T/neff.py pack work/ edited.neff
python $T/run_neff.py optimization-runs-HCCollapse/hc_collapse-full edited.neff \
       --sandbox neff_doc/artifacts/sandbox/my-edit --exec-timeout 60
```

`run_neff.py` copies the validator into a sandbox, puts `edited.neff` into the sandbox's compile
cache in place of the original, and runs the validator. The validator then loads and runs the
edited NEFF on all 4 ranks, compares rank 0's output with the golden, and profiles it. The
original run folder is not touched. Output: `passed`, `max_abs_err`, `cosine`, `latency_ms`
(fastest rank), `latency_rank_N_ms`, and `used_cached_neff = True` (proof that the edited NEFF
was run, not a recompile).

**7. Check correctness.** `passed = 1` means the output meets the module's tolerance. For an edit
that should not change the numbers (reordering, removing dead work, synchronisation changes),
also compare bit for bit with the unedited NEFF's output: run the unedited NEFF once the same way
and compare the two `sandbox/build/rank0_output.pt` files with `torch.equal`. A race shows up as
a difference that may change from run to run; run twice when in doubt.

**8. Measure.** `latency_ms` comes from the device profile (`total_exec_time` of the fastest
rank). Run-to-run spread measured here:

| Kernel | Latency | Spread of unchanged-output runs |
|---|---|---|
| HCCollapse | 0.055-0.062 ms | 12% (10 runs) |
| 40-MoE | 4.56 ms | 0.12% (4 runs) |

So: run the baseline and the edit at least 3 times each and compare medians. On a kernel of tens
of microseconds, a change below about 10% cannot be seen in one run.

## 13.3 When something goes wrong

| Symptom | Likely cause | Chapter |
|---|---|---|
| `NRT_TIMEOUT ... status_code=5` after the exec timeout | a wait that is never satisfied (wrong count, removed producer, missing DMA) | 10.6 |
| `NRT_EXEC_OOB status_code=1006`, `... out-of-bound access ... neff instruction index = N` | a DGE DMA ran with bad addresses (often a removed wait before register arithmetic) | 10.6 |
| runs, `passed = 0`, output wrong | a wait too low (race), a wrong field, a missing instruction | 10.4 |
| runs, output differs between two runs | a race | 10.5 |
| `used_cached_neff = False` | the cache entry was not used; the validator compiled `source.py` instead (check the sandbox cache path) | 13.2 |

After a timeout or an out-of-bounds error the device works normally for the next run.
**Verified.**

## 13.4 Profiling an edited NEFF

The validator already captured profiles (`sandbox/build/profile_rank_*.ntff` next to
`sandbox/build/profiled.neff`). To get the per-instruction table:

```bash
neuron-explorer view -n sandbox/build/profiled.neff -s sandbox/build/profile_rank_0.ntff \
    --output-format=parquet --output-file=$PWD/prof --ingest-only
```

`prof/Instruction.parquet` has per instruction: `engine`, `pcore_idx` (0 = `sg00`), `pc`,
`compiler_pc` (slot index; see the GID-load quirk in chapter 11.4), `opcode`, `operands`,
`start_ts`, `end_ts`, `duration_ns`. Sum `duration_ns` per engine to find the busy engine
(chapter 14).

## 13.5 Running a NEFF without a validator

For a NEFF that has no frozen validator, compile-and-run through a harness of your own that loads
inputs of the right shape and calls the kernel through torch_xla with
`NEURON_CC_FLAGS=--cache_dir=<dir>` (example: `neff_doc/tools/compile_hc_attn_in_full.py` for
HCSublayerIn); then put the edited NEFF into that cache directory the same way. For timing only,
`neuron-explorer capture -n x.neff --io-from=runtime` runs a NEFF with runtime-made inputs (for a
4-rank collective NEFF add `--collectives-worker-count 4 --collectives-workers-per-node 4
--collectives-worker-start-id 0 --collectives-profile-id all`, as the validators do).

## 13.6 Worked example: a verified edit on 00-Attention-B

`artifacts/experiments/a01_attnB_output_swap/` runs the whole procedure on the 7.8 MB
00-Attention-B NEFF (317,017 slots, 11.5 ms) and is the template to copy.

**The edit.** The kernel ends with an all-reduce into `cc_dst`, then eight DGE DMA writes copy
`cc_dst` into `output0` in blocks of 1024 rows of the (8192, 5120) bf16 output: four written by
`sg00` (rows 0-2047, 6144-8191) and four by `sg01` (rows 2048-6143). In `sg00` only, the
**source** offsets of the two writes whose destinations are rows 0-1023 and 1024-2047 were
exchanged (`src_start_addr_addr_var_offset`: 0x0 <-> 0xA00000). Nothing else changed: same slots,
opcodes, destinations, waits and updates, so `semaphores.py` output was identical to the control.

**The prediction**, made before the run and needing no model of the kernel's arithmetic:

```
edited[0:1024] == baseline[1024:2048],  edited[1024:2048] == baseline[0:1024],
edited[2048:8192] == baseline[2048:8192]
```

**The result.** All four checks held bit-exactly, and exactly 2048 of 8192 rows differed from the
baseline.

| | control (unchanged repack) | swap (edited) |
|---|---|---|
| `passed` | 1 | 0 |
| `max_abs_err` vs golden | 0.0859 | 4.648 |
| `cosine` | 0.999943 | 0.872 |
| `pass_fraction` | 1.0 | 0.805 |
| `latency_ms` | 11.5059 | 11.5067 |
| `used_cached_neff` / `compiled` | True / False | True / False |

Three independent pieces of evidence that the edited file is what ran:

1. `used_cached_neff = True` with `compiled = False` — the validator loaded the NEFF from the
   cache entry the sandbox had replaced, and did not recompile `source.py`.
2. The profiler's disassembly of the executed instructions shows the exchanged physical source
   addresses (`0x8005a3300000` <-> `0x8005a3d00000`, one block apart) with destinations unchanged
   (`check_profile.sh`; remember the slot-label shift of chapter 11.4, which this NEFF shows
   because it contains `PSEUDO_GID_LOAD`).
3. The device output equals the prediction bit for bit.

**The diff.** `tools/neff_diff.py` on the shipped and edited files reports: 64 of 65 package
members identical, 1 changed (`sg00/Pool0.bin`); 2 of its 745 instruction slots changed; and in
each of those exactly **one byte** differs, byte 18, the third byte of the little-endian u32
`src_start_addr_addr_var_offset` at bytes 16-19:

```
slot 738  - ... 00261000 00000000 18000010 ...      slot 739  - ... 00271000 0000a000 18000010 ...
          + ... 00261000 0000a000 18000010 ...                + ... 00271000 00000000 18000010 ...
  bytes 16..19 src_start_addr_addr_var_offset: 0 -> 0xa00000     ... 0xa00000 -> 0
```

Byte 23 is the address marker, `0x10` = the variable form (`PSEUDO_ADDR8_MARKER_VAR_ID_BIT`), and
bytes 20-22 give `var_id` 24 = `inst__I-23-0:cc_dst_sg0000` — so these two bytes are the only
thing that moved, and they moved which part of `cc_dst` each transfer reads. The header's
`payload_size` and both copies of `payload_md5` also change, as they must for any content change
(chapter 2.1); `tools/neff.py pack` recomputes them.

Two side notes worth copying:

* The control repack came out **byte-identical** to the shipped NEFF, so its device output is a
  true baseline. Predict against that baseline, not against the golden: this kernel's unedited
  output already differs from the golden by up to 0.0859 (it passes on tolerance, not bit-exactly).
* Latency moved by 0.007% — a pure address change moves the same bytes. Do not read anything into
  a latency difference that small (chapter 13.2 step 8).

## 13.7 Which edit proves best that a change took effect

Ranked by how convincing the evidence is:

1. **An exactly predictable structural change to the output** (as in 13.6): permute, duplicate or
   re-point whole blocks of the output through one DMA field. The prediction follows from the
   baseline output alone, it is checkable bit for bit, noise cannot imitate it, and the edit is one
   field, so a surprise can only come from the edit itself.
2. **An exact power-of-two scale on the last arithmetic step before the output** (`output == 2 x
   baseline`, exact in floating point). Needs a field that scales: `ACTIVATE` has
   `scale_value_imm_fp32`, and a tensor-scalar has `imm0`. Not available in 00-Attention-B, whose
   final PSUM drain compiles to `CAST`, which has no scale field (checked over the corpus).
3. **Changing an operand's dtype or an ALU op** — a large, obvious change in the numbers, but the
   exact expected value is harder to state in advance.
4. **A numeric tweak inside the module's tolerance** — weak: the validator can still report
   `passed = 1`, so it proves nothing by itself.
5. **A latency-only edit** — weakest as proof. Run-to-run spread is 0.12% on 40-MoE but about 12%
   on a 55 us kernel, so a single run cannot tell a small win from noise.

Deliberate failures (an unreachable wait, chapter 10.6) do prove the NEFF is being executed, but
they destroy the run, so use them only to check the harness itself.

## 13.8 Verifying an edit: expected versus actual

"The edit works as intended" is four claims, and each needs its own evidence. A run that merely
finishes proves none of them. `tools/verify_edit.py` checks them link by link and prints PASS/FAIL
per claim, so a failure names the broken claim instead of just saying the answer is wrong.

| # | Claim | Evidence | What it rules out |
|---|---|---|---|
| 1 | The file is well formed | `neff.py verify`: `data_size` and both MD5 copies match the payload | a truncated or mis-packed file |
| 2 | Only the intended bytes changed | `neff_diff.py`: which members, slots and fields differ | collateral edits elsewhere in the NEFF |
| 3 | Synchronisation is still sound | `semaphores.py`: no wait exceeds its supply; ideally identical to the control | hangs (`NRT_TIMEOUT`) |
| 4a | My file is the one that ran | `sha256(sandbox/build/profiled.neff) == sha256(edited.neff)`; `used_cached_neff = True`; `compiled = False` | the runtime having run a cached original, or torch_xla having recompiled `source.py` |
| 4b | The profile describes my file | the profile's `NeffHeader` `uuid` and `hash` equal the edited file's header, and neuron-explorer accepted the NEFF/NTFF pair (it refuses a uuid mismatch) | reading a stale profile from an earlier run |
| 4c | The hardware ran the edit | the profiler's disassembly of the edited slots shows the new operands | the edit being ignored at load time |
| 5 | The effect is exactly as intended | the device output against a prediction made in advance, bit for bit | "something changed" being mistaken for "the right thing changed" |

**The a01 edit, expected versus actual.** Predictions were written down before the run.

| | Expected | Actual |
|---|---|---|
| Bytes changed | 1 member, 2 slots, 1 byte each | `sg00/Pool0.bin`, slots 738 and 739, byte 18 each; 64 of 65 members untouched |
| Header | `data_size` and MD5 recomputed | 7805177 -> 7805178; MD5 `78c95059...` -> `9a8dcc71...` |
| Load | accepted; no hash check (r01) | loaded from the cache, `compiled = False`, `profiled.neff` sha256 equal to the edited file |
| Slot 738 executes | reads `cc_dst` + 0xA00000, writes output + 0x0 | reads `0x8005a3d00000`, writes output + 0x0 (baseline read `0x8005a3300000`) |
| Slot 739 executes | reads `cc_dst` + 0x0, writes output + 0xA00000 | reads `0x8005a3300000`, writes output + 0xA00000 (baseline read `0x8005a3d00000`) |
| Slots 740, 741 | unchanged | `0x8005a6f00000` and `0x8005a7900000`, as in the baseline |
| Output rows 0-1023 | baseline rows 1024-2047 | equal, bit for bit |
| Output rows 1024-2047 | baseline rows 0-1023 | equal, bit for bit |
| Output rows 2048-8191 | unchanged | equal, bit for bit |
| Rows differing | 2048 of 8192 | 2048 |
| Validator verdict | `passed = 0` on tolerance, large error | `passed = 0`, `max_abs_err` 0.0859 -> 4.648, `cosine` 0.99994 -> 0.872, `pass_fraction` 1.0 -> 0.805 |
| Latency | unchanged (same bytes moved) | 11.5059 (control), 11.5067, 11.5175 ms: spread 0.10%, no effect |

Note what `passed = 0` means here: the **module's numerical tolerance** failed, which is the
intended outcome of a probe designed to change the answer. It is not a runtime error, and the
validator's non-zero exit code only reflects that verdict. For a real optimization the target is
the opposite: `passed = 1`, output bit-identical to the baseline, and a lower median latency.

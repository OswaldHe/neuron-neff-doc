# Review of iter-0

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports the header identical, 34 of 34 members identical and "payloads are identical". `cmp` shows the files are byte-identical. `notes/` holds only the template README, so no edit is claimed and none was made. This iteration is a null edit. `profile_latest.json` was not present, so I re-profiled the unchanged NEFF.

## 2. Is there a latency change?
No. The baseline is 602.637 us with a spread of 0.052% over 5 captures (min 602.517, max 602.829). My re-profile of the unchanged file gave 602.312 us. That 0.05% difference is the noise floor. **Run-to-run spread on this kernel is about 0.05-0.1%, so any win above about 0.3% is real.** The spread is not the problem. The lever is.

## 3. Safety
There is no edit, so there is nothing to loosen. `semaphores.py` on the unpacked baseline shows supply equal to demand for the DMA queue semaphores, with no wait above its supply. Baseline verify error is 6.245e-02 against a limit of 0.075 (83%). That is about one bf16 ulp at magnitude 8-16, so only about 0.0126 of headroom remains. Any change to arithmetic dtype or rounding risks a second ulp or a flipped rounding, which fails the verifier.

## Where the time goes (measured)
Per-instruction durations come from `prof/Instruction.parquet`.

| engine | work | busy |
|---|---|---|
| DVE | 600 TENSOR_TENSOR (588 fp32 + 12 bf16-in), mean 672 ns, 512 elems, 128 partitions: 403 us | 592 us (98.3%) |
| DVE | 192 TENSOR_TENSOR_SCAN (MULT/ADD, fp32), mean 1268 ns: 243 us | |
| DVE | 395 EVENT_SEMAPHORE, mean 54 ns: 21.5 us | |
| PE | 384 MATMUL, 591 ns | 219 us (36%) |
| ACT | 204 ACTIVATE, 723 ns | 146 us (24%) |
| Pool | 192 COPY and 60 DMA triggers | 100 us (17%) |

The kernel is DVE-bound. Everything else is mostly idle, and the exec time (602 us) is DVE busy plus about 10 us.

Structure, from `isa.py` on DVE0.bin: 6 channel tiles x 2 chunks of 512 x 16 states = 192 scans. Per (tile, chunk, state) the DVE runs:
- `MULT` (dt*x * B-broadcast) to make dBx
- `SCAN` h = dA*h + dBx
- `MULT` C-broadcast * h
- `ADD` into the y accumulator at SBUF 18752

dA comes from ACT (exp) and the B and C broadcasts come from PE and the Pool COPY. Each DVE op waits on a semaphore: sem 3, 4, 5 or 6 (6 is its own completion).

Timing experiments run on copies in /tmp, timing only, outputs wrong, `kernel.neff` untouched:

| experiment | result |
|---|---|
| All fp32 TT ops: 512 elems -> 256 | TT 672 -> 405 ns, exec 602.6 -> 445.0 us. Cost model: **about 1.04 ns/elem plus a fixed cost of about 139 ns per op** |
| All TT ops with dtype in/out set to BF16 (still 512 elems) | TT 672 -> 407 ns (about 0.52 ns/elem), exec 445 us |
| Existing 12 TT ops with bf16 inputs and fp32 output | still 680 ns, so bf16 speed needs more than bf16 inputs. Whether it needs bf16 output is untested |
| Scan input dtype fp32 -> BF16 | scan 1268 ns unchanged, exec 602.7 us. **Scan speed is dtype-insensitive** |
| Scan 512 -> 256 elems | 1268 -> 736 ns, so about 2.08 ns/elem plus a fixed cost of about 203 ns |

## 4. Ranked candidates for iter-1

### C1. Merge the two chunk scans into one 1024-element scan (cheap, low risk)
- **Engine and slots:** DVE0.bin, the 192 TENSOR_TENSOR_SCAN_ARITH slots, as 96 pairs. Each pair is the chunk-0 and chunk-1 scan for the same (tile, state). Examples: decoded slots 61 and 66 look like the pair family (src0 20800 vs 33088 etc.).
- **Change:** in the first slot of each pair, set `src0/src1/dst_mem_pattern_num_elem` to [1024,1], provided chunk-0 and chunk-1 buffers are adjacent. Otherwise use a 2-D pattern with a stride step. The second slot becomes a NOP or a pure-semaphore EVENT_SEMAPHORE that keeps the instruction count and the sem-6 completion count.
- **Mechanism:** the scan carries state across the whole free dimension. The fixed cost is measured at about 203 ns (see the experiment above).
- **Gain and ceiling:** about 96 x 203 ns, about 19 us (3%). The ceiling is that fixed cost. The bytes moved are unchanged.
- **Risk:** the dA/dBx/h buffers are rotating buffers (20800, 33088, 45376, 24896, ...) that are not adjacent. The pair therefore needs contiguous buffers, which is an allocation change. The wait and update semaphores of the two ops must also be merged. A wrong semaphore count can pass the verifier once and race later. A cross-chunk carry error would show up in the verifier.
- **Cheapest falsifier:** first check from the decoded slots whether chunk-0 and chunk-1 sources are contiguous in SBUF. If they are not, C1 is dead in 5 minutes.

### C2. Make the fp32 TT ops bf16 where the precision allows (largest ceiling, high precision risk)
- **Engine and slots:** DVE, the C*h MULT and the dBx MULT, among the 588 fp32 `TENSOR_TENSOR_ARITH_OP` slots. Producers must write bf16: the scan's `out_dtype` (scan speed is dtype-insensitive) and the Pool COPY's PSUM->SBUF output for the broadcast operands.
- **Mechanism:** the experiment above measured TT at 407 ns when all dtypes are BF16, against 672 ns for fp32, which suggests about 2x elem/cycle. Only a full bf16 setting was tested, so whether mixed settings work is unknown.
- **Gain and ceiling:** at most 588 x 265 ns = 156 us (26%) if every TT went to bf16. That is not achievable within the error budget. Converting only the C*h MULT (192 ops) caps it at about 51 us (8.5%).
- **Risk:** high. Verify error is already 83% of the limit with only 0.0126 of headroom, and bf16 rounding of h before the readout adds error. The verifier uses fixed inputs. The recurrence state stays fp32 if only the readout is converted.
- **Cheapest falsifier:** a timing-only patch of just the C*h MULT slots (out_dtype BF16, stride adjusted) and a verify run. Kill it if max_abs_err exceeds 0.07 or if the TT stays at 680 ns, which would mean the fast path needs a bf16 output specifically.

### C3. Merge the 16 per-state TT ops into fewer, larger ops (medium-high gain, high effort)
- **Engine and slots:** DVE, the dBx MULT, C*h MULT and accumulate ADD slots, 588 in total.
- **Change:** use a 2-D pattern [512, k] with a stride between the per-state buffers to amortize the roughly 139 ns fixed cost per op. For example, the 2-way ADD becomes one op, which needs the h tiles to be equally strided.
- **Gain and ceiling:** 2-way merge saves about 41 us (7%). The ceiling is 588 x 139 ns = 82 us (13%) for an infinite merge.
- **Risk:** needs equally spaced rotating buffers (SBUF allocation change) and merged semaphore waits. A semaphore error can pass verify once and race later.
- **Cheapest falsifier:** the same buffer-spacing check as C1.

### C4. Offload the 204 ADD accumulations from DVE to the PE (large ceiling, large effort)
- **Engine and slots:** PE0.bin, replacing idle EVENT_SEMAPHORE slots (the PE has 347 of them and is 64% idle), plus an identity lhsT. The 204 ADDs on DVE0.bin (`dst=18752`, 18752 += tile) become matmuls that accumulate P_n into PSUM (identity x P_n, start/stop flags). A final PSUM->SBUF copy is also needed.
- **Mechanism:** the PE is only 36% busy, and y is a sum over 16 states.
- **Gain and ceiling:** up to 204 x about 680 ns = 139 us (23%) of DVE time, less whatever the new critical path costs.
- **Risk:** fp32 matmul is multi-pass on the PE and adds rounding. New instructions need slots, which are fixed in the stream, so the free slots must be found. Whether the PE can absorb 192 fp32 [128x512] matmuls is unknown.
- **Cheapest falsifier:** measure the duration of the existing 384 matmuls and whether fp32 doubles it. Kill it if 192 extra matmuls exceed about 280 us of PE time.

### C5. Remove the DVE self-wait EVENT_SEMAPHORE slots (cheap to test, low ceiling, loosens sync)
- **Engine and slots:** DVE0.bin, the `EVENT_SEMAPHORE W 6 GE_IMM n` slots. There are 395 EVENT_SEMAPHORE slots with mean 54 ns, about 21.5 us in total.
- **Ceiling:** about 21 us (3.5%).
- **Risk:** this is removed synchronisation. It can pass once and race later, and the verifier cannot catch it. I would use it only after C1-C3.
- **Cheapest falsifier:** NOP a handful of the same-engine sem-6 waits. A WAR/RAW hazard on the shared 18752 accumulator shows up immediately as wrong output.

## Closed (measured, do not redo)
- **Scan dtype change for speed:** 1268 -> 1268 ns, and exec 602.7 -> 602.7 us. Scan is not faster with bf16.
- **Cutting the scan work itself:** the scan costs about 2.08 ns/elem, which is 204 us of the 243 us scan total and cannot be shrunk below the elem cost without changing the arithmetic.
- **ACT or Pool taking the TT ops:** not tested. See the unknowns below.

## Unknown (measure these first)
- **Does bf16 TT speed need a bf16 output, or only bf16 inputs?** Settle it by patching the C*h MULT out_dtype to BF16 and reading the duration.
- **Are the chunk-0 and chunk-1 scan buffers adjacent in SBUF, and is each chunk-1 scan seeded from the chunk-0 last column?** Settle it from the `imm0` pointers and the src/dst addresses of the decoded scan slots.
- **Can Pool run an fp32 TT 512 at DVE speed?** Pool is 83% idle (about 500 us), and 588 TT x 672 ns = 395 us of DVE work is a candidate. Settle it by finding an existing Pool TENSOR_TENSOR example, or measuring a Pool COPY patched into TENSOR_TENSOR. Moving an op needs a new slot and semaphore rewiring in the Pool stream, which has 42 EVENT_SEMAPHORE slots and 1 NOP.
- **Whether the DVE self-waits are required.** Settle it with the NOP test in C5.

## Recommendation
Do C1 as the first test. The buffer-adjacency check takes five minutes and is worth about 19 us. If the pairs are not adjacent, take C2 with the C*h MULT only, since the 407 ns bf16 timing was measured on the device. If the verifier does not accept the extra bf16 rounding, fall back to C3. Do not start with C4.

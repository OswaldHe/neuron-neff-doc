# Review of iter-4 (kernel.neff vs HEAD 119bfcf, and vs baseline)

I diffed against HEAD (the previous accepted iteration), because the baseline diff also contains iterations 1-3.
`notes/` is empty: the agent wrote nothing, so "consistent with the notes" cannot be checked.
The only claim available is the HEAD commit message (40.5 us). I did not re-measure HEAD.

## 1. What changed

10 members changed, with identical shapes in `sg00` and `sg01`. Both subgraphs got the same edit.
`Activation0` is +1 slot (70 -> 71), `DVE0` is +1 slot (122 -> 123), `Pool0` is 7 -> 10 slots, and `PE0` and `SP0` have the same slot count.

| Member | Change |
|---|---|
| `Activation0.bin` | Slots 13-32 reordered: one more `ACTIVATION_TABLE_LOAD` (ACT_TABLE_LOAD total 7 per core, was 6) plus a CAST/ACTIVATE/READ_ACCUMULATOR shuffle. The per-head `func30` accumulate, `READ_ACC`, `func29` (scale 1/128, bias eps) chain is re-pointed. `events_wait_idx`, `events_semaphore_value` and `events_update_*` are renumbered. |
| `DVE0.bin` | One extra TENSOR_TENSOR. LOAD_MASK/STREAM_SHUFFLE, SUBTRACT/MULT and `src1` operands are re-addressed (PSUM `0x2001000`-style vs SBUF). Wait values are changed. |
| `PE0.bin` | 39 of 80 slots differ, with no change in count. |
| `Pool0.bin` | 3 more DGE descriptors. The final state write-back is split into 4 + 4 + 2 + 2 heads (`src_num_elem [4,128]`, `[4,128]`, `[2,128]`, `[2,128]`), each waiting on sem 6 at a different count. |
| `SP0.bin` | One slot changed. |

Net effect on the ACT stream is a re-ordering, not a removal. ACT instruction count is 193 vs 385 at baseline. The profile gives scalar 193 / vector 298 / tensor 215 / gpsimd 72 instructions.
The verifier-facing arithmetic is unchanged: `output_0` max_abs_err is 0.02322 and `output_1` is 0.00163, both identical to the baseline to all printed digits.
That is 86% of the 0.027 limit. It has not moved since iteration 0, so the error is a property of the kernel and not of the edits.

## 2. Is the latency change real?

Yes.
- Measured now: samples 38.559 / 38.714 / 38.464 / 38.739 / 38.808, median 38.71, spread 0.89%.
- My own re-run (3 repeats): 38.52 / 38.59 / 38.84, median 38.59, spread 0.84%.
- Against HEAD's recorded 40.5 us, that is about -1.9 us (-4.7%). It is more than 2x the total spread (about 0.35 us), so it is not noise.
- The baseline is 76.05 us, so the total is 1.97x.
- At this size the spread is under 1%, so the loop is not chasing noise. Anything below about 0.4 us is indistinguishable, though.

Note `total_exec_time` (38.6) is the kernel body only.
In the per-instruction table the body runs from about 12.1 us (after the runtime preamble) to about 50.6 us, when the last output DMA completes.
The roughly 2 us of end-of-program semaphore/drain after that, and the roughly 12 us preamble before it, are outside the metric.

## 3. Safety beyond the verifier

`semaphores.py` for old/new, `sg00` and `sg01`:
- Every semaphore has `max_wait <= total_inc`, and no row is flagged.
- sem 4 went 40 -> 41 (extra ACT CAST) and sem 6 went 78 -> 79 (extra DVE TT). Waiters are renumbered accordingly.
- sem 27 went 16 -> 64 (4 write-back DMAs x 16) with a single waiter at 64, in both subgraphs. That is the Pool end-of-kernel wait, so it is correct.
- `sg00` and `sg01` have identical semaphore tables. I found nothing that makes the two cores disagree.

Concerns:
- `max_wait == total_inc` shows only that no wait can hang. It does not show that waits are correctly ordered.
  A cumulative-count wait that is too small passes early on a slow run and races. `verify.py` ran once (`"runs": 1`) on one input set.
  Before accepting a reorder of this kind, run `verify.py --runs 5` or more and compare the outputs bit for bit.
- The error figures are bit-identical to HEAD (per the verify output). That is the right signal here, because a race would usually show up as a changed output.
- The write-back split puts the last chunk's DGE issue behind the last head's sem-6 wait. That is correct as long as sem 6 is cumulative. It is.

## 4. Where the remaining time is (sg00; sg01 is within 0.1 us on every number)

- Body is 12.1 -> 50.6 us.
  - Prelude 12.1 -> 28.0 (about 16 us): ACT is almost continuously busy, mostly table loads.
  - Steady state 28.8 -> 49.4 (about 21 us): 12 heads.
  - Tail: last ACT 48.7, last DMA done 50.6 (about 1.9 us).
- ACT_TABLE_LOAD: 7 x 1.283 us = 8.98 us per core. Table sequence `1,0,2,0,4,3,2`; 0 and 2 each appear twice.
  They sit in the window 12.1-26.8 in which ACT is busy more than 90% of the time.
- DVE busy 34.0 us (88%), ACT 33.1 us (86%), PE 18.7 us (48%), DMA 19.3 us (50%).
- DVE: 53 TENSOR_TENSOR x 330 ns = 17.5 us.
  - About 48 of these are per-head `channels=1`, 128-element ops at about 280 ns.
  - The measured cost per element on a single-lane op is about 1.1 ns (the 1536-element TT took 1.7 us).
  - So a 128-element single-lane op is about 140 ns of work plus about 140 ns of fixed overhead.
- ACT per head: `func30` accumulate (266 ns) + `READ_ACCUMULATOR` (295 ns) + `func29` on one element (200 ns), about 0.76 us x 12 = 9.1 us. Plus 14 CASTs at 366 ns.
- Input state DMA (786 KB per core) lands 14.8 -> 20.7. The first state consumer starts at about 24.7, so input DMA is not critical.

## 5. Candidate edits, best first (gain per risk and effort)

| # | Candidate | Ceiling |
|---|---|---|
| 1 | Remove or merge duplicate `ACTIVATION_TABLE_LOAD`s in the ACT prelude | <= 2.6 us (2 loads x 1.283) |
| 2 | Batch the 12 `func29` single-element ACTIVATEs in groups of 4 | <= 1.6 us ACT busy |
| 3 | Collapse per-head DVE single-lane TT/SCALAR_TT ops into 4-head strided ops | <= 5 us DVE busy; end to end about 2-3 us |
| 4 | Fuse the twin q/k `func30` + `func29` ACTIVATEs (slots 6/7 with 17/20) | about 1.8 us |
| 5 | Partition-parallel layout for the per-head vectors | unknown, possibly large (about 10 us of DVE busy) |
| 6 | Shorten the output tail (split the `output_0` write or issue it earlier) | <= 0.5 us |

### 1. Dedupe ACT table loads in the prelude (ACT, both subgraphs)

- **Where.** `sgNN/Activation0.bin`, slots 1-19: the loads at slots 1, 3, 5, 8, 10, 13, 19, with `act_tbl_sel` 1, 0, 2, 0, 4, 3, 2. Also the ACTIVATEs that follow each load.
- **Change.** Group the two func-7 activations (slots 4 and 9, both tbl 0) back to back, and group the func-30/29 activations (slots 6/7 and 17/20, tbl 2) back to back.
  That removes the second tbl-0 load and the second tbl-2 load (7 -> 5 loads).
  Better, find which functions are really table-specific. The act-function sets in `act_info.json` list `square`, `identity` and `copy` in every table, so func 30 (accumulate) may not need tbl 2 at all.
  If func 29 is also available in the current table, slot 19 can be dropped outright.
- **Mechanism.** Chapter 3.5 / 7.4: `ACTIVATION_TABLE_LOAD` just makes the set active. A load is only needed when the next function is not in the active set.
  The measured cost is 1.283 us each (`Instruction.parquet`).
  The func-id to set mapping is NOT yet established. I could not resolve func ids 29, 30 and 36 to table names from the docs. Check `generated/enums.md` / chapter 3.5 first.
- **Gain and ceiling.** About 1.3 us per load removed, 2.6 us for two. The ceiling comes from ACT being the busiest engine in the window 12.1-26.8 (9.0 us of loads in 14.7 us).
  It only pays off if ACT is on the critical path there. The first steady-state ACTIVATE waits 2.5 us on PE, so the gain could be smaller.
- **Risk.**
  - Moving ACTIVATEs changes the cumulative sem-4 counts that every later waiter on sem 4 (72 waiters) uses. That needs careful renumbering in `Activation0`, `DVE0`, `PE0` and `Pool0`.
  - The data hazards are the same as before (reads of `0x35600`, `0x35700`, `0x37e00`, `0x37f00`).
  - The verifier would catch a wrong table (wrong values), but not a too-small wait. Run `verify.py` several times.
- **Cheapest falsifier.** On one subgraph pair, replace slot 19 (tbl 2 load) with a NOP, or with `act_tbl_sel` set to the already-active table, and run `verify.py`.
  If the output is wrong, funcs 29/30 need that table and only the reorder variant remains. If it is right, the load can go outright.

### 2. Batch the per-head rsqrt (ACT)

- **Where.** `Activation0` func-29 ACTIVATEs (slots 26, 30, 35, 39, 44, 46, 50, 53, 61, 62, 67, 70). Each is one element, `src == dst` in SBUF at `0x4000`, `0x4008`, ...; scale 0.0078125, bias 1e-6-ish.
- **Change.** Process 4 heads per ACTIVATE with `src_mem_pattern_num_elem [4,..]` and step 2 (the 8-byte spacing of the accumulators).
  The consumers on DVE already wait on sem 4.
- **Mechanism.** ISA patterns allow multi-element strides (the 1536-wide silu in slot 11 shows an ACTIVATE already runs wide).
- **Gain and ceiling.** 8 of 12 ACTIVATEs gone, about 8 x 0.2 us = 1.6 us of ACT busy. The ceiling is the measured 197-202 ns per func-29. Pipeline latency (head h waits for head h+3's accumulate) may eat part of it.
- **Risk.** It serialises 4 heads, so DVE may stall. Numerics are unchanged (same function and operands). The verifier would catch a wrong stride.
- **Cheapest falsifier.** Do one group of 2 heads and check `verify.py` and the profile.

### 3. Collapse per-head DVE ops into 4-head strided ops (DVE)

- **Where.** `DVE0` per-head groups, for example the SUBTRACT (`src0` PSUM `0x36a00`, `0x36c00`, `0x36e00`, `0x37000`, stride 0x200) followed by MULT (`src1` `0x4098`..`0x40a4`, 4-byte stride, step 0 over 128).
  Also the SCALAR_TENSOR_TENSORs (`0x36400`, imm ptr `0x4000`/`0x4008`, ...) and the output MULTs.
  The destination addresses are also contiguous (`0x22400`, `0x22600`, `0x22800`, `0x22a00`).
- **Change.** One TT per 4 heads, with the head count moved into an outer `num_elem` dimension. Use the pattern-3D struct fields (`S3S3D3`).
- **Mechanism.** The same trick iteration 2 used to collapse the 36 scale-pointer ops to two 1536-element ops (HEAD~1 commit message).
- **Gain and ceiling.** At about 140 ns of fixed overhead per op, going from 48 ops to 12 saves about 36 x 0.14 = 5 us DVE busy. End to end it is probably 2-3 us, because ACT (86%) and the PE chain are close behind.
  The ceiling is the measured 1536-element cost of 1.7 us (about 1.1 ns per single-lane element), which means batching cannot beat the per-element cost.
- **Risk.** Waits become coarser (a group completes only when all 4 inputs are ready) and the pipeline stalls more. Sem-6 renumbering across ACT/PE/Pool waiters is needed.
  The verifier will catch address mistakes but not too-small waits.
- **Cheapest falsifier.** Group just the SUBTRACT+MULT pair of heads 0-3 and read the DVE span between 27.8 and 34.5 us. If it does not shrink by about 0.4 us, kill it.

### 4. Fuse the q/k twin ACTIVATEs (ACT, PE)

- **Where.** `Activation0` slots 6/7 and 17/20: func 30 then func 29 on `0x37e00` and `0x37f00` (256 B apart, 12 elements per partition each), plus the matching PE sum matmuls.
- **Change.** One 2x12-element func 30 and one func 29, and one matmul instead of two. It also removes the second tbl-2 load.
- **Gain and ceiling.** About 1.8 us (2 ACTIVATEs + 1 table load + the PE slot). Ceiling: ACT activity in the window 12.1-26.8, which is also bounded by when the k DMA lands (sem 26 at about 16.4).
- **Risk.** Higher effort (three engines, both subgraphs). Hazard: the k DMA (Pool DGE #3) is the last to issue (15.2-16.4) and gates the merged ACTIVATE.
- **Cheapest falsifier.** Check that the two source tiles can share one AP with step 128 elements, then try the func-30 merge alone.

### 5. Partition-parallel per-head vectors (PE, DVE)

- **Observation.** Nearly every per-head DVE op reports `channels=1`: the 128-vector sits on one partition, so one head costs a full op.
  If head h's result lived on PSUM/SBUF partition h, one TT with `channels=12` would cost about 300 ns instead of 12 x 280 ns.
- **Change.** Start each head's matmul output at PSUM partition h (the dst start address partition field of the MATMUL), and put the per-head scalars (func-29 results) on partition h too.
- **Gain and ceiling.** Unknown. The DVE TT time (17.5 us busy) bounds it above, the ACT per-head chain below. Possibly up to about 10 us of DVE busy.
- **Risk.** Highest. Whether the PE dst partition field can be set per-matmul is not established. It also needs the scalar broadcast to be per-partition. The verifier would catch wrong values but not hazards.
- **Cheapest falsifier.** One matmul with a changed dst partition start and one TT with `channels=2`. Check by verify on a single head.

### 6. Output tail

- **Where.** `output_0` goes out on `qSyncIO0` (issue 49.5, data 50.2-50.6) after the last TT.
- **Gain and ceiling.** At most 0.5 us. The write-back DMA is already split in 4 chunks. Not worth the risk unless everything else is exhausted.

## 6. Closed (already refuted or already done, with numbers)

- ACT silu hoist and table-load reduction 22 -> 6: done in commit df60817 (76.0 -> 55.6 us).
- Scale-pointer ACTIVATEs to DVE, wide S->S_bf cast: done in 400b689 (55.6 -> 45.0 us).
- Deleting the `o_sb` PSUM->SBUF copies, `z` first in the input DMA order, S_bf cast ACT -> DVE: done in 119bfcf (44.8 -> 40.5 us).
- This iteration: write-back DMA split 4/4/2/2 plus ACT prelude reorder: 40.5 -> 38.6 us.
- Re-ordering the Pool input DGE descriptors so the state DMA is first: state lands 14.8-20.7 and the first state consumer starts at about 24.7. Not critical, expected gain about 0.
- Using bf16 / `performance_mode` 2x on the single-lane fp32 TTs: changes numerics. The verifier error is already at 86% of the limit, so it is excluded.
- DMA: 50% busy, 20 of 38 us is the 786 KB state in plus out. Not the bottleneck.

## 7. Unknown (measure these)

- Function-id to table mapping (ids 5, 7, 9, 29, 30, 36 vs `act_tbl_sel` 0-4). It decides how many of the 7 table loads are really needed (candidate 1). Measure by decoding `activation_func` against `exp_and_others.json` / `act_info.json`, or by the NOP test.
- Whether ACT is on the critical path in 12.1-28.0. Measure by deleting one `ACTIVATION_TABLE_LOAD` (invalid numerics are fine for a timing probe) and reading `total_exec_time`. A drop of 1.2 us confirms it.
- Whether the PE MATMUL dst partition can be varied per head (candidate 5).
- HEAD's median under the same session. I did not re-measure HEAD, so the 40.5 -> 38.6 comparison uses the recorded number.
- Run-to-run safety of the new write-back split: `verify.py --runs 5`.
- The agent left no notes. The next agent should write one file per line of attack as the README asks.

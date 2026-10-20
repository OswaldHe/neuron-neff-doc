# iter-1 notes — gdn_tkg bh24 k128 v128, LNC=1 (24 heads on ONE core)

## Result
**153.58 us -> 79.59 us, -48.1% (1.93x).** `verify.py` PASS on every kept step with
`max_abs_err` **exactly** the baseline values (2.322e-02 / 1.633e-03) — every kept edit is pure
schedule / placement / data-movement, the arithmetic is bit-unchanged. `launch_rules.py` PASS
(1 subgraph, num_tpb 1, 10 inputs, 2 output heads). `semaphores.py` clean,
`scripts/cycle.py` acyclic after every step.

Run-to-run spread of `profile.py --repeats 5` here is **0.15–0.5 %** (~0.1–0.4 us), so ~0.5 us is
the believable threshold. Two of my "obvious" head-latency edits regressed by 3+ us — measure,
don't reason.

| # | edit | total us | ACT busy | note |
|---|---|---|---|---|
| 0 | baseline | 153.58 | 143.5 | ACT 93.5 % busy, saturated |
| 1 | 24 silu f36 -> ONE 3072-elem ACTIVATE at the front, private dst; 46 -> 7 `ACT_TABLE_LOAD` | **98.22** | 87.8 | -36 %, the big one |
| 2 | delete the 24 ACT `COPY out_p(PSUM)->o_sb(SBUF)`; both readers read PSUM | **93.22** | 82.8 | -5 us |
| 3 | share one `natural_log_exp` load between the two f7 (7 -> 6 loads) | 93.15 | 81.9 | ACT -0.9 us, total ~0 (became idle) |
| 4 | 24 ACT `k_f[h] *= k_scale[h]` -> ONE DVE `TENSOR_TENSOR(MULT)` | **86.19** | 71.1 | -7 us |
| 5 | 24 ACT `S_all[h] *= exp_g[h]` -> ONE DVE `TENSOR_TENSOR(MULT)` | **84.44** | 63.5 | -1.75 us (7.6 us of ACT, rest became idle) |
| 6 | move the [t4, merged silu] pair off the ACT head | 87.73 | 63.7 | **REGRESSION +3.3, reverted** |
| 7 | move the merged k_f DVE op after the `a` broadcast shuffles | 84.07 | 63.6 | noise-level, kept |
| 8 | hoist [t0, f7(A_log)] to the very front of ACT | — | — | **verify FAILED (race), reverted** |
| 9 | split the 1.5 MB state-out DGE DMA into 4 / 8 per-head-group transfers | 80.41 / **80.14** | 63.5 | -3.9 us; 8-way beats 4-way by 0.27 (ranges disjoint) |
| 10 | reorder the ACT head: t1 f30 f9 / t0 f7(A_log) before the silu | 83.58 | 65.2 | **REGRESSION +3.4, reverted** |
| 11 | swap input DMA transfers 3 (v,k) and 4 (norm_weight,z) so z arrives first | **79.48** | 63.7 | -0.66, ranges disjoint |
| 12 | merge the 24 `S_bf[h]=bf16(S_all[h])` CASTs into one | 79.54 | **60.2** | ACT -3.5 us, **total unchanged** -> reverted |
| 13 | split the merged `S*=exp_g` DVE op in two halves + 2 ACT barriers | 79.52 | 63.5 | **no change, reverted** (see Closed) |

The apply scripts for every step are in `iter-1-scripts/` next to this file
(`apply1.py` … `apply12.py`, plus `common.py`, `cycle.py`, `sbuf.py`, `chain.py`, `dump.py`).
They run against `work/payload/sg00` after `neff.py unpack`, in numeric order, each on top of the
previous. `apply6/8/10/12` are the reverted ones — **do not re-apply 6, 8 or 10**.

## The single most important structural fact about THIS NEFF
It is **not** the LNC2 build. With 24 heads on one core the compiler already vectorised every
*scalar* op across all 24 heads (`num_elem=[24,1,1]`), so the per-head work that remains on ACT is
exactly four families, and three of them turned out to be collapsible into **one** instruction
each because the per-head quantity is a **contiguous array of fp32 scalars** and the per-head
tensor slice is **contiguously strided**:

| family | count | ACT cost | what it is | what I did |
|---|---|---|---|---|
| f36 silu(z) | 24 | 11.8 us + 40 table loads | `z` only, 128 fp32, nch=1, src = `8605184 + 512h` contiguous | one 3072-elem ACTIVATE, private dst `0x20000` |
| f128 nch=1 bf16 | 24 | 11.7 us | `k_f[h] *= k_scale[h]`, src=dst=`209920+256h`, `sptr=16968+4h` | one DVE `TENSOR_TENSOR(MULT)` |
| f128 nch=128 fp32 | 24 | 11.6 us | `S_all[h] *= exp_g[h]`, src=dst=`114688+512h`, `sptr=16576+4h` | one DVE `TENSOR_TENSOR(MULT)` |
| f128 nch=1 fp32 misc | 24 | 11.7 us | per-head delta/beta, scattered src/dst | **not touched** |

### The broadcast trick that makes a per-head pointer-scale op mergeable  (NEW, verified)
An ACT `ACTIVATE f128` with `scale_src = POINTER_IMMEDIATE` reads **one** fp32 per partition at
`p*0x40000 + scale_value_imm_ptr`, so 24 heads need 24 instructions. But a DVE
`TENSOR_TENSOR_ARITH_OP(MULT)` can do all 24 in one slot because the access pattern is 4-D with
signed per-dimension steps and **step 0 repeats an element**:

```
src0 = tensor base,   num_elem=[128,24,1], step_elem=[1,128,1]   # 24 contiguous head slices
src1 = scalar array,  num_elem=[128,24,1], step_elem=[0,1,1]     # broadcast scalar h over 128 elems
dst  = tensor base,   num_elem=[128,24,1], step_elem=[1,128,1]
in0_in1_dtype_dtype_lo = src0 dtype, _hi = src1 dtype   (BFLOAT16 x FP32 -> BFLOAT16 works)
op = MULT, num_active_channels = 1 or 128
```
**The result is bit-identical to the ACT op** (`max_abs_err` unchanged to the last digit on both
outputs, twice). Build the slot by `isa.decode(existing_TT.encode())` and set fields: the
`*_addr_immediate` field is the whole 4-byte address, so writing a value < 2^24 clears the marker
byte for you. `bias_src=INSTRUCTION_IMMEDIATE, bias_imm_fp32=0.0` on the ACT side means there is
no addend to reproduce.

### Where the ACT table loads went: 46 -> 6
`ACTIVATION_TABLE_LOAD` costs a fixed **1283 ns** and the baseline had **46 = 59 us = 38 % of the
kernel**. The thrash is f29 (rsqrt, only in `reciprocal_sqrt_and_small`) alternating with f36
(silu, only in `silu_and_others`). Hoisting all 24 f36 into one instruction removes the
alternation entirely; afterwards the whole tail of the stream only needs f29/f30/f128, all of
which are in `reciprocal_sqrt_and_small`.
* `act_tbl_sel` is an index into the 5 tables **this NEFF ships**, not the 21 vendor sets. Derive
  the mapping from the stream (each load's value must contain the funcs that follow it). Here:
  `0=natural_log_exp_and_others, 1=softplus_and_others, 2=reciprocal_sqrt_and_small,
  3=sigmoid_and_others, 4=silu_and_others`.
* All 46 loads have **no wait and no update**, so they can be deleted/moved/re-pointed for free.
* Final order is `t4(silu) t1(f30,f9) t2(f29) t0(f7,f7) t3(f5,…) t2(everything else)` = 6.

### SBUF free space in this NEFF (partition-offsets, union of all 5 streams)
used `0x04000-0x0efe0` (sparse), `0x1c000-0x1f000` (S_all, 24x512 B), `0x33100-0x37e60`
(a/b/k/v/z/norm_weight), `0x37f00-0x37f60`.
**free: `0x0efe0-0x1c000` (53 KB) and `0x1f000-0x33100` (82 KB)**. I used `0x20000-0x23000` for
the 24 private silu buffers. (`iter-1-scripts/sbuf.py` recomputes it; it scans compute streams
only — cross-check `SP0.json` `to_off` by hand, and note **`to_sizes`/`from_sizes` in the DMA JSON
are BYTES, not elements**.)

## Edits that worked, with the safety argument that made them safe
1. **silu hoist + merge.** The 24 f36 depend only on the `z` input DMA; the blocker was 8 reused
   dst buffers. Each silu has exactly one consumer: the DVE `TENSOR_TENSOR` whose `dst =
   25197024 + 256h`, reading the silu as **src1**; the head index comes from
   `(silu_src - 8605184)/512` and matches `(dve_dst - 25197024)/256` one-to-one for all 24. Give
   each a private dst, re-point the 24 `src1`, merge, delete the dead loads.
2. **Deleting the 24 `COPY out_p(PSUM) -> o_sb(SBUF)` (6 us).** Identify the two readers by the
   chain `COPY -> f30 ZERO_ACCUMULATE reduce (same src) -> ACTIVATION_READ_ACCUMULATOR -> f29
   (reads the acc dst) -> the DVE SCALAR_TENSOR_TENSOR whose s4 wait resolves to that f29 and
   whose src0 is the same o_sb`. All 24 resolve uniquely. Then:
   * re-point both readers at the PSUM bank,
   * **transplant the COPY's own `s5` wait onto the f30 reduce** (the reduce's old wait pointed at
     the COPY, so retargeting it conservatively would have produced a self-wait = deadlock),
   * all 24 PSUM WAR constraints (reader before the next MATMUL writing that bank) were **already
     implied** by the happens-before graph — checked by reachability, zero added waits.
3. **Split the state-out DGE DMA (-3.9 us).** `Pool@4 PSEUDO_DMA_DIRECT2D` writes all 1.5 MB of
   `S_all` (src pattern `step=[512,262144] num=[24,128] elem=512`, src addr is the VAR_ID form with
   `var_offset=114688`; dst is the DGE-table form, `table_index=3`, `table_offset=0`,
   `step=[65536,512]`). Its `s6>=90` wait was already tight on the last `S += outer`. Split into
   N copies with `src_start_addr_addr_var_offset += 512*PER*g`,
   `dst_start_addr_addr_table_offset += 65536*PER*g`, `src/dst_num_elem=[PER,128]`, each gated on
   the last `S += outer` of its own head group, all sharing semaphore 27 with `sem_increment=16`;
   the final `NOP` then waits `s27 >= 16*N`. **Inserting extra `PSEUDO_DMA_DIRECT2D` slots on Pool
   is fine** (the LNC2 note that Pool insertions hang applies to *other* opcodes). 8-way (3 heads
   each) beat 4-way by 0.27 us; it costs ~0.7 us of Pool time per extra trigger and Pool is idle.
   The 24 DVE `S += outer` are in head order, so head groups are contiguous s6 ranges:
   gates for 4 groups were s6>=33,53,71,90.
4. **Swap input DMA transfers 3 and 4 (-0.66 us).** qSPIO0 runs its transfers in order; `z`
   (transfer 4) sat behind `v,k` (transfer 3) and the merged silu is the *first* ACT instruction,
   so it stalled 2.3 us. Swap the two `desc` lists in `SP0.json` and swap `events_wait_idx`
   11<->12 on all five waiters (2 ACT, 3 DVE). Every waiter then still waits for the transfer that
   carries its own data, so the swap is exact regardless of what each waiter needs; the only thing
   to check is that no ACT instruction before the s11 barrier reads transfer-3 data.

## Tooling that paid for itself (in `iter-1-scripts/`)
* `common.py` — load/save all 5 streams as `isa.Instruction` lists; `waits()` / `updates()`.
* **Identity-keyed wait rewriting** — resolve every `W[s{3,4,5,6}>=v]` to its *producer object*
  before the edit, reorder/insert/delete freely, then recompute `v` from the new positions. 12
  edits, ~250 wait values rewritten per edit, zero arithmetic by hand.
* `cycle.py` — nodes `(engine, slot)`, edges = program order + producer->waiter for s3/s4/s5/s6 +
  `SP PSEUDO_DMATRIGGER_0..k -> waiters on queue semaphore 8+k` + `Pool DGE trigger -> waiters on
  24+k`. Reports unmet waits and the cycle. Zero hangs this iteration.
* A **live-activation-table checker**: walk the ACT stream, track the last `act_tbl_sel`, assert
  every `ACTIVATE`'s `activation_func` is in that set's `profile_meta_data[].func_id`.

## THE TRAP that cost me one failed verify — read this before any reorder
**A wait is often a combined guard for more than one operand.** Examples found here:
* `PE LDWEIGHTS` has the wait; the `MATMUL` right after it has **none** and relies on it. So an
  LDWEIGHTS' `s4` wait guards *both* its own `k_f` tile and the MATMUL's moving operand.
* `DVE@5 SCALAR_TENSOR_TENSOR` has one wait that covered **both** `src0=16384` (written by ACT f9)
  and `src1=16480` (written by ACT f7(A_log)), because f7 happened to be later in ACT order.
Edit 8 hoisted f7(A_log) in front of f9 and so broke `output_1` (max_abs_err 0.609). Rules:
* **Deleting** an s4-updating instruction: retarget each wait that resolved to it to the *nearest
  surviving updater at or after it* in the original order (preserves "ACT has progressed this
  far"); if that lands at/after the waiter itself on the same engine, fall back to the last
  surviving updater before the waiter, and justify the data dependency by hand.
* **Moving an instruction earlier**: every wait that resolved to it gets weaker. Enumerate those
  waits and check each operand by hand. There were only 2 here, and I still got it wrong.
* The data guarantee a deleted instruction used to give must be replaced explicitly — I used one
  inserted `EVENT_SEMAPHORE` barrier on the consumer engine (PE for `k_f`, ACT for `S_all`), which
  works because engines complete in program order, so one barrier covers all 24 heads.

## What is CLOSED (measured on this NEFF, do not retry)
* **Moving the merged silu off the head of the ACT stream** — tried twice (edit 6: to just before
  the last table load, +3.3 us; edit 10: with the head's producers hoisted in front of it,
  +3.4 us). The silu's 2.8 us at the front is *absorbing* head latency; anywhere else it adds to
  a region that is on the critical path. **The merged silu belongs at slot 1-2.**
* **Hoisting [t0, f7(A_log)] to the very front** to start the DVE setup chain ~9 us earlier —
  races (see the trap above) and, even with the race, bought nothing (83.84 vs 84.07).
* **Tightening ACT's exp_g wait** (`s6>=8`): already tight. `a` is broadcast to 128 partitions by
  **four** DVE `STREAM_SHUFFLE`s of `nch=32` (partitions 0-31/32-63/64-95/96-127 all at offset
  209664, via addresses `209664 + p*0x40000`), and exp_g reads it with `nch=128`, so it genuinely
  needs the last of the four.
* **Tightening the state-out DMA trigger** (`Pool s6>=90`): already exactly the last `S += outer`.
* **Removing more ACT work**: edit 12 took ACT busy 63.7 -> 60.2 us and total did not move
  (79.54 vs 79.48). ACT is now 80 % busy over a 78.7 us span inside a 79.5 us kernel, and the
  idle is created by cross-engine dependencies, not by ACT. **ACT work reduction is now free but
  worthless.** The remaining ACT families (72 CASTs = 21 us, 24 misc f128 = 11.7 us, 24 f30 +
  24 READ_ACCUMULATOR = 15 us) are no longer worth attacking for their own sake.
* **Splitting the merged `S*=exp_g` DVE op to shorten ACT's 5.1 us barrier stall** — edit 13 did
  exactly candidate 1 below (two halves of 12 heads, one ACT `EVENT_SEMAPHORE` barrier each,
  placed before the first ACT reader of heads 0-11 and of 12-23). Verified, acyclic, WAR-checked —
  and **79.52 us, i.e. no change at all**. So that stall is set by *when* the DVE op can start
  (exp_g, i.e. the four `a` shuffles, i.e. ACT's own f7(A_log) ~11 us in), **not** by its 3 us
  duration. Candidate 1 below is therefore CLOSED in the form I wrote it; what is left is the
  circular head dependency itself: ACT -> f7(A_log) -> DVE@5 -> 4 shuffles -> ACT exp_g -> DVE
  `S*=exp_g` -> ACT. Edits 6, 8, 10 and 13 are four different attacks on that loop and all four
  failed. Whoever tries again should first build a critical-path trace (longest path through the
  happens-before graph weighted by measured `duration_ns`) instead of guessing which link is hot;
  `iter-1-scripts/cycle.py` already produces the graph.
* Merging the 24 f30 square-reduces or the 24 `ACTIVATION_READ_ACCUMULATOR`: the accumulator is
  **per channel** and all 24 `out_p` live on partition 0 (`nch=1`), so one instruction would sum
  all heads together. Closed by the hardware model.
* Merging the per-head pointer-scale ACTIVATEs **on ACT**: impossible, the scale is one scalar per
  partition. (On DVE it is possible — see the broadcast trick.)

## Ranked candidates for iter-2
The kernel is now **dependency-chain bound**, not engine bound: ACT 63.8 us (80 %), DVE 40.4
(51 %), PE 20.8 (26 %), DMA 20.7, Pool 15.1, SP 3.3, in a 79.5 us window whose ACT span is 78.7.
So the only thing that pays is **shortening the critical path or filling ACT's 15 us of idle**.

1. **Build a weighted critical-path trace first.** Current ACT gaps (>150 ns, total ~13.5 us):
   * **5.1 us** before the ACT `EVENT_SEMAPHORE` that waits for the merged `S*=exp_g` DVE op.
     Splitting that op in half changed nothing (edit 13), so the gate is the head dependency loop
     `ACT f7(A_log) -> DVE@5 -> 4x STREAM_SHUFFLE -> ACT exp_g -> DVE S*=exp_g -> ACT`. Four
     different attacks on it failed (edits 6, 8, 10, 13). Do not attack it again by inspection —
     compute the longest path through the happens-before graph with measured `duration_ns` as edge
     weights and find out which link actually dominates.
   * **2.3 us** before the merged silu (the `z` DMA). Edit 11 already moved `z` one transfer
     earlier. Making `z` transfer 0 means `A_log` moves later, which feeds the DVE chain — likely
     a wash, but it is a 10-line JSON edit if someone wants to measure it.
   * ~4.9 us in 10 small (0.3-0.7 us) gaps, each an ACT CAST waiting on `s6` (a DVE
     `S += outer` / RMSNorm op). These are the real per-head critical path; they would need the
     DVE stream re-ordered against its own stalls (the LNC2 iter-7 lever, worth ~1.5 % there).
2. **Re-apply `apply12.py` (merge the 24 `S->S_bf` CASTs) only after 1 succeeds.** It is correct
   and verified; it just has nothing to pay for right now. It frees 3.5 us of ACT work that would
   become real once the stalls above are gone. Same for a line-300 CAST merge (24 more CASTs,
   ~4 us) — but those depend on the 24 individual `S += outer`, so they would serialise the tail;
   merge them in 2-4 groups, not 1.
3. **DVE is now the second engine at 40.4 us.** Do not move more work onto it blindly. The 24
   remaining misc `f128` ACT ops (11.7 us) have scattered per-head src/dst and are mid-chain —
   the one family that genuinely resists the broadcast trick.
4. **Pool/DMA**: the state-out split is done; the tail is only ~1.4 us now. Nothing left there.
5. The baseline spends **83 %** of the error budget, so arithmetic still cannot be touched at all;
   and `verify.py` reporting `max_abs_err` *exactly* 2.322e-02 / 1.633e-03 is the signal that an
   edit is schedule-only. Treat any change in those digits as a failure even if it passes.

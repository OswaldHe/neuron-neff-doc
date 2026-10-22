# iter-6 (ssd_mamba2_370m_s2048): 2988.1 -> 2950.5 us (-1.26%); against iter-0 5248.9 (1.7790x)

**SHIPPED, two edits, both bit-exact (`max_abs_err` identical to the last digit to iters 1-5:
output_0 2.058e-07 = 89% of the limit, output_1 1.993e-07 = 87%), `verify.py --runs 3` all three
runs bit-identical, `semaphores.py` clean (sems 4 and 6 keep `max_wait == total_inc` at
5928 / 4846), `launch_rules.py` pass. Members changed: `sg00/Activation0.bin` (1099 slots) and
`sg00/DVE0.bin` (332 slots), no slot-count change. md5 `dcf3b8b93b18da3c3c82dfa1f94a99b6`.**

| step | edit | median us (5 rep) | spread | ACT busy |
|---|---|---|---|---|
| control (iter-5 binary `1f51a6160f2661428b75f14c145bf0f8`) | - | 2988.1 | 0.11% | 1977.9 |
| **6a** | 671 of 768 `Y_intra` copy+scale folds | **2964.296** | 0.09% | 1813.6 |
| **6b** | 326 ES stand-ins with no semaphore update made fully inert | **2950.524** | 0.16% | 1812.5 |
| 6b re-measured at the end of the iteration | (same binary) | **2950.093** | 0.11% | 1811.4 |

Reproduce from the iter-5 binary:
```bash
neff.py unpack kernel.neff work/
profile.py --repeats 1 --parquet prof       # the plan builder needs the current schedule
ROOT=work PROF=prof python iter-6-mkW.py    # /tmp/W.pkl, the SBUF writer table (iter-5-live.py)
ROOT=work PROF=prof python iter-6-yintra.py            # 6a: 671 units
NOGATE=1 ROOT=work python iter-6-inert.py              # 6b: 326 ES
neff.py pack work/ kernel.neff
```

---

## 0. THE HEADLINE, and it is a negative result worth more than the 1.26%

**ACT-side work has stopped converting into wall time.** Three measured points on the same
kernel, same loop, same kind of edit:

| iteration | what it removed from ACT | wall | conversion |
|---|---|---|---|
| iter-5 | 146 us busy (`Y_off` scale) | -83 us | **57%** |
| iter-6a | 164 us busy (`Y_intra` scale) | -25 us | **15%** |
| iter-6b | 99 us of ACT *stall* (measured, §2) | -14 us | **14%** |

So **budget 0.15 us of wall per 1 us of ACT time removed.** An ACT-side edit now has to delete
>=200 us of ACT time to be worth an iteration, and iter-6a is the proof: it deleted the single
biggest remaining ACT instruction class (768 x 283 ns in-place multiplies, the second half of
iter-5's idiom) and bought 0.83%.

**Why**: iter-6a's deleted multiply sat behind a stall, and the stall simply moved to the
`EVENT_SEMAPHORE` that replaced it (§2 measures this exactly: in-place-multiply sem6 stall
192.2 -> 42.6 us, ES sem6 stall 140.9 -> 325.7 us, net -7 us of stall for -164 us of busy).
The wall is now a **cycle** ACT -> PE -> DVE -> ACT, not a single busy engine
(ACT 61%, DVE 56%, Pool 53%, PE 43%).

§5 is the critical-path partition of the 2948 us, which is the thing to plan from.

## 1. §6a: fold `Y_intra = copy(psum)` + `Y_intra *= exp_cs_col` (ssd_block.py 476+477)

Exactly iter-5's §2 shape, one line earlier in the source, and **cleaner numerically**: there is
no add-of-zero, it is `copy then scale` -> `scaled read of the PSUM`, one fp32 rounding either
way, and the device agreed (bit-exact on both outputs).

The 768 copies at line 476 split into **411 on DVE and 357 on ACT** (the compiler picked per
unit; `nisa.tensor_copy` can land on either engine). Two different edits:

* **group A (C on DVE, 332 shipped):** the DVE `COPY` slot is overwritten with a
  `TENSOR_TENSOR_ARITH_OP MULT` cloned from one of iter-5's own edited slots, with
  `src0 = exp_cs_col` (`num_elem [32,1,1]`, **`step_elem [0,1,1]`** = iter-5's per-partition
  broadcast), `src1 = Y_intra_psum` (`marker 2`), `dst = Y_intra`, and `C`'s events verbatim.
  DVE cost is unchanged (COPY 190 ns -> MULT 187 ns); the whole 283 ns ACT multiply disappears.
* **group B (C on ACT, 339 shipped):** `ACT COPY` has **no scale field at all** (checked: the
  decoded struct has no `scale_src`/`scale_value_imm_ptr`), so the COPY slot is overwritten with
  a **clone of `R`'s own ACTIVATE** whose `src` is repointed at the PSUM (`marker 2`) and whose
  events are `C`'s. The PSUM read stays in the same slot, so this adds **no PSUM WAR exposure**
  (that is why I did not do the opposite, keep `R` and repoint it at the PSUM: that moves the
  PSUM read later and needs iter-2's `guard_psum`). Net ACT saving per unit is 283 - 95 = 188 ns.
* **both:** `R` (the line-477 ACTIVATE) becomes an `EVENT_SEMAPHORE` carrying `R`'s events plus
  iter-5 §2.4b's mandatory extended `sem4 >= cum4[R]` self-wait.

Gates (`iter-6-yintra.py`, all exact, no timestamp margins):
1. `R` provably after `C`: group A needs `R.wait == (sem6, v)` with `v >= need6(C)`; group B is
   free (`C < R` in ACT program order, ACT is in-order). Rejects 1.
2. `C` provably after the `exp_cs_col` producer `pi`: group A needs `gov4_DVE[C] >= need4(pi)`;
   group B is free (`pi < C` in ACT program order). Rejects 11.
3. exactly one in-place 128x32 ACT multiply at `Y_intra`'s address in `(C.end, R.start]`
   (this is what keeps a *`Y_off`* multiply from being mistaken for `R` -- the two buffers share
   24 of their 40/41 rotating addresses).
4. nothing else writes the 128-byte `Y_intra` tile in the window (writer table).
5. `exp_cs_col` not rewritten in the window. Rejects 8.
6. nothing else *reads* the `Y_intra` tile in the window (new: a reader table built the same way
   as the writer table, 16468 entries).
   -> **748 of 768 pass.**
7. iter-5 §1's ES gate on the inherited sem-4 increment: drain covers / no increment / nobody
   listens. **671 admitted** (574 + 92 + 5), 77 rejected.

**Do not need iter-5's empirical `0..187` prefix clause here** -- all 671 are admitted by stated
rules and the result is bit-exact over 4 device invocations. (iter-5's unexplained per-unit
failures were specific to `Y_off`; they did not reappear for `Y_intra`.)

## 2. §6b: an ES stand-in that carries no semaphore update is pure dead stall

New, reusable, and the cheapest 0.47% in this loop so far.

When iter-3/4/5/6a replace an ACT instruction `R` by an `EVENT_SEMAPHORE`, the stand-in inherits
`R`'s wait. If `R` carried **no semaphore update at all** (`events_update_mode NONE`), the
stand-in has *no observable effect whatsoever* except blocking ACT's in-order stream -- and with
the iter-5 self-wait drain on top it blocks twice. Measured on the 6a binary: the 671 new ES's
stall ACT **152.1 us**, and **all of it (151.9) is on group A**, i.e. ACT waiting for DVE to
finish the multiply it just handed over. Splitting those 332 group-A units by what their
increment is worth:

| class | n | ACT stall |
|---|---|---|
| `R` has **no update at all** -> the ES is a no-op | 237 | **99.5 us** |
| every waiter on `cum4[R]+1` is a DVE slot after `C` | 16 | 9.7 us |
| somebody really does key on `cum4[R]+1` | 79 | 43.2 us |

`iter-6-inert.py` finds every ACT `EVENT_SEMAPHORE` with (no primary update, no extended update,
a `sem6` primary wait, a `sem4` extended wait) -- that pattern is exactly the iter-5/iter-6
stand-ins and nothing else -- and zeroes **both** wait blocks. **326 of them** (237 from 6a,
89 inherited from iter-5, which left this on the table) -> **-13.8 us of wall, bit-exact**.

**The residual risk I did not close, stated plainly.** A pure barrier on an in-order engine can
be load-bearing for the instructions *behind* it even when it has no update of its own: ACT slot
2771 (§4) is exactly such a barrier. I built the gate for this (`iter-6-inert.py` without
`NOGATE=1`): window = `(R, J]` where `J` is the next ACT slot whose own sem6 wait is `>= R`'s,
and admit only if no ACT instruction inside reads any SBUF region DVE ever writes. The window is
short (median 2 slots, max 11) but the DVE-written set is 16% of SBUF, so the gate admits only
**125 of 326**. I shipped all 326 because (a) every one of them replaced an instruction whose
wait was explained by *its own* operand (it read the tile DVE had just written), not a hoisted
barrier, and (b) 3 device runs are bit-identical and bit-exact. **If a future iteration sees
anything inexplicable, revert to the 125 gated ones first** (`iter-6-inert.py` with no
`NOGATE`), it is worth ~5 us of the 13.8.

## 3. Refuted this iteration (measured, do not redo)

**a. `exp_cs_last` is `exp_cs_col[127]`, but you cannot use that.** `exp_cs_last = exp(cs_row[0,Q-1])`
(768 ACT ops, `7,1,1,SBUF,IMM`, 201 ns busy + 89 us of sem6 stall, 70.4 us on the critical path)
and `exp_cs_col = exp(transpose(cs_row))`, so `exp_cs_last == exp_cs_col[Q-1,0]` and the ACT op
is redundant. **It dies on the `STREAM_SHUFFLE` encoding**: the broadcast reads the **base
partition of a 32-partition group** (`shuffle_mask=[0]*32`; the mask is *not* in the 64-byte slot
-- all 64 bytes are accounted for by named fields, the mask comes from the paired
`LOAD_MASK_SELECT`), and all 4608 shuffles in this NEFF use a 32-aligned base (32 or 96 only).
Partition 127 is not 32-aligned, so there is no way to point a shuffle at `exp_cs_col[127]`.

**b. iter-5 §3's proposed fix for the redundant `exp_cs_last` broadcast is impossible: SBUF is
full.** iter-5 left "give the Q broadcast a private per-head destination" as the way to unlock
the 1536 N-group shuffles (227 us of DVE, **263 us of the critical path** -- 151.1 busy + 112.1
gap, the third biggest item in §5). I built the free-space map (`iter-6-sbfree.py`: every
compute operand, every scale/bias pointer, every DGE endpoint and every queued-DMA offset, as a
128 x 57344 bitmap): **the highest byte touched is 0x37F04 and the highest base is 0x37F00 =
0x38000 - 256, i.e. the compiler packed SBUF to exactly 224 KB per partition and there are
0 free 4-byte words that are free on all 128 partitions.** (Partition stride is 0x40000, so
0x38000..0x3FFFF *may* be physical SBUF -- but the allocator stopping dead on 0x38000 is strong
evidence it is not. Do not gamble a whole iteration on it; if you want to test it, do it with a
single-unit probe first.)
Without a private destination the dedup needs the long window `[Q broadcast, ACT line 535]` to be
free of other units' shuffles to the same offset, which is iter-5's 679-of-747 rejection.

**c. The `cs_row -> cs_col` transpose is already minimal.** iter-5 §7 item 3 priced
`ssd_utils.py:67` at 349 us of PE for transposing one 128-vector. The `LDWEIGHTS` already has
`num_active_rows 1`, `num_active_cols 128`, `transpose_mode ENABLED`, `src num_elem [128,1,1]`
with `step_elem [-1,0,0]`: it loads **one** partition. 222 ns + 233 ns is pure issue overhead,
not work. Nothing to shrink.

**d. Why PE is expensive, and why you cannot fix it.** Each `nc_matmul` is **2 `LDWEIGHTS` +
2 `MATMUL`** (`fp32_mode LOW` / `LOW_HIGH`) -- fp32 is emulated in two passes. Lines 470/491/525
cost 1446 / 1112 / 795 us of PE summed. Dropping a pass halves PE but destroys precision, and
the error budget is at 89%. Closed.

**e. Folding line 454 (`dtx = x*dt_col`) into 463/514** would delete 768 more ACT multiplies
(217 us) by precomputing `dt_col*exp_neg_cs_col` and `dt_col*decay_factor`, but (i) it is
`(x*dt)*e` vs `x*(dt*e)`, not bit-exact, with 11% of the error budget left, and (ii) there is no
free DVE/ACT slot at the right program position for the two new column products. Same for
`dtx_decayed = X_scaled * exp_cs_last` (which would also kill line 229's TT). **The error budget
is the binding constraint on every remaining arithmetic rearrangement in this kernel.**

**f. Moving `log_decay` to DVE: attempted on the device, 4 runs, NOT solved -- but every
mechanical doubt is now closed and exactly one question is left.** See §7, it is the best-priced
open lever in the kernel (305 us of ACT busy, 72.4 us of the critical path).

## 4. The 111 us Pool barrier is still there and is still the biggest single item

`Activation0.bin` slot 2771, `wait s23>=800`, no update: **111.4 us** (t=751.7..863.0), during
which GpSimd is **111.7 us busy** issuing 181 back-to-back DGE triggers (Pool slots 822..1002,
84 x line 203 `dt_col` + 83 x line 447 `x_sb`) and **Scalar 0.5 / Vector 1.6 / Tensor 0.2 /
Sync 0.0 us busy**. Unchanged by both of this iteration's edits -- it is 3.8% of the wall sitting
in one slot. iter-4 §4.1-4.3 is still the state of the art on it: Pool arrives 186 transfers
behind because its output stores head-of-line-block its loads, and hoisting is capped at ~20
slots by the round-robin per-queue order, i.e. ~10-20 us of the 111. **Queue reassignment is
still the only thing that opens it** (iter-4 §4.3), and the variant I would try is: give *all*
the line-572 stores one queue and the loads the other 15, so that no load is ever behind a store
in its own queue -- then loads can be hoisted arbitrarily. Price the single store queue first
(768 stores x ~16 KB at ring depth 1).

## 5. Where the 2948 us goes (`iter-6-cpath.py`, iter-3's tracer; measure your own)

Clipped partition of the wall: **covered busy 2210.1 us + gaps 738.4 us = 2948.4**.

| on-path busy | us | | on-path gap (blamed on the waiter) | us |
|---|---|---|---|---|
| Scalar `ACTIVATE` | 821.3 | | Scalar `ACTIVATE` | 192.3 |
| Tensor `MATMUL` | 546.0 | | Scalar `EVENT_SEMAPHORE` (111 of it = §4) | 167.1 |
| Vector `TENSOR_TENSOR` | 171.1 | | Tensor `LDWEIGHTS` | 113.2 |
| Vector `TENSOR_TENSOR_SCAN` | 166.7 | | Vector `STREAM_SHUFFLE` | 112.1 |
| Vector `STREAM_SHUFFLE` | 151.1 | | Vector `TENSOR_TENSOR` | 40.4 |
| Tensor `LDWEIGHTS` | 148.6 | | GpSimd `EVENT_SEMAPHORE` | 37.1 |
| Scalar `EVENT_SEMAPHORE` | 83.3 | | Tensor `EVENT_SEMAPHORE` | 26.2 |
| Vector `MEMSET` | 33.8 | | GpSimd `DMA_DIRECT2D` | 13.3 |

by engine, on-path busy: **Scalar 942.0, Tensor 696.6, Vector 546.3, GpSimd 25.1**.

Scalar on-path time by decoded class `(func, nch, ne0, src marker, scale)`:

| us on path | n | class | source |
|---|---|---|---|
| 444.2 | 1659 | `(128,128,32,SBUF,PTR)` | 454 `dtx`, 463 `X_scaled`, 514 `dtx_decayed` (+ the 341 un-folded in-place) |
| 103.7 | 495 | `(128,64,32,SBUF,PTR)` | **535** `state_sb *= exp_cs_last_N` |
| 99.3 | 866 | `(7,128,1,PSUM,IMM)` | the two `exp(cs)` / `exp(-cs)` per unit |
| 83.3 | 1914 | `EVENT_SEMAPHORE` | |
| 72.4 | 288 | `(128,1,128,SBUF,PTR)` | 102 `log_decay` (§3f) |
| 70.4 | 412 | `(7,1,1,SBUF,IMM)` | 226 `exp_cs_last` (§3a) |
| 24.0 | 98 | `(128,128,32,PSUM,PTR)` | **new in 6a**: group B's folded copy+scale |

Whole-engine gap attribution (`iter-6-gaps.py`, every gap >100 ns labelled by the semaphore it
waits on and the engine that owns it):

| engine | total gap | top blame |
|---|---|---|
| ACT | 882.9 | DVE/sem6 434.8, PE/sem5 313.3, **Pool/sem23 111.4 (one slot)** |
| DVE | 1058.6 | **PE/sem5 676.6** (743 gaps, mean 910 ns), ACT/sem4 376.8 |
| Pool | 1386.2 | DVE/sem6 1003.9, ACT/sem4 97.7 |
| PE | 1611.7 | **ACT/sem4 1202.4**, DVE/sem6 276.9 |
| SP | 2275.9 | ACT/sem4 743.5, then its own DGE rings |

Read it as one cycle: **ACT -> PE (1202) -> DVE (677) -> ACT (435)**. Every engine spends more
time waiting on the next one than it spends working. Time with exactly one engine busy:
GpSimd 160.5, Scalar 125.9, Tensor 98.9, Vector 93.8, Sync 12.7; with none busy 32.7.

## 6. Ranked plan for iter-7

1. **Finish `log_decay` -> DVE (§7b/§7c).** 305 us of ACT busy, 72.4 us of the critical path,
   bit-exact by construction, all the ISA mechanics already proven on the device, and one
   per-unit discriminator left to find. Start with the 15 marker-0 units -- that is a single
   40-second probe and it either opens the whole lever or eliminates the prime suspect.
   **Stop deleting other ACT instructions** unless the edit is worth >=200 us of ACT (§0).
2. **Pool queue reassignment** (§4) -- 111 us in one slot, and the only item left that is worth
   >3% on its own. Census first (iter-4 §4.3): does any single wait cover several transfers of
   one queue?
3. **`state_sb` WAR relayout.** Line 535 (`ACT`, in-place on `state_sb`) stalls **185-313 us on
   PE/sem5**, because it must wait for the line-491 matmul to finish *reading* `state_sb`. Give
   ACT a scratch destination and point line 541's DVE `src0` at the scratch; then ACT's wait can
   become `sem6 >= (line 541 of the previous chunk for this head)`, which covers the RAW on
   `state_sb` *and* the WAR on the scratch in one wait, and the PE stall disappears. Needs
   64 partitions x 128 B of scratch per head -> blocked by §3b until #1 answers, and chunk 0
   needs a different wait (the state DMA).
4. The 97 un-folded `Y_intra` units (~28 us of ACT, 77 of them behind iter-5's ES gate) and the
   79 group-A ES's that really are listened to (43 us of ACT stall): both need iter-2's
   "raise the consumer's wait" pattern. ~10 us of wall between them. Side dishes only.
5. **Do not** redo anything in §3 here, in iter-5 §5, iter-4 §5, iter-3 §4, iter-2 §4, iter-1's
   list.

## 7. NEW ISA/structure facts, and the one open question that is worth an iteration

### 7a. An engine's instruction stream can change length. Both directions, verified on device.

This was listed as unknown by every previous iteration and it is the key that unlocks §6 items
1-3.

| test | DVE slots | result |
|---|---|---|
| delete 38 provably dead `Y_off` memsets (`iter-6-killmemset.py`) | 15169 -> **15131** | **VERIFIED, bit-exact** |
| insert 758 fully inert `EVENT_SEMAPHORE`s, one every 20 slots (`iter-6-grow.py`) | 15169 -> **15927** | **VERIFIED, bit-exact** |

So `neff.py pack` is sufficient: nothing in the NEFF references an absolute instruction index.
Semaphore accounting survives because it is by increment *count*, so a deleted/inserted slot must
simply not change any engine's increment sequence (delete only event-free slots; insert only
event-free slots, or move the events with the slot).
**Caveat measured the hard way: an engine that executes an instruction it cannot honour just
stops, and then the whole machine deadlocks on its semaphore** -- you get
`NMGR:kmgr_exec Failed to execute model: 10001, error: 5` followed by
`Model stop timeout (30000 ms) for TPBs`, and `verify.py` reports `exit 1, ~71 s`. A 71-second
`[FAIL] run 0` is that, not a wrong answer.

Only 38 of the 768 line-486 memsets are event-free; 486 carry events (-> make them
`EVENT_SEMAPHORE`s, 34 ns each) and 244 belong to units iter-5 did not fold (still live).
Total value ~20 us of DVE -- below this kernel's 4.7 us measurement noise once converted, which
is why I did not ship it.

### 7b. A DVE `TENSOR_TENSOR_ARITH_OP` can replace the ACT `log_decay` op

`log_decay = dt_row * A_head` (`ssd_utils.py:102`) as a DVE `MULT` with
`num_active_channels 1`, `src0 = A_head` (`num_elem [128,1,1]`, `step_elem [0,1,1]`),
`src1 = dt_row`, `dst = log_decay`: **96 units (two whole chunks) are bit-exact on the device
with the ACT op deleted**. That closes four separate doubts at once:

* a DVE TT works with **all three operands on different partitions** (here dst p0, src1 p96,
  src0 p32) -- ACT's cross-partition behaviour is not special;
* `src_mem_pattern_start_addr_addr_reg_marker == 1` (750 of the 768 ACT log_decay ops use it,
  as do all 4608 `STREAM_SHUFFLE` destinations) **works unchanged on a DVE source**;
* `step_elem 0` broadcasts over **128** free elements, not just iter-5's 32;
* an inserted slot executes where you put it.

### 7c. ...and it is correct for 96 units with the ACT op deleted. The blocker is now one
thing only: the insert-point gate is schedule-dependent.

**The bug that ate three device runs was mine, and it is a trap anyone editing addresses will
hit: the `..._reg_marker` field IS address bits 24..25** (0 = low SBUF, 1 = high SBUF,
2 = PSUM), and `set('..._immediate', a)` already writes them. My builder then did
`set('..._reg_marker', 0)` "for safety", which cleared bit 24 and turned `A_head` at
`0x181c300` into `0x081c300`. It read a wrong scalar for 750 of the 768 units -- and the one
unit I happened to probe first was one of the 18 whose `A_head` is below `0x1000000`, so the
probe was bit-exact and pointed the diagnosis at the ISA for an hour. Verified directly:
`set(0x181c300)` -> `immediate 0x181c300, marker 1`; `set(0x2004000)` -> `marker 2`.

With that fixed, measured in order:

| variant | result |
|---|---|
| 96 units, `KEEPACT=1` (ACT also still writes the value) | **VERIFIED, bit-exact** |
| **96 units, ACT op replaced by an ES** | **VERIFIED, bit-exact** |
| 192 units | FAIL `output_0` 1.259e-02 |
| 288 units | FAIL 2.551e-02 |
| 540 units (all that pass the gate) | FAIL 2.886e-02 |
| 518 units (540 + iter-5 §1's ES gate on top: 494 drain-covers, 24 no-inc, 22 rejected) | FAIL 2.972e-02 -- **so the ES hazard is NOT the discriminator here** |
| 389 units, same but with a 20 us forward liveness margin on `dt_row`/`A_head` (`GUARD=20000`, which rejects 123 more units for "dt_row rewritten: queued") | FAIL 2.969e-02 -- **so it is not a thin liveness margin either** |
| **540 units with `KEEPACT=1`** (ACT still writes the correct value, so the only way to be wrong is for the inserted MULT to overwrite it with garbage) | FAIL 2.886e-02 -- **decisive: the defect is the MULT's own operands, the ACT deletion is innocent** |

**96 = 2 x 48 heads = chunks 0 and 1.** The boundary is a chunk boundary, and the error grows
monotonically with the number of units, which is the signature of the one gate clause that is
**not** schedule-independent: clause (2), "no writer of `dt_row`/`A_head` between the ACT op and
the insert point", is evaluated against the *old* profile's timestamps. Clause (1)
(`gov4_DVE[s] >= need4(li)`) and clause (3) are static and survive any reschedule; clause (2)
does not, and deleting 397 ns x N of ACT work moves everything. **The edit invalidates its own
gate**, so a prefix works and the prefix length is however far the schedule stays put.

One more fact that the next attempt needs, from `prof/DmaPacketAggregated.parquet`: **`dt_row`
and `A_head` arrive in the *same* queued transfer** (`qSyncIO0`, `write_offset
[0x181c400, 0x181c300]`, `write_shape [[512 1] [4 1]]`, 516 bytes, `variable "dt, A"`), and the
**two buffers swap roles between units** -- `0x181c000` is `dt_row` for one unit and `A_head`
for another. So any liveness argument has to treat the 516-byte pair as one object and cannot
assume an address keeps its meaning.

**What iter-7 should do (and it is a short list):**
1. Re-derive clause (2) in program order instead of timestamps. The writers of `dt_row` and
   `A_head` are queued DMAs; what is needed is "the DMA that next overwrites this buffer is
   provably not yet triggered at the insert point", which is a statement about the Pool/SP
   trigger program positions and the queue thresholds already in the stream -- the same kind of
   argument iter-3/4 made for the `state_sb` bounces.
2. Or sidestep it: `A_head` and `dt_row` have **382 and 395 distinct addresses for 768 units**
   (~2 uses each), so the real exposure is small; find, per unit, the *latest* slot that still
   has clause (1) and then take the minimum over a margin, and re-measure the gate on the
   *edited* schedule (iterate the plan twice: apply, profile, re-gate, re-apply).
3. Or ship the prefix. 96 units is only 38 us of ACT and below this kernel's noise, but the full
   540 is 214 us of ACT busy and ~50 us of the critical path.

## 8. Tool notes (adds to iters 1-5)

* **`ACT COPY` has no scale operand** (no `scale_src`, no `scale_value_imm_ptr` in the struct) --
  to give an ACT copy a scale you must overwrite the slot with a cloned `ACTIVATE`. Both are
  `header_inst_word_len 16` = 64 bytes, so the swap is in place.
* An `ACTIVATE` and a `TENSOR_TENSOR_ARITH_OP` have **no `events_extended_*` block**; only
  `EVENT_SEMAPHORE` does. So you cannot give a working instruction a second wait -- if an edit
  needs two orderings on one slot, it has to be rejected or re-derived.
* The address fields alias: `..._immediate`, `..._reg_regnum` and `..._reg_reserved0` are three
  views of the same bytes (0x8390 -> regnum 0x90, reserved0[0] 0x83). Setting `_immediate` is
  enough; the other views follow.
* `STREAM_SHUFFLE`'s 32x5-bit `shuffle_mask` is **not in the instruction** -- the 64 bytes are
  fully accounted for by the named fields. It comes from the paired `LOAD_MASK_SELECT` (4 ns,
  one per shuffle). The shuffle source is the **base partition of a 32-aligned 32-partition
  group** (observed: only 32 and 96 in this NEFF).
* **The `..._reg_marker` field of any address is just address bits 24..25** (0 = low SBUF,
  1 = high SBUF, 2 = PSUM), aliased over the same bytes as `..._immediate`; `set('..._immediate',
  a)` writes them for you. **Never `set('..._reg_marker', 0)` after setting an immediate** -- it
  silently drops 16 MB off the address. This cost me three device runs (§7c).
* `prof/SbufUsage.parquet` is useless for finding free SBUF (it reports ~0.006% utilisation).
  Build the map from the streams instead: `iter-6-sbfree.py`.
* `Instruction` has no `.get()`; use `name in {f['name'] for f in x.fields()}`.
* iter-3's `iter-3-cpath.py` still works unmodified (needs `scripts/common.py` and
  `prof/SemaphoreUpdate.parquet`); it is the only tool here that partitions the *wall* rather
  than summing per-engine busy, and it is what §5 is built from. Copy kept as
  `iter-6-cpath.py`.
* Device costs today: `verify.py` 40 s, `verify.py --runs 3` 80 s,
  `profile.py --repeats 1 --parquet` 50 s, `--repeats 5` 85 s.

Scripts next to this file: **`iter-6-yintra.py` is §1** (env `ROOT PROF DRY GRP=ab GATE LIM ONLY
SKIP DRAIN`), **`iter-6-inert.py` is §2** (`NOGATE=1` for all 326, default for the gated 125),
`iter-6-gaps.py` is the per-engine gap/blame table, `iter-6-cpath.py` + `iter-6-common.py` is
§5's critical-path partition, `iter-6-sbfree.py` is the SBUF free-space map of §3b,
`iter-6-eswaiters.py` enumerates who waits on an ES's increment, `iter-6-mkW.py` rebuilds
`/tmp/W.pkl` from `iter-5-live.py`, **`iter-6-logdecay2.py` is §7b/§7c** (env `LIM ONLY KEEPACT
DRAIN`; `iter-6-logdecay.py` is the earlier insert-at-the-consumer version kept for its two
recorded failures), `iter-6-killmemset.py` and `iter-6-grow.py` are §7a's two stream-length
tests.

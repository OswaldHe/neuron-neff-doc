# iter-4 notes — gdn_tkg bh24 k128 v128, LNC=1 (24 heads on ONE core)

## Result

**72.604 us -> 69.03 / 69.05 us, -4.9%** (all measured today, `profile.py --repeats 5`; the
final NEFF was measured twice, 69.030 and 69.053, so the step is solid).
`verify.py --runs 4` PASS, **all 4 runs bit-identical**, `max_abs_err` **exactly** the baseline
digits (2.322e-02 / 1.633e-03) and `output_0` bit-identical head by head to the pre-edit NEFF.
That 4-run check matters this iteration: three of the six kept edits only *relax* a semaphore
wait, which is the class of edit that passes once and races later. `launch_rules.py` PASS.
`semaphores.py` balanced (s4 120/120, s5 74/74, s6 103/103). `cycle.py` acyclic.
Spread on the final NEFF 0.21–0.44 % (0.15–0.3 us), so ~0.4 us is the believable threshold.
Final NEFF md5 `f2089be5712709641c1ced4cd4b03992`, **2.223x** baseline (153.46).

Changed members vs iter-3: `Activation0.bin` (170 slots, permuted + 1 moved table load),
`DVE0.bin` (172 -> 173 slots: 22 ops replaced by EVENT_SEMAPHOREs, 1 wide op inserted),
`PE0.bin` (wait values only), `SP0.bin` (one wait value).

Apply order (`iter-4-scripts/REPRO.sh`, run against `work/payload/sg00` after `neff.py unpack`):

```
applyJ.py work/payload/sg00 1     # J1: PE outer-product gate -> the f128 that wrote its operand
applyL.py work/payload/sg00       # L: ACT f29 bubbled left past the f128s
applyK.py work/payload/sg00       # K: DVE output-op s4 fence -> its own producer's fence
movetbl.py work/payload/sg00 24 20 # N: body ACT_TABLE_LOAD moved into the 0.62us ACT idle
applyP.py work/payload/sg00       # P: ACT head permutation 0,3,5,1,2,6,4,7,8,9,10
applyT.py work/payload/sg00 5 no 22 # T22: fuse the two DVE output ops, heads 0..21
```

| step | edit | total us | verdict |
|---|---|---|---|
| in | iter-3 result, re-measured today | 72.604 | |
| J1 | 16 PE `LDWEIGHTS` gating the `k (x) w` matmul: wait `s4 >= <ES fence>` -> `s4 >= <the f128 that wrote the matmul's src>` | 72.463 | noise alone, **kept** (prerequisite) |
| J2 | hoist all 24 DVE `S+=` ops ahead of the RMSNorm output ops | 73.470 | **+0.87 REGRESSION**, dropped |
| J2' | same, restricted to DVE slots <= 141 | 72.597 | wash, dropped |
| J2'' | same, restricted to DVE slots >= 142 (the drain) | 72.131 | **+0.7 REGRESSION**, dropped |
| **K** | 16 DVE output ops (`M`): s4 fence relaxed from the ACT `EVENT_SEMAPHORE` to the s4 value its own `N` already waits on | **72.200** | **-0.40**, kept |
| **L** | 7 ACT `ACTIVATE f29` bubbled left past the `ACTIVATE f128` that preceded them | **71.431** | **-0.77**, kept |
| M | same as L but also bubbling f29 past `CAST` | 71.591 | +0.16, dropped |
| **N** | the 6th `ACTIVATION_TABLE_LOAD` (body, 1.283 us) moved from ACT@24 to ACT@20, into the 0.62 us ACT idle in front of `ES@20` | **70.820** | **-0.61**, kept |
| **P** | ACT head permuted to `0,3,5,1,2,6,4,7,8,9,10` so the recip_sqrt table load happens while ACT is waiting for the `s24` DMA | **70.005 / 69.912** | **-0.82**, kept |
| Q | bubble the 24 DVE delta ops left past the `S+=` ops | — | **DEADLOCK** (semaphore cycle, device NRT timeout), dropped |
| R | generalise J1: relax *every* PE matmul gate to the ACT op that wrote its operand (4 more sites, all `S'_bf^T k`, jumps of 25-78 s4 slots) | 70.138 | **no gain and the dropped fence is unexplained**, dropped |
| S | move `ACT@19 f30` (nch=128, gated on `s26`) into the 1.14 us `f7@10` hole | 73.263 | **+3.35 REGRESSION**, dropped — this measures `s26` landing at ~24.5 us |
| T | fuse the two DVE output ops, **all 24 heads** (see below) | — | **verify FAILED**, only head 23 wrong |
| T' | same, 23 heads (last pair left unfused) | — | **verify FAILED**, still only head 23 wrong |
| T'' | same, 1 head | 70.0 | PASS bit-identical — mechanism is sound |
| **T22** | same, **heads 0..21** (last two pairs left unfused) | **69.030 / 69.053** | **-0.88**, kept |
| J2 again | re-ran the DVE `S+=` hoist *on top of* T22, now that the output ops in front of it are half as long | 70.201 | **+1.17 REGRESSION**, closed for good |

---

## The model of this kernel that the edits came from (read this first)

24 heads, **fully independent** (each has its own 128x128 state). The body is a software
pipeline the compiler built 2 heads deep. Per head the chain is

```
PE: S'_bf^T k  ->  DVE D: delta = v - that        (nch=1)
ACT f128: w = beta*delta (490ns)
PE: k (x) w  ->  DVE U: S += outer                (nch=128, src0==dst, src1=PSUM)
ACT CAST: S -> S_bf (298ns, nch=128)
PE: o = S_bf^T q  ->  ACT f30 (266) -> READ_ACC (293) -> ACT f29 (202) = rstd
DVE N: tmp = (o * rstd) * norm_weight@216064      (STT, src0 = PSUM o)
DVE M: out[h] = tmp * silu(z)[h]                  (TT, writes output_0 directly, bf16)
```

Per 2-head cycle: ACT 3.18 us of work, DVE 2.43, PE 1.1. Measured cycle at the start of this
iteration was 4.08 us, i.e. **0.9 us of ACT idle per cycle**, all of it in two holes:

* ~0.70 us between the last `f128` of the cycle and the first `CAST` — ACT waiting for
  `PE k(x)w -> DVE U`;
* ~0.16 us before the first `f30` — ACT waiting for `PE o`.

**The key discovery: the 0.70 us hole was NOT the PE/DVE round-trip latency, it was DVE
program order.** `U` sits behind `N` and `M` in the DVE stream, `N` waits on ACT's `f29`, and
`M`'s s4 fence pointed at the ACT `EVENT_SEMAPHORE` that only fires after the *next* cycle's
last `f128`. So the chain was

```
ACT f128(last) -> ACT ES -> [M unblocked] -> DVE M -> DVE U -> ACT CAST
```

with the PE matmul (which is what CAST's data actually needs) long finished. Edits K+L+J1
removed the three artificial links: the hole is now 0.30-0.36 us per cycle.

### Useful classifier (in `iter-4-scripts/applyQ.py`, reused everywhere)

Decode the DVE body by address, not by opcode:
* `U` = `TENSOR_TENSOR_ARITH_OP`, `nch=128`, `src0 == dst` (S at 118784..126975), `src1 >= 0x2000000`
* `D` = `TENSOR_TENSOR_ARITH_OP`, `nch=1`, `src1 >= 0x2000000`, dst in the delta ring
  (25182888 / 25182936 / 25183448 / 8604672)
* `N` = `SCALAR_TENSOR_TENSOR_ARITH` with `src0 >= 0x2000000`; dst is one of a 6-slot
  rotating ring 16392 / 17112 / 17880 / 18392 / 24032 / 25304
* `M` = `TENSOR_TENSOR_ARITH_OP` whose `src0` equals some `N`'s dst; `src1` = silu(z) at
  131584 + 512*h; dst = output_0 at 25197280 + 256*h

PSUM banks (base 0x2000000, 2 KB each): `o` lands at banks 7/1/5 rotating, the `S'^T k` matmul
at 2/4/0, the `k (x) w` outer product at 3/6. **ACT never writes PSUM**, so the PE outer-product
gate can be relaxed to its true producer without a WAR risk against the f30 that reads `o`.

### The s4 / s6 ring buffer that f30 shares with the delta ops

`ACT f30` (`accumulator_cmd=3`, square+accumulate) writes its 128 squared values into
**25182936** — the same 512-byte slot the DVE delta ops write. That is the "partition-96
scratch ring" iter-2/iter-3 could not place; it is now identified. It is why moving `f30`
around is dangerous, and it is why edit L moves only `f29` (which lives entirely in the small
16392/17112/... scalar ring: `RA` writes the sum there, `f29` overwrites it with rstd, `N`
reads rstd as its scale pointer *and overwrites the slot* with its 128-element result, `M`
reads it). Four different owners of the same 512 bytes in sequence.

---

## Edit J1: PE outer-product gate -> the f128 (prerequisite, no gain alone)

Generic rule: for every `PE MATMUL` whose `src_mem_pattern_start_addr` equals the `dst` of an
ACT `ACTIVATE f128`, lower the governing `LDWEIGHTS` wait from `s4 >= <whatever>` to
`s4 >= rank(that f128)`. Hits 16 of 24 (the 16 iter-2 fused to write bf16 into 0x26000+256n;
the other 8 go through an intermediate ACT `CAST`, so their matmul src is the CAST's dst —
extending the rule to CASTs is untried).
Alone it is noise (the DVE order, not PE, was the limiter) but it is what makes K and L pay.

## Edit K: the DVE output op's s4 fence (-0.40 us)

Each `M` waits `s4 >= (its N's s4 wait) + 1 or + 2`. The extra 1-2 ACT slots are an
`EVENT_SEMAPHORE` and/or the next `f128`; `M` reads none of them (it reads `N`'s dst, the silu
buffer and writes output_0). `N -> M` is separately enforced by the DVE `EVENT_SEMAPHORE`
between them **and by DVE program order anyway**. Relaxing `M` to `N`'s own value unblocks
`U` (and so ACT's `CAST`) ~0.5 us earlier per cycle. 16 of 24 relaxed; `applyK.py` skips one
(DVE@157, whose matching `N` is 90 slots back — buffer reuse, the naive dst->N map is wrong
there) and leaves the 7 whose `M` already waits on s6.

## Edit L: bubble `f29` left past `f128` (-0.77 us)

ACT order per cycle was `... f30 RA f29 | f30 RA | f128 f29 f128 | ES ES | CAST CAST`. The
second head's `f29` sat *behind* a 490 ns `f128`, so its `N -> M -> U` chain started 0.5 us
late and `U` (which gates the `CAST`) with it. Swapping gives `... f30 RA | f29 | f128 f128 |`.
7 of the 24 `f29` match the pattern. `f128`'s own consumer is a PE matmul and PE is 27 % busy,
so delaying it costs nothing.
Only two s4 ranks swap per site; retarget with the **direct producer** rule (`ACT@124` fence
-> the f128's new rank, `DVE N` -> the f29's new rank). The *conservative* "max over the old
prefix" rule is wrong here: it keeps `N` waiting for the `f128` and the whole gain vanishes,
and for the ACT self-fences it would **deadlock**.

## Edit N: the 6th table load into an ACT idle (-0.61 us)

6 `ACTIVATION_TABLE_LOAD` x 1.283 us = 7.7 us, 14 % of all ACT busy. The count is a hard floor
(see below) but the *placement* is not: an `ACTIVATION_TABLE_LOAD` has **no wait and no
update**, so it can be moved anywhere in the ACT stream with zero semaphore work — only the
table-residency constraint applies. The body load was at ACT@24, immediately in front of the
first body `f29`; ACT had a 0.62 us hole at 27.6-28.2 just before it (`ES@20` waiting on
`s6>=18`). Moving the load to ACT@20 absorbs the hole. `movetbl.py <from> <to>` does this and
asserts the instruction carries no semaphore.

## Edit P: permute the ACT head (-0.82 us)

Table-set membership (from the 5 `*_and_others.json`, field `profile_meta_data[].func_id`):

| table | func ids |
|---|---|
| silu_and_others | 1 2 3 4 6 11 12 13 19 30 31 33 35 **36** 128 130 |
| softplus_and_others | 1 2 3 4 **9** 11 12 13 30 31 33 35 128 130 |
| reciprocal_sqrt_and_small | 1 2 3 4 11 12 13 **29** 30 31 33 35 128 130 |
| natural_log_exp_and_others | 1 2 3 4 **7** 10 11 12 13 30 31 33 35 128 130 |
| sigmoid_and_others | 1 2 3 4 **5** 6 11 12 13 21 28 30 31 33 35 128 130 |

The kernel uses f36, f9, f29, f7, f5 — **each in exactly one table** — plus f30/f128 which are
in all five. The program order f36 -> f9 -> f29 -> f7 -> f5 -> (body f29) therefore needs
exactly **6** loads and cannot be reduced: f29@7 feeds the `b` broadcast shuffles which gate
f5, so f29 cannot move after f5. **This closes "cut a table load" for good.**

What *is* free is where the loads sit. The old head was
`load(silu) f36 load(softplus) f30 f9 load(recip) f29 load(natlog) f7 f7 ...` and
* `f36` waits for the `z` input DMA (lands 12.6 us) — ACT idles 1.19 us before it;
* `f30@4` waits for `s24` (the 24x128 transposing DGE input, lands 17.4 us);
* `f29@7` only needs `PE@3`, ready 17.9, but was stuck behind `load(recip)` until 19.1;
* `f7@9` only needs the `dt_bias/A_log` DMA (11.4 us) but was stuck behind `load(natlog)`.

So 2.5 us of table load sat in front of already-available operands while 1.2 us of ACT idle
sat in front of the one op that was waiting for a DMA. New prefix
`0 SET | 3 load(softplus) | 5 f9 | 1 load(silu) | 2 f36 | 6 load(recip) | 4 f30 | 7 f29 |
8 load(natlog) | 9 f7 | 10 f7`:
`f9` only needs `s6>=4` (ready 12.2) so it goes first; `load(recip)` finishes at 17.83, just as
`s24` arrives for `f30`; `f29` then runs at 18.33 instead of 19.09 and `f7` at 19.83 instead of
20.58. Legality: every ACTIVATE is still inside the load that owns its function (f30 is in all
five). Only one s4 wait changes (`PE@2`: 2 -> 3).
Body now starts at 28.24 instead of 28.99.

## Why hoisting the DVE `S+=` ops (edit J2) loses — three measurements

This looks like the obvious fix for the 0.70 us hole and it is wrong in all three variants:

* **all 24** (73.47, +0.87): the `N` ops move later, and two ACT `EVENT_SEMAPHORE` fences
  (`Activation@87` waits `s6>=60`, `Activation@112` waits `s6>=72`) resolve to `N` ops. Their
  waits must *increase*, and each becomes a ~0.95 us ACT stall. Those two fences cancel the
  whole body gain even though the 2-head cycle genuinely fell 3.94 -> 3.57 us.
* **slots <= 141 only** (72.60, wash): same two fences, same cancellation.
* **slots >= 142 only, the drain** (72.13, +0.7): no ACT fence involved, but in the drain the
  `N -> M` chain *is* the critical path to the end — `M` writes output_0 and the last `M`
  gates `SP@6`, the output DMA trigger. Delaying `N/M` delays the kernel end 1:1.

Generalisation: **in the body the `U` chain is critical, in the drain the `N/M` chain is.**
Any DVE reordering has to respect that switch-over.

**Re-tested after T22 and still a regression (70.20 vs 69.03).** T22 halves the output work
sitting in front of the next cycle's `U` (the DVE cycle is now `U,U,D,D,N,N`, 678 ns of `N`
instead of ~1.3 us of `N,ES,M,N,ES,M`), which was supposed to be the thing that made J2 lose.
It was not enough. **This lever is closed; do not spend another iteration on it.**

## Edit Q (DEADLOCK — be careful)

Bubbling the 24 DVE `D` (delta) ops left past the `U` ops is address-safe (both only read
PSUM, they write disjoint SBUF) but creates the cycle
`Activation@160 -> PE@155 -> PE@156 -> DVE@165 -> DVE@166 -> Activation@160`
after s6 renumbering. The device hung (`NRT close timeout`) and `verify.py` failed; it
recovered on its own, no reboot needed.
**`cycle.py` printed `CYCLE: ...` but still exits 0** — in a `cmd && cmd` chain it will not
stop you. Grep its output for `acyclic` instead.

## Edit T22: fuse the two DVE output ops, heads 0..21 (-0.88 us, KEPT)

The biggest remaining lever, and it landed for 22 of the 24 heads. DVE busy falls
**40.7 -> 38.2 us** and the wall 69.91 -> 69.03. `output_0` is bit-identical head by head, so
the reassociation it performs is numerically free in bf16 (that was the main unknown — it is
now answered: **this reassociation costs nothing**).
The two heads that are still unfused are worth ~0.07 us; see "the head-23 bug" below for the
cause, which is a 4-line fix away for whoever wants it.

What the two ops are, exactly (all 24 are uniform — verified field by field):
```
N : SCALAR_TENSOR_TENSOR_ARITH  dst = (src0 op0 imm0) op1 src1   op0=op1=6(MULT), rev=0
    src0 = PSUM o (bank 0/1/2/3/4/5/7)   imm0_src=1 -> imm0_imm_ptr = rstd scalar
    (rstd ring: 16384 / 16392 / 24024 / 210944)   src1 = norm_weight @216064 (128 fp32)
    dst = a 2-deep ring of 512-byte scratch slots {16392,17112,17624,17880,18136,18392,
          18648,24032,25304,30168}   out_dtype = 10 (fp32)   nch=1   num_elem=[128,1]
M : TENSOR_TENSOR_ARITH_OP  dst = src0 * src1   op=6
    src0 = that N's dst    src1 = silu(z)[h] = 131072 + 512*h
    dst  = output_0 @ 25197024 + 256*h    out_dtype = 6 (bf16)   nch=1
```
`silu(z)` is one contiguous 3072-fp32 block [131072,143360) written by the single merged
`ACT ACTIVATE f36`, and **read by nothing but those 24 M ops** (checked exhaustively) — so it
can be scaled in place.

The transform `applyT.py` applies:
1. one wide DVE `TENSOR_TENSOR` inserted at **DVE@5** doing `silu *= norm_weight` in place:
   `src0 = 216064 step [1,0,1] num [128,24,1]` (zero stride on the head axis),
   `src1 = dst = 131072 step [1,128,1] num [128,24,1]`, `out_dtype=10`, `nch=1`,
   wait `s4 >= rank(f36)` = 2, no update. **This slot is free**: DVE is idle from 12.20 us
   (DVE@4 ends) to 19.86 us (the `a` shuffles, woken by `f7@9`), and the silu is ready at
   16.53 — a 7.6 us hole nobody has used.
2. per pair: `N.src1 := M.src1`, `N.dst := M.dst`, `N.out_dtype := M.out_dtype`.
3. `M` -> `EVENT_SEMAPHORE` (clone one from the DVE stream), keeping M's s6 update where it had
   one and waiting `s6 >= rank(N)` in that case, so s6 ranks are untouched.

**Address-field gotcha that cost me time, now solved:** `*_start_addr_addr_immediate`,
`_reg_regnum`, `_reg_reserved0[]` and `_reg_marker` are *overlapping views of the same 4 bytes*
(`immediate = regnum | reserved0[0]<<8 | reserved0[1]<<16 | marker<<24`; `marker` is just the
top byte, 0x00 = low SBUF, 0x01 = SBUF above 16 MB, 0x02 = PSUM at 0x2000000 — it carries no
extra semantics). So `ins.set('..._immediate', full_address)` sets everything; do **not** also
write regnum/reserved0/marker or you clobber it.

`applyT.py <sgdir> <insert-pos> [skiplast] [only-N]` — `applyT.py work/payload/sg00 5 no 22`
is the kept form. It asserts the pairing and the silu stride, so it fails loudly if the
schedule changes under it.

### The head-23 bug (why only 22 of 24 are fused)

Bisected with `verify.py --keep` + a per-head bf16 compare of `.verify/run0/output_0.bin`
against a saved reference (this is the diagnostic that cracked it — **always do this instead
of reading the aggregate `max_abs_err`**):

| fused heads | result |
|---|---|
| all 24 | FAIL, **only head 23 wrong** (127 of its 128 bf16 words differ), 23 heads bit-identical |
| 0..22 (head 23's pair left alone, its silu left unscaled) | FAIL, **still only head 23 wrong, identical numbers** |
| 0..0 | PASS, bit-identical |
| 0..21 | PASS, bit-identical |

So fusing **head 22** is what corrupts **head 23**, and it is not head 23's own pair.
Head 22's `M@169` is one of the four M ops that carry an `s6` update (its rank is 100), and
`ACT@166` (the last `CAST nch=128`) waits `s6 >= 100`. The original `M@169` waited on an **s4**
fence (`s4 >= 117`) and took 284 ns; my replacement `EVENT_SEMAPHORE` waits `s6 >= 99` (its N)
and takes 24 ns, so the `s6 = 100` update now fires several hundred ns earlier *and* without
the ACT-progress fence. Whatever `ACT@166` really needed from that fence is no longer there.
The error is not a missing gamma factor — the per-element ratio ref/new for head 23 is not a
consistent vector (it changes sign), so head 23's `o`/`rstd` input itself is wrong, consistent
with a PSUM or scratch WAR let loose by the early `s6 = 100`.
**The 4-line fix to try first:** for every M that carries an s6 update, give the replacement ES
**M's original wait** instead of `s6 >= rank(N)` (the N->M visibility no longer matters, since
the ES reads no memory — the only thing the update has to mean is what the original meant).
There are only 4 such sites (`M@89, M@153, M@169, M@171`); `M@65`'s update already works with
the `s6` form, so test both forms.

---

## Where the time is now (69.03 us; trace span 10.57 -> ~81 + ~1.4 us tail)

Engines: ACT **56.0 (81.2 %)**, DVE 38.2 (55.4), DMA 19.8, PE 19.5 (28.3), Pool 15.6, SP 3.4.
ACT is now over 80 % busy, so from here either ACT work has to be deleted or its last ~7 us of
idle has to be filled. The gap numbers below were taken on the 69.91 us NEFF (before T22); T22
shortens the body cycle slightly but does not change which gaps exist.

* **0 -> 10.57 us: fixed startup.** `Q14-E65..E68`, 4 x 33792 = 135 KB arriving 0.04-9.80 us,
  then ACT@0 at 10.57. That is almost certainly the 5 activation tables' `*_bkt.bin` +
  `*_ctrl.bin` being loaded to iram/SBUF. All five tables are used, so it is not removable by
  deleting one; shrinking it would mean editing the DMA descriptors (untried, risky).
* **Head 10.57 -> 28.24 us.** ACT busy 14.0, idle 2.2. Remaining idle: 0.32 (f9 waiting
  `s6>=4`), 0.27 (f29 waiting PE@3), **1.14 (f7@10 waiting for the `a` shuffles — structural,
  nothing table-compatible is ready)**, 0.52 (f5 waiting for the second `b` shuffle group).
  After `f7@10` the window 21.4-28.2 is **both** engines near-dense: DVE has 4 exp_g chunks
  (3.76) + shuffles (1.4) = 5.16 us, ACT has load(sigmoid) 1.28 + 4 wide casts 3.33 +
  f5 + f30 + load(recip) 1.28 = 6.32 us in a 6.80 us window. Moving the shuffles before
  exp_g chunk 3 to kill the 0.52 us f5 gap just pushes chunk 3 (and so cast 3) back by the
  same amount — computed, not measured, but DVE has no slack there.
* **Body 28.24 -> 78.70 us, ACT idle 7.2 us**, now unevenly distributed:
  * the 8 steady-state cycles have 0.22-0.36 us holes each (~2.4 us) — close to the ~0.14 us
    floor set by `f128(last) -> ES -> PE -> DVE U -> CAST`;
  * the fill cycles (ACT@29/30/46/52/64/76) have 0.41-0.67 us holes (~2.9 us);
  * the drain (ACT@136/150/151/160/161/166/167) has 0.32-0.81 us holes (~3.8 us), where
    the pipeline is down to one head in flight and the full per-head latency is exposed.
* **Tail ~1.4 us**: last DVE op -> `SP@6` trigger (0.11) -> output_0 DMA. Breakdown from
  `DmaPacket.parquet`: 0.58 us trigger-to-first-packet, 0.17 us walking 15 engines that each
  move 4 dummy bytes, 0.25 us for the real 6148-byte packet on `qSyncIO0-Q1-E65`, 0.35 us to
  the semaphore. Splitting the transfer cannot help: the last chunk still pays the ~1.1 us
  fixed part. Closed.

## Ranked candidates for iter-5

0. **Finish edit T for heads 22-23** (~0.07 us, i.e. below the noise floor on its own, but it
   removes an asymmetry that will get in the way of anything else touching the drain, and the
   fix is 4 lines — see the edit-T section).
1. **Deepen the pipeline to 3 heads in the fill/drain regions (~3-4 us).** The biggest block
   left *inside* the kernel proper (candidate 4 is bigger but is startup, outside it): after
   T22 the 8 steady-state cycles idle ACT only 0.20-0.28 us each (~2 us total, near the
   structural floor), while the 4 fill cycles still idle 2.5 us and the 6 drain cycles 3.4 us.
   In the drain the pipeline is down to one head and **DVE is idle too** (gaps of 0.7-1.4 us at
   69-79 us), so it is pure cross-engine latency, not throughput: ~1.95 us/head with ACT doing
   only 1.3 us of it. The
   blocker there is that the delta matmul for the last heads is gated on a `CAST nch=128`
   (e.g. `PE@157` waits `s4>=113`), i.e. a PSUM-bank WAR, so their `D` cannot be computed
   early. Check whether a *different free PSUM bank* exists for the last 2-3 heads (banks
   0,2,4 are used by `S'^T k`, 3,6 by the outer product, 1,5,7 by `o` — the map is in
   "PSUM banks" above). If one is free in the drain window, re-point the matmul dst and drop
   the WAR fence.
2. **Extend edit J1 to the 8 f128s that go through a `CAST`** (their matmul src is the CAST's
   dst, not the f128's). Same rule, 8 more sites; worth ~0.1-0.3 us but it is 5 minutes.
   Note edit R tried the *blanket* generalisation and it was worthless (70.14) and unexplained,
   so restrict it to the CAST-mediated `w` buffers specifically.
3. **Extend edit L to the `f29` that are blocked by something other than `f128`.** Bubbling
   past `CAST` as well measured +0.16 (edit M, dropped), so the remaining ones need a
   case-by-case look, not a blanket rule.
4. **Startup (10.57 us) and the 135 KB table load.** Nobody has looked at the DMA descriptors
   for this. It is 15 % of the runtime and it is the single largest block left. Unknown
   whether the 4-engine, 13.9 GB/s aggregate can be widened by editing the queue/descriptor
   layout in `def.json` / the DGE tables.

## Closed / re-confirmed

* **The 6 `ACTIVATION_TABLE_LOAD`s are a hard floor** — proven from the table membership
  tables above, not just from scheduling. Do not revisit. Their *placement* is free (no
  semaphores) and was the source of 1.4 us this iteration.
* **Hoisting the DVE `S+=` ops** — three variants, all regressions, see above.
* **Bubbling DVE delta ops past `S+=`** — deadlocks.
* **Bubbling ACT `f29` past `CAST`** — +0.16 us.
* **The blanket "relax every PE gate to its operand's writer" rule (edit R)** — 70.14, no gain,
  and the 4 fences it drops (on the `S'_bf^T k` matmuls, jumps of 25-78 s4 slots) are
  unexplained. I could find no ACT-side reader of those PSUM banks before the fence, so it may
  be safe, but it buys nothing. Do not ship it.
* **Moving `ACT@19 f30` into the `f7@10` hole (edit S)** — +3.35 us, which tells you `s26`
  (the third Pool DGE input) only lands at **~24.5 us**. Useful number: nothing gated on s26
  can be hoisted into the head.
* **Splitting the output_0 DMA / shortening the tail** — the 1.4 us is ~1.1 us fixed
  (trigger-to-packet + 16-engine walk + semaphore), measured from `DmaPacket.parquet`.
* Everything iter-1/2/3 closed stays closed. iter-3's candidate 2 ("share one `READ_ACC`
  between two heads with `nch=2`") is **newly refuted on structure**: every head's `o`, its
  `silu(z)`, its rstd scalar and its output row all live on **partition 0** (output_0 is a
  single contiguous 6148-byte run in one partition — see the single DMA packet), and a matmul
  cannot put two different heads' `o` on two partitions because each needs a different moving
  operand. There is no 2-partition form of that op.
* iter-3's "relocating a chain link between ACT and DVE conserves on-path work" still holds,
  and the J2 re-test above is a second independent confirmation.

## The lesson of iter-4

Two kinds of edit paid, and neither is "move work to a less busy engine":

1. **Delete the artificial fence in front of work that is already ready.** Three of the six
   kept edits (J1, K, L) changed nothing but *which instruction a wait resolves to* — not one
   instruction moved, not one byte of data moved. Together they are -1.17 us. The pattern to
   look for: an instruction whose wait resolves to an `EVENT_SEMAPHORE` or to an op it reads
   nothing from, while its real producer finished much earlier. `semmap.py` prints exactly
   that (`wait s<N> >= v <- (eng, slot) end <t>` next to the waiter's own start time); any row
   where the producer's end is far before the waiter's start is a satisfied wait and fine —
   the ones to attack are where the resolved producer is *not* the data producer.
2. **Put a fixed-cost op into an engine's idle window, and prefer idle windows that are gated
   by a DMA.** Edits N and P (-1.43 us) only moved `ACTIVATION_TABLE_LOAD`s and ACTIVATEs
   inside the ACT head; T22 (-0.88 us) hoisted a 2.4 us precompute into a 7.6 us DVE hole that
   nobody had noticed. **Look for idle windows first, then ask what can legally go in them** —
   the inverse search (find expensive work, then a home for it) produced edits S and M, both
   regressions.

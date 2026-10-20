# iter-2 notes — gdn_tkg bh24 k128 v128, LNC=1

## Result

**79.594 us -> 74.47 us, -6.4%.** `verify.py --runs 3` PASS with `max_abs_err` **exactly** the
baseline digits (2.322e-02 / 1.633e-03) and bit-identical across runs; `launch_rules.py` PASS;
`semaphores.py` balanced; happens-before graph acyclic.

| step | edit | total us (5 reps) | ACT busy | verdict |
|---|---|---|---|---|
| in | iter-1 result | 79.594 | 63.73 | |
| A | 24 pre-outer `S_bf[h]=bf16(S_all[h])` CASTs -> 4 merged casts of 6 heads, each at the position of its group's FIRST head | 79.761 | 60.16 | **no change** (ACT -3.6 us, wall +0.2) — kept only because B/D build on it |
| B | merged `k_f *= k_scale` DVE op split 8 ways and 7 of the 8 chunks moved out of the head into DVE's body idle gaps | 77.46–77.92 | 60.2 | **-1.9 us**, real |
| D | 16 of the 24 `ACTIVATE f128` (scale-multiply) fused with the `CAST fp32->bf16` that reads them: ACT writes bf16 straight out, CAST deleted | **74.50** | **55.16** | **-3.0 us**, the big one |
| E | ACT head reordered to `t1 f9 t2 f30 f29 t0 f7 f7exp t4 silu t3` so two table loads fall in the 11.9-17.5 us `s24` wait and [t4, silu] falls in the ACT idle window after `f7(exp_g)` | 74.22 / 74.45 / 74.63 | 55.23 | **wash, NOT kept** (see below) |

Spread of `profile.py --repeats 5` on the final NEFF is 0.6–1.4 % (0.5–1.0 us), noticeably worse
than iter-1's 0.15–0.5 %. **Do not believe anything under ~0.7 us from a 3-repeat run** — two of
my placement sweeps reordered between a 3-repeat and a 5-repeat measurement.

Apply scripts are in `iter-2-scripts/` next to this file; run them against `work/payload/sg00`
after `neff.py unpack`, in the order

```
applyA.py 6
applyB2.py "3@keep,3@36,3@47,3@49,3@52,3@56,3@61,3@65"
applyD.py 0x26000 "0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16"
```

Final NEFF md5 `f7dadc8f6bf7df9a1f4f80be37fcb262`, 133140-byte payload, ACT 170 slots (was 191),
DVE 166, PE 161.

`applyE.py` is the head reorder that verified but measured as a wash (do not re-apply blindly).
`cpath.py` is the tool that made this iteration work (see below). `applyC.py` is the *broken*
first version of the fusion — kept as the record of what fails.

---

## The one tool that mattered: `cpath.py` (measured critical path)

iter-1's closing advice was "build a weighted critical-path trace instead of guessing". Done, and
it is cheap: `tools/profile.py --repeats 1 --parquet prof` gives `Instruction.parquet` where
`compiler_pc` **is the slot index** in `<Engine>0.bin`. Join it with the happens-before graph from
`iter-1-scripts/cycle.py`, start at the instruction with the largest `end_ts`, and walk backwards
always choosing the predecessor (program-order previous slot on the same engine, or the semaphore
producer it waited on) with the **largest `end_ts`**. That is the real critical path, measured.

Then classify each edge: if the program-order predecessor is the one that bound, the node is
**engine-occupancy bound** (removing work from that engine shortens the path); otherwise it is
**data bound** on another engine.

On the iter-1 NEFF this gave: path span 80.1 us = ACT-occupancy 50.0 + DVE-occupancy 12.8 +
PE 2.6 + 13.4 data-bound + 11.4 us of gaps. On the final iter-2 NEFF the gaps are down to
3.5 us, i.e. the schedule is now tight and further wins have to come from **removing or
cheapening work that is on the chain**.

**The classification is necessary but not sufficient** — see edit A.

---

## Edit A: group-merging the pre-outer S casts — ACT work removed, wall unchanged (AGAIN)

iter-1 edit 12 merged all 24 into ONE and got ACT -3.5 us / wall 0. I blamed the serialisation
(one wide cast right after the `S*=exp_g` barrier delays head 0 by ~2.9 us) and redid it as 4
groups of 6, each placed at the **original slot of its group's first head**, which delays nothing
(all 24 of these casts have every data dependency satisfied at the `s6>=10` barrier; they are
pure filler in ACT program order). ACT went 63.73 -> 60.16 (exactly the predicted -3.6) and the
wall went 79.594 -> 79.761.

So the explanation was wrong, twice over. **Removing ACT work that is not itself a link in the
per-head data chain does not shorten this kernel**, even though `cpath.py` calls those slots
"ACT-occupancy bound". What happens instead is that the ACT idle inside the body grows (3.5 us ->
7.2 us) and the `('DVE','Activation')` data-bound share of the path grows from 4.0 to 6.8 us: the
body is a tightly coupled ACT<->PE<->DVE chain of ~2.4 us per head and ACT's spare slots are
absorbed by waiting for the next link. Contrast edit D, which removed the *same amount* of ACT
time but from instructions that **are** links in the chain, and converted it ~1:1 into wall time.

**Rule learned: before removing an ACT op, check whether the op's output is read by the next link
of the chain. If it is not (it is filler), the removal is free but worthless.**

(Edit A is kept in the stack only because B and D were derived and measured on top of it; on its
own it is a wash. Someone could try dropping it to buy back SBUF and 4 slots.)

## Edit B: the head is DVE-serial, so get `k_f *= k_scale` out of it  (-1.9 us)

Measured head structure (all times from the iter-1 NEFF):

```
ACT   10.5 t4 | 13.3-16.1 merged silu | t1 | f30(s24) f9 | t2 | f29 -> k_scale | t0 | f7(A_log) 20.73
DVE                                                                      20.76 @5 + 6 shuffles 21.83
DVE                                                   @18 k_f*=k_scale  21.86-25.21   (3.35 us)
DVE                                                   @20 S*=exp_g      25.14-28.48   (3.34 us)
DVE                                                   @21-@33 trailing shuffles  -> 28.8
ACT                                                   body starts 29.86 (f5 waits s6>=14 = DVE@33)
```

The head is **9.1 us of serial DVE work** that cannot start before ACT reaches `f7(A_log)` at
20.73. `k_f *= k_scale` is 3072 bf16 on ONE partition (`nch=1`), so it is 3.35 us of pure
element-rate and it sits *in front of* `S*=exp_g`, which is what the body waits for.

Only the first few heads of `k_f` are needed early: the first `LDWEIGHTS` that reads `k_f` is
`PE@13` (head 0) at ~30 us, head 6's is ~50 us, head 12's ~76 us. So split the op and move all
but the leading chunk into DVE's body idle gaps. Useful facts:

* `k_f` is `209920 + 256h` (bf16, 24 x 128 on partition 0); `k_scale` is `16968 + 4h` (fp32);
  the op is `src0=dst=209920, src1=16968 step [0,1,1], num_elem=[128,24,1], nch=1`. Chunking is
  just `num_elem[1] = n` plus offsets `209920+256*h0` / `16968+4*h0`.
* `LDWEIGHTS` reads the k_f tile **reversed**: `src = 209920 + 256*h + 254`. Get this wrong and
  the head mapping is off by one.
* **The only waiter on the merged op is `PE@12 EVENT_SEMAPHORE W[s6>=9]`** — the compiler already
  put one PE barrier in front of all 24 `LDWEIGHTS`. Shrinking the op in place keeps that barrier
  valid by identity; each moved chunk needs one new `EVENT_SEMAPHORE W[s6>=chunk]` inserted
  before the first `LDWEIGHTS` of its head range.
* DVE body idle gaps (slot numbers in the post-edit-A DVE stream): after @36 (2.24 us), @47
  (0.67), @49 (0.90), @52 (0.64), @56 (1.17), @61 (1.05), @64 (1.24), @65 (0.83).

Placement sweep (3 repeats each, so ±0.5 us):

| spec | us |
|---|---|
| 2@keep, 22@36 | 78.21 |
| 3@keep, 21@36 | 78.16 |
| 12@keep, 12@36 | 78.11 |
| 6@keep, 18@36 | 77.92 |
| 6@keep, 18@49 | 78.47 |
| 18@keep, 6@36 | 79.13 |
| 6@keep, 9@36, 9@56 | 77.69 |
| 6@keep, 6@36, 6@49, 6@61 | 77.62 |
| **3@keep x8 across the gaps** | **77.46** |
| 12 chunks of 2 | 77.91 |

So: the leading chunk wants to be small *but not minimal*, and the tail wants to be spread over
several gaps rather than dumped in one. The 8x3 and 4x6 variants are within noise of each other.
Re-measured on top of edit D, the 8x3 spec ending at `@65` and the one ending at `@64` were
74.468 vs 74.575 — indistinguishable.

## Edit D: fuse `ACTIVATE f128` with the `CAST fp32->bf16` that reads it  (-3.0 us)

The per-head chain contains `ACTIVATE f128` (`out = scale_ptr[h] * src`, `nch=1`, 128 elements,
**490 ns**) immediately followed — within a couple of slots — by `CAST fp32->bf16` (**295 ns**)
whose source is exactly the ACTIVATE's destination and whose destination is read by one
`MATMUL`. Both are links in the chain. Make the ACTIVATE write BFLOAT16 straight out and delete
the CAST: 785 ns -> 490 ns + a 40 ns barrier, per head.

**The arithmetic is bit-identical.** ACT computes `scale*src` in fp32 and rounds once on output,
exactly as the separate CAST did. 24 f128 exist; 3 are excluded because another engine touches
their intermediate buffer; of the remaining 21, **16 verify bit-identically and 5 do not**.

Construction that works (`applyD.py`):
* the fused ACTIVATE stays **at the ACTIVATE's own slot**. Moving it to the CAST's slot is wrong:
  its `src` buffers (e.g. 25183448) are rotated by DVE between the two slots, so it would read a
  newer value — that version failed with `max_abs_err` 6.46 on output_0 *and* 0.73 on output_1.
* it writes a **private 256-byte buffer** (`0x26000 + 256n`, 16 of them) and the single
  `MATMUL` that consumed the CAST's destination is re-pointed. (This turned out not to be
  necessary — the shared destination gives the same errors — but it is strictly safer and costs
  4 KB of the 82 KB free SBUF.)
* an `EVENT_SEMAPHORE` carrying the CAST's **wait and update** is left at the CAST's slot, so the
  s4 increment count is unchanged and every downstream `s4>=k` keeps exactly its old meaning. No
  wait value in the whole NEFF had to change (`wait values rewritten: 0`).
* pairing: for each f128 writing `A`, require that `A` is touched by no other engine, then take
  `nw` = next ACT writer of `A` and demand **exactly one** ACT reader of `A` in `(i, nw)`, and
  that it is an `nch`-matching 128-element `CAST FP32->BFLOAT16` starting at `A`. The consumer of
  the CAST's destination is identified the way `apply12.py` does it (unique reader reachable from
  the CAST such that every later writer of that address is reachable from the reader).

### The 5 pairs that are WRONG and were excluded — unexplained

By ACT slot in the post-A/B stream: `(f128@96, CAST@98)`, `(134,136)`, `(137,139)`, `(144,149)`,
`(145,150)`. Each **fails on its own** (e.g. pair `(96,98)` alone: output_0 `max_abs_err` 2.327,
cosine 0.982 — about one head's worth of values), and each fails identically with the shared
destination, with a private 256-byte-strided destination and with a private 512-byte-strided one.
`verify.py --runs 2` reports the wrong outputs are **bit-identical across runs**, so these are
deterministic data-flow errors, not timing races — which is also the reason I am confident the 16
accepted pairs are genuinely correct and not lucky: they reproduce the golden to the baseline's
exact last digit, deterministically, over 6 device runs.

What I checked and ruled out for the failing pairs: no other engine touches the intermediate; the
intermediate has exactly one reader in the interval; the fused destination has no reader between
the two slots and all other readers are ordered either before the ACTIVATE or after the CAST;
the consumer is reachable from the fused op in the happens-before graph; private destinations with
both 256- and 512-byte stride. **The mechanism is still unknown.** Four of the five are the last
f128s in the stream (slots 134-150, the last heads), which is suggestive — maybe the output/RMS
tail re-uses something — and pair `(96,98)` is the only failing pair with an `ACTIVATE f29` in
between. Worth ~0.25 us each if someone cracks it, plus the 3 excluded ones.

**Do not widen the accepted pair set without re-running `verify.py`.** The selector is a hardcoded
index list for a reason.

## Edit E: the ACT head reorder that finally *works* and still buys nothing (0.0 us, not kept)

`applyE.py` permutes ACT slots 1-11 from `t4 silu t1 f30 f9 t2 f29 t0 f7 f7exp t3` to
`t1 f9 t2 f30 f29 t0 f7 f7exp t4 silu t3`. It verifies **bit-identically** (so unlike iter-1's
edit 8, the permutation is legal — the trick is that `f7(A_log)` must stay after `f9`, because
`DVE@5`'s single `s4` wait is a combined guard for both, and the silu must stay before `t3`,
because `f5` needs t3 live). Measured interleaved against the kept NEFF, 5 repeats each:

```
kept   74.366   74.719
edit E 74.447   74.626
```

Indistinguishable. **Not kept** — same NEFF, less risk.

Why it does nothing, from the profile of the edit-E NEFF (this is the useful part):

```
ACT  10.65 t1 | 12.61 f9 | 12.85 t2 | 17.50 f30 (s24!) | 17.99 f29 | 18.19 t0 | 19.49 f7(A_log)
ACT  20.85 f7(exp_g, waits the shuffles) | 21.09 t4 | 22.38-25.15 silu | 25.16 t3 | 26.46 cast | 27.47 f5
DVE  19.74 @5 | shuffles -> 20.82 | 20.85 k_f chunk0 | 22.64 ES(s25!) | 22.73-26.09 S*=exp_g
```

`f7(A_log)` did move from 20.73 to 19.49 and the body gate moved 29.9 -> 27.5, i.e. the head
really did get 2.4 us shorter — but the wall did not, because the head is now bound on **both**
sides at once:
* **ACT** has 9.17 us of irreducible head work (4 table loads + the 2.76 us merged silu) and
  cannot start before 10.6, so it is busy continuously from 21.1 to 27.3.
* **DVE** `S*=exp_g` is now gated by `s25 >= 16` (the fp32 DGE input = `state_in`) at ~22.6, and
  takes 3.36 us, finishing 26.1.

So the head floor is ~max(ACT 27.3, DVE 26.1). To go below it you must remove ACT head *work*
(the only candidate is one of the 6 table loads, and the mix needs all 6 — see below) or make
`state_in` land earlier. **This also explains iter-1's edits 6/8/10 once and for all:** `ACT@4
f30` waits on `s24`, which is satisfied at 16.3-17.5 us, so anything in front of it is free.

`state_in` arriving earlier is now a live lever (it was not before edit E): the three DGE input
transfers are serial on the single `qPoolDynamic` queue and complete at 17.5 / 22.6 / 24.0, with
the fp32 `state_in` second. Reordering the three `Pool PSEUDO_DMA_DIRECT2D` triggers would move
`state_in` earlier but `s24` later, and `s24` gates `ACT@4 f30` which starts the k_scale chain, so
it is a genuine trade, not a free win — I ran out of time to measure it. Note the semaphore index
(24/25/26) is assigned by **trigger order**, so swapping triggers means swapping every waiter's
`events_wait_idx`, exactly like iter-1 edit 11 did for `qSPIO0`.

---

## Facts about this NEFF that are now measured (use these, don't re-derive)

* **Input DMA completion times** (`SemaphoreUpdate.parquet`, first timestamp where a queue
  semaphore reaches 16): `s8` dt_bias+A_log 11.05, `s9` a 11.59, `s10` b 11.78, `s11`
  norm_weight+z 12.21, `s12` v+k 12.94; Pool DGE inputs `s24` 16.32, `s25` 21.80, `s26` 24.00;
  state-out `s27` first chunk 38.9. Output_0's DMA runs 89.8-90.2 in the iter-1 NEFF.
  `Pool0.bin` slots 1/2/3 are the three DGE input transfers (bf16, fp32, bf16) and 4..11 are the
  8 state-out chunks iter-1 created.
* The **three DGE input transfers are serialised on one queue** (`def.json` has only
  `qPoolDynamic` and `qSPIO0`), 1.5 MB total at ~120 GB/s, finishing at 24.0 us. `s25` (the fp32
  one, almost certainly `state_in`) at 21.80 is *not* currently binding — `S*=exp_g` cannot start
  before `exp_g` at ~22.0 anyway. So **input DMA bandwidth is NOT the head bottleneck**; iter-1's
  open question 4 is answered and closed as a non-lever at the current speed. It will become one
  if the head gets ~2 us faster.
* **The ACT head cannot be reordered to start the DVE chain earlier**, and now I know why
  iter-1's edits 6/8/10 regressed: `ACT@4 f30` waits on `s24 >= 16` which is only satisfied at
  **16.32 us**, so the 1.28 us table loads and the 2.76 us merged silu in front of it are
  *absorbing DMA latency*, not adding to the path. Pulling them out just converts busy time into
  idle. Closed.
* **Activation-function -> table membership** (from the five `*_and_others.json` in the payload):
  each of f36 (silu), f9 (softplus), f29 (rsqrt), f7 (natural_log_exp), f5 (sigmoid) lives in
  **exactly one** table, while f30, f128, f1-f4, f11-f13, f31, f33, f35, f130 are in **all five**.
  The head therefore needs 4 distinct loads (t4,t1,t2,t0) and the body needs t3 for the single f5
  plus t2 again afterwards = **6 loads is the floor for this instruction mix**, 5 is not
  reachable because f5 waits on `s6>=14` (~27 us) and so cannot be hoisted before the body's t2.
  Closed unless an f5/f29 use is eliminated.
* Per-op costs on this engine set: ACT `CAST` 295 ns (128 elems, any `nch`), ACT `ACTIVATE` with a
  pointer scale **490 ns**, ACT `ACTIVATE f29/f30` ~200/265 ns, `ACTIVATION_READ_ACCUMULATOR`
  292 ns *for one element*, `ACT_TABLE_LOAD` 1283 ns; DVE `TENSOR_TENSOR` ~285 ns for 128
  elems/partition and ~1.09 ns per extra element on one partition; PE `LDWEIGHTS` ~90 ns + MATMUL
  ~275 ns. ACT fixed overhead is ~180 ns, element rate ~0.9 ns per element per partition.
  Cross-engine hand-off after a semaphore update is **35-50 ns**.
* **SBUF now in use by edits:** `0x20000-0x23000` (iter-1 silu), `0x24000-0x25800` (iter-2 edit A
  group casts), `0x26000-0x27000` (iter-2 edit D private buffers). Free: `0x0efe0-0x1c000` and
  `0x27000-0x33100`.

## Where the time is in the FINAL NEFF (`cpath.py`, 74.5 us)

path span 74.6 us = ACT-occupancy **37.3** + DVE-occupancy **15.1** + PE 4.2 + 14.1 data-bound
+ 4.6 us of gaps. Engines: ACT 55.2 (74 %), DVE 40.5 (54 %), DMA 20.1, PE 18.8, Pool 14.7, SP 3.3.

On-path cost by op (count, total):

| op | n | us |
|---|---|---|
| DVE `TENSOR_TENSOR` | 38 | 14.45 |
| ACT `ACTIVATE f128` (the fused scale+cast) | 19 | 9.32 |
| ACT `CAST` | 24 | 8.67 |
| ACT `ACT_TABLE_LOAD` | 5 | 6.41 |
| ACT `ACTIVATION_READ_ACCUMULATOR` | 21 | 6.14 |
| ACT `ACTIVATE f30` | 22 | 5.81 |
| ACT `ACTIVATE f29` | 22 | 4.36 |
| PE `MATMUL` | 15 | 4.12 |
| DVE `SCALAR_TENSOR_TENSOR` | 10 | 3.22 |
| ACT `ACTIVATE f36` (merged silu) | 1 | 2.76 |
| DVE `STREAM_SHUFFLE` | 16 | 2.76 |

Gaps > 0.1 us: 1.35 (`SP@6` output_0 doorbell -> `Pool@12` sees `s13`, i.e. DMA trigger latency
plus the 404 ns transfer — not editable), 1.26 (ACT waiting for the `z` DMA in front of the
silu), 0.48, 0.46, 0.27, 0.18. **The schedule is tight; 4.6 us of gap is all that is left.**

Note ACT occupancy fell 50.0 -> 37.3 over the iteration while DVE occupancy rose 12.8 -> 15.1:
the two engines are converging, and the 7 `k_f` chunks edit B added are now themselves on the
path (DVE `TENSOR_TENSOR` on-path count 31 -> 38). ACT is no longer the only thing to attack.

## Ranked candidates for iter-3

1. **Crack the 5 failing fusion pairs** (and the 3 excluded ones) — ~2 us, and more importantly
   it would validate or refute the whole fusion model. What I already ruled out by dumping the
   *measured* timeline of both rotating buffers for pair `(96,98)`: the producer/consumer pairing
   is unambiguous — on 17624 the 7 writers and 7 readers strictly alternate in time
   (`DVE@55 34.1 -> DVE@59 35.4 | ACT@72 47.7 -> PE@65 48.0 | ACT@84 52.0 -> PE@78 52.8 |
   ACT@98 56.9 -> PE@92 57.3 | ACT@110 61.2 -> PE@104 62.1 | ACT@124 66.1 -> PE@117 66.7 |
   ACT@149 75.9 -> PE@143 76.3`), and on the intermediate 25184472 likewise
   (`ACT@71 -> ACT@72 | ACT@96 -> ACT@98 | ACT@121 -> ACT@124 | ACT@144 -> ACT@149`, plus one
   dead store at ACT@53). So it is not a mis-pairing. Remaining untested holes, cheapest first:
   (a) accesses that *partially overlap* the 512-byte intermediate — my checks only ever compared
   exact start addresses, and the three intermediates 25183448 / 25183960 / 25184472 are
   **exactly 512 bytes apart**, i.e. a contiguous ring, so a single access with `num_elem > 128`
   would straddle two of them and I would not have seen it; (b) `scale_value_imm_ptr` is read at
   `p*0x40000 + sptr`, so the *partition group* of the operands matters — the originals keep src
   and dst in the same group (`0x1800000` = partitions 96-127 for pair 11) while the fused op
   writes partition 0, which is what the deleted CAST used to do. Pairs 7 and 11 have the *same*
   src and dst addresses and one passes, so group alone is not the discriminator, but a
   group-dependent scale read combined with something else could be.
2. **The same fusion in the other direction: `CAST -> ACTIVATE`.** 34 ACT CASTs remain (12.8 us,
   and 24 of them are the two `S_all[h] -> bf16` families). Look for a CAST whose only consumer is
   an ACT `ACTIVATE`/`f30` that could take the fp32 input directly, or an `ACTIVATE` whose input
   CAST could be dropped by setting the ACTIVATE's input dtype. Same mechanism, same payoff shape.
3. **`ACTIVATION_READ_ACCUMULATOR`: 21 on the path, 292 ns each for a single element = 6.1 us.**
   This is pure fixed cost and it is a chain link (f30 ZERO_ACCUMULATE -> READ_ACC -> f29). iter-1
   closed *merging* them (the accumulator is per channel and all 24 live on partition 0), but
   nobody has checked whether the `f30` could write its reduction straight to SBUF (`dst` is
   already set to a PSUM-ish address) and drop the READ_ACC, or whether two heads' reductions can
   share one READ_ACC by using two channels. Biggest single remaining block of fixed cost.
4. **The 8 remaining `ACTIVATE f128` at 490 ns** (5 failing pairs + 3 excluded): if the fusion
   cannot be made to work for them, move them to DVE as `TENSOR_TENSOR(MULT)` with the iter-1
   broadcast trick (`src1 step_elem [0,...]`): 490 ns on ACT -> ~285 ns on DVE + ~85 ns of
   hand-off, and DVE still has ~34 us idle. ~1 us.
5. **Re-tune edit B now that the body is 5 us shorter** — the DVE gap table above was measured
   before edit D, and DVE `TENSOR_TENSOR` is now the single biggest on-path cost (38 ops,
   14.45 us), 7 of which are the `k_f` chunks edit B added. Re-measured on top of D, `...3@61,3@65`
   and `...3@61,3@64` were 74.47 vs 74.58 (indistinguishable) and `6@keep,6@36,6@49,6@61` was
   75.32 — so 8x3 is still right, but a fresh gap dump might find better slots. Low value.
5b. **Make `state_in` land earlier** — now a real lever (see edit E): after edit E the head's DVE
   side is gated by `s25` at 22.6 us. Even without edit E it is only ~0.8 us of slack away. The
   three DGE input triggers are serial on one queue; swapping `Pool@1` and `Pool@2` moves
   `state_in` first but delays `s24`, which gates `ACT@4 f30` -> `PE@3` -> `f29` -> `k_scale`.
   Worth one measurement. Remember the queue semaphore index follows trigger order, so every
   `events_wait_idx` in {24,25,26} must be permuted with the triggers.
6. **Try dropping edit A** (it is a wash on its own and its 4 wide casts are 2.4 us of ACT that
   could instead be 24 small casts spread through the body). Needs B2/D re-derived on the
   un-merged stream; B2's spec is given in DVE slot numbers, so it would have to be re-read off
   the un-A'd DVE stream.
7. The 1.34 us gap between `SP@6` writing the output_0 doorbell and `Pool@12` seeing `s13>=16` is
   DMA trigger + 404 ns of transfer; the ~3 us of runtime drain after it is not editable. Closed.

## Still closed from iter-1 (re-confirmed or untouched)

Moving the merged silu off the ACT head (now *explained*: `s24` at 16.32 us); hoisting
`f7(A_log)`; tightening the exp_g or state-out-DMA waits; splitting `S*=exp_g`; merging the f30
reduces or the READ_ACCUMULATORs *into one*; merging the per-head pointer-scale ACTIVATEs on ACT;
Pool as a host for CAST or `TENSOR_TENSOR`.

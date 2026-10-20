# iter-6 notes — gdn_tkg bh24 k128 v128, LNC=1 (24 heads on ONE core)

## Result

**66.282 -> 62.545 us, -5.6 %** (`profile.py --repeats 5` three times: 62.543 / 62.547 /
62.545, spread 0.06-0.36 %, so ~0.25 us is the believable threshold; the 2-edit intermediate
measured 63.83 and 63.91). `verify.py` PASS with `output_0`
**byte-for-byte identical** to the iter-5 NEFF (`cmp` of `.verify/run0/output_0.bin`;
`max_abs_err` still exactly 2.322e-02 / 1.633e-03), `--runs 2` bit-identical to itself,
`launch_rules.py` PASS, `semaphores.py` balanced, `cycle.py` acyclic.
Changed members: `DVE0.bin`, `Activation0.bin`, `PE0.bin`, `Pool0.bin`.
Scripts in `iter-6-scripts/`; the exact recipe:

```
applyAA.py work/payload/sg00 99 0,1,2      # 63.91  (-2.4)   move 21 S->S_bf CASTs to DVE
applyAF.py work/payload/sg00               # 63.82  (free)   fence the `o` PSUM banks
applyAE.py work/payload/sg00 99 3,4,6,7,9  # 62.54  (-1.3)   finish edit W on 7 more f128
```

| | ACT busy | DVE busy | wall |
|---|---|---|---|
| iter-5 | 48.75 | 41.26 | 66.28 |
| iter-6 | 39.49 (-9.3) | 46.66 (+5.4) | **62.54** |

---

## Edit AA: take the per-head `S -> S_bf` CAST off the Activation engine (-2.4 us)

Per head the body had `DVE U (S = S -/+ psum, fp32, in place)` -> `ACT CAST (S -> bf16 S_bf,
297 ns)` -> `PE o matmul (o = S_bf^T q)`. The CAST existed only because the matmul needs bf16:

* the **U op** keeps everything (wait, s6 increment, hence its rank) but gets
  `dst = S_bf` and `out_dtype = bf16` — one DVE op now produces the bf16 tile directly;
* an **exact clone of the original U** is inserted immediately *after* it (so it still reads
  the un-updated fp32 `S` and the same PSUM), writing fp32 `S` in place for `output_1`,
  **with no s6 update** — therefore *no rank anywhere is renumbered*, and DVE's in-order
  execution makes "the next s6 update has fired" imply "the clone has run";
* the **ACT CAST becomes an EVENT_SEMAPHORE** that keeps its wait (`s6 >= rank(U)`) and its
  s4 increment, so every PE gate that waits on its s4 rank is still fenced on S_bf —
  *zero PE edits*;
* the only waiters that needed touching are the **8 Pool state-out DGE DMAs**, which wait on
  exactly `rank(U)` and need the *fp32* S: bumped to `rank(U)+1`.

Bit-exact because a DVE op that rounds its fp32 result to bf16 on output gives the same value
as writing fp32 and casting afterwards (iter-2 for ACT, iter-5's edit W for DVE, and a byte
compare here).

Why it pays where iter-5's "move work to the other engine" lost: it **deletes an instruction
and a cross-engine hop** (`DVE -> ACT -> PE` becomes `DVE -> ACT(ES, 40 ns) -> PE`) instead of
relocating work. 5.9 us of ACT work removed, 3.7 us of DVE added, 2.4 us of wall.

**Refuted on top of it (edit AB, `applyAB.py`):** replacing the `o` matmul's gate
`s4 >= rank(ES)` with `s6 >= rank(U)` (same data guarantee, one hop less) **breaks one row**.
That s4 rank is doing **double duty**: it also fences the matmul's *PSUM output bank* against
the ACT `f30`/`READ_ACC` still reading the previous content. Keep the s4 wait.

**Sites 0,1,2 (S heads 0,1,2) must be skipped**: with all 24 converted, output **row 0** is
wrong. Skipping site 0 alone does not fix it; skipping 0,1,2 does. The clones add ~0.29 us of
DVE each and three of them land before the row-0 `N` op, so the fill region's races flip.
Fencing PSUM 0x2000000's next writer against that N op (`applyAC.py`) does **not** fix it;
the rstd-ring race below is the better suspect. Worth ~0.3 us.

## Edit AF: the fences the kernel never had (free, and it unlocks AE)

`applyAF.py` inserts 34 PE `EVENT_SEMAPHORE`s (no s5 update -> nothing renumbered) in front of
every PE matmul that **rewrites an `o` PSUM bank**, waiting on
(a) `s6 >= rank(N op that still has to read it)` and (b) `s4 >= rank(that head's READ_ACC)`.
Measured **63.820 vs 63.830 us and bit-identical** — i.e. free, because the ordering already
held; it just was not enforced. Keep this pattern in mind: **PE ES inserts are free and can
add a second wait to any gate**, which is how to get round "an instruction has only one wait".

## Edit AE: finish iter-5's edit W on 7 of the 12 remaining `f128` (-1.3 us)

`applyAE.py` re-derives the pairing from the **delta op** instead of from the PE gate, which
is what blocked iter-5 on half of these sites:

```
delta op = DVE TENSOR_TENSOR op=SUBTRACT nch=1, src0 = v_h (216576+512h), src1 = PSUM
           -> the head index h, unambiguously
f128     = the ACT ACTIVATE func=128 whose pointer scale is beta_h (16672+4h)
           (cross-check: its src must equal the delta op's dst -- holds for all 12)
U ops    = every DVE TT nch=128 with src0 == S_h and a PSUM src1  -- after AA that is TWO
           ops (the bf16 S_bf one and its fp32 clone) and BOTH must be flipped ADD->SUBTRACT
gate     = optional: if a PE LDWEIGHTS / ACT CAST waits on exactly the f128's s4 rank it is
           retargeted to s6 >= rank(delta op); if none does, the f128's EVENT_SEMAPHORE keeps
           its own s6 wait and its s4 increment and ACT's in-order execution makes any later
           s4 rank imply "the delta op has run".  **iter-5's "6 sites have no gate" wall is
           therefore not a wall at all.**
```

All 12 sites pair cleanly. 7 convert correctly (**S heads 3, 4, 7, 15, 18, 21, 22**); the
other 5 are:

* **heads 8, 12, 16** — each breaks the output row of *another* head, and the victim pattern
  is exact: **the victim is the previous head that uses the same rstd ring slot**, i.e. the
  consecutive pairs (6,8), (10,12), (14,16) of the `16384` list
  (`16384: 0,3,6,8,10,12,14,16,18,20` / `16392: 1,4,7,11,15,19` / `24024: 2,5,9,13,17,21`).
  The victim row comes out **exactly 0**, which means its `rstd` was 0, which means its sum of
  squares overflowed — i.e. **head X's `f30` squared a clobbered PSUM bank**, not that its
  rstd was overwritten. Two fixes were tried and **both failed**, so the rstd ring is *not*
  the mechanism:
  * `applyAG.py` — fence every rstd-ring slot properly (an ACT ES, no s4 update, in front of
    the later writer's `f30`, waiting on the previous user's `N` op): bit-identical but
    **+1.22 us** (62.54 -> 63.76). So ACT overwriting that slot right behind the DVE read is
    *load-bearing* for the current schedule; enforcing the order costs more than the 3 sites
    are worth. Useful datum: that is ~19 fences x ~60 ns of real stall.
  * `applyAH.py` — widen the ring instead: give heads 8/12/16 private rstd words so no WAR
    exists (**16396 and 16448..16476 are 9 free fp32 words**: the head's 24-element `a` vector
    at 16384..16479 is dead after `DVE@6`, and only 16400..16447 is re-used, for the bf16 `q`
    weights that `DVE@58` writes and the PE LDWEIGHTS read). Three field edits per head (the
    RA's dst, the 1-element `f29`'s src/dst, the N op's `imm0_imm_ptr`), nothing renumbered,
    asserts all clean — rows 6/10/14 still break identically.
  * `applyAI.py` — fence the *outer-product* banks against AA's clone (the clone extends each
    bank's read by ~0.3 us and nothing waits for it): 9 fences, no effect.

  **What a profile of the broken NEFF rules out** (`profbad/`, the 11-site variant, 62.17 us):
  every resource row 6 depends on is correctly ordered in time — its `o` bank `0x2003800` is
  written at 40.46, read by its `f30` at 41.11 and by its `N` op at 42.48, next written at
  44.88; its rstd `0x4000` is written by `RA` at 41.39, finalised by the 3-wide `f29` at 42.37,
  read at 42.48, next written at 45.02; its `S_bf` `0x97e0` is written at 40.05 and read at
  40.46; its `g` slice `0x20c00` has exactly one toucher. A static audit also shows **every**
  `U` op's `s5` wait equals the rank of its *own* outer-product matmul (no off-by-one fence).
  So the next iteration should stop looking for a WAR on those four and instead diff the
  *values*: dump `S_bf`, `o` and `rstd` for the victim head out of a device run (e.g. point a
  spare DMA at them, or make `output_1` carry them) and see which one is wrong. The victim is
  always ~2 heads before the converted one and the relation that fits the banks is
  **"the converted head's outer-product bank is the victim's `o` bank"** (head 12's outer
  product and row 6's `o` are both `0x2003800`; head 8's outer and row 10's `o` are both
  `0x2001800`), so the clobber is one PSUM generation further out than `applyAF` fences.
* **head 17** — converts fine *only together with head 16* (their f128s are the adjacent
  `ACT@122/@123` and their delta ops the adjacent `DVE@155/@156`). With 16 excluded,
  converting 17 corrupts row 16 with a *varying* ratio (not 0) — that is the w-ring version
  of iter-5's "ring producer" trap: the converted delta op writes the shared `w` slot ~500 ns
  earlier than the f128 did, before the unconverted neighbour's outer product has read it.
* **head 20** — unfenceable: its `f128` (`ACT@143`) has **no wait at all** and its delta op
  (`DVE@177`) has **no s6 update**, so nothing can order the w write against its reader.

`output_1` (the fp32 state) stayed perfect in *every* failed variant, which is the fastest way
to tell "the arithmetic and the S update are right, a downstream read raced" from "the algebra
is wrong".

---

## Deletion probes: where the wall time is (2 minutes each, do these before building anything)

`probe_del.py <engine> <slots>` replaces slots with EVENT_SEMAPHOREs that keep the wait and
the update (wrong results, timing only). On the **iter-5** NEFF (66.28):

| deleted | ACT busy | wall | gain | wall per ns of ACT |
|---|---|---|---|---|
| `ACT@4` silu (2.76 us, 3072 elem, 1 partition) | -3.0 | 65.93 | **-0.35** | 0.12 |
| all 6 `ACT_TABLE_LOAD` (7.7 us) | -11.4 | 63.25 | -3.03 | 0.27 |
| 24 body `S->S_bf` CAST (7.1 us) | -6.0 | 62.91 | -3.37 | 0.56 |
| 24 `READ_ACCUMULATOR` (7.0 us) | -6.9 | 61.25 | **-5.03** | 0.73 |

On the **AA** NEFF (63.91): 24 `READ_ACCUMULATOR` -> 59.90 (**-4.00**); the 12 remaining
`ACTIVATE f128` -> 61.68 (**-2.22**, of which AE has now banked 1.3); AA's own 21 clones ->
63.08 (-0.82, i.e. the DVE work AA adds costs only 0.8 us of wall).

**Ops on the per-head dependency chain convert ACT work into wall at 0.55-0.75; off-chain DVE
work at 0.22; anything in the head at 0.12.**

### Review candidate 1 (scatter `z` so `silu` is multi-channel) is DEAD

The whole op is worth **0.35 us**, not 2.4: deleting `ACT@4` outright only moves the median to
65.93. The head is **not** ACT-throughput-bound even though ACT is back-to-back there — the
four input `DMA_DIRECT2D` on Pool do not finish until 16.78 us and the
`ACT@6 f30 -> PE -> ACT@7 f29 -> DVE shuffles -> ACT@10 f7` chain starts from them.

## Ranked leads for iter-7

1. **Find what breaks AE's heads 8/12/16/17 and AA's sites 0/1/2** (~0.8 us if cracked).
   One output row comes out *exactly 0*, ~2 heads before the converted one. Four mechanisms
   are already ruled out by device runs (rstd fence AG, rstd-ring widening AH, `o`-bank fence
   AF, outer-product-bank fence AI) and the broken NEFF's own profile shows correct ordering
   on every resource that row depends on (details above). **Next step is a value diff, not
   another fence**: get the victim's `S_bf` / `o` / `rstd` off the device and see which is
   wrong before building anything. `rstdmap.py` prints the whole
   head -> (f30, RA, rstd slot, f29, N op) table; `pairs.py` the cast/U/gate table.
   Both edits are one command away: `applyAE.py <sg> 99 9` converts all 11.
2. **The 24 `f30` + `READ_ACCUMULATOR` pairs (-4.0 us ceiling).** `o` is 128 fp32 in *one*
   PSUM partition; `f30` (square, `accumulator_cmd=CMD_3`) + `RA` reduce it to one scalar.
   293 ns to read one float, on the chain, 24 times.
   * **CLOSED this iteration (one device run):** the accumulator is **one value per
     partition, zeroed at instruction start**. `applyAD.py` gave one `f30` a 2-group source
     pattern over two heads' banks (`num=[128,2] step=[1,-1536]`) and its `RA` a 2-element dst
     (`num=[2] step=[-2]`, `dst_element_count=2`): the RA wrote both slots but with the
     **same combined sum**, and exactly the two merged rows came out scaled wrong (0.549 and
     0.853 of the correct value). So batching the reduce across heads is impossible *as laid
     out*; it would need the two heads' `o` in different *partitions*, i.e. one matmul with a
     128x2 weight, which cannot be done because the moving operand `S_bf` differs per head.
   * Moving the reduce to DVE (`TENSOR_TENSOR MULT` + `accumulator_cmd` + `DVE_READ_ACCUMULATOR`)
     is a bad trade now: DVE is the busier engine (74.6 %) and it *adds* a hop.
3. **The 6th `ACT_TABLE_LOAD` (~0.5 us).** The 5 declared tables are each needed by exactly
   one function (`f9` softplus, `f36` silu, `f29` recip_sqrt, `f7` exp, `f5` sigmoid; `f30`
   and `f128` are in every set) and the required order is sp, silu, **rs**, exp, sig, **rs**.
   Collapsing it needs `ACT@7 f29` moved after `ACT@18 f5`, which needs the chain-A DVE
   shuffles (`DVE@7/@8`, `@11/@12`, `@28/@29`, all fed by `0x4248`) moved after the chain-B
   ones (`@9/@10`, `@13..@18`), or `ACT@10`'s `s6 >= 18` deadlocks.
4. **Heads 20's `f128` and the 3 AA fill sites** are the last stragglers (~0.4 us) and both
   need a *new* semaphore (the delta op at `DVE@177` has no s6 update — giving it one means
   renumbering every s6 wait, which `applyY.py` from iter-5 shows how to do).

## Map / facts (added or confirmed this iteration)

* **SBUF address = `partition * 0x40000 + offset`.** `z` is `partition 32, offset 0x34E40`;
  `silu(z)*norm_weight` -> `partition 0, 0x20000`; fp32 `S_h = 0x1C000 + 512h` on all 128
  partitions; the 24 **new** `S_bf` tiles are at 24 *distinct* scattered offsets (29656,
  25000, 44000, 38112, ...) — **no ring reuse**, which is why AA has no WAR hazard there; the
  head's 4 wide CASTs build the *old* `S_bf` at `0x24000 + 256h` (that is what the delta
  matmul reads); `w_n = 155648 + 256n` (6-slot ring, 2 writers each); `v_h = 216576 + 512h`;
  `beta_h = 16672 + 4h`; rstd ring = 16384 / 16392 / 24024 / 210944.
* **Output row index == S head index.** (iter-5's warning was about the v and w ring indices.)
  Build the map by sorting the `N` ops (DVE STT, `out_dtype` bf16, dst >= `96*0x40000`) by dst
  -> row; the `o` matmul is the PE MATMUL whose src is that head's `S_bf` and whose PSUM dst
  is the N op's src0. `o` PSUM banks: `0x2000000` (h0), `0x2000800` (1,4,12,15,20),
  `0x2001800` (2,5,10,13,18,21), `0x2002800` (3,8,11,16,19,22), `0x2002000` (7),
  `0x2003800` (6,9,14,17), `0x2001000` (23).
* **Per-head body chain now** (2 heads in flight, body 28 -> 74 us): `DVE delta (333)
  [-> ACT f128 (491) for the 5 unconverted heads] -> PE outer -> DVE U (S_bf, bf16, 287)
  -> ACT ES (40) -> PE o (267) -> ACT f30 (265) -> ACT RA (293) -> ACT f29 (203)
  -> DVE N (344)`. Body busy: **DVE ~81 %**, ACT ~65 %. DVE is now the busier engine — do not
  move more work onto it; the next wins have to *delete* instructions or hops.
* `probe_del.py`, `pairs.py`, `w6.py`, `head.py`, `cat.py` (in `iter-6-scripts/`) are the
  generic "size the lever before building it" / "who else waits on this rank" tools.

## Closed / re-confirmed (do not redo)

* Everything iter-1..5 closed stays closed.
* **Scattering `z` / making `silu` multi-channel** — the whole op is worth 0.35 us.
* **The head is not ACT-bound** — input DMA (ends 16.78) plus a 5-hop chain.
* **Batching the ACT accumulator across heads** — the accumulator is one value per partition,
  zeroed per instruction (measured, `applyAD.py`).
* **Retargeting the `o` matmul's PE gate from s4 to s6** (AB) — that s4 rank also fences the
  matmul's PSUM output bank.
* **Merging AA's clones pairwise** (would save 1.8 us of DVE work) — all 21 clones together
  are worth only 0.82 us of wall, so the most it can return is ~0.4 us, and it re-opens a
  PSUM-lifetime race.
* **Widening the `v *= beta` ops** — only runs of *converted* consecutive heads can be
  covered (an unconverted head's `v` must stay unscaled), so with 19 of 24 heads converted
  there are a handful of merges worth ~1 us of DVE work = ~0.2 us of wall. Low value.
* **Fencing the rstd ring** (AG) — correct, bit-identical, **+1.22 us**. The kernel's speed
  depends on that unfenced WAR.
* **Giving heads 8/12/16 private rstd slots** (AH) — does not fix their victim rows, so the
  rstd WAR is not what breaks them.
* The free scratch words in the head's dead `a` vector are **16396 and 16448..16476**
  (16400..16447 is the bf16 `q` weight array). Useful for any future ring widening.

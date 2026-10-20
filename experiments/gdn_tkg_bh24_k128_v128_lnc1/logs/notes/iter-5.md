# iter-5 notes — gdn_tkg bh24 k128 v128, LNC=1 (24 heads on ONE core)

## Result

**69.030 -> 66.31 / 66.41 us, -3.8 %** (`profile.py --repeats 5`, measured twice: 66.309 and
66.410, spread 0.29-0.30 %, so ~0.3 us is the believable threshold; the 9-site intermediate was
also measured twice, 67.437 / 67.417).
`verify.py --runs 4` PASS, all runs bit-identical, `output_0` **bit-identical head-by-head**
to the iter-4 NEFF (`max_abs_err` still exactly 2.322e-02 / 1.633e-03). `launch_rules.py`
PASS. `semaphores.py` balanced (s4 120/120, s5 74/74, s6 103/103). `cycle.py` acyclic.
Final md5 `14925985fccdd73b8f89e157a5d32eb1`.
Changed members: `Activation0.bin` (17 slots -> `EVENT_SEMAPHORE`, 5 retargeted),
`DVE0.bin` (172 -> 184 slots: 12 delta ops rewritten, 12 `S+=` ops flipped to `-=`,
12 inserted), `PE0.bin` (10 gates moved from s4 to s6).

Three kept edits (`iter-5-scripts/REPRO.sh`):

| edit | what | wall |
|---|---|---|
| **V** | fuse 2 more `ACTIVATE f128` + `CAST` pairs (iter-2 did 16; now 18 of 24) | |
| **U** | merge 3 pairs of body `ACTIVATE f29` (rsqrt) into one 3-element op | |
| U+V | ACT busy 56.09 -> 54.80 | **69.03 -> 68.15** |
| **W** | fold `w = beta*delta` into the DVE delta op (delta op emits `-w`, the matching `S += outer` becomes `S -= outer`) — 9 heads | **68.15 -> 67.44** |
| **W** | + 3 more heads whose `w` is CAST-mediated (12 of 24) | **67.44 -> 66.31** |
| all | ACT busy **56.09 -> 48.62**, DVE 38.19 -> 41.38 | **-2.6 us** |

Edit W is the important one: it is the **first edit in this experiment that shortened the
per-head dependency chain**, and it paid where the two "move work to the idle engine" edits
(X, Y) and the "reorder ACT" edit (Z) all lost. Details, and the 12 sites it has not
reached yet, below.

---

## The one number that should drive iter-6

I moved 14 of the 24 `ACTIVATE f128` (`w = beta*delta`, 490 ns each) from ACT to DVE
(edit Y, bit-identical, acyclic, verified). That took

```
ACT busy 54.80 -> 47.18 us (-7.6 us, 68 % busy)   DVE 38.19 -> 41.91 (+3.7, 61 % busy)
wall     68.15 -> 69.00 us  (+0.85, a REGRESSION)
```

**7.6 us of ACT work removed bought nothing.** Combined with edit X (below, +3.7 us wall from
*removing* 1.3 us of ACT work), the conclusion is:

> The body is a **latency chain**, not a throughput problem. Both big engines now have
> >30 % idle. Per head the chain is
> `PE delta-matmul -> DVE D -> ACT f128 -> PE outer -> DVE U -> ACT CAST -> PE o ->
>  ACT f30 -> ACT READ_ACC -> ACT f29`
> = ~2.6 us of work + ~8 cross-engine hops ≈ 3.4 us per head, and only **2 heads are in
> flight**, so the body runs ~1.75 us/head. 24 heads x 1.75 = 42 us of the 49 us body.

Two later measurements sharpen this. Edit Z (below) moved a drain CAST **earlier in the ACT
queue** without changing any work: +0.48 us, because the f30/RA/f29 triple it jumped is on the
*other* critical chain. And edit X made the cycle 0.3 us longer and cost 3.7 us. So:

> **ACT's queue order in the body/drain is already near a local optimum.** Every ACT slot in
> the body belongs to one of two chains that are both critical (`...->U->CAST->o->f30` and
> `f29->DVE N->...->output DMA`), so *reordering* ACT trades one for the other 1:1, and
> *deleting* an ACT op only pays if it is not a chain link (U and V deleted non-links).

So iter-6 should spend its whole budget on **shortening that chain or deepening the
pipeline**, and should *not* spend it on deleting or rebalancing engine work. Concretely:

1. **Get a 3rd head in flight.** iter-4's candidate 1. The blocker is PSUM: `o` uses banks
   1/5/7, the `S'^T k` delta matmul 0/2/4, the outer product 3/6 — all 8 banks are live, so a
   third head has nowhere to put its matmul output. But `o` is only 512 B of a 2 KB bank and
   `PsumUsage.parquet` is in the profile: check whether two heads' `o` can share one bank at
   different offsets (a matmul dst does not have to be bank-aligned — Pool@1..4 show DGE
   writing arbitrary offsets). If yes, 2-3 banks free up and the fill/drain (5.9 us of ACT
   idle) collapses.
2. **Delete a hop, not an op.** Every edit that removed a *cross-engine* link paid (iter-4
   J1/K/L, -1.17 us); every edit that moved *work* between engines lost (iter-5 X and Y,
   +4.5 us together). The remaining hops worth attacking: `ACT f29 -> DVE N` (the last link,
   drain-critical) and `DVE U -> ACT CAST -> PE o` (two hops for one 297 ns cast).
3. The f128->DVE move (edit Y) is **ready to use** and bit-identical — if a future edit makes
   ACT the bottleneck again, `applyY.py work/payload/sg00 99 "7"` re-applies it in one line.
4. **Finish edit W (kept here for 9 of 24 heads, -0.71 us).** It is the proof of the point
   above: it deletes the ACT `f128` *and* two cross-engine hops per head, and it is edit Y
   minus the thing that made Y lose. 15 more heads are available — 9 of them need the
   CAST-mediated retarget (~10 lines), 3 carry a latent fence, 3 are ring producers. Worth
   ~1.3 us. See the edit-W section.

---

## Edit U: merge pairs of body `f29` rsqrt (kept, 3 of 12 pairs)

24 body `ACTIVATE func 29` x 202 ns = 4.85 us of ACT. Each is a **1-element in-place** rsqrt
(`dst = rsqrt(src*0.0078125 + 1e-6)`, src == dst) of the scalar `READ_ACCUMULATOR` just wrote.
The 24 live in a 4-slot ring: **16384, 16392, 24024, 210944**, and the pairs of a pipeline
cycle always use two *different* slots, so two f29 can be done by ONE op if the two
addresses can be covered by one access pattern.

The merged op must sit at the **second** f29's slot (ACT is in-order; the first slot cannot
wait for the second `READ_ACC`). The first slot becomes an `EVENT_SEMAPHORE` that keeps its
s4 increment — **no s4 rank is renumbered** — and the DVE consumer of the first rstd (the
`SCALAR_TENSOR_TENSOR` whose `imm0_imm_ptr` is that address) gets its s4 wait **raised** to
the merged op's rank. All 24 consumers wait on exactly their own f29's rank, so the retarget
is unambiguous (`applyU.py` asserts it).

`applyU.py <sg> <npairs> <skip-list>`; it reverts any pair that `cycle.py` says would
deadlock. Result: **3 pairs merge cleanly** (`f29@68+@71`, `@93+@96`, `@118+@121`), -162 ns
each.

### Why only 3 of 12 — and the addressing law that explains it

* **5 pairs are (16384, 24024) or (24024, 16392)**, i.e. 1910 elements apart. A strided
  2-element pattern **does not work on ACT**: with `num_elem=[2,1,1] step_elem=[1910,1,1]`
  *and* with `[1,2,1]/[1,1910,1]` the device produced **identical wrong output, exactly one
  head wrong** (the head whose rstd slot was the second address — it kept the raw
  accumulator sum, so only element 0 was processed). Per-head bf16 diff against a saved
  reference is what proved this; the aggregate `max_abs_err` would not have.
  **Law: an `ACTIVATE`'s element count is `num_elem[0]` and its `step_elem[0]` is ignored
  (treated as 1).** Only a *contiguous* run works.
* The 4 pairs that are **(16384, 16392)** — 2 elements apart — merge as a **3-element
  contiguous** op covering the dead scratch word at **16388** (verified free: the only thing
  that ever touches it is the head's 24-element vector at 16384..16479, dead after DVE@6 at
  20.0 us). 1 of those 4 (`f29@142+@147`, the drain pair) makes **head 19** wrong — see
  "latent fences" below. The other 3 are the kept edit.
* 1 pair is 48640 elements apart (> the signed-16-bit step limit) and 2 more would create a
  semaphore cycle (`cycle.py` catches both: `ACT@39/40 <-> DVE@62..64` and
  `ACT@48..51 <-> DVE@72`).

**To finish this lever** (worth ~0.8 us of ACT, ~0.5 us wall): retarget the `READ_ACCUMULATOR`
dst of the second head of each (16384, 24024) pair to **16392** (and its DVE consumer's
`imm0_imm_ptr` with it), which turns it into a mergeable (16384, 16392) pair. Risk: 16392
would then be written every cycle instead of every other cycle, tightening a WAR window that
has no explicit fence.

## Edit V: fuse the remaining `f128` + `CAST` pairs (kept, 2 of 8)

8 heads still had `ACTIVATE f128` (fp32 out) followed by a `CAST` to bf16; iter-2 fused the
other 16. Fusing = give the f128 the CAST's `out_dtype`/`dst` and turn the CAST into an
`EVENT_SEMAPHORE` that keeps its s4 increment (every existing wait stays valid: the data is
now ready *earlier* than the rank it is tied to, and only the one PE matmul reads it).

`applyV.py` finds the 8 pairs and refuses any whose fp32 staging buffer is read by something
other than that CAST before the next writer (the staging buffers are a rotating ring).
6 pass that filter, and on the device **3 of the 6 are wrong** (`f128@136` -> head 18, and
two of `@143/@144/@155` -> heads 19, 21). The 2 that are safe (`f128@97`, `f128@134`) are the
kept edit. The failures are all in the **drain**, and all are the "latent fence" class below.

## The latent-fence class (both edits hit it, and so did iter-4's edit T)

Every failure in this iteration had the same shape: an edit that only *deletes ACT work* or
*moves a write 300 ns earlier* corrupts **one specific drain head**, deterministically, while
every other head stays bit-identical. The common factor is that ACT advances earlier relative
to PE/DVE, and some buffer in the drain is protected only by ACT's own progress rate, not by a
semaphore. iter-4 found the same thing on head 23 and called it the same.

**The diagnostic to use first** (it is what made this iteration tractable):

```
$PY tools/verify.py --keep --runs 1          # once on the pre-edit NEFF -> save output_0.bin
...apply edit...
$PY tools/verify.py --keep --runs 1
# then: per-head uint16 compare of .verify/run0/output_0.bin against the saved reference
```
A single head wrong = a fence; all heads of the edited set wrong = wrong arithmetic/addressing.
`try.sh` / `try2.sh` in the worktree do apply+check+per-head-diff in one command.

## Edit X (REFUTED, +3.7 us): merging the per-head S->S_bf `CAST` pairs

The 24 body `CAST` (nch=128, 297 ns, fp32 S at `114688+512h` -> bf16 S_bf) can be merged
pairwise into one 2-group CAST (~400 ns) — **9 pairs merged, output bit-identical, and the
wall went 68.15 -> 71.82 us.** ACT busy fell 54.80 -> 53.54 and the kernel got 3.7 us slower.

Mechanism: the merged cast cannot run until *both* heads' `DVE U` are done, so head A's
`S_bf` lands ~300 ns late, and `CAST -> PE o -> ACT f30` is the cycle-limiting loop — 12
cycles x 0.3 us = 3.6 us. **Do not merge anything across the two pipeline stages.** This is
the measurement that proves the body is latency-bound.

### But it settled the access-pattern semantics (valuable, reusable)

`step_elem` is **in elements, and the offset is the plain sum `sum_k idx_k * step_k`** — the
first attempt used `step[1] = distance/num_elem[0]` ("stride multiplier") and *all 16* heads
of the merged pairs came out wrong; `step[1] = distance_in_elements` is bit-identical.
Negative steps work (one pair has a dst distance of -5888 B -> `step[1] = -2944`).
**Unexplained and worth 10 minutes of someone's time:** the four head-region casts
(`ACT@14..17`, `num_elem=[128,6,1,1] step_elem=[1,1,1,1]`, 831 ns, fp32 S -> bf16 at
147456+1536n) cannot be doing 768 contiguous elements under sum semantics — yet the kernel
is correct. Either those casts are partly dead or the 4-D CAST pattern has an extra rule.

## Edit Y (REFUTED, +0.85 us): moving `w = beta*delta` from ACT to DVE

14 of 24 `ACTIVATE f128` replaced by a new DVE `TENSOR_TENSOR_ARITH_OP` inserted right after
the delta op, **bit-identical output**, and 0.85 us slower (numbers at the top). Keep the
script: it is the only worked example in this experiment of *inserting* instructions that
carry a semaphore.

Three things it establishes:

1. **A step-0 broadcast multiply on DVE reproduces an ACT scale-multiply bit-exactly.**
   `TENSOR_TENSOR_ARITH_OP op=MULT(6)`, `src0 = delta` (step [1,1,1], num [128,1,1]),
   `src1 = the f128's own scale pointer` with **step [0,0,0]**, num [128,1,1], `out_dtype` =
   the f128's. The f128's per-head beta lives at `16672 + 4h` and `ACTIVATE.scale_src = 0`
   means **POINTER**, not instruction immediate (`scale_value_imm_ptr` holds the address;
   the decoded `scale_value_imm_fp32` then looks like a denormal ~2.3e-41 — that is the
   tell-tale of a pointer).
2. **`SCALAR_TENSOR_TENSOR_ARITH` with `op1 = BYPASS (0)` does NOT yield `(src0 op0 imm0)`.**
   The first attempt used that form and all 15 heads were wrong (and `output_1` too). Use the
   step-0 TENSOR_TENSOR instead.
3. **Inserting DVE instructions with an s6 increment is mechanical and safe.** s6 is only
   incremented by DVE, so its ranks are just DVE program order: build the pre-insert list of
   updating slots, build the post-insert list, map old rank -> new rank by instruction
   identity, and rewrite `events_semaphore_value` on **every** s6 waiter in all five streams
   (the Pool state-out DGEs and `SP@6` wait on s6 too — do not forget them).
   `semaphores.py` then balances at 117/117 and `cycle.py` stays acyclic.
   The PE gate is retargeted from `s4 >= rank(f128)` to `s6 >= rank(new DVE op)`.
   One site (`ACT@61`, head 6) is wrong — same latent-fence class, skip it with `"7"`.

## Edit Z (REFUTED, +0.48 us): bubbling the drain's `w` CAST left

In the drain `PE@153` (the outer product for head 22) waits on `ACT@159`, the CAST that writes
its bf16 `w`, and that CAST **had no wait of its own** yet ran 3.7 us after its data was ready
— it was sitting in the ACT queue behind an `f30 / READ_ACC / f29` triple (762 ns) belonging to
a *different* head. `applyZ.py <sg> 159,164` bubbles such a CAST left past that triple and
recomputes/remaps every s4 rank by producer identity (clean, bit-identical, acyclic,
5 waits remapped). Measured **68.15 -> 68.62 us**.

Why it loses: the triple it jumps ends in the `f29` that feeds `DVE N`, and in the drain the
`N` chain is what gates `SP@6` (the output_0 DMA trigger). You buy ~1 us on the `w` chain and
pay it straight back on the `N` chain. **The drain needs both chains shortened at once, or
neither.** `applyZ.py` is a clean, reusable "bubble an ACT op left and fix the ranks" tool.

## Edit W (KEPT, -0.71 us on 9 of 24 heads — finish it, it is worth ~1.3 us more)

Goal: delete the ACT `f128` from the per-head chain **without adding a DVE instruction**.

The identity that makes it free: let the delta op produce **-w** instead of `delta`, and flip
the matching `S += outer` to `S -= outer`. Sign flips are exact in floating point (negating a
bf16 is exact, negating one matmul operand negates every product and their fp32 sum exactly),
so `S - (k (x) (-w)) == S + (k (x) w)` **bit for bit**.

```
delta op (DVE TENSOR_TENSOR  v - kv)      ->  SCALAR_TENSOR_TENSOR  (kv * beta) - (v*beta)
ACT f128 (w = beta*delta)                 ->  EVENT_SEMAPHORE (keeps its s4 increment)
DVE U    (S += outer, op=ADD(4))          ->  op = SUBTRACT(5)
PE LDWEIGHTS gate  s4 >= rank(f128)       ->  s6 >= rank(delta op)
```

`applyW.py` does all of that for the sites that have an unambiguous producer and a PE gate,
and `cycle.py`/`semaphores.py` stay clean (the delta op keeps its own wait **and** its s6
update, so **nothing is renumbered**). The pairing code is worth keeping: f128 -> delta op via
the f128's own s6 wait, delta op -> PE gate via `s4 == rank(f128)` + the MATMUL's src, PE gate
-> `U` op via `s5 == rank(that MATMUL)` + `src0 == dst` + `nch=128`.

**The piece that makes it correct:** `SCALAR_TENSOR_TENSOR` computes
`(src0 op0 imm0) op1 src1` — the scalar multiplies **src0 only**. So the form above yields
`beta*kv - v`, not `beta*kv - beta*v = -w`. **`v` must be pre-scaled by beta.** The kept
version inserts a per-head in-place `v *= beta` immediately before each converted delta op
(`TENSOR_TENSOR op=MULT`, `src0 = dst = v_h` step [1,1,1] num [128,1,1], `src1 = beta_h` with
**step [0,0,0]** num [128,1,1], nch=1, fp32, `wait s4 >= 12` = the rank of `ACT@18 f5` which
writes the beta vector, **no s6 update** — DVE program order already orders it before the
delta op, so nothing is renumbered). That costs ~280 ns of DVE per head, and the edit still
wins, which says the chain matters more than the engine load. **The cheap improvement left:
replace the 24 per-head v-scales by 4 wide 6-head ops** (`src0 = dst = 216576`
step [1,128,1] num [128,6,1], `src1 = 16672` step [0,1,1] num [128,6,1]), each ~700 ns
instead of 6x280 = 1.7 us, placed just before the first delta op of its group:

* `v` is **216576 + 512*h, h = 0..23, contiguous** (one 12 KB block in one partition), read by
  nothing but the 24 delta ops (and overwritten later by `ACT@161` and `DVE@79`, both *after*
  the last read of the slots they touch — so **in-place scaling is safe**).
* `beta` is **16672 + 4*h**, written by `ACT@18` (`f5`, 24 elements, s4 rank 12) which ends at
  **26.65 us**; the first delta op runs at ~27.5 us.
* so the op is `TENSOR_TENSOR op=MULT, src0 = dst = 216576 step [1,128,1] num [128,24,1],
  src1 = 16672 step [0,1,1] num [128,24,1], nch=1, out_dtype 10, wait s4>=12, no s6 update`
  (no update needed: the delta ops are later in DVE program order, which already orders them).
  **That is 3072 elements ~ 2.5 us and it does NOT fit in the 0.85 us window between beta and
  the first delta op** — so split it into 4 chunks of 6 heads (~700 ns each) and insert chunk
  k just before the first delta op that reads heads 6k..6k+5. Inserting DVE ops with no s6
  update is free (iter-4's T22 did it).

**Trap 1 (cost me two device runs):** the delta buffers are a **4-slot ring**
(8598528 / 8604672 / 25182888 / 25182936 / 25183448 / 25189032) and 11 of the 24 delta ops
write `8604672`. If you convert a delta op whose slot is still read by an **unconverted** f128
(e.g. `ACT@143`, which has **no s6 wait**, so its producer can only be identified by measured
time — it is `DVE@132`), that f128 reads a stale slot and **every** head downstream of it
breaks. Converting only site 0 (`ACT@22`, whose slot `8598528` has exactly one writer and one
reader) isolates the arithmetic from this effect — do that first.

**Trap 2:** pick the STT template by `imm0_src == 1` (POINTER_IMMEDIATE). The first STT in the
DVE stream (`DVE@6`) uses an *instruction* immediate of -1.0, and cloning it silently turns
`beta` into the denormal `2.3e-41`.

**Trap 3:** `op1 = BYPASS (0)` on an STT does **not** pass `(src0 op0 imm0)` through (measured:
all 15 heads wrong). If you need a plain scale-multiply on DVE, use `TENSOR_TENSOR op=MULT`
with a **step-0 broadcast** of the scalar (edit Y proved that bit-identical).

**Which sites are in, and what is left.** `applyW.py <sg> 99 "7,15,17"` converts **12 of the
24 sites** (-1.84 us in two steps: -0.71 for the 9 that have a PE gate, then -1.13 for the 3
more whose `w` goes through an ACT `CAST` — for those the CAST, not the PE, carries the gate
and gets the s6 retarget; the delta op then writes fp32 and the CAST still runs on ACT).
The remaining 12, in order of how to attack them:

* `99 ""` -> 12 sites but **output rows 6, 14, 16 wrong**; excluding sites **7, 15, 17** fixes
  exactly those three. Site 7 is the *same site* edit Y had to drop (it broke the same row by
  a completely different mechanism), so **those 3 sites carry a latent fence of their own**,
  not an algebra bug. Worth ~0.25 us if cracked. Note the naming trap that cost me 4 runs:
  the **output row index is NOT the v index and NOT the w-ring index** — `v_h = 216576+512h`
  and `w_n = 155648+256n` diverge at the first unconvertible head. Build the map by walking
  `delta op -> PE outer -> U -> CAST -> PE o -> f30/RA/f29 -> DVE N(dst = 25197024+256*row)`.
* 3 sites are excluded automatically as **ring producers**: a converted delta op stops writing
  its slot in the 6-slot delta ring, so if an *unconverted* f128 resolves to that producer it
  reads a stale slot (this is what broke all 24 rows on the first try). `applyW.py` resolves
  each unconverted f128's producer through its own s6 wait and protects it; `ACT@143` has no
  s6 wait, so all 5 candidate writers of its slot are protected (conservative).
* **6 sites have no gate that my search can retarget** (`ACT@30, @59, @97, @134, @143, @144,
  @155`): the op that consumes their `w` waits on an s4 rank that is *not* the f128's, because
  the `w` ring slot has two writers and the consumer is gated on the later one. An instruction
  has only one wait, so you cannot add `s6 >= rank(delta op)` without dropping that s4 fence
  — resolving them means first working out what the s4 fence is actually protecting (iter-4's
  edit R hit the same wall). ~3 us of ACT sits here.

Full-coverage payoff estimate: another ~1-1.5 us on top of the 2.72 already banked.
**Moving the `v *= beta` ops earlier is already refuted** (`VK=8 applyW.py`, bit-identical,
66.63 vs 66.31 — +0.32 us): the delta op waits on its PE `kv` matmul anyway, so the v-scale
runs inside that shadow where it is *not* on the chain; moving it earlier only crowds a busier
part of the DVE queue. **Widening** them (one op per run of consecutive heads, 190 + 110/head
instead of 280/head) is still untried and would cut DVE work without touching placement.

**Two fences the v-scale depends on — do not break them:**
* beta comes from `ACT@18` (`f5`, 24 elements at 16672, s4 rank 12), ready at 26.65 us; each
  v-scale waits `s4 >= 12` explicitly.
* `v` itself arrives by DMA on **sem 12**, and the only DVE op that waits for it is **`DVE@41`**
  (`EVENT_SEMAPHORE`, `s12 >= 16`), immediately before the first delta op. So a v-scale is
  only safe **after DVE@41**; `applyW.py` computes that floor. Placing one before it scales
  garbage and **all 24 heads come out wrong** — that is how I found the fence.

## Map of the kernel confirmed/extended this iteration

* **fp32 state S lives at `114688 + 512*h`** (24 heads x 128 fp32 per partition, 128
  partitions) and **`output_1` (state_out) is DMA'd straight out of it in fp32** by the 8
  `Pool PSEUDO_DMA_DIRECT2D` at Pool@5..12 (3 heads each, `in_dtype = out_dtype = 10`,
  gated on `s6 >= {30,42,56,63,74,81,92,100}` = the DVE `U` ops). So **the fp32 S write
  cannot be removed** and the 24 S->S_bf casts cannot be folded into `U` (I checked this
  specifically — it was the most promising "delete 7 us of ACT" idea and it is dead).
* `SP0.json` DMA descriptors support **`"op": "cast"`** (z and v arrive bf16->fp32 that way).
  That is the lever for the "scatter z over 24 partitions so `silu` becomes a 24-channel
  op" idea (ACT@4 `f36` is **2762 ns**, 3072 elements in ONE partition, and it is on the head
  critical path) — `z` is `to_off 8605184, to_sizes [8192,1], to_steps [1,1]`, a single
  partition. Untried. It would also need `norm_weight` broadcast to 24 partitions and the 24
  DVE output ops repointed at partition h. ~2.5 us of ACT, in the head where ACT *is* the
  bottleneck (the head has only 2.2 us of ACT idle in 17.7 us).
* ACT per-instruction cost model (fits every op in the kernel to ~5 %):
  **`190 ns + 0.85 ns x elements-per-partition`**, independent of `num_active_channels`.
  So a 1-channel 128-element op (f128, f30) is ~55 % fixed overhead and a 128-channel one
  (CAST) is ~37x more efficient per element. Merging ops is worth 190 ns each — *if* the
  merge does not lengthen the chain.
* ACT work inventory at 68.15 us (54.80 us busy): `f128` 11.8, body `CAST` 7.1,
  `ACT_TABLE_LOAD` 6x1.283 = 7.7, `READ_ACC` 24x0.293 = 7.0, `f30` 6.4, `f29` 4.4,
  4 wide head casts 3.3, `f36` silu 2.8, 8 w-casts 1.8, head ops 1.5, `EVENT_SEMAPHORE` ~1.2.
* ACT idle is now **11.5 us** in the 66.3 us ACT window, and only 3 gaps are over 0.6 us:
  **1.14 us at 19.8** (`f7@10` waiting for the DVE `a`-broadcast shuffles — structurally
  serial: `ACT f7@9 -> DVE shuffles -> ACT f7@10 -> DVE exp_g -> ACT wide casts`), and
  0.70/0.67/0.67/0.60 in the **drain** (74-76.6 us, one head in flight).

## Closed / re-confirmed (do not redo)

* Everything iter-1..4 closed stays closed.
* **Merging per-head CASTs across the pipeline pair** — +3.7 us (edit X).
* **Moving `w = beta*delta` to DVE** — +0.85 us (edit Y), despite -7.6 us of ACT busy.
* **Folding the S->S_bf cast into the DVE `U` op** — impossible: `output_1` is DMA'd from the
  fp32 S (descriptor evidence above).
* **Strided (non-contiguous) `ACTIVATE` patterns** — only `num_elem[0]` contiguous elements
  are processed; `step_elem[0]` is ignored.
* **`SCALAR_TENSOR_TENSOR` `op1 = BYPASS`** — does not pass the left operand through.
* **Bubbling an ACT op left in the drain** (edit Z) — +0.48 us, both drain chains are critical.
* 6 `ACT_TABLE_LOAD` is still a hard floor (iter-4 proved it from table membership).

## Tooling added (in `iter-5-scripts/`, all take `<sgdir>` first)

| script | what it does | verdict |
|---|---|---|
| `applyU.py <sg> <n> <skip>` | merge adjacent body `f29` pairs into one contiguous op; cycle-guarded, auto-reverts a pair that would deadlock | **kept, 3 pairs** |
| `applyV.py <sg> <skip>` | fuse `f128`+`CAST`; refuses any whose fp32 staging slot has another reader before the next writer | **kept, 2 sites** |
| `applyX.py <sg> <skip>` | merge consecutive-head S->S_bf CASTs pairwise (dst distance in the dim-1 step) | refuted |
| `applyY.py <sg> <n> <skip>` | move `f128` to DVE as a step-0 broadcast multiply; **includes the full s6-rank renumbering for inserted DVE updates** | refuted, reusable |
| `applyZ.py <sg> <slots>` | bubble an ACT op left past f30/RA/f29 and remap all s4 ranks by identity | refuted, reusable |
| `applyW.py <sg> <n> <skip>` | fold `beta*` into the delta op + flip `U` to SUBTRACT, + the per-head `v *= beta`; protects ring producers, asserts each `U` is claimed once | **kept, 9 sites** |
| `try.sh` / `try2.sh` | apply + semaphores + cycle + verify + **per-head bf16 diff** in one command | use these |

The per-head diff needs a reference: `tools/verify.py --keep --runs 1` on the unedited NEFF,
then copy `.verify/run0/output_0.bin` somewhere (the scripts expect `/tmp/ref_verify/`).

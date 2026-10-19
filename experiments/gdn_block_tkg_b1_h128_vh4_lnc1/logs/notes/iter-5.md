# iter-5 — 60.04 → 51.03 us (1.18x on top of iter-4; 4.28x vs the 218.17 us baseline)

Start-of-iteration re-measure of the iter-4 file: **60.039 us** (spread 0.47%, 5 repeats).
Shipped file sha256 `063887b45f2e68ad9bb8a6ba50108a149015573a8b748843bdfeb0076675ae7c`.
Shipped medians of 5, three times: **51.115 / 51.029 / 51.088** — stable to 0.09 us.
Spread on this device was 0.3–0.7%
for everything up to `n2`; the shipped `n4` is **bimodal** — four captures at ~51.0 and one at
53.3–53.9 in each run of five, which pushes the reported spread to ~5%. The outlier's position
within the five varies (capture 2, 3 and 4 across the three runs), so it is a genuine
run-to-run bimodality and not a fixed profiler artefact; see §8. The median is the metric and
reproduces to 0.09 us, but **re-measure with 5 repeats before trusting a sub-1-us win on top
of this file**, and treat chasing the outlier as a real candidate (§5 candidate 0).
`verify.py` gives **exactly the baseline errors** (2.999e-03 / 1.928e-03 / 6.033e-04) after
every shipped edit — all arithmetic bit-identical, and `--runs 3` passes on the shipped file.
`semaphores.py` clean. `launch_rules.py` passes (1 subgraph, num_tpb 1, 11 inputs, 3 output
heads, LNC=1, one rank).

| state | what | median of 5 |
|---|---|---|
| iter-4 (HEAD) | — | **60.039** |
| k2 | epilogue merge, skipping *middle* queue blocks | **HANG** (§1) |
| k1 | epilogue merge keeping every trigger | **59.352** |
| h2 | private dst for Act 100/101 + move the three func-36 ops + delete 2 ACT_TABLE_LOADs | **56.473** |
| k3 | epilogue: trigger only the first block, skip the trailing ones | **55.501** |
| k4 | stop triggering the 10 dead dummy qSPSpillReload0 blocks 5–14 | **55.267** (+0.23, marginal) |
| m1/m2 | 4-tile + 2-tile merge of the input-2 tile loads, Pool guard ESs clobbered | **HANG** (§4) |
| m3/m4 | same with the Pool ES updates kept | verify FAILS (§4) |
| m5 | only the (slot 7, slot 8) 2-tile merge | bit-identical, **55.178** (noise, not shipped) |
| m6/m7 | 3 and 2 of the 2-tile merges | verify FAILS (§4) |
| n1 | transpose group 2 → 128 descriptors, gather as ONE strided CAST | verify FAILS (§3) |
| n2 | same with the gather as FOUR single-element CASTs | **53.867** / 54.019 |
| n3 | transpose group 1 → 128 descriptors | verify PASSES, **profiler HANGS** (§8) |
| n3g | probe: only n3's new SP gate `W[59]>=32` | 55.081 (costs 1.06, no hang) |
| **n4 (shipped)** | n3 + SP 19/20/21 gated on the merged transfer's own semaphore | **51.029** / 51.115 |
| n5 | transpose pair A (SP 34+36) merged the same way | 51.162 (neutral, reverted) |
| n6/n7 | transpose pair B (SP 46+50), scratch at 246144 | verify FAILS — **scratch too high (§9)** |
| n7' | pair B with scratch at 244096 | bit-identical, **53.527 (+2.5 REGRESSION, §9)** |

---

## 0. THE HEADLINE FINDING: iter-4's blocked func-36 move was a 512-vs-256-byte alias

iter-4 spent its last hours on `g7`/`g7b` — moving the three `ACTIVATE func 36` ops at
Activation slots 100/101/102 up next to slot 82 so the `LOAD2` and `LOAD4` after them become
redundant — and got `output_0` off by 8348% with *no identified cause*, leaving an
"ACTIVATE may accumulate" hypothesis. Both parts of that are now settled:

* **The accumulator hypothesis is dead and needs no experiment.** `ACTIVATE` has a field
  `accumulator_cmd`: it is `ZERO_ACCUMULATE` on exactly the `func 30` ops that precede an
  `ACTIVATION_READ_ACCUMULATOR` (slots 94/103/115/124) and `IDLE` on every `func 36`/`func
  29`/`func 128`. So an `IDLE` op cannot perturb a `READ_ACC`, and moving one across a
  `READ_ACC` boundary is free. **Check `accumulator_cmd` before worrying about this again.**
* **The real cause: `ACTIVATE` slot 101 writes 128 fp32 = 512 bytes at `p0+17416` (0x4408),
  and that range covers `0x4500` = 17664, which Activation slot 87 fills with 128 **bf16**
  (256 bytes) as a matmul operand that `PE` reads 0.35 us later.** In the compiler's schedule
  slot 101 runs at 59.4, long after PE's read at 54.1, so the overwrite is deliberate buffer
  reuse. Moving slot 101 earlier makes it land *before* slot 87, so half of slot 101's own
  512-byte result is then overwritten by slot 87's 256-byte bf16 tile. Same story for slot
  100 (dst 16384 covers 16640, read by `PE 106 MATMUL` at 52.465 — only **55 ns** after
  Activation slot 85 writes it, with no semaphore between them).

**The general rule this gives you, and it is the most reusable thing in this note:**

> On this engine an fp32 `[128,1,1]` result occupies 512 bytes while the bf16 tiles around it
> occupy 256, so **a destination that looks private by start address is routinely half of
> another live buffer.** Before moving any compute op, scan the *whole* `[dst, dst+512)`
> range, not the start address — and scan the profile's `operands` strings, because `PE`
> `LDWEIGHTS`/`MATMUL` reads are ordered by PE program position alone and carry no wait that
> static analysis can see.

How I found it, which is the method to reuse: grep `prof/Instruction.parquet`'s `operands`
column for every hex literal in the range and sort by `start_ts`.

```python
import pandas as pd, re
d=pd.read_parquet('prof/Instruction.parquet'); t0=d.start_ts.min()
a=d[d.execution_unit!='sequencer'].copy(); a['s']=(a.start_ts-t0)/1000
a['h']=a.operands.map(lambda o:[h for h in re.findall(r'0x([0-9a-f]+)',str(o)) if LO<=int(h,16)<=HI])
print(a[a.h.map(len)>0].sort_values('s')[['engine','pc','opcode','s','operands']].to_string())
```

### The fix, and the two loads it deletes (−2.77 us, the biggest single win)

`g7a` (already in iter-4's notes, verified bit-identical) gives slot 100 a private in-place
destination `8602880` and re-points `DVE 106`'s `src1`. Do the **same for slot 101**: dst
`17416 → 8603392` (its own source, which has exactly one writer `Act 76` and one reader) and
re-point `DVE 112`'s `src1`. Slot 102 already writes in place (iter-4 `g4`). With all three
destinations private the move is bit-identical. I landed them just after slot 85 (before the
`sel 4` load at 86); landing position is irrelevant — iter-4's slot-83 landing fails for the
same alias reason, not because of the position.

**Bisect order that localised it in two runs:** move slot 102 alone → PASS; add slot 101 →
FAIL with iter-4's exact error. One op at a time is the right granularity here.

Tail table loads are now **4 → 2** (`sel 4` at slot 89 and the head's `sel 2` at 59).

## 1. THE SECOND FINDING: a queued DMA block can be left untriggered iff it is *trailing*

The epilogue was 9 queued DMAs for 1 KB of output: `qSPSpillReload0` 15 (a 2-byte dummy),
16–19 (four 256-byte SBUF→SBUF staging copies), then `qSPIO0` 2–5 (four 256-byte
SBUF→`output_0` stores), with ~0.16 us per SP trigger and a 0.77 us wait in the middle. The
source `SB p32+18184` is **1024 contiguous bytes** and `output_0` is 1024 contiguous bytes, so
one descriptor does the whole thing.

* `k2` put the merged store in `qSPIO0` **block 5** and turned the triggers for blocks 2/3/4
  (and `qSPSpillReload0` 15–19) into `EVENT_SEMAPHORE`s → **the device hung** (model stop
  timeout).
* `k3` put the merged store in `qSPIO0` **block 2** — the next block in that queue's ring
  order after the already-triggered 0 and 1 — and skipped 3/4/5. **Works.**

> **A queue's `PSEUDO_DMATRIGGER`s must stay a prefix of its block-id order.** The block id in
> `count_union_block_id` does not random-access the ring; skipping blocks 2–4 and triggering 5
> hangs. Skipping *trailing* blocks is free. (Doc ch. 9.3 reads like absolute addressing — it
> is not, at least not for skipped blocks.)

Consequence, and this is a cheap lever anyone can reuse: **iterations 1–4 left ~15 queued
blocks as 2-byte "dummies" that are still triggered and still cost ~0.15 us of SP plus a DMA
slot.** `k4` stopped triggering `qSPSpillReload0` 5–14 (all dummies, all with 0 waiters except
SP 72's own guard on s36) — legal because blocks 0–4 stay and 15–19 were already dropped by
k3, so the triggered set is still the prefix {0,1,2,3,4}. Only +0.23 us (marginal vs the
0.4% spread) but it removes 10 DMAs, SP busy 11.1 → 10.3 and DMA busy 21.4 → 20.8.
Epilogue total: last compute → last store was 3.0 us, now ~0.6 us.
Also needed: `Pool 71`'s final barrier `W[45]>=16` re-pointed to `W[42]>=16` (block 2's
semaphore) — **keep the barrier on the block that actually carries the output**, or the host
can be told the run is done before the store lands.

## 2. THE THIRD FINDING: an `EVENT_SEMAPHORE` used as a queue guard usually also *supplies* a count

`m1`/`m2` hung for a reason that has nothing to do with what they were testing: Pool slots
13–20 look like pure guards (`W[8x]>=16`) but each also carries
`SEM_ADD_IMM_COMPLETE[20/21/22/23/8/9/10/11] += 16`, and my "make it a no-op ES" helper
cleared `events_update_mode` along with the wait. **Before no-op'ing any ES, print its update
field.** Re-pointing the *wait* and leaving the update intact (`m3`) turned the hang into a
plain wrong answer, which is a much better failure mode to debug from.

## 3. THE FOURTH FINDING: 4× fewer DGE descriptors on a partition→row transpose (−1.40 us)

iter-4 established that the DGE retires ~1536 two-byte transpose descriptors at ~8.4 ns each
between 38 and 51.5 us and that **only removing descriptors pays**. Here is a way to remove
three quarters of them for one group:

> A `128 partitions × 2 bytes → 1 row of 128 bf16` transpose costs 128 descriptors **per
> column**. Four such columns cost 512. If the four columns are first packed, *within each
> partition*, into 8 contiguous bytes by compute, **one** 128-descriptor transfer of 8-byte
> elements moves all four, and the consumers read the interleaved row with a stride of 4
> elements. 512 → 128 descriptors.

Applied to SP slots 27/32/37/38 (`src` per-partition 18432/18688/18944/19200 at a 256-byte
stride → `p0` rows 218368/217600/216832/216064). Measured DGE time for the group was
0.53+1.00+1.17+0.63 = 3.3 us; it is now one transfer. **−1.40 us, bit-identical.**

The pieces (script in §6, `an/n2.py`):
1. four single-element `CAST`s on DVE after slot 52 (the last producer) packing the four
   values into `G=241024 .. +7` in every partition, the last one adding 16 to the **unused
   semaphore 46**;
2. SP slot 27 rewritten in place: `src G`, `src_elem_size 8`, `src_step_elem [262144,1]`,
   `dst 240000`, `dst_elem_size 8`, `dst_step_elem [8,1]`, `num [128,1]`, semaphore 58 kept;
3. SP slots 32/37/38 → `EVENT_SEMAPHORE`s that keep their wait and add 16 to their own
   semaphore (`events_semaphore_value` is shared, and wait == add == 16 here, so one slot
   suffices);
4. Activation CASTs 74–77 re-pointed to `240000 + 2*c` with `src_step_elem [4,1,1,1]`;
5. `Act 72`'s extended wait on `s60` dropped and `Act 73` no-op'd — after the merge only
   `s58>=32` still means "the data landed", and Activation is in order so one gate covers all
   four rows. **This is the trap iter-4 flagged: when you turn a DMA into a semaphore add,
   every wait on that semaphore stops meaning "the bytes are there".**

Three details that cost me runs:

* **`PSEUDO_DMA_DIRECT2D` has no `events_extended_*` fields** (`isa.py` raises `KeyError`), so
  a DGE DMA can hold only one wait. I needed a second gate (the gather) and got it free:
  **SP slot 23 is `ES W[74]>=16` with no update and slot 24 repeats the same wait**, so slot 23
  is a spare gate slot. Look for such duplicated-wait ESs instead of inserting slots.
* **`src_step_elem` on a compute op is NOT reliably in elements.** `n1` used iter-4's
  never-tested one-CAST strided form (`num_elem=[4]`, `src_step_elem=[128]` for a 256-byte
  stride) and produced a wrong, permutation-looking answer (`output_0` cosine 0.809). The same
  edit with **four single-element CASTs** (iter-4 `f8`'s proven form) is bit-identical. Either
  the unit is bytes or multi-element strided reads behave differently; I did not resolve which,
  I just used the proven form. **Do not trust the "steps for compute are in elements" row of
  iter-4 §2 for `num_elem > 1`.**
* SBUF is **0.07% utilised** (`prof/SbufUsage.parquet`) and **nothing in the whole NEFF
  touches any offset ≥ 229,400** — so there is free scratch above that, which makes the
  "hunt for a provably dead buffer" of earlier iterations unnecessary. I used 240000 (1 KB at
  p0, interleaved) and 241024 (8 B per partition). **But the window is not the full 0x40000:
  see §9 — it ends between 245,124 and 246,143.**

## 3b. The same cut applied to transpose group 1 — the 7.3 us idle (−2.99 us)

Group 1 is `SP 18/19/20/21`: per-partition bf16 at 218880/218112/217344/216576 → four 256-byte
rows at `p0` 212224/212480/212736/212992, 512 descriptors, executing 38.1 → 42.4 us and the
direct cause of Activation's 7.3 us idle. iter-4 listed it as "multi-hour"; it took one script
because **the gather already exists**:

> iter-4's `f8` inserted four Pool `CAST`s (slots 53–56) that copy exactly those four values
> into **215808/215810/215812/215814 — 8 contiguous bytes per partition** — and Pool slot 56
> already signals `s59 += 16` when the group is done. So steps (1) and the "find a free
> staging buffer" problem of §3 were already solved; only the transfer and the consumers
> needed editing.

Consumers (both have to change, and the second is the one iter-4 thought blocked this):
* `Activation 55–58`: `ACTIVATE func 128`, **in place**, `num_active_channels=1`,
  `num_elem [128,1,1]` → src *and* dst become `BASE + 2c` with `step_elem [4,1,1]`.
* `PE 44/46/55/61`: `LDWEIGHTS src=212478/212734/212990/213246`,
  `step_elem [-1,0,-32]`, `num_elem [32,1,4]` — each reads one row **backwards**.
  The fix is just to scale every stride by the interleave factor:
  `src = BASE + 1016 + 2c`, `step_elem = [-4,0,-128]`, `num_elem` unchanged.
  **A reverse-stride 3-D `LDWEIGHTS` tolerates a 4× stretch** — that is the finding that
  unblocks iter-4's candidate 3.

Gate: `SP 18` must wait for Pool's gather, and a `PSEUDO_DMA_DIRECT2D` has no extended wait,
so I used the spare-ES trick again — **`SP` slots 14 and 16 are bare `EVENT_SEMAPHORE`s with
no wait and no update**, free gate slots. Slot 16 became `W[59]>=32` (= Pool 56 has retired).
It costs 1.06 us of SP delay on its own (`n3g`) and buys 4 us of DGE time.

## 4. Refuted this iteration (do not redo without reading this)

* **Merging the eight bf16→fp32 tile loads of input 2 into 2-D DGE DMAs: wrong answer, and I
  could not find out why.** The eight tiles are a perfectly uniform pattern (`src` +1024,
  `dst` −768: src 0x0/0x400/.../0x1c00 → dst 225280/224512/.../219904), six issued by
  Activation slots 3–8 (sems 82–87) and two by Pool (sems 11,12), and the only waiters on
  sems 80–87 are the Pool guard ESs of §2. Replacing a group with one `num_elem [128,n]`
  transfer (negative step on src *or* dst, addresses taken verbatim from the slot holding the
  group's last tile so no address byte is touched) gives:
  * `n=2` on Activation **slot 8** (the last DGE slot): **bit-identical** (`m5`).
  * `n=2` on slot 4 or slot 6: **wrong** (`m6`, `m7`), cosines 0.9998/0.9990/0.9999.
  * `n=4` on slot 3 or slot 6: **wrong** (`m3`, `m4`), same error for both step directions.
  So the 2-D form itself is fine (and Pool slot 1 has shipped a `[128,8]` one since iter-3) —
  what fails is a merged transfer that is **followed by another DGE DMA on the same engine**.
  Next step if anyone retries: test `n=2` on slot 6 with slots 7 and 8 left as DMAs but
  *reordered* so the merged one is last, to confirm "only the last DGE slot tolerates it".
  `m5` alone measured 55.178 vs 55.267 — inside the noise, so **Activation's prologue issue
  time is not on the critical path** (see §5) and this whole line is low value anyway.
* ~~The descriptor cut is blocked by PE's reverse-stride `LDWEIGHTS`~~ — **wrong, it works;
  see §3b.** Scaling `step_elem` by the interleave factor is enough.
* **`PSEUDO_DMATRIGGER`s cannot be sparsified in the middle of a queue** (§1, `k2` hang).
* **`accumulator_cmd` rules out the iter-4 accumulator hypothesis** (§0) — do not spend an
  experiment on "does `ACTIVATE` accumulate".

## 8. THE FIFTH FINDING: replacing a DMA with a semaphore add can make a *third* wait pass early

`n3` (§3b without one detail) **passed `verify.py` and then hung the profiler** — twice, with
"Failed to schedule execution on XU[1]" and a 30 s model-stop timeout, while `n2` profiled
fine straight afterwards, so the device was healthy and the fault was n3's. The cause, and it
generalises to every "DMA → `SEM_ADD_IMM_COMPLETE`" substitution this loop has made:

> `s68`, `s69`, `s70` each had `{dge:SP 16, SP:EVENT_SEMAPHORE 16}` = 32 total with waiters at
> **both** `>=16` and `>=32`. Turning the DGE transfer into a semaphore add left the totals
> unchanged — `semaphores.py` is happy — but now **the `>=32` threshold is reachable from the
> two event semaphores alone**, so `SP 63/65/67` ("the transpose has landed") passed
> immediately, their `+32` to `s58/s59/s60` fired early, and that in turn satisfied
> `Activation 52`'s `s59>=32` *and my own new gate* before Pool's gather had retired. The
> merged transfer then read un-gathered data, and the resulting garbage hung the run.

**The rule:** when you replace a transfer with a semaphore add, list every waiter on that
semaphore *and every threshold they use*, then make the add wait for whatever still carries
the data guarantee. Here the one-line fix was to give the three `EVENT_SEMAPHORE`s
`WAIT_FOR_SEM_GE_IMM s67 >= 16` — the merged transfer's own completion — instead of their
inherited queue guards (`n4`, −2.99 us). iter-4's §3/f3b note said the same thing about a
single wait carrying several guarantees; the new part is that **the danger also runs through
semaphores you never touched**, via a threshold that used to need the DMA's increment.

Two corollaries worth keeping:
* `semaphores.py` checks supply ≥ demand and so cannot see this at all. The review's standing
  warning about "sems with more increments than max wait" is exactly this hazard; this is the
  first time it actually bit.
* **`verify.py` passed the broken file.** With `--runs 3` it still passes `n4`, but a single
  verify is not evidence. **Profile before believing an edit is safe** — the profiler's extra
  notification traffic perturbs timing enough to expose races that verify misses, which makes
  it the better (if accidental) race detector of the two.

Residual: `n4`'s captures are bimodal (4 × ~51.0, 1 × ~53.4, the outlier always the third
capture of five). It may just be the profiler, but the honest reading is that the 43→51 us
region now has two schedules. Arithmetic is bit-identical in all 65+ executions.

## 9. THE SIXTH FINDING: free SBUF scratch stops working somewhere in 245124..246143

I ran the §3b recipe on the other two transpose groups as well, and both results are useful:

* **Pair A (`SP 34` + `SP 36`, sems 61/62, producers `Activation CAST 48/49`, consumers
  `DVE 92/93 src0`): bit-identical, 51.162 vs 51.029 — neutral, reverted.** Those two complete
  at ~45–46 while the tail's gate is `s71` at ~50.5, so their descriptors are not on the
  critical path. Removing DMA work only pays where the DGE is the *binding* resource.
* **Pair B (`SP 46` + `SP 50`, sems 71/56, producers `Activation CAST 60/63`, consumers
  `DVE 94/113 src0`): bit-identical and a 2.5 us REGRESSION (53.527).** This one looked like
  the best of the three because `s71>=32` *is* the Activation tail's gate — but iter-4's `f6`
  deliberately hoisted this transfer so it could issue as soon as `Act 60` had written
  217088, without waiting for `Act 63`. Merging the pair re-couples it to `Act 63` (~47.5),
  which delays `s71` by more than the 128 saved descriptors buy.
  **The rule: merging two transfers makes both wait for the later producer. Check the
  producers' timestamps before merging, not just the consumers' dataflow.**

And the reason both pair-B attempts first failed verify, which matters for every future edit:

> I had been treating "any offset ≥ 229,400 and < 0x40000" as free scratch because nothing in
> the NEFF uses it and ch. 5.3 says each partition has a 256 KiB window. **That is only true
> up to about 0x3C000.** Pair B at `BASE=246144 / G=247168` gave `output_0` cosine 0.9977 and
> `output_2` 0.9995 — partial corruption, not a fault. The *same script* with
> `BASE=244096 / G=245120` is bit-identical. So the usable top lies between **245,124 and
> 246,143**; writes above it do not land where the address says.

Known-good scratch, all verified bit-identical over many runs: **240000–241031 and
242048–243071 (used by the shipped file) and 244096–245123 (used by `n5`/`n7'`).** Treat
~245,120 as the ceiling, and if a new edit comes out *slightly* wrong with a cosine near 1,
suspect the scratch address before suspecting the logic.

## 5. Where the time is now (51.03 us; trace body ~12.0 → ~65, metric ≈ end − 14.0)

Engine busy on the shipped file: **Activation 39.3 us (77%)**, Pool 23.8, DMA 18.7, DVE 17.9,
PE 7.3, SP 6.7. Over the five iterations DMA busy has gone 52.3 → 18.7 and SP 27.1 → 6.7,
while **Activation has barely moved (41.8 → 39.3) and is now 77% of the wall clock.** The next
big win almost certainly has to come out of Activation's 39.3 us itself, not out of the
plumbing around it: of that, ~9 us is the 7 remaining `ACT_TABLE_LOAD`s (but they sit in
idle windows, §5) and the rest is ~110 real compute ops of 0.2–0.5 us each. Nothing in this
loop has yet tried to move Activation compute onto DVE or Pool at scale (iter-4's review
candidate 2); with Activation at 77% and DVE/Pool at 35%/47%, that is where the headroom is.

Shape of the critical path, measured on `prof3` (the 55.27 state; n2 shifts the 43→51 block
~1.4 us earlier but the structure holds):

| window | us | what binds |
|---|---|---|
| 12.5 → 19.7 | 7.2 | Activation issues its 8 prologue DGE DMAs; **not critical** (see below) |
| 13.1 → 21.1 | 8.0 | **Pool issues 11 prologue DGE DMAs, ~0.7 us each, serially** |
| 18.1 → 23.3 | | PE `LDWEIGHTS`/`MATMUL` → DVE → Activation, gated by Pool's completions |
| 21.1 → 37.0 | 16 | Activation dense head (gaps 1.56 and 0.53) |
| 37.0 → 44.3 | **7.3** | **Activation idle**: SP 18–21's 512 descriptors (38.1→42.4) then a dummy — **`n4` cuts this group to 128 descriptors; re-measure this window first** |
| 44.3 → 55.9 | 11.6 | Activation + the remaining DGE transposes, 1.24 us gap at 49.8 |
| 57.2 → ~65 | 8 | Activation tail, ~95% busy |
| → ~68 | 3 | DVE/PE tail then the one output store and the semaphore range clears |

**Table loads are no longer worth attacking.** 7 remain (`sel 0,1,2,3,4` in the head, `sel 2`
at 59, `sel 4` at 89) and every one of them is followed by an Activation idle at least as long
as the 1.283 us it costs, so deleting one gains ~0. Only 5 are strictly needed
(`f9→T0, f7→T1, f36→T2, f5→T3, f29→T4`; `f30`, `f128`, `CAST`, `COPY` are table-agnostic);
merging the two `sel 2` or the two `sel 4` regions would need the head's `f5`/`f29` ops moved
across data they are not ready for. **This lever is exhausted — say so and move on.**

### Ranked candidates for iter-6

0. **First action: re-profile with `--parquet` and redo §5's table.** Everything below was
   measured at 54–55 us, and `n4` removed 384 descriptors from the 38–42 us window, so the
   37.0 → 44.3 idle and the whole 43 → 51 block have moved. Also settle `n4`'s bimodality
   (§8) — if the 53.4 us outlier is a second schedule rather than a profiler artefact, finding
   it is worth ~2.4 us on its own and is the cheapest thing on this list.
1. ~~The remaining four bf16→fp32 transposes~~ — **done, and both pairs are closed: pair A is
   neutral and pair B regresses 2.5 us (§9).** The descriptor-cut lever is now exhausted on
   this kernel: of the three transpose groups, the two that were on the critical path are cut
   and the third is not worth cutting. `an/n5.py` and `an/n7.py` are kept in case the balance
   changes after candidate 2.
2. **Pool's 11 serial prologue DGE DMAs (13.1 → 21.1 us) are the start-up critical path**, and
   six of them are the 256-byte weight tiles of input 0, which form the *same* uniform pattern
   as §4's: `src 0x100*k → dst 0x37e00 − 0x100*k`, k=0..7, two on Activation (sems 80, 81) and
   six on Pool. Destinations 0x37700..0x37eff do not overlap, so one `num_elem [128,8]` 2-D
   DMA should replace all eight. The obstruction is §4's unexplained "only the last DGE slot"
   behaviour — **settle that first** (the falsifier is in §4) because it unlocks both groups.
   Ceiling: PE's first `LDWEIGHTS` waits on `s8>=16` at 18.1 while Pool is still issuing, so
   ~4 us. Note iter-4's `g8` (permuting these) corrupted the weights; merging is not
   permuting, but verify each group separately.
3. **The 7.3 us Activation idle at 37.0 → 44.3 — largely addressed by `n4`, re-measure.** Its
   proxy gate is a 2-byte dummy (`qSPSpillReload0` block 4, `s28`) while the true gate was
   SP 18–21's completion; with those now one 128-descriptor transfer the dummy may become the
   binding gate, in which case iter-4's `f9` (replacing Activation's `s28` wait with direct
   waits on the transposes) is worth retrying — it was refuted twice, but both times because
   the DGE was saturated, and it no longer is.
4. **Activation's own 8 prologue DGE issues are NOT worth merging** — `m5` removed one and
   measured inside the noise, because Activation's first dependent compute waits on `s6>=3`
   from the DVE/PE chain, which Pool's DMAs gate (candidate 1).
5. **Move Activation compute to DVE or Pool.** Activation is 77% busy and is now the metric;
   DVE is 35% and Pool 47%. This is iter-4-review candidate 2 and no iteration has tried it at
   scale. The ~10 `CAST`s and `COPY`s in Activation's dense 21 → 37 us region are the obvious
   targets (a `CAST` is table-agnostic, so moving one changes no table state); the recipe is
   iter-3's e1/e4 and iter-4's f8, and §8's rule covers the semaphore bookkeeping. Note
   Activation's `s4` is a plain instruction counter, so **an op you move off Activation must
   leave an `EVENT_SEMAPHORE` behind if it carried `SEM_INC_COMPLETE[4]`**, or every later
   `s4>=N` shifts by one.
6. **Do not bother with:** table loads (§5); re-ordering or relaxing waits around the DGE
   transposes (iter-4 §1, confirmed again — `k4` removed 10 DMAs for +0.23 us); the output
   epilogue (now one DMA and ~0.6 us); `dma_configs_dma_transpose_2d_tile` (iter-4 `g1`);
   permuting Pool's prologue DMAs (iter-4 `g8`).

## 6. Reproducing

Scripts take the unpacked root as `argv[1]`; apply in this order from a fresh unpack of the
iter-4 NEFF, then `pack` → `semaphores.py` → `verify.py` → `profile.py --repeats 5`:

```
g7a.py work          # iter-4's enabler: Act 100 writes in place, DVE 106 re-pointed
h2.py  work          # Act 101 in place + DVE 112 re-pointed  (the new half)
mv.py  work 100,101,102 del    # move the three func-36 ops to slot 86, NOP the 2 loads
k2.py  work k1       # merged 1024-byte output store in qSPIO0 block 5, dummies elsewhere
k3.py  work          # move it to block 2 and stop triggering the trailing blocks
k4.py  work          # stop triggering qSPSpillReload0 5-14
n2.py  work          # transpose group 2: 512 -> 128 descriptors
n4.py  work          # transpose group 1: 512 -> 128 descriptors  (= n3 + the §8 fix)
```

`mv.py` takes a comma list so a move can be bisected one op at a time — that is what found
§0. `m3..m7.py` are the refuted tile merges, `n1.py` the refuted strided gather.
The analysis helpers are `d.py` (compact per-engine decode), `full.py` (all fields of chosen
slots, differences only), `addr.py LO HI` (every instruction field and JSON descriptor
touching an SBUF range), `waiters.py sems` (every wait/update on a semaphore set),
`tl.py ENGINE` (per-engine timeline with gaps from `prof*/Instruction.parquet`),
`gaps.py` (gaps > 0.4 us on every engine). `d.py`/`addr.py`/`waiters.py` are the three that
did the real work; the parquet `operands` grep in §0 is the fourth and was decisive.

### an/h2.py — the new half of §0
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
for slot,src in ((100,8602880),(101,8603392)):
    x=isa.decode(bytes(act[slot*64:(slot+1)*64])); a=x.as_dict()
    assert a['activation_func']==36 and a['src_mem_pattern_start_addr_addr_immediate']==src \
       and a['dst_mem_pattern_start_addr_addr_immediate']==(16384 if slot==100 else 17416)
    x.set('dst_mem_pattern_start_addr_addr_immediate',src)   # in place on its own source
    act[slot*64:(slot+1)*64]=x.encode()
open(ap,'wb').write(act)
dp=f'{ROOT}/payload/sg00/DVE0.bin'; dv=bytearray(open(dp,'rb').read())
for slot,fld,old,new in ((106,'src1',16384,8602880),(112,'src1',17416,8603392)):
    z=isa.decode(bytes(dv[slot*64:(slot+1)*64]))
    k=f'{fld}_mem_pattern_start_addr_addr_immediate'
    assert z.as_dict()[k]==old
    z.set(k,new); dv[slot*64:(slot+1)*64]=z.encode()
open(dp,'wb').write(dv)
```

### an/mv.py — move a subset of slots 100/101/102 to just after slot 85
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]; WHICH=[int(x) for x in sys.argv[2].split(',')]; DEL=len(sys.argv)>3 and sys.argv[3]=='del'
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
sl=lambda i: bytes(act[i*64:(i+1)*64])
for i in WHICH:
    a=isa.decode(sl(i)).as_dict()
    assert a['activation_func']==36 and a.get('events_update_mode') in (None,0,'NONE'), i
NOPT=sl(98)
body=b''.join(sl(i) for i in WHICH)
out=bytearray()
for i in [j for j in range(len(act)//64) if j not in WHICH]:
    if i==86: out+=body
    out+=sl(i)
act=out
if DEL:
    NOP=isa.decode(NOPT)
    for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
                ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
                ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
        NOP.set(k,v)
    nop=NOP.encode()
    for i,selv in ((100,2),(106,4)):
        y=isa.decode(bytes(act[i*64:(i+1)*64]))
        assert y.name=='ACTIVATION_TABLE_LOAD' and y['act_tbl_sel']==selv
        act[i*64:(i+1)*64]=nop
open(ap,'wb').write(act)
```

### an/k2.py (MODE 'k1') — merge the four output stores
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
SR={b['id']:b for b in d['dma'] if b['queue']=='qSPSpillReload0'}
IO={b['id']:b for b in d['dma'] if b['queue']=='qSPIO0'}
DUMMY=SR[9]['desc'][0]
IO[5]['desc']=[{"op":"copy","from":"SB","from_dtype":"bfloat16","from_off":8406792,
                "from_sizes":[1024,1],"from_steps":[1,1],
                "to":"output_0","to_dtype":"bfloat16","to_off":0,
                "to_sizes":[1024,1],"to_steps":[1,1]}]
for k in (2,3,4):       IO[k]['desc']=[json.loads(json.dumps(DUMMY))]
for k in (16,17,18,19): SR[k]['desc']=[json.loads(json.dumps(DUMMY))]
json.dump(d, open(p,'w'), indent=1)
sp=f'{ROOT}/payload/sg00/SP0.bin'; s=bytearray(open(sp,'rb').read())
for slot in (81,82,83,84):      # their waits were on the now-dummy staging copies
    x=isa.decode(bytes(s[slot*64:(slot+1)*64]))
    x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
    s[slot*64:(slot+1)*64]=x.encode()
open(sp,'wb').write(s)
```

### an/k3.py — move the store to block 2 and drop the trailing triggers
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
IO={b['id']:b for b in d['dma'] if b['queue']=='qSPIO0'}
assert IO[2]['semaphore']==42
IO[2]['desc']=[{"op":"copy","from":"SB","from_dtype":"bfloat16","from_off":8406792,
                "from_sizes":[1024,1],"from_steps":[1,1],
                "to":"output_0","to_dtype":"bfloat16","to_off":0,
                "to_sizes":[1024,1],"to_steps":[1,1]}]
json.dump(d, open(p,'w'), indent=1)
sp=f'{ROOT}/payload/sg00/SP0.bin'; s=bytearray(open(sp,'rb').read())
ES_T=bytes(s[57*64:58*64])
def es(wait=None):
    x=isa.decode(ES_T)
    for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
                ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
                ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
        x.set(k,v)
    if wait:
        x.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_wait_idx',wait[0])
        x.set('events_semaphore_value',wait[1])
    return x.encode()
s[76*64:77*64]=es((6,63))                                  # keep the data gate
for slot in (77,78,79,80,82,83,84): s[slot*64:(slot+1)*64]=es()
x=isa.decode(bytes(s[81*64:82*64]))                        # slot 81 still triggers block 2
x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
s[81*64:82*64]=x.encode(); open(sp,'wb').write(s)
pp=f'{ROOT}/payload/sg00/Pool0.bin'; po=bytearray(open(pp,'rb').read())
y=isa.decode(bytes(po[71*64:72*64]))                       # final barrier -> block 2's sem
assert y.as_dict()['events_wait_idx']==45
y.set('events_wait_idx',42); po[71*64:72*64]=y.encode(); open(pp,'wb').write(po)
```

### an/k4.py — stop triggering the ten dead dummy blocks
```python
# SP slots 39,40,41,42,51,52,53,54,71,72 trigger qSPSpillReload0 blocks 5..14; every block is
# a 2-byte dummy left by iter-1/f1/f3b/f5 and sems 29-36/33 have no waiter but SP 72 itself.
# Replace all ten triggers with bare EVENT_SEMAPHOREs (template = SP slot 57, all event
# fields cleared).  Blocks 0-4 stay, so the triggered set is still a ring prefix.
```

### an/n2.py — transpose group 2: 512 → 128 descriptors (§3)
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
G, BASE, SEM = 241024, 240000, 46        # nothing in the NEFF uses any offset >= 229400
dp=f'{ROOT}/payload/sg00/DVE0.bin'; dv=bytearray(open(dp,'rb').read())
for slot,off in ((42,18688),(50,18432),(51,18944),(52,19200)):
    assert isa.decode(bytes(dv[slot*64:(slot+1)*64]))['dst_mem_pattern_start_addr_addr_immediate']==off
blob=b''                                  # FOUR single-element CASTs, not one strided CAST
for k,(src,sem) in enumerate(((18432,None),(18688,None),(18944,None),(19200,SEM))):
    g=isa.decode(bytes(dv[50*64:51*64]))
    g.set('events_wait_mode','NONE'); g.set('events_wait_idx',0)
    if sem is None:
        g.set('events_update_mode','NONE'); g.set('events_update_idx',0); g.set('events_semaphore_value',0)
    else:
        g.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); g.set('events_update_idx',sem)
        g.set('events_semaphore_value',16)
    g.set('num_active_channels',128); g.set('in_dtype','BFLOAT16'); g.set('out_dtype','BFLOAT16')
    g.set('src_mem_pattern_start_addr_addr_immediate',src)
    g.set('src_mem_pattern_num_elem',[1,1,1,1]); g.set('src_mem_pattern_step_elem',[1,1,1,1])
    g.set('dst_mem_pattern_start_addr_addr_immediate',G+2*k)
    g.set('dst_mem_pattern_num_elem',[1,1,1,1]); g.set('dst_mem_pattern_step_elem',[1,1,1,1])
    blob+=g.encode()
dv[53*64:53*64]=blob
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
for slot,old,c in ((74,218368,0),(75,217600,1),(76,216832,2),(77,216064,3)):
    x=isa.decode(bytes(act[slot*64:(slot+1)*64])); a=x.as_dict()
    assert x.name=='CAST' and a['src_mem_pattern_start_addr_addr_immediate']==old \
       and a['num_active_channels']==1 and a['src_mem_pattern_num_elem']==[128,1,1,1]
    x.set('src_mem_pattern_start_addr_addr_immediate',BASE+2*c)
    x.set('src_mem_pattern_step_elem',[4,1,1,1])
    act[slot*64:(slot+1)*64]=x.encode()
x=isa.decode(bytes(act[72*64:73*64]))        # only s58>=32 still means "the bytes are there"
assert x.as_dict()['events_wait_idx']==58 and x.as_dict()['events_extended_wait_idx']==60
x.set('events_extended_wait_mode','NONE'); x.set('events_extended_wait_idx',0)
x.set('events_extended_sem_wait_value',0)
act[72*64:73*64]=x.encode()
y=isa.decode(bytes(act[73*64:74*64]))        # Activation is in order, so this gate is spare
assert y.as_dict()['events_wait_idx']==63 and y.as_dict()['events_extended_wait_idx']==64
for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
            ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
            ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
    y.set(k,v)
act[73*64:74*64]=y.encode()
open(ap,'wb').write(act); open(dp,'wb').write(dv)
sp=f'{ROOT}/payload/sg00/SP0.bin'; s=bytearray(open(sp,'rb').read())
x=isa.decode(bytes(s[27*64:28*64])); a=x.as_dict()
assert a['semaphore']==58 and a['src_start_addr_addr_tbl_offs_offset_immediate']==18432 \
   and a['dst_start_addr_addr_tbl_offs_offset_immediate']==218368 and a['src_elem_size']==2
x.set('src_start_addr_addr_tbl_offs_offset_immediate',G); x.set('src_elem_size',8)
x.set('src_step_elem',[262144,1]); x.set('src_num_elem',[128,1])
x.set('dst_start_addr_addr_tbl_offs_offset_immediate',BASE); x.set('dst_elem_size',8)
x.set('dst_step_elem',[8,1]); x.set('dst_num_elem',[128,1])
s[27*64:28*64]=x.encode()
ES_T=bytes(s[57*64:58*64])
g23=isa.decode(bytes(s[23*64:24*64]))        # spare gate: slot 24 repeats slot 23's wait
assert g23.name=='EVENT_SEMAPHORE' and g23.as_dict()['events_wait_idx']==74 \
   and g23.as_dict()['events_update_mode'] in (None,0,'NONE')
assert isa.decode(bytes(s[24*64:25*64])).as_dict()['events_wait_idx']==74
g23.set('events_wait_idx',SEM); g23.set('events_semaphore_value',16)
s[23*64:24*64]=g23.encode()
for slot,sem in ((32,60),(37,63),(38,64)):
    a=isa.decode(bytes(s[slot*64:(slot+1)*64])).as_dict()
    assert a['semaphore']==sem and a['events_wait_idx']==sem and a['events_semaphore_value']==16
    e=isa.decode(ES_T)
    e.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); e.set('events_wait_idx',sem)
    e.set('events_semaphore_value',16)
    e.set('events_extended_wait_mode','NONE'); e.set('events_extended_wait_idx',0)
    e.set('events_extended_sem_wait_value',0)
    e.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); e.set('events_update_idx',sem)
    s[slot*64:(slot+1)*64]=e.encode()
open(sp,'wb').write(s)
```

### an/n4.py — transpose group 1 (§3b + §8)

Identical to `an/n3.py` in §6's listing except that the three `EVENT_SEMAPHORE`s replacing
SP 19/20/21 wait on the merged transfer instead of inheriting their own queue guards:

```python
for slot,sem in ((19,68),(20,69),(21,70)):
    a=isa.decode(bytes(s[slot*64:(slot+1)*64])).as_dict()
    assert a['semaphore']==sem and a['events_semaphore_value']==16
    e=isa.decode(ES_T)
    # MUST be s67 (the merged transfer), not a['events_wait_idx']: otherwise s68/69/70 reach
    # their 32-threshold from two event semaphores alone, SP 63/65/67 pass early and their
    # +32 to s58/59/60 lets Activation 52 and the slot-16 gate run before Pool's gather. §8
    e.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); e.set('events_wait_idx',67)
    e.set('events_semaphore_value',16)
    e.set('events_extended_wait_mode','NONE'); e.set('events_extended_wait_idx',0)
    e.set('events_extended_sem_wait_value',0)
    e.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); e.set('events_update_idx',sem)
    s[slot*64:(slot+1)*64]=e.encode()
```

The rest of `n4` = `n3`: Pool 53–56's existing gather asserted, `SP 16` → `W[59]>=32`,
`SP 18` rewritten to `src 215808 / elem 8 / step [262144,1]` → `dst 242048 / elem 8 /
step [8,1]`, `SP 28` no-op'd (SP 29's extended `W[67]>=16` is the real gate),
`Activation 55–58` src+dst → `242048 + 2c` with `step [4,1,1]`, `PE 44/46/55/61`
src → `242048 + 1016 + 2c` with `step_elem [-4,0,-128]`.

### an/n5.py and an/n7.py — the refuted pair merges of §9

Both follow §3b exactly. `n5` (pair A, neutral): two single-element `CAST`s inserted at
Activation slot 50 pack 218624/217856 into `G+0/G+2`; `SP 30` (a bare spare ES) becomes the
gate on the gather's semaphore 47; `SP 34` becomes `src G / elem 4 / step [262144,1]` ->
`dst BASE / elem 8 / step [8,1] / num [128,1]`; `SP 36` becomes an ES waiting `W[61]>=16`
(s61's first 16 is the DMA's here, so `>=16` is correct) and adding 16 to s62;
`DVE 92/93` `src0` -> `BASE + 4c` with `src0_mem_pattern_step_elem [2,1,1]`.

`n7` (pair B, +2.5 us): same with the gather at Activation slot 64 over 217088/216320,
`SP 42` as the gate, `SP 46` as the merged transfer, and - the part that differs - the
`s56 += 16` split across **two** SP slots because s71's first 16 comes from `SP 22`'s event
semaphore, so the real gate is `s71 >= 32` and `events_semaphore_value` cannot be both the
threshold 32 and the amount 16:

```python
w=isa.decode(bytes(s[49*64:50*64]))        # SP 49's W[66]>=32 is dead after the merge
w.set('events_wait_idx',71); w.set('events_semaphore_value',32)
s[49*64:50*64]=w.encode()
e=isa.decode(ES_T)                          # SP 50: no wait, just the +16
e.set('events_wait_mode','NONE'); e.set('events_wait_idx',0)
e.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); e.set('events_update_idx',56)
e.set('events_semaphore_value',16)
s[50*64:51*64]=e.encode()
```

Both use `BASE=244096 / G=245120`. With `BASE=246144 / G=247168` the identical script fails
verify - that is how §9 was found.

## 7. Final state

Changed members vs the iter-4 file: `sg00/Activation0.bin` (127 slots, same length),
`sg00/DVE0.bin` (115 → **119**), `sg00/PE0.bin` (4 slots), `sg00/Pool0.bin` (one wait),
`sg00/SP0.bin` (~30 slots), `sg00/SP0.json` (`qSPIO0` block 2). `neff_diff.py` vs the
*original* baseline reports 6 of 46 members changed and no member added or removed.
**Free SBUF scratch used: `p0` 240000–241023 and 242048–243071, plus 241024–241031 per
partition.** Those are above the 0x37FFF that every compiler-emitted offset stays below, but
well inside the usable top established in §9 (~245,120) and they round-trip bit-identically
over 100+ executions. **Do not go above ~245,120** — §9 shows 246,144 silently corrupts.

Check order after every edit: `semaphores.py` → `pack` → `verify.py` → `profile.py
--repeats 5`. **Three hangs this iteration** (§1 `k2`, §2 `m1`, §8 `n3`) — each cost ~90 s of
device timeout, `semaphores.py` was clean for all three, and `verify.py` *passed* the third.
So: `semaphores.py` is not a hang detector and **neither is a single `verify.py` run** — only
the profiler caught `n3`. Always profile.

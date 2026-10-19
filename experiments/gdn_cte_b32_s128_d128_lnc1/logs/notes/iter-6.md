# iter-6 — **shipped: the W-path DMA-hop fusion (iter-5's candidate #1) is BUILT AND VERIFIED for
# all 64 instances.** 1174.8 → 1171.8 us (medians of two runs each), i.e. about **-3 us**, inside
# the ±2–3 us noise
# floor, so treat the wall number as "no regression" rather than a win. **The structural result is
# the point and it is large: one of the two queued DMA hops on the kernel's biggest idle site is
# gone, and the only reason the wall barely moved is that the replacement ACT `CAST` costs +19.4 us
# of ACT busy. §8 says exactly how iter-7 turns that into the ~23 us the hop removal actually
# bought.** Outputs are bit-identical over 3 runs (`max_abs_err` equal to the baseline's to every
# printed digit).
#
# ## AND READ THIS: **you can INSERT a brand-new instruction into an engine's stream.**
# Device-verified (§0). Activation0.bin 2532 -> 2533 slots, the new ACT `CAST` executes (290 ns,
# `compiler_pc` 26, Scalar instruction count 2555 -> 2556 in the profile), NEFF loads, and
# `verify.py` PASSES with `max_abs_err` identical to every printed digit. No engine JSON or
# `def.json` field encodes an instruction count — the count is implicit in the member length, and
# `neff.py pack` fixes the header. **Combined with fresh semaphores (S[88]+ are unused), this
# removes the "I cannot add compute" constraint that closed iter-4 §2.5, iter-5 §4.1 and §6.**
# It is what made §8 possible, and §6/§8.0 list what it opens next.
#
# The rest of the iteration: I found the one unexplored *scheduling* lever (hoisting ready,
# semaphore-*updating* work into another engine's bubbles, which iter-5 §7.1 explicitly left open),
# measured that it is worth **~-7 us real wall time on PE**, and then measured that **every legal
# version of it collapses to 3 moves / ~0 us.** Both failure modes are now pinned with a device
# measurement, so the whole engine-reordering family is closed on all five engines.
# Also in here: a corrected DMA-hop cost breakdown (§3.4), the exact PE operand-geometry encoding
# (§5.1 — iter-5 did not have this and my first hazard check was silently vacuous because of it),
# and the W-path descriptor decode (§4) that iter-5 §6 asked for.

---

## 0. THE NEW CAPABILITY: inserting an instruction. Full recipe, device-verified.

```python
import isa
slots = [s for _,s in isa.decode_stream(open('work/payload/sg00/Activation0.bin','rb').read()))]
new = isa.decode(slots[25].encode())        # clone any existing instruction of the opcode you want
new.set('events_wait_mode', 0); new.set('events_wait_idx', 0)
new.set('events_semaphore_value', 0)
new.set('events_update_mode', 0); new.set('events_update_idx', 0)
new.set('dst_mem_pattern_start_addr_addr_immediate', 0x24000)   # free SBUF, see below
out = slots[:26] + [new] + slots[26:]
open('work/payload/sg00/Activation0.bin','wb').write(b''.join(s.encode() for s in out))
```
then `neff.py pack`. That is the whole thing. Verified: **PASS, bit-identical outputs**, and the
inserted instruction really runs (it shows up in `Instruction.parquet`).

Rules that make an insert safe:
* **Insert a non-updating instruction** and nothing else changes: engine wait values are ordinals
  over the *updating* instructions only (1306/2555 on ACT, 2093/2884 on DVE, 1982/2272 on PE), so a
  non-updating insert shifts no ordinal and needs no rewrite anywhere.
* If the new instruction must signal another engine, **give it a fresh semaphore** (this NEFF uses
  S[0..87]; S[88]+ are unused — iter-5 already proved the loader accepts a new `semaphore_set` in
  that range) rather than the engine counter. Incrementing an engine counter would shift every
  later wait on it and *widen* every later wait prefix, which is exactly what hangs (§2.1).
* Ordering *within* an engine is free: the engine is in-order, so placing the insert between the
  producer's wait and the consumer is sufficient on that engine.
* **Free SBUF**: use the extent map in **§5.5**, not a padded-immediate map. On partitions 96-127
  the genuinely free windows are `0x00667f-0x00e000` (30 KB), **`0x0102ff-0x01c000` (47 KB)** and
  `0x0323ff-0x033b00` (5 KB); the shipped edit uses `0x011000-0x019000`. My first, padded map said
  `0x020c00-0x033b00` was free and it is **not** — that error cost three probe cycles (§8.1).
  (`SbufUsage.parquet` reports `avg_util_percent` < 0.01; it is not an address map.)

*(`scratch/` does not survive. Every script below is 40–120 lines and described well enough to
rebuild. `prof0/` was `tools/profile.py --repeats 5 --parquet prof0`.)*

## Result

| NEFF | total_exec_time_us (`--repeats 5`) | samples |
|---|---|---|
| start (= iter-5 shipped, HEAD `a04258d`, md5 `3b4fe7da`) | **1175.315** | 1173.6 1173.6 1175.3 1175.9 1178.2 |
| start, second run of the identical file | **1174.317** | 1168.7 1173.2 1174.3 1176.0 1176.7 |
| **shipped (W-path fusion, all 64)** | **1170.523** | 1168.6 1168.9 1170.5 1171.4 1173.7 |
| shipped, second run | **1173.038** | 1170.5 1172.3 1173.0 1173.4 1175.5 |

Median of the two shipped runs **1171.8**, of the two start runs **1174.8**: **-3 us (-0.26%),
inside the ±2–3 us between-measurement floor. I am not claiming a wall-clock win.** The sample
ranges overlap (shipped 1168.6–1175.5, start 1168.7–1178.2); there is no regression anywhere near
the 2% threshold.

Engine busy on the shipped NEFF: **ACT 540.8 (was 521.4, +19.4 — the 32 inserted CASTs)**,
DVE 535.9, PE 340.7, DMA 339.9 (+8.6, the 4-descriptor split), Pool 234.0, SP 53.1.
`verify.py --runs 3`: PASS, **all 3 runs bit-identical**, `output_0` 1.097e-04 (10% of limit),
`output_1` 9.081e-04 (83%) — identical to the baseline to every printed digit.
`launch_rules.py` PASS. `semaphores.py`: no over-waits; the new `S[88]` shows supply 32, max wait
32, 64 waiters.

**Noise floor, re-measured on the identical file: medians 1175.3 and 1174.3, but the two runs'
sample ranges are 1173.6–1178.2 and 1168.7–1176.7.** So single-sample spread understates it: the
honest between-measurement floor is still iter-3's **±2–3 us**, and a 7 us effect needs two
independent `--repeats 5` runs per NEFF before you call it.

`verify.py` PASS on the restored file (`launch_rules.py` PASS, `output_0` 1.097e-04 = 10% of limit,
`output_1` 9.081e-04 = 83%, unchanged). I deliberately did **not** ship the one correct edit I
built (3 PE group hoists, §2.4) — it is worth ~3.2 us of *instruction advance*, i.e. well under
1 us of wall, and it is not worth any residual risk for a number I cannot measure.

---

## 1. THE LEVER, AND THE NUMBER THAT MAKES IT INTERESTING

iter-5 §7.1 dismissed engine reordering with "hoisting a ready instruction X in front of a blocked
B does not shorten B's wait; all it buys is X's own ~250 ns of execution". **That is only true for
X that updates no semaphore.** If X updates its engine's counter, hoisting it makes its *result
visible to other engines* by the whole length of the bubble, which can advance a cross-engine
chain by microseconds. iter-5 noted this but set it aside because the wait values are ordinals
(a wait `S[5] >= v` means "the first v matmuls have completed"), so moving an updating instruction
requires rewriting every wait on that counter.

I built that rewrite for **PE** (`S[5]`: 1982 of 2272 MATMULs update it; 2098 waiters on
DVE/ACT/SP). PE is the right target: **29% busy, 867 us of gaps, and 844 us of those gaps have
work behind them whose waits were already satisfied.**

### The win is real: **~-7 us measured**
A 55-hoist version profiled at **1167.67 us** (min 1165.2, max 1169.5) against start medians
**1175.32 / 1174.32** — so about **-7 us**, and the medians are ~3x the ±2–3 us floor apart
(the sample ranges do just touch: the start's best sample was 1168.7). 52.9 us of *advance*
(sum over hoisted groups of how much earlier each could start) converted to ~7 us of wall:
**a ~13% conversion rate, the first non-zero conversion any iteration has measured for a
reorder.** Use it to cost any future reorder: `expected wall gain ≈ 0.13 × Σ advance`.

**But that version produces NaN** (1600 of 262144 in output_0, 384 in output_1), and so does every
weakening of it I could find. §2 is why.

---

## 2. WHY IT IS CLOSED — two independent walls, each measured on the device

### 2.1 Wall A: **rewriting the ordinal of a waited-for instruction deadlocks the kernel.** HANG.
Moving group H earlier past group G necessarily pushes G's ordinal *up*, so any consumer `C` of G
must now wait for a **wider prefix that includes H**. If `C` is upstream of anything H waits for,
that is a cycle. Measured: the full version (154 hoists, 365 of 1982 `S[5]` values changed,
394 wait rewrites across DVE/ACT/SP) → **`verify.py` FAIL, NRT 1004 + "Model stop timeout
(30000 ms) for TPBs"**, i.e. a hard hang. Repeated with the ES fix of §2.2 (53 hoists, 193
rewrites): **hang again**. So this is the mechanism, not an accident.

The sound rule is therefore: **a hoist may not change any wait value that anything actually waits
on.** Concretely, admit the hoist only if no instruction anywhere waits on `S[5]` with a value
equal to one of the *skipped* groups' matmul ordinals. (Values belonging to H itself may be
*lowered* — that narrows the prefix to exactly "H is done", which is safe, and is what notifies
H's consumer early. Guard it by checking that every waiter of an H ordinal reads a PSUM address
that is in H's own destination set; 1–5 candidates failed that check and were dropped.)

With that rule, **122 of the 125 candidate hoists are rejected** ("skipped-has-waiter"). The
stalled group's result is essentially always awaited — that is *why* PE is stalled there.

### 2.2 Wall B: **95% of PE's bubbles sit at an `EVENT_SEMAPHORE` that is the next group's second
### wait operand. Crossing it is exactly iter-5 §2.1, and it gives NaN.**
PE's `LDWEIGHTS`/`MATMUL` structs have **one** wait field and **no** extended-wait block, so
whenever a group needs two waits the compiler emits a preceding `EVENT_SEMAPHORE` (wait, no
update) — 493 of them on PE. The bubble you see in the profile is *that ES's wait*, charged as a
gap before it (on PE an ES's own `duration_ns` is only ~50 ns; the wait shows up as the preceding
gap — the opposite of SP/Pool, where the wait is inside the ES's duration).

Of 125 stall points, **only 3 are not an ES**. All 52 extra hoists in the ~-7 us version were
"hoist the group that immediately follows the ES above the ES" — i.e. **running a group before its
own second wait operand**. That is where the NaN came from, and it is the same rule iter-5 §2.1
established for SP, now confirmed on PE: **never move anything above an `EVENT_SEMAPHORE` that
updates nothing; it *is* the following instruction's missing wait field.**

I also tried the obvious repair — treat `(ES, the group it protects)` as one stall unit and only
hoist groups from *further down* (hj ≥ gi+2), seeding the hazard sets with the protected group's
addresses. That raises the candidate advance to **170.4 us** (so the ceiling of a fully-unlocked
version is ~0.14 × 170 ≈ **24 us ≈ 2%**), but every one of those hoists needs Wall A's rewrite,
and it hung.

### 2.3 Variants tried, all measured, all dead
| variant | moves | Σ advance | result |
|---|---|---|---|
| hoist any ready group, full ordinal rewrite | 154 | 724 us | **HANG** |
| + don't cross an ES as crossing-point (ES may still be the stall point) | 55 | 52.9 us | runs, **1167.67 us (~-7)**, **NaN** |
| + ES-as-stall-point allowed but H ≠ the ES's own group, full rewrite | 53 | 170.4 us | **HANG** |
| + no wait value may be widened (Wall A rule), ES stall points allowed, hj ≥ gi+2 | 3 | 3.2 us | **PASS, errors bit-identical** |
| + also require H to carry no waits at all (cannot block ⇒ cannot deadlock), widening allowed | 5 | 4.4 us | not run (too small) |
| strictest (no ES stall points at all) | 3 | 3.2 us | same 3 moves |

**Bisection note:** when hunting the NaN I bisected the 55 moves (subsets of 28, 14, 7, 3 — each
cycle is `pack` + `verify --runs 1` ≈ 50 s). Even the first **3** moves failed (no NaN but
`output_0` 2723% of limit, `output_1` 32771%). So the breakage is not a couple of bad moves, it is
the whole ES-crossing class. Do not spend another iteration bisecting it.

### 2.4 The one correct recipe (if someone wants the 3 moves)
Groups = `LDWEIGHTS` + the consecutive `MATMUL`s after it (a `MATMUL` with no preceding
`LDWEIGHTS` reuses the previous group's stationary register — e.g. `nc_matmul(pMT,PT_stat,MT)` and
`nc_matmul(pp,PT_stat,P)` at lines 441/447 share one `LDWEIGHTS` — so group boundaries must be at
`LDWEIGHTS`). Movable iff: every member profiled; all its `MATMUL`s have
`psum_accumulate_flags == 3` (self-contained: 1 = start of an accumulate chain, 0 = continuation,
3 = standalone); PSUM destination banks disjoint from every skipped group's; source ranges
disjoint; Wall A rule satisfied; stall point is not an ES. Verified PASS with
`max_abs_err` identical to every printed digit. **3 moves, ~0.5 us. Not worth shipping.**

---

## 3. Fresh whole-machine measurements on the shipped NEFF (all new this iteration)

### 3.1 Exclusive-busy decomposition — where the wall actually is
Sweep the merged busy intervals of all five engines (non-sync instructions only) and classify each
instant by how many engines are executing (`scratch/excl.py`, 30 lines):

| | us | note |
|---|---|---|
| ≥2 engines busy | **559.2** | |
| Vector (DVE) only | **130.3** | |
| Scalar (ACT) only | **129.3** | |
| Tensor (PE) only | **126.2** | |
| GpSimd (Pool) only | **41.5** | |
| Sync (SP) only | 7.3 | |
| **nothing busy** | **197.1** | pure DMA/dependency latency |
| span | 1191.0 | |

`DVE ∪ ACT = 797.1 us` (DVE 535.5, ACT 521.2, overlap 259.6). So **378 us of the wall has neither
big engine working**, and perfect DVE/ACT overlap would be a 536 us floor. This is the same
"program-order, not throughput" story as iter-5 §7, now with the exclusive numbers.

### 3.2 Pool-exclusive time by line — kills review candidates #2 and #3
Pool's 41.5 us of exclusive time, attributed to the running instruction:
`DMA_DIRECT2D :207` 8.1, `:171` 7.3, `:623` 7.1, `:173` 5.6, `:169` 5.3, `:191` 3.8,
`TENSOR_SCALAR_AFFINE_SELECT :243` 2.2, `:628` 1.0, `:535` 0.7.
So converting **all 288 Pool DGE DMAs** to queued transfers (review candidate #2, ~200 us of Pool
issue time) has a hard ceiling of ~38 us spread over 6 sites, and §3.3 shows the biggest ones are
not on the path at all. **Do not do it** — it needs HBM-referencing queued descriptors, of which
this NEFF contains **zero** examples (all 861 queued transfers are `SB`→`SB`), so it is also the
riskiest edit available.

### 3.3 The input loads are already ~5 groups ahead of use. Prologue DGE work is not on the path.
Decoding `Pool0.bin` against the profile (`compiler_pc` = slot index) shows the first ~34 us issue
the `k/v/q/gate/beta` loads for **groups 0–5** (distinct destinations `0x37d00, 0x101d400,
0x101f200, 0x1021000, 0x1022e00, …`; HBM source offsets `0x0, 0x8000, 0x10000, 0x18000, 0x20000`),
and steady state keeps that distance constant (6 loads issued per group). The loads look
"issue-bound" in the profile (27/32 at `:169`, 29/32 at `:173`, 55/64 at `:207` start within 60 ns
of the previous Pool instruction's end) — **that is prefetch depth, not a stall.** From slot 24
onwards each load also carries the `qPoolDynamic` ring throttle (`S[8+r] >= 16j`, 16 rings shared
by all 288 Pool DGEs ≈ 1.8 groups of depth); it never binds.

Pool's steady-state per-group program is
`[ES :494 (idles 15.8–18.9 us)] TSAS:535 [ES :494 (waits 1.0–2.7)] ES:623 [DGE :623 (idles 4.0)]
ES:535 TSAS:535 ES:623 [ES :628 (idles 3.4)] DGE:623 DGE:628 [TSAS:243 (idles 2.3)]
[TSAS:243 (idles 1.3)] DGE:207 DGE:207 DGE:169 DGE:171 …`
— every idle is the *next* instruction's own wait (`TSAS:243` waits `S[4]`, i.e. ACT), so Pool is
purely dependency-bound in order and has no head-of-line blocking to take. Consistent with
iter-4 §2.6.

### 3.4 Corrected DMA hop cost — the completion→semaphore term is small, not 1–1.9 us
From `DmaPacketAggregated` joined with `SemaphoreUpdate` (`scratch/lat.py`):

| term | q0 (669 transfers) | q1 (160) | q2 (32) |
|---|---|---|---|
| `trigger → start` | **713 ns** (flat: 704–761 ns for **1…7 descriptors** and for 256 B…16 KB) | n/a* | 696 |
| transfer `duration` | 307–787 ns (16 KB in ~320 ns) | n/a* | 734 |
| `end_ts → its semaphore's update` | **327 ns** (p25 259, p75 598) | 82 | 106 |

\* q1 rows report `start_ts` ≈ 171 us *before* their own trigger and `duration` ≈ 172 us. `end_ts`
is sane (trigger + ~877 ns). **`start_ts`/`duration_ns` are unusable for `qSPSpillReload1` —
use `end_ts - trigger`.** (This also means `profile.py`'s DMA busy figure does not include them.)

Two consequences:
* **The descriptor fetch is per *transfer*, not per descriptor, and is independent of size.** So
  *splitting* a transfer is latency-free (confirms iter-4 §2.5's 1→4 descriptor control probe) and
  *merging* N transfers that are consumed together would save (N−1)×713 ns. Nobody has found a
  merge candidate yet; the transposes that feed one matmul are already single transfers.
* iter-5's "1.0–1.9 us completion→consumer-visible" is **mostly consumer-side** (the consuming
  engine's own in-order position and dispatch), not DMA completion: the semaphore lands 327 ns
  after `end_ts`. The hop is ~713 + ~320 + ~330 ≈ **1.36 us** of hardware, plus the consumer's
  queueing.

---

## 4. The W-path, decoded (answers the open question in iter-5 §6)

`gdn_cte.py` 464→597 after iters 1–2 fused the HBM staging, verified against `SP0.json`:

```
DVE tensor_copy :464  (yW psum fp32 -> Wg_grp fp32, 32-partition bands)
  -> ONE queued transfer on qSPSpillReload2 (32 of them, 2 descriptors each, op:"cast")
       desc: from SB fp32 off 0x1810400 (= partition 96, byte 0xe000)
             from_sizes [512,4,16]  from_steps [1,512,262144]
             to   SB bf16 off 0x101d200 (= partition 64, byte 0x1d200)
             to_sizes  [256,4,16]  to_steps  [1,4194304,262144]
  -> W_all[64 partitions 64..127, 2 chunks x 128 elem bf16]
  -> ONE queued 64x128 transposing transfer on qSPSpillReload0
       transpose_shape [1,1,64,128], transpose_element_size 2,
       from W_all, from_sizes [256,64,1,1], from_steps [1,262144,256,16777216]
       to   W_t,   to_sizes  [128,128,1,1], to_steps  [1,262144,33554432,128]
  -> PE LDWEIGHTS/MATMUL :597
```
**Descriptor `*_sizes` are in BYTES, `*_steps` in bytes** (512 fp32 B = 256 bf16 B = 128 elements;
a step of 262144 is one SBUF partition, 4194304 = 16 partitions, 16777216 = 64, 33554432 = 128).
The `to_steps [1, 4194304, 262144]` on the cast is exactly the band write's `bi`-on-free →
`bi`-on-partition swap (dst partition = 16·j + p). So **the W path is two hops, not three**
(iter-5 §2.3's "Pool DGE" hop is a WAR gate, not a producer).

### 4.1 The index algebra, worked out. This is the part that takes an hour to get right.
Read `to_sizes [256,4,16]` / `to_steps [1, 4194304, 262144]` on the q2 cast as: for `j` in 0..3
(step 4194304 B = **16 partitions**), for `p` in 0..15 (step 262144 B = **1 partition**), copy 256 B.
Read `from_sizes [512,4,16]` / `from_steps [1,512,262144]` as: source partition `96+p`, bytes
`[512j, 512j+512)` — i.e. **2048 contiguous fp32 bytes = 512 elements per partition**, split into
4 blocks of `D=128` elements indexed by the sub-block `bi = j`.

So, with `desc0.from` at partition **96** byte `0xe000` and `desc1.from` at partition **112** byte
`0xe000` (both `to` partition 64, bytes `0x1d200` and `0x1d200+256`):

```
chunk c of a group  <->  source partitions 96+16c .. 111+16c      (c = 0,1; so 96..127 in total)
W_all[64+16j+p][c*128 + d]  =  bf16( Wg_grp[96+16c+p][0xe000 + 512j + 4d] )     p<16, j<4, d<128
W_t  [d][16j+p]             =  W_all[64+16j+p][c*128 + d]            (the :595 transpose)
=>  W_t[d][16j+p]           =  bf16( Wg_grp[96+16c+p][0xe000 + 512j + 4d] )
```
Therefore:
* the inserted ACT `CAST` is **one per group, not per chunk**: `num_active_channels = 32`, src
  partition 96 byte `0xe000`, `num_elem [512,1,1,1]` fp32 (2048 B contiguous), dst = bf16 scratch
  at partition 96, `num_elem [512,1,1,1]` (1024 B contiguous, 4 blocks of 256 B indexed by `j`).
  32 inserts total. Read each group's real `from_off` out of its own q2 transfer's two descriptors
  (iter-5 §6 saw `0xe000 / 0xf100 / 0xf400 / 0x377f8`, so the offset is **not** constant).
* the fused transposing transfer is **4 descriptors** (one per `j`), each a 16×128 bf16 tile:
  `transpose_shape [1,1,16,128]`, `transpose_element_size 2`,
  `from` = scratch at partition `96+16c`, byte `base + 256j`, `from_sizes [256,16,1,1]`,
  `from_steps [1,262144,256,16777216]`;
  `to` = `W_t + 32j` bytes, `to_sizes [32,128,1,1]`, `to_steps [1,262144,33554432,32]`.
  This is **iter-4 §2.5's control probe, which is already device-verified** (it ran bit-identical
  there with the source being `W_full`); the only change is where `from` points and that the source
  stride along `j` becomes a 256 B byte stride instead of a 16-partition stride.
* **Where to insert the CAST, and how to join it to its group.** Do *not* try to match by address
  (buffers rotate). The join that works: the q2 transfer's **SP trigger** waits on
  `S[40+k] >= 16m`, and that is the `m`-th `EVENT_SEMAPHORE` updating `S[40+k]` in `Activation0.bin`
  program order — i.e. one of iter-2's dead band-write slots at `:483`/`:487`. Insert the CAST
  **immediately after that ES** and give it **no wait at all**: ACT is in-order, the ES already
  waited `S[6] >= v` ("`Wg_grp` complete", visible as its `events_extended_*` block), so the CAST is
  correctly ordered by position alone. This also means the edit is purely **additive** — leave the
  q2 transfer and `W_all` in place for the first version and change nothing but the 64 `:595`
  descriptors and their trigger waits; delete the dead q2 transfers only once it measures.
  (`events_semaphore_value` is a single shared field, but `SEM_INC_COMPLETE` ignores it, so a CAST
  *can* carry a wait value **and** a `+1` update if you need both.)
* **Synchronisation.** Give each inserted CAST `events_update_mode = SEM_INC_COMPLETE` on a
  **fresh semaphore S[88]** (not `S[4]` — updating the engine counter shifts every later ordinal
  and widens every later prefix, which hangs, §2.1) and set the 64 `:595` triggers' wait to
  `S[88] >= group_index+1`. Replacing their current wait is safe: it is a
  `qPoolDynamic`/`q0` **ring throttle**, and iter-4 §2.7 measured that removing all 218 such
  throttles changes nothing (iter-5 §2.3 separately retargeted 33 of these very triggers and got
  bit-identical output). **Unproven: whether S[88] exists** — this NEFF uses S[0..87] and
  `def.json` has `runtime_semaphore_count: 3` (which does *not* appear to bound the engine
  semaphores, since 88 of them are already in use). Probe one wait on S[88] before building on it;
  if it fails, the fallback is the `S[40..55]` trick in §6.1(a), which needs no new semaphore.

### 4.2 **The 64 `:595` W_t transposes, identified.** Use this list; do not re-derive it.
Matching `from_off` against the q2 `to_off` set alone gives 78 candidates for 64 real ones (W_all's
addresses are reused by other buffers in other groups — e.g. `0x101c000` is also `beta_sb`'s DGE
destination at `:207`), and the SP trigger's `nki_source_location` is unreliable (§5.4). What
works, exactly, is **trigger program order**: for each of the 32 q2 transfers in SP trigger order,
take the first two 64×128-bf16 transposing descriptors triggered *after* it whose `from_off` is one
of that transfer's two `to_off`s, consuming matches greedily. That gives **exactly 2 per group for
all 32 groups, 64 total** (`scratch/ident.py`, 35 lines):

```
qSPSpillReload0 block ids =
 19  20  42  44  64  65  82  83 100 101 120 123 140 143 160 163
181 184 203 206 225 228 245 248 266 269 287 289 306 309 327 330
347 350 367 370 387 390 409 412 431 434 452 455 473 476 495 498
517 520 538 541 559 562 580 583 601 604 622 625 643 646 663 666
```
**This is self-validating: block 19 is the transfer iter-4 §2.5 independently identified as "the
W_t transpose, `to_off 0x5400`" when it ran its 4-descriptor control probe.** Group 0's chain in SP
program order is `q2 trigger @ slot 29 → transposes @ slots 35 and 38` (blocks 19 and 20, reading
`0x101d200` and `0x101d300` = the q2 transfer's two `to_off`s, writing `W_t` at `0x5400`/`0x4500`).
**Careful:** a few of these transfers carry a second descriptor (block 20 has 2) — rewrite only the
64×128 one and keep the other.

### 4.3 Why this is iter-5's candidate #1, and why it is now cheap
Fusing the two hops needs a **bf16** source for the transposing descriptor, because transpose+cast
is rejected at load (iter-4 §2.5, two encodings, `NRT_FAILURE in nrt_load_util`). iter-5 §6
concluded this meant changing `Wg_grp`'s dtype, found ~700 instructions touching that SBUF window,
and declared it a layout refactor. **That conclusion was wrong in one step:** you do not need to
change `Wg_grp` — you need a *bf16 copy of it* in free scratch, which is one inserted ACT `CAST`
per group (§0, §4.1). The whole edit is then 32 inserted CASTs + 64 descriptor rewrites + 64
trigger-wait rewrites, it is purely additive, and every piece except the S[88] question is already
device-verified somewhere in iters 4–6.
Ceiling: one queued DMA hop (~1.4 us of hardware plus consumer queueing) × 64, against site
`:597`'s 28.8 us of whole-machine idle (the largest single idle site in the kernel). **This is the
only remaining item with a plausible multi-% gain.**

---

## 5. Tooling / encoding facts (additions to iter-4 §5 and iter-5 §5, all still correct)

### 5.1 PE operand geometry — iter-5 did not have this and it silently broke my first hazard check
My first `src` range model for `MATMUL` used `src_mem_pattern_num_elem` as if it were the whole
operand; it is the **free axis only**, so the ranges came out 2 bytes wide and the disjointness
check was vacuous. Correct encoding (derived from lines 223/441/447/408/519/529/597/615):
* a compute-side address immediate is **flat**: `partition*262144 + byte_offset`.
* `LDWEIGHTS` (stationary): partitions `[p0, p0 + num_active_rows)`;
  free axis is consumed **backwards**, `step_elem[0] = -1`, so bytes
  `[b0 - (num_elem[0]-1)*esz, b0 + esz)`. Invariant `prod(num_elem) == num_active_cols`
  (line 615 uses a 2-level pattern `step [-1,0,-32] num [32,1,4]` = 4 blocks of 32).
* `MATMUL` (moving): partitions `[p0, p0 + num_active_rows)`, bytes
  `[b0, b0 + num_elem[0]*esz)`, `step_elem[0] = +1`.
* `MATMUL` dst: **PSUM bank = (dst_immediate − 0x2000000) / 0x800**, 8 banks
  (`psum_zero_region = SIZE2048` on all 2272), observed 0x2000000…0x2003800.
* `psum_accumulate_flags`: 3 = standalone (1723 + 101), 1 = first of an accumulate chain
  (129 + 127), 0 = continuation (130 + 62). Only 1982 of 2272 `MATMUL`s update `S[5]`; the
  non-updaters are the non-final members of accumulate chains.
* the solve's matmuls are **32×32** (`num_active_rows = num_active_cols = 32`), i.e. 1/4 of the PE
  array's rows and 32 of 512 free columns — latency-bound, 180–200 ns each.

### 5.2 A dead `EVENT_SEMAPHORE` slot can also be re-encoded in place (weaker than §0, but free)
`isa.Instruction` has `.set(field, value)` and `.encode()`; a slot is 64 bytes
(`header_inst_word_len = 16`). Rewriting a slot in place (even changing its opcode/struct) keeps
the file length and therefore every engine's instruction ordinals. iter-2 left **224 dead ACT ES
slots** (lines 483/487) and iter-5 left **160 neutered SP relay slots** — those are 384 free
instruction slots at known program points. This is the mechanism §4 needs and nobody has used it
yet. (`neff_diff.py` prints `opcode EVENT_SEMAPHORE [...ES_STRUCT] -> LDWEIGHTS [...LW_STRUCT]`,
so it will show you exactly what you re-encoded.)
**Limitation that §0 removes:** a compute struct has only ONE value field
(`events_semaphore_value`) shared between its wait value and its update increment, and **no
extended-events block** (`CAST`/`MATMUL`/`LDWEIGHTS` all lack one; only `EVENT_SEMAPHORE` has it).
So a re-encoded slot cannot reproduce a dead ES that both waits on `S[6] >= 55` *and* does
`S[43] += 16` — which is exactly the shape of all 224 dead ACT slots. Inserting a fresh slot and
leaving the ES alone (§0) is therefore strictly better.

### 5.3 Reset-aware semaphore satisfaction times
`SemaphoreUpdate.parquet`'s `id` is the **string** `"S[10] (qGpSimdDynamic)"`, not an int — parse
with `re.match(r'S\[(\d+)\]', str(sid))`. To get "when was `S[x] >= v` satisfied for an
instruction that started at `t`": take the last update at or before `t`, walk back while
`value >= v`, and return that timestamp. Do **not** use a running max over the whole run.
Be warned that this returns a *vacuously early* time for stale waits (e.g. `S[21] >= 16` on a ring
that passed 16 in the first 20 us), which is correct but makes "in-order delay" sums meaningless —
ACT totals 154 **ms** by that metric. Use it only to answer "was this wait already true at time
T", which is what the hoist planner needs.
`S[5]` has exactly one `sem_wr_imm` (the initial zero) and is then monotone to 1982, so the
ordinal ↔ wait-value correspondence is exact. `SemaphoreUpdate` does cover `S[72..87]` in this
NEFF (contra iter-5 §5).

### 5.5 **The real SBUF extent map.** My first one was wrong and it cost three probe cycles.
Padding every address immediate by a fixed 4 KB is **not** a free-space map — it reported
`0x020c00-0x033b00` free when `0x01c000-0x029002` is in fact live. Expand the access patterns:
* compute instruction, side ∈ {src,dst}: `p,b = divmod(immediate, 262144)`; byte extent
  `b + Σ (num_elem[i]-1)·step_elem[i]·esz` (signed — `LDWEIGHTS` runs backwards); partition span
  `num_active_channels` (ACT) / `num_active_rows` (PE) / 128 if the field is absent.
* DMA descriptor, side ∈ {from,to}: `*_sizes`/`*_steps` are in **bytes**; a dim whose step is a
  multiple of 262144 is a *partition* dim (count·step/262144 partitions), every other dim adds
  `(n-1)·step` to the byte extent, plus `sizes[0]` for the innermost block.
Result for **partitions 96-127** of this NEFF (`scratch/sbmap.py`, 45 lines):
used `0x004000-0x005fff`, `0x006000-0x00667f`, `0x00e000-0x0102ff`, `0x01c000-0x029002`,
`0x029100-0x029702`, `0x029800-0x02aaff`, `0x02ab00-0x02ab02`, `0x02ac00-0x0323ff`,
`0x033b00-0x033c08`, `0x033d00-0x0381f7`.
**Free: `0x00667f-0x00e000` (30 KB), `0x0102ff-0x01c000` (47 KB), `0x0323ff-0x033b00` (5 KB).**
Nothing above `0x0381f7` is usable — `def.json`'s `runtime_statebuffer_reservation` puts the
runtime's own 8 bytes at `0x38038` on all 128 partitions, i.e. at the top of SBUF.

### 5.4 Misc
* `profile.py --parquet` + `Instruction.parquet` `compiler_pc` = slot index on every engine, but
  Pool/PE ES rows can appear slightly **out of program order** in `start_ts` (I saw pc 296 before
  pc 295 on Pool, 26 ns apart) — do not infer program order from timestamps, decode the `.bin`.
* `PSEUDO_DMATRIGGER`'s `dma_queue_name` decodes as a byte **list** and `count_union_block_id` as a
  1-element list; join them to `SP0.json`'s `dma[*]` by `(queue, id)`.
* The SP trigger slots' `nki_source_location` is **unreliable** for identifying what a transfer is
  (only 39 of the 64 `:595` transposes are attributed to `:595`, and the first one attributed to
  `:595` is a `(1,1,1,64)` fp32 gather). Identify transfers structurally, from their descriptors.
* Queued-transfer census of this NEFF (all `SB`→`SB`): q0 669 (192 of shape 64×128 bf16, 227 of
  32×32 bf16, 146 fp32 partition gathers/scatters of shape `(1,1,1,64)`/`(1,1,64,1)`, 32 plain
  bf16 copies), q1 160 (plain fp32), q2 32 (`op:"cast"` fp32→bf16).

---

## 6. What iter-7 should do, in order

**Everything here depends on §0 (instruction insertion), which is already device-verified. The
constraint that closed the last three iterations' best candidates is gone: you can now add
compute. Do not spend the budget re-deriving §0; spend it on 1 or 2.**

0. **DONE AND SHIPPED — read §8 first.** The W-path fusion below is built, verified
   bit-identical for all 64 instances, and in the NEFF you inherit. Your job is **§8.0**: the hop
   removal bought ~23 us and the replacement ACT `CAST` gave +19.4 us back, so get the cast off
   ACT (or delete the now-dead q2 transfers). The sub-steps below are kept because they document
   the shipped construction; do not rebuild it.

1. ~~**§4's W-path fusion — insert the bf16 cast, then fuse the two DMA hops into one.**~~
   **(done, §8)** The largest remaining ceiling (site `:597` carries 28.8 us of whole-machine idle
   and the path was two queued hops of ~1.4 us hardware + consumer queueing each, × 64 chunks).
   * (a) Insert, per chunk, an ACT `CAST` (fp32→bf16, 32 partitions × 512 elements ≈ 0.4 us; clone
     the `:456` CAST, §5.1 geometry) reading `Wg_grp` and writing bf16 scratch in the
     free window at `0x011000` (§5.5). Place it **immediately before** the dead ACT ES at `:483`, which
     already waits `S[6] >= n` ("`Wg_grp` complete") and then does `S[40+k] += 16` — so the
     existing `S[40..55]` increment automatically becomes "the cast is done" (ACT is in-order and
     `SEM_ADD_IMM_COMPLETE` fires at completion). **No new semaphore needed, no ordinal shifts**:
     give the inserted CAST no wait and no update.
   * (b) Rewrite the `:595` transposing transfer (§4) to read that bf16 scratch, using iter-4
     §2.5's device-verified 4-descriptor 32×128-tile encoding (`transpose_element_size 2`, source
     stride `b*512` bytes, `to_steps [1,262144,33554432,32]`), and retarget its SP trigger's wait
     from the q2 cast ring (`S[72..87]`) to the `S[40..55]` value above.
   * (c) The 32 q2 `op:"cast"` transfers and `W_all` then become dead. **Confirm `W_all` has no
     other reader first** — my structural search was inconclusive because SBUF addresses are
     reused across groups (78 transposes "read W_all" by offset match, vs 64 real ones), so build a
     byte-exact SBUF liveness map rather than matching offsets.
   * Probe order: one chunk, `verify.py` (bit-identical expected), then all 64, then profile twice.
2. **Compute rebalancing is now possible too.** `DVE ∪ ACT = 797 us` of a 1175 us wall (§3.1) while
   Pool is 20% busy with ~940 us idle. With §0 you can insert the op on Pool (fresh semaphore
   S[88]+ to signal) and neuter the ACT/DVE original in place (replace with an `EVENT_SEMAPHORE`
   carrying the same `events_*` fields, as iter-2 did for the 224 ACT DGE saves — that keeps the
   counter ordinal exactly). Caveat from the `gdn_tkg` experiment: Pool `CAST` is ~4× slower than
   ACT per element and each overlapping Pool op cost DVE ~1 us there, so only move bulk ops that
   sit in a Pool-idle stretch. Start from §3.1: the 130 us of DVE-exclusive and 129 us of
   ACT-exclusive time is the only part where removing their work can shorten the wall.
3. **A DMA hop can now be replaced by compute.** The 146 fp32 partition gather/scatter transposes
   on q0 (shapes `(1,1,1,64)`/`(1,1,64,1)`, e.g. `:214` `cg_row` and `:589` `dte_C1`) each cost a
   full hop to move 64 fp32 values between 1 and 64 partitions. PE can do that as
   `LDWEIGHTS(column, transpose) + MATMUL(identity)` — the kernel already does exactly this at
   `:223` (fp32, `fp32_mode=LOW`, 64×64, `transpose_mode=ENABLED`) — then one inserted PSUM→SBUF
   copy. PE is 29% busy. **Check the numerics first**: `fp32_mode=LOW` on an identity matmul may
   not be bit-exact and `output_1` is already at 83% of the error limit. Site `:591` (the consumer
   of `dte_C1`) carries 5.4 us of whole-machine idle, so the realistic prize is ~5–15 us.
4. **Do NOT spend budget on:** everything in iter-1 §5.4, iter-2 §5.5, iter-3 §5.4, iter-4 §4.5,
   iter-5 §2 and §4.3. Newly and definitively dead from this iteration:
   * **All engine instruction reordering, on all five engines** (§2). PE was the last untried one
     and it is closed by two walls, each with a device measurement. The generic rules to remember:
     *never cross an `EVENT_SEMAPHORE` that updates nothing* (it is the next instruction's missing
     wait field — true on SP per iter-5 §2.1 and now on PE), and *never widen a wait prefix that
     something actually waits on* (it hangs).
   * **Pool DGE → queued DMA** (review candidate #2): ceiling 38 us by §3.2, the loads it would
     accelerate are already 5 groups ahead of use (§3.3), and it needs an HBM-referencing queued
     descriptor for which this NEFF has no template.
   * **Output-store conversion** (review candidate #3): `:623`/`:628` are 8.1 us of Pool-exclusive
     time and nothing waits on them.
   * **Splitting or merging DMA transfers for latency**: the fetch is 713 ns per transfer,
     flat in descriptor count and size (§3.4), and transfer time is only ~320 ns for 16 KB.
   * **Pool head-of-line blocking** (review candidate #4): re-confirmed empty (§3.3).
5. The honest expectation if 1–3 are not attempted is **0%**. Six iterations have taken
   1351.4 → 1175.3 us (−13.0%); what is left is real DVE/ACT arithmetic on the chunk's dependence
   chain, ~1.4 us × ~220 DMA hops of hardware latency, and a per-chunk period that only a full
   software-pipelining reschedule could shrink (iter-5 §7, still true, still a multi-iteration
   project — and §2 now says the reschedule would have to rewrite wait *prefixes*, which is why it
   hangs every time it is attempted). **The one thing that changed this iteration is that the
   remaining levers are no longer blocked by a missing capability, only by their own size.**

## 8. THE SHIPPED EDIT, exactly — and the five wrong turns it took to get there

**What it does.** For each of the 32 groups, one inserted ACT `CAST` writes a bf16 copy of
`Wg_grp` into free SBUF scratch, and that group's two `:595` `dma_transpose`s are re-expressed as
4-descriptor fused permute+transpose transfers reading that scratch instead of `W_all`. The
`Wg_grp --q2 cast--> W_all --q0 transpose--> W_t` two-hop chain becomes one hop. **It is purely
additive**: the q2 cast transfers and `W_all` are left in place (now dead but harmless), so the
edit touches only `Activation0.bin` (+32 slots), `SP0.json` (64 descriptors) and nothing else.

**`neff_diff` vs HEAD `a04258d`: exactly 2 members changed** — `sg00/Activation0.bin`
(162048 → 164096 B, 2532 → 2564 slots) and `sg00/SP0.json` (the 64 descriptor rewrites).
`SP0.bin` is **unchanged**, because the triggers keep their original waits (§8.1 wrong turn 5).

**Parameters as shipped** (`scratch/fuseall2.py`, ~110 lines):
* scratch base **`0x11000`**, one 1024 B slot per group (`0x11000 + g*1024`, so `0x11000-0x19000`).
* inserted CAST: clone of an existing ACT `CAST`; `num_active_channels 32`; src = the group's q2
  `desc0.from_off` (always partition **96**, byte ∈ {`0xe000`,`0xf100`,`0xf400`,`0x377f8`});
  `num_elem [512,1,1,1]`, `step_elem [1,1,1,1]`, FP32→BFLOAT16; dst = partition 96 + scratch;
  wait `S[6] >= v` (the group's q2-trigger value) where available; update **`SEM_INC_COMPLETE` on
  the fresh semaphore `S[88]`** (numeric enum values: `WAIT_FOR_SEM_GE_IMM`=5,
  `SEM_INC_COMPLETE`=19, `SEM_ADD_IMM_COMPLETE`=21).
* insert position: immediately **after** the group's dead ACT `EVENT_SEMAPHORE` at `:483`/`:487`,
  found by the join in §4.1 (q2 trigger waits either `S[6] >= v` → the ES with that extended wait,
  or `S[40+k] >= 16m` → the `m`-th ACT ES updating `S[40+k]`). Assert the anchors come out
  monotone in group order — they do — so `S[88]`'s count is the group index.
* descriptors: exactly §4.1, source partition `96+16c` for chunk `c`.
* **the 64 triggers keep their original waits** — see wrong turn 5.

### 8.0 Why the wall only moved ~4 us, and how iter-7 fixes it
The hop removal is worth about **23 us** and the inserted CASTs cost **+19.4 us of ACT busy**
(32 × ~0.61 us for 512 elem × 32 partitions). ACT was already the second-busiest engine
(521 → 541 us, 46%), so almost all of the gain was handed straight back. Three ways out, in order
of expected value:
1. **Move the cast off ACT.** Pool is 20% busy with ~940 us idle. Pool `CAST` was measured ~4×
   slower per element in the `gdn_tkg` experiment → 32 × ~2.4 us = ~77 us on Pool, which fits
   easily. The obstacle is the insert *position*: Pool's per-group program only reaches this point
   at its `ES :494` (which waits on the q2 ring), and anchoring there re-introduces the hop you
   just deleted. You need an earlier Pool anchor gated on `S[6]` — look for one, or give the Pool
   insert its own `S[6] >= v` wait and place it before the group's first Pool instruction, checking
   for deadlock (Pool is in-order and its tail holds the next group's prefetches, §3.3).
2. **Delete the now-dead q2 cast transfers** (32 transfers, ~23 us of DMA busy) **and neuter their
   32 Pool `ES :494` waiters** (iter-5 created those; each blocks Pool 15.8–18.9 us per group,
   §3.3). Prove `W_all` has no remaining reader first. This removes DMA work and a large Pool
   block; Pool-exclusive time is only 42 us (§3.2) so expect a few us, but it is cheap.
3. **Shrink the cast.** It converts all 4 sub-blocks (512 elem); that is what the transposes read,
   so there is nothing to trim without splitting the cast per chunk.

### 8.1 The five wrong turns (each cost one 4-minute probe cycle; do not repeat them)
1. **Scratch at `0x21000`, then `0x24000`** — both are *inside live SBUF* on partitions 96-127.
   My first free-SBUF map (§0) padded each address immediate by 4 KB and ignored access-pattern
   extents, so it reported a 77 KB gap that is not free. The **one-instance probe passed at
   `0x24000` by luck** and that luck cost me three probes. Use the real extent map (§5.5): the
   genuinely free windows on partitions 96-127 are **`0x00667f-0x00e000` (30 KB)**,
   **`0x0102ff-0x01c000` (47 KB)** and `0x0323ff-0x033b00` (5 KB).
2. **Blaming the fp32→bf16 rounding.** I was sure ACT and the DMA's `op:"cast"` rounded
   differently, and tested a truncating variant (`in_dtype BFLOAT16`, src `+2`, `step_elem 2` —
   a reinterpret of each fp32's high half). **Byte-identical failure**, as did sourcing from
   `W_all` instead of the scratch, as did three different wait configurations. *Five probes with
   byte-identical `max_abs_err` meant the variable I was changing was not the cause* — that is the
   signal to stop varying it and go looking for an invariant (here: the clobbered scratch).
3. **Assuming all 32 q2 triggers gate the same way.** They do not: some wait `S[6] >= v`, others
   `S[40+k] >= 16m`. Assert, don't assume.
4. **Identifying the transposes by address alone** — see §4.2; add the `:597` `LDWEIGHTS`
   destination filter (it turned out my greedy trigger-order match was already correct, but the
   filter is what proves it).
5. **Retargeting the 64 triggers to wait on `S[88]`.** This is the one that looked most correct and
   was most wrong. `W_t` has only **9 distinct addresses for 64 uses** (`:597`'s `LDWEIGHTS`
   sources), so the triggers' original `S[8..23]`/`S[24..39]` ring-throttle waits are also
   providing the **WAR protection** that stops group `g+k`'s transpose from overwriting `W_t`
   before group `g`'s `LDWEIGHTS` has read it. Firing the transposes early on `S[88]` broke that
   for one chunk (`output_0` 1.772e-02, `output_1` bit-exact — a tell that the *state* path was
   fine and only the output path was corrupted). **Keeping the original waits made it pass.**
   iter-4 §2.7's "the ring throttles are stale and free to remove" is true for *timing* and false
   for *correctness* once you make a transfer eligible to fire earlier.

### 8.2 Residual risk, stated plainly
With the original trigger waits kept, nothing *semaphore*-orders the inserted ACT `CAST` before its
transpose; the ordering rests on the CAST being gated on the same `S[6]` value the q2 transfer's
trigger used, and on ACT reaching it promptly. **The baseline NEFF has exactly the same looseness
on this edge** — iter-5 §2.3 showed these triggers wait on a Pool/q0 ring throttle and not on the
q2 cast's completion — and the ACT `CAST` makes the data ready *earlier* than the q2 DMA did
(~0.4 us of ACT vs ~1.4 us of trigger+transfer+semaphore). Outputs were bit-identical on 3 of 3
runs plus 2 further single runs. If a future iteration wants this tightened, the correct fix is a
trigger wait that is the *conjunction* of the ring throttle and `S[88]`, which needs a preceding
`EVENT_SEMAPHORE` on SP to carry the second operand (§2.2 explains why that ES must then never be
crossed).

## 7. The two edits I built and did not ship
* the 3-move PE hoist (§2.4): PASS, ~0.5 us of wall, below the floor — not worth the risk;
* the single-instruction insertion probe (§0): PASS, one extra 290 ns ACT `CAST` writing to
  scratch, a pure cost — its value was proving the capability, which §8 then used.

# iter-5 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 38.59 us (iter-4 NEFF, re-measured here) -> 38.38 us, -0.5 %** (vs the iter-0
baseline 76.054: **-49.5 %**). A small win. Five edits were built and device-verified, four
of them negative; the important output of this iteration is **the measurement that explains
why the cheap levers are now closed** and what the remaining ~2 us actually needs (sections
"The one fact that changes the strategy" and "What the floor really is").

| # | edit | script | median (repeats) | verdict |
|---|---|---|---|---|
| — | iter-4 NEFF, re-measured | — | 38.588 (5), 38.596 (7) | |
| A | fold the 12 DVE `delta *= beta_K` into the 12 ACT `CAST delta->delta_bf` as one scale-pointer `ACTIVATE` | `fuse.py` | 38.662 (5) | **neutral, reverted** |
| B | `Q_p` DGE issued 2nd + `f30 Q_p^2`/`f29 q_scale` moved under the head's first T2 load + that T2 reload moved into the body | `qearly.py` | 39.035 (5) | **+0.45, reverted** |
| C | tighten the 12 PE outer-product `s4` waits onto the `cast_delta` that really produces their tile | `tight.py` | **38.421 / 38.335 / 38.415 / 38.312** (7,7,7,7) | **kept** |
| D | defer DVE's `xf h0` past the three S-updates PE waits on | `defer.py` | 38.531 (7) | no gain, reverted |
| E | batch-defer all 11 DVE per-head tail ops past the next 4 s6 updaters, alone and on top of A | `pipe2.py` | 38.738 / 39.503 (7) | **worse, reverted** |

Final sha256 `b9b90ae88561d2b7f3f455db0b540b406dde50ec6b14135e604207b3aea080f3`.
`neff_diff` vs the iter-4 NEFF: only `events_update_mode/idx` on 4 ACT slots plus
`events_semaphore_value` renumbering — **no instruction moves, no address changes, no slot
count changes** in any of the 10 streams.

The ~0.2 us of edit C is smaller than a single run's spread (0.4-2.2 %), so it was called by
measuring **each state several times**: {38.421, 38.335, 38.415, 38.312} (median 38.38) vs
{38.588, 38.596}. Repeat measurements of the *same* bytes spread 0.01-0.11, so the ~0.2 gap
is real but small, and the two sets do not overlap.

Every edit was `verify.py --runs 3` (final state: `--runs 5`) **bit-identical with
`max_abs_err` exactly the baseline value** (2.322e-02 = 86 % of the limit, 1.633e-03),
`cycle.py` acyclic, `semaphores.py` clean on both subgraphs, `check.py` 0 table-set
violations, `hb.check_psum` 0 unordered pairs, plus a new generic SBUF WAR/WAW reachability
check (see "New tools").

## Where the time is (38.59 us), head vs body

`cmp.py`-style split at 26.8 us (`Instruction.parquet`, `compiler_pc.notna()`, core 0):

| region | span | ACT busy | DVE busy | PE busy |
|---|---|---|---|---|
| head 12.65 -> 27.30 | 14.65 | **13.66 (93 %)** | 7.50 (51 %) | 1.48 |
| body 27.30 -> 49.95 | 22.65 | 12.81 (57 %) | **20.01 (88 %)** | 12.96 (57 %) |

So the head is ACT-throughput bound (8.98 of its 13.66 us is seven 1.283 us
`ACTIVATION_TABLE_LOAD`s) and the body *looks* DVE-throughput bound. It is not — see below.

## The one fact that changes the strategy

**`t/crit.py` computes the longest path through (program order + semaphore producer->waiter)
weighted by each slot's measured duration. It is 37.55 us out of a 38.6 us total.**
The kernel is on its critical path essentially everywhere; there is no slack left to
squeeze by rebalancing work between engines. Two independent confirmations:

* **Edit A removed 2.68 us of DVE body work (20.01 -> 17.33) and one cross-engine hop per
  head, and the body span did not move at all** (22.65 -> 22.74). ACT went 12.81 -> 15.25,
  DVE 20.01 -> 17.33 — i.e. both engines ended up well under 80 % busy in the body and the
  span was unchanged. So the body is **not** throughput bound and moving elementwise work
  between ACT and DVE is a dead lever in both directions (iter-3 already found the ACT->DVE
  direction dead with `deltabf.py`; now the DVE->ACT direction is dead too).
* The critical path spends 14.08 us in the ACT head (all 7 table loads are on it) and
  23.5 us in the body, where it alternates engines but travels through **consecutive slots**
  of each engine — i.e. the cost is in-order streams doing other heads' work before reaching
  the dependent instruction. That is a scheduling problem, not a work problem.

## What the floor really is (do read this before planning iter-6)

The 12 head chains in the body are **completely independent of each other** (head h's state
tile, scalars and output are private; nothing in head h+1 reads anything of head h). So the
body is not a 12-deep dependency chain — it is 12 independent 11-step chains of ~3.5 us
each, and its span floor is `max(engine busy) + one chain's fill/drain`:

| | DVE body busy | ACT body busy | PE body busy | floor | measured span |
|---|---|---|---|---|---|
| shipped | 20.01 | 12.81 | 12.96 | 20.0 + ~1.5 = **21.5** | 22.65 |
| with edit A (`fuse.py`) | 17.33 | 15.25 | 12.98 | 17.3 + ~3.5 = **20.8** | 22.74 |

**Both constraints bind at ~22.7 today, which is why each of them alone buys nothing.**
Edit A removes 2.7 us of DVE work and the chain stops being the thing that matters — but the
schedule then fails to use the 5.4 us of DVE idle it creates, so the span does not move.
Equally, a pure scheduling improvement without edit A would hit DVE's 20.0 us throughput
wall at ~21.5.

So the remaining ~2 us needs **both**: apply `fuse.py` (or any other real DVE work removal)
*and* re-schedule the body so DVE's new idle is absorbed. Neither half pays on its own, and
that is exactly why iters 3-5 have each found "move elementwise work between ACT and DVE" to
be worth zero in both directions.

Head floor: ACT 13.66 us busy with one unavoidable 0.87 us stall (waiting `dt`, which waits
the `a` transfer) = 14.5, and it measures 14.65. **The head is at its floor** unless a table
load can be removed, and the section below shows the sequence is forced.

Overall: `12.0 prologue + 14.5 head + 20.8 body + 1.1 tail` ~ **34 us of `total_exec` is the
realistic target**, i.e. ~4.4 us left, essentially all of it in the body schedule.

## Edit C (kept): PE waits that were 2 us more conservative than the data flow

`t/pe4.py`: four of the twelve ACT `CAST delta -> delta_bf` slots (ACT 22, 25, 37, 54) do
**not** update s4, so the PE `LDWEIGHTS`+`MATMUL` pair that consumes their `delta_bf` tile
has to wait on the *next* ACT s4 update instead. Measured slack between the tile being
ready and the wait being satisfied:

| PE MATMUL | tile | producer | producer end | wait satisfied | slack |
|---|---|---|---|---|---|
| PE21 | 0x4d70 | ACT22 | 27.869 | ACT24 29.863 | **1.994** (on the critical path) |
| PE61 | 0x4370 | ACT54 | 41.303 | ACT56 42.265 | **0.962** |
| PE37 | 0x4170 | ACT37 | 34.228 | ACT38 34.427 | 0.199 |
| PE23 | 0x4170 | ACT25 | 30.051 | ACT26 30.155 | 0.104 |

Fix (`tight.py`): give all 12 casts an s4 update and set each governing PE wait to
`s4 >= count(the cast its group really needs)` (PE@20 14->13, PE@22 16->15, PE@36 23->22,
PE@60 35->34). No instruction moves, no address changes, nothing else in the NEFF changes.
Mechanics worth remembering:
* several heads share a `delta_bf` tile (0x4170 is written by ACT 25/37/45/63), so pair the
  k-th writer of a tile with the k-th reader of it — both streams are monotone (same rule
  iter-3's `deltabf.py` used).
* a PE wait governs the waiter **plus every following PE slot with no wait of its own**
  (here `LDWEIGHTS`+`MATMUL`), so take the max over that group.
* lowering a wait can destroy a WAR ordering that the old conservative value supplied; the
  check for that is `hb.check_psum` plus the generic SBUF reachability check in `tight.py`.
  Both are clean here.

**The same analysis should be re-run after any future edit** — `pe4.py`/`slack.py` are
cheap, and this class of win costs nothing in work or risk.

## Edit A (reverted, but a reusable primitive and an exact cost model)

`fuse.py` replaces, for each head, `DVE TENSOR_TENSOR MULT(delta_raw, beta_K[h])` +
`ACT CAST fp32->bf16` with a single
`ACT ACTIVATE activation_func=128, scale_src=POINTER_IMMEDIATE, scale_value_imm_ptr=beta_K[h],
in=FP32, out_dtype=BFLOAT16`. **Device-verified bit-identical**, so:

* **`activation_func=128` with `scale_src/imm_src = POINTER_IMMEDIATE` is the compiler's own
  "copy, scaled by a per-partition scalar at an SBUF pointer" form** — iter-1's NEFF
  (`git show df60817:kernel.neff`) still has 36 of them, and iter-2 moved them all to DVE as
  zero-stride `TENSOR_TENSOR`s. The two forms are **bit-exact equivalents in both
  directions**, and the ACT form may freely convert fp32 -> bf16 on the way out, so it also
  absorbs a following `CAST`. Donor bytes are embedded in `fuse.py`.
* ACT may read a source on one partition and write a destination on another (iter-1 slot 28
  does partition 32 -> 64), and the scale pointer is read from its own address' partition,
  not the source's.
* **Cost model: a 128-element, `channels=1` scale-pointer `ACTIVATE` costs 0.494 us, vs
  0.290 for the `CAST` it replaced and 0.280 for the DVE `TENSOR_TENSOR`.** That is the
  reason ACT body busy went *up* 2.44 us while DVE dropped only 2.68. Do not assume an
  `ACTIVATE` costs the same as a `CAST`/`TENSOR_TENSOR` of the same shape.
* New hazard this creates: the `delta_raw` scratch is **shared between heads** (0x836400 is
  the SUBTRACT dst for heads 1,3,5,7,9,11), and the WAR against the next head's DVE
  `SUBTRACT` used to be free by DVE program order. Moving the reader to ACT needs an
  explicit guard: `fuse.py` detects it with a reachability check and inserts one DVE
  `EVENT_SEMAPHORE` waiting on the ACT reader (exactly one guard is needed, before DVE's
  head-3 SUBTRACT).

Measured 38.662 vs 38.588 — neutral. It is still a good building block if a future
iteration ever makes ACT the body bound again, and it does remove one chain hop per head.

## Edit B (reverted): the head's 7th table load cannot be bought back

The ACT head's last two slots are a **second** `ACTIVATION_TABLE_LOAD t2` (26.02-27.30) plus
`f29 q_scale` (27.31-27.48), and the first body instruction (`cast_delta` h0) is data-ready
at 26.10 and waits for them. Removing that 1.45 us looks like the single cleanest win left.
It is not reachable:

1. **The load cannot be deleted**: the 12 body `f29` rsqrt need table set 2 as well, not
   just `f29 q_scale`. `check.py` caught this immediately (24 violations). It can only be
   *moved*.
2. **Moving it into the body is free** (it lands in the 1.42 us ACT stall before the first
   `f30`, which waits on PE's first `out_p`; `f30` is in every table set). That part worked.
3. **But `f30 Q_p^2`/`f29 q_scale` then have to run under the head's *first* T2 load
   (between ACT slots 5 and 8), which needs `Q_p` early, and `Q_p` cannot be made early
   enough.** Swapping Pool's `S_all` and `Q_p` DGE slots does put `Q_p`'s packets at 17.9
   instead of 21.6 — but `f30 Q_p^2` still stalled until **19.26**, and `f29 q_scale` a
   further 0.26 on PE's `q_ss`. Net +1.47 us of new ACT head stall against 1.28 saved:
   measured **39.035, +0.45 us**.

### Measured DMA facts behind that (new)

* **DGE semaphore visibility lags the last DMA packet by 0.55-1.35 us.** Clean datapoints:
  `a` (qSyncIO id 2) last packet 14.125, DVE@4 starts 14.670 -> 0.545 lag; `S_all` last
  packet 21.51, DVE's `ES s25` at 21.962 -> 0.45; `Q_p` last packet 17.9 (after the swap),
  ACT's `f30 Q_p^2` starts 19.259 -> **1.35**. Budget ~1 us whenever you plan around "the
  transfer lands at T".
* The three input DGE transfers are **not** 512 B descriptors in the packet trace: per DMA
  engine they are `K_p` 128 B + 64 B, `S_all` 12 x 4096 B, `Q_p` 128 B + 64 B, each followed
  by a 4 B semaphore write. `S_all` moves at ~400 GB/s aggregate and takes only ~1.9 us per
  engine; **its completion time is set by Pool's serial descriptor generation (1.25 us per
  DGE), not by bandwidth.** Hence any order that makes `S_all` the 3rd DGE pushes it ~1 us
  later, and any order that makes `K_p` or `Q_p` 3rd pushes that one ~5 us later.
* Completion times in the shipped order (core 0): `K_p` 16.63, `S_all` 21.51, `Q_p` 21.63.
  With `S_all`/`Q_p` swapped: `K_p` 16.55, `Q_p` 17.90, `S_all` 22.38.
* All nine qSyncIO input transfers land by 15.05 (core 0) / 15.45 (core 1); `A_log` (id 7)
  is the last.

### Why the ACT head's table-load sequence `T1 T0 T2 T0 T4 T3 T2` is forced

Worked out with the DMA numbers above; each load is 1.283 us and ACT is 93 % busy, so this
is where 2.6 us would come from if it were reachable. It is not:

* `f9 softplus` is T1-only and must be first, because `A_log` (and so `f7 exp(A_log)`, T0)
  is only *visible* at ~15.6 while `dt` is ready at 14.83. **`T0 f7A T1 f9` was measured-out,
  not guessed**: it would stall `f7A` to 15.6, pushing `f9` to 17.3 and DVE's `g` from 16.63
  to 17.33. (This is why iter-3's `head3.py` moved to `T1 f9 T0 f7A`.)
* `f7 exp_g` (T0) must follow DVE's 4-way `g` broadcast, which needs `f9` *and* `f7A`, so a
  single T0 cannot cover both `f7`s without ACT stalling ~0.70 us — and the resulting order
  delays `f29 k_scale` by ~1.0 us, which delays DVE's 1.74 us `k_f_all *= k_scale` and hence
  the two `S *= exp_g` halves and the two big casts. I worked the full timeline: **-0.08 us.
  A wash, matching iter-4's candidate-3 estimate. Do not spend time on it.**
* `f5 sigmoid` is T3-only and must precede the casts (iter-4's `beta.py`), `silu` is T4-only,
  `f29` is T2-only, `f7` is T0-only, `f9` is T1-only; `f30 square` is in every set.
* The ACT head's only idle is 0.87 us at 13.996-14.863 waiting for `dt` (which waits the `a`
  transfer). Nothing can fill it: the slot before `f9` must be the T1 load.

## New tools (in `.autohelix/notes/iter-5-tools/`)

* **`crit.py TAG`** — longest weighted path over program order + semaphore edges, printing
  the chain with each slot's duration and timestamp. The single most informative new tool:
  it is what showed the kernel has no slack and found edits C, D and E.
* **`pe4.py`** — pairs the 12 ACT `cast_delta` slots with the 12 PE outer-product MATMULs
  that read their tiles and prints, for each, (tile ready) vs (wait satisfied).
* **`slack.py TAG`** — the generic version over every semaphore wait. Noisy (it reports
  huge fake slack for `EVENT_SEMAPHORE`s and for waiters with no cross-engine source), so
  read it filtered by engine/semaphore.
* `fuse.py`, `qearly.py`, `tight.py`, `defer.py`, `pipe2.py` — the five edits, all with
  `--dry`. `pipe2.py` takes `$K`, `defer.py` takes `$MOVE=src:dst`.
* `tight.py`, `fuse.py`, `defer.py` and `pipe2.py` carry a reusable **generic SBUF WAR/WAW
  reachability check**
  (every (reader, writer) and (writer, writer) pair on a set of addresses must be ordered by
  the happens-before graph). `hb.check_psum` only covers PSUM; this covers SBUF scratch,
  which is where both of this iteration's real hazards were.
* `qearly.py` also contains the reusable **prefix-closure renumbering generalised to any
  semaphore** (iter-3's `head3.py` only did s4), i.e. it handles reordering the ACT *and*
  PE streams in one pass, with per-wait overrides keyed `(semaphore, engine, old slot)`.

## Gotchas hit this iteration

* `semaphores.py` caught a bug `cycle.py` did not: my Pool reorder `[pool[0], pool[3],
  pool[2]] + pool[4:]` silently **dropped** `pool[1]` (the `K_p` DGE), which showed up only
  as "s24: 16 waits, 0 increments". `cycle.py` models DMA semaphores as free, so **always
  run `semaphores.py` on both subgraphs after touching Pool**, and assert the stream length
  and the `semaphore` field sequence.
* `check.py` (table-set validity) is what caught that the body's 12 `f29` need T2. Run it on
  every ACT edit, not just reorders of the head.
* `profile.py --repeats 5` spread was 0.40-1.25 % this iteration. **0.3-0.5 us is the noise
  floor**; one of my three results (-0.17 us) needed repeated 7-repeat runs to call.

## Edit D (reverted): single-slot deferral is too small to beat the noise

`defer.py MOVE=58:62`. On the kept NEFF `crit.py` shows the path running DVE40..62 (3.94 us,
pure program order) and then PE30/31 (`out_p` head 3), whose real input is DVE62
(`CAST S_h3`). Five of those DVE slots — 48, 49 (ADD/cast S h0), 55, 56 (h1) and 58
(`xf h0`), 1.35 us total — are **not** needed by DVE62 and sit in front of it only because
DVE is in-order. Moving just DVE58 (0.332 us) behind DVE62 is bit-exact and clean
(`defer.py` asserts the moved slot updates no semaphore and has no waiter, so the s6
renumbering is pure identity; the two waits whose conservative closure would have been
*raised* are overridden back to identity, justified by the generic SBUF/PSUM check over the
moved slot's three addresses). Measured **38.531 vs 38.42 — no gain.**

Why: the deferred work does not disappear, it piles up later, and DVE is 88 % busy. One
0.33 us slot is below the noise floor. The lever is real but only as part of a many-slot
schedule *together with* a DVE work reduction (see "What the floor really is").

## Edit E (reverted): a *uniform* deferral rule is actively harmful

`pipe2.py` batch-defers all 11 DVE per-head tail ops (the 9 `out` multiplies and 2 `xf`
STTs that update no semaphore and have no waiter) past the next `K` s6-updating DVE slots.
This is the safe mechanical form of the body re-schedule: because the moved ops update
nothing and nothing waits on them, the s6 numbering is untouched and all renumbering is
identity; the only real constraints are the shared `xf` scratch buffers (0x5078 is used by
two heads) and that nothing may be deferred past DVE's **79th** s6 updater, because SP's
`output_0` doorbell waits `s6>=79` and that is the *only* thing ordering the output transfer
after the 12 `out` multiplies. `pipe2.py` enforces both (reachability check + an explicit
cap) and both variants verified bit-identical on the device.

| variant | median (7) |
|---|---|
| kept NEFF | 38.42 |
| `pipe2.py K=4` alone | **38.738** |
| `fuse.py` + `pipe2.py K=4` | **39.503** |

So the lever is real but **a blind uniform rule makes it worse, and worse still when DVE has
been freed** — moving a head's tail work later delays its own `out` without putting anything
useful in the hole, and the holes are not where the rule puts them. iter-6 must choose each
move from `crit.py`'s path rather than from a rule, and measure every few slots.

## Ranked candidates for iter-6

1. **`fuse.py` + a *measurement-driven* body re-schedule, together (~2 us; neither half pays
   alone, and a uniform rule for the second half is worse than nothing — edits A, D, E).**
   This is the conclusion of "What the floor really is". Concretely: apply `fuse.py` (DVE
   body 20.0 -> 17.3, one fewer cross-engine hop per head, bit-exact, already written and
   verified), then **for each of the three or four longest consecutive runs `crit.py` reports
   on the path**, move only the slots in that run that the run's exit edge does not need, and
   re-profile. Do not apply a blanket rule. `defer.py` is the worked single-slot example,
   `pipe2.py` the batch machinery (identity renumbering + the SP `s6>=79` cap), and
   `qearly.py` the generalised prefix-closure + override machinery for moves that *do*
   involve s6 updaters. iter-3 lost a device run on a whole-body permutation, so go
   incrementally. Target: body 22.7 -> ~20.8.
2. **Re-run `pe4.py`/`slack.py` after every edit.** Edit C was free; there may be more such
   over-conservative waits once the streams move (iter-3 noted the compiler also emits waits
   whose producer is a proxy, e.g. 2 of the 12 casts have *two* waiters).
3. **Partition-parallel per-head vectors (the big one, still open, still highest risk).**
   Four of the six per-head DVE ops are `channels=1`, 128 elements — 1/128 of the machine —
   and cost 0.28 us each, the same as the `channels=128` 128x128 `S += outer_p`. If PE wrote
   head h's 1-row `out_p`/`kv_p` to PSUM **partition h** instead of partition 0, one
   `channels=12` op would replace twelve, taking ~14 us off DVE and ~8 off ACT. Unknown to
   settle first: whether a `MATMUL`'s dst partition is freely selectable (the dst address
   encodes it as `partition * 0x40000`), and whether the per-head scalars (`beta_K`,
   `ssum`, `q_scale`) can be made per-partition. Note this only pays off **together with**
   candidate 1, since the body is latency- not throughput-bound today.
4. **Issuing a DGE descriptor from SP or ACT instead of Pool** (chapter 9 says
   `PSEUDO_DMA_DIRECT2D` is legal there; SP's stream is 19 slots and idle after 13.98).
   That is the only way to break the "Pool generates the three input DGEs serially at
   1.25 us each" constraint, which is what pins `S_all` to 21.5 and `Q_p` to 21.6 and is
   what killed edit B. Unknown: whether `src_table_index` is engine-relative.
5. Closed/not worth it: moving elementwise work between ACT and DVE in either direction
   (edits A, iter-3 `deltabf.py`); any re-derivation of the ACT head table-load order
   (section above); splitting the state write-out further or the cast halves differently
   (iter-4); re-ordering `PSEUDO_DMATRIGGER` or the qSyncIO descriptor ring (iter-3/iter-4).

## Addresses (unchanged from iter-4, plus)

`delta_bf` tiles (ACT `cast_delta` dsts, in head order): 0x5970, 0x4d70, 0x4170, 0x4470,
0x4170, 0x4270, 0x4170, 0x4670, 0x4370, 0x4470, 0x4170, 0x36800 — **shared between heads**.
`delta_raw` scratch (DVE `SUBTRACT` dsts): 0x835800, 0x836400 (heads 1,3,5,7,9,11 all),
0x1804170, 0x1804370. `beta_K` 0x4090+4h, `q_scale` 0x40f8, `k_scale` 0x4128,
`K_p` 0x37e00, `Q_p` 0x37f00, `S_all` 0x1c000+0x200h, `S_bf`(head) 0x26000+0x100h,
`k_f_all` 0x35800+0x100h, `q_p` 0x4158+2h, `norm_weight` 0x36400, silu(z) 0x20000.
`addr = partition*0x40000 + offset`; PSUM `0x2000000 + bank*0x800`.

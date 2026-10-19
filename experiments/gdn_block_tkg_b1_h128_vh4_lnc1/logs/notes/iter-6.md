# iter-6 — 51.07 → 47.8 us (1.07x on iter-5; 4.57x vs the 218.17 us baseline)

Start-of-iteration re-measure of the iter-5 file: **51.070 us** (spread 0.71%, 5 repeats).
Shipped file sha256 `04897daa9a4ae15c12a4f753422bf42d96b2007c623314667c1c951d76738693`.
Shipped medians of 5, four times: **47.802 / 47.596 / 47.722 / 47.918** (mean 47.76). Two of the
four runs were 5/5 in the fast mode (spread 0.20–1.2%); the other two caught one ~52.6 us
capture, which is the standing bimodality (§10) and drags the reported median up by ~0.2 and the
spread to ~10%. `neff_diff` vs the original baseline: the same 6 of 46 members as iter-5
(`Activation0.bin` 105→119 slots, `DVE0.bin` 114→127, `PE0.bin`, `Pool0.bin`, `SP0.bin`,
`SP0.json`), no member added or removed.
`verify.py` gives **exactly the baseline errors** (2.999e-03 / 1.928e-03 / 6.033e-04) after every
shipped edit, so all arithmetic is bit-identical; `verify.py --runs 4` reports **all 4 runs
bit-identical**, so no latent race is visible on this input. `semaphores.py` clean,
`launch_rules.py` passes (1 subgraph, num_tpb 1, 11 inputs, 3 output heads, LNC=1, one rank).

| state | what | median of 5 |
|---|---|---|
| iter-5 (HEAD) | — | **51.070** |
| **p1** | Act 54's gate `W[28]>=16` → `W[67]>=16` (one field!) | **48.697** (−2.37) |
| **p2** | 8 Activation CASTs (68–71, 74–77) moved to DVE | **48.400** (−0.30) |
| p3a | SP: let the 6 DGE DMAs issue back-to-back | verify PASSES, **profiler HANGS** (§5) |
| p3b | hoist Act CAST 62 above the T2 `ACT_TABLE_LOAD` | verify FAILS (§3) |
| p3c | hoist Act CAST 60 above the same load | bit-identical (kept) |
| p4 | move Act 71/72's barriers past the silu cluster | verify FAILS (§4) |
| **p5 (shipped)** | p3c + hoist the 4 silu `ACTIVATE`s above those barriers | **47.802 / 47.596 / 47.722 / 47.918** |
| p16 | move the T0 `ACT_TABLE_LOAD` after the `func 128` copies (no table active) | verify FAILS (§2) |
| p6/p7 | move an Activation or Pool DGE DMA onto SP / DVE | verify FAILS (§2) |
| p8-m4, p10, p11 | merge 4 (or 2+2) Activation tile loads | verify FAILS (§2) |
| p9 | merge Act tile loads 7+8 on slot 7, drop slot 8 | bit-identical, **49.594 (+1.8 REGRESSION)** |
| p12 | defer Act DMA slots 7,8 past the first two computes | bit-identical, 48.097 (+0.3, reverted) |
| p13 | SP 19/20/21/29 relocated with **every field preserved** | verify PASSES, **profiler HANGS** (§5) |
| p14 | de-interleave the merged transpose so Act 55–58 read contiguously | bit-identical, 47.803 (neutral, §8) |
| p15 | split the final 256 KB output store into 2 DMAs, 1st issued 10 us earlier | bit-identical, 47.648 (neutral, §9) |

**Engine occupancy on the shipped file — use the UNION of instruction intervals, not the sum of
durations.** `profile.py`'s "busy" and a naive `sum(duration_ns)` both double-count: Activation,
DVE and PE overlap consecutive instructions (they are pipelined). Over the body (11.9 → 61.0):

| engine | sum of durations | **union busy** | overlap | % of its 49 us span |
|---|---|---|---|---|
| Scalar (Activation) | 44.0 | **38.9** | 5.1 | **79%** |
| Vector (DVE) | 23.1 | 18.8 | 4.3 | 38% |
| GpSimd (Pool) | 23.4 | 22.3 | 1.1 | 46% |
| Tensor (PE) | 13.4 | 9.7 | 3.7 | 20% |
| Sync (SP) | 7.5 | 7.5 | 0.0 | 15% |

Activation is still the metric at 79%, with 10.1 us of real idle in 10 gaps (§7). But the 5.1 us
of overlap is why removing Activation work pays less than its duration suggests (§8).

---

## 0. THE HEADLINE: a 3.7 us Activation idle was gated by a DEAD 2-byte dummy DMA (−2.37 us, one field)

`Activation` slot 54 is a bare `EVENT_SEMAPHORE W[28]>=16` sitting immediately before slots
55–58, which read/write `242048` — the destination of iter-5's merged transpose (`SP` slot 18).
**s28 is `qSPSpillReload0` block 4: a 2-byte copy `SB 212224 → inst__I-16-0:_mem_110_105`,
an iter-1 leftover writing a dead scratch buffer.** It is triggered by `SP` slot 31, which SP
only reaches *after* slot 18's own completion has been reported, so the dummy's completion
lands ~2.6 us after the data actually arrived:

```
SP 18 DMA   35.80 → 36.41      (the real transfer)
s67 += 16          37.80       (≈1.4 us completion-report hop)
SP 31 trigger      39.00
s28 += 16          40.39       ← Activation slot 54 released here
Act 55             40.44
```

Fix: `events_wait_idx 28 → 67`. s67's first +16 can only come from slot 18's DGE completion
(SP is in-order and slot 19 blocks on `W[67]>=16`, so SP slot 56's later `s67+=16` cannot race
ahead), so the gate is strictly tighter than the dummy. **−2.37 us for one changed field.**

> **The reusable move: for every `EVENT_SEMAPHORE` gate that sits in a measured idle window,
> print the semaphore's producer. If it is a queued-DMA block, open `<Eng>0.json` and look at
> the descriptor — this NEFF is full of 2-byte dummies from iterations 1–5 that are still used
> as proxy gates, and a proxy costs an extra trigger + an extra completion hop.**
> I then audited every other Activation gate this way and **this was the only proxy left** —
> see §7 candidate 1 for the per-gap table, and do not spend time re-auditing them.

This was iter-4's `f9` / iter-5's candidate 3. It failed twice before and works now, exactly as
iter-5 predicted: the gate only became binding once the DGE stopped being saturated.

## 1. THE SECOND FINDING: `def.json['dma_queue']` — DGE queues are OWNED per engine

This single JSON key explains three separate families of failure from iter-4, iter-5 and this
iteration. **Read it before touching any DGE DMA.**

```
qActDynamicHW     type dynamic  owner act   sems 72..87
qPoolDynamic      type dynamic  owner pool  sems  8..23   (+ SB_scratch for descriptors)
qSPDynamicHW      type dynamic  owner sp    sems 56..71
qSPIO0            type in       owner sp    sems 40..55
qSPSpillReload0   type data     owner sp    sems 24..39
```

A `PSEUDO_DMA_DIRECT2D`'s `semaphore` field *is* its queue (position within the owner's
`semaphore_set`). Consequences, all measured this iteration:

* **You cannot relocate a DGE DMA to another engine's stream.** Activation slot 1 or slot 8
  (sems 80/87, act-owned) placed in `SP0.bin` or `DVE0.bin` → `output_1` off by 26000%.
  Rewriting the semaphore to an SP-owned queue (49, `qSPIO0`) fails too, so it is not simply
  "the sem must belong to the issuing engine" — a *dynamic* DGE DMA seems to need its own
  queue group. One contrary data point: `Pool` slot 12 (sem 19) moved to `SP` **did** verify
  bit-identical. One run, not reproduced, do not build on it.
* **Reordering DGE DMA slots inside their own engine is fine.** Activation slots 7 and 8
  moved after slot 13 (keeping their own semaphores): bit-identical. So the *order* is free;
  the *engine* and the *semaphore* are not.
* **Merging: only the single TRAILING queue can be absorbed, and the surviving instruction
  must be the EARLIER one.** The one form that works:
  ```
  base = Act slot 7 (sem 86, src 0 → dst 225280); drop slot 8 (sem 87)
  src_num_elem [128,2]  src_step_elem [8, +1024]      # steps are in BYTES
  dst_num_elem [128,2]  dst_step_elem [262144, -768]
  Pool 20's guard W[87]>=16 → W[86]>=16
  ```
  Everything else fails verify with an identical error triple
  (`2.720e-02 / 9.436e-01 / 3.176e-02`): n=4 on slot 3 dropping 4,5,6 (**middle** queues);
  two pairwise merges at once; and any reassignment of a DGE semaphore to a different tile.
  Building the merge on the *later* slot with reversed step signs also fails — that is why
  iter-5's `m5` recipe ("addresses from the group's last tile", "only the last DGE slot")
  could not be reproduced; the correct statement is **"keep the lowest-addressed slot, drop the
  trailing queues"**, the DGE analogue of iter-5 §1's queued-DMA ring-prefix rule.
* **And the one legal merge is a 1.8 us REGRESSION** (47.80 → 49.59; 4 of 5 captures in the
  slow bimodal mode). Collapsing two tiles into one queue makes the single completion later and
  forces both consumers to wait for it. **Do not redo the tile merges.**

## 2. THE THIRD FINDING: an Activation `CAST` is NOT safe to move across an `ACTIVATION_TABLE_LOAD`

iter-2/iter-5 recorded "30, 128, CAST and COPY are table-agnostic". That is true for *picking a
table*, but not for *crossing a load*:

| edit | result |
|---|---|
| `60, T2load, 62, 61` (CAST 62 after the load) | **bit-identical** |
| `60, 62, T2load, 61` (CAST 62 before the load) | `output_0` off **3823%** |
| CAST 60 moved above the load | **bit-identical** |

Slot 60 is `bf16 → bf16`; slot 62 is `fp32 → bf16`. And `p16` closed the other half: moving the
**T0 load itself** from slot 10 down to just before the `func 9` op at slot 22 — so that slots
11–21, which are all `ACTIVATE func 128` (`copy_1p`, bf16 → FP32), run with **no table active at
all** — gives `output_0` off 5245%.

> **Working rule: every dtype-converting Activation op (`CAST` *or* `ACTIVATE`, including
> `func 128` = copy) reads the active table set. The only op proven safe to move across an
> `ACTIVATION_TABLE_LOAD` is a same-dtype `CAST`.** iter-2's "30/128/CAST/COPY are
> table-agnostic" is true only for *choosing which set to load*, never for *reordering across a
> load*. This is the main thing standing between the next iteration and the 2.57 us in §5.

## 3. THE FOURTH FINDING: Act slots 55–58 are a per-row SCALE, not a copy (so don't delete them)

`activation_func = 128` is `copy_1p`, and these four read and write the same address, so they
look like deletable identities (2.55 us!). They are not: `scale_src = POINTER_IMMEDIATE` with
`scale_value_imm_ptr = 23920 / 23924 / 23928 / 23932` — **exactly the four fp32 words that
slot 53 (`ACTIVATE func 29`, rsqrt) writes**. So `dst = src * scale[c]`.

> **Before calling any `ACTIVATE` a no-op, print `scale_src`, `scale_value_imm_ptr`, `bias_src`
> and `bias_imm_fp32`.** `dst = func(src*scale + bias)` and both can be memory pointers.

Note also what iter-5's descriptor cut cost: these four now read at `step_elem 4` and take
**0.637 us each** instead of ~0.2 contiguous, i.e. ~1.7 us of Activation time was traded for
~3.0 us of DGE time. Now that Activation is 86% busy that trade is close to break-even, and
de-interleaving (§10 candidate 3) would buy it back.

## 4. THE FIFTH FINDING: Act slots 71/72's barriers are DVE's gate, not Activation's

`Act 71 = ES W[61]>=32 XW[62]>=32` and `Act 72 = ES W[71]>=32 XW[62]>=32` (p2 numbering) were
the binding gate for the silu cluster at 43.0 us. They are *not* protecting Activation's own
reads: they exist so that `Act 73`'s `s37 += 16` — which releases `DVE 91` and hence
`DVE 92/93/94` reading `p0 213760 / 214272 / 214784` — cannot fire before `SP` 34/36/46's
transposes have landed. Moving both barriers past the silu cluster **fails verify** (`output_0`
4343%).

What *is* legal: the four `ACTIVATE func 36` whose only input is the DVE gather of §6
(slots 74, 78, 79, 80 = the ops reading `8602368 / 8602880 / 8603392 / 8603904`) can be hoisted
**above** the barriers. Bit-identical, and it fills 1.2 us of the 1.63 us idle: **−0.60 us**.

## 5. Refuted / closed this iteration (do not redo)

* **SP must not have two DGE transfers in flight — this is now CLOSED, two hangs paid for it.**
  SP blocks at slots 19–21 (`W[67]>=16`, i.e. its own slot-18 completion) for 1.44 us before
  issuing slot 27, and that ripples all the way into Activation's 42.7 → 43.8 us gap (§7).
  Two ways of lifting that block both **pass `verify.py` and then hang the device**
  ("Failed to schedule execution on XU[1]" + 30 s model-stop timeout):
  - `p3a`: move the three `s68/69/70 += 16` adds to SP slots 39–41 and no-op SP 29's
    `W[68]>=16 XW[67]>=16`;
  - `p13`: a **pure relocation** of slots 19, 20, 21, 29 to after slot 38 with *every field
    preserved* (so no wait is weakened and no semaphore total changes).
  Since p13 changes nothing but the position, the hang is not iter-5 §8's "a wait passes early":
  **the compiler's `W[67]>=16` at SP slot 19 is there to stop SP from issuing slot 27 before
  slot 18 has completed, and that constraint is real.** Pool happily issues 11 back-to-back,
  but Pool is on `qPoolDynamic` (descriptors staged in an `SB_scratch` region, §2) while SP and
  Activation are on `*DynamicHW` queues. Working model: **an engine on a `DynamicHW` queue group
  can only have one transfer in flight at a time.** (Activation's 8 prologue DMAs are
  back-to-back but each uses a *different* queue of the group, 80..87; SP's 6 also use different
  queues, so if this model is right the limit is per group, not per queue — either way, do not
  try to overlap SP's.)
* **Merging / relocating Activation's 8 prologue DGE DMAs: closed by §2.** This is the one
  remaining ~2 us at the front (§7) and both routes to it are now known-dead.
* **Deferring Activation DMA slots 7,8 past slots 11/13** (so the first compute starts at
  18.8 instead of 20.8): bit-identical, 48.097 vs 47.802 — the delayed tile completions cost
  more than the earlier start buys.
* **Table loads cannot go below 7 without moving slots 52–58.** `act_info.json` is decisive:
  **no set contains both `silu` and `reciprocal_sqrt`** (`silu_and_others` = silu/sin/tanh/…,
  `reciprocal_sqrt_and_small` = reciprocal_sqrt/…). The five needed sets are
  T0 `softplus_and_others`(f9), T1 `natural_log_exp_and_others`(f7), T2 `silu_and_others`(f36),
  T3 `sigmoid_and_others`(f5), T4 `reciprocal_sqrt_and_small`(f29), and the order of use is
  T0 T1 T2 T3 T4 T2 T4 = 7 loads. The two reloads are forced because f29 is needed at ~36.5 us
  (slot 53 → the scale for 55–58, which PE reads at 45.2) and f36 again at ~43.
  **Correction to iter-5 §5: the loads ARE on the critical path now** — all 7 are immediately
  followed by the next Activation op with no idle, so each is worth a full 1.283 us if a legal
  reordering is ever found. The 5-load schedule needs slots 51/52/53/55–58 pushed past the
  second T2 block, which delays `PE 44 LDWEIGHTS` from 45.2 to ~50 — I judged that a net loss
  and did not test it. It is the single biggest unexplored item (2.57 us).

## 6. Semaphore-hop cost, and when moving work off Activation pays

Measured on this NEFF:

| edge | latency |
|---|---|
| DGE DMA completion → a waiter on its semaphore | **1.3–4.0 us** (bimodal, §11) |
| compute op's `SEM_ADD_IMM_COMPLETE` → a waiter on another engine | **0.05–0.2 us** |
| `ACT_TABLE_LOAD` | 1.283 us |
| Activation DGE issue 0.59–1.11 us; Pool 0.66–0.76; SP ~0.6 | |

That is why `p2` removed **1.96 us** of Activation work and moved the wall clock only **0.30 us**:
the 8 CASTs' consumer (`Act 82`) is 7 slots later, so the move inserts a hop, and the real
binder in that window turned out to be §4's barriers. **Rule: only move an op off Activation if
(a) its consumer is far away in Activation's program, or (b) the target engine can start it
materially earlier.** The recipe itself is cheap and works (DVE happily ran the `step_elem 4`
strided CASTs byte-for-byte), so the remaining ~20 Activation CASTs are still a lever — just
pick ones whose consumers are distant.

Mechanics that worked for `p2` (reusable):
1. clone the Activation slot's 64 bytes into `DVE0.bin` (insert before DVE slot 91 — the
   `ES W[37]>=16` that gates DVE's tail), clearing the event fields;
2. first clone of a group carries the gate (`W[58]>=32` for the merged transpose,
   `W[4]>=24` for "Act 65 has retired" — s4 is Activation's own instruction counter, count the
   `SEM_INC_COMPLETE[4]` slots to get the threshold);
3. last clone of a group carries `SEM_ADD_IMM_COMPLETE[free] += 16` (47 and 48 were free);
4. delete the Activation originals outright (they had no `SEM_INC_COMPLETE[4]`, so the counter
   does not shift) and repoint the consumer's wait to the new semaphore. `Act 82`'s `W[38]>=16`
   and `Act 92`'s `W[37]>=16` were **vacuous self-waits** (s38/s37 are supplied only by
   Activation's own slots 78/81), so repointing them is free — always check whether a wait's
   only producer is the same engine before preserving it.
5. Free semaphores after this iteration: **0,1,2,7,49–55,88–127** (47 and 48 are now used).

## 7. Where the time is now (~47.8 us; body 12.0 → 58.4, metric ≈ end − 14.0)

| window | us | what binds |
|---|---|---|
| 12.1 → 17.8 | 5.7 | Pool slot 1 (merged conv_weight) issue + its 3.7 us completion; **not editable** |
| 17.8 → 18.7 | 0.9 | PE slot 2 → DVE slots 4,5,6 |
| 18.7 → 20.8 | **2.1** | **Activation still issuing its 8 prologue DGE DMAs + the T0 table load.** `Act 11` is ready at 18.75 and starts at 20.83. Both routes to this are closed (§2, §5) |
| 20.8 → 22.7 | 1.9 | `Act 11` → DVE slots 8–19 (1.56 us of DVE work) → s6>=11 → `Act 15` |
| 22.7 → 36.5 | 13.8 | Activation dense head, 3 small gaps (1.55 / 0.53 / 1.00) |
| 36.5 → 37.7 | 1.2 | the s67 completion hop after SP 18 (§0) — intrinsic |
| 37.7 → 48.7 | 11.0 | Activation ~100% busy: 4 scale ops (2.55), 2 table loads (2.57), ~13 CASTs, the silu cluster |
| 48.7 → 50.0 | 1.3 | `W[5]>=22` — PE |
| 50.0 → 58.5 | 8.5 | the 4-head recurrence tail (COPY→f30+ZERO_ACCUMULATE→READ_ACC→f29, Act/PE/DVE serial), 1.95 us of gaps |
| 58.5 → 61.8 | 3.3 | Pool slot 70's 256 KB output store (0.94 issue + ~3.2 completion) + range clear; **not editable** |

### Ranked candidates for iter-7

1. ~~Audit the remaining `EVENT_SEMAPHORE` gates the way §0 did~~ — **I did this for all ten of
   Activation's gaps and they are now ALL real cross-engine data dependencies.** Don't redo it;
   the per-gap findings are:
   | Activation gap | binding gate | verdict |
   |---|---|---|
   | 21.4→22.7 (1.28) before slot 15 | `W[6]>=11` = DVE 19 (input-2 tile COPY) | real, see candidate 5 |
   | 23.9→24.5 (0.53) before slot 22 | `W[6]>=18` = DVE 32 | real |
   | 26.9→27.9 (1.00) before slot 27 | `W[6]>=24` = DVE 48, which ends at 27.8 | real (slot 28's `W[20]>=32` was ready at 25.7, it is *not* the blocker) |
   | 36.4→37.6 (1.21) before slot 55 | `W[67]>=16` = SP 18's own completion hop | irreducible (§0) |
   | 42.7→43.8 (1.07) before slot 78 | SP 46's completion → `s71` | closed (§5) |
   | 47.9→49.2 (1.30) before slot 88 | `W[5]>=22` = PE 52, 36 ns of slack | real (candidate 2) |
   | 4 tail gaps (1.97 total) | DVE/PE in the recurrence | real |
   The whole kernel is now one interlocked Activation↔DVE↔Pool↔PE chain: Act 37 → Pool 45/47/49
   → Pool's gather → Act 52/53 → DVE 76..83 → Act 55.. , with DMA completion hops (1.3–4 us) at
   three points. There is no slack left to reclaim by re-pointing waits.
2. **The 5-table-load schedule (2.57 us) — the biggest single item left, fully scoped below.**
   The 7 loads are all on the critical path (§5) and the schedule that needs only 5 is:
   ```
   34, 45(T3), 46(f5), 50, 35(T2), 36..44(f36), 47,48,49, 59, 61, 62(f36), 63..74(f36),
   75,76,77, 78,79,80, 51(T4), 52, 53, 54, 55..58, 82, ...        # delete slots 60 and 81
   ```
   Everything about it has been checked against §3's table rule except one thing:
   - slots 47/48/49 move T3 → T2 but are **same-dtype CASTs** (safe);
   - slot 50 is a **converting** CAST so it must travel with T3 (hence its position above);
   - slot 59 moves T4 → T2 but is bf16→bf16 (safe); 61, 78, 79, 80 stay under T2; 82+ stay
     under T4; 52/53/55–58 stay under T4.
   **I measured the PE slip and it is NOT free — read this before starting.** `s5` thresholds map
   to PE slots as `{18:43, 19:45, 20:47, 21:50, 22:52, 23:54, ...}`, and the 45–50 us stretch is
   a fully tight Activation↔PE ping-pong with **zero slack**:
   ```
   Act 80 CAST 44.81-45.10 -> PE 44/45 (W[4]>=27) 45.16-45.46
   Act 82 CAST 46.42-46.71 -> PE 46/47 (W[4]>=28) 46.76-47.06
   Act 83 CAST 46.74-47.03 -> PE 49/50 (W[4]>=29) 47.51-47.70
   DVE  ...              -> PE 51/52 (W[6]>=53) 48.89-49.16
   PE 52 retires 49.162  -> Act 88  (W[5]>=22) 49.198      <-- 36 ns of slack
   ```
   Slots 55–58 would move from 37.7–39.5 to ~46.5–48.0, and `PE 44/46/55/61 LDWEIGHTS` read
   their output (`243064 + 2c`), so PE 44 cannot run until ~48 and the whole ping-pong — and
   `Act 88`, and the entire tail — slips with it. Rough net: +2.57 saved, ~0.8–1.5 given back.
   **Plus a second obstacle I did not see at first:** slots 52 and 53 both carry
   `SEM_INC_COMPLETE[4]`, so moving them renumbers Activation's instruction counter for every
   slot in between (breaking `PE 36 W[4]>=19`, `DVE 76 W[4]>=20`, `PE 44 W[4]>=27`, …). Leaving
   `EVENT_SEMAPHORE`s behind to keep the count fires those increments *earlier than the data*,
   which is the iter-5 §8 hazard (and `DVE 76` really does wait for slot 53 — it runs at 36.57,
   0.13 us after slot 53 retires). So this needs the s4 thresholds recomputed, not placeholders.
   My judgement: worth attempting only with a fresh budget and only after finding a way to keep
   slots 55–58 early (e.g. compute the rsqrt scale for them from a *different* function that
   lives in `silu_and_others`, which would remove the need for T4 at 37 us entirely).
3. ~~De-interleave the merged transpose so slots 55–58 read contiguously~~ — **tried (`p14`),
   neutral (47.803 vs 47.60), reverted. See §8 — it is the lesson, not a candidate.**
4. **Keep moving Activation CASTs off, but only distant-consumer ones (§6).** ~20 remain,
   ~5 us of Activation time; Pool is idle 44.2 → 57.2 (12.9 us!) and DVE 38.6 → 43.5.
   Candidates with no semaphore complications left in the tail: the three 128×128 fp32→bf16
   CASTs reading `115976 / 116488`. **Caution: their destinations (`21256 / 22280 / 23560`) are
   read by PE `LDWEIGHTS`, which carries no visible wait (iter-5 §0).** Verify with the
   `operands` grep before moving any of them.
5. **The 1.28 us gap before `Act 15`** is the only remaining front slack and it is one hop wide:
   `Act 11` (20.83–21.21) → `DVE 8..19` (21.18–22.74, 1.56 us of DVE work) → `s6>=11` →
   `Act 15` (22.68). Nothing is idle in that chain, so the only lever is making `Act 11` start
   at its ready time of 18.75 — which needs 1.5 us out of Activation's prologue, and §2/§5 close
   every route I found. If someone finds a legal way to delete ~2 of Activation's 8 prologue
   DGE DMAs (or to run `Act 10/11/13` before them), it is worth ~2 us. One untried, cheap test:
   move the T0 `ACT_TABLE_LOAD` to *after* slot 21 so slots 11–21 (all `ACTIVATE func 128`) run
   with no table active. Likely breaks for §3's reason, but it is a single verify.
6. **Do not bother with:** any DGE DMA relocation, semaphore reassignment, or tile merge (§2);
   overlapping SP's DGE transfers (§5, two hangs); deferring Activation's prologue DMAs (§5);
   moving Act 71/72's barriers (§4); deleting Act 55–58 (§3); de-interleaving for cheaper
   ACTIVATEs (§8); splitting or moving the output store (§9); the epilogue trigger chain
   (0.09 us, §9); hoisting a dtype-converting CAST over a table load (§3);
   `dma_configs_dma_transpose_2d_tile`.

## 8. THE SEVENTH FINDING: for a run of same-shaped Activation ops you buy the ISSUE RATE, not the duration

`p14` did exactly iter-6-review-candidate-3: four DVE CASTs de-interleave the merged transpose
from `242048 + 2c` (stride 4) into four contiguous 256-byte rows at `244096 + 256c`, Activation
slots 55–58 then scale in place with `step_elem [1,1,1]`, and `PE 44/46/55/61` go back to
`src = 244096 + 256c + 254, step_elem [-1,0,-32], num_elem [32,1,4]` (iter-4's pre-n4 values,
scaled to the new base). **Bit-identical, and completely neutral.** Why:

```
                       per-op duration    start-to-start    slots 55-58 span
interleaved (stride 4)   0.667 / 0.637        0.39 us        37.64 -> 39.48
contiguous   (p14)       0.492 / 0.477        0.40 us        38.59 -> 40.24
```

**The four ACTIVATEs overlap each other** — the next one starts 0.39 us after the previous one
starts, whichever layout is used. Shortening each instruction by 0.16 us bought nothing because
the engine was already issuing at its own rate, while the DVE de-interleave + its semaphore hop
pushed Activation's gate from 37.59 to 38.54 (+0.95). Net: slightly worse.

> **Two rules out of this.** (1) Use the *union* of instruction intervals to decide how much of
> an engine is reclaimable — `sum(duration_ns)` over-counts Activation by 5.1 us, DVE by 4.3 and
> PE by 3.7. (2) Before "making an op cheaper", check the start-to-start spacing of its
> neighbours in `Instruction.parquet`: if the ops are already pipelined back-to-back, only
> *deleting* one (or removing a gate) moves the wall clock.

This also explains `p2` (1.96 us of CASTs moved off Activation → 0.30 us of wall clock) and
means the §7 estimate of what is reclaimable from Activation should be read off the gaps
(10.1 us of real idle), not off the 44 us duration sum.

## 9. THE EIGHTH FINDING: a DGE DMA's completion report is ~2.5 us of LATENCY, not bandwidth — and the tail is irreducible

`Pool 70` is the kernel's big store: `src SB 114952, elem 512, num [4,128], step [512,262144]`
→ `dst table_index 9 off 0, step [65536,512]` = **256 KB**, gated on `W[6]>=60`. The four 512-byte
chunks are written by `DVE 103 (114952, s6=49)`, `DVE 104 (115464, s6=50)`,
`DVE 119 (115976, s6=57)` and `DVE 123 (116488, s6=60)`, i.e. chunks 0/1 are final ~10 us before
the store is issued. `p15` split it into two 128 KB DMAs — chunks 0,1 on `W[6]>=50` and chunks
2,3 on `W[6]>=60`, both on Pool DGE queue sem 8 — plus `Pool`'s end-of-program
`NOP W[8]>=64 → >=80` (s8's total is exactly 64 = Pool 2 DGE + Pool 17 ES + Pool 34 CAST +
Pool 70 DGE, so **that NOP is precisely "the last store has landed" and its threshold must be
raised by 16 per added part**). Bit-identical, and **neutral**:

```
                 store issue ends   Pool's s42 barrier   delta
whole 256 KB         57.36                59.88          2.52
half 128 KB          57.96                60.50          2.54
```

Halving the bytes did not change the delay at all, so the ~2.5 us is a fixed notification
latency. Nothing to win by splitting or by moving bytes around.

And the tail is genuinely finished: `Act 118` (58.12) → `DVE 125/126` (58.35–59.01) →
`SP 76` (`W[6]>=63`, 59.03) → 4 no-op ESs → `SP 81`'s trigger of `qSPIO0` block 2, the 1 KB
`output_0` store (59.26–59.43) → `s42` at 60.50 → `Pool`'s barrier → ~1.4 us of runtime
teardown. Every edge there is already tight; iter-5's candidate 4 (hoisting the trigger above
the four no-op ESs) is worth 0.09 us. **Treat 58.4 → end as fixed.**

## 10. Bimodality (unchanged from iter-5 §8, now diagnosed one level deeper)

Captures are still bimodal: the slow mode is ~+4.8 us and shows up in roughly 1 capture in 5.
In the slow capture the cause is visible — **`SP` slot 18's DMA completion report takes ~4.0 us
instead of ~1.4 us** (SP 18 ends 36.22, `s67` lands 40.6 instead of 37.6), and everything
downstream shifts. Hypothesis worth one experiment: at 35.8–36.4 there are three Pool DGE
transfers in flight (Pool 45/47/49, issued 32.6–34.5), so SP 18's completion notification may be
queueing behind them; nudging Pool's three earlier or later might remove the collision and is
worth ~1 us of *median* plus much stabler measurements.

Practical rules: **always run `--repeats 5` and look at `max`**, treat anything under ~0.4 us as
noise, and remember **`profile.py --parquet` writes only the LAST capture — check its `end`
timestamp (fast ≈ 60.9–61.9, slow ≈ 66.4) before trusting any gap analysis.** I lost one round of
reasoning to a slow-mode parquet.

## 11. Reproducing

Scripts take the unpacked root as `argv[1]`. From a fresh unpack of the iter-5 NEFF:

```
p1.py  work                      # Act 54: W[28]>=16 -> W[67]>=16            (-2.37)
p2.py  work                      # Act CASTs 68-71,74-77 -> DVE (sems 47,48) (-0.30)
p3c.py work                      # Act CAST 60 hoisted above the T2 table load
p5.py  work 74,78,79,80          # the 4 silu ACTIVATEs hoisted above Act 71/72's barriers (-0.60)
```
then `pack` → `semaphores.py` → `verify.py` → `profile.py --repeats 5`.
Refuted: `p3.py work a` and `p13.py` (both hang the device), `p3b.py work 62` (CAST over table
load), `p4.py` (barrier move), `p6.py`/`p7.py` (DGE DMA relocation), `p8.py m4`/`p10.py`/`p11.py`
(illegal merges), `p9.py work 7 8 1024 -768 2` (the only legal merge — regresses 1.8 us),
`p12.py work 13` (deferring DMA issue), `p14.py` (de-interleave, neutral),
`p15.py work 0:2:50,2:2:60` (split output store, neutral), `p16.py` (T0 table load moved late).

**Check order after every edit: `semaphores.py` → `pack` → `verify.py` → `profile.py
--repeats 5`.** Three device hangs this iteration (`p3a`, `p13`, and iter-5's lesson repeated):
`semaphores.py` was clean for all of them and `verify.py` *passed* two of them. Only the
profiler catches them. Keep the last good `kernel.neff` next to the unpacked tree
(`cp work-good.neff kernel.neff` is the whole rollback) because a hang costs ~2 min of device
timeout and you will want to get back to a known state fast.

Analysis helpers rewritten this iteration (they did all the real work):
`d.py ROOT ENG [lo hi]` compact per-engine decode with waits/updates/addresses;
`full.py ROOT ENG a,b,c` every field of chosen slots, differences marked;
`waiters.py ROOT sems` every wait/update/DGE-sem/queue-block touching a semaphore set;
`semuse.py ROOT` which semaphores are used and which are free;
`gaps.py prof [thresh]` per-engine busy time and every gap;
`tl.py prof [ENG]` per-engine timeline.
The three decisive moves were: `gaps.py` to find the idle window, `waiters.py` to find what its
gate really was, and `json.load(def.json)['dma_queue']` to find out why DMAs cannot be moved.

### an/p1.py — §0, the whole −2.37 us
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
x=isa.decode(bytes(act[54*64:55*64])); a=x.as_dict()
assert x.name=='EVENT_SEMAPHORE' and a['events_wait_idx']==28 and a['events_semaphore_value']==16
assert a.get('events_update_mode') in (None,0,'NONE')      # iter-5 §2: never no-op an ES that updates
x.set('events_wait_idx',67)        # SP 18's merged transpose, not qSPSpillReload0 block 4's dummy
act[54*64:55*64]=x.encode(); open(ap,'wb').write(act)
```

### an/p2.py — §6, the CAST→DVE recipe
```python
GRP1=[74,75,76,77]   # strided CASTs 240000+2c -> 8602368+512c  (gate s58>=32, signal s47)
GRP2=[68,69,70,71]   # 1-elem CASTs 19968/20480 -> 114944+2k     (gate s4>=24,  signal s48)
blob  = clone(74, wait=(58,32)) + clone(75) + clone(76) + clone(77, upd=(47,16))
blob += clone(68, wait=(4,24))  + clone(69) + clone(70) + clone(71, upd=(48,16))
dv[91*64:91*64]=blob                                  # before DVE 91 (its ES W[37]>=16)
for slot,old,new in ((82,38,47),(92,37,48)):          # both were vacuous Activation self-waits
    repoint(slot, old, new)
delete Activation slots 68,69,70,71,74,75,76,77       # none carries SEM_INC_COMPLETE[4]
```
`clone(i, wait, upd)` copies Activation slot `i`'s 64 bytes, clears every event field, then sets
`WAIT_FOR_SEM_GE_IMM`/`SEM_ADD_IMM_COMPLETE`. `events_semaphore_value` is shared between the
wait threshold and the add amount, so put the wait and the add on *different* slots.

### an/p3c.py + an/p5.py — §3 and §4
```python
# p3c: Activation order ... 58, 60, 59(T2 ACT_TABLE_LOAD), 62, 61, 63 ...
#      CAST 60 (bf16->bf16) may cross the load; CAST 62 (fp32->bf16) may NOT.
# p5:  hoist slots 74,78,79,80 (ACTIVATE func 36, accumulator_cmd IDLE, no update,
#      srcs 8602368/8602880/8603392/8603904) to just before slot 71 (ES W[61]>=32).
order=[i for i in range(len(act)//64)]
for m in (74,78,79,80): order.remove(m)
k=order.index(71); order[k:k]=[74,78,79,80]
```

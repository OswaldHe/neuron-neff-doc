# iter-2 — four levers, all cross-engine scheduling, none touching arithmetic:
# (1) three Activation TENSOR_SCALARs executed on the idle DVE engine,
# (2) a queued trigger hoisted out from behind a DGE descriptor row,
# (3) three staging DGE rows re-hosted from Activation onto idle Sync,
# (4) a late WAR guard rotated behind the two casts it was stalling.

## Result: **43.4 → 40.9 us (−2.5 us, −5.7%)**, outputs bit-identical

Interleaved A/B pairs, 7 captures each, back to back on the same device:
* iter-1's bytes vs config F: **43.433 vs 41.353** (−2.08) and **43.345 vs 41.437** (−1.91);
* config F vs the shipped bytes (H): **41.573 vs 40.839** (−0.73).
Ranges never overlap. The shipped bytes measured **40.859 / 40.839 / 41.137** (the last run late in
the session, when the device had drifted up ~0.3 us); iter-1's bytes **43.657 / 43.433 / 43.345**.
Quote **40.9**, not 40.84. Against the iter-0 baseline: 48.519 → 40.9, **−15.7% (1.186x)**.

| step | edit | median (7 captures) | Δ vs prev | verdict |
|---|---|---|---|---|
| — | iter-1's bytes, measured at session start (5 reps) | **43.657** | — | anchor |
| **A** | `ACT[11]`, `ACT[13]`, `ACT[16]` executed on DVE (§2) | **42.526** | −1.13 | **kept** |
| **B** | blk3 trigger `SP[9]` hoisted above the sem-58 DGE row (§3) | **42.112** | −0.41 | **kept** |
| C1 | sems 44/45/46 DGE rows → Sync, inserted **before** `SP[22]` | 42.670 | +0.56 | **REFUTED (§4.1)** |
| **C2** | same, inserted **after** the blk14 trigger `SP[23]` (§4) | **41.555** | −0.56 | **kept** |
| **D** | sem-45 and sem-46 rows swapped (§5) | **41.441 / 41.438 / 41.353 / 41.354** | −0.11 | **kept** |
| E | D + sem-44 row hoisted above `SP[22]`/`SP[23]` (§5.1) | 41.513 / 41.519 | +0.08 | **REVERTED** |
| **F** | the composite `S[14]` fence split: `Pool[15]` no-oped, blk17 given its own `ES(S[44]>=16)` on Sync (§5.2) | **41.346 / 41.342 / 41.573** | −0.01 | **kept** |
| G | F + the blk16 trigger pair hoisted above the blk15 trigger (§5.3) | 41.502 | +0.16 | **REVERTED** |
| **H** | `ACT[26]` (a WAR guard with a 1 us-late wait) moved behind the two casts it was stalling (§5.5) | **40.859 / 40.839 / 41.137** | **−0.73** | **kept (SHIPPED)** |

Every Δ in the table is against the *same-session re-measurement* of the previous config, not
against a remembered number — the device drifted ~0.3 us upward over the ~2 h session (config F
measured 41.342 early and 41.573 late; iter-1's bytes 43.657 early and 43.345 late).

`verify.py --runs 3`: PASS, all 3 runs bit-identical, `max_abs_err` **identical to the baseline to
every printed digit** after every kept step (output_0 6.119e-05 = 17% of the limit, output_1
2.864e-04 = 82%). `launch_rules.py` PASS. `semaphores.py` no over-wait at any step; **S[4] stays
41/41 and S[6] 66/66 through every edit**.
`neff_diff` vs iter-1: 4 of 34 members — `sg00/{Activation0.bin (9 slots), DVE0.bin 94→100,
Pool0.bin (1 slot), SP0.bin 37→42}`. sha256
`3cae70386f6d726e73e3567f793a45a69a634e90d8adcdd656cf7e0a5498eccb`.
Engine busy on the shipped bytes: **Vector 20.28 (49.0%), Scalar 19.89 (48.1%)**, DMA 13.99 (33.8),
Tensor 11.94 (28.9), Sync 9.55 (23.1), Pool 9.09 (22.0).
**Scalar is no longer the busiest engine — Vector is.** That changes which lever is next (§8).

---

## 0. The single most useful thing: read the *sibling's late* notes, not just its early ones

`neff_doc/experiments/gdn_cte_b1_s256_d128_lnc1/logs/notes/iter-6.md` and `iter-7.md` contain the
two levers that produced everything in this iteration, and iter-1 of **this** kernel had not used
either (it only mined the sibling's iter-1/iter-2):

1. **An Activation `TENSOR_SCALAR_ARITH_OP` / `CAST` slot is byte-compatible with DVE** — same
   struct `v2.NEURON_ISA_TPB_S3D3_TS_STRUCT` / `common.NEURON_ISA_TPB_S4D4_TR_STRUCT` — so a raw
   64-byte slot copy executes on either engine, **including `imm_src = POINTER_IMMEDIATE`**, which
   the compiler never emits on DVE. Device-verified again here (3 more ops). `ACTIVATE` is
   Scalar-only; Pool only runs `TENSOR_SCALAR_AFFINE_SELECT`.
2. **The `EVENT_SEMAPHORE` stand-in.** Replace the vacated Activation slot with
   `ES{wait S[new]>=1, update S[4] SEM_INC_COMPLETE}` and the `S[4]` count sequence is *byte*
   identical, so no waiter anywhere needs renumbering — and the ordering is enforced by
   construction, not by timing.
3. **Fence the consumer's engine, not the engine you are relieving** (iter-7 §3). Doing it the
   other way puts the stall straight back.

The sibling's `dead2.py` lever (duplicate same-engine instructions) is **dry on this kernel**: the
only same-(engine, opcode, operands) duplicates are `LDWEIGHTS`/`MATMUL` pairs with real
intervening writes to the PE array. Ran it on the shipped bytes too — still empty. 10 minutes, worth
re-running after any reorder, but do not budget more.

## 1. Rebuild these four scripts first (~35 min, and they found everything)

`scripts/` is not preserved.

* **`cpath.py`** — backward critical-path walker. For each instruction the two candidate
  predecessors are (a) the previous instruction on the same engine **sorted by `start_ts`**, and
  (b) the first `SemaphoreUpdate` timestamp with `value >= threshold` for each `S[n]>=v` found in
  the **`operands` text** (regex `S\[(\d+)\][^>=<]*>=\s*(\d+)`); the later one binds. Two things
  make it not dead-end:
  - a **DGE** semaphore resolves by searching the instruction table for `semaphore=<n>` in
    `operands` (the `DMA_DIRECT2D` row); print `row_end → semaphore landing` as the *flight*;
  - a **queued** semaphore resolves through `DmaPacketAggregated` (`semaphore_id`, `trigger`,
    `start_ts`, `end_ts`), and the trigger instruction is the row whose `[start_ts, end_ts]`
    brackets `trigger`. **A queued trigger is reported by the profiler as opcode `WRITE`**
    (`data=0x4 size=4 dst=0x801002084038`), not as anything with "TRIGGER" in the name — that cost
    me 10 minutes.
* **`occ.py`** — union occupancy per 1 us bucket per engine (merge intervals inside the bucket,
  do not sum durations). This is what points at the window to attack.
* **`scan.py`** — "who touches this address": regex the hex address over the `operands` column of
  every instruction. **Do not mask the top bits**: an address is `partition<<18 | offset`
  (0x1004d00 is partition 64 offset 0x4d00, a *different* buffer from 0x4d00), DGE operands print
  as `0x80200<addr>`, and PSUM is `>= 0x2000000`.
* **`d2.py`** — raw-byte/field dump of a stream (`isa.iter_stream`), printing
  `events_{wait,update}_{mode,idx}`, `events_semaphore_value`, `semaphore`, `count_union_block_id`
  and every `*_addr_addr_immediate` in hex.
* **`reorder.py`** — ~50 lines, and it found edit H (§5.5, −0.73 us) in one pass. For every
  update-less slot, compare the landing time of **its own** wait with the landing time of the waits
  of the next few slots on the same engine; if a later slot's gate lands earlier, the update-less
  slot is stalling it. Read `events_*` from the *bytes*, timing from the parquet. 34 hits on the
  shipped bytes, 33 of them real fences — §5.6 lists the three false-positive shapes.

Slot byte layout, re-confirmed: byte 0 opcode, byte 1 `0x10` (= 16 words), **byte 4 wait mode
(0x05 = `WAIT_FOR_SEM_GE_IMM`), byte 5 wait idx, byte 6 update mode (0x13 = `SEM_INC_COMPLETE`),
byte 7 update idx, bytes 8–11 wait threshold**; extended wait at offsets 32/33 + 36. The pure
no-op is `a0 10` + 62 zero bytes. **Always re-decode what you built and assert the decode**
(`scripts/edit.py:es()` does).

## 2. Edit A — three Activation `TENSOR_SCALAR`s on DVE: **−1.13 us**, the biggest single win

`occ.py` on iter-1's bytes: **Scalar is 90–96% busy from 5.3 to 9.9 us (kernel-relative) while
Vector is 20–30% busy**, and `cpath.py` runs straight through `Scalar[5..19]` → `SP[7]` (sem-58
DGE) → trigger → 2.3 us queued flight → everything else. 2.4 us of that stretch
(`ACT[11]`…`ACT[19]`) is **pure Activation program order** — every wait in it is already satisfied.

Moved (all `TENSOR_SCALAR_ARITH_OP`, `imm_src=PTR,INSTRUCTION`, 128 elem × 64 channels):

| ACT slot | dur | src / imm ptr → dst | guards needed on DVE | new sem |
|---|---|---|---|---|
| `ACT[16]` | 386 | `0x37e00` (sem 9) / `0x101c100` (sem 11) → `0x1004700` | `S[9]>=16` + `S[11]>=16` + `S[4]>=1` (WAR vs `ACT[2]`'s read) | 47 |
| `ACT[13]` | 327 | `0x1004a00` (`S[6]#11`) / `0x1004310` (`ACT[7]`) → `0x1004210` | `S[6]>=11` + `S[4]>=3` (covers `ACT[7]`, **and** the WAR against `ACT[6]`'s read of the overlapping `0x1004208`) | 48 |
| `ACT[11]` | 384 | `0x1037d00` (sem 12) / `0x101c000` (sem 10) → `0x1004d00` | `S[12]>=16` + `S[10]>=16` | 49 |

DVE insertions (6 slots, 94 → 100): `[es(S[11]>=16, x S[4]>=1), copy16]` before old `DVE[13]`
(runs in the 5.35–6.41 us Vector hole), and `[es(S[10]>=16), copy11, es(S[4]>=3), copy13]` before
old `DVE[18]` (the 7.10–7.45 hole, then straight after). Vacated Activation slots:

* `ACT[11]` → `ES{wait S[12]>=16}` — **keep the wait**: `ACT[12]`'s source `0x1037e00` is in sem
  12's span and the compiler was relying on `ACT[11]`'s wait for it. A bare no-op here is a race.
* `ACT[13]` → `ES{wait S[48]>=1, update S[4]}`, `ACT[16]` → `ES{wait S[47]>=1, update S[4]}` —
  so `S[4]#6` and `S[4]#8` still mean "that op completed", and `Pool[12]` (`S[4]>=6`) and
  `ACT[19]` (`S[4]>=8`) need no change.
* `SP[5]`, the sem-56 DGE that reads `ACT[11]`'s output, gets an inserted `ES{wait S[4]>=5}`
  in front of it (preserving the `ACT[12]` guard `SP[6]` used to inherit) and **its own wait
  retargeted `S[4]>=6 → S[49]>=1`**, i.e. onto the real producer. Fence on the consumer's engine.

**A DVE TENSOR_SCALAR of this shape runs 254–288 ns where Activation takes 327–386** — same
~25% as the sibling measured. `S[4]#10` landed 9.850 → 8.930, the sem-58 DGE row and the blk3
trigger followed, metric −1.13 (predicted −0.9).

### 2.1 What makes a candidate real (the checks that mattered here)
* **The op's own wait is not its only input.** Every one of these three needed a *second* guard
  that Activation program order had been providing for free. Resolve `src`, `imm0_imm_ptr` **and**
  the destination's WAR separately, with `scan.py`.
* **A DGE input load's landing time is the cap.** `ACT[12]` looks like the same kind of candidate
  but `S[12]>=16` only lands at 6.23, by which time Vector has no hole — moving it would have
  *cost* 0.3 us. Print `first timestamp with value >= 16` for every input-load semaphore before
  picking the set (here: sem 9 → 3.49, 10 → 4.29, 11 → 5.00, 12 → 6.23, 13 → 6.63).
* **Vector program order is often a *stronger* guard than what the op had on Activation.** `copy16`
  overwrites `0x1004700..0x10048ff`, which `Vector[11]` reads at 5.08 — inserting at DVE slot 13
  makes that ordering structural. Same for `copy13` vs `Vector[12]`'s MEMSET of `0x1004208`.
* **`ACT[9]` is NOT a candidate** even though it is 388 ns and update-less: `ACT[10]` is gated by
  `S[6]>=10` (a Vector op) and would not start any earlier, so `ACT[9]` is running inside a hole
  Activation would otherwise waste. Check the *next* instruction's gate, not just the candidate's.

## 3. Edit B — **a queued trigger was sitting behind a 0.6 us DGE descriptor row: −0.41 us**

`SP[9]` triggers blk3 (var8's SB→SB reload, sem 27) and waits `S[6]>=13`, which landed at 9.046.
It fired at **9.631** because `SP[8]` (the sem-58 DGE row, 601 ns) is in front of it in Sync program
order. A plain **swap of `SP[8]` and `SP[9]`** (no insertion, no renumbering — a trigger has no
semaphore update) fired it at 9.2 and moved the 1.14 us all-engine idle gap behind it down to 0.5.

Safety: blk3 writes SBUF `to_off` 227072 (p0+0x37700) and 4421408 (p16+0x37720) — read the span out
of `SP0.json`, not the `#comment` — while the sem-58 DGE *reads* `0x4d00`. Disjoint, so the only
thing the swap removes is an ordering that was never a dependency.

> **Generalisable and cheap: for every `PSEUDO_DMA_TRIGGER`, compare its fire time with the landing
> time of its own wait. Any slack means a row or a fence in front of it on the same engine is the
> real gate, and triggers are free to move as long as the transfer's own source/destination ranges
> are disjoint from what sits between.** This is `cpath.py`'s `[dma ...]` / `[dge ...]` lines.

## 4. Edit C2 — **three staging DGE rows moved off Activation onto idle Sync: −0.56 us**

At 26.6–29.4 us Activation runs four 607–670 ns `DMA_DIRECT2D` descriptor rows (sems 43, 44, 45, 46
— the var11/var12 staging saves iter-1 created) **interleaved with** the four compute slots that
carry the critical path:

```
ACT[56] row sem43 | ACT[57] row sem44 | ACT[58] TS | ACT[59] row sem45 | ACT[60] ACT | ACT[61] row sem46 | ACT[62] | ACT[63]
                     path:  ACT[58] -> ACT[60] -> S[4]#31 -> DVE[83] -> ACT[62] -> ACT[63] -> S[4]#32 -> DVE[85] -> S[6]#53 -> blk17 trigger
```

`ACT[58]`'s gate `S[5]>=57` landed at 26.674 and it ran at 28.056 — **1.4 us of pure row blockage.**
So `ACT[57]`, `ACT[59]`, `ACT[61]` became pure no-ops and their 64-byte slots were inserted verbatim
into `SP0.bin` (same `PSEUDO_DMA_DIRECT2D` struct on both engines; `semaphores.py` then reports
`dge:SP` instead of `dge:Activation`). The two rows that had no wait of their own
(`ACT[57]`, `ACT[61]`) got the guard they had been inheriting from Activation program order:
`S[6]>=49` (the last writer of p112+0xe000..0xe7ff) and `S[6]>=50` (p112+0xe800..0xefff). Sync is
idle 12.7 → 27.1, so the rows actually issue **earlier** than before.

### 4.1 Where to insert them — this is the whole difference between +0.56 and −0.56
I first put them in front of `SP[22]` (the `ES(S[6]>=51)` that opens Sync's trigger block). Verified
bit-identical, and the metric got **0.56 us worse**. Why, from the two profiles:

| | before `SP[22]` (C1) | after `SP[23]` (C2) |
|---|---|---|
| `S[4]>=31` (`ACT[60]`) | 28.66 (**−0.55, as designed**) | 28.66 |
| `S[4]>=32` (`ACT[63]`) | **31.04 (+0.89)** | 29.5 |

`ACT[63]` waits `S[38]>=16`, and sem 38 is blk14, **triggered by `SP[23]`** — which the three
inserted rows pushed out by 1.4 us. The relief I bought on Activation was paid back with interest
through a queued transfer on Sync.

> **Rule: before inserting a 0.65 us DGE row into a stream, list every `PSEUDO_DMA_TRIGGER` that
> comes after the insertion point on that engine and check whether anything downstream of the
> instruction you are relieving waits on that trigger's semaphore.** Sync is "idle", but its idle
> slots are load-bearing: its trigger block is a chain of `ES(ring guard)` + `trigger` pairs that
> the whole kernel hangs off.

## 5. Edit D — row order inside the burst, and what it is worth
Three rows on one engine land in issue order plus a flat ~1.7–2.1 us flight. Their consumers need
them in a different order: sem 44 first (`Pool[15]` → `Pool[16]` → `S[14]` → the blk16 trigger),
then sem 46 (`SP[32]`'s fence), then sem 45 (`SP[33]` → `S[60]`). Reordering the last two to
**44, 46, 45** moved `SP[32]` 30.77 → 30.24 and cost nothing: **−0.11 us**, reproduced
(41.441 / 41.438 vs 41.555). Small, kept because the mechanism is right and it does not regress.

### 5.1 Edit E — REFUTED: hoisting the sem-44 row above `SP[22]`/`SP[23]` is a wash
sem 44 is the binding input of `S[14]` → blk16 trigger → 1.95 us flight → `S[24]>=32` →
`Tensor[128]`, so issuing its row 0.65 us earlier looked free. It did exactly what it was designed
to do — in `profE` the two mid-kernel idle gaps waiting on `S[45]`/`S[46]` (1.01 + 0.33 us) are
**gone** — and the metric was **+0.08** (41.513 / 41.519 vs 41.441 / 41.438, both reproduced).
The row again displaced `SP[23]` (blk14 → sem 38 → `ACT[63]`), and a new 0.85 us gap opened waiting
on `S[39]` (blk15, whose trigger is gated by `ACT[63]`). **Same lesson as §4.1 from the other side:
on this kernel Sync's early slots are worth more as trigger-issue latency than as DGE-row capacity.**

### 5.2 Edit F — **the composite `S[14]` fence split: −0.9 us of local latency, −0.01 on the metric**
`Pool[15]` is `ES(S[44]>=16)` (no update) and `Pool[16]` is `ES(S[43]>=16) → S[14]++`, so **Pool
program order made `S[14]` mean "sem 43 *and* sem 44 landed"**. `SP[29]`, the blk16 trigger, waits
`S[14]>=1` but only needs sem 43 (it reads p64+0x1d200); the sem-44 consumer is blk17 (`SP[31]`,
reads p64+0x1d300), which was inheriting the guard through Sync program order behind `SP[29]`.
So: `Pool[15]` → pure no-op, and an explicit `ES(S[44]>=16)` inserted on Sync in front of `SP[30]`
(blk17's ring guard). Nothing else waits on sem 44 and no Pool slot ≥15 reads p64+0x1d300 — checked
with `scan.py` over the whole instruction table and over every `SP0.json` `to_off`/`from_off`.

Device: `S[14]` 29.70 → **28.83 (−0.87)** and every downstream landmark moved −0.35
(`S[24]>=32` 31.67→31.32, `Tensor[128]` 31.71→31.36, `Vector[87]` 32.07→31.71,
`Vector[94]` 35.95→35.56); all-engine idle **6.73 → 6.29 us**; profile span 42.17 → 41.91.
**And the metric did not move** (interleaved pair: D 41.354 vs F 41.342). Kept anyway because it is
a strict tightening (blk17 now states its own dependency instead of inheriting one), it is
device-verified bit-identical over 3 runs, and it removes a wall iter-3 would otherwise hit.

> **The metric does not track the profile span 1:1 on this kernel.** profD span 42.17 → metric
> 41.35; profF span 41.91 → metric 41.34. A −0.26 us span improvement showed as −0.01.
> `total_exec_time` is a vendor-computed field in `neuron-explorer`'s `summary-json` (also in
> `Summary.parquet`), **not** `last_end − first_start`; the same profiles report `total_time`
> 55.74 / 55.86 us (the whole capture including the ~12 us barrier preamble). So span − metric was
> 0.81 for profD and 0.56 for profF. **Do not substitute the span for the metric, in either
> direction**: edit F looked like a −0.26 us win on the span and measured 0, while edit H
> (§5.5) was −0.73 on both. Use the span to confirm an edit did what you designed, and the metric
> to decide whether to keep it.

### 5.3 Edit G — REFUTED: **firing a queued trigger earlier can make its own transfer land later**
`SP[29]` (blk16's trigger, gate `S[14]>=1` landing 28.83) fired at 29.48, held by Sync program order
behind `SP[27]` (blk15's trigger, gate `S[4]>=32` landing ~29.3). Textbook §3 hoist: move the pair
`[ES(S[24]>=16), blk16 trigger]` ahead of `SP[27]`, each keeping its own waits; blk15 and blk16
write disjoint SBUF (p0/p64+0x4500 vs p0+0x4600) so it is correct, and it verified bit-identical.

**Both transfers landed 0.32 us *later*** (`S[24]>=32` 31.32 → 31.64, `S[39]` 31.04 → 31.37) and the
metric went +0.16. **All 21 queued transfers share `queue_idx 3`** (iter-1 §5.6), so firing blk16
first does not give it a free ride — it just changes the queue order, and blk15 (2 descriptors, and
the one `Scalar[64]` waits on) ends up behind it instead of in front.

> **Rule: a trigger hoist only pays when what you hoist it over is an *engine-side* row or fence
> (§3: a 0.6 us DGE descriptor row). Hoisting it over another trigger on the same queue is
> zero-sum at best, because the single shared queue serialises the transfers either way.**

### 5.4 Closed by inspection: `sem 24`'s two blocks are not two concurrent transfers
Semaphores 24–28 are each reused by exactly two blocks — `{24:[0,16], 25:[1,17], 26:[2,18],
27:[3,19], 28:[4,20]}` — and each block contributes **16** increments regardless of its descriptor
count. So `S[24]>=32` is "blk0 (done by 6.5 us) **and** blk16", i.e. a plain single-hop wait on
blk16, not a join of two late transfers. Do not go looking for a second laggard trigger.

### 5.5 Edit H — **the second-biggest win of the iteration, −0.73 us, for a 3-slot rotation**
`ACT[26]` is `ACTIVATE COPY 0x1004200 → 0x37900`, update-less, with wait `S[31]>=16` landing at
**13.46**. The two slots behind it, `ACT[27]` and `ACT[28]` (`CAST` PSUM → `0x4300` / `0x4340`),
have gates `S[5]>=7` / `S[5]>=8` that land at **~13.45**, and `ACT[28]` produces `S[4]#11`, which is
the gate of `Tensor[18]` — the head of the whole 14–27 us Tensor/Vector pipeline. Activation was
reaching `ACT[27]` only at 14.36 because `ACT[26]` burned 0.3 us *after* a 0.6 us stall.

`ACT[26]`'s wait is genuinely needed — **blk7 reads p0+0x37900 (`from_off` 227584) and `ACT[26]`
overwrites it, so `S[31]>=16` is a WAR guard** (iter-1 §5.5's pattern) — but it is needed *by
`ACT[26]`*, not by anything behind it. So the slot was rotated to `[27, 28, 26]`, carrying its wait.
Numbering-neutral (`ACT[26]` and `ACT[27]` update nothing, `ACT[28]` keeps `S[4]#11`).
Checked before packing: blk7 writes `0x5200`/`0x5400` and reads `0x37900`/`0x37980`, while
`ACT[27]`/`ACT[28]` read PSUM and write `0x4300..0x437f` — no overlap either way, so they never
needed blk7's completion.

> **This is a distinct and cheap lever from §2 and §3: a *late WAR guard* parked at the head of a
> group. The reorder scan (`reorder.py`, §5.6) finds them in one pass — an update-less instruction
> whose own gate lands later than the gate of the slot(s) behind it on the same engine.**

### 5.6 The rest of the reorder scan: 34 hits, 33 false positives
`reorder.py` (§1) on the shipped bytes reports 34 candidates. **Every other one is a real fence**,
and the three shapes of false positive are worth knowing so iter-3 does not re-walk them:

* **`ES(sem) + the op that reads that transfer's output`** (`Tensor[141]`+`[142]`, `Tensor[118]`,
  `Tensor[121]`, `Scalar[22]`/`[24]`, `Vector[27]`/`[31]`, …). The fence *is* the next op's gate.
* **`LDWEIGHTS` → `MATMUL`**: data-dependent through the PE array, not through a semaphore.
* **A guard whose semaphore belongs to a *different* engine than the queue names suggest.**
  `Sync[22]`'s `ES(S[6]>=51)` in front of the blk14 trigger looks like a trivially-satisfied ring
  guard; it is not — `S[6]#51` is `DVE[81]`, which **reads p0+0x4200, blk14's destination**. It is
  the WAR guard for blk14 and must stay in front of the trigger.
* Two that look real and are not: **`ACT[7]`** (update-less, gate `S[10]>=16` at 4.29, runs 6.03)
  and **`ACT[9]`** both sit in holes where the *next* instruction is gated by a Vector semaphore, so
  rotating them gains exactly 0 — and `ACT[7]` would additionally break edit A's `copy13` guard
  (`S[4]>=3` = `ACT[8]` currently implies `ACT[7]` completed). Worked both out on paper.

## 6. Where the remaining 40.8 us goes

Span 41.36 us (kernel-relative; the profile has ~12 us of runtime barrier preamble before
`SP[0] SET_ORDERING_MODE` — anchor on that). **5.81 us of all-engine idle in 7 gaps > 200 ns, every
one of them DMA flight**, and no engine above 49% (shipped bytes, `profH`; Vector 20.14 us,
Scalar 19.90, DMA 14.30, Tensor 11.93, Sync 9.48, Pool 9.20):

| gap us | at | waiting for |
|---|---|---|
| 0.46 | 10.81 | blk3 (var8 reload) — 1.07 us slow scatter + 0.3 propagation |
| 0.25 | 29.12 | sem-46 DGE (`SP[33]`) |
| **0.84** | 29.41 | nominally `SP[34]` on `S[45]`; really **Tensor waiting blk16's `S[24]>=32`** and Scalar waiting blk15's `S[39]` |
| 0.21 | 30.36 | blk15 / `S[39]` → `ACT[64]` |
| 0.40 | 34.08 | blk18 / `S[26]>=32` |
| 0.69 | 36.40 | blk19 / `S[27]>=32` |
| **1.37** | 39.88 | the sem-17 output store: 0.66 us row on Sync + **1.35 us flight** → final fence |

Note the 0.60 us blk7 gap that used to sit at 12.86 is **gone** — edit H (§5.5) let the two casts
run during blk7's flight instead of behind it.

Plus a **2.33 us DRAM→SBUF flight** at the very head (`SP[1]`, the gate load, sem 8 → `DVE[8]`),
which is the first instruction of the kernel and has nothing to overlap with.

## 7. Closed this iteration (do not re-spend budget)

* **`dead2.py` / duplicate-instruction dead-code scan: empty on this kernel**, before and after.
* **`ACT[9]`** — update-less and 388 ns, but its successor is gated by a Vector semaphore, so the
  slot is free (§2.1).
* **`ACT[12]`** — a correct ACT→DVE candidate whose input semaphore (12) lands too late for any
  Vector hole (§2.1).
* **Inserting DGE rows in front of Sync's trigger block** (§4.1, measured +0.56).
* **Hoisting the sem-44 DGE row above the blk14 trigger** (§5.1, measured +0.08 twice).
* **Swapping `Pool[15]` and `Pool[16]`** to make `S[14]` land ~0.8 us earlier: **that swap is a
  race.** `Pool[15]` is `ES(S[44]>=16)` with no update and `Pool[16]` is `ES(S[43]>=16) → S[14]++`;
  Pool program order makes `S[14]` mean "sem 43 **and** sem 44 landed", and blk17 (`SP[31]`) reads
  p64+0x1d300 = sem 44's destination with no guard of its own other than being behind `SP[29]`'s
  `S[14]>=1`. `semaphores.py` cannot see this at all. The *correct* version of this edit is §5.2
  (give blk17 an explicit `S[44]>=16` and no-op `Pool[15]`) — shipped, and it buys the 0.87 us
  locally even though the metric did not move.
* **Reordering Sync's 9-trigger burst `SP[11..19]`** — already optimal: sems land 12.85, 13.24,
  13.44, …, 14.73 in issue order and the first consumer (`Tensor[15]`, `S[28]`) is the first one
  triggered.
* **Hoisting a queued trigger over *another queued trigger*** (§5.3, measured +0.16). One queue
  for all 21 transfers means that is zero-sum.
* **A queued transfer with a DRAM side**: there is no example in this NEFF to copy from — all 26
  descriptors are SB→SB and the engine `.json` carries no queue metadata (§8.2).
* Everything in iter-1 §7 still stands.

## 8. Ranked candidates for iter-3

0. **Re-run `reorder.py` (§1, §5.5, §5.6) on the new bytes before anything else — 15 minutes.**
   Every edit this iteration moved gates around, so the scan's output changes. One hit out of 34
   was worth 0.73 us and cost a 3-slot rotation. The discriminator that makes it cheap: a hit is
   real only if the stalled slots do **not** depend on what the stalling slot's wait protects —
   resolve that with `scan.py` on the transfer's `from_off`/`to_off` versus the stalled slots'
   `src`/`dst`/`imm0_imm_ptr`.
1. **Vector is now the busiest engine (20.14 us vs Scalar 19.90) — the direction of the §2 lever has
   flipped.** `occ.py` on the shipped bytes shows Vector at 83–93% from 16 to 26 us (kernel-relative)
   while Scalar sits at 24–51% there. The sibling's iter-7 §4 found Vector→Activation *worthless* in
   its own Vector-bound window **because every Vector op there was chained through `S[5]`/`S[6]`
   waits**; check that first on this kernel (read `evt_wait_time_ns`: only move an op whose own wait
   is ~0 out of a stretch whose neighbours' waits are also ~0). If the window is genuinely program-
   order-bound, the recipe in §2 runs in reverse — a `CAST` with `in_dtype == out_dtype` is a `COPY`,
   and `S4D4_TR` is byte-compatible both ways. Note `op0=MIN` does not appear in `Activation0.bin`,
   so check ALU support before moving a `TENSOR_SCALAR MIN`.
2. **The 1.37 us tail** (§6, unchanged from iter-1 and the sibling's iter-6 §9.1, still untried).
   It is the single largest idle block left — 24% of the remaining 5.81 us.
   Tail of config F: last compute ends 39.74 → `SP[41]` sem-17 DGE row 39.76–40.43
   (**662 ns**, `src 0x4c00`, `[262144,1][128,1]`, wait `S[4]>=41`) → **1.37 us flight** →
   `Pool[21]` fence (`S[17]>=16` ∧ `S[16]>=16`) 41.79 → `Pool[22]` NOP 41.91.
   sem 16's store (`Pool[19]`, 38.99–39.68) lands ~40.8, so **sem 17 is the laggard by 1.0 us**.
   Converting the row to a queued trigger costs ~50 ns instead of 662: append a block to
   `SP0.json["dma"]` and replace the row with a trigger. **I checked the blocker so iter-3 does not
   have to: all 26 descriptors in `SP0.json` are `{"from":"SB","to":"SB"}` — there is no DRAM-side
   descriptor anywhere in this NEFF**, and `SP0.json`/`Pool0.json` carry only
   `{instr, name, dma}` (no queue sizing), so the DRAM address would have to be expressed as an
   offset with no variable reference. That is the unknown, and it is why I did not attempt it with
   50 minutes left. Expect only the row, not the flight (sibling iter-7 §6). Hang-prone — do it
   first in the iteration with a `cp save/kernel.neff kernel.neff` ready.
   **Cheaper variant worth trying first:** split the 128-partition store into two 64-partition rows
   with **different** semaphores on two different hosts (Sync + Activation, both ~25 ns reaction),
   and give `Pool[21]` the second semaphore as its extended wait. The sibling's iter-6 §3 says two
   halves on the *same* semaphore serialise in one queue (the second row costs 1.4x), but on
   different semaphores they go to different queues. The row cost is ~650 ns regardless of payload,
   so this only wins if the 1.37 us flight is payload-dependent — measure
   `DmaPacketAggregated.duration_ns` for this transfer before building it.
3. **The `ACT[56]` sem-43 row (670 ns) still blocks `ACT[58]`.** It is the one of the four that
   stayed on Activation, deliberately: its semaphore feeds `Pool[16]` → `S[14]` → the blk16 trigger,
   and `S[14]` is the binding input of the biggest remaining mid-kernel gap. Moving it to Sync would
   make it the 4th row there (sem 43 lands ~2 us later ⇒ `S[14]` ~2 us later); moving it to Pool
   pays Pool's 0.4–0.5 us wake-up (sibling iter-6 §3b) and iter-1 §5.1 measured that as a loss.
   The only version worth a run is **Pool, inserted *before* `Pool[14]`** (Pool is idle 9.7→27.1 so
   the wake-up is unavoidable either way), which also needs `Pool[14]`'s consumer `DVE[82]` checked.
   Ceiling ~0.5 us, probability ~40%.
4. **blk3's 1.07 us scatter** (iter-1 §6.1) — still a data-layout change, still low probability.
5. **Double-buffer to break a WAR**, iter-1 §5.5's candidate 2, still untried and still needs an
   SBUF free-region map from `SbufUsage.parquet`. The live instances in the shipped bytes:
   `SP[27]` (blk15 trigger) waits `S[4]>=32` only because blk15 *overwrites* p0+0x4500, which
   `ACT[63]` reads; `SP[40]` (blk20) waits `S[4]>=38` for the same reason. Each is worth ~0.6 us
   and the fix is a different `to_off` plus a repointed consumer address.
6. **Done this iteration, do not redo:** splitting the composite `S[14]` fence (§5.2 — shipped);
   hoisting the blk16 trigger over the blk15 trigger (§5.3 — refuted, +0.16); looking for a second
   late transfer on `sem 24` (§5.4 — there is none).

## 9. Mechanics confirmed or learned
* `compiler_pc` **is** the post-edit slot index: after inserting 6 slots into `DVE0.bin` and 4 into
  `SP0.bin`, the profiler reported the new indices (`Vector[18]` was old `DVE[16]`). Only
  `nki_source_location` goes stale. Inserting slots stays free — nothing in the NEFF records an
  instruction count, and `neff.py pack` recomputes sizes and both MD5s.
* The profiler reports an `EVENT_SEMAPHORE` **out of program order** relative to the compute slot
  next to it (`Scalar[17]` printed between `Scalar[14]` and `Scalar[15]`). Sort holes by `start_ts`,
  and never conclude "the engine executed out of order".
* A DGE's flight (row end → semaphore landing) is **1.4–2.1 us for these 64 KB SB→SB staging
  transfers** and 2.33 us for the DRAM gate load; a queued transfer's trigger→start is ~0.7 us flat
  and its propagation 0.3–0.5 us. `DmaPacketAggregated` (21 rows: `trigger`, `start_ts`, `end_ts`,
  `block_id`, `queue_idx`, `semaphore_id`) is still the fastest way to split a gap into
  start-up / transfer / propagation.
* Free semaphores after this iteration: 0, 1, 7, 18–23, 40, 41, 42, 50–55, 61–63 (47/48/49 are now
  used by the three DVE copies; **2 is the runtime barrier — do not touch**).
* Device cost this session: 10 verifies (0 failures, every run bit-identical), 17 profiles, no hang,
  no silently wrong output. Keeping `save_{base,A,B,C,C2,D,E,F,G,H}.neff` made every revert one `cp`
  and made each of the three refutations (§4.1, §5.1, §5.3) cost about 4 minutes.
* Budget shape that worked: ~35 min rebuilding tools and reading the sibling's late notes, then
  eight 6-minute cycles of (analyse one window → one edit → `semaphores.py` → `verify --runs 2`
  → `profile --repeats 7 --parquet`). A verify is ~1 min, a 7-capture profile ~1.6 min, a
  parquet ingest ~0.5 min more.
* **Decide with an interleaved A/B pair, not with a remembered number.** The device drifted ~0.2 us
  across this session; the two refutations (+0.56, +0.08) and the 0.11 us edit D were each only
  readable because the comparison config was re-measured in the same few minutes.

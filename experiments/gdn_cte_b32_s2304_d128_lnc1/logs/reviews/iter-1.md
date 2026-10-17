# Review — iter-1

Verdict: **accept, but do not score it as a win.** The edit is behaviourally exact, the notes are
accurate (I re-derived every structural claim independently), and the iteration's real product is
a set of refuted hypotheses, not the −0.12%. One of the notes' conclusions is wrong, and its
"what I would do next" #1 is contradicted by the profile — see §4.

Measurements below are my own: a fresh `--repeats 1 --parquet prof_iter1` capture of the edited
NEFF, plus the agent's pre-existing `prof/` which I verified is a **baseline** capture
(`NeffHeader.hash = e78a3f0f` = baseline payload md5). Having both let me compare the edited and
unedited structure directly, which turned out to matter.

---

## 1. What changed, and does it match the notes?

Exactly what the notes claim. `neff_diff`: 3 of 34 members changed.

| member | A | B | slots differing |
|---|---|---|---|
| `sg00/Activation0.bin` | 30789 slots | 30789 (same size) | 2784 |
| `sg00/Pool0.bin` | 7584 slots | 8976 (+1392) | 8938 |
| `sg00/SP0.bin` | 17364 slots | 18756 (+1392) | 18735 |

All 2784 `PSEUDO_DMA_DIRECT2D` (DGE) slots on Activation became `EVENT_SEMAPHORE`
(`v3.NEURON_ISA_TPB_PSEUDO_DMA_DIRECT2D_STRUCT` → `v2.NEURON_ISA_TPB_CTRL_ES_STRUCT`), and the
DGE slots reappear verbatim in Pool/SP (e.g. Activation slot 78's payload tail is now Pool slot
38; the rest of the Pool stream is shifted down). Confirmed independently by the profiler's own
instruction counts: `scalar_engine_instruction_count` 30812 unchanged, `gpsimd` 7608→9000
(+1392), `sync` 17418→18810 (+1392).

The S[72] handshake is real and correctly built. `semaphores.py` on both unpacked trees differs in
exactly two ways: the 16 DGE sources on S[40..55] change from `dge:Activation` to
`dge:Pool`/`dge:SP` (2784 increments each, unchanged), and one new row appears:

```
  72      2784      2784    2784  {'Activation:EVENT_SEMAPHORE': 2784}
```

`max_wait == total_inc == 2784` and the waiter count equals the increment count — no over-wait, no
under-count. No semaphore anywhere lost an increment or a waiter. The notes' recipe (placeholder
keeps the original wait verbatim and adds `SEM_INC_COMPLETE`/`events_update_idx=72`; the moved slot
waits `S[72] >= n`; `semaphore`/`sem_increment` carried over unchanged) is what is actually in the
binary. Notes and artifact agree.

## 2. Is the latency change real?

**No — it is below the loop's own noise floor, and the loop should not be scored on it.**

| | median | min | max | spread |
|---|---|---|---|---|
| baseline | 14778.976 | 14778.273 | 14779.613 | 0.009% |
| iter-1 (agent, 3 rep) | 14760.853 | 14755.495 | 14767.632 | 0.082% |
| iter-1 (my 4th sample) | 14754.986 | — | — | — |

−0.12% (−18 µs). The stated threshold is ~0.5% (≈74 µs); this is a sixth of it. Four independent
samples of the edited NEFF all land below the baseline's minimum, and the agent's own re-measure of
the *unedited* NEFF came in at 14785.5 µs — above the recorded baseline — so the effect is probably
a real ~15–25 µs, not pure capture noise. But "probably real and 0.1%" is still noise by the rule
the loop was given, and the drift between captures of an identical binary (+0.03–0.05%, per the
notes) is the same order as the signal.

To the agent's credit, the notes say this plainly ("Below the 0.5% claim threshold, so **I do not
claim a win**"). The risk is not the agent's honesty, it is the harness: `profile_latest.json`
records `change_percent_vs_baseline: -0.12` and the loop scores on `total_exec_time_us`, so this
will be banked as an improvement regardless. **The loop is at risk of chasing noise** — not because
this iteration was dishonest, but because a −0.12% result is indistinguishable from drift and will
look like progress.

(My `--repeats 1` run overwrote `profile_latest.json`; I restored the agent's 3-repeat file.)

## 3. Is the edit safe beyond the verifier?

**No synchronisation was loosened.** This is the unusual case where the transformation is provably
conservative, and I checked the argument rather than taking it:

- `S[72] >= n` is satisfied only after placeholder *n* completes, which requires *n*'s wait (the
  moved DGE's original wait) to have cleared. Activation is in-order, so it also implies
  placeholders 1..n−1 completed. Each moved DGE therefore fires **no earlier** than it did before —
  strictly later, never sooner. No consumer wait changed: all ~2435 consumers still wait on
  S[40..55] at the same values.
- `semaphores.py` exits clean on the edited tree and flags nothing. (Its only check is
  `max_wait <= total_inc`; it cannot see capacity deadlock — the notes say so too.)

**The cross-execution reuse hazard is real in principle and I closed it empirically.** The whole
scheme depends on S[72] being zeroed between executions, and `verify.py` *cannot* test that: it
calls `D.capture(...)` with the default `num_exec=1`, so every verify run is a fresh model load.
Only `profile.py` runs 5 executions per load, and it checks no numbers. So if S[72] carried over,
executions 2–5 would have every `S[72] >= n` wait trivially satisfied, the moved DMAs would fire
unordered, and the verifier would never notice — while the *profiled* execution (the 5th) would be
the broken one. I checked the profiled 5th execution directly in `SemaphoreUpdate.parquet`:

```
ts 43262  S[72]  value 1      sem_inc     Scalar
...
ts 14735219  S[72] value 2784 sem_inc     Scalar
ts 14768392  S[72] value 0    sem_wr_imm  Scalar   <- epilogue clear
```

S[72] runs 0→2784 in the 5th execution and is then cleared. The clear is **pre-existing**: in the
baseline capture S[72] has exactly one row, that same `sem_wr_imm 0`. The agent picked a semaphore
the existing epilogue already zeroes, which is the right choice. Hazard closed.

**Error moved nowhere.** `max_abs_err` is bit-identical to baseline on both outputs
(1.372e-04 and 6.841e-04). output_1 sits at 86% of the 8e-4 limit, but that is a pre-existing
property of the baseline, not something this edit consumed. Outputs were bit-identical across 3
runs. No tolerance headroom was spent.

### The two genuine residual concerns

1. **Capacity, not ordering.** The agent *demonstrated* that this edit class can hang: all 2784 on
   Pool gives `NRT_TIMEOUT`. 1392/1392 works, but the limit is unexplained, so the accepted artifact
   sits an unknown distance from a cliff whose failure mode is a **hang**, not a wrong number — and
   `semaphores.py` provably cannot see it (it passed on the version that hung). Reproducibly fine
   here; worth stating as a standing caveat.
2. **It consumed Pool/SP slack and added head-of-line pressure.** SP busy 2776→3488 µs, Pool
   2751→3626 µs (genuine sequencer occupancy, ~610–630 ns per inserted DGE, matching the notes'
   fit). More specifically, each inserted DGE slot now *blocks* its new in-order stream until
   Activation reaches its placeholder, and **350 of the 1392 inserted SP slots sit immediately
   before a `PSEUDO_DMATRIGGER`**. The number of critical spill-reload triggers blocked behind a
   `DMA_DIRECT2D` predecessor rose 226→375. It cost nothing this time because SP/Pool had 11 ms of
   idle, but it is a real cost to later iterations, and the insertion rule should simply never place
   a moved DGE immediately before a trigger WRITE. Cheap fix, no downside.

The deadlock-freedom argument is, incidentally, sounder than the notes claim ("provably cannot
deadlock for one insertion"). The true data-dependency set is a subset of the observed
happened-before set, so ordering insertions by "ended before this DGE originally started" places
every real predecessor earlier in *any* run, not just the observed one. Ordering is fine; capacity
is the open risk.

---

## 4. The main answer: the next edit

First, two corrections to the notes, because they change the target.

**(a) The notes' recommended next step is not supported by the profile.** Their #1 is reordering the
Activation stream, on the reasoning that "3209 µs of PE idle is producer-not-finished". That number
conflates *producer* with *compute producer*. Decomposing PE's block on `qSyncSpillReload0` by what
the gating trigger actually waits on:

| gating trigger waits on | n | PE block | per transfer |
|---|---|---|---|
| compute S[3..6] | 1384 | 2331 µs | 1.68 µs |
| **a DGE semaphore** | **299** | **3206 µs** | **10.7 µs** |
| ring S[24..39] | 336 | 457 µs | 1.36 µs |

61% of it is DMA-waiting-on-DMA, not compute. Only **578 µs** of PE's block is gated by an
Activation producer at all, and the reorder ceiling on it — `sum min(Q, block)` where Q is the
producer's measured ready-but-queued time — is **112 µs**, not ≥1 ms. Scaling the resolved 29% to
the whole set gives ~700 µs as a strict upper bound that still ignores the cost of displacing what
the producer jumps over, on an engine that is 52.8% busy. Activation reordering is not the lever.

**(b) One lever the notes assumed is now closed by measurement.** They wrote that the S[24..39] ring
flow-control waits "must not be touched". Correct, and also pointless: for the 2554 ring-waiting
triggers the median slack between the wait clearing and the instruction issuing is **21.7 µs** — the
ring wait was already satisfied long before the trigger got there, binding in only 214 of 2554
cases, total held 280 µs. Relaxing ring depth is the one genuinely dangerous edit available (ring
overflow passes once, corrupts later) and it is worth ≤280 µs. Dead, with a number.

### Recommended edit: reorder the head-of-line-blocked trigger WRITEs on **Sync/SP**

Chapter 14 lever **14.3-2, "start long-latency work earlier"**, using the verified r08 mechanic
(reorder independent instructions on one engine). Applied to SP, not to Activation.

- **Engine / member:** Sync (SP), `sg00/SP0.bin`.
- **What changes:** slot *position* only. **No field edit at all.** The moved `WRITE` keeps its
  `events_wait_idx` / `events_semaphore_value` verbatim, so it still cannot fire before its data is
  ready.
- **Why SP is the right stream:** I verified that **all 18756 SP slots have
  `events_update_mode = NONE`** — SP increments no semaphore that anything waits on (increments come
  only from Pool→S[3], Activation→S[4]/S[72], PE→S[5], DVE→S[6], and the DMA/DGE queues). So
  reordering SP requires **zero wait re-pointing anywhere in the NEFF**: nothing can observe SP
  program order except through the DMAs it triggers, which carry their own semaphores unchanged.
  That makes this the cheapest and safest reorder available. SP is also only 23.6% busy
  (3488 µs busy / 11282 µs idle), so displacing SP work is nearly free — the opposite of Activation.
- **Slots, ranked.** 437 trigger WRITEs are head-of-line blocked (own wait satisfied before the
  preceding SP slot finished) *and* their transfer blocks PE. Ceiling = `sum min(H, PE_block)`:

| slot (`SP0.bin`) | recoverable | blocked by slot | predecessor |
|---|---|---|---|
| 7063 | 12.6 µs | 7062 | `PSEUDO_DMA_DIRECT2D` (waits S[61]≥928) |
| 2935 | 12.3 µs | 2934 | `PSEUDO_DMA_DIRECT2D` (S[67]≥384) |
| 12325 | 12.1 µs | 12324 | `PSEUDO_DMA_DIRECT2D` (S[66]≥1616) |
| 10549 | 11.8 µs | 10548 | `PSEUDO_DMA_DIRECT2D` (S[60]≥1392) |
| 8190 | 11.7 µs | 8189 | `PSEUDO_DMA_DIRECT2D` (S[71]≥1072) |
| 2874, 1704, 18140, 8716, 9895, 539, 15774, 12265, 6410, 16966, … | 6.7–9.7 µs each | n−1 | `EVENT_SEMAPHORE` (pure wait) |

  Cumulative: **top 5 ≈ 60 µs, top 20 = 172 µs (1.17%), top 50 = 304 µs (2.06%), all 437 = 536 µs
  (3.63%)**. Blocking predecessors: `EVENT_SEMAPHORE` 267, `DMA_DIRECT2D` 135, `WRITE` 35.
- **Start with the top 5**: each is a single pairwise swap of a trigger WRITE with the one DGE slot
  in front of it. Per-swap safety check is cheap and local — the swap is order-independent iff the
  trigger's wait semaphore differs from the preceding DGE's `semaphore` field (it does: those five
  wait on ring S[25..30], the DGE slots signal S[60..71]) and the DGE's destination range does not
  overlap the transfer's source range. 5 slots, ~60 µs (0.4%).
- **Expected:** top 50 for ~300 µs ≈ **2%**; realistically 50–70% of the 536 µs ceiling once
  displacement is paid for, so **~1.5–2.5%**. That clears the 0.5% noise floor, which nothing else
  on the table does.
- **Free side-fix, same edit:** move the 350 inserted DGE slots that currently sit immediately
  before a trigger WRITE one slot later. Pure refinement of this iteration's own insertion rule.

### Where the rest of the time goes, and why I would stop after this

Span 14769 µs. ≥1 engine busy 90.9%; **all engines idle only 1349 µs, shredded into 15183 gaps with
a maximum of 3.9 µs** — there is no bubble to remove. The residue is a serial dependency cycle:
PE waits 7073 µs on the spill-reload queue, 2398 µs on DVE, 2023 µs on Activation; Activation in
turn waits 3980 µs on PE and DVE 4893 µs on PE. On PE's critical path the spill-reload cost splits
into **1391 µs of fixed trigger→start latency** (median 724 ns, p25 719 / p75 734 — a hard pipeline
constant, reproduced exactly from the notes), 746 µs of transfer time, and 3857 µs of upstream
dependency. Compute-gated triggers issue a median **17 ns** after their wait clears, binding in 75%
of cases — the compiler's waits are exactly tight, independently confirming the notes' refutation #2.

That 1391 µs is the only large block of pure overhead left, and the route to it is closed: the
transfers are hardware transposes, DGE cannot express them (`dge_op=2` is rejected at load,
`NRT_INVALID`), and removing a round trip from the chain means re-tiling, which is a compiler
decision not reachable by slot edits. **After the SP reorder I would expect the remaining time to be
irreducible at the NEFF level**, and I would spend the next iteration's budget confirming that
rather than looking for another lever. Every ms-scale candidate is now measured and closed:
occupancy (−1331 µs busy bought −18 µs), trigger wait tightness (0 ns), ring flow control (≤280 µs,
never binds), producer reorder (≤112 µs on Activation), coalescing (impossible — 3-D permutation
into 2 descriptor dimensions), DGE transpose conversion (rejected at load).

One caveat on my own method: I tried to confirm this with a backward longest-path walk over the
observed execution, and it did not converge cleanly (the 16-wide semaphore fan-out on
`qSyncSpillReload0` makes value→transfer attribution ambiguous, and my DGE-producer resolver is not
reliable). The conclusions above rest on the aggregate per-engine gap attribution, which I verified
reproduces on both the baseline and the edited capture — the three-way split of PE's block is
2324/3239/446 µs in the baseline versus 2331/3206/457 µs edited, i.e. **this edit was
structure-neutral**, which is also the cleanest explanation of why it bought only 18 µs.

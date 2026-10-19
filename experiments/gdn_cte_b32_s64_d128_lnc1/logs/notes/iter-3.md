# iter-3 — **no change shipped** (`kernel.neff` is bit-identical to iter-2's, 835.7 us).
# Seven candidate levers measured and **closed** (three of them built, verified bit-exact on
# device, and still slower), the cost model the last two iterations ranked candidates with is
# **wrong**, and two tooling facts in iter-1/iter-2's notes are **unsafe**.
# Read §0 (what costs time), §9 bug 2 (`start_ts` is not what you think) and §9 bug 1
# (`events_semaphore_value` is shared) before planning anything.

Start-of-iteration measurement of the inherited NEFF (`profile.py --repeats 5`):
**835.673 us**, captures 834.5 / 835.0 / 835.7 / 836.1 / 836.9, spread 0.29%.
Engine busy: DVE 438.7 (52.5%), ACT 346.5, PE 262.7, DMA 200.7, Pool 166.8, SP 34.4.
Final `kernel.neff` sha256 `e051ff8e6422fede470dce39673e02cf4a6af77a8e6b713e19c82682705351f4`
== `git show HEAD:kernel.neff`. `verify.py` and `launch_rules.py` pass (unchanged artifact).

---

# §0 THE COST MODEL. `EVENT_SEMAPHORE`s AND DMA TRIGGERS ARE FREE.

`Instruction.parquet` has an **`execution_unit`** column with two values, `compute` and
`sequencer`. Split this NEFF's 835 us span by it:

| engine | compute slots | compute us | sequencer slots | sequencer us | `profile.py` "busy" |
|---|---|---|---|---|---|
| Vector (DVE) | 2186 | **461.7** | 189 | 16.3 | 438.7 |
| Scalar (ACT) | 1475 | **375.3** | 371 | 68.5 | 346.5 |
| Tensor (PE) | 3791 | **626.5** | 360 | 34.0 | 262.7 |
| GpSimd (Pool) | 292 | 166.5 | 853 | 160.6 | 166.8 |
| Sync (SP) | 1 | 0.1 | 1514 | 242.6 | 34.4 |

`profile.py`'s busy number is the **union of instruction intervals**, and a sequencer slot's
interval sits *inside* the compute pipeline's — so **deleting `EVENT_SEMAPHORE`s, DMA
triggers or relays removes no wall time at all**. Measured, not inferred: I deleted **125 ACT
`EVENT_SEMAPHORE`s** (§2) and ACT busy went 346.5 -> 346.0 (nothing) while the wall went
835.7 -> **849.8 (+14)**.

Consequences for the review's and iter-1/2's candidate lists:
* Review candidate 2 ("merge dead ACT `EVENT_SEMAPHORE`s, ceiling 66 us") is **worth 0 us**,
  and so is review candidate 1 (retire Pool's 224 DMA_DIRECT2D *issue* slots) to the extent
  it is argued from issue time. Pool's `DMA_DIRECT2D` are `compute`-unit, so that one is not
  dead, but Pool's 158 us of `EVENT_SEMAPHORE` is.
* iter-1's "SP issue time is not a lever" generalises: **sequencer time is never a lever.**
  SP is 242.6 us of sequencer and 0.1 us of compute, which is why emptying it bought nothing.
* The only thing that can be removed for wall time is **compute-unit work**.

**Per-instruction compute cost, measured (compute us / compute slots):**
**DVE 211 ns, ACT 254 ns, PE 165 ns, Pool 570 ns.** DVE is the *fastest* elementwise engine
in this NEFF, which is why §1 below fails: moving an op from DVE to ACT costs 43 ns more than
it saves before any scheduling effect.

---

# §1 REFUTED: offloading DVE casts to ACT. **+8.5 us.** iter-2's -45 us does not reproduce.

iter-2's candidate 1 ("BUILT AND MEASURED AT -45 us but it RACES"). I built it **with the
sound WAR gate iter-2 prescribed**, and it verifies perfectly — and it is *slower*.

| NEFF | total_exec_time | DVE busy | ACT busy |
|---|---|---|---|
| inherited | **835.7** | 438.7 | 346.5 |
| 191 DVE casts hosted on ACT, **with** WAR gates | **844.2** (5 rep, spread 1.5%) | 399.7 | 398.2 |
| the same, **without** any gate (racy, profile-only) | **843.2** (3 rep, spread 0.2%) | 398.7 | 398.1 |

* **The gates cost 1.0 us, i.e. nothing** — so iter-2's -45 us was not "the gain the race was
  buying". I cannot reproduce it; the only differences from iter-2's build are 191 casts
  instead of 224 (I excluded the 33 `num_active_channels=64` ones, whose destination boxes
  overlap Pool's DGE writes) and the gates. Treat iter-2's 790.6 us as unexplained.
* **Why it loses**: ACT is on the latency chain (`:451/:453` ACT casts feed PE `:447/:449`),
  ACT costs 254 ns/op against DVE's 211, and the DVE stub still has to wait for ACT's copy.
  Balancing the two engines at 398 us each is *worse* than 438/346. **Do not rank candidates
  by "which engine is busiest"** — the wall responds to the chain, not to the max.
* The edit itself is correct and worth keeping as a technique. `verify.py --runs 3`: PASS,
  all 3 runs bit-identical, `max_abs_err` identical to the baseline to every printed digit.

**The WAR gate that makes an offload sound** (this is the reusable part; iter-2 could not
finish it). For a DVE slot at stream position p with nominal start time T and destination box D:
1. host copy = the slot verbatim (identical op, src *and dst* address), `events_update_mode`
   `SEM_INC_COMPLETE` on a **fresh semaphore 104** (a free index past the 103 in use; it needs
   no `def.json` entry), primary wait = `max(original S[6] wait, S[6] gate)`;
2. in front of it, one `EVENT_SEMAPHORE` carrying up to two gates (primary + extended):
   for each engine E != host, `S[sem_E] >= V_E` where `V_E` is the semaphore ordinal of the
   **last instruction of E that touches D and ended by T** (`end_ts <= T`, read/write boxes
   from the mem-pattern fields). If that instruction does not itself increment `sem_E`, use the
   first later one that does **and require its `end_ts <= T` too** (else drop the candidate);
3. the DVE slot becomes an `EVENT_SEMAPHORE`: primary = the original wait, extended =
   `S[104] >= k`, update `SEM_INC_COMPLETE` on the original update index (preserves S[6]'s
   total and order exactly; `semaphores.py` 1528/1528);
4. insertion point = the position the host engine held at time T (binary search on host
   `start_ts`), which is monotone in p. **Touchers on the host engine need no gate** — they
   necessarily sit before that position.
**Why it cannot deadlock:** every added constraint names an event that happened before T and
the slot is placed where the host was at T, so the nominal schedule is itself a witness that
the constraint set is acyclic. (This argument is the whole trick; it is what iter-1 §7 and
iter-2 §3 were missing when they hung.)
Slack available to the WAR gate alone, measured over the 191 casts:
`T - end_ts(last toucher)` is p10 1.7 us, **median 3.2 us**, p90 23.6, min 0.28 — so the WAR
gate itself has room and does not bind. Toucher engines: PE 137, DVE 49, ACT 36, none missing.

**And here is why the offload can never win, which §4-§5 of iter-2 did not see.** The +8 us is
not ACT's load: §7's probe P3 shows removing **31.7 us of ACT compute work is worth 1.7 us of
wall** (ACT has slack). The cost is that DVE's stub waits for ACT to *arrive at and finish*
the copy. I therefore built the fix — place the copy where ACT stood at **T' = the time the
last of its gates (WAR gates **and** the copy's own `S[6]` RAW wait) became satisfied**, which
is still acyclic by the same witness argument, so ACT could run ahead and the stub would be
free (`scratch/offload2.py`). **The run-ahead is zero: median 0 ACT slots earlier, p90 1,
max 1.** The reason is structural — a `:407/:417` cast's source is written by the DVE
instruction *immediately before it*, so `T' == T` and no host engine can start earlier than
DVE would have. **Moving an op whose producer is its own engine's previous instruction always
replaces 211 ns of in-stream compute with a cross-engine handoff.** Offload is dead for this
op class; it could only ever pay for an op whose producer is on *another* engine (the 288
PSUM-sourced `COPY`s qualify — see §8 item 1, where the stub is the remaining blocker).

# §2 REFUTED: deleting the 160 neutered ACT `EVENT_SEMAPHORE`s. **+14 us.**

iter-1 turned 160 ACT DGE saves into `EVENT_SEMAPHORE`s that only add 16 to `S[40..55]` to
tell SP's staging triggers "ACT's stream got here". That is also expressible as
`S[4] >= n4`, where `n4` is the ordinal of the last ACT slot before the ES that increments
`S[4]` — **provided nothing between them writes anything**, which holds for 125 of the 160
(the gap to that slot is 1-5 slots, and 123 of the 162 in-between slots are themselves ES).
That gate is *implied by* the original one, so it is strictly weaker and cannot deadlock.
Built it: 125 ES deleted (ACT 1823 -> 1698), 383 waits retargeted onto `S[4]` across
SP/Pool/DVE, and ACT's own 115 waits on `S[40..55]` simply **cleared** (an ACT slot waiting on
a value its own earlier slot produced is a no-op). Surviving ES absorb the deleted increments
so every surviving cumulative value is unchanged; `semaphores.py` clean, `verify.py` PASS with
baseline-identical errors.
**835.7 -> 849.8 us (+14.1, 5 rep, spread 0.15%), ACT busy 346.5 -> 346.0.**
Two findings: (i) §0 — the slots were free; (ii) firing the staging DMAs *earlier* costs 14 us.
Combined with iter-1 §1/§6 (firing them **later** costs +16/+20 us), the compiler's gate point
is a **local optimum in both directions**. Stop touching the staging-trigger gate.

# §3 REFUTED: hosting offloaded work on Pool. **HANG** (and it was already numerically dead).

Same edit as §1 with `--host Pool` (Pool is the only engine with real slack: 166 us compute,
655 us idle): `NRT_TIMEOUT`, `semaphores.py` clean. Cause is structural, not a bug in the
placement: **Pool's stream hosts iter-1's 320 transpose relays, which PE waits on**, so a gate
`S[5] >= V` on Pool closes the cycle Pool -> PE -> Pool. §1's acyclicity argument needs the
host's own timestamps to place the slot, and only 897 of Pool's 1121 slots have one (see §6),
so the insertion point is approximate and lands on the wrong side of the PE event.
**Pool cannot host gated work while it hosts the relay chain.** (iter-2 had already shown Pool
computes a `CAST` *wrongly*: `max_abs_err` 2.3e-2.)

# §4 REFUTED: merging adjacent DVE instructions into wider ones. **HANG, at one pair.**
### A `num_elem[1] > 1` access pattern on a compute instruction does not execute on this part.

This is review candidate 3's cousin and iter-2's candidate 3 (scanned there at "96 pairs,
~10-14 us"). The encoding is available and the merge is easy to make *sound*; the hardware
refuses it.

* **Funnel (DVE, 2351 slots).** 762 adjacent same-opcode pairs -> **637 with an identical
  signature** (every field equal but the three pattern start addresses and the event block;
  iter-2's 637 reproduces exactly) -> 221 encodable (see the step-width limit below) -> after
  requiring the second member's wait to have been satisfied **before the first member started**
  (iter-1 §5's sound rule, resolved from the profile: a wait `(sem, val)` is satisfied at the
  `end_ts` of the slot where that engine's cumulative count first reaches `val`) and
  non-overlapping greedy selection: **48 runs / 60 slots saved** (one run of 14, 47 pairs),
  i.e. ~12.7 us of DVE compute. Blocking reason for the rest: `wait_late` 254.
* **`step_elem` is signed 16-bit** (`docs/06` line 117, and measured: a step of 6301666 reads
  back as 10210 = 6301666 mod 32768), so only deltas under +-32767 elements are expressible.
  That is what cut 637 -> 221: partition-crossing deltas (262144 B) cannot be a step.
* **S[6] renumbering is mechanical and safe to do**: dropping a member's `S[6]` increment and
  remapping every wait on `S[6]` program-wide (old ordinal -> new; a wait that named the dropped
  member now names the merged slot, which completes both writes, so no wait gets weaker) is
  1570 field edits across all five engines and `semaphores.py` stays balanced. **No slot that
  waits on `S[6]` has a value-consuming update** (checked: 1574 waiters, all either
  `SEM_INC_*` or no update), so the shared `events_semaphore_value` field is not a hazard here.
* **The refutation.** 54 pairs -> hang. Then **1 pair, with `S[6]` unchanged (1528 -> 1528) and
  `0` waits renumbered** -> **hang** (`Failed to execute model: 10001, error: 5`, then model
  stop timeout; `verify.py` [FAIL] run 0, 70 s). So neither the renumbering nor the selection
  is at fault: **setting `num_elem[1] = 2` with a `step_elem[1]` on a DVE compute slot hangs
  the engine**, even though the struct (`common.NEURON_ISA_TPB_S4D4_TR_STRUCT`) has four dims
  and `docs/14` says patterns may use four. Consistent with the corpus: **not one of this
  NEFF's 2186 DVE / 1475 ACT / 3791 PE compute slots uses a dim above 0** — the compiler never
  emits one. If a future iteration wants this lever it must first find the field that enables
  dim 1 (suspect: an opcode variant, `docs/06` line 117 lists `TENSOR2D/3D/4D` as *separate*
  encodings of length 12/16/20, so the dim count may be a property of the opcode, not a
  runtime field, and writing a 4D struct's dim 1 may simply be out of contract).
* **Widening dim 0 instead** (plain contiguous extension, which the compiler does use and
  which probe4 proved costs ~148 ns of fixed overhead per instruction) is **encodable and
  surely legal, but there is almost nothing to widen**: only **13** of the 637 same-signature
  pairs are dim-0 contiguous in *all* their patterns (the others have an address gap in a
  source pattern). ~2 us. Dead.
* ACT is dead for merging for a different reason: 455 same-signature adjacent pairs, 323 have
  the second member updating `S[4]`, and **all 132 remaining have a real blocking wait** on the
  second member (0 mergeable under any rule).

# §5 REFUTED (statically): redundant `LDWEIGHTS`. **0 candidates**, as on the s128 sibling.

PE has 1742 `LDWEIGHTS` but only **2 distinct `(row_grp, col_grp)` pairs**, and no two
consecutive loads into the same group have the same `(src addr, num_elem, step_elem, dtype,
num_active_rows/cols, transpose_mode, fp32_mode, perf_opt)`. The stationary operand is always
genuinely different. Review candidate 3 is closed for this kernel too.

---

# §6 TOOLING: how to map a slot to its profile row (do not redo this)

`Instruction.parquet`'s **`pc` column is the slot index plus a per-engine constant**, and the
constant is the length of the runtime's preamble (`TENSOR_LOAD`/`DRAIN`/`NOTIFY`/
`COMPARE_BRANCH`/`EVENT_SEMAPHORE` rows that are not in the stream). Find it by maximising
opcode-name agreement over `k in range(0, 220)`; for this NEFF:

| engine (profile name) | offset | slots matched |
|---|---|---|
| PE (`Tensor`) | **61** | 4130 / 4130 |
| DVE (`Vector`) | **63** | 2351 / 2351 |
| ACT (`Scalar`) | **63** | 1823 / 1823 |
| Pool (`GpSimd`) | 58 | 897 / 1121 |
| SP (`Sync`) | 209 | 434 / 1329 |

PE/DVE/ACT are exact. **Pool and SP are only partially resolvable** — the profiler does not
report iter-1/2's hand-built `EVENT_SEMAPHORE` relays and `PSEUDO_DMATRIGGER`s under names
that align, so any placement rule that needs Pool or SP timestamps is working on a
forward-filled approximation (this is what made §3 unsafe). Sorting rows by `start_ts` and
aligning does **not** work (the sequencer and compute rows interleave): use `pc`.
Opcode-name normalisation needed: `TENSOR_TENSOR`->`TENSOR_TENSOR_ARITH_OP`,
`TENSOR_SCALAR`->`TENSOR_SCALAR_ARITH_OP`, `ACT_TABLE_LOAD`->`ACTIVATION_TABLE_LOAD`.
Also: `in_dtype`/`out_dtype` come back as **raw ints** from `ins[name]` but as enum names from
`ins.as_dict()` (6=BFLOAT16, 10=FP32); the same is true of `events_update_mode` (19 =
`SEM_INC_COMPLETE`). Use `as_dict()` for anything you compare by name.

# §7 WHERE THE TIME IS, AND WHAT EACH ENGINE'S WORK IS WORTH (new probe numbers)

DVE compute 461.7 us over a 835.7 us span is the biggest pool of removable work:
`TENSOR_TENSOR` 928 slots / 187.6 us, `COPY` 518 / 129.6, `CAST` 544 / 114.6,
`MEMSET` 163 / 23.5, `TENSOR_SCALAR` 32 / 6.2.

Ceiling probes run this iteration (`scratch/probe.py`: set `num_elem[0] = 1` on a selected
set — keeps the slot, its semaphore, its ordinal and its wait, so only the *work* goes.
Results are wrong; **profile only, never verify, never ship**). Against 835.7 us:

| probe | slots | DVE busy | ACT busy | span | wall per us of busy |
|---|---|---|---|---|---|
| P2: DVE `CAST` whose wait is on `S[6]` (the `:407/:417` coupling casts) | 223 | 438.7 -> 422.8 | — | **818.6 (-17.1)** | **1.07** |
| P5: DVE `COPY` with a **PSUM source** (the matmul-output evacuations) | 256 | 438.7 -> 406.7 | — | **808.5 (-27.2)** | **0.85** |
| P3: **all 896 ACT `CAST`s** | 896 | — | 346.5 -> 314.9 | **834.0 (-1.7)** | **0.05** |

**DVE work converts to wall time at ~0.85-1.07; ACT work converts at ~0.05.** That is the
single most useful number pair in this file: ACT has 32 us of pure slack and DVE has none.
It explains §1 (an offload to ACT should be free on ACT's side, and it is — the loss is the
handoff) and it says plainly that **only DVE compute work is worth removing**.
iter-2's probe4 numbers on the same scale, for comparison: all `COPY`+`CAST` -60.9 us, all
`TENSOR_TENSOR` -40.6, the 384 `:407/:417` casts -26.5.
(Trap, from iter-2 and confirmed: shrinking *all* DVE compute, or clearing PE's `S[6]` waits
at lines 407-419, dies with `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`. Shrink subsets.)

### P6: the floor. **Every removable elementwise instruction's work together is worth 87 us.**
Shrinking **2982 slots at once** — all SB-sourced DVE `COPY`/`CAST`/`TENSOR_TENSOR`/
`TENSOR_SCALAR` (1606) plus all ACT `CAST`/`ACTIVATE`/`TENSOR_SCALAR` (1376), i.e. 168 us of
engine busy time (DVE 438.7 -> 341.4, ACT 346.5 -> 272.8) — gives **748.8 us, only -86.9**.
So **835.7 us is ~90% instruction issue and cross-engine chain latency, and at most ~87 us of
it is elementwise work that could in principle be deleted.** The two priced targets (P5's -27
and P2's -17) are already about half of that 87, and §8 item 2 is the only mechanism left that
can collect any of it. A future iteration should read this number before planning: there is no
large win hiding in the compute, and nothing else this loop has found converts at all.

# §8 WHAT TO TRY NEXT

Everything below rests on the two facts this iteration established: **only DVE compute work
is worth removing** (§7: DVE converts at ~1.0, ACT at 0.05, sequencer slots at 0.0), and
**an op can only be hosted elsewhere if its producer is on another engine** (§1).

0. **~~Move the 288 PSUM-sourced DVE `COPY`s to ACT~~ — BUILT, CORRECT, AND REFUTED at
   847.1 us (+11.4). See §9: it is bit-identical to the baseline and still loses, because each
   moved copy makes ACT block on PE inside ACT's in-order stream. The whole "host DVE work
   elsewhere" class is closed. What §9 leaves behind is valuable though: the `S[6]`
   renumbering + conjunction machinery is now device-proven, which is the risky half of
   item 2 below.** The paragraph that follows is kept only for its recipe.
1. ~~Move the 288 PSUM-sourced DVE `COPY`s to ACT. Prize measured at -27.2 us (probe P5);
   ACT's side is free (P3); and unlike §1 these qualify, because their producer is PE.~~
   This is the one candidate where the §1 refutation does *not* apply: the 232
   `('COPY', upd S[6], wait S[5], PSUM)` + 56 `('COPY', no upd, wait S[5], PSUM)` slots are
   the matmul-output evacuations, gated on `S[5]` (PE), so an ACT host can start the instant
   PE's matmul lands — exactly when DVE would — and ACT has 32 us of slack to absorb them.
   Use §1's WAR-gate recipe verbatim (it verifies bit-identical over 3 runs).
   **The one blocker, and it is the whole job: the DVE stub must go away.** With a stub, DVE
   waits for ACT to finish and the win is ~0 (that is §1's +8 us). Every cheap way to keep
   `S[6]`'s numbering fails for the same reason — a semaphore is a counter with no identity,
   so if ACT produces increment *v*, DVE's next incrementing op has to be held back until it
   does, which re-creates the stall. So:
   * delete the DVE slot outright and **renumber `S[6]` program-wide** (§4's machinery:
     old ordinal -> new; 1570 field edits; `semaphores.py` stays balanced; no `S[6]` waiter
     has a value-consuming update, so the shared value field is not a hazard). §4's hang was
     proven to be the access pattern, **not** the renumbering, so this part is untested but
     not implicated.
   * a wait `S[6] >= v` means "producers 1..v", so after deleting copies it becomes
     `S[6] >= omap[v]` **and** `S[104] >= (number of deleted copies with ordinal <= v)`.
     That is a conjunction, and only `EVENT_SEMAPHORE` has an extended block, so insert one
     two-wait ES in front of each `S[6]` consumer and clear the consumer's own wait
     (~1574 of them: PE 748, ACT 385, DVE 315, Pool 67, SP 59). **Those ES are free** (§0:
     sequencer unit), which is exactly why this is affordable; iter-1 §2 built this pattern
     for 32 PE consumers and it ran bit-identical.
   * build it incrementally and profile at 16 / 64 / all copies — the renumbering is global,
     so a partial build is still a complete, shippable edit.
2. **THE ONE REMAINING LEVER: iter-2's candidate 2, re-priced at -17.1 us (probe P2, the 223
   `:407/:417` casts) — and the only kind of edit left that *removes* DVE work instead of
   moving it. `scratch/offload3.py` already contains the `S[6]` renumbering this needs, and
   §9 proved it correct on device.**
   Delete them by making `:464/:465` (`tensor_copy(dst=Wg_grp[...], src=yW)`, DVE PSUM fp32 ->
   SBUF fp32 `COPY`) a `CAST` to bf16, so `Wg[bj]`/`Ug[bj]` is already the contiguous bf16
   `[16,128]` run the `:408/:418` coupling matmuls stream. Four-part edit (producer dtype, the
   halved row stride, the PE matmul source addresses, the 160 `qSPSpillReload1` regroup
   descriptors — 32 of which already cast fp32->bf16 and become plain copies), and it needs
   item 1's `S[6]` renumbering anyway. Do item 1 first: it is worth more, it is purely
   mechanical, and it builds the renumbering this one needs.
3. **Do not spend budget on**: everything in §1-§5 here; anything justified by an engine's
   `profile.py` busy number or by its `EVENT_SEMAPHORE`/trigger/relay count (§0); removing or
   rebalancing **ACT** work (§7 P3: 31.7 us of it is worth 1.7 us); hosting anything on Pool
   (§3: deadlocks against its relay chain, and it computes casts wrongly); widening a compute
   instruction's access pattern (§4: hangs at one pair); the staging-trigger gate point
   (§2 + iter-1 §1/§6: a local optimum in **both** directions); **hosting DVE compute on any
   other engine, in any form, with or without a DVE stub** (§1 +8, §9a +31, §9b +11.4, §3
   hang — four measured variants, and the cause is structural); and everything iter-1, iter-2
   and the s128/s2304 siblings closed.

# Device budget used
~2 h: 3 `profile.py --repeats 5` (one with `--parquet`), 1 `--repeats 3`, 4 `verify.py`
(1 of them `--runs 3`), 3 hangs at 70 s each. `verify.py` ~55 s, `--runs 3` ~2 min,
`profile.py --repeats 5` ~80 s. Spread this iteration: 0.15-0.29% on four of five runs and
1.53% once, so **re-measure anything under ~8 us**.

# Reusable scripts (in `scratch/`, reverted between iterations — rebuild from this file)
`common.py` (stream load/save, event-field helpers, dtype sizes incl. the int enum),
`sets.py` (`times()`: the §6 pc->slot map with timestamps; `rw()`: read/write boxes as
`(p0,p1,b0,b1)` from any mem-pattern prefix; `candidates()`), `dmacells.py` (expand a queued
descriptor's SB side into absolute byte runs), `offload.py` (§1, `--host/--ops/--nchan/
--nogate/--limit` — the sound WAR-gated offload, keep this), `dropes.py` (§2),
`merge.py`/`merge2.py`/`merge3.py` (§4: signature compare, cell-set verification, wait
satisfaction resolver, run finder, global `S[6]` renumbering).

---

# §9 §8 ITEM 1 IS BUILT, CORRECT, AND **STILL LOSES**: 847.1 us (+11.4).
# The whole "host DVE work on another engine" class is now closed, four variants deep.

`scratch/offload3.py` implements it end to end: the **288 PSUM-sourced DVE `COPY`s moved to
ACT with the DVE slot deleted outright** (no stub), `S[6]` renumbered program-wide, and every
`S[6]` consumer given the `S[6] >= omap[v]` **and** `S[104] >= cnt(v)` conjunction.
It **verifies bit-identical to the baseline** (`max_abs_err` 7.688e-05 / 5.842e-04, the same
digits as iter-0) and `semaphores.py` is clean.

| variant | ACT busy | DVE busy | span |
|---|---|---|---|
| inherited | 346.5 | 438.7 | **835.7** |
| §1: 191 casts to ACT, DVE **stub** waits for ACT (nominal placement) | 398.2 | 399.7 | 844.2 |
| §9a: 288 copies, stub-free, **conservative** companion (1930 inserted slots) | 416.4 | 367.0 | 866.8 |
| §9b: the same with the **box-aware** companion (564 inserted slots) | 414.0 | **366.7** | **847.1** |

**-72.0 us of DVE busy, +67.5 us of ACT busy, and the wall goes UP 11.4 us.** Probe P5 says
those 72 us of DVE work are worth -27 us of wall and probe P3 says ACT's side should cost
about +3; the missing ~70 us is the structure, and it is the same structure in all four
variants: **each moved copy makes ACT block on PE (`S[5] >= w`, the RAW on PE's PSUM output)
inside ACT's in-order stream, and ACT is on the latency chain** (`:451/:453` ACT casts feed
PE `:447/:449`). 288 new PE->ACT head-of-line blocks cost more than the work they save. This
is iter-1's SP head-of-line finding reappearing on an engine that is actually on the chain.
**Do not try to host DVE work anywhere.** The remaining way to collect P5's -27 us and P2's
-17 us is to make the work *not exist* (§8 item 2), not to move it.

### Four bugs found on the way; each cost a device run, and each generalises
1. **`events_semaphore_value` is shared between the primary wait value and the
   `SEM_ADD_IMM`/`SEM_SET` update amount.** Rewriting a primary wait on such a slot rewrites
   its increment: `S[40..55]` went 160 -> 1286-3046 and **`semaphores.py` did not warn**
   (max_wait <= total_inc still held). Guard: a slot is *value-locked* when its primary update
   is neither `NONE` nor `SEM_INC_*`; never touch its primary wait, hoist the wait into an
   inserted `EVENT_SEMAPHORE` and restore the value field. The **extended** block has its own
   wait/update value fields and is always safe to rewrite. **Always diff every semaphore's
   total (primary *and* extended channel) before and after an edit** — that is how I caught it.
   (My §4 check "no `S[6]` waiter has a value-consuming update" was wrong because it only
   looked at primary waits: the 160 ACT signalling ES wait on `S[6]` in their *extended* slot
   while adding 16 to `S[40..55]` on the primary one.)
2. **`start_ts` in `Instruction.parquet` is when a slot is issued and BEGINS WAITING, not when
   it starts executing** (`duration_ns` includes the wait; `evt_wait_time_ns` is the wait part).
   So "the position engine E held at time T" must be computed from **cumulative-max `end_ts`**,
   not `start_ts`: place at `bisect_right(cummax_end_ts, T)` = the number of slots that had
   *completed* by T. With `start_ts` the builder put copy 8 *after* ACT slot 48, which waits
   `S[6] >= 48` — a self-deadlock, and a 70 s hang. **This invalidates every placement rule in
   iter-1/iter-2's notes that says "the last slot whose `start_ts` precedes X".** Bisection:
   1/4/6/7 copies passed, 8/16/64/288 hung, and the static assertion below found it instantly.
3. **Assert, do not argue.** The check that found bug 2 in seconds: for every ACT slot *j*
   with a wait on `S[6] >= v`, assert `position(copy cnt(v)) <= j`. It reported exactly 1
   violation at 8 copies and 7 at 288, and 0 after the fix. It is now in the builder.
4. **A conservative companion is 20 us of wall** (§9a vs §9b, 1930 vs 564 inserted slots).
   A consumer needs the `S[104]` companion only if it actually touches a moved copy's data —
   its SBUF destination or its **PSUM source bank** (PE `MATMUL` declares a 1-element PSUM
   write but accumulates over `psum_zero_region = SIZE2048`). **But a slot with no address
   fields (`EVENT_SEMAPHORE`, `PSEUDO_DMATRIGGER`) must fall back to the conservative rule** —
   an SP trigger's transfer reads SBUF a moved copy writes, and skipping those 59 SP waits
   produced `max_abs_err 6.96e+03` (wrong, not hung).

### What is reusable and verified
The renumbering machinery itself is now **device-proven correct**: deleting DVE slots,
remapping 1600+ `S[6]` waits across all five engines, and emitting conjunctions produces
**bit-identical output**. §8 item 2 needs exactly this, so it is no longer the risky part.

# §10 TWO MORE STATIC CLOSURES (no device time spent)

* **Dead DVE stores: 3.** Of the 2185 DVE write boxes, only 3 are never touched by any read
  box anywhere in the program (8280 read boxes across all five engines). Nothing to delete.
* **Redundant DVE recomputation: 1.** 1456 DVE compute slots fall into 272 groups of
  **bit-identical instructions** (same opcode and every non-event field equal; group sizes 2
  to 23) — but that is just the loop body reusing the same buffer addresses across the 32
  batch items. Checking each adjacent pair in a group against **every write box in the
  program with its timestamp** (does anything clobber the second one's inputs between the
  two?), exactly **one** pair is a genuine recomputation (`COPY` at DVE slots 1194 / 2217).
  The compiler leaves no redundant work. Both scans are cheap; do not repeat them.

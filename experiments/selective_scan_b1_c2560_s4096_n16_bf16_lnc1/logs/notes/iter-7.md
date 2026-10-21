# iter-7 — selective_scan b1 c2560 s4096 n16 bf16

**Result: no change shipped.** `kernel.neff` is byte-identical to iter-5/6's (md5 `2d4a729e…`),
`verify.py` + `launch_rules.py` PASS (1.250e-01 / 1.812e-04), measured 6604.33 us at the start of
the iteration (samples 6603.29 .. 6604.60, spread 0.02%).

What this iteration produced instead of a win:

* **§0 — iter-6's headline closure is wrong.** Reordering DVE is free; moving a *PSUM read* earlier
  is what costs 526 ns. That re-opens iter-5 §4's 153 us scan-pair merge as the only >1% lever left
  (now blocked only by SBUF, not by reorder cost).
* **§1 — a quantitative DVE↔Pool cost model**, replacing the qualitative "the balance point is
  reached": the Pool exchange rate is asymmetric (+0.68 us of wall per us added, −0.15 per us
  removed) and the total remaining headroom is **113 us (1.7%)**.
* **§2 — four new editing primitives, all verified on device**, the most useful being that an
  `EVENT_SEMAPHORE` slot can carry `SEM_INC_COMPLETE`, so you can neutralise an instruction
  without renumbering its semaphore.
* **§3 — six levers closed by measurement** (one of them a fully built, bit-exact edit that is
  simply unprofitable), and the cross-engine aliasing fact that probably also explains iter-6's
  unexplained hazard in canonical phases 31..35.

Tools in `.autohelix/notes/iter-7-tools/` (`probes.py` reproduces every number in §0/§1).

---

## 0. THE HEADLINE — iter-6's "the DVE order is immutable" is wrong, and I have the measurement

iter-6 closed every reorder-based lever with one control: swapping two *adjacent, mutually
independent* DVE computes cost **572 ns per swap**, 420 times. Its control pair was `B(k)`↔`C(k-1)`
— **both of which read PSUM**. I split that experiment by operand location (body-only swaps:
bytes 0..3 and 12..63 exchanged, the events block left in each slot, so the sync is bit-identical
to the unedited stream; timing probes, outputs not checked; `sim.py` clean, 42026 steps both times):

| probe | swaps | wall | per swap |
|---|---|---|---|
| swap adjacent **non-PSUM** compute pairs (`y+=Cs` ADD ↔ `SCAN`, `dt*x` ↔ `SCAN`, MEMSET ↔ …) | 93 | **6600.8** (−3.5) | **≈ 0 ns** |
| swap adjacent `SCAN` ↔ **PSUM-reading** `TENSOR_TENSOR` (i.e. a PSUM read moves 1 slot earlier) | 2042 | **7678.6** (+1074) | **+526 ns** |

(DVE busy 6522 → 6519 and 6522 → **7609** respectively.)

**So the cost is not the permutation, it is moving a PSUM read earlier.** That is exactly iter-5
§2d's mechanism (PE is a strict follower with a *median 53 ns* of slack against DVE's PSUM-bank
releases, and the bank rotation is depth 4 with all 16 KB of PSUM in use, so it cannot be
deepened). The correct rule for future iterations is:

> **Reordering DVE is free as long as every PSUM-reading instruction keeps its position.**
> The 2560 `SCAN`s, the 306 `y+=Cs` ADDs, the 160 `dt*x` and the 46 MEMSETs are all non-PSUM and
> can be permuted at no cost; the 5120 `B`/`C` multiplies cannot be moved earlier at all.

I could not turn this into a win today, because no *mergeable* non-PSUM pair exists:
two `SCAN`s of different chains cannot merge (a 2-row scan chains the carry across the row
boundary — iter-5), two `y+=Cs` ADDs cannot merge (a 2-row TT with `dst` step 0 writes `y` twice
instead of accumulating), and the one real candidate — iter-5 §4's scan-pair merge along seq — is
still blocked because merging `scan(t,n)` with `scan(t+1,n)` drags `C(t,n)` (a PSUM read) along and
needs window-private SBUF that iter-6 proved does not exist. **But the lever is no longer closed by
reorder cost, only by SBUF**, which changes the ranking for anyone who finds spare SBUF or a way to
re-bank PSUM.

## 1. The DVE/Pool cost model — the numbers that decide what is left

All measured on this NEFF today, `profile.py --repeats 3/5`, spread ~1 us.

| probe (pure field edits, no sync changes) | Pool busy | DVE busy | wall | d(wall)/d(Pool) |
|---|---|---|---|---|
| **unedited** | 5610 | 6522 | **6604.3** | — |
| 2260 Pool `y+=Cs` ADDs 512→**64** elem (timing-only, wrong math) | 3634 (**−1976**) | 6442 | **6491.2** (−113) | 0.057 avg |
| 1587 Pool carry COPYs → **ES** (timing-only, wrong math) | 5292 (**−318**) | 6496 | **6557.7** (−47) | **0.148** |
| 2260 Pool ADDs 512→**576** elem (timing-only) | 5894 (**+283**) | 6685 | **6795.9** (+192) | **0.68** |
| 142 DVE `dt*x` moved to Pool (real, **bit-exact**) | 6112 (+501) | 6940 (+417) | **7053.2** (+449) | 0.89 |

1. **wall = DVE busy + 82 us still holds exactly** (6522+82 = 6604; 6442+49 = 6491; 6685+110 =
   6795). But part of "DVE busy" is Pool-induced stall: making Pool fast dropped DVE busy by 80 us
   with DVE's instruction stream untouched. **DVE's own work is ≈ 6442 us, so the floor for this
   instruction stream is ≈ 6491 us and the total remaining headroom is ≈ 113 us (1.7%).**
2. **The Pool exchange rate is sharply asymmetric: removing Pool work buys 0.15 us of wall per us
   (marginal, near the operating point); adding Pool work costs 0.68–0.89 us of wall per us.**
   The +0.68 figure comes from a probe that changes *one int16 field per instruction* and nothing
   else, so the penalty is intrinsic to the DVE/Pool dependency interleave, **not** an artifact of
   my semaphore edits. This explains iter-6's "balance point measured in both directions"
   quantitatively: moving one `y+=Cs` ADD either way loses
   (to Pool: −692 ns DVE, +2164×0.68 = +1472 ns ⇒ +780; back to DVE: +692, −2164×0.15 = −325 ⇒ +367).
3. **Therefore every "offload DVE → Pool" lever is dead**, and the only Pool lever that can pay is
   making Pool work *disappear* (fewer/cheaper Pool instructions), worth at most ~47 us for the
   carry COPYs and ~67 us for the DMA_DIRECT2Ds, inside the 113 us global budget.
4. Pool's per-element rate does **not** depend on the access-pattern shape: rewriting all 2260 ADDs
   from `[512,1,1]` to `[256,2,1]/step[1,256,1]` (same bytes, same order, bit-exact) measured
   Pool 5611.4 vs 5610.4 and wall 6603.8 vs 6604.3 — **zero**. A 512-elem Pool TT is 2164 ns,
   a 512-elem Pool COPY 1870 ns, a 1-elem COPY 184 ns, a Pool ES 19 ns.

## 2. NEW editing primitives, both verified on device

* **An `EVENT_SEMAPHORE` slot CAN carry `SEM_INC_COMPLETE`.** No ES in any of the 14 corpus NEFFs
  does, so this was unknown. I converted all **1587** Pool carry `COPY`s into ES slots that keep
  their original wait **and** their `SEM_INC_COMPLETE S[3]` update: the NEFF ran normally
  (6557.7 us, output wrong only because the copies no longer happen).
  **Consequence: you can neutralise or relocate an instruction without renumbering its semaphore.**
  This is the cheap replacement for every "delete N slots and remap S[k]" edit that cost iter-6 six
  70 s hangs. 19 ns per ES, i.e. free.
* **Shrinking a compute to `num_elem=[1,1,1]` is a safe way to keep a slot's ordinal while removing
  its work** (203 ns instead of its full duration). Verified bit-exact when the 1 element it still
  writes is the same value another engine wrote (same product ⇒ order-independent, no tearing).
* **Replacing an instruction's single wait by a different one, and putting the extra guarantee in
  the `events_extended_wait_*` fields of the ES slot that already follows it, is free.**
  Census of free capacity in this NEFF: of the 160 DVE slots after a `dt*x`, **157 are ES with an
  unused extended wait**; globally 2192 of 5946 DVE ES slots have both wait fields free, and 2541
  DVE slots already wait on S[4] (whose value you can simply raise to a max). Unlike iter-6's
  "~280 ns per *inserted* wait-carrying ES", reusing an existing ES costs nothing.
* **Pool executes a `BFLOAT16 × BFLOAT16 → FP32 TENSOR_TENSOR MULT` bit-exactly** (the product of
  two bf16 is exact in fp32). 3547 ns per 512 elements, vs 2164 for the fp32 ADD — bf16 operands
  are *more* expensive on GpSimd.

## 3. Levers closed this iteration (do not redo)

### 3a. `dt*x` (the 160 bf16 DVE multiplies, 108 us of DVE) → Pool. **+449 us. DEAD.**
Built and **verified bit-exact** for 142 of the 160 phases: DVE slot shrunk to 1 element, the full
`[512]` MULT inserted into Pool inside a measured Pool gap (median gap 2840 ns), WAR/RAW covered by
explicit `S[6]/S[4]/S[5]` ordinal waits on inserted Pool ES slots, and the hand-off back to DVE as
an `S[3] >= v` extended wait on the ES that already followed the `dt*x` slot (`v` = the value of
the first *existing* Pool S[3] incrementer at/after the insertion point, so **no S[3] renumbering
at all**). Everything worked; it is simply not profitable — see §1.2. `build7.py` reproduces it
(`MODE=shrink` / `MODE=es`).

### 3b. Pool carry COPYs (1587 slots, 318 us of Pool, worth −47 us) → ACT. **Numerically WRONG.**
Same machinery (`build9.py`): Pool slot → ES keeping its `S[3]` increment, the COPY inserted into
ACT (29% busy), and the consumer guarantee re-emitted as `S[4] >= v` on a DVE ES before the first
DVE slot that next writes the source or touches the destination. It simulated clean and ran, but
`verify.py` gave `max_abs_err 4.374e+01`. **Root cause, and the generally useful fact:**

> **The rotating compute region 16384..60224 is shared across *roles and engines*.** The same byte
> ranges that DVE scans write are also written by **ACT `ACTIVATE` (dA)** and by **Pool
> `TENSOR_TENSOR` (y_accum)**. For 60 sampled carry COPYs there were ~12300 ACT and ~9900 Pool
> instructions writing the COPY's *source* range and ~8000/6100 touching its *destination*.
> Any edit that moves a read or a write in that region **even 1 us** in time needs the WAR/RAW
> analysis over **all five streams**, not just DVE. iter-5/6's intra-DVE alias checks are not
> sufficient, and this is very likely also the unexplained hazard in iter-6's "canonical phases
> 31..35".

The fix is to extend the deadline to `min(next DVE toucher, next Pool toucher, next ACT toucher)`
and to add the WAR wait on the Pool/ACT side too; I ran out of time. The payoff is only ≈ 47 us
(0.7%) and only 478/1587 sites were placeable anyway (next bullet), so it is low priority.

### 3c. **ACT is not usefully idle.** ACT is 29% busy with 4.7 ms of "idle", but its `dA` production
runs in near-lockstep just ahead of DVE's scans. Inserting a copy before ACT's `S[4]` incrementer
`v` requires DVE progress `ord6(producer)`, while DVE already demands `S[4] >= v` at an *earlier*
DVE point (it needs `dA(v)`) ⇒ cycle. Formally, with
`D(v) = min{ ord6(k) : DVE slot k waits S[4] >= w, w >= v }`, a site is placeable only if
`ord6(producer) <= D(v)`; that admits **478 of 1587**. Any plan that wants to park work on ACT must
first buy ACT lead time, which means touching the `dA` schedule.

### 3d. Pool access-pattern reshaping. Zero (§1.4).

### 3e. Pool `PSEUDO_DMA_DIRECT2D` (680 slots, 447 us of Pool, worth ≈ −67 us). **DEAD, both routes.**
* **Cannot change engine.** Its DGE queues are per-engine: the profiler names the semaphores
  DVE/Pool/ACT wait on `S[8..23] (qGpSimdDynamic)`, i.e. they belong to GpSimd. Moving a trigger to
  SP or ACT would use a different queue and a different semaphore, breaking every waiter.
* **Cannot be merged either** — I ran the falsifier (no device time). The 680 slots are the x/dt
  input loads (`dge_op = DMA_DIRECT2D`, src HBM via a var marker, dst SBUF 114688.., 128 partitions
  × 1024 B, `sem_increment` 16), spread over the 16 queues as 42–43 each. Two blockers:
  (a) **every intermediate value is waited on** — each semaphore has exactly 43 distinct wait values
  across the 5 streams, `16, 32, 48, … 688`, i.e. one wait per DMA and 344 of the 680 are odd
  multiples of 16, so no pair can be fused into one `sem_increment = 32`;
  (b) same-semaphore DMA_DIRECT2Ds are **never adjacent** — the slot-distance histogram peaks at
  128–132 Pool slots, so a merge would also be a long-range relocation.

### 3f. Merging adjacent Pool carry COPYs. Only **149** of the 1587 small COPYs have another small
COPY in the next Pool slot (iter-3 already took the easy ones) ⇒ ≈ 27 us of Pool ≈ **−4 us of wall**.
Not worth the risk. (With the ES primitive it no longer needs an S[3] renumber: merge into slot `i`
and make slot `i+1` an ES that keeps its increment.)

## 4. Three facts worth keeping about sync and the tools

* **The x/dt input DMAs land a median of 104 us before the instruction that consumes them**
  (min 23 ns, p10 100 us), measured from `SemaphoreUpdate.parquet`: parse `id` as
  `S\[(\d+)\]` and find the first timestamp whose `value >= the instruction's events_semaphore_value`.
  So there is ~2.5 phases of input prefetch slack. Useful for any edit that wants to read an input
  earlier.
* **`sim.py`'s DMA model is over-conservative and will report false deadlocks.** It credits
  `+16` per `PSEUDO_DMA_DIRECT2D`, but the hardware increments those semaphores **once per AXI
  packet** (`update_source == AXI`, 90064 of the 109931 updates; S[8] reaches 688 in 689 updates).
  My first `dt*x` build was reported stuck on `S[15] >= 224 have 208` through a
  Pool→SP→PE→DVE→Pool cycle that the real semaphore timeline refutes (the value was reached 104 us
  earlier). Cross-check any DMA-semaphore wait against `SemaphoreUpdate.parquet` before believing
  the sim. The sim is still worth running: it caught two *genuine* cycles this iteration
  (§3c's ACT one twice).
* **DVE instruction timestamps are not monotone in slot index** (the profiler reports overlapping
  pipeline intervals; DVE ES slots show a 459 ns median "duration"). Ordering DVE instructions by
  `start_ts`/`end_ts` is wrong and it cost me two bad builds. Use the slot index, or a running max
  (`dmono[k] = max(start_ts[0..k])`) when you need a time.

## 5. Where the remaining 113 us is, and what I would try next

The honest picture: **wall 6604, DVE's own work ≈ 6442, floor ≈ 6491 ⇒ 113 us (1.7%) of
headroom**, of which nothing I could construct reaches more than ~13 us safely. Ranked by
(value × probability of landing):

1. **Find SBUF, then do iter-5 §4's scan-pair merge** — 1216 merges = **153 us (2.3%)**, and §0
   removes the reorder-cost objection that iter-6 used to close it. Everything else about it is
   already worked out in iter-5 §4/§4a. The *only* remaining blocker is 18 × 2 KB of window-private
   SBUF, and iter-6's exhaustive `freescan.py` says the only hole above 16384 is 1216 B. So the
   real first step is a **liveness analysis that reuses a compiler tile** (iter-6 says nobody has
   done one), or a PSUM re-bank that frees the `C` tiles. Budget a whole iteration; it is the only
   lever left that is worth >1%. Note the §0 corollary that makes it harder than iter-5 thought:
   the merge drags `C(t,n)` (a PSUM read) later, and §0's second row shows PSUM reads are the one
   thing you may not move — so verify the direction (later may be free; I only measured *earlier*,
   and only 8 adjacent pairs exist to test "later" at scale).
2. **Finish §3b with the 5-engine deadline** — ≈ −47 us ceiling, and only 28% of the sites are
   placeable (§3c), so realistically −13 us (0.2%). `build9.py` needs ~20 lines: extend the
   deadline search to Pool/ACT writers of the source and put the WAR wait on the Pool/ACT side too.
   Low value, but it is the only edit here that is nearly built.
3. Do **not** spend device time on: anything that moves work *onto* Pool (§1.2), anything that
   parks work on ACT without first buying ACT lead (§3c), Pool access-pattern reshaping (§1.4),
   moving a DVE PSUM read earlier (§0), merging or relocating Pool `DMA_DIRECT2D` (§3e), merging
   adjacent Pool carry COPYs (§3f, −4 us), or any time-shift inside 16384..60224 that has not been
   checked against all five streams (§3b).

## 6. Tooling — `.autohelix/notes/iter-7-tools/`

* `lib.py` — `decs(engine)` → `(raw bytes, [isa.Instruction])`; point `W` at your `work/`.
* `an1.py` — builds `an1.pkl` = (exact SBUF read/write ranges for every slot of all 5 streams,
  `(engine, slot) -> (start_ts, end_ts, duration)` from `prof/Instruction.parquet` via
  `compiler_pc`). Everything else consumes this. Enumerates dim≥1 offsets exactly (no bounding
  boxes) and includes the scan's `imm0_imm_ptr`.
* `sim.py` — iter-6's in-order 5-engine simulator, **now also checking extended waits** (it did
  not before; that was listed as a to-do in iter-6's notes). Unedited NEFF: 42026 steps, done.
* `build7.py` — the `dt*x` → Pool edit (§3a). `MODE=shrink|es`, `LEAD`, `POOLDUR`, `DMAMARG`,
  `LIMIT`. Reads `/tmp/w7/clean` and writes `/tmp/w7/work`.
* `build9.py` — the carry COPY → ACT edit (§3b), including the `D(v)` cycle test and the
  monotone/distinct-incrementer placement rule.
* `probes.py` — **all six timing probes of §0/§1 in one script**, with the measured result for each
  in its docstring. `PROBE=<name> SRC=<clean work> DST=<out work> python probes.py`, then pack and
  profile. Smoke-tested: the slot counts it reports (2260 / 2260 / 2260 / 1587 / 93 / 2042) are the
  ones the numbers in this note were measured with. Includes the **body-swap** technique, which is
  the cheap way to price any reordering question on this kernel: exchange bytes 0..3 and 12..63 of
  two adjacent slots and leave the events block (4..11) alone, so the semaphore stream is
  byte-identical, `sim.py` stays at 42026 steps, the probe cannot hang, and the only thing you have
  changed is which operands execute first.

Device cost unchanged: `verify.py` ~10 s, `profile.py --repeats 3` ~40 s, `--repeats 5` ~80 s,
`--repeats 1 --parquet` ~25 s. Process-to-process spread ~1 us.

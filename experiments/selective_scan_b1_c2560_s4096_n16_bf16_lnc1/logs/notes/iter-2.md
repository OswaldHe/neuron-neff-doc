# iter-2 — selective_scan b1 c2560 s4096 n16 bf16

**Result: 6886.4 -> 6818.8 us (-0.99% on iter-1, -13.90% on the iter-0 baseline, 1.1615x).**
Numerics bit-identical to the baseline (max_abs_err 1.250e-01 / 83% of limit, cosine 0.999944,
output_1 1.812e-04). 5 repeats, spread 0.01%. `semaphores.py` clean, `launch_rules.py` clean.

Engines: DVE 6610 -> **6454** us (94.7%), Pool 5579 -> **6069** us (89.0%), ACT/PE/SP/DMA unchanged.
**DVE and Pool are now nearly balanced** (385 us apart) — that is the new constraint.

---

## THE HEADLINE: iter-1's "closed item 2" is WRONG. S6-incrementing instructions CAN be moved.

iter-1 concluded *"moving any S[6]-incrementing instruction is a net loss"* because its only fix
for the early-retire problem was an `S6 >= k-1` **self-wait guard** on the replacement
`EVENT_SEMAPHORE`, which costs ~585 ns of DVE pipelining — more than the 548 ns saved. That closed
352 y-ADDs, 152 MEMSETs and 160 `dt*x` MULTs.

**The guard is not needed. Renumber S[6] instead.** Recipe, verified bit-exact on device:

1. Delete the S6 increment at the moved slot (the replacement ES has `events_update_mode = NONE`).
2. `newmap[k] = #surviving S6 updaters with old ordinal <= k`. Apply `newmap` to **every** S[6]
   wait, primary (`events_semaphore_value`) and extended (`events_extended_sem_wait_value`), in
   all five streams. A wait that was `S6 >= k` (k = the moved instruction's own ordinal) becomes
   `S6 >= k-1`, i.e. "everything on DVE before it" — strictly weaker, so it never stalls.
3. The dependency on the moved instruction itself is re-expressed as a **Pool `S[3] >= c` wait**
   placed by iter-1's greedy interval stabbing (latest legal DVE ES slot < deadline, value =
   exactly that ADD's S3 ordinal).

This moved 351 of the 352 remaining y-ADDs and gave **-68 us with DVE busy down 155 us**. No
pipelining penalty at all. **Whoever edits next: this unlocks the 146 S4-waiting y-ADDs, the 160
`dt*x` bf16 MULTs and the 152 Y-zeroing MEMSETs** — see "What to try next".

### The one bug that cost a device run (and the signature to recognise it)

First build: ran fine, `max_abs_err 1.471e+01`, **99.40% of elements in tolerance**. One 512x128
ADD term is exactly 65536/10485760 = **0.62% of output_0** — so "99.4% in tolerance" means
*exactly one* ADD term lost. Cause: a cross-engine consumer (ACT/PE/SP) whose wait is
`S6 >= k` with k == the moved instruction's own ordinal. iter-1's deadline code maps a
cross-engine waiter back to `idx_of[value]` and then takes `bisect_right(L, i)`, which **skips
the entry equal to i** — so that consumer's dependency is silently dropped and cannot be covered
(no DVE slot in `[i, i)`). Fix: `bisect_left` for the cross-engine list, then drop any candidate
whose deadline is `<= i`. Exactly 1 of 352 was affected. **Use the percentage-in-tolerance figure
to count how many terms you lost: 0.62% per 512x128 fp32 ADD.**

### Preserving the fp32 accumulation order (mandatory, and iter-1's data is not enough)

`y_accum += Cs` is summed over 16 states; the order is bit-visible. The Pool insertion position
must keep the global order. **Within one Y buffer the S[6] wait value does NOT order the ADDs**:
e.g. Y=18752 has Pool ADDs with waits 7,9,...,48 and DVE ADDs at slots 90 (wait 50) and 93
(wait 17) — slot 93 is *last* in program order but has the *smallest* wait (it is the
`+ D*x` ADD, 160 of those, whose wait is on `x`/`D`, not on a `Cs`). Sorting by wait value
reorders the sum and the arithmetic changes.

**Recover the true order from `baseline/kernel.neff`**, where all 2720 y-ADD-like TT ADDs are
still on DVE: key each ADD by `(dst, src1, events_wait_idx, events_semaphore_value)` — unique for
2712 of 2715 keys — and index it by baseline DVE slot order. Then for a candidate, the legal Pool
window is `(pos of last Pool ADD with same Y and smaller baseline index, pos of first with a
larger one)`, and the insertion point is the eager S6 key clamped into that window. All 352 were
placeable this way. (The naive filter "only move ADDs whose wait exceeds every existing Pool ADD
wait for that Y" keeps only 30 of 352 — do not use it.)

---

## REFUTED this iteration: moving the carry `COPY` off Pool (review candidates #2/#3)

### #3 — repointing the scan's `imm` carry pointer at the previous scan's last column: **dead**

Structure (measured, 2560 scans / 2560 Pool COPYs / 80 carry slots at 16640 + 4k):

* A scan chain = one (channel tile, state). Its 8 sequence chunks all use **one** carry slot
  `P`; the chunk `dst` buffer **rotates over 8 different addresses**; `dA` (`src0`) advances
  +2048 B; `dtB` (`src1`) is reused.
* After chunk c, the Pool `COPY` snapshots `scan_c.dst + 511*4` into `P`, and chunk c+1 reads `P`.
* **Why repointing fails:** by the time chunk c+1 runs, `scan_c.dst` has already been overwritten
  by a *different* chain's scan that shares that buffer (e.g. dst 26944 is written by DVE slots
  11, 26, 44, 62, 79, 97 — six different chains). The source of the carry is destroyed. The COPY
  exists precisely to snapshot it. Closed.

### #2-adjacent — moving the 2560 COPYs to the Activation engine: **built, verified, 2 variants, both slower**

`COPY` (0x46) **is** executed correctly by Activation (7316 instances in the corpus; here 2560
moved copies at **191 ns each**, output bit-exact). So ACT is a usable off-load target for
`COPY`, unlike `TENSOR_TENSOR` (iter-1 closed that). But:

| variant | total us | DVE busy | Pool busy | ACT busy |
|---|---|---|---|---|
| iter-1 (COPY on Pool) | **6886.7** | 6609 | 5582 | 1903 |
| COPY on ACT, Pool keeps an ES that **waits** `S4>=n` and increments S3 | 6978.8 (+92) | 6606 | 5007 | 2200 |
| COPY on ACT, every S3 wait naming a COPY **re-pointed to S4**, Pool ES has no wait | 6955.6 (+69) | 6637 | 5006 | 2200 |

* Variant 2: the Pool placeholder still costs **436 ns** of queue time waiting for ACT, so the
  latency DVE sees is unchanged. Pool occupancy -575 us, DVE 0.
* Variant 3: Pool ES really is 20 ns (Pool COPY 1760 -> 64 us) and Pool stops being the
  constraint (its TT duration rises 1551 -> 2066 ns because it now *waits* for DVE). But DVE's
  long ES stalls move from `S[3]` (438 us, 191 slots) to **`S[4]` (517 us, 1863 slots)**: the
  2560 inserted copies delay ACT's `ACTIVATE` signals, and 1996 DVE ES already wait on S[4].
  2560 x 191 ns = 490 us added to ACT ~= the 517 us of new DVE stall.
* **Conclusion: DVE, Pool and ACT run in lockstep with essentially zero slack.** The carry copy
  costs DVE ~170-200 ns of stall wherever it is executed; relocating it only moves which
  semaphore DVE blocks on. Do not retry unless the *total* instruction count on the chain drops.

**Cycle that must be respected if anything is ever put on ACT** (this is what made my first
attempt hang at 70 s): ACT waits on DMA semaphores 8..23, and those DMAs are triggered by
**Pool** `DMA_DIRECT2D` instructions. So "Pool waits on ACT" closes a loop. The safe insertion
rule that worked (2560/2560 feasible, no hang):

* `A_i` = max `S[4]` wait among Pool slots `< i`; insert the ACT copy right after ACT's `A_i`-th
  S4 updater — i.e. at the point where, in the original schedule, everything before it had
  already happened by the time Pool reached slot `i`.
* Then advance `p` while: `prefix max S6 wait <= v`; `prefix max S3 wait < c`; `prefix max
  DMA-producing Pool slot < i`; prefix S5 wait reachable with `S6 <= v`; until
  `R(ordinal at p) > v`, where `R(o)` = earliest DVE S6 count that needs ACT ordinal `o`
  (propagated through DVE's S4 waits, Pool's S4 waits -> its S3 ordinal -> DVE's S3 waits, and
  PE's S4 waits -> its S5 ordinal -> DVE's S5 waits).
* ACT's `S[6]` wait values and `S[3]` wait values are both **non-decreasing in slot order**
  (verified), which is what makes the bisects legal.

---

## Also measured: coalescing DVE `EVENT_SEMAPHORE` slots is nearly free, not worth much

DVE ES carry **no semaphore updates at all** (all 7382 had `update_mode = NONE`), so they can be
merged/deleted with **no renumbering**. Rules (all exact, since an ES has no side effect and two
adjacent waits on the same semaphore are equivalent to one on the max):

* maximal run of consecutive ES -> one `sem -> max value` dict -> `ceil(n_distinct/2)` slots
  (primary + extended wait per slot);
* drop a wait the **following** instruction already makes (same sem, `>=` value); if the
  following instruction waits on the same sem with a *smaller* value, **raise its value** and
  drop the ES wait (12 cases);
* non-ES instructions have **no extended wait fields at all** (`TENSOR_TENSOR`, `COPY`: confirmed
  empty), so a wait can never be absorbed into a compute slot — 0 of 7382 were.

Measured: 15929 -> 13886 DVE slots (**-2043**, 80 us of instruction duration) bought only
**-9.3 us** of wall time (6886.7 -> 6877.4). A *second* pass, run after the ADD move, removed 323
more slots and measured **6825.3 us vs 6818.8 — worse**, so it was reverted. ES slots run in the
shadow of the neighbouring TT/SCAN, so their 24-54 ns duration is almost entirely overlapped.
**Lever closed: DVE slot count is not the cost, DVE compute duration is.**

The shipped NEFF = iter-1 + **first** coalescing pass (DVE 15929 -> 13886) + the 351-ADD move.
`neff_diff` vs `baseline/kernel.neff`: 4 of 34 members changed — `DVE0.bin` (1019456 -> 888704 B),
`Pool0.bin` (245888 -> 410048 B), `Activation0.bin` and `PE0.bin` (same size, only
`events_semaphore_value` fields rewritten by the S6 renumbering). `SP0.*` and `def.json` untouched.

---

## Current profile of the shipped NEFF (6819.2 us median of 5, spread 0.02%)

| engine | profiler busy | union-of-intervals | note |
|---|---|---|---|
| DVE | 6454 us (94.6%) | span 6826, busy **6562**, idle **264** | critical engine |
| Pool | 6068 us (89.0%) | span 6824, busy 6072, idle 752 | 385 us behind DVE |
| ACT | 1903 us (27.9%) | | idle but in lockstep; only `COPY`/`ACTIVATE` run here |
| PE | 2972 us (43.6%) | | paced by PSUM banks, i.e. by DVE |

DVE instruction durations now: `TENSOR_TENSOR` 5435 x 670 = 3644 us, `TENSOR_TENSOR_SCAN`
2560 x 1297 = 3320 us, `MEMSET` 200 x 412 = 82 us, `EVENT_SEMAPHORE` 5698 x 649 = 3699 us.
Pool: `TENSOR_TENSOR` 2565 x 1534 = 3936, `COPY` 2560 x 683 = 1749, `DMA_DIRECT2D` 680 x 656 = 446.

**Do not read the ES duration as wall time.** Total DVE duration is now 10745 us inside a 6826 us
span (64% overlap) and 3682 ES report >200 ns "stall" summing to 3640 us. An ES's `duration_ns`
evidently starts at *dispatch*, not at head-of-queue, so it accumulates while earlier instructions
run. The trustworthy numbers are the union-of-intervals ones above. (This is also why removing
2043 ES slots bought only 9 us.)

**DVE is now compute-bound, not sync-bound.** TT + SCAN alone are 6964 us of duration, which
union-overlaps to roughly 6300 us inside a 6826 us span. With Pool unable to take any of it, the
whole remaining headroom is the 264 us of DVE idle plus whatever overlap is recoverable —
call it **~300 us (4.4%)** without changing the DVE instruction mix itself.

DVE duration composition before the move (sum 8022 us over a 6847 us union — 17% overlap):
`TENSOR_TENSOR` 5786 x 676 = 3910; `TENSOR_TENSOR_SCAN` 2560 x 1291 = 3305 (**DVE-only and
unmergeable**, iter-1); `EVENT_SEMAPHORE` 7390 x 97.7 = 722 (of which **438 us is stall** in 191
slots, waiting `S[3]` on Pool, and ~260 us is 24-54 ns base cost); `MEMSET` 200 x 412 = 82.

Attribution of the 438 us of DVE stall to what Pool was doing (iter-1 NEFF): TT 250.8, COPY
100.3, DMA_DIRECT2D 79.0, ES 6.8, **Pool idle 0.9** — Pool is 100% busy throughout every stall.

## Three more groups attacked after the win — all three closed

Pool had 385 us of headroom left and DVE/Pool balance (`DVE - 0.548x = Pool + 1.551x`) said
**x ~= 183 more ops** would equalise them at ~6354 us each (~100 us of upside). All three
remaining candidate groups were tried and none of them can supply those ops:

1. **The 146 y-ADDs that wait on `S[4]` — CLOSED, with the reason iter-1 could not find.**
   Moving them makes Pool wait on ACT, and ACT waits on DMAs triggered by Pool, so it deadlocks
   (70 s NRT timeout; reproduced). I built the full `A_i` happens-before cycle check (all 146
   "feasible" by it) and it **still hung** — the check misses one transitive hop: DVE's own
   progress to the ACT prefix's `S[6]` requirement depends on Pool via DVE's ~2600 `S[3]`
   hand-off waits, so the condition is a fixed point, not a prefix scan.
   Then the clean escape: replace the `S[4] >= n` wait with an equivalent **`S[6]` proxy** — the
   last DVE slot before it that already waits `S[4] >= n` or more, so Pool never waits on ACT.
   **Measured: 0 of 146 have such a proxy.** These ADDs *are* the first DVE instructions to need
   their ACT ordinal — they are the S4 synchronisation points themselves. There is no way to move
   them without putting Pool downstream of ACT. **Do not retry.**
2. **The 159 `dt*x` bf16 MULTs — CLOSED (built, ran, badly wrong).** 152 were placeable with a
   WAR guard (`S[6] >=` the last DVE slot touching the dst) and a Pool ES carrying the DGE DMA
   wait. Result: `max_abs_err 9.9e+03`, cosine 0.066, **only 41.5% of output_0 in tolerance** and
   output_1 wrong too — structural corruption, not a lost term. Cause: their `dst` addresses
   (16704, 31040, 35136, 53568 ...) are **the same scratch addresses as the `Cs` and `Y` buffers
   that Pool's own 2565 TT ADDs read**. Hazard analysis against the DVE stream plus cross-engine
   S6 waiters is *not sufficient* once Pool holds compute: **Pool's own program order has to be in
   the hazard graph.** With deadline slack of only 2-3 DVE slots there is very little room to fix
   it. Anyone retrying must add Pool-internal RAW/WAR/WAW on the shared scratch pool.
3. **The 152 Y-zeroing `MEMSET`s — CLOSED by the opcode table.** `MEMSET` (0x49) appears 1035
   times in the 14-NEFF corpus and **every one is on DVE**. No other engine is observed to run
   it, so there is nothing to probe.

## ANSWERED (the loop's biggest open question): **Pool/GpSimd CANNOT read PSUM.**

The review and iter-1 both left this "unknown"; iter-1's probe hung but the hang was explained by
its own sync, so nothing was concluded. **Settled now by a controlled pair of device runs.**

Subject: DVE slot 6741, a line-267 `Cs = C_full * state` MULT — `src0 = 0x2000800` (PSUM),
`src1 = 53568`, `dst = 16384`, `wait S6>=3961`, own S6 ordinal 3963. Chosen because exactly two
instructions wait on its S6 ordinal: Pool slot 3113 (the `y += Cs` ADD that consumes its output)
and PE slot 4932 (the PSUM-bank WAR). Construction (no S6 renumbering, PE untouched):

* insert the identical 64 bytes into Pool at position **3113** — immediately before its consumer
  ADD, so RAW on `Cs` and WAR against the previous Pool reader of 16384 (slot 3107) are both free
  from Pool's in-order queue; it is also exactly the eager S6 position (last Pool slot waiting
  `S6 <= 3961` is 3112);
* the freed DVE slot becomes an ES that **keeps the `S6++`** (so PE's and Pool's `S6>=3963` waits
  stay valid, no renumbering anywhere), with iter-1's `S6 >= 3962` self-wait guard plus
  `S3 >= 2447` for the Pool MULT;
* renumber only the S[3] waits for the one inserted S3 incrementer.

| run | source operand | result |
|---|---|---|
| A | `src0 = 0x2000800` (PSUM) | **NRT_TIMEOUT, 70 s** |
| B | `src0 = 53568` (SBUF), everything else byte-identical | **exit 0 in 10.9 s**, `max_abs_err 75.8`, 99.68% in tolerance — exactly "one state's Cs computed from the wrong operand" |

Run B proves the synchronisation is complete and live; the only byte that differs in run A is the
source address. Before concluding I also ruled out every engine-level cycle by hand: max S6 wait
among Pool slots < 3113 is 3961 (not 3963); max S4 wait among those slots is 1214 -> ACT slot
1445, whose prefix needs `S5>=2003` (PE slot 4052, before the blocked PE slot 4932), `S6>=3872`
(< 3963) and `S3>=2363` (< 2447) — all satisfiable. (A first attempt that *did* have a cycle is
also informative: making **PE** wait on Pool via an inserted `S3>=c` ES in `PE0.bin` deadlocks,
because PE is upstream of ACT which is upstream of Pool.)

**Consequence: the 5120 DVE `TENSOR_TENSOR` MULTs with a PSUM operand (lines 224 and 267, 3447 us
= ~50% of DVE busy) can never leave DVE.** Together with the 2560 DVE-only scans (3305 us), that
is **~6750 us of DVE instruction duration that is permanently DVE-resident**, which at the
measured 17% intra-DVE overlap is a floor of roughly **5.6-5.8 ms** for this kernel unless the
PSUM traffic itself is restructured. Delete candidate #4 from the review's list.

## What to try next (ranked)

**The DVE -> Pool off-load lever is finished.** Every SBUF-only DVE TT op has now been moved
(2214 in iter-1 + 351 here) or closed (146 S4-waiting + 159 bf16 MULTs + 152 MEMSETs + 5120 PSUM
MULTs). There is nothing left to move, so making Pool cheaper is also pointless now — Pool is
already 385 us *behind* DVE and its spare capacity has no use. (For the record: the 680 Pool DGE
triggers, 446 us, would be the cheapest Pool saving if that ever changes, and
`TENSOR_TENSOR_ARITH_OP` has **no performance-mode field** — only `in0_in1_dtype`, `out_dtype`,
`num_active_channels`, `op` — so Pool's 1534 ns for a 512x128 fp32 TT vs DVE's 670 ns is
intrinsic and not tunable.)

1. **Recover the 264 us of DVE idle (3.9%).** This is the only cheap money left. It is hand-off
   latency: DVE waiting on Pool's `S[3]` and ACT's `S[4]`. The binding constraint on the greedy
   stabbing is deadline slack, which is set by the 3-deep `Cs` rotation — see (2).
2. **A 4th `Cs` buffer (SBUF re-addressing).** The hand-off deadlines (median slack 8 DVE slots
   for these 352, min 2; only 2-3 for the bf16 MULTs) are set by the 3-deep `Cs` rotation, and the
   shared scratch pool is also what made group (2) above corrupt. The pool is 16384..58176 and
   inputs start at ~113k, so **there is ~55 KB/partition free**. Un-aliasing the families would
   both add hand-off slack and make Pool-internal hazards tractable. Biggest remaining item, and
   the only one that can go past ~6.6 ms, but it is a large coherent edit.
3. **Fewer, wider DVE TT instructions — now the best structural idea, and worth re-planning.**
   iter-1 built pairwise merging of the line-224 and line-267 MULTs end to end and found it
   bit-exact and arithmetically profitable (merged duration 1190 ns vs 2 x 676, i.e. **136 ns per
   pair**; 635 mergeable pairs for line 224, **761 for line 267**). It refuted line 224 by
   measurement (merging at the earlier slot pulls PE's `B_full` dependency one matmul earlier and
   PE/DVE run in lockstep: 7100 vs 6886). But it only refuted **line 267** because the merge had
   to happen at the *later* slot, which pushed the `Cs` write past the DVE slot where iter-1's
   greedy had already placed that state's hand-off wait — i.e. it collided with a *plan*, not with
   the hardware. With the S6-renumbering recipe in hand, the right order is now clear: **merge
   line 267 first, then compute the hand-off deadlines and run the stabbing on the merged
   stream.** Ceiling 761 x 136 ns = **103 us** of DVE duration, and it also removes 761 DVE slots.
   iter-1's renumbering recipe for a dropped merge partner (map every old S6 count to the new
   ordinal of its own instruction, or of the instruction it merged into) is the same machinery as
   this iteration's `newmap`.
4. The scan (3320 us, ~49% of DVE) is the hard floor: DVE-only, carry does not reset per
   access-pattern row (iter-1), and 2.06 ns/element is ~2 cycles for a dependent multiply-add.

## Tooling gotchas found this iteration

* `semaphores.py` **only counts primary waits.** After this edit S[3] showed `max_wait 0,
  waiters 0` in one variant while 1852 *extended* S[3] waits were still live. Never read it as
  "nothing waits on this semaphore".
* `prof/Instruction.parquet`: `compiler_pc` **is** the 64-byte slot index (verified: Vector
  0..15928 for 15929 slots), so the profile joins to the NEFF directly. `evt_wait_time_ns` is
  not a per-instruction stall (its DVE sum is 181 ms over a 6.9 ms span) — use `duration_ns`,
  which *includes* the stall, and union-of-intervals for the span.
* `events_semaphore_value` is shared between the wait comparand and the `SEM_ADD_IMM` operand
  (chapter 6.4), so you cannot both wait `S[a]>=v` and add a different immediate in one slot.
  That kills "give the next S6 updater a +2 increment" as an alternative to renumbering.
* Device-run cost this iteration: a working run is ~10 s, a deadlock is ~70 s, a wrong-arithmetic
  run is ~10 s and reports as a tolerance failure.

## Build recipe for the shipped NEFF (from iter-1's `kernel.neff`)

`/tmp/mv4.py` (candidate selection, baseline-order recovery, Pool window, deadlines) then
`/tmp/mv5.py` (Pool rebuild, S6 + S3 renumber, hand-off stabbing) — reproduced below in prose
because the scripts live outside the worktree:

1. Candidates: DVE `TENSOR_TENSOR_ARITH_OP`, `op=ADD`, `src0 == dst`, `num_elem[0]==512`, no PSUM
   operand, `events_update == S6++`, `events_wait == S6 >= v`. -> 352.
2. Baseline order for each by `(dst, src1, wait_idx, wait_value)` against `baseline/kernel.neff`;
   Pool insertion position = eager S6 key clamped into the same-Y baseline-order window.
3. Deadline = first later DVE slot referencing `Y` or `Cs` (candidates excluded), and for
   ACT/PE/SP slots referencing them, `idx_of[their S6 wait]` with **bisect_left**; drop candidates
   with deadline `<= i` (1 of 352).
4. Pool: insert the identical 64 bytes with `S6>=v` wait and `S3++` update, in baseline order
   within an insertion position.
5. DVE: candidate slot -> bare ES keeping `S6>=v`, **no update**.
6. Renumber every S6 wait by `newmap` and every S3 wait by the new Pool S3 ordinals (both primary
   and extended, all five streams).
7. Greedy stabbing of 308 `S[3] >= c` hand-off waits onto DVE ES slots that have a free wait
   field (1988 available).

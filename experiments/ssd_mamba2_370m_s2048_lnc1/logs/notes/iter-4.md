# iter-4 (ssd_mamba2_370m_s2048): 3106.2 -> 3071.8 us (-1.11%); against iter-0 5248.9 (1.709x)

**Shipped one edit: 120 more of the 166 leftover `state_sb` (head,chunk) bounces go single-hop,
i.e. iter-3 §1b/§1c applied to the remainder, admitted by a *substantively* tighter liveness gate
instead of a looser margin. Bit-exact: `max_abs_err` identical to the last digit to iter-1/2/3
(output_0 2.058e-07 = 89% of the 2.3e-07 limit, output_1 1.993e-07 = 87%), `verify.py --runs 3/4`
all runs bit-identical (three separate invocations), `launch_rules.py` pass, `semaphores.py`
clean (sems 40-55 still 816/816, no queue oversubscribed). md5
`6cc98083836757ee1b8de0eab881c009`. Members changed: `sg00/Activation0.bin`, `sg00/SP0.bin`.**

| step | median us (5 rep) | spread | delta |
|---|---|---|---|
| iter-3 control | 3106.231 | 0.15% | - |
| §1 102 pairs (iter-3's gate at `GUARD=5000`) | 3077.99 / 3076.88 (two runs) | 0.09% / 0.10% | -28.2 |
| §1 121 pairs (either gate) | 3068.76 | 0.08% | -37.5 |
| **§1 120 pairs (either gate, minus the one with a writer in its true window) -- SHIPPED** | **3071.83 / 3070.65** (two runs) | **0.12% / 0.08%** | **-34.4** |
| §2 Pool DGE trigger pair-merge (REFUTED, reverted) | 3133.1 (1 rep) | - | +27 |
| §2b `qPoolDynamic` -> hardware DGE in `def.json` (REFUTED) | does not load | - | - |

The 121- and 120-pair numbers differ by 3.1 us for one pair that should be worth 0.28 us, so
treat anything under ~4 us here as unresolved by 5 repeats. I shipped the 120 because that set
has a clean safety rule behind it (§1, last bullet).

**THE HEADLINE for whoever comes next: `Pool` is now a measured hard floor and the two obvious
ways to lower it are both closed (§2). A DGE trigger costs the issuing engine ~650 ns (Pool) /
~540 ns (ACT) / ~530 ns (SP) *per 128 descriptors*, flat from 1 to 128 and exactly linear above
it — so iter-3's "390 ns + 2.2 ns/desc" model is wrong, shrinking a transfer buys nothing, and
merging two transfers into one 2-D transfer buys nothing either. Pool issues 2385 triggers =
1550 us = 50% of the wall and that number cannot be moved by any descriptor-level edit.
The single biggest stall in the program (111 us, 3.6%) is ACT waiting for a Pool load that Pool
could in principle have issued 300 us earlier (§4.1), and §4.2 gives a reordering rule that is
semaphore-exact -- but §4.3 measures that the compiler's round-robin queue assignment pins any
reorder to ~20 slots, so that stall only opens if the *queue assignment itself* is rewritten.**

Scripts next to this file: **`iter-4-single2.py` is the shipped edit** (`MODE=or GUARD=20000
TGUARD=5000`); `iter-4-single-fixed.py` is iter-3's script with only the pairing fix of §1;
`iter-4-qstab.py` re-checks a shipped plan against any profile (§1, stability);
`iter-4-poolmerge.py` (§2, refuted); `iter-4-dgecost.py` (the cost measurement that killed §2 and
the descriptor-storm plan); `iter-4-poolprog.py` and `iter-4-qdist.py` (§4); `iter-4-common.py`.

---

## 1. §1 (shipped): the last 166 state bounces, and a gate that is tighter in substance

iter-3 converted 605 of 816 (head,chunk) bounces to single-hop (SP's reload reads `X[c-1]`
directly; the ACT save becomes an `EVENT_SEMAPHORE` doing `SEM_ADD_IMM_COMPLETE +16`), leaving
166 in the two-hop staged form, 28 on the original HBM path and 17 unpairable. iter-3's filter
rejected a pair if **anything touched** `X[c-1]` within `GUARD=20000` ns of the
`[save, reload]` window, and all 166 were rejected by it.

**First, a bug to know about: iter-3's pairing breaks on an already-edited stream.** The SP
reload's `events_semaphore_value//16` indexes the k-th *increment* of sem 40+i, but after iter-3
605 of those increments come from `EVENT_SEMAPHORE`s, not from `PSEUDO_DMA_DIRECT2D`s. Build the
per-sem list from **both** sources in program order (DGE `semaphore` field **and**
`EVENT_SEMAPHORE` with `events_update_mode==SEM_ADD_IMM_COMPLETE` on idx 40..55). With the fix
the list is 51 per sem (816 total) and the 605 already-converted pairs fall out by themselves as
`save already killed`.

**Two independent gates, and what each is worth** (all computed from the iter-3 control
profile at 3106 us):

| gate | pairs of the 166 |
|---|---|
| iter-3: any toucher, 20 us margin | **1** |
| iter-3: any toucher, 5 us margin | **102** |
| iter-3: any toucher, 0 margin | 151 |
| new: **writes only**, 20 us margin | **117** |
| new: writes only + **queued DMA writes**, 20 us margin | 79 |
| both gates (writes@20us AND touchers@5us) | 60 |
| either gate | 121 |
| **either gate, minus any pair with a writer in the true window -- SHIPPED** | **120** |
| both gates at 0 margin (the true window) | **150** |

* **Only a *write* to `X[c-1]` can break this edit.** SP's reload reads exactly the bytes the ACT
  save read, so a *reader* in the window is harmless. Filtering on writes only
  (`dst*_mem_pattern_*` + DGE `dst_*` endpoints with var marker 16 / var 0; PE's dst is PSUM,
  addr >= 0x2000000, and drops out) gives **117 pairs at iter-3's own conservative 20 us margin**,
  and 116 at 40 us -- i.e. the write gate is nearly margin-independent, which is the stability
  property a timestamp gate needs.
* **The 1149 queued DMA transfers are writers and iter-3's scan did not see them at all.** They
  only ever write SB (`dt -> SB 0x..1c000+`, `A -> SB 0x..7f00`), 512 B and 4 B. Adding them (from
  `prof/DmaPacketAggregated.parquet`: `write_offset` / `write_shape` / `dest` are **strings**
  like `'[0x181c400, 0x181c300]'`, parse with a regex) rejects 38 more pairs. **I could not settle
  whether those 38 are real:** the ACT save's source offsets are 0x4200..0x37e00 and the queued
  writes land at the same *offsets* but at `addr>>20 in {8,16,24}`, and compute instructions
  encode SBUF as `(partition<<20)|offset` while DGE/queued descriptors are linear with a 0x40000
  partition stride. Comparing them offset-only (partition masked out, which is what every script
  here does) may be a false positive. **Resolve this before relying on either answer** -- it is
  worth 38 pairs (~11 us) and it also decides whether iter-3's shipped 605 have a latent hazard.
* **The shipped set is the 121 of "either gate", minus every pair that has a *writer* of
  `X[c-1]` anywhere inside the TRUE window (margin 0) = 120 pairs.** That last rule is the one
  that is not a margin heuristic, so it is applied unconditionally in `iter-4-single2.py`; it
  rejects 7 pairs (4 queued-DMA writers, 3 DVE), one of which the margin gates had let through.
* **Stability check (`iter-4-qstab.py`), the thing that makes a timestamp gate defensible:**
  re-evaluating both gates for the shipped pairs against the profile of the **edited** stream
  (3067.6 us) still passes all of them, and 119 of 121 are clean in the true window on both
  schedules. The one remaining true-window hit on the edited schedule is a *reader* (ACT), which
  cannot break this edit by construction. So the gate's verdict does not depend on the schedule
  it was computed from.

Cost/benefit per pair is stable across iterations: iter-3 got 168 us from 605 pairs (0.28 us
each), iter-4 got 28.2 us from 102 (0.28 us each). **The remaining 64+45 pairs are worth ~30 us
and no more.**

## 2. §2 REFUTED, and the cost model it corrects: merging Pool DGE triggers

Hypothesis: Pool issues 2385 DGE triggers at ~650 ns = 1550 us (50% of the wall), and during the
111 us stall of §4 Pool is 100.9% busy while the 16 DMA engines are **10% busy** (`DmaPacket`
in that window: 6403 packets, 186 us of packet time over 16 engines, i.e. the DMA is not the
limit -- Pool's issue rate is). iter-3's note said a trigger costs "390 ns + 2.2 ns/descriptor",
so merging two 128-descriptor transfers into one 256-descriptor 2-D transfer should cost
390+2.05*256 = 915 ns instead of 1304 ns and save ~390 ns per pair.

`iter-4-poolmerge.py`: 102 adjacent same-queue same-shape load pairs merged by setting
`src_num_elem=[128,2]`, `src_step_elem=[step0, src_delta]`, `dst_num_elem=[128,2]`,
`dst_step_elem=[262144, dst_delta]`, `sem_increment=32` on the earlier slot and turning the later
slot into an empty `EVENT_SEMAPHORE`.

**Measured: the merged trigger takes 1311 ns, exactly 2x 652 ns, and `sbuf_write_bytes` doubles
(7670 -> 15340).** Pool busy 1577.0 -> 1581.6 us (worse), DMA busy +100 us, wall 3106 -> 3133 us.
So the 2-D encoding *works* (the second dimension is honoured and moves the right number of
bytes) but **the issuing engine's cost is per 128-descriptor row-group, not per trigger.**

The decisive static measurement (`iter-4-dgecost.py`, from `prof/Instruction.parquet` joined with
the decoded slots) is worth keeping:

| engine | descriptors | n | mean duration |
|---|---|---|---|
| Pool | 64 | 48 | **650 ns** |
| Pool | 128 | 2337 | **651-657 ns** |
| Pool | 256 (this experiment) | 102 | **1311 ns** |
| ACT | 64 | 211 | 544 ns |
| SP | 64 | 816 | 531 ns |

i.e. cost = `C_engine * ceil(ndesc/128)` with `C_Pool = 650`, `C_ACT = 544`, `C_SP = 531`.
**Consequences, all of them closing something:**
* Merging transfers on any engine is worthless (the cost follows the descriptors).
* Shrinking a transfer below 128 descriptors is worthless on the issuer (iter-3 §1b already
  measured this from the other side: 64 desc -> 1 desc moved ACT busy by 0.8 us over 605
  instructions). **Only deleting a trigger helps the issuer** -- iter-3 §1c's rule stands.
* **The 768 four-byte column loads (iter-3 §3a, the "descriptor storm", 128 descriptors to move
  512 B) cost Pool 500 us and would still cost 650 ns each as a single descriptor.** The
  relayout-plus-on-chip-transpose plan therefore buys *nothing on Pool* and only buys DMA
  descriptor time, and the DMA is 10-32% busy. **iter-3 §5 item 2 (the highest-ceiling candidate
  in the review) is closed.**
* Pool's floor is 2385 triggers = 1550 us whatever the layout. The wall cannot go below that plus
  the un-overlappable head/tail.
* **§2b, tried and REFUTED on the device.** `qPoolDynamic` is the only **software**-dynamic queue
  set in `def.json` (it carries `dynamic_dma: {SB_scratch_partition_offset: 0,
  SB_scratch_partition_size: 16384}`, and the profiler labels its packets `software_dynamic`;
  `qActDynamicHW` and `qSPDynamicHW` are `hardware_dynamic`) -- presumably why Pool pays 650 vs
  ACT/SP's ~540 ns, so making it hardware would have been worth ~260 us of Pool busy for a
  one-line JSON edit. **Dropping `dynamic_dma` makes the NEFF fail to load**
  (`NRT_INVALID in nrt_load_util`, "Failed to load nrt model"), both with the key renamed to
  `qPoolDynamicHW` and with the original name. The runtime requires the SB descriptor scratch for
  a pool-owned dynamic queue; GpSimd evidently has no hardware DGE path. `def.json` is only
  referenced from itself (`grep` of the whole unpacked tree), so the name is not the issue.

## 3. The per-engine picture after §1 (and what the critical path says)

`iter-3-cpath.py` still works unchanged and is still the right sizing tool (`profile.py
--repeats 1 --parquet prof`, then run it). On the 3106 us control it reproduced iter-3's
decomposition exactly:

| | us | % |
|---|---|---|
| Scalar/ACT busy on the path | 1160.7 | 37% |
| Tensor/PE busy on the path | 689.8 | 22% |
| Vector/DVE busy on the path | 504.1 | 16% |
| GpSimd/Pool busy on the path | 26.6 | 1% |
| gaps | 723.3 | 23% |

gaps by waiter: `Scalar ACTIVATE` 205.7 (producer latency on DVE sem 6 / PE sem 5),
**`Scalar EVENT_SEMAPHORE` 137.5** (111 of it in one wait, §4), `Tensor LDWEIGHTS` 118.9 (the
state reload, sems 56-71), `Vector STREAM_SHUFFLE` 100.0 (own program order),
`Tensor EVENT_SEMAPHORE` 41.0, `GpSimd EVENT_SEMAPHORE` 37.9 (Pool's own ring throttles).

ACT `ACTIVATE` on the path by class `(func, nch, num_elem[0], src marker, scale)`:

| class | n on path | us on path | ns each |
|---|---|---|---|
| `128,128,32,SBUF,PTR` (the `*= scale_col` multiplies) | 2342 | **637.9** | 272 |
| `7,128,1,PSUM,IMM` (the exps) | 1032 | 119.2 | 115 |
| `128,64,32,SBUF,PTR` | 495 | 105.8 | 214 |
| `128,1,128,marker1,PTR` | 285 | 83.9 | 294 |
| `7,1,1,SBUF,IMM` (`log_decay`) | 433 | 76.6 | 177 |
| `EVENT_SEMAPHORE` | 1286 | 50.2 | 39 |
| `DMA_DIRECT2D` (state save; 211 left before §1, 109 after) | 128 | 47.8 | 373 |

Note `128,128,32` costs 272 ns and `128,64,32` 214 ns -- same element count, half the channels --
so the ACTIVATE cost is **not** a fixed issue overhead; it tracks bytes written (16 KB vs 8 KB).
A 1-element 128-channel exp is 115 ns. So the multiplies are near a bandwidth bound, and iter-3
§4.1 already closed merging them. I did **not** run the review's candidate-#3 probe (make the
pointer `scale` an immediate and look at the duration) -- with the 272/214 pair above, the
pointer fetch cannot be more than ~60 ns of it, and the scale is real per-partition data anyway,
so even a positive result is not actionable. Treat candidate #3 as closed by inference, or spend
one profile run on it if you disagree.

## 4. Pool's *order*: the biggest single stall in the program, and why only queue reassignment opens it

### 4.1 The measurement
One ACT slot owns 3.6% of the wall. `Activation0.bin` slot **2771** is an `EVENT_SEMAPHORE` with
`wait s23>=800`, no update at all (a pure barrier), and it stalls **111.05 us** (t=787..898 us).
In that window GpSimd is 100.9% busy issuing 186 back-to-back DGE triggers (Pool slots ~809-1002,
652 ns each, zero gaps) and Scalar/Tensor/Sync are 0% busy. The 16 DMA engines are 10% busy, so
**Pool's in-order issue rate is the whole stall.**

Where did Pool fall behind? Before t=898 Pool was **busy 529 us and idle 356 us**, and nearly
every one of those idle chunks (4-10 us each) is Pool waiting on **sem 6 (DVE)** at an output
*store*. Pool's program interleaves `TBL->SB` loads (which depend on nothing but HBM input and
buffer liveness) with `SB->TBL` output stores (which wait for DVE/ACT to compute the data). The
stores head-of-line-block the loads behind them, Pool arrives at the burst 186 transfers behind
what ACT needs, and ACT eats the whole 111 us.

ACT's barriers on the Pool queues are **not** coarse: there are ~1286 of them, one every ~11 ACT
slots, each waiting one more increment on the next queue (s8..s23 round-robin, +16 each). So
there is nothing to relax: the data genuinely is not there.

### 4.2 The rule that makes reordering legal
A DGE queue semaphore is cumulative: a waiter on `q >= 16k` means "the first k transfers of queue
q have completed". Therefore **permuting Pool's instruction stream is semaphore-exact as long as
the relative order of the transfers *within each queue* is preserved** -- every wait threshold in
the program keeps its meaning, on every engine, with no renumbering at all. Pool's loads carry
*only* their own queue's ring throttle as a wait (measured ring depth is **1**: the throttle is
always `q >= 16*(own_count-1)`, i.e. "my queue's previous transfer has completed"), so a load may
be hoisted arbitrarily far earlier without touching its wait, as long as it does not cross
another transfer of the same queue. Hoisting makes data available *earlier* and signals *earlier*,
so it cannot deadlock (unlike sinking, which delays a signal and can close a
Pool-waits-DVE-waits-ACT-waits-Pool cycle -- that is the form I rejected on paper, 2215 of 2385
same-queue pairs have a sem-4/6-waiting store between them).

What a hoist does need is a **WAR check**: the load writes its destination earlier, and the
destinations are reused (1569 loads over only 300 distinct offsets, max multiplicity 39). Use the
same two gates as §1 (any-toucher for the hoist window, since here a *reader* of the previous
contents is exactly the victim) over the window [new position's time, old position's time].
A reuse happens every ~620 us on average, so a 10-50 us hoist should mostly pass; build an
offset-bucketed index first or the 1569 x 32k range tests will take minutes.

### 4.3 ...but the per-queue rule pins the schedule to ~20 slots. Measured, do not assume otherwise.
`iter-4-qdist.py`: the distance from a Pool transfer to the **previous transfer on its own
queue** is median **21 slots**, mean 21.2, p10 18, p90 24, **max 32** -- the compiler assigns
queues strictly round-robin, so per-queue order is very nearly the total order. A load can
therefore be hoisted **at most ~20 slots**, and a block of 16 consecutive loads (one per queue)
cannot be hoisted over *any* transfer at all. The reclaimable time is only the stall contained in
a 20-slot window, i.e. **~10-20 us for the whole program, not 100.** Reordering on its own is
not an iteration's worth.

**The thing that would unlock it is queue *reassignment*** (give a transfer a different
`semaphore`), because that is what decides the order constraint. It is a mechanical but wide
edit: for every wait `(q,16k)` on any engine, identify the transfer it refers to (the k-th on q
in program order) and rewrite it to that transfer's new `(q',16k')`. The hazard to check first is
whether any single wait is relied on to cover *several* transfers of the same queue (then it
cannot be rewritten as one threshold) -- census that before writing any code.

### 4.4 The path after the shipped edit (measure your own, but this is the shape)
`iter-3-cpath.py` on the edited stream (3068 us): covered busy 2367.2 + gaps 701.0 = 3068.2 us.

| busy on path | us | | gap (attributed to the waiter) | us |
|---|---|---|---|---|
| Scalar `ACTIVATE` | 1044.5 | | Scalar `ACTIVATE` (DVE sem 6 / PE sem 5 latency) | 205.9 |
| Tensor `MATMUL` | 529.2 | | Scalar `EVENT_SEMAPHORE` (111 of it = §4.1) | 138.4 |
| Vector `TENSOR_TENSOR_SCAN` | 160.2 | | Vector `STREAM_SHUFFLE` (own program order) | 104.2 |
| Vector `TENSOR_TENSOR` | 146.5 | | Tensor `LDWEIGHTS` (state reload, was 118.9) | 103.3 |
| Tensor `LDWEIGHTS` | 146.4 | | GpSimd `EVENT_SEMAPHORE` (ring throttles) | 37.4 |
| Vector `STREAM_SHUFFLE` | 133.2 | | Vector `TENSOR_TENSOR` | 32.5 |
| Scalar `EVENT_SEMAPHORE` | 52.1 | | Tensor `EVENT_SEMAPHORE` | 24.4 |
| **Scalar `DMA_DIRECT2D`** | **18.8** (was 47.8, 53 instances left) | | | |

**Honest read of the remaining headroom.** Per-engine on-path busy is Scalar 1146.9, Tensor 677.8,
Vector 517.9, GpSimd 24.5 = 2367 us of work that is on the chain, and every structural way to
reduce it is now closed by measurement: the 2339 ACT multiplies (643 us) cannot be merged
(iter-3 §4.1, both directions) and their cost tracks bytes written, not issue overhead (§3);
`LDWEIGHTS` has no duplicates (iter-3 §4.4); Pool's 1550 us is a floor (§2) though only 24.5 us of
it is on the chain. **So the only remaining material target is the 701 us of gaps**, and of those
the two largest (206 us of ACT waiting on DVE/PE, 138 us of ACT waiting on Pool's queues) are
pipeline-fill and issue-rate effects rather than removable work. Expect single-digit percent from
here, not another 5%, unless someone cracks the `state_sb` relayout (iter-3 §5 item 3, ~103 us of
`LDWEIGHTS` gap plus 146 us of SP issue time) or queue reassignment (§4.3).

## 5. Closed / do not redo (added to iter-1/2/3's lists)

* **Merging DGE transfers (any engine) -- closed by §2.** Cost is per 128-descriptor group.
* **Shrinking a DGE transfer to help the issuer -- closed** (same measurement, and iter-3 §1b).
* **The 4-byte column-load "descriptor storm" relayout + on-chip transpose (iter-3 §5 item 2,
  review candidate #2) -- closed by §2**: Pool pays 650 ns for 1 descriptor just as for 128, and
  the DMA engines are not the bottleneck anywhere (10% busy in the one window where Pool is
  pinned, 32% overall).
* **Relaxing ACT's barriers on the Pool queues -- closed by §4.1**: they are already one-per-
  transfer, and the data really is not there.
* **Sinking Pool stores / merging them -- rejected on paper** (deadlock: Pool would block earlier
  on a higher sem-6 threshold while the loads that let DVE reach it sit behind the block).
* `dma_configs_dma_transpose_2d_tile` -- still untried (1 bit, 0 on all 4017 triggers); after §2
  there is no reason to want it.
* **The review's candidate #1** (fold the 605 `EVENT_SEMAPHORE` stand-ins into the preceding
  `ACTIVATE`'s event block) -- not attempted, and the ceiling is now smaller than the review
  thought: those slots total **52.1 us** of path over 1336 instances (39 ns each), and every
  `ACTIVATE` already uses its one update slot for sem 4, so almost none can absorb a second
  update. Treat as ~0.

## 5b. What is left on the state bounce (exact inventory after this iteration)

Counted on the shipped streams:
```
ACT   46 PSEUDO_DMA_DIRECT2D  SB -> SB   (two-hop staging through T(head), iter-3 §1a)
      45 PSEUDO_DMA_DIRECT2D  SB -> var9 (the original HBM path)
     725 EVENT_SEMAPHORE      SEM_ADD_IMM_COMPLETE +16 on sems 40-55 (605 iter-3 + 120 iter-4)
SP   771 PSEUDO_DMA_DIRECT2D  SB -> SB   (725 single-hop reading X[c-1], 46 reading T(head))
      45 PSEUDO_DMA_DIRECT2D  var9 -> SB
```
The 90 remaining pairs are worth ~25 us and split as: **39** rejected because `X[c-1]` is genuinely
busy in both gates, **7** by the true-window writer rule, **17** whose reload waits on a semaphore
outside 40-55 (unpairable by iter-3's method -- pair them by matching the staging address instead,
~5 us), **28** whose save writes HBM `var9` at an offset that does not match the reload's source
(the semaphore pairing is right but the data comes from a *different* save, so converting them
would be unsound -- **leave these alone**).

## 6. Tool notes (additions)

* **`sem_increment` is a `uint8` and a value other than 16 works** (the refuted §2 shipped 32 on
  102 triggers and `semaphores.py` kept `total_inc` exact at 2384 per queue). The 2-D fields
  (`src_num_elem[1]`, `src_step_elem[1]`) are honoured by the hardware: bytes moved doubled
  exactly. No instruction in this NEFF uses `num_elem[1] > 1` natively, so that encoding is now
  verified on the device for the first time here.
* **Pool's DGE ring depth is 1** (every trigger waits for its own queue's previous transfer).
  16 queues -> at most 16 outstanding Pool transfers.
* `prof/DmaPacket.parquet` labels queues `software_dynamic` (Q1-E64..79 = Pool) /
  `hardware_dynamic` (Q2-E64..79 = ACT+SP) / `input` (Q3 = `qSPIO0`); `def.json`'s `dma_queue`
  gives owner and semaphore_set per queue set (ACT 40-55, Pool 8-23, SP HW 56-71, SP IO 24-39).
* `prof/DmaPacketAggregated.parquet` has exactly the 1149 queued transfers with
  `start_ts/end_ts/variable/dest/read_offset/write_offset/write_shape` -- the only practical way
  to put the queued DMAs into a liveness analysis (they are invisible to the instruction streams).
  **All four of those columns are strings, not lists.**
* The GpSimd timeline is **not** in program order in `Instruction.parquet` (764 of 3168
  consecutive rows go backwards in `compiler_pc`): Pool's cheap `EVENT_SEMAPHORE`s retire out of
  order with respect to its DGE triggers. Sort by `compiler_pc` for program order and use the
  `DMA_DIRECT2D` rows only when computing issue gaps.
* `isa.decode_stream` is a **generator**; `list(...)` it before indexing.
* Device-run costs today: `verify.py` ~40 s, `--runs 4` ~75 s, `profile.py --repeats 1 --parquet`
  ~35 s, `--repeats 5` ~80 s. 5-repeat spread was 0.09-0.15% (~3-4 us).
* Reproduce the shipped NEFF from iter-3's (`48972372faa61507c7b22203d7887d14` ->
  `6cc98083836757ee1b8de0eab881c009`):
  ```bash
  neff.py unpack kernel.neff work/ ; cp -r work work_pristine
  profile.py --repeats 1 --parquet prof          # the liveness gate needs the current schedule
  MODE=or GUARD=20000 TGUARD=5000 python scripts/single2.py      # = iter-4-single2.py, 120 pairs
  neff.py pack work/ kernel.neff
  ```
  `iter-4-single2.py` knobs: `MODE=or|and` (pass either gate / both), `GUARD` = margin for the
  write-only+queued gate, `TGUARD` = margin for iter-3's any-toucher gate, `PROF` = parquet dir,
  `DRY=1` for a census only. The true-window writer rule is unconditional. `iter-4-single-fixed.py`
  is iter-3's script with only the `bysem` fix (what the 102-pair row used).
  `iter-4-qstab.py` re-checks a shipped plan (`/tmp/plan2.pkl`) against any profile.

## 7. Where to go next, ranked by what is actually measured

1. **`state_sb` relayout (iter-3 §5 item 3) -- now the biggest single item that is not closed.**
   Prize is `Tensor LDWEIGHTS`'s 103.3 us of path gap (PE waiting for the reload on sems 56-71)
   plus 771 SP triggers x 531 ns = 410 us of SP busy (SP is 19% busy and ~0 us of the path, so
   only the gap converts). Work: repoint the 276 rotating state addresses at the per-head
   resident buffer `T(head) = 0xe780 + head*128` that iter-3 already created, per unit by program
   position, then delete both transfers of each pair. The check that gates it: for a unit, *all*
   accesses (PE `LDWEIGHTS` src, the ACT/DVE writers that update the state, the next unit's save)
   must be repointed together -- redirecting only the reader is wrong because the update is
   read-modify-write on the new buffer. Realistic 50-100 us (1.6-3%).
2. **Queue reassignment for Pool (§4.3)** to unpin the 111 us stall of §4.1. Ceiling ~138 us of
   ACT `EVENT_SEMAPHORE` gap plus 37 us of Pool ring gap. Needs the census in §4.3 first (does
   any single wait cover several transfers of one queue?). Highest ceiling, highest risk.
3. **The 17 unpairable reloads** (§5b), ~5 us, paired by staging address instead of by semaphore.
   Cheap and safe; do it as a side dish.
4. The 98%-error-budget variant of iter-3 §2 (`EXCL=5,6,7,8`, 11 recips, 16 table loads, ~5 us).
   Still deliberately declined by two iterations: it leaves no numeric headroom at all.
5. **Do not** redo: anything in §5, iter-3 §4, iter-2 §4, iter-1's closed list. In particular the
   descriptor-storm relayout and DGE trigger merging are now closed by the §2 cost measurement,
   and the Pool queue cannot be made a hardware DGE queue (§2b).

# iter-3 — selective_scan b1 c2560 s4096 n16 bf16

**Result: 6818.8 -> 6711.7 us (-1.57% on iter-2, -15.26% on the iter-0 baseline, 1.1800x).**
Numerics bit-identical to the baseline (max_abs_err 1.250e-01 / 83% of limit, cosine 0.999944,
output_1 1.812e-04). 5 repeats, spread 0.01-0.02%. `verify.py --runs 3`: all three runs
bit-identical (no race). `semaphores.py` balanced, `launch_rules.py` clean.
4 of 34 members differ from baseline (`DVE0.bin`, `Pool0.bin`, `Activation0.bin`, `PE0.bin`).

Two edits shipped, on top of iter-2:

| # | edit | us | DVE busy | Pool busy |
|---|---|---|---|---|
| — | iter-2 shipped | 6818.0 | 6455 | 6065 |
| 1 | 973 Pool carry `COPY`s merged pairwise (dim-1 strided) | 6801.5 | 6459 | 5765 |
| 2 | 151 `y+=Cs` Pool ADDs pulled **back** onto DVE (the ones DVE was stalling on) | **6711.7** | 6548 | 5539 |

**The kernel is now DVE-compute-bound with almost no slack left**: DVE union-busy 6651 us inside
a 6721 us span — **69.7 us of DVE idle, of which only 10.8 us sits in gaps > 400 ns**.
(iter-2 had 250 us of idle, 193 us of it in gaps > 400 ns.) Pool is 1.1 ms behind DVE and its
spare capacity now has no use. Anything further must **remove DVE instruction duration**, and the
one lever for that (merging DVE instructions) is refuted below with a mechanism.

---

## WIN 1 — Pool 1-element carry `COPY`s merge pairwise. Pool -300 us, wall -16.5 us

The 2560 Pool carry COPYs (`src = scan_c.dst + 2044`, `dst = carry slot`, 1 element x 128
partitions) are **adjacent in time and at a constant stride**: consecutive COPYs write
*consecutive* carry slots (`dst` delta always +4 B) while their sources are different scan-output
buffers. Any **two consecutive** COPYs therefore merge into one instruction:

```
src_mem_pattern_num_elem  = [1,2,1,1]   src_mem_pattern_step_elem = [1,(src_b-src_a)/4,1,1]
dst_mem_pattern_num_elem  = [1,2,1,1]   dst_mem_pattern_step_elem = [1,(dst_b-dst_a)/4,1,1]
start addresses = the *earlier* COPY's; the slot kept is the *later* one (keeps its wait + S3++)
```

* **`num_elem[1]=2` with a `step_elem[1]` works on Pool/GpSimd** — bit-exact, first try
  (8-merge probe, then 973). `step_elem` is signed int16 in **elements** (+-131068 B), which covers
  every delta here (max |delta| 26624 B). This is new: in this NEFF *no* compute slot on any
  engine except PE `LDWEIGHTS` uses a dim above 0, and the gdn_cte experiment reports dim-1
  **hanging** its DVE — it does not hang here (iter-1 of this loop also ran dim-1 on DVE).
* Feasibility: **973 of 1280** possible pairs. Rejections: 428 have a Pool `PSEUDO_DMA_DIRECT2D`
  between them (I treat a DMA trigger as an unknown-address barrier — decoding it would unlock
  them), 141 have an ACT slot waiting on exactly the earlier COPY's S3 ordinal (conservative
  reject, cycle risk), 44 have the earlier COPY updating S3 while the later one does not
  (72 of the 2560 COPYs carry no S3 update at all — probably last-chunk carries nobody reads).
* **0 pairs were rejected for a DVE-side cycle**: the carry hand-off has plenty of slack.
* Renumbering: `newmap[k] = #surviving S3 updaters with old ordinal <= k`, with the deleted
  COPY's ordinal mapped to the **merged (later) slot's** ordinal. Apply to every S3 wait,
  primary and extended, in all five streams (4373 DVE + 20 Pool + 252 ACT fields).
  WAR is preserved because the DVE scans' `S3 >=` waits are exactly what stops DVE overwriting
  a scan-output buffer before the carry is snapshotted; mapping onto the merged slot is strictly
  stronger.
* **Cost model measured: a 2-element Pool COPY saves only ~308 ns, not ~683.** Pool busy
  6065 -> 5765 us for 973 merges. So Pool's fixed overhead per COPY is ~310 ns and the rest is
  per-element/addressing work, i.e. **Pool's 683 ns 1-element COPY is not mostly fixed overhead**
  (iter-1's "299 ns idle / 688 ns loaded" numbers are consistent with that).
* **Wall gain only 16.5 us.** Pool occupancy was *not* what DVE was waiting for — see win 2.
  Further Pool-side savings (the 428 DMA-blocked pairs, triples, the 72 dead COPYs) are now
  **pointless**: Pool is 1.1 ms behind DVE.

## WIN 2 — the real cost of the Pool hand-off: 151 ADDs with 3 slots of slack belong on DVE

Profiling the gaps (union-of-intervals on `Instruction.parquet`, attributing each gap to the
instruction that starts after it) gave a very sharp picture of iter-2's 250 us of DVE idle:

| | |
|---|---|
| 151 gaps of 0.9-1.5 us (183 us, **73% of all DVE idle**) | a DVE `EVENT_SEMAPHORE` waiting `S[3] >= c` |
| what it waits for | always a **Pool `TENSOR_TENSOR` `y+=Cs` ADD**, which ends **37-38 ns before DVE resumes** |
| Pool idle immediately before that ADD | median **-17 ns** (back-to-back) -> Pool is **not** idle, and not reorderable |
| DVE slots from the ADD's producer to the waiting ES | median **3**, min 3, max 5 |

So these are not backlog, they are a **round trip**: DVE computes `Cs` at slot p, Pool's ADD needs
it, Pool's ADD takes 1.28-1.53 us, and the hand-off wait can only be placed 3 slots (~2 us of DVE
work) later because that is where the deadline is. DVE pays 1.3-1.5 us of stall to save 0.67 us of
work. **Moving those ADDs back to DVE is a 2:1 win.**

Recipe (all 151 passed every check; no device failures):

1. Take the awaited Pool ADD (`TENSOR_TENSOR_ARITH_OP`, `src0==dst`, wait `S6>=v6`, update `S3++`).
2. **Insert** it as a new DVE slot immediately **after** the hand-off ES, keeping its wait
   `S6>=v6` and with `events_update_mode = NONE`. (Inserting DVE slots is safe — no PC-relative
   fields. S6 numbering is untouched because the inserted ADD updates nothing.)
3. Delete it from Pool and renumber S3 with `newmap[k] = #surviving updaters with ordinal <= k`;
   for the deleted ADD this maps **down** to the previous survivor — which is exactly right here:
   the ES keeps its `S3` wait and now only has to wait for everything Pool did *before* that ADD,
   which still covers every earlier same-`Y` Pool ADD (Pool is in-order). That is where the
   1.3 us comes from.
4. Checks that make it safe (each one matters):
   * no DVE slot in `(idx_of[v6], ES]` references the ADD's `Y` or `Cs` (preserves the fp32
     accumulation order and the WAR window; guaranteed by iter-1's deadline construction but
     verified independently);
   * every DVE waiter on exactly that S3 ordinal is at a slot `>= ES` (else the ADD would have
     been needed before it now runs); no ACT/PE/SP/Pool waiter on that ordinal at all;
   * the **next Pool ADD with the same `Y`** must have its own producer (`idx_of[v']`) after the
     insertion point, so Pool cannot run it before the restored DVE ADD completes.
5. Result: DVE busy 6459 -> 6548 us (+89, i.e. +0.59 us per restored ADD, as predicted) and the
   wall 6801.5 -> 6711.7 (-90). DVE idle 250 -> 70 us. Bit-exact.

**Generalisable rule for this kernel: a Pool-resident op whose hand-off deadline is <= ~5 DVE
slots costs DVE more in stall than it saves in work.** Slack >= ~8 slots is where the iter-1/2
off-load pays. This also closes, without a device run, the idea of moving the 160 bf16 `dt*x`
MULTs to Pool (iter-2 measured their slack at 2-3 slots): even if the Pool-internal hazards that
corrupted iter-2's attempt were fixed, each one would buy 0.55 us of DVE work for ~1.3 us of
stall.

---

## REFUTED (built, ran, bit-exact, and 37% SLOWER) — merging adjacent DVE `TENSOR_TENSOR` pairs

This is review candidate "fewer, wider DVE TT instructions" / iter-2's next-step #3, taken as far
as it goes. **Do not retry.** The arithmetic saving is real and the mechanism of the loss is now
measured.

**It builds and it is bit-exact.** 2235 of 2793 possible adjacent pairs merge. Key enabler that
iter-1 missed: lines 224 and 267 put the PSUM operand in **opposite** operand slots
(`224: src0=dtx(SBUF), src1=B_full(PSUM)`; `267: src0=C_full(PSUM), src1=state(SBUF)`), so a
same-line-only search finds few pairs. Both ops are **commutative** (fp32 `MULT`/`ADD`), so the
merged instruction may assign either member's operands to either slot — pick the assignment that
puts PSUM in the same slot for both rows, and every adjacent pair becomes expressible:

```
merged (at the LATER slot i2, keeps its own wait and its own S6++):
  for each of src0/src1/dst: start = M1's address, num_elem=[512,2,1], step_elem=[1,(a2-a1)/4,1]
M1's slot i1 -> bare EVENT_SEMAPHORE carrying M1's original wait, no update
S6: drop M1's increment, newmap[ord(M1)] = new ordinal of the MERGED slot
```

* **Measured merged duration: 1220 ns** for 2x512 elements vs 2 x 688 ns — 156 ns saved per pair,
  exactly the fixed-overhead model. A dim-1 DVE pattern is **not** slower per element
  (1.19 ns/elem at 1024 vs 1.34 at 512).
* **And yet: 6711.7 -> 9346.9 us (+39%).** DVE busy 6548 -> 8752, PE busy 2969 -> 2622.
  +2203 us over 2235 merges = **+1.0 us of critical path per merge.**
* **Mechanism: the DVE<->PE PSUM lockstep.** 5120 of the 5586 DVE TTs read a PSUM bank, and PE's
  next matmul into that bank waits `S6 >= ord(that MULT)`. Merging at the later slot moves that
  release one instruction later, so PE's whole pipeline slips; DVE then waits on PE's `S[5]`.
  iter-1 refuted the *earlier*-slot variant for the mirror-image reason (it pulls PE's `B_full`
  dependency one matmul *earlier*: 6886 -> 7100). **Both directions lose, so TT merging is dead
  for any PSUM-operand instruction.** The non-PSUM TTs (306 fp32 ADDs + 160 bf16 MULTs) are never
  adjacent to each other in the stream, so that subset is **empty** (0 pairs).
* Two bugs worth knowing if anything like this is tried again:
  * **`newmap` direction matters.** For a *merge*, the dropped instruction's waiters must map
    **onto the merged slot**; iter-2's off-load `newmap` (map down to the previous survivor) gives
    `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR` / `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`,
    nrt 1004, in ~9 s — PE overwrites a bank the merged MULT has not read yet.
  * **Nothing that increments S6 may sit between the two members.** A cross-engine (or DVE)
    consumer keyed on an in-window ordinal — here a `MEMSET` and a `SCAN` between slots 1062 and
    1066 — is released before the merged slot has written M1's output. Bisected to exactly one
    pair out of 171: `max_abs_err 5.789e+00`, **99.91% of elements still in tolerance**,
    cosine 0.999910. With the `s6_between` rule added, all 2235 merges are bit-exact.
    (Checking only "waiters on `ord(M1)`" is **not** sufficient — that was the hole.)

## REFUTED (two device probes) — DVE's intra-engine pipeline overlap is already maximal

This was going to be next iteration's best idea, so it is settled here. **DVE overlaps exactly one
instruction, the compiler's barriers are tuned to exactly that, and there is nothing to relax.**

How DVE orders itself: it does **not** interlock its own RAW/WAR. Every real same-engine hazard is
an explicit wait on `S[6]` (DVE's own completion counter). Distribution of the 5897 DVE `S[6]`
waits by **lag** = `(#S6 updaters before the waiting slot) - (wait value)`:

| lag | meaning | count |
|---|---|---|
| 0 | wait for my *immediate* predecessor to complete (fully serial) | 393 (6.7%) |
| **1** | **allow exactly one predecessor to still be in flight** | **4955 (84.0%)** |
| 2-5 | allow 2-5 in flight | 549 (9.3%) |
| >=8 | — | 0 |

* **Probe A (relax).** Lower a lag-1 wait to lag-2 wherever the instruction whose completion is
  dropped **provably does not alias** the waiter or any following un-gated slot (full DVE
  read/write range check). 237 of 2218 primary lag-1 sites qualify (907 reject on a genuine
  alias — the barriers are mostly real). Applied: bit-exact, **6711.6 us vs 6712.4 — no change**
  (spread 1.1 us).
* **Probe B (tighten, the control).** Push **all 1422** lag-1 primary waits to lag-0, i.e. forbid
  the one allowed overlap. **6712 -> 6920 us, +208 us = +146 ns per site**, DVE busy +120 us.
* Conclusion: the waits really do gate overlap (B), and 146 ns per instruction is what the single
  in-flight slot is worth — but going from 1 to 2 in flight buys **zero** (A), so the hardware
  depth is 1 and the compiler is already at it. The 594 us of "intra-DVE overlap" visible in the
  profile is that one slot, fully exploited. **Delete this lever.**

## The exact threshold for "Pool or DVE?", measured — and what re-addressing has to buy

Why the 151 restored ADDs cannot go back to Pool even with a perfectly placed hand-off wait
(iter-1's greedy could only reach the 3rd slot because it may only reuse a *freed* ADD slot;
an inserted ES could reach the true deadline, so this was worth checking):

* Pool's measured latency chain for one of these ADDs: **1.15 us** from the DVE producer's end to
  Pool starting the ADD (Pool is back-to-back busy, not idle) **+ 1.28 us** for the ADD
  = **2.43 us** before DVE may proceed. The ADD itself is 0.67 us of the DVE work in that window,
  so the window must hold **>= ~3.1 us of DVE work** for the hand-off to be free.
* Measured window (producer -> true deadline, deadline including cross-engine ACT/PE references
  mapped back through their `S[6]` waits, DVE durations summed at 688/1297/412/35 ns):
  **median 2204 ns**, mean 2463; `>=2500 ns`: 29 of 156 sites; **`>=3100 ns`: 9 sites**.
* So the best case is 9 sites x ~0.3 us = **~3 us**. Dead as a scheduling edit, and this is the
  number re-addressing has to move: **the window must roughly double, from 2.2 to >3.1 us.**
  Free SBUF for that: the scratch pool tops out at 62272 and the input DMA region starts
  at ~113 k, so there is **~50 KB/partition free**, i.e. ~24 extra 2048 B buffers to rotate
  through (151 distinct fresh buffers would need 309 KB and is not available).
* Related micro-lever, also dead: **`MEMSET` merging.** The 200 MEMSETs sit ~87 slots apart
  (one per chunk-group); only **3** adjacent pairs have no S6 updater between them. ~0.5 us.

## Closed / refuted elsewhere this iteration

* **Merging scan chunks** (the idea that the 8 sequence chunks of one chain could become one
  wider `TENSOR_TENSOR_SCAN_ARITH`, since iter-1 proved the carry does not reset across
  access-pattern rows): **dead on addressing.** Dumped all 2560 scans grouped by carry pointer
  (80 pointers x 32 scans). Within one chain the chunk operands have **no constant stride** at
  all (`src0` 20800, 28992, 18752, 24896, 26944, ... and `dst` likewise), and consecutive chunks
  of a chain are ~86 DVE slots apart. There is no affine pattern to express.
* **Pool occupancy is no longer a lever** (1.1 ms behind DVE): the 428 DMA-blocked COPY pairs,
  COPY triples (116 exact `(+12288,+4)` runs of 3 exist), and the 72 S3-less COPYs are all
  worthless now.
* `performance_mode` is empty for every engine in `Instruction.parquet` — no perf-mode lever
  visible from the profile.

## Where the time is now (measured on the shipped NEFF)

| engine | profiler busy | union-of-intervals | note |
|---|---|---|---|
| DVE | 6548 us (97.6%) | span 6721, busy **6651**, idle **70** | the wall, with ~1% of slack |
| Pool | 5539 us (82.5%) | span 6719, busy 5551 | 1.1 ms behind DVE; irrelevant now |
| PE | 2969 us (44.2%) | | paced by PSUM banks, i.e. by DVE (and it paces DVE back) |
| ACT | 1903 us (28.4%) | | only `COPY`/`ACTIVATE` run here |

DVE instruction duration: `TENSOR_TENSOR` 5586 x 688 = 3843 us, `TENSOR_TENSOR_SCAN` 2560 x 1297
= 3320 us, `MEMSET` 200 x 412 = 82 us -> 7245 us of duration inside a 6651 us union, i.e. only
**8% intra-DVE pipeline overlap**. Of the 5586 TTs, **5120 have a PSUM operand** (immovable: Pool
cannot read PSUM, iter-2; unmergeable: the PE lockstep, above). The 2560 scans are DVE-only and
unmergeable. **That is 6.4 ms of permanently DVE-resident, un-restructurable work against a
6.71 ms wall.**

## What to try next (ranked, and honest about the size of what is left)

Every engine-level and sync-level lever is now closed by measurement. What is left needs
**addresses** to change, not schedules.

1. **Re-address the shared scratch pool — the only lever with >1% in it, and now quantified.**
   The deadline that limits the Pool hand-off is **buffer reuse**, measured directly: for 148 of
   the 156 restored DVE ADDs, the first later DVE slot touching their `Y` or `Cs` is a
   `TENSOR_TENSOR` that **writes `Y`** (i.e. recycles that address as a `dtB`/`Cs`/`Y` buffer),
   at a median of **8** slots (min 4, max 26); 8 of them are a write to `Cs`. The scratch pool is
   `16384..58176` (two interleaved families at +2048 B, which actually overlap each other, so
   live ranges are packed), inputs start at ~113 k, and there is **~55 KB/partition free**.
   Stretching the recycle distance (a 4-deep `Cs`/`Y` rotation) would:
   * let the 151 tight ADDs move back to Pool with real slack (DVE -100 us, Pool has 1.1 ms free);
   * make the Pool-internal hazards that corrupted iter-2's bf16 `dt*x` move tractable
     (another ~88 us of DVE work);
   * and relax the `s6_between`/consumer constraints that blocked the TT merges — though the PE
     lockstep refutation below is independent of addresses, so merging stays dead.
   It is a large coherent edit: re-point the ACT `dA` writes, the line-224 `dtB` writes, the scan
   operands and the `Y`/`Cs` families together, then re-run iter-1's deadline/stabbing.
2. **Same-PSUM-bank TT merges only** (small, risky). `B_full[b]` (`0x2000000+b*0x1000`) and
   `C_full[b]` (`0x2000800+b*0x1000`) live in the *same* 4 KB bank, so merging a 224/267 pair with
   the same `b` would not delay PE's bank release at all and the 156 ns would be free. Such pairs
   are **not adjacent** in the stream (the schedule interleaves `b` and `b+1`), so this needs
   non-adjacent merging, which reopens the `s6_between` hazard. Ceiling <=~100 us against a
   measured 1 us/pair penalty if the bank reasoning is wrong.
3. The 2560 scans (3320 us, 50% of DVE) and the 5120 PSUM MULTs remain the hard floor:
   ~6.4 ms of DVE-resident work against a 6.71 ms wall, with 1-deep pipeline overlap already
   fully exploited and 70 us of idle.

## Tooling notes

* `isa.py` encodes `*_mem_pattern_num_elem` / `*_mem_pattern_step_elem` from a Python list
  correctly, including **negative** steps (round-tripped), and setting
  `*_start_addr_addr_immediate` rewrites the whole address word (the `_reg_regnum` /
  `_reserved0` fields are aliased views of the same bytes, so they follow automatically).
  For a Pool `COPY` the only bytes that change for a pairwise merge are 18-19 (src `step_elem[1]`),
  26 (src `num_elem[1]`) and 58 (dst `num_elem[1]`).
* Attributing DVE idle: union-of-intervals over `Instruction.parquet` (`engine=="Vector"`), then
  map the gap to the instruction whose `start_ts` equals the gap's end and read **its own wait
  fields** out of the NEFF (`compiler_pc` is the slot index). This is what turned "250 us of
  idle" into "151 sites, one semaphore, 3 slots of slack" in a single pass — much more useful
  than `evt_wait_time_ns`.
* Watch out for `python ... 2>&1 | grep ...` swallowing an `AssertionError` after a progress
  print: one build "succeeded" silently without writing anything and the following `verify.py`
  happily re-verified the *previous* NEFF. Check a structural invariant of the edit
  (here: "TTs with `num_elem==[512,2,1]`") before trusting a device result.
* Device-run cost unchanged: ~10 s working, ~9 s PSUM fault, ~70 s deadlock. A full
  `profile.py --repeats 5` is ~70 s; `--repeats 1 --parquet` ~25 s.

---

## Build recipe for the shipped NEFF (from iter-2's `kernel.neff`, i.e. HEAD of this branch)

`neff.py unpack kernel.neff work/`, run the two steps, `neff.py pack work/ kernel.neff`.
Final measurement: **6712.4 us**, median of 5, min 6711.65 max 6712.61, spread 0.01%.

### Step 1 — merge Pool carry COPYs pairwise (973 merges)

```python
# candidates: consecutive Pool COPY slots a,b (COPY order, not slot order)
#  accept iff  b.wait_value >= a.wait_value  (both wait S6; wait of merged = b's)
#        and  (b.src-a.src)%4==0 and |(b.src-a.src)/4|<=32767, same for dst
#        and  neither src equals either dst (no intra-instruction alias)
#        and  no Pool slot strictly between them writes [a.src,+4) or touches [a.dst,+4)
#             (treat PSEUDO_DMA_DIRECT2D as an unknown-address barrier -> 428 rejects)
#        and  no DVE slot waiting S3 >= a.s3_ordinal is at a slot <= idx_of[b.wait_value]
#             (0 rejects here) ; no ACT slot waits on exactly a.s3_ordinal (141 rejects)
#        and  not (a updates S3 and b does not)                             (44 rejects)
# edit: slot b  <- src/dst start = a's, num_elem[1]=2 on src and dst,
#                  src step_elem[1]=(b.src-a.src)/4, dst step_elem[1]=(b.dst-a.dst)/4
#       slot a  <- deleted
# renumber: newmap[k] = #surviving S3 updaters with old ordinal <= k ;
#           then newmap[a.ord] = newmap[b.ord]      <-- onto the MERGED slot
#           apply to every S3 wait (primary + extended) in all five streams
```

### Step 2 — pull the stalling `y+=Cs` ADDs back from Pool onto DVE (151)

```python
# find the sites from a profile:  union-of-intervals over DVE rows of Instruction.parquet,
#   keep gaps > 900 ns whose following instruction waits S3 >= c (an ES), take the Pool slot
#   that supplies S3 ordinal c (always a TENSOR_TENSOR ADD; 151 sites, 183 us)
# accept iff  the Pool ADD waits S6 >= v6 only, src0 == dst, no extended wait
#       and  no DVE slot in (idx_of[v6], q] references its Y or Cs      (q = the waiting ES)
#       and  every DVE waiter on exactly c is at a slot >= q, and no ACT/PE/SP waiter on c,
#            and no Pool waiter on c in (pool_slot, q]
#       and  the next Pool ADD with the same Y has idx_of[its v6] > q
# edit: INSERT the identical 64 bytes as a new DVE slot at q+1 with events_update_mode=NONE
#       (keeps wait S6>=v6; inserting DVE slots is safe, no PC-relative fields;
#        sort multiple insertions at one position by their old S3 ordinal)
#       delete the Pool ADD
# renumber: newmap[k] = #surviving S3 updaters with ordinal <= k, and for the DELETED ADD
#           leave it mapping DOWN to the previous survivor (this is the point of the edit:
#           the ES then only waits for what Pool did before that ADD)
```

Both steps keep every instruction's 64 bytes byte-identical apart from the fields named, so the
arithmetic is a pure relocation/re-pattern and the output stays bit-exact.

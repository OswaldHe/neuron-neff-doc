# iter-3 — selective_scan b1 c768 s1024 n16 bf16

**Result: 516.456 -> 510.69 us (-1.12%). Cumulative from the unedited NEFF: 602.64 -> 510.69
(-15.26%, 1.180x).** Numerics unchanged: output_0 max_abs_err 6.245e-02 (83% of limit,
cosine 0.999998), output_1 4.268e-05 — the exact baseline figures. `verify.py --runs 3`:
**all 3 runs bit-identical** (no race). `semaphores.py` clean, `launch_rules.py` clean.
Three independent 5-repeat profiles of the shipped file (sha256 `0c18b05f...`, md5
`55a5c384...`): medians **510.673 / 510.690 / 510.622** (spread 0.04-0.08%), DVE busy
493.6-493.9 us, Pool 415.1-415.3.
vs the iter-2 artifact only **2 of 34 members changed**: `sg00/DVE0.bin` (1216 -> 1057 slots)
and `sg00/Pool0.bin` (2 slots, one field each).

Read `iter-1.md` (build scripts, calibration) and `iter-2.md` (the big refutation list) first;
everything in them still holds. **The one correction to iter-2 is R4 — see "The unlock" below —
but following that corrected lever to the end turned out to be a loss anyway (measured).**

## What shipped (two edits, both on the iter-2 artifact)

| # | edit | us | DVE busy | Pool busy |
|---|---|---|---|---|
| — | iter-2 artifact, re-measured | 516.456 | 502.29 | 415.21 |
| 1 | strip the 170 now-redundant S6 self-waits off the ex-ADD `EVENT_SEMAPHORE` slots, then delete the 159 that become pure no-ops | 515.882 | 501.86 | 414.91 |
| 2 | **collapse the 12 y-accumulator `MEMSET`s (512 fp32 x 128ch, 485 ns each) into one shared zero buffer** | **510.673** | **493.89** | 415.06 |

### Edit 1 — the 170 ex-ADD ES slots (worth only 0.57 us, but free)
iter-1 moved 170 `y+=Cs` ADDs to Pool and left an `EVENT_SEMAPHORE` on each freed DVE slot that
*kept the ADD's own `S6 >= v` wait* ("keep the ADD's own dependency"). That wait is redundant:
the Pool copy of the ADD carries the identical wait, and the ES has no update, so its only effect
is to stall DVE. Of the 170: 147 had no extended wait, 20 hosted the S3 hand-off, 3 hosted an S4
wait. Stripping the primary wait from all 170 and deleting the 159 that then carry nothing at all
(no wait, no update -> deleting them cannot change any ordinal) gave **-0.57 us** (515.882 vs
516.456, both 5-repeat medians, spread 0.04%). So DVE S6 self-waits on *idle* ES slots are nearly
free — iter-1's "4.7 us of S6 gap" is mostly on real compute slots, not these.
**After this edit there are 0 pure-no-op ES slots on any engine** (DVE 410 ES: 299 with primary +
extended wait, 100 primary only, 11 extended only; Pool 42, PE 347, ACT 28, SP 271 — all have at
least one live wait). The iter-2 review's candidate #1 is now closed.

**How to find the 170 slots again (exact alignment recipe).** The current DVE stream = baseline
DVE stream + 12 inserted ES - nothing deleted (before this iteration). The 12 inserted ES are
exactly the DVE ES slots whose **primary** wait is on S3 (`[102,202,302,403,504,607,719,808,921,
1011,1125,1212]` in the iter-2 artifact); remove those 12 and the stream aligns 1:1 with the
1204-slot baseline, with exactly 170 opcode mismatches, all at the baseline's moved-ADD slots and
all with matching wait values. (iter-1's `FRESH=[97,197,...]` are the *ADD* indices, **not** the
insertion points — build3 inserts at `c["d"]`, the deadline. Do not use FRESH for index mapping.)

### Edit 2 — one zero buffer instead of 12 memsets (worth 5.2 us)
The 24 DVE `MEMSET`s are 12 y-accumulators (`dst_element_count=512`, fp32, nac=128, `set_value=0`,
**485 ns each**) + 12 small carry/state inits (16 elem x 128ch and 128 elem x 1ch, ~119 ns).
Only the 12 big ones matter. `y` is accumulated as `0 + Cs_0 + Cs_1 + ...`, so if the group's
**first** y writer reads a zero buffer as `src0` instead of `y`, the memset is unnecessary:

* `MEMSET` slot 3 was re-targeted from `y=0x4940` to a **shared zero buffer at `0x15400`**
  (2048 B/partition, inside the free SBUF gap `0x15280-0x1c000`; iter-2 had already proved
  `0x15400` is free). It keeps its 512 elements, so it is the only full-size memset left.
* The other 11 were shrunk to `dst_element_count=1` / `num_elem=[1,1,1,1]` (~100 ns), everything
  else byte-identical. They still write `y[0]=0`, which the first writer then overwrites.
* Each group's first y writer got `src0 -> 0x15400`. Finding it: for each memset take its S6
  ordinal `o`, then the earliest writer of that y address with order key `> o`, where a DVE ADD's
  key is its own S6 ordinal and a Pool ADD's key is `its S6 wait value + 0.5`. Result: **10 of 12
  first writers are DVE ADDs** (slots 105,192,280,369,457,547,630,724,899,982) and 2 are Pool ADDs
  (slots 8, 303).
* Arithmetic is identical (`0.0 + x == x` in fp32; only a `-0.0` input would change sign, which
  cannot move `max_abs_err`). Output came out bit-identical.

Gain **-5.21 us** (515.882 -> 510.673), bigger than the 11 x 485 ns = 5.3 us of duration predicted,
because DVE MEMSET total duration fell 2556 ns vs ~7.2 us before and the freed slack also removed
a little gap. DVE busy 501.86 -> 493.89.

## The unlock, and the three refutations that used it

### U1. `EVENT_SEMAPHORE` **cannot** carry an `S6` increment that another engine's PSUM WAR guard depends on — but a 1-element `MEMSET` can. (NEW, and it corrects iter-2's R4.)
First attempt at edit 2 turned the 11 memsets into `EVENT_SEMAPHORE`s that kept the same wait and
the same `SEM_INC_COMPLETE S6`. Semaphore accounting was perfect (S6 still 640/640, every ordinal
preserved). Device: **`TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`, nrt 1004, 9.9 s.**
Mechanism: `S6` is `SEM_INC_COMPLETE`, and PE's PSUM WAR guards are `S6 >= k` meaning "DVE's read
of that bank has completed". An `EVENT_SEMAPHORE` **retires immediately**, so it increments S6
before the preceding PSUM-reading MULT has finished — PE then overwrites the bank.
**A real instruction's `SEM_INC_COMPLETE` is ordered behind the previous instruction's completion;
an ES's is not.** Replacing the same 11 slots with a **1-element `MEMSET` to scratch (~100 ns,
`num_active_channels=128`, `set_value=0`)** passes and is bit-exact.
=> **Reusable tool: to free a DVE compute slot while preserving its S6 ordinal *and* the
completion ordering every waiter relies on, put a 1-element MEMSET there, not an ES.**
This is ~100 ns instead of iter-1's "585 ns S6-self-wait guard", so it reopens every lever iter-1
and iter-2 closed with the argument "the replacement ES must also wait, which costs more than the
move saves". I reopened it and it still loses, for a different reason — R2 below.

### R2. Moving the remaining 25 movable `y+=Cs` ADDs to Pool is a **measured 10 us loss**. Closed.
(iter-2's R4 predicted a loss from the wrong mechanism; the conclusion survives.)
DVE still had 34 `y+=Cs` ADDs: 13 (wait S6, update S6), 9 (**wait S4**, update S6), 8 (wait S6,
src0 = the new zero buffer), 2 (no wait), 2 (no wait, src0 = zero buffer). Built the full move with
iter-1's `build3.py` machinery plus U1: Pool ADD gets `wait S6 >= (DVE ordinal just before it)` and
`update S3`, the DVE slot becomes a 1-element MEMSET keeping its S6 update, the S3 hand-off wait is
hosted as late as legal on a DVE ES's free extended wait (17 hosts) or a fresh ES (3), and all S3
waits are renumbered in all five streams.
* **25 moved (all except the 9 S4-waiting): verified bit-identical, and 520.679 us — a 10.0 us
  regression.** DVE busy fell 493.6 -> **476.6** (-17 us, exactly as predicted) but exec rose:
  DVE idle went from 11 to ~44 us. The hand-off deadlines are the problem — slacks were
  `[2,2,2,2,3,3,4,5,5,5,5,5,8,8,8,8,8,9,11,11,19,72,75,76,156]` DVE slots, and a Pool TT is
  ~2.0 us, so a 2-5-slot deadline is a guaranteed stall. Pool only reached 444 us busy, i.e. Pool
  was **not** saturated: this is a latency loss, not a throughput loss. Do not try to fix it by
  freeing Pool (moving its COPYs to ACT etc.).
* **Filtering to the high-slack ones does not help either:** slack is self-referential (an ADD left
  on DVE is itself a Y deadline for its neighbours), and the fixed point of
  "keep only slack >= 20" is **2 ADDs** (slacks 71, 72) = ~1.2 us of duration. Not worth a run.
* **All 34 moved (incl. the 9 that wait on S4): NRT_TIMEOUT, 70 s (deadlock).** The S4 dependency
  has to go on a Pool ES (`wait S4 >= v`) ahead of the ADD, and that closes a cycle: Pool blocks on
  ACT, ACT's `ACTIVATE`s wait on Pool's S3 / DVE's S6, DVE blocks on Pool's S3. This is iter-2's
  "S[4] ADDs hung the sibling", now with the reason. **Never put an S4 wait on Pool.**
* Balance model for the record (fits all three points): `DVE_busy ~ 494 - 0.68k`,
  `Pool_busy ~ 415 + 1.15k` for k ADDs moved, but exec `~ 510 + 0.4k`. **The DVE/Pool split is
  past its optimum; DVE idle, not DVE busy, is what the next us has to come from.**

### R3. Pair-merging the 192 `Cs = C_psum * state` MULTs is a **measured 76 us loss**. Closed, with the duration numbers that show why it looked attractive.
This was the iter-2 review's only remaining >2% lever (and iter-1's #2). Built end to end:
merge `Cs_{2k}`,`Cs_{2k+1}` **at the later slot** with `num_elem=[512,2,1]` and per-pair dim-1
steps (`src0` PSUM +1024 elem, `src1`/`dst` whatever the two buffers differ by), keep the later
slot's `S5` wait, put a 1-element MEMSET on the earlier slot to preserve its S6 ordinal, and
re-point every `S6` wait whose value equals the deleted read's ordinal to the merged slot's ordinal
(163-196 wait fields across the five streams).
* **1 pair: exit 0, bit-identical.** So a 2-row access pattern that crosses PSUM banks is legal,
  dim-1 steps on all three operands work, and re-pointing PE's WAR guard 3 ordinals later is safe
  at n=1.
* **85 pairs: exit 0, bit-identical, 586.523 us (+75.8 us, -15% of the win).** DVE busy went
  *up* 493.6 -> 525.1 and DVE compute union 497.7 -> 532.8 us, even though the merged instruction
  is genuinely cheaper: measured **1210 ns for 1024 elements vs 2 x 675 ns**, i.e. 157 ns fixed +
  1.018 ns/elem exactly as iter-1 calibrated. The merged read holds two PSUM banks for 1210 ns and
  `TENSOR_TENSOR_SCAN` mean duration rose 1275 -> 1462 ns (median unchanged at 1278, so it is
  blocking, not slower). **PE/DVE PSUM lockstep eats ~5x the 140 ns/pair the merge saves.** Same
  mechanism iter-1 measured for the line-224 merge at c2560 (214 us worse). Both merge directions
  for both MULT families are now refuted **on this shape**.
* Two legality rules found the hard way, worth keeping if anyone retries this:
  1. **Reject a pair if any DVE slot strictly between the two reads or writes the earlier pair's
     `state` or `Cs` buffer** (2 of 96 pairs). Without this: exit 0 but `max_abs_err 1.032e+01`,
     `cosine 0.994813`, 89.3% within tol — a stale-read, not a crash.
  2. **Reject a pair if any instruction anywhere holds an `S6` wait *strictly inside* the ordinal
     window `(ord(a), ord(b))` while referencing the pair's PSUM bank / state / Cs** (9 of 96).
     Bumping those waits up to `ord(b)` instead of rejecting the pair **deadlocks** (NRT_TIMEOUT,
     69.5 s): PE blocks on the delayed C-bank WAR guard and therefore never emits the B matmul
     that DVE needs in order to reach `ord(b)`.
  * The buffer layout is **not** a clean 3-deep rotation with a constant stride — per group the
    `state`/`Cs` triples differ and some are irregular (e.g. group 2's `Cs` goes
    `0x6800, 0xe340, 0xc800`). Observed pair strides: `(1024,3072,3072)`, `(1024,3072,3584)`,
    `(1024,3152,3152)`, `(1024,3584,4096)`, `(1024,4096,4096)`, `(1024,4176,4176)`. Groups are
    exactly 16 Cs MULTs each, in program order, so `cs[16*g + n]` is correct.

### R4. Letting Pool's `COPY`/`PSEUDO_DMA_DIRECT2D` runs overtake the `y+=Cs` ADDs is a **measured 21.7 us loss**. Closed.
This was the only lever the gap attribution pointed at (see below: 2.9 us of DVE gap is bound on
DGE queue semaphores S9/S11, and at one of them the Pool chain
`carry COPY (pc=70) -> carry spill DMA (pc=72) -> DVE wait S11>=32` is queued behind a 2.15 us
Pool ADD at pc=69). Generic edit: in the Pool stream, move each `TENSOR_TENSOR_ARITH_OP` after the
following maximal run of non-ADD slots whenever the run contains a DMA trigger, their touched SBUF
extents are disjoint (exact extents from `num_elem`/`step_elem`, not bounding boxes) and no run
member waits on an S3 ordinal `>=` the ADD's; then remap S3 ordinals by instruction identity in all
five streams. 30 ADDs moved, verified bit-identical, **532.412 us (+21.7 us)**. DVE busy rose
493.6 -> 515.7 while Pool busy *fell* 415 -> 393.7. The hand-off waits for the delayed ADDs bind
much harder than the DMA queues do. This confirms iter-1's refuted-item 2 in the opposite
direction: **Pool's instruction order is already at its optimum; the ADDs are more urgent than the
DMA triggers.**
Targeted follow-up (not built, closed by arithmetic): of the two DMA-bound gaps, only the
S11 one (2.0 us) is actually blocked by a Pool ADD — the other trigger (Pool pc=45, the y
output DMA, sem 9) sat waiting on its *own* `S6 >= 54` from t=58.36 to 60.46 with no Pool ADD in
front of it. So the targeted version's ceiling is ~1.4 us (0.27%), and the delayed ADD's own S3
hand-off is likely on DVE's critical path at that same boundary, so the net is probably ~0.

## Where the 510.7 us goes (fresh, shipped NEFF, `prof/Instruction.parquet`, `compiler_pc.notna()`)

| | |
|---|---|
| **DVE compute union (excl. `EVENT_SEMAPHORE`)** | **497.7 us** |
| DVE compute span | 508.6 us |
| DVE gap, **25** gaps | **10.95 us** |
| &nbsp;&nbsp;startup input DMA (before pc=5, `dtx = dt*x`, no supply entry -> DGE queue S9) | 3.63 us |
| &nbsp;&nbsp;before pc=178: DGE queue **S11>=32** (carry spill, Pool pc=72) | 2.00 us |
| &nbsp;&nbsp;before pc=10 and pc=38: **ACT S4** ordinals 1 and 7 (`dA` late, binding to 33 ns and 101 ns) | 2.03 us |
| &nbsp;&nbsp;before pc=97: DGE queue **S9>=32** (y output DMA, Pool pc=45) | 0.92 us |
| &nbsp;&nbsp;21 gaps of 130-335 ns (issue granularity) | 2.37 us |

**Gap attribution recipe that produced this (iter-1's, with the fix that matters):** build
`supply[(sem, ordinal)] = end_ts` by walking each engine's stream counting primary **and** extended
`SEM_INC_COMPLETE`, then for each DVE compute gap take the wait with the latest supply over all
slots in `(prev_pc, pc]`. **The profile's engine names are `Vector`/`GpSimd`/`Tensor`/`Scalar`/`Sync`,
not DVE/Pool/PE/ACT/SP** — match on those or you get zero supply entries. `compiler_pc` equals the
slot index exactly on all five streams. A wait on a DGE queue (S8..S23) has **no supply entry** (the
DMA hardware increments it), so a gap whose latest in-stream supply is >1 us before the gap ends is
DMA-bound — that is how the three DMA rows above were identified.

DVE duration census: `TENSOR_TENSOR` 430 x 696 ns = 299.3 us, `TENSOR_TENSOR_SCAN` 192 x 1275 ns
= 244.8 us, `MEMSET` 24 x 107 ns = 2.6 us, 410 `EVENT_SEMAPHORE`. Sum 546.7 us vs union 497.7 us,
i.e. **~76 ns of overlap per compute instruction** (645 of them). Per-engine busy: DVE 493.6
(96.7%), Pool 415.1, PE 218.7, ACT 145.9, DMA 107.0, SP 48.4.
Instruction census: DVE 1057 = 430 TT + 192 SCAN + 410 ES + 24 MEMSET + 1 SET_ORDERING_MODE;
Pool 389 = 170 TT + 112 COPY + 60 `PSEUDO_DMA_DIRECT2D` + 42 ES + 5 other.
The 430 DVE TT = 192 `dBx = dtx*B_psum` + 192 `Cs = C_psum*state` + 34 `y+=Cs` + 12 `dt*x`.

**Accounting that bounds everything: exec 510.7 = DVE compute union 497.7 + 10.95 us of DVE gap +
~2 us of prologue/drain lapping in. 384 of the 430 TTs read PSUM and cannot leave DVE (iter-1 #8,
iter-2 R1); the 192 scans cannot leave DVE (iter-2 R3), cannot merge (iter-1) and are
dtype-insensitive (iter-1 #6). That is 497 us of immovable DVE duration.**

## What I would try next — honestly, I think the kernel is done

My estimate of the remaining headroom is **~4 us (0.8%)**, all of it in DVE gap, and I could not
find a lever for any of it. Ranked:

1. **Nothing engine-level, nothing merge-level, nothing schedule-level.** R2, R3 and R4 close the
   last three candidates
   that iter-1/iter-2 and the iter-2 review ranked as the only >2% levers, both by device
   measurement on *this* NEFF (not inherited from the sibling). Combined with iter-2's R1/R3 and
   iter-1's #6/#8, every instruction DVE runs is now known to be un-offloadable, un-mergeable and
   dtype-locked.
2. The 2.03 us of ACT-bound gap (`dA` ordinals 1 and 7) is first-group pipeline fill: ACT's first
   two `ACTIVATE`s cannot start until `dt` lands. Same class as the startup DMA. **Closed.**
3. The 3.63 us startup input DMA: iter-2 closed it by arithmetic (~0.2 us of hoistable work).
   The only real fix is spreading the first 256 KB load over more DMA queues — a descriptor
   restructuring, not an instruction edit. Unattempted, and I would not attempt it.
4. Sub-us leftovers, each below the 0.04-0.17% measurement spread, listed only so nobody spends a
   run on them: moving the 2 high-slack `y+=Cs` ADDs (~1.2 us of duration, R2); shrinking the
   remaining 512-element zero-buffer MEMSET to 1 element by giving the 12 first-writer ADDs
   `src0_mem_pattern_step_elem[0] = 0` (~0.4 us, and step 0 is still unverified); dropping the 11
   placeholder MEMSETs to `num_active_channels=1` (~0 — 100 ns looks like the issue floor, the
   12 original small memsets measure 119 ns at 16-128 elements).
5. If a future iteration wants to burn a run on something: **re-point PE's PSUM WAR guards off S6
   onto Pool's S3** so DVE instruction removals stop being blocked by PE. That is the common root
   of R2's and R3's failures (PE and DVE are in lockstep on 8 PSUM banks). PE has 347 ES slots that
   could host such waits. It is a large, coherent edit with a high deadlock risk and I would put
   its expected value below 1%.

## Gotchas added this iteration

* `semaphores.py` reported perfectly balanced for **both** the PSUM-collision build and the
  deadlocking build. It stays a necessary, very weak check.
* `isa.py`: `TENSOR_TENSOR_ARITH_OP` (`S3S3D3_TT_STRUCT`) has **no** `events_extended_*` fields at
  all — `ins.set("events_extended_wait_mode", ...)` raises `KeyError`. Only `CTRL_ES` has them.
* `MEMSET` carries both `dst_element_count` and `dst_mem_pattern_num_elem`; set both.
  Its fill value is `set_value` (0 here), and `serialization_mode = SERIAL`.
* Address fields decode as three aliased views of the same bytes
  (`..._immediate`, `..._reg_regnum = addr & 0xff`, `..._reg_reserved0[0] = (addr >> 8) & 0xff`);
  setting `..._immediate` writes all of them correctly, but re-decode and assert after `encode()`
  — every script here does.
* Device run time is still the error oracle: ~10 s = ran; ~10 s with `PSUM_COLLISION`/nrt 1004 =
  WAR violation; ~70 s with `error: 5` = NRT_TIMEOUT = deadlock or an unretirable instruction.
  Wrong-but-finite output (exit 0) = a stale read. All four were seen this iteration.
* SBUF free map, re-derived from the union of every compute operand extent + `imm*_ptr` over all
  five streams: used `0x4000-0x15280`, `0x1c000-0x31a40`, `0x37200-0x38300`; free gaps
  `0x15280-0x1c000` (28032 B) and `0x31a40-0x37200` (22464 B). This NEFF now uses `0x15400`
  (2048 B zero buffer, read every group) and `0x15C00` (1-element scratch, written by the
  placeholder memsets).
* Build scripts for everything above lived in `xp/` (gone after the iteration):
  `apply_e1.py` (ES self-wait strip + delete), `apply_e2.py` (zero-buffer memset collapse, env
  `NEL=1` for the shipped shrink / `NEL=0` for the ES variant that collided), `apply_e4.py`
  (y-ADD move, env `SKIPS4`/`MINSLACK`/`NMOVE`), `merge_cs.py` (Cs pair-merge, env
  `NPAIR`/`POSONLY`), `gaps.py`, `scan*.py`. They are small variants of iter-1's `build3.py`
  skeleton, which is reproduced in full in `iter-1.md`; the two shipped edits are ~40 lines each
  and fully described above.

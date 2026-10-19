# iter-1 — three edits: (A) fused all 160 DRAM staging round-trips onto a new queued-DMA queue
# and neutralised the 160 now-dead ACT DGE saves; (B) replaced the 320 SP DGE 16x16
# `DMA_TRANSPOSE` slots with 161 *multi-tile* queued transposes; (C) hosted the resulting
# relay chain on Pool instead of SP

## Result: **873.1 us, -26.3%** vs my own re-measure of the baseline (1184.65 us), 1.356x

| NEFF | median us | captures / spread |
|---|---|---|
| baseline (re-measured, 5 rep) | **1184.646** | 1183.2 1183.8 1184.6 1185.2 1185.3 (0.18%) |
| after edit A only, run 1 (5 rep) | 1013.056 | 0.43% |
| after edit A only, run 2 (5 rep) | 1010.632 | 0.43% |
| after edit A only, run 3 (5 rep) | 1011.827 | 0.20% |
| A+B, 5 rep x3 | 881.260 / 878.887 / 881.507 | all captures 878.8-882.6 |
| **shipped (A+B+C), 5 rep x4** | **874.089 / 873.138 / 872.996 / 874.289** | all captures 871.7-877.5 |

Edit A alone: **1011.8 +- 1.5 us (-14.6%)**. A+B: **880 +- 2 us (-25.7%)**.
A+B+C: **873.6 +- 1.0 us (-26.3%)**; twenty captures of A+B+C (871.7-877.5) against ten of A+B
(878.8-882.6) do not touch, so the last 7 us is real. The baseline and shipped bands never touch
(noise is +-2 to 4 us). `verify.py --runs 3` on the shipped NEFF: PASS, **all 3 runs
bit-identical**, `max_abs_err` **identical to the baseline to every printed digit**
(output_0 7.688e-05 = 11% of limit, output_1 5.842e-04 = 82%). `launch_rules.py` PASS.
`semaphores.py`: `max_wait <= total_inc` on every semaphore.
Engine busy: **SP 480 -> 34 us (-93%)**, ACT 414 -> 347, DMA 308 -> 207, Pool 187 -> 167,
PE and DVE unchanged. **DVE (439 us, 50% of the span) is now the busiest engine by far.**
5 of 34 members changed: `sg00/def.json`, `SP0.json`, `SP0.bin`, `Pool0.bin`, `Activation0.bin`.

---

---

# EDIT B — the 320 SP DGE `DMA_TRANSPOSE` slots became 161 multi-tile queued transposes

This is where the second -130 us came from, and it is the opposite of what the engine-busy table
suggested. Three measurements, in the order I made them:

| variant | SP busy | DMA busy | total_exec_time |
|---|---|---|---|
| edit A only: 320 DGE transposes on SP | 421.5 us | 273 us | **1011.8** |
| all 320 converted 1:1 to queued transfers | **41.9 (-90%)** | 229 us | **1018.0 (+6)** |
| the same 320 tiles carried by **161** transfers (pairs merged) | 34.4 | 215 us | **881.3 (-131)** |
| by **131** transfers (longest (src,dst) progressions, up to 4 tiles) | 33.2 | 212 us | 882.0 (= same) |

**Read that table before planning anything on this kernel.** (i) Emptying SP of 380 us of issue
time is worth **nothing** — occupancy is not the limit. (ii) Halving the number of DMA
*transfers* that carry the same bytes is worth **131 us**. (iii) Halving it again is worth
**nothing** — the lever saturates between 10 and 20 transfers per ring. I cannot separate the
mechanism of (ii) from ring-depth head-of-line blocking (20 -> 10 per ring) versus per-transfer
queue-start cost (~730 ns x 160 = 117 us, which is the right order of magnitude). Either way the
knob is **transfers per ring**, not bytes, not slots, not engine busy time.

## The structure
All 320 DGE transposes are the same shape: a **16x16 bf16 tile**, `src_elem_size 32`,
`num_elem [1,1]`, `dst_elem_size 32`, with a `PSEUDO_EXTENSION` slot carrying
`tile_src_row_step 262144, tile_src_rows 16, tile_src_cols 16`
(`flags_indirect_mode SRC_INDIRECTION` is a don't-care for this op). They come in 24 windows of
10 program-order-adjacent slots (plus 7x6, 7x4, 2x3, 2x2); inside a window the src offsets are 10
of the 16 slots of a uniform 32-byte grid and the dst offsets a uniform 256-byte grid.

## The queued form (device-verified bit-exact, for 1, 2 and 4 tiles per descriptor)
Derived from this NEFF's own 224 `qSPSpillReload0` transpose descriptors; the convention
reproduces all three pre-existing shapes exactly:
```json
{"transpose_shape":[1,1,rows,cols],"transpose_op":"WZXY","transpose_element_size":2,
 "num_tiling_dimensions":2,
 "from":"SB","from_dtype":"bfloat16","from_off":src0,
 "from_sizes":[cols*2, rows, ntiles, 1],"from_steps":[1, 262144, src_tile_stride, rows*262144],
 "to":"SB","to_dtype":"bfloat16","to_off":dst0,
 "to_sizes":[rows*2, cols, ntiles, 1],"to_steps":[1, 262144, dst_tile_stride, cols*262144]}
```
* `sizes[0]` is the contiguous byte run on each side (`cols*2` in, `rows*2` out), `steps[1]` is the
  partition stride, and **`sizes[2]`/`steps[2]` are a per-tile dimension**: `ntiles` independent
  16x16 transposes at arbitrary byte strides. That is the whole trick. `steps[3]` keeps the
  single-tile value (`rows*262144` / `cols*262144`) with `sizes[3] = 1`.
* Tile strides seen: src +32..+160, dst +256..+1024, and negative strides work too.
* When replacing a DGE transpose you must **delete its `PSEUDO_EXTENSION` slot as well** — an
  orphan extension after an `EVENT_SEMAPHORE` is misparsed.

## The rewiring (per merged group, `scratch/tqm.py` / `tqm2.py`)
Emitted at the program-order-**last** member's slot (all members' waits are provably satisfied
there, so no cross-engine cycle):
1. if two or more members have real waits, one `EVENT_SEMAPHORE` carrying two of them
   (primary + extended) — this is how a trigger gets a *conjunction* of waits, since
   `PSEUDO_DMATRIGGER` has only one wait field;
2. the `PSEUDO_DMATRIGGER` for the merged transfer (queue `qSPSpillReload2`, sems 88..103 —
   7 queues x 16 = 112 rings is accepted by the runtime);
3. for **each** member tile, a guard `EVENT_SEMAPHORE` (`S[own_sem] >= 16*(ordinal-1)`) and a
   relay `EVENT_SEMAPHORE` (primary `SEM_ADD_IMM_COMPLETE` on its own DGE semaphore, value 16;
   extended wait = the merged transfer's completion). Every original consumer wait is untouched:
   one transfer can repay several DGE semaphores, which is what makes merging possible at all.

# EDIT C — the 320 transpose relays moved from SP to Pool: another -7 us (881.5 -> 873.1)

Edit B leaves SP running an **in-order chain of 448 relay `EVENT_SEMAPHORE`s, each blocking until
its own queued transfer completes**, and the profile showed PE losing **101 us** waiting not for
DMA but for *SP to arrive* at a relay slot. Three measurements say the chain, not the DMA, is what
costs:

| variant | transfers | rings | total_exec_time |
|---|---|---|---|
| 320 transfers, 16 rings (1:1 conversion) | 320 | 16 (20/ring) | 1018.0 |
| **320 transfers, 32 rings** (2 queues, same transfers) | 320 | 32 (10/ring) | **1022.4 — no change** |
| 161 transfers (pairs merged), 16 rings | 161 | 16 (10/ring) | 881.3 |
| 131 transfers, 16 rings | 131 | 16 (8/ring) | 882.0 — no change |
| **161 transfers, the 320 relays hosted on Pool** | 161 | 16 | **873.1** |

**Ring depth is not the knob** (320 transfers spread over 32 rings is exactly as slow as over 16),
so the 131 us that merging bought is not queue parallelism. What merging halved is the number of
**distinct transfer completions that an in-order relay chain has to stall on** — the two relays of
a merged pair wait on the *same* completion, so the second never stalls. Moving those stalls onto
an engine nobody waits for (Pool: 19% busy, 655 us idle) is worth a further 7 us.

**How**: emit each tile's guard+relay pair into `Pool0.bin` instead of `SP0.bin`, inserted after
the last Pool slot whose profile `start_ts` precedes the SP slot the group is emitted at (the same
placement rule as the triggers). Cross-engine ordering is safe **because of the guards**: each
relay waits `S[own_sem] >= 16*(ordinal-1)` before adding its own 16, so the increments on
S[56..71] stay in the original DGE order even though they now come from two engines
(`semaphores.py`: `S[56..71]` sources become `Pool:EVENT_SEMAPHORE 320 + SP:EVENT_SEMAPHORE 128`,
totals unchanged). SP drops to 1329 slots, Pool grows 491 -> 1121. `verify.py --runs 3`:
bit-identical, errors identical to the baseline. Script: `scratch/tqm_pool.py --relay-host Pool`.

## Where to host the relay chain: a four-point curve (all verified, all bit-exact)

| relay host for the 320 transpose relays / the 128 staging relays | SP slots | Pool slots | total_exec_time |
|---|---|---|---|
| SP / SP (edit B as built) | 1959 | 491 | 881.5 |
| **Pool / SP  <-- shipped** | **1329** | **1121** | **873.0** |
| Pool+SP alternating / SP | 1643 | 807 | 877.6 |
| Pool / Pool (all 448 on Pool, one monotone pass) | 1073 | 1377 | 920.9 |

So the chain has an **interior optimum**: offloading all 320 transpose relays to Pool is best, and
pushing the staging relays over too costs **48 us** — Pool's in-order stream then has to carry
every stall *and* its own 224 DGE input loads, which the relays now sit in front of. An even split
is also worse than the shipped asymmetric one. Tuning further is not worth it (the whole span of
the curve is 48 us and three of the four points are within 5 us of each other), but note that
`scratch/move_all.py`'s **monotone single-pass placement** is the safe way to move relays: assert
the host position is non-decreasing in the source stream's order and §7's hang cannot happen.

# WHERE THE TIME GOES NOW (measured on the A+B NEFF, `prof3/`, 879 us span; edit C then
# moved the relay chain to Pool and took another 7 us off, but the picture below still holds)

Engine busy: **DVE 439 (50%), ACT 347 (39%), PE 263 (30%), DMA 215, Pool 167, SP 34 (4%)**.
Idle per engine, attributed to the semaphore the next slot waits on (`scratch/peidle.py`):

| engine | idle | biggest buckets |
|---|---|---|
| PE | 605 us | **S[6] DVE 250.3**, qSPDynamicHW relays 188.3, qSPSpillReload0 89.5, S[4] ACT 76.8 |
| ACT | 519 us | **S[5] PE 262.1**, S[6] DVE 156.9, qSPSpillReload0 67.7 |
| DVE | 438 us | **S[5] PE 310.9**, qSPDynamicHW 45.0, S[3] Pool 37.5, S[4] 23.7 |
| Pool | 655 us | S[4] ACT 317.3, S[6] DVE 176.5, qPoolDynamic 149.0 |
| SP | 464 us | **ACT dead-save relays S[40..55] 242.5** (the 160 staging triggers), S[5] 113.5 |

The loop is now **PE <-> DVE <-> ACT** (PE waits 250 us on DVE, DVE waits 311 us on PE, ACT waits
262 us on PE) plus two DMA items on PE's path:
* **188.3 us** on the transpose relays (n=71), which splits (`scratch/pesplit.py`) into
  **101.0 us waiting for SP to arrive at the relay slot**, 82.5 us the relay waiting for its own
  transfer, 4.8 us after. SP is blocked 242 us of that time at the 160 staging triggers, waiting
  for ACT — so SP's *order* (not its occupancy) is what delays those relays. Note the identical
  99.6 us figure in the pre-B profile converted to **0 us of wall** when I removed it (§2 below),
  so a parallel path is likely; re-measure before spending an iteration on it.
* **89.5 us** on `qSPSpillReload0` (n=65), the kernel's original 222 queued transfers
  (224 descriptors, all SB->SB `WZXY` transposes, 14 per ring).

(The pre-B profile, for comparison: PE 732 us idle with 283.9 on S[56..71] of which 154.0 was the
320 DGE transposes' DMA latency — that is what edit B removed.)

# REFUTED THIS ITERATION (each with the number that closed it)

### 1. Trigger placement: early beats late, and the gating is already optimal
* **All 160 triggers moved late** (immediately before their own relay, which is safe by
  construction because the old DGE reload waited for the save's *completion* there):
  **1029.1 us vs 1013.1** -> **+16 us, worse**. Keep triggers at "the last SP slot that started
  before the save did".
* **Re-placing triggers from the post-edit profile**: the rule returns the *same* position for 123
  of 160 and up to 4 slots *earlier* for 37 — nothing to gain. (The measurement is
  self-referential: SP is blocked **at** the trigger, so no later SP slot starts before the save.)
* **Gating the trigger on the producer instead of on "ACT reached the save"** (s128 iter-1's
  variant): measured the head start in the profile — the producer of each save's own wait ends a
  **median 0.45 us *after* ACT reaches the save ES** (87 of 98 negative, total -42 us). There is
  **no head start to win**; ACT is not lagging behind its own data. Lever closed without a device run.

### 2. Taking PE's wait off the relay and onto the transfer itself — 0 us
Only `EVENT_SEMAPHORE` has an extended events block (checked every struct: ACT/DVE/PE/Pool
`CAST`/`COPY`/`MATMUL`/`LDWEIGHTS`/... do **not**), so a consumer cannot carry two waits. I
inserted an ES in front of each of the 32 PE `LDWEIGHTS` that wait on a relay, carrying
*both* `S[s] >= 16*(ordinal-1)` (keeps the original ordering) **and**
`S[72+ring] >= 16*k` (the exact transfer), and cleared the consumer's own wait — this deletes the
"wait for SP to arrive at the relay" term worth 99.6 us of PE idle.
**1012.2 us vs 1013.1 — zero.** PE ES does not increment S[5] so inserting PE slots is free and
legal (`semaphores.py` unchanged, verify bit-identical), but the 99.6 us has a parallel path.
Reverted. (Direct retargeting *without* the inserted ES is only provably safe for 49 of 191
candidate waits — the test is "an earlier wait on the same engine already covered ordinal j-1";
`scratch/retarget_apply.py` implements it.)

### 3. **SP issue time is not on the critical path at all** — the big one
I converted **all 320 SP DGE `DMA_TRANSPOSE` slots into queued transfers** on a 6th queue
`qSPSpillReload2` (sems 88..103, 96 rings total — accepted), each becoming
trigger + guard + relay. **The mapping is bit-exact** (see below), `verify.py` PASS with
`max_abs_err` identical to the baseline, and:

| | SP busy | DMA busy | total_exec_time |
|---|---|---|---|
| shipped (DGE transposes) | 421.5 us | 273 us | **1013.1 us** |
| all 320 transposes queued | **41.9 us (-91%)** | 229 us | **1018.0 us (+5)** |

**Removing 380 us of SP issue time made the kernel 5 us slower.** So after the staging fusion this
kernel has *no* engine-occupancy lever left: SP can be emptied completely and the wall does not
move. (s128's "issue time converts at ~40%" was true while ACT was 46% busy and on the path; it
does not transfer here.) The +5 us is the 448 relays in SP's in-order stream each blocking ~1.2 us
for their own transfer.

### 4. Queued transpose descriptors CAN express a DGE `DMA_TRANSPOSE` — verified, keep this
All 320 are the same shape: a **16x16 bf16 tile**, `src_elem_size 32`, `num_elem [1,1]`,
extension `tile_src_row_step 262144, tile_src_rows 16, tile_src_cols 16`,
`flags_indirect_mode SRC_INDIRECTION` (a don't-care for this op). The queued form, derived from
this NEFF's own 224 `qSPSpillReload0` transposes and **device-verified bit-exact** (16 instances,
then all 320):
```json
{"transpose_shape":[1,1,rows,cols],"transpose_op":"WZXY","transpose_element_size":2,
 "num_tiling_dimensions":2,
 "from":"SB","from_dtype":"bfloat16","from_off":src,
 "from_sizes":[cols*2, rows, 1, 1],"from_steps":[1, 262144, cols*2, rows*262144],
 "to":"SB","to_dtype":"bfloat16","to_off":dst,
 "to_sizes":[rows*2, cols, 1, 1],"to_steps":[1, 262144, cols*262144, rows*2]}
```
(The convention `steps = [1, partition, run_bytes, rows*partition]` on the from side and
`[1, partition, out_rows*partition, out_run]` on the to side reproduces all three existing
transpose shapes in this NEFF exactly.) When replacing a DGE transpose, **delete its
`PSEUDO_EXTENSION` slot too** — an orphan extension after an ES is misparsed.
`scratch/tq.py` does the whole conversion.

### 5. **Merging `qSPSpillReload0`'s 222 transfers the same way HANGS unless the waits are checked**
The obvious follow-up to edit B (PE idles 89.5 us on this queue, 14 transfers per ring) is to merge
its transfers in pairs too. 222 transfers carry only **7 distinct descriptor keys**, and with a
window of 8 triggers **108 mergeable (identical key + uniform `from_off`/`to_off` delta) pairs**
exist (222 -> 114 transfers). Built it (new queue `qSPSpillReload3`, sems 104..119, each removed
transfer repaid by a relay `EVENT_SEMAPHORE` on `S[24+id%16]` with a `>= 16*(k-1)` guard, merged
transfer emitted at the group's last trigger position with the conjunction of the members' waits):
**NRT_TIMEOUT / hang** (`verify.py` `[FAIL] run 0 ... exit 1, 70 s`), `semaphores.py` clean.
* **Cause**: merging *delays* the earlier member's transfer to the later member's trigger, and the
  later member's own wait can transitively depend on the earlier member's data — e.g. a wait
  `S[5] >= v` (PE) that PE only reaches after consuming the first transfer. The s128 rule ("a
  retargeted wait must name a producer already satisfied before the instruction ran") does not
  cover this case: nothing is retargeted, a *producer* is delayed.
* **The sound condition**: for every member, the wait must
  have been satisfied **before the group's *first* trigger started** in the baseline profile — then
  no member's condition can depend on any transfer in the group. The wait classes to resolve are
  `S[3..6]` 183 of 222 (v-th signalling instruction of that engine, `end_ts`), `qPoolDynamic` 34,
  `qSPSpillReload0` 5 (transfer `(v//16-1)*16 + (s-24)`).
* **CLOSED, with the number**: I then built the satisfaction model (`scratch/satcheck.py`: for a
  wait `(sem,val)`, the producing event is the `val//16`-th DGE/relay on that DMA semaphore, or the
  `val`-th signalling instruction of the engine for `S[3..6]`; resolves 217 of 222) and measured
  what merging would actually cost. Every q0 trigger starts a **median 79 ns** after its own wait
  is satisfied — the compiler already fires each transfer the moment its data exists. So for the
  108 candidate pairs, merging would **delay the earlier transfer by a median 22.9 us**
  (p10 2.3, p90 27.6, sum 1535 us; only 1 of 105 is <= 0), because the second member's data is not
  ready until much later. **The lever is closed by data dependency, not by tooling** — and that is
  also the precise reason the hang happened.
* **Why edit B is different, and the general rule**: merging transfers only pays when the members'
  data becomes ready **at the same time**. The merged 16x16 transpose tiles are produced by the same
  instructions (adjacent DGE slots, same window), so the merged group's wait equals each member's
  wait and nothing is delayed. Before merging any transfers, measure
  `sat(member_i) - start(first trigger)`; if it is positive and large, do not merge.
* The cheap sound shortcut — merge only members with **identical** waits — yields **0 pairs**.
  Every one of the 222 triggers has a distinct wait, so the timing check is unavoidable.
* Note that the same hazard did **not** bite edit B: a DGE transpose's wait is on the SBUF producer
  of its tile and the merged tiles are written by the same instructions, so delaying the group to
  its last member is safe there (and 3 runs are bit-identical).

### 6. Late staging triggers are worse in the post-B configuration too — +20 us
Re-ran refutation 1 on top of edit B (where SP is 96% idle, so the trade-off is different): all 160
staging triggers moved to just before their own relay, to stop them blocking SP ahead of the
transpose relays PE waits 101 us for. **898.4 us vs 878.9** — worse by 20 us. The rule holds in
both configurations: **trigger a queued transfer as early as its data allows, and accept the SP
head-of-line blocking.**


### 7. Moving the **128 staging relays** to Pool as well — HANG (ordinal order vs stream order)
The obvious continuation of edit C (it would leave SP with nothing but triggers). I moved the 128
staging relay+guard pairs out of `SP0.bin` into `Pool0.bin`, placed by the same timestamp rule but
in a **second** pass, on top of edit C's 320 transpose relays: **NRT_TIMEOUT / hang**,
`semaphores.py` clean.
* **Cause, and the rule that matters**: a guarded relay waits `S[s] >= 16*(ordinal-1)` before
  adding its 16, so **the host engine's program order must agree with the semaphore's ordinal
  order**. Two independent placement passes into the same stream do not guarantee that: as soon as
  Pool's order puts ordinal 5 before ordinal 3 on some `S[56..71]`, ordinal 5's guard waits for
  increments that can only be produced later in the same in-order stream — a self-deadlock that
  `semaphores.py` cannot see (it only compares totals).
* To retry: place **all** relays for a given semaphore in one pass, assert
  `host_position(ordinal k) < host_position(ordinal k+1)` for every semaphore, and only then
  choose positions by time. Expected payoff is small (edit C's whole move was worth 7 us), so do
  it only as cleanup.

# WHAT TO TRY NEXT, in order (ranked by this iteration's measurements, not by analogy)

**The four rules this iteration established for this kernel** (each with the number behind it):
* **Engine issue time / occupancy is not a lever.** SP went 421 -> 34 us of busy time for **+6 us**
  of wall (§3). Do not rank candidates by the engine-busy table.
* **Removing a dependency edge that has a parallel path converts at 0.** -99.6 us of PE idle bought
  **0 us** (§2).
* **Reducing the number of distinct DMA transfers that carry the same bytes converts enormously**
  (320 -> 161 transfers = **-131 us**, edit B) **but saturates** (161 -> 131 = 0). It is *not* ring
  parallelism: 320 transfers spread over 32 rings instead of 16 is **exactly as slow** (edit C
  table). The quantity that matters is how many distinct completions an **in-order relay chain**
  must stall on, which is also why hosting that chain on an idle engine is worth another 7 us.
* **Merging transfers only pays when the members' data is ready at the same time.** The
  `qSPSpillReload0` candidates would delay the earlier transfer by a median 22.9 us and one
  arrangement **hangs** (§5).

1. **Delete the relay chain instead of relocating it** — the one lever the measurements still
   point at. Edit C relocated it (-7 us) and §7 shows a second relocation pass hangs. The clean
   version: retarget every consumer wait on `S[56..71]` onto the queue semaphore of the transfer
   that now carries its data, then drop all 448 relays and their guards, so no engine ever stalls
   in order on a transfer completion. Two known obstacles: (i) a wait `S[s] >= 16j` means
   "producers 1..j", so each retargeted consumer needs a conjunction (the j-th transfer's
   completion **and** `S[s] >= 16*(j-1)`), and only `EVENT_SEMAPHORE` has an extended events block,
   so it needs ~400 inserted two-wait ES slots across PE/DVE/ACT/SP (`scratch/pe_ext.py` does
   exactly this for 32 of them); (ii) §2 measured the 32-consumer version at **0 us**, so the
   ceiling is unproven — do a cheap subset and measure before doing all 400.
2. **DVE is the top engine: 439 us busy = 50% of the 873 us span** (928 `TENSOR_TENSOR` 188 us,
   518 `COPY` 130 us, 544 `CAST` 114 us, 163 `MEMSET` 23 us, ~200 ns/slot), PE waits 250 us on DVE
   and DVE waits 311 us on PE, so **PE<->DVE is now the kernel**. There is no per-slot fat; only
   whole operations can go. The one concrete thread (inherited from the s128 sibling, still open):
   the DVE `COPY`s that evacuate PSUM (`src_mem_pattern_start_addr_addr_immediate >= 0x2000000`)
   would die if a queued descriptor could read PSUM, but no PSUM variable type exists in the
   corpus — `from:"SB", from_off >= 0x2000000` is untested, and deleting the copies would need the
   ~2250 waits on S[6] renumbered. Low probability, high payoff.
3. ~~Finish edit C's relocation (move the 128 staging relays off SP too)~~ — **done and refuted**:
   with the monotone single-pass placement that avoids §7's hang it measures **920.9 us, +48**.
   See the four-point host curve in the edit C section. PE does still lose ~62 us waiting for SP to
   reach a *staging* relay (`scratch/pesplit.py` on the shipped NEFF), so the stall is real — but
   moving it to Pool costs more than it saves, which leaves only lever 1 (delete the chain).
4. **Do not spend budget on**: everything in §1-§7 below (trigger placement early/late,
   producer-gated triggers, consumer-wait retargeting of a 32-wait subset, SP occupancy, 1:1
   DGE->queued conversion without merging, merging past ~10 transfers per ring, splitting a queue
   for ring depth, merging `qSPSpillReload0`); and everything the s128 sibling closed (instruction
   merging, `LDWEIGHTS` dedup, shrinking dead DGE transfers, ring depth/throttle as a JSON knob,
   DGE-side staging fusion, engine rebalancing of compute, queued DMA reading PSUM).

# Device-cycle budget actually used
~25 device runs in 2 h: 2 baseline profiles, 11 verifies (3 of them `--runs 3`, 2 of them hangs
at 70 s each) and 12 profiles. A `verify.py` run is ~55 s, `--runs 3` ~2 min,
`profile.py --repeats 5` ~80 s, `--parquet` adds ~20 s. Spread on this kernel is 0.18-0.43%
(+-2 to 4 us), so **nothing under ~7 us is claimable**.

# Reusable scripts (in `scratch/`, reverted between iterations — rewrite from these notes)
`compose.py` (two-hop -> one queued descriptor, with the dim-splitting and the exact verification),
`build.py` (the whole shipped edit, `--limit/--group/--tpos` for probes), `shapes.py`,
`peidle.py` (per-engine idle attributed to semaphores), `pesplit.py` (a PE stall split into
producer-wait / transfer / propagation), `tq.py` (DGE transpose -> queued), `tmerge.py`,
`retarget_apply.py`, `pe_ext.py` (inserted two-wait ES in front of a consumer),
`tqm_pool.py --relay-host Pool` (edit C), `tq2q.py` (the 2-queue ring-depth control),
`q0merge2.py` + `satcheck.py` (the refuted `qSPSpillReload0` merge and the wait-satisfaction
model), `move_staging.py` (the hang in §7).

# How the shipped NEFF was produced (exact chain, for reproduction after `scratch/` is reverted)
```bash
python neff_doc/tools/neff.py unpack baseline/kernel.neff work_orig/
python tools/profile.py --repeats 5 --parquet prof      # the baseline profile the placement needs
python scratch/build.py        --out work                # edit A (uses work_orig + prof)
python tools/profile.py --repeats 3 --parquet prof2      # profile of edit A (Pool placement needs it)
python scratch/tqm_pool.py --src work --out work_final --prof prof2 --relay-host Pool   # edits B+C
python neff_doc/tools/semaphores.py work_final sg00      # totals must balance; it cannot see ext waits
python neff_doc/tools/neff.py pack work_final/ kernel.neff
python tools/verify.py --runs 3 && python tools/profile.py --repeats 5
```
Both placement passes read a profile of the NEFF they are editing, so the order above matters:
`build.py` needs a profile of the **baseline** (to place the 160 staging triggers) and
`tqm_pool.py` needs a profile of the **edit-A** NEFF (to place the 320 relay pairs on Pool).
Shipped `kernel.neff` sha256 `d783fd3d9a96fcde...`; 5 of 34 members differ from the baseline
(`sg00/def.json`, `SP0.json`, `SP0.bin`, `Pool0.bin`, `Activation0.bin`).
Final state of the three new queues in `def.json`:
`qSPSpillReload1` (data, 16, sp, sems 72..87, 160 staging transfers) and
`qSPSpillReload2` (data, 16, sp, sems 88..103, 161 merged transpose transfers);
instruction counts ACT 1823 (unchanged), Pool 491 -> 1121, SP 1366 -> 1329.

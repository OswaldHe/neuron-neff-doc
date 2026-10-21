# iter-2 (ssd_mamba2_1b3_s2048): the DGE move is a dead end, and ACT busy time no longer buys wall time

Start of iteration: iter-1's NEFF. My own 5-repeat control of it: **5128.300 us** (min 5127.6, max
5129.4, spread 0.04%). ACT 3769 us busy (73.5%), DVE 2701, Pool 2072, PE 2013, DMA 1843, SP 804.

Shipped: 303 pairs of adjacent `ACTIVATE` merged into one (ACT 14377 -> 14074 slots, ACT busy
3769 -> 3672 us, **-97 us**). `verify.py` VERIFIED with `max_abs_err` 5.234e-07 / 3.334e-07,
identical to iter-1. Only `sg00/Activation0.bin` changed.

Two back-to-back 5-repeat pairs (control first, then merge, same session):

| pair | control | merge | delta |
|---|---|---|---|
| 1 | 5128.300 | 5126.696 | -1.6 |
| 2 | 5130.074 | 5126.607 | -3.5 |

So the sign is consistent but the size (~-2.5 us, -0.05%) is at the noise floor (spread
0.04-0.11%, i.e. 2-6 us). **Do not record this as a win.** I shipped it because it is strictly
less work for identical output and it banks 97 us of ACT time that will convert if a future
iteration lifts the co-constraint described next.

**THE HEADLINE RESULT -- read this before planning iter-3.**
Three independent, verified edits this iteration moved ACT busy time and none of them moved the
wall clock:

| edit | ACT busy | SP busy | wall (5-repeat median) |
|---|---|---|---|
| control (iter-1 NEFF) | 3769.2 | 804.2 | **5128.300** / 5130.074 (two runs) |
| 208 DGE triggers moved ACT -> SP | 3712.5 (**-57**) | 928.6 (+124) | 5133.522 (**+5.2**) |
| 280 provably-satisfied `EVENT_SEMAPHORE` waits deleted (ACT/DVE/Pool/PE) | 3770.5 (0) | 806.7 | 5130.119 (+1.8) |
| 303 adjacent `ACTIVATE` pairs merged | 3672.2 (**-97**) | 802.6 | 5126.696 (-1.6) |

**-97 us of ACT busy bought -1.6 us of wall: a sensitivity of ~1.6%.** iter-1's table-load removal
had a sensitivity of 81% (-2516 us ACT -> -2026 us wall). So the kernel crossed over during
iter-1: ACT is no longer the throughput constraint, and shaving ACT by tens or even a couple of
hundred microseconds is now worth nothing. **Do not spend another iteration on ACT instruction
count.** The review's ranked list (and iter-1's) is built on the pre-crossover assumption and is
now wrong at the top.

Where the wall actually is (backward blame walk over `prof/Flow.parquet` + measured timestamps,
`scripts/cp2.py`): the critical path is 13671 nodes, span 5143 us, of which **4544 us is
instruction duration and 1714 us is gap**. The path is essentially ACT's program order
(10641 `ACTIVATE` = 3723 us busy + **1402 us of gap**, 971 DGE triggers = 540 us) with short
excursions onto Sync/Vector/Tensor. Those 1402 us of ACT gap are the sem-6 (DVE) and sem-5 (PE)
handoffs iter-1 measured (866 + 411 us): ~5700 producer-consumer handoffs at ~300 ns each on a
per-chunk ACT<->DVE<->PE ping-pong. DVE is 53% busy and PE 39%, so neither is throughput-bound --
this is pure latency. Engine overlap confirms it: on a 50 ns grid, 4497 us of the 5147 us timeline
has >= 2 engines busy and **no engine is ever solo for more than 287 us**.

So the only levers left are ones that *remove handoffs* (fewer, larger steps per chunk) or remove
work on the scale of a millisecond. Shaving an engine is finished.

## 1. The DGE trigger move (iter-1's open item 1): closed, twice over

Target: 1088 `PSEUDO_DMA_DIRECT2D` on ACT, 558 ns each, 607 us. It fails for two separate reasons.

### 1a. `def.json` -> `dma_queue` is why it hangs. Nobody had read it.
```
qActDynamicHW  type dynamic  num_queues 16  owner act   semaphores 40..55
qPoolDynamic   type dynamic  num_queues 16  owner pool  semaphores  8..23  (+ SB_scratch 0/16384)
qSPDynamicHW   type dynamic  num_queues 16  owner sp    semaphores 56..71
qSPIO0         type in       num_queues 16  owner sp    semaphores 24..39
```
Each engine owns a 16-queue dynamic-DGE set and **the queue is the semaphore**: queue *i* of a set
signals `base+i`. Moving an ACT trigger to SP while keeping `semaphore=40..55` makes SP push
descriptors into a ring owned by ACT, which ACT is still pushing into -- two producers with
independent head pointers on one ring. Which explains the pass/fail pattern exactly: a moved
*prefix* of each ring is safe (SP writes ring slots 0..C-1 first, ACT then restarts from 0 on
already-consumed entries), a moved *suffix* is safe (ACT is long past), and a middle block is safe
only if SP's low ring slots no longer hold ACT descriptors in flight.

### 1b. It is not an ordering problem. I proved a placement rule and it hangs identically.
`scripts/move.py` (not kept -- rebuild from this). For trigger *k* at ACT slot T: `v` = ACT's
cumulative sem-4 increments strictly before T; `W` = first SP slot that waits for this transfer;
`A` = ACT's own waiter slot for it.
* **ACT side: just delete the slot.** No placeholder is needed, because ACT *already* waits on
  sems 40..55 1076 times (always >= 11 slots after its own trigger, median 214) -- that is the WAR
  guard on the spilled SBUF region. And the triggers carry `events_update NONE`, so deleting them
  leaves every sem-4 cumulative count byte-identical.
* **SP side:** insert the same 64 bytes with `wait sem4 >= v`, `update NONE`, at
  `P = min( first SP slot whose profile start_ts > start_ts(ACT slot T), suffix-min over k'>=k of W(k') )`.
  - the suffix-min makes P monotone in k *and* keeps `P <= W(k)`: SP must trigger a transfer
    before its own wait for it, or it self-deadlocks. This is the one real trap.
  - the timestamp term keeps SP at a position its original schedule reached before ACT reached T.
* **Why there is no new cycle:** at SP position `W(k')` the original program already blocked until
  ACT had reached `v(k')`, and `v(k) <= v(k')` for `k <= k'` since `v` is cumulative -- the new
  wait is *weaker than the wait that position already carried*. Two extra filters (max sem-4 wait
  value among SP slots before P must be <= ACT's count at A; no SP wait before P may depend on a
  higher-indexed transfer) reported **0 violations for all 1088 triggers**. `semaphores.py` clean.

Device results (`VERIFIED` means outputs byte-identical to iter-1, so when the move runs it is
bit-exact). `k` maps to `(sem = 40 + k%16, ordinal c = k//16 + 1)`.

| moved to SP | my rule | iter-1's rule |
|---|---|---|
| `0:1088` all | hang | hang |
| `0:544` | hang | - |
| `128:544` | hang | - |
| `608:1024` | hang | - |
| `0:96` (ordinals 1..6) | pass | pass |
| `0:112` (ordinals 1..7) | **pass** | - |
| `0:120` | **hang** | - |
| `0:128` | - | hang |
| `96:128` | hang | hang |
| `480:544` | **hang** | - |
| `544:608` (ordinals 35..38) | pass | pass |
| `608:672` | **hang** | - |
| `1024:1056` | - | hang |
| `1056:1088` (ordinals 67,68) | pass | pass |
| union `0:112,544:608,1056:1088` (208) | **pass** | - |

### 1c. And even when it works it is worthless
The 208-trigger union verified and profiled at **5133.5 us vs the 5128.3 control**: ACT busy -57,
SP busy +124, wall **+5**. SP is not a free parking space -- SP is the engine that waits on
sems 40..55 and then triggers the spill *reloads* that PE waits on (PE has ~1020 waits on 56..71),
so SP sits on the critical path and the added 558 ns per trigger comes straight back. Pool is worse
(ACT waits on Pool's 8..23 about 1900 times). **There is no idle engine to move DMA triggers to.**

If someone still wants the 607 us: the only clean form is to retarget the moved trigger's
`semaphore` from `40+i` to `56+i` so it uses SP's *own* ring, which means recomputing
`events_semaphore_value` for every waiter on 40..55 (ACT 1076, SP 1030, DVE 17, Pool 5) and on
56..71 (PE ~1020, SP ~1072, Pool ~60) from the new per-ring FIFO order -- ~4300 slots in 5 members,
and an SP transfer's waiter then also waits for ACT transfers queued ahead of it, which is a
*stronger* wait and can deadlock. Given 1c, the payoff is probably zero. I would not do it.

## 2. What I shipped: merging adjacent `ACTIVATE` pairs

iter-1 recorded "only 319 adjacent pairs are mergeable, ceiling ~30 us, closed". The count is
right (I reproduce 319) but the ceiling was wrong and the edit is trivial, so I did it.

Merge rule, over `sg00/Activation0.bin` only: for adjacent slots `i, i+1` both `ACTIVATE`, with
bytes `0:4`, `12:16`, `32:48` identical (so same opcode, `imm_src`/`scale_src`/`bias_src`,
`accumulator_cmd`, dtypes, `num_active_channels`, `activation_func`, **and the same scale/bias/imm
pointer**), identical `step_elem`/`num_elem` patterns with `step_elem[0] == 1` and dims 1,2 unused,
`src2 == src1 + n*4` and `dst2 == dst1 + n*4`, the same wait semaphore index, and not both
updating: emit one instruction at slot `i` with `num_elem[0]` doubled, `events_semaphore_value` =
max of the two values, and the update of whichever had one; delete slot `i+1`.
`both update` was 0 for all 319 pairs, so no sem-4 count changes -- verified 7360 before and after.
303 of 319 pairs have the same wait index and were taken; the other 16 would need an extra
`EVENT_SEMAPHORE` for the second wait.

Cost model for `f=128 nch=128`, from the measured data (two exact points):
`nelem=1 -> 162.3 ns`, `nelem=64 -> 434.3 ns`, i.e. **158.0 + 4.317 * nelem**. Merging 64+64 should
save `868 - 710 = 158 ns`, so 48 us predicted. Measured ACT busy fell **97 us** (320 ns per merge),
so a merged instruction also saves a per-instruction issue overhead the duration model does not
show. The wall did not follow (see headline).

## 3. ACT inventory (measured; use this instead of re-deriving it)

Per-opcode on ACT, from `prof/Instruction.parquet` with `duration_ns` summed per `compiler_pc`,
and the share-of-wall column from the issue-interval decomposition (`st[i+1]-st[i]`, which sums
exactly to ACT's 5067 us span):

| class | n | duration us | share of wall us |
|---|---|---|---|
| `ACTIVATE` f=128 nch=128 nelem=64, scale POINTER, src!=dst | 3072 | 1329 | 1482 |
| `ACTIVATE` f=128 nch=128 nelem=64, scale POINTER, **src==dst** (in-place `x = s*x`) | 3072 | 1339 | 1177 |
| `PSEUDO_DMA_DIRECT2D` (spill saves) | 1088 | 607 | 537 |
| `ACTIVATE` f=128 nch=1 nelem=128, scale POINTER | 1024 | 500 | 548 |
| `EVENT_SEMAPHORE` (standalone waits) | 1922 | 78 | 447 |
| `ACTIVATE` f=7 nch=1 nelem=1 | 1024 | 207 | 423 |
| `ACTIVATE` f=7 nch=128 nelem=1 (iter-1's exp pair) | 2044 | 397 | 248 |
| `ACTIVATE` f=128 nch=128 nelem=1, scale imm 1.0 (iter-1's PSUM->SBUF `prod`) | 1020 | 165 | 176 |
| `COPY`, `ACT_TABLE_LOAD`, f=129 | 110 | 30 | 28 |

Caveat: the `EVENT_SEMAPHORE` share is an artefact -- their own duration is 40 ns and the rest is
the *next* instruction's stall landing in their interval. Deleting 206 of them (section 5) changed
ACT busy by 0.0 us, which proves it.

## 4. Closed, with the measurement that closed it
* **DGE 2-D merge** (two 32 KB transfers in one instruction): dead. Trigger cost is linear in the
  **descriptor count** (`src_num_elem[0]`), not bytes. Pool, same `(128,1)` shape:
  `elem_size 4 -> 651.7 ns`, `256 -> 657.1 ns`, `512 -> 649.8 ns`. 128 descriptors is the floor for
  a 128-partition gather, so 2 transfers = 256 descriptors = the same total time.
* **Hoisting a DGE trigger earlier inside ACT to hide it in a stall**: dead. Total ACT idle
  immediately before a trigger is **8.7 us over all 1088** -- they never stall, and the stalls that
  exist are the sem-5/sem-6 waits that establish the trigger's upstream ordering, so it cannot be
  hoisted above them. (No ACT instruction writes a trigger's source within ~870 slots: the spilled
  data is produced by DVE/Pool/PE, and ACT's *position* is the only thing ordering the trigger
  after it.)
* **Folding a standalone `EVENT_SEMAPHORE` wait into the next instruction**: impossible. The events
  block has exactly one wait, **every** ACT instruction already carries a wait, and all 1922
  standalone waits are on a *different* semaphore than the instruction that follows them. That is
  precisely why the compiler emitted them.
* **Deleting provably-satisfied waits**: 206 on ACT, 52 on PE, 11 each on DVE/Pool (a wait is
  redundant if the same stream already passed a `>=` wait on the same semaphore; semaphores are
  monotone here -- the only `EVENT_SEMAPHORE_RANGE_CLEAR` on each engine runs at t=5145 us, after
  everything). Verified, and worth **0.0 us of ACT busy** and +1.8 us of wall. A satisfied wait is
  free. Closed.
* **Redundant `ACTIVATE` recomputation**: 4311 ACTIVATEs repeat an earlier identical instruction
  (same func/addresses/scale pointer). With range-based write tracking across all five engines
  **0 of 4311 survive** -- 4308 have their source rewritten in between and 3 have the destination
  clobbered. There is no redundant arithmetic on ACT. (An exact-address-only check wrongly leaves
  528 survivors; use ranges.)
* **ACT identity instructions**: none. No `ACTIVATE` has `scale==1.0 && src==dst`.
* **ACT performance mode**: `performance_mode` is NaN for every Scalar row; only 972 Vector rows
  carry `2x`. Nothing to flip.
* **`ACTIVATE` dtype**: everything is FP32 in and out, and `max_abs_err` is already 84% of the
  limit, so a precision change is not available.

* **Relaxing over-strict waits using `prof/Flow.parquet`**: do **not** trust it. The obvious lever
  on the 1402 us of ACT gap is to prove that an ACT `wait sem6 >= v` only really needs `>= k` for
  some `k < v` and lower it. Computing the minimal count from Flow's predecessors (all dependency
  types; for a predecessor that does not itself increment, taking the first incrementer at-or-after
  it in the owner's program order) says **2302 of 4349 sem-6 waits and 256 of 1087 sem-5 waits have
  slack, exactly 3 are tight, and 2875 have no predecessor on the owner engine at all**, with an
  "estimated removable stall" of 471 ms -- i.e. 90x the whole runtime. `Flow.parquet` is therefore
  not the synchronisation requirement: it carries only the dependences the profiler could attribute
  directly, while the compiler's waits enforce transitive/aggregate ordering. Any relaxation built
  on it is a silent race that the single-input verifier may well not catch. If this lever is worth
  trying, derive the dataflow from the NEFF's own `debug_info_backend_*.dbg` (ch 11.2,
  `dbginfo.py`), not from the profile.

## 5. Ranked for iter-3 (given the headline)
1. **Attack handoff count, not engine busy time.** 1714 us of the 5143 us critical path is gap, and
   it is ~5700 sem-4/5/6 handoffs at ~300 ns on a per-chunk ACT<->DVE<->PE ping-pong. Anything that
   merges two chained steps into one (so one handoff instead of two) converts directly. The
   `ACTIVATE` merge in section 2 is the shape of this when the two steps are on the same engine;
   the valuable version is when the *consumer* of an ACT result is ACT again two handoffs later.
   Start from `scripts/cp2.py`'s path and look at consecutive path nodes that alternate engines but
   do not actually need to.
2. **The 1024 `f=128 nch=1` ACTIVATEs (548 us of path share, 489 ns to move 512 B on one lane).**
   Cross-partition 128-element moves, e.g. `p24/0x1c300 -> p0/0x9c00`, several in a row writing the
   *same* destination `p0/0x9b00`: a serial "load the next row into scratch" chain. 127 of 128 ACT
   lanes idle. They cannot be merged (serial, same dst) and cannot become a DMA (they apply a
   per-partition scale, and a 1-descriptor DMA round trip is ~1.3 us against 489 ns). The win would
   be restructuring the scratch so one `nch=128` instruction does many rows -- which needs the
   consumers rewritten, and they are the handoff chain of item 1.
3. The remaining 16 of 319 mergeable `ACTIVATE` pairs (different wait semaphores, need one extra
   `EVENT_SEMAPHORE`): ~2.5 us of ACT busy, i.e. nothing. Skip.
4. Do **not** revisit: the DGE trigger move (section 1), DGE 2-D merge, wait folding, wait deletion,
   ACTIVATE redundancy, ACT dtype/perf-mode. All closed above with device measurements.

## 6. Tool notes (additions to iter-1's)
* `verify.py` prints `VERIFIED` on success and nothing at all on a device hang -- grep `^VERIFIED`.
  Pass ~40 s including the build, hang ~71 s, so a window scan is ~1 min/test. Run scans with
  `run_in_background` and poll the output file: **do not pipe through `sed`/`tail`**, they buffer
  and you lose everything (I lost one 4-test batch that way, same trap iter-1 hit).
* Field names, per engine, matter. ACT `ACTIVATE` is `v2.NEURON_ISA_TPB_S3D3_AC_STRUCT`:
  `src_mem_pattern_start_addr_addr_immediate` @16, `src_mem_pattern_step_elem` @20 (3 x int16),
  `src_mem_pattern_num_elem` @26 (3 x uint16), dtypes @32-33, `num_active_channels` @34,
  `activation_func` @35, `scale_value_imm_*` @36, `bias_imm_*` @40, `imm_imm_*` @44,
  `dst_mem_pattern_*` @48/52/58. DVE uses `dst_mem_pattern_start_addr_addr_immediate` too, but a
  naive search for `dst_start_addr_addr_immediate` (the DMA name) finds **only DMA triggers** and
  silently returns an empty write set -- that bug made a redundancy analysis report 4311 false
  positives. Match `^dst.*start_addr_addr_immediate$`.
* `prof/Flow.parquet` is the dataflow graph: `in_id`/`out_id` join to `Instruction.id`,
  `dependency_type` in {`FLOW_DEPENDENCE`, `ANTI_DEPENDENCE`, ...}. 160k edges. Combined with
  per-engine program order it gives a usable backward critical-path walk; candidates must be
  filtered to `start_ts[pred] < start_ts[cur]` (not `end <= start`, because ACT instructions
  overlap) or the walk dies after 11 nodes.
* `prof/SemaphoreUpdate.parquet` has `(timestamp, id, value, update_source)` per increment -- the
  cheap way to get "when did semaphore X reach v".
* `Instruction.parquet` has several rows per `compiler_pc`; sum `duration_ns` grouped by pc. The
  per-opcode totals then come to 4654 us against a reported ACT busy of 3769 us, because ACT
  pipelines and the durations overlap by ~900 us. Use the issue-interval decomposition
  (`st[i+1]-st[i]`) when you want numbers that add up to the wall.
* ACT instruction addresses are `(partition << 20) | byte_offset`; a DMA trigger's
  `src_start_addr_addr_var_offset` is the same `byte_offset` with the partition sweep in
  `src_num_elem[0]`/`src_step_elem[0]` (0x40000). Compare `addr & 0xFFFFF` against the DMA offset
  range to test SBUF overlap; masking off the partition is conservative (it over-reports aliasing).
* The ACT spill-save triggers' HBM destination (`var 9`, the 2 MB tmp buf) advances 32768 B per
  trigger and **wraps every 64 triggers** (wrap points at k = 69, 133, 197, ...).

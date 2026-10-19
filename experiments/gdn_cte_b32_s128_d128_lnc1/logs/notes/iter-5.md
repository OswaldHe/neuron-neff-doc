# iter-5 — removed iter-2's 160 SP relay `EVENT_SEMAPHORE`s by pointing the queued DMA at the
# consumer semaphores directly (restores the compiler's own DGE semantics).
# **Performance-neutral (1174.6 vs 1175.6 us, i.e. inside the noise band).**
# The value of this iteration is in §2: **five levers measured closed on the device**, including
# the whole SP-reordering family and iter-3's PSUM-evacuation candidate.

*(`scratch/` does not survive the iteration — every script referenced below is described well
enough in place to rebuild in a few minutes, and none is longer than ~60 lines.)*

## Result

| NEFF | total_exec_time_us (`--repeats 5`) | samples |
|---|---|---|
| start (= iter-4 shipped, HEAD `a8b683b`) | **1175.173** | 1172.9 1173.0 1175.1 1175.2 1180.9 |
| start, iter-4's own 4 runs | 1175.0–1176.6 | (median of medians 1176.0) |
| **shipped edit, run 1** | **1173.002** | 1170.9 1173.0 1173.0 1173.9 1176.9 |
| shipped edit, run 2 (`--repeats 3`) | 1174.838 | 1170.2 1174.8 1175.7 |
| shipped edit, run 3 | **1175.858** | 1170.9 1174.5 1175.9 1176.7 1177.6 |
| shipped edit, run 4 | **1174.304** | 1172.5 1173.0 1174.3 1174.8 1175.7 |

Median of the four shipped runs **1174.6**, of the two start runs **1175.6**. The between-measurement
floor on this kernel is ±2–3 us (iter-3 measured it twice on an identical NEFF), so **this is a
0-us result and I am not claiming a win.** `verify.py --runs 3`: PASS, all 3 bit-identical,
`max_abs_err` identical to the start NEFF to every printed digit (output_0 1.097e-04 = 10% of
limit, output_1 9.081e-04 = 83%). `launch_rules.py`: PASS. `semaphores.py`: zero over-waits.
Engine busy essentially unchanged (DVE 536.6, ACT 520.7, PE 340.5, DMA 332.5, Pool 233.4, SP 52.4).

**Why I shipped a 0-us edit:** it deletes 160 instructions, removes 110 us of SP blocking, and
replaces iter-2's relay hack with the semantics the compiler originally emitted (the DMA's own
completion increments the consumer's semaphore). It is a strictly simpler base for iter-6, and it
is measured not to regress. If the reviewer prefers a no-op iteration, reverting it costs nothing.

---

## 1. THE EDIT: a queued-DMA transfer can increment any semaphore set, so a relay is never needed

iter-2 moved 160 DGE reloads onto `qSPSpillReload1` (semaphores 72–87) and repaid each consumer
semaphore `S[56+k]` with an SP `EVENT_SEMAPHORE` relay. The relay is removable **outright**:
point the queue at `S[56..71]` and let the DMA do the `+16` itself. That is what the DGE queue
`qSPDynamicHW` (semaphore_set 56–71, now instruction-less and dead) did before iter-2.

### The constraint that makes it non-trivial
A queue's `semaphore_set[r]` is indexed by **ring = `transfer id % num_queues`**, and the ring is
served **FIFO by trigger order**, so a consumer's `S[56+k] >= 16(j+1)` means "the first j+1
transfers *triggered* on ring k are done". To preserve every consumer value unchanged you must put
exactly the 10 transfers that repay `S[56+k]` on ring k, **in their relay order**, and nothing else.

Measured structure of this NEFF (all verified, scripts in `scratch/`):
* 160 relays, exactly **10 per `S[56+k]`**, and the relay→`S[56+k]` map is **not** ring-identity:
  relay on queue-1 ring r repays an arbitrary `S[56+k]`. Even queue-1 rings carried 8 relays,
  odd ones 12.
* **relay program order == trigger program order on all 16 rings** (checked) — this is what makes
  the relabel value-preserving.
* 192 queue-1 transfers; the **32 that have no relay are block ids 4+6n** (`#comment "fused-varNN-0"`,
  the `op:"cast"` composites), and each is waited on by exactly **one Pool `EVENT_SEMAPHORE`**.

### Recipe (≈60 lines, `scratch/edit.py`)
1. Decode the relays: `b = (v/16-1)*16 + (r-72)` from the extended wait, `k = events_update_idx-56`.
   `P_k` = the 10 `b`s in program order.
2. `qSPSpillReload1`: `semaphore_set = [56..71]`; transfer `P_k[j]` gets **`id = j*16 + k`** (ids
   0–159) and `"semaphore": 56 + id%16`.
3. The 32 cast composites go to a **new 6th queue** `qSPSpillReload2` (`type data`, `owner sp`,
   `num_queues 16`, `semaphore_set [72..87]`), ids 0–31 in trigger order, `"semaphore": 72+id%16`.
   Rewrite the 32 Pool waits to `S[72 + n%16] >= 16*(n//16+1)`.
4. Rewrite all 192 `PSEUDO_DMATRIGGER`s' `count_union_block_id` (+ `dma_queue_name` for the 32).
5. Neuter the 160 relays in place: `events_update_mode = 0`, `events_extended_wait_mode = 0`
   (keeps the slot count, so nothing else shifts; they cost ~30 ns each now).
6. **Consumers of `S[56..71]` need no change at all** (370 waits on ACT/DVE/SP, values 16…160).
Then `pack`, `semaphores.py`, `verify.py`, `profile.py`. A 6th queue loads and runs (iter-3 §4.4
already proved this); two queues sharing no semaphores is fine, and the dead `qSPDynamicHW`
keeping its 56–71 set alongside the live queue1 was **accepted by the loader**.

### Why it bought nothing — the number that explains it, and it is the most reusable fact here
**A queued transfer's semaphore increment becomes visible to a waiting engine ~1.0–1.9 us after
the transfer's profiled `end_ts`, and the SP relay was hiding inside that window.**

| metric (160 queue-1 transfers) | with relay | relay removed |
|---|---|---|
| `dma end_ts` → **first consumer start**, median | 1938 ns | 1886 ns |
| same, mean / sum over 160 | 2421 / **387.4 us** | 2398 / **383.7 us** |
| relay "fire delay" (`relay end` − `dma end`), median / sum | 986 / **187.5 us** | — (gone) |

So deleting 187 us of relay fire delay moved the consumers by **4 us in total**. Before the edit,
68 of the 160 relays had their releasing consumer start within 600 ns of the relay — that number
*looked* like 88.9 us of recoverable latency and it converted at ~2%. Lesson:
**"consumer starts right after X" does not mean X is the gate**; the consumer is re-gated by the
semaphore-propagation floor the moment X is removed. This also definitively closes iter-4 §2.8
(hoisting relays earlier, "19.7 us available"): there is nothing there.

---

## 2. REFUTED / CLOSED THIS ITERATION — each with the measurement

### 2.1 **An SP `EVENT_SEMAPHORE` with a wait and NO update is a program-order barrier. Moving a
### trigger past one gives NaN. This closes the entire SP-reordering family, incl. iter-4's cand. #1.**
I re-ran iter-4 §1's head-of-line planner on the post-relay NEFF: 31 blocking ES slots with an
already-unblocked `PSEUDO_DMATRIGGER` run behind them, **26.8 us** of block. Applied the same
permutation that won iter-4 (-31 us). Result: **`verify.py` FAIL, output_0 all-NaN** (output_1 and
its error bit-exact, so the kernel ran and only the affected matmuls were wrong).

The cause is visible in the decode, and it is the thing to remember:

```
slot  268  EVENT_SEMAPHORE  wait S[37]>=112  extwait S[6]>=382   upd []   dur 781 ns
slot  269  PSEUDO_DMATRIGGER qSPSpillReload0 block 125  wait S[4]>=242
```
**`upd []` — the ES updates nothing.** Its only effect is to stall the in-order Sync engine, i.e.
it *is* the trigger's third and fourth wait operand (a `PSEUDO_DMA_TRIGGER_STRUCT` has **one**
wait field and no extended-wait block, so the compiler spends a barrier ES whenever a trigger
needs more than one). 25 of the 31 blockers are this exact shape (ring wait + `S[6]` wait, no
update), one per group.

iter-4's relays were safe to move precisely because they **did** have an update (they were pure
repayments and read nothing). The test is one line: **move an SP ES only if it updates something.**
Everything else on SP — iter-4's candidate #1 (hoisting the 27 W_t triggers past an `S[5]` waiter,
"expect ~10 us"), the generic trigger hoist (iter-4 §2.3), and this iteration's plain-ES hoist —
is the same illegal move. **Do not spend another iteration on SP instruction order.**

### 2.2 The barrier's ring wait is a stale throttle, but the `S[6]` (DVE) wait is the binding one.
On those 28 barriers I cleared the **primary** wait (`S[24..39] >= 16k`, the ring throttle iter-2
showed is stale in 218/260 cases) and kept the extended `S[6]` wait.
`verify.py`: **PASS, bit-identical.** `profile.py`: **1173.7 us = no change** (23.4 us of block
addressed, 0 us delivered). So the ~770 ns each of those barriers blocks SP is a **real DVE data
dependency**, and the queue-0 transpose behind it cannot be triggered any earlier. Reverted.

### 2.3 Retargeting the W_t transpose triggers to their real producer converts at 0%. Measured.
This was built from a **topology correction to iter-4 §3.2** (worth keeping): `W_all` is written by
the 32 `op:"cast"` composites and their waiters are **Pool `EVENT_SEMAPHORE`s**, not the
transposes. The real line-595/597 chain has **three** DMA hops:
`DVE Wg_grp (S[6]) → cast transfer (queue2) → Pool barrier → Pool DGE DMA (S[8..23]) →
queue-0 W_t transpose (S[24..39]) → LDWEIGHTS 597`.
I found the 65 queue-0 transposes whose **entire read set is covered by exactly one earlier cast
transfer** (cell-exact cover, with last-writer resolution by trigger order — `scratch/wfuse2.py`)
and retargeted the 33 of them whose trigger waited on a Pool DGE ring (`S[8..23]`) straight to the
cast's completion, deleting the Pool DGE hop from the chain.
`verify.py`: **PASS, bit-identical** (so the Pool DGE DMA really is not a producer and not a WAR
gate for these). `profile.py`: **1173.6 us = no change.** Reverted.
Reason, consistent with §2.1/§2.2: the trigger is not gate-bound, it is **SP-program-order bound**
behind a barrier whose DVE wait is binding. Making a trigger's own wait earlier cannot help while
a barrier in front of it is still waiting.

### 2.4 **PSUM evacuation has zero redundancy. iter-3's candidate #1 is CLOSED, statically.**
iter-3 called this "by far the biggest coherent block left" (1056–1536 instructions, 225 us of
engine busy, ~93 us of critical path) and left open whether any of it is duplicated. It is not:
* 1536 instructions read PSUM (`src ...start_addr_addr_immediate >= 0x2000000`): ACT CAST 640,
  DVE COPY 288, DVE TENSOR_TENSOR 192, ACT TENSOR_SCALAR 128.
* 994 of them share an **identical signature** (every non-event field) with another — but that is
  only buffer reuse across the 32 groups: of the **1140** same-bank pairs that are consecutive in
  time, **0** have no `MATMUL` rewriting that PSUM bank in between.
* Grouping evacuations by (bank, generation) where generation = #matmuls that wrote the bank
  before it: **1446 of 1452 groups contain exactly one evacuation**, and **0** pairs are identical
  except for their destination. There is nothing to delete and nothing to re-point.
PSUM has **8 banks** and a `MATMUL`'s destination immediate is the bank base, which makes this
generation count cheap and exact (`scratch/dupe3.py`, `dupe4.py`, ~40 lines).

### 2.5 Measurement hygiene note that cost me a wrong hypothesis
"68 relays have a consumer starting within 600 ns of them, worth 88.9 us" was the whole basis for
§1 and it converted at 2%. The honest form of that probe is: **remove the candidate and re-measure
the consumer's start time**, which is what §1's table does. Slack-based upper bounds on this kernel
over-predict by 20–50x because every dependency is re-gated by the ~1.5 us DMA/semaphore floor.

---

## 3. Where the remaining 1175 us is (measured on this iteration's NEFF, `prof2/`)

Whole-machine idle (no non-sync instruction running on any engine): **194.8 us** in 1453 gaps;
**102.4 us in the 68 gaps over 1 us**. Attributed to the first work instruction after the gap
(`gdn-bench/vendor/gdn/gdn_cte.py`):

| site | gaps >1us | us | what it is |
|---|---|---|---|
| PE `LDWEIGHTS` 597 | 20 | **28.8** | `nc_matmul(WS_p, W_t, state_bf16)` — the 3-hop W path (§2.3) |
| PE `MATMUL` 447 | 12 | 15.6 | doubling-solve squaring |
| SP `WRITE` (trigger issue, incl. 350/607/560/517/589/495) | 20 | ~32 | the §2.1 barriers waiting on DVE |
| ACT `TENSOR_SCALAR` 591 | 3 | 5.4 | `k_dec_full` |
| everything else | 13 | ~20 | nothing over 3 us |

Engine busy: DVE 536.6 (45.7%), ACT 520.7 (44.4%), PE 340.5, DMA 332.5, Pool 233.4, SP 52.4.
Nothing saturated. iter-3 §1's decomposition still governs: **795 us of real instruction execution
on the critical path**, ~220 us of DMA latency, ~105 us of semaphore blocking, ~98 us of gaps.

**The shape of the problem after five iterations:** every remaining gap resolves to either a real
DVE/ACT compute dependency or the ~1.5–2.6 us fixed cost of a DMA hop
(717 ns trigger→start + ~400 ns transfer + 1.0–1.9 us completion→consumer-visible). Both are
immovable by field edits. Scheduling is exhausted; only *fewer instructions* or *fewer DMA hops*
can move this kernel now.

## 4. Ranked candidates for iter-6

1. **Make `Wg_grp` bf16 at its producer** (the DVE `tensor_copy` of PSUM `yW`, line 466/467). Still
   the largest ceiling: it collapses the W path's cast hop and the transpose hop into **one bf16
   permute+transpose descriptor** (iter-4 §2.5 proved transpose+**cast** is rejected at load, so
   the cast must leave the DMA; a 16x128-tile, 4-descriptor transposing transfer is device-verified
   there). Site 597 carries 28.8 us of idle and the chain is 3 hops (§2.3) — removing two hops is
   ~3–5 us x 64. Cost: `Wg_grp` is fp32 with a 512 B row stride; writing bf16 into the same stride
   keeps every reader's *address* but changes its `in_dtype`/element stride (line 407 already casts
   `Wg[bj]` to bf16, so numerics should be unchanged). **Enumerate `Wg_grp`'s readers first** — if
   it is more than a handful, this is a layout refactor, not a field edit.
2. **Removing instructions from DVE/ACT is mechanically possible**, and I did not know that before
   this iteration: only 2093 of 2884 DVE and 1306 of 2555 ACT instructions update their engine
   counter (`S[6]`/`S[4]`), so a removal is legal if you then **decrement every later wait on that
   counter by one** (I have no script for this; it is ~30 lines). This is the gate on iter-3's
   candidate #2 (229 DVE SBUF→SBUF pure-relocation COPYs, 52 us busy, ~33 us of path). Needs the
   SBUF interval map to prove each destination's readers can read the source instead.
   Expected ≤10 us — do candidate 1 first.
3. **Do NOT spend budget on:** everything in iter-1 §5.4, iter-2 §5.5, iter-3 §5.4, iter-4 §4.5,
   plus §2 above. Newly and definitively dead: **all SP instruction reordering** (§2.1 — barrier
   ES, and this kills iter-4's candidate #1), **SP barrier ring-wait removal** (§2.2, 0%),
   **retargeting a trigger to its true producer** (§2.3, 0%), **PSUM-evacuation redundancy**
   (§2.4, 0 candidates), **relay removal / relay hoisting** (§1 and iter-4 §2.8, 0%).

## 5. Tooling notes (additions to iter-4 §5, which is all still correct)
* A transfer in `SP0.json` carries its own `"semaphore"` field *and* its queue carries
  `semaphore_set[ring]`. Keep them consistent; I never established which the runtime honours.
* `DmaPacketAggregated` contains **only the queued transfers** (669+160+32 here). Pool/ACT **DGE**
  DMAs are not in it — find those by decoding `PSEUDO_DMA_DIRECT2D` on the owner engine and reading
  its `semaphore` field.
* `isa.Instruction` has `.as_dict()` (no `.keys()`/`.fields` dict). Address fields are
  `{src,dst}_mem_pattern_start_addr_addr_immediate`; `..._addr_reg_marker` is just the top byte of
  that 4-byte immediate. PSUM source ⇔ immediate `>= 0x2000000`.
* `semaphores.py` reports `max_wait 0` for semaphores whose only waiters are Pool
  `EVENT_SEMAPHORE`s (it missed all 32 of mine). Do not rely on it to validate Pool waits — decode
  `Pool0.bin` yourself.
* Head-of-line blocking recipe (iter-4 §5) still works and is still the cheapest diagnostic, but
  **add the `updates != {}` filter** before you act on it (§2.1).
* A full probe cycle is `pack` 15 s + `verify.py --runs 1` 25 s + `profile.py --repeats 5` 80 s.
  Four of this iteration's five closures cost one cycle each.

## 6. Scoping data for candidate 1 (collected, so iter-6 does not have to): **`Wg_grp` is shared scratch**
`Wg_grp` is read by the 32 `op:"cast"` composites at **partitions 96-111 and 112-127**, offsets
`0xe000 / 0xf100 / 0xf400 / 0x377f8`, 2048 B each (`from_sizes [512,4,16]`, `from_steps
[1,512,262144]`, so 4x512 B contiguous x 16 partitions; note `0xf100` and `0xf400` *overlap*).
Counting every instruction whose src/dst immediate lands inside those 8 cells
(`scratch/wg2.py`; compute-side immediates are **flat** `partition*262144 + offset`, so the
`_addr_reg_marker` byte is already included):

| who | n | note |
|---|---|---|
| DVE `CAST` src (FP32 -> **BFLOAT16**) | 353 | the region is *already* being cast to bf16 on DVE elsewhere |
| DVE `COPY` dst (FP32) | 234 | the producers |
| queue-1 DMA `from_off` (fp32->fp32) | 54 | |
| queue-2 cast `from_off` (fp32->bf16) | 64 | the hop we want to delete |
| ACT `ACTIVATE` src / dst, DVE `TENSOR_TENSOR` dst | 4 / 2 / 2 | |

**So this window is shared scratch, not a private `Wg_grp` tile: ~700 instructions touch it.**
Changing its dtype is a layout refactor touching hundreds of readers, *not* the "line 466/467 +
re-address a handful of readers" that iter-4 §4.2 assumed. Before attempting it, build a
byte-exact SBUF interval map and re-scope to the cells one cast descriptor actually reads; if the
true `Wg_grp` tile still has hundreds of readers, candidate 1 is dead too and this kernel is done
for field edits.

## 7. The structural ceiling, measured: **chunks already overlap 2.3x, and the limiter is the
## per-chunk *period*, not any engine's throughput**
iter-4 §4.4 said "the 32 groups are independent, the schedule is near-sequential, perfect overlap
would be DVE-bound at ~535 us, a 2.2x". The first half is wrong and the number is worth having
exactly (`scratch/overlap.py`, 20 lines, uses `nki_source_location` + `start_ts`):

* 64 chunks (32 groups x 2). Per-chunk **latency** (line 221 start -> line 628 end) median
  **43.0 us** (min 37.3, max 77.2). Chunk **period** (median spacing of a once-per-chunk line)
  **18.4 us**. Wall = 43 + 63*18.4 = 1202 ~= 1175. **So ~2.3 chunks are in flight at all times.**
* Per-chunk engine busy: DVE 8.4 us, ACT 8.1, PE 5.3, Pool 3.7, SP 0.8. **The period is 2.2x the
  busiest engine's per-chunk work**, so no engine's throughput sets it — the period is set by how
  far the compiler interleaved consecutive chunks into each in-order engine's program.
* Reaching the DVE bound (536 us) needs overlap ~5.2x, i.e. a reschedule of 11k instructions.

### 7.1 And the "head-of-line blocking on the compute engines" prize is an illusion. Do not chase it.
Same recipe as iter-4 §1, applied to the work engines (`scratch/dvehol.py`):

| engine | total gap | gap in front of a slot that has **ready work** behind it | ready work followers / of which update no semaphore |
|---|---|---|---|
| PE | 867.2 us | **844.2** us (853 sites) | 2874 / 1162 |
| ACT | 663.0 us | **413.6** us (505 sites) | 992 / 273 |
| DVE | 664.7 us | **155.3** us (305 sites) | 828 / 91 |

Those look like enormous levers and they are not, for a reason that applies to every future
reorder of a *work* instruction: hoisting a ready instruction X in front of a blocked B **does not
shorten B's wait** — B still waits exactly as long. All it buys is X's own execution time
(~250 ns) moved into the bubble. So the ceiling is ~250 ns x n of *engine issue time*, and iter-3
§2.1 and iter-4 §2.4 both measured that removing ACT issue time converts at ~0%. On **PE** it is
worse than useless: 1162 of the 2874 non-updating ready followers are `LDWEIGHTS`, which load the
weight register the *preceding* `MATMUL` consumes — hoisting one is always wrong.
The only reorder that could raise the 2.3x overlap is moving a whole chunk's chain, which means
moving semaphore-*updating* instructions, which means rewriting every wait on that engine's
counter (`S[4]`/`S[5]`/`S[6]`) to the moved instruction's new ordinal. That rewrite is mechanical
(the wait value **is** the ordinal of the k-th updating instruction, and only 1306/2555 ACT and
2093/2884 DVE instructions update), but iter-3 §3b found that even a correct-looking DVE
dependency proof produces **hangs**, so this is a multi-iteration project, not an edit.

### 7.2 Resource headroom, in case someone does attempt §7
* **PSUM is not the blocker**: mean **2.1 of 8 banks live**, max 6 (`PsumUsage.parquet`;
  distribution of #banks live = {0:425, 1:1194, 2:1364, 3:909, 4:444, 5:167, 6:31}).
* The machine is nowhere near a hardware roof: `Summary.parquet` gives 864.5 MFLOP in 1.19 ms
  (~0.7 TFLOP/s) and `mbu_estimated_percent` 0.0074. 861 DMA transfers, 2080 PSUM->SBUF
  evacuations, 104 MB of SBUF reads, only 3.2 MB of HBM traffic each way.
* So the 2.3x overlap is purely a program-order property, not a capacity limit.

## 8. Bottom line for iter-6
Five iterations have now removed every *scheduling* inefficiency this NEFF had: 1351 -> 1175 us
(-13.0%). What remains decomposes into (a) real DVE/ACT arithmetic on the chunk's dependence
chain, (b) the fixed ~1.5-2.6 us cost of a DMA hop (717 ns trigger->start, ~400 ns transfer,
**1.0-1.9 us completion->consumer-visible**, §1), and (c) a per-chunk period that only a full
reschedule can shrink (§7). Every cheap lever is closed and recorded across iter-1..5 §"refuted".
**If iter-6 is not going to attempt the §7 reschedule or the §6 layout refactor, the honest answer
is that this kernel is finished for binary editing**, and the budget is better spent confirming
that than on another 0% experiment.

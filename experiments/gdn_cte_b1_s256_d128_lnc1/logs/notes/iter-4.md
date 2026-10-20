# iter-4 — the var11/var12 staging group: queue placement, two 4-to-1 fences, one needless DMA wait

## Result: **61.24 → 59.42 us (-1.8 us, -3.0%)**, outputs bit-identical
(means of repeated 7-capture medians: iter-3's NEFF 61.403 / 61.074; shipped 59.361 / 59.291 /
59.603)

Three edits, all kept. All measurements are 7-capture medians of `tools/profile.py`.

| step | edit | median us | note |
|---|---|---|---|
| 0 | iter-3's NEFF (`c8fefa05…`), measured twice this session | **61.403 / 61.074** | A-side |
| 1 | staging DGE: one payload moved to an earlier-issuing slot + the 8 DGEs spread over the three DGE queues | **60.686** | -0.72 |
| 2 | + retarget every staging-block reader at its own producer (kill two 4-to-1 fences) | **59.958 / 59.766** | -0.73 |
| 3 | + `PE[134]` gated on the staging DGE (`S[43]>=16`) instead of queued transfer 20, reading that DGE's block directly (§10) | **59.361 / 59.291 / 59.603** | -0.45 |

Interleaved A/B after step 2: iter-3 **61.074** vs **59.766** (-1.308); step 3 then took another
0.4-0.6 (and moved `PE[134]` 3.8 us earlier internally, see §10).
`verify.py --runs 3`: PASS after each edit, bit-identical across runs, `max_abs_err` identical to
the baseline to every digit (output_0 6.092e-05 = 11%, output_1 4.494e-04 = 83%).
`launch_rules.py` PASS. `semaphores.py` no over-wait. Members changed vs iter-3:
`sg00/{Activation0.bin, Pool0.bin, SP0.bin, DVE0.bin, PE0.bin}`.

---

## 1. The new fact that unlocked this: a DGE's **queue is chosen by its semaphore**, not by its host

`def.json`'s `dma_queue` block is the whole story:

| queue | owner | `semaphore_set` | profile name |
|---|---|---|---|
| `qActDynamicHW` | act | **40..55** | `qScalarDynamicHW` |
| `qPoolDynamic` | pool | **8..23** | `qGpSimdDynamic` |
| `qSPDynamicHW` | sp | **56..71** | `qSyncDynamicHW` |
| `qSPSpillReload0` | sp (data) | 24..39 | the 33 queued transfers |

A `PSEUDO_DMA_DIRECT2D` has no queue field — only `semaphore`. Its transfer runs on whichever
queue owns that semaphore, **independently of the engine whose stream the slot sits in**. The
unedited NEFF already mixes them (Pool slots 21/22/23 used sems 44/45/48 = the *Act* queue;
SP slots 64/65 use sems 20/21 = the *Pool* queue), so every host×queue combination is legal.

That is why iter-1/2/3's "move the DGE to an idle engine" edits kept hitting a wall and why
iter-2 §6 concluded "the completions land in the same order at the same time": **all eight
var11/var12 staging DGEs were on one queue**, so moving the *issue* never moved the *completion*.

Free semaphores in this NEFF (no inc, no wait, including extended waits): **22, 23, 40, 41, 42,
51, 52, 53, 54, 55, 68, 69, 70, 71** (65/66/67 are now used by edit 1). `semaphores.py` cannot
see extended waits — use the census in §9.

## 2. What the 48.6→55 us hole actually is (every compute engine idle ~6.4 us)

Eight DGEs copy var11 (group A) and var12 (group B) from the ACT output region to the layout
their consumers need. Decoded, they are one family — src `p(16g+j) + 0x6480` (fp32,
4x512 B/partition), dst `p(64+16i+j) + base + stride*g`, i.e. an **offset-index <-> partition-block
swap** that no compute instruction can do:

| slot (host) | sem | src | dst | dtype | read by |
|---|---|---|---|---|---|
| ACT[82] | 43 | p0+0x6480 | p64+**0x1CC00** | fp32->bf16 | queued transfer **20** |
| Pool[21] | 44 | p16 | **0x1CD00** | fp32->bf16 | queued transfer **22** |
| Pool[22] | 45 | p32 | **0x1CE00** | fp32->bf16 | queued transfer **26** |
| SP[30] | 46 | p48 | **0x1CF00** | fp32->bf16 | queued transfer **30** |
| ACT[88] | 47 | p64 | **0x1D000** | fp32 | **DVE[99]** |
| Pool[23] | 48 | p80 | **0x1D200** | fp32 | **DVE[110]** |
| SP[34] | 49 | p96 | **0x1D400** | fp32 | **DVE[120]** |
| ACT[94] | 50 | p112 | **0x1D600** | fp32 | **DVE[129]** |

**Every block has exactly one reader.** (`scripts/scan.py`: every instruction field in
`p64+0x1CC00 .. p64+0x1D7FF` across all five streams — only the 8 DGE writes and the 4 DVE reads
appear; the 4 transfer reads come from `SP0.json`'s `from_off`.)

Timing facts (iter-3's NEFF, `prof/`):
* group A's sources are ready at `S[4]>=42` (ACT[85], 50.5 us), group B's at `S[4]>=43`
  (ACT[87], 51.2) except p112 which needs ACT[93] (52.5).
* a DGE row costs 610-670 ns on its host; the three hosts issue the 8 rows between 49.2 and 53.0.
* **row end -> semaphore visible is ~1.40 us when the fabric is quiet and 1.9-2.4 us when the
  group is in flight.** The 8 completions landed 51.3, 52.6, 53.9, 53.9, 54.1, 54.4, 54.6, 54.8 —
  i.e. ~3.5 us to drain 256 KB read / 192 KB written = **~54 GB/s aggregate, bandwidth-limited**.
* consumers: `Pool[24,25,26]` waited on all four of 43/44/45/46 and produced `S[16]`, which
  `SP[35]` waited on, immediately in front of `SP[36]` = the trigger of transfer 20.
  `SP[37,38,39]` waited on all four of 47/48/49/50 and produced `S[64]`, waited by `DVE[98]`,
  immediately in front of `DVE[99]`.
* `PE[134] <- S[28]>=32` (transfer 20) was the gate that held every engine idle.

## 3. Edit 1 — earlier slot for the fence-critical payload + three queues (-0.72 us)

Two independent mechanisms, applied together:

**(a) payload swap.** Slots issue at fixed times; payloads can be exchanged between slots. The
fence-critical group-A payload (src p32) sat in Pool[22], whose row ends last (52.25 us), while a
group-B payload sat in ACT[88], whose row ends at 51.54. ACT[88] and Pool[22] differ **only** in
the address/dtype/semaphore fields (identical events block, sizes and steps), so a raw 64-byte
slot swap exchanges the two transfers *with their semaphores*, leaving every consumer's wait
valid. Guards after the swap: ACT[88] needs `S[4]>=42` (ACT program order already implies
`>=43`); Pool[22] needs `S[4]>=43` (its old guard was ACT program order after ACT[87] — the same
thing). I wrote both waits explicitly rather than relying on program order.

**(b) queue split.** sem 46->**22**, 45(on ACT[88])->**65**, 49->**23**, 47(on Pool[22])->**66**,
48->**67**, so the 8 transfers sit 3/2/3 on qAct/qPool/qSP, with the four group-A ones on three
different queues. Each consumer's wait index was moved with it.

Measured effect (per-semaphore completion times, `prof` vs `prof2`): the last group-A completion
54.14 -> 53.37, the Pool fence output 54.28 -> 53.49, the transfer-20 trigger 54.38 -> 53.59,
**`PE[134]` 56.07 -> 55.06 (-1.01 us)**, last instruction end -1.02 us. Metric -0.72 us.
Most of it is (a): the per-transfer latency stayed ~1.9-2.0 us even on a private queue, so the
queue split alone is worth only ~0.2-0.4 — **the window is bandwidth-limited, not queue-limited**.
Do not expect more from queue juggling here.

## 4. Edit 2 — the two fences were 4-to-1 barriers in front of the *earliest* consumer (-0.73 us)

Using §2's table, each reader now waits on its own producer:

* `SP[35]`: extended wait `S[16]>=1` -> **`S[43]>=16`** (transfer 20 reads only 0x1CC00).
  The full fence `S[16]>=1` moved to `SP[42]`'s free extended wait — `SP[42]` is the slot just
  before `SP[43]`/`SP[49]`/`SP[59]`, the triggers of transfers 22/26/30 which read the other
  three blocks. Net: the transfer-20 trigger fires **1.5 us earlier** (its remaining gates are
  `S[4]>=44`, the WAR guard on its destination, and `S[28]>=16`).
* `DVE[98]` (was the `S[64]>=1` fence, sits right in front of DVE[99]): wait -> **`S[66]>=16`**.
  The three later readers got their producer's semaphore on the nearest preceding slot with a
  free wait field: `DVE[107]` xwait `S[67]`, `DVE[117]` xwait `S[23]`, `DVE[128]` wait `S[50]`.
  (DVE[107]/[117] run at 57.7/61.6 us and DVE[128] at 66.8, so these waits cost nothing.)
* `SP[37,38,39]`'s waits deleted — nothing waits on `S[64]` any more and they sat in front of
  `SP[41]`, the transfer-21 trigger. `SP[39]` still increments the now-unwaited `S[64]`.

**Trap:** `TENSOR_TENSOR_ARITH_OP` has no extended-events fields (same as `MATMUL`/`LDWEIGHTS`,
iter-3 §10), so you cannot give a DVE compute op a second wait — use the nearest preceding
`EVENT_SEMAPHORE`, or a slot whose normal wait is unused.

Measured: the transfer-20 trigger another 1.5 us earlier, but `PE[134]` only 0.2 us earlier,
because transfer 20 now runs *inside* the DGE flood and its trigger->consumer latency grew
1.47 -> 2.20 us. The DVE side gained nothing at all for the same reason (`S[66]` lands at 54.81
in `prof3` where `S[64]` landed at 54.83 in `prof2`). The -0.73 us is real (two 7-capture medians
plus the A/B) but **the mechanism is "fewer serial hops", and it is now capped by fabric
bandwidth in that window.**

## 5. What is closed now (do not retry)

* **Rebalancing DGE *hosts* in the 49-55 window** — closed by iter-2 §6 and re-confirmed: the
  completion order follows the fabric, not the issue engine. Only the payload<->slot assignment
  (§3a) matters, and it is now optimal: group A's four rows end at 49.7/51.0/51.4/51.4, which is
  the earliest the three hosts can issue them after `S[4]>=42`.
* **More queue splitting in that window** (§3b): bandwidth-limited, ~54 GB/s aggregate.
* **Over-synchronisation of the var11/var12 staging group**: both fences are now 1-to-1. There is
  no fence left to relax there; `semaphores.py` + the §9 census show every staging semaphore has
  exactly one waiter at `>=16`, each the real reader.
* **Merging the staging DGEs**: impossible. The mapping needs three index dimensions
  (g, i, j) on each side; a `DMA_DIRECT2D` has two plus `elem_size`.
* **Fusing the DGE cast into the transposes** (one hop instead of two): the transposes are
  hardware `transpose_op: WZXY` queued descriptors; and iter-3 §3 proved descriptors inside one
  queued transfer run concurrently, so a cast-then-transpose inside one transfer is a race.
* **The var10/var9 reload fence at 28-32 us** (sems 56-59 / 60-63, the `ACT[39,40,41]` and
  `DVE[33,34,35]` stacks): looks like the same 4-to-1 pattern and `S[59]` does land 4 us after
  `S[56]`, **but it has ~2.2 us of slack**. `ACT[41]` (the reader) finishes at 32.06 while
  `ACT[42]`, the next link, is gated by `PE[23]` (the transfer-5/6 chain) until 34.26. Measured,
  not guessed. The critical path there is still iter-3 §2's DMA chain.
* **Tail output stores** (`SP[64]` sem 20, `SP[65]` sem 21, the last program instructions before
  `Pool[35,36]`): `SP[65]`'s row waits 181 ns for `SP[64]`'s row to finish, and its own gate
  (`S[6]>=93`) lands at 70.57. Moving it to an appended ACT slot would buy **~240 ns**, and Pool
  is the wrong host (iter-2 §3's ~530 ns start-up gap after an idle stretch applies — Pool is
  idle 67.8->73.0). Not worth the slot-append risk; I measured the numbers and dropped it.

## 6. The 9 do-nothing `EVENT_SEMAPHORE` slots (unused lever, measured but not taken)

`scripts/noop.py` finds every `EVENT_SEMAPHORE`/`NOP` with no wait, no extended wait and no
update — almost certainly carcasses of DGEs that iter-1/2/3 moved off an engine:
ACT **83, 84, 86, 89, 90** (40 ns each, ~90 ns apart, in the 49.7-51.6 window);
Pool **33, 34** (19 ns, at 66.1); SP **14, 15** (32/38 ns, at 31.7).
The tkg kernel's iter-7 got -2.6% by deleting 32 such slots, but there each one sat on a
24-head serial cycle. Here ACT's five cost ~450 ns of *sequencer* time, and the sequencer runs
ahead of the compute pipeline (a DGE row even overlaps the following `ACTIVATE`), so only
`ACT[89,90]` plausibly delay anything (~180 ns before `ACT[91]`). SP[14,15] and Pool[33,34] are
off-path. I left them; a future iteration could delete all nine in one edit (shortening a stream
is fine — `neff.py pack` recomputes sizes; the `.dbg` members only carry profile labels).
**But note Pool[33]/[34] are the only free slots in the right place for a tail-store move (§5).**

## 7. Where the remaining time goes (shipped NEFF, `prof3`)

Tensor is the busiest engine in the body (67%); its idle is two holes and a string of ~1.4-1.9 us
PE<->DVE<->ACT handoffs:

| Tensor idle | window | what it waits for | status |
|---|---|---|---|
| 9.9 us | 21.8 -> 31.7 | transfer 5 (32 KB scatter copy) -> transfer 6 -> `PE[21]/[23]` | iter-3 closed every lever |
| **6.4 us** | 48.6 -> 55.1 | the 8 staging DGEs, now bandwidth-limited | this iteration took 1.3 us out of it |
| 1.9 / 1.7 / 1.4 / 1.4 | 65.7 / 58.3 / 61.5 / 67.7 | `S[6]>=91`, `S[6]>=76`, ... = DVE links of the rigid cycle | untouched |

The kernel is **DMA-latency bound end to end**: a queued transfer costs ~1.1-1.4 us
trigger->consumer for 220-380 ns of actual fabric time, a DGE ~1.4 us row-end->semaphore, and the
critical path crosses at least a dozen of them. Everything structural left is "remove a hop",
and the three hop-removal mechanisms (fuse into the consumer, split across rings, add a
transfer) are refuted in iter-3 §3/§4/§6.

## 8. Ranked candidates for iter-5

1. **Hunt for more instructions that wait on a DMA they do not need** (§10.7). That is what the
   shipped step-3 edit was, and the recipe is mechanical: list every wait on a queued-transfer
   completion, take its idle time from the profile, and run the one-run sensitivity control
   (point the operand elsewhere; if the outputs do not move, the wait is free to retarget at the
   real producer). The whole ACT and DVE side is unexamined. `transpose_mode` itself is closed
   (§10.5-10.6) — do not retry it, and do not retry the throttle or the wait retargets on
   `PE[142]`/`PE[151]`; five device runs refuted those.
2. **Delete the nine no-op slots** (§6). Cheap, ~0.2 us at best, and it frees Pool[33]/[34] for
   candidate 3.
3. **Tail store split** (§5, ~240 ns as a host move; more if the 64 KB `SP[64]` store is split in
   half across two queues so the two halves' ~1.34 us HBM latencies overlap — untested, and
   iter-3 §4 warns that splitting one transfer across rings shared a throughput budget).
4. **Re-check the var9/var10 fence at 28-32 us** (§5) only if the transfer-5 chain ever gets
   faster — it has 2.2 us of slack today, which would vanish first.
5. **Do not** retry anything in §5, iter-3 §9.3, iter-2 §10.4 or iter-1 §7.

## 9. Method notes

* `scripts/` (not preserved): `d2.py` stream dump with waits/extended-waits/updates/DGE payload
  (`WORKDIR=workN` to point at another unpacked tree), `waits.py` the full wait+increment census
  **including extended waits** (`semaphores.py` misses those — sems 45/49/58/62 look unwaited and
  are not), `scan.py`/`scan2.py` every instruction field pointing into an SBUF window (this is
  what found the 1-reader-per-block map), `noop.py`, `gaps.py`, `win.py` a time-ordered dump of
  one window, `sem.py` semaphore reach times, `scan3.py`/`scan4.py` readers/writers of one buffer or PSUM
  bank with their profile times, `edit1.py`..`edit8.py` one per edit and probe, all self-asserting.
* **`SemaphoreUpdate.parquet` is the best tool in the box for DMA work.** `id` is a string like
  `S[44] (qScalarDynamicHW)`, so it names the queue as well as the semaphore, and `value==16`
  rows give you each DGE transfer's exact completion time. `DmaPacketAggregated` only covers
  *queued* transfers, never DGEs.
* **Normalising a profile by `start_ts.min()` is not safe for A/B.** The pre-`PE[21]` region
  drifts +-400 ns run to run (iter-2 §9), so landmark-vs-landmark comparisons need an anchor
  *inside* the rigid region (I used `Tensor` pc 133) or they show phantom shifts. Compare
  `last_end - Te133`, not raw times.
* `neff.py unpack` writes `payload/{info.json,kelf-0.json,neff.json}` next to `payload/sg00/`.
  If you build a new work tree by copying only `payload/sg00`, `pack` silently drops three
  members and the NEFF is 3 members short — `neff_diff` catches it ("3 only in A"). Copy the
  whole tree.
* Run `verify.py --runs 3` after every edit (iter-3 §0); all three shipped edits passed it.
* And pair any operand/address change with a **sensitivity control** (§10.5) — a PASS after
  changing an operand means nothing until you know the output depends on that operand.
* Spread this session: iter-3's NEFF measured 61.403 and 61.074; the step-2 NEFF 59.958 and
  59.766; the shipped one 59.361, 59.291 and 59.603. So +-0.35 us on a 7-capture median — anything under
  ~0.5 us needs an interleaved A/B.
* Device runs this iteration: 11 verifies + 6 profiles, no hang. Every failed edit was a wrong
  *result* (NaN), never a timeout, so the cost of a bad PE/semaphore edit is ~4 minutes. Within
  your own session, keep a copy of the last good `kernel.neff` and one unpacked work tree per
  accepted step (I used `work3`, `work6`; nothing outside `kernel.neff` survives the iteration),
  so every revert is one `cp` and every new variant is one `pack` away.

## 10. `LDWEIGHTS.transpose_mode` — the lever is NOT proven; the win came from somewhere else

**Read §10.5 before anything else in this section.** The shipped step-3 edit is correct and its
gain is real, but the mechanism is *not* "PE does the transpose". `PE[134]`'s weight operand turns
out not to influence the outputs at all, so the equivalence I thought I had verified was vacuous,
and the same edit on the two sites whose operands *do* matter produces NaN.

### The pattern
Four queued transfers exist only to transpose one 64x128 bf16 block that a group-A staging DGE
just wrote, for one `LDWEIGHTS` each:

| staging block | sem | queued transfer (ring sem) | writes | the one `LDWEIGHTS` that reads it | its wait | its idle wait in `prof3` |
|---|---|---|---|---|---|---|
| p64+0x1CC00 | 43 | **20** (`S[28]>=32`) | p0+0x4500 | `PE[134]` src `0x457e` | `S[28]>=32` | **6248 ns** |
| p64+0x1CD00 | 44 | **22** (`S[30]>=32`) | p0+0x4000 | `PE[142]` src `0x407e` | `S[30]>=32` | 706 ns |
| p64+0x1CE00 | 65 | **26** (`S[34]>=32`) | p0+0x5500 | `PE[151]` src `0x557e` | `S[34]>=32` | 1556 ns |
| p64+0x1CF00 | 22 | **30** (`S[38]>=32`) | p0+0xE100 | `PE[161]` src `0xE17e` | `S[38]>=32` | 195 ns (no slack) |

All four transfers are byte-for-byte the same shape: `from_sizes [256,64,1,1]`,
`transpose_shape [1,1,64,128]`, `transpose_element_size 2`, `num_tiling_dimensions 2`.
Each `LDWEIGHTS` is `num_active_rows=128, num_active_cols=64, src_num_elem=[64,1,1],
src_step_elem=[-1,0,0]` reading the *transposed* copy.

### The encoding (device-verified twice)
Corpus evidence first (`artifacts/corpus/*.parquet`, `raw[34]` is `transpose_mode`): 16913 of
686403 `LDWEIGHTS` have `transpose_mode=ENABLED` (in `01-Engram`, `01-MoE-long8192`,
`20-Attention`, `40-DSparkAttention`, `40-MoE`, `HCSublayerIn`), and a transposed load uses an
**identical** `src_mem_pattern` to a plain one (same `num_elem`, same `step_elem [-1,0,-32]`,
same rows/cols) — so the flag changes only how the array is filled, not how SBUF is walked.

What works, on `PE[134]`:
```
src_mem_pattern_start_addr_addr_immediate :  p0 + 0x457e  ->  p64 + 0x1CC00 + 254
src_mem_pattern_num_elem                  :  [64,1,1]     ->  [128,1,1]
src_mem_pattern_step_elem                 :  [-1,0,0]     ->  unchanged (still reverse)
num_active_rows / num_active_cols         :  128 / 64     ->  unchanged
transpose_mode                            :  DISABLED     ->  ENABLED
```
i.e. **`num_elem` = elements per partition = the row count when transposing** (it was the column
count when not); the partition count comes from `num_active_cols`; the start address is still the
*high* end of the run because the step stays -1. Keep rows/cols describing the array, not the
source. Verified two ways: (i) with the original `S[28]>=32` wait left in place, so only the
addressing changed — `max_abs_err` identical to the baseline to every digit; (ii) then with the
wait retargeted to `S[43]>=16`, `verify.py --runs 3` bit-identical.

### The gain and the limit
`PE[134]`'s wait `S[28]>=32` -> `S[43]>=16` makes it fire at the staging DGE's completion instead
of after transfer 20's extra ~1.4-2.2 us round trip: `PE[134]` moved **3.8 us earlier**
(anchor-relative 50865 -> 47046) and the last instruction ~1.0 us earlier, for -0.45 us on the
metric. It is capped downstream by `DVE[98] <- S[66]` (the group-B DGE at ~54.8 us), which is
bandwidth-bound (§3/§4). Transfer 20 still runs — I did **not** neutralise its trigger (iter-3 §6
makes touching queued-transfer plumbing look hang-prone, and nothing else reads its output).

### It works on `PE[134]` and NOT on `PE[142]`/`PE[151]`, and the reason is DATA, not timing
Four device runs, all **deterministic all-NaN** (16384-24576 NaNs; `--runs 3` bit-identical, so
not a race):

| attempt | result |
|---|---|
| all three (`PE[134]`+`[142]`+`[151]`) | NaN |
| `PE[151]` alone (load + wait -> `S[65]>=16`) | NaN |
| `PE[151]` + a throttle: inserted a bare `EVENT_SEMAPHORE` with the old `S[34]>=32` **after** `PE[152]`, so only the pair runs early | **still NaN** -> "PE runs ahead" is NOT the cause |
| `PE[142]`: transposed load from `0x1CD00` with its **wait left at `S[30]>=32`** (addressing changes, timing does not) | **NaN** -> the cause is the *data*, not the schedule |

So `PE[134]`'s success is a special case: for that block, the PE array's transposed load of
`p64+0x1CC00` happens to be equivalent to the DMA's `WZXY` transpose of the same bytes, and for
`0x1CD00`/`0x1CE00` it is not — even though the four DGE writes and the four queued transposes
are byte-for-byte the same shapes (I checked), each block has exactly one writer and one reader
(§2), nothing else writes into `p64+0x1CC00..0x1D7FF` (no instruction field and no queued
`to_off` lands there), and the array geometry fields are identical (`rows=128, cols=64,
row_grp=col_grp=15`).

Leading hypotheses for iter-5, in order:
1. **`WZXY` is not a plain matrix transpose** of a 64x128 bf16 tile (it is a 4-D dimension
   permutation, `transpose_shape [1,1,64,128]`), and the block `PE[134]` reads has structure that
   makes the two orders agree — e.g. a symmetric or block-diagonal `(I - tril)`-style matrix,
   which is exactly what a chunked delta-rule kernel builds. **Settle it by dumping the bytes**:
   repoint one of the two unwaited tail output DGEs (`SP[64]`, `SP[65]`) at `p64+0x1CC00` and at
   `p0+0x4500` in two runs and compare the arrays offline. That is a ~10 minute experiment and it
   converts this whole lever from guesswork into arithmetic.
2. My `num_elem`/partition-count model for a transposed load is wrong and only coincidentally
   worked for `PE[134]`. Variants worth one device run each on `PE[142]` (keep its wait, so only
   correctness is in play): forward traversal (`step=[1,0,0]`, start `0x1CD00`); `num_elem` left
   at `[64,1,1]`; `num_active_rows/cols` swapped to 64/128.
3. If (1) is the answer, the lever is only worth `PE[134]`'s 0.45 us and should be marked closed.

Measured ceilings if it ever generalises (`prof3`, and these are the caps, not the gaps):
`PE[151]` ~1.3 us (its `S[34]>=32` wait is a pure 1.4 us transfer-26 round trip whose trigger
`SP[51]` itself waits on PE's own `S[5]>=63`); `PE[142]` ~0.45 us (its `MATMUL` partner's moving
operand `p0+0x5880` is not written until `DVE[105]@57.06`, so it cannot start before then);
`PE[161]` zero (its wait is already satisfied, 195 ns). Safety data already gathered:
PSUM `p128+0x1800`'s last reader before `PE[143]` is `DVE[86]@47.8`; PSUM `p128+0x1000`'s is
`DVE[101]@56.0`; `PE[152]`'s moving operand `p0+0x4480` is shared with `PE[150]`, so it is live.
Note also that `PE[150]` is a `MATMUL` with **no `LDWEIGHTS` of its own** (it reuses `PE[148]`'s
weights), so any edit that lets `PE[151]`'s weight load issue earlier must not overtake it.

### 10.5 The control that reframes all of §10: `PE[134]`'s weight operand is irrelevant
Took the shipped NEFF and moved `PE[134]`'s transposed load to the **neighbouring block**
(`p64+0x1CD00` instead of `0x1CC00`) — completely different data, same shape. `verify.py`:
**PASS, `max_abs_err` identical to the baseline to every digit.** So `PE[134]`/`PE[135]`'s matmul
result does not reach either output (dead, or masked downstream — a chunked delta-rule kernel
computes a diagonal-block correction and then masks it, which fits). Three consequences:

1. **The shipped edit is still correct and its gain is real**, but the correct description is:
   *`PE[134]`'s gate on queued transfer 20 was a 1.4-2.2 us round trip in front of a matmul whose
   operand does not matter; retargeting the gate to the DGE semaphore `S[43]>=16` (and reading
   that DGE's block, which is defined and stable from ~51 us) deletes the round trip from the
   critical path.* It is bit-identical across 3 runs and the whole tail moved ~1.0 us earlier.
   (The transposed-load encoding is incidental here — it reads defined data at a defined time,
   which is why I kept it rather than leaving the stale `0x4500` read in place.)
2. **`transpose_mode=ENABLED` is unproven on this kernel.** Every attempt on a site whose operand
   *does* matter (`PE[142]`, `PE[151]`) gives deterministic **NaN** — including with the timing
   held fixed and including forward traversal (`step=[1,0,0]`, start at the block base). NaN
   rather than wrong-but-finite values suggests the load reads out of range, i.e.
   `num_elem=[128,1,1]` with `num_active_cols=64` is probably an illegal combination: for a
   transposed load the partition count and the element count may both have to come from the
   array geometry, so a 64x128 source may simply not be loadable into a 128x64 array this way.
   The corpus only ever uses it on square 128x128 tiles (`num=[32,1,4]`, `step=[-1,0,-32]`,
   `rows=cols=128`), which is consistent with that reading.
3. **Any correctness claim about a PE edit must be paired with a sensitivity control.** A
   `verify.py` PASS after changing an operand proves nothing until you have shown the output
   *does* depend on that operand. One extra device run; it cost me an hour of wrong conclusions.

**For iter-5:** the realistic reading is that `PE[142]`/`PE[151]`'s ~1.75 us is **not** reachable
through `transpose_mode` with a non-square tile. Before spending more on it, check whether those
two sites' operands matter either (same control: point them at the neighbouring block) — if a
site's operand is irrelevant, its queued-transfer gate can be retargeted at the DGE semaphore the
way `PE[134]`'s was, which is the cheap version of the same win and needs no transpose at all.
That control is one run per site and is the highest-value next experiment in the whole kernel.

### 10.6 ...and the control is already done for the other two sites, which closes the lever
Changing `PE[142]`'s and `PE[151]`'s operand to the pre-transpose block **changed the outputs**
(NaN), which is exactly the sensitivity control: **their weight operands do matter**, so their
`S[30]>=32` / `S[34]>=32` waits are genuine data dependencies on the queued transposes and cannot
be retargeted at the DGE semaphore. Combined with 10.5(2) — `transpose_mode` does not work for a
64x128 source into a 128x64 array — **this lever is closed at the NEFF level** unless someone
works out the correct non-square transposed-load encoding (and the corpus has no example of one).

So the residual ~1.75 us behind transfers 22 and 26 is only reachable by making those two
transfers *cheaper or earlier*, not by removing them. Both are 16 KB, ~250 ns of fabric time and
~1.1-1.4 us of trigger->consumer overhead, and their triggers (`SP[43] <- S[5]>=58`,
`SP[51] <- S[5]>=63`) already fire within ~60 ns of PE's own matmul that gates them. That is the
same per-hop overhead wall as §7.

### 10.7 The generalisable next experiment
`PE[134]`'s win came from finding an instruction that *waits on a DMA it does not need*. Look for
more of them systematically: list every instruction on every engine whose wait is a queued-transfer
completion (`S[24..39] >= 16/32/48`), take its `evt_wait_time_ns` from the profile, and for each
one with a big idle wait run the one-run sensitivity control (point its operand somewhere else and
see whether the outputs move). In `prof3` the PE candidates are `PE[151]` (1556 ns, operand matters
-> closed), `PE[142]` (706, closed) and `PE[161]` (195, no slack); **the ACT and DVE sides are
unexamined** — `ACT[96]` (`S[24]>=32`), `ACT[103]` (`S[27]>=32`), `ACT[108]` (`S[33]>=32`),
`ACT[114]` (`S[37]>=32`), `ACT[117]` (`S[39]>=32`), `DVE[103]` (`S[29]>=32`), `DVE[117]`
(`S[33]>=32`), `DVE[75]`/`DVE[84]`, and `PE[166]` (`S[24]>=48`) all wait on a queued transfer and
none has been checked.

## 11. Where the idle time is in the **shipped** NEFF, and what it waits for (`scripts/idle.py`)

Per-engine gaps > 500 ns in `prof4`, with the wait of the instruction that was blocked. This is
the list to work from next time (times are `prof4`-raw; subtract ~12 us for iter-3's base):

| idle ns | slot | waits for | read as |
|---|---|---|---|
| 17099 | `POOL[20]` | `S[6]>=58` | Pool is idle anyway, off-path |
| **9726** | `PE[21]` | `S[29]>=16` (queued transfer 5) | the 32 KB scatter copy; iter-3 closed every lever |
| 9552 / 4575 / 2784 | `SP[23,26,24]` | `S[4]>=29/40`, `S[6]>=49` | SP waiting to trigger; off-path |
| 8319 | `SP[6]` | `S[4]>=10` | first var10 reload DGE, off-path (§5) |
| 6148 | `POOL[35]` | `S[21]`,`S[20]` | the two tail output stores (§5, ~240 ns available) |
| **3347** | `PE[136]` | `S[6]>=68` (DVE) | **the new biggest tail idle — see below** |
| 3785 / 3499 | `POOL[29,21]` | `S[6]>=80`, `S[4]>=42` | DGE hosts waiting for their source data |
| 2670 / 2137 | `ACT[95,110]` | `S[5]>=58/67` (PE) | the rigid PE<->ACT cycle |
| 2557 | `PE[134]` | `S[43]>=16` | this iteration's edit: was 6248 ns on transfer 20 |
| 2477 | `SP[64]` | `S[4]>=60` | tail output store, gated by ACT's last compute |

**`PE[136]`'s 3.3 us is the new top target.** Chain: `PE[136] <- S[6]>=68` (DVE) `<- DVE[99]/[100]
<- max(PE[135] via S[5]>=56, DVE[98] via S[66]>=16)`, and `S[66]` is the group-B staging DGE that
lands at ~54.8 us in the bandwidth-limited window of §2-§4. So after this iteration the critical
path runs *through the group-B DGE completion*, exactly where §3/§4 measured ~54 GB/s of fabric.

Two concrete follow-ups, in order:
1. **Retire queued transfer 20.** It is now **dead**: its only consumer was `PE[134]`, which no
   longer reads its output (`0x4500` is next written by `DVE[124]`@65.6 and next read by
   `PE[160]/[162]`@66.2, both after), and nothing waits `S[28]>=32` any more (`PE[19]` and
   `SP[35]` want `>=16`). Neutralising its trigger `SP[36]` (make it a bare `EVENT_SEMAPHORE`)
   removes 16 KB read + 16 KB written from the fabric *inside* the window that now gates
   `PE[136]`, plus 53 ns on SP. I did **not** try it: iter-3 §6 shows queued-transfer plumbing can
   wedge the device (`rmmod neuron`/`modprobe` to recover), and I was too close to the deadline to
   risk leaving the device wedged for the harness's own measurement. Expected ~0.2-0.3 us; do it
   early in an iteration, not late.
2. **Reduce the group-B staging traffic.** 4 x (32 KB read + 32 KB written) of pure fp32
   partition/offset redistribution for `DVE[99]/[110]/[120]/[129]`. Only `DVE[99]` is on the
   critical path; the other three have 2-10 us of slack (`DVE[107]`@57.7, `DVE[117]`@61.6,
   `DVE[128]`@66.8). If those three DGEs could be *delayed* out of the 51-55 us window (their
   semaphores are now per-reader after edit 2, so there is nothing structural stopping it — move
   the slots later in their hosts' streams), `DVE[99]`'s block would get the fabric to itself and
   `S[66]` could land ~1 us earlier. **This is the cheapest shot at the remaining hole and it
   needs no new semaphores — just three slot moves.**

### 11.1 REFUTED on the spot: moving a group-B DGE out of the window does not help
I tried §11 candidate 2 on one of the three: `SP[34]` (sem 23, 64 KB, read only by `DVE[120]`
at ~61.6 us) moved from SP slot 34 to just after the transfer-22 trigger, so its transfer leaves
the 51-55 us window entirely. `verify.py --runs 2` PASS bit-identical; the move did exactly what
it was designed to do (`S[23]` 54028 -> 58561) and **bought nothing**: metric 59.532 vs
59.291/59.361 (neutral to slightly worse), and the spacing of the transfers that stayed behind is
unchanged — `S[66] - S[43]` was 3047 ns before and 2987 ns after.

So the 49-55 us window is **not** aggregate-bandwidth-limited in a way that removing traffic
helps; each DGE transfer carries its own ~1.4-2.0 us trigger->semaphore latency and they already
overlap. Combined with §3 (private queues bought ~0.2-0.4) and iter-3 §4 (splitting one transfer
across rings shared one budget), the honest model is: **a DMA hop costs ~1.4 us of latency that
nothing in the NEFF can remove, and the only wins are removing hops or starting them earlier.**
Candidate 2 of §11 is therefore closed; candidate 1 (retire the now-dead transfer 20) is still
open but is worth only its ~50 ns of SP time if the window is latency-bound, so it is no longer
attractive either — which leaves `PE[136] <- S[6]>=68 <- DVE[99] <- S[66]` gated by a single
irreducible DGE hop, and the kernel close to what NEFF-level scheduling can do in that region.

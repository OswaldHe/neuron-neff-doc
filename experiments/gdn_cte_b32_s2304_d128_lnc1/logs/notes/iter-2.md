# iter-2 — the DGE host matters: Activation+SP beats both all-Activation and Pool+SP

## Result: 14501 us, **-1.88% vs baseline** (14779), **-1.8% vs the iter-1 NEFF** (14770.6)

| config: who issues the 2784 moved DGE slots | total_exec_time_us | note |
|---|---|---|
| 2784 Activation (= baseline, unedited) | 14778.98 | `baseline/profile_baseline.json` |
| 1392 Pool + 1392 SP (= iter-1) | **14770.56** | I re-measured it myself, 3 rep, spread 0.83% |
| **1392 Activation + 1392 SP (= what I shipped)** | **14501.3** (medians 14503.8 / 14501.4 / 14506.3 / 14501.3) | four independent 3-rep captures |
| 0 Activation + 2784 SP | 14869.41 | runs and verifies, but **+0.61% vs baseline** |

Twelve captures of the shipped NEFF: 14503.8 14502.4 14510.7 | 14501.4 14513.6 14500.9 |
14651.4 14506.3 14502.1 | 14496.6 14501.3 14504.4. Eleven of twelve lie in 14496.6..14513.6;
one is a +145 us outlier. The four medians agree within 5 us. The win is 3.7x the 0.5% claim
threshold (278 us vs 74 us), so it is not noise.
`verify.py --runs 3` passes, outputs **bit-identical across runs**, `max_abs_err`
identical to baseline to 4 digits (1.372e-04 / 6.841e-04), so the edit is behaviourally exact.

## What I shipped

A **partial revert of iter-1 plus a keep**: of iter-1's 2784 relocated
`PSEUDO_DMA_DIRECT2D` slots, the 1392 that went to **Pool are restored to Activation**
(byte-for-byte from `baseline/kernel.neff`, same slot indices, original waits, no S[72] gate);
the 1392 on **SP stay there** with the S[72] handshake.

`neff_diff` vs baseline: 32 of 34 members identical, 2 changed —
`Activation0.bin` (same size, 1392 slots are still `EVENT_SEMAPHORE` placeholders) and
`SP0.bin` (17364 -> 18756 slots). **`Pool0.bin` is bit-identical to the baseline.**

Mechanics (all mechanical, no new semaphore):
1. The 2784 Activation placeholders are `EVENT_SEMAPHORE` with
   `events_update_mode=SEM_INC_COMPLETE, events_update_idx=72`, in program order -> n = 1..2784.
   Each relocated DGE waits `S[72] >= n`, so n identifies its placeholder.
2. For the 1392 n's whose DGE sat on Pool: overwrite Activation slot `ph[n-1]` with the
   **baseline bytes of the same slot index** (iter-1 replaced in place, so indices align — I
   asserted the baseline slot decodes as `PSEUDO_DMA_DIRECT2D` and that its `semaphore`,
   `src_start_addr_addr_immediate` and `dst_start_addr_addr_immediate` equal the Pool copy's).
3. Delete those 1392 slots from `Pool0.bin` -> byte-identical to baseline (checked with `cmp`).
4. Renumber: the 1392 surviving placeholders are re-ranked 1..1392 in Activation program order
   and each SP-hosted DGE's `events_semaphore_value` is set to its new rank.
   `semaphores.py`: `S[72] 1392 inc / 1392 max wait`, nothing else moved.

Script: `scratch/hybrid.py` in this worktree (not preserved; the recipe above is complete).

## Why: the DGE host is a first-order choice, and the curve is convex

Measured per-engine DGE issue delay (time from the slot's wait clearing to it issuing) on the
iter-1 config: **Pool median 1157 ns (mean 2107, max 8984), SP median 595 ns (mean 917)**.
Pool is the worst host for two reasons:

* Pool's queue `qPoolDynamic` has the slowest DGE start latency (578 ns vs SP 146 ns vs
  Activation 17 ns — iter-1's table, still true).
* **Pool's in-order stream is the one that carries the critical spill chain.** Inserting 1392
  DGE slots there cost Pool 1241 us of `S[72]` stream-wait + 877 us of sequencer time, directly
  in front of the spill-reload DGEs that gate PE.

The three corners (2784/0/0, 1392/1392/0, 0/2784/0 on Act/SP/Pool) fit a convex parabola in
"slots on SP" with its minimum at **x = 1294, f = 14502 us** — i.e. the 50:50 Act/SP split I
shipped is within ~2 us of the optimum of that axis. **Do not spend budget re-tuning the ratio.**
The effect is load balancing of in-order sequencer slots across two engines, not per-slot cost:
a per-slot cost model cannot produce a minimum in the interior.

## Hypotheses REFUTED this iteration, with the proofs (do not repeat)

### 1. "Head-of-line blocking" measured as `start_ts - own_wait_satisfied` is MOSTLY FAKE on this NEFF
This is the most important finding, and it invalidates the iter-1 review's recommended edit.

A `PSEUDO_DMA_DIRECT2D` or `PSEUDO_DMATRIGGER` whose **own wait is only queue flow control**
(`S[8..23]`/`S[40..71]` for DGE, `S[24..39]` for the spill-reload ring) has its **real data
dependency carried by a preceding `EVENT_SEMAPHORE` barrier on the same engine**. The compiler
routinely splits one logical "wait for A and for B" over two consecutive slots because
`PSEUDO_DMATRIGGER` and `PSEUDO_DMA_DIRECT2D` have **no extended-wait field** (only
`EVENT_SEMAPHORE` does: 1206 of SP's 18756 slots use `events_extended_wait_*`).
So `start - own_wsat` is huge for those slots and **none of it is recoverable**.

Proof I ran, on the single biggest case (Pool slot 8618, DGE on S[18], `start - own_wsat`
= 97 us, gating a trigger that holds PE idle 15 us):

* 8618 reads a 256 KB region at `src_imm = 1152924816026632192`, pattern
  `step=[4096,1] num=[64,1] esz=4096` -> contiguous `[base, base+262144)`; it writes SBUF
  (`dst step=[262144,1]`), i.e. it is a spill **reload**.
* The four `EVENT_SEMAPHORE` barriers immediately in front of it (Pool slots 8611/8612/8615/8616,
  waiting `S[55]/S[53]/S[43]/S[41]`) resolve to DGE transfers whose **destinations are
  `base+512`, `base+1536`, `base+2560`, `base+3584`** — inside the exact region 8618 reads.

So hoisting it would read spill data before it is written. **The 1649 us of "recoverable PE idle"
I measured this way is a real data dependency, not a schedule artefact.**
Corollary, measured: splitting the critical Pool DGEs by their own wait,
**`data`-wait ones (S[3..6]) have 5.3 us of recoverable time in total and `flowctl`-wait ones
have 1648.7 us** — i.e. the compiler places every explicitly-gated DGE exactly where it becomes
ready, and all the apparent slack belongs to the implicitly-gated ones. A simulation of the Pool
stream confirms: hoisting only the data-wait DGEs gains **0.0 us**.

### 2. The iter-1 review's recommended edit (reorder head-of-line-blocked trigger WRITEs on SP) is worth far less than 536 us, for the same reason
I reproduced the measurement (437 -> I find 3549 triggers with `block > 300 ns`, 206 ms of
"block") and then checked what the blocking predecessor is. Example, the largest: SP slot 2874
waits `S[4] >= 2148` (Activation data) and its predecessor `EVENT_SEMAPHORE` waits
`S[35] >= 1264` (spill-reload **ring** flow control) — the pair is one logical trigger with two
waits. The dual pattern also occurs (trigger holds the ring wait, predecessor holds the data
wait: SP slot 16077 waits `S[29] >= 7088`, predecessor waits `S[6] >= 18579`).
**Hoisting either member of such a pair breaks a dependency or the ring flow control.**
The provably safe residue is tiny: same-semaphore-dominance swaps (T's wait on the same
semaphore with a value >= the predecessor's) = **12 slots, 0.9 us**.

### 3. Reverting iter-1's Pool insertions is worth 91 us of Pool stream time, not the 1241 us of S[72] wait
Simulating the Pool stream with the 1392 inserted slots deleted (semaphore times held fixed)
moves the critical DGEs earlier by only **90.9 us** in total; "hoist the critical DGE above the
immediately preceding run of inserted slots" gains **66.5 us**. The `S[72]` stream-wait overlaps
waits the stream would have had anyway. (The 275 us the revert actually bought therefore comes
mostly from the *queue* change Pool -> Activation and from Pool's position in the critical chain,
not from the sequencer slots.) Same result on SP: removing all 1392 inserted SP slots moves
triggers earlier by **268 us** summed over all 8285 triggers (mean 32 ns, median 0).

### 4. Replacing the `S[72]` gate on the SP-hosted DGEs with their ORIGINAL wait is unsafe
It would let them fire as soon as their data is ready (order within each semaphore is still
guaranteed, because all 174 transfers of a semaphore sit on one engine in their original relative
order). But their original waits are: **1086 of 1392 on `S[41..55]` = their own queue's ring flow
control**, 170 on `S[6]`, 129 on `S[4]`, 7 on none. So 78% are implicitly gated (refutation 1)
and dropping Activation program order would break them. Dead.

### 5. Pushing the SP-hosted DGEs later so the gate is already satisfied when the stream reaches them
Simulated: triggers 129.6 us **earlier** but 332.2 us **later**, and the moved transfers 3400 us
later. Net worse. Dead.

### 6. A 3-way split (a DVE or PE DGE host) is IMPOSSIBLE — settled on device
`def.json` has exactly **4** DMA queues, and each is `{type, num_queues: 16, owner, fabric_path,
semaphore_set[16]}`:

| queue | type | owner | semaphore_set |
|---|---|---|---|
| `qActDynamicHW` | dynamic | act | 40..55 |
| `qPoolDynamic` | dynamic | pool | 8..23 (+ `dynamic_dma.SB_scratch_partition_*`) |
| `qSPDynamicHW` | dynamic | sp | 56..71 |
| `qSPSpillReload0` | data | sp | 24..39 |

There is no dynamic queue for DVE or PE, so I tried adding one. Three device runs:

1. Add `qDveDynamic` (owner `dve`, semaphore_set 73..88) **and** move all 174 S[40] DGE slots from
   Activation to DVE behind a second `S[89]` handshake (same recipe as S[72]; S[89] is inside the
   Scalar epilogue clear range 54..104). `semaphores.py` clean (`S[89] 174 inc / 174 max wait`,
   `S[40]` now `dge:DVE`). -> **`NRT_INVALID in nrt_load_util`**.
2. Add `qDveDynamic` and **change nothing else** (no instruction edits at all).
   -> **`NRT_INVALID`**. So it is the queue table, not the DVE instructions.
3. Add a 5th queue with a **legal** owner (`qSPDynamicHW2`, owner `sp`), nothing else.
   -> **`NRT_INVALID`**. So it is not the owner string either.

The queue table is validated at load and a 5th entry is rejected; `grep` over all 34 package
members shows the queue names appear **only** in `def.json` and `SP0.json`, so there is no other
record to keep consistent. **Three DGE hosts are not reachable by a NEFF edit. Do not retry.**

### 7. All 2784 on SP does NOT hang (iter-1's capacity theory is partly wrong)
iter-1 found all 2784 on **Pool** hangs with `NRT_TIMEOUT` and concluded ~1400 extra transfers
per queue is the limit. All 2784 on **SP** (2464 native + 2784 = 5248 on `qSPDynamicHW`) **loads,
runs and verifies bit-identically** — it is just slower (14869). So the Pool hang is not a simple
per-queue outstanding-transfer limit; it is specific to `qPoolDynamic` (or to Pool's own 3136
native transfers: 3136+2784 = 5920). Treat the Pool limit as real, SP's as >= 5248.

## Where the time goes now (span 14514 us, hybrid config)

| engine | busy | idle | top waits (idle attributed to the wait of the slot that follows the gap) |
|---|---|---|---|
| PE / Tensor | 7067 | 11676 | **ring S[24..39] 6103 us**, S[6] DVE 2799, S[4] Act 2763 |
| DVE / Vector | 7161 | 8288 | S[5] PE 4147, S[4] Act 1771, dge S[56..71] 905 |
| Act / Scalar | 9177 | 7623 | S[5] PE 3418, S[6] DVE 1741, ring 873 |
| Sync / SP | 3424 | 11082 | S[4] 3270, S[6] 2596, S[5] 1684, **S[72] 1599** |
| Pool / GpSimd | 2905 | 12603 | S[6] 8838, S[4] 2157, dge S[40..55] 977 |

(These busy numbers are my own per-slot sums from `Instruction.parquet`; `profile.py` reports
somewhat lower figures — 8249/5674/3193/2744/2728 — so compare like with like.)
The serial cycle iter-1 and the review described is unchanged: **PE waits 6.1 ms on
`qSPSpillReload0`**, and the chain is
`Act/DVE compute -> DGE spill-save -> EVENT_SEMAPHORE barrier -> DGE spill-reload ->
PSEUDO_DMATRIGGER -> queued transpose -> PE`.

## What I would do next, in order

1. **Nothing on the reorder axis.** Between iter-1, the iter-1 review and this iteration, every
   schedule-level lever is now measured and closed: engine occupancy (-1331 us busy -> -18 us),
   trigger wait tightness (0 ns), ring flow control (<=280 us, never binds), Activation producer
   reorder (<=112 us), Pool DGE hoist (0 us for the explicitly-gated ones, and the rest is a
   proven real dependency), SP trigger hoist (0.9 us provably safe), DGE coalescing (impossible),
   DGE transpose conversion (rejected at load), DGE host ratio (at the optimum, +-2 us).
2. The third-host idea that the convexity suggested (worth maybe another 100-200 us) is **closed**
   by refutation 6 — a 5th `dma_queue` is rejected at load. The remaining `def.json` knobs are
   `fabric_path` (always `"main"`) and `qPoolDynamic`'s `dynamic_dma.SB_scratch_partition_*`
   pair, which only that queue carries; both unexplored.
3. The other untouched block is the **1391 us of fixed trigger -> DMA-start latency** on
   `qSPSpillReload0` (median 724 ns x ~1900 transfers on PE's path). Unexplored opcodes that
   might touch it: `PSEUDO_DMAREARM` (194), `PSEUDO_DMABARRIER` (195),
   `PSEUDO_DMASWAP_QUEUE_SET` (207), all "not in corpus".
4. **Methodology warning for whoever profiles next:** never rank candidates by
   `start_ts - (time its own wait cleared)`. On this NEFF that metric is dominated by slots whose
   own wait is queue flow control, whose real dependency is on the preceding slot. Classify every
   candidate by its own wait first (`S[3..6]` = explicit data dep, `S[8..23]`/`S[40..71]`/
   `S[24..39]` = flow control -> the dependency is elsewhere), and confirm with the
   src/dst addresses before believing any "recoverable" number.
5. Capture-to-capture drift on an identical binary is ~0.05%, but I saw one capture of the iter-1
   NEFF come in 120 us low (14651 in a set whose median was 14770, spread 0.83%). Always take the
   median of 3 and look at the spread; a single outlier low capture is not a win.

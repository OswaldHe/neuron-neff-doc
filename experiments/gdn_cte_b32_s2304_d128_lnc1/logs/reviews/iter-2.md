# Review of iter-2 — `kernel.neff` (gdn_cte b32 s2304 d128)

Checked against `baseline/kernel.neff`, `profile_latest.json` / `baseline/profile_baseline.json`,
`verify_latest.json`, `.autohelix/notes/iter-{1,2}.md`, plus my own device capture
(`prof/*.parquet`, 1 repeat: 14500.127 us) and a static re-decode of both NEFFs.

---

## 1. What changed in the NEFF, and does it match the notes?

`neff_diff`: 32 of 34 members identical; only `sg00/Activation0.bin` (same size, 1392 of 30789
slots differ) and `sg00/SP0.bin` (17364 → 18756 slots) changed. **`Pool0.bin` is byte-identical
to the baseline**, so iter-1's Pool insertions are fully reverted. That is exactly what the notes
claim.

I re-decoded both streams and verified the mechanics slot by slot:

| check | result |
|---|---|
| Activation slots changed | 1392, all `PSEUDO_DMA_DIRECT2D` → `EVENT_SEMAPHORE` |
| placeholders that preserve the original wait (`events_wait_mode`/`_idx`/`events_semaphore_value`) | 1392 / 1392 (0 deviations) |
| placeholders doing `SEM_INC_COMPLETE` on `events_update_idx=72` | 1392 / 1392 |
| SP slots inserted | 1392, all `PSEUDO_DMA_DIRECT2D`, each waiting `S[72] >= n` |
| SP DGE bodies vs the baseline Activation DGE they replace (all 64 bytes except 4..11) | **byte-identical, 0 mismatches** |
| ranks n | 1..1392, unique, non-decreasing in SP program order |
| completion semaphore / `sem_increment` | unchanged (`sem_increment=16`) |

The split is cleaner than the notes describe it: it is **per semaphore group**, not an arbitrary
half. All 174 transfers of each *odd* queue semaphore `S[41,43,…,55]` moved to SP; all 174 of each
*even* `S[40,42,…,54]` stayed on Activation (8 × 174 = 1392 each way). Keeping a whole
flow-control group on one engine is what preserves the per-semaphore ordering, so this grouping is
the reason the edit is sound — worth stating explicitly in the notes.

`semaphores.py` diff vs baseline is exactly two things: `S[41..55 odd]` sources change
`dge:Activation` → `dge:SP` (totals unchanged at 2784 each), and `S[72]` appears
(`max_wait 1392 / inc 1392 / 1392 waiters`). No "wait exceeds increments" anywhere.

Consistent with the notes. The one quantitative claim I would soften: the notes present the
Act/SP convexity fit as the explanation; the profile says the mechanism is **queue** choice, not
sequencer load (see §4) — Q0 `qScalarDynamicHW` is the fastest and least-loaded dynamic queue, so
splitting the transfers over Q0+Q2 beats putting them all on either one.

## 2. Is the latency change real?

Yes, clearly.

| | median | min..max | spread |
|---|---|---|---|
| baseline | 14778.976 | 14778.3..14779.6 | 0.009% |
| iter-2 | **14505.815** | 14499.1..14519.5 | 0.14% |

Δ = **−273.2 us = −1.85%**, i.e. ~13× the observed spread and 3.7× the 0.5% noise threshold. My
own independent capture landed at 14500.127 us, and the notes report 12 captures with 11 in
14496.6..14513.6. Engine busy moves the expected way: Activation 8943 → 8255 us (−688, the 1392
DGE slots at ~630 ns each), Sync 1871 → 2730 (+859). **Not noise.** The loop is not chasing noise
this iteration.

## 3. Is the edit safe beyond the verifier?

**Synchronisation: nothing was loosened.** Every moved DGE kept its original wait — the wait is
now evaluated by the Activation-side placeholder and forwarded through `S[72] >= n`. Because the
placeholders execute in Activation program order and each carries the original wait,
`S[72] >= n` is *strictly stronger* than the wait it replaces (original wait **and** all earlier
placeholders done). Rank order is monotone in SP program order, so the handshake cannot deadlock.
No wait was removed, no threshold reduced, no semaphore count unbalanced.

**Numerics: bit-exact.** `max_abs_err` is identical to the baseline to every printed digit
(1.3723596930503845e-04 / 6.8411976099014280e-04). Nothing moved toward the limit. Note for later
iterations, not for this one: `output_1` already sits at **86% of the 8e-4 limit in the baseline**,
so any future edit that changes numbers at all has almost no headroom there.

**The one real residual risk — ring accounting, not semaphores.** A DGE's queue is implied by the
issuing engine, but its completion semaphore is in the instruction. So the 1392 moved transfers now
execute on `qSPDynamicHW` (queue_idx 2) rings while their flow control is still accounted on
`qActDynamicHW`'s `S[41..55]`. Queue 2 therefore carries 2464 native + 1392 moved = 3856 DGE
transfers (308480 packets, 194 MB in my capture) whose outstanding count is bounded by *two
independent counters* instead of one. It works — 12+ runs, bit-exact — but the margin is empirical,
not structural, and this is precisely the failure mode that gave `NRT_TIMEOUT` when iter-1 put 2784
extra transfers on `qPoolDynamic`. Practical rule for the next iterations: **do not add further DGE
slots to the SP stream**, and do not treat "all 2784 on SP loaded fine" as headroom.

## 4. The most promising next edit

### Where the 14.51 ms actually goes

I built the real critical path (34521 nodes) by resolving, for every instruction, which of
(program-order predecessor, each `S[x] >= v` release from `SemaphoreUpdate.parquet`) cleared last,
then walking that blame edge back from the last instruction. It reconstructs the span exactly
(wait 7424 + own duration 8722 − overlap 1636 = 14510 us):

| class on the critical path | wait us | own dur us | total |
|---|---|---|---|
| on-path instruction execution + sequencer | 765 | 5954 | **6719** |
| **queued transpose on `qSyncSpillReload0`** (ring latency + transfer) | 3832 | 879 | **4711** |
| DGE completion `S[40..55]` (Act queue; spill-save → reload turnaround) | 965 | 100 | 1066 |
| spill-reload ring flow control `S[24..39]` | 706 | 215 | 921 |
| compute producer `S[5]` (PE) | 248 | 650 | 898 |
| compute producer `S[6]` (DVE) | 189 | 543 | 732 |
| **DGE completion `S[8..23]` (Pool queue)** | **558** | 20 | **578** |
| compute producer `S[4]` (Act) | 41 | 175 | 216 |
| `S[72]` handshake (this iteration) | 4 | 118 | 122 |
| DGE completion `S[56..71]` (SP queue) | 106 | 15 | 121 |

Busiest engine: Scalar 8420 us union-busy (58%) — `ACTIVATE` 3401, `TENSOR_SCALAR` 3310,
`CAST` 1192, `DMA_DIRECT2D` 880. All-engine-idle is only 1351 us (9.3%), so the kernel is a
cross-engine dependency chain, not an occupancy problem. That matches iter-1's refutation
(−1331 us of Activation busy bought −18 us), so **do not take compute off Activation**.

### The lever: `qPoolDynamic` is the only queue on the *software* dynamic-DMA path

This is the one knob with a measured cost attached that nobody has touched.

* `prof/DmaPacket.parquet`, `queue_type` per queue: queue 0 `qScalarDynamicHW` =
  `hardware_dynamic` (111360 packets, 79.6 MB), queue 2 `qSyncDynamicHW` = `hardware_dynamic`
  (308480, 194.5 MB), **queue 1 `qGpSimdDynamic` = `software_dynamic`** (240640, 97.3 MB).
* `def.json` → `dma_queue`: `qPoolDynamic` is the **only** one of the four queues carrying
  `"dynamic_dma": {"SB_scratch_partition_offset": 0, "SB_scratch_partition_size": 16384}`.
  The other three entries are `{type, num_queues, owner, fabric_path, semaphore_set}` only.
* DGE start latency I measured in *this* NEFF (instruction end → first packet on its queue,
  median): **Act Q0 20 ns, SP Q2 142 ns, Pool Q1 578 ns**. The Pool DGE *instruction* is also the
  most expensive of the three (656 ns median vs 547–624), consistent with the engine staging
  descriptors into that SBUF scratch partition. Q1 also moves the smallest packets
  (404 B/packet avg vs 714 on Q0), i.e. ~1.7× more packet overhead per byte.
* It is on the critical path: 158 path nodes wait **558 us** on `S[8..23]`, and 95 of them
  (435 us, median 1.7–5.7 us each) are the **Sync doorbell `WRITE`s for the transposes that gate
  PE** — i.e. the chain is `Pool-queue DGE reload lands → SP doorbell → 725 ns ring latency →
  transpose → PE`. Two serial DMA hops, and the first one is on the slow queue.

**The edit.** `def.json`, member `sg00/def.json`, key `dma_queue.qPoolDynamic`: delete the
`dynamic_dma` block so the entry has exactly the shape of `qActDynamicHW`/`qSPDynamicHW`. No
instruction slot changes, no semaphore rebalancing, `semaphores.py` unchanged. Chapter-14 lever:
this is 14.1/14.3 applied to the DMA path rather than to an engine — make the transfers on the
critical path start sooner, in the spirit of 14.3.2, with the DMA-queue mechanics of chapter 9.5 /
13 (r09 "re-point or reshape a queued DMA", Verified mechanics).

**Expected change:** 558 ns earlier start × 158 on-path events ≈ 90 us as the defensible floor,
plus the per-byte efficiency difference on 97 MB of Q1 traffic → realistic band **−100 to −300 us
(0.7–2%)**. Two outcomes are possible and both are cheap to learn: the queue table is validated at
load (iter-2 got `NRT_INVALID` for a 5th queue, a new owner, and an owner rename), so this may be
rejected immediately — one edit, one ~10 s load. If it loads, `verify.py` must stay bit-exact
(the data movement is identical; only the descriptor path changes).

**Fallback if it is rejected at load — a zero-net-occupancy host swap.** Move whole Pool semaphore
groups whose completions gate the path onto Activation with the already-proven `S[72]`-style
handshake (use a fresh `S[73]`), and move an equal number of *off-path* Activation DGE slots the
other way, so Activation's slot count stays at 1392 and the Q0/Q1 loads stay balanced — iter-2
measured that Activation past ~1392 DGE slots costs +278 us, so a one-way move is not available.
Concrete slots: `Pool0.bin` holds 3136 DGE slots (indices 1..7581), exactly 196 per completion
semaphore `S[8..23]`; the on-path wait is spread over all 16 groups, with `S[21]` (79 us),
`S[13]` (48), `S[18]` (47), `S[22]` (43) the largest. Swap those four groups (784 slots) against
784 Activation slots from the `S[40..54 even]` groups. Expected ≈ **−90 us (0.6%)** — at the edge
of the threshold, so it needs ≥5 repeats to call.

### What is closed, with the measurement that closes it

Everything else in the table above has either a proven-closed lever or is irreducible:

* **Queued transposes, 4711 us (32%)** — 1344 us of it is the fixed doorbell→data-start latency
  (median 725 ns, p25 718, **min 693** across all 8285 transfers: a hardware floor, and the ring
  is 83% idle with 0.24 mean transfers in flight, so it is not congestion). The only route around
  it, `dge_op = DMA_TRANSPOSE`, is settled-rejected at load twice (iter-1 refutation 7), and
  merging transfers is dead (every transfer has ≥1 waiter; 12 safe pairs = 9 us).
* **Ring flow control `S[24..39]`, 921 us** — loosening a flow-control wait is the forbidden class
  (ch. 14.2 "remove waits on an engine's own completion semaphore" = Verified negative) and risks
  exactly the ring overflow discussed in §3. Do not.
* **DGE host/ratio, 1066 + 578 + 121 us of completion waits** — the four measured corners
  (14779 / 14501 / 14771 / 14869) bracket the optimum; marginal cost of a slot is positive on both
  SP and Pool. Closed, apart from the software-vs-hardware queue path above.
* **Compute producer waits, 1908 us, and the 6719 us of on-path execution** — real dataflow; engine
  occupancy is demonstrably not the binding constraint.

If the `dynamic_dma` edit is rejected at load and the swap lands inside the noise band, I would
report the NEFF-level budget as exhausted: what is left is 6.7 ms of on-path compute, ~1.3 ms of
hardware DMA-start latency and ~1.9 ms of genuine producer latency, and the next real win needs the
kernel (fewer spill/transpose round-trips), not the NEFF.

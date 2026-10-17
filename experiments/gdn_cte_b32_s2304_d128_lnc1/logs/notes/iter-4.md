# iter-4 — a calibrated schedule simulator, and merging 993 redundant instruction slots

## Result: 14231 us, **-1.13% vs the iter-3 NEFF** (14393), **-3.71% vs the iter-0 baseline** (14779)

| NEFF | total_exec_time_us | captures |
|---|---|---|
| iter-3 NEFF = my starting point | **14392.9** (med 3) | 14387.1 14392.9 14509.7 |
| **+ 993 merged instruction slots — shipped** | **14231.0** (med 5) | 14214.4 14221.6 14226.3 14231.0 14233.7 14235.0 14235.3 14237.9 14239.0 14240.2 / 14635.5 (outlier) |

Three independent capture sets, medians **14231.0 (5 rep) / 14239.0 (3 rep) / 14235.3 (3 rep,
spread 0.05%)**. Ten of eleven captures lie in 14214..14240 and every one is below every capture
of the starting NEFF (14387..14510); the gap is ~155 us against a ~10 us cluster width.
`verify.py`: PASS, `max_abs_err` **identical to the baseline to every printed digit**
(1.372e-04 / 6.841e-04, 17% and 86% of the limit). `single_core.py`: PASS.
`semaphores.py` output is **byte-identical** to the starting NEFF — not one count moved.

**Two working tools are parked next to this file** (both re-run and checked today, so start from
them rather than rebuilding):
* `iter-4-sim.py` — the schedule simulator of §1 plus the §2 sweep, self-contained. Needs
  `work/` unpacked and `prof/` ingested from the *same* binary. Reproduces this NEFF's span to
  +0.79%.
* `iter-4-merge.py` — the shipped edit of §3, verbatim. Re-run against the iter-3 NEFF from
  `git show HEAD:kernel.neff` it reproduces **all five shipped streams byte for byte**; run
  against the shipped NEFF it is a no-op (every run is length 1), so it is safe to re-run after
  any future edit to pick up newly-adjacent runs.

`neff_diff`: 2 of 34 members changed. `Activation0.bin` 30789 -> 30069 slots,
`DVE0.bin` 29896 -> 29623 slots. `Pool0.bin`, `PE0.bin`, `SP0.bin`, `SP0.json`, `def.json` and
every other member byte-identical.

---

## 1. THE MOST USEFUL THING IN THIS FILE: a schedule simulator that predicts edits to ~1%

Do not pick a lever on this kernel by reasoning about engine busy time. Build this and measure.
It predicted **-143 us** for the edit I shipped; the device gave **-162 us**. Two independent
calibrations (iter-3 NEFF and the shipped NEFF) landed 0.94% and 0.58% above the observed span.
It evaluates any lever in ~40 s, which is ~20x cheaper than a device cycle, and it killed three
candidate edits before I wrote a line of patch code.

### The model

Nodes are instruction slots in program order per engine. Two timelines per engine, because
**PE pipelines and the other four engines do not**:

```
start_i      = max(t_issue_free[eng], latest time any of its waits was satisfied)
t_issue_free = start_i + issue_i      # when the engine can start slot i+1
end_i        = start_i + dur_i        # when its semaphore increment lands
```

* `dur_i` = observed `end_ts - start_ts` from `prof/Instruction.parquet` (`compiler_pc` == slot
  index; all five streams align exactly, SP has 32 duplicate rows — dedupe by slot).
* `issue_i` is the *sequencer* cost and must be calibrated separately: take
  `start_{i+1} - start_i` for pairs where slot i+1 was sequencer-bound (its ready time
  <= `end_i`), and the per-(engine,opcode) median of that sample otherwise.
  **Using `dur_i` as the serial cost overestimates the span by 23%** — it makes PE look like
  7583 us of work when its real sequencer occupancy is 1347 us (MATMUL issues in **2 ns** and
  LDWEIGHTS in 30 ns while their durations are ~213 ns).
* Semaphore supply: every slot with `events_update_mode != NONE` adds 1 to
  `events_update_idx` at `end_i`. DGE and queued transfers add 16 after a per-slot observed
  latency (below). Waits: `events_wait_idx >= events_semaphore_value`, plus
  `events_extended_wait_*` on EVENT_SEMAPHORE.
* Event loop: apply pending increments <= now; run every engine head whose waits are satisfied
  and whose `t_issue_free <= now`; else advance `now` to the earliest of (next pending
  increment, min feasible head start).

### Mapping a DMA completion to the slot that issued it (needed for the latencies)

This is the part that is not obvious and that `DmaPacketAggregated` alone cannot give you
(it only covers `S[24..39]`, the queued ring).

* `SemaphoreUpdate.parquet` is the key table: `timestamp`, `id`, `value` (absolute, post-update),
  `update_source` (`Scalar`/`Vector`/`Tensor`/`GpSimd`/`Sync`/`AXI`). **`id` is sometimes an int
  and sometimes the string `"S[6] (Vector)"` — parse both.** Build, per semaphore, the running
  maximum as `(value, time)` pairs; then `t_reach(sid, v)` is a `searchsorted`.
* **Each DGE semaphore is owned by exactly one engine and its AXI increments are exactly 16x
  its slot count**, so the n-th DGE slot naming S in program order completes at
  `t_reach(S, 16n)`. Verified for all 48 DGE semaphores:
  S[8..23] 196 slots on Pool, S[40..55] 174 (Activation even / SP odd), S[56..71] 154 on SP.
* Queued: `DmaPacketAggregated.block_id` is the `SP0.json` transfer id and equals the
  `PSEUDO_DMATRIGGER` program order; completion is `t_reach(S[24 + id%16], 16*(id//16+1))`.
* Latency stored on the node = `completion - end_i`, so an edit that moves the slot moves its
  transfer with it.

### Caveat
`issue_i` for `EVENT_SEMAPHORE` is contaminated: its `duration_ns` **includes** the semaphore
wait, so the model over-credits removing them (it claims 88% conversion and -805 us for DVE's
2736 EVENT_SEMAPHOREs — do not believe that row). Compute-opcode and DGE rows are sound.

## 2. The lever table, measured on the SHIPPED NEFF

"issue" = sequencer time the lever removes; "conv" = simulated gain / issue removed. These are
**upper bounds**: each zeroes the work with no cost at a destination and no back-pressure.
Numbers below are from a run whose simulated span was 14322 us; a second profile capture of the
same binary gave 14348 us and every row within 3 us, so the ranking is capture-independent.

| lever | n | issue | simulated gain | conv |
|---|---|---|---|---|
| Act ALL non-signalling compute | 7761 | 2054 us | **-634 us (-4.4%)** | 31% |
| **SP DGE, all** | 3856 | 2352 us | **-423 us (-3.0%)** | 18% |
| Act ACTIVATE non-signalling | 3139 | 872 us | **-382 us (-2.7%)** | 44% |
| SP DGE, native S[56..71] only | 2464 | 1501 us | -308 us (-2.2%) | 21% |
| Act TENSOR_SCALAR non-signalling | 2524 | 808 us | -279 us (-1.9%) | 34% |
| Act DGE S[40..54] | 1392 | 884 us | -180 us (-1.3%) | 20% |
| DVE all non-signalling compute | 3628 | 729 us | -134 us (-0.9%) | 18% |
| Pool DGE | 3136 | 1562 us | -130 us (-0.9%) | 8% |
| SP trigger WRITEs | 8285 | 394 us | -111 us (-0.8%) | 28% |
| **Act CAST non-signalling** | 1879 | 328 us | **-21 us (-0.1%)** | 6% |
| Act COPY non-signalling | 219 | 46 us | -1 us | 2% |

Per-engine, zeroing *all* issue cost: Activation -1526 us, DVE -1092, SP -571, Pool -103.
No engine is occupancy-bound (Act 56.9%, DVE 39.8%, DMA 38.0%, PE 22.4%, SP 19.4%, Pool 19.3%).

## 3. What I shipped: merge affine runs of identical instructions (ch. 14.3 lever 4)

**Find maximal runs of consecutive slots on one engine that are identical in every field except
the two base addresses, where the addresses form an arithmetic progression. Widen the first
slot's access pattern to cover the whole run and delete the rest.**

993 slots deleted: 720 Activation (697 `ACTIVATE`, 23 `COPY`) + 273 DVE (245 `COPY`, 28 `CAST`).
Nearly all are **single-element `ACTIVATE`s** — `num_elem=(1,1,1)`, `num_active_channels=64`,
~290 ns of sequencer time to move one element per partition. 157 us of Activation issue and
51 us of DVE issue removed; **the device gave back 162 us, i.e. ~78% of it, and at the margin
the Activation part converted at essentially 1:1.** Those slots were squarely on the critical
path even though the class average for `ACTIVATE` is 44%.

### Selection rules (all necessary)
1. Same opcode, and every decoded field equal except
   `{src,dst}_mem_pattern_start_addr_addr_immediate`, the event fields and the debug bits.
   **You must also ignore every `*_addr_reg_*` field** — they are a union view of the same
   address bytes and differ whenever the address does. Forgetting this finds 0 runs.
2. Followers have `events_update_mode = NONE` and **no wait at all**, so deleting them changes
   no semaphore count and crosses no ordering point. The head may carry the run's wait.
3. `accumulator_cmd == 'IDLE'` (an accumulating ACTIVATE would change its accumulator grouping).
4. `header_inst_word_len == 16` (= one 64-byte slot; **not** 1 — this bit me).
5. `src` and `dst` base deltas both constant across the subrun; split at any change of delta.

### The pattern rewrite (steps and counts are in **elements**, dim 0 innermost, ch. 5.4)
With k members and byte deltas `ds`, `dd`, element sizes `se`, `de`:
* **case 1** (`num_elem[0] == 1`): `num_elem[0] = k`, `step_elem[0] = ds/se` (resp. `dd/de`).
  697 of the ACTIVATEs. Requires `|step| <= 32767` (signed 16-bit) and `k <= 65535`.
* **case 2** (`step_elem[0] == 1` and the delta is exactly `num_elem[0]` elements):
  `num_elem[0] *= k` — a single longer contiguous run.
* **case 3** (dim 1 unused): `num_elem[1] = k`, `step_elem[1] = ds/se`. All 296 DVE merges.
* src dtype is `in_dtype` **or** `in0_in1_dtype_dtype_lo` **or** `in_bias_dtype_dtype_lo`
  depending on the struct; dst is always `out_dtype`.
* 13 runs were unexpressible (both dim 0 and dim 1 already in use) and were skipped.

Deleting slots is safe in this NEFF (iter-1: no branches, no `PSEUDO_EXTENSION`, no member
records stream length). `semaphores.py` must come out byte-identical — if it does not, rule 2
was violated.

### This lever is now exhausted, under provable safety
* Allowing **signalling followers** (replace them with an `EVENT_SEMAPHORE` carrying the
  increment instead of deleting): only **160** more slots, ~30 us. Not worth a device cycle.
* **Non-consecutive** merging (move members together past independent slots): the safe
  condition is *the window must contain no semaphore increment*, because a non-signalling slot's
  cross-engine ordering is carried only by a later increment on its engine. Under it, only
  **106** more slots (72 Act + 34 DVE) — 1908 Activation candidate windows are blocked by an
  increment. Without that condition it is 514 slots (~0.5%) but then correctness rests on a
  hand-rolled alias check against a single fixed input; I judged that a bad trade.
* Consecutive `EVENT_SEMAPHORE` pairs can in principle fold into one via
  `events_extended_wait_*` (exactly equivalent: the conjunction of two sequential waits).
  Only **343** qualify, all on Activation; SP's 123 runs of 4 and DVE's 256 runs of 4 all have
  two updating members, which would need `events_extended_sem_update_value` (unused anywhere in
  this NEFF, unverified). ~30 us. Parked.

## 4. Hypotheses REFUTED this iteration (with the evidence; do not repeat)

### 1. The iter-3 review's recommended edit — move non-signalling Activation CASTs to Pool — is dead, twice over
* **Simulated, idealised** (zero cost at the destination, no back-pressure):
  1879 CASTs, 328 us of issue, **-21 us (-0.14%)**. The review's "430 us ceiling (-3.0%)" is off
  by a factor of 20. Its premise (PE's S[4] waits are 2 counts apart, so Activation sequencer
  occupancy is on PE's path) is true but does not imply the time is recoverable — CAST slots
  specifically sit where Activation has slack. The *measurement* this reproduces is iter-1's
  refutation 1 (-1331 us of Activation work bought -18 us), which it predicts at -18.5 us on the
  iter-3 NEFF. Both numbers are right; the inference from them was wrong.
* **Simulated, realistically** (placeholder costs 90 ns, insert into Pool at the
  schedule-consistent point, Activation placeholder waits on the Pool copy's handshake):
  **+26 us, a regression.** Same for TENSOR_SCALAR to Pool at cost parity: **+15 us**. Only if
  Pool were ~1.8x *faster* would TENSOR_SCALAR relocation win (-101 us at scale 0.55).
* **Measured on device — Pool is slower, and for ACTIVATE it is impossible.** I inserted
  harmless probe instructions into `Pool0.bin` and read their cost out of the profile:

  | opcode | Pool, 1 elem | Pool, 64 elem | implied ns/elem | Activation, for comparison |
  |---|---|---|---|---|
  | `TENSOR_SCALAR_ARITH_OP` | 266 ns | 1152 ns | **14.1** | 278 + 0.83/elem |
  | `CAST` | 189 ns | 378 ns | 3.0 | 141 + 1.02/elem |
  | `COPY` | 193 ns | 383 ns | 3.0 | ~200 fixed |
  | `TENSOR_SCALAR_AFFINE_SELECT` (native) | — | 239 ns median | — | — |
  | `ACTIVATE` | **hangs** (status 1004, model stop timeout) | | | Activation-only |

  Pool's fixed cost is competitive but its per-element rate is **17x worse** than Activation's,
  and `ACTIVATE` is not executable there at all (it needs the activation table). DVE is the only
  engine that is *faster* than Activation at these opcodes (TENSOR_SCALAR 152 ns, CAST 206,
  COPY 209) and the simulated relocation to DVE is worth only -50 us, below the noise floor.
  **ch. 14.3 lever 3 (move work off the busy engine) is now closed on this kernel.**

### 2. `ACTIVATE` cannot be downgraded to a cheaper opcode
Activation `ACTIVATE` costs ~303 ns of sequencer time against `CAST`/`COPY`'s ~200 ns, so
replacing the 4357 instructions that look like plain copies would be worth ~435 us of issue.
It does not work: **`activation_func = 7` is `exp`, not identity** (`docs/generated/
activation_functions.md`: 1 identity, 7 exp_400p, 128 copy_1p; the profiler misnames 7 as
`SOFTPLUS`). The func-7 instructions carry `scale=1.0, bias=0.0` and are real `nl.exp` calls.
The func-128 (`copy`) ones carry `scale` != 1 (0.0883883 = 2^-3.5, or -1.0) or a
`POINTER_IMMEDIATE` scale, i.e. they are affine, and `COPY` has no scale/bias field.

### 3. Merging DGE slots is not reachable — the candidates are never adjacent
I tested this because §2 ranks it as the biggest remaining block (4.8 ms of sequencer time over
8384 slots at ~650 ns each). It fails on *placement*, not on the access pattern:

* **Of the 8383 adjacent DGE pairs in the three hosting streams, 8381 differ in their decoded
  key and 0 are mergeable** (SP 3855/3855 `diffkey`, Activation 1391/1391, Pool 3135/3135).
  Consecutive DGE slots round-robin over the 16 semaphores of their queue, so they are never the
  same transfer shape on the same semaphore.
* Same-semaphore DGE slots are **40 to 200 slots apart** (S[41] on SP: slots 21, 61, 185, 304,
  425, 535). Bringing two together is the reordering problem of §3 bullet 2, and here the safe
  condition can never hold: **every DGE slot increments its semaphore**, so any window spanning
  two of them crosses ordering points by construction. Their own `S[40..71]` ring flow-control
  waits also depend on the count.
* The access pattern is *not* the blocker for all of them, which corrects the scope of iter-1's
  refutation 3: the `(4,16)` spill shapes do have both dimensions in use
  (`src step [512, 262144] num [4,16]`), but the `(64,1)` shapes — 2464 on Pool, 400 on SP,
  240 on Activation — have **dim 1 free** and would accept a second dimension. It is purely
  that no two of them are adjacent.

So the 4.8 ms of DGE sequencer time is not attackable by merging. The only remaining route to it
is reducing the *number* of staged transfers, i.e. iter-3's unfinished save/reload fusion
(iter-3 §"the next step I got 90% of the way through"), which needs a legal way to retire a DGE
slot — and iter-3 showed shrinking one hangs, so it means deleting it and repaying its +16.

### 4. The 45,696-byte free SBUF run is GONE — iter-3 spent it
iter-3's notes advertise `[68736, 114432)` as free scratch in every partition. **iter-3's own
phase 2 then allocated all 11 of its 4096-byte windows** to the 640 staging vars. I wrote probe
output to `68736 + 8192` and got 114,688 NaNs. What is left is the 128 bytes at 57216 and
`[229368, 262144)`, which is past the runtime's reserved 0x37FF8. **Any future iteration that
needs scratch SBUF must re-derive the free map from the current streams, not from that note.**

## 5. What I would do next, in order

1. **Build the static SBUF alias checker, then finish the `ACTIVATE` merge.** This is the
   highest-value reachable item and hints.md already asks for the tool. The prize is the 1908
   Activation windows that §3 bullet 2 had to reject: **514 more slots (346 Act + 168 DVE), worth
   ~104 us (~0.7%)** at this iteration's measured 78% conversion. What blocks them is that the
   window crosses a semaphore increment, which can release another engine that writes the moved
   instruction's source — a real race, not a conservatism. The check that unblocks it is iter-3's
   intactness method, which the iter-3 review independently re-derived and confirmed: for every
   writer in the program (compute `dst`, DGE `dst`, queued `to`) take its `(start_ts, end_ts)`
   from the profile and ask whether it overlaps the moved instruction's range in partitions
   **and** bytes **and** in the time interval between its original and its new position. The
   schedule is static and every address is a literal, so one profile's order is the order of
   every execution. The missing piece is decoding DGE addresses (iter-3's `dge2.py` recipe:
   `PSEUDO_ADDR8` marker byte 0x10 = VAR, 0x20 = TBL, and the `DGETable0` list in `def.json`
   maps a table index to a var) plus the `SP0.json` descriptors. Budget the tool, not the edit:
   it also unlocks every future reordering lever, and `DmaPacketAggregated` only covers the 8285
   queued transfers, not the 8384 DGE ones, so the profile alone will not give it to you.
2. **Finish iter-3's save/reload fusion** (its §"the next step I got 90% of the way through").
   Now that compute relocation (§4.1), `ACTIVATE` downgrading (§4.2) and DGE merging (§4.3) are
   all closed, this is the only remaining route to the 4.8 ms of DGE sequencer time, and it
   removes a whole hop plus its completion from 319 places on the dependency chain. 319 groups
   are already analysed and the rewritten saves are proven to run on device; the one open
   question is retiring the reload slot (delete it and add its +16 from an `EVENT_SEMAPHORE`'s
   `events_extended_sem_update_value`, since shrinking it hangs). Note that this NEFF uses
   `events_extended_sem_update_value` **nowhere**, so probe it on 8 slots before 319.
3. **Do not** spend budget on: moving compute between engines (refutation 1 above, closed),
   `ACTIVATE` -> cheaper opcode (refutation 2), Activation/DVE/Pool/SP engine occupancy as a
   proxy for latency (iter-1 refutation 1 and the whole of §2 — use the simulator instead),
   DGE host ratios (iter-2, at the optimum), `def.json` DMA queues (iter-2 refutation 6,
   iter-3 refutation 1), DMA priority (iter-3 refutation 2), trigger/wait tightness (iter-1
   refutation 2), `dge_op = DMA_TRANSPOSE` (iter-1 refutation 7), queued-path SBUF sources
   (iter-3, +199 us).
4. The transpose lever (hints.md) is still open but the iter-3 review scoped it to the **689**
   transposes that feed a stationary operand exclusively (866 us of queue time, a few hundred µs
   for PE) because 4433 of the 8125 feed both operands. It needs a whole iteration and it is now
   ranked below item 1.

## 6. Tooling notes

* `scratch/` is not preserved, which is why §1 and §3 are parked as runnable scripts beside this
  file (see the top). The only script not parked is the Pool cost probe of §4.1: it builds a
  template instruction from an existing slot, clears its waits and updates, points both
  patterns at scratch SBUF, and inserts N copies into `Pool0.bin` in blocks; the profile then
  reports its cost per instruction. **It needs genuinely free SBUF (see §4.4) — I used iter-3's
  stale free window and got 114,688 NaNs. The timing is still valid when the data is wrong,
  which is how the §4.1 table was measured, but do not confuse that run with a candidate edit.**
* `isa.Instruction.set` takes a whole list for array fields. `header_inst_word_len` is **16**
  for a normal slot. `as_dict()` does not include the mnemonic — carry `ins.name` yourself.
* Capture noise on an identical binary is still ~1 in 6 captures landing +100..400 us high
  (14635 in a set whose other members were 14221..14239). Take the median of 5 and read the
  cluster. The 0.5% claim threshold is ~71 us at this span.
* `prof/SemaphoreUpdate.parquet` is the most under-used table in the profile — it is what makes
  §1 possible, and `Flow.parquet` is still not trustworthy (iter-1).

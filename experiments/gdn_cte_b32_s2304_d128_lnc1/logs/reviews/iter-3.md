# Review of iter-3 — SBUF-staged DGE reshapes

Verdict: **accept.** The edit is address-only, the semaphore model is byte-identical, the win is
real (−0.86% vs iter-2, −2.69% vs baseline), and the notes are accurate. One unenforced hazard
is worth recording. The main answer (§4) rejects both of the agent's proposed next steps and
names a different one, with the measurement that distinguishes them.

---

## 1. What actually changed

`neff_diff.py` HEAD(iter-2) → working tree: 3 of 34 members, 31 identical.

| member | slots | changed |
|---|---|---|
| `sg00/Activation0.bin` | 30789 | 1308 |
| `sg00/Pool0.bin` | 7584 | 160 |
| `sg00/SP0.bin` | 18756 | 3780 |

5248 slots, matching the notes exactly. Decoding both streams independently:

* **all 5248 are `PSEUDO_DMA_DIRECT2D` in both versions** — no slot inserted, deleted, retyped
  or reordered (`SP0.bin` is the same length in both; `PE0.bin`, `DVE0.bin`, all `*0.json`,
  `def.json` and the `.dbg` members are untouched);
* the only decoded fields that differ are `{src,dst}_start_addr_addr_*` and
  `{src,dst}_step_elem`. Per side: Activation 1308 `dst`; Pool 160 `src`; SP 1316 `dst` +
  2464 `src`;
* **zero** changes to `semaphore`, `sem_increment`, `events_wait_*`, `events_update_*`,
  `in_dtype`/`out_dtype`, `num_elem`, `elem_size`, `dge_op`, or any `dma_configs_*` field.

The address edit is exactly what the notes describe: a `PSEUDO_ADDR8` VAR address
(`0x1000000b00000200` = marker 0x10, var 11, offset 0x200) becomes
`0x1000000000012e80` (var 0 = SBUF, offset 0x12e80 = 77440), and the steps change from DRAM
strides (`[65536, 4096]`) to SBUF partition strides (`[4194304, 262144]` = 16 partitions ×
one partition). Every new offset lands in `[68736, 114432)`, the 45,696-byte free run the notes
claim. Notes and NEFF agree.

## 2. Is the latency change real

Yes, though it is only just past the bar.

| | median | samples | spread |
|---|---|---|---|
| baseline | 14778.976 | 14778.3 / 14779.0 / 14779.6 | 0.009% |
| iter-2 (start point) | 14505.815 | 14499.1 / 14505.8 / 14519.5 | 0.14% |
| **iter-3** | **14381.433** | 14378.2 / 14381.4 / 14386.8 | 0.06% |
| my own 4th capture | 14369.015 | — | — |

Every iter-3 sample (14378–14387, plus my independent 14369) is below every iter-2 sample
(14499–14519). The separation is 112–150 us against a 0.06–0.14% spread (≈9–20 us) — about
10× the noise. **−0.86% vs iter-2, −2.69% cumulative.** Not noise, but 0.86% is close enough to
the 0.5% threshold that the agent was right to take 11 captures.

The clock is corroborated by counters that do not depend on it: `hbm_read_bytes` 217.5 → 66.5 MB
and `hbm_write_bytes` 181.4 → 30.4 MB, while `dma_transfer_total_bytes` (121,241,600) and
`dma_transfer_count` (8285) are *unchanged* — precisely the signature of re-pointing DGE
addresses from DRAM into SBUF without touching the queued path. The notes' own honest result
(the last 91 MB of removed DRAM writes bought 0 us) is the right reading: the win is shorter
save→reload completion chains, not bandwidth.

## 3. Safety beyond the verifier

**Synchronisation: untouched, verified.** `semaphores.py` on unpacked copies of the iter-2 NEFF
and this one is **byte-identical** — same 69 semaphores, same `max_wait`, `total_inc` and waiter
counts. No wait was removed or reduced. (Against the *baseline*, the only differences are
iter-1/2's: S[41..55] odd entries re-owned `dge:Activation` → `dge:SP`, and the new
`S[72] 1392/1392/1392 {Activation:EVENT_SEMAPHORE}` handshake. Nothing new this iteration.)

**Error: no headroom consumed.** `max_abs_err` is identical to the baseline to every printed
digit across all four iterations — 1.372360e-04 and 6.841198e-04, cosine identical to 9 dp.
`output_1` sits at 86% of the 8e-4 limit, but that is the *compiler's own* bf16 rounding, not
something this iteration moved.

**What the verifier cannot catch is aliasing, not synchronisation.** I re-derived the free-window
claim from scratch and it holds:

* 164,849 compute `mem_pattern` references across all five streams: **zero** inside
  `[68736, 114432)`. Nearest access below ends at ≤68736; nearest above starts at exactly 114432.
* unedited DGE slots: zero inside; nearest SBUF offsets 63488 below, 114688 above.
* 8285 queued descriptors in all five `*0.json`: zero overlapping; highest SB end below the
  window 66688, lowest start above 114432.

So the window is genuinely unallocated and nothing is being clobbered.

**The residual hazard** is window *reuse*. All 11 windows are used, each by 270–339 edited
references — roughly 58 of the 640 staging live ranges per 4096-B window — and the users of one
window span three engine streams (window 0: 133 Activation, 15 Pool, 191 SP). Ordering between
successive users of the same window rests entirely on greedy interval colouring of **one
profiled timeline**; no semaphore edge was added between them. The notes' justification ("the
schedule is static, so one profile's order is the order of every execution") is true for
instruction *issue* order within one engine, but the thing that decides when a window is free is
DMA *completion* order across three streams and 16 DGE rings, which is not statically ordered.
The claimed margin (≤3 live ranges at once, median live range 6.8 us, 11 windows) is probably
ample, but it is unmeasured and unenforced — and it will silently break if a later iteration
changes DGE hosting or adds DMA pressure. **Worth one cheap follow-up:** record the minimum
inter-user time gap per window in `notes/`, and allocate windows round-robin by first-use order
instead of greedily, so reuse distance is maximised rather than minimised.

## 4. The next edit

### Where the time is (fresh profile, span 14383.6 us)

No engine is occupancy-bound: Scalar 58.5%, Vector 45.4%, DMA 37.5%, Sync 24.2%, PE 22.8%,
Pool 19.9%. All-engine-idle is only 1285 us (8.9%) spread over 14911 gaps with p50 27 ns and
max 3.3 us — there is no stall to remove. Per-engine idle, blamed on the wait of the instruction
that ends each gap:

| engine | idle | biggest causes |
|---|---|---|
| **Tensor (PE)** | 11101 us | **6746 queued:qSyncSpillReload0**, 2266 Vector, 2084 Scalar |
| Vector | 7850 us | 4017 Tensor, 1684 Scalar, 779 dge:Sync |
| Scalar | 5970 us | 3247 Tensor, 1042 Vector, 865 queue |

The queue (`DmaPacketAggregated`): 8285 transfers, trigger→start **6824 us** total (median
725 ns, p10 713 — a hardware pipeline floor), transfer time 3438 us, union-busy 2398 us at
concurrency 1.43. Latency-serialised, as iter-1 found.

### Do not spend the iteration on the transpose lever (hints.md / the notes' step 1)

Two measurements kill most of it:

* **`MATMUL` has no `transpose_mode` field** (field enumeration over the PE stream); only
  `LDWEIGHTS` does. PE contracts the partition axis of *both* operands.
* Classifying all 8125 transposes by which PE operand role reads their destination:
  **4433 feed both the stationary and the moving operand** — e.g. transfer 0 writes SB 0x4200
  and is read by `LDWEIGHTS` slot 7 (`0x427e`, backwards) *and* by `MATMUL` slot 8 (`0x4200`);
  ~2000 feed only compute engines (`ACTIVATE`/`TENSOR_SCALAR`, which cannot cross partitions at
  all); and only **689** feed a stationary operand exclusively with no other consumer. Those 689
  carry 596 us of trigger→start + 270 us of transfer = **866 us of queue time, 8.5% of the
  transposes**. PE's share of that is a few hundred µs at best.

One correction to the notes while here: the stated blocker ("the reversal cannot be pushed
through `transpose_mode`") is not the blocker. `transpose_mode = ENABLED` coexists with the
reversed read in this NEFF — all 1152 ENABLED `LDWEIGHTS` have *exactly* the DISABLED pattern,
`step_elem=(-1,0,0) num_elem=(64,1,1)`. What distinguishes them is **dtype**: all 1152 ENABLED
are `in_dtype=FP32`, rows=64 cols=64, and they wait on **S[4] (Scalar)** — they read a tile
Activation just wrote, with no DMA transpose at all. All 3960 transpose-consuming `LDWEIGHTS`
are `BFLOAT16`. So the real open question is whether ENABLED is legal for bf16 (a 1-byte probe
answers it) — but even "yes" caps the lever at those 689 transfers.

### Also do not spend it on reordering (ch. 14.3 lever 2)

I measured its ceiling: for every PE gap whose releasing producer ran back-to-back inside a busy
run on its own engine, min(gap, run-before) sums to **2676 us** (Scalar 1623 us / 2214 instances,
Vector 1053 us / 2757), concentrated in 832 Scalar `ACTIVATE`, 2119 Vector `CAST`, 961 Scalar
`CAST`. That looks like 18.6% of the span, and it is a mirage. The largest single site:
Activation slot **11595** (`ACTIVATE`, `wait S[0]>=0` — no dependency at all,
`upd=SEM_INC_COMPLETE@4`) gates PE slot **13174** (`LDWEIGHTS S[4]>=5209, fp32@0x477c`) for
8204 ns, sitting 23 slots / 8.5 us deep in a Scalar run behind eight `num_elem=[1,1,1]`,
`upd=NONE` ACTIVATEs it provably does not alias. Hoisting it is legal and free. It buys **zero**:
PE's *next* wait, `S[4]>=5211`, is satisfied at t=5425317 — 1.1 us *after* the original stall
ended. **PE waits on 4506 distinct cumulative S[4] values spaced a median of 2 counts apart**, so
PE needs the count, not that instruction. This is the mechanism behind iter-1's refutation 1 and
it should be written into `notes/` as closed.

### THE EDIT: move non-signalling Activation compute to Pool (ch. 14.3 lever 3)

The same fact that kills reordering points at the right lever. Because PE's S[4] waits are spaced
2 counts apart, **30781 of the 30789 Activation slots lie between two counts PE needs** — the
in-order Activation *sequencer* is on PE's critical path at 2-instruction granularity. And
Activation instructions are 72–78% fixed overhead (weighted fits over the observed element
counts: `ACTIVATE` 291 ns + 0.85 ns/elem, `TENSOR_SCALAR` 278 + 0.83, `CAST` 141 + 1.02).

Of the 7343 us of Activation sequencer time in those intervals, **3758 us is on slots that never
increment S[4]** — PE never waits for them, yet they hold the sequencer between two counts it
does. The precisely delimited, zero-renumbering subset of that:

| opcode | slots | Activation time | example slots in `sg00/Activation0.bin` |
|---|---|---|---|
| `TENSOR_SCALAR_ARITH_OP` | 2524 | 949.9 us | 10, 14, 25, 40, 44, 64, 68, 72, … |
| `CAST` | 1879 | 515.6 us | 82, 84, 98, 103, 104, 108, 119, 121, … |
| `COPY` | 219 | 65.4 us | 2, 87, 118, 128, 314, 323, 331, 545, … |
| | **4622** | **1530.9 us** | |

Selection criteria, all three necessary:
1. `events_update_mode = NONE` (does not increment S[4]) — iter-1's zero-renumbering
   precondition: no wait anywhere names them by count;
2. a **real** semaphore wait, not `S[0]>=0` (1681 CAST and 1277 TENSOR_SCALAR wait on
   S[3..7] engine completions, 1005 on `dge:Pool`, 237 on the queue, 198 on `dge:Sync`) — so the
   relocated copy starts when its data lands, not when the Activation sequencer arrives;
3. an opcode ch. 14.3 #3 lists for other engines. `ACTIVATE` is excluded — Activation-only
   (activation table), which is why the 3139 no-S[4] ACTIVATEs (1132 us) are not candidates.

**Host: Pool (GpSimd)** — 19.9% busy, 11516 us idle in 6450 gaps. **Start with the 1879 CASTs
only** (516 us), taking Pool to ~23.5%: comfortably under the 913 us of added Pool work that made
iter-1's Pool-hosted-DGE variant lose, and iter-2's finding that Pool's in-order stream carries
the S[3] chain that gates Vector is the reason to stage it rather than do all 4622 at once.

**Mechanics** — iter-1's recipe, already verified on device 2784 times bit-exactly:
* in `Activation0.bin`, rewrite each of the 1879 slots to `EVENT_SEMAPHORE` keeping the original
  wait and adding a wait on a fresh `S[73] >= k` (S[73]..S[255] are free);
* insert the real instruction into `Pool0.bin` carrying the **original wait verbatim** plus
  `events_update_mode = SEM_INC_COMPLETE, events_update_idx = 73`;
* insertion point schedule-consistent: after the last Pool slot whose `end_ts <=` the moved
  slot's original `start_ts` (`prof/Instruction.parquet`, `compiler_pc` == slot index).

**No semaphore count renumbering anywhere.** The moved slots increment neither S[4] (so no S[4]
wait names them) nor S[3] (so Pool's 2403 S[3] waiters are untouched), and the Activation
placeholder preserves Activation program order exactly — anything later in that stream which
reads the result still runs after the placeholder, which runs after the Pool copy.
`semaphores.py` should show exactly one new row (S[73], n increments, n waiters) with every
other row unchanged; if any other row moves, the edit is wrong.

**Expected change.** Each relocated slot drops ~274 ns of Activation sequencer time (CAST median
on Activation) to ~40 ns (Activation `EVENT_SEMAPHORE` median): 1879 × 234 ns ≈ **430 us ceiling
(−3.0%)** for the CAST batch; all 4622 ≈ 1300 us (−9%). Discount it: PE also waits 6746 us on the
queue and 2266 us on Vector, so part of each advance is absorbed. Realistic expectation
**−1% to −4%**, i.e. 1–5× any single previous iteration.

**Why iter-1's refutation 1 does not apply.** iter-1 removed *DGE* slots from Activation and got
−18 us — but each one was replaced by an `EVENT_SEMAPHORE` that still occupied the stream *and*
the moved transfer then paid Pool's 578 ns queue start latency instead of Activation's 17 ns.
iter-2 measured that trade directly (Activation+SP beat Pool+SP by 270 us). A relocated *compute*
instruction has **no queue-start-latency term at all**, so the trade is one-sided. That
difference is the whole hypothesis, and it is what the experiment tests.

**Falsification first, cheaply.** Do 200 of the 1879 CASTs (≈55 us of Activation time, ≈45 us
expected ceiling) before writing the full pass. If the measured change is under ~10 us, the
hypothesis "Activation sequencer occupancy between PE-needed S[4] counts is on the critical path"
is dead, and with it ch. 14.3 levers 3 and 4 on this kernel — which, combined with the transpose
scoping above, would justify recording that the remaining 14.4 ms is irreducible at the NEFF
level. Either outcome is a result worth having; do not skip straight to all 4622.

# iter-1 — where the time actually goes (and two refuted hypotheses)

Baseline I measured myself (3 repeats, unedited NEFF): **14785.5 us**
(range 14784.2 / 14785.5 / 14789.7). `baseline/profile_baseline.json` says 14778.98.
Inter-capture offset on an *identical* binary is ~+0.03..0.05%, as iter-0's review warned.

## What I shipped

Moved **all 2784 `PSEUDO_DMA_DIRECT2D` (DGE) slots off the Activation stream**, split
1392 to `Pool0.bin` and 1392 to `SP0.bin`. Measured three times: **14767.8 (3 rep),
14761.5 (5 rep), 14768.6 (3 rep)** vs baseline **14785.5 (3 rep)** -> about **-0.1 .. -0.16%**.
Activation busy 8942 -> 7612 us (**-14.9%**). `verify.py --runs 3` passes with
**bit-identical outputs across runs** and `max_abs_err` identical to baseline to 4 digits
(1.372e-04 / 6.841e-04), so the edit is behaviourally exact and race-free.
Below the 0.5% claim threshold, so **I do not claim a win** -- but all 5 repeats of the
edited NEFF came in below the baseline's minimum (14784.2), so it is not a regression.
`neff_diff`: 3 of 34 members changed (`Activation0.bin` same size, 2784 slots become
`EVENT_SEMAPHORE`; `Pool0.bin` 7584->8976 slots; `SP0.bin` 17364->18756 slots).
**The important result is the negative one: removing 1331 us of Activation work bought
~18 us. Activation occupancy is NOT the constraint.** Script: `scratch/move_dge.py` in
the iter-1 worktree (not preserved) — the recipe is written out below.

### The recipe (reusable, verified on device)
A DGE slot can be relocated to another engine with **zero semaphore renumbering**:
1. All 2784 Activation DGE slots have `events_update_mode=NONE` — they increment no
   engine semaphore, so deleting them from Activation touches no S[4] wait, and inserting
   them into Pool/SP touches no S[3] wait.
2. **In place of each one, put an `EVENT_SEMAPHORE` that keeps the original wait verbatim
   and adds `events_update_mode=SEM_INC_COMPLETE, events_update_idx=72`** (S[72] was free;
   S[3]..S[71] are all used, and 72 falls inside the runtime's Scalar epilogue clear range
   54-104, so it is zeroed between the 5 executions profile.py does).
3. **The moved slot's own wait becomes `S[72] >= n`** (n = 1-based placeholder index).
   The placeholder completes only after the original wait *and* all earlier Activation
   instructions (same-engine program order), so one wait field reproduces both the data
   dependency and Activation program order. This is what makes it bit-exact.
4. `semaphore` / `sem_increment` (40..55 / +16) are carried over **unchanged** — so the
   ~2435 consumer waits on S[40..55] need no re-pointing.
5. Insertion point in the host stream: **schedule-consistent** — insert after the last
   host slot whose `end_ts` <= the moved slot's original `start_ts` (from
   `prof/Instruction.parquet`; `compiler_pc` == slot index, no GID-load quirk in this NEFF).
   That provably cannot deadlock for one insertion and keeps both sequences time-ordered.

### VERIFIED facts from this
* **A DGE instruction's `semaphore` field is honoured from any engine's queue.** A DGE
  issued from Pool/SP that names S[43] (Activation's `qActDynamicHW` set) still increments
  S[43]. Tested with an 8-slot probe first, then 2784. This was the open question in
  iter-0's review — it is now answered yes.
* Deleting/inserting slots is safe: no stream in this NEFF has any branch
  (`COMPARE_BRANCH` exists only once on SP, in the epilogue), no `PSEUDO_EXTENSION` slots,
  and no metadata records stream length (`kelf-0.json`/`def.json`/`*0.json` carry no counts).
  Stale `.dbg` debug info does not bother the runtime or neuron-explorer.
* **All 2784 -> Pool only HANGS** (`NRT_TIMEOUT`, model stop timeout). 1392/1392 across
  Pool+SP is fine. Most likely the DGE queue's outstanding-transfer capacity: the moved
  transfers keep their own S[40..55] flow control, which does not coordinate with the host
  queue's native S[8..23] flow control, so the host ring can be asked to hold ~2x the
  outstanding transfers it was built for. **Do not put more than ~1400 extra DGE transfers
  on one queue.**

## Where the time actually goes (span 14800 us)

| engine | busy | idle | top thing it waits on |
|---|---|---|---|
| Scalar/Act | 9057 us (61%) | 5735 us / 22871 gaps | S[5] PE 3127 us, S[6] DVE 1025, q:Spill 817 |
| Vector/DVE | 6729 us (45%) | 8067 us | S[5] PE 4054 us, S[4] Act 1681, dge:SP 975 |
| Tensor/PE | 3288 us (22%) | 11512 us | **q:SPSpillReload 7006 us**, DVE 2423, Act 2079 |
| GpSimd/Pool | 2905 us | 11888 us | — |
| Sync/SP | 2776 us | 12016 us | S[4] Act 3890, S[6] DVE 2709, S[5] PE 1661 (all before triggers) |

Time with >=1 engine busy: 13539 us (91.5%); **all-engines-idle only 1261 us**. No engine is
near 100%: this kernel is **dependency-bound**, not throughput-bound.

**PE is the hub and it waits 7006 us (47% of the span) on the `qSPSpillReload0` queued DMA**
(8285 transfers, 8125 of them hardware DMA *transposes*, SB->SB, 121 MB). Decomposition of
those 7006 us (2550 PE gaps resolved to a specific transfer via S[24+id%16] >= 16*(id/16+1)):

| component | us | per transfer |
|---|---|---|
| PE ready, producer not finished yet (true data dep) | 3209 | 1258 ns |
| **trigger -> DMA start (fixed ring-fetch latency)** | **2002** | **785 ns** |
| the transfer itself | 1208 | 474 ns |

The queue is only 13% busy with mean concurrency 0.64 — it is **latency-serialised, not
bandwidth-limited** (398 MB total HBM traffic in 14.8 ms = 27 GB/s, nowhere near the limit).

## The one big unexploited lever: DGE start latency vs queued start latency

Measured (first DMA packet on the queue after the issuing instruction / after the trigger):

| path | start latency (median) |
|---|---|
| `qScalarDynamicHW` (DGE from Activation) | **17 ns** |
| `qSyncDynamicHW` (DGE from SP) | **146 ns** |
| `qGpSimdDynamic` (DGE from Pool) | 578 ns |
| `qSPSpillReload0` (queued, trigger -> start) | **725 ns** (p25 718 — a fixed pipeline latency) |

So the DGE path starts **~600 ns sooner** than the queued ring path. Converting the
spill-reload transposes that gate PE to DGE DMA would recover much of the 2002 us (**~13% of
the span**). What stands in the way, all established this iteration:

* The transfers are hardware transposes. A 2-D DGE pattern *can* express a transpose only
  with `elem_size=2`, which explodes to 8192 descriptors per tile — useless. The real route
  is **`dge_op = DMA_TRANSPOSE` (DGE_OPCODE value 2), struct
  `v3.NEURON_ISA_TPB_DMA_DIRECT2D_XPOSE_STRUCT`** (plus `DMA_GATHER_TRANSPOSE` = 3). Both are
  marked *"not in corpus"* in `neff_doc/artifacts/reference/opcode_struct_v3.json`, so the
  field semantics are unverified — that is the research task. `nisa.dma_transpose` in NKI
  always compiles to the queued trigger path, which is why no example exists.
* You cannot simply stop triggering a converted transfer: **8274 of the SP triggers wait on
  S[24..39] as ring flow control** (transfer k waits for k-16), so an untriggered transfer
  hangs the chain. Workaround: keep the queued transfer but shrink its descriptor to a
  1-byte dummy copy so it still signals, move the real data via a DGE instruction with a
  fresh semaphore, and re-point that transfer's consumers (median 2 waiters, see below).
* Free semaphores for this: **S[72]..S[255]** (S[3]..S[71] are all used; S[72] is now used by
  my handshake). Each can host many transfers at 16, 32, 48, ...

## Hypotheses REFUTED this iteration (do not repeat)

1. **"Activation is the busiest engine, so take work off it."** (iter-0 review's top
   recommendation.) Done in full: -1331 us of Activation busy time, -18 us of latency.
   Activation has >=1.3 ms of slack. Same conclusion applies to DVE (45% busy) and to
   Pool/SP (<20%). *Engine occupancy is not the metric on this kernel.*
2. **"The compiler over-synchronises; tighten the DMA trigger waits."** Measured properly:
   for the 5025 SP triggers whose predecessors are all ordinary compute instructions, I
   computed the earliest wait value that still covers every `FLOW_/ANTI_/OUTPUT_DEPENDENCE`
   predecessor (mapping each predecessor to the next increment of its engine's completion
   semaphore, which is program-order-robust). **The gain is exactly 0 ns for every single
   one** — the compiler's wait values are already precisely the true dataflow requirement.
   The other 3189 triggers wait on S[24..39] ring flow control and must not be touched.
3. **Coalescing the 2304 `(4,16)x512` Activation spill-save DGE slots (iter-0 review's
   fallback, lever 4).** Impossible: within a group of 8 the src->dst index map is a genuine
   3-D permutation (`dst = 512g + 4096b + 65536a`, `src = 4194304g + 262144b + 512a`) and
   `src_num_elem`/`dst_num_elem` have only **2** dimensions. No pair of the three indices
   merges into one affine dimension. Also pointless: DGE cost is ~**fixed per instruction**,
   not per byte — `(64,1)x4096` (256 KB) costs 547 ns and `(4,16)x512` (32 KB) costs 606 ns.
   Fit on Pool: `~630 ns + 0.5 ns x ndesc`.
4. **Merging queued spill-reload transfers to save trigger latency.** Every one of the 8285
   transfers has >=1 real waiter (histogram: 285 have 1, 5826 have 2, 1767 have 3, ...;
   **zero** have none), so merging always delays a consumer.
5. **Head-of-line blocking on SP delaying the triggers.** All 8321 trigger `WRITE`s have a
   non-zero idle gap before them — each issues as soon as its own wait clears. SP program
   order is not a constraint.
6. `dma_configs_priority_class` is 0 and `dma_configs_dma_transpose_2d_tile` is 0 on all
   8384 DGE slots, and the latter is a **1-bit** field — no free throughput knob there.
7. **Converting the queued spill-reload transposes to the (600 ns faster) DGE path is
   IMPOSSIBLE — settled on device, twice.** (a) Setting
   `dma_configs_dma_transpose_2d_tile = 1` on an Activation spill-save DGE slot (slot 140,
   a 1-byte diff) ran fine and produced the **same** `max_abs_err` on both outputs: the bit
   alone does not alter data movement. (b) Setting `dge_op = 2`
   (`v3.NEURON_ISA_TPB_DGE_OPCODE_DMA_TRANSPOSE`, from `isa_dwarf.json`) on the same slot is
   **rejected at load**: `Failure: NRT_INVALID in nrt_load_util`, `Failed to load nrt model`.
   A pseudo slot cannot carry it anyway: the transpose parameters `tile_src_rows`,
   `tile_src_cols`, `tile_src_row_step` exist only in the post-rewrite
   `v3.NEURON_ISA_TPB_DMA_DIRECT2D_XPOSE_STRUCT`, at byte offsets 60/61/36 that collide with
   `dst_elem_size`/`in_dtype`/`src_elem_size` in `PSEUDO_DMA_DIRECT2D`. So only
   `dge_op` 0/1 are loadable. **Do not spend iteration budget here.**
8. **Merging pairs of queued transfers to halve the 785 ns start latency: no material
   candidates.** 1284 of the 8285 transfers *already* carry more than one descriptor — the
   compiler has already coalesced what it can. Of the 8284 consecutive-id pairs, only
   **12** have both an identical trigger wait (so merging would not delay the first
   consumer) and an identical descriptor shape. 12 x 785 ns = 9 us. Dead end.

## Method notes / tooling that paid off

* `prof/Instruction.parquet`: `duration_ns` **excludes** the semaphore wait for compute and
  DGE slots (so DGE really costs ~600 ns of sequencer time) but **includes** it for
  `EVENT_SEMAPHORE` (median 24 ns, mean 409 ns on DVE). `evt_wait_time_ns` looks like a
  cumulative counter — do not read it per instruction.
* `prof/DmaPacketAggregated.parquet` is the gold mine: `block_id` **is** the `SP0.json`
  transfer `id`, and it carries `trigger`, `start_ts`, `end_ts`, `semaphore_id`,
  `is_transpose_mode`. That is what makes the trigger/latency/transfer split above possible.
* Per-engine idle attribution (gap -> wait field of the instruction that follows it, via
  `compiler_pc` -> stream slot) is the single most useful view on this kernel. The
  `Flow.parquet` longest-path walk is *not* trustworthy on its own: pure-wait
  `EVENT_SEMAPHORE` slots have no flow predecessor, so the walk blames them for gaps far
  larger than any real gap on that engine (it claimed 8367 us at Scalar/EVENT_SEMAPHORE
  when Scalar's largest actual idle gap is 36 us).
* `semaphores.py` passed on both the working edit and the one that hung — it only catches
  unsatisfiable waits, not capacity deadlock. Always run `verify.py` on device.

## Honest summary: this NEFF looks close to a local optimum for NEFF-level edits

Of the 14800 us span, I can account for it as a tight dependency chain through PE:
~3.3 ms of PE work, ~3.2 ms of genuine producer latency, ~2.0 ms of fixed queued-DMA
start latency, ~1.2 ms of transfer time, and the Act/DVE compute that feeds it. Every
*schedulable* slack I looked for turned out to be zero (trigger waits exactly tight,
no head-of-line blocking, nothing coalescable, engine occupancy irrelevant). Four separate
levers from the iter-0 review and from chapter 14 are now closed experimentally.

## What I would do next, in order

1. **Reordering the Activation stream** to put producers that gate a DMA trigger ahead of
   Activation work nothing waits on. This is the last lever with a plausible >=1 ms
   ceiling: 3209 us of PE idle is "producer not finished yet", and the median gating
   Activation producer sat ~1.3 us ready-but-queued. **Caveat / partial evidence against
   it:** my edit already made every Activation slot in that window ~540 ns cheaper
   (-14.9% busy) with essentially no latency change, which means making Activation work
   *cheaper* does not help; only changing the *order* could. Mechanism: *pairwise swaps of
   two S[4]-signalling instructions* (verified as r08 in chapter 10.3) so only the two wait
   values that named them need re-pointing, instead of renumbering all 15436 S[4] waiters.
   Pick the pairs from `prof/Instruction.parquet` + the per-engine gap attribution above.
2. Measure whether the ~725 ns queued-DMA ring-fetch latency has *any* knob. Unexplored
   opcodes that might: `PSEUDO_DMAREARM` (194), `PSEUDO_DMABARRIER` (195),
   `PSEUDO_DMASWAP_QUEUE_SET` (207) — all "not in corpus". Also unexplored:
   `def.json`'s `dma_queue` fields (`fabric_path: "main"`, and the `dynamic_dma`
   `SB_scratch_*` pair that only the *dynamic* queues carry). Low confidence, but it is the
   only remaining way to touch the 2.0 ms.
3. Accept that `total_exec_time` may be within a few percent of what this NEFF can do, and
   spend the budget confirming rather than chasing: any claimed win under ~0.5% (74 us) on
   this kernel is noise.
4. Do not re-measure anything on the engine-busy axis, and do not re-try items 1-8 under
   "Hypotheses REFUTED".

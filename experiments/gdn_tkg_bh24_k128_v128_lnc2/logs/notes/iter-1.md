# iter-1 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 76.054 us (baseline) -> 55.595 us, -26.9%.** Verified; `max_abs_err` on both outputs is
*identical to baseline* (2.322e-02 / 1.633e-03), i.e. the arithmetic is unchanged — every edit is
a pure schedule/placement edit.

## The one fact that drives everything

Profile the ACT (Scalar) stream and look at **gaps**, not just busy time. In the baseline,
`Activation` is 100% busy and has **zero idle after the kernel body starts** (the first ~12.4 us of
the profile span is the runtime prologue: DRAIN / COMPARE_BRANCH / NOTIFY / TENSOR_LOAD /
EVENT_SEMAPHORE with `compiler_pc = NaN` — those are *not* in the NEFF). So in the baseline
**every ns of ACT work removed is a ns off `total_exec_time`, 1:1.** That is why this iteration
worked. It is no longer fully true at 55.6 us (see "where the time is now").

`ACT_TABLE_LOAD` costs a **fixed 1283 ns** and the baseline had **22 per core = 28.2 us = 37% of the
kernel**. That was the whole opportunity.

## Mechanics that made the edits safe (reusable)

1. **`ACTIVATION_TABLE_LOAD` touches no semaphore at all** (`events_wait_mode=NONE`,
   `events_update_mode=NONE`; the slot is all zeros except opcode and `act_tbl_sel` at byte 35).
   So loads can be deleted, moved or re-pointed with **zero** semaphore work. Only invariant:
   when an `ACTIVATE` with `activation_func=F` runs, the currently-loaded set must contain `F`.
   `ACTIVATE` has **no** table-select field — the table is engine state.
2. **S[4] only constrains the relative order of s4-*updating* ACT instructions.** A wait
   `s4>=v` means "the v-th ACT instruction that updates s4 has completed". So: resolve every
   `s4>=v` to its producer slot, reorder, recompute, rewrite. `scripts/apply*.py` in this
   iteration do exactly that and it is fully mechanical (100 updaters, 101 waits across all 5
   streams, every one resolvable).
3. **A cycle checker is the single most valuable tool here.** Nodes = (engine, slot); edges =
   program order within an engine + (producer -> waiter) for every wait on s3/s4/s5/s6. If
   that graph has a cycle the NEFF **hangs** (30 s NRT_TIMEOUT, ~1 min wasted + device recovery).
   It caught 2 of my 4 proposed orders before any device run. **Write this first next time.**
   (`scripts/cycle.py`, ~60 lines.)
4. Useful side tools: `scripts/check.py` (asserts every ACTIVATE's func is in the live table set,
   using `<set>.json:profile_meta_data[].func_id`), `scripts/sbuf.py` (union of used SBUF
   partition-offsets across all 5 streams **and** the SP0.json DMA descriptors -> free gaps).

## What the kernel actually does (source is on disk!)

`nki_source_location` in `prof/Instruction.parquet` points at
`/home/ubuntu/workspace/gdn-bench/vendor/gdn/gdn_tkg.py` — **the NKI source exists on this
machine** (lines ~195-340). Read it; it answers every dataflow question in seconds.
12 heads per core, Python-unrolled, so the compiler interleaved 12 independent head streams.
Per head on ACT: `tensor_scalar(k_f*=k_scale)`, `tensor_scalar(S*=exp g)`, CAST S->S_bf,
`tensor_scalar(delta=diff*beta)`, CAST delta, CAST S->S_bf, COPY out_p->o_sb,
`activation_reduce(square,+)` (= ACTIVATE f30 ZERO_ACCUMULATE + ACTIVATION_READ_ACCUMULATOR),
`activation(rsqrt)` (f29, T2), `activation(silu)` (f36, T4).

The **f29/f36 alternation** is the thrash: f29 is only in `reciprocal_sqrt_and_small` /
`abs_reciprocal_sqrt_and_small`, f36 only in `silu_and_others` / `derivative_silu_and_others` —
**no table set contains both** (checked all 21 vendor sets in
`neuronxcc/pwp/pwp_bin_trainium`). f30 (square) and f128 (copy) are in every set, so they never
force a load.

## Edits applied (final NEFF), in order of discovery

| # | Edit | ACT loads/core | measured |
|---|---|---|---|
| 0 | baseline | 22 | 76.054 |
| 1 | move the 2 f36 that don't update s4 next to another f36; drop 4 now-dead loads | 18 | **70.594** (-7.2%) |
| 2 | **hoist all 12 silu f36 into one block** under one T4 load, with 12 **private** dst buffers at a fresh SBUF offset (`0x20000`), and the 12 DVE consumers' `src1` re-pointed | 8 | **58.534** (-23.0%) |
| 3 | collapse those 12 silus into **one** `ACTIVATE` with `num_elem=[1536,1,1]` (z_all and the new dst region are both contiguous) | 8 | **57.586** (-24.3%) |
| 4 | move `[T4 load, merged silu]` to the **very front** of the ACT stream so it fills the head's input-DMA stall; one T2 at old slot 22 then covers every f29 | 7 | **55.841** (-26.6%) |
| 5 | share one T0 load between the two `exp` (f7) calls: move slot 2 after slot 5, give slot 2 an s4 update, re-point DVE@5's wait to it | 6 | **55.595** (-26.9%) |

Final NEFF measured three times (same sha256 `4e389b19…`): **55.595 / 55.607 / 55.367 us**
(spread 0.5-2.0% per run). ACT busy 54.3-55.3 us, 98-99.5% of the span.
Steps 4-5 together only bought ~0.25 us of *total* time beyond step 3 even though they removed
2.5 us of ACT work — see "where the time is now". Step 4 is the valuable one (it re-used dead
time); step 5 is near-noise and could be dropped if its hand-repointed wait ever looks risky.

**Exact reproduction of the final ACT order** (baseline slot numbers; `m` = slot 46, the merged
silu; `scripts/apply8.py` in this iteration):
`[0, 45, m, 3, 4, 5, 6, 7, 8, 2]` then every remaining slot in ascending order, **dropping**
slot 1 (2nd T0 load) and the 14 tail loads
`{50,60,64,74,88,96,108,111,114,120,136,146,151,160,163}` and the 11 collapsed silus
`{63,75,87,97,112,121,125,134,147,148,161}`. Plus: slot 46 gets
`src/dst_mem_pattern_num_elem=[1536,1,1]` and `dst=0x20000`; slot 87's `events_wait_mode`
cleared; slot 2 gains `U[s4] SEM_INC_COMPLETE`; the 12 DVE consumers' `src1` set to
`0x20000 + (z_src - 8611328)`; `DVE@5`'s wait hand-pointed at slot 2; every other s4 wait
remapped mechanically; the 11 collapsed silus' waits all resolve to slot 46. Identical in sg00
and sg01.

Step 2 is the big one and the key realisation: **the 12 silus depend only on the `z` input DMA**
(one transfer, `qSPIO0` id 7, semaphore 15, `cast z -> SB 8611328`, 6144 B = all 12 head slices),
so they are independent of the whole recurrence. The *only* thing stopping the hoist was that the
compiler reused 8 SBUF buffers for the 12 `silu_z` values (21112 used 3x, 24176 and 18032 2x).
Giving each a private buffer removes the WAR hazard entirely, after which:
* `ACT@87`'s `W[s6]>=30` was purely that WAR guard — **dropped** it (its only real dep is the z DMA).
* no DVE wait needed hand-editing: `DVE@51` already waits for the first silu, and `DVE@52` waits
  for an ACT slot *after* the block, so by ACT program order all 12 are done for every later DVE
  consumer. Mechanical s4 remap handled the rest.

SBUF map (partition-offset, union over both subgraphs, all streams + DMA descriptors):
used `0x4000-0x9078`, `0x1c000-0x1d800`, `0x35500-0x37ff8`; **free: `0x9078-0x1c000` (77 KB) and
`0x1d800-0x35500` (97 KB)**. I put the silu block at `0x20000` (12 x 512 B). Addressing is
`addr = partition*0x40000 + offset`, PSUM is `0x2000000 + offset`.

## Levers that are CLOSED (do not retry)

* **Merging the rsqrt and silu table sets into one new set** — no vendor set has both, and a
  hand-merged set would need ~2126 bkt entries while the vendor's largest is 1447 (ctl max 254,
  suspiciously near 256). Capacity unknown and probably exceeded. Also unnecessary now.
* **Grouping the f36 *without* re-allocating their dst buffers** — blocked by buffer reuse +
  the WAR waits that protect it. (This is what killed review-iter-0's candidate 1 as stated.)
* **Grouping the 12 f29 (rsqrt)** — they are the serial tail of each head's recurrence
  (`matmul -> COPY -> square-reduce -> READ_ACCUMULATOR -> rsqrt`) and they share 4 reused
  scalar buffers (16384/16392/20592/220160). No hoist possible.
* **Moving `f29@7` (q_scale rsqrt) after `f7@9`** — **HANGS**. `ACT@9` waits `s6>=8`, DVE's 8th
  s6-update is behind `DVE@6`, and `DVE@6` waits for `ACT@7`. Cycle. Likewise `f29@7` must stay
  before `ACT@16`. The forced chain is `f9(T1) ... f29@7(T2) ... f7@9(T0) ... f5@16(T3) ...
  f29@23(T2)`, so **6 table loads is the structural floor** (T4,T1,T2,T0,T3,T2).
* **Deferring `f5@16` (sigmoid -> beta_K) past the rsqrts** to drop the 2nd T2 — beta_K's first
  consumer is `ACT@21`, which reads it through `scale_value_imm_ptr=16528`, not through a
  `src_mem_pattern` address. **Scalar operands are passed by pointer in
  `scale_value_imm_ptr` / `bias_imm_ptr` — a dependency scan that only looks at
  `src/dst_mem_pattern_start_addr` will MISS them.** I nearly broke this.
* `ACT_TABLE_LOAD` has no size/flush knob that changes its 1283 ns (slot is all zeros;
  `hint_or_notific_inst_flush_flags_flush_mode=FULL` is just the encoding of 0).

## Where the time is now (55.6 us, ACT 99.5% busy)

ACT per core: 6 x ACT_TABLE_LOAD = 7.7 us; ~58 ACTIVATE ~24 us; 36 CAST = 10.5 us;
12 COPY = 3.0 us; 12 ACTIVATION_READ_ACCUMULATOR = 3.5 us. ACT is still the only engine that
matters (DVE 54%, PE 38%, DMA 33%, Pool 17%, SP 11%).

ACT instruction cost is almost all **fixed overhead**: 128 elements on 1 channel costs ~290 ns
(~400 cycles at 1.4 GHz); 128x128 on 128 channels costs ~490 ns. The merged 1536-element silu
costs 1482 ns, i.e. ~0.8 ns/elem marginal — so **merging instructions pays, but less than
linearly**.

### ALSO CLOSED: moving the 12 `o_sb` COPYs off ACT (I tried both engines; both fail)

Target was the 12 `COPY out_p(PSUM) -> o_sb(SBUF)` (line 307, 12 x 249 ns = 3.0 us). I built it
fully: private `o_sb` buffers at `0x22000`, both readers re-pointed (the ACT square-reduce,
identified exactly because its `s4` wait resolves to its own COPY; and the DVE
`SCALAR_TENSOR_TENSOR`, matched through the chain COPY -> f30-reduce -> READ_ACCUMULATOR -> f29
-> the DVE instr whose s4 wait resolves to that f29), `events_update_idx` switched to the new
engine, and an **identity-keyed** semaphore remapper (key producers by `id(instruction)`, not
`(engine, slot)` — then an instruction that changes engine is remapped automatically, including
its wait *index* s4->s3/s6). That machinery works; the placement does not.

* **-> DVE: structurally impossible.** The per-head chain is
  `PE matmul_k -> COPY_k -> ACT f30-reduce_k -> ... -> (ACT work PE needs)`. On ACT this is fine
  because ACT runs in order and matmul_k's ACT prerequisite precedes the COPY. Once COPY_k is on
  DVE, every DVE instruction *before* it must not depend on ACT at/after f30-reduce_k — and DVE
  is interleaved with ACT so tightly that pulling the late heads' COPYs early enough makes DVE
  block on `s5>=N` (a late PE matmul) which needs ACT progress that DVE is itself blocking.
  I automated a search (cycle checker as oracle, pull offending COPYs earlier each round): it
  walked all 12 down to DVE index ~10 and hit an irreducible cycle
  `ACT@124 -> PE@78 -> PE@79 -> DVE@11..15 -> ACT@124`. Only the first ~3 heads could move
  (~0.7 us, inside the noise). Not worth it.
* **-> Pool: HANGS (NRT_TIMEOUT) even though my cycle checker said acyclic and
  `semaphores.py` was balanced** (s3: 12 updates / 12 waiters / max_wait 12). `COPY` is
  `common.NEURON_ISA_TPB_S4D4_TR_STRUCT` on every engine (859 Pool instances in the doc corpus),
  so the encoding is not the problem. Pool's whole stream is
  `SET_ORDERING_MODE, 3x PSEUDO_DMA_DIRECT2D, PSEUDO_DMA_DIRECT2D(W s6>=49), EVENT_SEMAPHORE, NOP`
  and the only legal window is *between* the pseudo-DMA triggers (before them deadlocks the input
  DMAs; after the 4th one is behind `s6>=49`, far too late). My best guess is that **inserting
  real slots between `PSEUDO_DMA_*` instructions breaks the runtime's expansion** (the runtime
  does track a "neff instruction index"). If someone retries this, first test the hypothesis with
  a *single* harmless inserted slot on Pool and see whether it still runs.

**Two lessons worth more than the 3 us:**
1. **My cycle checker is incomplete: it models DMA semaphores (8-16, 24-27) as free.** They are
   not — they are produced by a *triggering* instruction on SP (queued DMA) or Pool (DGE). Add
   those edges: `trigger instruction -> every waiter on that DMA semaphore`. Without them a
   schedule can look acyclic and still hang.
2. An acyclic semaphore graph is necessary but **not sufficient** — pseudo-instruction streams
   have constraints outside the semaphore model.

### Ranked candidates for iter-2

1. **Move ACT work to DVE, but only work that is *not* in the per-head serial chain.** The COPY
   failed because it sits mid-chain. Look instead for ACT instructions whose producer and
   consumer are both already on DVE (or are early/global), so the DVE insertion window is wide.
   Scan for that mechanically before investing. The 36 `tensor_scalar` ACTIVATEs
   (lines 274/277/290, ~16 us on ACT, `TENSOR_SCALAR_PTR_ARITH_OP` on DVE/Pool) are the big
   prize but need real encoding work, not a verbatim slot copy, *and* most of them are mid-chain.
2. **Merge the 12 line-280 `CAST S -> S_bf` into one 128x1536 CAST (~2.4 us).** This is the
   best-scoped remaining lever and I measured all the facts for it (did not attempt it):
   * `S_all` **is** contiguous: 114688 + 512h, h=0..11, `nch=128`, `num_elem=[128,1,1]` fp32
     (matches the used SBUF range 0x1c000-0x1d800 exactly).
   * The 12 **line-277** ops (`S *= exp(g)`, ACTIVATE f128 **in place** on S_all, `nch=128`,
     `scale_value_imm_ptr = 16480 + 4h`) are at baseline ACT slots
     **12, 13, 18, 24, 53, 54, 77, 78, 100, 103, 128, 131** and are **mutually independent** —
     each needs only the `S_all` bulk DMA and `exp_g_K` (= f7@9, slot 12's `s4>=4` points at it).
     So all 12 can be hoisted into one block right after slot 9. They cannot themselves be
     merged (per-head scalar pointer).
   * The 12 **line-280** CASTs (the ones whose wait points at their own line-277 op) are slots
     **14, 15, 26, 28, 58, 59, 81, 93, 106, 110, 130, 150**; the 12 **line-300** CASTs (wait on
     s6 = the DVE `S += outer`) are **30, 40, 44, 49, 76, 79, 98, 107, 129, 132, 152, 159**.
   * **The blocker is the destinations, not the sources**: line-280 dsts are
     `16752, 20080, 28280, 23920, 31864, 35192, 29304, 23416, 35960, 21360, 21360, 30584` —
     scattered, and 21360 is used by **two** heads. So you must first move the 12 `S_bf` tiles to
     a fresh strided region (12 x 256 B bf16 = 3072 B contiguous; use the free gap at
     `0x1d800-0x35500`), re-pointing the matching **line-300** CAST dst and the **PE matmul**
     operands that read each `S_bf` tile. After that the merge is one instruction with
     `num_elem=[1536,1,1]`, `nch=128`.
   * Ordering requirement: all 12 line-277 ops before the merged CAST, and the merged CAST before
     all 12 DVE `S += outer` (step 4) writes to `S_all`. The 12 `kv_p` matmuls then all run
     early; PE has slack (41% busy). Run `cycle.py` on the candidate order first.
   * Expect ~2.4 us of ACT work removed (12 x 291 ns -> ~1.1 us) and **check for a new stall** —
     twice this iteration a removal of ACT work converted straight into idle instead of time.
3. **Spread work into whatever stall remains.** The head stall is now filled; re-measure gaps
   before assuming any removal of ACT work converts 1:1. In plan "B" I removed 2 head loads
   (2.57 us) and got **nothing** because they were hidden behind the Q input DMA and a DVE wait.
   Always check: `body idle = span - busy` on the Scalar rows with `compiler_pc` not null.
4. The 1482 ns merged silu is 1-channel. Rewriting the `z` DMA descriptor (SP0.json, doc r09
   verified) to scatter z across 12 partitions would let one ACTIVATE use 12 channels
   (~1.3 us), but the DVE consumers and `out_sb` are partition-0 — probably not worth it.

### Small dead ends (checked, closed)

* **Looked for redundant ACT work** (same opcode+func+src+dst+pattern+scale pointer appearing
  twice). 5 groups turned up but **none is redundant**: the three f29 groups are different heads
  reusing the same 4-byte `ssum` scratch buffers, and the two CAST pairs
  (`25183088->16752` at slots 38/83, `117760->29304` at 74/87) both have their **source
  rewritten in between** (ACT@81 and DVE@65 respectively), so they re-materialise a changed
  value. No free instruction to delete.
* **Making the merged silu use more channels** (rewrite the `z` DMA descriptor to scatter z
  across 12 partitions so one ACTIVATE uses 12 channels instead of 1, ~1.1 us): closed, because
  the silu result is multiplied into `out_sb`, which lives on partition 0 — the output has to
  stay on one partition.

### Encoding trick worth remembering

Access patterns are up to 4-D with signed `step_elem` **in elements**, so a single instruction
can cover 12 *non-contiguous* equally-spaced slices: e.g. the 12 `S_bf` tiles are 3328 B apart,
which is `num_elem=[128,12], step_elem=[1,1664]` in bf16. Contiguity is **not** required for a
merge — only a constant stride. (The silu merge happened to be fully contiguous, so it only
needed `num_elem=[1536,1,1]`.)

## Safety evidence for the final NEFF

`verify.py --runs 5`: all 5 device runs **bit-identical**, `max_abs_err` exactly the baseline
value on both outputs (2.322e-02 = 86% of the limit, and 1.633e-03). `semaphores.py` clean on
sg00 and sg01 (s4: 95 updates / max_wait 95). `cycle.py`: acyclic, no unmet waits.
`launch_rules.py`: 2 subgraphs, num_tpb 1, inputs 10, output heads 4, unchanged. So the edit is
schedule-only and race-free under the fixed input — which is what you want, because the baseline
already spends 86% of the error budget and **any edit that perturbs the arithmetic will fail**.

## Method notes

* `profile.py --repeats 5` spread is 0.5-1.1% (~0.3-0.6 us). Treat anything under ~0.7 us as
  noise. **ACT busy time is far less noisy than `total_exec_time`** — use it to tell a real work
  reduction from a stall reshuffle (plan C 55.520 us busy vs plan D 55.300 us busy decided a tie
  the totals could not).
* sg00 and sg01 ACT streams are **identical** in slot indices, addresses and waits for every slot
  touched here, so the same edit list applies to both (semaphores are per-subgraph but the
  numbers come out the same). Always re-run `semaphores.py` on both anyway.
* After reordering, the profiler's `nki_source_location` / `compiler_pc` attribution is shifted
  and per-line tables become misleading; per-opcode totals stay correct.

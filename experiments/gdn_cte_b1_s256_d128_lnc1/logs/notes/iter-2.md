# iter-2 — took two reload DGEs out from behind a DMA wait they did not need

**Result: 61.817 -> 60.93 us (-1.44%)**, output bit-identical.
Shipped NEFF measured three times (7 captures each): **60.866 / 60.930 / 61.017**
(median of medians 60.930). iter-1's NEFF, re-measured at the start of this iteration:
**61.817** (and 61.830 / 61.803 after the two neutral tail edits). The two bands do not overlap.

| step | edit | median us | kept? |
|---|---|---|---|
| 0 | iter-1 NEFF, re-measured | 61.817 | — |
| 1 | output_1 store DGE `Pool[31]` -> SP (appended) | 61.830 | kept (see §3) |
| 1b | + output_0 chunk-4 store `Pool[32]` -> SP | 61.803 | kept (see §3) |
| **2** | **two var9 reload DGEs `SP[14]`/`SP[15]` -> Pool, given their own guard `S[4]>=18`** | **60.866 / 60.930 / 61.017** | **kept** |
| 2' | step 2 applied to the iter-1 NEFF *without* steps 1/1b | 61.158 | not shipped (§3) |
| 3 | gate-load DGE `Pool[1]` split in two, halves on SP + Activation | 61.149 / 61.143 | **reverted (§5)** |
| 4 | staging-save DGE `Pool[22]` (sem 45) -> Activation | 61.455 | **reverted (§6)** |
| 5 | gate-load DGE split in two, **both** halves on SP | 61.012 | **reverted (§5a)** |

`verify.py --runs 3`: PASS, outputs bit-identical across runs, `max_abs_err` **identical to the
baseline to every printed digit** (output_0 6.092e-05 = 11% of the limit, output_1
4.494e-04 = 83%). `launch_rules.py` PASS. `semaphores.py`: no over-wait.
Members changed vs iter-1: `sg00/{Pool0.bin, SP0.bin}`. Final sha256 `c8fefa05…`.

---

## 0. Tooling: iter-1 §9's warning is WRONG, and that is worth 30 minutes to you

iter-1 warned that `Instruction.parquet.compiler_pc` goes stale after you insert slots. **It does
not.** `compiler_pc` is derived from the raw `pc` column (pc of the instruction minus the pc of
the engine's first kernel instruction), so **`compiler_pc` == my slot index exactly, on every
engine, after any number of insertions**. Verified by comparing the decoded stream's opcode
against the profile's for all five engines at every step: 0 mismatches other than cosmetic
renames (`PSEUDO_DMATRIGGER` -> `WRITE`, `ACTIVATION_TABLE_LOAD` -> `ACT_TABLE_LOAD`,
`PSEUDO_DMA_DIRECT2D` -> `DMA_DIRECT2D`). Only `nki_source_location` is stale. Slot-indexed
joins are safe; do not waste time rewriting `.dbg`.

**Normalise timestamps to Pool[0] `SET_ORDERING_MODE` before comparing two profiles.** The
runtime preamble varies by ±600 ns run to run; raw timestamps differ by that much and you will
chase phantom regressions (I did, for ten minutes). All timestamps below are normalised.

`scripts/` in this worktree (~250 lines, rebuild in 20 min, not preserved):
`d2.py` compact stream dump (strips `reserved*`/name arrays), `alldge.py` every DGE descriptor
resolved against `def.json`, `waits.py` every wait **including extended** grouped by semaphore,
`q.py` stream x profile join with `why <eng> <slot>` / `tl <lo> <hi>` / `reach` / `prod`,
`path.py` the critical path flattened with the stall on each edge, `cmp.py` two profiles
normalised and compared, `nop.py` the self-asserting no-op builder (§4), `edit*.py` each edit,
self-asserting.

**Caveat on my own tool:** `q.py`'s backward walk resolves `S[3..6]` producers by "the v-th
signalling slot on that engine" and queued-ring producers by `id % 16`. Both are approximations
and I caught one case where it named a producer that provably ran *after* the consumer
(`SP[55]` "waiting" for `DVE[118]`). Treat a path node as a hypothesis and cross-check the
binding semaphore's reach time against `DmaPacketAggregated` (which has `trigger`, `start_ts`,
`end_ts` per transfer) before editing. Also beware: `SemaphoreUpdate.id` is sometimes
`"S[36] (qSyncSpillReload0)"` and sometimes a bare number, and conflating the two gives wrong
reach times.

## 1. Where the 61.8 us goes: DMA completion latency, not engine busy time

No engine is above 53%. The backward walk from the final Pool fence gives 171 nodes and the sum
of all attributable gaps (engine idle before a node whose wait had already cleared) is only
6.1 us. The chain is saturated with real work **plus DMA latency**:

| edge on the critical path | cost |
|---|---|
| `Pool[1]` gate-load DGE: row end -> `S[8]` visible | **2263 ns** |
| `SP[9]` trigger -> queued transfer 5 -> `S[29]` | **2920 ns** |
| `SP[17]` trigger -> queued transfer 8 -> `S[32]` | **2856 ns** |
| trigger -> transfer 6 -> `S[30]` (after edit 2) | 2187 ns |
| trigger -> transfer 20 -> `S[28]`; -> transfer 28 -> `S[36]` | ~1430 ns each |
| `Pool[32]` row end -> `S[21]` -> final fence | 1259 ns |

That is ~12 us of the 62. Measured sub-structure of a queued transfer:
**trigger -> DMA start 700-950 ns, the transfer itself 187-530 ns, DMA end -> semaphore visible
320-1374 ns.** The fixed overhead dominates and bytes barely matter (16 KB WZXY transposes run
in 229-315 ns). The end->semaphore part grows when several transfers land together (transfer 8:
1374 ns with 7 in flight; transfer 24: 229 ns alone).

**So the candidate lever is: never let a DGE or a trigger sit behind an instruction it does not
depend on.** Reducing engine busy time is not the job. But read §5a and §6 before trusting the
backward walk: **only one of the four places this table points at turned out to be elastic.**
Issuing a DMA earlier helps only where (a) the region has no slack and (b) the DMA is
issue-limited rather than fabric-limited. The one that was both is §2.

## 2. What worked: edit 2, -0.89 us (-1.44%)

`SP[14]` and `SP[15]` are two of the four var9 reload DGEs iter-1 created. They carry **no wait
of their own** — iter-1 relied on SP program order, because the first of the four (`SP[11]`)
carries the guard `S[4]>=18` (the neutralised staging save's own wait). But in program order they
sit *behind* `SP[13]`, a trigger that waits `S[29]>=16`, i.e. for queued transfer 5 — a 2.9 us
wait they have nothing to do with. Measured consequences:

* `SP[14]`/`SP[15]` issued at 19321 and 19974 instead of ~17800, so `S[62]`/`S[63]` were only
  visible at 22.6 us, gating `Activation[46]`;
* and they pushed the triggers for transfers 7 and 8 (`SP[16]`, `SP[17]`) 1.25 us later, so
  `S[32]` landed at 23.56 us and gated `Activation[49]`.

**Fix:** move both DGEs to Pool (idle 16.0-36.2 us), inserted right after `Pool[17]`, and give
each the guard `S[4]>=18` explicitly so they no longer depend on SP program order. Both pay-offs
land at once. Address-hazard check done first: the moved DGEs read SBUF `0x4d80`/`0x4f80` and
write partitions 96-127 at `0x1c308`; transfers 7-12 read `0x37100`-`0x37680` and write
`0x5900`-`0x6200` — disjoint. (Transfer 13 writes `0x4000..0x7fff`, which *does* overlap the
moved DGEs' source, so they must stay ahead of `SP[22]` — they do, by 20 us.)

**Generalisation worth reusing on any kernel iter-1 touched:** when iter-1 fused a DRAM
round-trip it left the fused reloads depending on program order for their guard. Any such reload
is free to move to any idle engine **once you give it the guard explicitly**. Look for DGEs with
`events_wait_mode == NONE` sitting behind a long wait on their host engine; `alldge.py` lists
them in one pass.

## 3. The two tail edits (1 and 1b): 0 on their own, ~0.25 us in combination — kept

The last two stores, `Pool[31]` (output_1, 64 KB) and `Pool[32]` (output_0 chunk 4, 16 KB), were
serialised on Pool, and the final fence `Pool[33]` waits for **both** (`S[21]` primary,
`S[20]` extended), so the program ends when the later of the two lands. `Pool[32]` was ready
752 ns before it issued. Moving `Pool[31]` to SP (edit 1) removed that block and **gained
nothing**, because:

> **A Pool DGE that follows an idle stretch issues ~531 ns after its wait clears.** (`S[6]>=93`
> at 59885, Pool idle since 56190, `Pool[32]` issued 60416.) iter-1 §8 suspected this was a
> profiler artifact. **It is real** — it only became visible once the head-of-line block in
> front of it was removed. An SP DGE after an idle stretch issues ~20 ns after its wait, and
> `Pool[1]` pays the same gap (654 ns) at the very start of the kernel.

Moving both stores to SP (edit 1b) moved the fence 210 ns earlier (62338 -> 62128, confirmed in
the profile) but `total_exec_time` only 14 ns, because **the tail is largely absorbed by the
runtime epilogue**: after `Pool[33]` there is a ~1.7 us cascade (per-engine barrier
`EVENT_SEMAPHORE`s, `DRAIN`, `EVENT_SEMAPHORE_RANGE_CLEAR`, `NOTIFY`, `COMPARE_BRANCH`); the last
engine to release is Tensor, ~470 ns after the Pool fence, with slack to spare.

They are kept because the measurement says the combination is better: edit 2 **on top of**
1+1b measures 60.866/60.930/61.017, edit 2 applied to the iter-1 NEFF **without** them measures
61.158. One sample vs three, so this is suggestive rather than settled; a future iteration that
wants a smaller diff can drop them and re-measure (they are `Pool0.bin` slots 31/32 -> no-ops
plus two appended `SP0.bin` slots).

## 4. A trap that cost two device runs — ALWAYS assert your no-op is bare

The "replace an instruction with an `EVENT_SEMAPHORE` that does nothing" idiom needs the
*template* slot to be free of an **extended** events block. I copied Pool slot 24 — which after
edit 2's insertion was the old `Pool[22]`, `wait S[46]>=16` **plus extended `S[45]>=16`** —
cleared only `events_wait_*` and `events_update_*`, and left the extended wait. Dropped at Pool
slot 1 it waits, at the very start of the kernel, for a semaphore produced by a Pool DGE 40 us
later in Pool's own stream: **guaranteed deadlock, 70 s `NRT_TIMEOUT` per attempt.**

Two runs were spent "refuting" two perfectly good hypotheses (a DRAM-source DGE on SP; two DGEs
sharing a semaphore) before I found it. `scripts/nop.py` builds the no-op and asserts
`events_extended_wait_mode == NONE` on the **re-decoded** bytes. More generally: after any field
edit, re-decode the encoded slot and assert the fields you did not mean to touch.
`semaphores.py` cannot catch this — it ignores the extended events block entirely.

## 5. Refuted: splitting the gate load (edit 3), +0.28 us — but the mechanism works

The gate load is the first thing the kernel does and the whole compute chain hangs off it:
`Pool[1]` DGE 702-1496, `S[8]>=16` visible 3759, first compute `DVE[8] CAST` 3790. 256 scattered
2-byte DRAM reads into 64 partitions x 4 fp32, plus Pool's 654 ns start-up gap.

Split into two 32-partition halves, one on SP (idle 0-4510) and one on Activation (inserted ahead
of `ACT_TABLE_LOAD`, whose first consumer is at 4517). It did exactly what was predicted locally:
halves issued at 11-793 and 187-881, `S[8]>=32` at **3122 instead of 3759 (-641 ns)**,
`DVE[8]` 3145 instead of 3787, instruction span 357 ns shorter. And yet `total_exec_time` went
**up** 0.28 us, reproduced in two independent 7-capture runs (61.149, 61.143), with Activation
busy +0.59 us. Reverted.

**Three DGE mechanisms newly verified on the device** (bit-identical output), worth keeping even
though the edit lost:

1. **A DGE can be split into N smaller DGEs sharing one `semaphore`, each with
   `sem_increment = 16`; the increments accumulate.** Two DGEs with `semaphore = 8, +16` and the
   single consumer's wait raised `S[8]>=16` -> `>=32`. `semaphores.py` reports
   `8  32  32  1  {'dge:Activation': 16, 'dge:SP': 16}`.
2. **A DGE whose source is a DRAM variable (address marker 32, `DGETable0` index) runs correctly
   from the SP and Activation hosts**, not only from Pool, even though every compiler-emitted
   DRAM-addressed DGE in this NEFF is on Pool. (Pool's `qPoolDynamic` is the only queue with a
   `dynamic_dma` SB-scratch block in `def.json`; it turns out not to matter.)
3. **A DGE can be inserted ahead of `ACTIVATION_TABLE_LOAD` on the Activation stream.**

Recipe for splitting a 2-D DGE along one dimension: element `(i, j)` sits at
`src_off + i*src_step[0] + j*src_step[1]` (and likewise dst), with `num_elem = [ni, nj]`, all in
bytes. To split `j` into `[0, m)` and `[m, nj)`: set both `num_elem[1] = m`, and for the second
half add `m*src_step[1]` / `m*dst_step[1]` to the respective `*_start_addr_addr_var_offset`.
Setting `_var_offset` leaves the address marker and the table index intact — assert them after.

## 5a. Edit 5 (both halves on SP): neutral, and it closes the gate-load lever

Same split, both halves on SP (idle 0-4.5 us, no start-up gap, no busy engine disturbed).
Measured **61.012** — dead centre of the shipped band (60.866/60.930/61.017), i.e. nothing.
The profile says why: `S[8]>=16` at 2525 but `S[8]>=32` at **3710**, only **53 ns** better than
the unsplit 3763. The halves issued 650 ns apart and their *transfers* ran back-to-back.

> **DGEs issued from the same host engine share that engine's dynamic DMA queue and their
> transfers serialise; DGEs on different hosts use different queues and overlap.** Edit 3
> (SP + Activation -> `qSPDynamicHW` + `qActDynamicHW`) got `S[8]>=32` to 3122, -641 ns;
> edit 5 (both on SP -> one queue) got 53 ns. Splitting a DMA only buys concurrency if the
> pieces go to different *hosts*.

And the gate-load lever looks closed, because **moving `S[8]` earlier did not move the finish**:

| | `S[8]>=32` | `DVE[8]` start | total_exec_time |
|---|---|---|---|
| shipped | 3763 | 3787 | 60.866 / 60.930 / 61.017 |
| edit 3 (SP + Activation) | 3122 (-641) | 3145 | 61.143 / 61.149 |
| edit 5 (SP + SP) | 3710 (-53) | 3734 | 61.012 |

Edit 3 bought 641 ns at the gate load and the finish got 0.2-0.3 us *worse*; edit 5 bought
nothing and the finish did not move. §9 says why: the whole 0-17 us region is essentially
deterministic and the finish time is set from ~21.4 us onwards.

**Correction to a claim I made mid-iteration:** I briefly read edit 5's profile as moving
`Sync[10]` 1.9 us earlier. That was an index-shift artifact — edit 5 inserts two slots at SP 1-2,
so edit 5's `Sync[10]` is the shipped NEFF's `Sync[8]`, a different instruction. When you compare
two profiles slot-by-slot, **offset the indices of every engine you inserted into.** `cmp.py`
does not do this for you.

## 6. Refuted: rebalancing the staging-save DGEs (edit 4), +0.53 us — and WHY it is closed

Eight staging-save DGEs become ready within ~2 us of each other at 50.9 us, 2-3 deep on the only
three engines that can host a DGE. Their consumers do not need them equally soon: the Pool fence
chain (`Pool[24]` -> `Pool[25]` -> `Pool[26]` -> `SP[36]` -> `SP[37]` -> queued transfer 20 ->
`PE[134]`) needs `S[43]`/`S[44]`/`S[45]`/`S[46]`, and `Pool[24]` ends on its *extended* wait
`S[45]>=16` at 54260 — `S[45]` being the **second** DGE on Pool while the other three are each
the first on their host and land 53.2-53.6 us.

So I moved `S[45]`'s DGE to Activation, whose first DGE (`S[43]`, waiting `S[6]>=62`) is finished
1.3 us before `S[4]>=42` clears, making it effectively a "first" DGE too. It issued **exactly**
when predicted (50750, the moment `S[4]` hit 42, 444 ns earlier than before) — **and `S[45]`
still arrived at ~54230, unchanged.** `Pool[24]` still ended at 54250.

> **Conclusion: in the 50-54 us window the eight SBUF->SBUF staging DMAs (256 KB) are
> DMA-fabric/queue limited, not issue limited.** Moving one DGE's *issue* earlier does not move
> its *completion* earlier; the completions land in the same order at the same time. The
> "~1650 ns DGE row-end -> semaphore" figure is a queueing effect in that window, not a fixed
> latency. The 0.53 us regression is just the 653 ns the DGE row costs on Activation's critical
> chain. **Do not rebalance DGE hosts in this window again.**

## 7. Activation is the wrong DGE host on this kernel

Both edits that put a DGE on Activation regressed (edit 3 +0.28, edit 4 +0.53) even though each
did precisely what it was designed to do locally. Activation is the busiest engine (53%) and its
instruction chain is on the critical path for most of the kernel, so the ~650 ns a DGE row costs
its host is paid in full there. **Prefer SP (17.6% busy, idle 0-4.5 us and from 56.9 us), then
Pool (26%, idle 16.0-36.2 us).** Note the asymmetry with §3: SP is also the only host that does
not pay a ~530 ns start-up gap on its first DGE after an idle stretch.

## 8. Facts about this kernel worth keeping

* **Engine idle windows** (normalised to Pool[0]): Pool **16.0-36.2** and 36.2-39.7 us;
  SP **0-4.5** us and from **56.9** us; PE **37.6-44.3** us. Activation and DVE have no window
  longer than ~1 us.
* **Only ACT, Pool and SP can host a DGE.** `PSEUDO_DMA_DIRECT2D` never appears on PE or DVE in
  the whole 14-NEFF corpus, and `def.json` declares DMA queues only for owners
  `act`/`pool`/`sp`. The eight-DGE save group can never be spread wider than three engines.
* **A DGE row in the profile is ~650 ns of its host engine's issue slot** regardless of size
  (512 B: 794 ns; 32 KB: 617-667 ns). A *compute* instruction behind it cannot issue until the
  row ends; `EVENT_SEMAPHORE`/`MOVE` slots can overlap it, and a DGE row can start while the
  previous compute instruction is still finishing.
* **DGE-table index != `var_id`.** In a DGE the address's `var_id_lo` is an index into
  `def.json`'s `DGETable0.list` (`[5, 2, 4, 3, 1, 6, 7]`), so `v0`->gate, `v1`->k, `v2`->beta,
  `v3`->v, `v4`->q, `v5`->output_0, `v6`->output_1. The address **marker** (32 = table var,
  16 = plain SBUF) is what distinguishes a DRAM var from an SBUF address, not the index.
  `alldge.py` naming the wrong var cost me a few minutes of confusion.
* **`in_dtype`/`out_dtype` sometimes decode to raw ints** (6 = BFLOAT16, 10 = FP32) rather than
  symbols — assert against both forms.
* **The queued transposes are cheap; the one `op:"copy"` is not.** Transfer 5 (the var8 reload
  iter-1 fused) is 4 descriptors, 32 KB, `num_tiling_dimensions: 1`, innermost contiguous run
  only **32 bytes**: 1891-2079 ns of DMA and 2920 ns trigger-to-semaphore, on the critical path.
  Every other transfer is `transpose_op: WZXY` and runs 187-530 ns for 0.25-32 KB. The slow part
  is the destination pattern, and it is the same pattern the pre-iter-1 DRAM reload used, so this
  is not a regression iter-1 introduced (iter-1 measured its var8 fusion as worth 0 on its own).
* **Queued-descriptor fields are in bytes**, not elements (checked against transfer 1:
  `from_sizes [256,64,1,1]` -> 16384 = `transfer_bytes`).

## 9. THE ELASTICITY MAP — read this before picking a target (10 captures, shipped NEFF)

Ten single-capture profiles of the *same* NEFF (`profile.py --repeats 1 --parquet elN`, ~13 s
each, 3 minutes total), then for each landmark: its mean time (normalised to `Pool[0]`), its
standard deviation across the ten captures, and its correlation with that capture's
`total_exec_time`. This is the cheapest high-value measurement in this whole loop and I wish I
had run it first.

| landmark | mean | sd | corr(total) |
|---|---|---|---|
| `Vector[8]` first compute, `Tensor[3]`, `Scalar[2]` | 3786-4509 | **6-7** | -0.26..-0.36 |
| `GpSimd[14]`, `GpSimd[19]`, transfer-5 trigger | 10666-16299 | **6-7** | -0.04..-0.12 |
| transfer 5 end | 19068 | 45 | 0.20 |
| **`S[29]`** (transfer 5 complete) -> `Sync[13]`, all the transfer 6-13 triggers | 19582-20260 | **128** | 0.37 |
| transfer 6/7/8/12/13 end | 20584-21507 | 134-197 | 0.62-0.80 |
| **`S[30]`** (transfer 6 complete) -> **`Tensor[23]`** | **21411 / 21453** | **312** | **0.90** |
| every landmark after `Tensor[23]`, to the last instruction | 21453-63416 | 312-383 | **0.91-1.00**, slope **1.00-1.22** |

Two things fall out:

1. **The kernel is rigid from `PE[23]` (21.4 us) to the end.** 40 us of instructions, 170+
   landmarks, every one of them with the same 312 ns sd and `corr >= 0.90`, slope ~1.0. Delay
   `PE[23]` by N and the kernel finishes N later; nothing downstream has any slack to absorb it.
   That is also why edit 2 converted its local gain into total time almost 1:1.
2. **All of the run-to-run variance is created in two steps, both of them DMA completion
   semaphores:** `S[29]` (transfer 5, sd 7 -> 128) and `S[30]` (transfer 6, sd 134 -> 312). The
   jump happens in the *DMA-end -> semaphore-visible* step, not in the transfer.
   The 0-17 us region, including the gate load, has sd 6-7 ns — it is clockwork, and its
   correlation with the finish is slightly *negative*, i.e. it does not drive the end.
   (Caveat: zero variance cannot *prove* zero elasticity — there is nothing to correlate. But
   edits 3 and 5 tested that region directly and got nothing, so both lines of evidence agree.)

**Therefore the single target for iter-3 is the chain
`transfer-5 trigger (16299) -> S[29] (19582) -> transfer-6 trigger -> S[30] (21411) -> PE[23]`:
5.1 us of strictly serial DMA sitting immediately upstream of a perfectly rigid 40 us tail.**
Transfer 6 reads what transfer 5 writes, so the serialisation is a true data dependency —
which is exactly what §10.1's fusion removes. Everything else in this kernel is either rigid
(nothing to win) or deterministic-and-off-path (nothing to win).

## 10. Ranked candidates for iter-3

1. **Fuse queued transfer 5 away** — the single biggest item left: 2.9 us of pure critical path
   at 16.0-19.2 us, and transfer 6 (another 2.2 us) is strictly serial behind it because it reads
   what transfer 5 writes. Transfer 5 is a plain `op:"copy"` SB->SB staging hop; transfers 6-12
   all read inside its destination. This is the iter-1 §2(a) reload-side rewrite applied to a
   *queued* consumer, and queued descriptors have 4 dims, so the composed pattern should fit.
   **§9 confirms this is the right target and the only one:** transfer 5's trigger is
   deterministic at 16299, its completion `S[29]` is where variance first appears, and from
   `PE[23]` (21453, gated by transfer 6, which waits for transfer 5) the kernel is rigid to the
   end with slope 1.0. So the 2.9 us transfer 5 costs plus the 1.8 us transfer 6 costs are
   elastic almost 1:1 — the largest remaining prize by far.
   **Do the arithmetic first.** Transfer 5 desc *g* (g = 0..3) maps
   `dst = base_g + d0 + 128*d1 + 512*d2 + 262144*d3` (base_0 = `0x36f00`, base_1 = `0x436f20`,
   base_2 = `0x836f40`, base_3 = `0xc36f60`, i.e. partitions 16g..16g+15, offset `0x36f00 + 32g`)
   from `src = 0x1005600 + 128g + d0 + 32*d1 + 4194304*d2 + 262144*d3`, sizes `[32,4,4,16]`.
   Note the source's `d2` advances 16 **partitions** while the destination's advances 512
   **bytes** — it is a partition<->offset reshuffle, not a copy.
   **The one thing you must settle first (I could not, and it is the whole edit).** Reading
   `sizes`/`steps` as bytes with every dim contributing independently, transfer 5's destination
   byte set per partition is `{0..31} + 128*m, m = 0..15` (because
   `{128*d1} + {512*d2}, d1,d2 < 4` is exactly every multiple of 128 up to 1920), with each
   16-partition group shifted by a further 32 bytes (`to_off` = `0x36f00`, `0x436f20`,
   `0x836f40`, `0xc36f60` = partitions 0-15/16-31/32-47/48-63 at offsets `0x36f00 + 32g`).
   That makes the destination **sparse**: 32 valid bytes in every 128. But transfer 6 reads
   `0x36f00 + 0..127` **contiguously** on partitions 0..63 (`from_sizes [128,64,1,1]`,
   `from_steps [1,262144,128,...]`, 8192 bytes). Those two cannot both be right.
   The pre-iter-1 descriptor (in `baseline/kernel.neff`, `SP0.json` id 5) settles that the
   *source* side was a complete dense cover — `from` = var8 (32 KB), `from_off` `0x0`/`0x80`/
   `0x100`/`0x180`, `from_steps [1, 32, 8192, 512]`, same sizes: per desc that is 128 contiguous
   bytes at `8192*d2 + 512*d3 + 128g`, and the four descs tile var8 exactly. So the transfer
   really does move all 32 KB, and **my reading of the `to` side must be wrong** — most likely
   the dim pairing or the partition split for a `num_tiling_dimensions: 1` descriptor.
   Decisive cheap experiment (chapter 13.6 / e01 style): perturb one field of transfer 5's
   descriptor by a known amount (e.g. desc 0's `to_off` by +0x20, or swap desc 0's and desc 1's
   `from_off`) and read which output elements change. One device run tells you the traversal
   order, after which the composition into transfers 6-12 is mechanical.
2. **Splitting a queued transfer across ring slots is CLOSED.** A transfer's `id` is positional in
   `SP0.json.dma`, its semaphore is `semaphore_set[id % 16]` with +16 per completion, and waits
   are absolute counts. Inserting ids in the middle renumbers 27 transfers. Appending ids 33+ is
   worse than useless early in the kernel: id 33 lands on ring slot 1, whose transfer 17 has not
   run yet, so `S[25]` would reach 32 early and `Activation[91]`'s `S[25]>=32` wait would release
   before transfer 17 wrote its data — a correctness bug the fixed-input verifier might not
   catch. Only appending transfers that trigger *after* every existing transfer on their ring
   slot has completed is safe.
3. **Run the §9 elasticity map first, before any edit.** Three of my five edits did exactly
   what they were designed to do locally and bought nothing (§3, §5a, §6). Ten
   `profile.py --repeats 1 --parquet elN` captures cost 3 minutes and would have ruled out the
   tail, the gate load and the save group before I spent four device runs on them. Re-run it
   after each accepted edit: the rigid/elastic boundary moves.
4. **Do not revisit:** the gate load / the first 17 us (§5a — two experiments, has slack),
   the tail / final fence (§3), the Pool DGE start-up gap (§3 — real, but
   cheaper to avoid by choosing SP as the host than to remove; iter-1's edit7 already refuted
   hoisting the wait), rebalancing the save-group DGE hosts (§6), putting a DGE on Activation
   (§7), merging the beta loads (iter-1 §7), swapping the two tail stores (iter-1 §7),
   reload-side fusion of var11/var12 (iter-1 §7), lowering `output_1`'s error margin (83% of the
   limit already).
5. **Measurement discipline.** The 7-capture median itself moves ±0.15 us run to run and the
   within-run spread is 0.6-1.8%. Anything under ~0.5 us needs 2-3 repeats of the 7-capture
   median on both sides before you believe it; I spent four device runs this iteration doing
   exactly that and it changed two decisions.

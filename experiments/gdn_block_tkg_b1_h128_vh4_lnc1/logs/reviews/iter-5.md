# Review of iter-5 (kernel.neff vs HEAD e5fb209; HEAD reported 60.0 us, baseline 218.2 us)

`notes/` contains only the stock README, so there is no agent write-up to check. The only claim to compare against is the HEAD commit message, which describes the *previous* iteration (74.6 -> 60.0 us). The diff below is what this iteration changed on top of it.

## 1. What changed (neff_diff.py HEAD -> working tree)

6 of 46 members changed: Activation0.bin (28 slots), DVE0.bin (66 slots, 115 -> 119 slots), PE0.bin (4), Pool0.bin (1), SP0.bin (30), SP0.json (DMA blocks). The instruction count only grows on DVE.

- **SP:** 6 `PSEUDO_DMA_DIRECT2D` and 8 `PSEUDO_DMATRIGGER` became `EVENT_SEMAPHORE` (signal-only). These are the SBUF->DRAM->SBUF scratch round trips and the 256-element transposes.
  - SP0.json: blocks 16 and 18 (sems 24 and 26) shrank from 256-element SB->SB copies to a 2-element copy into the named scratch `inst__I-16-0:_mem_110_105`. Nothing waits on them any more.
  - SP slot 18 changed `src_elem_size` and `dst_elem_size` 2 -> 8 and `dst_step_elem` 2 -> 8. Its destination moved to 0x3b180, which is the packed region (`step_elem` 4) that the compute engines now read.
- **Activation:** slots 55-58 and 74-77 changed their `step_elem` to 4 and their addresses to the 0x3b180 / 0x3a980 regions. Slots 72-73 lost their SP waits (sems 58, 60, 63, 64 at value 32).
  - Slots 86-95: one `ACTIVATION_TABLE_LOAD` moved (and one ACTIVATE was reordered next to a CAST), a CAST was swapped with an ACTIVATE, and one `ACTIVATE` func=30 -> func=29.
  - Slot 94 now reads 4 elements at SBUF 0x2000000 and writes with `activation_func=29` and a bias immediate. This is the `rsqrt(x+eps)` reached through a different address.
- **DVE:** 4 new slots. Six `SCALAR_TENSOR_TENSOR`/`STREAM_SHUFFLE`/`LOAD_MASK_SELECT` slots became CAST/COPY reading bf16 from the 0x3ad80 region. Sem waits were rebalanced (events_wait_idx 12 -> 10 and so on).
- **PE:** 4 matmul sources changed from the 0x33d00 area (`step -1`, `-32`) to 0x3b578 (`step -4`, `-128`), i.e. a re-packed layout.
- **Pool:** 1 wait index changed, 45 -> 42.

The pattern is consistent with the earlier iterations' technique: replace DRAM round trips with direct SBUF->SBUF compute, and re-lay-out so the consumer engines read the new region. I could not check against the agent's own notes because none exist.

## 2. Is the latency win real?

Yes.

- **Remeasured (5 repeats):** median 51.05 us, min 50.99, max 51.52, spread 1.05%. This is the same device as the saved result.
- **Saved profile_latest.json:** median 51.175 us, min 50.86, max 53.57, spread 5.3% (one 53.6 outlier). Without the outlier the spread is about 0.7%.
- **Against HEAD:** 60.0 -> ~51.1 us, a gain of about 8.9 us. That is roughly 10x the spread, so it is not noise. Against baseline: 218.2 -> 51.1 us, 4.27x.
- **At this size noise matters:** a gain below about 0.5 us cannot be told from noise at ~1% spread, so the next agent should use `--repeats 5` or more.

## 3. Is it safe beyond the verifier?

- **Error:** the verifier passes, with max_abs_err 3.0e-3 (83% of the 3.6e-3 limit) on output_0, 1.9e-3 (54%) on output_1 and 6.0e-4 (17%) on output_2. The baseline had exactly the same errors, so none of the edits moved the numerics. Output_0 sits at 83% of the limit in the unedited NEFF too, so there is no headroom to spend on any approximate substitution (for example replacing rsqrt with a different formulation).
- **Semaphores (semaphores.py, HEAD vs now):** no new "wait exceeds increments" rows. There are three observations.
  - Sems 24 and 26 now have zero waiters, with 16 increments each from the neutralised DMAs. Their destination is the dead scratch `mem_110_105`, so there is no data hazard. Sems 28-39 and 42-45 were deleted along with their DMAs.
  - Sems 15, 22, 57-61 and 62-66 have `total_inc` greater than `max_wait`, and sem 6 has 73 waiters against 63 increments. All of these rows are unchanged from HEAD, and they are the usual overcount from multi-queue DMAs, but they do not guard against a late hazard.
  - Sem 46 is a new DVE:CAST-to-single-waiter edge. It is balanced, 16 increments against a wait of 16.
- **Residual risk:** the loosened waits (Activation 72-73 dropping their SP waits) pass on one fixed input set. They are safe only if the removed producers truly were dead or already ordered through another sem. The mapping above indicates they were, but there is no hardware-visible check beyond "output matches on this input".
- **Unwaited DMA at kernel end:** blocks 16/18 write a 2-byte DRAM scratch with no waiter. If the runtime ever stops waiting for all DMA queues before finishing, this is harmless because the buffer is dead.

## 4. Next candidates

Timeline from `prof/Instruction.parquet` (kernel body ≈ 12.5 -> 62 us; 0-12.5 and 64.8-66 us are runtime pre/post-amble that cannot be edited). Activation is 77% busy, Pool 47%, DMA 37%, SP 13%.

Activation time breakdown: 9.0 us of ACT_TABLE_LOAD (7 x 1.283), 6.8 us of DMA issue (8 x 0.88), 8.7 us of CAST/COPY (36 ops, median 0.24 us), 17 us of ACTIVATE/READ_ACC. DVE CAST takes 0.14 us and Pool CAST 0.18 us.

### Ranking (best expected gain per effort/risk first)

**1. Spread the prologue DMA issue onto the idle SP engine.**
- *Where:* Pool0.bin slots 74-188, 16 DMA_DIRECT2D issues from 13.2 to 27.0 us at 0.66-0.95 us each, plus Activation0.bin slots 1-8 (8 DMAs at 0.88 us each, 12.6 -> 19.8 us). SP is idle in that interval (busy 6.8 us total, first useful op after 47 us).
- *Change:* move 4-6 of the later Pool weight DMAs (the ones issued at 19.8-27.0) and 2-3 of the Activation ones to SP. This is a structural change: insert DMA_DIRECT2D slots in SP0.bin using SP's unused queue/semaphore indices, then re-point the consumer waits. Delete the originals or turn them into EVENT_SEMAPHORE.
- *Mechanism:* iter-2 did exactly this to Activation's DGE queues ("inserting instruction slots into a stream works", sems 80-87), and the DGE-issue cost is per-engine serial. Established on this device in commits fa6ac0a and d596d33.
- *Ceiling:* Pool's last prologue DMA is issued at 27.03 us and Pool's next op starts at 29.31, i.e. it is gated by data completion, not issue. The first dependent compute is the ACTIVATE at 23.67. A perfect three-way split would shorten the front from 13.8 to about 6-7 us. I'd bound the gain at ≤ ~5 us, but the total weight bytes (~680 KB HBM read) may impose a bandwidth floor that I have not measured.
- *Risk:* moderate. The DMA completion semaphores are re-pointed, which a fixed-input run can pass while a late wait is wrong; the DGE transposes are a single serial resource (see "closed"), so SP must not add transposes in the 34-50 us window.
- *Cheapest falsifier:* move only Pool's two DMAs at 19.80 and 20.44 us (slots 105 and 111) to SP and see whether Pool's ops at 29.3 us move earlier. Dies in one profile run if the DMA hardware, not the issue port, is the limit.

**2. Move Activation CAST/COPY that do not read PSUM to DVE or Pool.**
- *Where:* Activation0.bin slots 60, 62, 63, 66-71, 74-77 (done in iter-5), 85, 90, 91, 93, 108, 110, 111, 117, 120. These are 32 CASTs at 0.237 us median (7.7 us total on the busiest engine).
- *Change:* swap the opcode/engine for those whose source is SBUF. DVE takes them at 0.14 us, Pool at 0.18 us. Pool is idle from 47.8 to 60.7 us.
- *Mechanism:* the iter-3/4 recipe (commit e5fb209: "4 Pool CASTs"), plus the known address immediate with partition encoded in the linear address.
- *Ceiling:* the Activation tail chain at 45-62 us is latency-bound (many instructions have evt_wait 500-1400 ns), so only CASTs on the critical path pay. Upper bound is the sum of their savings ≈ 36 x ~0.07 = ~2.5 us if all of them were critical. Realistic ~0.5-1.5 us.
- *Risk:* low if the destination dtype/engine is PSUM-free; moving the cast changes which engine a sem edge originates from, so the per-semaphore consumer waits need re-indexing. A wrong wait here could pass once and race later.
- *Falsifier:* move one tail CAST (slot 117 or 120) to DVE, re-profile.

**3. Delete one or more of the 7 ACT_TABLE_LOADs by checking which tables the oddball funcs share.**
- *Where:* Activation0.bin slots 10, 24, 35, 45, 51, 59, 89 (1.283 us each, 9.0 us on Activation total). The loads at 45 and 51 are the suspect pair: func=5 at slot 46 sits between the silu (36) cluster and a single rsqrt (func 29) at slot 53, which sits again before the silu cluster at 61.
- *Change:* replace slot 45 or 51 with an EVENT_SEMAPHORE no-op (no-op whose table is already resident, per the e5fb209 finding). Alternatively reorder slot 53 next to the other func-29 users (94, 99, ...), if its data dependency allows.
- *Mechanism:* "an ACT_TABLE_LOAD whose table is already resident can be replaced by a no-op" (HEAD commit); funcs 36 = silu and 29 = rsqrt are table-specific, while 30, 128, CAST and COPY are table-agnostic. The tables for funcs 5, 7 and 9 are not recorded anywhere I could find.
- *Ceiling:* 1.283 us per load actually removed from the critical chain, ≤ ~2.6 us for two.
- *Risk:* moderate. If the table is wrong the activation produces garbage, which the verifier does catch; the issue would be if it silently drifts within the tolerance, so also check error against the baseline's 83%.
- *Falsifier:* swap slot 45 for a no-op and run verify.py (one run).

**4. Trim the tail signalling chain.**
- *Where:* SP0.bin slots 76-84. After SP's wait on sem 6 = 63 at slot 76, it runs 4 no-op EVENT_SEMAPHOREs (slots 77-80) before the PSEUDO_DMATRIGGER at slot 81. That takes the trigger from 62.67 to 62.93 us. Also Pool slot 70 issues the output DMA at 60.7 us (waiting for sem 6 = 60), which completes about 64.2 us.
- *Change:* turn slots 77-80 into the trigger directly, or hoist the trigger above them (the waits on 0 are no-ops).
- *Ceiling:* about 0.15-0.25 us (62.67 -> 62.93 us). The final DMA completion hop (63.39 -> 64.7 us, ~1.3 us) and the 1.3 us postamble are not editable.
- *Risk:* low. *Falsifier:* tail is 62.9 -> 64.7 us in the profile; if the trigger moves earlier and the end does not, it is dead.

**5. Reorder Pool's DMA issue so the first-consumed weights go first.**
- *Where:* Pool0.bin slots 74-188 (the 16 DMA issues). The Activation pc 15 ACTIVATE at 23.67 waits 3.5 us for data; Pool's DMAs run in id order, not consumption order.
- *Ceiling:* unknown. It is a subset of candidate 1's ceiling, and the previous iteration already put conv_weight first. Only worth doing if candidate 1 shows the issue port is the limit.

If the remaining chain is dominated by the 4-head recurrence (COPY -> ACTIVATE func 30 -> READ_ACC -> ACTIVATE func 29 -> DVE ops, ~3 us per head, 47 -> 62 us), that part is serial dependency through PE/PSUM/Act/DVE and I would treat it as near-irreducible at the NEFF level.

### Closed (do not redo)

- DGE transposes: the DGE is a single serial resource retiring ~1536 two-byte descriptors at ~15 ns each between 34 and 50 us. Three edits that only let a transfer start earlier all lost time (e5fb209).
- `dma_configs_dma_transpose_2d_tile` is inert (e5fb209).
- Per-partition SBUF->SBUF DMAs: 0.55 us of issue plus a 2-5 us completion hop each versus 0.03 us for a semaphore add; all removed (d596d33, fa6ac0a).
- The SBUF->DRAM->SBUF round trips (~135 ns/element, ~18 us per 128-element write): all six removed (480fb81) and the remaining three in e5fb209.
- Dead qSPSpillReload0 triggers and iter-1 dummies: removed (fa6ac0a).

### Unknown (measurements that would change the ranking)

- **Which activation table funcs 5, 7 and 9 live in.** Settle by checking whether each ACTIVATE still passes `verify.py` when its preceding table load is replaced by a no-op (one run per load).
- **Whether prologue DMA is issue-bound or bandwidth-bound.** Compare `DmaUsage.parquet` / `DmaPacket.parquet` timestamps for the prologue: if the DMA rings are idle between issues, candidate 1 is worth a lot; if they are saturated, the ceiling is ~0.
- **Which Activation CASTs are on the critical path.** Check `evt_wait_time_ns` per CAST (many in the tail are ≤ 2 ns, i.e. already queued) before moving any.
- **Whether the removed SP waits are safe for any input.** The mapping says the producers were dead, but only a second input set (re-run `verify.py` with a different seed in `bench/inputs`) would test it.

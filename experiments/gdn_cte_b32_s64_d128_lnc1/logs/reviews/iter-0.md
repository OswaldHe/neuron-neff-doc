# Review of iter-0 (gdn_cte_b32_s64_d128)

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports all 34 members identical ("payloads are identical"), and `cmp` says the files are byte-identical. `notes/` holds only the template README. There is no `profile_latest.json` or `verify_latest.json` in the worktree. This iteration is a no-op, so there is no claim to check against the notes.

## 2. Is the latency change real?
There is no change. I re-profiled the unedited NEFF myself (3 repeats): 1185.551 us median (1182.795 to 1186.940, spread 0.35%). The recorded baseline is 1184.963 us (spread 0.145%). The difference is +0.05%, which is noise. **Noise floor: about 2-4 us (0.15-0.35%), so claim nothing under about 7 us.**

## 3. Safety
Nothing was edited, so nothing was loosened. The baseline verifier error is output_0 7.688e-05, 11% of the 0.00071 limit.
`semaphores.py` on the unpacked NEFF reports supply >= demand everywhere. I only inspected the tail of its output.

## 4. Candidate edits for iter-1 (ranked)

### Basis
This kernel is the same nkilib `gdn_cte` as `neff_doc/experiments/gdn_cte_b32_s128_d128_lnc1`. That loop took 1351 to 1166 us (-13.7%) in 7 iterations, and most of the gain came from DRAM-staging fusion. I checked that the structure is present here:
- `def.json` has 160 `tmp-buf` DRAM vars.
- The only data queue is `qSPSpillReload0` (16 rings, semaphores 24-39). The dynamic queues use semaphores 8-23, 40-55 and 56-71, so **72-87 look free**. Confirm against `runtime_semaphore_count`.
- `SP0.json` has 222 queued `SB->SB` transfers (op unset, i.e. copy or transpose).
- Classifying the 160 vars by who touches them gives **128 vars with one ACT DGE save and one SP DGE reload each, and 32 vars with one ACT DGE save and one Pool DGE reload each.**

This differs from s128, which had 2 saves per var in some groups. The descriptor shapes therefore have to be re-derived. Do not copy the s128 constants.

Profile, 1184 us span, busy time by opcode:

| Engine / opcode | Slots | Busy |
|---|---|---|
| PE `MATMUL` | 2048 | 457 us |
| SP `DMA_TRANSPOSE` | 320 | 392 us |
| DVE `TENSOR_TENSOR` | 928 | 188 us |
| ACT `CAST` | 896 | 177 us |
| PE `LDWEIGHTS` | 1742 | 170 us |
| Pool `DMA_DIRECT2D` | 256 | 169 us |
| ACT DGE save `DMA_DIRECT2D` | 160 | 91 us |
| SP DGE `DMA_DIRECT2D` | 128 | 74 us |

Engine busy time is at most 40% on every engine, so the limit is dependence-chain latency, not throughput. The s128 loop also concluded this. `hbm_read_bytes` / `hbm_write_bytes` (6.0 / 7.1 MB) and `spill_*` of 3.9 / 4.5 MB confirm the staging traffic.

| # | Edit | Mechanism | Ceiling |
|---|---|---|---|
| 1 | **Replace the DRAM staging round-trips with direct SB->SB queued transfers on a new queue.** | s128 iter-1 and iter-2 | see below |
| 2 | **Neutralise the 160 dead ACT DGE saves.** | s128 iter-2 step 3 | about 91 us ACT busy; the on-path part is unknown |
| 3 | **Remove the relay `EVENT_SEMAPHORE`s from edit 1 by retargeting the queue's completion semaphores to the consumers.** | s128 iter-4 | about 2% in s128 |
| 4 | **Fuse the W-path hops.** | s128 iter-5 to iter-7 | about 0.9% in s128 |
| 5 | **Pool DGE input loads to queued DMA.** | review candidate in s128 | at most about 3% on s128's measurement; unmeasured here |

#### 1. Staging fusion (Engines: ACT, SP, Pool; members: `sg00/SP0.json`, `SP0.bin`, `Pool0.bin`, `Activation0.bin`, `def.json`)
- **What changes.**
  - Add `qSPSpillReload1` (16 rings, semaphores 72-87) to `def.json`. Transfer ids are per queue, so no existing wait value moves.
  - For each var, append a queued `SB->SB` transfer in `SP0.json`. Its source is the save's SBUF source (`So + i*S`, per the s128 iter-1 recipe). Its destination is the reload's SBUF destination. For the 32 Pool vars the reload is fp32 to bf16, which needs `op:"cast"` (s128 iter-2 section 2.2, bit-exact).
  - Add a `PSEUDO_DMATRIGGER` on SP that waits for the save's own producer wait (the DVE/ACT semaphore the save waited on), not the save's completion.
  - Turn the old DGE reload into an `EVENT_SEMAPHORE` that repays its +16 through the extended events block.
  - Start with the 32-var Pool group or a 16-var subset, as an in-iteration probe.
- **Mechanism.** s128 iter-1 and iter-2 notes: a DGE pattern allows only one partition-crossing dimension, but a queued descriptor does not, so the 2-D composite is expressible only as queued. A descriptor-only change measured +0.3% (nothing). The wait retarget is what removes the hop.
- **Ceiling.** In s128, fusing the 32 vars whose reader was already queued gave -38 us. Fusing all 160 vars and killing the dead saves gave -143 us in total (1351 to 1208 us), about 10.6%. Here: 160 vars, and per-hop savings of about 3.7 us (581 ns save issue + 1.5 us save transfer, replaced by a roughly 2 us queued hop). If every hop were on the path, the bound is roughly 160 x 3.7, about 590 us, which is clearly an overestimate. **Realistic ceiling: 60-140 us (5-12%), by analogy to s128 only. Unmeasured on this NEFF**; the critical-path walk (see Unknown) settles it.
- **Risk.**
  - A wait that is too weak races: the reload reads SBUF earlier than before. The verifier uses one fixed input set, so it may pass and still be wrong. Run `verify.py --runs 3` and require bit-identical outputs.
  - A wait retargeted to a producer that was not already satisfied before the old trigger ran **deadlocks** (s128 iter-1: NRT_TIMEOUT). `semaphores.py` cannot see this, because it only checks supply >= demand. The wait must name a producer that was already satisfied before the relaxed instruction ran in the baseline.
  - A removed wait also dropped the implicit ordering of every waitless slot after it on SP. Re-install the old wait on the first waitless follower.
- **Cheapest falsifier.** Apply only the 32-var Pool group (or 16 vars) and run `profile.py --repeats 5`. If the median does not move by more than 7 us, the staging hops are not on this kernel's critical path and the lever is dead in minutes. If it works, widen to the 128 SP vars.

#### 2. Dead ACT DGE saves
- **What changes.** After edit 1 no descriptor reads the 160 `tmp-buf` vars, so the 160 ACT DGE saves are dead. Replace each with an `EVENT_SEMAPHORE` that repays its +16 via `SEM_ADD_IMM_COMPLETE` (about 40 ns vs about 580 ns).
- **Mechanism.** s128 iter-2 step 3: -27 us after only 94 saves were killed. Shrinking the save (`num_elem` to 1) is **not** a lever, because the per-slot cost is fixed (s128 iter-1 section 4.4).
- **Ceiling.** At most 91 us of ACT issue time (profile: ACT `DMA_DIRECT2D` 160 slots). The on-path part is unknown, because it depends on edit 1.
- **Risk.** The semaphore count must stay the same so the other waiters are not broken. `semaphores.py` checks this. The verifier would catch wrong data, but not a missing +16 that only hangs under different timing.
- **Falsifier.** Do it for 16 saves after edit 1 and look at the median.

#### 3. Relay removal
- **What changes.** After edit 1, retarget the new queue's completion semaphore set to the consumers' semaphores, so no SP relay `EVENT_SEMAPHORE` stands in the in-order SP stream. This needs a ring relabel (s128 iter-4).
- **Why.** The SP relays block SP's in-order stream (130 us in s128).
- **Gain.** -2.3% in s128.
- **Risk.** Ring order must equal trigger order. Check it with `semaphores.py` and `verify.py --runs 3`.

#### 4. W-path hop fusion
- **What changes.** Fuse the SP `DMA_TRANSPOSE` chain that feeds the W_t matmul (SP `DMA_TRANSPOSE` is 320 slots, 392 us busy here), following s128 iter-5/6/7 (inserted ACT `CAST` into bf16 scratch, then one 4-descriptor permute+transpose).
- **Ceiling.** s128 gained -0.9% in total. This is only worth doing if the walk shows the W path is on the critical path here.
- **Risk.** Inserting instructions into a stream (s128 iter-6) is device-verified, but the ordinals of waited-for instructions must not be rewritten (that deadlocks).

#### 5. Pool DGE input loads to queued DMA
- Pool has 256 `DMA_DIRECT2D` slots (169 us busy), of which only about 32 are staging reloads.
- s128 section 3.2 bounded the equivalent at about 38 us, and showed input loads run about 5 groups ahead of use, so it is **probably not on the path**. Keep this last, and only if the walk shows otherwise.

## 5. Closed (in s128; recheck with one probe before relying on them)

| Lever | Number that closed it |
|---|---|
| DGE-side staging fusion (rewrite a DGE reload instead of a queued one) | `max_abs_err` 1.155e-2 (1050% of limit), deterministic mis-map; a DGE pattern allows only one partition-crossing dim |
| Merging affine runs of identical instructions | s128: 57 removable slots, about 10 us at best; later device test about 0% |
| Dedup of PE `LDWEIGHTS` | 0 of 1877 consecutive pairs identical |
| Shrinking dead DGE saves | 1313.7 vs 1312-1313 us (0%) |
| Reordering SP triggers past an `EVENT_SEMAPHORE` that updates nothing | gives NaN |
| Retargeting `W_t` transposes to the real producer | 0% |
| Engine rebalancing / compute moves | all directions about 0%; wall responds to latency removed from the chain, not to busy time |
| Queued DMA reading PSUM | NRT_INVALID at load |
| Queued DMA doing transpose + cast in one descriptor | closed on device |
| `fabric_path` or queue type as a knob | static: no effect |
| Ring throttle / ring depth | 0 |
| DVE `TENSOR_TENSOR` merges | wrong answer or hang |

## 6. Unknown (would reorder the list)

1. **Critical-path decomposition of this NEFF.** Run the s128 iter-1 section 1 walk (backward walk over S[3..6] and DGE and queued semaphores, with slot index == `compiler_pc`) on `prof/Instruction.parquet` + `SemaphoreUpdate.parquet`. The question is how much of the 1184 us is DMA hop latency at the 160 staging sites. If it is under 100 us, demote #1-#3 and report the kernel near-irreducible. If it is over 300 us, #1 is the whole iteration.
2. **Whether semaphores 72-87 are free** and whether 80 rings are accepted. Check `runtime_semaphore_count` in `def.json` and load the NEFF.
3. **Descriptor shapes of the 128 SP-reload vars and the 32 Pool-reload vars.** Print `SP0.json` and the DGE slots. This kernel's 1:1 save-to-reload shape differs from s128.
4. **Whether the 320 SP `DMA_TRANSPOSE` slots are all W_t.** This decides #4.

If the walk shows hop latency is a small fraction, the honest conclusion is that this NEFF is close to irreducible; do not invent further levers.

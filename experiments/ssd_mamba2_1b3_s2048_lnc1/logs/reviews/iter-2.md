# Review: iter-2 (ssd_mamba2_1b3_s2048)

## Summary

The iteration-2 edit is a small cleanup of the Activation stream. It is a gain of 2.8 us on a 5129 us kernel, which is inside the run-to-run spread, so it is noise. `notes/` is empty, so there is no claim to check the edit against. The real result is the one already committed: 7154.9 -> 5130 us from iteration 1.

## 1. What changed

**Baseline to current**
- Only `sg00/Activation0.bin` differs. The other 36 members are byte-identical.
- It goes from 15397 slots to 14074 slots, and 15386 of the baseline's slots differ because the stream shifted.
- This matches the iteration-1 commit message: 1020 ACTIVATE slots were deleted, and the exp(x)+reciprocal pairs became `activation_func` 7 (exp) with `scale_value_imm_fp32 = -1.0`.
- A table-load slot (`ACTIVATION_TABLE_LOAD`) is replaced by an ACTIVATE, with `activation_func` 7 -> 128 on some neighbouring slots.

**Committed HEAD (iteration 1) to working tree (iteration 2)**
- `kernel.neff` is modified against HEAD. Again only `Activation0.bin` differs, with 14377 -> 14074 slots, so 303 slots were removed.
- Visible edits, for example slots 141-144:
  - merged `num_elem` 64 -> 128
  - `events_update_mode` NONE -> SEM_INC_COMPLETE and a bumped `events_semaphore_value` (63 -> 64) to keep counts right
  - a few ACTIVATEs turned into plain 1-element ones
- This is a slot-merge and dedupe pass on the ACT stream. Nothing was written in `notes/`, so the intent cannot be checked, and the commit message does not cover it.

`profile_latest.json` has neff_sha256 `1bd995d8...`, which matches the working `kernel.neff`.

## 2. Is the latency change real?

I measured the committed HEAD NEFF again with `profile.py --repeats 5`.

| NEFF | median (us) | min-max (us) | spread | ACT busy (us) |
|---|---|---|---|---|
| baseline | 7154.9 | 7154.9-7156.4 | 0.02% | 6283.9 |
| HEAD (iteration 1) | 5129.5 | 5125.8-5132.8 | 0.14% | 3769.3 |
| working tree (iteration 2) | 5126.7 | 5123.2-5130.6 | 0.15% | 3669.7 |

- The iteration-2 gain is **-2.8 us (-0.05%)**. The spread is about 7 us, so this is noise.
- The iteration-1 gain of 28% is real. Its spread is 0.02%, against 2030 us.
- ACT busy dropped 100 us (-2.6%) with no wall-clock change. **Reducing Activation busy time does not by itself shorten this kernel.** Candidate 1 below is conditional on this.

## 3. Safety

- `semaphores.py` on the unpacked NEFF:
  - `max_wait <= total_inc` for every semaphore.
  - Sem 4 (ACT self-completion) has 7360 incs and a max wait of 7360.
  - There are no over-waits.
- `launch_rules.py` passes.
- Verifier error:

| output | baseline | HEAD / working | limit used |
|---|---|---|---|
| output_0 | 5.122e-7 | 5.234e-7 | 84% |
| output_1 | 3.166e-7 | 3.334e-7 | 54% |

  - The exp(-x) rewrite changed the arithmetic slightly (1/exp(x) vs exp(-x)).
  - There is about 16% headroom left on output_0. **Further numerics-touching edits can fail the verifier.**
- The merged `num_elem` 64 -> 128 and the semaphore renumbering in iteration 2 were not checked by the verifier on more than one fixed input set. Any further ACT merging has to keep the semaphore counts by hand.

## 4. Where the time goes (current NEFF, 5127 us)

**Utilisation**
- Engine busy: ACT 71.6%, DVE 52.7%, PE 39.3%, Pool 40.4%, DMA 36.3%, SP 15.6%.
- No engine is saturated. The kernel is 1024 (head, chunk) units at about 5.0 us each.
- That is a dependency and latency chain across engines, not a throughput limit.

**ACT contents**
- 14.4k Scalar instructions.
- `ACTIVATE` is 10957 instructions. The fn=`COPY(scale/bias enabled)` ones account for the largest part of Scalar time: 6865 instructions at about 445 ns each, about 3.05 ms of summed duration.
- Plain `COPY` is 1020 instructions at 162 ns.
- `SOFTPLUS` is 3068 instructions at 197 ns.
- Many are 1-element ACTIVATE pairs (fn 7 with scale -1.0, then +1.0), about 350 ns each. Per the decoded Activation0.bin there are about 3000 of these, about 1.1 ms. They look like the serial per-chunk decay chain.
- 64- or 128-element ACTIVATEs with `imm_src/scale_src = POINTER_IMMEDIATE` and fn 128 (scale from SBUF, bias imm) number about 5430 and cost about 1.86 ms.
- **Caveat:** the profile `pc` does not line up with the slot index in this stream. Mapping the profile to slots gave inconsistent durations. Treat the per-category numbers above as approximate. `decode -> profile` joins were not reliable.

**DMA issue is serial software DGE**
- `PSEUDO_DMA_DIRECT2D` costs about 0.55-0.65 us each on the issuing engine:
  - Pool: 3169 x 655 ns = 2.08 ms
  - Scalar: 1088 x 560 ns = 0.61 ms
  - Sync: 1088 x 542 ns = 0.59 ms
- The ~5.3k triggers per 1024 units are spread unevenly across the three engines that may issue them.

**DVE**
- STREAM_SHUFFLE 8192 x 147 ns plus LOAD_MASK_SELECT 8192 = 1.25 ms.
- TENSOR_TENSOR 4128 x 203 ns = 0.84 ms.
- TENSOR_TENSOR_SCAN 1024 x 464 ns = 0.47 ms.

**PE**
- LDWEIGHTS 7233 x 305 ns and MATMUL 7233 x 411 ns.

**HBM**
- Total traffic is about 143 MB, of which 35.6 MB is spill save/reload. That is about 28 GB/s average, nowhere near bandwidth-bound.

## 5. Candidate edits for iteration 3 (ranked)

**Do first (a minutes-long probe, applies to candidates 1 and 2): find the real wall-clock limiter.** Iteration 2 showed ACT busy -100 us gives 0 wall. Before any move-to-another-engine edit, do a timing-only probe that does not need to pass verify:
- Shrink `src_mem_pattern_num_elem` of about 1000 of the 64-elem fn-128 ACTIVATEs (wrong numbers, same semaphore counts) and measure the wall time.
- If wall time does not drop by at least about 30% of the removed ACT time, the kernel is latency-chain-bound on something else (DMA issue, PE->DVE->ACT chain), and candidates 1-2 are dead.

| # | Engine / member | Change | Mechanism | Expected gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | ACT -> DVE/Pool, `Activation0.bin`, the ~5430 fn-128 POINTER_IMMEDIATE-scale ACTIVATEs (64/128 elem) | Replace them with `EVENT_SEMAPHORE` on ACT (keeps the sem-4 count). Insert a `TENSOR_TENSOR`/`TENSOR_SCALAR` multiply on DVE or Pool with a zero-stride broadcast src1, with waits on the producer. Consumers wait on the new engine's sem at the new count. | `14-optimization-playbook` 14.3(3); precedent: gdn_tkg iteration 2, 55.6 -> 44.6 us, 36 scale-pointer ACTIVATEs moved to DVE as bit-identical TENSOR_TENSOR | Ceiling: ACT summed time ~1.86 ms. DVE (2.70 ms busy) and Pool (2.07 ms, mostly DGE issue) have maybe 1.5 ms combined headroom. Realistic: a few hundred us only if ACT is on the critical chain. Iteration 2 argues it may not be. | Rewiring sems wrongly passes one input set and races later. Rounding: fp32 multiply should match, but err is at 84% of limit. | The shrink-num_elem probe above. If wall time is unchanged, drop it. |
| 2 | Scalar-issued `PSEUDO_DMA_DIRECT2D` (1088 x 560 ns = 0.61 ms) -> SP stream (15.6% busy) | Re-home the Scalar-queue DGE triggers onto SP (a legal issuer per `08`/`09`). Keep each DMA's waits/updates and semaphores. | `09-dma.md` 9.4: SP, Activation and Pool may all issue DGE; `08-pseudo-instructions` | Ceiling: 0.61 ms of ACT stream time. The same caveat applies: only helps if ACT is binding. | Queue ids and DGE `semaphore=` numbers belong to a given issuing engine's ring. Edit only the slot's engine placement, which has not been established on this device. | Move 20 triggers, check the run completes and verify passes, then check wall time. |
| 3 | Pool DGE issue, `Pool0.bin`: 3169 triggers x 655 ns = 2.08 ms | Spread some Pool-issued DGE DMAs to idle SP (and, if ACT is not binding, to ACT). Alternatively merge adjacent same-shape transfers across chunks into one (fewer, larger descriptors, 14.3(4)). | Same as 2; merging is also in 14.3(4) | Ceiling: Pool stream is 40% busy but serial on a ring; if the DMA ring is the pacing chain, rebalancing could cut Pool-ring latency. Unknown. | Same as 2, plus merged transfers must preserve the HBM address strides and SBUF layout (verifier would catch a wrong layout; a race could pass.) | Time only: how much does wall change if 10% of Pool triggers go to SP? |
| 4 | DVE `STREAM_SHUFFLE`/`LOAD_MASK_SELECT` (8192 pairs, 1.25 ms) | Batch the 8 shuffle pairs per unit into fewer instructions | `14-optimization-playbook` 14.3(4) | Ceiling: at most ~1.2 ms of DVE, but DVE is 52.7% busy and not the limiter; unknown whether it is on the critical chain. | High: touches data layout. | Not worth starting before the probe shows DVE is critical. |

I could only rank 1 and 2 from the NEFF; 3 and 4 rest on weaker evidence.

**Closed** (do not redo):
- exp/reciprocal table reloads: 2040 of 2046 table reloads and 1020 slots deleted in iteration 1, 7154.9 -> 5130 us. The remaining 6 `ACT_TABLE_LOAD` cost 7.7 us in total.
- Trimming ACT slot count alone: iteration 2 removed 303 more slots and 100 us of ACT busy for -2.8 us wall (inside 0.15% spread).
- HBM bandwidth: 143 MB over 5.1 ms is far below peak; spill traffic is not the limiter.

**Unknown** (would change the ranking):
- What the actual critical path is. Settle with the shrink-num_elem timing probe above, or a per-unit critical-path walk of `evt_wait_time_ns` and sem producers in `prof/Instruction.parquet`.
- Whether the profile `pc` can be mapped to slot index. Without it, per-slot durations are unreliable. Settle by matching a handful of unique instructions (e.g. the 6 table loads) to their slot indices.
- Whether a DGE trigger's issuing engine can be changed without touching the ring/semaphore assignments. Settle with the 20-trigger test in candidate 2.
- Whether the 1-element fn=7 (-1.0/+1.0) pairs (~1.1 ms of ACT) can be merged, for example computing both signs from one source in one 2-element instruction. Settle by checking whether the +1 and -1 variants read the same source address.

## 6. Bottom line

- Do not accept iteration 2 on its own merits: it is within noise. The net result from the baseline is still the iteration-1 win, and the working tree is not worse than HEAD.
- Writing `notes/` for the next agent is the biggest missing piece.
- Run the timing probe first. If it shows ACT is not binding, the remaining time is a cross-engine latency chain and the remaining NEFF-level levers are small.

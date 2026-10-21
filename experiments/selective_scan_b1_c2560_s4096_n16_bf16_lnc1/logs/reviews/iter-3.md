# Review of iter-3 (working-tree `kernel.neff` vs HEAD e7f3a7a = 6819.0 us; baseline 7919.96 us)

## 1. What changed (diffed against HEAD, since HEAD already holds iter-1/2)
`neff_diff HEAD baseline`: 3 members changed (Activation0, DVE0, Pool0); PE0, SP0 and all data members identical.

| member | slots HEAD -> now | change |
|---|---|---|
| Pool0.bin | 6407 -> 5283 | Pool `COPY` 2488 -> 1573 (-915). Surviving COPYs have `src/dst num_elem [1,1,1,1] -> [1,2,1,1]` and `src step [1,3072,1,1]`, so adjacent single-element copies were fused into one 2-element strided copy. Pool `TENSOR_TENSOR ADD` 2565 -> 2414 (-151). |
| DVE0.bin | 13886 -> 14037 (+151) | The 151 ADDs that left Pool were moved back to DVE (`S[6]` incrementers unchanged at 5428 TT, 2560 SCAN, 186 MEMSET). Waiters on S[6] drop 14391 -> 13418 (Pool-side waits attached to the removed instructions). Every DVE slot shifts, so 13974 slots differ. |
| Activation0.bin | 252 slots | Only `events_semaphore_value` (46 slots) and a few wait-index and address fields, retargeting waits on S[3] after the COPY count changed. |

Semaphore S[3] (Pool) now has 1573 COPY + 2414 ADD = 3987 increments. HEAD had 5053.

**Consistency with the notes:** `notes/` contains only the template README, so this iteration is undocumented. The latest commit message describes the previous iteration (S[6] renumbering). The actual change is a COPY fusion, a partial rebalance of ADDs from Pool to DVE, and a rewrite of S[3] wait thresholds. The next agent must write this down.

## 2. Is the latency change real?
`profile_latest.json` gives a median of 6712.06 us, min 6711.02, max 6712.39 (spread 0.02%, about 1.4 us), 5 repeats. HEAD's committed number is 6819.0, so the delta is -107 us (-1.57%), about 75x the spread. It is real, not noise. Against the baseline it is -15.25%.

The sampled profile spread is the run-to-run spread within one process. I did not re-profile HEAD in this session. The 6819.0 figure comes from the commit message.

## 3. Safety beyond the verifier
- The verifier passes. output_0 max_abs_err is 0.125 (83% of the 0.15 limit), cosine 0.99994. output_1 error is 1.8e-4. I could not tell whether 0.125 is unchanged from baseline, because only the latest verify JSON is present. The next agent should record the baseline value. The claim of bit-exact arithmetic is plausible, since no ALU op or operand dtype changed, but it is unproven here.
- `semaphores.py` on an unpacked copy: for every semaphore, max_wait <= total_inc. S[3] is 3986 vs 3987 and S[6] is 8174 vs 8174. The +1 on S[3] also exists in HEAD, and baseline had equality on S[3] (2488), so it is not new. No semaphore is under-incremented.
- **Wait removal.** S[6] waiters fell by 973 and S[3] waiter count is unchanged (4339). Removal of waits that lived on deleted instructions is benign. Waits are not removed from surviving instructions, but I did not prove this slot by slot. A fused COPY must wait for the later of the two producers it merges. Its source is the final column of a scan, and a scan increments S[6]. If the fused copy kept the first copy's S[6] threshold, it could read the second column before the scan wrote it. The verifier on one fixed input could still pass if the timing happens to hold. **Cheapest check:** for every fused COPY, confirm its S[6] wait value is >= that of both original copies, and that no consumer of S[3] now waits for a smaller value than before.
- The 151 ADDs moved back to DVE are a placement change only. Check the same wait/increment ordering rule on them.

## 4. Where the time is now (from `prof/Instruction.parquet`, current NEFF)
- DVE compute instructions (TT 5586 x ~0.67 us, SCAN 2560 x ~1.27 us, MEMSET 200 x 0.41 us) are issued back-to-back. The sum of their durations is 7156 us, minus ~592 us of pipeline overlap, plus ~146 us of positive gaps, which fills the 6710 us span. DVE is the critical engine.
- Per state, DVE does 1 SCAN + 2 broadcast multiplies, and 2560 states x (1.27 + 2 x 0.67) = 6.68 ms. That is nearly the whole runtime. Only ~466 TT (ADDs) and 200 MEMSETs remain as non-essential DVE work.
- Pool is 83% busy: 2414 ADDs at ~1.0-2.2 us each (4706 us), 1587 COPYs (426 us), 680 DMA_DIRECT2D (448 us). Pool gaps total 1168 us, spread across 518 gaps > 1 us.
- PE is 44% busy and ACT 28% busy. Both are mostly idle.
- Pool ADD duration is bimodal: 709 ops at 1.0-1.8 us and 1704 at ~2.17 us. Every sampled Pool ADD overlaps a DVE compute instruction.

## 5. Candidate edits for the next iteration (best first)

| # | Edit | Mechanism | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Move DVE `MEMSET` (200 slots, value=0, dst `fp32@0x6800` 512 elem) to Pool or ACT.** DVE0.bin MEMSET slots, S[6]-incrementing. Re-issue on the other engine with a new semaphore or reuse S[3]/S[4], then retarget the first y_accum ADD's wait. | The memset is y_accum zero-init, off the scan chain. Same renumbering technique as iter-2 (HEAD message). | At most 200 x 0.485 = 97 us gross (82 us from profile sum). Net 60-80 us (~1%) if the memset is cheap on its new engine. Bounded by DVE MEMSET total 82.3 us. | Medium. An engine must support MEMSET (Pool does for NKI memset; unconfirmed in the NEFF ISA table). A missed wait gives a stale y_accum, caught only if the race is hit. | Decode the Pool opcode table in `isa.py` for MEMSET. Then make the edit on one chunk and diff the output on the fixed inputs. |
| 2 | **Replace "MEMSET 0 + first ADD" with a single `TENSOR_COPY`/ADD-with-zero eliminated: write the first product straight into y_accum.** Fold the 200 memsets and 200 first ADDs away (the first ADD in each 16-state group sees `0 + x`). | `0 + x` is exactly `x` in fp32, so numbers are bit-identical. | Removes 200 memsets (82 us on DVE) and 200 Pool ADDs (~430 us Pool). DVE ceiling ~80 us. Pool gets slack, which allows more ADDs to move off DVE. | Medium. The multiply writes to `0x6800` instead of the temp, so the temp-buffer free/WAR semaphore chains change. | Pick one group, change the 16-state head only, and check the output. Verify the S[3]/S[6] waits cover the multiply's new dst. |
| 3 | **Move more of the ~466 remaining DVE ADDs (those still on `TENSOR_TENSOR ADD` on DVE) to Pool** (rebalance, the same hand-off HEAD used). | DVE-eq model: DVE 6.56 ms, Pool 5.58 ms non-wait. Moving x ADDs gives DVE -0.67x and Pool +(1.0 to 2.17)x. Pool cost depends on which of the two modes the ADD lands in. | At most 466 x 0.67 = 312 us DVE if Pool keeps up. Pool headroom is ~1.1 ms of gaps minus waits, so realistically 150-250 ops move: **-100 to -170 us**. Iter-3 moved 151 ADDs the other way and still gained, so the optimum is non-monotonic. Treat this as an empirical sweep. | Low-medium. Same placement-only change as iter-1/2 (arithmetic order preserved). | Move 100 ADDs and re-profile. If Pool becomes the critical engine (Pool span > DVE), stop. Binary-search the count. |
| 4 | **Find out why Pool ADD is 1.0-1.8 us for 709 ops and 2.17 us for 1704 ops, then steer more ADDs into the fast mode.** Candidate causes: SBUF port contention with a concurrent DVE op, or address alignment/bank of `src0`/`dst` (`0x6800` and `0xa000` vs `0x7000`). Edit: change the temp operand address field of the slow ADDs (and the producing multiply's dst) to a buffer that behaves like the fast ones. | Known only as an observation in this profile. No documentation or on-device experiment in `notes/` establishes the cause. | Up to 1704 x (2.17 - ~1.3) = ~1.5 ms of Pool time. That lowers Pool load and permits candidate 3 to move far more ADDs. DVE gain is bounded by the ~312 us in candidate 3. | High until the cause is known. A buffer move is an allocation change that one input set cannot validate. | Join `Instruction.parquet` Pool ADDs on `operands` (src/dst addresses) and `duration_ns`, and see if address alone separates the two modes. No device run needed. |
| 5 | **Fuse the remaining 1573 Pool COPYs** (one 2-element copy per state now). Group several states' 2-element copies into one wider strided COPY. | Same fusion already used (915 removed, S[3] thresholds remapped). | Pool COPY total is 426 us; fusing gains only Pool time, not DVE time. Worth it only if Pool turns critical after candidate 3. Ceiling: 0 us DVE-side. | Medium, same hazard as section 3: fused COPY needs the max of its producers' waits. | Do this only after 3 shows Pool is the limit. |

Candidates 1-3 compete for the same ~312 us DVE-ADD+MEMSET headroom. **The total remaining upside is about 150-250 us (2-4%), below ~6460 us.** Beyond that DVE is at the SCAN + 2 mults floor of ~6.68 ms (see section 4).

## Closed (measured, do not redo)
- Moving DVE ADDs to Pool is not monotone. 2214 moved (7919.6 -> 6886.3), the last 351 renumbered (-> 6819.0), then 151 moved back with COPY fusion (-> 6712.1). Pool alone cannot take all the work.
- Self-wait guard of 585 ns on S[6] (iter-1 variant) was replaced by renumbering. Do not reintroduce it.
- DVE is not idle: positive gaps total 146 us out of 6710 us. Reducing semaphore-wait latency on DVE gains at most that 146 us (91 us of it before TT).
- PE (44%) and ACT (28%) cannot take the SCAN or the two per-state multiplies. SCAN is DVE-only, and the multiplies are elementwise in both channel and time (ACT `scale` is per-partition only).

## Unknown (would change the ranking)
- The cause of the bimodal Pool ADD duration (1.0-1.8 vs 2.17 us). Settled by the address/operand join in candidate 4, then one device run moving an ADD between layouts.
- Whether Pool and DVE share an SBUF port such that moved ops slow DVE. Measure the DVE TT duration (656-678 ns) while a Pool ADD overlaps vs alone. Everything sampled overlaps, so alone-samples must be found first.
- The baseline `output_0` max_abs_err. If it is far below 0.125, something is not bit-exact. Run `tools/verify.py` on `baseline/kernel.neff`.
- Whether bf16 operands for the DVE multiplies would double DVE throughput. That changes numbers (error is already at 83% of the limit), so I rank it out of scope unless the limit is confirmed to allow it.

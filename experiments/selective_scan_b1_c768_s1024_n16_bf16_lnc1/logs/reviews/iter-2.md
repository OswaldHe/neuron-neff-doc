# Review, iter-2 (selective_scan b1 c768 s1024 n16 bf16)

Reviewed: `kernel.neff` at commit 6e7af17 (iteration 1's NEFF, 602.7 -> 516.5 us). `notes/` holds only the template README, so there are no agent notes to compare against; the commit message is the only claim.

## 1. What changed

`neff_diff.py` reports 3 of 34 members changed. Header and 31 other members are identical.

| member | slots | change |
|---|---|---|
| `sg00/DVE0.bin` | 1204 -> 1216 (+12); 1154 differ | 170 `y += Cs` fp32 512-elem `TENSOR_TENSOR_ARITH_OP` ADDs turned into `EVENT_SEMAPHORE`. Many S[3] hand-off waits (`events_extended_wait_*` on S3) added to ES slots. 12 fresh ES slots inserted before the group-boundary deadlines. The diff is large because every slot after an insertion shifts. |
| `sg00/Pool0.bin` | 296 -> 389 (+93); 382 differ | The 170 ADDs inserted as `TENSOR_TENSOR_ARITH_OP`. The pairwise-merged carry COPYs removed 77 slots. S[3] increments follow the new ordering. |
| `sg00/Activation0.bin` | 234 -> 234; 17 differ | Only `events_semaphore_value` changed, for example 14 -> 22 and 43 -> 68. These are S[3] waits renumbered because Pool now increments S[3] more often. |

This matches the commit message. Semaphore accounting confirms the structure:
- `semaphores.py` on the new NEFF shows S3 max_wait 282 = total_inc 282, with sources `Pool:COPY` 112 and `Pool:TENSOR_TENSOR_ARITH_OP` 170.
- S6 640/640, S5 374/374 and S4 194/194 are all balanced.
- The profile counts agree: Pool TT 170 and COPY 115; DVE TT 430 (baseline 600).
- The DVE stream has 146 ES slots with an S3 extended wait, which fits "170 ADDs, mostly one wait each, placed as late as legal".

## 2. Is the gain real?

Yes.
- Baseline median is 602.637 us, spread 0.052% (about 0.3 us).
- Latest median is 516.796 us, min 516.675, max 517.109, spread 0.084%.
- My own 1-repeat profile gave 517.07 us.
- The win is 85.8 us (14.2%), roughly 250 times the noise. The loop is not chasing noise here, and a gain of about 1 us or more would still be distinguishable.
- Mechanism is visible: DVE busy 592.4 -> 501.7 us, Pool busy 101 -> 415 us, PE/ACT/SP unchanged. The Pool ADD duration is a median of 2162 ns against about 694 ns on DVE.

## 3. Safety

- **Numerics:** `verify_latest.json` gives output_0 max_abs_err 6.245e-02 (83% of the 0.075 limit) and output_1 4.27e-05. These are identical to the baseline, which is consistent with the same fp32 adds run in the same order, only relocated. No error moved.
- **Sync accounting:** `semaphores.py` reports no wait that exceeds its supply on any semaphore.
- **Not covered by that tool:** it cannot see WAR or cycle hazards. In the sibling c2560 experiment the hand-off wait had to sit before the first DVE slot that reads or overwrites `Y` or `Cs`, and the Pool TT is not allowed to run past the 3-deep `Cs` rotation. The agent says it placed waits "as late as legal" using deadlines, which is the right method, but there is no written proof.
  - Not verified here: only 1 verifier run was done (`"runs": 1`). I did not re-run it multiple times.
  - A WAR race would show up as nondeterministic output, not as a deterministic failure. **Before the next edit is stacked on this one, run `verify.py --runs 3+` and compare outputs bit-for-bit.** The sibling check was "bit-identical across runs".
- **Risk assessment:** the profile shows the Pool TT gaps are real waits, so the hand-offs are doing something. I found nothing wrong, only that this is unproven by repetition.

## 4. Where the time goes now (from `prof/Instruction.parquet`, 517 us)

- **DVE:** 501 us busy (97%). Union idle is only 12 us over 48 gaps. The two gaps over 1.5 us are a 3.8 us prologue wait on the first input DMA and one 1.9 us gap at 100.9 us.
- **DVE composition (sum of durations):**

| op | n | mean ns | total us |
|---|---|---|---|
| TENSOR_TENSOR | 430 | 694 | 298 |
| TENSOR_TENSOR_SCAN | 192 | 1271 | 244 |
| EVENT_SEMAPHORE | 577 | 58 | 34 |
| MEMSET | 24 | 302 | 7 |

- **Pool:** 415 us busy, 106 us idle in gaps (88 us of it in 37 gaps over 1 us). 170 TT at 2008 ns mean (median 2162, min 1258).
- **Others:** PE 42%, ACT 28%, DMA 20%, SP 9%.
- **DVE is still the critical engine.** Its work is 192 scans, 192 `dtx*B_full` MULTs, 192 `C_full*state` MULTs, 22 remaining `y+=Cs` ADDs, 24 other small TT, and ES overhead. The scan is not movable (the sibling experiment showed the scan carry chains across access-pattern rows, so scans cannot be merged and only DVE runs them).
- The DVE floor with the current work split is about 500 us. Further gains need either less DVE work or a cheaper per-instruction cost.

## 5. Ranked candidates for the next iteration

Calibration carried over from `experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/logs/notes/iter-1.md`. That kernel is the same code at a different shape, so check each claim on this NEFF.

| # | Candidate | Engine / member / slots | What changes | Mechanism, where established | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Verification debt first**: re-run `verify.py --runs 5` and bit-compare outputs; then remove the ES slots that carry neither wait nor update. | DVE0 | Delete (or turn into NOP-free removal) any DVE ES slot with `events_wait_mode=NONE` and `events_update_mode=NONE`. Sibling iter-1 found 40 such slots. | Sibling iter-1: a no-wait ES costs 35 ns. Here ES mean is 58 ns, 577 slots, 34 us total. | Ceiling is the ES duration total (34 us) less overlap, so realistically 1-5 us. Gain unknown until measured. | Low if only no-wait slots go. Removing a slot shifts every later slot index, so branch targets and S6 renumbering matter (sibling had to renumber 19924 waits for slot deletion; this kernel has no branches inside the loop body but check `COMPARE_BRANCH`). | `isa.py` count of ES with both modes NONE. If it is under 10, drop it. |
| 2 | **Slack-tune the Pool ADD hand-off to fill DVE's 12 us of idle and Pool's 88 us of gaps.** | DVE0 + Pool0, the 146 S3-ext-wait ES slots | Re-place each wait at the latest legal slot using per-ADD deadlines (greedy interval stabbing, as in sibling iter-1). It already claims this; verify by measuring `evt_wait_time_ns` on those ES slots in the profile. | Sibling iter-1 "second most important fact": wait value equal to only the ADD being covered gave 6886 us; over-synchronised gave 7377. | Ceiling is DVE union idle = 12 us (2.3%), unmeasured beyond that. Realistically under 5 us. | Medium: WAR race on the 3-deep `Cs` rotation. | `evt_wait_time_ns` sum on DVE ES slots is 14.1 ms across ES in the table, but most is the baseline's own waits; list the top 10 DVE ES slots by wait and see whether any wait on S3. If none, closed. |
| 3 | **Merge pairs of `C_full*state` MULTs (line 267) into 2-row TTs, planned together with the ADD hand-off.** | DVE0 | Two consecutive 512-elem MULTs -> one `num_elem=[512,2,1]` with strides PSUM +0x1000, state/Cs +12288 B. Deleted slot becomes an ES with its original wait and no S6 update, then renumber all S6 waits in the 5 streams. | Sibling "Pairwise MULT merging": merged duration is 157 + 1024*1.018 = 1190 ns, saving 136 ns per pair; bit-exact. | Ceiling is about 96 pairs x 136 ns = 13 us (2.5%) before side effects. | High and costly: sibling line 224 merge was 214 us slower (PE lockstep) and line 267 merge deadlocked against the hand-off. The saving is small against that history. | Merge one pair, check bit-exactness and the delta; if the whole-kernel change is not at least -0.1 us, stop. |
| 4 | **Pool TT speed**: figure out why Pool ADDs take 2162 ns when isolated Pool TT is 1271 ns. | Pool0 | Measure whether the slow cases are overlapped with DVE reading the same SBUF partitions or are due to the alternating 1-element COPYs (sibling: Pool is about 2x slower under that mix). If copies are the cause, move the remaining 115 Pool COPYs to ACT (probe first; ACT silently drops opcodes it cannot run). | Sibling "third fact": loaded Pool TT 1551 ns vs 1271 ns isolated. | Shortening Pool TT only helps if Pool or the hand-off latency is binding. DVE busy 501 vs span 526 says DVE is the bound, so ceiling is near zero for latency (about 12 us DVE idle). Gain is indirect: it would allow moving more ADDs, but the 22 remaining ADDs all increment S6 (see closed). | Low to try, low value. | Compute from the parquet how many of the 170 Pool TT overlap a Pool COPY neighbour; the min of 1258 ns shows a fast path exists. |
| 5 | **Trim the prologue**: DVE waits 3.8 us at 10.6 us for the first input DMA. | SP/Pool DMA triggers, `Pool0` `PSEUDO_DMA_DIRECT2D` order | Issue the first DMA descriptors earlier or split the first load so DVE starts sooner. | Unknown: no documented mechanism, one 3.8 us gap in the table. | Ceiling 3.8 us (0.7%) at most. | Low-medium. | Check which DMA queue the first DVE wait is on and its length. |

Honest assessment: the remaining DVE time is dominated by work that cannot leave DVE (192 scans = 244 us, 384 MULTs = 258 us). The only substantial lever left (moving more adds) is closed below. **I would not expect more than 1-3% more from NEFF edits.** If the next agent finds #1 and #2 give under 2 us, it should stop rather than invent levers.

### Closed (with the number)

- **Moving S[6]-incrementing DVE instructions to Pool** (the 22 remaining `y+=Cs` ADDs, the 24 MEMSETs, 160 line-182 MULTs): the replacement needs an `S6>=k-1` self-wait costing about 585 ns, more than the 548 ns marginal DVE saving of removing a TT (sibling iter-1: DVE busy went up 6607 -> 6679 us). Not re-measured on this NEFF.
- **ACT running `TENSOR_TENSOR_ARITH_OP`:** sibling probe ran with exit 0 and silently dropped the term (max_abs_err 0.125 -> 12.7). So ACT's 72% idle cannot take TT work.
- **Merging two states into one scan:** the scan carry does not re-seed across access-pattern dim 1 (sibling: bit-exact when split into 2 rows, so it is one linear scan). 244 us of scan is a floor.
- **Merging line-224 MULTs:** sibling measured 7919 -> 7100 us, i.e. 214 us worse than 6886, because PE and DVE run in lockstep on PSUM bank availability.
- **Adding the S[4] (ACT-waiting) ADDs to the moved set** hung the sibling at 70 s; worth under 4 us there.

### Unknown (would change the ranking)

- **Pool reading PSUM.** The sibling probe hung from its own sync mistake, not a verdict. Subject to try is a line-267 MULT whose only wait is `S6>=k`. If Pool can read PSUM at about 2 us, the 192 `C_full*state` MULTs (about 133 us of DVE) become movable, but they increment S6, so the 585 ns guard applies. Measure: one slot, one run.
- **DVE 2x/4x perf mode for 16-bit operands.** `performance_mode` column is empty and the error budget is 83% used, so any dtype change is likely to fail.
- **Whether PE can accumulate `y += Cs` in PSUM** (fp32 matmul rate). Measure one fp32 512-column identity matmul on PE; if over 1.2 us, drop. It also changes the PSUM layout, so it is high risk.
- **Repeatability of the current NEFF** (see section 3): `verify.py --runs 5`, bit-compare. This needs to be done before more edits go on top.

# Review of iter-5 (commit 51c6a8e): 6711.7 -> 6603.9 us

Baseline is 7919.96 us. Unless noted, "previous" means HEAD~1 (fa7f641).

## 1. What changed

I diffed the previous NEFF against this one with `neff_diff.py`. Four members changed. The other 30 are identical, so DMA descriptors, weights and the header are untouched.

| Member | Slots (prev -> now) | Slots differing | What |
|---|---|---|---|
| DVE0.bin | 14037 -> 14139 | 14087 | The prior-to-new slot index shifts after the MEMSET deletions and ES insertions, so the raw differing count is inflated. The real edits are those below. |
| Pool0.bin | 5283 -> 5283 | 4135 | Mostly `events_semaphore_value` rewrites (about 60% of the sampled diffs are only bytes 8..11). A few have `events_wait_idx` changes. |
| Activation0.bin | 3040 -> 3040 | 2038 | Only `events_semaphore_value` (bytes 8..11), by a constant offset. This is the renumbering of S[6] after the MEMSETs were deleted. |
| PE0.bin | 10135 -> 10135 | 4958 | Same: semaphore wait value renumbering only. |

- **Claimed deletion:** 154 y_accum MEMSETs are removed. DVE `MEMSET` count is now 46 in the profile, and S[6] total increments are 8022 (34 MEMSET).
- **Claimed rewrite:** the first Pool `y+=Cs` ADD becomes a 512-element fp32 COPY. Pool COPY is now 1741 (about 973 merged carry copies plus these), and Pool TENSOR_TENSOR is 2260.
- **Claimed moves:**
  - 144 DVE adds hop one slot past their S[3] wait.
  - 68 of them use a Cs buffer moved to SBUF address 62272.
  - 112 S[6] waits move onto a new ES.
  - DVE slots end up +102 net: about 256 added and 154 removed.
  - The diff shows `src0/src1_mem_pattern_start_addr` register fields on the moved slots, which matches the Cs buffer relocation.
- **Consistency:** the notes are empty. `notes/README.md` is the 5-line template, so the only description is the commit message. The slot-level evidence is consistent with that message: mostly semaphore-value renumbering, a few address-register edits, and DVE slot growth by about 102. I did not map each of the 154/144/68/112 counts to slots individually.
- **Process problem:** nothing is recorded for the next agent. The notes directory is the only thing that survives, so the next agent must write per-lever notes.

## 2. Is the gain real?

- **Latest run:** median 6604.44 us, samples 6603.27 to 6604.49, spread 0.019%.
- **Baseline:** median 7919.96 us, spread 0.024%.
- **This step:** -108 us (-1.6%) against a spread of about 1.3 us. That is about 80x the noise floor, so it is real.
- **Check on the latest run:** the profile is internally consistent. DVE busy is 6523 us (98.8%), Pool busy is 5616 us (85%), and the instruction counts match the claimed edits.
- **Caveat:** HEAD~1's samples are not stored in the worktree, so the 6712.4 us figure comes only from the commit message. Re-profile HEAD~1 once if exact attribution matters.
- **Diminishing returns:** gains per iteration are now -1.6% and the scheduling edits are fiddly. Further gains need the structural levers below.

## 3. Safety

- **Verifier:** all pass. `max_abs_err` is 0.125 on output_0, which is 83% of the 0.15 limit. The baseline gives the same number (`verify_baseline.json` is identical: 1.250e-01, cosine 0.999944). So the error is inherent to the kernel and has not moved. Output_1 is 1.8e-4 in both.
- **Semaphores:** `semaphores.py work` shows no "wait exceeds increments" flag on the unpacked current NEFF. Examples:
  - S[6]: max_wait 8022 against 8022 increments.
  - S[3]: 3986 against 3987.
  - S[5] and S[4]: equal.
- **Remaining semaphore risk:** these are totals, so they cannot show ordering races.
  - The risky edits are the 144 "hop past wait" adds and the relocated Cs buffer at SBUF 62272.
  - Moving a consumer one DVE slot past its S[3] wait is a race-sensitive change.
  - The Cs buffer move depends on the old buffer no longer being recycled underneath it.
  - The verifier runs one fixed input set, and timing-dependent races would pass it. Three identical runs reduce the odds but do not prove safety.
  - Cheap extra check: run verify 5 to 10 times and also with `bench/inputs` perturbed, and require bit-identical outputs each time.
- **Bit-exactness:** the error is unchanged against baseline, which is good evidence that the fp32 accumulation order is preserved.

## 4. Ranked candidates

The profile shows both busy engines stalling. This is not a pure throughput limit.

- **DVE** has 855 us in 1451 gaps over 200 ns:
  - 916 gaps (528 us) are an ES followed by TENSOR_TENSOR, i.e. the y+=Cs adds waiting on a hand-off.
  - 390 gaps (240 us) are an ES followed by SCAN, i.e. waiting on PE or ACT.
- **Pool** has 1011 us in 1239 gaps.
- **Work totals:** the DVE and Pool work totals are DVE TT 5586 x 0.67 us, SCAN 2560 x 1.32 us, Pool TT 2260 x 1.98 us, and Pool COPY 1741 x 0.42 us. PE (2.97 ms) and ACT (1.9 ms) are far from the limit.

| # | Candidate | Engine / members / slots | Change | Mechanism and evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Remove the remaining stall behind DVE ES -> TT gaps (916 gaps, 528 us) | DVE0.bin: the TT slots after the gap. Find them from `Instruction.parquet`: `gap>200`, `prev==EVENT_SEMAPHORE`, `opcode==TENSOR_TENSOR`; map with `compiler_pc` to slot. | For each, find the wait's source (S[3] from Pool, or S[6]) and either (a) hop the dependent add one more compute slot later, or (b) move that add to a different producer so the wait is already satisfied. This is the same recipe as iter-5 on the remaining slots. | Established on device in iter-4/5 (hop past wait: 144 slots, 112 S[6] waits). The same mechanism applies where `evt_wait_time_ns` is large. | At most 528 us. Realistic is 20 to 40% of it (about 100 to 200 us), since deadline slack is 3 DVE slots (commit fa7f641) and Pool is itself 85% busy. | Medium: race on S[3]. The verifier may not catch a latent race. | Sort the 916 gaps by the waited semaphore and value. If the producing Pool instruction ends later than the DVE slot's start in the profile for the majority, the gap is Pool latency and hopping cannot help, so kill it. Takes about 10 minutes. |
| 2 | Re-balance the y+=Cs adds between DVE and Pool using the real costs | DVE0 TT and Pool0 TT / COPY. Pool does 1.98 us per add and DVE 0.67 us. | Move adds back from Pool to DVE only where the Pool add is on the critical path, or move a few where Pool idles. The measured balance point is DVE 6523 us vs Pool 5616 us. Moving k adds from DVE to Pool balances at about k=340, but Pool work per add is 3x DVE's, so it is almost worthless unless the gaps close first. | Established in iter-1..4: moves off DVE gave -13%, -1%, -1.6%. Each later move cost more Pool time than it saved. | Small. DVE critical-path saving is at most about 0.67 us x k, so about 100 to 200 us even in theory. | Medium. The S[6] renumbering has to stay exact. | Count how many of the 2260 Pool adds have a successor within less than their duration. If under about 100, the lever is empty. |
| 3 | Overlap-rather-than-serialise the SCAN wait gaps (390 gaps, 240 us) | DVE0 SCAN slots; the waits are PE MATMUL (S[5]) or ACT (S[4]) outputs. | Check whether PE/ACT results are produced late relative to DVE (S[5] has 8859 waiters). If they are early, move the SCAN's wait onto a separate earlier ES so DVE is not stalled by the wrong semaphore. If they are late, nothing to do. | Same mechanism as the iter-5 "move S[6] waits onto a fresh ES". Not yet applied to S[4]/S[5]. | At most 240 us. PE is 45% busy and ACT 29%, so the producers have slack. Realistic is 50 to 100 us. | Low to medium: the wait must stay before the consumer. | Compare producer end_ts to consumer start_ts for the 390 gaps. If the SCAN starts within a few hundred ns of the producer ending, it is a true dependency and it dies. |
| 4 | Shrink Pool fixed overhead: the 1741 COPYs at 0.42 us and the 680 DMA_DIRECT2D at 0.66 us | Pool0.bin COPY slots and DGE descriptors. | Merge more carry COPYs into strided dim-1 copies, as done for 973 of them in iter-3. Check whether the remaining singles can be paired. | Established in fa7f641 (-300 us of Pool time). Fixed per-instruction cost dominates, since a 1-element COPY is mostly overhead. | Bounded by Pool COPY total of 732 us. Pairing gives at most about 250 us of Pool time. It speeds the wall clock only if Pool is on the critical path, which is unknown. | Low: carry copies are small and verifiable. | First check whether Pool time is on the critical path: shorten one COPY and see if the end-to-end time moves. If not, kill it. |
| 5 | Fold a scalar-engine-owned piece of the work onto idle ACT or PE | Activation0.bin (28.8% busy), PE0.bin (44.9%). | Move a TT-like add (for example a y+=Cs partial) to ACT as an ACTIVATE with a bias, or to PE as an accumulating matmul in PSUM. | Not established. It would need a different fp32 accumulation order, so bit-exactness is at risk, and the error is already at 83% of its limit. | Unknown. ACT has about 4.7 ms idle and PE about 3.6 ms, but an ACTIVATE costs 0.72 us per op and there is no evidence it can take an fp32 add of two tensors. | High: changes numerics and could push error toward 0.15. | Check `isa.py` for a two-tensor-input ACTIVATE or tensor_scalar form on ACT. If none, kill. |

Ranking rationale: #1 and #3 reuse a lever that already worked and have measured gaps behind them. #2 and #4 are smaller and depend on the critical path. #5 is speculative and is listed last.

### Closed (do not redo)

- **Moving ADDs wholesale from DVE to Pool.** Done in iter-1..3 (7919.6 -> 6886 -> 6819 -> 6712). DVE busy went from 7860 to 6523 us, but Pool is now 85% busy (5616 us). More bulk moves would make Pool the bottleneck (1.98 us per add vs 0.67 us on DVE).
- **y_accum MEMSETs.** Gone (46 left, down from 200). No more headroom: 46 x 0.17 us is under 8 us.
- **Merging 1-element carry COPYs** (973 merged). The remaining ones are in #4.
- **S[6] renumbering to avoid the self-wait guard.** Done in e7f3a7a. Do not reintroduce the 585 ns guard.
- **DMA and PE as limits.** DMA is 21.7% busy and PE 44.9%. Both are idle most of the time.

### Unknown

- **Critical path.** It is unclear whether DVE or Pool bounds the finish. Settle it by extracting the last-finishing engine's chain from the profile (end_ts of the last DVE and Pool instructions, plus the final waits). If Pool finishes first, work on DVE gaps (#1, #3) is what matters.
- **Pool-latency share of the DVE gaps.** Settle it with the per-gap analysis in #1: producer end_ts vs consumer start_ts.
- **Reduced-precision TT.** Whether the TT and SCAN operands could be bf16 for a faster mode. This would break the bit-exact claim and the error budget is already at 83%, so it is probably closed, but the throughput is unmeasured. One profile of a modified single slot answers it.
- **Latent race risk of the iter-5 hops.** Settle it with 10 repeated verify runs and a perturbed input.

I did not execute any of the candidates. All gain figures are upper bounds from the profile, not measurements.

# Review of iter-6 (kernel.neff == commit 51c6a8e, iter-5's NEFF): 6603.6 us

## 1. What changed

**Nothing in this iteration.** `git status` is clean, and the worktree's `kernel.neff` is the one committed in 51c6a8e. The agent's `.autohelix/notes/iter-6.md` says the same ("no change shipped, byte-identical to iter-5"). The worktree's own `notes/README.md` is still the empty template; the real notes are in the main `.autohelix/notes/`.

`neff_diff.py baseline/kernel.neff kernel.neff` therefore shows iter-1..5 cumulatively:

| Member | Slots (base -> now) | Slots differing |
|---|---|---|
| DVE0.bin | 15929 -> 14139 | 15926 |
| Pool0.bin | 3842 -> 5283 | 5276 |
| PE0.bin | 10135 -> 10135 | 4961 |
| Activation0.bin | 3040 -> 3040 | 2290 |

- PE and ACT differ only in `events_semaphore_value` (the S[6] renumber).
- The other 30 members, including all DMA descriptors, are identical.
- The only code-like changes are DVE/Pool slot moves, deletions and insertions.
- Consistency with the notes: the slot counts match. DVE went down by 1790 (154 MEMSETs deleted, ADDs moved to Pool). Pool went up by 1441 (moved ADDs, minus merged carry copies). In the profile, DVE MEMSET is down from 186 to 46 and Pool TENSOR_TENSOR is 2260.
- iter-6's real output is negative results plus tooling in `notes/iter-6-tools/`.

## 2. Is the latency real?

- The baseline median is 7919.96 us (spread 0.024%).
- The latest median is 6603.58 us. The samples are 6603.53 to 6603.95, a spread of 0.006%. My own `--repeats 1` run gave 6603.76 us.
- The gain over baseline is -16.6%, about 1300 us against a spread of about 1 us. It is real and deterministic.
- There is no new win this iteration, so nothing here is noise-chasing. Iter-5's step (-108 us) was about 80x the spread.
- The loop is now at a point where the remaining ideas each produce a few tenths of a percent, roughly 5 to 25 us. That is still well above the 1 us spread, but device-run cost is high.

## 3. Safety

- **Verifier:** pass. output_0 `max_abs_err` is 0.12497, which is 83% of the 0.15 limit. The baseline has the identical number, so it is inherent to the kernel and has not moved. output_1 is 1.8e-4.
- **Semaphores:** I unpacked both NEFFs and ran `semaphores.py` on sg00.
  - S[3] has max_wait 3986 against 3987 increments.
  - S[4] has 2555 against 2555, S[5] 5023 against 5023, and S[6] 8022 against 8022.
  - The DMA and DGE semaphores are unchanged from baseline.
  - No wait exceeds its increments.
- **Limits of that check:**
  - The tool counts only primary waits, and the notes record 3567 DVE extended waits on S[6] and 164 on S[3].
  - It also cannot see ordering races.
  - The risky pieces are still the iter-5 "hop past S[3] wait" ADDs (144) and the Cs buffer relocated to SBUF 62272. Three identical runs make a latent race unlikely but do not rule it out.
  - Cheap extra guard before accepting anything built on this base: run `verify.py` about 10 times and compare the outputs bit for bit, ideally also with perturbed inputs.
- **Hang risk:** iter-6 found that an unsatisfiable wait hangs silently for about 70 s without an error, and that removing ES slots corrupts output (`max_abs_err` 19) while the verifier's tolerance fraction stays at 99.6%. Any further edit must keep every ES slot, and must run `notes/iter-6-tools/sim.py` first. Extend it to read extended waits.

## 4. Ranked candidates

The honest picture is that the kernel is close to the floor for NEFF-level edits. The profile numbers below are from this kernel and are the same as iter-5's review.

Profile summary:
- DVE is 98.8% busy. It has 855 us in 1451 gaps over 200 ns. Of these, 916 gaps (528 us) are an ES followed by TENSOR_TENSOR, and 390 gaps (240 us) are an ES followed by SCAN.
- Pool is 85% busy. It has 1040 us in 1379 gaps. The big ones are TT->TT (488 gaps, 118 us), TT->DMA_DIRECT2D (123 gaps, 150 us), DMA->COPY (164 gaps, 212 us), and COPY->COPY (185 gaps, 155 us).
- PE is 45% busy and ACT 29%.

| # | Candidate | Slots / members | Change | Mechanism and evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Per-gap triage of the 916 DVE ES->TT gaps (528 us), then fix only the sites where the producer finished early | DVE0.bin. Find slots via `Instruction.parquet`: `engine=='Vector'`, gap>200, prev=EVENT_SEMAPHORE, next=TENSOR_TENSOR. `compiler_pc` is the slot index. | For each site, identify the semaphore the ES waits on (S[3] from Pool, or S[6]). If its producer ended well before the ES started, the wait is over-conservative: move the wait onto an earlier ES, or hop the dependent add one more slot. Otherwise leave it. | Same recipe as iter-5, which worked on device (144 hops, 112 S[6] waits moved). Moving a wait to an earlier ES is not reordering. iter-6's -280 ns cost applies to inserted wait-carrying ES slots, so reuse existing ES slots. | At most 528 us. Realistic 30 to 100 us. iter-5 already took the sites with slack, and deadline slack is about 3 DVE slots. | Medium. A race on S[3] can pass the verifier. | A 10-minute join of each gap to the end_ts of the Pool instruction that supplies the awaited ordinal. If most producers end at or after the gap's end, the gap is Pool latency and the lever is dead. |
| 2 | Same triage for the 390 ES->SCAN gaps (240 us) | DVE0.bin SCAN slots. The waits are on PE (S[5]) or ACT (S[4]). | Check whether the PE matmul or ACT activation feeding each scan finished before DVE started waiting. If it was early, the wait is already satisfied and the gap is an artifact. If late, widen the PE/ACT lead. | Not applied yet to S[4]/S[5]. PE has 3.6 ms idle and ACT 4.8 ms idle, so producer slack exists. iter-4 measured ACT binding stall at only 1.2 us, so most of these are probably true dependencies. | At most 240 us. Realistic is near 0, per iter-4's ACT result. | Low to medium. The wait must stay before the consumer. PE's S[6] PSUM-bank release waits must use the running max (iter-6 fact). | Compare producer end_ts with consumer start_ts on 390 gaps. If the scan starts within about 300 ns of the producer ending, it is a true dependency. Dead. |
| 3 | Shrink Pool fixed overhead: remaining 1-element carry COPYs (1741 COPY, 0.42 us each) | Pool0.bin COPY slots | Pair more carry COPYs into dim-1 strided copies, using iter-3's rule set. | Established in fa7f641 (973 merged, -300 us of Pool time). The remaining ones were rejected for alias or S[3]-waiter reasons. | iter-5 §2 put the remaining ceiling at about 153 us of Pool time (1216 merges). Wall effect is unknown, because it is not shown that Pool is the pacing engine. | Low. Carry copies are small and the output is bit-exact. | Delete the time of one COPY, or merge a handful, and check whether the end-to-end time moves. If not, Pool time is not on the critical path and it is dead. |
| 4 | Shorten Pool DMA_DIRECT2D / DGE fixed costs (680 at 0.66 us, plus about 210 us of gaps before COPYs) | Pool0.bin DMA_DIRECT2D slots and their `semaphore`/`sem_increment` fields | Check whether the prefetch DMAs can be issued earlier (up to the 8 semaphore queues' max waits) so DMA->COPY gaps shrink. | Unknown. Not established. No earlier note touches the issue order. | Unknown. Bounded by the 212 us in DMA->COPY gaps. | Medium. A DMA issued too early overwrites a live buffer, and the verifier can pass this. | Look at 20 DMA->COPY gaps and see whether the COPY waits on a DMA completion or only on a Pool issue slot. If it is a DMA completion wait, the DMA queue is the limit and this is dead. |

I have only four, because reorder, merge and reallocation levers are all measured shut (see closed). The gain figures are upper bounds from the profile, not measurements. I executed none of them.

### Closed (do not redo)

- **Reordering DVE computes at all.** iter-6 measured 572 ns per swap of two adjacent independent computes (420 swaps = +322 us, bit-exact, instruction count unchanged). Closes any reorder-based lever.
- **"2 DVE computes per state" (merge C(m) into B(m+3)).** Built, bit-exact on 30 phases, measured +3.5% to +3.9%. The merge itself is worth only about 80 ns, not 138. A hazard also remains in phases 31..35.
- **Removing ES slots.** Output goes wrong (`max_abs_err` 19) even when every per-semaphore guarantee is preserved. ES slots are pipeline padding as well as waits. Only add them.
- **Conservative wait re-emission.** About 280 ns per inserted wait-carrying ES, which makes it fatal at 40 per phase.
- **Fresh SBUF buffers.** The only free region above 16384 is 113472..114688 (1216 B), too small. 0..16383 hangs when used for a buffer family. Note that iter-5's "free hole at 62272" is now used by the Cs relocation.
- **Moving y+=Cs ADDs wholesale DVE <-> Pool.** The balance point is measured in both directions (590 ns DVE work against 521 ns stall). DVE is 6521 us, Pool 5611 us.
- **y_accum MEMSETs.** 46 remain, worth under 8 us.
- **ACT as a bottleneck.** Binding stall is 1.2 us. DMA (21.9% busy) and PE (45%) are idle most of the time.
- **S[6] self-wait guard.** Replaced by renumbering in e7f3a7a.
- **Reduced precision (bf16 TT/SCAN).** Error is already 83% of the limit, so it is out.

### Unknown (would change the ranking)

- **Which engine bounds the end.** The wall is 6603 us, DVE span 6612 us and Pool 6610 us, so both finish at the end. Settle it by finding, in each gap, whether DVE waits on Pool or Pool waits on DVE. The ES->TT triage in candidate 1 answers this.
- **Whether Pool time is on the critical path.** A 10-run test shortening a few Pool COPYs settles it. It gates candidate 3.
- **Whether iter-5's hops hide a race.** Settle it with about 10 repeated verify runs and perturbed inputs.
- **Why a merged `[512,2]` TT is worth only 80 ns.** Widen 40 unmerged C-multiplies and read their `duration_ns` from `Instruction.parquet`. This only matters if a merge is found that needs no reordering, and iter-3 says none is left.

## Bottom line

This iteration produced no NEFF change, and the notes support that. The 16.6% total is real and the error is unchanged. The remaining headroom the profile points to is the 755 us of DVE ES gaps (528 + 240), the most that candidates 1 and 2 could reach. Realistically it is 0 to 100 us. The next agent should spend its first 10 minutes on the gap-to-producer join in candidates 1 and 2, and stop if the stalls are true Pool or PE latency.

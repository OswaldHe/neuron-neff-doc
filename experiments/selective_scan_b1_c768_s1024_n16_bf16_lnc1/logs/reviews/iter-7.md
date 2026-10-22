# Review: iter-7 worktree (HEAD 7358c9d + uncommitted kernel.neff), selective_scan b1 c768 s1024 n16

I diffed against `git show HEAD:kernel.neff` (sha 07f0638c), not against `baseline/`, because the baseline diff is cumulative over six iterations. `notes/README.md` is still the empty template, so there is nothing to cross-check the edit against. Everything below comes from the diff.

## 1. What changed

3 of 34 members changed: `DVE0.bin` (31 slots), `Pool0.bin` (6 slots), `PE0.bin` (1 slot). `Activation0.bin` is identical to HEAD. Slot counts are unchanged, so this is all in-place rewriting.

| change | slots | detail |
|---|---|---|
| 2 more first-y writers retargeted into y | DVE 14; DVE 798/799 with Pool 8 and Pool 303 | DVE 14 `dst` goes 0x7940 -> 0x4940, which is y. DVE 798/799 swap places, and the Cs MULT now writes 0xa940, where it wrote 0x8000 before. The MEMSET placeholder with the same `dst` now sits before the MULT, which is the correct order. The dead Pool 512-element COPYs (7940->4940 and 8000->a940) become 1-element COPYs, `num_elem` 512 -> 1, at the 0x16040 and 0x16080 scratch addresses. With these, all 12 first-y COPYs are gone: Pool now has 0 COPYs of 512 elements, against 2 before. |
| DVE instruction reorders at group boundaries | DVE 610-613, 785-788, 804-807, 633/634 | Slots 610-613 and 785-788 move the bf16 `dt*x` TT (it waits on a DMA semaphore, S8 >= 48 and S18 >= 48) from the front to the back of its 4-slot window. The ES wait on S6 and the two TTs it guards move forward by one slot. Slots 804-807 swap an ES and TTs in the same way. Slots 633/634 swap an ADD and a MULT. |
| S6 ordinal compensation | DVE 614, 635, 789, 811; Pool 243, 307, 361; PE 572 | `events_semaphore_value` is +1 or +2 on the waiters of the TTs whose ordinal moved. |

The mechanism is DVE slot reordering plus compensating wait values. The slots are not widened, fused or moved between engines. The shape of the diff is internally consistent: S6 total increments are unchanged at 640, with 420 TT, 192 SCAN and 28 MEMSET. S3 stays at 282 and S5 at 374, so none of the counts moved.

## 2. Is the latency change real?

Yes.
- `profile_latest.json`: median 497.88 us, min 497.78, max 498.07, spread 0.06%.
- My own re-profile: 497.858 us.
- The previous accepted value was 501.3-501.4 us (iter-6 noted a noise floor of about 0.3 us). The gain is 3.5 us, about 10x the noise.
- DVE busy fell 482.3 -> 478.6 us, which matches the gain.
- The union DVE span is 4.36 -> 510.59 us, with only 8.8 us of true idle inside it. Of that, 4.13 us is the startup gap at 10.7 us and the rest is scattered, none above 0.39 us. DVE is saturated.
- Cumulative gain: 602.64 -> 497.88 us (1.210x).
- I cannot split the 3.5 us between the retargets and the reorders. The agent did not write notes. See candidate 1, step 0.

## 3. Safety beyond the verifier

- **Error:** output_0 max_abs_err 6.245e-02 (83% of the 0.075 limit), identical to the baseline figure. output_1 is 4.27e-05. Nothing has moved.
- **Counts:** `semaphores.py` flags no wait exceeding its increments. S6 max_wait is 640 and total_inc is 640.
- **Not checked, and the verifier cannot check it:** `semaphores.py` checks counts, not which instruction each waiter is meant to see. This edit changes which instruction retires at which S6 ordinal. Slot 613 is the bf16 `dt*x` that now retires two ordinals later. Slot 611/612 retire one earlier. Seven waiters were re-valued by hand (DVE 614, 635, 789, 811; Pool 243, 307, 361; PE 572). A waiter that was left alone and sat on the old `dt*x` ordinal would now be satisfied one or two instructions early. That would pass `verify.py` if PE or Pool happened to be late. iter-6 §3 found exactly this hazard, in a build that passed `verify.py` bit-identically.
- **Requested audit (one script):**
  - For every S6 waiter in all five streams, derive the instruction that retires at its wait value in the new order. Compare it with the instruction that retired there in HEAD, mapped through the permutation. They must be the same.
  - Run the NEFF 20 or more times and compare outputs bit-for-bit.
- **Retargeted y writers:** the Cs MULT now writes straight into y at 0x4940 and 0xa940. Check that no later reader of the old 0x7940 / 0x8000 Cs buffers expected the value there.

## 4. Candidates for the next iteration

Context: iter-6 argued that this NEFF is within about 1% of the instruction-level floor, and this iteration found another 0.7%. DVE is the wall, as the iter-6 note says: about 384 PSUM MULTs x 0.677 us + 192 scans x 1.27 us. That is about 506 us of DVE compute, and the 36 SBUF-only ADDs are the only structurally movable part. The wall is DVE duration plus the group-boundary stalls. All remaining levers are small.

| # | candidate | change | ceiling | risk | falsifier |
|---|---|---|---|---|---|
| 1 | **Extend the `dt*x` reorder to the remaining sites** | In `DVE0.bin`, move the bf16 `dt*x` TT after the two following TTs in its window, as done at 610-613 and 785-788. Still preceded by an ES, so not yet done: slots 5, 87, 262, 350, 438, 527, 702, 878. Slot 174 is preceded by a TT and 963 is already moved. Re-value the S6 waiters: DVE 614 style +2 on the `dt*x` consumers, and -1 or the equivalent on the two TT consumers. | About 0.5-1.7 us per site, because 2 sites gave part of the 3.5 us. The group-boundary tax is 1-2 us per boundary (iter-6 M2b), so the best case is 8 x 1.5 = about 12 us. Expect 2-5 us. | Moderate. A mis-valued S6 wait races silently. | **Step 0:** revert the reorders and keep the retargets, profile it, and learn which half gave the 3.5 us. **Step 1:** apply one site (say 87), profile, and expect at least 0.5 us. If not, close this. |
| 2 | **Startup gap, 4.13 us of DVE idle at 10.7 us** | Reorder or split the first 128 KB input load: add one extra DGE trigger that splits the load by partition, so the first MULT's inputs arrive first. | At most about 1.5 us (0.3%). Source: `Instruction.parquet`, DVE start 4.36 us with the first compute only after the 4.13 us gap. | Moderate. Queue semaphore counts shift (S8-S23). A mistake hangs, so the verifier catches it. | Find in `SemaphoreUpdate.parquet` the semaphore the first DVE MULT waits on. If it is satisfied within 1 us of the gap's end, close this. |
| 3 | **Tail DMA, last y store** | Split the last group's y DMA across two queues (same engine only, because a trigger cannot change engine). | 1 us or less. | Low. | If the DMA is latency-bound rather than bandwidth-bound, close it. |

Candidates 2 and 3 are listed only because iter-6 left them open. Each is under 0.3%.

### Closed (do not redo)

All numbers are from iter-6 `notes/iter-6.md` unless noted.
- **Moving the `y+=` ADDs to Pool:** E1 (24 ADDs) +8.3 us, E6 (12 ADDs) +0.62 us. DVE duration is converted into blocking at about 1:1 (M1).
- **Pairwise-merging TT pairs on DVE or Pool:** `dBx` merge +26.1 us, `Cs` merge +83.4 us, Pool merge +62 us, DVE `y+=` merge +0.33 us (M2). Saved duration per merge is only 67-126 ns.
- **Fusing the two seq chunks into a 1024-element scan:** blocked by PSUM capacity, not scheduling. Scan fixed cost is 205 ns, marginal rate 2.08 ns/element.
- **DGE trigger placement:** the 15 big loads later +1.06 us, earlier +11.6 us (M3).
- **Re-hosting the Pool DGE queue onto SP or ACT:** host segfault or NRT_TIMEOUT in all four variants (M4).
- **Operand swap, dtype, or ES-slot trimming:** zero effect.

### Unknown (would change the ranking)

1. **Which part of this iteration's 3.5 us came from the retargets and which from the reorders.** Measure by reverting each half. This sets candidate 1's per-site yield.
2. **Whether the 168 Pool ADDs (2.16 us each vs 1.26 us when DVE is quiet) are slowed by SBUF contention with DVE.** If yes, the real lever is the overlap between Pool and DVE SBUF accesses, not instruction counts. Settle it by comparing the fast and slow ADDs for temporal overlap with DVE computes in `Instruction.parquet`.
3. **Whether the S6 re-valuing in this iteration is race-free.** The audit in section 3 settles it, and a hidden race would invalidate 497.9 as a baseline.

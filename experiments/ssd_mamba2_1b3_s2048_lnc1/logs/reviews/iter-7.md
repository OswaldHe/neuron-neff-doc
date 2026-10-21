# Review of iter-7 (kernel.neff vs HEAD c7383fe; baseline = 7154.9 us)

Method: `neff_diff.py` against HEAD and baseline, a difflib slot alignment per engine stream (`/tmp/rv/al*.py`), `semaphores.py` on both unpacked NEFFs, and a fresh `tools/profile.py --repeats 1 --parquet` run (reproduced 4251.1 us).

## 1. What changed

All of it is relative to HEAD (4683 us in the last commit message). `notes/README.md` is the empty template, and the HEAD commit message describes the previous iteration. **There is nothing to check this iteration's claims against.** The changes below are what I could reconstruct from the NEFF.

| Member | Slots HEAD -> now | Change |
|---|---|---|
| DVE0 | 24525 -> 18590 | Pure deletion, no replace: **3031 LOAD_MASK_SELECT** and **2821 EVENT_SEMAPHORE**. All 3031 LMS and 2745 of the ES have no wait and no update, i.e. they were the leftovers from the previous iteration's retired STREAM_SHUFFLEs. 76 deleted ES carried a wait (see section 3). STREAM_SHUFFLE count is unchanged at 4818. |
| Activation0 | 14074 -> 13668 | 5199 ACTIVATEs flip `imm_src` POINTER_IMMEDIATE -> INSTRUCTION_IMMEDIATE (the only field that changes in 1452 single-slot edits and in the same-length groups). About 190 `[ES, ACTIVATE]` or `[ACTIVATE, ES, ACTIVATE]` groups collapse by one slot, and 8 ES are deleted outright. ES count 4574 -> 4168. |
| PE0 | 14865 -> 14813 | 52 wait-only ES deleted. All 52 waited on the sem 56-71 family. |
| Pool0 | 4192 -> 4181 | 11 wait-only ES deleted. |
| SP0 | unchanged | |

The `imm_src` flip is, I believe, harmless: the third immediate operand has `imm_imm_ptr = 0` and `imm_imm_fp32 = 0.0`, and the activation func (128 = COPY or scale) doesn't use it. This is the same family as the earlier "what does `imm_src=PTR, imm_ptr=0` read" question in the neff_doc experiments. It is bit-exact in effect only if nothing consumed that operand. It is also unverified here whether it gives any speedup, because it is mixed with the deletions.

The change is consistent with the pattern in the previous commits: dead slots are removed, increments are kept, and total increments are unchanged. `semaphores.py` totals and max_wait are identical for every semaphore between HEAD and now. Only the waiter counts dropped (sem4 -37, sem6 -246, and the DGE sems).

## 2. Is the latency change real

Yes. Baseline is 7154.9 us with 0.021% spread. HEAD, per its commit message, is 4683 us. Now it is 4250.97 us, with samples 4249.7 to 4251.6 (spread 0.045%). My independent single-repeat run gave 4251.108 us. The ~432 us step over HEAD is about 10,000x the spread, so it is not noise. Cumulatively that is -40.6%.

DVE busy fell 2733 -> 2132 us and ACT busy fell 6284 -> 2550 us. Every engine is now at 30-60% busy, so the kernel is latency-bound on semaphore handoffs, not on any one engine.

The conversion rate matters for ranking. About 5935 DVE slots removed bought about 430 us of wall time, roughly **70 ns of wall per removed DVE slot**. That is far better than the earlier ACT experience, where -97 us of ACT busy bought -1.6 us.

Reading the timeline, a DVE LOAD_MASK_SELECT takes about 80 ns of in-order issue time even though its recorded duration is 1-15 ns. Slot count on the critical engine matters more than recorded `duration_ns`.

## 3. Safety beyond the verifier

- **Error moved.** The cap is 6.2e-7. output_0 is 5.234e-7 (84%) against the baseline's 5.122e-7 (83%). output_1 is 3.334e-7 (54%) against the baseline's 3.166e-7 (51%). Earlier commits are described as bit-identical, so a deterministic bit-identical NEFF would give the same errors as the baseline. The small shifts mean one of three things:
  - the output is not bit-identical (some edit changed a few ULPs);
  - the run is nondeterministic (a latent race);
  - the verifier's input differs from the baseline's.

  The magnitude is tiny, but an unexplained change should be checked before more edits stack on top of it. **Next agent: dump `output_0`/`output_1` bytes for baseline, HEAD, and now, and `cmp` them.** If HEAD matches the baseline and now does not, bisect the `imm_src` flip against the ES/LMS deletions. If two runs of the same NEFF differ, there is a race.
- **Waits loosened.** Of the deleted ES that carried a wait, 76 were on DVE, 52 on PE and 11 on Pool. Only 47 of the 76 DVE waits are covered by a wait of the same sem and an equal or higher value on the next retained instruction. The other 29 DVE waits (19 on sem6, the DVE's own semaphore; the rest on DGE sems 10-23) plus all 52 PE and 11 Pool waits are not covered locally.
  - The PE and Pool waits are on sem 56-71, the state_sb DGE sems, which were made moot by the earlier "pin state_sb in SBUF" commit.
  - The sem6 DVE self-waits would only matter if they were acting as a RAW fence.
  - `semaphores.py` flags sems 40-71 as "wait exceeds increments". This is true of HEAD and baseline as well and is an artefact of SEM_ADD_IMM(+16) placeholders that the tool does not count, so it is not new.
  - A race can pass the fixed verifier input once and still be wrong. The deleted waits are mostly on dead-DMA sems, so I rate this low to medium risk, but it is not proven.
- **Not checked:** a second verifier input. `tools/verify.py` uses one fixed input set.

## 4. Candidate edits for the next iteration

Ranked by expected gain per unit of risk and effort. The gains rest on one calibration point: -5935 DVE slots bought -432 us.

### C1. Delete the remaining LOAD_MASK_SELECT slots on DVE
- **Where:** `sg00/DVE0.bin`, 5115 LMS slots. Every one is identical: a single distinct mask and param, all zero. 3764 have no wait and precede a STREAM_SHUFFLE with no wait, so they can be deleted outright. 1054 carry a wait and precede a STREAM_SHUFFLE with no wait, so move the wait into the shuffle's own wait field, then delete. Another 289 precede COPY, TENSOR_TENSOR or ES with their own waits, so they are placeholders. Delete them if the next instruction's wait covers theirs, otherwise convert to a shrunk ES or leave them.
- **Mechanism:** a dead or identical instruction can be deleted with the stream compacted, as long as increments and wait thresholds are preserved. That was established on device in this iteration (3031 LMS were removed from the same stream). The open question is whether STREAM_SHUFFLE needs an LMS immediately before it. If the mask register persists, keeping the first LMS is enough. If the mask is all zero, it may not matter at all. Keeping one LMS per run of shuffles is the safe version.
- **Gain / ceiling:** 5115 slots x about 80 ns of issue time is about 0.4 ms of DVE issue time. At the 70 ns/slot calibration, 3764 outright deletions would be worth up to about 260 us, and the full set up to about 350 us. A realistic expectation is 100-250 us. The bound is DVE busy (2132 us) and the 4251 us wall.
- **Risk:** medium. If STREAM_SHUFFLE needs its own LMS, the output is wrong. The verifier would catch that, because the shuffles feed the decay broadcasts. A wait moved onto the shuffle is a pure relocation, so it is safe.
- **Cheapest falsifier:** delete the LMS before the second, third and fourth shuffle of one 4-shuffle group, for one head only. Run verify (30 s) and compare the output bytes to the current NEFF. If they are bit-identical, scale up.

### C2. Fold ACT wait-only ES pairs and no-wait neighbours
- **Where:** `sg00/Activation0.bin`, 4168 ES. 2241 are wait-only and 1927 are wait+update placeholders. By neighbour: 1124 are followed by another ES, 232 are covered by the next instruction's wait on the same sem, and 110 are followed by an instruction with no wait.
- **Mechanism:** ES carries two waits (`events_wait_*` and `events_extended_wait_*`) and two updates (the decoded fields exist). The 232 covered ones are redundant. The 110 can move their wait into the next instruction. An ES;ES pair with the first having no update can become one ES that uses the extended wait. The extended-wait semantics are not yet established on device. Test with one pair.
- **Gain / ceiling:** ACT ES busy is 236 us, so the ceiling is about 236 us of ACT busy. Realistically 1466 slots x 40-56 ns is 60-80 us if the ACT is on the critical path. The earlier ACT-busy to wall conversion was poor (-97 us for -1.6 us), so a few tens of us is more likely.
- **Risk:** medium. Merging waits can reorder against other engines. The verifier may not catch a weakened wait, so run the semaphore accounting on every change.
- **Cheapest falsifier:** do the 232 covered deletions first (zero synchronisation risk) and profile. If wall time doesn't move, drop C2 and ACT-side slot removal.

### C3. Remove the second LDWEIGHTS of each fp32 matmul pair
- **Where:** `sg00/PE0.bin`. Of 7232 consecutive LDWEIGHTS pairs, 3558 have identical `src` patterns. In the timeline each fp32 matmul is `LDW(A); MM(acc_flags=1); LDW(A); MM(acc_flags=2)` with the same moving data and the same PSUM destination.
- **Mechanism:** unknown. fp32 matmul looks like a 2-pass operation, and the second LDW may be required for the second pass. This is the weakest-evidenced candidate.
- **Gain / ceiling:** LDW busy is 2221 us over the whole run. Deleting 3558 slots of about 300 ns bounds PE busy at about -1.07 ms. PE is only 45% busy, so wall gain is uncertain, probably small.
- **Risk:** low to detect, because the verifier would show a numerics change at once.
- **Cheapest falsifier:** shrink one second LDW to a no-op ES in one unit and run verify. If the output changes, close C3.

### C4. Merge adjacent same-source SOFTPLUS ACTIVATEs
- **Where:** `sg00/Activation0.bin`, 3068 SOFTPLUS (func 7), each 1 element and 160-200 ns. 971 adjacent pairs read the same source address.
- **Mechanism:** the 2-group 3-D access pattern on ACTIVATE (established last iteration) and the "shrink, keep the compute-completion increment" rule from the previous commit. Iter-1 merged 303 identical ACT pairs, and that gave only -1.6 us wall.
- **Gain / ceiling:** 971 x about 170 ns is about 165 us of ACT busy. Given the iter-1 conversion, expect little unless ACT is critical.
- **Risk:** low to medium. Semaphore thresholds must be preserved.
- **Cheapest falsifier:** merge ten pairs and look at the profile gap. Do this only after C1 or C2 show that ACT slot removal converts.

### Not recommended as the first move
An ablation of the whole class is cheaper and more informative than any single edit. See "unknown" below.

## Closed

- **Lower-precision matmul or activation.** Output error is already 84% of the cap (5.23e-7 against 6.2e-7). `perf_opt` is NONE and `in_dtype` is FP32. Not a lever.
- **ACT busy-time reduction as such.** Iter-1 merged 303 ACT pairs for -97 us ACT busy and -1.6 us wall (HEAD commit message; no note file).
- **state_sb HBM round-trip DMAs.** Done (commit 73236ba, 5107 -> 4856 us).
- **Per-head cs_col PSUM->SBUF copies.** Done (86c8a41, 4856 -> 4777 us).
- **Redundant second exp_cs_last broadcast, exp_cs_col multiply merge, Y_off dead pair rewrite.** Done (c7383fe, 4777 -> 4683 us).
- **DVE dead LMS and ES left by retired shuffles.** Done in this iteration (-432 us).
- **DMA volume.** DMA is 30% busy and 528 KB of DMA payload per the profile summary. Not a bottleneck.

## Unknown, and the measurement that would settle each

- **Whether the bit-level output changed vs baseline.** Dump and `cmp` the output bytes for baseline, HEAD, now. This should be done before the next edit.
- **Which engine is on the critical path now.** Ablate: delete all 4818 STREAM_SHUFFLE and 5115 LMS, then profile (numerics are wrong, only time matters). Then ablate all ACT ES. The wall drops show the ceiling of each lever before any careful editing.
- **Whether STREAM_SHUFFLE requires its own LMS.** C1's one-head test.
- **Which edit in this iteration gave the speedup.** The `imm_src` flip (5199 ACT slots) versus the DVE deletions. Revert the flip alone and profile (a 5-minute test).
- **Whether the sem6 DVE self-waits and the sem 56-71 waits removed here were fences.** Run a second verifier input (different seed) and a 5-repeat run; compare bit-for-bit.
- **Extended-wait semantics on ES.** One merged ES;ES pair plus a profile.
- **fp32 two-pass matmul need for the second LDWEIGHTS.** C3's one-unit test.

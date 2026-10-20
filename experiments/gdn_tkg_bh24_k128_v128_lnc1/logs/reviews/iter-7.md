# Review of iteration 7 (kernel.neff, working tree vs HEAD = iter-6 commit 8d63265)

All numbers below come from `neff_diff.py` of HEAD's kernel.neff vs the working one, `semaphores.py` on both unpacked copies,
`profile_latest.json`, `verify_latest.json`, and a fresh `profile.py --repeats 1 --parquet` run (total_exec 60.943 us).
`notes/` contains only the README, so the agent left no claim to check; the commit message for this iteration does not exist yet.
Everything under "what changed" is therefore inferred from the NEFF, not compared with a claim.

## 1. What changed

Only two package members differ from HEAD (44 of 46 are byte-identical; PE, Pool, SP and all JSON files are untouched):

| member | slots | opcode census, before -> after |
|---|---|---|
| `sg00/Activation0.bin` | 170 -> 156 | EVENT_SEMAPHORE 67 -> 54 (13 pure wait fences deleted), ACTIVATE 59 -> 58, CAST 13 and ACT_TABLE_LOAD 6 unchanged |
| `sg00/DVE0.bin` | 213 -> 196 | EVENT_SEMAPHORE 46 -> 28 (18 pure wait fences deleted), SCALAR_TENSOR_TENSOR 45 -> 46, TENSOR_TENSOR 86 -> 86 |

Beyond the fence deletion there are two functional edits:

- **One ACT op moved to DVE.** One per-head ACTIVATE (func 29, reading a PSUM tile) is gone from the ACT stream. A DVE `SCALAR_TENSOR_TENSOR` (src0 = PSUM 0x2002000, src1 = 0x37600) replaces it, and the preceding DVE TT now multiplies by the SBUF scalar at 0x4170 instead of reading PSUM. ACT -1 ACTIVATE, DVE +1 STT.
- **Five DVE TENSOR_TENSOR slots reordered.** They are insert/delete pairs in a difflib alignment, i.e. earlier positions for no-update TTs.

The ACT/DVE wait count dropped by 13. `semaphores.py` waiter counts: sem4 149 (was 153), sem6 137 (was 144), sem12 2 (was 3), sem25 4 (was 5). Total increments and max_wait are unchanged on every semaphore (sem4 120, sem6 103, ...), so no counter was left unreachable.

## 2. Is the speedup real?

Yes.

- HEAD measured 62.546 us (from its commit message). Now the median is 60.941 us, samples [60.924, 60.941, 61.178, 60.940, 60.942], spread 0.417%.
- My re-run gave 60.943 us.
- The gain is 1.6 us (2.6%), about 6x the spread. It is not noise.
- Versus baseline 153.463 us (spread 0.79%) the speedup is 2.518x.

Engine busy in the new profile: DVE 46.8 us (76.8%), ACT 39.1 us (64.2%), PE 18.7 us, Pool 15.5 us, DMA 21.7 us.

## 3. Safety

- **Verifier.** output_0 max_abs_err 2.322e-02 (83% of the 0.028 limit); output_1 1.633e-03. Both equal the baseline, so numerics did not move. The margin to the limit is a property of the kernel's bf16 rounding, not of this edit. Any edit that is not bit-identical must be checked against that 17% headroom.
- **Loosened synchronisation.** I walked each engine's cumulative "waited-for" semaphore values, aligned by instruction sequence. Two waits were removed that matter:
  1. ACT no longer waits `sem25 >= 32` (the second half of the 1.5 MB state_in DGE; was a fence at old ACT slot 12).
  2. ACT no longer waits `sem12 >= 16` (an input DMA; was a fence at old ACT slot 23, directly before a PSUM-reading ACTIVATE that needs no DMA data).

  Argument for (1): the four ACT casts that follow wait `sem6 >= 11..13`. sem6 is DVE's completion counter, and the DVE chunk ops that produce those values are themselves behind the DVE fences on `sem25 >= 16/32` (DVE slots 20..26). So the DMA completion is implied transitively through DVE. Argument for (2) is that nothing downstream on ACT reads that DMA's data directly; the DVE op that does read it waits on sem12 itself (DVE slot 19).
  Both are *probably* safe, but the verifier uses one fixed input and cannot tell a dropped-but-implied wait from a race. They were not proven by the notes.
- The one real loosening on DVE (PE `sem5 >= 60` reduced to `>= 59`) is the TT that no longer reads PSUM; the new STT carries the `sem5 >= 60` wait. This is correct.
- **Next step for any further deletion:** re-run `semaphores.py` and the cumulative-wait comparison above. The removed-fence pattern has now been used twice (iter 6 and 7), and the fences removed so far were the cheap ones.

## 4. Where the remaining time is (fresh profile, trace-clock us; trace span 73.5, exec 60.9)

| window | what limits it |
|---|---|
| 0 - 10.3 | fixed prologue (TENSOR_LOAD, semaphore resets, ordering-mode); not kernel code |
| 10.9 - 16.8 | Pool issues 4 input DGE DMAs serially, 1.25-1.74 us each (state_in is 1.5 MB) |
| 13.4 - 16.2 | ACT pc67: `ACTIVATE func36`, 1 channel, 3072 fp32 elements, 2.76 us (waits sem11 DMA). DVE is idle 12.0 - 16.2 |
| 16.2 - 19.3 | DVE pc68: `TENSOR_TENSOR` [128,22], 3.08 us, waits sem4 >= 2 (the ACT op above). This is the only slow outlier (others of the same shape take 0.94 us) |
| 19.7 - 70.0 | **DVE body, 50.3 us, essentially DVE-throughput bound**: 79% busy (busy 50.3 of 55 in 15-70), positive stalls 10.7 us total, of which 15 stalls >= 0.2 us sum to 6.6 us |
| 70.0 - 71.2 | tail: last state-out DGE (Pool, issued 69.05) and output_0 SP trigger (70.04); last DMA packet ends 71.15 |
| 71.2 - 73.5 | fixed epilogue |

DVE composition: 86 TENSOR_TENSOR (31.3 us), 46 SCALAR_TENSOR_TENSOR (15.3 us), 16 STREAM_SHUFFLE (2.8 us). The per-head [128,1,1] ops cost 0.27-0.34 us each, which is fixed overhead, not data.

## 5. Ranked candidates for iteration 8

| # | candidate | expected gain | ceiling and where it comes from | risk |
|---|---|---|---|---|
| 1 | Fuse the 20 per-head `v *= beta` TTs into 1-2 wide TTs | 1-2 us | 5.5 us = summed duration of the 20 TTs; the wide op costs about 3 us (3072 elems on one partition; compare pc68's 3.08 us for 22x128), so net DVE time saved is about 2.4 us at most | low |
| 2 | Pull no-update DVE TTs forward into the 15 stalls >= 0.2 us | 0.5-1 us | 6.6 us total stall time; 41 no-update TTs (11.3 us) are available, but stalls are dependency-bound. The sibling kernel got 0.54 us from 4 sinks | low |
| 3 | Pipeline the front chain ACT pc67 -> DVE pc68 (split both into 2-4 pieces) | 1-2 us | DMA lands 13.4, DVE body starts 19.7: 6.3 us window; a pipelined floor is about 4 us | medium |
| 4 | Split the last state-out DGE (Pool pc70) so only a 1-head piece remains after the last DVE op | 0.3-0.7 us | tail between last DVE op end (70.0) and last DMA end (71.15) is 1.15 us, of which the minimum DMA latency is unknown | low |
| 5 | Issue two of the four input DGEs from SP/ACT instead of Pool | 0-3 us, unknown | Pool serialises 4 DGEs over 10.9-16.8; whether a DGE on SP works for this ring is the unknown named in the LNC2 sibling's iter-5 notes | high until probed |

### Candidate 1 (best): hoist the 20 `v *= beta` DVE TTs into wide ops

- **Engine / member / slots:** DVE, `sg00/DVE0.bin`. The 20 `TENSOR_TENSOR` MULTIPLY slots of the form `src0 = fp32@0x34e00 + 0x200*k  (1 partition, 128 elements)`, `src1 = fp32@0x4120 + 4*k` with step `[0,0,0]` (scalar broadcast), `dst = src0`. They are the `W[sem4>=12]` TTs visible in the profile (pc 147, 155, 174, 189, 227, 237, 258, 276, 320, 348, 351, 385, 424, 427, 497, 516, 524, 562, 574, 590).
- **Change:** replace them by one (or two, halves of 12 heads) TT whose `src0` is the contiguous run starting at 0x34e00 (`num_elem` up to 24*128), `src1` is the beta row with a 2-D pattern `[[1,24],[0,128]]` (step 1 over heads, step 0 over the 128 elements), and `dst = src0`. The 20 narrow slots become deletions with fences/wait re-pointing, or NOP-equivalents that keep sem counts (none of the 20 has a sem update; all wait sem4 >= 12 only).
- **Why it should work:** the per-head operands are exactly contiguous: v rows at stride 0x200 (128 fp32), beta at stride 4 B (0x4120 .. 0x417c = 24 scalars). All 20 wait the *same* value, sem4 >= 12, so there is no ordering reason for them to be spread over the body. The multiplication is identical, so the result is **bit-identical**. The multi-group broadcast `src1` is the same idiom the LNC2 sibling verified (its iter 2: "two groups collapsing into one 1536-element instruction", docs 14.3 #4).
- **What stays the same:** DVE-to-DVE ordering needs no semaphore (the existing TT -> STT pairs already rely on in-order execution). The first consumer STT must come after the wide op in program order.
- **Placement:** put the first half right after the sem4 >= 12 point and before head 0's STT, the second half into a stall (see candidate 2).
- **Measured ceiling:** 5.5 us summed duration of the 20 TTs (profile). Net gain is bounded by the cost of the wide op, hence 1-2 us after allowing for the stalls absorbing part of it.
- **Risk:** low. The verifier would catch a wrong stride (output wrong), and a missing ordering would show as a hard wrong value in the first head. A leftover narrow slot that now double-multiplies also shows as a wrong value. A race cannot be introduced because no semaphore is changed.
- **Cheapest falsifier (minutes):** put the wide op in, leave the 20 narrow TTs as no-ops by turning them into fences, and look at `duration_ns` of the wide op in a one-repeat profile. If it is above about 4.5 us (> the 5.5 us it replaces by too little), kill it. Four heads (those at 0x41e8 etc.) are handled by a different path; check by diffing the dst extents that no head is multiplied twice.

### Candidate 2: forward the no-update sinks into DVE stalls

- **Where:** `DVE0.bin`; the 41 `TENSOR_TENSOR` slots with no `events_update_mode` (the fp32 S clone `S -= outer` that only feeds state_out, and the v*beta TTs) move to just before the 15 stalls >= 0.2 us. The stall instructions are at trace times 26.5, 32.4, 34.3, 39.1, 41.6, 42.2, 48.8, 54.3, 54.9, 57.9, 62.6, 65.0, 65.7, 67.7, 69.4 (pc 105, 131, 138, 155, 165, 167, 192, 213, 214, 224, 240, 247, 249, 253, 257).
- **Mechanism:** this is the lever that gave the LNC2 sibling its iter 7 (-0.54 us for 4 sinks); docs 14.2 lists "reorder independent instructions on one engine (re-point waits)" as verified (r08).
- **Ceiling:** 6.6 us of stall >= 0.2 us, but each stall is waiting for ACT or PE. The tail stalls (64.99-69.36, 2.1 us) are the end of the dependency chain and cannot be filled. Realistic 0.5-1 us.
- **Risk:** low if the byte extents of the moved sink's source are not written in the hop window (the LNC2 notes list that check) and its dst is not read. A mistake shows as wrong output or none.
- **Falsifier:** compute, per stall, whether a movable sink exists before it in program order; if fewer than about 6 pairs exist, drop.

### Candidate 3: pipeline the front chain

- **What:** ACT pc67 (`ACTIVATE func36`, src 0x834e00, dst 0x20000, 3072 elements, 1 channel, 2.76 us) and DVE pc68 (`TENSOR_TENSOR` [128,22], src0 0x34c00 step 0, src1 0x20000 step 128, 3.08 us) form a serial chain 13.4 -> 19.3 with DVE idle before it. Split the ACT op into 2-4 element ranges and the DVE op into matching pieces so DVE starts after the first ACT piece.
- **Ceiling:** the window from the data landing (13.4) to the DVE body (19.7) is 6.3 us; two serial ops cost 5.8 us; a pipelined version would be about 4 us. Unknown: whether the 3.08 us duration scales with element count (it is 3x the 0.94 us of the same-shaped ops) or comes from the strided read of src1 (step 128 elements = 512 B).
- **Risk:** medium: new slots and sem4/sem6 renumbering (waiters of sem4 >= 2 would have to wait the last piece). A missed wait would pass if timing is lucky and fail later.
- **Falsifier:** one-repeat profile of the ACT op alone with half the `num_elem`: if the duration does not halve, splitting does not help.

### Candidate 4: shrink the tail

Split the last state-out DGE (Pool pc70, issued 69.05, 0.88 us generation) into a smaller final piece. The LNC2 sibling's iter 4 did the same with the whole state-out DMA. The ceiling is the 1.15 us between the last DVE op ending at 70.0 and the last DMA packet ending at 71.15, minus the minimum DMA latency (unknown). Pool is idle then, so extra DGEs cost nothing. The sem27 total must stay at 128. Falsifier: check the last DMA packet end time against the last DVE op end in the profile; if the gap is under 0.5 us, drop it.

### Candidate 5: DGE from SP or ACT

The LNC2 iter-5 notes (candidate 4) named this as the only way to break the "Pool generates the input DGEs serially at 1.25 us each" limit; chapter 9 says `PSEUDO_DMA_DIRECT2D` is legal on SP and Activation. It was never tried. The unknown is whether `src_table_index` is engine-relative. Falsifier: move one small input DGE (a 24-element one) from Pool to SP and check that the output is unchanged. If it fails with an OOB status the lever is dead. The gain is only possible if the 4 DGEs (10.9 -> 16.8) gate DVE, and they do not obviously: DVE does not start until 16.2 anyway because of the ACT chain in candidate 3, so this only pays after candidate 3.

## 6. Closed

- **Pool as host for CAST or per-head TENSOR_TENSOR** (LNC2 sibling, iter 5/6): Pool CAST is about 4x slower than ACT (3.35 and 2.71 us vs 0.81 and 0.83 us), a running Pool op steals bandwidth from DVE, and a Pool TT gave output_0 error 4.818, VERIFY FAILED. Do not redo. Also: Pool cannot read PSUM.
- **Moving ACT <-> DVE elementwise work blindly:** closed in the sibling (iters 3/5/6), but in *this* kernel iters 5 and 6 moved ACT work to DVE and won (66.3 -> 62.5 us). That worked because ACT was on the critical chain. DVE is now 76.8% busy and ACT 64.2%, so the next move in that direction needs a measurement, not a guess.
- **Removing waits on an engine's own completion semaphore:** wrong output on DVE, OOB on Pool (docs 14.2, o01).
- **Re-deriving the ACT head table-load order, and further splitting of the state write-out in halves:** done in the sibling (iter 4/5) and in iter 3 and 4 of this loop (commit 66f6259).
- **Reordering PSEUDO_DMATRIGGER or the qSyncIO descriptor ring:** closed in the sibling (iters 3/4).

## 7. Unknown (would change the ranking)

- **Is the 3.08 us pc68 duration per-element or per-stride?** Settle by splitting it and reading the new durations. It moves candidate 3 up or down.
- **Do the 4 remaining heads without a `v *= beta` TT (STT at 0x41e8 etc.) use a different beta layout?** Settle by diffing their dst extents; it decides whether the wide op covers 20 or 24 heads.
- **Minimum latency of a small tail DMA:** one 24-element DGE measured in the profile gives it. It bounds candidate 4.
- **Can bf16/2x DVE perf mode be used for the [128,1,1] ops?** Header advertises `v3-2x-4x-dve-perf-mode`; docs 14.3 #5 says unmeasured, and it changes the numbers, so it would need a check against the 17% error headroom (output_0 is at 83% of the limit).
- **Is the (psum - x) * s fold of the `v*beta` TT into the following `MULTIPLY,SUBTRACT` STT acceptable on error?** It removes the TT entirely (a 20 x 0.27 us = 5.5 us saving, more than candidate 1's net) but is not bit-identical (`a*s - x*s` vs `(a-x)*s`). It only pays if the output_0 error stays below 0.028, which sits 17% above the current value. Test it on a single head and read verify's max_abs_err before extending.

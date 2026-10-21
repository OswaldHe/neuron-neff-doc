# Review of iter-6 (kernel.neff vs HEAD 6d77c44)

## 1. What changed

`neff_diff.py HEAD kernel.neff`: 3 members changed, all in sg00; no arithmetic, no address or AP field touched.

- **Activation0.bin** (82 -> 83 slots)
  - Slots 77..79 rotated: `TENSOR_SCALAR, ACTIVATE, EVENT_SEMAPHORE` -> `ACTIVATE, EVENT_SEMAPHORE, TENSOR_SCALAR`.
  - The old ACT[81] `TENSOR_SCALAR` (wait S[4]>=0x42) is now slot 82.
  - Slot 81 is now a `PSEUDO_DMA_DIRECT2D` row (wait S[6]>=0x41, i.e. 65).
- **Pool0.bin** slot 20: that same DGE row (wait S[6]>=65) is replaced by an update-less `EVENT_SEMAPHORE`.
  The final-output store DGE row moved from Pool to Activation.
- **PE0.bin** slots 141/142: the two waits swapped (S[6]>=64 and S[27]>=32).
  PE is in-order, so the combined precondition for the LDWEIGHTS in slot 142 is unchanged.

`semaphores.py` HEAD vs new differs in one line: sem 16 source `dge:Pool` -> `dge:Activation`.
Every semaphore keeps the same total_inc and max_wait (S[16] 16/0 before and after, S[4] 41/40, S[6] 66/66).
No wait was removed or reduced; the PE swap only moves where each wait sits.

`notes/` contains only the template README, so there is nothing to compare the change against.
The edit is a plain "move a DGE row to a different engine" plus a local reorder.
This matches the style of the earlier commits (e833bbc, ac545ca).
It breaks the README rule that notes record every attempt.

## 2. Is the latency change real? No: it is neutral.

- `profile_latest.json`: median 38.224 us (min 37.78, max 38.34, spread 1.46 %). That is -21.2 % vs the original 48.519 us baseline, but that gain comes from iterations 1-5.
- Against HEAD (6d77c44, commit message says 38.138 us) I ran interleaved 3x3 captures:

| NEFF | repeat medians | per-repeat range |
|---|---|---|
| iter-6 | 38.281, 38.202, 38.224 | 37.78 - 38.34 |
| HEAD | 38.180, 37.978 | 37.94 - 38.37 |

- HEAD is 0.0 to 0.2 us *faster* than the iteration, within the 0.2-1.4 % spread.
- The evidence says the delta is zero.
- Moving the store row from Pool to Activation changes nothing on the critical path.
  The Scalar DGE fires at 48.317 (S[6]=65 lands at 48.296), but the exit is set by the other store, Sync[40], which waits on DVE[100] (48.70) and issues at 48.718.
- The iteration should not be credited with a win. Keeping it is harmless but does nothing.

## 3. Safety

- Sync is intact (see section 1). The moved DGE row keeps its own S[6]>=65 wait and its sem-16 update (16 increments, no waiter, same as before).
- Verifier: max_abs_err 6.119e-05 against a limit of 3.5e-4 (17 %), cosine 0.999995, atol/rtol check at 100 %.
- Error is unchanged from earlier iterations (the arithmetic is untouched).
- Residual risk: a single fixed input set cannot show a race.
  Nothing was loosened here, so the risk is low.
- The sem-16 completion has no waiter. Program end relies on the runtime's implicit DMA drain.
  That was already true at HEAD.

## 4. Where the remaining time is (profile, not intuition)

The metric window is about 12.14 us (first SET_ORDERING_MODE) to about 50.8 us (Pool EVENT_SEMAPHORE #22), i.e. 38.7 us. The first 12 us of the trace is runtime preamble.

Three DMA phenomena dominate:

- **Queued-transpose DMA hop.** The Sync WRITE trigger -> DMA start takes about 0.72 us, the transfer 0.25-0.3 us, and the completion lag about 0.3 us. Total is 1.3-1.5 us per hop (e.g. trigger 45.801 -> S[27]=32 at 47.287).
- **DGE row cost.** Each `DMA_DIRECT2D` row blocks its engine for 0.60-0.73 us. A queued WRITE costs 44-70 ns.
- **HBM store completion.** About 1.75-2.05 us from row issue to completion (S[17] first update 50.56, exit 50.77).

Critical tail, 45.8 -> exit:
`DVE CAST[95]` -> `Sync WRITE[37]` trigger (45.801) -> 8 KiB transposing DMA -> `PE[142]` (47.328) -> `PE[143]` -> `DVE[98]` TS -> `DVE[99]` TT -> `PE[145]` -> `DVE[100]` TT (-> 48.69) -> `Sync[40]` store (48.718) -> exit 50.77.
The co-critical branch is the 256 B S[28] DMA (46.338 -> 47.607) -> `ACT[80]` (47.63-48.02) -> `DVE[99]`.

Mid-kernel there is a 4.2 us PE hole (38.07 -> 42.26). It is filled by four DGE/queued-DMA triggers (38.39, 38.61, 38.69, 39.31) plus their latency.

Startup, 12.2 -> 19:

- Five input DGE rows are serial on Pool (pc 7-11, 13.43 -> 16.98, 0.66-0.73 us each). They are programmed behind `AFFINE_SELECT` pc 3/5/6 (12.52-13.47).
- Sync is idle 12.97 -> 16.05 and Scalar is idle 13.64 -> 16.06 (after ACT_TABLE_LOAD).
- Completion times per row (first update of S[9] ... S[13]) are 15.34, 15.93, 16.61, 18.14, 18.49.

## 5. Ranked candidates for the next iteration

| # | Candidate | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|
| 1 | Parallelise the startup input DGE rows | 0.5-1.5 us. Ceiling: S[12]/S[13] land 18.1/18.5 against 15.3 for S[9]; if all five rows landed with the first, 2-3 us is the absolute bound, and only if the inputs gate the 19-22 us compute | Low | Move only Pool row pc 11 to Sync and see if S[13] completion drops by about 0.7 us in `SemaphoreUpdate` |
| 2 | Hoist the Pool DGE rows above the `AFFINE_SELECT`s | Up to 0.7-0.9 us on the first load (13.43 -> about 12.55) | Low | Profile and read the pc-7 start time; first consumer Vector[8] at 15.326 must move |
| 3 | Re-home the 4-5 `DMA_DIRECT2D` rows in the 38-40 us PE hole | 0.3-0.7 us. Ceiling: the hole is 4.2 us, about 1.5 us of it intrinsic DMA latency | Low-medium | Look at which slot gates the 38.39-39.31 chain |
| 4 | Replace the tail's queued-transposing DMA with a PE transpose | Up to about 1.0 us (hop 1.49 us vs PE ident-matmul 0.3 + evac 0.25) | High (new PE and DVE instructions, PSUM allocation, semaphore counts) | Check whether an identity tile and a free PSUM bank exist before starting |
| 5 | Pre-programme the final output store as a queued WRITE trigger | Up to 0.6 us (667 ns row vs about 50 ns WRITE) | Likely infeasible (output address is runtime-bound) | Does `def.json` / `SP0.json` have a ring for the output variable? If not, drop it |

### Details for the top three

**1. Parallelise the startup input DGE rows**

- *Where:* `Pool0.bin` compiler_pc 7-11 (the five serial rows). Move to `SP0` (Sync) and `Activation0` (Scalar), which are idle 12.97-16.05 and 13.64-16.06.
- *What:* move 2-3 of the five rows. Each carries its own `dge:<engine>` semaphore, so rewrite the source engine's sem-naming exactly as e833bbc did for sem-43/sem-45. Re-point the consumers' waits only if the destination engine's semaphore index changes.
- *Mechanism:* DGE row relocation between engines is verified, see commits e833bbc and ac545ca.
- *Check:* each of S[9]..S[13] keeps total_inc 16, and `semaphores.py` must show `dge:Sync` / `dge:Activation` as the new source.
- *Verifier:* these are read-only HBM loads, so a wrong address would be caught. A missing wait cannot arise as long as no wait changes.

**2. Hoist above the AFFINE_SELECTs**

- *Where:* Pool pc 3/5/6 delay row 7 until 13.43. Pool pc 2/4 are MOVEs; check whether they set DGE registers before moving the row over them.
- *Mechanism:* "Reorder independent instructions on one engine" is verified (doc 14.2, r08). The S[3] threshold for the selects is unchanged as long as the DMA rows do not touch sem 3.

**3. Re-home the DMA rows in the 38-40 us PE hole**

- *Where:* Pool pc 14, Sync pc 24/25, Scalar pc 66 at 38.39-39.31 (each blocks its own engine for about 0.64-0.69 us).
- *What:* spread them so they fire together. Only the gating wait decides the start, so this is bounded by the earliest wait (about 38.1).
- *Mechanism:* same DGE relocation as in 1.

## 6. Closed

- **Moving the output-store DGE row Pool -> Activation (this iteration):** 0.0 +/- 0.15 us, not on the critical path. Medians 38.2-38.3 vs HEAD 38.0-38.2.
- **Adding more filler / stand-in EVENT_SEMAPHORE reorders in the ACT/DVE tail:** iteration 5 got only -0.37 % (0.14 us) for four edits. The remaining ACT/DVE gaps are DMA-latency gaps, not ordering gaps.
- **ACT_TABLE_LOAD (1.283 us at 12.36):** off the critical path. Scalar is not needed until 16.06.

## 7. Unknown

- **Do the startup input rows gate the first compute?** Settled by finding the waiters of S[12]/S[13]; the cheapest test is candidate 1's falsifier.
- **Is there slack in DMA trigger latency (0.72 us)?** Compare queued WRITE -> start for the fastest and slowest examples in `DmaPacketAggregated`.
- **What the 256 B DMA S[28] at 46.338 is for** (it feeds `ACT[80]`) and whether it could be done as a copy on an idle engine.
- **What gates Pool DGE[18] (44.562)**, which completes S[15] only at 46.5.

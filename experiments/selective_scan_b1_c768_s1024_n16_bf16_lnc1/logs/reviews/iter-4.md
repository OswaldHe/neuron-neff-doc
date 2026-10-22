# Review of iter-4 (selective_scan b1 c768 s1024 n16 bf16, LNC=1, sg00)

Baseline 602.64 us, HEAD (commit 5e1f980) 510.7 us, this iteration 508.77 us.
`notes/` is empty (README template only), so the claims are checked against the HEAD commit message and the NEFF.

## 1. What changed (vs HEAD; vs baseline it is the three commits stacked)

Diff against HEAD `kernel.neff`: 2 members, 13 slots.

| member | slot(s) | change |
|---|---|---|
| `DVE0.bin` | 105, 192, 280, 369, 457, 547, 630, 724, 899, 982 (10 slots) | `TENSOR_TENSOR_ARITH_OP` (y = zero + Cs) becomes `COPY` (y = Cs). The two source operands collapse to one, and the dst/src patterns are rewritten as `S4D4`. Bytes 4..11 are untouched, so the wait/update (events) block is unchanged. |
| `Pool0.bin` | 8, 303 (2 slots) | Same ADD to COPY rewrite. |
| `DVE0.bin` | 3 | The shared zero-buffer MEMSET at 0x15400 goes from 512 elements to 1 (`dst_element_count` 512 to 1, `dst_mem_pattern_num_elem[0]` 512 to 1). |

This is consistent with the preceding commit's design. That commit made the first `y+=Cs` writer of each of the 12 groups read a shared zero buffer.
This iteration drops the zero operand, so the first writer becomes a COPY and the zero buffer is no longer needed.
I scanned all five streams for the 0x15400 address: the only remaining reference is the 1-element MEMSET in DVE slot 3.
The buffer is now write-only dead state.

The sync fields are identical, so no semaphore count changed.
The change is bit-exact in principle (0 + x = x in fp32); the result is the same value and the same semaphore ordering.
Nothing is written to `notes/`, so there is no written account of the iteration, the 12 sites, or the measurement.

## 2. Is the latency change real?

| | median (us) | min..max | spread |
|---|---|---|---|
| baseline | 602.637 | 602.517..602.829 | 0.052% |
| this iteration (`profile_latest.json`) | 508.77 | 508.714..508.95 | 0.046% (0.24 us) |
| my re-profile (`--repeats 1`) | 508.904 | | |

HEAD's own number was 510.7, so the gain from this iteration is about 1.9 us (0.37%).
That is roughly 8x the 0.24 us spread and agrees with an independent single run, so it is real, if small.
The sample sets do not overlap.
HEAD's 510.7 comes from the commit message, not from a re-measured profile of the HEAD NEFF.
So the loop is not chasing noise at this size, but it is now in the range where each win is under 0.5%.

The profile shows DVE at 491 us busy (96.6%) and Pool at 416 us (81.8%) of a 509 us wall.
So wall = DVE busy + about 18 us.
This is the same DVE-bound shape that the c2560 sibling case ended up in (wall = DVE busy + a constant).

## 3. Safety beyond the verifier

- `semaphores.py` on the unpacked NEFF: for every compute semaphore (S3 Pool, S4 ACT, S5 PE, S6 DVE) `max_wait` equals `total_inc` (282, 194, 374, 640).
  The DGE semaphores S15 and S18 have max_wait 48 against 64 increments; that is a ring-wrap pattern and is not attributable to this change, because the diff does not touch events fields.
- No wait was loosened in this diff: bytes 4..11 are identical for all 12 converted slots, so every wait and every update is as before.
  The removed dependency is the read of the zero buffer, and that buffer's only writer is a 1-element MEMSET.
  The first writer of y therefore no longer waits on the MEMSET at all, and it does not need to, since it no longer reads it.
- The COPY reads the same Cs range the ADD read as src1. The dst is unchanged. A hazard would require that src0 of the old ADD was also the destination, which is not the case here.
  I did not run a full five-stream alias scan (the c2560 sibling's iter-7 notes show this region is shared across engines), but nothing moved in time, so the risk is low.
- Error: `max_abs_err` is 6.245e-02 on output_0, which is 83% of the 0.075 limit (cosine 0.999998), and 4.27e-05 on output_1.
  The commit chain describes the output as bit-identical to baseline, so this is the baseline bf16 rounding error and it did not move.
  The headroom is not related to these edits.
- The verifier uses one fixed input set. These edits are value-preserving (x + 0 vs copy of x), so a data-dependent failure is not plausible.
  Only a race could pass once and fail later; I found no new race surface.

Verdict: accept. The gain is small but real and the edit is safe.

## 4. Ranked candidates for the next iteration

Context from the profile and NEFF: DVE has 420 `TENSOR_TENSOR` at about 0.69 us (290 us), 192 `TENSOR_TENSOR_SCAN` at 1.27 us (245 us), 10 COPY, 24 MEMSET and 418 ES slots.
Pool has 168 TT at 2.0 us (338 us), 117 COPY (39 us) and 60 `DMA_DIRECT2D` (39 us).
The sibling case's measured model (c2560 s4096, `neff_doc/experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/logs/notes/iter-5..7.md`) is: DVE duration = 0.95 ns/element-op + 203 ns/instruction.
I confirmed that the same shape shows up here: TT of 512 elements runs at 690 ns and a scan at 1275 ns, in line with that model.
The model transfers by analogy only. Nobody has re-measured it on this c768 NEFF.

The gains below are small. If the next agent wants a clean stop, "at the floor" is a defensible outcome.

| # | Candidate | Ceiling and source | Risk | Cheapest falsifier |
|---|---|---|---|---|
| 1 | **Retarget the Cs producer straight into y, drop the 12 COPYs.** Engines: DVE slots 105, 192, 280, 369, 457, 547, 630, 724, 899, 982; Pool slots 8, 303. The COPY source is the Cs tile (for example 0xc140 in slot 105; destination 0x9140). Find the slot that writes that Cs range (the C(psum) x h `TENSOR_TENSOR` shortly before it), set its dst to the y accumulator, and turn the COPY into an `EVENT_SEMAPHORE` that keeps its events block. Neutralising a slot into an ES that keeps its `SEM_INC_COMPLETE` was verified on the c2560 sibling (iter-7 section 2). | 10 DVE COPY x 415 ns = 4.2 us DVE, plus the 2 Pool COPYs. Wall gain is at most about 4 us (0.8%) if wall tracks DVE 1:1; I would expect 2 to 3 us. Bound: profile COPY duration sum (4156 ns DVE). | Medium. The Cs tile may also feed the next group state or a carry. Dtype of the producer dst must match fp32 y. Pool-side hand-off waits sit on the old ES slots. The verifier would catch a wrong value but not a race. | Before editing: for each of the 12 sites, list every reader of the Cs source range across all five streams. If any other slot reads it, kill the site. Takes minutes, no device time. |
| 2 | **Probe the Pool exchange rate on this NEFF before moving anything between engines.** Timing-only probe: shrink about 100 Pool ADDs from 512 to 64 elements (wrong math), profile, and compare the wall change against the Pool change. | Unknown. It prices levers 3 and 4 here and settles whether wall = DVE busy + 18 us holds. On the sibling the marginal rate was 0.15 us of wall per us of Pool removed, 0.68 to 0.89 per us added. | Zero (not shipped; measurement only). | The probe is the falsifier: if the wall does not move, Pool is not on the critical path and the whole Pool family is closed. |
| 3 | **Merge DVE scan pairs along the sequence axis.** The 192 scans are non-PSUM DVE computes, so they can in principle be paired. | Ceiling 96 merges x 203 ns = 19.5 us (3.8%); realistic about 80 ns each = 8 us (1.5%) from the sibling's measured -31 us over 390 merges. Source: sibling iter-5 section 4 and iter-6. | High. The sibling never got this working: it needs window-private SBUF, a C(PSUM-read) tile dragged later, and a carry chain across the row boundary. A hazard can pass one fixed input and still be wrong. Verify bit-exactness. | Check first whether this NEFF has free SBUF for the merged tile (the sibling's `freescan.py` found only 1216 B free). If there is none, kill it. |
| 4 | **Pool carry COPY (117 slots, 38.8 us Pool) to ACT or ES.** | At most 0.15 x 38.8 = about 6 us, and the sibling found only 28% of sites were placeable (ACT lead cycle). Realistic 1 to 2 us. | High for the return. The shared region is written by ACT, Pool and DVE; a five-stream alias check is mandatory. | The sibling's `D(v)` placeability test over the 117 sites. If under 30% pass, drop it. |
| 5 | **Dead zero-buffer cleanup.** DVE slot 3 (the 1-element MEMSET at 0x15400) now has no reader. Make it an ES that keeps its S6 increment. | Under 0.2 us. | Low. | Confirm no reader again with the same grep and check S6 `total_inc` stays 640. Free, but only worth doing together with candidate 1. |

## Closed (do not redo)

- **Remove the S6 self-waits and fold the zero memsets** into one buffer: done in 5e1f980 (516.5 to 510.7 us). This iteration finishes it (-1.9 us).
- **Move the y+=Cs ADDs from DVE to Pool**: done in 6e7af17 (602.7 to 516.5 us). Pool is now 81.8% busy. On the sibling case, moving more work onto Pool cost 0.68 to 0.89 us of wall per us added, and moving `dt*x` to Pool was +449 us. Do not move more DVE work onto Pool without the probe in candidate 2.
- **Reordering DVE slots that read PSUM** (sibling iter-6/7): moving a PSUM read earlier costs about 526 ns per swap; non-PSUM reorders are free. Do not move B/C multiplies.
- **Reshaping the Pool access patterns**: zero effect (sibling iter-7 section 1.4).
- **Merging or relocating Pool `DMA_DIRECT2D`**: dead on the sibling (queues are per engine, every intermediate value is waited on).
- **Merging adjacent Pool carry COPYs further**: the sibling found only 149 of 1587 sites; about -4 us.
- **ES and 1-element slots as cheap neutralisers**: ES costs about 19 ns; this is a tool, not a lever.

## Unknown (would change the ranking)

- **Is wall still DVE busy + a constant here?** Settled by the probe in candidate 2. If Pool is on the critical path, candidates 1 and 4 change value.
- **How much of the 491 us DVE busy is real DVE work versus Pool-induced stall?** The same probe answers it (on the sibling, 80 us of the 6522 us was stall).
- **What reads the Cs tiles apart from the 12 COPYs?** Candidate 1 depends on this; a static scan of the five streams settles it.
- **Free SBUF for candidate 3.** A free-range scan of the five streams (allocation intervals above 16384) settles it. Not measured here.
- **Whether the 18 us wall-minus-DVE gap includes a startup/tail floor** (the sibling had 10 us head and 4 us tail). Check the first and last DVE `start_ts` in `Instruction.parquet`: DVE first start is at 4.3 us and the last end at 521.3 us of a 521.2 us span, so head is about 4 us.

## Bottom line

This iteration is a real but tiny win (-1.9 us, 0.37%) from a value-preserving edit. The remaining levers are each at most a few microseconds, apart from the scan-pair merge, which is high risk.
If candidate 1's reader scan fails and the probe in candidate 2 shows no headroom, the kernel is at its NEFF-editing floor (about 509 us) and the loop should stop.

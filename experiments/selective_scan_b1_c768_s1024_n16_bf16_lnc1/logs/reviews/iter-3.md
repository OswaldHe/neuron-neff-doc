# Review: iter-3 (uncommitted `kernel.neff` on top of 6e7af17)

Measured by me: `tools/profile.py --repeats 1` on this NEFF gave 510.694 us, in line with `profile_latest.json` (median 510.633, samples 510.43..511.05).

## 1. What changed (vs HEAD 6e7af17, and vs baseline)

**vs HEAD (this iteration's edit).** Only `sg00/DVE0.bin` and 2 slots of `sg00/Pool0.bin` differ. Activation, PE and SP are identical.
- **DVE slot count 1216 -> 1057.** The compute mix is unchanged: 430 TENSOR_TENSOR, 192 SCAN, 24 MEMSET. `EVENT_SEMAPHORE` slots went 569 -> 410, so 159 ES slots were deleted.
  - 1201 of the 1216 old slots differ by index, which is just the shift.
  - Surviving ES slots take over waits through `events_extended_wait_{mode,idx,sem_wait_value}`, e.g. slot 22 gains `WAIT_FOR_SEM_GE_IMM S[4]>=4`.
  - `events_semaphore_value` is renumbered on the surviving slots.
- **S[6] waiters** (DVE's own completion semaphore) fell 1388 -> 1218, i.e. 170 fewer. S[3], S[4], S[5] are untouched.
- **Zero buffer.** DVE slot 3 (a 512-element fp32 MEMSET 0) moved its dst 0x4940 -> 0x15400 (87040). 10 DVE adds and Pool slots 8 and 303 (all `src0`) now read 0x15400 as a shared zero buffer instead of a per-group y_accum.
  - I scanned all five instruction streams. Nothing else touches 0x15400..0x15bff, so it is a free hole.
  - I did not check DGE/queue DMA descriptors for that range. They stage at 114688+ and 223808+, so this is probably fine.

**vs baseline.** The committed iter-1 work is still there: 170 y+=Cs ADDs moved DVE->Pool, Pool carry COPYs merged (192 -> 115), and waits moved onto ES extended-wait fields.

**Consistency with the notes.** Not checkable. `notes/` contains only the stub README, so this iteration wrote no notes, and there is no commit message for it yet.
- The diff reads as "delete redundant DVE ES slots and wait fields, and re-point the y_accum zero-init". That fits the sibling case's iter-4 edit (`neff_doc/experiments/selective_scan_b1_c2560...`).
- **Ask the agent to write the notes before the next iteration.**

## 2. Is the 516.5 -> 510.6 us gain real?

Yes.
- HEAD was 516.5 (the commit says 516.79). The new median is 510.63 over 5 runs, spread 0.12% (about 0.6 us).
- The step is about -6 us, roughly 10x the spread. Total vs baseline (602.64, spread 0.05%) is 1.180x.
- Engine busy: DVE 494.3 us (96.8%), Pool 415 (81%), PE 219, ACT 146.
- Output 6.245e-02 is identical to baseline, so the arithmetic is unchanged.

The loop is not chasing noise yet, but the gains are shrinking: -86 us, then -6 us.

## 3. Safety beyond the verifier

- **Semaphore balance.** `semaphores.py` shows max_wait == total_inc for S[3..6], with no wait exceeding its supply. No hang risk.
- **Coverage check.** I computed the cumulative wait guarantee at each of the 646 DVE compute slots, HEAD vs now.
  - Identical at 642 of 646. The 170 removed S[6] waits were all dominated by waits that remain.
  - Weaker at 4 slots, all on S[6], by 1-2 completions: compute ordinals 10 (prologue), 372, 480 and 587.
  - I hand-checked the SBUF ranges for 372, 480 and 587 against the newly uncovered predecessors. The 2 KB buffers are adjacent, not overlapping, so no RAW/WAR/WAW. Ordinal 10 (the prologue) I did not check.
  - The remaining checks are the all-five-stream alias checks that the sibling notes (iter-7 s3b) say are required in the 16384..60224 rotating region.
- **Race exposure.** The edit now relies on in-order DVE issue and on data-address disjointness rather than on waits. The verifier uses one fixed input and cannot see a timing race.
  - Cheap guard: run `verify.py --runs 3` and require bit-identical output across the runs.
- **Error margin.** Unchanged: output_0 is 83% of the limit (0.0624/0.075), identical to baseline. No error moved.

## 4. Ranked candidates for the next iteration

**Ceiling logic.** Everything below is bounded by this fact: DVE is the wall.
- DVE busy is 494 us and its span is 14.9..518.8 us, so only about 10 us of in-span idle. Wall is roughly DVE busy + 16.
- DVE work is fixed: 192 SCAN at 1274 ns (245 us), 192 BMUL at about 656 ns (about 126 us), 192 CMUL at about 677 ns (about 130 us), 34 y-adds at 692 ns (23.5 us), 12 dt*x (8 us), and MEMSET/ES.
- Per the sibling kernel, DVE time is 0.95 ns per element-op + 203 ns per instruction.
- The sibling's measured headroom was 1.7% of wall (113 us of 6604). The analogue here is about 9 us. Treat anything promising more as suspect.
- The sibling's mechanisms were measured on a c2560/s4096 NEFF with the same code structure. I confirmed the instruction mix and the 3-op-per-state structure here, but I did not re-measure its constants on this NEFF.

| # | Candidate | Engine / slots | Change | Mechanism and source | Ceiling (from) | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Prune the remaining non-binding DVE S[6] self-waits | DVE0.bin. 424 S[6] waiters remain (lag histogram: 7 at lag 2, 219 at lag 3, 19 at lag 4, 124 at lag 5, 22 at lag 6, 30 at lag 7, 3 at lag 8). Prefer lag >= 5 first (about 180) | Delete the wait, or move it to the first slot that actually reads or writes the producer's range. Keep ES slots as no-op carriers for the S[6] ordinal, or set `SEM_INC_COMPLETE` on the ES to keep counts (sibling iter-7 s2) | This edit's own evidence: 170 S[6] waits removed gave -6.2 us, about 36 ns per wait. Doc ch.10 and the sibling iter-5 s2a (hardware RAW turnaround 118-170 ns) | At most 424 x 36 ns = 15 us; realistically 2-5 us, since most remaining waits protect real PSUM->SBUF RAWs. Bound: DVE busy 494 | Medium-high. A removed wait that was real passes the fixed input once and races later. Needs the exact-range alias check over all five streams (sibling iter-7 s3b: ACT dA and Pool y_accum write the same region) | Timing-only probe: strip every lag >= 5 S[6] wait (output may be wrong). If wall drops < 1.5 us, the candidate is dead. If it drops, build the real version with alias checks |
| 2 | Scan-pair merge along seq: `scan(t,n)` + `scan(t+1,n)` -> one `[512,2]` scan | DVE: 96 of 192 SCANs (6 groups x 16 states; only chunks 0 and 1 per chain). Pool: the 96 carry COPYs between chunks go away. PE: re-bank | Merge the two scans into one instruction with a 2-D pattern. The carry flows across the dim boundary, and the sibling iter-1 showed `[256,2]` is bit-exact | Sibling iter-5 s4 and iter-7 s0: removes the 203 ns fixed cost per instruction minus the 77 ns boundary overlap. Reordering is free if PSUM reads keep their positions (iter-7 s0). The remaining blocker was SBUF | 96 x 126 ns = 12 us (2.4%); Pool also sheds about 96 x 0.3 us of copies. Bound: DVE busy | High. Needs window-private dBx/h/Cs buffers (the sibling found no free SBUF). This kernel has more room: the hole near 0x15400+ is partly in use now, and the free region starts higher. Large edit across DVE, Pool and PE. A timing-correct but arithmetically wrong build can pass | Falsifier first: count free SBUF (exact reference counts, not bounding boxes, `freescan` style) for 2 x 2 KB x ~6 buffers. If there is none, drop it. Then a timing-only probe: shrink 96 SCANs to half-length and see whether wall falls by about 10 us |
| 3 | Tail: shorten the last output DMA / issue it earlier | Pool DMA_DIRECT2D at 519.19 us (last two), SP DMA triggers; DVE ES at 518.9..522.0 | Split the last chunk's final output store so the first half leaves before the last y-add. Or point the last DMA at the earliest ready tile | Profile: last DVE compute ends 518.83; the DVE end ES waits 3.06 us for the final output DMA; postamble to 523.46 | About 1-1.5 us (0.3%). Bound: DMA latency of about 2.4 us after the last add | Medium. Touches DMA queue semaphore counts (S[8..23] have per-DMA wait values) | Count the elapsed time from the last DVE compute end to the last DMA completion. If the DMA cannot be sooner, drop it |
| 4 | Prologue: first DVE compute at 14.86 us | Pool `DMA_DIRECT2D` (x/dt loads) issue at 11.1 us; DVE slots 4-5 wait S[9]>=16 and S[8]>=16 | Trigger the first x/dt loads before Pool's other prologue work | Profile: runtime preamble ends about 10.4 us. The first DMAs complete about 14.8 us, so DMA latency is about 3.6 us | At most about 1 us. Floor is about 13.7 us | Low. Wait values are unchanged | Compare the S[8]/S[9] update times in `SemaphoreUpdate.parquet` with 11.1 + 3.4 us. If already at the floor, dead |

**Hypotheses I tried and rejected from the evidence on hand:**
- A single-instruction gap census of DVE shows about 16 apparent 1.05-1.44 us "gaps", but they are profiler overlap artifacts. The preceding SCAN is still running when the next one starts. Do not chase them.
- Streaming the y-sum through PE (identity matmuls into PSUM) would remove all 204 adds. But fp32 matmul runs about 4x slower than the 585 ns bf16 K=1 matmuls, giving about 2.3 us x 204, which is about 480 us of PE. PE would then be the bottleneck, and PSUM is fully used by the B/C bank rotation. Not worth a probe.

## Closed (with the number that closed it)

- **More DVE->Pool ADD moves** (the remaining 34 DVE y-adds). Sibling iter-7 s1 cost model: adding Pool work costs 0.68 us of wall per us (a probe with +283 us of Pool gave +192 us of wall). Moving the DVE adds saves 0.69 us each of DVE but adds 2.01 us each on Pool: about +780 ns net each. Pool is at 81% here. This was also the sibling's iter-3 finding in the other direction.
- **dt*x -> Pool:** sibling iter-7 s3a, bit-exact but +449 us.
- **Carry COPYs -> ACT:** sibling iter-7 s3b, numerically wrong (max_abs_err 43.7) because the rotating region is shared across engines. Only 28% of sites are placeable anyway (s3c).
- **Pool access-pattern reshaping:** 0 us (sibling iter-7 s1.4).
- **Merging or relocating the Pool DMA_DIRECT2Ds:** dead (sibling iter-7 s3e: per-engine queues, and a distinct wait value per DMA).
- **Inserting or removing no-op ES slots as such:** free (sibling iter-5 s2b: +4000 no-op ES gave 0). The gain from this iteration's edit is from the removed waits, not from fewer slots.
- **Moving a PSUM read earlier:** +526 ns per move (sibling iter-7 s0). Keep PSUM reads at their positions.
- **Offloading BMUL/CMUL to ACT:** ACT cannot do tensor x tensor (sibling iter-5 s2e).
- **Rebalancing y-adds at 692 ns on DVE vs 2010 ns on Pool by hand:** the 170-add split in HEAD is already the measured optimum. I saw no evidence that it can be improved.
- **MEMSET removal:** 24 MEMSETs of about 106 ns each is at most 2.5 us. The loop ones re-set the bf16 ones vector (6 times), the fp32 zero carry (6 times) and single-element zeros. Not worth the sync risk.

## Unknown (would change the ranking if measured)

1. **How many of the 424 remaining S[6] waits are data-independent?** Measure: the exact SBUF/PSUM range alias check over all five streams (sibling `an1.py` and `sim.py`). If more than 100 are removable, candidate 1 is worth a full iteration.
2. **Does this NEFF have the sibling's constants?** Measure: single-instruction durations (done: SCAN 1274, BMUL about 656, CMUL about 677, ADD 692 on DVE; ADD 2010 on Pool) match the sibling's within 2%. The 0.68 us per us Pool exchange rate and the 82 us offset are not re-measured. A probe that widens 170 Pool ADDs from 512 to 576 elements would settle it (the sibling's was +0.68).
3. **Free SBUF for candidate 2.** Measure: exact-reference free-region count. This kernel has fewer live tiles (6 groups, 2 chunks), so the answer could differ from the sibling's "none".
4. **Cause of the 4 weakened S[6] sites.** Ordinal 10 (prologue) was not hand-checked. The three checked sites show no overlap, but the in-flight window grew by 1.
5. **Whether DGE descriptors touch 0x15400** (the new zero buffer). I scanned instruction streams only.

**Bottom line.** The remaining headroom is small, about 9 us (under 2%) by the sibling's model. Candidate 1 is the best gain per risk because the falsifier is a quick timing-only probe. Candidate 2 is the only one that could clear 2% but needs an SBUF liveness answer first. If both die, this NEFF is near its editing floor (about 495-500 us).

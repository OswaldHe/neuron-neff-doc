# Review of iter-7 (`kernel.neff`, uncommitted on top of 5585a33)

Measured here on the device unless marked "notes". The agent's `notes/iter-7.md` exists and was read.

## 1. What changed, and does it match the notes?

`neff_diff` against HEAD (5585a33) shows 4 members changed. `def.json`, `SP0.json` and `PE0.bin` are untouched.

| member | change |
|---|---|
| `Activation0.bin` | 2564 -> 2532 slots. The 32 inserted ACT `ACTIVATE` casts (iter-6) are deleted, and the rest of the stream shifts. |
| `DVE0.bin` | 2860 -> 2892 slots. 32 `CAST` (struct 0x47, fp32 -> bf16) are inserted, with `SEM_INC_COMPLETE` on the fresh semaphore S[88]. |
| `Pool0.bin` | 32 slots re-fielded. The extended wait changes from `S[72+r] >= 16p` (the dead q2 cast DMA) to `S[6] >= v_g` (g2: `S[42] >= 32`). |
| `SP0.bin` | 60 slots re-fielded: 32 chunk-0 `W_t` triggers now wait on `S[88] >= g+1`, and 28 preceding `EVENT_SEMAPHORE`s carry a `S[5] >= n` WAR guard. |

This matches the three edits in the notes. I checked the Activation, DVE and SP changes against the diff.

`semaphores.py` on unpacked prev and cur copies:
- Semaphores that change: S[6] waiters 2224 -> 2193, S[88] 0 -> 32, and about 12 of the Pool-fed S[8..21] (`max_wait` 320 -> 304, fewer waiters).
- No "wait exceeds increments" line anywhere.
- No semaphore with increments and zero waiters, except the already-dead S[72..87] (q2 / `qSPSpillReload2`).
- S[88] is now a real producer/consumer pair, 32/32/32. In iter-6 it had 0 waiters.

## 2. Is the latency change real?

Interleaved A/B on the same device session, 5 captures each, medians in us:

| NEFF | run 1 | run 2 |
|---|---|---|
| HEAD (A) | 1170.9 | 1171.4 |
| iter-7 (B) | 1169.9 | 1166.7 |

- That is -1 and -4.7 us, about -2.9 us on average, against a within-run sample range of ±4 us.
- The B and A sample ranges overlap (B 1163.0-1170.6, A 1163.8-1174.1).
- The agent's own medians are 1165.9 against 1170.5, and `profile_latest.json` says 1166.3 (spread 0.72%).
- The direction is probably real but it is at the noise floor: roughly 0.25%.
- The accumulated gain over the last four iterations is also tiny: iter-4 1176.9, iter-5 1172.4, iter-6 1170.3, iter-7 about 1166.
- Do not read any single step of iter-5 onward as a repeatable win.
- Total against baseline is 1351.4 -> about 1166, -13.7%. That comes almost entirely from iterations 1, 2 and 4.

## 3. Safety beyond the verifier

- **Error:** `output_0` is 1.097e-04 (10% of the limit) and `output_1` is 9.081e-04 (83%). Both are bit-identical to the baseline, so the edit did not move error. The 83% on `output_1` is inherent to the baseline, not iter-7.
- **Loosened sync, edit 1:**
  - The Pool wait now equals the dead DMA's own trigger condition, which is satisfied no later than the old one.
  - There is no new deadlock edge.
  - Nothing reads `W_all` (the notes show a device probe that never fires the q2 transfers and still passes).
- **Loosened sync, edit 2:** the DVE casts have no wait and rely on in-order program position after the `Wg_grp` producer. That is sound for DVE-produced data. It also fixes iter-6's g2 cast, which had no wait at all.
- **Tightened sync, edit 3:** the chunk-0 `W_t` triggers now wait on the cast's own completion, plus an explicit PE-ordinal WAR guard. This closes the iter-6 review's hole.
- **Remaining timing-only edges:**
  - Chunk-1 `W_t` triggers (source partition 112) keep their original ring-throttle waits. Their RAW against the cast rests on timing (the notes measure a minimum cast-end to DMA-start margin of 1.9 us).
  - Four groups (g0-g3) have no WAR guard. The notes call them first use of their address, so none is needed.
  - Neither would be caught by a fixed-input verifier if the race were tiny. 3 of 3 bit-identical runs is decent evidence.
- **`W_t` reuse:**
  - `W_t` has only 9 distinct addresses for 64 uses, so the WAR window is structurally tight.
  - The `S[5]` ordinals form a clean arithmetic progression (step 62), which is a good self-check.
  - Treat any further change to these triggers as high-risk.

Verdict: accept. It is safe, slightly positive, and strictly better synchronised than HEAD.

## 4. Where the time is

- Per-engine busy from `profile_latest.json`: PE 340 us (29%), ACT 521 (45%), DVE 549 (47%), Pool 234 (20%), DMA 330 (28%), SP 53 (5%).
- The wall is 1166 us.
- In my 100 us time bins, ACT, DVE and PE are uniformly loaded (ACT ~50 us busy per bin, DVE ~50) from start to finish. This is a steady-state pipeline of 32 groups at about 36 us each, not one hot phase.
- The notes (iter-7 section 2) show wall moves about 0.34 times added engine busy and 0 times removed engine busy.
- The notes also show 185 us of whole-machine idle spread over 1442 gaps, the largest being 3.8 us.
- So the remaining time is dependency latency spread across many small hops, not a single lever.

## 5. Ranked candidates for iter-8

Ranked by expected gain per unit of risk and effort. Honest ceiling for the whole list: roughly 5-10 us, which is the noise floor. Stop after these.

| # | Candidate | Where | What changes | Mechanism / evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Critical-path decode first (measurement, not an edit)** | `Instruction.parquet` + `SemaphoreUpdate.parquet` | For each instruction with a wait, find the update that satisfied it. Walk back from the last instruction over 32 groups and sum latency by hop type (DMA trigger, ring throttle, PE ordinal, DVE -> ACT). | Establishes which hop type dominates. Notes show busy-removal gives 0 wall and latency removal gives wall. | Unknown by design; it bounds all later picks. | None (read-only). | If the walk shows a flat mix with no hop above about 2% of the wall, stop and declare the NEFF at its floor. |
| 2 | **Hoist chunk-0 `W_t` trigger past the blocking trigger, with a full-engine WAR guard** | `SP0.bin`, 32 pairs of (guard `EVENT_SEMAPHORE`, trigger) | Move each pair above the `PSEUDO_DMATRIGGER` waiting on `S[6] >= v_g+3`. Build the guard from every engine's `src` extent overlapping the transpose's write region. | The notes (3.4) measure SP blocked ~975 ns on that trigger. Naive hoist failed with NaN in `output_0` (3.5). | ~1.1 us x 32 = 35 us of DMA advance, about 4-5 us of wall at 13% conversion. | High: a partial guard passes some inputs and corrupts others. `output_0` NaN would catch the gross case only. | One group: hoist only g5 with the full guard and run `verify --runs 3`. A NaN or a diff kills it in under 5 minutes. |
| 3 | **Decode `MATMUL :447` gaps** | `PE0.bin`, the pcs after the 28 largest gaps | Identify what the `LDWEIGHTS` / `EVENT_SEMAPHORE` before it waits on. Likely an over-conservative wait value or ordinal. | Notes 3: 27.7 us of whole-machine idle over 28 gaps, never decoded. | Under 28 us, realistically 2-5 us. | Medium. PE ordinal waits are exact once divided by 16 (notes 3.2). Reducing a wait that something relies on hangs or races. | Check whether the waits there are already exact. If the binding wait is the real DVE/ACT producer, the gap is true dependency and the candidate dies. |
| 4 | **Neutralise the 32 q2 triggers** | `SP0.bin`, the 32 `qSPSpillReload2` `PSEUDO_DMATRIGGER` | Re-encode each as a neutral `EVENT_SEMAPHORE` (wait NONE, update NONE). | Notes 4: device-verified, DMA busy 326 -> 311 us, SP 53 -> 50, wall equal. | About 0 us wall. | Low. | Already done by the previous agent (wall 1168.0 / 1166.5). Take it only as hygiene, with candidate 2 or 3 in the same edit. |
| 5 | **Fold the cast into the producer** | `DVE0.bin`: each group's last band `COPY` (pre-cast anchor) | Make the final `Wg_grp` write also produce a bf16 copy at the scratch address, deleting the separate cast. | Saves one instruction per group. | Under 14 us of DVE busy, which the notes show converts to ~0 wall. | Medium: the fp32 `Wg_grp` still has consumers, so a dtype change on the same write is not enough. | Needs a second output on one instruction, which the ISA struct does not allow. Likely dead on arrival; check `num_elem` / dtype fields before trying. |

## 6. Closed (do not redo)

| Lever | Closed by |
|---|---|
| Compute rebalancing between ACT / DVE / Pool | Notes iter-7 section 2: 32 dummy ACT casts gave +6.6 us wall; moving them ACT -> DVE gave 0; repositioning inside ACT gave 0. |
| Pool head-of-line retarget as a lever | Removing the retargeted wait entirely measures 1166.4 (notes 1, last para). |
| Fusing gather -> scatter DMA pairs on `qSyncSpillReload0` | Gather end to scatter trigger is 32-326 us (median 104); the real producer is compute (notes 7). |
| Shaving `LDWEIGHTS :597` wait values | Exact once divided by 16 (notes 3.2). |
| Splitting or merging DMA transfers for latency | Fetch is 713 ns flat in descriptor count and size (notes iter-6 section 4). |
| Deleting the dead q2 transfers for wall | Wall equal at 1168.0 / 1166.5 (notes 4). |
| Whole-schedule reorder / software pipelining | Hangs because it must widen wait prefixes (iter-6 section 2). |
| Chunk-1 `W_t` trigger retarget | PE consumers wait only 170-260 ns, no prize (notes 6). |

## 7. Unknown (would change the ranking)

| Unknown | Measurement that settles it |
|---|---|
| Which hop type dominates the dependence chain | Candidate 1, the semaphore critical-path walk. |
| Whether `MATMUL :447` gaps are real dependency or over-sync | Decode the waiters of the 28 largest gaps in `PE0.bin` and check the releasing instruction's end time. |
| Whether the SP head-of-line wait on `S[6] >= v_g+3` can be avoided cheaply | Count how many of those four-slots-ahead triggers share a ring with their own transpose; if none, a pure reorder of the trigger only (no guard ES) may suffice. |
| True noise floor of this machine | Re-profile the identical NEFF 4 times; the A medians above span 1170.9 and 1171.4, but the agent saw 1167-1176 across sessions. |

If candidate 1 shows no single hop above about 2% of wall, the kernel is at its NEFF-level floor and the loop should stop.

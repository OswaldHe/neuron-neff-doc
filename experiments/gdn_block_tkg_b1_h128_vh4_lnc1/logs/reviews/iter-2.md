# Review of iter-2 (kernel.neff, gdn_block_tkg_b1_h128_vh4)

Method: `neff_diff.py` against `baseline/kernel.neff` and against `git HEAD:kernel.neff` (the 95.3 us commit).
The iteration's own change is the HEAD-vs-working-tree diff. I also ran `semaphores.py` on all three unpacked
copies, `verify.py --runs 10`, and `profile.py --repeats 1 --parquet prof` (per-instruction table).
`notes/` contains only the README stub, so there is nothing to compare the diff against. The only written claim
is the HEAD commit message, which covers the previous iteration (the six SBUF->DRAM->SBUF reroutes).

## 1. What changed (HEAD -> working tree)

Three members changed: `Activation0.bin` (15 slots), `Pool0.bin` (19), `SP0.bin` (14). `SP0.json` is unchanged
against HEAD. Against baseline it also differs, because of the previous iteration.

| Engine | Slots | Change |
|---|---|---|
| Activation | 35, 39, 40, 41, 52, 55, 58, 59 | `PSEUDO_DMA_DIRECT2D` (SBUF->SBUF partition-aligned copy, 128x2 B) became `CAST`. The wait is kept on sem 4 (Activation's own counter). The `CAST` adds `events_semaphore_value` to the old DMA semaphore (72..79) with `SEM_ADD_IMM`. |
| Activation | 4, 6, 8, 10, 12, 15, 17, 67 | Waits re-pointed to a single semaphore: `wait_idx` 12-17 -> 13, `value` 32 -> 16. Slot 67 changed 61 -> 37 and 48 -> 16. |
| Pool | 6 | The 128-element DMA became a 2-D DMA `[128,8]`, `src_step [16,-2048]`, `dst_step [262144,768]`. It replaces 8 narrow DMAs. |
| Pool | 7, 21-26, 34, 36, 38, 40, 43, 51, 53, 55 (15 slots) | `PSEUDO_DMA_DIRECT2D` became `EVENT_SEMAPHORE` that adds 16 to the old DMA semaphore. Slots 7 and 21-26 have no wait. Slots 34-55 keep a wait, re-pointed to sems 56-61, 66, 71. |
| Pool | 44, 46, 48 | Waits changed from sems 13/14/15 (value 32) to sem 13 (value 16). |
| SP | 13, 15, 24, 26, 44, 46, 52, 55, 62, 64, 66, 68, 75 | DMA became `EVENT_SEMAPHORE`: waits re-pointed to sems 72-79 (the Activation `CAST` results), and the DMA's semaphore is bumped by 16. |
| SP | 50 | The `PSEUDO_DMATRIGGER` wait on sem 32 was removed (`WAIT_FOR_SEM_GE_IMM` -> `NONE`). |

In total, 36 queued DMAs were removed. Eight were replaced by real Activation copies and one by a merged strided
DMA. The other 27 were turned into bare semaphore bumps, so the data movement they did is no longer in the
program. Two things to know:

- I could not prove from the diff that the data those 27 moved is still delivered. The outputs are
  bit-identical across 10 runs and equal the baseline's to 4 digits, so it was evidently dead or merged.
  Nothing documents which.
- The edit matches the HEAD commit's "reroute to cheaper path" direction. No commit message or note describes
  this iteration, so "consistent with the notes" cannot be assessed. Write the notes.

## 2. Is the latency change real?

Yes.

- `profile_latest.json`: median 83.176 us, samples 82.73-83.27, spread 0.65%.
- Baseline: median 218.17 us, spread 1.30%.
- My independent single-repeat profile gave 83.53 us.
- HEAD was 95.3 us per its commit message. The 12 us step is about 18x the spread.
- Total speedup is 2.62x.

The attribution of the 12 us among the three parts is unmeasured: the Activation `CAST`s, the Pool merge and
the SP/Pool DMA deletions. Nobody profiled them separately.

## 3. Safety beyond the verifier

- **Error:** unchanged. The `max_abs_err` values are the baseline's: 3.0e-3 (83% of the 3.6e-3 limit), 1.9e-3
  and 6.0e-4. Output 0 was already at 83% before any edit, so there is no new margin problem, and little margin
  to spend. Ten runs were bit-identical.
- **`semaphores.py`:** no wait exceeds its supply. Sems 15, 57-61 and 72-79 have supply above the max wait,
  which is benign for `>=` waits.
  - **Soft spot 1 (Pool 7 and 21-26):** these bump sems 12-17 with no wait while the merged DMA at Pool 6
    increments only sem 13, once. The consumers of the other 7 DMAs' data are released by an instruction that
    does not wait for the merged DMA's completion. It holds only because slot 6 is issued earlier in the same
    queue (about 7 us before slot 21) and the DMA takes about 2 us. That is a timing assumption, not
    synchronisation. It can pass once and fail under a delayed or contended queue, and a fixed-input verifier
    cannot see it.
  - **Soft spot 2 (SP 50):** a wait on sem 32 was removed (its only waiter). Sem 32 now has 16 increments and
    0 waiters.
  - **Soft spot 3 (the `CAST` slots):** the value field is both wait threshold and add amount. Slot 35 now
    waits sem 4 >= 18 (was 13), and slots 39, 40 and 41 got equally larger thresholds. Larger means later, so
    it is safe, but it costs time. Sems 72-79 receive 16-24 where consumers wait for 16, which is fine with a
    single add.
- **Cheap hardening:** give the bumps at Pool 7 and 21-26 a `WAIT_FOR_SEM_GE_IMM` on sem 13 >= 16 (the merged
  DMA). The expected cost is about 0, because the DMA is long done by then. This makes the edit sound rather
  than lucky.

## 4. Where the remaining time is (from `prof/Instruction.parquet`, exec 83.5 us, body about 12.4 -> 97 us)

| Window | What happens | Binding resource |
|---|---|---|
| 13-31 us | Pool issues 25 `DMA_DIRECT2D` serially at about 0.66 us each, 100% busy, `evt_wait` about 22 ns. Sync and Activation are about idle. | Pool DGE issue |
| 30-40 us | Activation: table loads 0-4 (1.28 us each) and activates. | Activation |
| 33-64 us | About 25 Sync and 13 Pool narrow DMAs, with long idle gaps on every engine (Vector idle 38.7-51.7, PE 32-51 and 53-71). The Q2 queue carries 249 x 16 packets of about 14 ns each, only about 3.5 us of active time per queue over 30 us. | DMA latency/dependency chain, not bandwidth |
| 70-91 us | Activation 80-100% busy. 8 of the kernel's 13 `ACT_TABLE_LOAD`s (1.283 us each, 10.3 us) are here, alternating table 2 and table 4 with one or two instructions between them. | Activation |
| 91-97 us | Output DMAs (about 4-5 us) and semaphore drain. | fixed latency |

Busy time: Activation 51 us, Vector 32 us, Pool 31 us, Sync 24 us, PE 25 us (of which 14 us is
`EVENT_SEMAPHORE` stall). `ACT_TABLE_LOAD` is 16.7 us in total.

## Ranked candidates for the next iteration

| # | Candidate | Engine / slots | Change | Mechanism and source | Gain and ceiling | Risk | Falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Cut Activation table-load thrash in the 70-91 us tail | Activation0.bin, `ACT_TABLE_LOAD` at compiler_pc 64, 75, 82, 85, 89, 96, 99 (and 51). Tables alternate 2, 4, 2, 4, 2, 4, 2, 4 | (a) Reorder independent Activation work across the 4 v-heads so same-table instructions are adjacent, then delete the redundant loads (turn them into `EVENT_SEMAPHORE` keeping counts). (b) First test whether the `UNKNOWN`-function activates between loads exist in both tables, and drop loads outright. | Playbook ch14.3 item 2 (reorder with waits re-pointed, "verified mechanics" r08, ch10.4 R4). ch14.3 item 3 if work moves engine. | Ceiling 8 loads x 1.283 = 10.3 us, all on the Activation critical path (Activation 100% busy 80-90 us). Realistic 5-8 us if per-head chains permit batching. Whether chains permit it is unknown. | A wrong table gives wrong activations, and the verifier catches that (output depends on it). A reorder race is invisible to it. | Decode the function ids of the `ACTIVATE`s around the loads with `isa.py` (2 min). If the instructions after a table-4 load need table 2 data-dependently on the previous one, batching is impossible. Otherwise NOP one pair (pc 82 and 85) and run `verify.py`: pass plus a drop of about 2.5 us, or the candidate dies. |
| 2 | Convert the remaining partition-aligned SBUF->SBUF DMAs to engine copies | SP0.bin slots 5, 6, 7, 8, 9, 12, 14, 16, 17, 22 (`src/dst step [262144,1] x [128,1]`, wait 0, SBUF to SBUF); likewise SP 53 and 56 if SBUF-only. Target engines: Pool and Vector (Pool idle 40-53 and 58-87, Vector idle 38-53). | Replace each DMA with a `COPY`/`CAST` on the engine idle at that time. Keep the wait. Make the copy's `SEM_ADD_IMM` equal to the old DMA semaphore (16) and re-point consumers. Delete the DMA (turn it into an `EVENT_SEMAPHORE`). | The same transformation this iteration used on 8 Activation DMAs (Activation 35-59 DMA -> `CAST`). A copy costs 0.18-0.3 us, versus 0.6 us issue plus about 2 us DMA latency. ch14.3 item 3, ch10 (the `CAST` value field doubles as wait threshold and add). | Ceiling unknown: the 33-64 us phase is 30 us with many chained hops. Each converted hop is worth about 2 us only if it is on the critical path. | Moderate, since semaphore re-pointing. A mis-pointed wait is caught by the verifier only if the data is stale in the fixed input. Run `verify --runs 10` plus `semaphores.py`. | Convert one hop on the chain with the largest `evt_wait`, run once. If the 33-64 window does not shrink by about 1.5 us, the hops are not latency-bound, and stop. |
| 3 | Merge the remaining same-shaped DRAM->SBUF gathers on Pool | Pool0.bin slots 1, 3, 8, 10, 11, 12, 13, 14 (src step 2048, src 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x0, 0x100; dst 0x37c00, 0x37b00, 0x37a00, 0x37900, 0x37800, 0x37700, 0x37e00, 0x37d00, i.e. dst stride -0x100) and slots 4, 5, 15, 16, 17, 18 (src 0xc00, 0x800, 0x1000, ..., stride 0x400, dst stride -0x300) | One `[128,8]` / `[128,6]` DMA, as the iteration did at Pool 6 (negative step already works). Re-point every consumer's wait to the merged semaphore at its full count. Do not use bare bumps without a wait. | Same pattern as Pool 6 this iteration (8 DMAs -> 1, measured as part of 95 -> 83 us). Doc ch9, ch14.3 item 4. | The Pool window is 17.4 us at 100% busy, issue-limited at 0.66 us per DMA. Removing 12 DMAs frees at most about 8 us of Pool issue time. Whether that gates ACT/PE start (first PE matmul at 16.9, Activation at 29.7) is unmeasured, so the gain could be 0. | Moderate. A narrower consumer set is easy to get wrong, but a stale-data error would show up in the verifier. | Check whether the first consumers of the Pool DMAs wait on the last one (look at the `wait_idx` of the first Activation/PE users). If consumers wait only on early DMAs, issue time is off the critical path and the gain is 0. |
| 4 | Distribute DMA issue across idle queues | Move about half of the Pool 13-31 us DMAs onto SP or Activation (both idle then); SP0.bin and Activation0.bin need inserted slots | `PSEUDO_DMA_DIRECT2D` can be issued from any DGE engine (the Activation stream held DMAs before this iteration). Move non-waiting loads, rewriting `semaphore`/`wait_idx`. | ch13 insert/reorder and ch9 DGE. | Ceiling about 8 us of queue time, the same bound as candidate 3, so the two overlap. Do candidate 3 first, then see if Pool is still binding. | Insertion shifts slot indices (branch offsets, `compiler_pc`), higher edit risk than in-place. | If candidate 3 already makes Pool less than 80% busy in 13-31 us, drop this. |
| 5 | Move `CAST`/`COPY` off Activation in the tail | Activation `CAST` (pc 63, 65, 66, 68, 84, 87, 88, 94, 97) and `COPY` (pc 71, 77, 90, 101) in 70-91 us; Pool is 0% busy there and Vector 17-36% | Replace each with the same op on Pool/Vector, wait on the producer's semaphore, bump a fresh semaphore, and re-point consumers (ch14.3 item 3). | ch14.3 item 3. | Ceiling: 9 x 0.29 + 4 x 0.26 = about 3.7 us of Activation time, and only the part on the critical chain counts. | Moderate. | Dependent on candidate 1. If the table loads are removed, Activation is no longer the critical engine in 70-91 us. Re-profile after 1. |
| 6 | Replace the SBUF column->row transposes with a PE or DVE transpose | The 13 DMAs with `dst_step [2,1]` / `[4,1]` (SP 18-21, 27, 30, 32, 34, 36, 37, 38, 58, 60) | Insert a matmul against an identity (PE is only 9% busy) plus a PSUM->SBUF copy, or a `STREAM_SHUFFLE` transpose on DVE. | Not established on this device. The 28 existing matmuls show the PE path works for this kernel, but the identity tile and PSUM space would need to be created. | Ceiling unknown, bounded by the 30 us 33-64 phase. | High: needs new SBUF/PSUM allocation. An allocation or race error can pass fixed inputs. | Do not start without a numeric spike: one transposed column replaced in isolation, `verify --runs 10`. Low priority. |

Ranking logic: 1 has the largest measured bound (10.3 us on a binding engine) with a two-minute decode test. 2
is the proven transformation, with cheap per-hop gains. 3 and 4 depend on a gating question that has not been
measured. 5 and 6 depend on the earlier ones or need new structure.

## Closed (do not redo)

- **Six DRAM scratch round-trips** (tmp-bufs 6_1, 52_47, 7_2, 111_106, 110_105, 109_104): rerouted to SBUF->SBUF
  by the previous iteration. 215.2 -> 95.3 us (commit 480fb81).
- **The eight Activation SBUF->SBUF DMAs** (pc 35, 39, 40, 41, 52, 55, 58, 59) are now `CAST`s. 8 -> 1 merge at
  Pool 6 is done.
- **Removing waits on an engine's own completion semaphore** (doc ch14.2, ch10.5): refuted by o01. DVE gave wrong
  output, Pool a runtime out-of-bounds error, and Activation no gain. Do not strip own-semaphore waits.
- **PE `perf_opt`:** not a lever (doc ch14.3).
- **PE and DVE are not bottlenecks:** PE is 9% and DVE 22% busy. Do not chase matmul or DVE arithmetic cost.

## Unknown (would change the ranking)

1. **Gain attribution of this iteration's three parts** (Activation `CAST`s, Pool merge, the 27 DMA deletions).
   Measure by reverting each part separately on a copy, 3 profiles each (about 10 min). It tells whether
   candidates 2 and 3 are worth the effort.
2. **What data the 27 deleted DMAs moved**, and whether it is still delivered by something else. Decode src/dst
   of the HEAD versions (`/tmp/r/head`) against the variable table. This also settles whether soft spot 1 is a
   latent bug. Do this before the next edit builds on it.
3. **Which function ids occupy tables 2 and 4**, and whether the `UNKNOWN`-function activates exist in both.
   Decode with `isa.py`. This sets the real ceiling of candidate 1.
4. **Whether Pool's 13-31 us issue window is on the critical path.** The first Activation/PE consumers' wait
   indices will say. It sets the ceiling of candidates 3 and 4.
5. **The floor.** The fixed epilogue (97 -> 98.6 us) plus 4-5 us of output DMA latency is not editable, so the
   floor is at least about 5 us above the last compute. Compute plus table loads in 70-91 us is the last
   adjustable stretch.

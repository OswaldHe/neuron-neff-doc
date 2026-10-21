# Review of iter-3 (working-tree `kernel.neff` vs HEAD `ac545ca`, and vs baseline)

`notes/` is empty (only the README). The agent left no record of what it tried, so everything below is from the binary and from my own profile.

## 1. What changed

Diff against HEAD (iter-2, 41.056 us). 4 members changed: `Activation0`, `DVE0`, `Pool0`, `SP0`.

* **Two DVE CASTs moved to Activation.**
  * DVE slots 49 and 60 (fp32→bf16 `0x4500→0x4b08`, 128 elem, and `0xe28001→0x4800`) became `EVENT_SEMAPHORE` stand-ins. They keep the S[6] increment, so S[6] stays 66/66. They also wait on the new semaphores S[50] and S[51].
  * Activation gained three slots (77→80). Two are the CASTs, with `SEM_INC_COMPLETE` on S[4] so S[4] stays 41/41. The third is an `EVENT_SEMAPHORE` that takes the S[4] count in the slot the CAST vacated. ACT slots 36–42 and 53–55 were rotated or shifted.
  * S[50] and S[51] are new, with 1 inc, 1 wait and value 1 each. This is the same "numbering-neutral stand-in" trick as iter-2/iter-7 of the s256 run, but in the opposite direction, ACT→DVE becoming DVE→ACT. It rebalances the two engines: busy time is now ACT 20.1 us and DVE 20.05 us.
* **DGE hosts rotated.**
  * Semaphore 43's DGE row moved from Activation to Pool (a new `DMA_DIRECT2D` slot, Pool 23→24).
  * Semaphore 45's DGE row moved from SP to Activation.
  * SP lost a slot (42→41), with 16 SP slots rewritten (trigger and fence reshuffle).
* The `semaphores.py` diff against HEAD shows exactly this: S[6] waiters 74→77, S[43] and S[45] hosts changed, and S[50] and S[51] added. **No wait threshold exceeds its supply**, and no wait was removed or lowered. The tool lists S[16] with 0 waiters. That is its blind spot for the extended wait on Pool `EVENT_SEMAPHORE`, and it was already there in HEAD.
* Is this consistent with the notes? There are no notes, so there is nothing to compare against. The commit message of HEAD describes iter-2 only.

## 2. Is the latency change real?

* `profile_latest.json`: median 40.094 us, min 39.98, max 40.627, spread 1.6%, 7 samples.
* My own re-run: **40.045 us**, which reproduces.
* HEAD measured 41.056 us in the iter-2 session. The delta is about −1.0 us (−2.4%), which is bigger than the 0.65 us spread.
* I did not run an interleaved A/B of HEAD against the current NEFF. Session drift on this device has been about 0.5 us in the s256 run's notes. So treat the win as **probably real, 0.5–1.0 us**. Run one interleaved A/B (7 captures each, alternating) before quoting 1.0.
* The loop is not chasing noise at this step, but it is close to the noise floor.

## 3. Safety

* The sync edits are additive. Each moved op gets a fence on a new semaphore, and every wait stays at or below its supply.
* **Red flag: the output is not bit-identical to the baseline.**

| | baseline | latest |
|---|---|---|
| `output_0` `mean_abs_err` | 2.05150e-06 | 2.05116e-06 |
| `output_0` `max_abs_err` | 6.119e-05 (17% of limit) | 6.119e-05 |
| `output_1` `max_abs_err` | 2.864e-04 (82% of limit) | 2.864e-04 |

  * Earlier iterations claimed "bit-identical". A different `output_0` mean means at least some elements changed.
  * Either the baseline is itself non-deterministic (a race already in the compiled code that is exposed by timing), or iter-3 introduced a hazard on `output_0`. Moving a CAST between engines changes only timing, not arithmetic, so a changed value points at a read-before-write race.
  * `verify.py` ran only `runs: 1`.
  * Before accepting, run `verify.py --runs 5` on both HEAD and the current NEFF and compare the raw output bytes. The verifier would not catch a rare race.
* `output_1` sits at **82% of the error limit** and is unchanged to every printed digit. That is the compiled kernel's own error, not new.
* The two moved CASTs are the prime suspects if there is a hazard. The destinations are `0x4b08` and `0x4800`, and both are reused as buffers by later ops. Check the WAR (write-after-read) fences on them.

## 4. Candidates for iter-4 (ranked by gain per risk and effort)

Method note: I did not build the backward critical-path walker (the s256 notes iter-5 §1 describe it). Rebuilding it takes about 90 lines and should be the first 30 minutes of the next iteration.
Facts below come from `Instruction.parquet`, with the run starting at 12.5 us (first compute) and ending at 53.1 us.

| # | Candidate | Where / what changes | Mechanism | Gain and ceiling | Risk | Cheapest falsification |
|---|---|---|---|---|---|---|
| 0 | **Determinism check** (not an edit) | `verify.py --runs 5` on HEAD vs current NEFF, byte-compare outputs | n/a | Decides whether iter-3 is accepted | n/a | One command |
| 1 | **Skip the final identity op.** The last store waits on a ×1.0 copy. | Scalar compiler_pc 79 is `TENSOR_SCALAR ×imm=1`, fp32 `0x4710→0x4c00`, 128 ch, 298 ns. Sync slot 40 (`DMA_DIRECT2D` sem 17) reads `src_addr_imm=0x…4c00` and waits `S[4]>=41`. Change the DGE `src_addr` to `0x4710` and its wait to `S[6]>=66`, the Vector TT 99 that writes `0x4710`. Keep Scalar 79 in place so S[4] stays 41/41. | Same bytes (×1.0 on fp32 is exact). Removes a Vector→Scalar hop. Device-established idiom: re-pointing a DGE guard to the true producer (s256 iter-4/5). | Ceiling ≈ one op plus one semaphore hop: Vector 99 ends 50.64, Scalar 79 runs 50.67–50.97, Sync row starts 50.995. Realistic **0.25–0.35 us**. | Low. Need `0x4710` not overwritten before the DGE reads it; it is the last op. The verifier would catch a wrong address (output_1 would be garbage). | Edit one DGE and one wait field, run verify and one profile. Dies in about 5 minutes. |
| 2 | **Split the tail output-store row.** | Sync slot 40 is 667 ns (128 partitions × 512 B), followed by about 2 us DMA flight to 53.03. Replace it with two 64-partition rows hosted on Sync and Activation, or Pool for the sem 16 store (Pool row 20 is 672 ns). Each row increments the semaphore by 8 and the consumer wait stays 16. | The s256 iter-5 §7 split of the last store across two engines; the semaphore still gets 16. | Ceiling ≈ half the row issue time, **0.2–0.35 us**. s256 got −0.21. DMA flight is irreducible. | Medium: the semaphore increment must total 16, or the Pool exit fence hangs. `semaphores.py` checks supply. | Inspect `sem_increment` in the profile and check S[17]/S[16] reach 16. |
| 3 | **Split the first-input DGE (sem 8).** | SP slot 1 (`DMA_DIRECT2D` sem 8): src_elem_size=2, dst fp32, pattern `[4,262144][2,64]` (tiny elements, bf16→fp32 cast in the DMA). It issues at 12.61 us and lands at 15.7 us, a 3.1 us flight (other loads, with 256 B elements, take about 2 us). Vector CAST 8 is gated on it (2106 ns wait), and the whole gate prelude (12.6→20.1 us, tiny dependent 64-partition ops) hangs off it. Split it into two rows of 64 partitions each on SP and Pool, increment 8 each. | Fewer descriptors per queue should cut flight. **This scaling is unproven.** | Ceiling unknown. At most about 1 us (the part of the 3.1 us above the 2 us of a normal load). The prelude is serial after it. | Medium: two rows and the semaphore total must stay 16. | Compare flight time of a split vs unsplit on the profile. If flight does not drop, kill it. |
| 4 | **Rebalance Scalar vs Vector further, or hunt dead ops.** | ACT 20.1 us and DVE 20.05 us, each about 50% busy, but whole-kernel stretches in 24–38 us are bound by ACT/DVE alternation (Scalar CAST 26–36 and Vector COPY/CAST at 28–36). Move or remove ops that sit on the walker's path. Scan the table: Vector COPY fp32 PSUM→SBUF followed by a CAST of the same SBUF (e.g. Vector 38→40/43, 39→41/46, 54→56/59, 68→70, 69→72). | Dead-op removal: s256 iter-6 removed two provably dead CASTs (−0.43 us). Engine moves: iter-6/7. | **Unknown** until the walker runs. The only measured bound is the 0.43 us per dead CAST in s256, plus 0.2 us per moved op. | High: the pointer-operand trap (s256 iter-5 §2 broke output_1 to 29551% of the limit by moving a fence past a ptr operand). Only the verifier plus two or more runs would find it. | Check each candidate against all `dst=` operands in `Instruction.parquet` and DGE `to_off` spans (s256 iter-7 `dead2.py` method). |
| 5 | **Trigger-train serialisation on Sync at 24.0–24.6 us.** | Nine `WRITE` queue doorbells at 43–72 ns each (SP compiler_pc 11–19) sit behind `S[27]>=16`. They feed queued blocks 4–12 (2–4 KB each, landing 24.9–25.5), after which Tensor resumes at 25.28. Move a few triggers to Pool (idle after 21.4). | A trigger is an ordinary instruction on any engine. s256 iter-5/6 rehosted DGEs, not WRITEs. Pool answers semaphores about 0.5 us slower than Sync. | At most about 0.3 us (9 × ~60 ns). Pool's slower wake probably eats it. | Low–medium. | Move one trigger and watch whether block start times move. |

Ranking rationale: #1 is mechanically certain and small. #2 and #3 are small and unproven. #4 may be larger but needs the walker first, and carries the most risk. #5 is probably noise-limited.

**Closed (from this run's profile or earlier reviews):**
* Parallelising the Pool input DGE train (sems 9–13): the rows issue at 13.8, 14.6, 15.3, 15.9 and 16.7 us. Their consumers run at 17.6–19.8 us, which is later than the landings, and Pool is idle after 17.4 us. Ceiling ≈ 0.
* The ending: Pool `EVENT_SEMAPHORE`/`NOP` at 53.03 us is the exit fence waiting on the last stores. It is not a lever.
* `iter-2 → iter-3` direction DVE→ACT has now been done; engine balance is 20.1/20.05 us. There is no more slack to equalise.
* From s256 iter-7: moving `ACT[14]` to DVE-style hosts (verify failed with a race), and re-hosting a staging DGE on Activation (+0.68 us, reverted). These are different NEFFs, so re-check before relying on them.

**Unknown:**
* Whether the `output_0` mean change is run-to-run noise in the baseline or an iter-3 hazard. Settle it with `verify.py --runs 5` on both NEFFs.
* The true critical path. Settle it by rebuilding `cpath.py` (backward walker, `SemaphoreUpdate.parquet`, queued-transfer resolution).
* Whether DMA flight (about 2–3 us per DGE) scales with descriptor count or with bytes (for #2 and #3). Settle it by comparing flight time of existing rows of different size: sem 56–59 (32 KB each) against sem 8 (tiny, 256 descriptors).
* HEAD vs current in an interleaved A/B, to firm up the real size of the iter-3 win.

# Review of iter-6 (`kernel.neff`, uncommitted on top of a04258d)

All numbers below were measured in this review, on the device, unless marked otherwise.
`notes/` holds only the template README: **this iteration left no notes**, so the "claim" being checked is
only the (absent) commit message; what the edit does is reconstructed from the diff.

## 1. What changed

Diffed against the committed HEAD NEFF (`git show HEAD:kernel.neff`); only two members differ.

* `sg00/Activation0.bin`: 2532 -> 2564 slots (+32). Every slot index after ~21 shifts, so `neff_diff` reports
  ~2500 "different" slots; this is a pure insertion. Opcode histogram: `CAST` 896 -> 928, everything else unchanged.
  The 32 new instructions are `CAST` fp32->bf16 (in_dtype 10, out_dtype 6) that each `SEM_INC` **semaphore 88**
  (`events_update_idx 88`, mode 19), and each waits on DVE semaphore 6 (`events_wait_idx 6`, value = the next
  ACTIVATE's wait value minus 1) or has no wait at all (slot 217: wait mode 0). `semaphores.py`: S6 waiters 2193 -> 2224
  (+31), S88 `max_wait 0, total_inc 32, waiters 0`.
* `sg00/SP0.json`: 64 of the 861 `dma` entries changed (the `SP0.bin` trigger stream is byte-identical). 62 are
  bf16 `dma_transpose` descriptors in `qSPSpillReload0` that were one `[1,1,64,128]` transpose reading partitions
  64.. (`from_off ~16.89M`) and are now **four `[1,1,16,128]` transposes** reading a new staging area at
  `from_off 25235456...` (partition 96 and up; `to_off` split in 32-byte pieces); 2 more are the same inside a mixed entry.
  Queue, id and completion semaphore (24..39) are unchanged, and so are the 32 trigger slots' waits (sem 21 >= 32, from Pool DGE).
* `def.json`, all other streams: identical to HEAD.

So the idea is: let ACT produce the bf16 transpose source in a layout that lets four 16-row transposes replace one
64-row transpose. Nothing in the NEFF says why that should be faster, and the DMA profile says it is not (see 2).

## 2. Is the latency change real? No: it is inside the noise.

| NEFF | medians of 5-run profiles (us) | all samples |
|---|---|---|
| HEAD a04258d (re-measured, 2 x 5, interleaved) | 1172.6, 1173.2 | 1170.3 .. 1176.4 |
| iter-6 (agent's `profile_latest.json`) | 1170.3 (spread 0.85%) | 1166.3 .. 1176.2 |
| iter-6 (re-measured 2 x 5, +1 single in my parquet run) | 1169.2, 1172.1, (1166.4) | 1166.3 .. 1175.4 |

Mean difference ~2.5 us (0.2%); the within-run spread is 5-10 us (0.4-0.85%) and the two NEFFs' sample ranges
overlap almost completely. The commit log itself already records a +/-2-3 us band. The mechanism does not predict
a gain either: per-packet DMA durations in `DmaPacketAggregated` are 0.36 us (`[[128,64]]`, 118 transfers) vs 0.37 us
(`[[128,16]]x4`, 62 transfers), i.e. identical, and the trigger->DMA-start latency is a constant 0.72 us. One more ACT
CAST per piece (+32 x ~0.22 us on a ~46%-busy engine) is also not free. The loop is chasing noise here; do not
count this as an improvement over iter-5. Versus the unedited baseline (1351.4 us, spread 0.2%) the cumulative win
(~-13.4%) is real and is entirely from iters 1-5.

## 3. Safety

* **Synchronisation hole (real).** The 32 DMA transposes now read data written by an ACT `CAST` whose completion
  semaphore (S88) has **zero waiters**. The SP trigger still waits on S21 (the Pool DGE that produced the old
  source) and on the previous queue entry; nothing orders "ACT CAST done" before "DMA reads". Bit-exact output on
  every run only shows that ACT happens to win the race with ~1-2 us margin (ACT CAST waits on DVE S6; the trigger waits on Pool S21,
  an unrelated chain). It will pass the fixed-input verifier and can fail under any timing shift
  (other bench input, thermal, DVE slower). `semaphores.py` prints no warning for it because a never-waited
  increment is not flagged. Slot 217's CAST has no wait at all, so it can read its fp32 source before it is written.
  Fix if kept: make each trigger wait on S88 (+1 per piece, i.e. value k) or make the CAST wait on the producer's actual semaphore.
* The other two checks are fine: `semaphores.py` has no "wait exceeds increments" line for base/HEAD/iter-6; error is
  unchanged vs. baseline and HEAD: output_0 1.097e-4 (10% of the 1.1e-3 limit), output_1 **9.08e-4 (83% of the limit)**.
  output_1 is bit-identical to baseline, so the 83% is inherited from the original kernel's bf16 numerics, not from edits, but it means any
  edit that changes rounding of output_1 (e.g. moving a cast, changing dtype) has only ~17% headroom.
* Recommendation: **revert this edit** (keep HEAD): no measurable gain, new unsynchronised producer.

## 4. Where the time is (fresh per-instruction profile of iter-6, span 1182 us in the parquet capture)

* Busy by engine (profile.py, us): ACT 540, DVE 537, PE 340, Pool 233, DMA 332, SP 52. Nothing is above 46%.
  Per-100us windows are flat (ACT 55-63, DVE 46-54, PE 59-72 us of 100): a steady-state pipeline of ~32 repeating units, not a
  single hot spot. The time is dependency latency, not throughput.
* Cross-engine wait time (instruction `evt_wait_time_ns` > 300 ns, summed, engine waiting on which producer): PE on the
  SP-queued DMAs 488 us (183 waits, mean 2.7 us), ACT on PE 495 us (450), DVE on PE 480 us (732), ACT on DVE 313, PE on DVE 280, PE on ACT 154.
  These overlap, so they are not additive; they show the ring DMA-transpose -> PE -> ACT/DVE -> DMA chain.
* One SBUF->SBUF DMA hop costs: trigger -> DMA start 0.72 us (all 669 transfers, median = mean), transfer ~0.4 us (max 1.7), plus the
  documented 1.0-1.9 us before the semaphore becomes visible, ~2.2-3 us per hop. 637 of the 669 `qSyncSpillReload0` transfers are transposes.
* Pool: 647 instructions, only 31 us is compute (131 `AFFINE_SELECT`), 202 us is 288 `DMA_DIRECT2D` at 0.70 us each. Pool compute is ~free.
* ACT has 224 update-only `EVENT_SEMAPHORE` (40 ns each, stand-ins for deleted DGE saves), SP has 160 pure no-op `EVENT_SEMAPHORE`
  (no wait, no update).
* `qSyncSpillReload1` (160 cast/copy transfers) shows start_ts ~170 us before its trigger in `DmaPacketAggregated`: a
  profile artefact of pre-armed rings; do not read those as 170 us transfers.

## 5. Candidates for the next iteration (best first)

Rank is by expected gain per risk/effort. Honest summary: all remaining cheap NEFF-local levers look worth a few us
at most (the noise band is 2-5 us); only #1 and #2 have a ceiling large enough to matter, and both need a measurement first.

| # | Candidate | Engine / member / slots | Change | Mechanism and evidence | Expected gain / ceiling | Risk, would verifier catch it | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Software-pipeline the PE stream: hoist matmul groups that do not depend on the awaited DMA transpose ahead of those that do** | PE0.bin; the ~183 instructions (MATMUL/LDWEIGHTS with `events_wait_idx` in the `qSyncSpillReload0` set, sems 24-55) with a wait >300 ns, plus the matmuls that follow them in program order | Move independent matmul+LDWEIGHTS pairs (different batch/chunk, different PSUM bank) in front of the blocked one; re-point `events_wait_idx/value` and PSUM accumulate `acc_flags` ownership (R4, chapter 10.4); semaphore totals unchanged | "Reorder independent instructions on one engine" is Verified mechanics (playbook 14.2, r08); iter-3 (a8b683b) already got -2.6% by the same permutation on SP. Dependencies must come from BIR predecessors (chapter 11.2), not guessed. | Ceiling: PE waits on queued DMA = 488 us summed (183 waits); but ACT/DVE wait on PE 495/480 us, so the real gain is bounded by how much of that is on the critical path. Unknown; plausible 10-40 us, lower bound 0. | PSUM bank reuse (start/stop flags in `acc_flags`) and WAR hazards on PSUM/SBUF tiles; a wrong reorder passes on a fixed input only if the data race is benign, so also run 3 inputs. | Take the 10 longest PE waits on sems 24-55 (list from Instruction.parquet), hoist only the next independent matmul over each; if median does not move by >5 us over 3x5 runs, stop. |
| 2 | **Offload ACT/DVE elementwise to Pool** | ACT CAST (928 x 221 ns = 206 us), DVE COPY/CAST, ACT/DVE `TENSOR_SCALAR`; Pool has only 31 us of compute | Move a block of independent fp32->bf16 `CAST`/`COPY` from the ACT or DVE stream to Pool0.bin per playbook 14.3 #3: put an update-only `EVENT_SEMAPHORE` in the old slot (keeps its count), insert the op in Pool with waits for its inputs, make consumers wait on the Pool semaphore | Playbook 14.3 #3 (mechanics Verified, speed Unknown); Pool compute idle per the profile (31 us of 1170). The 0.7-1.0 us cross-engine semaphore hop is paid twice, so only worth it for ops off the critical chain. | Ceiling: ACT and DVE are 46% busy, so the ceiling is only what the moved ops contribute to the critical path. Unknown; worth at most a few us per moved block unless Pool is faster than ~1 element/cycle per lane. | Pool is slower on some dtypes; a rounding-mode change on cast would hit output_1 (83% of the limit). Bit-exact outputs vs HEAD on 3 inputs is the check; verifier catches rounding but not a timing race. | Move 32 independent CASTs from ACT to Pool; read their Pool `duration_ns` in the parquet. If mean > 0.5 us, dead. |
| 3 | **Fold ACT update-only EVENT_SEMAPHOREs into neighbouring instructions' `events_update_*`** | Activation0.bin: the 224 update-only `EVENT_SEMAPHORE` (events_update_mode set, wait mode 0) | When the previous ACT instruction has no completion update of its own, move the +16 increment into its `events_update_idx/mode`; delete the slot and shift/re-point the trigger positions | Delete-and-rebalance is Verified (r05, r06). Each stand-in costs 40 ns on ACT (profile min=median=40 ns). | Ceiling 224 x 40 ns = 9 us of ACT time; ACT is 46% busy so expect <= 2-4 us, i.e. inside noise. Not worth risking `sum>2us` gaps. | Moving an increment earlier makes the consumer see the "done" signal before the producing DMA's data exists; the increments at stake represent DMA completion, so only fold when the neighbour is the real producer. Not caught by the verifier. | Count how many of the 224 sit right after an ACT instruction with a free update field; if < 100, drop it. |
| 4 | **Delete the 160 no-op SP `EVENT_SEMAPHORE`** | SP0.bin (160 slots, no wait, no update) | Delete and shift SP slots; no semaphore count changes | Zero-effect stand-ins left by iter-4. SP is 4.5% busy and these are 40 ns each (6 us total, off-critical). | Ceiling ~0: SP is not on the critical path except through triggers, and the earlier trigger permutation already removed the blocking ones. | Very low: no semaphore effect. Verifier would catch a miscount. | Not worth running; list only so nobody re-discovers it. |

Not recommended: re-doing the 4x16 transpose split or the ACT staging cast of this iteration (see 2 and 3).

## 6. Closed (do not redo)

* **4x16-row DMA transposes with an ACT cast stage (this iteration):** DMA durations 0.37 vs 0.36 us, median latency 1170.3
  vs HEAD 1172.6/1173.2 (noise 5-10 us), plus an unsynchronised producer. Closed.
* **Relay EVENT_SEMAPHOREs on SP for retired DGE reloads (iter-2 to iter-5):** removed in a04258d; 1174.6 vs 1175.6 us, no
  change, because the ~1.0-1.9 us semaphore visibility after `end_ts` hides the relay. Closed.
* **Moving DRAM round-trips to SBUF->SBUF queued DMA:** done, -2.9% and -8.2% (d6d348b, f8380ab). Nothing left in `spill_reload_bytes` (now 0).
* **SP head-of-line trigger permutation:** done, -2.6% (a8b683b), pure permutation.
* **Removing waits on an engine's own completion semaphore:** negative in the doc set (wrong output on DVE, OOB on Pool, no gain on ACT).
* `MATMUL.perf_opt`: not a lever (bf16 already `NONE`).
* Consecutive `LDWEIGHTS` with identical operands: 0 of 1878 in the profile, nothing to remove.

## 7. Unknown (would change the ranking)

* **Which of the PE-waits-on-DMA chains are on the critical path.** Measure: build the per-unit dependency chain from
  `Flow.parquet` / semaphore ids and compute the longest path; or reorder one chain (candidate 1 falsifier) and see if the total moves.
* **Pool speed for `CAST`/`COPY`/`TENSOR_SCALAR` vs ACT/DVE on these shapes (bf16, 32-128 partitions).** Measure: candidate 2 falsifier, one block.
* **Whether the 32 repeating units are data-independent** (so cross-unit interleaving is legal). Measure: BIR predecessors (chapter 11.2) / DMA
  source-dest address overlap between consecutive units; if they chain through the state, #1 has no ceiling.
* **Run-to-run noise at the 5-run level.** Re-measured spread 0.4-0.85% (5-10 us) on identical NEFFs; any claimed win under ~6 us should be
  re-run at 3 x 5 interleaved against HEAD before being accepted.
* If the answer to the second bullet is "chained through the state", the remaining time is dominated by the recurrence's DMA hop latency
  (~2.2-3 us per hop) and is largely irreducible at the NEFF level without replacing DMA transposes by PE transposes (needs an identity tile in SBUF and PSUM space: a structural change, not a field edit).

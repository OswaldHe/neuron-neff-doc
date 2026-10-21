# Review of iter-4 (working-tree `kernel.neff` vs HEAD fa7f641 = 6712.06 us; baseline 7919.96 us)

I diffed with `neff_diff.py`, read `profile_latest.json`, `verify_latest.json` and `.autohelix/notes/iter-4.md` (the iteration's notes), and queried `prof/Instruction.parquet` for the numbers in sections 4 and 5. `semaphores.py` was run on an unpacked copy of the current NEFF. I did not re-run the device, the profiler or the notes' deadlock simulator myself.

## 1. What changed, and does it match the notes?

`neff_diff HEAD kernel.neff`: 4 members changed (Activation0, DVE0, PE0, Pool0). SP0 and all data members are identical.

| member | slots HEAD -> now | change |
|---|---|---|
| DVE0.bin | 14037 -> 14139 (+102; the 154 MEMSET deletions are more than offset by the inserted bare ESs) | See the four edits below. |
| Pool0.bin | 5283 -> 5283 | 154 `TENSOR_TENSOR ADD` slots rebuilt in place as 512-element fp32 `COPY` (opcode 0x46). The diff shows the src1->src0 address-register moves and the opcode/struct change. Pool now has 1727 COPY + 2260 ADD = 3987 S[3] increments, the same total as before. |
| PE0.bin / Activation0.bin | same slot count | Only `events_semaphore_value` (S[6] / S[3] thresholds) and wait-index fields rewritten after the DVE renumbering (4958 PE slots, 2038 ACT slots touched). No opcode, address or dtype change. |

The four edits (notes' table, which matches the slot-level diff):
1. Delete 154 DVE `MEMSET` (y_accum zero-init, fp32, 512 elem, value 0). The group's first Pool `y+=Cs` ADD becomes `y = Cs` as a Pool COPY. `0 + x == x` in fp32, so this is exact. S[6] is renumbered. -18.7 us.
2. Hop 76 stalling DVE `y+=Cs` ADDs one compute slot later and clear the extended S[3] wait on the gating ES. A fresh bare ES carrying the S[3] wait goes in front of the moved ADD. -29.7 us.
3. For 68 more sites, re-point the recycled `Cs` buffer to a fresh 2 KB buffer from the free SBUF hole at 62272..114688, then hop. -49.9 us.
4. Move 112 lag-0 S[6] waits from the gating ES to a fresh ES just before the real consumer. No renumbering. -9.7 us.

The four savings sum to 108 us, which is the measured delta. The notes match the diff. The commit-summary text should be taken from the notes, not from HEAD's message. Unlike iter-3, the iteration is documented (`notes/iter-4.md`). It lives in `.autohelix/notes/`, not the repo `notes/`.

## 2. Is the latency change real?

Yes. `profile_latest.json`: median 6604.127 us, samples 6603.703 / 6603.737 / 6604.127 / 6604.358 / 6604.604, spread 0.014% (about 1 us). The HEAD figure is 6712.06, so the delta is -108 us, about 100x the spread. Against baseline 7919.96 it is -16.6%. The loop is not chasing noise at this size. Each of the four wins (-9.7 to -49.9 us) is still 10x or more above the 1 us spread, but each was measured in a single 5-repeat run, not re-measured across processes.

## 3. Safety beyond the verifier

- **Output error.** `output_0` max_abs_err 1.250e-01 (83% of the 0.15 limit), cosine 0.999944. `output_1` is 1.812e-04. Both are identical to `baseline/verify_baseline.json` (I checked), so the error has not moved. The notes report 3 runs bit-identical.
- **`semaphores.py` on the unpacked NEFF.** Every semaphore has total_inc >= max_wait: S[3] 3987/3986, S[4] 2555/2555, S[5] 5023/5023, S[6] 8022/8022. S[15] has total_inc 688 but max_wait 672, an overshoot that was already there.
- **That tool is blind to extended waits.** `EVENT_SEMAPHORE` slots carry a second wait at bytes 32-43. `semaphores.py` does not read it: it said "balanced" for a NEFF that deadlocked in this iteration. The notes' 40-line in-order simulator, which includes extended waits, is the real deadlock check. I did not re-run it. The current NEFF runs and finishes, so a deadlock is excluded. A latent race would not show up that way.
- **Where an unseen race could hide.**
  - Edits 2-3 clear WAR/RAW waits and rely on static read/write-set checks. Two earlier attempts got `max_abs_err` 14.3, which is what a missed hand-off looks like. The final set passes the notes' checks (`imm0_imm_ptr` carry operand, Pool-read WAR), and the 3-run bit-identity supports it. The fixed-input verifier cannot prove it, because a missed WAR that never loses the race on the benchmark inputs passes.
  - Edit 4 depends on the hardware overlap depth being exactly 1 (the only instruction that can still be running when slot q starts is q's predecessor). This was established in iter-3 from the profile, not from a document.
  - Edit 3's fresh buffers (25 x 2 KB at 62272) rely on "no instruction field and no DMA descriptor references the hole". This was checked on the NEFF's fields. The runtime reservation of 8 bytes at 229368 is outside it. Spare SBUF is also what the next iteration wants, so the next agent must track which of the 25 buffers are now used.
- **Verdict:** safe as far as I can check. The residual risk is timing-dependent hazards that the fixed inputs cannot expose, so any further edit of this kind should keep running `verify.py --runs 3`.

## 4. Where the time is now

From `prof/Instruction.parquet` (one profile of the current NEFF):

| engine | busy | note |
|---|---|---|
| DVE | 6523 us (98.8%) | TT 5586 x ~670 = 3760 us, SCAN 2560 x 1269-1316 = 3370 us, 77 ns pipeline overlap at every boundary. Total gap 82 us. |
| Pool | 5615 us (85%) | 2260 ADD (median 2162 ns), 1741 COPY, 680 DMA_DIRECT2D (447 us) |
| PE | 2968 us (45%) | 5120 bf16 rank-1 matmuls (K=1 broadcasts of B and C into PSUM, 585 ns each) |
| ACT | 1903 us (29%) | SOFTPLUS and EXP, bf16->fp32 512-wide, 722 ns |

- Head 10 us, tail 4 us. Nothing to gain at the ends.
- **DVE EVENT_SEMAPHORE slots cost no time.** Compute-to-compute start gaps are -77 ns whether 0 or 1 ES sits between them (median -77, n = 2477 and 5482). Only gaps with a still-unsatisfied wait are positive. Deleting satisfied ESs gains nothing.
- **DVE TT duration does not depend on operand location or dtype:**

| class | n | median |
|---|---|---|
| PSUM src1 (the B multiply) | 2560 | 656 ns |
| PSUM src0 (the C multiply) | 2560 | 677 ns |
| SBUF-only fp32 ADD | 306 | 692 ns |
| bf16,bf16 -> fp32 | 160 | 676 ns |

  So there is no PSUM-read penalty and no bf16 speedup. DVE runs at 1 element per cycle regardless.
- DVE total is the wall. Of the 6604 us, 6523 us is DVE work, and the remaining gap budget is the 82 us in 356 gaps (S[6] lag-0 34.8, S[3] primary 24.2, S[3] extended 17.0, misc 5.4, per the notes' gap attribution).

**Remaining realistic upside is about 100-150 us (1.5-2.3%). All of it needs an address-level edit. Most of the scheduling slack is gone.**

## 5. Candidates for iter-5 (best first)

| # | Edit | Mechanism and where established | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Retire the 306 remaining DVE `y+=Cs` ADDs to Pool, with `Cs`/`Y` rotated through the free SBUF hole.** Engine: DVE0 (ADD slots with SBUF-only fp32 operands, src0 == dst == Y) to Pool0 (same ADD template). Re-address the Cs/Y buffer that the next DVE slot recycles ("~8 slots later") to a fresh 2 KB buffer from the 25 free ones at 62272..114688. The S[3] hand-off wait on the consumer then relaxes by k ordinals. | Re-addressing recipe from iter-4 win 3 (68 sites, -49.9 us, 734 ns/site). The old "at the optimum" result (590 ns DVE work vs 521 ns stall) was measured before buffer rotation lengthened the window. | DVE removes 306 x 692 = 212 us gross. Pool gains 306 x 2.16 = 661 us, so Pool goes 5615 -> 6276 (95%, wall 6604). Realistic **60-120 us**. Ceiling 212 us DVE, capped by Pool occupancy. | Medium. Needs the iter-4 WAR/RAW checks (Pool-read WAR on every hopped slot) and the extended-wait rewrite. Pool at 95% might become co-critical. The verifier cannot catch a missed WAR on fixed inputs. | Move 40 ADDs with fresh buffers, re-profile. Pass if DVE busy drops ~27 us and DVE gap total does not rise. Also check 25 buffers are enough. |
| 2 | **Finish win 3 on the last 15 S[3]-extended sites (17 us).** DVE0 sites blocked by a SCAN in the hop window, a Pool read of `y_accum` (fix: fresh buffer), or by the stalling compute being a SCAN. The scan can be hopped like a TT, and the carry slot at 16640+4n can get a second buffer. | Same recipe as wins 2-3, plus the notes' deadlock simulator. | Ceiling **17 us** (S[3] extended gap total = 17.0 us). Realistic 8-12 us. | Low-medium. The scan's `imm0_imm_ptr` carry is the hazard that already corrupted output once. | Hop one site, run `verify.py --runs 3`, expect about 1 us. |
| 3 | **Hoist the satisfied S[3] primary waits (124 gaps, 24.2 us).** Move the wait from the ES directly before the consumer to an earlier ES while DVE is still busy. | The notes show median lateness -22 ns, i.e. Pool is not late. The 118-131 ns is the time for DVE to observe an already-satisfied wait. | Ceiling **24 us** (S[3] primary gap total). Realistic 5-10 us. | Low. Waiting earlier is a stronger wait, so it is safe against races but can create a stall if Pool is late. | Hoist 20 sites by one slot and compare the gap sum. |
| 4 | **Merge scan chunk pairs of the same chain** into one `[1024]` scan. DVE0 `TENSOR_TENSOR_SCAN` (2560 slots) by `imm0_imm_ptr`. Requires placing chunk k and k+1 `dA`, `dBx` and `h` buffers adjacently and deferring chunk k's scan. | The carry does not reset across access-pattern dims (iter-1: `[256,2]` bit-exact), so a wider scan chains chunks correctly. The probe gives 220 ns fixed per scan. | Gross 1280 x 220 = **282 us**. The real ceiling is unknown. Chunk k's `C`-multiply is delayed ~86 slots, which delays PE's C_full bank release (the DVE<->PE lockstep that made TT merging 37% slower in iter-3). | High. Every operand moves, and the 25 free buffers are shared with candidate 1. It also needs the deadline analysis from scratch. | Time a single `[1024]` scan with a probe: if it is < 2 x 1269 - 220 ns the saving exists. Then check by simulation whether the deferred C-multiply lets PE's PSUM bank release stall. If it does, drop it. |

Rank reasoning: 1 has the best gain per unit of risk because the recipe is proven. 2 and 3 are the cheapest but small. 4 is large but uncertain. I would not invent a fifth item.

**Closed** (measured, do not redo):
- **DVE EVENT_SEMAPHORE slots are free** (gaps -77 ns with or without an ES between). Removing satisfied ESs gains 0.
- **PSUM->SBUF copy by ACT** to speed the 5120 DVE multiplies: PSUM-source TT is 656-677 ns, SBUF-only TT 692 ns. There is no PSUM penalty.
- **bf16 operands do not speed up DVE TT** (bf16 pairs 676 ns, same as fp32). The scan must stay fp32 for the error budget (83% of limit used).
- **TT merging** (iter-3): 2235 merges built, 37% slower, +1.0 us of critical path per merge from the DVE<->PE PSUM lockstep.
- **Scan merging across different states**: carry chains across rows (iter-1).
- **Lag-0 -> lag-1 S[6] relaxation**: all 222 sites alias (iter-4).
- **ACT as a bottleneck**: 1.2 us of binding stall, not 73 us.
- **Pool ADD shape** `[256,2]` vs `[512,1]`: identical duration and the wall regressed 6603.9 -> 6665.5.
- **Moving ADDs between Pool and DVE at the old window**: at the optimum (590 vs 521 ns).
- **PE fp32 accumulation of `y`**: all 5120 matmuls are bf16, so an fp32 matmul has no microcode entry on this NEFF.
- **Pool MAX ALU op**: hangs the engine (never present in the NEFF). Treat the set of (opcode, alu_op, dtype) triples present as the whole ISA.
- **Head/tail**: 10 us / 4 us.

**Unknown** (would change the ranking):
- Whether Pool MULT exists in the microcode tables. If it did, Pool slack (~1 ms) could take ~300 DVE multiplies, but they read PSUM and Pool cannot (iter-2). The measurement is a single Pool MULT probe slot; a hang costs ~70 s.
- Pool occupancy ceiling: candidate 1 pushes Pool to ~95%. Measure Pool's utilisation after a 40-ADD trial.
- Whether Pool-DVE SBUF port contention slows DVE TT while Pool runs a long ADD. DVE ADDs at 692 vs 656-677 ns for the PSUM multiplies hint at it but nothing isolates it.
- Whether candidate 4's PSUM-bank release stall appears: needs the one-`[1024]`-scan probe plus a simulation.
- Whether the 25 free 2 KB buffers cover candidates 1, 2 and 4 together (the 52416-byte hole).

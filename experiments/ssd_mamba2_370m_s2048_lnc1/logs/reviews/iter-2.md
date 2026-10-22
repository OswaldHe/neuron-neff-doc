# Review of iter-2 (kernel.neff vs HEAD 200d241; HEAD is itself 1.449x over baseline)

`git status` shows only `kernel.neff` modified relative to HEAD, so "this iteration" = HEAD -> working tree.
`baseline/` vs working tree also includes iter-1's edits, so I diffed against `git show HEAD:kernel.neff`.
I re-ran `tools/profile.py --repeats 1 --parquet /tmp/r/prof` (3290.19 us) to get a current per-instruction table;
`prof/` in the worktree is stale (17:15, older than the NEFF).

## 1. What changed

Two members changed; 35 of 37 are identical.

**`sg00/Activation0.bin`**: 11973 -> 10493 slots (-1480). The "11965 slots differ" line from `neff_diff.py` is an index-shift artifact. By opcode:
- ACTIVATE (0x21): 8313 -> 7692 (-621)
- EVENT_SEMAPHORE (0xa0): 2212 -> 1591 (-621)
- ACTIVATION_TABLE_LOAD (0x23): 265 -> 27 (-238)

A `difflib` alignment shows the edit patterns. Most are inside the recip(exp(cs)) units that iter 1 rewrote as Exp(scale=-1):
- `ACTIVATE func=128(copy) cs->scratch ; ACTIVATE func=7 scale=-1 scratch->dst` becomes one `ACTIVATE func=7 scale=-1`. It reads `cs` directly from the original source (pointer `0x2000000`, marker 2), with the same `events_semaphore_value`.
  - 362 are 1->1 replacements, 331 are 3->1 (two ACTIVATEs and a trailing self-wait EVENT_SEMAPHORE), 285 are 4->2.
  - The scratch-SBUF copy iter 1 introduced is gone for 633 units.
- The `ACT_TABLE_LOAD` pairs around those units are deleted: 122 pure deletes, plus the replace patterns that drop one.
- The trailing `EVENT_SEMAPHORE wait sem4>=N` slots are the ACT stream waiting on its own sem 4. They were deleted together with the copy they covered.

**`sg00/PE0.bin`**: 288 MATMULs changed, all identically: `events_wait_mode NONE -> WAIT_FOR_SEM_GE_IMM`, `wait_idx 0 -> 4`, `semaphore_value 0 -> N`. These are new, tighter waits on the ACT semaphore, compensating for the sync the deleted ACT slots used to provide.

**Notes:** `notes/README.md` is the empty template. The commit message describes only iter 1. The agent has recorded nothing for this iteration, so the next agent has to take this section as the record. That violates the notes rule, and "consistent with the notes" cannot be checked.

## 2. Is the latency change real

Yes.
- Median 3290.28 us, min 3289.04, max 3292.88, spread 0.117%, 5 samples.
- HEAD was 3622.2 us (iter 1 recorded 3624.21), so -332 us (-9.2%) is about 80x the spread.
- Cumulative: 5248.9 -> 3290.3 us (1.595x). Baseline spread was 0.022%.
- Engine busy (profile): ACT 4722 -> 2308 us, DVE 1689 -> 1653, PE 1255 -> 1304, Pool 1559 (unchanged), DMA 1193 -> 1140.
- Instruction counts agree with the diff: ACT 11996 -> 10516 in the summary.
- Nothing here looks like noise.

## 3. Safety

**Semaphores** (`semaphores.py` on unpacked HEAD and new):
- sem 4 total increments are unchanged at 5928, and `max_wait` is unchanged at 5928.
- Waiters on sem 4 dropped 8262 -> 7308 (-954 = the 621 removed EVENT_SEMAPHOREs + removed waits on deleted instructions, minus the 288 added PE waits).
- Every other semaphore is identical.
- No "wait exceeds increments" flag.
- The reduced waits are on the ACT stream waiting on its own sem 4, which are in-order and covered by dependencies now folded into one instruction.
- The added PE waits tighten ordering.
- I see no loosened cross-engine wait. But the verifier covers one input set, so a missing wait would show as a rare race rather than a failure.
- Cheap confirmation: run `verify.py` 5+ times, and with `--runs` if available. All outputs must stay bit-identical run to run.

**Error moved toward the limit.** Limit is 2.3e-07:

| output | baseline | now |
|---|---|---|
| output_0 | 1.881e-07 (82%) | 2.058e-07 (89%) |
| output_1 | 1.192e-07 (52%) | 1.993e-07 (87%) |

- output_1 error rose 67% from the exp(-cs) rewrite (iter 1 and this iteration). That is not bit-exact vs baseline.
- Margin is 11-13%.
- Any further numerical change to the decay path (fp32 -> lower precision, another algebraic rewrite) has little room.
- Treat the remaining ~11% as spent.
- The exp(-x) vs 1/exp(x) difference is not a rounding-only wobble. It is the same ~1 ulp-level difference per element, but it accumulates through the scan.

## 4. Candidates for the next iteration

Where the time is (current profile, `/tmp/r/prof`; busy = sum of durations, which can overlap; the `engine_busy` percentages in `profile_latest.json` are the non-overlapped numbers):
- ACT is still the busiest engine at 2308 us, 70.1% of 3290.
- DVE is 50%, Pool 47%, PE 40%, DMA 35%, SP 17.5%.
- ACT instruction time: ACTIVATE 7692 x 284 ns = 2182 us, DMA_DIRECT2D 816 x 535 ns = 437 us, EVENT_SEMAPHORE 1599 x 70 ns = 112 us, COPY 366 x 187 ns = 68 us, ACT_TABLE_LOAD 27 x 1283 ns = 35 us.
- ACTIVATE shape: about 4400 are 32-element/128-channel fp32 ops at ~317 ns; about 1400 are 1-element SOFTPLUS-class ops at ~165 ns; channels=1 ones ~200 ns. A minimum single-element op is ~143 ns, so ~150 ns of every 317 ns op is per-instruction overhead.
- SP has 17.5% busy and 2624 us of gaps. It is the one engine with room.

| # | Edit | Mechanism / evidence | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Move the 816 ACT-queue `PSEUDO_DMA_DIRECT2D` triggers into the SP stream** (`Activation0.bin` -> `SP0.bin`). The ACT queue ones use semaphores 40-55 (`dge:Activation`). SP already issues the 816 sem 56-71 (`dge:SP`) the same way. | Each costs ACT 535 ns of issue time (816 -> 437 us). SP issues the same DMA shape at 524 ns. Semaphore accounting (`semaphores.py`) attributes by increment, not issuer, and waiters on sem 40-55 do not care who posted. Precedent: this iteration shrank the ACT stream without breaking launch rules. | Upper bound 437 us of ACT busy (13% of runtime); realistic maybe 100-300 us, because ACT is not 100% on the critical path. It is bounded below by ACT's 2308 - 437 = 1871 us busy. | Medium-high. The DMA trigger's queue/ring may depend on the issuing engine (check the `queue`/DGE fields in the descriptor and the NEFF's DMA ring JSON members). An ACT-resident DMA is implicitly ordered after the preceding ACT ops; at SP it needs an explicit `wait sem4>=N`, and it has only one wait slot per instruction (the existing wait may already be in use). A missing wait is a race that the verifier may not catch on one input set. | Move ONE trigger (e.g. the first), re-run verify and profile. If device errors or output changes, dead in minutes. Also check first whether the existing wait slot is free on those 816 slots (decode `events_wait_mode`). |
| 2 | **Delete the remaining 27 `ACTIVATION_TABLE_LOAD`s** by finishing the exp rewrite for the 135 units not converted (768 - 633) and grouping function use so the table is not switched. | 1283 ns each, measured in the profile (27 x 1.283 = 34.6 us). Pattern is established by iter 1 and this iteration (238 deleted this time). | <= 35 us (1.1%). | Low; same pattern as before (the 135 left may be ones where the source is not directly readable, which is why they were skipped). Error budget: each additional exp(-x) swap has already raised error; output_1 is at 87%. | Convert 5 of the 135, check max_abs_err on output_1 doesn't move beyond 90%. |
| 3 | **Remove ACT `EVENT_SEMAPHORE` slots that are pure self-waits** (ACT waiting on sem 4 that only ACT increments) where the next ACT instruction already carries the wait in its own header. | 1599 ACT EVENT_SEMAPHORE at 70 ns = 112 us in the profile. The 621 deleted this iteration are the proven case. | <= 112 us (3.4%) in the best case; many remaining ones guard cross-engine data (DMA/Vector/PE), so the safe fraction is unknown. | Medium. Only the ones whose wait target is sem 4 (ACT's own) are safe to drop. Anything waiting on 5/6/DMA sems is a real hazard and a race that one input set won't show. | Count how many of the 1591 EVENT_SEMAPHOREs wait on sem 4; if few, kill it. |
| 4 | **Merge the many small 32-element ACTIVATEs** into fewer instructions with a larger free dimension (e.g. two adjacent heads -> `num_elem [64,...]`). | 317 ns for 32 elem vs 165 ns for 1 elem: roughly half the cost is fixed overhead. ~4400 such ops = 1.4 ms of ACT time. | Unknown. If adjacent-address pairs exist, up to ~150 ns x 2200 = 330 us of ACT time; whether wall time follows depends on ACT being critical. | High effort and risk: needs contiguous src/dst/scale/bias, semaphore re-accounting (one increment instead of two, or update values), and each merge changes the stream. Verifier catches wrong addresses but not a dropped wait. | Check whether consecutive same-function ACTIVATEs have src/dst addresses adjacent with stride = 32 elements. If not, dead. |

I considered and am not recommending: splitting PE work (PE is 40% busy), DVE STREAM_SHUFFLE (4608 x 148 ns = 681 us, DVE 50% busy, not critical), and Pool DMA triggers (47% busy, not the binding engine).

Ranking by gain per risk and effort: #1 has the highest ceiling and a cheap falsifier, so try it first. #2 is almost free but caps at 1%. #3 is only worth it after a count. #4 is last.

### Closed (don't redo)
- `imm_src` POINTER -> INSTRUCTION on all 6144 ACTs: done in iter 1, -8.3%, bit-exact.
- recip(exp(cs)) -> exp(-cs) through a scratch SBUF copy: done in iter 1; this iteration removed the copy for 633 units (-332 us).
- Table loads: 1531 -> 27 remain (35 us left).
- Error headroom: output_1 error 52% -> 87% of limit. No more precision-reducing edits to the decay path.

### Unknown (would change the ranking)
- **Is ACT actually on the critical path?** ACT is 70% busy, so there is up to 983 us of ACT idle. Measure: per-engine wait-for-producer breakdown in `Instruction.parquet` (`evt_wait_time_ns` for Scalar sums to 64.8 ms, far above wall time, so it is cumulative; look at what ACT waits on, per semaphore). This settles #1 and #4.
- **Do DMA triggers depend on issuing engine** (queue/ring in the NEFF JSON members)? Decode the 816 ACT-queue descriptors against the 816 SP ones; diff in fields settles #1.
- **How many ACT EVENT_SEMAPHOREs wait on sem 4 only?** One decode loop settles #3.
- **Run-to-run determinism with the new PE waits.** Run `verify.py` 5+ times for bit-identical outputs, because the verifier tests only one input set.

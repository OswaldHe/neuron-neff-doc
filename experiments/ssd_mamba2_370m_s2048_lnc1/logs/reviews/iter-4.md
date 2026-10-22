# Review of iteration 4 (kernel.neff, uncommitted on top of 1db4928)

Note: `notes/` holds only the stub README, so there is no agent claim to check. The comparison below is against HEAD (iter 3, 3106.0 us). The diff against `baseline/` is cumulative over iterations 1-4. I also unpacked both NEFFs and ran `semaphores.py` on each.

## 1. What changed

Two members changed, 120 slots each, nothing else (35 of 37 members identical to HEAD).

- **`sg00/Activation0.bin`, 120 slots**: `PSEUDO_DMA_DIRECT2D` (opcode 0xd4, the ACT-queue DGE staging copy, SBUF 0x47xx..0x4bxx -> SBUF 0xe780..0xeb80) -> `EVENT_SEMAPHORE` (opcode 0xa0).
  - The replacement is an ES that does a semaphore add of 16. Its bytes follow the same pattern as the iter-3 replacements.
  - Some slots, e.g. 227 and 1312, also keep the old instruction's wait field in the second half of the slot.
  - So the copy is deleted and its +16 semaphore increment is now issued immediately instead of at DMA completion.
- **`sg00/SP0.bin`, 120 slots**: the matching `DMA_DIRECT2D`'s `src_start_addr` is redirected from the staging address (0xe780..0xeb80) to the original state_sb address (0x4700, 0x4a00, 0x4900, 0x4b00 ...). Only byte 16/17 of the source address changes.

This is the same pattern iter 3 applied to 605 of the 816 (head,chunk) bounces, now applied to 120 of the remaining 211. About 90 ACT `DMA_DIRECT2D` staging copies remain (`Scalar DMA_DIRECT2D` count = 90 in the new profile).

Semaphore accounting is unchanged. `total_inc` and `max_wait` per semaphore are identical; only the source mix moves from `dge:Activation` to `Activation:EVENT_SEMAPHORE`, so counts balance.

## 2. Is the gain real?

Yes. Median 3071.04 us (min 3069.11, max 3072.41, spread 0.107% = 3.3 us) against HEAD's recorded 3106.03 us. The 35 us (-1.1%) is about 10x the run-to-run spread. It is also plausible mechanistically: the ACT lost 120 DMA triggers at about 0.3-0.5 us each, plus the latency of the removed hop on the SP chain.

Caveat: the baseline profile's spread is 0.02%, so measurement noise is not the issue. The loop is not chasing noise yet, but returns are shrinking: -24.7%, -9.2%, -5.6%, -1.1%.

## 3. Safety

- **Hazard I could not rule out.** The staging copy snapshotted state_sb. Now the SP DMA reads 0x4700 etc. directly, at some later time. If the ACT, DVE or PE rewrites that tile before the SP transfer executes, the old value is lost.
  - The +16 is still issued at the ACT program point, so the order is ACT-at-that-point -> SP. Nothing constrains the writer of the *next* version of the tile.
  - A race like this gives wrong data intermittently, not a one-time fail, and one fixed input set can pass it.
- **Error budget.** `max_abs_err` is 2.058e-7 on output_0 (89% of the 2.3e-7 limit) and 1.993e-7 on output_1 (87%).
  - Baseline was 1.881e-7 (82%) and 1.192e-7 (52%).
  - Output_1 rose 67% over the iterations.
  - A pure move or reordering of data cannot change fp32 values. A growth in error therefore hints at a stale or partially stale read in some earlier iteration, or at an exp(-x) vs 1/exp(x) rounding change (iters 1-2). That second cause is expected and legitimate.
  - Headroom is thin. No further numerics-changing rewrite should be attempted without re-measuring.
- **Check before accepting.**
  - Run `verify.py` with at least 3 repeats on the device. `verify_latest.json` shows `runs: 1`.
  - If the harness allows it, run with different random inputs. Compare output_1 error against HEAD's verify (I have not seen it) to see whether this iteration moved it.

## 4. Candidate edits for iteration 5 (ranked)

Context from `prof/Instruction.parquet` (3082 us span):
- ACT busy is 2124-2468 us of 3071. Of the ACT busy time, `ACTIVATE` is 2181 us over 7692 instructions at about 284 ns average, EVENT_SEMAPHORE 146 us over 2335, COPY 68 us over 366, DMA 49 us over 90.
- The ACT idle gaps total 1131 us.
- All five engines have 1.1-2.4 ms of idle gap and 11-66 ms of summed `evt_wait` (wait events overlap, so this is not additive).
- The kernel is a latency-bound chain of tiny ops, not a throughput-bound one.
- ACT table loads are essentially closed: 17 left, 22 us total, of which one is a 51 us start-up load.

| # | Candidate | Where / what | Mechanism | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Finish the staging-copy removal (remaining ~91 bounces) | Activation0.bin: ~90 ACT `DMA_DIRECT2D` (source lines ssd_block.py:149/226/514, ssd_utils.py:100/119) -> `EVENT_SEMAPHORE` SEM_ADD_IMM +16. SP0.bin: matching `DMA_DIRECT2D` src to the original SBUF address. Where the dst of the SP transfer needs the staged-resident SBUF buffer (the iter-3 "211 stage" cases), the copy cannot be removed and must stay. | Same mechanism as iter 3/4 (SBUF->SBUF SP DMA reading directly). | Measured per-slot gain: 35 us / 120 = 0.29 us per slot -> about 26 us more. Ceiling: 48.7 us of ACT busy in those 90 DMAs. | Same stale-read hazard as §3, higher because these are the ones iter 3 chose to *stage*, possibly for a reason (the source is overwritten before the SP transfer). Verifier may not catch it. | Before editing, for each of the 90, find the next writer of the source tile (DVE/PE/ACT instruction with the same address) and check its program order against the SP DMA via the semaphore table. If any writer is unordered w.r.t. the SP DMA, drop that slot. |
| 2 | Remove redundant ACT `EVENT_SEMAPHORE` waits | Activation0.bin: 2335 ES (146 us busy; most are 40 ns waits, plus the new SEM_ADD_IMM ES). Use `semaphores.py` to find waits whose threshold is already implied by an earlier wait on the same semaphore in the same engine stream. | A wait on a monotone semaphore with threshold <= an earlier wait is a no-op. Documented in the semaphore model used by `semaphores.py`. | Ceiling 146 us of ACT busy (minus the genuine waits). Realistic: tens of us, if the ACT is on the critical path. | Low if strictly dominated waits only; a mistake becomes a race that can pass the verifier. Never remove a non-dominated wait. | Count dominated waits statically. If under ~100, the ceiling is under ~5 us. Stop. |
| 3 | Merge the 40-ns ES and ACTIVATE sequencing: fold waits into the ACTIVATE's own wait field | Activation0.bin: an ES followed immediately by an ACTIVATE that has no wait of its own -> move the wait into the ACTIVATE's semaphore-wait fields and delete the ES. | Instructions carry their own wait fields (see how `Activation:ACTIVATE` already has `S[4]>=524` in operands). | Up to ~40 ns x number of pairs; at most about 90 us if all 2335 were mergeable. Probably 20-50 us. | Medium: field encoding must be exact, and only one wait per instruction is allowed. Fails to load if wrong, so it dies quickly. | Hand-edit one pair, run on device; the profile should show the ES count drop by one and the output stay identical. |
| 4 | Cheaper `COPY(scale/bias enabled)` on ACT | 5376 ACTIVATEs at ssd_block.py:454: those with a per-partition pointer scale cost about 320-400 ns, against about 170 ns for immediate-scale SOFTPLUS/COPY. | If the scale is a constant per head, an immediate scale (`imm_src` INSTRUCTION_IMMEDIATE, as iter 1 did for the pointer imm) saves the pointer read. If it is data-dependent (it is: `scale=[0x4688]`), it cannot. | Ceiling: 5376 x ~150 ns = 0.8 ms of ACT busy, but it only applies if the scale is a compile-time value. It most likely is not. | High: a wrong immediate silently changes values. | Dump the scale values at 5 addresses across runs. If any differ across inputs, drop this one. |
| 5 | Re-balance the GpSimd DGE load | Pool0.bin: 3193 instrs, 1562 us of `DMA_DIRECT2D` busy (software DGE, ~2400 per sem queue). | Moving some DMA triggers from Pool to the idle SP (SP is 19% busy) takes load off Pool. | Unknown. Pool is 51% busy and has 1655 us of idle gaps, so it is probably not the critical path. | Medium: queue/semaphore renumbering. | Build the critical-path chain from `SemaphoreUpdate.parquet` / `Flow.parquet` first. If Pool is never the last waiter before ACT stalls, skip. |

Rank rationale: #1 has a measured conversion rate and a mechanical edit, so it comes first. #2 and #3 share a static analysis that is cheap to run, so they come next. #4 is potentially the largest but probably not legal. #5 is speculation until the critical path is known.

Realistic total available from #1-#3: roughly 30-80 us (1-2.5%). The rest is probably dependency latency in a chain of ~40k tiny instructions.

## Closed (do not redo)

- ACT table reloads: 1531 -> 17 (22 us total). Nothing left to win.
- The ACTIVATE pointer `imm` operand: already turned off on all 6144 (iter 1, -8.3%).
- The 621 scratch-SBUF copies and barriers: deleted in iter 2.
- recip(exp) -> exp(-x) rewrite: all done. The last 8 units went in iter 3.
- Most SBUF->SBUF bounces: 605 + 120 of 816 done.

## Unknown (would change the ranking)

- **Critical path.** Which engine's instruction is the last waiter at each stall? Settle it with `SemaphoreUpdate.parquet` plus `Flow.parquet`, walking back from the end. This decides whether #1-#3 (ACT) or #5 (Pool/SP) matter at all, given that ACT is idle 1131 us.
- **Whether output_1 error growth (52% -> 87%) is numerics or a race.** Settle it by verifying the HEAD NEFF and the baseline NEFF with 3 repeats each and different seeds.
- **Why iter 3 left 211 bounces staged.** Check whether it was a hazard or just the SBUF space limit. This decides how many of #1's 90 slots are legal.

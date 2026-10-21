# Review of iter-5 (kernel.neff vs HEAD e6372de; HEAD itself is iter-4's accepted 38.33 us)

Note: `neff_diff.py baseline/kernel.neff kernel.neff` shows the *cumulative* edit of all five iterations
(6 members, 55 ACT slots). I diffed against `git show HEAD:kernel.neff` to isolate this iteration.
`notes/README.md` is still the empty template: the agent wrote no notes, so there is nothing to
cross-check the change against. The commit messages are the only claim record.

## 1. What changed (HEAD -> working tree): 4 members, 7 slots

| member | slot | change |
|---|---|---|
| Activation0 | 76 <-> 77 | ACTIVATE (wait S6>=60, upd S4) and TENSOR_SCALAR (wait S5>=62, no upd) swapped |
| Activation0 | 78 | ACTIVATE (identity x1.0) wait S6>=64 -> S6>=63 |
| Activation0 | 79 | TENSOR_SCALAR (wait S5>=63, upd S4) -> EVENT_SEMAPHORE, **no wait**, SEM_INC S4 (stand-in) |
| DVE0 | 96..99 + new 100 | CAST/TT swapped (96/97); the ACT[79] TENSOR_SCALAR is re-hosted as DVE[98] (wait S5>=63, update **new sem S61**); DVE[99] TT gains wait S61>=1 (was wait-less on that edge); old 98/99 shifted to 99/100 |
| PE0 | 142 | LDWEIGHTS wait S6>=63 -> 64 |
| SP0 | 39 | DMA trigger wait S4>=38 -> 39 |

This is the same two idioms as iter 3/4 (re-host an op onto an idle engine with an EVENT_SEMAPHORE
stand-in; reorder two ops and renumber the S4/S6 thresholds). No data fields (addresses, dtypes,
scalars) were touched except that the TS moved intact. There is no commit message yet, so I cannot
check a narrative, but the diff is internally consistent: thresholds shift by exactly the one
count that the swap/stand-in moved.

## 2. Is the latency change real?

- This iteration `profile_latest.json`: median 38.170 us, min 38.035, max 38.657, spread 1.63 % (0.62 us).
- HEAD as committed: 38.33 (message) / 38.417 (trailer).
- I re-measured both, 7 repeats each, interleaved A/B/A/B (copy of tools in /tmp):
  HEAD 38.377, 38.415; iter-5 38.109, 38.224 (plus 38.174 from a 1-repeat run). Paired gain = **~0.2 us (0.5 %)**.
- Verdict: the sign is consistent over 3 vs 2 runs, but the gain (0.2 us) is a third of the
  single-run spread (0.62 us). Treat as "probably real, ~0.5 %, not distinguishable from noise on its own".
  The loop's returns are shrinking: -10.3 %, -5.7 %, -1.8 %, -4.7 %, now -0.5 %. Cumulative 48.52 -> 38.17 (-21.3 %).

## 3. Safety

- `semaphores.py` on the unpacked NEFF: no wait exceeds its supply (S4 max_wait 40/total 41, S6 66/66, S5 64/64,
  new S61 1/1, S60 1/1). No hang risk.
- The one real change in sync semantics: ACT[79] stand-in now increments S4 *without waiting* (before: after S5>=63).
  Only S4 waiters with threshold >= 39 can observe this: SP[39] (>=39), PE[144] (>=40), DVE[96] (>=37, unaffected).
  SP[39]'s source was ACT[78] (count 38); it now waits on the stand-in, which in-order ACT fires right after ACT[78]
  completes, so the effective condition is unchanged. PE[144] waits on ACT[80] (count 40), which is later. The TS's
  real consumers are DVE[99] (now gated on S61, correct) and in-order DVE successors. I found no consumer that
  can now see the S4 increment before the data. This is a count-renaming that only holds while ACT is strictly
  in-order and ACT[78] precedes the stand-in; any later reorder of ACT[78]/[79] must re-derive it.
- DVE[98] now reads PSUM/SBUF written by PE with the same S5>=63 gate as before: fine. DVE-vs-ACT numerics for
  TENSOR_SCALAR: output error identical to the baseline to 4 digits (output_0 6.119e-05, output_1 2.864e-04), consistent with bit-identical.
- **Error headroom**: output_1 max_abs_err is 2.864e-04 = **82 % of the 3.5e-04 limit**, but this is the *baseline's* value
  (baseline/verify_baseline.json is identical), so the loop has not moved it. Any edit that changes arithmetic
  (precision, order of accumulation) has only 18 % headroom; all scheduling edits are free of that.
- The verifier uses one fixed input set; WAR/RAW races introduced by moving waits would pass it only by timing luck.
  Run verify 3x after any edit that removes a wait.

## 4. Ranked candidates for iter 6

Where the time goes (prof/Instruction.parquet, 1 capture; times are timeline-relative, body ends ~50 us on this clock):
the tail from 43.1 us is a serial chain
`DVE[90..97] (gapless, ~2.7 us) -> SP[37]/ACT[78]->SP[39] DMA triggers (45.33/45.91) -> SBUF->SBUF DMA (sem 27/28, ~1.3-1.4 us)
-> PE[141..143] (46.74-47.14) -> DVE[98..100] (47.18-48.32) -> Sync[40]/Pool[20] output DGE (0.67 us issue) -> store DMA lands ~50.0`.
Engine busy: Scalar 53.7 %, Vector 51.9 %, DMA 39 %, PE 31 %, Pool 25 % (idle in the tail), Sync 23 %.
Pool/GpSimd is the only engine not yet used as a compute re-host target (iter 3/4 used it only for DGE rows).

| # | Candidate | Engine/slots | What changes | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Re-host one *waiting* tail Vector CAST (DVE[92], then [95]/[97]) on idle Pool | DVE0 slots 92/95/97 -> Pool0 new slot; DVE keeps EVENT_SEMAPHORE stand-in that does the same wait/update | Move the 64-B slot (same CAST opcode struct), keep wait on the original sem, make DVE stand-in wait on a fresh sem the Pool op increments (same trick as S61 this iter), keep S6 count by having the stand-in do the S6 increment | Established on device in iter 3-5 (Vector->Scalar, ACT->DVE, DGE->Pool/Sync with stand-ins; exp. in neff_doc/experiments). **Not established**: that Pool executes CAST at similar speed | Each CAST is 0.19-0.22 us on DVE inside a gapless chain; best case ~0.2 us per op moved, 0.4-0.6 us if two. Cap: DVE chain 43.12-45.80. If Pool is 2x slower per op the move loses (chain waits for Pool) | Race if the fresh sem threshold is wrong; Pool/DVE share SBUF port so DVE may slow. Verifier will not reliably catch a race | Move only DVE[92]; profile once. If Pool CAST duration > ~0.45 us, or Pool refuses the opcode (hang/err), stop |
| 2 | Same idiom for the update-less ACT TENSOR_SCALARs in the tail (ACT[75] 0.39 us, ACT[77] 0.35 us) -> Pool or DVE | Activation0 slots 75/77 | Re-host with stand-in; ACT chain 43.0-45.88 is also serial and gates SP[39] | Same as this iter's ACT[79]->DVE[98] | Bounded by the next gate: ACT[78] waits S6>=63 (DVE chain); ACT currently starts 78 at 45.63 vs DVE ready ~45.55-45.80, so ceiling is **~0.1-0.25 us**, not 0.35 | Low-moderate; stand-in ordering must be redone for S4 thresholds (SP[39]>=39, PE[144]>=40, DVE[96]>=37) | Do the count arithmetic with `semaphores.py` first; if the S6 gate of ACT[78] is the binder, drop it |
| 3 | Shorten the 2 SBUF->SBUF DMA hops in the tail (sem 27/28 triggered at 45.33 / 45.91; arrive 46.74 / ~47.2) by splitting each over more queues or putting the second on Pool/Act DGE | SP0 slots 35/37/39 (PSEUDO_DMATRIGGER, ring "qSPSpillReload0") | Re-point 39 to a queue whose descriptors are already pre-generated, or fire 37 and 39 from different engines in parallel | DMA doc ch.9 (queued vs DGE); queued triggers were hoisted successfully in iter 1/2 | Latency of one hop is ~0.85-1.4 us and the tail contains two in series -> **ceiling unknown; ~0.3-0.6 us plausible**. Source of bound: PE[141] idles 45.95-46.74 and ACT[80] idles 45.88-47.26 waiting on them | Descriptor surgery in the variable table; the verifier will catch a wrong address but not a short transfer on bytes it never reads | Check in DmaPacket parquet whether hop latency is queue-wait or transfer: if the packets of the queue start within 0.1 us of the trigger and the cost is the 2.6 KB x 16 descriptors, the lever is dead |
| 4 | Halve the issue time of the final output stores (Sync[40] 668 ns, Pool[20] 672 ns DMA_DIRECT2D) | Sync0 slot 40, Pool0 slot 20 | Split each DGE row over fewer partitions/descriptors per engine (e.g. two engines each issue half of Sync[40]'s rows) so the last store finishes sooner | DGE row duration ~ 634-687 ns for all 16-row DMAs on every engine, suggesting ~40 ns/descriptor; hypothesis only | ~0.3 us if issue time scales with rows; 0 if it is fixed. Bound: store lands ~50.0 vs Sync[40] gated at 48.35 | Descriptor rewrite, sem_increment must still sum to 16 on S17 (waiters: Pool[22]) -> easy to hang | Compare the duration of DMA_DIRECT2D rows of different row counts already in the NEFF; if durations are flat, kill it |
| 5 | Find why Pool[20] starts at 48.209 although its only wait (S6>=65, DVE[99]) was satisfied at 47.78 (0.43 us lag) | Pool0 slot 20 (and 19, 21) | Likely an unaccounted gate (DMA queue slot / in-order behind Pool[19]); move the output DGE row to another engine whose predecessor is not busy | Observation from the per-instruction table | It is parallel to Sync[40] (ends 49.02 vs Pool 48.88), so the gain is **<=0.1 us** unless Sync[40] is also moved | Low | Look at evt_wait_time_ns of Pool[20] in Instruction.parquet; if the lag is semaphore propagation it is not fixable |

Rank rationale: 1 and 2 reuse a mechanism proven on this device three times; 3-5 need descriptor surgery with an
unknown ceiling. Expect each of 1/2 to deliver about the size of this iteration (0.2-0.5 us) -- i.e. at the edge of the measurable.

If the next agent does not reach a >=0.5 us gain in one try on #1, the remaining time looks close to irreducible
by scheduling: the critical path is a dependent chain of PE/DVE/ACT ops and two DMA hops, and the three cheap
rewriting idioms (cast re-host, reorder, DGE re-host) have been spent on the wait-less cases.

## Closed (do not redo)

- SBUF->DRAM->SBUF staging round-trips: fused to direct SBUF->SBUF (iter 1, 48.52 -> 43.50).
- Gate-load / output-store DGE rows on Pool -> SP; hoisting queued triggers above unrelated fences (iter 1).
- Wait-less Vector CASTs -> Scalar: all done (iter 3 DVE[49],[60]; iter 4 the last two). No wait-less CAST remains.
- sem-43 / sem-45 DGE re-hosting (iter 3), sem-17 output store reading true producer (iter 4).
- Split of Pool's composite S[14] fence (iter 2).
- Arithmetic/precision changes: not an option -- output_1 sits at 82 % of the error limit already in the baseline.
- ACT[79] TS -> DVE[98] re-host: done this iter, worth ~0.2 us (3 vs 2 runs).

## Unknown (would change the ranking)

- Pool CAST/COPY throughput vs DVE for these tiles -> decides #1. Measure: one CAST re-host, read `duration_ns` of that Pool slot.
- Whether DGE issue time (634-687 ns) scales with row count -> decides #4. Measure: tabulate duration vs rows over the DMA_DIRECT2D slots in Instruction.parquet.
- Composition of the ~1.3 us hop latency of the sem-27/28 DMAs (trigger-to-first-packet vs transfer) -> decides #3. Measure: DmaPacket.parquet start_ts of those queues vs SP trigger times (the queue naming "Q1-E64.." did not map cleanly in my one capture).
- Run-to-run noise at this size: spread is 1.6 % (0.6 us); every candidate above is at or below that. Use >=3 interleaved A/B runs of 7 repeats before accepting.
- What the ~3 us post-body tail (48.9-52.2: drains, EVENT_SEMAPHORE_RANGE_CLEAR, notify) contributes to `total_exec_time_us`; if it counts and is editable (it is runtime-generated postamble), it would dwarf all of the above, but I did not establish that it is editable.

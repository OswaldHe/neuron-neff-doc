# Review of iter-3 (kernel.neff, branch autohelix-iter-3)

Compared against HEAD (iter 2, 3289.6 us) and against `baseline/`.

## 1. What changed

Diff against HEAD touches 2 members. All other members are identical.

**`sg00/Activation0.bin`: 789 of 10493 slots differ.**
- 605 slots go `PSEUDO_DMA_DIRECT2D` -> `EVENT_SEMAPHORE`. The DMA is removed, but a semaphore op with the same semaphore, same index and value 16 stays in its place.
  - `semaphores.py` now lists sems 40-55 as `dge:Activation` plus `Activation:EVENT_SEMAPHORE`, e.g. 192+624 = 816.
  - So `total_inc` and `max_wait` are unchanged (816 each).
- 166 slots stay DMAs. Their `dst_start_addr` moves from a tag-9 spill tensor (`0x1xxxx`, `...09000010`) to an on-chip scratch address (0xe780-0xeb80 range). Their `dst_step_elem` goes 128 -> 262144.
- 10 slots `PSEUDO_DMA_DIRECT2D` -> `EVENT_SEMAPHORE` (opcode 35 -> 160) and 5-8 `ACTIVATE`/`EVENT_SEM` fields near slots 659/3887 change. These look like tail/boundary units.

**`sg00/SP0.bin`: 771 of 3119 slots differ.**
- 605 + 166 = 771, a one-to-one match with the ACT DMAs above.
- Each is an SP `PSEUDO_DMA_DIRECT2D` (the reload). Only the `src_start_addr` bytes change, e.g. slot 54: 0x12000 (tag-9 DRAM spill) -> 0xec00 (on-chip).

**In plain terms.** The ACT->DRAM->SP spill/reload round trip for the per-unit scratch values was shortened. 605 ACT-queue DMA triggers are deleted and replaced by a bare semaphore increment. The SP reloads now read an on-chip address directly. The remaining 166 are retargeted to on-chip scratch.
- Profile effect: `spill_save/reload_bytes` 6.68 MB (baseline) -> 0.37 MB. ACT `DMA_DIRECT2D` fell to 211.
- `hbm_read/write` fell 21.2 -> 14.9 MB and 19.7 -> 13.3 MB. DMA busy 1193 -> 1029 us.

**Consistency with the notes.** I cannot check it. `notes/README.md` is the empty template and the commit message for this iteration does not exist yet. The agent recorded nothing about intent, slots touched, or why 0xec00 holds the right data.
- This is itself a finding. The next agent starts with zero written knowledge of the aliasing assumption.
- The semaphore bookkeeping is internally consistent. Nothing else in the diff is unexplained.

## 2. Is the latency change real?

Yes.
- `profile_latest.json`: median 3106.03 us, min 3102.79, max 3107.10, spread 0.139%. My re-run measured 3105.2 us.
- HEAD was 3289.6 us (3290.28 in the commit trailer). That is -5.6% vs HEAD, a ~184 us step.
- Versus baseline 5248.9 us it is 1.69x (-40.8%).
- The spread is ~4 us and the win is ~184 us, about 45x the spread. The loop is not chasing noise.
- The device is deterministic here (baseline spread 0.022%). Wins under ~10 us are only distinguishable with 5 repeats and should not be accepted on 1 sample.

Per-engine busy (us): ACT 4722 -> 2151, DVE 1689 -> 1651, Pool 1559 -> 1559, PE 1255 -> 1275, SP 575 -> 583, DMA 1193 -> 1029.
- ACT is still the busiest engine at 69% of exec.

## 3. Safety beyond the verifier

**Error moved toward the limit.** The limit is 2.3e-7.

| output | baseline | now |
|---|---|---|
| output_0 | 1.88e-7 (82%) | 2.06e-7 (89%) |
| output_1 | 1.19e-7 (52%) | 1.99e-7 (87%) |

- output_1 error rose 67% over iterations. That is the exp(-x) vs 1/exp(x) rounding change from iters 1-2, not a new change in iter-3.
- There are only ~0.25e-7 (~11%) of headroom left on output_0. Any further numeric change (another approximation, fp32->bf16 anywhere) will likely fail.
- Cheap check: compare latest outputs bit-for-bit with HEAD's. If iter-3 is purely a data-movement change, the output should be *identical* to iter-2 output. That would prove the DMA removal is aliasing-safe on this input. `verify_latest` shows no separate number for this; **it was not measured**.

**Synchronisation.** `semaphores.py` shows per-semaphore `total_inc` and `max_wait` unchanged. So the wait counts are not reduced and no waiter can pass early because of count drift.

The residual risk is semantic, not counting:
- The deleted ACT DMA used to complete asynchronously (~539 ns) before the SP-side wait for that semaphore could be satisfied. Now ACT's `EVENT_SEMAPHORE` signals as soon as the ACT queue reaches it.
- The SP reload therefore reads 0xec00 *earlier relative to the data being valid* and *without a writer-completion guarantee*. It is correct only if what lives at 0xec00 was already produced by an earlier-completed op.
- The data at the slot's source is not the same address ACT DMA used to read (0x4f00 in slot 175), so there is a hidden assumption that the two locations hold the same value. I could not establish that from the NEFF within the review. It is guaranteed by the verifier only for the one input set.
- Write-after-read: if the 0xec00-region scratch is reused (the retargeted ACT DMAs write there, 0xe780-0xeb80), a late SP reload could read a later unit's data. Count-correct semaphores do not protect against that.
- The ACT queue is in-order and the semaphore increments are per unit, so the margin is probably large (the ACT->SP latency is ~µs). A race would show as intermittent wrong output.

**Cheapest safety check, not yet done:** run `tools/verify.py` with `runs>=3` and on a second random-input seed. A race that passes once is the failure mode named in the task.

## 4. Ranked candidates for iter-4

Profile facts the ranking rests on (per `/tmp/prof3/Instruction.parquet`, re-run at 3105 us):
- ACT `ACTIVATE` x 7692 = 2181 us.
  - 5376 are `COPY` with a **pointer scale** (`scale=[0x...]`): mean 329 ns, 1768 us in total. 4608 of them are 32-element, ~317 ns.
  - 2296 are `SOFTPLUS` with an immediate scale: mean 178 ns, 410 us.
- ACT `EVENT_SEMAPHORE` x 2214 = 139 us. ACT `DMA_DIRECT2D` x 211 = 114 us. ACT `ACT_TABLE_LOAD` x 17 = 22 us. Total ACT instruction time 2527 us, of which only ~1132 us is gaps.
- Pool `DMA_DIRECT2D` x 2385 at a constant ~655 ns = 1562 us. Sync `DMA_DIRECT2D` x 816 at ~534 ns = 436 us.
- PE `MATMUL`+`LDWEIGHTS`: 5441 each, 350 and 277 ns average.
- DVE `STREAM_SHUFFLE` x 4608 at 148 ns = 681 us.
- Engine busy is 69 / 53 / 50 / 41 / 33 / 19 % (ACT / DVE / Pool / PE / DMA / SP). No engine is at ~90% any more, so the critical path is a cross-engine chain, not one saturated engine. I have not measured the chain.

| # | Candidate | Engine / member / slots | Change | Mechanism and where established | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Make the 605 new `EVENT_SEMAPHORE` stand-ins free: fold the +16 into the preceding ACT instruction's `events_update_*` or delete it and move the increment onto the SP DMA's `sem_increment` | ACT `Activation0.bin`, the 605 slots converted this iteration (list by diffing HEAD vs now) | Remove the bare semaphore op. Either set `events_update_mode = SEM_ADD_IMM`, idx 40-55, value 16 on the previous `ACTIVATE`, or drop the op and raise the waiter's count only if the SP waits can be re-expressed | `semaphores.py` accounting; iter-2 notes in the commit log say "moving ACT's semaphore increment" was rejected -- the precedent shows it was illegal for the PSUM guard, not here | At most 605 x 63 ns = **38 us** of ACT instruction time, and only if ACT is on the critical path. An `ACTIVATE` already updates sem 4 (see `5906 5928` totals), and each instruction updates one semaphore, so only the ACTIVATEs *without* an update can absorb it. Fewer than 605 qualify | Medium: wrong sem or count deadlocks (a hang is caught immediately); loosening a wait would pass once | Count how many of the 605 have a preceding ACTIVATE with `update_mode` empty. If under ~300, drop the candidate |
| 2 | Delete the remaining ~166 retargeted ACT DMAs the same way | ACT + SP, the 166 slots listed in the diff, ACT dst 0xe780-0xeb80 | Same pattern as iter-3: `PSEUDO_DMA_DIRECT2D` -> `EVENT_SEMAPHORE`, retarget SP `src_start_addr` to the original producer address | Same mechanism as this iteration; **its aliasing proof is unwritten** (see section 3) | 166 x (539-63) = ~79 us of ACT time, 166 x 534 ns = 89 us of SP time (SP is 19% busy, so SP is not on the path). Realistic: 0-79 us | Medium-high: the 166 survived because deleting them was not safe, or the agent stopped early. **Find out why** (the retargets suggest the source was overwritten before the reload, i.e. real WAR) | Diff the 166: is the SP src address written by anything between the ACT DMA and the SP reload? If yes, it is closed |
| 3 | Probe whether the pointer `scale` operand is what costs 329 ns vs 178 ns | ACT, any 20 of the 4608 32-element `COPY` slots | Set `scale` to an immediate (wrong value is fine for a timing probe) and re-profile | The iter-1 win of -8.3% came from turning the unused `imm` pointer off on all 6144 ACTIVATEs (commit 200d241). The same class of cost may apply to a *used* pointer scale | Ceiling **unknown**. If the 150 ns gap SOFTPLUS vs COPY were all pointer cost: 5376 x ~100 ns = up to ~500 us of ACT time. Most of it is probably the 32-vs-1 element count, so expect much less | Low for the probe (timing only, output will be wrong, never accepted). The real edit is only possible if the scale value is a data-independent constant. The scale is per-head `dA` (data), so the final edit likely **cannot** be made | Patch 100 of the slots, profile only the durations: if mean stays ~320 ns, pointer scale is not the cost and this is closed in 5 minutes |
| 4 | Look for the remaining redundant `SOFTPLUS` units | ACT, 2296 `SOFTPLUS` (410 us) | Check whether they are 1-element ops on values that already hold the result; if repeated per chunk, drop the repeats | The iter-2 trick (read PSUM directly) removed copies of exactly this kind | Up to 410 us ACT time, but they probably compute dt per head once. 2296/48 = 47.8 per head, so many look like per-chunk recomputation. Ceiling unknown | **High**: removing a real computation changes numerics; verifier may pass on one input | Decode 20 consecutive SOFTPLUS operands: if src addresses repeat unchanged between them, they are redundant. Otherwise close |
| 5 | Shorten the Pool software-DGE DMA chain | Pool `Pool0.bin`, 2385 `PSEUDO_DMA_DIRECT2D` | Merge two adjacent DGE descriptors that move contiguous ranges into one (halve the count) | Fixed ~655 ns each (std 5.8 ns, i.e. pure issue cost, independent of size) | 1562 us Pool time; halving gives ~780 us. But Pool is 50% busy, so the gain depends on whether Pool is on the critical path. **Unknown** | High: changes the queue-semaphore structure (sems 8-23, each counting 2384); reduces and rebalances waits, which is exactly the race-prone edit | Check in the parquet whether ACT/DVE idle intervals line up with waits on sems 8-23. If not, close |

Rank rationale: #1 and #2 reuse a mechanism proved on-device by this iteration, but both have small, bounded ceilings. #3 is a 5-minute probe whose result decides whether anything big exists on ACT. #4 and #5 have larger ceilings but unproven structure and higher risk.

## Closed

- **Activation table reloads.** 1531 x 1283 ns -> 17 x 1283 ns = 22 us left (profile `ACT_TABLE_LOAD`). Done in iters 1-2. Nothing left to gain.
- **Unused `imm` pointer on ACT.** Turned off on all ACTIVATEs in iter 1 (-8.3%, bit-exact).
- **ACT-queue spill DMAs.** 605 of 816 removed (this iteration); `spill_*_bytes` 6.68 MB -> 0.37 MB. Only the ~211 remainder is open (candidate 2).
- **Scratch SBUF copies for exp(-cs).** Removed in iter 2 (621 copies and barriers).
- **Numerical headroom.** output_0 is at 89% and output_1 at 87% of the limit. Anything that changes arithmetic is effectively closed.

## Unknown (would change the ranking)

1. **Is the iter-3 output bit-identical to iter-2?** Settles whether the DMA removal is purely a data-movement change. Method: run both NEFFs on the same input and compare outputs byte for byte.
2. **What is at 0xec00 and who wrote it?** Settles the aliasing risk behind candidates 2 and the existing edit. Method: trace writers of that SBUF range in `Activation0.bin`/`DVE0.bin` with `semaphores.py` ordering.
3. **Which engine's chain is the critical path?** ACT is 69% busy, but 1132 us of ACT gaps and 1689 us of Pool gaps suggest cross-engine waiting. Method: from `Instruction.parquet`, take each instruction's `evt_wait_time_ns` (Scalar sum 66 ms vs Sync 61 ms is an aggregate over ~10k wait events, not a time on the path) and follow the semaphore that released it back to its producer.
4. **Is the 32-element ACT `COPY` cost fixed latency or the pointer operand?** Settled by candidate 3's probe.
5. **Is the kernel near a floor?** Not established. DMA busy is 33%, PE 41%, so no engine's *throughput* is the limit. The ~3.1 ms looks latency-bound by a ~10k-instruction ACT chain. Remaining floor is unknown; further gains probably need the ACT per-op cost (~320 ns x 4608) to drop, which a NEFF-level edit may not allow.

## Process note

`notes/README.md` is empty. The next agent should write notes before touching anything: the aliasing assumption behind the 605 removed DMAs, and the slot lists, are currently recoverable only by re-diffing.

# Review: iter-4 (working-tree `kernel.neff`, uncommitted on top of f8380ab)

## 1. What changed

`neff_diff baseline kernel.neff` shows the cumulative edit (def.json, SP0.json/bin, Activation0.bin, Pool0.bin).
HEAD (f8380ab) already contains the qSPSpillReload1 queue, the DGE->EVENT_SEMAPHORE replacement and the
SBUF->SBUF fusion, so the commit message describes those.

**This iteration's own change is uncommitted. It touches `sg00/SP0.bin` only.** The diff against HEAD's
kernel.neff has 148 of 1507 slots changed. The slot count is unchanged and 33 of 34 members are identical.
Decoding the slots shows it is a pure reorder of the SP instruction stream:

- The change is in the "relay" `EVENT_SEMAPHORE`s. These are the ES slots whose encoded word is `0x15xx` (e.g. `0x153b`, `0x153c`, `0x153f`, `0x1541`...). They repay the +16 for a retired reload.
- Each relay ES that used to sit in front of a run of `PSEUDO_DMATRIGGER qSPSpillReload0` slots (slots 14, 33, 52, 80, 103, 128, 171, ...) was moved to after that run. Example: slot 14 `ES 0x153b` / triggers ring 6-13 became triggers ring 6-13 followed by `ES 0x153b` at slot 22.
- The triggers therefore issue earlier on SP, and the repay lands later.
- Trigger ring numbers and counts are unchanged. A few unrelated ES slots (sem field 0) were shifted by one position along the way.

`notes/README.md` is the empty template. The agent recorded nothing for this change: no slots, no rationale, no measurement. This does not match the "consistent with notes" test.

## 2. Is the speedup real?

| | median us | min-max | spread |
|---|---|---|---|
| baseline | 1351.4 | 1349.1-1352.1 | 0.22% |
| HEAD (per commit message) | 1207.4 | - | - |
| this iteration, `profile_latest.json` | 1176.9 | 1172.6-1178.8 | 0.53% |
| my re-run (`--repeats 1`) | 1175.4 | - | - |

- This iteration gives -30 us (-2.5%) against HEAD. That is about 50x the 0.5% (about 6 us) spread, and a second independent run reproduces it. It is real, not noise.
- Cumulative change vs baseline is -12.9%.
- Engine-busy figures are consistent with the change: SP/Sync busy is 131 -> 52 us and DMA busy is 420 -> 341 us versus baseline.
- One caveat on the HEAD number: it is quoted from the commit message, not re-measured here. The -30 us is only as good as that figure.

## 3. Safety

- **Counts:** `semaphores.py` output for HEAD's NEFF and this NEFF is byte-identical, including per-semaphore max_wait, total_inc and sources. Increments and waits are conserved. It is order-blind, so it cannot see this class of change.
- **Verifier:** max_abs_err is 1.1e-4 against a limit of 1.1e-3 (10% of the limit). No error issue.
- **Residual race risk (medium):** the edit lets a trigger run before an ES that sits on the same SP queue.
  - If that ES's wait gated ring or semaphore-slot reuse, for example waiting for the previous reload of the same destination to retire, the trigger can now overwrite in-flight data.
  - The device ran once and matched, but a timing-dependent hazard may not show on a fixed input.
  - The verifier would not catch it deterministically.
- **Check to run:** for each of the 7+ moved groups, confirm that no ES moved past a trigger waits on a semaphore that the trigger's destination or source depends on. Run verify 5-10 times, not once.

## 4. Candidate edits for the next iteration

Profile facts that bound everything:

- Wall time is about 1176 us.
- Per-engine instruction-level busy (Instruction.parquet, 100 us bins) is flat across all 32 batches: PE 798 us (68%), ACT 680 (58%), DVE 592 (50%), Pool 297, SP 212.
- The same ratios hold in every 100 us window, so there is no serial prologue or epilogue to attack.
- No engine is saturated. The kernel is dependency-latency bound across a per-batch pipeline.
- The PE floor, set by MATMUL 540 us plus LDWEIGHTS 208 us, is about 750-800 us. That is the soft lower bound without changing the schedule.

| # | Edit | Where | Mechanism | Gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Finish the SP hoist this iteration started. Move triggers ahead of relay ES slots, and coalesce adjacent ES slots, in the remaining groups. SP shows 645 ES vs 669+192 triggers in an alternating `tete` pattern. | SP0.bin, remaining `ES 0x15xx` / `TRIG` groups | Same as the measured win: reorder within the SP queue, no new instructions. Established on-device this iteration (-30 us for 148 slots, about 0.2 us/slot). | Ceiling unknown. SP has 82 us of ES waits above 1 us and 13 us of short ones. Linear extrapolation to the untouched groups gives about 10-30 us. | Medium: the hazard in section 3. Verifier is blind to races. | Apply to 5 groups. If total moves by less than 3 us (spread is about 6 us), stop. |
| 2 | Apply the same hoisting on Pool, whose 288 `DMA_DIRECT2D` take 202 us (about 700 ns each, serial). Issue the DGE descriptors that gate PE/ACT input loads earlier in the Pool stream. | Pool0.bin DMA_DIRECT2D slots, and the ES before them (Pool has 208 ES, 61 us, 52 us of that in waits above 1 us) | Same as #1: instruction-order change on an engine that is only waiting. Pool is 20% busy, so it has slack. | Unknown. Needs a stall attribution (see unknown list). | Medium: DGE ring or semaphore reuse. | Check that Pool DMA issue-to-consumer-start latency is nonzero on the critical path. If consumers never wait on a Pool-issued DMA, drop it. |
| 3 | Convert the remaining DGE loads on Pool (the `dma` entries still on `qPool*` queues) to pre-built queued transfers, the way qSPSpillReload1 did for staging. This frees the 700 ns per descriptor of Pool/DGE time. | Pool0.bin / Pool0.json / def.json (new queue on free semaphores). Semaphores 72-87 are used; check what is free beyond 87. | Same mechanism as f8380ab (queued DMA on its own semaphore set). | At most 202 us of Pool time moves off Pool. Wall gain is unknown, probably small because Pool is not critical. | High: new queue, semaphore rebalancing. | Count how many of the 288 DMA_DIRECT2D are on the critical path first. If few, skip. |
| 4 | Trim redundant LDWEIGHTS on PE (1878 of them, 208 us). Where two consecutive MATMULs reuse the same weights, drop the second LDWEIGHTS. | PE0.bin | PE is the most-used engine. LDWEIGHTS can be skipped when stationary operands repeat. This is a structural change. | Ceiling is 208 us only if every one is redundant, which is unlikely. Realistic ceiling is unknown. | High: the PE queue is 4665 slots and a slot removal shifts every `pc` and semaphore wait. A wrong removal gives wrong numbers, which verify would catch. | Diff consecutive LDWEIGHTS operands in PE0.bin. If none repeat, dead. |
| 5 | Stop here. Remaining reordering gains are small and the kernel is latency-bound across the pipeline. | n/a | n/a | n/a | n/a | n/a |

Recommended order: #1, then #2 only after the unknown-list stall attribution says Pool matters.
#3 and #4 are speculative and should not be started without the measurements below.
I did not find a lever with a measured ceiling above about 30 us.

### Closed (do not redo)

- **Short ACT ES removal:** Scalar EVENT_SEMAPHOREs of 60 ns or less total 16.4 us (410 instrs). Deleting them all caps at 16 us wall, probably less. Not worth a slot shift.
- **DGE save -> ES swap on ACT:** already done in f8380ab (224 slots).
- **DRAM staging round-trips:** already fused and moved to qSPSpillReload0 and qSPSpillReload1 (d6d348b, f8380ab).
- **Tensor ES:** 501 ES at 53 ns, 24.7 us total on PE. Cap is low.

### Unknown (would change the ranking)

- **Critical path:** which engine's stall chains actually bound wall time. Settle with a per-instruction stall breakdown: `start_ts - prev end_ts` per engine, joined to `SemaphoreUpdate.parquet` for the waited semaphore. Use `tools/profile.py --repeats 1 --parquet prof`.
- **Whether the moved relay ES gated ring reuse (section 3):** settle with a multi-run verify on varied inputs, or by reading the wait field of the moved ES.
- **Whether PE LDWEIGHTS are redundant (candidate #4):** settle by dumping consecutive operand pairs from PE0.bin.
- **Whether Pool DMA_DIRECT2D issue latency sits on a critical path (candidate #2):** settle with the stall breakdown above.

## Housekeeping

- `notes/README.md` is still the empty template. The next agent needs the slot ranges of this reorder (slots 14-22, 33-36, 52-54, 80-84, 103-105, 128-132, 171-175, 196-203, 213-217, ... all in SP0.bin) written down.
- The change is uncommitted. HEAD's NEFF is 1207 us and the working tree is about 1176 us.

# Review of iter-1 (ssd_mamba2_1b3_s2048)

## 1. What changed

Only `sg00/Activation0.bin` changed (985408 -> 920128 bytes, 15397 -> 14377 slots). It is the only member that differs.
`notes/` holds only the README template. The agent wrote no notes, so there is no claim to check the change against. The description below is reconstructed from the slots.

Opcode counts, baseline -> new:

| opcode | baseline | new |
|---|---|---|
| `ACTIVATION_TABLE_LOAD` (0x23) | 2046 | 6 |
| `ACTIVATE` (0x21) | 10240 | 11260 |
| function 129 (`reciprocal_400p`) | 1024 | 4 |
| function 7 (`exp_400p`) | 2048 | 3068 |
| function 128 (`copy_1p`) | 7168 | 8188 |

All other opcodes, including the 0xa0 `EVENT_SEMAPHORE` slots, are unchanged in count.

Per chunk (1024 chunks), the edit replaces the `exp -> table switch -> reciprocal -> table switch` sequence with three ACTIVATEs in the exp table set. Baseline slots 6-9:
- slot 6: `exp(PSUM x)` -> `0x9610`, wait sem5>=3, inc sem4.
- slot 7: `exp` accumulate, unchanged.
- slot 8: `TABLE_LOAD` (reciprocal set).
- slot 9: `recip(0x9610)` -> `0x9618`, wait sem4>=3, no inc.

New slots 6-9:
- slot 6: `copy(PSUM x)` -> `0x9610`, same wait and inc as before.
- slot 7: unchanged.
- slot 8: `exp(-1 * 0x9610)` -> `0x9618`, wait sem4>=3, no inc. This is the old reciprocal output, now computed as exp(-x) = 1/exp(x).
- slot 9: `exp(+1 * 0x9610)` -> `0x9610` in place, wait sem4>=3, no inc.

The same rewrite repeats for 1020 of the 1024 chunks. The four remaining reciprocals and 6 table loads are untouched.

The other opcodes (0xd4, 0x46, 0xa0, 0xb1) are unchanged. The 4 slots whose 129 changed to 7 are consistent with the above.

## 2. Is the latency change real?

Yes.
- Baseline median is 7154.9 us (spread 0.02%).
- New median is 5129.0 us (samples 5126.3-5132.6, spread 0.124%).
- The delta is -2026 us (-28.3%), about 600x the spread. My own re-profile gave 5131.0 us.
- Mechanism is visible in the profile: Activation busy dropped 6284 -> 3768 us, and the 2040 removed table loads account for it. Everything else is unchanged within 1-2%: PE 1967 -> 2013 us, DVE 2733 -> 2701 us, DMA 1879 -> 1874 us.

## 3. Safety

**Semaphores.**
- `semaphores.py` on both unpacked NEFFs: sem 3-N all identical except the waiter count on sem4 (8678 -> 9698, the 1020 new same-engine waits). No "wait exceeds increments" flag.
- Total increments are unchanged (sem4 7360 = 7360).
- The only increment-carrying slot (old slot 6) keeps its `events_update` and its sem5 wait.

**Consumers.**
- I scanned all engine streams for the SBUF addresses involved. `0x9610` is touched only by Activation slots 6/8/9. The consumers on DVE (slot 101) and PE (118 slots) read `0x9618`, as in baseline.
- So the cross-engine hand-off still goes through `0x9618`, whose producer (slot 8) has the same wait and no increment, as before. I see no new race.

**Dead write.** Slot 9 writes `exp(x)` to `0x9610`, and nothing reads it afterwards. It is a leftover that costs about 194 ns x 1024.

**Error.**
- output_0: 5.122e-7 -> 5.234e-7 (84% of the 6.2e-7 limit).
- output_1: 3.166e-7 -> 3.334e-7.
- Both are still near the limit, so a margin of about 16% remains. Accumulating changes of this kind will not fit.
- exp(-x) vs 1/exp(x) matches in the overflow/underflow limits (inf and 0). The verifier uses a single fixed input, so it can't exercise extreme `x` — the risk is small but not tested.

## 4. Next candidates (ranked)

Profile facts for sizing (Scalar engine, per-instruction, one run):
- 11260 ACTIVATEs, 3.94 ms summed.
- 6144 PSUM->SBUF `copy` ([64] elems, 128 channels) at 435 ns each = 2.67 ms.
- 1024 `copy` of 128 elems on 1 channel at 489 ns = 0.50 ms.
- 3068 tiny exp ([1] elem) at ~195-200 ns = 0.60 ms.
- 1024 tiny copy at 163 ns = 0.17 ms.
- Scalar idle 1.62 ms of a 5.14 ms span.
- Scalar `DMA_DIRECT2D` 1088 at 559 ns = 0.61 ms.
- 6 `TABLE_LOAD` = 7.7 us in total.

| # | Candidate | Engine / member / slots | Change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Delete the dead in-place exp | Activation0.bin, new slot 9 of each chunk (1020 sites; find with `f==7, dst==src, scale=+1, wait 4/x, upd 0`) | Turn into NOP, or `EVENT_SEMAPHORE`, keeping the slot count so no branch/PC offsets move. If the wait field matters for later in-order reasoning, keep it. | Nothing reads `0x9610` after slot 8. This was established by my address scan (DVE/PE/Pool/SP/ACT, offsets 16-28/48), not on the device. | 1024 x ~194 ns = 0.20 ms (3.9%) if Scalar is fully on the critical path; likely 0.1-0.2 ms (Scalar took ~77% of the table-load saving: the saving was 2.0 ms of ~2.6 ms of loads). | Low. No consumer to starve, no semaphore changed. Verifier would catch a missed reader. | Re-scan addresses; apply to 1 chunk and run verify; profile Activation busy drops by ~0.2 ms or not at all. |
| 2 | Fold the PSUM copy into the exp | Same 4-slot window | Make old slot 6 `exp(scale=-1 * PSUM x) -> 0x9618`, keeping its wait (sem5>=3) and inc (sem4). Then slots 8 and 9 become NOPs. Baseline slot 6 already read that PSUM address with exp, so PSUM source is legal. | Same-address PSUM read is already done by baseline. Moving the producer of `0x9618` earlier is safe only if the slot-8 wait (sem4>=3) isn't a WAR guard on `0x9618`. | 1024 x (163 + 194 + 194) ns = 0.56 ms if all on the critical path; realistic 0.3-0.4 ms | Medium. `0x9618` ownership changes; the sem4>=3 wait on slot 8 may guard against the previous chunk's readers still using `0x9618`. Verifier on one input may pass with a race. | `semaphores.py` diff after edit; first run 20 repeats of verify with `use_*` unchanged; check slot 8's wait target against the producer of the previous chunk's readers on DVE/PE. |
| 3 | Same trick for the remaining 4 reciprocals and 6 table loads | Activation0.bin: the 4 slots with f=129, plus remaining `TABLE_LOAD` | Rewrite as exp(-x) in the exp set | Same identity | 7.7 us | Trivial | None needed; low priority. Do it only for the single-table-set side effect. |
| 4 | Drop the 128-element, 1-channel copies | Activation0.bin, `copy` with `channels=1`, 128 elements, scale from pointer (e.g. slots 12, 32, 44) | Unknown: replace by a single DVE/Pool op or merge | Not established | 1024 x 489 ns = 0.50 ms (upper bound, Scalar only) | High: cross-engine move needs semaphore rewrites; whole edit lives in several members | Dump 3 consecutive chunks of this op's producers/consumers; if the output is read by one engine only, test routing there |
| 5 | Move Scalar-issued `DMA_DIRECT2D` to a quiet queue | Activation0.bin + SP0.bin | Reissue the 1088 DMAs on SP (15.6% busy) | Not tried on this device | <= 0.61 ms, unknown whether on critical path | High (queue/semaphore bookkeeping across members) | Check whether Scalar `DMA_DIRECT2D` time overlaps ACTIVATE time in `start_ts/end_ts` |

Recommendation: do #1 first (minutes, near-zero risk), then #2 once the sem4>=3 wait is understood. #4 and #5 are worth only a measurement for now.

**Closed**
- Table-load removal from the exp/reciprocal switch: done in this iteration, 2046 -> 6 loads, -2026 us. Nothing left except the 4 reciprocals (#3, 7.7 us).
- No earlier iterations or notes exist for this kernel.

**Unknown**
- Is Scalar the critical path? The per-instruction `start_ts/end_ts` of Scalar vs the waits on sem5 (PE) and sem6 (DVE) would settle it. Scalar is 73.5% busy with 1.62 ms idle gaps; whether those gaps are waiting on PE/DVE decides the realised gain of #1/#2.
- What the 128-element, 1-channel copies are (candidate 4): needs the producing and consuming instruction on each side.
- Whether any kernel input produces |x| large enough for exp(-x) vs 1/exp(x) to differ. The verifier checks one input.

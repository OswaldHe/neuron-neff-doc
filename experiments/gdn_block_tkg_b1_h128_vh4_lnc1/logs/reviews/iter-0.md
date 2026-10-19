# Review of iter-0

`notes/` holds only the README template. There is no `profile_latest.json` or `verify_latest.json` in the worktree. Everything below comes from the NEFF plus a fresh profile I ran (`tools/profile.py --repeats 3 --parquet prof`).

## 1. What changed

Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports an identical header, 46 of 46 members identical, and "payloads are identical". This iteration did not edit the NEFF, so there is nothing to compare to the notes (which are empty).

## 2. Is the latency change real?

There is no change.

| run | median (us) | min–max (us) | spread |
|---|---|---|---|
| baseline JSON | 218.17 | 215.63–218.48 | 1.30% |
| my re-profile of the same NEFF | 219.87 | 218.23–222.49 | 1.94% |

The +0.78% between the two is run-to-run noise, since the bytes are identical. Any win under about 2% (about 4 us) needs 5 repeats and a clear gap before it counts.

## 3. Safety

- No edit, so no loosened synchronisation. `semaphores.py` on the unpacked baseline reports every sem with `max_wait` <= `total_inc`. Sems 33, 34 and 35 have `max_wait` = 0, which is harmless.
- **Error budget:** `verify_baseline.json` shows `max_abs_err` 2.999e-3 against a limit of 3.6e-3 (83%). Any edit that changes the arithmetic (dtype of a cast, an activation table, accumulation order) will probably fail. Restrict edits to scheduling, placement and data movement.

## 4. Where the time goes (baseline, sg00; the NEFF has one subgraph)

Total span is 232.6 us. Engine busy: DMA 52, Scalar 42, Pool 36, SP 27, DVE 19, PE 7.6 us. No engine is above 24% busy, so the kernel is latency-bound, not throughput-bound.

The timeline has three compute bursts separated by long dead windows:

| window (us) | what happens |
|---|---|
| 0–9 | runtime preamble (not editable) |
| 9–36 | weight prefetch (Pool, 49 DMAs, ~684 KB) plus first compute |
| 36–60 | short burst |
| 60–128 | **dead**: Scalar idle 64.8→128.1, Vector 61.3→128.8, Tensor 36→128. SP and Pool only run small DMAs. |
| 128–145 | burst |
| 145–204 | **dead**: Scalar idle 143→204, Vector 142→204, Tensor 130→210 |
| 204–226 | Scalar-bound tail (see candidate 1) |
| 226–232 | epilogue (semaphore waits plus a runtime reset, ~5 us) |

The dead windows are chains of small DGE DMAs: 256 B transposes done through an HBM scratch buffer, with `src_elem_size=2` and `src_pattern=[2,1][128,1]`, plus SBUF→TMPBUF→SBUF spill/reload.

What I measured on those chains (profile `SemaphoreUpdate.parquet` against `DmaPacket.parquet`):

- The data movement is fast. The `Q2` descriptors are 2 B each at about 8 ns, and each engine's whole DMA busy time is about 3 us.
- The **completion semaphores lag the data by 10–30 us**. For example:
  - Act's DMA issued at 58.6 us has its packets on `Q0` at 60.0, but sem 72 reaches 16 only at 78.2.
  - SP's group-3 use of sem 56 has its packets done by 167 us, but its 16 increments spread over 169.5–198.5.
  - Early in the run the same kind of DMA completes in 0.2–0.5 us (for example the sems 8–23 group 0).
- These lags directly set the restart time of the next burst. Sync gap 38.0→78.2 waits on sem 72. Sync 90.5→107.4 waits on sems 69 and 70. The tail trigger at 198.6 coincides with sem 56 ≥ 48 at 198.5.

I could not establish why the completion semaphores lag. That is the main unknown (see below).

## 5. Ranked candidates for iter-1

Slot numbers are SP/Pool/Activation/DVE streams in `payload/sg00/*0.bin` (decode with `isa.py`). Profile `compiler_pc` runs about 80 slots ahead of the stream slot number for DMA instructions, so check by opcode before editing.

| # | Candidate | Ceiling and source | Risk | Cheapest falsifier |
|---|---|---|---|---|
| 1 | **Cut ACT table loads and offload Scalar work in the 204–226 us tail** | ≤ 9 us. Scalar runs 204.4→225.5 with no gaps, and 7 `ACT_TABLE_LOAD` at 1.283 us each = 8.98 us (profile). Realistic 3–6 us. PE, DVE and Pool are mostly idle there. | Low–moderate. The verifier checks values. Wrong dependencies show as wrong numbers or hangs (run `semaphores.py`). | Decode the table `sel` and function id for `Activation0.bin` slots ~64–104. If the order is already A B A B with no legal regrouping, and no ACTIVATE is a plain multiply movable to DVE, it is dead in 10 minutes. |
| 2 | **Remove the identity spill round trip on the output path** | ~2.5 us (1.1%). `DmaPacketAggregated` shows SB(partition 32, 0x804708, 1024 B) → TMPBUF at 227.0, then 4×256 B TMPBUF → SB at 228.2–229.3, then 4×256 B SB → `output_0` at 229.5–230.3. The offsets 0, 0x100, 0x200, 0x300 are the same before and after, so it is a copy. | Moderate. Needs editing the queued-DMA descriptor blocks (ch. 9) for `qSPIO0` blocks 2–5 and `qSPSpillReload0` blocks 15–19. Wrong descriptors give wrong output, which the verifier catches. | Dump the descriptors for those blocks. If the output source can't be pointed at the SB source (partition 32 vs 0/96 mismatch) or TMPBUF is also used elsewhere, drop it. |
| 3 | **Fire the tail's cast reloads earlier** | ≤ 5 us. The two spill-reload cast DMAs (`mem_111_106`, `mem_112_107`, 3.6 us each, 199.3–203) are triggered at 198.6, but the data was spilled at 153–160. Sync then idles 198.9→204.1 (5.2 us) waiting for them. | High. It moves a trigger earlier, so the destination SBUF (0x34300, 0x834300) must be free between 160 and 198. A liveness error can pass the fixed-input verifier, or fail only intermittently. | Search every immediate operand in all engines for 0x34300–0x34b00 and 0x834300 between those times. Any hit kills it. |
| 4 | **Break DMA-ring semaphore reuse waits** | Unknown. SP slots 24–70 and Pool slots 17–32 each wait on `s(N) >= 16/32/48` before re-issuing a DMA that increments the same sem N. That is slot recycling, not a data dependency. Unused sems exist (46–55, 80+). | Moderate–high. Consumers wait `>=16/32/48`, so changing which sem a DMA increments needs matching changes to every consumer wait. Races would not necessarily show in the verifier. | Count how many stalls >1 us in the Sync/Pool/Act streams are ring waits (wait idx == own `semaphore` field). If under ~3 us total, close it. |
| 5 | **Understand and shorten DMA completion lag** | Possibly large: 10–30 us per hop across ~5 hops in the dead windows. This is a probe, not a candidate edit. | None to run. | Join `DmaPacket` end times to `SemaphoreUpdate` per DMA. Test whether lag correlates with the number of DMAs outstanding on the same queue rings. Try issuing the burst of small DMAs from one engine instead of three. |

Notes on ranking: 1 is cheap, low-risk and has a measured ceiling. 2 is a small but certain gain. 3 and 4 carry the highest race risk. 5 could change everything but is not yet an edit.

## 6. Closed (by measurement)

- **DMA bandwidth or descriptor count:** not a limit. 11k packets but ~3 us busy per engine; engine busy is 24% at most.
- **PE (7.6 us, 3.5%):** not a lever.
- **Weight prefetch on Pool (12.5→33 us):** completes before the first dead window; not what the dead windows wait on.
- **Arithmetic changes:** closed by the 83% error budget.

## 7. Unknown (would change the ranking)

- **Why DMA completion semaphores lag the data by 10–30 us** (item 5). If the cause is queue contention or ring ordering, a reissue order change could win more than candidates 1–3 combined. The join described above settles it.
- **Whether the Scalar tail is on the critical path to the end.** Check by delaying one tail ACTIVATE by ~1 us and seeing whether total time moves; if it doesn't, candidate 1's ceiling is much lower.
- **Whether sg00 weight loads gate the first burst.** Compare `Tensor` LDWEIGHTS start times against the matching Pool DMA completion sems (8–23, group 0).

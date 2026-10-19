# Review of iter-4 (kernel.neff vs HEAD fa6ac0a at 74.6 us, and vs baseline at 218.2 us)

## 1. What changed in the NEFF

Compared with HEAD (the accepted iter-3 NEFF); 5 members changed, 41 identical:

| Member | Slots | What |
|---|---|---|
| `Activation0.bin` | 113 -> 127 (59 differ) | ~14 new slots, mostly in-engine `CAST`s that each add +1 to semaphores 72-79 (those sems now get 16-24 increments, waiters wait for 16). `ACTIVATE`s were regrouped by LUT, so `ACTIVATION_TABLE_LOAD` count fell **13 -> 9**. Two `DMATRIGGER`-dummy replacements: sems 37/38 are now incremented by `Activation:EVENT_SEMAPHORE`, not a DMA. |
| `Pool0.bin` | 71 -> 75 (22 differ) | More per-partition DGE DMAs / waits turned into `CAST`s (sems 59, 63, 65 now have `Pool:CAST` sources). |
| `DVE0.bin` | 114 -> 115 (7 differ) | A few `TENSOR_TENSOR` / `SCALAR_TENSOR_TENSOR` slots reordered, plus one `EVENT_SEMAPHORE`. |
| `SP0.bin` | 85 -> 85 (20 differ) | `PSEUDO_DMATRIGGER` / `DGE` slots turned into `EVENT_SEMAPHORE` with `SEM_ADD_IMM_COMPLETE` (sems 68 and 70 now get 16 from SP and 16 from the DGE instead of 32 from the DGE). SP waits 63 -> 57. |
| `SP0.json` | +4.2 KB | DMA block descriptors. |

**Consistency with the notes: cannot be checked.** `notes/README.md` is still the 5-line template. The iteration left no note and no commit message. The decoded change is consistent with a continuation of iter-2/3 (DMA to compute/semaphore substitution, fewer ACT table loads, work moved to Pool). Tell the next agent that a note is required.

## 2. Is the latency change real?

Yes. `profile_latest.json` has median 60.157 us (samples 59.90-60.32, spread 0.69%). I re-measured with 3 repeats: median 59.892 (59.82-60.14, 0.55%). HEAD was 74.6 us, so the gain is about 14.5 us, which is more than 30 times the spread. The sha and size in `profile_latest.json` match the current file. Baseline is 218.2 us (spread 1.3%), so the cumulative speedup is 3.6x.

## 3. Safety beyond the verifier

- Error is **unchanged from baseline**: 2.999e-3 (83% of the 3.6e-3 limit), 1.928e-3 and 6.03e-4, identical to `verify_baseline.json`. The arithmetic is bit-identical on the fixed input. Output 0 sits at 83% of the limit, but that is the baseline's own error, not drift.
- `semaphores.py` reports no never-satisfiable wait. Compared with HEAD, only sems 24/26/33/37/38/56/58/59/61/63/68/70/71 changed. Waiters on 24, 26 and 33 were dropped (sem 33 now has 48 increments and 0 waiters).
  - Sem 63: max wait 16 -> 32 (tighter). Sems 68 and 70: a DGE increment became an SP semaphore add (equivalent).
  - Sems 72-79: 16-24 `Activation:CAST` increments against waits of 16. This is safe only because Activation is in-order and the intended CAST is among the first 16. If a CAST is moved before another increment of the same sem, the wait can pass early and the verifier will not notice, because the data is usually ready.
  - Sems 15, 22, 57-62 and 64 also have increments above max wait. They are unchanged from HEAD, so they are not new.
- Table loads were removed by regrouping, not deleted. Funcs 36 / 30+29 / 5 still run under tables 2 / 4 / 3. I did not verify the table-to-func map for the moved `ACTIVATE`s; bit-identical output says it is right on this input.
- Residual risk: the dead qSPSpillReload0 triggers still run (see candidate 5).

## 4. Ranked candidates

Timeline anchors (`prof/Instruction.parquet` and `DmaPacket`, µs on the profile clock; kernel work runs ~12.9 to 72.1): weight DMAs issue 13.1-20.3, `Q0` weights land 20.9, Act dense region 23.9-36.6, SP DGE chain 34.65-48.84, dependent tail 49.6-72.1.

### 1. Merge the 18 weight-load DGE DMAs into ~3 (best gain per risk)

| Field | Detail |
|---|---|
| Where | `Activation0.bin` slots 1-8 and `Pool0.bin` slots 1-12 (the 8 x 256 B tile loads, src offsets 0x000..0x700 of input 0, are Act slots 1,2 + Pool 2,4,8,10,11,12; the 8 x 8 B->16 B loads, src stride 0x400, are Act 3-8 + Pool 5,6). |
| Change | Each group becomes one `PSEUDO_DMA_DIRECT2D` with dim0 = 128 partitions and dim1 = 8 tiles. The first group needs a dst tile step of -0x100 (dst descends 0x37e00..0x37700). The second group needs src +0x400 and dst -0x300 per tile. Keep the elem sizes (8->16 for the cast). The other sems the consumers wait on get an `EVENT_SEMAPHORE` + `SEM_ADD_IMM_COMPLETE` after a wait on the merged sem (iter-2 trick), or the waiters are re-pointed. |
| Mechanism | **Verified on this NEFF:** Pool slot 1 is already a 2-D `[128,8]` DGE with a negative tile step (`[16,-2048]`). A DGE costs ~0.55-0.9 us of issue regardless of size (commit d596d33; profile: Act 8 DMAs = 6.92 us, Pool ~6 us). |
| Gain / ceiling | Up to ~4 us. Bound: Act spends 6.9 us and Pool ~6 us issuing before `Q0` lands at 20.9, and the first dependent ACTIVATE is at 21.1. The floor is one issue (~0.8 us) plus ~1.5-2 us of transfer. |
| Risk | Medium. A wrong dst order or step silently permutes weights. The verifier would catch that, because every weight tile feeds the output. Wrong sem accounting would pass on a lucky run, so check with `semaphores.py` and `--runs 5`. |
| Cheapest falsifier | Merge only the Act 3-8 group (the 8 x 8 B ones). Check `verify.py` is bit-identical and `DmaPacketAggregated` shows one transfer. If the DGE rejects the pattern, the device errors at once. ~10 min. |

### 2. Move Act CASTs / COPYs in the dense Act region to Pool

| Field | Detail |
|---|---|
| Where | `Activation0.bin` compiler pcs 33, 37, 43, 47, 48-50 (CASTs at 29.8-34.6) to `Pool0.bin`. |
| Change | Replace each with an `EVENT_SEMAPHORE` that keeps Act's sem count. Insert the `CAST` on Pool with waits for its inputs. Consumers wait on the Pool sem at the new count. |
| Mechanism | Playbook ch. 14.3(3), and **done this iteration** (Pool `CAST` sources for sems 59/63/65). |
| Gain / ceiling | ~1.5-2 us. Act is the busiest engine (41.6 us, 69%). 23.9-36.6 has no waits (engine-bound), with 5 table loads (6.4 us) + ~7 CASTs (~1.6 us). Pool idles 28.7-47.7. Ceiling = those CASTs' durations. |
| Risk | Low to medium. A Pool CAST has a different latency. A wrong count passes by luck. |
| Falsifier | Move one CAST (pc 33). Check Act's following instruction start shifts by ~0.2-0.3 us and the output is unchanged. |

### 3. Collapse the SP DGE transposes (34.65-48.84, 11 DGEs)

| Field | Detail |
|---|---|
| Where | `SP0.bin` DGE slots 18-21 (bf16 -> bf16, src stride -768 B, dst +256 B), then 34/36/46/50 (bf16 -> fp32, dst step 4), then 27/32/37/38. |
| Change | Each group becomes one DGE with a tile dim, or the group is spread onto Pool's and Act's DGE queues. Use fresh sems (46-55 are unused). |
| Mechanism | Same 2-D pattern as candidate 1. DGE = ~0.6 us issue each (observed back-to-back at 34.65, 35.31, 35.94, 36.59, 42.79, 43.48, 44.17, 44.81, 45.46, 47.42, 48.25). |
| Gain / ceiling | ≤ ~3 us. SP's 11 DGEs = ~6.6 us of issue, and the next consumer starts at 49.6. Part of this is gated by producer arrival (SP idles 38.1-42.4), so only the 18-21 group is certain. |
| Risk | Medium to high. Slots 18-21 had all producers ready by 34.65. Other groups wait on different sems, so merging forces a wait on the latest producer. |
| Falsifier | Merge 18-21 only and check the SP slot 27 start moves earlier by ~1.8 us and the output is bit-identical. |

### 4. Hoist `ACTIVATION_TABLE_LOAD` into idle time (after candidate 1)

| Field | Detail |
|---|---|
| Where | The first `ACTIVATION_TABLE_LOAD` (`Activation0.bin` slot 10, 19.81-21.10) moves ahead of the DMA block. |
| Change | Swap slot order (it has no wait). Do not do this before candidate 1: it would delay Act's own 8 DMAs by 1.28 us. |
| Mechanism | Act is in-order, and the table load has `events_wait_mode` NONE. |
| Gain / ceiling | ≤ 1.28 us. There are 9 loads x 1.28 = 11.55 us on Act. The tail ones (57.6, 61.2) are already hidden behind waits; the 25.3-34.9 ones are inline in the dense region. |
| Risk | Low. A table must be loaded before its funcs, which in-order issue keeps. |
| Falsifier | After candidate 1, check the first ACTIVATE start vs the `Q0` end. |

### 5. Remove the dead qSPSpillReload0 DMA triggers

| Field | Detail |
|---|---|
| Where | `SP0.bin` DMATRIGGERs that feed sems 29-35 and 39 (now 0 waiters; e.g. slots 31, 39-42, 51-54, 71-72). |
| Change | Make them `NOP` or `EVENT_SEMAPHORE` adds. |
| Mechanism | Their 2-byte writes to a DRAM scratch (the "dead buffer" in the iter-3 message). Profile: packets 52.40-55.40 (3.0 us) and 55.28-56.34 (1.06 us) occupy the 16 DMA engines. |
| Gain / ceiling | **Unknown**, ≤ ~1 us. SP cost is ~0.1 us per trigger and none is on SP's critical path; the only mechanism is DMA-engine contention with live transfers. |
| Risk | Low, if the scratch tmp-bufs are confirmed to have no readers. |
| Falsifier | NOP all of them and profile. If the median moves < 0.3 us (the spread), drop it. |

## Closed (do not redo)

- **Narrow DRAM scratch round-trips** (SBUF -> DRAM -> SBUF, and writing outputs straight from the per-partition layout): ~135 ns/element, ~18 us per 128-element transfer (commit 480fb81, 215 -> 95 us). This also closes skipping the final SBUF->SBUF 256 B hops before the output DMA (70.0-70.7 us).
- **Many tiny DGE DMAs** cost 0.55-0.9 us issue each plus a 2-5 us completion hop: already replaced by CAST / semaphore adds (d596d33, fa6ac0a). Merge them rather than replace them.
- **Weight loads on Pool's serial prologue**: already split Pool / Activation (fa6ac0a). Still unused: SP's DGE queues (SP is idle 12.3-22.5).
- **Duplicate and dummy DMAs from iter-1**: removed.
- **Removing waits on an engine's own completion semaphore**: verified negative (ch. 14.2).
- **ACT table-load regrouping**: done this iteration (13 -> 9). Remaining alternations 2 <-> 4 follow data dependencies.

## Unknown (measure first)

- **What gates the first compute (~21 us): weight arrival (`Q0` end 20.9) or ACT table load?** Delay an Act DMA by 1 us and see if the median moves. This decides how much of candidate 1 lands.
- **Shared DMA-ring contention from dead scratch writes**: candidate 5's NOP experiment settles it.
- **Is the 12.9 us before the first engine instruction inside `total_exec_time`?** The timeline spans 74.4 us but the metric is 59.9 us, and the `Q14` loads finish at 11.8 us. Compare the metric with timeline start/end on an empty-ish edit.
- **The fixed trigger-to-start-to-complete chain per DMA stage** (about 0.6 + 0.2 + hop): sets the floor for the 49-72 us tail. Measure the best case with one DMA.

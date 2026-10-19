# Review: autohelix iter-2 (gdn_cte_b32_s64_d128)

Compared `kernel.neff` (working tree, uncommitted) against HEAD (86fffa6, iter-1 = 873.6 us) and against `baseline/`.

## 1. What changed

**Against HEAD (this iteration's actual edit): 2 members, nothing else.**
- `sg00/SP0.bin`: 438 of 1329 slots differ, same slot count. Opcode changes are 10 EVENT_SEMAPHORE→PSEUDO_DMATRIGGER and 4 the reverse. The rest are field changes:
  - `dma_queue_name` on 4 slots (qSPSpillReload1 ↔ qSPSpillReload2).
  - `events_wait_idx` on 5 slots, `events_update_idx` on 1, `events_extended_wait_idx` on 1.
  - `count_union_count` on 4.
  
  This is a re-balance and re-ordering of the SP queued-DMA relay triggers between the two queues, not a new mechanism.
- `sg00/SP0.json`: 225 leaf values differ across about 161 DGE/queued transfers. `desc[0].from_off` is re-based by multiples of 4194304 (e.g. 114688→16794368, 115200→16796416) and `from_steps[2]` is 160/128→4194336/4194304. Those are the iter-1 multi-tile transposes with their source partition base and step re-expressed. `to_*` and sizes are untouched. I did not decode the address space, so I can't say whether the byte stream read is the same.
- `def.json` and Activation, DVE, PE and Pool streams are unchanged vs HEAD. No arithmetic instruction was touched.

**Consistency with notes:** `notes/` contains only the README template. The iteration wrote no notes and no commit message, so the claim can't be checked. The diff is consistent with "re-spread the SP relay chain over the two reload queues and re-address the transposes". The next agent should write the notes. This iteration's lever is otherwise undocumented.

**Against baseline** (iter-1 plus iter-2): Activation0 has 160 PSEUDO_DMA_DIRECT2D slots turned into EVENT_SEMAPHORE. def.json has new queues qSPSpillReload1/2 with 16 rings and semaphore sets 72-103. Pool gains 662 instructions (relay chain). SP is 1067→1515 instructions, but its busy time falls 479.7→34.5 us.

## 2. Is the gain real?

- Latest profile: median **836.40 us**, samples 838.1 / 836.4 / 836.3 / 832.6 / 839.0, spread 0.77%. My re-run: 837.0 / 836.4 / 835.0, median 836.37, spread 0.24%.
- Baseline is 1184.96 us (spread 0.15%).
- Iter-1 committed 873.6 us (874.5 in the commit trailer). Going from 873.6 to 836.4 is -37 us (-4.3%). That is about 5× the spread, so it is real.
- The sibling experiments saw a ±2–3 us floor at 1170 us, and 2% regression gates. Neither is near this gain.
- Total vs baseline: -29.4%, 1.417×.
- Engine busy (profile.py): SP 479.7→34.5, DMA 308→190, ACT 414→346, Pool 187→166, DVE ~437→439 (unchanged), PE ~263→263.
- Limitation: the iter-2 gain over HEAD is not attributed. I did not measure partial variants of the edit.

## 3. Safety beyond the verifier

- `semaphores.py` on the unpacked copy: no wait exceeds its supply. Every queued-DMA semaphore (S8-14, S72-103) has max_wait == total_inc (256 or 160), so no wait can hang.
  - `max_wait == total_inc` means every wait sits at the full count. That is zero slack, so no pass is possible at an earlier value.
  - **Count-race risk:** a `>=16` wait can be satisfied early if increments from a different DMA ring land on the same semaphore. The semaphore sets for Reload1 and Reload2 are disjoint (72-87 vs 88-103), so I found no sharing.
  - I did not do a full WAR audit of the re-based transposes.
- Verify: all pass. output_0 max_abs_err 7.69e-05 (11% of the 0.00071 limit). output_1 5.84e-04 (**82% of the limit**).
  - Baseline output_1 is the same 5.84e-04, so error did not move; iter-1 and iter-2 are bit-identical to baseline and the margin is inherited.
  - Any edit that changes accumulation order or a cast will fail on output_1 first.
- Verifier gap: one fixed input set. Scheduling or queue-ordering changes can race rarely. I'd run `verify.py --runs 3` or more before accepting a further queue edit.

## 4. Where the time is (from `/tmp/r2/prof/Instruction.parquet`, 850 us span)

- PE true busy (MATMUL 455 us + LDWEIGHTS 170 us) ≈ **625 us = 75%** of span. It is the busiest engine.
  - **Qualification:** the 568 us of PE gap is waits.
  - 2048 MATMULs: 1152 are 16×16 (`row_grp=q0 col_grp=q0`, ~165-180 ns each, source lines gdn_cte.py:441/447/449). 704 are 64/128-wide at 260-315 ns.
  - 1742 LDWEIGHTS at ~98 ns.
- DVE: TENSOR_TENSOR 188 us + COPY 129 + CAST 114 + MEMSET 23 ≈ 460 us (55%). ACT: CAST 177 + TENSOR_SCALAR 102 + ACTIVATE 95 ≈ 375 us (45%).
- Per-100-us-window busy (instruction durations) is flat from 100 to 800 us. That is steady-state, so the loop is a repeating chain across 32 batch items, not a prologue/epilogue problem.
- Pool: DMA_DIRECT2D 224 × ~660 ns = 149 us (descriptor generation, SWDGE), EVENT_SEMAPHORE 155 us.
- SP is idle (4%). DMA is 23%.
- Sum of all compute is far under 836 us, and no engine is above 75%. The critical path is the cross-engine dependency chain (the sibling notes measured ~250 ns per handoff).

## 5. Ranked candidates for the next iteration

Ordering is by expected gain per risk and effort. The honest headline: the SP/DMA overhead is now mostly gone (SP 4%, DMA 23%), and what's left is compute plus handoff latency. The sibling kernels' notes, which this kernel shares source with, say the cheap levers there are closed, and measurement shows only small wins after the first big one. Expect the list below to yield low single-digit percent, not another 4%.

| # | Candidate | Engine / member / slots | Change | Mechanism and evidence | Ceiling (and its source) | Risk and verifier coverage | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Retire the 224 Pool `DMA_DIRECT2D` SWDGE descriptor-generation slots by converting them to queued `PSEUDO_DMATRIGGER` on a pre-built ring | Pool0.bin, the 224 DMA_DIRECT2D (148.7 us, max 792 ns, plus Pool EVENT_SEMAPHORE 155 us) | Replace each DIRECT2D with a DMATRIGGER on a queue whose descriptors are prebuilt in Pool0.json (the same trick iter-1 used for SP and Activation: `dma_queue_name` + def.json queue + `dma` entry) | Established in this loop at iter-1: DIRECT2D→DMATRIGGER + queued ring removed 480 us of SP busy. Queue def is already present | Upper bound ~149 us of Pool issue time. Pool is only 20% busy, so wall gain ≤ the part on the critical chain: unknown, likely 5-20 us | Medium: new ring needs semaphore sets, and tx count must match. Verifier catches wrong data, not a rare race | Convert 8 of the 224 and re-run profile; if median moves <2 us, drop it |
| 2 | Re-spread the ACT-side CAST/TENSOR_SCALAR waits: look for ACT EVENT_SEMAPHORE slots that only forward a semaphore (dead ES) and merge them into the neighbouring instruction's wait field | Activation0.bin, 356 EVENT_SEMAPHORE (66 us busy, 9022 us cumulative wait) | Delete or neuter an ES whose wait is subsumed by the next instruction's wait, via the notes' "re-encode in place" trick (sibling iter-6 §5.2) | Sibling iter-6 shows an ES is re-encodable in place; baseline→iter-1 already turned 160 DIRECT2D slots into ES, so some may now be dead. 224 dead ES seen on the sibling kernel | Ceiling ≤ 66 us ACT busy; each ES ~185 ns, only those on the critical chain count: unknown, probably <10 us | Low if dead (no consumer); a removed wait that a later reader depended on is a real race the verifier may not hit | Run `semaphores.py` and look for ES slots with `waiters 0` on the semaphore they increment; if none, drop |
| 3 | Rebalance the 16×16 matmul group to cut LDWEIGHTS | Tensor0.bin lines :441/:447/:449, 1152 matmuls, 1518 LDWEIGHTS at q0/q0 (92.9 ns avg) | Skip an LDWEIGHTS when the immediately preceding one loaded the same stationary (same src address, no intervening write) | Needs checking first: the sibling s128 notes report **0 adjacent identical LDWEIGHTS+MATMUL pairs** over 2272 matmuls (iter-4 line 154), so this is likely refuted here too | Ceiling ≤ 170 us LDWEIGHTS, realistically ≈ 0 if same result as sibling | Verifier would catch a wrong weights change | Count adjacent identical (src, shape, row/col_grp) LDWEIGHTS pairs in the Tensor stream; 0 pairs = dead |
| 4 | Move the SP/Pool relay-trigger placement so the 161 transposes' triggers sit ahead of the engine that consumes them (earlier prefetch) | SP0.bin (~150 triggers), SP0.json 161 transposes | Reorder PSEUDO_DMATRIGGER slots earlier in SP program order, wait fields unchanged | This is what iter-2 plausibly did; sibling iter-5 §2.1 says **all SP instruction reordering is dead** (barrier ordering; head-of-line blocking) — so only ring/queue re-assignment is likely to pay | Unknown; iter-2 itself is the first measure (~37 us if that was the whole edit) | Hang if a trigger moves ahead of its producer | Move 1 trigger; if it hangs or no change, stop |
| 5 | Insert a fresh `ACT CAST` (clone + neuter) earlier to remove head-of-line delay | Activation0.bin, the 160 converted ES slots | Sibling iter-7 §2: clone-and-reposition to a bubble | Iter-7 measured 1172.3 vs 1172.3 (no gain) on s128 | Measured 0 on the sibling | — | Do not run unless item 1 shows ACT is on the critical chain |

### Closed (do not redo)

- **SP-side relay chain / SBUF→DRAM→SBUF staging round-trips:** DONE (baseline SP busy 479.7 us → 34.5 us now).
- **Queue/ring type changes and ring depth** (`fabric_path`, queue `type`, ring depth): closed on the sibling s128 (iter-3 §2.3, iter-4). A "ring of its own" per transfer measured nothing.
- **Remove all 218 waits entirely:** on the sibling that gave 1178.9, slightly worse.
- **All SP instruction reordering:** dead on sibling (iter-5 §2.1, iter-4).
- **Rewriting the ordinal of a waited-for instruction:** deadlocks (sibling iter-6 §2.1, HANG).
- **Dummy ACT/DVE compute rebalancing:** closed (sibling iter-7: dummy casts +6.6 us, repositioning +0).
- **`LDWEIGHTS transpose_mode` / `dge_op = DMA_TRANSPOSE` / merging queued transfers:** dead on s2304 (iter-6 §3, §6).
- **Dead DGE write retirement:** exhausted, ≤16 us, inside noise (s2304 iter-6).

Caveat: the closed items are measured on other kernels (s128, s2304) of the same source. This kernel is s64, so they are priors, not proof for this NEFF.

### Unknown (would change the ranking)

1. **Critical path:** which engine's wait gates the chain per batch item (≈26 us/item). Measure by following `evt_wait_time_ns` and the semaphore that unblocks each PE/DVE/ACT stall in the Instruction parquet. Settles whether items 1-2 can pay at all.
2. **What iter-2 actually bought.** Revert `from_off`/`from_steps` only and re-measure to separate address change from queue/trigger re-spread. It decides whether item 4 has more headroom.
3. **Adjacent LDWEIGHTS reuse** in the Tensor stream (item 3 falsifier).
4. **Semaphore sharing** between Reload1 and Reload2 beyond what `semaphores.py` reports.

If the unknowns come back negative, the remaining time (PE 75% busy with 2048 matmuls and handoff latency) is **close to irreducible at the NEFF level**. Items 1-2 are the only ones with a known mechanism, and each is probably worth under 1%.

# Review of iter-7 (kernel.neff vs HEAD 88ded90 = iter-6 NEFF; baseline 76.05 us)

## 1. What changed
`neff_diff HEAD -> kernel.neff`: 87 of 89 members identical; only `sg00/DVE0.bin` and `sg01/DVE0.bin` changed (same size, 124 slots, 42 slots differ). I decoded the slots myself:

- The multiset of 64-byte DVE slots is **identical** to HEAD in both subgraphs, and the permutation is identical in sg00 and sg01. Activation, PE, Pool and SP are byte-identical in both subgraphs. So no one-sided edit, and no field changed; DVE slots were only reordered.
- The permutation moves five `out[h] = out_norm * silu_z[h]` sinks later in DVE program order (they are TENSOR_TENSOR MULT, 1 channel, 128 elements, dst = `0x18060xx+` partition 96, `events_update_mode` NONE). HEAD slot -> new slot: 64->71, 82->105, 104->108, 111->116, 113->117. The slots they hopped over shift down by one.
- This matches `.autohelix/notes/iter-7.md`: "5 sinks moved, nothing else". The notes are in `worktrees/iter-7/.autohelix/notes/`, not in `notes/`, which is only the template. The note's "forward" means "deferred", not "earlier".

## 2. Is the latency change real?
Probably yes, but small.

- `profile_latest.json`: median 36.175 us, samples 36.007-36.190, spread 0.51%. The agent's own final run gave 36.106 (spread 0.84%).
- The iter-6 binary measured 36.805 in this iteration and 36.772 in its own iteration. The gap is about -0.6 to -0.7 us (-1.7%), roughly 3x the spread, and consistent across two binaries and two sessions. That is real, but it is the size of the noise floor of one 5-repeat median plus run-to-run drift.
- Do not trust any single claimed gain below about 0.3 us. Versus baseline: 76.05 -> 36.2 us (2.10x).
- `verify_latest.json` has `runs: 1`. The notes say 3 bit-identical runs.

## 3. Safety
- `semaphores.py` on both unpacked subgraphs: output for HEAD and for this NEFF is **byte-identical**, and sg00 equals sg01. No semaphore's wait or increment changed (sem4 45/45, sem5 38/38, sem6 80/80). The moved slots have no semaphore update and no wait on a different engine's counter, so there is no renumbering that could be off by one.
- Hazard check, done independently. The failure mode is a deferred sink reading a scratch buffer that someone overwrote in the meantime (iter-7's rejected edit E did exactly this, error 4.4).
  - For all five moved sinks I compared **byte extents** (512 B for the fp32 `out_norm` buffer) against every writer in the DVE, ACT, PE and Pool streams. No DVE slot in the hop window `[old, new)` writes into a sink's source extent.
  - The only overlapping later writer is DVE slot 110 (an op5 STT writing 0x5078). It sits after sink 105 (h5, reads 0x5078) and before sink 117 (h9, reads the 0x5078 that slot 110 produced), so the order is right. DVE is in-order, so no cross-engine race is possible for that pair.
  - No ACT, PE or Pool slot writes into any sink's source extent.
  - The output doorbell (`SP@9`, s6>=80) is still gated by the last DVE updater (slot 123). All 12 sinks are before slot 123 in the in-order stream, so the output DMA can't fire before a sink retires.
- Error: `max_abs_err` 2.322e-02 (86% of 0.027) on output_0 and 1.633e-03 on output_1. These equal the baseline exactly, so the arithmetic is bit-identical. The 86% is inherent to the kernel, not drift.
- Both cores run the same program, so they cannot disagree. The notes report pcore 1 finishing 0.34 us before pcore 0.
- Residual risk: the 5 moves are safe by the in-order argument above, but the verifier checks one fixed input set. The check that matters is extent overlap. Any further reorder must repeat it with extents, not start addresses.

## 4. Ranked candidates for iter-8

Context from the profile and the notes:
- DVE is 95% busy (34.1 of 36.2 us) and ACT is 92% (33.4). The body is a two-sided wall.
- Remaining DVE body stall is 1.69 us in total. About 3.1 us of tail after the last DVE slot is mostly runtime drain.
- DVE cost model measured by the agent: ~150 ns + 1 ns x elements per channel. 1 channel x 128 elements = 278 ns. 1 channel x 1536 elements = 1740 ns.
- I did not re-profile this iteration. The per-slot numbers below come from `notes/iter-7.md` and my decode of the DVE stream.

| # | Candidate | Where | What changes | Mechanism / evidence | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Split the output DMA and the last state-out DGE so the tail overlaps compute** | `sg00`+`sg01` `SP0.json`/`SP0.bin` (output_0 `PSEUDO_DMATRIGGER`, SP@9), `Pool0.json`/`Pool0.bin` (state-out DGE heads 10-11, Pool@7, and its waits Pool@8/@9) | Output_0: heads 0-10 triggered when s6 covers head 10's sink, plus a separate 256 B transfer for head 11. State-out: heads 10 and 11 as two DGE transfers (one per head), the second after `S+=` h11. Wait values on Pool@8/@9 and the semaphore totals (s16, s27) change accordingly. | Precedent: 7bc704d split state-out 1 -> 4 transfers (-5.8% jointly with the cast hand-off). Tail timing from notes: output DMA 47.39-47.65 (0.72 us launch latency); state DMA ~2.1 us for 128 KB, landing ~48.0; the two paths are within 0.09 us of each other. | Ceiling ~0.3 us total. Both paths must shorten or the other one binds. That is under about 1x the noise of one run. | A DMA triggered before its data is final would pass one input and race. A wrong sem total deadlocks (visible). | Re-read the tail from `prof/Instruction.parquet` and `DmaPacketAggregated`. Confirm `SP@9` ends ~46.67 and Pool@9 waits to ~48.0. If the output DMA wait is not the last event, stop. |
| 2 | **Batch the 12 `* beta_K` DVE ops across heads** (3 groups of 4 heads, or 12) | `sg00`+`sg01` `DVE0.bin`: the op6 MULT slots at 66, 68, 80, 85, 92, 95, 107, 115 (+4 more). Their sub partners (op5) at 65, 67, 79, 84, 91, 94, 106, 114 (+4 more) | The twelve `delta_raw` tiles are currently written alternately to partition 96 (0x1804370) and 32 (0x836400). Repoint the op5 dsts into one contiguous partition-0 scratch (free SBUF 0x27200-0x35500 per sbuf.py). Replace each group of K op6 MULTs by one op with `num_elem [128,K]` and `src1 step [0,1]`. `delta` is already contiguous at 0x22800+0x200h. | Same technique as 400b689 (zero-stride broadcast src1 collapsing 36 scale ops). Cost model: K=4 costs 150+512 = 0.66 us vs 4x0.279 = 1.12 us. | DVE busy -0.45 us per group, ~1.4 us total (K=12: ~1.7). **The wall then moves to ACT (33.4 us busy, within 0.7 us of DVE), so realistic wall gain is ~0.3-0.7 us, not 1.4.** The ACT `cast_delta` ops (12 x 0.297) would need batching too to go further. Unknown whether the Activation engine's wide cast scales like DVE. | Group fill delays the head's chain (the first head in a group waits for 3 more subs). The `delta_bf` scratch is reused across heads, so batching needs distinct tiles, and PE `s4` waits need re-deriving. The verifier catches wrong numerics but not a PE read before the cast. | Prototype one 4-head group at the end of the body (heads 8-11). If wall time is not lower, kill it. |
| 3 | **Partition-spread `z` so ACT `silu` becomes a 12-channel op** | `sg00`+`sg01` `SP0.json` `z` descriptor (`to 0x836600 [6144,1]`), `Activation0.bin` `silu` ACTIVATE (the 1.48 us slot), and the 12 sink/`silu_z` src1 addresses in `DVE0.bin` | Rewrite `z`'s `to_sizes/to_steps` as `[128,12]/[1,0x40000]` (12 partitions x 128). Change `silu` to `num_active_channels=12` with 128 elements. The sinks read `silu[h]` at `h*0x40000+...` | Descriptors step by 0x40000 per partition (docs ch. 5.3). `step_elem` has no partition axis, so this must go through the DMA descriptor. Cost model: silu 1.48 us -> ~0.3 us. | On paper -1.2 us on the ACT head chain, but iter-6 measured ACT and DVE arriving at the body boundary within 0.14 us, so the DVE head chain binds. **Realistic ceiling ~0.15-0.5 us unless DVE head work also shrinks.** Unknown. | Unproven that a partition-strided descriptor is legal for this input. Wrong data is caught by the verifier immediately (`z` gates `out`). | Change only the `z` descriptor in a scratch NEFF, keep `silu` 1-channel reading partition 0, and see whether output changes. If it breaks, kill. |
| 4 | **Defer the 12 op5 STTs (339 ns each) into DVE stalls, as the sinks were** | `DVE0.bin` op5 slots (SCALAR_TENSOR_TENSOR, e.g. 64, 69, 74, 81, 86, 93, 98, 109, 110, 120, 122) plus the waiters of their s6 updates in PE/ACT/Pool/SP | Move op5s ahead of stall slots; carry each op's s6 update and renumber every waiter on sem 6 | Same stall-filling idea as this iteration. Remaining stalls total 1.69 us (0.24+0.15+0.15+0.23+0.17+0.19+0.36). | Ceiling is the 1.69 us of stall, less what is already hidden. Realistic 0.3-0.7. | High. 10 of 12 update s6 and have waiters. A wrong renumber races silently and passes one input. | Do it for one op whose s6 update has exactly one waiter, check `semaphores.py` for identical totals, measure. |
| 5 | Give op5 h4/h7 private 512 B `out_norm` buffers so those two sinks can also be deferred | `DVE0.bin` op5 dst addresses at h4/h7 + sink positions | Move the two blocked sinks after the buffer fix | iter-7 found these are blocked only by 512 B vs 256 B overlap with `delta_bf` tiles; 0x28000 area is confirmed free | ~0.15 us (notes' own estimate). Below noise. | Same extent hazard that broke edit E. Needs an extent check, not start-address check. | Re-run `stall.py` and see whether any stall capacity is still unfilled. If it isn't, skip. |

Candidates 4 and 5 are weak. If candidates 1-3 each fail their falsifiers, the honest conclusion is that the remaining time (~36 us) is close to the DVE/ACT throughput wall for this schedule (DVE 95% busy plus ~3 us of drain), and further NEFF-level gains are all under about 1 us.

### Closed (do not redo)
- Delete all 12 sinks via a precomputed `norm_weight*silu_z` wide op (`fold.py`): 36.805 -> 37.174 (+0.37). The 1.74 us block cannot be hidden.
- Move ACT `T4+silu` to the stream front so the precompute hides (`head7.py`): 37.728 (+0.55 worse). The ACT head is a chain feeding DVE at four points.
- Sinks `out[h4]`/`out[h7]` deferred without fixing buffers: output_0 error 4.425 vs limit 0.027 (edit E). `semaphores.py`, `cycle.py`, `hb.check_psum` and the reachability check all missed it.
- Pool as a host for CAST or per-head TENSOR_TENSOR: Pool CAST is ~4x slower than ACT and each overlapping Pool CAST costs DVE ~1 us. Pool TT gave output_0 error 4.818 (iter-6).
- `fuse.py` and any ACT<->DVE elementwise move: 3 iterations measured zero (36.772 vs 36.828 in iter-6).
- Head table-load sequence (7 loads, 5 is the floor); input DMA order; stride-across-partitions access patterns (`step_elem` has no partition axis); further PE wait tightening (`pe4.py` slack 0.000).
- The ~3 us runtime drain after the last DVE slot and before ~12 us of preamble is outside the editable stream.

### Unknown (would change the ranking)
- **Whether the wide ACT `cast_delta` batch is cheap.** Settle: time one 4-head (512-element) ACTIVATE/CAST in a scratch NEFF. It decides whether candidate 2 can go beyond a 0.7 us ceiling.
- **Whether a partition-strided `z` descriptor is legal.** Settle: candidate 3's one-descriptor falsifier.
- **What binds at the body boundary after this edit.** The 0.14 us tie from iter-6 may have changed. Settle: `crit.py` on a fresh `prof/` parquet and check whether ACT `f29 q_scale` or DVE `delta h1` ends later. This bounds candidate 3.
- **Whether the tail is DMA- or DVE-bound.** Settle: compare SP@9 and Pool@9 end times, as in candidate 1's falsifier.

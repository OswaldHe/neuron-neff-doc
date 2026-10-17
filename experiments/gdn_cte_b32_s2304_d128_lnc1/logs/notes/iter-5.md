# iter-5 — `SEM_ADD_IMM_COMPLETE` works on the device, which finishes iter-3's save/reload fusion

## Result: 14039 us, **-1.41% vs the iter-4 NEFF** (14240), **-5.01% vs the iter-0 baseline** (14779)

| NEFF | total_exec_time_us | captures |
|---|---|---|
| iter-4 NEFF = my starting point | **14240.3** (med 3) | 14238.9 14240.3 14250.8 |
| + 707 dominated `EVENT_SEMAPHORE`s deleted | 14244.3 (med 5) | 14239.7 14241.2 14244.3 14244.3 14245.7 — **neutral, see §2** |
| + 68 save/reload groups fused | 14177.9 (med 5) | 14174.8 14175.6 14177.2 14177.9 14178.6 14181.4 |
| **+ 120 more groups (188 total) — shipped** | **14039.3** (med 5) | 14026.6 14026.8 14039.3 14041.1 14044.8 |
| the same shipped NEFF, independent second capture set | **14028.3** (med 5) | 14023.8 14026.4 14028.3 14030.4 14034.3 |

`verify.py`: PASS, `max_abs_err` **identical to the baseline to every printed digit**
(1.372e-04 / 6.841e-04, 17% and 86% of the limit). `single_core.py`: PASS.
`semaphores.py`: every semaphore's `max_wait` and `total_inc` **unchanged**; only the *source*
attribution moves (`dge:SP 2464` -> `dge:SP 2352 + SP:EVENT_SEMAPHORE 112`).
`neff_diff` vs baseline: 5 of 34 members changed (cumulative iters 1-5).

All **ten** shipped captures across two independent sets lie in 14024..14045, and every one is
below every capture of the starting NEFF (14239..14251): a ~205 us gap against a ~21 us cluster
width. The win is real. (I quote the first set's 14039.3 as the result, the more conservative of
the two medians.)

**Nine scripts are parked beside this file as `iter-5-*.py`** — the whole fusion pipeline in
dependency order (`grp2` -> `fuse2` -> `qacc` -> `fuse3` -> `prov`/`prov_half`/`filt` -> `apply`),
plus `delredun.py` and `cpath2.py` (the critical-path tool of §1). They need `work0/` unpacked
and `prof/` ingested **from the same binary**. `apply.py` takes `WK=` and `PLAN=` env vars.

---

## 1. THE NEW MEASUREMENT TOOL: an exact backward critical-path walk (`iter-5-cpath2.py`)

iter-4 built a forward simulator. This is cheaper and exact: it needs no calibration, runs in
~60 s, and accounts for **14158 of the 14251 us (99.3%)** of the observed span.

Walk backwards from the last-finishing instruction. For slot *i* on engine *E* the candidate
binding predecessors are (a) `EN[E][i-1]` (same-engine serial) and (b) `t_reach(sid, val)` for
each of its waits. Take the **latest**; that is what made it start. Attribute
`(EN[i]-ST[i])` to "exec" and `(ST[i]-t_bind)` to "stall", then hop to the producer:

* `S[3..6]` -> the `val`-th signalling slot of Pool/Activation/PE/DVE in program order.
* `S[24..39]` (queued ring) -> transfer id `(ceil(val/16)-1)*16 + (sid-24)`, i.e. the id-th
  `PSEUDO_DMATRIGGER` on SP.
* everything else (DGE) -> the `ceil(val/16)`-th `PSEUDO_DMA_DIRECT2D` naming `sid`, in program
  order **on its owning engine** (I re-verified: no DGE semaphore is hosted by two engines).
* For a DMA hop also charge `t_reach - EN[producer]` as `dma_latency`, or the walk loses it
  (this was 4.6 ms of my first run's missing time).

### What it says about this kernel (iter-4 NEFF, span 14251 us)

| us | term | on-path nodes |
|---|---|---|
| **2874** | SP `dma_latency` queued | 2631 triggers, 1092 ns each |
| 1622 | Activation exec `ACTIVATE` | 4802, 338 ns |
| 1472 | Activation exec `TENSOR_SCALAR` | 4093, 360 ns |
| **1140** | SP exec `PSEUDO_DMA_DIRECT2D` | 1858, 613 ns |
| 1079 | PE exec `MATMUL` | 3724, 290 ns |
| 960 | DVE exec `TENSOR_TENSOR` | 3621 |
| **939** | SP `stall_serial` | ~161 ns per SP slot |
| **906** | SP `dma_latency` DGE | |
| 715 | DVE exec `EVENT_SEMAPHORE` | 628 — contaminated, its duration includes the wait |
| 531 / 412 / 410 | DVE `CAST` / PE `LDWEIGHTS` / DVE `COPY` | |
| 388 / 360 | Pool / Activation `dma_latency` DGE | |
| 295 / 250 / 230 / 128 | Pool DGE / Act DGE / SP `EVENT_SEMAPHORE` / SP trigger exec | |

**SP owns ~6220 us (44%) of the critical path**, Activation ~3600, DVE ~2650, PE ~1500,
Pool ~840. Use this table to size a lever before building it — it killed four of my candidates
in minutes. Source-line attribution (`nki_source_location` in `Instruction.parquet`) shows the
path spread over ~30 `gdn_cte.py` lines with no single line above 180 us: **there is no hotspot,
only a wide flat chain**, which is why every remaining lever is worth <2%.

## 2. What I shipped

### (a) Delete `EVENT_SEMAPHORE`s whose wait is already satisfied — 707 slots, worth NOTHING

Semaphores here are monotone counts since program start (ch 10.1; no loops, no in-program
`RANGE_CLEAR`) and one engine's instructions complete in program order. So if an earlier slot on
engine E waited `S[k] >= V'` with `V' >= V`, a later wait `S[k] >= V` is **provably** already
satisfied. A pure `EVENT_SEMAPHORE` (no primary and no extended update) all of whose waits are
dominated is a no-op and can be deleted with no semaphore accounting change at all.

707 such slots exist (Act 129, DVE 100, Pool 30, PE 231, SP 217). Deleting them:
`semaphores.py` `max_wait`/`total_inc` unchanged, verify bit-exact, and the device gave
**14244.3 vs 14240.3 — nothing** (5 captures, 14239.7..14245.7). Those slots were not on the
critical path. **Measured, kept because it is free and provably safe, but do not count it as a
win and do not spend a cycle on this class again.** Folding is also dead: of the pure-ES runs
(Act {1:2479, 2:66, 4:228}, SP {1:6034, 4:123}, PE {1:3243}), the runs of 4 already carry 2 waits
each, so `ceil(W/2)` equals the run length — total extra deletions from `events_extended_wait_*`
folding is **0**. Re-running `iter-4-merge.py` after the deletion finds only **8** new runs.

### (b) **The real win: fuse 188 save/reload staging pairs** (iter-3 §"the next step")

This is iter-3's unfinished work, and the thing that unblocked it is:

> ### `events_update_mode = SEM_ADD_IMM_COMPLETE` (21) with `events_semaphore_value = 16`
> ### WORKS on the device, from an `EVENT_SEMAPHORE`, on a DGE semaphore. **VERIFIED.**

iter-3 and iter-4 both parked their biggest lever on this question. The probe (2 minutes, zero
risk): insert 8 `EVENT_SEMAPHORE`s with `update_mode=21, update_idx=73, semaphore_value=16` early
in `Activation0.bin` and one `EVENT_SEMAPHORE` near the end that waits `S[73] >= 128`. S[73] is
used by no NEFF instruction and lies in the runtime's Scalar clear range 54-104. It **ran in
10.3 s and stayed bit-exact**, so the 8 slots delivered 128. (A *passive* probe does not work:
`prof/SemaphoreUpdate.parquet` tops out at `sid` 71 — it does not record S[72] or S[73] even
though iter-1's S[72] scheme demonstrably works. Make something wait, do not try to read it.)
`tools/semaphores.py` already models mode 21 as `+value` with no changes.

**The edit.** For a staging group of N DGE "saves" writing an SBUF window plus exactly one DGE
"reload" copying that whole window to a destination:

1. Re-point every save's destination at the reload's destination — `dst_start_addr_addr_immediate`
   (= `marker << 56 | offset`; `marker 0x10` = VAR with `var_id 0` = SB, flat address
   `partition*0x40000 + offset`; `marker 0x20` = TBL, `| table_index << 32`, and
   `def.json var.DGETable0.list[idx]` gives the var id), plus `dst_step_elem`, `dst_elem_size`
   and `out_dtype`. Setting the 8-byte immediate updates every union view — assert
   `dst_start_addr_addr_var_offset` on a re-decode.
2. Replace the reload slot with **two** `EVENT_SEMAPHORE`s:
   * `ES_a`: the reload's own wait verbatim in the primary slot, **plus** `S[sid] >= 16*(m-1)`
     in the extended slot, where *m* is the reload's 1-based index among DGE slots naming `sid`
     on its engine. **The second wait is not optional**: without it the `+16` could push the
     count to `16*(m-1)` while transfer *m-1* is still in flight and release its waiter early.
   * `ES_b`: no wait, `SEM_ADD_IMM_COMPLETE +16` on `sid`.
   One value field per slot is why this takes two slots: the primary wait value and the ADD_IMM
   operand are the same bytes. (`events_extended_sem_update_value` is a *separate* field from
   `events_extended_sem_wait_value`, so one slot could in principle do 2 waits + 1 add, but the
   extended update mode is used nowhere in this NEFF and I did not probe it.)
3. Correctness: `ES_a` sits exactly where the reload sat, so the preceding `EVENT_SEMAPHORE`s
   that carried the group's other waits still gate it; when `ES_a` completes every save is done,
   so the data is already at the destination and releasing the consumers at `ES_a`'s completion
   instead of the reload's is sound — and ~1-2 us earlier, which is the win.

Net per group: a 128-256 KB SB->SB DGE transfer **and its completion latency** disappear;
613 ns of host issue becomes ~2x130 ns. 188 groups: 68 on SP (plain copy) + 120 on Pool
(with the FP32->BF16 cast folded into the saves, `dst_elem_size` halved, `out_dtype` ->
`BFLOAT16` — iter-3 had already proven such a save runs). 1280 saves re-pointed.
DMA busy 5476 -> 5290 us (-3.4%), Pool busy -2.8%.

### Finding the groups (and the two checks that are NOT optional)

* Group by **time**, not by var: after iter-3's phase 2 the DGE addresses are plain SB offsets
  and the `tmp-buf` var identity is gone. Take every DGE whose src or dst lands in a 4096-byte
  staging window of `[68736, 114432)`, sort by `start_ts`, and cut at each W->R->W transition.
  Shapes found: (8,1)x132, (10,1)x78, (4,1)x56, (9,1)x40, (1,8)x39, (2,8)x19, ... —
  **320 groups with >=2 saves and exactly 1 reload**, which is iter-3's number exactly.
  Each window holds ~230 writers and ~60-90 readers over the program, i.e. ~30 time slices.
* Of the 320: **229 reloads permute partitions** (`q_i != p_i`), which is the whole reason the
  staging exists; a general affine remap of the save's `dst_step_elem` recovers most of them
  (my `fuse3.py` does this and only 0 are lost to non-affineness) — after that the rejections are
  **101 hazard** and **8 save-outside**, leaving 211, then 23 lost to save overlap -> 188.

**Check 1 — the time-windowed destination hazard (iter-3's criterion).** Nothing other than the
reload's own write may touch the destination region between the first save's `start_ts` and the
reload's `end_ts`. I first ran this over compute slots and DGE only and **the device produced
1.47 M NaNs**: the 8285 `qSPSpillReload0` descriptors read that region too — `dma[0]` reads
`SB 227072`, which is inside the first group's destination `[225272, 229368)`. Adding the queued
descriptors (`SP0.json dma[i].desc`, `from_off`/`from_sizes`/`from_steps` are in **bytes**;
window = trigger `start_ts` .. `t_reach(S[24+id%16], 16*(id//16+1))`) rejected 33 more.
Of the 101 hazard rejections, 82 involve a queued read or another DGE write — those are precise,
not conservatism; only ~21 are compute-only and might be false.
**If you extend this edit, add the queued descriptors before you touch the device.**

**Check 2 — save overlap.** 23 groups have two saves writing the same window byte. In the window
the later write wins by time; at the destination the two saves are on different engines and
semaphores, so the order is not pinned. Reject them. (Dropping this check alone was the other
half of the NaNs.)

**Check 3 — byte/element provenance (free, catches everything before a device run).** Build
`dest -> source` for both schemes and compare. For a plain copy do it per byte; for a cast
reload do it per **element** (FP32 element *j* -> BF16 element *j*) — a byte-level model flags all
120 cast groups spuriously. `iter-5-prov.py` / `iter-5-prov_half.py`. This is what told me the
84 pure groups were already correct and that the failure had to be a hazard, without burning a
device cycle.

## 3. Hypotheses REFUTED this iteration (with the numbers; do not repeat)

1. **Deleting dominated `EVENT_SEMAPHORE`s is worth 0 us** — 707 slots, measured, §2(a). And
   `events_extended_wait_*` folding adds 0 more slots.
2. **Replacing a DGE with a compute instruction is nearly unreachable.** Only **169 of 8384**
   DGE transfers are identity-partition SB->SB (SP 120, Act 37, Pool 12); 5079 permute
   partitions and the rest touch DRAM/inputs/outputs. Compute engines cannot cross partitions,
   so the rest can never become a `COPY`. In particular Pool's 1152 `beta`->SB transfers
   (`src esz 4`, 647 ns each, 750 us of Pool issue) are a **partition scatter** of 4 bytes from
   256 contiguous DRAM bytes — not compute-expressible.
3. **Merging adjacent DGE slots is still dead, even with the semaphore problem solved.**
   `ADD_IMM` removes iter-4's "different semaphore" blocker, but adjacency does not survive it:
   only **262 followers** are in consecutive same-key runs with `num_elem[1]` free
   (Pool 262, SP 0, Activation 0; lens mostly 2), 256 of which carry a wait that then needs its
   own pre-`EVENT_SEMAPHORE`. Costed out: ~1050 ns for a merged pair against 1300 ns, i.e. ~50 us
   of Pool issue over all 194 pairs, against a Pool critical-path budget of only 840 us. Not worth
   the hang risk.
4. **iter-4's consecutive-run merge left nothing on the table.** I checked whether splitting runs
   at every delta change lost merges that a 2-level nest would catch: only **13 runs of 2**
   survive anywhere (the 13 "unexpressible" ones iter-4 recorded), and 2-level nesting adds
   **0** deletions. That lever is exhausted.
5. **PE has no perf-mode headroom to take.** `MATMUL`/`LDWEIGHTS` carry a `perf_opt` field
   (`PERF_OPT_MODE`: `DOUBLE_ROW`/`DOUBLE_COLUMN`/`DOUBLE_PIXEL`) and it is `NONE` on all 31840
   of them, but those modes change the expected operand packing, so flipping the field alone
   changes results. `performance_mode` in the profile is reported only for DVE `CAST`/`COPY`/
   `TENSOR_SCALAR` and is **already `2x`** there; DVE `TENSOR_TENSOR` (11584 slots, the biggest
   DVE block) reports none and has no mode field in its struct.
6. **The SP DGE host axis stays closed** (iter-2 swept it; its sim says removing all 1392 moved
   SP slots advances the triggers by only 268 us, mean 32 ns). My critical path agrees that SP's
   DGE exec is 1140 us on the path, but there is nowhere to put it: Pool is worse (iter-2),
   a 3rd host is impossible (only 4 queues, iter-2 refutation 6), and the Act/SP 50:50 split is
   within 2 us of the optimum of that axis.

## 4. What I would do next, in order

1. **Extend the fusion.** 132 of the 320 groups are still unfused: ~21 rejected only by my coarse
   compute ranges (`crange` in `iter-5-fuse2.py` takes one bounding interval per partition over
   all pattern dims x `num_active_channels` — enumerate the outer dims instead and most should
   clear), 23 by save overlap (could be admitted by ordering the overlapping saves on one
   engine), 8 by `save_outside`. At the measured ~1.07 us per group that is **another ~140 us
   (~1%)** and the whole pipeline is parked and working. *Cheapest next step on the whole list.*
2. **Apply the same trick to the `(1 save + N reloads)` groups** (39 of shape (1,8), 19 of (2,8),
   1 of (1,4)). Those are the mirror case: one save feeds many reloads, so instead of folding the
   cast into the save you would fold the *gather* into each reload's source — which is exactly
   iter-3's phase 1 and is already done for 1832 of them. Check first whether any are left.
3. **Delete the now-dead staging saves.** With `ADD_IMM` proven, a dead DGE can be retired
   (replace with the same `ES_a`/`ES_b` pair; the WAR barrier its completion provided becomes
   vacuous once nothing reads its destination). Needs a read/write-set sweep to prove dead —
   `iter-5-fuse2.py` + `iter-5-qacc.py` already build the whole access map, you only need the
   "no reader anywhere" query. iter-3 warns the traffic alone bought 0 us; the slot and the
   completion are the prize.
4. **Do not** spend budget on: dominated-wait deletion (§3.1, measured 0), DGE adjacency merging
   (§3.3), the consecutive-run instruction merge (§3.4, exhausted), PE/DVE perf modes (§3.5),
   compute-for-DMA substitution (§3.2), the DGE host ratio (§3.6), and everything iter-2/3/4
   already closed (engine occupancy as a proxy for latency, `ACTIVATE`->cheaper opcode,
   moving compute between engines, `def.json` DMA queues, DMA priority, trigger/wait tightness,
   `dge_op = DMA_TRANSPOSE`, queued-path SBUF sources).
5. The transpose lever (hints.md / iter-3's `transpose_mode` analysis) is still the only
   multi-ms item and still blocked on the free-axis reversal. The critical path now puts a number
   on it: **2874 us of queued `dma_latency`** sits on the path over 2631 transfers at 1092 ns
   each, of which iter-1 measured ~785 ns as the fixed trigger->start ring-fetch floor. So even a
   perfect reordering of those transfers cannot win; only **deleting** them can.

## 5. Tooling notes

* `prof/Instruction.parquet` has `compiler_pc` as **float with NaNs** — filter `notna()` before
  `astype(int)`, and dedupe (SP has duplicate rows).
* `prof/SemaphoreUpdate.parquet` covers `sid <= 71` only. S[72]/S[73] never appear even when they
  are demonstrably working. Its `id` is sometimes an int and sometimes `"S[6] (Vector)"`.
* DGE `step_elem`/`elem_size` are in **bytes**; compute `step_elem`/`num_elem` are in **elements**.
  Queued descriptor `sizes`/`steps` in `SP0.json` are in **bytes** (`from_sizes [256,64]` with
  `from_steps [1,262144]` is 256 bytes in each of 64 partitions = a 64x128 bf16 tile).
* Compute addresses: marker byte (bits 56-63 of the immediate) `0x00` = immediate, and
  SBUF address = `partition * 0x40000 + offset`, `num_active_channels` partitions.
* `isa.Instruction` has `set` / `encode` / `as_dict` / `fields` (no `pack`, no `raw`).
* Capture noise on an identical binary is still ~1 in 6 captures landing high; the five shipped
  captures spanned 18 us, so a median of 5 plus "every capture below every baseline capture" is
  the right standard. 0.5% is ~70 us at this span.

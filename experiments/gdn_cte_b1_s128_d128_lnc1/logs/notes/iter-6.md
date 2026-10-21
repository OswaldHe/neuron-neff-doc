# iter-6 — **the metric's window was identified, the tail's whole budget was itemised, and five
# of the six candidates left by the iter-5 review were refuted on the device.** Shipped a 3-edit
# package worth ~0.06 us. The remaining 38.1 us is **2.6 us of fixed DMA latency in the tail plus
# a two-branch diamond that is balanced to within 0.16 us** — scheduling has essentially run out.

## Result: anchor **38.142** -> shipped **38.113** (medians of **11** 7-capture runs each; means
## 38.153 vs 38.104). On the **order-balanced subset only** (two ABBA blocks, §8) it is
## **38.153 -> 38.084, −0.069**, which matches the 0.06 the landmark table predicts.
## **Quote −0.03 to −0.07 us (0.1-0.2%), not more.** `verify.py --runs 2/3` PASS at every step,
## `max_abs_err` identical to the baseline to every printed digit (output_0 6.119e-05 = 17%,
## output_1 2.864e-04 = 82%), all runs bit-identical.

| step | edit | verdict |
|---|---|---|
| — | iter-5's bytes (anchor), 11 separate 7-capture runs | 38.011 / 38.063 / 38.066 / 38.091 / 38.112 / **38.142** / 38.149 / 38.193 / 38.231 / 38.269 / 38.350 |
| — | shipped, 11 separate 7-capture runs | 37.894 / 37.961 / 38.050 / 38.076 / 38.091 / **38.113** / 38.148 / 38.192 / 38.194 / 38.203 / 38.226 |
| **E2** | ACT: rotate the update-less `TENSOR_SCALAR` (slot 77, dst p0+0xe000) behind the SOFTPLUS + stand-in | **KEPT**, blk20's trigger −0.047 (§3) |
| **E3** | PE: swap the waits of the fence `ES` (slot 141) and the `LDWEIGHTS` behind it (slot 142) | **KEPT**, blk19's branch −0.063 — *but blk19 does not bind, so worth 0 today* |
| **E5** | sem-16 output-store DGE row: Pool slot 20 → Activation (new slot 81) | **KEPT**, ~−0.02; the Q1/Q2-contention theory it was built on is **refuted** (§4) |
| E1 | DVE: rotate the wait-less `TENSOR_TENSOR` (slot 94) behind slots 95/96, renumber S[6]#61-63 | **0.000 — REFUTED** (slot 94 was never the blocker; slot 95 is, and it is gate-bound) |
| E4 | sem-17 store row `sem_increment` 16 → 8, exit fence `S[17]>=16` → `>=8` | **REFUTED**: row 667 → **619 ns** only (≈6 ns per increment), packets ended 0.10 *later* (§5) |
| E6a | the 192 ns Vector `CAST` (DVE slot 95) → **Activation** + fresh `S[62]` for blk19's trigger | **REFUTED**: blk20 −0.099 but blk19 **+0.206** → S[17] row +0.004 (§6) |
| E6b | same `CAST` → **Pool** instead | **REFUTED**: Pool *does* run it, bit-exact, but in **372 ns**; blk19 +0.250 → S[17] row +0.063 (§6) |

Shipped: `neff_diff` vs iter-5 — `sg00/{Activation0.bin 82→83 slots, PE0.bin, Pool0.bin}`.
md5 of the shipped kernel is in `.autohelix/observations/iter-6/`.

---

## 1. THE FINDING THAT CHANGES HOW TO READ THE METRIC

`total_exec_time` (the metric) **ends at the last DMA *packet* of the sem-17 output store, not at
`S[17]#16` and not at the epilogue.** On the anchor capture: last packet 38.340, `S[17]#16` 38.702,
last instruction 40.501. The ~1.8 us `DRAIN` / `$S[2]` daisy-chain-barrier / `EVENT_SEMAPHORE_
RANGE_CLEAR` epilogue after `S[17]` is therefore **outside the metric — do not spend time on it**
(I was about to: reordering the barrier so Pool's two steps come last, re-balancing the
`RANGE_CLEAR` ranges off the slow PE, etc. All worthless.)

**The exact formula, confirmed to 1 ns on two independent captures of two different NEFFs:**

```
total_exec_time = max(DmaPacket.end_ts) − (SET_ORDERING_MODE.start_ts + 79 ns)
   capture A: 50787 − (12447 + 78) = 38262 ns  (profiler printed 38.262)
   capture B: 49814 − (11918 + 79) = 37817 ns  (profiler printed 37.817)
```
i.e. the window is **[the first DGE row's issue on Sync (t = +0.079 in `SP0.bin`'s second body
slot), the last DMA packet of the sem-17 output store]**. `last_packet − total_exec_time` is *not*
a constant (12525 vs 11997) — it is the drifting prologue; the *relative* offset is constant.
Two consequences: (1) the epilogue is free, see above; (2) **the window's left edge is the first
DMA, so delaying the kernel's start would shrink the metric without making anything faster** —
do not do that, and do not be fooled if a head-region edit reads as a win.

* `SemaphoreUpdate` timestamps run **~0.36 us later** than the `DmaPacket` timestamps of the same
  increment. Compare like with like.
* **Normalise every relative delta on an upstream landmark** — `S[6]#60` is the right one for the
  tail; it read 33.150 / 33.047 / 33.080 / 32.920 / 32.823 over five captures of five different
  NEFFs, i.e. **0.23 us of capture-to-capture drift on an op nothing touched.** Gate-relative
  starts (`start − own gate`) are stable to ±0.005 and are what the §3/§6 tables are built from.

## 2. The DMA facts, measured (these decide everything in the tail)

**A queued block's cost from trigger to its 16th increment is `0.72 + transfer + 0.30..0.47`.**
`trigger → first packet = 0.716..0.729` for every block, 256 B or 16 kB (iter-5 §1 said 0.72; confirmed).

**The 16 increments track transfer *progress*, they are not a burst at completion.** blk3
(1.1 us transfer, 512 packets) lands increments #1..#12 *during* the transfer (10.32-10.53, transfer
9.94-11.04) and #13..#16 after. For the small tail blocks the increment latency (0.3-0.5 us) exceeds
the transfer duration (0.2 us) so all 16 *appear* after the end — **that is a coincidence of size,
not a guarantee.** So **lowering a consumer's threshold from `>=32` to `>=17` is a real race**
(worth 0.15-0.19 us per hop; margin only 0.1-0.3 us). I did not take it. Recorded so the next
iteration does not have to re-derive it.

**A DGE row costs `0.60..0.76 us` of sequencer time, then `+1.03..1.93` to the first increment, and
the row's *end* gates its packets** (sem-17: row 36.640-37.307, first packet 37.958 = row end
+0.65; row *start* +1.32). Row duration is **independent of payload AND of `sem_increment`** (E4:
16→8 bought 48 ns). It is ~600 ns of fixed cost. **Review candidate #4 is dead.**

**A DGE row's hardware queue is chosen by its host engine**, not by the transfer: the sem-16 store
ran on `Q1 (software_dynamic)` from Pool and on `Q0 (hardware_dynamic)` from Activation; the sem-17
store on `Q2 (hardware_dynamic)` from Sync. All three use 16 channels `E64..E79`.

## 3. The tail's full budget (shipped bytes, relative to `S[6]#60` = 0)

```
0.000  S[6]#60  (DVE slot 93 TT, dst p64+0x4008)
       |                                               \
       | ACT slot 76 ACTIVATE 438ns -> S[4]#37          | Pool slot 19 AFFINE_SELECT 235ns -> S[3]#7
0.472  | DVE slot 95 TT 220ns  (BLOCKED 0.079 by        0.329 DVE slot 94... no: slot 95 is the CAST
       |   the CAST in front of it) -> S[6]#62          0.521 DVE CAST 192ns p64+0x5100->p64+0x37900
0.700  | ACT slot 77 SOFTPLUS 254ns -> S[4]#39          0.548 blk19 TRIGGER
1.083  | blk20 TRIGGER (gate-bound +0.026)              1.934 blk19 LANDS (1.386 = .72+.21+.46)
2.362  | blk20 LANDS   (1.279 = .72+.22..32+.30)        1.976 PE 141/142/143 LDW+MATMUL -> S[5]#63
2.385  | ACT slot 81 TENSOR_SCALAR 390ns -> S[4]#40     2.434 DVE slot 97 TS 283ns -> S[61]
2.830  | PE 144/145 LDW+MATMUL 290ns -> S[5]#64         2.776 DVE slot 98 TT 285ns -> S[6]#64
3.100  +------------------ max(3.100, 3.061) -----------+ 3.061
3.140  DVE slot 99 TT 292ns -> S[6]#65            (the two branches are 0.16 apart)
3.477  SP slot 40 DGE row 667ns  (gate-bound +0.017)
4.144  row end -> +0.65 first packet -> 0.38 packet span
5.145  LAST PACKET = THE METRIC'S END
```
Itemised: **2.63 us of fixed DMA latency** (blk19 or blk20 1.28-1.39, output store 0.65+0.38, plus
the 0.72 inside the block), **1.88 us of compute** (438+220+254+390+290+292 ns, every op 64-128
elements, all fixed-overhead-dominated), **0.67 us of DGE row**, **~0.3 us of gates/dispatch**, and
only **0.079 us of engine-scheduling fat** (the one marked BLOCKED). Everything else is `+0.02..0.05`
after its gate.

> **The tail is a diamond and both arms arrive within 0.16 us.** An edit that speeds one arm by
> more than its margin buys nothing. That is why E3 (−0.063 on blk19) and E1 are worth 0, and why
> E6 — which trades 0.21 off one arm for 0.10 on the other — is a *loss*. **Compute the margin
> first; it is `blk20_branch_end − blk19_branch_end`, 0.16 us on the shipped bytes.**

## 4. E5 and the refuted Q1/Q2-contention theory

The sem-17 store's 16 channels start **staggered over 0.145 us** (E64 first, E79 last) and each
channel then runs 8 packets over ~0.21 us; the last channel's last packet is the metric's end.
On the anchor the stagger order matched exactly the order in which the **sem-16** store's channels
freed up (per-channel gap +0.07..+0.15), which looked like per-channel serialisation worth 0.145.
**It is not:** with the sem-16 store moved to Activation (E5) its packets finish at 37.474, a clear
0.24 us before the sem-17 store's first packet at 37.716 — and **the 0.145 stagger is unchanged**
(37.716 → 37.861). The stagger is intrinsic to the queue's descriptor dispatch. E5 is still kept
(it costs nothing, Activation reacts to a DGE row in +0.02 vs Pool's +0.42, and it takes the sem-16
store off the metric's critical resource) but it is worth ~0.02, not 0.145.

Also measured and now closed: the two stores do **not** contend for bandwidth — the sem-17 store
moved 40 kB at 163 GB/s while the sem-16 store was active and only 135 GB/s after it finished.

## 5. E4 — `sem_increment` does not buy the row back

`sem_increment` is a real field of `PSEUDO_DMA_DIRECT2D` (`v3.NEURON_ISA_TPB_PSEUDO_DMA_DIRECT2D_
STRUCT`), independent of `src_num_elem`/`dst_num_elem`. Setting it to 8 (and the exit fence
`Pool[22]` to `S[17]>=8`) **verified PASS, all 128 data packets still transferred**, 8 increments
instead of 16 — and the row only shrank 667 → 619 ns. The row's cost is fixed, not per-descriptor.
`Pool[22]`'s second guard is `events_extended_wait_idx/events_extended_sem_wait_value` (note: the
*wait* value field is `events_extended_sem_wait_value`, not `events_extended_semaphore_value`).

## 6. E6 — and the engine-capability finding that outlives it

The 192 ns Vector `CAST` (DVE slot 95, `p64+0x5100 → p64+0x37900`, blk19's source) is the only op
in the tail that blocks another: it costs the `TENSOR_TENSOR` behind it 0.079 us, and that TT gates
blk20's trigger. Moving it off Vector needs a fresh semaphore for blk19's trigger (the S[6] *order*
would otherwise depend on which engine wins a 40 ns race) — done with `S[62]`, all S[6] thresholds
above the removed increment decremented by one (SOFTPLUS 63→62, `PE[141]` 64→63, sem-16 row 65→64,
sem-17 row and the dead identity TS 66→65). Both hosts verified bit-exact. Both lose:

| host of the CAST | duration | reaction | blk19 trigger | blk20 trigger | S[17] row |
|---|---|---|---|---|---|
| **Vector** (as shipped) | **192 ns** | +0.024 | **0.548** | 1.083 | **3.477** |
| Activation (behind the 438 ns ACTIVATE) | 241 ns | +0.024 | 0.754 | **1.031** | 3.494 |
| Pool (behind the AFFINE_SELECT that writes its src) | **372 ns** | **+0.104** | 0.798 | 1.027 | 3.540 |

> **NEW: Pool/GpSimd executes `CAST` correctly and bit-exactly.** iter-5 §5 concluded
> "Activation↔Vector is the only compute-move axis" — that is **wrong for CAST**. Pool is a third
> host, at **~1.9x Vector's and ~1.5x Activation's time** for a 64-partition x 1-element cast, plus
> a **+0.10 us reaction** (vs +0.05 for its AFFINE_SELECT, +0.02 for its DGE rows). So Pool is only
> worth using for an op with >0.2 us of slack. `TENSOR_TENSOR` and `ACTIVATE` were *not* tested on
> Pool; `TENSOR_SCALAR` was not either.

## 7. Where the remaining 38.1 us is, and what is left

* **Head 0 → 13.1 us**: two serial DMA hops, structural (iter-5 §4), and iter-5's edit P proved a
  head gain does not reach the exit. Closed.
* **Body 13.1 → 32.8**: Activation 53% / Vector 51% busy; the critical path runs
  `blk18 trigger 31.13 → blk18 lands 32.47 (1.45) → PE 142/143-equivalents → S[5]#61 → DVE slot 93
  → S[6]#60`. blk18's trigger is immovable (iter-5 §1, a disguised WAR on `PE`'s read of p0+0x4780).
* **Tail 32.8 → 38.1**: §3. 2.6 us of it is DMA latency nobody can edit; 0.08 us is schedulable.
* The *only* unexplored items with a ceiling above 0.1 us:
  1. **Make blk20 unnecessary.** It is a 256-byte 1→64-partition scatter (`p0+0x5100 → p0+0x37b00`,
     `SP0.json` id 20) whose only consumer is `ACT slot 81`'s **per-partition scale pointer**
     (`imm0_imm_ptr=0x37b00`, `imm_src=PTR`). It costs **1.28 us** to move 256 bytes. If that
     `TENSOR_SCALAR` could take its scale from somewhere already laid out one-value-per-partition,
     the whole hop leaves the critical path (−1.3 us). I found no existing copy of those 64 values
     in per-partition layout (`p64+0x5100` is a *different* vector — Pool's AFFINE_SELECT output,
     not the SOFTPLUS's). A PE matmul against a one-hot/identity weight could do the scatter, but
     that is new arithmetic in PSUM and output_1 has only 18% of the error budget left.
  2. **A second DMA queue.** All 21 queued blocks use `qSPSpillReload0`, `queue_idx 3`. blk19 and
     blk20 did not actually contend in any capture I took (blk20's start is always
     `trigger+0.72`), so this is probably worth 0 — but it is the only structural DMA lever left.
  3. **Two output stores in one row.** The sem-16 and sem-17 stores are two DGE rows of ~667 ns
     each on different engines; they already overlap, so this is worth ≤0.
* **Do NOT retry** (measured here or earlier): the epilogue/barrier (§1 — outside the metric);
  splitting or re-hosting the sem-17 store (iter-3 §5, iter-4 §8, and §2/§4 here); `sem_increment`
  (§5); lowering a queued block's wait below its 16th increment (§2 — a real race); moving the
  Vector CAST (§6); rotating DVE slot 94 (E1); anything on the blk18 or blk14 triggers (iter-5
  §1/§2); the head (iter-5 §4).

## 8. Method / tools (rebuild, ~25 min; `scripts/` is not preserved)

`scripts/lib.py` (load/save a stream as 64-byte slots; `brief()` prints waits/extended waits/
updates/addresses per slot; **slot index ≠ profile `pc`** — the offset is per engine and constant
within the kernel body: ACT +63, DVE +63, PE +61, Pool +58, SP +63 on these bytes, but *derive it
by matching one operand string*, do not trust those numbers), `scripts/an.py`
(`occ|gaps|tl ENG lo hi|land S:thr|sem S|dma` over `prof/`, normalised to
`SP SET_ORDERING_MODE`; the wait regex must be `S\[(\d+)\][^\s]*\s*\([^)]*\)\s*>=\s*(\d+)` — the
profiler prints `S[6] (Vector)>=60` **without** a `$`), `scripts/cp.py` (backward critical path:
for each op, predecessor = latest of {producer of each wait, previous op on the same engine}),
`scripts/edit{1..5}.py` (each asserts every pre-image field and re-decodes after `encode()`).

* **Build the package, then A/B the package** — and *first* compute which arm of the diamond binds.
* **ABAB pairs are not good enough on this device; use ABBA blocks.** Across 6 plain A/B pairs the
  config that ran **first in the pair read ~0.10 us lower** in 5 of 6 pairs, which is the same size
  as the effect I was measuring: plain pairs read **5 of 6 against** the edit even though the
  order-balanced blocks read −0.069 **for** it. In an ABBA block (positions 1,4 = A; 2,3 = B) the
  linear part of that drift cancels exactly:
  `anchor 38.091 / v3 38.194 / v3 38.076 / anchor 38.193` → A 38.142, B 38.135;
  `anchor 38.231 / v3 37.961 / v3 38.091 / anchor 38.112` → A 38.172, B 38.026.
  **Two ABBA blocks (8 runs, ~14 min) resolved a 0.07 us difference that 12 runs of ABAB could not.**
* **Do not run a profile from a background shell while the foreground also profiles.** Both loops
  `cp` into `kernel.neff` and both drive the one device: my pair-7 reading is garbage and four
  foreground runs returned empty. One profiler at a time, always.
* A single capture's DMA *transfer duration* is unreliable: blk20's 256-byte transfer measured
  219 / 322 ns and blk18's 255 / 405 ns on identical bytes. Never read a 0.1 us chain delta off one
  capture — read the *gate-relative* start of each op instead, which is stable to ±0.005.
* Device cost: 8 verifies (0 failures), 16 profiles, no hang.
* Free semaphores after this iteration: 0, 1, 7, 18-23, 40-42, 54, 55, 63 (**61 is iter-5's,
  62 is E6's — E6 was reverted so 62 is free again**; 52/53 iter-4's, 50/51 iter-3's,
  47/48/49 iter-2's; **2 is the runtime barrier — do not touch**).

# iter-7 — **shipped: −4.6 us (1170.5 → 1165.9, three interleaved 5-capture medians each).**
# Three edits, all device-verified bit-identical:
#   1. **the win:** 32 Pool `EVENT_SEMAPHORE` waits retargeted off the now-dead q2 "cast" DMA onto
#      that DMA's own trigger condition (`S[6] >= v_g`) — deletes a ~1.4 us queued-DMA hop from
#      `DVE Wg_grp → Pool → Pool DGE loads → q0 ring throttle → W_t dma_transpose → PE LDWEIGHTS :597`;
#   2. iter-6's 32 inserted fp32→bf16 `CAST`s moved **ACT → DVE** (2× faster per element there,
#      −19.5 us of ACT busy; neutral on the wall but it replaces iter-6's cross-engine race with
#      same-engine program order, incl. the g2 cast that had **no wait at all**);
#   3. the 64→32 chunk-0 `W_t` `dma_transpose` triggers now wait on the cast's own fresh semaphore
#      **S[88]**, with the q0 ring throttle they displaced reinstated explicitly as a PE-ordinal
#      WAR wait (`S[5] >= n`) on the `EVENT_SEMAPHORE` that precedes each trigger. Neutral on the
#      wall (the trigger is SP-arrival-bound, §3.4) but it **closes the unsynchronised-producer
#      hole the iter-6 review flagged**: the DMA now provably cannot read the scratch early.
#
# ## THE HEADLINE NEGATIVE RESULT — measure this before planning anything: **the wall does not
# ## respond to engine busy time. It responds only to latency removed from the dependence chain.**
# Three device measurements, same session (§2):
#   * +19.6 us of ACT busy (32 dummy CASTs, provably semantics-free) → **+6.6 us** of wall;
#   * move those 19.6 us ACT → DVE (ACT 540→521, DVE 536→549) → **0 us**;
#   * move them later *within* ACT, out of the head-of-line slot → **0 us**.
#   Adding work costs ~1/3 of it; removing it gains nothing, because the critical path immediately
#   hands over to another engine. **Compute rebalancing is closed by measurement** (review
#   candidate #2, iter-6 §6.2) — do not spend another iteration on it.

---

## Result

| NEFF | `profile.py --repeats 5` medians (us) | samples |
|---|---|---|
| HEAD `5585a33` (= iter-6) | **1171.8, 1169.2** | 1167.3 … 1176.2 |
| **shipped** | **1165.9, 1164.2, 1167.4** | 1163.2 … 1168.8, spreads 0.23–0.42% |
| intermediate (edits 1+2 only) | 1166.5, 1165.2, 1167.2, 1167.0 | indistinguishable from shipped |
| shipped-family + q2 triggers neutralised | 1168.0, 1166.5 | equal; see §4 |

≈ **−4.6 us (−0.4%)**, measured interleaved HEAD/shipped as the iter-6 review asked; the
between-measurement floor is still ±2–3 us, so this is a small but repeatable win (shipped's worst
median 1167.4 ≤ HEAD's best 1169.2). Cumulative vs the unedited baseline (1351.4): **−13.7%**.

`verify.py --runs 3` PASS, **all 3 runs bit-identical**, `output_0` 1.097e-04 (10% of limit),
`output_1` 9.081e-04 (83%) — identical to the baseline to every printed digit. `launch_rules.py`
PASS. `semaphores.py`: no over-waits; `S[88]` now shows `max_wait 32, total_inc 32, waiters 32`.
Engine busy: DVE 549.7, ACT 520.7, PE 340.3, DMA 333.2, Pool 233.8, SP 53.1.

`neff_diff` vs HEAD: 4 members. `Activation0.bin` 2564→2532 slots (32 deletions),
`DVE0.bin` 2860→2892 (32 insertions), `Pool0.bin` 32 slots re-fielded, `SP0.bin` 60 slots
re-fielded (32 triggers + 28 guards). `SP0.json`, `PE0.bin`, `def.json` untouched.

---

## 1. EDIT 1, the win, exactly (30 lines of script; do not re-derive)

iter-6 fused the W path so `W_all` (the q2 `op:"cast"` DMA's destination) has no reader — but it
left the 32 q2 transfers running **and left Pool waiting on them**. Pool's per-group
`EVENT_SEMAPHORE` at `gdn_cte.py:494` had an `events_extended` wait `S[72+r] >= 16·pass`
= "the g-th q2 cast transfer completed". That transfer's own SP trigger waits `S[6] >= v_g`
(31 of 32) or `S[42] >= 32` (g2 only). So Pool paid a full queued-DMA hop (713 ns trigger→start
+ transfer + 330 ns semaphore ≈ 1.4 us) for data nobody reads.

**Edit:** copy the matching q2 trigger's wait into each Pool ES's *extended* wait field:

```
Pool0.bin slots  39  88 134 157 174 191 208 224 240 258 278 297 314 329 347 362
                379 397 415 436 456 471 484 502 520 533 546 559 572 585 598 612
  events_extended_wait_idx       : 72+r -> 6   (g2: 42)
  events_extended_sem_wait_value : 16·p -> v_g (g2: 32)
```
The ring↔group map is exact and self-checking: Pool slot order == q2 block order, and
`block = (value/16 − 1)·16 + (idx − 72)` reproduces the index for all 32 — assert it.
`v_g` comes from `SP0.bin`'s `qSPSpillReload2` `PSEUDO_DMATRIGGER` slots (`count_union_block_id`
is the block id; join to `SP0.json`).

**Why it is sound, statically:** the new condition is exactly what the deleted DMA's trigger
waited for, so it is satisfied *strictly earlier in time*, and Pool blocks at the same slot — no
new deadlock edge, no widened wait prefix. The only things it could break are a RAW on `W_all` or
a WAR on `Wg_grp`; §4 shows on the device that neither exists.

**Why it pays:** after the edit Pool's ~19 us per-group block moves one slot earlier, onto
`TENSOR_SCALAR_AFFINE_SELECT` at Pool slot 395 waiting `S[6] >= 1165` — a genuine DVE
dependency. The ES then runs with its wait already true, so Pool's `S[16..17] += 16` and its
`DMA_DIRECT2D` input loads issue ~1.4 us earlier, satisfying the **q0 ring throttles** earlier,
which is what releases the `W_t` `dma_transpose`, which is what `PE LDWEIGHTS :597` waits for.
(After the edit Pool's binding wait is a DVE dependency, and removing the retargeted wait
*entirely* measures the same 1166.4 us — so Pool is done as a lever.)

## 2. The three measurements that close "rebalance the engines"

All on this NEFF family, `--repeats 5`, HEAD baseline 1171.8/1169.2.

| probe | what it changes | wall |
|---|---|---|
| 32 **dummy** ACT `CAST`s (clone of the inserted cast; wait/update neutered; dst = dead scratch `p96+0x19400`) inserted right after each real one | ACT busy 540 → 557 (+19.6 us), **zero** semantic effect | **1178.4** (+6.6) |
| reposition the 32 inserted ACT casts later inside ACT, past every already-ready compute op up to the next bubble, capped by the measured DMA-read deadline | ACT busy unchanged, head-of-line delay removed | **1172.3** (= 0) |
| move the 32 casts off ACT onto DVE (§5) | ACT 540 → 521, DVE 536 → 549 | **1172.8** (= 0) |

The dummy probe is the cheapest "is this engine on the critical path" instrument available:
**clone the instruction you are about to move, neuter its events, point it at dead scratch, insert
it beside the original, profile.** It is provably semantics-free (bit-identical output), so it
isolates pure occupancy. Measured conversion here: **wall ≈ 0.34 × added engine busy and
0.0 × removed engine busy.**

## 3. The W_t chain, fully instrumented

Whole-machine idle (merge all five engines' non-ES intervals): **185 us over 1442 gaps, largest
single gap 3.8 us — there is no hot spot**, only diffuse dependency latency. Attributed to the
instruction that starts at the end of each gap:

| us | n | site |
|---|---|---|
| **30.3** | 28 | `Tensor LDWEIGHTS :597` (the W_t consumer) |
| **27.7** | 28 | `Tensor MATMUL :447` ← **nobody has decoded this one yet** |
| 14.4 | 55 | `Tensor LDWEIGHTS :609` |
| 8.5 | 26 | `Sync WRITE :350` |
| 6.5 | 55 | `Tensor LDWEIGHTS :441` |

### 3.1 The q0/q1/q2 completion semaphores increment by **16**
`S[24..55]` etc. add **16** per completed transfer, so `S[27] >= 32` means "the **2nd** transfer on
ring 27 is done": **wait value / 16 = the transfer's index in that ring's trigger order.** This is
the decoding that makes every `S[8..55]` wait in this NEFF readable; without it the values look
absurd (up to 320 on a 42-transfer ring).

### 3.2 PE's `:597` waits are therefore EXACT — there is no slack to shave
With §3.1, all 64 `LDWEIGHTS :597` waits land exactly on their own `W_t` transpose (verified for
14: `value/16` ∈ the W_t indices on that ring, every time). Do not look for over-synchronisation
there.

### 3.3 `W_t` is only double-buffered
The 64 `:597` `LDWEIGHTS` read just **9 distinct source immediates**, and two of them account for
53 of the 64 uses (`0x547e` ×28, `0x4d7e` ×25). The address immediate is the **end** of the range
(`step_elem[0] = −1`), so `0x547e` means bytes `[0x5380, 0x5480)` on the LDWEIGHTS side — use
interval overlap, not equality, when matching it to a transpose's `to_off` (`0x5400`).
Reuse distance ≈ 2 chunks ⇒ the WAR on `W_t` is structurally tight, which is why iter-6 §8.1
wrong turn 5 broke. In *time* it is loose: the previous reader of the same region finishes ~30 us
before the transpose fires.

### 3.4 After edit 1, the `W_t` trigger is **SP-arrival-bound**, not wait-bound
Before edit 1 the chunk-0 trigger's `qPoolDynamic` ring wait released it with **17 ns** to spare
(median over 32; so SP was not late and the throttle was binding). After edit 1 that wait is no
longer binding and edit 3 (retarget to `S[88]`) changes the trigger time by **nothing**
(trigger − cast_end: 1158 → 1186 ns). Decomposing that window: SP is **busy only 232 ns and
idle/blocked ~975 ns**, blocked at the `PSEUDO_DMATRIGGER` **4 slots ahead**, which waits
`S[6] >= v_g+3` — i.e. for three more DVE instructions than the cast needs. That is pure
head-of-line blocking and it is the remaining ~1.1 us per group on this chain.

### 3.5 **The obvious fix for 3.4 is REFUTED on the device — read this before trying it**
I hoisted each `(guard-ES, W_t trigger)` pair above that blocking trigger — a pure permutation of
2 slots × 32 groups, units kept intact (never splitting a wait-only ES from the instruction it
guards), with both of the pair's other guards verified satisfied **14–30 us earlier** so SP could
not block at the new position, and the deadlock argument checked (the jumped triggers wait on
`S[6] >= v_g+3`, i.e. on DVE work that happens *after* the cast, so the cast cannot depend on
them). It **FAILS: 2048 NaN in `output_0`, `output_1` bit-exact** — the same signature as iter-6
wrong turn 5 (state path fine, output path corrupted).
* Not the ring order: no jumped trigger shares our transfer's ring semaphore (checked for 5
  groups; ours 25/27/34/36, jumped 24/26/29–33/58–70).
* So firing the transpose ~1.1 us early clobbers a reader that the q0 ring throttle was
  (incidentally) ordering against, and that reader is **not** a `:597 LDWEIGHTS` — my WAR guard
  only covered those. `W_t`'s 9 destinations live in the heavily reused window partition 0
  `0x4300–0x5500`, which ACT/DVE also read.
* **To retry this, the WAR guard must cover every reader of the transpose's write region on every
  engine** (build it from all engines' decoded `src` extents, not from `:597`), and a single
  preceding ES has only one spare wait field — so you would need the max over readers to live on
  one engine, or a second inserted ES. That is the remaining ~1.1 us × 32 ≈ 35 us of DMA advance
  (≈ 4–5 us of wall at iter-6's 13% conversion). It is the best-sized item left, and it is
  **strictly harder than it looks**: two iterations have now been burnt on the unguarded version.

## 4. Device-verified: the 32 q2 `op:"cast"` transfers and `W_all` are **DEAD**
Two independent probes, both `verify.py --runs 2` PASS with bit-identical output:
1. edit 1 itself — if anything read `W_all`, or needed the WAR on `Wg_grp` that the q2 DMA's read
   provided, this reads it too early;
2. **re-encoding all 32 `qSPSpillReload2` `PSEUDO_DMATRIGGER` slots in `SP0.bin` as neutral
   `EVENT_SEMAPHORE`s** (clone any SP ES with wait NONE / update NONE), so the transfers are never
   triggered. DMA busy 326 → **311 us**, SP 53 → 50 us, wall **1168.0 / 1166.5 — equal**, so I did
   not ship it.
   This answers iter-6 §6.1(c)/§8.0(2)'s open question; the byte-exact liveness map it asked for is
   unnecessary, the device answered it. **A future iteration can take this for free** if it wants
   the DMA headroom or a cleaner NEFF. Neutralising the trigger slot (rather than deleting the
   `SP0.json` entry) avoids all block-id renumbering risk.

## 5. EDIT 2: the ACT → DVE cast move (neutral on the wall, strictly better hygiene)
* **`CAST` is not one struct.** ACT's "CAST" rows in the profiler are
  `ACTIVATE v2.NEURON_ISA_TPB_S3D3_AC_STRUCT` (header byte 0x21); DVE's are
  `CAST common.NEURON_ISA_TPB_S4D4_TR_STRUCT` (0x47). **You cannot copy an ACT cast slot into
  `DVE0.bin`.** Clone a DVE `CAST` (e.g. slot 40: 32 ch, 128 elem, FP32→BFLOAT16) and set
  `src/dst_mem_pattern_start_addr_addr_immediate`, `*_num_elem = [512,1,1,1]`,
  `*_step_elem = [1,1,1,1]`, `num_active_channels = 32`. `addr_reg_regnum/reserved0/marker` are a
  **union view of the same 4 address bytes** — set the immediate and they follow.
* **Speed:** DVE does 512 elem × 32 ch in **423 ns** (measured, max 424); ACT took 613 ns. Fits:
  ACT ≈ 185 + 0.84·n ns, DVE ≈ 165 + 0.42·n ns. **DVE is 2× faster per element for fp32→bf16.**
* **Placement (worth copying).** The ACT cast waited `S[6] >= v_g`, and `v_g` is the ordinal of a
  DVE `COPY` (the last `Wg_grp` band write) — the producer is DVE itself. Put the cast at DVE slot
  `anchor+2`, `anchor = upd[v_g − 1]` where `upd` = DVE slots with `events_update_idx == 6,
  mode SEM_INC_COMPLETE` (2093 of 2860). All 32 anchors come out `COPY` with `num_elem
  (128,1,1,1)`, spacing 87–94 — self-consistent. Give the cast **no wait**: DVE is in-order, so
  program position *is* the dependency, and a no-wait instruction can never deadlock.
* **g2's ACT cast had `events_wait_mode = NONE`** (the review's "slot 217 can read its source
  before it is written"). Its q2 trigger waits `S[42] >= 32`, not `S[6]`, so the ordinal trick does
  not apply; found its anchor by address — the last DVE compute instruction whose `dst` extent
  overlaps the cast's `src` and that ends before the ACT cast started → DVE slot 264, which lands
  exactly on the 88/94 spacing between 176 and 358. **Now correctly ordered.**
* Deleting the 32 ACT casts is free: they updated **S[88]**, not the ACT counter `S[4]`, so no ACT
  wait ordinal moves.
* Data-ready margin (`cast end → DMA start`) after everything: **min 1901 ns**, p10 1921, median
  2940 — vs HEAD's 1829 ns, so nothing was tightened.

## 6. EDIT 3: making the W_t transposes properly synchronised (shipped, neutral)
* 32 DVE casts given `events_update_mode = SEM_INC_COMPLETE` on the fresh semaphore **S[88]**
  (numeric 19; they are in group order so `S[88] == g+1`). A fresh semaphore shifts no ordinal.
* The **32 chunk-0** `W_t` triggers (source partition 96) now wait `S[88] >= g+1`. Their slot order
  is group order — assert it. Liveness is safe: SP cannot reach the trigger earlier than before, so
  no new deadlock edge; the chunk-1 triggers (source partition 112) are untouched — their PE
  consumers wait only 170–260 ns, so there is no prize there and no reason to take the risk.
* The q0 ring throttle thus displaced is reinstated as the **WAR guard**: every one of the 32
  triggers is preceded by an `EVENT_SEMAPHORE` that already carries the trigger's second wait
  operand in its *main* field and has its **`events_extended` wait free** — put `S[5] >= n` there,
  `n` = the `S[5]` ordinal of the first `MATMUL` at or after the `LDWEIGHTS :597` that last read
  that `W_t` region. 28 of 32 get a guard; 4 (g0–g3) are the first use of their address and need
  none. The `n` values come out as a clean arithmetic progression (57, 299, 361, 423, … step 62),
  which is the check that the pairing is structural and not a timing artefact.
* Neutral on the wall (§3.4) but it closes iter-6 §8.2's residual race: the transfer can no longer
  read the bf16 scratch before the cast wrote it.

## 7. Refuted / closed this iteration (each with its measurement)
* **Engine-compute rebalancing in every direction** — §2. Closed by measurement, not estimate.
* **Repositioning a compute insert inside ACT** — §2. Also: ACT/DVE/PE have **two execution units**
  (`Instruction.parquet.execution_unit` ∈ {`compute`, `sequencer`}); `EVENT_SEMAPHORE` rows run on
  the sequencer and routinely **overlap the compute instruction before them** (a 40 ns ES starting
  mid-CAST). **Filter `execution_unit == 'compute'` before reasoning about ACT/DVE ordering** — an
  inserted compute op delays only the next *compute* op, by (its duration − that op's remaining
  `evt_wait_time_ns`).
* **Shaving `LDWEIGHTS :597` wait values** — §3.2, exact once divided by 16.
* **Reordering SP to issue the W_t trigger earlier** — §3.5, NaN. Needs an all-engine WAR guard.
* **Fusing gather→scatter DMA pairs.** `qSyncSpillReload0` has 64 `transpose_shape [1,1,64,1]`
  gathers (64 partitions × 1 fp32 → 64 contiguous fp32 on partition 0) and 128 `[1,1,1,64]`
  scatters (the inverse); 32 exact region matches exist, so a pair *looks* like a fusable two-hop
  round trip whose composition is a single plain 64-partition 4-byte copy. **It is not:** the
  measured gap from the gather's `end_ts` to the scatter's `trigger` is **32–326 us** (median 104).
  The scatter's real producer is ACT/DVE compute reading and rewriting the packed form on partition
  0. This also closes iter-6 §6.3 ("replace the 146 fp32 partition gathers with a PE transpose")
  as a latency lever.
* **Deleting the dead q2 transfers** — §4: verified safe, measures the same. Optional.

## 8. Facts for the toolbox (additions to iter-4 §5 / iter-5 §5 / iter-6 §5, all still correct)
* Completion semaphores increment by **16** per transfer (§3.1). `value/16` = index in ring trigger
  order. This is the key to reading every `S[8..55]` wait in this NEFF.
* `profile.py`'s per-engine busy ≠ Σ`duration_ns` (ACT: Σ 700 us, merged intervals 598 us,
  `profile.py` 540 us) — the two execution units overlap. Compare like with like.
* `SemaphoreUpdate.parquet`: `id` is the string `"S[6] (Vector)"`; filter `sid.notna()` **before**
  `.astype(int)` (the raw column has NaN and raises).
* A Pool `EVENT_SEMAPHORE`'s wait shows up as a gap before the **preceding** slot in `start_ts`
  order as often as inside its own `duration_ns`. Attribute Pool stalls by decoding `Pool0.bin`
  around the gap, never by trusting which row the gap sits in front of.
* `PSEUDO_DMATRIGGER` slots are self-describing (`dma_queue_name` = 32-byte list,
  `count_union_block_id` = 1-element list); `SP0.json`'s `dma[*]` joins by `(queue, block id)` and
  **not** by slot index, so SP slots may be permuted, re-encoded or inserted without touching the
  JSON.
* `semaphores.py` does **not** flag a semaphore with increments and **zero waiters** — grep its
  table for `waiters 0` after any edit that removes a consumer. That is how a dead DMA hides
  (S[72..87] had been dead since iter-6).
* The `LDWEIGHTS` address immediate is the **end** of its byte range (`step_elem[0] = −1`).

## 9. What iter-8 should do, in order
1. **§3.5 with a complete WAR guard.** ~1.1 us × 32 of DMA advance ≈ 4–5 us of wall. Build the
   guard from **all** engines' `src` extents overlapping each transpose's write region, not just
   `:597`. Budget the guard, not the permutation — the permutation is 10 lines and already written.
2. **Decode `MATMUL :447`** — 27.7 us of whole-machine idle over 28 gaps, the second-largest site,
   and nothing in iters 1–7 has looked at it. Its `events_wait_idx` is mostly 0, so the gap belongs
   to the `EVENT_SEMAPHORE`/`LDWEIGHTS` before it; start by decoding those (`PE0.bin`, the pcs that
   follow the 28 largest gaps).
3. Free and verified if you want it: neutralise the 32 q2 triggers (§4), −15 us of DMA busy.
4. **Do not** spend budget on: iter-1 §5.4, iter-2 §5.5, iter-3 §5.4, iter-4 §4.5, iter-5 §2/§4.3,
   iter-6 §6.4, and §7 above. In particular all five engines' instruction reordering (iter-6 §2)
   and now all compute rebalancing (§2) are closed by measurement.
5. Honest expectation: 185 us of the 1166 us wall is whole-machine idle spread over 1442 small
   gaps, and DVE∪ACT busy is ~800 us. Without a full software-pipelining reschedule (which
   iter-6 §2 showed hangs, because it must widen wait prefixes) what is left is item 1's ~5 us plus
   whatever item 2 turns up.

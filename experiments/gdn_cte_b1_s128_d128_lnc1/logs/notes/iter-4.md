# iter-4 — one lever paid three times, 0.5 us each: **an op parked in front of the chain that feeds
# a queued trigger**. Plus the tail output store re-pointed at its true producer, and the last two
# wait-less Vector CASTs moved to Scalar. Four refutations, one of them closing a whole class of
# edit (every SB->SB DMA block on this kernel is a *transpose*, not a copy).

## Result: **40.21 → ~38.35 us (−1.85 us, −4.6%)**, `max_abs_err` identical to the baseline

| step | edit | median (7 captures) | Δ vs prev | verdict |
|---|---|---|---|---|
| — | iter-3's bytes, measured at session start | **40.210** | — | anchor |
| **A** | sem-17 output store re-pointed at `0x4710` / `S[6]>=66` (§2) | **39.922** | −0.29 | **kept** |
| B | blk19 replaced by a duplicate Scalar `CAST` (§3) | — | — | **REFUTED: verify FAIL, output_0 cosine 0.76** |
| **C** | `ACT[70]`↔`ACT[71]` so blk18's trigger is not behind a 259 ns op (§4) | **39.420** | **−0.50** | **kept** |
| **D** | `DVE[87]`↔`DVE[88]`: wait-less `MEMSET` hoisted (§5) | **38.903** | **−0.52** | **kept** |
| **E** | wait-less/update-less `ACT[68]` rotated behind `ACT[70]` (§6) | **38.799** | −0.10¹ | **kept** |
| F | `ACT[67]` (blk15's consumer) rotated behind the blk18 chain (§7) | 38.482 | (−0.32) | **REFUTED: verify FAIL, output_0 3.3e4** |
| G | F with blk18's trigger left on `S[4]>=35` (§7) | — | — | **REFUTED: byte-identical failure** |
| H | the minimal half of F: `ACT[67]`↔`ACT[68]` only (§7) | — | — | **REFUTED: byte-identical failure** |
| **I** | `DVE[43]` (wait-less `CAST`) executed on idle Scalar (§12) | — | (−0.0, unresolvable alone) | **kept** |
| **I+J** | `DVE[59]` too (§12) | **38.333 / 38.353 / 38.227** | **−0.41** | **kept (SHIPPED)** |

**Interleaved A/B, alternating which config ran first, 7 captures each, back to back:**
* iter-3's bytes **40.104 / 40.338** vs edit E **38.567 / 38.707** → **−1.54 / −1.63**;
* edit E **38.847 / 38.714 / 38.576** vs the shipped bytes **38.333 / 38.353 / 38.227** →
  **−0.51 / −0.36 / −0.35** (every shipped capture below every E capture in the same half hour).
Session drift was ≤0.2 us (iter-3's bytes read 40.210 early and 40.10/40.34 late).
**Quote 40.21 → 38.33, −1.88 us (−4.7%).** A 7th independent run of the shipped bytes read 38.402
(min 38.197, max 38.927). Against iter-0: 48.519 → ~38.35, **−21% (1.27x)**.

¹ E's *chain* shortened by the designed 0.30 us (landmark table, §6) but blk15's DMA flight happened
to land 0.33 us later in that capture, so the metric only showed −0.10. Keep it: the landmarks say
it did what it was built to do, and it never regressed.

`verify.py --runs 2/3` PASS after every kept step, all runs bit-identical, `max_abs_err`
**identical to the baseline to every printed digit** (output_0 6.119e-05 = 17% of the limit,
output_1 2.864e-04 = 82%). `launch_rules.py` PASS. `semaphores.py` clean at every step: `S[4]`
41 inc / max_wait **40** (edit A dropped the only `S[4]>=41` waiter), `S[5]` 64/64, `S[6]` 66/66.
`neff_diff` vs iter-3: **3 of 34 members**, `sg00/{Activation0.bin 80→82 slots, DVE0.bin 6 slots,
SP0.bin 2 slots}`.
sha256 `b116c199ab9fdffc3bd3a74803662abbc58951aacd80cf6a75fc194440797155`
(edit E alone, if ever needed: `bfd0da20966bdf77b8ec0af4ba2f968bcd9e1910ab80d61dd439792657733201`).
Engine busy on the shipped bytes: **Scalar 20.60 (53.6%)**, Vector 19.64 (51.1), DMA 15.27,
PE 11.95, Pool 9.78, Sync 8.87. All-engine idle **4.93 us in 5 gaps** (was 5.62), span 38.97
(was 40.60). **Scalar is now the busiest engine by 0.96 us** — edits I/J moved 0.44 us of CAST onto
it (a Scalar CAST of this shape is 295 ns where Vector takes 219), so the Vector→Scalar direction
should not be pushed further without re-reading `occ`.

---

## 0. The lever that paid, and how to find the next one in 10 minutes

> **For every `PSEUDO_DMATRIGGER`, walk *backwards* from its wait through the producing chain and
> print `slack = start − gate` for every op on the way. The first op with slack > 0.2 is stalled by
> whatever sits in front of it on *its own engine*; if that blocker is not an input of the chain,
> move it behind. Each one of these was worth 0.5 us on this kernel.**

Why it is worth so much here and not elsewhere: `blk18 → blk19 → (blk20) → the output store` is a
**serial train of queued DMA hops at the end of the kernel**, and every hop costs a flat
**0.73 us trigger→start + 0.2–0.4 transfer + 0.47–0.51 propagation ≈ 1.45 us** (measured on all 21
blocks, `an.py dma`). Nothing overlaps them, so *any* nanosecond taken off the chain in front of
blk18's trigger comes straight off the total. C and D are the same edit at two different links.

`an.py dma` (`DmaPacketAggregated`: `trigger`, `start_ts`, `end_ts`, `semaphore_id`) is the single
most useful table for this: it splits a hop into issue / transfer / propagation and the 0.73 us
trigger→start is *flat and payload-independent* (256 B and 16 kB alike).

## 1. Rebuild these (~20 min, `scripts/` is not preserved)

`scripts/lib.py` (load/save a stream as 64-byte slots, `es()`, `nop()`), `scripts/d2.py`
(per-slot waits/updates/addresses), `scripts/scan.py` (regex an address over `operands`),
`scripts/an.py` (`gaps | occ | slack ENG | tl lo hi | land S:thr | dma`, all normalised to
`t0 = SP[0] SET_ORDERING_MODE`).

Traps, in addition to the five in iter-3 §1 (all still true):
* **`ins["field"]` returns the RAW value; `ins.as_dict()["field"]` returns the symbolic name.**
  `ins["events_wait_mode"] == "WAIT_FOR_SEM_GE_IMM"` is always False (it is `5`). Assert on
  `as_dict()`.
* **`events_extended_semaphore_value` is absent on some `EVENT_SEMAPHORE` slots** even when
  `events_extended_wait_mode` is present — guard the key before formatting.
* **A DGE row's source offset has five alias field names.** Setting
  `src_start_addr_addr_immediate` to `0x10000000000<off>` updates `table_offset`, `var_offset`,
  `tbl_offs_offset_immediate`, `reg_reg_hi`… all at once and keeps the `0x10` marker. `neff_diff`
  prints every alias for the same bytes; that is not five changes, it is one.
* **The profiler's `start_ts` origin drifts between captures** (`Vector[99]` was at 50.16 in one
  profile and 50.89 in the next with a 0.3 us *faster* metric). Never compare absolute profiler
  times across two profiles — normalise on `SP[0]`, and compare **landmark deltas**.

## 2. Edit A — the output store reads the producer's buffer, not the ×1.0 copy: **−0.29 us**

`ACT[79]` is `dst fp32@0x4c00 = fp32@0x4710 × 1.0` (128 ch × 128 elem, `imm0_imm_arith_fp32 = 1.0`,
FP32→FP32 — an exact copy), and `SP[40]`, the sem-17 output-store DGE row, waited `S[4]>=41`
(= `ACT[79]`) and read `0x4c00`. Re-pointed the row at **`0x4710`** and its wait at **`S[6]>=66`**
(= `DVE[99]`, the real producer of `0x4710`). **4 bytes changed in one slot.**
`ACT[79]` is left in place so `S[4]` stays 41/41; it now runs in parallel with the row and is dead.

Checks: `scan.py` says `0x4710` is touched by exactly `DVE[99]` (write) and `ACT[79]` (read), and
`0x4c00` by `PE[30] MATMUL` at 26.3 (blk5's data, long gone) and `ACT[79]`. No DMA block writes
into `0x4710..0x490f`. Device: row start → gate slack **0.357 → 0.027 us**, `S[17]` −0.33.

> Generalise: **any `×1.0` / same-dtype copy whose only consumer is a DGE row or a trigger is a free
> 0.3 us** — re-point the consumer at the copy's source and leave the copy in place for the
> semaphore count. Check `imm0_imm_arith_fp32` and both dtypes in the slot, not the profiler text.

## 3. Edit B — REFUTED, and it closes a whole class: **every SB→SB DMA block in this NEFF is a transpose**

blk19 (`p64..127+0x37900 → p0..63+0x4100`, 128 B × 64 partitions, bf16) looked like a pure
partition shift of `DVE[95]`'s output, and **cross-partition compute writes exist all over this
NEFF** (`ACT[2]` p64→p0 ch=64, `ACT[23]` p96→p0 ch=32, `DVE[33]` p96→p32 ch=32), so I duplicated
`DVE[95]`'s `CAST` on an idle Scalar hole with `dst = p0+0x4100` and re-gated blk19's only waiter
(`PE[141]`, the `ES(S[27]>=32)` in front of `PE[142] LDWEIGHTS 0x417e`) on the duplicate. blk19 was
left running, which is harmless if it writes the same bytes.
**It does not. `SP0.json` descriptor field `transpose_shape` = `[1,1,64,64]` with
`transpose_op = "WZXY"` and `transpose_element_size = 2`: blk19 is a 64×64 bf16 TRANSPOSE.**
verify FAIL, `output_0` cosine 0.7586, deterministic over 3 runs.

Surveyed all 26 descriptors afterwards (one line of python over `SP0.json`):
* `transpose_shape` with **two** entries > 1 (`[1,1,64,64]`, `[1,1,64,128]`, `[1,1,32,32]`) = a real
  transpose: blk1, blk2, blk4–blk13, blk15-d0, blk16, blk17, blk18-d1, blk19.
* `transpose_shape = [1,1,1,64]` or `[1,1,64,1]` = a **1↔64 partition scatter/gather** (256 B in one
  partition ↔ 4 B in each of 64): blk0, blk14, blk15-d1, blk18-d0, blk20.
* blk3 has **no** `transpose_*` key at all (it is the var8 reload) — the only plain copy, and it is
  a 1.0 us scatter nobody can shorten.
**So there is no DMA hop on this kernel that a compute instruction can replace. Do not retry this.**
Also note `from_sizes`/`to_sizes` in `SP0.json` are **BYTES, not elements** (blk20 reads 256 B =
64 fp32 from `p0+0x5100`, which `ACT[76]` writes as 64 fp32) — I first mis-sized blk19 by 2×.

## 4. Edit C — **blk18's trigger was behind a 259 ns op: −0.50 us**

```
ACT[69] (488, gate S[6]>=55 @31.27, gate-bound) -> S[4]#34 @31.617
ACT[70] TENSOR_SCALAR  (259, gate S[5]>=59 @31.900)   31.923-32.182  -> S[4]#35   (writes p0+0x5280)
ACT[71] ACTIVATE SOFTPLUS (305, gate S[4]>=34 @31.795) 32.206-32.511 -> S[4]#36   (writes p96+0x4400)
SP[35] blk18 trigger  waits S[4]>=36, gate-bound, fires 32.537 -> blk18 @34.103 -> the whole tail
```
`ACT[71]`'s gate landed 0.41 us before it ran. Swapped the two slots, which renumbers `S[4]#35/#36`,
so the **two** waiters in `[33,37]` traded thresholds: `SP[35]` (needs `ACT[71]`'s `p96+0x4400`)
`>=36 → >=35`, `DVE[91]` (needs `ACT[70]`'s `p0+0x5280`) `>=35 → >=36`. Addresses disjoint both ways.
Device, exactly as designed: `S[4]#35` −0.24, **`S[26]#32` (blk18) 34.103 → 33.411 (−0.69)**,
`S[27]#32` (blk19) −0.62, `S[17]` −0.61, metric −0.50, and the 0.50 us all-engine gap at 33.6 is gone.

**iter-3 §5 dismissed this exact swap as "worth only 0.10 us".** That was wrong: it priced the
*second* op's end (−0.13) instead of the *first* op's end, which is what the trigger waits on, and
the swap also absorbs a 0.14 us Scalar idle gap in front. **Re-derive a "closed, too small" claim
from the landmark it actually gates before believing it.**

## 5. Edit D — **a wait-less `MEMSET` parked in front of a 284 ns op: −0.52 us**

The same chain one link further back:
```
S[5]#58 @30.530 -> DVE[87] TENSOR_TENSOR (284) -> DVE[88] MEMSET 1.0 -> p96+0x4180 (165, NO WAIT)
  -> S[6]#55 @31.095 -> ACT[69] ACTIVATE COPY p96+0x4180 -> p96+0x4b00 (488) -> S[4]#34 -> ...
```
`DVE[88]` has **no wait at all** and `ACT[69]` (its only consumer) is the next link, so the MEMSET
was 165 ns of pure latency sitting behind an op it does not depend on. Swapped `DVE[87]`/`DVE[88]`;
renumbers `S[6]#54/#55`, and again exactly two waiters trade: `ACT[69]` `>=55 → >=54`,
`DVE[89]` (needs `DVE[87]`'s `p64+0x4800`) `>=54 → >=55` (the latter is redundant — `DVE[89]` is
behind `DVE[87]` in Vector program order either way — but keep the semantics exact).
`p96+0x4180..0x437f` is touched by **nothing but** the MEMSET and `ACT[69]`; `ACT[69]`'s other input
is a `scale_src = POINTER_IMMEDIATE` at `p0+0x51fc`, which is blk0's output from ~6 us and is never
written by an instruction. Device: `S[6]#54` landed **29.587** (the MEMSET ran as soon as Vector was
free, 1.5 us early), `S[26]#32` −0.52, `S[27]#32` −0.52, `S[17]` −0.53, metric −0.52.

> **`events_wait_mode == NONE` on a compute slot is a flag, not a detail.** Such a slot is held only
> by its engine's program order, so it can be hoisted over any neighbour it does not alias — and if
> its consumer is on the critical chain, the hoist is worth the neighbour's whole duration.

## 6. Edit E — the one wait-less *and* update-less Scalar slot, rotated out of the chain: −0.30 on the chain

After D, `ACT[69]` had **0.91 us of slack** (gate 29.587, ran 30.497) because `ACT[67]`
(gate-bound on blk15's `S[39]`) and then `ACT[68]` were in front. `ACT[68]` is
`ACTIVATE COPY p64+0x5400 → p64+0x4000` (1 elem × 64 ch) with **no wait and no update**, and its
consumers (`ACT[74]`'s `imm_ptr=0x1004000`, `Pool[19]`) are ~3 us later. Rotated it behind
`ACT[70]`: **numbering-neutral, because it updates nothing.** Chain `S[39] → S[4]#35` shortened
1.575 → 1.272 us (−0.30) as designed; metric only −0.10 because blk15's own flight drifted +0.33 in
that capture. 10 slots of `Activation0.bin` later, `ACT[70]` is the filler.

## 7. Edit F — REFUTED, and the cause is NOT identified. **This is the first thing to debug in iter-5.**

Same lever one step further: `ACT[67]` (`TENSOR_SCALAR bf16 p0+0x37d00 × ptr p64+0x4500 →
p0+0x4880`, 388 ns, gate `S[39]>=16`) is **not an input of the blk18 chain**, it is merely in front
of it, and its own consumer `PE[133] LDWEIGHTS 0x497e` had 0.85 us of slack and is PE-sequencer
bound. Rotating `67,68,69 → 68,69,67` renumbers `S[4]#33/#34/#35`; the three waiters were retargeted
`ACT[68]`'s own wait `>=34→33`, `SP[35]` `>=35→34`, `PE[133]` `>=33→35`, and those four (with
`DVE[91]` at `>=36`) are the only `S[4]` waiters in `[32,38]`.
**Metric 38.482 (−0.32) and verify FAILED: `output_0 max_abs_err 3.3e+04`, cosine 0.0045**
(`output_1` stayed exactly at its baseline 2.864e-04, so only the sem-16 store, `Pool[20]`
`src = p64+0x4000`, is corrupted).

What I checked and ruled out before giving up on it (so iter-5 does not redo it):
* the `S[4]` renumbering is right — in `profF`, `PE[133]`'s gate `S[4]>=35` lands 30.849 and
  `ACT[67]` (now third) ends 30.840, i.e. the threshold resolves to exactly the intended op;
* `p0+0x37d00` (`ACT[67]`'s source) is **read-only** after the head gate load (`ACT[8]` and
  `ACT[67]` are its only two touchers), so moving the read later is safe;
* `p96+0x4b00` is touched only by the two ops that moved earlier; `p0+0x4880` only by `ACT[67]`
  and `DVE[92]` (which still runs 1.1 us later), and `PE[133]` still reads it in between;
* blk18's trigger moved 31.376 → 30.452, so its transfer ran **31.170–31.436** instead of
  32.106–32.390. The destinations are `p*+0x4600..0x4603` (d0) and `p*+0x4700..0x477f` (d1), and
  I could not find any reader of either in 31.17–31.44: `PE[131]` reads `0x4500..0x457f`,
  `PE[132]`/`PE[134]` read `0x4780..0x487f`, `PE[133]` `0x4880..0x497f`, `DVE[89]` writes
  `0x4780..0x487f`. `PE[128] LDWEIGHTS 0x467e` (the one real overlap, `0x4600..0x467f`) finished at
  30.285, 0.89 us before the transfer.
* lead (a) is **ruled out**: blk18-d1's source `p0+0x37c00` *is* written, by the **sem-13 DGE row
  `Pool[11]`** (`dst 0x37b00`, `dst_elem 0x100`, `dst_num [2,64]`, `dst_step [256,262144]` → it
  covers `p0..63 + 0x37b00..0x37cff`), which lands at ~6.6 us. Same trick explains blk1/blk2's
  sources: `Pool[7]` (sem 9) covers `p0+0x37d00..0x37eff`. **A DGE row's `dst_num[0] × dst_elem`
  spans more than its printed offset — always expand the pattern before concluding "nobody writes
  this".**

### 7.1 Two more variants, both REFUTED with the **byte-identical** error — this is a deterministic
### ordering bug, not a DMA race, and I could not localise it

* **G** = F with `SP[35]` left on `S[4]>=35` (one byte different): the blk18 trigger then still waits
  for all three Scalar ops, firing at ~30.87 instead of 30.45. `max_abs_err` **3.302e+04,
  cosine 0.004481 — identical to F to every digit.** So **the failure is not the DMA firing early.**
* **H** = the minimal half: swap `ACT[67]`↔`ACT[68]` only (order `68, 67, 69`), `ACT[69]` 34→33,
  `PE[133]` 33→34, trigger untouched. `ACT[69]` (which writes blk18-d0's source `p96+0x4400`) stays
  third. **Identical failure again.** So it is not `ACT[69]` running early either.
* What all three share: **`ACT[68]` (`ACTIVATE COPY p96+0x4180 → p96+0x4b00`) runs ~0.93 us earlier,
  ahead of `ACT[67]`, and `ACT[67]` (`TENSOR_SCALAR bf16 p0+0x37d00 × ptr p64+0x4500 → p0+0x4880`)
  runs ~0.4 us later.** Only `output_0` (= `var5`, the two Pool stores) breaks; `output_1` (`var6`,
  the `SP[40]` store) stays exactly at its baseline error.
* Checked and ruled out: the `S[4]` renumbering (verified against the profile — `#33`/`#34`/`#35`
  resolve to exactly the intended ops in every variant, and in H both retargeted thresholds are
  *equivalent-or-stronger* than the originals); `p0+0x37d00` is read-only after the head gate load;
  a **slot-level** scan of every `*_addr_immediate` / `*_imm_ptr` field in all five streams found no
  reader of `p96+0x4b00..0x4cff` other than `ACT[69]` (and `ACT[57]`, which finishes at 25.6) and no
  reader of `p0+0x4880..0x497f` between `PE[110]` (23.7) and `PE[133]`.
* **Leads for iter-5, in order:**
  1. **`ACT[68]`'s second input is the `scale_src = POINTER_IMMEDIATE` at `0x51fc`, and I never
     established which partition that offset resolves against.** `ACT[68]`'s src/dst are p96 and
     `ACT[75]` carries the *same* `scale_value_imm_ptr = 0x51fc` with src p32 — so either the
     pointer is absolute `p0+0x51fc` (blk0-d1's output, written once at ~6 us, which is what I
     assumed) **or it is partition-relative**, in which case `ACT[68]` reads `p96+0x51fc` and
     `ACT[75]` reads `p32+0x51fc` — and `p32+0x5100..0x51ff` is **rewritten by `DVE[84] MEMSET` at
     28.02**, i.e. right in the window the reorder crosses. **Settle how an `ACTIVATE` pointer
     operand resolves its partition; that is the one unexamined input left.**
     (*Dead end, already checked:* `imm_imm_ptr = 0` is **0x0 on all 18 `ACTIVATE`s** in the stream,
     including the 10 with `imm_src = POINTER_IMMEDIATE`, and nothing in any stream or descriptor
     writes SBUF offset 0 — it is an unused field, not a hidden input.)
  2. `ACT[67]`'s `imm0_imm_ptr = p64+0x4500` is **blk15-d1's destination** (4 B × p64..127), and
     `ACT[69]` writes `p96+0x4400..0x45ff`, which **overlaps `p96+0x4500`**. So `ACT[67]` must read
     before `ACT[69]` writes — true in E and in H, so it is not H's bug, but **any reorder of these
     three must keep `ACT[67]` ahead of `ACT[69]`**, which rules out F/G's order permanently.
  3. The prize, and it is bigger than F measured, because **two further edits are sitting behind the
     same wall** (both worked out below in §8 as "worth 0 today"): the whole tail is
     `blk15 @29.85 → ACT[67] (388) → ACT[68] (488) → ACT[69] (306) → S[4]#35 → blk18 trigger`,
     1.5 us of serial Scalar work after blk15 lands, and `ACT[68]`'s own gate lands **0.5 us before
     ACT[67] even starts**. Fix the F/G/H dependency and edits K and L below become worth ~0.5 us
     each on top of F's own 0.3.

## 8. Closed this iteration (do not re-spend budget)

* **Replacing any DMA block with a compute copy (§3).** All of them are transposes or 1↔64
  partition scatters. One python line over `SP0.json` `transpose_shape` proves it.
* **Splitting / re-hosting the sem-17 output store.** Measured again here: the row is 667 ns and the
  flight `row end → S[17]#16` is **1.39 us**; the sem-16 store (half the payload, on Pool) has the
  same 1.39. Payload-independent, so two halves on two queues buy ≲0.1 (iter-3 §5 agrees).
  `Pool[20]`'s sem-16 store is **not** the laggard — `S[16]` lands 0.48 us before `S[17]`.
* **The `ACT[65]`/`ACT[66]` swap in front of `S[4]#32` / blk15** — iter-3 §5's race, re-derived and
  still a race: `ACT[65]` writes `p96+0x4700` = blk15-d1's source, and splitting blk15 does not help
  because the gap's consumer `ACT[67]` needs **d1** (`imm_ptr = p64+0x4500`), the half that needs
  `ACT[65]`. iter-3 §7 candidate 2 is therefore worth ~0 for `S[39]`, not 0.38.
* **Rotating `ACT[73]` (update-less, 1.37 us slack) and swapping `DVE[96]`/`DVE[97]`** to pull
  blk20's trigger earlier: worked out on paper to −0.18 on blk20, but **blk19's branch binds the
  tail by 0.1–0.6 us at every step**, so the gain is ~0. Only worth doing *after* something makes
  blk19 land earlier.
* iter-3's "`ACT[68]`/`ACT[69]` swap is worth only 0.10" — **wrong, it was 0.50** (§4).
* **Edit K — swap `SP[32]`↔`SP[33]`: correct, and worth 0 *today*.** `SP[32]` is `ES(S[46]>=16)`
  (no update) and `SP[33]` is `ES(S[45]>=16) → S[60]++`; Sync program order makes `S[60]` mean
  "sem 45 **and** sem 46 landed". `SP[33]`'s own gate lands 29.13 and it runs 29.64 — **0.51 us
  stalled by the sem-46 fence in front of it** — and `S[60] → DVE[86] → DVE[87] MEMSET → S[6]#54
  → ACT[68]` is the blk18 chain. The fix is iter-2 §5.2's: swap them and give sem 46's only real
  consumer its own guard — `DVE[94]` (`src0 = p64+0x1d600`, currently wait-less) can carry
  `S[46]>=16` itself. **Measured gain today: 0, because `ACT[68]` is blocked by `ACT[67]` until
  30.50 anyway, 0.5 us after `S[6]#54` already lands.**
* **Edit L — hoist `DVE[87]` (the MEMSET, after edit D) above `DVE[86]` (`ES(S[60]>=1)`).**
  Numbering-neutral (the `ES` has no update), and the MEMSET reads nothing, so it is trivially
  safe. Would move `S[6]#54` ~0.4 us earlier. **Also worth 0 today, same reason.**
* Both K and L only pay once §7.1 is fixed. Do them *in the same iteration* as the §7.1 fix and
  A/B the package (see §11 on stacking).

## 9. Where the remaining ~38.35 us goes

Span 38.97, **4.93 us of all-engine idle in 5 gaps, every one a DMA flight**, no engine over 54%:

| gap us | at | waiting for |
|---|---|---|
| 0.55 | 10.82 | blk3 (var8 reload) — the only non-transpose block, a 1.0 us scatter; a layout change |
| 0.50 | 12.47 | blk4 `S[28]>=16` → `PE[15] MATMUL` |
| 0.44 | 28.76 | `S[46]` (a DGE row) — consumer has slack, but the gap is real idle |
| 0.58 | 34.42 | blk19 `S[27]>=32` → `PE[141]` — **the binding hop of the tail** |
| 1.38 | 37.47 | sem-17 output store: 667 ns row + **1.38 us flight** → the exit fence |

The tail, every hop gate-bound to ±25 ns:
`blk18 → PE[138/139] → S[5]#61 → DVE[88] → S[6]#60 → Pool[19] → S[3]#7 → DVE[90] CAST → S[6]#62
→ SP[37] trigger → blk19 → PE[141/142/143] → S[5]#63 → ACT[77] → (blk20 joins) → ACT[78]
→ S[4]#40 → PE[144/145] → S[5]#64 → DVE[99] → SP[40] row → 1.38 flight → Pool[22]`.
**Four queued hops × ~1.45 us + the 1.38 us store flight ≈ 7.2 us of the 38.4 is DMA latency on the
serial chain**, and §3 says none of them can be turned into compute. The ways left are
(i) shorten the compute between two hops (§0, which is what C/D/E did — the blk18→blk19 segment is
still ~1.1 us of compute for 906 ns of work), (ii) fix §7/§7.1, or (iii) find a second queue.

## 10. Ranked candidates for iter-5

1. **Finish §7.1's diagnosis — the single highest-value open question.** `ACT[68]`'s gate lands
   **1.4 us** before it runs, and three different reorders that exploit that all fail with the
   *byte-identical* error, so one unresolved dependency is hiding there. Start with "what does an
   `ACTIVATE` with `imm_src = POINTER_IMMEDIATE, imm_imm_ptr = 0` actually read?". Worth
   **−0.5 to −0.9 us** on the whole tail if it can be made safe.
2. **After §7.1: `ACT[73]` rotation + `DVE[96]`/`DVE[97]` swap** (§8), **−0.18**, but only once the
   blk19 branch stops binding.
3. **Re-run the §0 backward walk from the blk19 and blk15 triggers.** The blk18→blk19 segment
   (`PE[139] → DVE[88] → Pool[19] → DVE[90] → trigger`, 1.12 us for 906 ns of work) has four
   cross-engine hops; `Pool[19]` (`TENSOR_SCALAR_AFFINE_SELECT`, Pool-only) and `DVE[88]`
   (`TENSOR_TENSOR`, Vector-only) cannot move, but check their slack after any new edit.
4. **Vector→Scalar `CAST` moves: both remaining candidates were built and measured — see §12.**
   Do not spend budget re-deriving them; the verdict is in §12.
5. **blk3's 1.0 us scatter** (iter-1 §6.1) — the only non-transpose block, still a layout change.
6. **A second DMA queue.** All 21 transfers share `queue_idx 3`; four serial hops × 1.45 us is the
   floor while that holds. `SP0.json` blocks carry a `"queue": "qSPSpillReload0"` string — whether
   a second queue name is legal, and whether the runtime would bind it, is the one untested
   structural question with a >1 us ceiling. Hang-prone; keep a `cp save/kernel.neff` ready.
7. **The 0.50 us gap at 12.47 (blk4's `S[28]>=16` → `PE[15] MATMUL`)** is new at the top of the
   list after this iteration's tail work; it was 0.14–0.37 in earlier profiles and is worth one
   `an.py slack` pass on blk4's trigger chain.

## 11. Method notes

* **All four refutations were caught by `verify.py`, never by `semaphores.py`.** `semaphores.py` was
  clean for edit B (a transpose replaced by a copy) and for F/G/H. It only checks supply ≥ demand.
  **Run `verify.py` before `profile.py`, always** — edit F's 38.482 would have been a 0.3 us "win".
* **Save every step** (`save/kernel_{start,A..J}.neff`): each revert was one `cp` + unpack, and the
  four refutations cost ~6 minutes each.
* **The landmark table is what decides.** `an.py land 4:35 26:32 27:32 28:32 17:16` printed for two
  profiles side by side showed C and D doing *exactly* what they were designed to do (−0.52 on
  every landmark downstream) and showed E doing it too even though the metric said −0.10.
* Device cost: 11 verifies (4 failures, all caught by `verify.py`), 24 profiles, no hang.
  verify ~1 min/run, 7-capture profile ~1.5 min, +0.5 for the parquet ingest.
* **Stacking two same-sized edits before A/B-ing them is the cheap way to beat the noise floor.**
  Edit I alone was unresolvable in two pairs (−0.11, +0.08); I+J was unambiguous in two
  (−0.51, −0.36) and confirmed in a third (−0.35). At 0.2 us per edit and ~0.2 us of run-to-run
  spread, measure the pair, not the member.
* Budget shape: 20 min tools + reading iter-2/iter-3 notes, then six ~12 min cycles, then the
  interleaved A/B and the notes. **Writing the notes before the last A/B was right** — I had the
  whole iteration recorded before the clock mattered.

## 12. Edits I and J — the last two wait-less Vector `CAST`s on Scalar: **−0.41 us for the pair**

Both are the iter-3 §4 recipe, and both **verify bit-identical with `max_abs_err` unchanged**, so
the safety arguments below are device-confirmed and can be reused:

| edit | slot | the one wait that covers RAW *and* WAR | Scalar insertion |
|---|---|---|---|
| **I** | `DVE[43]` `CAST p96+0xe800 fp32 → p0+0x4300 bf16` (32 ch × 128 elem, 219 ns, no wait, no update) | **`S[6]>=18`** — RAW: `DVE[38] COPY` writes `0xe800` and is S[6]#18 (lands 16.097). WAR: the previous `p0..31+0x4300..0x43ff` was last read by `DVE[35]` (14.46, which precedes `DVE[38]` in Vector program order), `PE[27] MATMUL` and `PE[24] LDWEIGHTS 0x43fe`; `DVE[38]` waits `S[5]>=18` and **`PE[27]` is only the 13th S[5]-updating matmul by end time**, so `S[6]>=18 ⟹ S[5]>=18 ⟹ PE[27]` done. No Scalar slot anywhere reads `0x4300..0x43ff`. | index 37 (hole 16.13–17.04), update `S[52]` |
| **J** | `DVE[59]` `CAST p96+0xea00 → p0+0x4700 bf16` (32 ch × 128 elem, 219 ns, no wait, no update) | **`S[6]>=28`** — RAW: `DVE[54] COPY` writes `0xea00`, S[6]#28 lands 19.630. WAR: the previous `p0..31+0x4700..0x47ff` was last read by `PE[58] MATMUL` (ends 18.319, `acc_flags=1`, **no S[5] update**) and `PE[36] LDWEIGHTS 0x47fe` (≤15.2); `DVE[54]` waits `S[5]>=31` = `PE[67]` (19.298) and **MATMUL end times are monotonic in pc** here (56→17.813, 58→18.319, 60→18.345, … 67→19.298), so PE program order closes it. | index 46, **after** iter-3's `S[51]` CAST at `ACT[45]` — inserting before it pushes `S[51]` out and stalls the Vector `ES` that waits on it | 

Vacated Vector slot in both cases: `ES{wait S[new]>=1}` **with no update**, which is what keeps every
later Vector `S[6]` increment implying "the CAST completed" (the consumers `PE[52]` and `PE[96]`
have no guard of their own). `S[52]`/`S[53]` each 1 inc / 1 wait.

**Measured, interleaved, alternating which config ran first:**

| pair | A | B | Δ |
|---|---|---|---|
| E vs **I** | 38.688 | 38.578 | −0.11 |
| E vs **I** | 38.507 | 38.588 | **+0.08** |
| **I+J** vs E | 38.333 | 38.847 | −0.51 |
| **I+J** vs E | 38.353 | 38.714 | −0.36 |
| **I+J** vs E | 38.227 | 38.576 | −0.35 |

**I alone is not resolvable (one pair each way); I+J together is, and it is worth ~0.4 us.** Every
`I+J` capture (38.33, 38.35, …) is below every `E` capture taken in the same half hour
(38.51–38.85). So each move is worth ~0.2 us after all — the same ~0.15–0.2 iter-3 measured — and
**the lesson is about measurement, not about the lever: at this kernel size a single 0.2 us move
needs 4–5 interleaved pairs, but two of them stacked need only two.** Build the pair of edits, then
A/B the pair.

Semaphores `S[52]` (edit I) and `S[53]` (edit J) each 1 inc / 1 wait. Free after this iteration:
0, 1, 7, 18–23, 40–42, 54, 55, 61–63; `2` is the runtime barrier; 47/48/49 iter-2's, 50/51 iter-3's.
`TENSOR_TENSOR` is still not available on Scalar, so the 274–292 ns Vector ops cannot move at all,
and `DVE[43]`/`DVE[59]` were the **last** wait-less, update-less Vector `CAST`s — **this lever is
now exhausted by inventory, not by payoff.**

# iter-6 (ssd_mamba2_1b3_s2048): three bit-exact edits — merged the two per-unit `*= exp_cs_col`
# ACT multiplies into one **3-D access-pattern** ACTIVATE, retired the 1040 dead `Y_off` MEMSETs by
# **shrinking them instead of deleting them**, and deleted the redundant second `exp_cs_last`
# broadcast by **repointing its consumer at the first broadcast's buffer**.
# Also: **the `exp_cs_last` recomputation lever is dead** (pairing solved, then device hang).

**Shipped: 4777.0 -> 4683.0 us (-1.97%). `verify.py` VERIFIED, `max_abs_err` 5.234e-07 /
3.334e-07 — byte-identical to iter-5 (no edit changes any arithmetic). Members changed:
`Activation0.bin` and `DVE0.bin` only. Against iter-0: 7154.9 -> 4683.0, **-34.6% (1.528x)**.
md5 `d415202bd55f9725949c3990569cf158`.**

My 5-repeat control on iter-5's NEFF (md5 `ba28564445…`): **4776.995 us** (min 4776.5, max 4778.3,
spread 0.04%). ACT 3344.0 busy (70.0%), DVE 2427.2, Pool 2072.1, PE 1921.7, DMA 1321.2, SP 279.3.
After: ACT **3112.4** (66.5%), DVE **2175.5** (46.5%), everything else flat.

| step | median us | ACT busy | DVE busy | delta |
|---|---|---|---|---|
| iter-5 (control, 5 repeats) | 4776.995 | 3344.0 | 2427.2 | - |
| §0 Y merge, 367 contiguous sites (3) | 4757.397 | 3227.7 | - | -19.6 |
| **§0 Y merge, 708 sites (5)** | 4731.489 | 3113.0 | 2424.9 | **-45.5** |
| **+ §9 `Y_off` MEMSET shrink, 1040 sites (5)** | 4708.946 | 3113.0 | 2339.7 | **-22.0** |
| + §10 second broadcast, 408 sites incl. timing-only ones (3) | 4681.930 | - | 2167.6 | -27.0 |
| **+ §10 second broadcast, 384 statically-fenced sites (5, shipped)** | **4683.020** | **3112.4** | **2175.5** | **-26.0** |

Spread 0.03-0.07% (2-4 us), and each step was reproduced on two independent profile runs. Dropping
the 24 sites that only had a timing argument cost **nothing measurable** (4681.93 -> 4681.94 on the
same 3-repeat setting), so the shipped version is the statically safe one.

## READ THESE SEVEN FIRST

0. **§9 — when an instruction is dead, SHRINK it, do not replace it.** A `MEMSET` cut to
   `dst_element_count=1, nch=1, num_elem=[1,1,1,1]` still fires its `SEM_INC_COMPLETE` **on
   compute completion**, so it needs no `EVENT_SEMAPHORE` handshake (iter-5 §1) and no threshold
   renumbering, and it costs ~30 ns instead of 112 ns. This turned iter-5 §4b's **+28.9 us
   regression into -22.0 us** on the identical site list. Try this first on any future dead
   instruction. (Corollary from §10: when the dead instruction has **no** semaphore update at all —
   5400 of the 6012 `STREAM_SHUFFLE`s do not — a plain `EVENT_SEMAPHORE` with its wait copied and
   *no* update is equally safe and cheaper, because an ES with no update cannot release anything.)
0b. **§10 — redirect the consumer, do not rebuild the value.** iter-3 deduped the second
   `exp_cs_last` broadcast by replacing it with a DVE `COPY` *of the first broadcast* and its wider
   version made the wall go **up**; repointing the consumer's address at the first broadcast's
   buffer instead adds no instruction and no dependency, and is worth **-26 us**. The guard that
   makes it sound is a WAR check on the surviving buffer (§10a) — 71 of 479 sites fail it and
   shipping them gives `max_abs_err 7.9e-02`.

1. **§1 — a 2-group 3-D access pattern on `ACTIVATE` WORKS on this device** (`num_elem=[64,2,1]`,
   `step_elem=[1,delta/4,1]`, verified bit-identical at 283 non-contiguous sites). No instruction
   in the unedited NEFF uses a 2-D pattern, so this was unproven; it is now proven and it is a
   general tool for "merge two identically-configured ops on tiles that are **not** adjacent".
2. **§2 — put a merged instruction at the LATER slot, not the earlier one.** That is the one-line
   fix for iter-4 §5's "40 merges FAILED on device (8.2e-03)": merging at the earlier slot reads
   the second tile before its producer has written it. At the later slot nothing moves in the
   semaphore stream (the earlier slot carries no sem-4 update at **any** of the 708 sites) and no
   threshold changes. iter-4 §5's "ACTIVATE merging is exhausted, only 40 pairs survive" is
   **wrong** — 708 survive.
3. **§3 — `exp_cs_last` is DEAD. The pairing is now SOLVED (1024/1024, exact, static) and the
   lever is closed anyway by a hardware deadlock.** I built the join iter-4/iter-5 could not, then
   a numerics-preserving *cost* probe (add only the ordering the real edit needs) **hung the device
   (NRT_TIMEOUT)**. Do not spend another iteration on `exp_cs_last`. §3 has the join (reusable for
   any per-unit question) and why the hang is structural.
4. **§4 — ACT busy still converts to wall at 69% for element-time and only 20% for
   instruction-count**, because the deleted instruction mostly ran inside an existing stall window.
   Re-measured this iteration; it is the number that explains why this edit is worth 45 us and not
   150 us.
5. **§5 — ACT's remaining 1.6 ms of idle is 1284 stalls, not handoff latency**: 727 us waiting on
   DVE (sem 6), 257 us on PE (sem 5), and only 132 us in the 5547 sub-50 ns gaps. Nulling
   redundant waits is therefore worth ~nothing (measured: -2 us for 201 sites, §6).

## 0. FIRST SHIPPED EDIT: the two `*= exp_cs_col` multiplies merged (-45.5 us)

`ssd_block.py:477` `Y_intra *= exp_cs_col` and `:504` `Y_off *= exp_cs_col` are two ACT
`ACTIVATE func=128 nch=128 scale_src=POINTER_IMMEDIATE` with the **same `scale_value_imm_ptr`**
(the unit's `exp_cs_col` column), the same everything else, both in place (`src == dst`), both
`num_elem=[64,1,1]`. A pointer scalar is **per partition and does not advance along the free
dimension**, so one instruction covering both 64-element tiles multiplies both by the identical
`exp_cs_col[p]`: **bit-exact** (confirmed, `max_abs_err` unchanged to the last digit).

Per site, slots `si < sj` (median `sj-si = 2`, max 15):

| slot | new content |
|---|---|
| `sj` (later) | the merged `ACTIVATE`: `src`/`dst` start = `min(dst_i,dst_j)`; if `|delta|==256` just `num_elem[0]=128`, else `num_elem=[64,2,1]`, `step_elem=[1,|delta|/4,1]`. **Its own wait and its own sem-4 update are kept verbatim.** |
| `si` (earlier) | `EVENT_SEMAPHORE` with `si`'s wait copied (bytes 4,5,8..11), `events_update_*` zeroed |

Byte offsets used: `num_elem[0]` at 26/58, `num_elem[1]` at 28/60, `step_elem[1]` at 22/54
(signed int16, 3 entries starting at 20/52), addr immediates at 16/48 (low 20 bits are the byte
offset).

**Why this needs no semaphore work at all.** `si` carries a `SEM_INC_COMPLETE` on sem 4 at **0 of
the 708 sites** (`sj` carries one at 666). So deleting `si`'s payload removes no increment, moves
no increment, and changes no threshold — the whole class of hazards iter-4 §2 / iter-5 §1 is about
simply does not arise. The only requirement is that `sj`'s wait be strong enough for both tiles,
and it is: of the 613 sites where both waits are on the same semaphore, `wj >= wi` at **613**.
The 95 sites with waits on *different* semaphores are also fine, because **the placeholder keeps
`si`'s wait**, so ACT still cannot pass `si` until `wi` is satisfied and `sj` is after it.

Site census (`iter-6-yio.py` + `iter-6-merge.py`, `MODE=all`): 1024 units ->
**708 with two compatible uses** (all accepted: 425 contiguous `|delta|=256`, 283 needing the 3-D
pattern), **301 already single `num_elem=[128,1,1]`** (iter-2 merged those), 15 odd shapes
(3 users, or one of the 64/60-user shared-projection groups). After this iteration **1009 of 1024
units** have a single merged Y multiply.

### 0a. Measurement

| NEFF | median us (repeats) | ACT busy | sites |
|---|---|---|---|
| iter-5 (control) | 4776.995 (5) | 3344.0 | - |
| contiguous only | 4757.397 (3) | 3227.7 | 367 |
| **+ 3-D pattern (shipped)** | **4731.489 (5)** / 4731.646 (3) | **3113.0** | **708** |

230 us of ACT busy removed, 45.5 us of wall: **325 ns of busy and 64 ns of wall per merge**
(20% conversion).

## 1. THE NEW PRIMITIVE: a 2-group 3-D access pattern on ACTIVATE

`docs/05-memory-and-addresses.md` §5.4 documents `step_elem[n]`/`num_elem[n]` as a 3-D pattern
with dimension 0 innermost and address = `start + i0*step0 + i1*step1 + ...`, **but no instruction
in this NEFF uses `num_elem[1] > 1`** (I checked all five engines: zero 2-D patterns). It works:
283 sites with `|delta|` from 7448 to 38144 bytes verified **bit-identical** with
`num_elem=[64,2,1]`, `step_elem=[1,|delta|/4,1]` on both `src` and `dst`.

Consequences worth carrying forward:
* **Two ops on arbitrarily placed tiles can be merged without relocating anything.** Every earlier
  iteration's merge attempt required the tiles to be contiguous (iter-2) or proposed a relayout
  (iter-4 §5). Neither is necessary.
* `step_elem` is **signed int16**, so the usable displacement is ±131068 bytes — the whole SBUF
  offset range in this kernel. Use `step = |delta|/4` with the start address at `min` of the two.
* Require `step >= num_elem[0]` so the two groups do not overlap.
* **`src` and `dst` have independent `step_elem` arrays** (offsets 20..25 and 52..57), so the two
  merged tiles do **not** need the same displacement in the source and the destination. Both
  `iter-6-merge.py` and `iter-6-merge-generic.py` require `src_delta == dst_delta` out of caution;
  that restriction is unnecessary and dropping it would admit pairs whose source and destination
  come from different allocation pools (it only costs 9 ACT pairs here, but it is the difference
  between "merge in-place ops" and "merge any two identically-configured ops").
* The pointer scalar (`scale_value_imm_ptr`) is **not** advanced by the pattern; it stays
  per-partition. That is what makes the merge bit-exact, and it is why only pairs that share the
  *same* scale pointer can be merged.

## 2. Place a merged instruction at the LATER slot

iter-4 §5 merged 40 pairs at the earlier slot and got `max_abs_err 8.2e-03`; its static test was
"nothing on any engine touches the **earlier** tile between the two slots", which never checks
that the **later** tile's producer has run. Merging at the later slot makes that check unnecessary:
the later slot's own wait already gates its own tile, and the earlier tile has been sitting ready
since before `si`. The static test that *is* needed is the mirror image — **nothing may read or
write the earlier tile between `si` and `sj`** (it is now multiplied later). 0 of 708 sites failed
it, which also says the compiler's live ranges are generous here.

A merged instruction may only move to a later slot if the earlier slot carries no semaphore
update, otherwise the increment would have to move later too and ACTIVATE cannot carry a
handshake (iter-4 §1: no `events_extended_*` on `ACTIVATE`). Here it never does.

**ACT merging is now exhausted for real.** The five remaining per-unit pointer multiplies
(`dtx` `:454`, `X_scaled` `:463`, `dtx_decayed` `:514`, `state_sb` `:535`, and the merged Y) each
use a *different* scale pointer, so no two can share one instruction. A generic pass over all
pointer-scalar ACTIVATEs (`iter-6-merge-generic.py`) finds only **733** pairs — the same 708 plus
25 — and rejects 2863 for "earlier slot has a sem-4 update"; those 2863 are **cross-unit** pairs
that reuse the same addresses with different values (iter-4 §4's trap) and merging them would be
wrong. Do not chase them.

## 3. `exp_cs_last`: PAIRING SOLVED, LEVER DEAD (device hang)

### 3a. The join iter-4 and iter-5 could not build — it is exact and purely static
`iter-6-join.py`. Two occurrence-index joins over address classes, **zero count mismatches,
1024/1024 units, 1024 distinct `exp_cs_col`**:

```
lasts[k] = ACT ACTIVATE func=7 nch=1            (exactly 1024)
cols[k]  = ACT ACTIVATE func=7 nch=128 scale=+1 (exactly 1024)
tmm      = PE MATMUL with num_active_rows == 1  (exactly 1024)  <- the nc_transpose
step 1: the LDWEIGHTS in front of each tmm has src == lasts[].src  (the compiler elided
        `padded` and transposes cs_row in place; the address is cs_row+127*4, i.e. *exactly*
        exp_cs_last's src).  75 address classes, counts agree exactly -> last <-> transpose.
step 2: tmm[].dst is the PSUM word; its unique per-unit ACT consumer is either the +1.0 exp
        (folded units) or iter-5's f=128 scale-1.0 copy (the 36 unfolded ones).  8 PSUM
        classes, counts agree exactly -> transpose <-> exp_cs_col.
```
Timestamp cross-check: `exp_cs_last.start - exp_cs_col.end` has median **-5633 ns**, i.e.
exp_cs_last runs about one unit period *before* its own exp_cs_col, and `lasts` is **before**
`cols` in ACT slot order at 1022 of 1024 sites. That is why every slot-order and
address-arithmetic pairing in iter-4/iter-5 failed, and it is the reusable part of this section:
**on this kernel, join units through `cs_row` address classes and the `num_active_rows==1`
MATMUL, never through slot distance.**

### 3b. Why the lever is closed
The broadcast's `STREAM_SHUFFLE` group starts **before** this unit's `exp_cs_col` finishes at
**1024 of 1024 sites** (`tcol - t_shuffle_start`: p10 332 ns, median 4486 ns, max 9200 ns).
So pointing the shuffles at `exp_cs_col[127]` requires DVE to wait ~4.5 us for ACT.

I probed exactly that cost without changing any arithmetic (`iter-6-last-HANGS.py`): keep
`exp_cs_last` and the shuffles as they are, and only raise the first `LOAD_MASK_SELECT` of each
group to `sem4 >=` the value produced at the unit's `exp_cs_col` (1015 of 1503 groups; 44 already
covered, 444 have a non-sem-4 wait and were skipped). Result: **`NRT_TIMEOUT` / model-stop
timeout — the device hangs.** There is a genuine cycle: DVE blocked at the LMS cannot run the work
ACT needs in order to reach `exp_cs_col`. iter-4 saw the same hang and patched round it with a
profile-timing guard that admits **0 sites** in the current NEFF.

So iter-5 §3's **-194.8 us is unreachable** and iter-5 §6 item 1 should be struck. The only way in
would be to move the shuffle slots later in the DVE stream, which means shifting PCs — not
something any iteration has shown to be possible.

## 4. Conversion rates, re-measured on this iteration's NEFF

| probe (destructive, numerically wrong, ~3 min) | ACT busy | wall | conversion |
|---|---|---|---|
| shipped | 3113.0 | 4731.5 | - |
| `num_elem[0] = 1` on all 6157 multi-element ACTIVATEs | 2683.0 (-431) | **4435.0 (-296.6)** | **69%** |
| (this iteration's real edit, for comparison) | -230 | -45.5 | **20%** |

The gap between 69% and 20% is the useful finding. The merge deletes an instruction that was
*already executing inside a stall window*: ACT arrives at `si`, waits `sem6 >= wi`, and in the old
code it filled part of the `wi -> wj` window with `si`'s 430 ns of work. Removing the work does not
remove the wait. **Shortening every ACTIVATE converts at 69%; deleting an ACTIVATE that sits in a
stall converts at 20%.** Any future ACT edit should be priced by *where* the instruction sits, not
by its busy time (iter-5 §0a said the same thing with different numbers: 190 ns on the head chain
vs 79 ns off it).

## 5. Where ACT's 1.6 ms of idle actually is (`iter-6-stall.py`)

ACT: 9265 `ACTIVATE` (mean 360 ns), 4582 `EVENT_SEMAPHORE` (mean 55 ns, 252 us total), span 4731,
idle 1314 us distributed as:

| gap size | n | total |
|---|---|---|
| < 50 ns | 5547 | 131.9 us |
| 50-150 ns | 1184 | 76.2 us |
| 150-400 ns | 457 | 124.5 us |
| 400-1000 ns | 956 | 645.6 us |
| > 1 us | 328 | 483.2 us |

**1284 stalls of ~880 ns carry 1129 us of it.** By the semaphore the next instruction waits on:
sem 6 (DVE) **727 us** (369 at an `EVENT_SEMAPHORE`, 358 at an `ACTIVATE`), sem 5 (PE) **257 us**,
one-off DGE stalls 90 us. So ACT is latency-bound on DVE and PE roughly once per unit, and the
8000 sub-150 ns gaps — the thing iter-3 called "semaphore handoff latency" — are worth only
208 us in total. **Do not build anything that targets handoff latency.**

Classified by which per-unit op stalls (`iter-6-stall2.py`, gaps >= 300 ns; `role()` in that
script decodes an ACT slot into its source-level identity from
`(activation_func, num_active_channels, scale_src, who-writes-src, who-writes-the-scale-pointer)`
— reuse it, it is the fastest way to read an ACT profile):

| us | n | wait | op |
|---|---|---|---|
| 379.1 | 476 | sem6 | `EVENT_SEMAPHORE` — 440 of them are this iteration's merge placeholders |
| 213.5 | 322 | sem6 | the **merged Y multiply** (n=128), waiting for DVE's `Y_off` add / `Y_intra` copy |
| 158.1 | 169 | sem5 | `state_sb *= exp_cs_last_N` — **WAR on `state_sb`**: PE must finish reading it as the `Y_off` matmul's moving operand before ACT may overwrite it |
| 100.1 | 119 | sem5 | `dtx_decayed = dtx * decay_factor` |
| 79.1 | 109 | sem6 | `exp_cs_last`, waiting for the DVE scan (head chain) |
| 62.2+91 | 92 | sem6/DGE | `log_decay` |

**So 592 us of the 1129 us is ACT waiting at the merged-Y site** for the
`PE matmul -> PSUM -> DVE copy/add -> SBUF -> ACT multiply` chain. Removing the DVE hop is
hardware-refuted (iter-3 §3b, `PSUM_COLLISION`: ACT may not drain a `LOW_HIGH`/accumulating
matmul's PSUM) and moving the multiply to DVE costs 1 ulp (iter-4 §6). The rest is genuine data
dependency (including the `state_sb` WAR, which is intrinsic to the recurrence, not an artefact of
iter-4's relayout). **The remaining bubble can only be filled with work from a different unit,
i.e. by moving instructions across large slot distances — which needs the sem-4 renumbering and is
the one big thing nobody has tried.**

## 6. BUILT, MEASURED, NOT SHIPPED: freeing the merge placeholder's wait (-2 us)

440 of the 448 `EVENT_SEMAPHORE`-on-sem-6 stalls (365 us!) are at the placeholders this iteration
created: ACT stops there for ~825 ns and does nothing. Nulling that wait should let ACT run the
instruction between `si` and `sj` during the stall. `iter-6-esnull-MARGINAL.py` does it at the
**201** sites where it is sound, and the result is **4729.5 vs 4731.5 — about -2 us**, inside/at
the edge of the 2 us spread. Not shipped. Two things to carry forward:

* **Soundness rule for deleting a wait.** `ACTIVATE` has exactly **one** wait field, so an
  instruction needing two semaphores *must* rely on an earlier instruction's wait for one of them.
  Therefore you may only drop instruction X's wait on sem S if every instruction between X and the
  next wait that covers it either (a) states its own wait on **the same semaphore** S, or
  (b) references no address that any instruction on S's owning engine ever writes. With that rule
  201 of 708 sites qualify; the naive "it is covered later" rule admits 502 and is **not** safe.
* **Why it buys nothing:** the in-between instruction cannot proceed either — its own wait is also
  unsatisfied. The `wi -> wj` window is a genuine data-dependency bubble, not a scheduling
  artefact. A generic "null the 1175 ACT waits that an earlier ACT wait already implies" pass is
  for the same reason worth ~0 (ceiling 1175 x 30 ns = 35 us) and I did not run it.

## 9. SECOND SHIPPED EDIT: the dead `Y_off` MEMSET, retired by shrinking (-22.0 us)

iter-5 §4b had all the analysis right and the wrong implementation. `num_dstate_tiles == 1`, so
`ssd_block:486-503` is `memset(Y_off, 0); Y_off = Y_off + Y_off_psum`, i.e. `Y_off = copy(psum)`
(`0.0 + x == x` for finite fp32). `iter-6-yoff.py` (derived from `iter-5-yoff-REGRESSION.py`,
same 1040-site list, zero rejections — 1024 `Y_off` + 16 `CB_accum`):

1. The DVE `TENSOR_TENSOR_ARITH_OP ADD` (`src0 == dst == Y_off`, `src1` marker 2 = PSUM) is
   rewritten in place as a DVE `COPY` from the PSUM word, **sync bytes verbatim**. iter-5 already
   verified this bit-identical; it is confirmed again here.
2. The `MEMSET` is **kept and shrunk**, not replaced: `dst_element_count = 1`, `nch = 1`,
   `num_elem = step_elem = [1,1,1,1]`. It still zeroes `Y_off[0][0]`, which the `COPY` immediately
   overwrites, so the result is bit-exact, and because it is still a real compute instruction its
   sem-6 increment keeps firing when the datapath finishes — **no semaphore byte changes at all**.

DVE busy 2424.9 -> 2339.7 (-85.2 us) for -22.0 us of wall (**26% conversion** — far better than
iter-3's 8% for DVE work, and in line with iter-5's 28% probe).

### 9a. Two REFUTED implementations of the same edit, both device hangs
Both cost ~90 s each; record them so nobody repeats them.

* **`MEMSET` -> `EVENT_SEMAPHORE` with no update + a sem-6 renumbering. HANGS.** The remap
  `T' = T - |{m in M : m < T}|` is exact for waits whose producer survives, but a wait whose
  producer *was* the deleted MEMSET maps to the **next** increment — which on DVE is frequently
  the rewritten `COPY` in the same group, so DVE ends up waiting on itself. Mapping the other way
  (`side='right'`, down to the previous surviving increment) is not safe either: a DVE instruction
  with no increment of its own is covered by the *first increment at or after it*, so lowering a
  threshold past it releases its consumers early. **Conclusion: you cannot renumber a
  completion-counter semaphore by deleting increments unless you also know, for every waiter,
  which instruction's write it needs.** The machinery (assertions that sem 6 is DVE-only, has no
  extended updates and only `WAIT_FOR_SEM_GE_IMM` waits; 8964 waits rewritten including extended
  ones) is in `iter-6-yoff.py` without `SHRINK=1` if a future iteration wants to retry with real
  per-waiter reasoning.
* **Shrinking `num_elem` but forgetting `dst_element_count`. HANGS.** `MEMSET`
  (`v2.NEURON_ISA_TPB_D4_MR_STRUCT`) carries a **separate total count at offset 28**
  (`dst_element_count == product(num_elem)`, 4 bytes) on top of `dst_mem_pattern_num_elem`
  (offset 56, **4** entries, not 3) and `step_elem` (offset 48, 4 entries). Leaving offset 28 at 64
  while `num_elem` says 1 hangs the engine. `num_active_channels` is at offset 34. The compiler
  itself emits 16 MEMSETs in the exact `(1, nch=1, [1,1,1,1])` form, so copying that form is safe.

## 10. THIRD SHIPPED EDIT: the redundant second `exp_cs_last` broadcast (-26.0 us)

`exp_cs_last_Q = broadcast_scalar_to_column(exp_cs_last, Q)` (`ssd_block:227`, consumed by
`decay_factor = exp_neg_cs_col * exp_cs_last_Q`) and
`exp_cs_last_N = broadcast_scalar_to_column(exp_cs_last, n_size)` (`:533`, consumed by
`state_sb *= exp_cs_last_N`) broadcast the **same fp32 scalar** into two different 128-partition
buffers. 479 units still had both (iter-3 replaced one at the other 545). `iter-6-bcast.py`:

1. Cluster the 8 `STREAM_SHUFFLE`s that read this unit's `exp_cs_last` **by destination byte
   offset** (not by time order — time order mixes the two groups at 89 of 479 sites) into two
   groups of 4; `g1` is the one that finishes first, `g2` the other.
2. Find the single remaining consumer of `g2`'s buffer and **repoint its address field at `g1`'s
   offset** — bit-exact, same value, and the operand is now produced *earlier*, so no wait moves.
   The consumer is `ACT scale_value_imm_ptr` at some units and `DVE src1` at others (278 of 479):
   whichever of `:227`/`:533` happens to be scheduled second, so **do not assume which group is
   which** — read the consumer and handle both kinds.
3. `g2`'s four shuffles now write a buffer nobody reads. Replace each one that carries **no** sem-6
   update with an `EVENT_SEMAPHORE` keeping its wait and no update (1194 of 1536); leave the ones
   that do carry an update alive and untouched, so no threshold moves.

`LOAD_MASK_SELECT` is **6 ns** (8192 of them, 49 us total) — not worth touching. A
`STREAM_SHUFFLE` is 147 ns and essentially all fixed cost (32 partitions x 1 element), so there is
nothing to shrink; deletion is the only lever, and 5400 of 6012 have no semaphore update, which is
what makes it cheap.

DVE busy 2339.7 -> 2175.5 (-164 us) for -26 us of wall (**16% conversion**).

### 10a. The guard that matters: a WAR check on the surviving buffer
The two buffers exist because their live ranges differ — `exp_cs_last_Q` is consumed early and its
SBUF block is recycled for a later unit, while the state multiply runs late. Repointing without
checking that gives **`max_abs_err 7.9e-02`** (shipped by accident on 479 sites; cosine 0.984).
Two conditions, both needed:
* **Observed**: no write to `g1`'s offset between `g1`'s last shuffle and the repointed consumer's
  read (71 of 479 sites fail — that is the whole error above).
* **Static**: the *next* write to `g1`'s offset after the consumer must be provably ordered after
  it. Either the consumer is on DVE and that writer is a **later DVE slot** (in-order), or the
  consumer is on ACT at slot `ck` and the clobbering DVE slot's **governing sem-4 wait** (the
  running max of sem-4 wait values down the DVE stream) is `>= cum4[ck]`-at-or-after. 384 of 408
  qualify; the other 24 are worth 0.0 us, so there is no reason to ship a timing-only argument.

The same recipe would apply to the 545 iter-3 units (retire the dedup `COPY`), but only 73 of the
545 copies have no sem-6 update, so it is worth ~2 us. Not done.

## 11. Ranked for iter-7

State after this iteration: **ACT 3112 us busy / 66.5%**, DVE 2176 / 46.5%, Pool 2072 / 44.2%,
PE 1919 / 41.0%, DMA 1362, SP 279, span 4683. Nothing cheap is left on ACT's instruction count; the
measured levers are now ACT *element* time (69% conversion, -297 us if it were free) and the
DVE/PE latency ACT waits on (984 us). Only bit-exact edits are viable (iter-4 §6: `max_abs_err`
is at 84% of the limit). §4/§5's numbers were measured on the 708-merge NEFF (4731 us) and so are
the right order of magnitude but not re-measured after §9/§10 — re-profile before using them.

**The two primitives this iteration added, in the order to try them on any new target:**
shrink a dead instruction to 1 element (§9) > drop it to a no-update `EVENT_SEMAPHORE` (§10) >
merge it into a neighbour with a 2-group 3-D access pattern at the *later* slot (§1, §2) >
redirect its consumer's address so it becomes dead in the first place (§10). Never move a
semaphore increment and never renumber a completion counter (§3b, §9a).

1. **Software-pipeline / shorten the serial head chain.** §5 is the first direct accounting of it:
   ACT waits 727 us on DVE and 257 us on PE, ~1284 stalls, ~1 per unit. The chain is
   `dt_row(DMA) -> log_decay(ACT) -> scan(DVE) -> transpose(PE) -> 2 exps(ACT) -> broadcast(DVE)
   -> decay_factor(DVE) -> dtx_decayed(ACT)`; every arrow is a semaphore handoff plus one
   instruction. Largest remaining lever and still the most expensive to build. A probe that
   *lowers* (not raises) a head-chain wait is numerically wrong but cheap and would size it.
2. **The 15 unmerged units and the 25 extra pairs the generic pass finds** (~40 x 325 ns). Warm-up
   only, ~2 us.
3. **A sem-4 renumbering pass is the enabling primitive nobody has built.** sem 4 is purely
   ACT's completion counter (7360 increments, nothing else touches it, every wait is
   `WAIT_FOR_SEM_GE_IMM`), so if you delete the increment that sits at ACT slot index `m`, the
   remap `T -> (T-1 if T > m else T)` makes **every** wait in the program refer to exactly the
   same ACT slot as before — this is *exact*, unlike iter-4 §2's DGE remap which failed because
   those increments stood for distinct transfers. ~9700 waits to rewrite (ACT 4241, DVE 2746,
   PE 2465, SP 226, Pool 20). That unlocks (a) merging a pair whose earlier slot carries an
   increment, and (b) moving an ACT instruction to a different slot, which is what §5 says is
   needed to fill the 592 us bubble. Build and validate it on a single site first.
4. **The 3-D pattern (§1) on DVE is not worth it: only 136 mergeable pairs** (`iter-6-dvem.py`;
   4778 pairs rejected because the earlier DVE slot carries a sem-6 increment — DVE increments on
   nearly every slot — and 1245 because a `TENSOR_TENSOR` needs *three* operand deltas to agree).
   136 x ~200 ns of DVE busy at iter-3's 8% / iter-5's 28% conversion is 2-7 us, under the noise.
5. **Merging the two per-unit DVE PSUM->SBUF copies is NOT worth it (~3 us). Checked, not built.**
   `iter-6-dcop.py` joins each merged Y `ACTIVATE` to the two DVE `COPY`s that drain
   `Y_intra_psum` and `Y_off_psum`: **1004 of 1011 pairs found**, configs identical, DVE slot
   distance median 3, PSUM displacement always a multiple of 2048 (the bank size, both signs) so a
   2-group pattern would have to span banks. The blocker is the semaphore, not the pattern: both
   copies carry a sem-6 update, so the merged copy must go at the **earlier** slot (putting it late
   would let the early increment release PE's PSUM reclaim before the second tile is read) and the
   earlier slot's single wait field must then cover both producers' waits — true at only
   **362 of 1004** sites ((5,5) 285 and (6,6) 77; the rest wait on *different* semaphores). And the
   late slot can only be **shrunk**, not deleted, so the saving is ~57 ns x 362 = 21 us of DVE busy
   = ~3 us of wall. If the sem-4/sem-6 increment problem is ever solved generally, revisit: deleting
   1004 x 207 ns = 208 us of DVE busy would be worth ~-30 us.
6. **A proper dead-store scan needs interval overlap, not exact-address match.**
   `iter-6-dead.py` looks for "writer W1 of address A, then writer W2 of A, no reader of A in
   between" and reports 1438 `STREAM_SHUFFLE`, 953 `TENSOR_TENSOR` and 859 `TENSOR_TENSOR_SCAN`
   candidates. **They are all false positives**: the reader of `cs_row` is at `cs_row + 508`
   (`exp_cs_last`'s src and the transpose's `LDWEIGHTS`), and the broadcast buffers are read at a
   different partition encoding, so an exact-address reader set misses them. A future scan must
   compare `[start, start + bytes)` intervals per partition. Worth doing properly — it is what
   found §9's and §10's wins by hand.
7. **Do NOT redo:** `exp_cs_last` *recomputation* (§3, device hang — note §10 is a different
   edit on the broadcast and did ship); ACTIVATE merging beyond §0 (§2);
   handoff-latency / redundant-wait edits (§5, §6); iter-5 §4b (`Y_off` memset+add) and §5
   (Pool/DMA merging); iter-4 §2 (DGE threshold remaps), §5 (merging at the earlier slot), §6
   (ACT->DVE multiplies cost 1 ulp); iter-3 §3b (ACT may not drain a `LOW_HIGH` matmul's PSUM).
   And `semaphores.py`'s `sems 40-71 ... wait exceeds increments` alarm is still iter-4's
   documented false positive.

## 12. Tool notes (additions to iter-5 §7)

* **`tools/neff.py` is not in this repo** — the unpack/pack/diff/semaphore tools live only in
  `/home/ubuntu/workspace/neff_doc/tools/`. `tools/` here has just
  `verify.py`, `profile.py`, `launch_rules.py`, `device.py`, `neffio.py`.
* Run order (`iter-6-try.sh`): `git show HEAD:kernel.neff > kernel_start.neff`,
  `neff.py unpack kernel_start.neff work/`, `profile.py --repeats 3 --parquet prof`,
  `scripts/<edit>.py`, `neff.py pack work/ kernel.neff`. Every script here loads from `work/`, so
  **always re-unpack before a dry run** or you will decode your own previous edit.
* **Reproducing the shipped NEFF takes three passes, each on a fresh profile of the previous
  result**, because every site selection uses timestamps from `prof*/Instruction.parquet`:
  1. unpack iter-5's NEFF, `profile.py --parquet prof`, `MODE=all iter-6-merge.py`, pack
     (-> md5 `fffbf39d093eb902fdc68c1785a31f72`, 4731.5 us);
  2. unpack that, `profile.py --parquet prof2`, `SHRINK=1 iter-6-yoff.py`, pack
     (-> `583d1b58f0e439b42472d2d6df28b451`, 4708.9 us);
  3. unpack that, `profile.py --parquet prof3`, `STATIC=1 iter-6-bcast.py`, pack
     (-> `d415202bd55f9725949c3990569cf158`, 4683.0 us).
  `iter-6-try.sh` / `iter-6-try5.sh` wrap one pass each. Running a later pass against an older
  profile will pick wrong sites.
* The shipped NEFF was `verify.py`-ed **three times** (all VERIFIED, identical `max_abs_err`), as
  the loop asks for any edit that touches ordering.
* `nki_source_location` in the parquet is useless in this build (it is the same vendor path for
  every row). Classify an ACT slot by decoding it instead: `(activation_func,
  num_active_channels, scale_src, scale_value_imm_ptr, num_elem)` identifies every per-unit op.
* `ACTIVATE` has **no `perf_mode` field** on this ISA version, so there is no throughput mode to
  flip for fp32.
* A hang costs about 90 s of wall (`nrt_execute` fails, then a 30 s model-stop timeout and two
  100 ms close timeouts) and leaves the device usable — `verify.py` passed on the very next run.
  A deadlock probe is therefore affordable; budget ~2 minutes for it.

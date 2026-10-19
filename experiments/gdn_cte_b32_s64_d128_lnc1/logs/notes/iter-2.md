# iter-2 — **fused the A_full -> Ball -> transpose two-hop queued-DMA chain into one hop**
# **876.0 -> 835.6 us (-4.6%, 1.048x; 1.419x vs the iter-0 baseline)**, outputs bit-identical.
#
# Read these three things first:
#  * the **ceiling probe** (clear/lower an engine's waits, profile, never ship) — it found this
#    edit and closed five other candidates, at 90 s each. Section "THE MEASUREMENT THAT FOUND IT".
#  * **DVE busy time converts to wall time at 0.7-1.0** on this kernel, unlike SP's (iter-1).
#  * **candidate 1 below is already built and measured at a further -45 us, but it races.**

## Result

| NEFF | total_exec_time_us | samples |
|---|---|---|
| start (= iter-1 shipped), 5 rep | **875.980** | 872.2 873.8 876.0 876.8 878.0 (0.66%) |
| shipped (fusion), 5 rep run 1 | **836.707** | 832.7 833.1 836.7 838.4 838.9 (0.73%) |
| shipped, 3 rep (parquet run) | **834.807** | — |
| shipped, 5 rep run 2 | **835.300** | 834.5 835.1 835.3 835.4 836.2 (0.21%) |

Three independent measurements of the shipped NEFF: 836.7 / 834.8 / 835.3 (take **835.6**).

The two bands do not touch (838.9 < 871.6 min of the start's own later run). `verify.py`: PASS,
`max_abs_err` **identical to the baseline to every printed digit** (output_0 7.688e-05 = 11% of
the limit, output_1 5.842e-04 = 82%). `launch_rules.py` PASS. `semaphores.py` zero warnings.
Engine busy: DVE 438.8 (unchanged), ACT 346.6, PE 262.6, **DMA 208.7 -> 198.5**, Pool 166.7,
SP 34.2. Only `sg00/SP0.json` + `sg00/SP0.bin` change (2 of 34 members).

---

# THE EDIT (full recipe; `scratch/` does not survive, this is enough to rebuild it in ~20 min)

## What the chain was
`nisa.dma_transpose(dst=opt, src=op_view)` at **gdn_cte.py:350** (320 instances = 32 groups x 10
`(bi,bj<=bi)` slots) reads `op_view = Ball[:, slot*16 : +16]`. `Ball` is **itself a queued DMA
copy** of `A_full`: iter-1's edit A fused the `A_full -> A_hbm -> Ball` DRAM round trip into one
SB->SB `qSPSpillReload1` transfer. So PE's doubling solve sat behind **two chained DMA hops**:

```
ACT/DVE write A_full (partitions 64..127)
   -> q1 transfer  "Ball"   (A_full -> partitions 0..15, pure bf16 copy, 16x512 B)
   -> q2 transfer  "opt"    (16x16 bf16 transposes out of Ball)       <-- waited for Ball
   -> PE LDWEIGHTS at :408/:418/:441/:447/:449
```
The second hop's gate was **not a semaphore on the trigger** (158 of 161 q2 triggers have
`events_wait_mode = NONE`): it was **SP's program order** — the q2 triggers sit immediately after
the q1 Ball transfer's *relay* `EVENT_SEMAPHORE`, which blocks SP until Ball completes.

## The fix, in two parts
1. **Compose the two access patterns** so each q2 descriptor reads `A_full` directly.
   The producer descriptor is a pure same-dtype SB->SB copy, so the map is invertible:
   `translate(addr, prod_desc)`: decompose `addr - to_off` over the `to_sizes/to_steps` dims
   (greedily, strides descending; reject if the reconstruction does not match), then apply
   `from_off + sum(idx[k]*from_steps[k]) + rest`. Then for a consumer descriptor:
   `from_off' = translate(from_off)` and, for each dim k>0 with `size>1`,
   `from_steps'[k] = translate(from_off + from_steps[k]) - from_off'`. **Verify by expanding the
   full cell set both ways** (`cells(new,'from') == {translate(c) for c in cells(old,'from')}`).
   All 161 q2 transfers composed as a *single* descriptor each — no descriptor split needed.
   Worked example (real): `from_off 114688 -> 16794368`, `from_steps [1,262144,160,4194304] ->
   [1,262144,4194336,4194304]` (tile stride 160 = one 128-B block + 32 B maps to
   one 16-partition block + 32 B = 4194304+32). `to_*` untouched. dtype unchanged (bf16->bf16),
   so this is *not* the transpose+cast case that s128 §2.5 proved the loader rejects.
2. **Move each q2 trigger in `SP0.bin` to immediately after its producer's q1 trigger** and give
   it the q1 trigger's own wait (`S[40..55] >= v`, "ACT reached the dead save" = A_full ready).
   That is exactly the guarantee the q1 copy had, and the fused descriptor reads exactly the
   cells the q1 copy read, so the wait is sound and nothing else is needed.
   **`assert` that ring order is preserved**: `sorted(plan, key=ci) == sorted(plan, key=(pi,ci))`.
   This is critical — a queued ring is FIFO and every relay resolves "transfer m on ring r" by
   *trigger order*, so permuting two triggers of the same ring silently remaps every relay.
   (It held here for all 161.)

Rebuild: `neff.py unpack`, `profile.py --parquet prof`, the two steps above on `SP0.json` +
`SP0.bin`, `semaphores.py`, `neff.py pack`, `verify.py`, `profile.py --repeats 5`.
The *placement* step needs no profile — it is purely structural (producer trigger position).

## Why it is worth 39 us
Removing one DMA hop from the chain removes its fixed latency, which s128 §2.7 measured as
**717 ns trigger->start + transfer + 384-518 ns completion->consumer ≈ 1.0-1.8 us**, once per
group-entry into the solve. `DMA` busy dropped 208.7 -> 198.5 us and PE's idle on the relay
semaphores `S[40..71]` dropped **187.9 -> 142.6 us**; total PE idle 584.5 -> 541.9 us.

---

# THE MEASUREMENT THAT FOUND IT: a *ceiling probe*

Rather than reasoning about which stall is on the critical path, **delete the waits and profile**.
Clearing every PE wait on `S[40..71]` (321 waits, `events_wait_mode = NONE`; incorrect, so
profile only, never verify or ship) gave:

| NEFF | probe: PE waits on S[40..71] cleared | shipped |
|---|---|---|
| iter-1 (start) | **775.9** | 876.0 |
| iter-2 (fused) | **782.3** | 836.7 |

So the DMA chain feeding PE was worth **-100 us** of wall before the fusion and is worth
**-54 us** after. **This probe is the single most useful tool of the iteration — it converts a
stall attribution into a number in 90 s, and it refuted iter-1's conclusion that the lever was
closed.** Use it before building anything.
(Caveat: a probe can fault. Clearing PE's `S[6]` waits at lines 407-419 dies with
`TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION` — those waits also order PSUM accumulation, so the
DVE-chain prize could not be sized this way.)

---

# REFUTED THIS ITERATION (each with the number that closed it)

### 1. Retargeting PE's waits off the relay chain onto the queue semaphore: **+3 us, i.e. nothing.**
This is review candidate #1 / iter-1's "lever 1" (delete the relay chain). Built it properly and
soundly, twice (before and after the fusion), and it does not pay.
* **How** (`scratch/retarget.py`, keep this, it is correct): PE only ever *reads* SBUF (it writes
  PSUM), so every PE wait on `S[40..71]` is a pure RAW edge and can be retargeted without a
  conjunction. For each PE slot waiting `S[s] >= 16j`: expand its read set
  (`LDWEIGHTS`: `src_..._addr_immediate` +- `num_elem[0]*step_elem[0]*elemsize`, over
  `num_active_rows` partitions at stride 262144; an `EVENT_SEMAPHORE` inherits the read set of the
  next addressed slot), resolve `(s, ordinal o)` -> transfer via the relay inventory (**the guard
  `EVENT_SEMAPHORE` immediately before each relay carries `S[s] >= 16*(o-1)`, so the ordinal is
  readable straight out of the bytes** — 432 of 448; the 16 ordinal-1 relays have no guard),
  intersect, and take the **largest** intersecting ordinal <= j (earlier hits are the same SBUF
  address written for a *previous group* — buffer reuse, not a live producer).
  321 of 321 PE waits resolve (111 to a unique transfer, 210 to a buffer-reuse chain).
  `semaphores.py` clean, `verify.py` PASS bit-identical.
* **Measured**: iter-1 base 876.0 -> 874.2. Fused base 836.7 -> 833.1. Both inside the +-3 us
  between-run floor. **Reverted** (it also relies on an unproven cross-ring WAW argument).
* **Why**: split PE's `S[40..71]` idle into three terms (`scratch/tsplit.py`): of 142.6 us,
  **84.4 us is before the transfer's trigger has even been issued** (SP is still waiting for
  A_full — a genuine upstream dependency), 54.5 us is trigger -> relay-done, and only 3.7 us is
  relay -> PE start. The relay is **not** the cost; the hop is. My earlier "waiting for the host
  engine to ARRIVE at the relay" metric (95-110 us) is **misleading** and should not be used: SP
  not having reached the relay usually means SP has not reached the *trigger*, i.e. the data does
  not exist yet. Note the probe's -54 us agrees with the 54.5 us trigger->relay term to 1%.

### 2. There are **no more real two-hop DMA chains**. CLOSED statically.
A generic scan (`scratch/chk.py`: for every transfer, the latest earlier transfer whose SB write
set *contains* its read set) reports 255 more pairs — q0<-q0 52, q2<-q0 140, q0<-q1 35,
q1<-q0 28 — but **0 of them have the consumer trigger waiting on the producer's ring semaphore**,
and the gate is not SP program order either (the q0 triggers wait on `S[3..6]`, i.e. compute).
They are **stale-buffer coincidences**: the same SBUF tile written by a previous group. Containment
alone is not evidence of a chain — require a semaphore edge or an SP-order gate.
Also: my naive rewiring (move the consumer trigger after the producer's) **breaks ring order** for
all four of those sets, so it could not have been applied anyway.

### 3. Head-of-line demotion (s128 iter-4's lever) is small here. Not built.
`scratch/plan.py` with s128's exact rule (move a blocking slot forward past the maximal run of
unblocked following slots that contains >=1 DMA trigger and does not touch a semaphore it
updates): **SP 13 relay moves holding 18.4 us** (21 moves / 36.5 us if any `EVENT_SEMAPHORE`
qualifies), versus 44 moves / 59.8 us in s128 where it was worth -31 us. On **Pool** the rule is
empty for relays (0 moves); the only movers are the 224 `PSEUDO_DMA_DIRECT2D` input loads.
Built the 19-move version on top of the fusion anyway (pure permutation, `semaphores.py` clean,
same slot count): **NRT_TIMEOUT / hang**. So s128 iter-4's lever is not merely small here, it is
**unsafe on this NEFF** — the rule "jump only unblocked slots that do not touch a semaphore the
mover updates" is **not sufficient** once the relays are split across two host engines (SP 128 /
Pool 320) with guards enforcing a shared ordinal order on `S[56..71]`. I did not isolate which of
the 19 moves deadlocks (a hang costs ~70 s of device time and the lever is worth <= 5 us).
Do not retry without first proving the mover's ordinal order is preserved across *both* hosts.

### 4. Demoting blocking *compute* slots on the bottleneck engine. Sized, not built.
`scratch/dveplan.py`: slots that block >200 ns and have a following run of ready slots that do
**not** increment the engine's own semaphore (so the ordinal order is preserved) hold
**DVE 143.7 us (166 slots, run length 1-2), ACT 62.2, PE 121.4**. Looks tempting, but the gain per
move is bounded by the *execution* time of the 1-2 jumped slots (~250 ns), i.e. <= ~40 us of DVE
idle filled, and the two analogous experiments on the s128 sibling (§2.3 trigger hoisting, 285 us
addressed; §2.4 ACT ES demotion, 71 us addressed) both measured **exactly 0**. It also needs a real
SBUF address-hazard check. Low expected value.

---

# WHERE THE TIME GOES NOW (`prof2/`, 834.8 us span)

Engine busy: **DVE 438.8 (52.5%)**, ACT 346.6 (41%), PE 262.6 (31%), DMA 198.5, Pool 166.4,
SP 34.2. Union-busy (overlap removed): PE 270, DVE 438, ACT 357, Pool 331, SP 269.
Idle >200 ns attributed to the semaphore the next slot waits on (`scratch/attr.py`):

| engine | idle | biggest buckets | biggest (sem, NKI line, op) |
|---|---|---|---|
| PE | 541.9 | **S[6] DVE 236.4**, S[4] ACT 73.9, S[56..71] relays 142.6 | :418 LDW S[6] **96.0**, :223 LDW S[4] 54.0, :441 ES S[6] 41.8, :195 LDW S[6] 36.0, :441 LDW S[56..71] ~140 total |
| DVE | 351.0 | **S[5] PE 245.3**, S[3] Pool 36.5, S[4] 21.8 | :443 TT S[5] ~130, :225 TT S[5] 68, :254 CAST S[3] 37 |
| ACT | ~520 | S[5] PE ~273, S[6] DVE ~162 | :451 CAST S[5] 157, :458 CAST S[6] 116, :453 CAST S[5] 67 |
| Pool | ~606 | S[4] 185 (its 224 DGE input loads), S[6] 162 | — |
| SP | ~557 | S[5] 260 at the :350 triggers | — |

**The kernel is now a pure three-engine latency chain, DVE-led.** Two serial cycles, ~2 cross-engine
hops each (~300-500 ns per hop is the hardware floor, s128 §2.7):
* the **doubling solve** (`:441` PE matmul -> `:443` DVE tensor_tensor -> `:441`, and
  `:447/:449` PE -> `:451/:453` ACT PSUM->SBUF casts -> `:447/:449`), 512 rounds;
* the **coupling loop** (`:407/:417` DVE fp32->bf16 casts of `Wg[bj]`/`Ug[bj]` -> `:408/:418` PE
  matmuls), 192 each.
Measured proof that these are real and not scheduling: of PE's 236.4 us on `S[6]`,
**218.6 us is before the producing DVE instruction has even ended**; and DVE reaches the
`:407/:417` casts with a **median block of 31 ns** (it is not late at the slot, it is busy).
The compiler already interleaves 2-3 solve instances in PE's stream (pc 2076-2099 mixes
`:441/:447/:449` from different instances), so naive interleaving is not the missing idea.

---

# THE BIG FINDING OF THIS ITERATION: **DVE busy time converts to wall time at 0.7-1.0**

iter-1's rule "engine issue time / occupancy is not a lever" was measured on **SP** and **does not
transfer to DVE**. Measured with a new probe (`probe4.py`): shrink a set of DVE compute slots to
`num_elem[0] = 1` (keeps the slot, its semaphore, its ordinal and its wait — only the element count
drops, so results are wrong and this is profile-only), against the shipped 835.6 us:

| shrunk | slots | DVE busy | span | DVE us saved -> wall us saved |
|---|---|---|---|---|
| the `:407/:417` coupling casts | 384 | 438.8 -> 411.3 | **809.1 (-26.5)** | 27.5 -> 26.5 = **0.96** |
| every `COPY` + `CAST` | 998 | 438.8 -> 352.4 | **774.7 (-60.9)** | 86.4 -> 60.9 = **0.70** |
| every `TENSOR_TENSOR_ARITH_OP` | 928 | 438.8 -> 389.2 | **795.0 (-40.6)** | 49.6 -> 40.6 = **0.82** |
| every DVE compute slot | 2121 | — | **faults** (PSUM collision) | — |

**DVE is 438.8 us busy against a 835.6 us span (52.5%) and it is the lever.** Also, from row 2:
998 slots cost 234.4 us at 128 elements and 148.0 us at 1 element, so **a DVE instruction is
~148 ns of fixed overhead + ~86 ns of work for a 128-element row — about 2/3 of DVE's time is
per-instruction overhead.** DVE has 2151 compute slots (928 `TENSOR_TENSOR` 187.7 us, 518 `COPY`,
544 `CAST`, 163 `MEMSET`).

# WHAT TO TRY NEXT, in order

**Use the ceiling probe first, every time.** It costs 90 s per answer and it settled six candidates
this iteration, two of which the review had ranked top. `probe1.py` clears an engine's waits on a
semaphore range; `probe3.py` lowers a wait to an earlier ordinal; `probe4.py` shrinks an
instruction's element count. Pack, `profile.py --repeats 3`, **never verify or ship a probe.**

| probe | span | verdict |
|---|---|---|
| PE waits on `S[40..71]` cleared, on the **iter-1** NEFF | 775.9 | the DMA chain was worth **-100 us** -> became this iteration's edit |
| PE waits on `S[40..71]` cleared, on the **shipped** NEFF | 782.3 | **-54 us left**; equals the measured trigger->relay term to 1% |
| PE waits on `S[4]` (ACT) cleared | 822.2 | the whole PE<->ACT edge is worth only **-14.5 us** |
| PE `:408/:418` waits moved off the DVE cast onto the cast's **source** (191 of 295) | 836.5 | **0 — parallel path** |
| PE waits on `S[6]` at `:407-:419` cleared / DVE `S[5]` cleared / ACT `S[5..6]` cleared | **fault** | `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION` — these waits also order PSUM accumulation, so **lower** them, never clear them |

1. **Offload DVE compute slots to ACT with a semaphore stub. BUILT AND MEASURED AT -45 us, but
   it RACES — finishing this is the whole next iteration.**
   `835.6 -> 790.6 us (-5.4%)` with DVE 438.8 -> 392.8 and ACT 346.6 -> 406.8 (the two engines end
   up balanced, which is why this is close to the right size). `verify.py --runs 3`: **FAIL, the
   3 runs are not bit-identical and output_0/1 contain 73216/146432 NaNs — a race.** Treat the
   -45 us as a ceiling-probe number, and fix the race.
   * **What works (device-verified, keep all of it).** `scratch/offload.py`:
     - copy the DVE slot verbatim into `Activation0.bin` (identical op, source **and destination
       address**), clearing its `events_update_*`;
     - after it, an `EVENT_SEMAPHORE` with `events_update_mode = SEM_ADD_IMM_COMPLETE`,
       `events_update_idx = 104`, `events_semaphore_value = 1` — **a fresh semaphore index past
       the 103 in use, so no existing ordinal sequence changes anywhere**;
     - **replace the DVE slot by one `EVENT_SEMAPHORE`** carrying: the original slot's primary
       wait, `events_extended_wait = S[104] >= k`, and
       **`events_update_mode = SEM_INC_COMPLETE` on `S[6]`**. `S[6]`'s total and order are then
       exactly preserved (`semaphores.py` 1528/1528) and DVE pays ~40 ns instead of ~215 ns.
     - place the ACT copy after the last ACT slot whose profile `start_ts` precedes the DVE slot's
       `start_ts` (monotone in k, so ACT's order matches the stub order).
   * **Two encoding facts this cost a device run each — do not rediscover them:**
     - **`events_extended_update_mode` does not work.** A stub using
       `events_extended_update_mode = SEM_ADD_IMM_COMPLETE` on `S[6]` leaves `S[6]` one increment
       short (`semaphores.py` reports 1528 waited / 1527 incremented) and the whole kernel
       mis-numbers. Only the **primary** update channel updates. Use `SEM_INC_COMPLETE`, whose
       +1 does **not** consume `events_semaphore_value`, so the primary wait can still use it
       (this NEFF's own compute slots prove it: `('CAST','SEM_INC_COMPLETE',6,224)` where 224 is
       the wait value).
     - **Pool cannot execute a DVE `CAST`.** The identical edit hosted on Pool verifies as
       `max_abs_err 2.327e-02` / `4.528e-01` — deterministically wrong, bit-identical across two
       different semaphore encodings, so Pool runs the slot but not the same way (engine-specific
       microcode tables, chapter 1/3). **ACT is the only proven host**; it already runs 610 `CAST`s.
   * **The race, and exactly how to fix it.** The stub preserves the *forward* (RAW) edge —
     consumers waiting `S[6] >= v` still see the data. What is lost is the **anti-dependency
     (WAR/WAW) that DVE's in-order stream provided for free**. Confirmed statically: the 224
     offloaded casts write only **25 distinct destination addresses**, the hottest **reused 30
     times** (`Wj`/`Uj` are fresh `nl.ndarray`s per loop iteration but the compiler recycles ~25
     tiles). So ACT can run cast k+1 and overwrite the tile while PE is still reading version k,
     and nothing in the edit stops it.
     **The fix, and why the obvious shortcut does not work**: gating the ACT copy on "DVE reached
     the stub" (`S[6] >= n6(p)`, where `n6(p)` counts `S[6]` increments among DVE slots before p)
     is sound but starts ACT exactly when DVE would have, so DVE then blocks on ACT and the gain
     is zero or negative. The gate must instead be **the previous reader of the destination**:
     for each offloaded cast with destination D, find the last instruction *on any engine* that
     reads D before it (profile time order) and wait on that engine's semaphore ordinal. A `CAST`
     has **no extended events block** (iter-1 §2 checked every struct — only `EVENT_SEMAPHORE`
     does), so it cannot carry both that gate and its original wait: **insert an
     `EVENT_SEMAPHORE` before the ACT copy** carrying the two waits (primary + extended) and clear
     the copy's own wait — iter-1's trigger-conjunction pattern. Budget: 3 ACT slots + 1 DVE stub
     per offloaded instruction, so ACT grows ~9 us of `EVENT_SEMAPHORE` overhead for 224 slots,
     against the 48.7 us of DVE work moved. Readers to resolve: PE `LDWEIGHTS`/`MATMUL`
     (read-set code already in `scratch/retarget.py`) **and** ACT/DVE/Pool `CAST`/`COPY`/
     `TENSOR_TENSOR`/`TENSOR_SCALAR` sources — do not assume PE is the only reader.
   * **Process lesson that matters more than the edit**: the **1-slot** version of this edit
     verified **bit-identical** and `semaphores.py` was clean at 224 slots. Only
     **`verify.py --runs 3`** caught it, via the "all runs bit-identical" check. A single
     `verify.py` run is not evidence for an edit that moves work between engines.
   * Inventory for next time: 324 DVE `CAST` slots qualify as candidates (increment `S[6]`, carry
     a primary wait, non-PSUM source) for 67.0 us of DVE duration; the 224 whose wait is on `S[6]`
     (48.7 us, almost all `:407/:417`) are the set measured above. Offloading **all 324** (which
     includes 8 waiting on `S[4]`, ACT's own semaphore, and 63 on `S[3]`) **fails to run at all** —
     an ACT slot must never wait on `S[4]`, and the 63 `S[3]` (Pool) ones are the safest subset to
     try first since ACT cannot be blocked behind itself by them.

2. **Delete the 384 `:407/:417` DVE casts by making `Wg_grp`/`Ug_grp` bf16 at their producer.
   Sized at exactly -26.5 us by probe px1** — the only candidate in this file with a measured
   number attached. `:464/:465` (`tensor_copy(dst=Wg_grp[:, bi*D], src=yW)`, DVE PSUM fp32 ->
   SBUF fp32 `COPY`) becomes a same-cost `CAST`, and then `Wg[bj]` is already a contiguous bf16
   `[16,128]` run the coupling matmul can stream, so `:407/:417` vanish. Numerically free
   (`:407/:417` already round to bf16, and the W/U regroup path rounds to bf16 too — s128 §2.5).
   **The prize is DVE busy time, not PE idle**: probe px3 showed PE's 96 us of idle at `:418` is a
   parallel path worth 0. Costs: the row stride halves (2048 -> 1024 B), so the 160
   `qSPSpillReload1` regroup transfers move (32 of them already cast fp32->bf16 and become plain
   copies) and an SBUF liveness check is needed; absorb the removed `S[6]` increments with the
   `SEM_ADD_IMM_COMPLETE` stub from candidate 1 instead of renumbering.
3. **Merging adjacent DVE instructions into wider ones: mostly CLOSED, scanned statically.**
   `scratch/dvemerge2.py` on the shipped NEFF: of 2350 adjacent DVE pairs, **637 have an identical
   signature** (same opcode, dtypes, `num_active_channels`, ALU op, `step_elem`, `num_elem` —
   compare everything except `*_addr_immediate` and `*_addr_reg_*`, which are just the top bytes of
   the same immediate). Every one of the 637 has a free `num_elem` dim, so the merge *encoding* is
   always available (set some dim j>=1 to size 2 with that pattern's own `step = off_b - off_a`;
   the deltas need **not** agree between src0/src1/dst, each pattern has its own `step_elem`).
   The blocker is **the second member's wait**: only **96** pairs have the second member's wait
   absent or dominated by the first's (13 `MEMSET`, 26 `COPY`, 57 `CAST`); **457 + 80 = 537 are
   blocked by a real dependency** the second member waits on, so merging them would advance work
   past its producer (iter-1 §5's hang class). 96 pairs x ~148 ns = ~14 us of DVE busy
   = **~10-14 us of span, right at the ~8 us claimable threshold** — do it only as a top-up after
   candidates 1 and 2, and cell-verify the merged access pattern. 79 of the 96 need no
   `S[6]` stub at all (the second member does not increment `S[6]`).

4. **The last DMA hop (-54 us ceiling) is pure fixed latency. Probably dead.** The `Ball` hop
   (`A_full` partitions 64..127 -> partitions 0..15) **cannot be removed**: s128 §2.1 showed a PE
   stationary operand is indexed by its *absolute* SBUF partition and every `LDWEIGHTS` here has
   `addr_marker 0` (partition < 64) — that copy is why the operand is loadable at all. And it
   cannot be made faster: from `DmaPacket`, a `Ball` transfer is **8 KB = 4 packets of 2048 B at
   84-107 ns each**, while `trigger -> packet start` is a rock-solid **722 ns median (p90 760,
   max 880)**. More rings would add a completion, not remove latency.
   *(Trap: `DmaPacketAggregated.duration_ns` reads 44-52 **us** with a negative `start_ts -
   trigger` for the 160 `qSPSpillReload1` transfers. That is the profiler mis-aggregating iter-1's
   hand-built `num_tiling_dimensions: 1` descriptors — ignore it; use `DmaPacket` or the relay's
   `end_ts`.)*
5. **Do not spend budget on**: everything in "REFUTED" above and the whole probe table; everything
   iter-1 closed (trigger placement early/late, producer-gated triggers, SP occupancy, 1:1
   DGE->queued conversion, merging `qSPSpillReload0`, merging transfers past ~10 per ring,
   splitting a queue for ring depth, moving the staging relays to Pool); and everything s128 closed
   (`LDWEIGHTS transpose_mode` as a `dma_transpose` replacement, adjacent-`LDWEIGHTS` dedup, the
   `S[24..39]` ring throttle, `fabric_path`/queue type, transpose+cast in one descriptor, queued
   DMA reading PSUM). Also closed: **reduced precision of the matmul operands** — output_1 sits at
   82% of `max_abs_err_limit`, and *raising* precision (fp32 operands) changes what the kernel
   computes.

# Device budget used
~2h: 2 profiles of the start NEFF (one with `--parquet`), 3 `verify.py`, 1 `verify.py --runs 3`,
13 `profile.py` (8 of them ceiling probes, 4 faulted) and 1 hang (70 s); plus 4 verifies of the offload experiment (3 of them FAIL). `verify.py` ~55 s, `--runs 3` ~2 min,
`profile.py --repeats 5` ~80 s, `--parquet` adds ~20 s. Spread on this kernel is 0.56-0.73%
(+-3 to 6 us), so **nothing under ~8 us is claimable**; use `--repeats 5` and compare bands.

# Reusable scripts (in `scratch/`, reverted between iterations — rebuild from this file)
`probe1.py` (**the ceiling probe**: clear an engine's waits on a semaphore range),
`fuse.py` (the shipped edit: compose + move the triggers, with the ring-order assert),
`fuse2.py` (the generic any-queue version), `chk.py` (is a containment pair a real chain?),
`retarget.py` (sound PE wait retargeting; correct but worth 0), `attr.py` (per-engine idle by
semaphore and NKI line), `tsplit.py` (a PE DMA stall split into trigger-issue / transfer / relay),
`split.py` (the misleading arrive-vs-transfer split — read §1 before trusting it),
`plan.py` (head-of-line demotion planner), `dveplan.py` (same, sized for compute engines),
`probe3.py` (lower a wait to an earlier ordinal), `probe4.py` (**shrink a DVE op's element count — the DVE-work probe**), `chain2.py`/`chain.py` (transfer producer-consumer graph), `dve.py` (is the producer late, or the
engine?), `lines.py` (instruction cost by engine/NKI line/opcode), `dvemerge_scan.py`/`dvemerge2.py` (the
DVE adjacent-pair merge scan), **`offload.py` (the DVE->ACT offload with the semaphore stub —
`--host`, `--ops`, `--waitsem`, `--n`; the -45 us racy edit, and the base for candidate 1)**.

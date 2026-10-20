# iter-1 — removed the DRAM staging round-trips, then rebalanced the DGE hosts

## Result: 67.525 -> **61.905 us (-8.32%)**, output bit-identical
(shipped NEFF measured three times: 61.812 / 62.074 / 61.905; see §10)

| step | edit | total_exec_time_us | vs base |
|---|---|---|---|
| 0 | baseline (re-measured twice: 67.568, 67.525) | 67.525 | — |
| 1 | fuse the **var8** round-trip (queued reload) | 67.659 | +0.2% (nothing) |
| 2 | + fuse **var10** and **var9** (DGE reloads) | 65.501 | **-3.00%** |
| 3 | + fuse **var12** (save-side rewrite) | 64.334 | **-4.73%** |
| 4 | + fuse **var11** (save-side, incl. its fp32->bf16 cast) | 62.716 | **-7.12%** |
| 5 | + move 2 ACT DGE saves to Pool / Sync | 61.821 | **-8.45%** |
| 6 | + move 3 more ACT DGE saves to Pool / Sync | 61.812 | -8.46% (neutral vs step 5, kept) |
| 7 | hoist the Pool store DGEs' waits into preceding `EVENT_SEMAPHORE`s | 62.513 | **refuted, reverted** (§8) |

`verify.py --runs 3`: PASS, bit-identical across runs; `max_abs_err` **identical to the baseline
to every printed digit** (output_0 6.092e-05 = 11% of limit, output_1 4.494e-04 = 83%).
`launch_rules.py`: PASS. `semaphores.py`: no over-wait at any step.
Members changed: `sg00/{Activation0.bin, DVE0.bin, Pool0.bin, SP0.bin, SP0.json}`.

---

## 1. The iter-0 review's "no staging round-trips here" is WRONG — there were five

The review checked `DmaPacketAggregated` and concluded all 33 transfers were SB->SB. That table
only lists **queued** transfers. The staging *saves* are **DGE instructions**
(`PSEUDO_DMA_DIRECT2D` whose dst is a `tmp-buf` var), which never appear there. `def.json` has
five tmp-bufs and every one of them was a pure SBUF -> DRAM -> SBUF staging hop:

| var | size | save(s) | reload | cost on the critical path |
|---|---|---|---|---|
| var8  | 32 KB  | ACT slot 37 (`:317`)        | **queued** id 5 (`:340`)  | 4.6 us |
| var9  | 128 KB | ACT slot 38 (`:367`)        | SP DGE 11,12,14,15 (`:378`) | ~2.4 us |
| var10 | 128 KB | ACT slot 26 (`:368`)        | SP DGE 6,7,8,10 (`:382`)  | ~2.1 us |
| var11 | 128 KB | ACT 82,83,84,86 (`:483`)    | Pool DGE 21 (`:494`)      | ~1.5 us |
| var12 | 128 KB | ACT 88,89,90,94 (`:487`)    | SP DGE 37 (`:495`)        | **4.5 us** |

**Always enumerate the DGE descriptors** (decode every `PSEUDO_DMA_DIRECT2D`, resolve
`*_start_addr_addr_var_var_id_lo` against `def.json`'s var list and `DGETable0`) before deciding
a kernel has no round-trips. `scripts/alldge.py` in this worktree does it in one line per DGE.

## 2. Two fusion directions. Pick by which side fits the dimension budget

`DMA_DIRECT2D` patterns are **strictly 2-D** (`src_num_elem`/`src_step_elem` are 2-element
arrays). A *queued* descriptor has 4 dims. That alone decides which end you rewrite.

### (a) reload-side rewrite — var8, var9, var10
Save is always `src=SB off=So step=[S,1] num=[N,1] es=E` -> `dst=VAR off=0 step=[E,1] num=[N,1]`,
i.e. DRAM byte `i*E + c` **is** SBUF byte `So + i*S + c`. So for each reload dim with step `s`:
`s % E == 0` -> new step `(s//E)*S`; else require `s + es <= E` and keep `s`.
New source offset `= So + (F//E)*S + (F%E)`, source becomes `SB` (`var_id 0`).
Then **retarget the reload's wait** from the save's completion semaphore to *the save's own wait*
(the compiler's own statement that the SBUF source is ready), and neutralise the dead save with an
`EVENT_SEMAPHORE` that keeps that wait. Both halves are needed; (a) without the wait retarget is
worth 0 (same finding as `gdn_cte_b32_s128` iter-1).

### (b) save-side rewrite — var11, var12  (new here; the s128 run only did (a))
Here the *reload* is linear (`num=[64,1] step=[M,1] es=M`) and the *saves* are 2-D, so the
composed reload map needs 4 dims — impossible for a DGE. Rewrite the **saves** instead: reload
`src=VAR off=0 step=[M,1] num=[N,1] es=M` -> `dst=SB off=D step=[J,1] es=M/ratio` means DRAM byte
`j*M + m` is SBUF byte `D + j*J + m/ratio`. For each save: dst step `s` -> `(s//M)*J`,
dst offset `F` -> `D + F/ratio`, `dst_elem_size /= ratio`, `out_dtype = reload's out_dtype`.
`ratio` = the reload's own dtype conversion (var11: fp32->bf16, ratio 2; var12: 1).
**The dtype conversion moves into the save and the numbers come out bit-identical.**
Then the reload is dead: replace it with an `EVENT_SEMAPHORE` that keeps its wait and does
`SEM_INC_COMPLETE` on the reload's old semaphore, and lower the consumer's wait **16 -> 1**
(a DGE adds +16, one instruction adds +1). Keep the compiler's own fence instructions in front of
it untouched — they are what guarantee all four saves landed (and note SP slot 35 carries a
*second* wait in its extended events block, so the fence really does cover all four).

## 3. Three mechanisms verified on the device for the first time

1. **A DGE can be SBUF->SBUF, including with a dtype conversion.** No compiler-emitted DGE in
   this NEFF was SB->SB; all 9 rewritten ones work and the output is bit-identical.
2. **A DGE keeps its own `semaphore` field when moved to another host engine.** Pool- and
   SP-issued DGEs correctly increment `S[44]`, `S[45]`, `S[46]`, `S[48]`, `S[49]` — semaphores of
   **qActDynamicHW**, a queue they do not own. (`s2304` iter-2 swept "DGE host ratios"; this
   confirms the mechanism on this kernel.) This is the cheapest way to take work off a locally
   saturated engine: a DGE costs **0.60-0.70 us of its host engine** almost independently of size.
3. **An instruction stream can be grown by inserting slots mid-stream.** Pool 32 -> 35 slots,
   SP 62 -> 64. Nothing else in the NEFF references slot indices (no branches in the kernel
   streams), and `neff.py pack` recomputes sizes/MD5s.
   *Side effect:* the `.dbg` debug info goes stale, so `Instruction.parquet.nki_source_location`
   is shifted for every slot after an insertion point. Re-derive slot->meaning from the decoded
   stream, not from the profile's `nki` column, after any insertion.

## 4. Tooling that made this possible (rebuild these first next iteration)

* **A backward critical-path walk** (`scripts/cp.py`): for every slot, candidate predecessors are
  (previous slot on the same engine, its end) and (first time each of its waits was satisfied);
  the binding one is the latest. Semaphore reach times come from `SemaphoreUpdate.parquet`
  (`id` is either `3` or `"S[24] (qSyncSpillReload0)"`; take the **first** timestamp whose value
  >= v). Producer rules: `S[3]/S[4]/S[5]/S[6]` = the v-th slot of Pool/ACT/PE/DVE with
  `events_update_mode != NONE`; DGE sems `S[8..23]` Pool, `S[40..55]` ACT, `S[56..71]` SP;
  queued `qSPSpillReload0` = `S[24..39]`, transfer id k signals `S[24 + k%16]` by +16.
  Every single edit in this iteration was chosen from this walk; the engine-busy table was
  useless (nothing above 56%).
* **Pitfalls.** `semaphores.py` ignores the **extended** events block, so a semaphore read only
  by an extended wait shows `max_wait 0` and `waiters 0` (e.g. `S[16]`, `S[20]`). Write your own
  scan over `events_extended_wait_*`. `Instruction.parquet.evt_wait_time_ns` is garbage (a
  constant 10.417 on most rows) — use an `EVENT_SEMAPHORE`'s `duration_ns` as its block time.
* `json.dumps(J, separators=(",",":"))` reproduces `SP0.json` byte-identically — assert that
  before editing so the diff stays minimal.
* `isa.Instruction.set` takes a whole list for array fields. Setting
  `*_start_addr_addr_var_var_id_lo` + `*_start_addr_addr_var_offset` is enough to repoint an
  address; the `_immediate`/`_table_*`/`_tbl_offs_*` names are aliases of the same 8 bytes.
* Scripts are parked in `scripts/` in this worktree (not preserved): `alldge.py` (all DGE
  descriptors), `cp.py` (critical path), `tl.py` (timeline window), `gaps.py` (low-occupancy
  stretches), `waits.py` (all waits incl. extended, by semaphore), `join.py` (stream x profile),
  `edit1..6.py` (the edits, each self-asserting).

## 5. Measured costs on this kernel (for budgeting the next edit)

| item | cost |
|---|---|
| DGE instruction issue, on its host engine | 0.60-0.70 us (64 desc: 0.65; 256 desc: 0.79) |
| DGE slot end -> its semaphore visible | 1.3-2.3 us |
| queued transfer: trigger instruction | 0.04-0.07 us on SP |
| queued transfer: trigger -> semaphore visible | 1.2-2.6 us |
| cross-engine semaphore handoff on the path | ~24 ns |
| run-to-run spread, 7 captures | 0.5-1.4% (±0.3-0.9 us) |

## 6. What is still open, ranked

1. **The 47-56 us ACT window.** 4 DGE saves are still hosted on ACT (slots 82, 88, 94 of the
   `:483`/`:487` groups). Converting them to **queued** transfers on `qSPSpillReload0` would
   replace 0.65 us of issue each with a 0.05 us trigger — and queued descriptors have 4 dims, so
   they can express patterns a DGE cannot. Append ids 33.. (semaphore = `semaphore_set[id%16]`,
   so the waits become `>= 48` on ring `id%16`; check the ring's trigger order stays program
   order) and retarget the existing fence waits. ~2 us of ACT time, worth maybe 1-1.5 us.
2. **Find out why edit6 did not land.** It moved the whole 50-56 chain 1.1 us earlier
   (ACT slot 94 ends 53.71 vs 54.85; `S[50]` at 55.51 vs 56.43; PE `:597` at 56.42 vs 57.14) and
   ACT busy fell 1.5 us, yet the end did not move at all. Something between 56 and 74 us became
   binding. Walk the path from the last slot on the *current* NEFF — that is where the next 1-2 us
   is, and it is the single most valuable thing to do first.
3. **The head is 3.9 us of pure DMA latency.** Pool slot 1 (gate load, `:191`) issues at 12.9,
   ends 13.7, and `S[8]` is not visible until 15.95; the first DVE `CAST :193` starts at 15.97 and
   the whole compute chain hangs off it. Nothing is reorderable — slot 1 is already first. The
   only idea left is to make that one transfer's completion cheaper (it scatters 256 bf16 elements
   into 64 partitions as fp32).
4. **var8's fusion is still worth 0 on its own** (step 1 above: +0.2%). Its gain only appeared
   once var9/var10 were fused too. Do not re-test it in isolation.

## 7. Closed — do not spend budget on these

* **Swapping the two tail stores** (Pool `:628` state store and the last `:623` output store).
  The output store is blocked 0.64 us by the state store's issue, but the final fence
  (Pool `EVENT_SEMAPHORE`, primary `S[21]>=16` **and extended `S[20]>=16`**) waits for both, and
  the state store is 4x the bytes — putting it second makes *its* completion later. Analysed from
  the profile, net negative, not worth running.
* **Merging the 4 `beta` input loads** (Pool slots 8-11, `:207`). They *are* mergeable into one
  2-D DGE (src `num=[64,4] step=[4,256]`, dst `num=[64,4] step=[262144,256]`; then re-point the
  ACT waits on `S[11]`, `S[12]`, `S[13]` to `S[10]`) and it would save ~1.95 us of Pool time. But
  measured: **nothing downstream is gated by them** — `S[10..13]` are waited at 18.8/21.3/21.8/24.1
  while the merged load would land at ~17.5, and the 13.9-16.0 "only Pool busy" stretch is the
  whole machine waiting for the gate load's DMA anyway. Pool busy is not the constraint there.
* **The 4-us SP `EVENT_SEMAPHORE` blocks** (SP slot 33 in the baseline). They look like the
  biggest single stall in the profile and they are not on the critical path: the primary wait is a
  long-satisfied ring throttle and the real condition is in the **extended** events block; in
  every case the instructions queued behind it were blocked by their own waits at the same time.
* **var11 and var12 reload-side fusion.** Needs 4 dims in a DGE. Impossible; use direction (b).
* **Hoisting a Pool DGE's wait into a preceding `EVENT_SEMAPHORE`** so the DGE issues back-to-back
  (edit7): **+0.4 to +0.7 us, refuted** (§8). The ~0.5 us gap before a Pool DGE is not caused by
  its wait, and is probably not real at all — see §8 for why a Pool DGE's profile row is not its
  issue window.

---

## 8. A Pool DGE pays ~0.5 us of start-up that an ACT or SP DGE does not

Found at the very end, measured only on the **baseline** profile (clean `compiler_pc`), not yet
exploited. For each DGE, the gap between "its wait was satisfied" (`SemaphoreUpdate` first-reach)
and its profile `start_ts`:

| DGE | host | engine idle before it | gap |
|---|---|---|---|
| Pool slot 1 (gate load `:191`) | Pool | yes (since `SET_ORDERING_MODE` ended 12.232) | **0.65 us** |
| Pool slot 24 (`:623` output store) | Pool | yes (since 65.440) | **0.46 us** |
| Pool slot 28 (`:628` state store) | Pool | yes (since 73.920) | **0.49 us** |
| Pool slots 8-11 (`beta` loads) | Pool | no, back-to-back | 0.005 us |
| SP slot 6 (`:382`) | SP | **yes, idle 10.7 us** (since 17.118) | **0.02 us** |
| ACT slot 26 (`:368`), ACT slot 37 (`:317`) | ACT | no | 0.02 us |

SP idle for 10.7 us still issues its DGE 20 ns after the wait clears, so this is **not** a generic
"cold queue" effect — it is specific to the Pool host. (Consistent with the `s128` measurement of
per-host DGE issue cost 581/698/595 ns for ACT/Pool/SP: Pool is the worst host on both axes.)

### I tried to cash this in and failed — read this before trying again

**Hypothesis tested (edit7, reverted):** the gap appears only after a *wait*, so hoisting each
store DGE's wait into a cheap `EVENT_SEMAPHORE` placed immediately in front of it (leaving the DGE
with no wait, so it follows its predecessor back-to-back) should remove it. Mechanically perfect —
identical wait set, identical semaphore table, `verify.py` PASS, numbers bit-identical — and
**measured 62.513 us vs 61.81/62.07, i.e. +0.4 to +0.7 us. Refuted.**

**So the cause is not the wait.** The likeliest remaining explanation is that it is a *measurement
artifact*: the profiler does not timestamp a Pool DGE row the way it timestamps other rows. Direct
evidence from the baseline profile — Pool slot **1** (the DGE) is reported at 12.887-13.680 while
Pool slot **2** (a `MOVE`, which comes *after* it in program order) is reported at 12.858-12.902.
A Pool DGE row therefore cannot be the instruction's issue window; it looks like the DMA-side
window. If so there is no 0.5 us to recover on Pool and the whole of §8 is a mirage.

**If you still want to test it,** the one experiment that distinguishes the two stories is moving a
`:623` output store from Pool to **SP** (§3.2/§3.3 mechanics, already device-verified here). If the
0.5 us is real it comes back as ~0.5 us of total time; if it is an artifact, nothing moves. Cost
~15 min. I ranked it below §6.1 and §6.2 after edit7, and ran out of budget for it.

## 9. WARNING: `compiler_pc` in the profile is wrong after you insert slots

After edits 5 and 6 inserted 3 slots into `Pool0.bin` and 2 into `SP0.bin`, the profiler's
`Instruction.parquet.compiler_pc` for those engines still follows the **stale `.dbg`** numbering:
for Pool it equals `my_slot - 3` for every slot at or after the first insertion point (verified by
matching the opcode/duration sequence: profile pc23 d=0.687 is the DGE at my slot 26, etc.), and
`nki_source_location` is `NaN`/shifted the same way. Any slot-indexed join is silently wrong for
the engines you grew, which is why I stopped editing rather than target blind.

**Fix this first next iteration** (30 min, unblocks everything else): either
(a) rewrite `sg00/debug_info_backend_*.dbg` / `debug_info_asm_*.dbg` to match the new streams
    (`neff_doc/tools/dbginfo.py` decodes them; chapter 11.5 says they are optional), or
(b) join on the decoded opcode + duration sequence instead of `compiler_pc`, or
(c) prefer *replacing* slots over inserting them: there are now several 40 ns no-op
    `EVENT_SEMAPHORE`s on ACT (slots 26, 37, 38, 83, 84, 86, 89, 90) that a future edit can
    overwrite in place, keeping all stream lengths and therefore all slot indices fixed.
Option (c) is the cheapest and is how I would do the §8 edit: the five Pool store DGEs can be
*swapped* with no-ops already present on the target engine instead of inserted.

## 10. Final measurement of the shipped NEFF

Three independent 7-capture runs: **61.812**, **62.074** and **61.905 us** (median 61.905,
= **-8.32%**; the run-to-run spread of the 7-capture median itself is ~0.26 us = 0.4%).
Baseline, re-measured twice this iteration: 67.568 and 67.525 us.
The gap between the two bands is ~5.5 us, roughly 20x the spread, so the win is not in doubt.

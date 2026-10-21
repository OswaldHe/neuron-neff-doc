# iter-1 — removed all five DRAM staging round-trips, moved the gate load off Pool, unblocked two queued triggers

## Result: 48.519 → **43.504 us (−10.3%, 1.115x)**, output bit-identical

| step | edit | median (7 captures) | vs prev |
|---|---|---|---|
| 0 | baseline, re-measured twice this session | **48.702**, **48.424** | — |
| A+B+C+D | fuse the var9/var10/var11/var12 staging round-trips | **46.338** | −4.9% |
| +E | fuse the **var8** round-trip (queued reload → SB) | **45.066** | −2.7% |
| +F | gate-load DGE `Pool[1]` → `SP[1]` | **44.484** | −1.3% |
| +G | var12 save pair `ACT[59]/[61]` → Pool | 45.636 | **+2.6% REFUTED, reverted (§5)** |
| +H | hoist the id16/id17 trigger pairs above the var12 fences | **43.779 / 43.679** | −1.6% |
| +I | output_0 store `Pool[20]` → end of SP | **43.504 / 43.463 / 43.530** | −0.5% |

Shipped NEFF measured **three** independent 7-capture runs: 43.504 / 43.463 / 43.530
(21 captures, 43.229–43.892). Baseline, re-measured twice in the same session:
48.702 and 48.424 (14 captures, 48.207–49.025). **The bands are 4.5 us apart, ~10x the
spread**, so the win is not in doubt.

`verify.py --runs 3`: PASS, all 3 runs bit-identical, `max_abs_err` **identical to the baseline
to every printed digit** (output_0 6.119e-05 = 17% of the limit, output_1 2.864e-04 = 82%).
`launch_rules.py` PASS. `semaphores.py` no over-wait at any step.
`neff_diff`: 5 of 34 members changed — `sg00/{Activation0.bin, DVE0.bin, Pool0.bin, SP0.bin,
SP0.json}`; 47 slots touched; SP0.bin grew 35 → 37 slots. sha256 `cacd5b79…`.

---

## 0. The single most useful thing: this kernel is the s256 sibling, scaled down exactly

`neff_doc/experiments/gdn_cte_b1_s256_d128_lnc1` has the **same five tmp-bufs with the same
descriptor shapes**, at half the size. Its `optimized.neff` is on disk and is a working,
device-verified reference for every edit in §1–§2 — **unpack it and diff it against its
`baseline.neff` before writing any code** (`scripts/sdiff.py` below does it in one line).
That turned a 2-hour design problem into a 40-minute transcription.

**Also: the b32_s128 notes' §4.1 "a DGE pattern allows only one partition-crossing dimension on
the SBUF side" DOES NOT APPLY HERE.** It says `src_step_elem = [4194304, 262144]` is impossible;
the s256 optimized NEFF uses exactly that on 8 SB→SB DGEs and it is bit-identical on the device,
and so are mine. b32's refutation was about a *different* composition (its reload's dst pairing
forced two partition dims at once). **Check a "closed" claim against the shipped artifact.**

## 1. The structure (identical to s256, verify with `scripts/dge.py`)

| var | size | save(s) | reload(s) | direction used |
|---|---|---|---|---|
| var8  | 16 KB | `ACT[21]` (wait `S[6]>=13`, sem 42) | **queued id 3** (sem 27), trigger `SP[7]` | (a) queued, §2c |
| var9  | 64 KB | `ACT[20]` (wait `S[4]>=10`, sem 41) | `SP[6]`, `SP[8]` DGE | (a) reload-side |
| var10 | 64 KB | `ACT[14]` (wait `S[4]>=5`, sem 40)  | `SP[4]`, `SP[5]` DGE | (a) reload-side |
| var11 | 64 KB | `ACT[56]` (sem 43), `ACT[57]` (sem 44) | `Pool[16]` DGE (fp32→bf16) | (b) save-side |
| var12 | 64 KB | `ACT[59]` (sem 45), `ACT[61]` (sem 46) | `SP[24]` DGE | (b) save-side |

All five semaphores had **exactly one waiter**, which is what makes the edits cheap.
Every fused reload is fenced by the pattern `ES(second save's sem)` immediately before
`DGE/consumer(first save's sem)` — `Pool[15]`+`Pool[16]` for var11, `SP[23]`+`SP[24]` for var12.
**Leave those fence ES slots alone**; they are what guarantees both halves landed.

## 2. The edits, with the exact arithmetic (all bit-identical)

### (a) reload-side, var9 + var10 — `SP0.bin` only
Save is `src=SB off=So step=[S,1] num=[64,1] es=E` → `dst=VAR off=0 step=[E,1] num=[64,1]`, so
DRAM byte `i*E+c` **is** SBUF byte `So + i*S + c`. For each reload dim with step `s`:
`s%E==0` → `(s//E)*S`, else keep `s` (and assert `s*count <= E`).
New `src` offset `= So + (F//E)*S + (F%E)`, `src var_id = 0` (SB).
Concretely `E=1024, S=262144`: `[16384,1024] → [4194304,262144]`; var10 `So=16796928`
(F=0 → 16796928, F=512 → 16797440), var9 `So=19712` (→ 19712 / 20224).
Then **retarget the leading reload's wait** from the save's completion to the save's own wait and
replace the dead save with a bare `EVENT_SEMAPHORE` keeping that wait. The second reload is
already waitless and rides on the first.

> **I used `S[4]>=6` for var10, not the compiler's `S[4]>=5`.** `ACT[13]` is the 6th `S[4]`
> incrementer and is the last ACT slot before `ACT[14]`, so `>=6` is the *full* ACT-program-order
> condition and is strictly safer than the compiler's own wait, at zero cost (`ACT[13]` completes
> immediately before). var9's `S[4]>=10` was already program-order-complete. **Do this
> routinely**: count the `S[4]` increments up to the save and use that number.

### (b) save-side, var11 + var12 — `Activation0.bin`, `Pool0.bin`, `SP0.bin`, `DVE0.bin`
Reload is `src=VAR off=0 step=[M,1] num=[64,1] es=M` → `dst=SB off=D step=[J,1] es=M/ratio`.
For each save: dst step `s → (s//M)*J`, dst offset `F → D + F/ratio`,
`dst_elem_size /= ratio`, `out_dtype =` the reload's `out_dtype`.
`M=1024, J=262144`; var11 `D=16896512 ratio=2` (fp32→bf16: `es 512→256`, `out_dtype FP32→BFLOAT16`,
offsets 16896512 / 16896768); var12 `D=16897024 ratio=1` (offsets 16897024 / 16897536).
**The dtype conversion moves into the save and the numbers come out bit-identical.**
Then the reload is dead → `EVENT_SEMAPHORE` keeping its wait and doing `SEM_INC_COMPLETE` on the
reload's old semaphore, and **lower the single consumer's wait 16 → 1** (a DGE adds +16, one
instruction adds +1): `SP[26]` `S[14]>=16→>=1`, `DVE[80]` `S[60]>=16→>=1`.

### (c) var8's queued reload — `SP0.json` + `SP0.bin`  (worth −2.7% HERE, unlike s256)
s256 iter-1 measured var8's fusion at **+0.2%, i.e. nothing**, and the review repeated that.
**On this kernel it is the second-biggest single win.** Measure, do not inherit.
Why: PE was idle 19.4 → 26.5 us waiting for it — `ACT[21]` save end 22.90 → `SP[7]` trigger 24.38
(1.48 us) → `S[27]` 26.47 (2.03 us) → `PE[13]`. 3.6 us of pure round-trip on the critical path.
The descriptor (`op:"copy"`, sizes/steps **in bytes**) is the b32_s128 shipped case verbatim:
`from_sizes [32,4,4,16]`, `from_steps [1,32,4096,256] → [1,32,4194304,262144]`,
`from_off 0/128 → 18944/19072`, `from "SB"`, `E=256 S=262144`. Keep `num_tiling_dimensions`,
`from_dtype` and the whole `to` side. Retarget `SP[7]`'s wait `S[42]>=16 → S[6]>=13`.
`json.dumps(J, separators=(",",":"))` reproduces `SP0.json` byte-identically — assert it first.

### (F) the gate-load DGE belongs on SP, not Pool — **−1.3%, and s256 declared this closed**
s256 iter-2 §5/§5a closed the gate-load lever ("moving `S[8]` earlier did not move the finish").
**That verdict does not transfer.** Here `Pool[1]`'s row occupies Pool until 13.24 us and
head-of-line blocks `Pool[3]`, `Pool[5]`, `Pool[6]` and the five q/k/v/beta input loads
`Pool[7..11]`. Moving the single DGE verbatim to a slot inserted at **SP index 1** (SP is idle
11.77–16.23) and leaving a bare no-op at `Pool[1]`:
* the gate transfer starts ~0.6 us earlier (Pool pays ~0.65 us of DGE start-up, SP ~0.02 — s256
  iter-2 §3 is right that this is **real**, not a profiler artifact);
* and Pool's whole head runs ~1.0 us earlier.
**Do not split it** (s256 §5/§5a: splitting across two hosts bought 641 ns locally and lost
0.28 us overall; both halves on one host bought 53 ns because one host = one queue).

### (H) two queued triggers were stuck behind fences they do not need — **−1.6%, the best us/effort**
Post-F SP order was
`[24] ES S[46]`, `[25] ES S[45]→S[60]`, `[26] ES S[24]`, `[27] trig id16 S[14]>=1`,
`[28] ES S[25]`, `[29] trig id17 S[6]>=53`.
`[24]/[25]` fence **var12**, whose only consumer is `S[60] → DVE[80]` (which has ~2 us of slack).
id16 reads SBUF per-partition bytes 119296–119551 and id17 reads 119552–119807 — that is
**var11**'s region, already guarded by `S[14]` (and `Pool[16]`'s ES sits behind `Pool[15]`'s
`S[44]` fence, so both halves landed). Neither touches var12's 119808–120831.
So reorder `[24..29] → [26,27,28,29,24,25]`: each trigger keeps its own **ring guard**
(`ES S[24]` / `ES S[25]`, i.e. the WAR guard for the previous transfer on that ring) immediately
in front of it and its own data wait. Both triggers fire 0.78 us earlier.

> **Reusable recipe.** On SP the compiler emits `ES(ring guard)` + `trigger` pairs and leans on
> SP program order. Any such pair whose data wait is already satisfied, and whose transfer's
> **source bytes** are disjoint from what the fences in front of it protect, can be hoisted.
> Check the source range from `SP0.json` `from_off`/`from_sizes`, not from the comment.

### (I) output_0 store `Pool[20]` → end of SP — +0.25 us, kept (3 measurements)
Appending to the end of SP shifts **no** slot index. ~0.43 us of Pool DGE start-up gap plus it
stops sharing `qPoolDynamic` with `Pool[19]` (the second gate store). Small and partly absorbed
by the ~1.7 us runtime epilogue (s256 iter-2 §3 saw 210 ns of fence movement → 14 ns of total);
measured 43.504 / 43.463 vs H's 43.779 / 43.679, i.e. it won both head-to-head pairs.

## 3. Mechanics / tooling notes

* All slots are 64 bytes on every engine, `header_inst_word_len = 16`, including
  `EVENT_SEMAPHORE`. Replacing a DGE in place is a straight 64-byte overwrite.
* Address bytes: `*_start_addr_addr_var_offset` is the u32 at bytes 0–3 of the 8-byte address,
  `var_var_id_lo` is byte 4, `var_marker = 0x10` is byte 7. Setting offset + `var_id_lo`
  (+ `var_id_hi`) is enough; every `_immediate` / `_table_*` / `_reg_*` name aliases the same
  8 bytes, so `neff_diff` will report a pile of cosmetic field renames — that is expected.
* **ALWAYS assert your no-op is bare on the *re-decoded* bytes** (s256 iter-2 §4: an inherited
  extended wait = guaranteed deadlock). `scripts/edit.py`'s `make_es()` does it.
* `compiler_pc == slot index` even after insertions (s256 iter-2 §0 corrects iter-1 §9).
  Confirmed again here after inserting into SP twice. Only `nki_source_location` goes stale.
* Normalise profile timestamps to `Pool[0] SET_ORDERING_MODE`; the runtime preamble moves ±0.6 us.
* `semaphores.py` ignores the **extended** events block, so `S[16]` shows `max_wait 0`
  (its only waiter is `Pool[21]`'s extended wait). Scan `events_extended_wait_*` yourself.
* Scripts in `scripts/` (not preserved, ~150 lines, 20 min to rebuild):
  `dge.py` every DGE resolved against `def.json`; `dump.py` a stream's waits/updates/extended;
  `waits.py` every semaphore's increments and waiters incl. extended; `sdiff.py` slot-by-slot
  diff of two unpacked NEFFs; `tl.py` timeline window; `edit.py` steps A–I, each self-asserting.

## 4. Where the remaining 43.5 us goes

Kernel span 44.06 us, **5.03 us of all-engine idle in 8 gaps**, no engine above 51%
(ACT 22.3, DVE 19.5, DMA 15.2, PE 12.0, Pool 9.3, SP 7.6). Every one of the 8 gaps is a
**queued-transfer or DGE completion latency on a serial chain**, 1.3–2.2 us per hop:

| gap us | unblocks | what it is waiting for |
|---|---|---|
| 1.244 | `SP[10]` | var8's queued id 3 (now SB→SB, still 2.2 us trigger→semaphore) |
| 0.363 | `PE[15]` | queued id 4/5 |
| 0.594 | `ACT[26]` | queued id 7 (`S[31]`) |
| 0.240 | `SP[28]` | var12b's DGE (`S[46]`) |
| 0.632 | `ACT[64]` | queued id 15 (`S[39]`), trigger `SP[23]` gated by `ACT[63]` |
| 0.608 | `ACT[69]` | queued id 18 (`S[26]>=32`), trigger gated by `ACT[68]` |
| 0.579 | `PE[141]` | queued id 19 (`S[27]>=32`), trigger gated by DVE |
| 1.369 | `Pool[21]` | the output_0 store's DMA (final fence) |

The 20 remaining queued transfers are all **SB→SB `WZXY` transposes** (`SP0.json`) — real work.
Only id 3 was a DRAM staging hop and it is now fused.

## 5. Refuted this iteration, with the number that closed it

### 5.1 Moving the var12 save pair off ACT (edit G): **45.636 vs 44.484, +2.6%. Closed.**
Moved `ACT[59]`/`ACT[61]` to two slots inserted after `Pool[14]` (whose `S[6]>=51` covers
`ACT[59]`'s own `S[6]>=50`), the leader carrying `S[4]>=31` as the ACT-program-order guard.
Correct and bit-identical; it lost for two measured reasons:
1. **The guard is later than the data.** On ACT, `ACT[59]` issues *during* `ACT[60]`
   (34.86 vs 35.52–35.96), so it never actually waited for `S[4]>=31`. Forcing
   `S[4]>=31` delayed it 0.7 us. **A waitless DGE's true guard can be weaker than
   "the previous signalling slot on its host completed" — the profile tells you which.**
2. **Pool adds its ~0.4 us DGE start-up gap**, and var12b's row ended 1.08 us later ⇒ `S[46]`
   1.15 us later ⇒ `SP[28]` 1.15 us later. That is the whole regression.
3. The prize was never there: freeing ACT of two DGE rows moved `ACT[62]`/`ACT[63]` only
   **0.38 us** earlier, not 1.26. **An ACT DGE row is mostly not exclusive ACT occupancy** —
   `ACT[58]`/`ACT[60]` run *concurrently* with `ACT[59]`/`ACT[61]`'s rows in the baseline.

### 5.2 Hoisting the id4–id12 triggers above `SP[10]`'s `S[27]` wait — 0 candidates. Closed.
The 9 triggers `SP[10..18]` all fire in one 0.65 us burst behind `SP[10]`'s wait on var8's
reload, and `SP[11..18]` are waitless. I checked every one: **all nine read SBUF partitions
0–31 inside var8's reload destination** (offsets 227072–228095 vs id3's writes
227072–227872). They genuinely need it. Nothing to hoist.

### 5.3 Freeing ACT's var11 pair instead — analysed, closed without a run.
`ACT[58..63]` cannot start before `S[6]>=50` clears at 34.8 regardless, and then
`ACT[59]→[60]→[61]→[62]→[63]` is a tight serial chain in which the DGE rows already overlap the
compute. Moving `ACT[56]`/`ACT[57]` to SP would leave `ACT[63]` *later* (37.2 vs 36.5).
**The whole 34.8–36.5 ACT window is closed to DGE rehosting.** (Agrees with s256 iter-2 §6/§7.)

## 6. Ranked candidates for iter-2

**Read §5.4 and §5.5 first — the two obvious follow-ons are already closed**, and the
`slack.py` scan in §5.4 is the tool to rebuild before anything else (5 minutes, and it answers
"is this instruction blocked by its wait or by its engine's program order?" for all 378 slots).

1. **Reshape var8's queued id 3 so the transfer itself is cheaper** (the 1.244 us gap). See §5.6:
   splitting it across rings is **dead** — all 21 transfers run on one `queue_idx`, and the cost
   is not two serialised descriptors but **one slow scatter: 1022 ns for 16 KB, vs 256 ns for
   another 16 KB transfer in the same NEFF.** The destination is 16 runs of 32 bytes per
   partition with 32-byte holes (`to_sizes [32,4,4,16]`, `to_steps [1,64,256,262144]`), which is
   what makes it slow. Merging dim 0 (step 1, count 32) into dim 1 (step 64) is impossible — the
   hole is real. So this needs the *consumers'* layout changed too (ids 4–12 all read it, §5.2),
   i.e. it is a data-layout change, not a descriptor edit. Low probability, high cost.
2. **Double-buffer the SBUF scratch that forces the WAR chain in §5.5** (ceiling ~1.2 us: the
   0.632 + 0.608 us gaps). `id15` must wait for `ACT[63]` only because it *overwrites* the
   buffer `ACT[63]` reads (offset 17664). Give `id15` a different `to_off` in a provably free
   SBUF region and re-point `ACT[64]`'s `src_mem_pattern_start_addr_addr_immediate` to match;
   then `id15` can be triggered as soon as its own source is ready. Same for `id18`/`ACT[68]`.
   **Needs an SBUF free-region map first** (`SbufUsage.parquet` is in the profile; the sibling
   iter-3 built one) — do not guess, a wrong region silently corrupts another tile.
3. **`Pool[19]` (second gate store) → SP** — now that `Pool[20]` moved, `Pool[19]` is the only
   DGE left on `qPoolDynamic` in the tail and is *not* binding (`S[17]` lands after `S[16]`).
   Expect ~0. Listed only so nobody spends a run on it.
4. **The head's 2.3 us gate-transfer latency** (`SP[1]` row end → `S[8]` → `DVE[8]`). 128 scattered
   2-byte reads expanded to fp32 across 64 partitions. Splitting is closed (§F). The only
   untried idea is making the *transfer* cheaper, e.g. expressing it as a queued descriptor
   (4 dims, and queued trigger→semaphore measured 1.2–2.6 us vs this DGE's 2.3) — speculative.

### 5.4 No more hoistable triggers exist — closed by a full scan, build this tool first
`scripts/slack.py`: for every slot with a wait, compute `start_ts − t_reach(wait)` from
`SemaphoreUpdate.parquet` (first timestamp whose `value >= v`; `id` is sometimes `"3"` and
sometimes `"S[24] (qSyncSpillReload0)"`, parse both). Slack > ~0.1 us ⇒ the slot is held by its
**engine's program order**, not by its wait, so it is a reorder candidate.
Run on the shipped NEFF: **every `PSEUDO_DMATRIGGER` has < 0.3 us of slack except `SP[4]`
(3.0 us, in the head, and the two triggers in front of it have no slack either).** The five
`EVENT_SEMAPHORE` ring guards show 23–26 us of slack — they are trivially satisfied WAR guards
and are *not* what blocks their trigger. So §H exhausted this lever on this kernel.
The other big-slack rows are PE `LDWEIGHTS`/`EVENT_SEMAPHORE` (2.5–8.4 us) and `DVE[12]`
(3.9 us), all blocked by their own engine's program order, i.e. they would need compute
reordering, not a wait change.

> **Gotcha that cost 10 minutes:** `pandas` resolves `df.sem` to `DataFrame.sem()` (standard
> error of the mean), not to a column named `sem`. `df[(df.sem==s)]` silently yields an
> all-False mask and the scan prints nothing. Name the column anything else.

### 5.5 The `ACT[63] → id15 → ACT[64]` serial chain is a **WAR hazard**, not a missing hoist
`SP[23]` (the id15 trigger) waits `S[4]>=32` = `ACT[63]`'s completion, and that looked like a
free 0.632 us. It is not. Decoding the addresses:
`id14` reads SB (96, 20224) → writes **(0, 16896)**; `ACT[63]` reads `src` offset **17664** and
`imm0_imm_ptr` **16896** (id14's output) and writes offset 20992; `id15` desc 1 **writes 17664**.
So id15 overwrites the buffer `ACT[63]` is still reading — `S[4]>=32` is a **WAR guard on SBUF
offset 17664**. The same shape holds for `id18`/`ACT[68]` (the 0.608 us gap).
These hops are only breakable by double-buffering (candidate 2 above), not by reordering.
**General lesson: before calling a wait "unnecessary", decode the consumer's `src` /
`imm0_imm_ptr` and the transfer's `to_off`. Several of this kernel's waits are anti-dependences,
and `semaphores.py` cannot see them at all.**

### 5.6 The queued-DMA cost model on this kernel (from `DmaPacketAggregated.parquet`)
This table has `trigger`, `start_ts`, `end_ts`, `block_id` (== transfer id), `queue_idx` and
`semaphore_id` per transfer — **21 rows, the whole queued picture in one read.** Use it instead
of guessing. Measured on the shipped NEFF (times normalised to `Pool[0]`):

| stage | cost |
|---|---|
| `trigger` → DMA `start_ts` | **0.71–0.77 us, flat** (0.709/0.713/0.720/0.769; var8's 0.966 is the outlier) |
| DMA `end_ts` → consumer start | 0.36–0.47 us |
| ⇒ one queued hop, end to end | **1.3–2.4 us** |

* **`queue_idx` is 3 for all 21 transfers** even though `qSPSpillReload0` has `num_queues: 16`.
  So "split it across two rings for concurrency" has no basis on this kernel. Transfers *do*
  pipeline on that one queue (ids 4–12 overlap each other by ~0.2 us).
* **Transfer duration is driven by the access pattern, not the byte count.** Five 16 KB
  transfers measured 256 / 266 / 338 / 421 / 611 ns; var8's 16 KB `op:"copy"` scatter measured
  **1022 ns**. The `completion` column is all `NaN` — do not rely on it.
* Corollary: the `DmaPacketAggregated` read is the *first* thing to do when a gap looks like
  "DMA latency". It splits the gap into start-up (fixed, unavoidable), transfer (pattern-driven,
  sometimes fixable) and semaphore propagation (fixed).

## 7. Closed — do not spend budget on these
* var11/var12 **reload-side** fusion: needs 4 dims in a DGE, impossible. Use the save-side rewrite.
* **Merging the 4 `beta`/k input loads** (`Pool[8..11]`): nothing downstream is gated by them —
  `S[9..13]` land 2–6 us before their waiters even run. (s256 §7 agrees.)
* **Hoisting a Pool DGE's wait into a preceding `EVENT_SEMAPHORE`** (s256 iter-1 edit7, +0.4–0.7 us).
* **DGE rehosting in the 34.8–36.5 us ACT window** (§5.1, §5.3).
* **Hoisting the id4–id12 triggers** (§5.2) and any other SP trigger (§5.4, full scan).
* **Treating `SP[23]`/`SP[31]`'s waits on ACT as removable** — they are WAR guards (§5.5).
* **Splitting the gate load** across hosts (§F, s256 iter-2 §5/§5a).
* **Splitting a queued transfer across rings for concurrency**: every transfer already shares
  one `queue_idx` (§5.6).
* **Swapping the two tail stores**: the final fence waits for both and the state store is 4x the
  bytes.

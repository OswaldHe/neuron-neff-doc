# iter-6 — the critical path is now gap-free; the wins came from "get a 650 ns row out from in front of the node that signals" and "move an update-less op onto the idle engine"

## Result: **57.5-58.0 → 55.1 us (-2.4 us, -4.2%)**, outputs bit-identical

Six kept edits. All numbers are 7-capture medians of `tools/profile.py` in one session.

| step | edit | median us | Δ |
|---|---|---|---|
| A | iter-5's NEFF, measured at session start | **57.521** | — |
| B | tail: ACT[118]↔ACT[119] swap; sem-20 store back on Sync; sem-21 store un-split onto ACT (§3) | **56.824** | -0.70 |
| C | sem-18 output DGE moved off Pool onto an inserted Sync slot (§4) | **56.742** | -0.08 (but -0.79 anchor-relative) |
| D | sem-17 output DGE moved off Pool onto an inserted Sync slot (§4) | **56.293** | -0.45 |
| E | DVE[127]↔DVE[128]: the update-less `0x4780→0x4980` COPY moved behind the op that signals `S[6]#89` (§5) | **55.473** | -0.82 |
| F | two more numbering-neutral swaps of the same shape (DVE[109]/[110], ACT[99]/[100]) | 55.875 | **+0.40 — REVERTED (§6)** |
| G | **`ACT[11]` (TENSOR_SCALAR) executed on DVE instead of Activation** (§8) | **55.199** | -0.27 |
| H | + the same for `ACT[16]` (§8) | **55.128 / 55.146** | -0.07 / -0.17 (two pairs, both negative) |

**Interleaved A/B** (iter-5's bytes vs the shipped bytes, 7 captures each, alternating):
round 1 (vs E) **57.693 / 55.467**, round 2 (vs E) **58.011 / 55.772**, round 3 (vs H, shipped)
**57.563 / 55.133** → **-2.23 / -2.24 / -2.43 us**. The individual step deltas of ≤0.5 are inside
the ~±0.4 spread; the total is 5x it and reproduced three times.

`verify.py --runs 2`: PASS after every kept edit, bit-identical across runs, `max_abs_err`
identical to the baseline to every digit (output_0 6.092e-05 = 11%, output_1 4.494e-04 = 83%).
`launch_rules.py` PASS. `semaphores.py` no over-wait.
Members changed vs baseline: `sg00/{Activation0.bin, DVE0.bin, PE0.bin, Pool0.bin, SP0.bin, SP0.json}`;
vs iter-5: `Activation0.bin, DVE0.bin, Pool0.bin, SP0.bin`. Slot counts now
ACT 121, **DVE 136** (133 in iter-5), Pool 37, **SP 69** (67): four insertions this iteration.

## 0. Session numbers

Same-session ladder: A 57.521 → B 56.824 → C 56.742 → D 56.293 → E 55.473 → G 55.199 → **H 55.128**
(F 55.875, reverted). Direct G/H pairs: 55.199/55.128 and 55.319/55.146.
Last run on the shipped bytes: **55.235 us, -18.20% vs the iter-0 baseline (67.525)**.
Engine busy at the end: Scalar 60.1%, Vector 52.5%, DMA 36.5%, Tensor 31.8%, Pool 25.9%, Sync 21.5%
— **Scalar and Vector are now both over half busy, which is why §8 (move work between them) is the
lever and why removing stalls is finished.**
Interleaved A/B vs iter-5: 57.693/55.467, 58.011/55.772, **57.563/55.133 (the shipped bytes)**
→ -2.23 / -2.24 / **-2.43 us**.
Note iter-5's own bytes measured 57.521, 57.693, 58.011 and 57.563 over the session:
**the machine drifts ~0.5 us, so only interleaved A/B means anything.**

---

## 1. Rebuild the tools first — 30 minutes, and everything in this iteration came out of them

`scripts/` is not preserved. Rewrite these three; they are short and they are the whole method.

* **`cpath.py`** — the backward critical-path walker (iter-5 §1). Mine is ~110 lines and now also
  resolves **queued-transfer semaphores**, which iter-5's version could not: build
  `block id -> (sem, 16*nth-use-of-that-sem)` from `SP0.json["dma"]` (33 blocks, all on
  `qSPSpillReload0`, semaphores 24..39 reused 2-3x each), invert it, and map a wait
  `S[q]>=thr` with `q in 24..39` to `PSEUDO_DMATRIGGER` whose `count_union_block_id` is that
  block. Then the walk never dead-ends and you get the *flight time* (trigger end → semaphore
  landing) printed on every DMA hop. Field names that cost me time:
  `events_wait_idx` (not `events_semaphore_idx`), `events_semaphore_value` is the threshold
  **and** the (ignored) update operand, `isa.decode(raw)` picks the struct from the opcode,
  `isa.iter_stream(path)` yields `(idx, Instruction)`.
* **`d2.py`** — compact stream dump. Print `imm0_src`/`imm0_imm_ptr` always (iter-5 §2) and note
  that for `TENSOR_TENSOR_ARITH_OP` the sources are `src0_/src1_mem_pattern_*`, which a generic
  dumper misses — I nearly mis-analysed DVE[128] because of that. Dump the full `as_dict()` for
  any instruction you are about to move.
* **`reorder.py`** — new, and the one that found edit E: for every engine, every instruction `X`
  with **no semaphore update** that is followed by an instruction `Y` that does update and is on
  the critical path, compare the profile times at which each one's own wait was satisfied. If
  `X`'s gate is later than `Y`'s, `X` is stalling `Y` for nothing. **Filter the output by hand**:
  `LDWEIGHTS→MATMUL` pairs and any `Y` with no wait at all (printed as gate -1) are false
  positives, because `Y` depends on `X` through data, not semaphores.

## 2. What the kernel looks like now (this is the headline for iter-7)

`cpath.py` on the shipped NEFF: **164 nodes, and the sum of all gaps > 200 ns on the path is 0.**
Every node starts 10-50 ns after its binding predecessor. There is no stall left to remove. The
57.5→55.5 window decomposes into

* **~42 us of dependent compute**: Scalar 19.2 us over 61 path nodes, Vector 14.1 us over 62,
  Tensor 5.0 over 26, Sync/Pool ~4 us of triggers and rows;
* **~14 us of DMA flight on the path**, in 8 hops that cannot overlap each other:

  | hop | flight ns | gated by | status |
  |---|---|---|---|
  | input DGE `S[8]` → `DVE[8]` | 2340 | — | closed, iter-5 §6 (fixed ~2 us DRAM→SBUF latency) |
  | queued **blk5** → `Sync[14]` | 3100 | `S[6]>=19` | 64 KB 4-descriptor redistribute, ~21 GB/s; iter-3 closed merge (race) and split (hang) |
  | queued **blk6** → `Tensor[23]` | 1500 | blk5's output | blk6 transposes exactly what blk5 wrote — strictly serial |
  | queued **blk17** → `Scalar[90]` | 1250 | `S[4]>=41` | `p96+0x4900 → p0+0x5900` row→partition scatter, read as `ACT[90].imm0_imm_ptr=0x5900` |
  | queued **blk21** → `Vector[103]` | 1900 | `S[4]>=45` | same shape from `p96+0x4600`; the `DVE[103]` fence iter-5 §10 could not explain — see §7 |
  | queued **blk28** → `Tensor[157]` | 1580 | `S[6]>=82` | 64x64 bf16 transpose for one `LDWEIGHTS` |
  | queued **blk31** → `Scalar[117]` | 1230 | `S[4]>=58` = `ACT[116]` writes `p0+0x5600` | real RAW, see §7 |
  | output DGE `S[21]` → end barrier | 1550 | `S[6]>=93` | ~1.09 us to the first of 16 `+1` landings, ~0.18 us for all 16 |

  A DGE row costs **600-700 ns on its host engine regardless of payload** (32 partitions 8 KB =
  649 ns, 128 partitions 64 KB = 647 ns), and a **queued trigger costs only 43-61 ns**. That
  asymmetry is the biggest un-exploited lever left: see §8.

So iter-7 cannot win by removing stalls. It has to **remove a hop, or move compute off Scalar**.

## 3. Edit B — the tail, and the rule that the fastest tail is the fewest DGE rows on the two fast hosts

iter-5 split the final 16 KB output store into two 32-partition halves (ACT + Sync) and left the
64 KB store on Sync in front of one of them. Measured tail in that NEFF: last compute
`DVE[132]` end → kernel end = **2559 ns**, of which

* `Sync[65]` (sem 20, 64 KB) 649 ns row, then `Sync[66]` (sem 21 half B) **934 ns** row — the two
  are serial on Sync, and half B's row is 934 while the *identical* half A on ACT is 649, because
  both halves carry `semaphore 21` and therefore land in the **same DGE queue** and contend;
* 1.5 us of flight.

Three sub-edits, in the order I found them:

**(a) `ACT[118]` ↔ `ACT[119]` (kept, ~430 ns).** `ACT[119]` (PSUM `0x2002000 → 0x4d00`) was behind
`ACT[118]` (`0x4580 → 0x4f00`) although `ACT[119]`'s gate `S[5]>=71` lands 220 ns *before*
`ACT[118]`'s `S[6]>=92`, and `DVE[132]` — which writes `0x4b00`, the source of the whole sem-21
store — waits on `ACT[119]`. Writes are adjacent but disjoint (`0x4d00..0x4eff` vs
`0x4f00..0x50ff`). Both update `S[4]`, so the swap renumbers: I moved the two waiters' thresholds
with them (`DVE[132]` 61→60, the sem-20 store 60→61). Measured: `Tensor[167]→DVE[132]` link
1028 → 611 ns.

**(b) Pool is the wrong host for a tail DGE (refuted, -0 ns).** I first moved the sem-20 store to
`Pool[33]` to unblock Sync. The row then started **541 ns after its gate landed** (`S[4]>=61`
at 68010, row 68551-69246), against 27 ns for the same gate on Sync. iter-2 §3's "~530 ns
start-up gap after an idle stretch" is real and it is specifically **Pool**: Pool had been idle
since 55.7 us. Net effect on the metric: zero, and `S[20]` became the laggard.

**(c) un-split the sem-21 store, host the two stores on Sync and ACT (kept, ~250 ns).** Merged the
two 32-partition halves back into one 64-partition row on `ACT[120]` (`src/dst_num_elem
[32,1] → [64,1]`, barrier wait `S[21]>=32 → >=16`), left the sem-20 store on `Sync[65]`,
no-oped `Sync[66]`. Both rows now start within 30 ns of their gate on different engines and cost
~640 ns each. Tail 2559 → **2341 ns**, which is about the floor (row 640 + flight 1550 + barrier 150).

**Rule:** with two stores and two fast hosts (Sync, Activation), one row each is optimal.
Splitting a store only helps if the two halves get *different semaphores* (hence different
queues); on the same semaphore they serialise in the queue and the second row costs 1.4x.

## 4. Edits C and D — **a DGE row hosted on Pool costs the critical path ~1.2 us; move it to Sync**

Two output stores (`sem 17` at `Pool[27]`, `sem 18` at `Pool[29]`) were hosted on Pool. Each one
cost the path twice:

* Pool's ~500 ns wake-up after an idle stretch (measured: `S[6]>=71` landed 54686, `Pool[27]`'s row
  started 55176; `S[6]>=80` landed 58769, `Pool[29]`'s row started 59274);
* the 680 ns row itself, which sat in front of `Pool[28]`/`Pool[30]`
  (`TENSOR_SCALAR_AFFINE_SELECT`, both on the path, both feeding `S[3]` → a `DVE` CAST).

Both moved to **inserted** Sync slots, keeping their own `S[6]` guards, with `Pool[27]`/`Pool[29]`
replaced by the pure no-op (`a010` + 62 zero bytes). Placement matters:

* sem 18 → new slot **before the blk-28 trigger**: its gate `S[6]>=80` lands 666 ns before the
  trigger's `S[6]>=82`, so the row fits in the hole and only pushes the trigger out by ~120 ns.
* sem 17 → new slot **before `Sync[45]`** (the `S[31]>=16`/`S[6]>=74` guard, which blocks Sync
  until 56.4 us anyway), *not* just before its consumer `Sync[48]`: Sync does not reach slot 48
  until 56.5, so a row placed there would have run too late to help.

Effect of C, anchor-relative (`last_end − Tensor[23].start`, the only honest A/B measure — see §9):
`Vector[116].end → Vector[118].end` 1551 → 528 ns, kernel **-788 ns**. D: metric -0.45 us.

**Generalisable:** the three legal DGE hosts are act/pool/sp (iter-5 §7.1), the queue follows the
semaphore (iter-4 §1), and now: **the host's *reaction time* differs by 20x.** Sync and Activation
answer a semaphore in ~25 ns; Pool takes ~500 ns if it has been parked. Pool is only a good host
for a DGE that is issued back-to-back with other Pool work.

## 5. Edit E — the best edit of the iteration: a pure COPY was stalling the op that signals the tail

`DVE[127]` was `COPY 0x4780 → 0x4980` (128 partitions, FP32→FP32, **no semaphore update**, gate
`S[4]>=57` = `ACT[115]`), and `DVE[128]` was the `TENSOR_TENSOR SUBTRACT` that produces
`S[6]#89` — the gate of `ACT[116]`, which gates the blk-31 trigger, which gates the whole tail.
Vector issues in order, so the COPY's late gate (62511) was holding a signal whose own data
(`0x4008` from `ACT[111]`, `S[4]>=56`, at **61809**) had been ready for 700 ns.

Three facts made the swap safe and cheap:

1. `DVE[127]` updates **nothing**, so exchanging it with `DVE[128]` does not perturb the `S[6]`
   count sequence. *Any* reorder that moves only update-less instructions is numbering-neutral —
   this is the cheapest class of edit in the whole NEFF.
2. `DVE[128]`'s wait `S[50]>=16` is **not its own dependency**: it is iter-4 §4's *parked* guard
   for `DVE[129]`'s staging block. Its real data gate is `ACT[111]`'s `0x4008`. I retargeted its
   wait to `S[4]>=56` and **re-parked `S[50]>=16` on a newly inserted bare `EVENT_SEMAPHORE`**
   between the COPY and `DVE[129]`. Inserting a slot is free (§9).
3. Addresses are disjoint: COPY `0x4780→0x4980` vs TT `0x4008`+`0x5600` → PSUM `0x1804000`; and
   the WAR on `0x5600` (which `ACT[116]` overwrites) is still held by `ACT[116]`'s own
   `S[6]>=89` wait, i.e. by the TT op's increment.

Measured: `ACT[116]` stopped being semaphore-bound and became program-order-bound behind
`ACT[115]`; metric **-0.82 us** (predicted 0.5).

**When you retarget a wait, always ask first whether it is a parked guard for a later
instruction.** iter-4 parked four of them (`DVE[107]` xwait `S[67]`, `DVE[117]` xwait `S[23]`,
`DVE[128]` wait `S[50]`, `SP[42]` xwait) and they look exactly like real dependencies.

## 6. Edit F — REFUTED: the same swap twice more bought nothing, and this is the shape of the wall

`reorder.py` found two more numbering-neutral swaps:
`DVE[109]` (update-less COPY `0x5a08→0x5c08`, gate `S[4]>=49`) ↔ `DVE[110]` (TT, gate `S[5]>=60`,
signals `S[6]`), and `ACT[99]` (update-less ACTIVATE → `0x37300`, gate `S[4]>=48`) ↔ `ACT[100]`
(TENSOR_SCALAR → `0x5a08`, signals `S[4]#49`). Both verified bit-identical, both did exactly what
they were designed to do: **`Vector[111]` and `Tensor[144]` moved 626 ns earlier**.

By `Scalar[117]` the gain was down to **-44 ns** and the kernel end to **-52 ns** anchor-relative
(metric 55.875 vs 55.473, i.e. noise-up). Reverted, to keep the diff small.

So: in the 56-62 us region a 626 ns head start is completely absorbed, and the absorbing node is
somewhere between `Tensor[144]` and `Scalar[117]` — a stretch that the walker reports as 100%
back-to-back `Vector`/`Tensor`/`Scalar` compute. **The middle of the kernel is throughput-bound,
not latency-bound. Only the tail and the Scalar-saturated stretches are worth attacking.**

## 7. Hops re-examined and closed this iteration

* **blk31 (`p0+0x5600 → p0+0x37300`, flight 1230)**: reads 256 fp32 from *one* partition and
  scatters 4 per partition across 64; `ACT[117]` consumes it as a per-partition scalar
  (`imm0_imm_ptr=0x37300`). No compute instruction can do that, and its trigger's `S[4]>=58`
  is `ACT[116]` writing the first 256 B of the source. **Real, irreducible.**
* **blk17/blk21 (`p96+0x49xx/0x46xx → p0+0x59xx/0x5a00`, flights 1250/1900)**: same 4-per-partition
  scatter. `ACT[90].imm0_imm_ptr=0x5900` and `ACT[100].imm0_imm_ptr=0x5a00` are the consumers, and
  those two `ACT` instructions have **no wait of their own** — they are ordered only through
  `ACT[97]`'s `S[6]>=73`, which is produced by a `DVE` op downstream of the `DVE[103]` fence.
  **That answers iter-5 §10 candidate 2: `DVE[103]`'s `S[29]>=32` fence is the only thing that
  orders `ACT[100]`'s read of blk21's output.** It is a real cross-engine guard; do not move it.
* **`ACT[115]` cannot be hoisted ahead of `ACT[114]`**: `ACT[115].imm0_imm_ptr = 0x4200` is exactly
  blk29's destination (`p96+0x4b00 → p0+0x4200`), and `ACT[114]` is the `S[37]>=32` fence for it.
  Another instance of iter-5 §2's pointer-operand trap; the scan caught it before the device did.
* **`DVE[126]` waits `S[3]>=11` = `Pool[31]`**, so Vector cannot reach slot 127 before ~62.0 us
  however early its gate lands. Pool's `AFFINE_SELECT` chain is a real part of the tail.

## 8. Edits G and H — **an Activation instruction can be executed on DVE instead; this is open and it worked**

This is the lever iter-7 should spend most of its time on, so here is the whole recipe and every
fact I verified on the device.

**The structural fact:** Activation is the busiest engine (59%) and contributes **19.2 us to a
42 us compute path**. Two stretches are pure back-to-back Activation while Vector *and* Pool are
mostly idle: `ACT[6..17]` (17.9→22.7 us) and `ACT[20..36]` (22.6→27.5 us). Symmetrically,
`DVE[44..79]` is 9.4 us of back-to-back Vector while **Activation is idle 35.2→44.0 us**.

**The encoding fact:** `TENSOR_SCALAR_ARITH_OP` is `v2.NEURON_ISA_TPB_S3D3_TS_STRUCT` on *both*
Activation and DVE, and `CAST` likewise, so **a raw 64-byte slot copy of an Activation
TENSOR_SCALAR into a DVE slot executes correctly**. Device-verified twice, bit-identical, and
crucially: **`imm0_src = POINTER_IMMEDIATE` works on DVE** even though the compiler never emits
it there (all four original `DVE` TENSOR_SCALARs use `INSTRUCTION_IMMEDIATE`; all ~25 Activation
ones use the pointer form). That was the one assumption I was worried about and it is now settled.
`ACTIVATE` is Scalar-only (no `ACTIVATE` anywhere in the DVE/Pool/PE streams) and Pool only ever
runs `TENSOR_SCALAR_AFFINE_SELECT` (`S2D2_TS_AS_STRUCT`), so **Activation↔DVE is the only pair.**

**The recipe (both edits used it verbatim):**
1. Pick an Activation op with **no semaphore update** → the `S[4]` count sequence is untouched and
   no waiter anywhere needs renumbering. In the two Scalar-saturated windows the no-update ops are
   `ACT[6], [8], [11], [12], [15], [16], [18]` and `ACT[21], [22], [23], [24], [26], [29], [33]`;
   of those only the `TENSOR_SCALAR`s are movable (`ACT[11], [16], [21], [23], [24], [33]`).
2. Check the op needs **exactly one gate**. `ACT[11]` reads `p64+0x4b00` (`ACT[9]` = `S[4]#3`) and
   `imm0_imm_ptr p64+0x200` (`ACT[3]`, earlier on the same stream), so `S[4]>=3` — its existing
   wait — covers both. `ACT[16]`: `p64+0x4d88` (`ACT[14]` = `S[4]#6`) + ptr `p64+0x4d80` (`ACT[4]`),
   so `S[4]>=6`. Ops that need two gates (`ACT[21]` wants `S[14]` *and* `S[10]`; `ACT[23]/[24]`
   have **no wait at all** and rely purely on Activation program order) need a second inserted
   `EVENT_SEMAPHORE` in DVE — doable, just more surgery.
3. Check the **consumers**. `scan.py` over the destination: `ACT[11]`'s `p0+0x4980` is read only by
   the DGE `SP[12]` (gate `S[4]>=18`); `ACT[16]`'s `p0+0x4b80` only by the DGE `SP[13]`
   (`S[4]>=18`). Both are gated at an `S[4]` value far above the moved op's slot, so step 4
   orders them. Watch the partition bit: `ACT[27]` writes `p64+0x4980` and `Pool[15]` reads it —
   a different partition range from `p0+0x4980`, no conflict.
4. Give the DVE copy `events_update_mode = SEM_INC_COMPLETE` on a **free semaphore** (68, 69 used
   here; 70, 71, 22, 23, 40, 41, 42, 51-55 still free) and **replace the vacated Activation slot
   with a bare `EVENT_SEMAPHORE` waiting `S[f]>=1`**. That keeps every later Activation
   instruction — and therefore every `S[4]` increment from the moved op's position onward —
   ordered after the DVE copy. **Ordering is then enforced by construction, not by timing**, which
   is what makes this safe against the race class that `verify.py` cannot see.
5. Place the DVE copy where **DVE has a hole at the moment the gate lands**, otherwise the
   Activation fence stalls and you lose. `ACT[11]`: gate lands 18.91, `DVE[15]` ends 18.88 and
   `DVE[16]` does not run until 19.77 → inserted at DVE slot 16, ran 19.09-19.38, fence cost 40 ns,
   nothing displaced. `ACT[16]`: gate lands 20.90 and the only hole is *before* `DVE[18]`, so the
   copy pushed `DVE[18]/[19]` by ~210 ns, and the gain fell from a predicted 346 to a measured
   **135 ns** (every Scalar landmark from `ACT[17]` through `ACT[36]`, and the blk-5 trigger
   `Sync[10]`, moved -135 ns). That is why G was worth 0.27 and H only ~0.1: **the gain is
   `op duration − fence − whatever you displace on the receiving engine`.**

**For iter-7:** `ACT[21]/[23]/[24]/[33]` are the remaining candidates in the 22.6-27.5 us window
(370-384 ns each), and the *reverse* direction is untouched and larger: `DVE[52], [54], [55],
[56], [59], [68], [71], [72]` are eight update-less `CAST`/`COPY` ops (~1.9 us total) inside the
9.4 us Vector-saturated stretch where **Activation is completely idle**. `COPY` does not appear on
Activation in this NEFF, but a `CAST` with `in_dtype == out_dtype` is the same thing, and
`DVE[52]/[55]/[71]/[72]` are exactly FP32→FP32 copies. Same five steps, mirrored. Also worth
10 minutes: check whether `DVE[54]` (`src 0x6480 → dst 0x5880`) is simply a repeat of `DVE[46]`
(same src, dst and shape) — if nothing writes `0x6480` in between, it is dead and can be no-oped.

### 8.1 The next concrete instance, already scoped (I ran out of session, not out of lever)

`DVE[57]` in the shipped NEFF (`COPY p64+0x1c708 → p64+0x5200`, 224 ns, nac=64, FP32→FP32,
**no update**, single wait `S[4]>=25`). In `prof8`: `S[4]>=25` lands at **38259** but `DVE[57]`
does not run until **38949** (Vector is saturated 38093→39636 with 13-30 ns gaps), and
**Activation has a hole 38250→39264** — so an Activation copy with the same `S[4]>=25` wait would
run 38280-38490 and be finished before Vector reached the slot. Net ~200 ns. Two things to clear
first, which is exactly why I did not ship it:

* **the WAW hazard.** `p64+0x5200` is a reused scratch buffer written by *four* DVE ops (old
  indices 49, 55, 65, 71). The fence left in the vacated DVE slot only orders DVE slots ≥ 57;
  if the earlier writer (old `DVE[49]`, gate `S[5]>=25`) can still be running when the Activation
  copy lands, the final value is wrong — and `verify.py` would very likely still pass on one
  input. Check `Vector[51]`'s end time against `S[4]>=25` before touching it, or give the
  Activation copy a wait that covers the earlier writer too.
* **does Activation accept `COPY`?** There is no `COPY` anywhere in `Activation0.bin`. `DVE`'s
  `COPY` is `common.NEURON_ISA_TPB_S4D4_TR_STRUCT`; compare it with an Activation `CAST`'s struct
  and, if they match, flip the opcode byte to `CAST` with `in_dtype == out_dtype == FP32`, which
  is the same operation.

## 9. Ranked candidates for iter-7

1. **Turn an output store from a DGE into a queued transfer.** A DGE row costs 600-700 ns of host
   engine time; a `PSEUDO_DMATRIGGER` costs 43-61 ns. The tail's floor is row (640) + flight (1550)
   + barrier (150) = 2340 ns and **640 of that is pure descriptor-generation on Activation**. The
   edit is: append a 34th block to `SP0.json["dma"]` (`queue qSPSpillReload0`, a free semaphore,
   one descriptor `SB p0+0x4b00 [64 partitions, 256 B] → DRAM var5+0xc000`), replace `ACT[120]`
   with a trigger for it, and point the end barrier at the new semaphore. Unknown: whether the
   runtime sizes the ring from the JSON (it must — `neff.py pack` rewrites the member) and whether
   `to: "DRAM"` descriptors are legal in that queue (31 of the 33 existing blocks are SB→SB;
   check blocks 0-4 for a DRAM one first). iter-3 found queued-transfer plumbing hang-prone, so do
   this **early in the iteration**, with `cp save/kernel.neff kernel.neff` ready. Ceiling ~0.6 us
   on the tail, and the same trick applies to the sem-17/18/19/20 stores.
2. **Continue §8 — move more update-less ops to the idle engine.** Proven on the device twice
   this iteration (edits G and H, -0.3 us). The *reverse* direction is the bigger half and is
   completely untouched: eight update-less `CAST`/`COPY` ops in `DVE[52..72]` (~1.9 us) inside the
   9.4 us Vector-saturated stretch where Activation is idle 35.2→44.0 us. Read §8 for the five-step
   recipe and the exact candidate lists; the only new unknown in that direction is whether
   Activation accepts a `COPY` (it has none in this NEFF — use a `CAST` with
   `in_dtype == out_dtype`, which is what `DVE[52]/[55]/[71]/[72]` already are).
3. **blk5 (3.1 us) + blk6 (1.5 us) = 4.6 us, the biggest single block of path time.** blk5 is 4
   descriptors, 64 KB SBUF→SBUF, measured ~21 GB/s — so roughly 1.4 us latency + 1.7 us transfer;
   blk6 transposes exactly what blk5 wrote, so they are strictly serial. The only shape of win
   left is to make blk5's *bandwidth* better (it is the only transfer in flight at that moment —
   `Sync[11],[12],[13]`'s DGE rows run alongside but their payloads are small). iter-3 closed
   merging and splitting; what is *not* tried is moving two of blk5's four descriptors into a
   second block with a different semaphore so they occupy two queues, and gating blk6 on both.
   Note all 33 queued blocks share `qSPSpillReload0`, so a second queue would have to be a DGE,
   and blk5's descriptors are 4-D (DGE `DIRECT2D` has 2). Probably closed; 20 minutes to confirm.
4. **The ~120 ns still lost to the inserted sem-18 row** in front of the blk-28 trigger (§4).
   Moving it one slot later gains ~120 ns on `Tensor[157]` but delays `S[18]`, and `S[18]>=16`
   binds the blk-30 trigger within 17 ns, whose consumer `Tensor[161]` only has 433 ns of slack.
   Measured; net negative. Leave it.
5. **Do not retry**: everything in iter-1 §7, iter-2 §10.4, iter-3 §9.3, iter-4 §5/§10.6/§11.1,
   iter-5 §2/§6/§7.1, and from this iteration: hosting a tail DGE on Pool (§3b), splitting one
   store across two hosts on the *same* semaphore (§3), hoisting `ACT[115]` (§7), the
   `DVE[109]`/`ACT[99]` swaps (§6).

## 10. Mechanics confirmed or learned this iteration

* **Inserting slots is free and now proven 4x.** `SP0.bin` 67→69 and `DVE0.bin` 133→136 this
  iteration. The engine `.json` files only carry `{"instr": "<name>.bin", "name": ..., "dma": ...}`
  — `instr` is a **filename, not a count** — and `def.json` names only the `.bin`. Nothing records
  an instruction count, and `neff.py pack` recomputes sizes and both MD5s.
* **`last_end − min(start_ts)` ≈ `total_exec_time`** (the ~12.2 us before `Sync[0]
  SET_ORDERING_MODE` is not counted), but it is **not** usable for A/B: the pre-`Tensor[23]`
  region drifts ±700 ns run to run. Use `last_end − Tensor[23].start_ts`. Config C looked like
  -0.08 us on the metric and was -0.79 us anchor-relative; the difference was entirely upstream
  noise. Conversely config F looked like +0.40 on the metric and -0.05 anchor-relative.
  **Decide with the anchor, report with the metric, and run both.**
* A DGE's `sem_increment=16` lands as **16 separate `+1` updates** ~12 ns apart in
  `SemaphoreUpdate.parquet`; the first arrives ~1.09 us after the row ends and the 16th ~0.18 us
  later. Two DGEs on the same semaphore interleave their 32 updates and the queue serialises them.
* `TENSOR_TENSOR_ARITH_OP` / `COPY` / `CAST` have no extended-wait fields (only `EVENT_SEMAPHORE`
  does), so a second wait needs a neighbouring `EVENT_SEMAPHORE` slot — and inserting one is cheap.
* To find which `ACT` instruction is `S[4]#n`, enumerate the stream in slot order counting
  `events_update_mode != NONE && events_update_idx == 4`. Did this programmatically; worth 2 lines.
* Device cost this session: 6 verifies (~1.2 min each at `--runs 2`), 10 profiles (~1.7 min each),
  **no hang and no wrong output** — every edit was address-checked before packing.
* Keep a `save_<step>.neff` per accepted step and a `work<X>/` tree per step; each new variant is
  one `pack` away and each revert is one `cp`.

## 11. Honest model after six iterations

67.5 → 55.1 us (-18.4%). The critical path has **no idle gaps left** — `cpath.py` reports 0 ns of
gap > 200 ns on 164 nodes. It is ~42 us of dependent compute plus ~14 us of DMA flight in 8
non-overlappable hops, and the hops are all genuine redistributions (row→partition scatters and
64x128 transposes) that no compute instruction can replace. So synchronisation is done; what is
left attacks the *compute*:

* **move update-less ops off the saturated engine onto the idle one** (§8). Proven, Activation↔DVE
  only, ~150-350 ns per op, ~2-3 us of candidates left. This is the main lever.
* **replace a 650 ns DGE row with a ~50 ns queued trigger** (§9.1). Unproven, needs a new
  `SP0.json` dma block, ~0.6 us, hang-prone.

Each costs one inserted slot and at most one free semaphore, which this iteration showed is
routine (four insertions, no hang, no wrong output). Expect the next iteration to be another
five 200-300 ns edits rather than one big win; below ~52 us I would expect to need the kernel
source. **And measure with `last_end − Tensor[23].start_ts` as well as the metric: the pre-anchor
region (the blk5/blk6 DMA chain) swings ±700 ns run to run and it disagreed with the metric on
two of the eight edits this session.**

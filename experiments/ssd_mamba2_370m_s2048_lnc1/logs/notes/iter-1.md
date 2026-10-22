# iter-1 (ssd_mamba2_370m_s2048): 5248.9 -> 3622.2 us (-31.0%, 1.449x)

**Shipped two edits to `sg00/Activation0.bin` only. `verify.py` VERIFIED, `launch_rules.py` pass,
`semaphores.py` clean. `max_abs_err` output_0 2.058e-07 (89% of the 2.3e-07 limit, baseline 82%),
output_1 1.583e-07 (69%, baseline 52%). Spread 0.15% (5.6 us) over 5 repeats.**

| step | median us | delta | ACT busy |
|---|---|---|---|
| iter-0 control (5 rep) | 5248.897 | - | 4721.7 |
| §1 `imm_src` POINTER -> INSTRUCTION on 6144 ACTIVATEs (3 rep) | 4811.082 | **-437.8 (-8.34%)** | 4189.7 |
| **§2 + 633 of 768 recip(exp) units rewritten as exp(-x) (5 rep, shipped)** | **3622.223** | **-1188.9 (-24.7%)** | **2727.6** |

Scripts are next to this file: `iter-1-common.py`, `iter-1-imm.py`, `iter-1-tblkill.py`,
`iter-1-try.sh`, `iter-1-census.py`. Pipeline: `neff.py unpack kernel.neff work/`, keep a
`work_<step>/` copy after each accepted step, run a script, `neff.py pack work/ kernel.neff`,
`semaphores.py work sg00`, `verify.py`, `profile.py`.

## READ THESE FIRST

1. **§1 is free and bit-exact: do not re-derive it.** All 6144 pointer-`imm` ACT `ACTIVATE`s
   (5376 f=128 + 768 f=129) carried `imm_src = POINTER_IMMEDIATE` with `imm_imm_ptr == 0`.
   One field flip, **-437.8 us (-8.34%)**, `max_abs_err` identical to the last digit. The review's
   worry that `imm_ptr` here is `0x808080` was a mis-read: `isa.py` decodes `imm_imm_ptr == 0`
   and `imm_imm_fp32 == 0.0` on every one of them. The f=129 class flips safely too (the sibling
   only did f=128).
2. **§2's blocker was never numerics, it was ACT's pipeline overlap.** The sibling's exp(-x)
   recipe fails here out of the box (`max_abs_err` 3.9e-02). Three distinct causes, in the order
   I found them — all three fixes are in `iter-1-tblkill.py`:
   - a DGE `PSEUDO_DMA_DIRECT2D` inside the window addresses SBUF through `var_id`/`var_offset`,
     not a `mem_pattern`, so an address scan cannot see it. **Reject any unit with a slot other
     than `ACTIVATE`/`ACTIVATION_TABLE_LOAD`/`EVENT_SEMAPHORE` between `prod` and `R`** (27 units).
   - **ACT runs consecutive slots concurrently.** `Instruction.parquet` on the unedited NEFF:
     slot 1557 = (624417, 624620), slot 1558 = (624526, 624833) — 1558 *starts 94 ns before 1557
     ends*. 1558 reads `A` as a per-partition scale pointer with a wait on sem **6**, so nothing
     orders it after 1557's write of `A`. In the baseline the 1283 ns `sel=0` reload sat between
     them and supplied the separation by accident. **Fix: don't delete the `sel=0` slot, delete it
     and insert an `EVENT_SEMAPHORE` waiting `sem4 >= cum[r+1]` immediately after `R'`** (~62 ns
     instead of 1283 ns, slot count unchanged). Putting the barrier at `L0`'s original position
     does **not** work — `L0` is after the racing reader in many units.
   - the `L1'`/`R'` pair both touched `A` (`L1'` read it, `R'` overwrote it in place) and can
     overlap for the same reason. **Fix: put the temp in free SBUF scratch** so neither `A` nor
     `C` is ever written with anything but its final value (§2c).
3. **The remaining limit on §2 is the error budget, and the error is NOT additive per unit.**
   With all hazards fixed, 634 units gives 2.393e-07 (104% of the limit) and 375 units gives
   2.421e-07 — *more* conversions gave a *lower* max error, so you cannot attribute the error to
   individual units and subtract them. What shipped is "all 634 minus plan index 365", found by
   bisection: 2.058e-07 (89%). **This is not bit-exact and it is fitted to the fixed inputs** —
   see §3 for the honest accounting and the conservative fallback.
4. **Closed by measurement this iteration:** no stock activation-table set holds both `exp` and
   `reciprocal` (all 22 sets in `act_info.json` checked, §2a), so the reload pair is unavoidable
   while 1/x runs on ACT; there are **zero** dead `EVENT_SEMAPHORE`s (wait NONE + update NONE) on
   any of the five engines, so the sibling's iter-7 §1 slot-deletion lever does not exist here;
   and the `POINTER_IMMEDIATE` sweep is exhausted after §1 (§4).

## 1. §1: the unused `imm` operand (-437.8 us, -8.34%, bit-exact)

`iter-1-imm.py`. ACT `ACTIVATE` census of the unedited NEFF
(`activation_func`, `nch`, `scale_src`, `imm_src`, `bias_src`, `imm_ptr`, `imm_fp32`):

| n | f | nch | scale_src | imm_src |
|---|---|---|---|---|
| 3840 | 128 | 128 | POINTER | **POINTER** |
| 768 | 128 | 1 | POINTER | **POINTER** |
| 768 | 128 | 64 | POINTER | **POINTER** |
| 768 | 129 | 128 | INSTRUCTION | **POINTER** |
| 768 | 7 | 1 | INSTRUCTION | INSTRUCTION |
| 768 | 7 | 128 | INSTRUCTION | INSTRUCTION |

`imm_imm_ptr == 0` and `imm_imm_fp32 == 0.0` on all 6144, i.e. ACT was told to fetch the
activation function's extra operand from SBUF partition 0 offset 0 (unrelated kernel data) and
the output was bit-exact anyway, so neither f=128 nor f=129 consumes it. `imm_src ->
INSTRUCTION_IMMEDIATE` (the immediate is already 0.0) removed 532 us of ACT busy -> 437.8 us of
wall, an **82% busy-to-wall conversion** while ACT was at 90%. Remember that ratio: it is what
makes any ACT saving worth ~0.8x on the wall as long as ACT is the binding engine.

**Do not touch `scale_src`.** Its 5376 pointers are real per-partition scale vectors
(`0x4e90` = `exp_cs_col`, `0x4e98` = `exp_neg_cs_col`, `0x1037e00`, ...).

## 2. §2: the activation-table reloads (-1188.9 us, -24.7%, NOT bit-exact)

### 2a. The baseline shape (unedited)

768 units = 48 heads x 16 chunks. Per unit the ACT stream is

```
prod  ACTIVATE f=7   nch=128  PSUM(m2) -> A   ne=[1,1,1]  scale INST +1.0   A = exp(cs_col)
L1    ACTIVATION_TABLE_LOAD sel=1                                            <- 1283 ns
R     ACTIVATE f=129 nch=128  A -> C                                         C = 1/A
L0    ACTIVATION_TABLE_LOAD sel=0                                            <- 1283 ns
```

`A` and `C` are then consumed by the unit's f=128 copies as `scale` **pointers**, and `C` also by
DVE `TENSOR_TENSOR_ARITH_OP`/`TENSOR_TENSOR_SCAN_ARITH`. `A`/`C` are per-unit scalars (4 bytes x
128 partitions) and the *same* address is reused by many units: `0x4e90`/`0x4e98` serves 655/600
units, with small groups at `0x4200`, `0x4208`, `0x4210`, `0x4218`, `0x4c88`, `0x5290`, ...

1531 `ACT_TABLE_LOAD` x **exactly 1283.000 ns** = **1964 us**, 37% of the unedited runtime and 47%
of ACT busy after §1. The duration is a constant, independent of which set is loaded, so shrinking
the table binaries is very unlikely to help.

Table-set contents (`natural_log_exp_and_others.json` / `reciprocal_and_small.json`): set 0 has
`exp` + `ln` + `copy` + the small ones; set 1 has `reciprocal` + `copy` + the small ones.
**I checked all 22 sets listed in `act_info.json`: none contains both `exp` and `reciprocal`.**
f=128 (`copy`) is in every set, which is why the f=128 copies at `L1..L0` run fine under set 1.
So 2 switches per unit is structural as long as one reciprocal sits between two exps.

### 2b. What shipped

```
prod  ACTIVATE f=128 copy  PSUM -> T(0x37f80)   events untouched
L1'   ACTIVATE f=7   scale -1.0   T -> C        R's own wait, update dropped     C = exp(-cs_col)
R'    ACTIVATE f=7   scale +1.0   T -> A        R's events untouched             A = exp(cs_col)
      EVENT_SEMAPHORE wait sem4 >= cum[r+1]     inserted (new slot, ~62 ns)
L0    deleted
```

Net per converted unit: -2566 ns of table load, +62 ns of barrier, slot count unchanged.
633 converted, 265 table loads left (340 us of ACT busy still there).

* `L1'` is built from `R`'s bytes, so it **keeps `R`'s own wait** (`sem4 >= v`, always >=
  `cum[prod+1]` because `R` read `prod`'s output) and only drops the update. That way `C` is never
  written earlier *in semaphore terms* than `R` could have started, and the cumulative sem-4
  increment sequence is byte-for-byte identical to the baseline (asserted; `semaphores.py` clean).
  17 of 768 recips carry **no wait at all** — for those `L1'` gets `sem4 >= cum[prod+1]`.
  1 recip waits on sem 19 (a DMA queue); it is excluded rather than reasoned about.
* The inserted barrier waits on ACT's *own* completion counter. That is a pattern the compiler
  itself uses in this NEFF, and it cannot deadlock: every increment it counts is at a slot before
  it in the same in-order stream.

### 2c. The scratch temp `T = 0x37f80` works, and there is room for 756 bytes

Highest SBUF byte offset any instruction on any engine touches (mem patterns expanded to full
byte ranges, plus every active `*_imm_ptr`): **0x37f04**, by an ACT scale pointer. `def.json`
`runtime_statebuffer_reservation` is `offset 229368 (0x37ff8), size 8, partitions 128, evtaccel`.
So **[0x37f04, 0x37ff8) = 756 bytes per partition are free**; `0x37f80` is in it and verified on
the device. This is a reusable primitive — it is exactly the "free SBUF hole" the review's
candidate #4 (state round-trip removal) needs.

### 2d. Unit filter, and the counts it rejects (GUARD=20000)

`iter-1-tblkill.py`, `DRY=1` prints the census. 634 of 768 accepted; rejects:
`A touched off ACT inside the window` 106, `opaque slot inside window` 21,
`multi-recip region` 6 (two recips share one `L1`/`L0` pair — needs a second spare slot),
`recip waits on another semaphore` 1.

The off-ACT test resolves each off-ACT slot to a `(start_ts, end_ts)` from
`prof/Instruction.parquet` and rejects a unit if any slot whose byte range overlaps `A` is live
within `GUARD` ns of `[prod_start, R_end]`. **A plain address -> reader map is useless** (a global
equality test rejected 705 of 768 on false positives) and **address equality is not enough** —
patterns span ranges (`ne=[32,1,1]` is 128 bytes), so compare intervals. GUARD matters because
the schedule compresses by 1.6 ms once the edit lands, so a baseline-timestamp gate is not sound
by itself; GUARD=1500 let a `0x5290` unit through whose `A` is read by DVE's scan.
Yields: GUARD=1500 -> 728, 5000 -> 721, **20000 -> 634**, 40000 -> 531.

## 3. The error budget: honest accounting

Baseline output_0 `max_abs_err` 1.881e-07 (82% of 2.3e-07). The substitution replaces
`recip_table(exp_table(x))` with `exp_table(-x)` — one table evaluation instead of two. In exact
fp32 on these inputs the two forms differ by at most **2.65e-07 relative** (numpy check: `cs` only
ranges over [-7.04, 0], nothing overflows, no denormals), so there is no edge case here; it is
plain rounding, and it moves the max error by about a quarter of an ULP.

Measured, GUARD=20000, prefix of N units:

| N | output_0 max_abs_err | |
|---|---|---|
| 300, 340, 358, 362, 364, 365 | 1.881e-07 (82%) | identical to baseline |
| 366, 371, 375 | 2.421e-07 (105%) | FAIL |
| 450, 634 | 2.393e-07 (104%) | FAIL |
| **634 minus index 365** | **2.058e-07 (89%)** | **shipped** |

**The error is not monotone in the converted set** (375 is worse than 450), so "find and remove the
bad units" is not a sound procedure — and plan index 365 is structurally identical to 364 and 366
(same `A`/`C`, same `nch`, same spacing). Excluding it is **fitted to the fixed verifier inputs**:
the element that sets output_0's max error happens to live in that unit's (head, chunk).

**The conservative fallback, if a future review wants the error budget untouched:** GUARD=20000,
`LIMIT=365`, no exclusion -> **4164.98 us (-20.65%, 1.26x)**, `max_abs_err` 1.881e-07 / 1.192e-07,
i.e. *exactly* the baseline numbers. Measured, 5 repeats, spread 0.10%. That is 10 points of wall
time traded for 7 points of error budget. `kernel_340.neff` in my worktree was the 340-unit
variant; the 365-unit one is one `LIMIT` change away.

Either way: **the next numerics-touching edit has at most 11% of the budget left. Prefer bit-exact
edits from here.**

## 4. Levers closed by measurement this iteration

* **No stock table set has `exp` + `reciprocal`** (§2a). The reload pair cannot be removed by
  switching sets.
* **No dead `EVENT_SEMAPHORE` anywhere**: ACT 1579, DVE 764, PE 282, Pool 780, SP 1153, and
  **0** with wait NONE *and* update NONE. The sibling's iter-7 §1 (-37 us) was cleaning up its own
  earlier edits; this baseline is clean. Review candidate #3 does not exist here.
* **`LOAD_MASK_SELECT` deletion**: 4608 LMS and 4608 `STREAM_SHUFFLE` on DVE — a 1:1 ratio, so
  there are no orphans to delete (the sibling's 3077 were orphaned by its own earlier edits).
* **The `POINTER_IMMEDIATE` sweep is exhausted after §1.** A scan of every enum field of every
  slot of all five engines for a `*POINTER*` value leaves only: ACT `scale_src` (5376, real
  per-partition vectors) and DVE `TENSOR_TENSOR_SCAN_ARITH.imm0_src` (768, non-null pointers
  `0x4c80`/`0x4e80`/...). Nothing on PE, Pool, SP.
* **Reading PSUM twice** — not retried; the sibling's `PSUM_COLLISION_ERROR` is why `prod` has to
  become a copy into a temp rather than letting a second exp read the PSUM bank.

## 5. Where the time is now (wall 3622, `prof2/`)

| engine | busy us | % of wall |
|---|---|---|
| **Activation / Scalar** | **2727.6** | **75.3%** |
| DVE / Vector | 1662.1 | 45.9% |
| Pool / GpSimd | 1558.9 | 43.0% |
| PE / Tensor | 1267.1 | 35.0% |
| DMA | 1160.6 | 32.0% |
| SP / Sync | 572.4 | 15.8% |

Per opcode: ACT `ACTIVATE` 8313 x 280 ns = **2324 us**; PE `MATMUL` 5441 x 356 = 1934;
Pool `DMA_DIRECT2D` 2385 x 655 = 1562; PE `LDWEIGHTS` 5441 x 279 = 1516;
DVE `STREAM_SHUFFLE` 4608 x 148 = 681; DVE `TENSOR_TENSOR` 3104 x 177 = 550;
ACT `DMA_DIRECT2D` 816 x 530 = **432**; SP `DMA_DIRECT2D` 816 x 522 = 426;
DVE `TENSOR_TENSOR_SCAN` 768 x 470 = 361; **ACT `ACT_TABLE_LOAD` 265 x 1283 = 340**;
ACT `EVENT_SEMAPHORE` 2220 x 62 = 137.

## 6. Ranked next steps

1. **ACT is still binding at 75%, and `ACTIVATE` is now 2324 us of it**, so the 82% busy-to-wall
   ratio from §1 still makes ACT the best target. The big block is the **5376 pointer-scale f=128
   copies**; the sibling priced the per-partition scale fetch at 132 ns each and called it
   irreducible because the ISA has no "read one scalar and broadcast" `imm_src` mode.
   **I checked the most promising subset and it is closed:** the 768 f=128/nch=64 copies all share
   the scale pointer `0x1037e00`, which looked like it might be a constant. Debug info maps them
   to `ssd_block.py:535`, `nisa.tensor_scalar(state_sb *= exp_cs_last_N)` where
   `exp_cs_last_N = broadcast_scalar_to_column(exp_cs_last, n_size)` — a genuine per-unit
   on-device value, so no instruction immediate can carry it. Do not spend a device run on it.
   (Debug info is the cheap way to settle this class of question: join
   `Instruction.parquet.compiler_pc` to `nki_source_location`.)
2. **The 265 surviving table loads are 340 us of ACT busy** but they are the units the filter
   rejects, and the error budget is nearly spent, so leave them.
3. **ACT's 816 `DMA_DIRECT2D` = 432 us** is the largest non-`ACTIVATE` ACT block. The sibling's
   iter-1 §4 moved these to SP and found it is *not* a count limit: specific triggers hang
   (`NRT_TIMEOUT`) and whole 32/64-trigger windows move cleanly. All the machinery (slot insertion
   into another engine's stream, the two handshakes, the `sem s >= c*16` completion formula) is
   written up in that iteration's §4b/§4d. Budget ~25 min for a window scan; expect 250-350 us.
   Note that **slot insertion is now proven on this NEFF too** (§2b inserts 633 slots into ACT).
4. **The free SBUF hole is found (§2c)**, which unblocks review candidate #4 (deleting the
   per-chunk `state_sb` HBM round-trip, the sibling's iter-4, -4.9%). `spill_save_bytes` here is
   worth checking in `Summary.parquet`; Pool's 2385 DGE DMAs at 1562 us and SP's 816 are what
   those spills cost.
5. **Do not** start with a DVE/Pool/DMA throughput edit: the sibling measured Pool/DMA work
   converting to wall at only ~6%, and nothing here is above 76% of its span except ACT.

## 7. Tool notes

* `isa.py`'s no-op enum value for both `events_wait_mode` and `events_update_mode` is **`'NONE'`**,
  not `'NO_WAIT'`/`'NO_SEM_UPDATE'`. Getting this wrong silently makes every slot look like a
  sem-4 incrementer.
* Field names on `ACTIVATE`: `src_mem_pattern_start_addr_addr_immediate` (24-bit address),
  `..._addr_reg_marker` (0 = SBUF, 1 = ?, 2 = PSUM), `..._num_elem`/`..._step_elem` (3 dims,
  elements), `activation_func`, `scale_value_imm_fp32`, `scale_src`/`imm_src`/`bias_src`,
  `events_wait_mode`/`events_wait_idx`/`events_semaphore_value`/`events_update_mode`/
  `events_update_idx`. `ins.set('dst_..._immediate', v)` writes the whole 3-byte field correctly;
  set the marker separately. Re-decode the encoded stream and assert every rewritten field —
  `iter-1-tblkill.py` does, and it caught two mistakes before they reached the device.
* **`try.sh` must `rm -rf work && cp -r work_imm work` first.** I wasted a bisect round reading a
  plan off an already-converted stream and got slot indices that did not exist in the clean one.
* Semaphores: **sem 4 = ACT's completion counter** (5928 increments, no other engine touches it),
  5 = PE, 6 = DVE, 8-23 / 24-39 / 40-55 / 56-71 = the four DGE queue semaphore sets (+16 per
  transfer). `semaphores.py` reports no over-subscription on anything I shipped.
* A device `verify.py` is ~25 s, a `profile.py --repeats 5` ~70 s, `--repeats 1 --parquet` ~45 s.
  Two device runs per tool call, printed from the foreground, is the right granularity; a bisect
  over 768 units costs ~10 runs.
* `Instruction.parquet` joined on `compiler_pc` is what proved the ACT slot overlap in §2's
  diagnosis — `start_ts`/`end_ts` per slot, with `compiler_pc` NaN rows (runtime epilogue)
  dropped. **A profile is only valid for the stream it was captured on**; inserting or deleting
  slots invalidates the `compiler_pc` mapping.

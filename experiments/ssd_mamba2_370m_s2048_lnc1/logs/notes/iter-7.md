# iter-7 (ssd_mamba2_370m_s2048): 2952.0 -> 2905.1 us (-1.59%); against iter-0 5248.9 (1.807x)

**SHIPPED, one edit in two steps, bit-exact (`max_abs_err` identical to the last digit to
iters 1-6: output_0 2.058e-07 = 89% of the limit, output_1 1.993e-07 = 87%), `verify.py --runs 3`
all three runs bit-identical, `semaphores.py` clean (sem 4 = 5928, sem 6 = 4846, both still
`max_wait == total_inc`), `launch_rules.py` pass. Members changed: `sg00/DVE0.bin`
(15169 -> 13498 slots) and `sg00/Activation0.bin` (same size). md5
`3c05e45912fb64665180acc360fa38ed`.**

Reproduce from the iter-6 binary (`dcf3b8b93b18da3c3c82dfa1f94a99b6`):
```bash
neff.py unpack kernel.neff work/
profile.py --repeats 1 --parquet prof      # the plan builder needs the current schedule
SHIFT=1 ROOT=work PROF=prof python iter-7-bcast2.py
neff.py pack work/ kernel.neff
```

| step | edit | median us (5 rep) | spread | DVE busy |
|---|---|---|---|---|
| control (iter-6 binary) | - | 2952.0 (1 rep) / 2950.5 (iter-6) | 0.11% | 1655 |
| **7a** | 649 `exp_cs_last_N` broadcasts privatised away, 851 shuffles + 500 mask loads deleted | **2914.428** | 0.17% | 1505.8 |
| **7b** | +160 more shuffles deleted by *moving* their sem-6 increment one slot later | **2905.136** | 0.12% | 1477.5 |

---

## 0. THE HEADLINE: **SBUF IS NOT FULL. iter-6 §3b is wrong and I proved it on the device.**

iter-6 concluded "0 free 4-byte words that are free on all 128 partitions" and used that to close
three levers. That conclusion is an artifact of its own tool: `iter-6-sbfree.py` marks
`DmaPacketAggregated`'s **`read_offset`** as SBUF, but for an HBM->SBUF load the read side is an
**HBM** address, and `o % 0x40000` of an HBM address lands anywhere. That one line marks all
57344 words of every partition. Remove it and keep only the side whose `source`/`dest` column
says `SB`, and you get:

**512 bytes per partition free, in four 128-byte runs, free on all 128 partitions:**
```
0x32480-0x324FF   0x32980-0x329FF   0x33380-0x333FF   0x33880-0x338FF
```
**Device-verified usable**: step 7a writes 649 units' `exp_cs_last_Q` into words taken from those
runs and reads them back from two different engines; `max_abs_err` is bit-exact. (`iter-7-sbmap.py`
is the corrected map: compute operands + `scale/bias_imm_ptr` + in-slot DGE descriptors +
`DmaPacketAggregated` filtered by direction. Per-partition SBUF really is `0x38000`:
`neff_doc/docs/16` and the 1b3 experiment both say so, and offsets below 0x4000 **are** used by
this kernel, so there is no low hole either.)

Only **8 of the 128 words** are actually needed for step 7a (`POOLK=8` assigns all 649), so
**three full 128-byte-per-partition scratch buffers are still free** for the next iteration.
That is enough for e.g. iter-6 §6 item 3's `state_sb` relayout (one 64x32 fp32 tile is exactly
128 B/partition, and a 3-deep rotating pool is plenty) -- but see §4, I measured that lever and
it is *not* where the time is.

## 1. §7a What shipped: kill the `exp_cs_last_N` broadcast (iter-5 §3, re-opened and landed)

Per unit the source does `exp_cs_last_Q = broadcast(exp_cs_last, 128)` (4 `STREAM_SHUFFLE`,
offset `O_Q`, partitions 0/32/64/96; sole consumer the `decay_factor` DVE
`TENSOR_TENSOR` at `ssd_block.py:229`) **and** `exp_cs_last_N = broadcast(exp_cs_last, 64)`
(2 shuffles, offset `O_N`, partitions 64/96; sole consumer the ACT op at
`ssd_block.py:535`, through its `scale_value_imm_ptr = 64*0x40000 + O_N`). Same scalar, so
pointing 535 at the Q buffer is bit-exact and the N group becomes dead. iter-5 refuted this on
liveness (679 of 747 rejected) because `O_Q` is one of only **20** rotating offsets.

The fix is a private destination, which iter-6 thought impossible. Edit per unit:
`4 Q shuffle dsts -> dp*0x40000 + P`, `decay_factor TT src1 -> P`,
`535's scale_value_imm_ptr -> 64*0x40000 + P`, then delete the 2 N shuffles.

### 1a. Unit identity: anchor on the `exp_cs_last` ACT op, not on the shuffle's `src`
There are only **13 distinct `exp_cs_last` addresses for 4608 shuffles** (one is used 3372
times), so iter-5's "the Q group with the same `src` just before this N group" cannot identify a
unit -- my first attempt that way produced a pairing whose "lifetime" was *negative by 1.4 ms*.
What works: the 768 ACT ops of class `(func 7, nch 1, num_elem[0] 1, src SBUF, scale 0)` are
exactly `ssd_block.py:226`. For op *k* writing address `S`, the window
`(end of k, start of the next op writing S)` contains exactly the shuffles of unit *k* that read
`S`. Checked, not assumed:
* 2 shuffles with `dp in (0,32)` in every window (766/768; 2 fail "bad group shape");
* **exactly 1** candidate `decay_factor` TT in `(Q group, next write of O_Q)` for 641 of 649, and
  exactly 1 ACT64 reader in `(N group, next write of O_N)` for **649 of 649**;
* every consumer is claimed by at most one unit (asserted).

**408 of 768 units have the N group scheduled *before* the Q group on DVE** even though the
source has 227 before 533 -- the compiler reorders them, so do not gate on source order.

### 1b. The ordering proof (schedule-independent, no timestamp margins)
* `decay_factor` TT reads `P`: its DVE slot is after the Q group's last slot, DVE is in-order.
* `535` reads `P`: needs `gov6_ACT[ai] >= need6(max Q slot)` where `gov6_ACT[i]` is the running
  max of ACT's sem-6 waits before slot *i* and `need6(s) = cum6[j+1]` for the first DVE slot
  `j >= s` carrying a sem-6 increment. **649 of 768 pass**; 117 fail (of those, 101 wait on
  sem 5, 16 have no wait).
* WAR on the private word: assignment is greedy over the pool and a word is only reused when
  (i) the previous owner's `decay_factor` TT is at a **lower DVE slot** than the new owner's
  first Q write, and (ii) `gov4_DVE[new Q slot] >= need4(previous owner's 535)`. Both are proofs,
  not margins; with 128 words the observed slack is 489 us anyway.

### 1c. Deleting the N shuffles: **the ES stand-in fails here, deleting works**
* 851 of the 1298 N shuffles carry **no semaphore update** -> plain deletion (plus the paired
  `LOAD_MASK_SELECT` immediately before, when it is event-free: 500 of them). Bit-exact.
* The other 447 carry a sem-6 `SEM_INC_COMPLETE`. Replacing them with an `EVENT_SEMAPHORE` that
  inherits the increment **FAILS on the device** (`output_0` 3.04e-01, `output_1` 2.67e-02) --
  iter-5 §2.4b's hazard, now confirmed on **DVE**: a 34 ns ES fires the increment ~110 ns earlier
  than the 148 ns shuffle did, and a consumer released early reads SBUF that a *preceding* DVE
  instruction is still writing. (I isolated this with two probes: Q-side-only and
  ACT64-repoint-only both PASS bit-exact, so the address, the pairing and the scale-pointer
  rewrite are all innocent.)
* **§7b, new and reusable: move the increment instead of inheriting it.** If the first surviving
  DVE slot after the shuffle has **no update and no wait**, give *it* the shuffle's
  `events_update_mode/_idx/_semaphore_value` and delete the shuffle. Nothing in between carries
  an increment (everything in between was deleted and was event-free), so the whole sem-6 count
  sequence is unchanged and the increment simply fires **one instruction later** -- strictly
  safer than the ES, which fires earlier. **160 of 447 qualify, bit-exact, -9.3 us.**
  Do not relax "no wait" on the slot you land on (`events_semaphore_value` would clash) and do
  not skip *over* a waiting slot (that can close a cycle through the increment you are delaying).

## 2. Measured but not shipped (all offline, no device time needed to redo)

| extension | units / slots | why not shipped |
|---|---|---|
| transitive proof for the 101 ACT64s that wait on sem 5 (find the PE slot that produces that sem-5 count, require `gov6_PE` there `>= need6(Q group)`) | **+6 units** | 1.8 us of DVE, below the 3.5 us spread. `PEPROOF=1` |
| *set* the ACT64's sem-6 wait where it has none (safe: adding a wait loses no ordering, and the value is produced earlier in the same unit) | **+16 units** | 4.7 us of DVE. `FIXWAIT=1` |
| `SEM_ADD_IMM_COMPLETE` merge: fold the deleted shuffle's +1 into the next slot's own sem-6 +1 as a `+2` (mode 21), which only ever releases waiters *later* | **5 candidates** | almost every blocking successor has a sem-4/sem-5 **wait**, not an update. `MERGE=1` |
| ES stand-in for the 287 unshiftable shuffles under iter-5 §1's "drain covers" clause (`ESGATE=1`) | **139, device-VERIFIED bit-exact**, DVE busy 1477.5 -> 1457.7 | **measured 2904.796 us vs 2905.136 -- no wall change at all.** See §2a. Not shipped: zero gain, and it adds an early-firing-increment risk class |
| delete the dead `Y_off` memsets (`/tmp/mem.py`, census below) | **441 of 768 provably dead**; 244 belong to units iter-5 never folded, 83 have a reader inside the window (46 PE, 37 ACT); **only 26 are event-free** | the other 415 need `SHIFT`, which lands ~36% of the time -> ~176 slots x 85 ns = 15 us of DVE -> ~4 us of wall, below the 3 us spread |

### 2a. **On DVE, deleting a slot converts to wall time; making a slot cheaper does not.**
Three measured points on the same binary family:

| edit | DVE busy removed | DVE slots removed | wall |
|---|---|---|---|
| 7a (851 shuffles + 500 mask loads deleted) | 150 us | 1351 | **-37.6 us (0.25 us/us)** |
| 7b (160 more shuffles deleted) | 28 us | 160 | **-9.3 us (0.33 us/us)** |
| `ESGATE` (139 shuffles -> 34 ns ES **in place**) | 20 us | **0** | **-0.3 us (noise)** |

So the DVE cost that matters is **issue slots, not occupancy**. Do not spend an iteration making
DVE instructions shorter; only deleting them pays. (ACT behaves the same way per iter-6 §0 --
0.15 us/us there and the biggest ACT edits were in-place rewrites.)

### 2b. A `cum6` trap that costs a 70 s hang
The ES stand-in's extended `sem6 >= cum6[slot]` self-wait (iter-5 §2.4b) must use the increment
count **in the final stream**. `SHIFT` moves increments *later* and deletes slots, so the
original `cum6[slot]` can become unreachable before that slot -- the machine then stops and the
run ends as `[FAIL] run 0 ... exit 1, 70.56s`. Recompute the prefix sum after all moves and
deletions (`cum6f` in `iter-7-bcast2.py`). With the original value: hang. With `cum6f`: PASS,
bit-exact. `semaphores.py` does **not** catch this -- totals were conserved in both.

## 3. §7c The big open lever I characterised but did not crack: `exp_cs_last` is redundant

`ssd_block.py:226` costs **768 x 280 ns = 215 us of ACT busy and 129 us of ACT pre-gap**, and
`exp_cs_last == exp_cs_col[Q-1]` (both are ACT `exp` of the same fp32 scalar: `cs_col` is
`transpose(cs_row)` so `cs_col[127] == cs_row[0,127]` bit-for-bit). iter-6 §3a closed this on
"the shuffle reads the base partition of a 32-aligned group". **Three of the four unknowns are
now closed:**

1. **`exp_cs_col`'s address is identifiable, 452/768 units, cross-validated with 0 errors.**
   Per unit: the `decay_factor` TT's *other* source is `exp_neg_cs_col`; take the 4 ACT
   `(func 7, nch 128, num_elem[0] 1, src PSUM)` ops before the unit's 226, one of them is
   `exp_neg_cs_col`, the other (unique for 452 units) is `exp_cs_col`. **All 452 land in the set
   of 13 addresses that iter-5/6's folded DVE MULTs use as their `exp_cs_col` broadcast source**
   -- independent confirmation. (`EXPLAST=1` in `iter-7-bcast2.py`.)
2. **Ordering is free**: `exp_cs_col` is written by an ACT op that precedes 226 in ACT program
   order, so whatever already orders a shuffle after 226 orders it after that op too. No wait
   changes needed.
3. **The shuffle mask lives in `LOAD_MASK_SELECT` bytes 32..63** (`immediate_uint8[32]`,
   `offset 32`), and writing it **does** change the result (probe: keep the original `src`, set
   all 32 bytes to 31 -> wrong answer, where unmodified is bit-exact). **`mask_enable` on the
   shuffle is irrelevant**: `ELME=0` and `ELME=1` give *byte-identical* `max_abs_err`.
4. **Still open: the lane encoding.** Five device probes, all `FAIL`, all with the ACT op still
   in place (so the only defect is the value the shuffle read):

   | probe | `output_0` max_abs_err | `output_1` |
   |---|---|---|
   | `src` = partition **127** of `exp_cs_col`, mask untouched | 5.750e+00 | 2.177e+00 |
   | `src` = partition 96, mask = 31 byte-per-lane | 5.586e+00 | 2.120e+00 |
   | `src` = partition 96, mask = 31 packed 5-bit (bytes 32..51 = 0xFF) | 6.874e+00 | 1.876e+00 |
   | `src` = partition **0**, mask untouched (tests `exp_cs_last == exp_cs_col[0]`) | 7.180e+00 | 2.414e+00 |
   | `src` kept, mask = 127 byte-per-lane | 2.816e+00 | **8.941e-02**, cos 0.536 |

   **How to close it without more blind guesses** (and this is the cheapest 1.5% left in the
   kernel): the golden inputs are in `bench/inputs/`, so compute, in numpy, for every candidate
   partition *j* what `max_abs_err` *would* be if `exp_cs_last` were replaced by
   `exp_cs_col[j]` for those 452 units, and match the five measured numbers above against that
   table. That tells you exactly which partition each encoding actually read -- i.e. it decodes
   the mask from data already paid for. (iter-5 §2.4c used the same "fingerprint on the error
   value" trick successfully.)

## 4. Where the 2905 us is, measured on the shipped binary (`prof7/`), and what it rules out

Engine busy: ACT **1809 (62.3%)**, Pool 1559 (53.6%), DVE 1478 (50.9%), PE 1269 (43.6%),
DMA 988, SP 588. Wall / max-engine-busy = **1.6**, so ~1100 us is dependency slack, not work.

**The single biggest item in the program is still iter-4 §4.1's Pool barrier, and it is bigger
than any instruction class.** At t = 848..960 us, **four engines stall simultaneously**:
`Scalar` 111.5 us (pc 2771, the `EVENT_SEMAPHORE` with `wait s23>=800`), `Tensor` 112.8 (pc 2969),
`Vector` 112.1 (pc 3544), `Sync` 122.6 (pc 838). In that window Pool issues ~130 DGE triggers and
is ~76% busy (it has ~27 us of its *own* sem-6 gaps inside the stall, i.e. the store/load
head-of-line block is still live). Every engine's next-largest gap is **4-20 us**, so this one
event is 3.9% of the wall and nothing else is close.

Pool trigger cost is flat: `ssd_block.py:572` 768 x 662 ns, `:447` 768 x 651, `:203` 768 x 651.
There is no cheap trigger to merge and no expensive one to fix.

**Refuted this iteration: the `state_sb` relayout (iter-6 §6 item 3) is not worth doing.**
It is unblocked now (§0 gives the scratch), but the prize isn't there: the 768 line-535 ACT ops
have a combined pre-gap of **279.9 us**, and the part of that gap during which PE *and* DVE *and*
Pool are all under 20% busy is **0.6 us**. The stall is real but it is concurrent with other
engines' work, so removing it buys almost nothing. (Also note 114 of the 768 have no wait at all
and 30 wait on sem 6, so only 624 are even WAR-stalled on PE.)

## 5. Ranked plan for iter-8

1. **Close the mask encoding and delete `ssd_block.py:226`** (§3). 215 us of ACT busy + 129 us of
   ACT gap, bit-exact by construction, identification already done and cross-validated, ordering
   free, and the numpy fingerprint in §3.4 closes the last unknown with **zero** device time.
   This is the best-priced lever in the kernel.
2. **The Pool barrier** (§4, iter-4 §4). 112 us in one event, 3.9%. Nothing else is above 1%.
   iter-4's queue-reassignment program is still the only stated way in; one idea it did not
   price: moving the 768 `:572` **stores** off Pool entirely onto SP (684 us busy / 2230 us gap,
   it already issues 816 DGE triggers on sems 56-71) would stop stores from head-of-line-blocking
   loads *in Pool's own in-order stream*, which is where the blocking actually is -- queue
   reassignment alone does not fix that, because Pool still issues in program order.
3. **Do not** finish §7a's tail. `ESGATE=1` is verified correct and buys **nothing** (§2a), the
   dead-memset deletion is ~4 us (§2), and `PEPROOF=1 FIXWAIT=1` is +22 units = ~6 us of DVE
   busy, i.e. ~2 us of wall. This lever is done at 649/768.
4. **Do not** redo: iter-6 §3 (except §3a/§3b, both of which this iteration overturned),
   iter-5 §5, iter-4 §5, iter-3 §4, iter-2 §4, iter-1's list. And do not re-derive the SBUF
   free-space map from `DmaPacketAggregated` without filtering by direction (§0).

## 6. Tool notes (adds to iters 1-6)

* **`DmaPacketAggregated` has `source` / `dest` columns** (`[[INPUT]]`, `[SB]`, ...) and
  `read_sbuf_partitions` / `write_sbuf_partitions`. Only the side whose column says `SB` is an
  SBUF address; the other side is HBM. iter-6's free-space map ignored this and its conclusion
  ("SBUF is full") was wrong -- which cost three iterations' worth of closed levers.
* **Free SBUF in this NEFF**: `0x32480`, `0x32980`, `0x33380`, `0x33880`, 128 B each, on all 128
  partitions. Device-verified. 8 words are in use by the shipped edit.
* `LOAD_MASK_SELECT` (opcode 0x69) pairs 1:1 with `STREAM_SHUFFLE` and always sits **immediately
  before** it (4608/4608). Its payload is all zeros in this NEFF = `shuffle_mask [0]*32`; the
  mask bytes are `immediate_uint8[32]` at slot **offset 32**, and they are live (§3.3).
  `STREAM_SHUFFLE.mask_enable` is 0 everywhere and setting it to 1 changes nothing.
* `events_semaphore_value` on an instruction that has **both** a wait and an update is the *wait
  threshold* (4872 ACTIVATEs in this NEFF do that); an instruction with an update and no wait has
  `events_semaphore_value == 0`, i.e. `SEM_INC_COMPLETE` is an implicit +1. So you may set a wait
  threshold on a slot that already updates, but you may not put an `ADD_IMM` amount on a slot
  that waits.
* `Instruction.fields()` entries carry `offset`/`width`/`stride`/`count`, which is how to find
  where to patch raw bytes when there is no named scalar field.
* `profile.py --parquet`'s `Instruction.parquet` has **one `pcore_idx` per engine** here, but
  instruction intervals on the same engine **overlap** (PE `LDWEIGHTS`/`MATMUL` pipeline), so
  `sum(duration_ns)` per engine can exceed the wall (Tensor: 3493 us of 2905). Use
  `max(start - prev_end, 0)` for gaps -- that is still exact -- but never sum durations over a
  window and call it busy time.
* Device costs today: `verify.py` 40 s, `--runs 3` 100 s, `profile.py --repeats 1 --parquet` 50 s,
  `--repeats 5` 90 s. Spread on this kernel is now 0.12-0.17% = 3.5-5 us, so an edit must be
  worth >10 us to be separable in one measurement.

Scripts next to this file: **`iter-7-bcast2.py` is the whole iteration** (env
`ROOT PROF DRY LIM ONLY POOLK MARGIN SOUND SHIFT WIDE MERGE ESMODE DRAIN NOKILL NOACT FIXWAIT
PEPROOF EXPLAST ELPART ELLANE ELME ELPACK ELKEEPSRC`; the shipped build is `SHIFT=1` and nothing
else), **`iter-7-sbmap.py` is §0's corrected SBUF free-space map**.

# iter-4 — selective_scan b1 c768 s1024 n16 bf16

**Result: 510.558 -> 508.8 us (-1.7 us, -0.34%). Cumulative from the unedited NEFF:
602.64 -> 508.8 (-15.6%, 1.184x).** Numerics bit-identical: output_0 max_abs_err 6.245e-02
(83% of limit, cosine 0.999998), output_1 4.268e-05 — the exact baseline figures.
`semaphores.py` clean, `launch_rules.py` clean. vs the iter-3 artifact only **2 of 34 members
changed**: `sg00/DVE0.bin` (11 slots) and `sg00/Pool0.bin` (2 slots); no slot inserted or
deleted, no semaphore ordinal touched.
Measurements of the shipped file (md5 `8ae099ad...`): three independent 5/5/3-repeat medians
**508.920 / 508.826 / 508.651** us (spreads 0.14 / 0.04 / 0.06%), DVE busy 491.2-492.0.
iter-3's artifact re-measured at the start of this iteration: **510.558** (spread 0.08%).

Read `iter-1.md` (build scripts + calibration), `iter-2.md` and `iter-3.md` first. Everything in
them still holds. This iteration adds **one new lever (DVE/Pool `COPY` is a 2-stream op and is
261 ns cheaper than `TENSOR_TENSOR`)** and **five device-measured refutations** that between them
close the whole "make Pool cheaper" family, which is where the last 10 us of accessible time
lives.

---

## THE ONE BIG NEW FACT: DVE is ~2% SBUF-bandwidth-limited by Pool's y-accumulation

Timing-only probe, Pool's 170 `y += Cs` `TENSOR_TENSOR`s shrunk in `num_elem[0]` (output wrong,
nothing else changed — this scales Pool's SBUF traffic without touching instruction count,
order, addresses or semaphores):

| Pool ADD elements | Pool traffic | exec us | DVE busy us | Pool busy us |
|---|---|---|---|---|
| 512 (shipped) | 1.00 | 510.56 | 493.97 | 415.40 |
| 256 | 0.50 | 501.35 | 486.48 | 331.92 |
| 1 | ~0.00 | **497.39** | **483.34** | 242.28 |

So **Pool's SBUF traffic inflates DVE's own busy time by 10.6 us and exec by 13.2 us**, and the
relation is *convex* in traffic (0 -> 0.5 costs 3.1 us, 0.5 -> 1.0 costs 7.5 us). This is the
mechanism behind iter-1's observation that DVE `TENSOR_TENSOR` went 672 -> 687 ns when the ADDs
moved to Pool. It is also the reason exec (508.8) sits 13 us above DVE's own floor.

**It is the largest accessible number left in the kernel — and every route to it is now closed
(R1-R5 below).** The decisive facts: the contention is *raw SBUF bandwidth*, not addresses (R3),
and Pool's widest arithmetic op takes 2 inputs, so an n-term sum costs exactly `3(n-1)` SBUF
units and cannot be made cheaper (R1 + R2).

Pool duration model, fitted on the same three points (useful, and not in the earlier notes):
**Pool `TENSOR_TENSOR` = 0.98 us fixed + 2.0 ns per element** (512 fp32 x 128ch = 2.0 us). The
fixed cost is ~5x DVE's. Pool non-ADD work is ~75 us (117 COPY at 186 ns + 60 DMA_DIRECT2D at
657 ns).

---

## What shipped: `ADD -> COPY` for the 12 zero-buffer first writers (-1.7 us)

New calibration that makes this work (measured on this NEFF, `prof/Instruction.parquet`):

| DVE, 512 elem, fp32, nac=128 | median duration | operand streams |
|---|---|---|
| `TENSOR_TENSOR_ARITH_OP` | **677 ns** | 3 (2 reads + 1 write) |
| **`COPY` (0x46)** | **416 ns** | **2 (1 read + 1 write)** |
| `MEMSET` (512) | 485 ns (iter-1) | 1 |
| `TENSOR_TENSOR_SCAN_ARITH` | 1271 ns | 3, but latency-bound |

i.e. a DVE 512-element fp32 operand stream costs ~261 ns, and **dropping one stream is worth
261 ns.** `COPY` is enabled on DVE *and* Pool in this NEFF's opcode table (it was previously used
only on Pool).

iter-3 made each group's first y writer read a shared zero buffer at `0x15400` as `src0`
(`y = 0 + Cs_k`). That is a 3-stream `TENSOR_TENSOR` doing the work of a 2-stream `COPY`. The
edit:

* the 12 first writers — DVE slots **105, 192, 280, 369, 457, 547, 630, 724, 899, 982** and Pool
  slots **8, 303** (exactly iter-3's list) — rewritten as `COPY` (`src` = the ADD's `src1` = `Cs`,
  `dst` = the ADD's `dst` = `y`, `num_elem=[512,1,1,1]`, `step_elem=[1,1,1,1]`, `in/out_dtype`
  FP32, `nac` 128, `op_dim` UNUSED), keeping the identical `events_*` wait / update / value.
  Built by cloning an existing Pool `COPY`'s 64 bytes as a template and overwriting fields, so
  the struct-specific bits (`op_dim`, `header_inst_word_len=16`) are right by construction.
* the one surviving 512-element zero `MEMSET` (DVE slot 3) shrunk to 1 element
  (`dst_element_count` **and** `dst_mem_pattern_num_elem`): nothing reads `0x15400` any more.

Accounting: 10 x 261 ns (DVE) + 385 ns (MEMSET) = 2.99 us predicted; DVE busy fell
493.97 -> 491.2-492.0 (**-2.4 us**), exec -1.7 us. The 2 Pool conversions are worth ~0.7 us each
of Pool duration (invisible in exec — see R4).

Build script is ~45 lines; reproduce with iter-1's `build3.py` skeleton plus the template trick.
The script used was `xp/zcopy.py` (gone after the iteration); its whole body is described above.

**Where else does the `COPY` trick apply? Nowhere, I checked.** The full DVE census after the edit
is 384 `MULT` (PSUM x SBUF -> SBUF), 192 `SCAN`, 24 `y+=Cs` `ADD`, 12 `dt*x` `MULT`, 10 `COPY`,
24 `MEMSET`, 410 `EVENT_SEMAPHORE`, 1 `SET_ORDERING_MODE`. Every remaining `TENSOR_TENSOR` has two
genuinely live operands, so no further stream can be dropped.

---

## Refutations, each with the device measurement that closed it

### R1. **Pool cannot execute `TENSOR_REDUCE_ARITH_OP` (0x42)** — NRT_TIMEOUT, 70 s. CLOSED.
This was the only way to cut the y-accumulation's SBUF traffic (an n-wide reduce costs `n+1` units
instead of `3(n-1)`). One Pool `y += Cs` ADD replaced by an equivalent 2-input reduce
(`src num_elem=[2,512,1,1]`, `step=[(Cs-y)/4,1,1,1]`, `op_dim=X`, `dst num_elem=[1,512,1,1]`,
same wait/update) -> **70 s NRT_TIMEOUT**, iter-2's R3 signature for "the engine cannot retire
this opcode". Consistent with the corpus: 0x42 has **9738 slots, all DVE, 0 Pool**.

**The encoding is worth keeping** (it took a while to find and `isa.py` does not expose it):
`S4D4_TR_STRUCT` has **no `op` field in `isa.py`** — the ALU op lives at raw **byte offset 36**
(`ALU_OP`: 4 = ADD, 8 = MAX). `isa.py` leaves byte 36 unnamed, so you must patch the encoded
bytes directly. `op_dim` (`TENSOR_SUBDIM`, byte 37) is cumulative-from-dim-0 (`X`, `XY`, `XYZ`,
`XYZW`) — you can only reduce the *innermost* dims, so a reduce must put the summed axis in
`num_elem[0]`. Reference instance: `artifacts/unpacked/00-Attention-B/payload/sg01/DVE0.bin`
slot 1582.

### R2. **Pool's complete arithmetic repertoire is 2-input, so the y-sum's traffic is irreducible.** CLOSED.
Corpus census of every opcode with any Pool slot: `TENSOR_TENSOR` (0x41), `TENSOR_SCALAR`,
`COPY`, `CAST`, `TOP_K`, `TENSOR_SCALAR_ADDR`, `CROSS_LANE_REDUCE` (reduces across *partitions*,
not along the free axis), `IOTA`, `TENSOR_SCALAR_AFFINE_SELECT`, plus control/DMA.
**No `TENSOR_REDUCE`, no `SCALAR_TENSOR_TENSOR` (0x9d), no `TENSOR_CUMULATIVE`.** So summing 17
terms on Pool costs exactly 16 x (2 reads + 1 write) and there is no wider instruction to use.
Together with R1 and R3 this closes the 10.6 us.
(For the record the same is true of the alternatives: ACT cannot do `TENSOR_TENSOR` at all
(iter-1), and moving the sum to the DMA engines does not help because *the contention is SBUF
bandwidth, not engine occupancy* — the same bytes move either way.)

### R3. **The contention is raw SBUF bandwidth, not bank/address placement.** CLOSED.
Timing-only probe: all 170 Pool ADDs repointed to `src0 = dst = 0x16000`, `src1 = 0x18000` (the
free SBUF gap), same traffic / count / order / semaphores. **510.96 us, DVE busy 493.6** — i.e.
indistinguishable from the shipped 510.56 / 493.97. So there is no cheap win from relocating
`y`/`Cs` into a different SBUF region, and **the whole "re-address the rotating buffer families"
lever that iter-1 ranked #2 and iter-2 ranked #1 has no bandwidth component** (its only remaining
rationale would be loosening WAR deadlines, which R4 shows is worthless).

### R4. **Pairwise row-merging the Pool `y+=Cs` ADDs is a 62 us loss.** CLOSED — and this is the
### most useful negative result, because it shows Pool *latency*, not Pool *duration*, is what DVE sees.
Pool instruction count is expensive (0.98 us fixed each), so merging looked free. Built: 75
adjacent ADD pairs merged into one `[512,2]` 2-row `TENSOR_TENSOR` at the later slot
(`dst`/`src0` `step_elem[1] = 0`, `src1 step_elem[1] = (Cs_b - Cs_a)/4`), earlier slot deleted,
all S3 ordinals remapped by instruction identity in all five streams, with a deadlock guard
(every exact S3 waiter of the deleted ordinal must be on DVE at a slot whose S6 count-before is
>= the merged ADD's own S6 wait value; 154 of 169 adjacent pairs pass).
* **Verified bit-identical** — so the mechanism works (see the Unlock below) — and **573.24 us
  (+62.7 us)**. Pool busy fell 415.4 -> 356.8 (-58, i.e. 0.77 us saved per merge) but **DVE busy
  rose 493.97 -> 552.9 (+59)**, which on this profiler is DVE *blocking* counted into instruction
  duration: 59/75 = 0.79 us of stall per merge.
* Mechanism, exactly: the merged ADD can only start after DVE's `Cs_b` MULT, and the DVE waiter it
  now gates is the old `Cs_a` WAR deadline, which with the compiler's **3-deep `Cs` rotation** is
  only 2 DVE states (~2.65 us) later. The merged ADD costs ~3.0 us plus Pool queueing. 3.0 - 2.65
  ~ 0.4-0.8 us of stall. Same mechanism as iter-1 refuted-item 1, iter-2 R4 and iter-3 R2/R4.
* **Therefore: reducing Pool's instruction count or duration buys nothing (R3 says DVE only feels
  bytes), while increasing any single Pool ADD's latency costs DVE directly.** Any future wider
  Pool op would need a deeper `Cs` rotation first — and by R1+R2 there is no wider Pool op, so
  deepening the rotation is pointless. This also means **the 5 pairs with no exact S3 waiter are
  the only safe merges and they are worth ~0.**

### R5. **The review's candidate #1 (prune the remaining 424 DVE S6 self-waits) has a 0.94 us ceiling.** CLOSED.
Timing-only probe: `events_wait_mode = NONE` on **all 420** DVE primary waits on S6 (364 on
`EVENT_SEMAPHORE`, 56 on compute). **509.62 us, DVE busy 493.4** — a total of **0.94 us, i.e.
2.2 ns per wait**, not the 36 ns/wait the review extrapolated from iter-3's edit (that -6.2 us was
the MEMSET collapse, not the waits). So an already-satisfied DVE semaphore wait does **not** break
the ~76 ns instruction overlap. Even stripping every one of them — which would be a real race, since
most are genuine intra-DVE RAW guards — cannot pay for itself. **Do not spend another iteration here.**

---

## The unlock, for whoever finds a use for it

**A `TENSOR_TENSOR` with a 2-D access pattern and `dst`/`src0` `step_elem[1] = 0` performs the
rows sequentially and sees its own writes.** Verified bit-identical on **Pool** with 75 merged
accumulations (`y = y + Cs_a` then `y = y + Cs_b` as one `[512,2]` instruction). So multi-row
read-modify-write accumulation into a single destination is legal and ordered. It is useless
*here* (R4), but it is a general way to fold k accumulations into one instruction on any engine,
and `step_elem = 0` is now verified (iter-3 listed it as unverified).

---

## Where the 508.8 us goes (shipped NEFF, `prof/Instruction.parquet`, `compiler_pc.notna()`)

| | |
|---|---|
| **DVE compute union (excl. `EVENT_SEMAPHORE`)** | **495.2 us** |
| DVE compute span | 506.4 us |
| DVE gap, 24 gaps | **11.2 us** |
| &nbsp;&nbsp;startup input DMA | 4.32 us |
| &nbsp;&nbsp;DGE queue S11 (carry spill, Pool DMA) | 1.97 us |
| &nbsp;&nbsp;3 gaps of 837-935 ns (ACT S4 `dA` x2, DGE S9 y-output DMA) | 2.70 us |
| &nbsp;&nbsp;19 gaps of 130-350 ns (issue granularity) | 2.21 us |

DVE duration census: `TENSOR_TENSOR` 420 x 677 ns (mean 697) = 292.7 us, `TENSOR_TENSOR_SCAN`
192 x 1271 = 244.9, `COPY` 10 x 416 = 4.2, `MEMSET` 24 x 65 = 2.1, 410 `EVENT_SEMAPHORE` x 53.
Sum 544.6 vs union 495.2 => ~76 ns of overlap per compute instruction (646 of them) — unchanged
from iter-3, and R5 shows waits are not what sets it.
Pool: `TENSOR_TENSOR` 168 x 2163 ns = 338.2 us, `DMA_DIRECT2D` 60 x 657 = 39.4, `COPY` 117 x 186
= 38.8. Per-engine busy: DVE 491.9 (96.7%), Pool 417, PE 218.8, ACT 146.0, DMA 105.6, SP 48.3.

**The accounting that bounds everything.** Fitting the stream model to the census reproduces DVE
busy to 0.1 us: `(384+24+12) x 677 + 192 x 1271 + 10 x 416 + 24 x 65 - 76 ns x 646 = 491.4` vs
measured 491.9. Of that, 384 PSUM `MULT`s (260 us) cannot leave DVE (iter-1 #8, iter-2 R1) and
192 `SCAN`s (245 us) cannot leave DVE (iter-2 R3), cannot merge (iter-1), and are dtype-insensitive
(iter-1 #6). The scan is **latency-bound, not bandwidth-bound**: 1271 ns / 512 steps = 2.48 ns =
~3.5 cycles per serial step at 1.4 GHz, and the carry is a single chain across all access-pattern
dims (iter-1's falsifier), so no interleaving of independent chains is expressible.
**505 us of DVE duration is immovable; 495.2 us is the measured union. The kernel is at its floor.**

---

## What I would try next — I believe there is nothing left above 1%

My honest estimate of remaining headroom is **~2 us (0.4%)**, all of it inaccessible with
instruction-level edits. Ranked, with why each is dead:

1. **Nothing.** The three families the earlier reviews ranked as >2% are now all closed by device
   measurement on this NEFF: engine rebalancing (iter-1 #8, iter-2 R1/R3, iter-3 R2), instruction
   merging (iter-3 R3 for `Cs`, R4 here for the y-ADDs, iter-1 for the scans and `dBx`), and
   re-addressing / bank placement (R3 here). The 10.6 us of Pool-traffic inflation is closed by
   R1+R2+R3.
2. The 12 placeholder 1-element `MEMSET`s (DVE slots 3, 100, 188, 275, 365, 452, 542, 623, 719,
   800, 894, 975) exist only to carry S6 ordinals. 9 of the 12 have exactly 1 exact S6 waiter;
   3 (slots 3, 623, 975) have **no S6 ordinal at all** and are dead code. Deleting all 12 and
   renumbering S6 would be *safe in the usual way* — a `V == ord` waiter automatically re-points to
   the following `COPY`, which is strictly later and is the dependency it actually wants — but at
   65 ns each and ~76 ns of instruction overlap, the net is **~0.4 us** for ~1800 rewritten wait
   fields. Not worth it.
3. The 1.97 us DGE-S11 carry-spill gap: iter-3's R4 measured the generic fix at +21.7 us and
   bounded the targeted fix at <=1.4 us with an expected net of ~0. The 4.32 us startup DMA and the
   2.03 us ACT `dA` pipeline fill were closed by arithmetic in iter-2/iter-3.
4. If a future iteration insists on burning a run: `dBx` into PSUM instead of SBUF would move 2 of
   the 12 SBUF units per state off SBUF (~7 us of DVE by the traffic curve), but it needs 3 free
   PSUM banks and all 8 are held by the 4-deep B and C rotations, which iter-3 R3 showed is already
   the binding PE/DVE constraint. I would not attempt it.

## Gotchas added this iteration

* **`isa.py` does not expose the reduce ALU op.** `S4D4_TR_STRUCT` (`TENSOR_REDUCE_ARITH_OP`,
  `COPY`, `CAST`, `TENSOR_CUMULATIVE_*`, `STREAM_*`) has an **unnamed byte at offset 36** holding
  `ALU_OP` (4 = ADD, 8 = MAX). Patch the encoded bytes. `op_dim` at byte 37 is
  cumulative-from-dim-0, so the reduced axis must be `num_elem[0]`.
* **Build a new-opcode instruction by cloning an existing one of that opcode**, then overwrite
  fields. Constructing from zeroed bytes silently leaves struct-specific fields (`op_dim`,
  `header_inst_word_len`) wrong, and `isa.py` will not complain.
* `MEMSET` carries both `dst_element_count` and `dst_mem_pattern_num_elem` — set both (iter-3 said
  this; confirmed again).
* **Profiler "engine busy" includes blocking time**: in R4 DVE busy rose 59 us with no change to
  what DVE computes. Read a busy-time rise on an unmodified engine as a stall, not as slower work.
* Device-time oracle, re-confirmed: ~10 s = ran; ~10 s + `PSUM_COLLISION`/nrt 1004 = WAR violation;
  **~70 s + `error: 5` = NRT_TIMEOUT = deadlock or an opcode the engine cannot retire.**
* `tools/profile.py --repeats 2` (~40 s) is enough for a probe that is expected to move >5 us;
  use `--repeats 5` (~75 s) and repeat it for anything under 2 us. Spread on this NEFF is
  0.03-0.14%, i.e. 0.15-0.7 us, so **do not claim anything below ~1 us from one run**.
* Probe scripts for all of the above lived in `xp/` (gone after the iteration): `probeA.py`/
  `probeA2.py` (Pool ADD `num_elem` scaling — the traffic curve), `probeB.py` (strip DVE S6
  waits), `probeC.py` (relocate Pool ADD operands), `merge_add.py` (the Pool ADD row-merge with
  the S3 remap + deadlock guard), `red_probe.py` (Pool `TENSOR_REDUCE`), `zcopy.py` (the shipped
  edit), `census.py`/`sizes.py`/`waiters.py`/`msets.py` (analysis). Each is 30-100 lines built on
  iter-1's `build3.py` skeleton, which `iter-1.md` reproduces in full.

# iter-5 — selective_scan b1 c768 s1024 n16 bf16

**Result: 508.712 -> 501.33/501.50 us (-1.44%). Cumulative from the unedited NEFF:
602.637 -> 501.4 (-16.8%, 1.2021x).** Numerics bit-identical at every step: output_0
max_abs_err 6.245e-02 (83% of limit, cosine 0.999998), output_1 4.268e-05 — the exact baseline
figures, including output_1 after relocating the carry array it is spilled from.
`semaphores.py` clean (S3 282/282, S4 194/194, S5 374/374, S6 640/640), `launch_rules.py` clean.
3 of 34 members changed: `sg00/DVE0.bin` (1055 slots, -2), `sg00/Pool0.bin`, `sg00/Activation0.bin`.
Shipped md5 `c34c2e23...`. Three independent 5-repeat medians **501.332 / 501.496 / 501.483**
(spread 0.04-0.06% each), plus `verify.py --runs 4` all bit-identical; the iter-4 artifact
re-measured at the start of this iteration: **508.712** (spread 0.07%, DVE busy 491.1).
Per-edit ladder, each a 5-repeat median: 508.712 -> **505.510** (Cs producer -> y)
-> **503.626** (carry array de-aliased) -> **502.333** (y/state de-aliased)
-> **501.540** (ACT table load hoisted) -> **501.33/501.48** (ACT dA writers hoisted).

Read `iter-1.md` (build-script skeleton + calibration), then `iter-4.md` (the Pool-traffic curve
and the five refutations that close the "make Pool cheaper" family). Everything in them still
holds. **iter-3 and iter-4 both concluded "the kernel is at its floor"; that was wrong by 1.4%,
and the reason is instructive: all four previous iterations attacked DVE *duration*, and the
remaining time was in DVE *gap* — specifically in three compiler barriers that exist only
because the compiler aliased two live buffers into the same SBUF bytes.** The new lever is
**"relocate the buffer, then delete the barrier"**, and it is still only half-mined.

---

## THE NEW LEVER: a DVE stall on a DGE queue is usually a WAR on an *aliased* SBUF buffer

Three of the four large DVE gaps iter-3/iter-4 catalogued as "DMA-bound, irreducible" were not
DMA-bandwidth at all. In each case a DVE instruction waits on a DMA-queue semaphore because
**the 2 KB extent it is about to write overlaps a small live buffer that a DMA is still reading.**
Move the small buffer into free SBUF and the wait becomes vacuous; delete it and the gap goes to
zero. This is cheap (a handful of address fields), bit-exact, and `verify.py` catches a missed
reference immediately.

| gap (iter-4 artifact) | what it really was | fix | measured |
|---|---|---|---|
| 2.01 us, DVE pc 176->178, `ES W=S11>=32` | the `output_1` carry spill DMA (Pool slot 72, `semaphore 11`, `+16`) reads the 16-word carry array at **0x4100**; DVE 178 writes a 512-elem fp32 dBx buffer at **0x4000**, whose extent 0x4000-0x47FF *contains* 0x4100-0x413F | carry array -> 0x16000 (53 operand fields + the DMA's `src_start_addr_addr_immediate`), then `events_wait_mode = NONE` on the ES | **503.626 (-1.88)** |
| 0.92 us, DVE pc 95->97, `ES W=S9>=32` | the group-0 **y** output DMA (Pool slot 45) reads y at **0x4940**; DVE 97's scan writes its state output to the same 0x4940 | scan 97 `dst` and MULT 104 `src1` -> 0x16100, Pool 48's merged carry `COPY` `src` 0x513c -> 0x168fc with `step_elem[1]` 2560 -> -15344, then drop the ES wait | **502.333 (-1.29)** |
| 0.87 us, DVE pc 8->10, `W S4>=1` | ACT's 1.28 us `ACTIVATION_TABLE_LOAD` sat *behind* an `ES W=S10>=16`, so ACT's first `dA` finished 1.28 us late. **The table load has no address operands at all** (`act_tbl_sel 0`) and the S10 DMA loads **A** (it is the `scale_value_imm_ptr = 0x37f00` of the `dA` ACTIVATE), so the ES does not guard it | swap ACT slots 1 and 2 (both have no semaphore update -> pure 128-byte swap, no renumbering) | **501.540 (-0.79)** |
| 0.64 us, DVE pc 36->38, `W S4>=7` | ACT's 7th-9th `dA` writers sat behind `ES W=S17>=16` + an unrelated ACTIVATE, so they issued at t=31 instead of t=20 | reorder ACT slots 9..13 to `[11,12,13,9,10]` and give slots 11/12 the same conservative `W S6>=10` slot 13 already had | 501.332 (**-0.21, inside the spread**; the *gap* did close — see R3) |

**After these four, DVE's total gap is 5.9 us over 17 gaps, and the only one above 0.35 us is the
4.21 us startup input DMA.** `exec 501.3 = DVE compute union ~493 + 5.9 us gap + ~2.4 us
prologue/drain`.

### Recipe for finding the next one (10 minutes, no device time)
1. Gap-attribute DVE (iter-3's recipe; script below). Any gap whose binding wait is on
   **S8..S23** is a candidate.
2. Decode the ES/compute slot that carries the wait and the DVE slot *immediately after* it.
   Compute that slot's **full dst extent** = `start + num_elem[0] * sizeof(dtype)` (512 fp32 =
   2048 B). Do **not** compare start addresses: every one of these hazards was invisible at
   start-address granularity.
3. Find the DGE DMA that increments that semaphore to that value (`PSEUDO_DMA_DIRECT2D` carries
   `semaphore` and `sem_increment` as decoded fields; walk Pool in order counting increments per
   semaphore). Its `src_start_addr_addr_immediate & 0xffffffff` is the SBUF address at risk.
4. If the DMA's source range lies inside the next DVE write's extent, relocate the *smaller* of
   the two into free SBUF and delete the wait.

### Gotchas this lever has (both cost me a failed device run or nearly did)
* **A buffer's last element is referenced by a different address.** The Pool carry `COPY`s read
  `base + 511*4`, so y at 0x4940 is also referenced as **0x513c**. My first attempt relocated
  only the exact value 0x4940, passed output_0, and broke output_1
  (`max_abs_err 1.153e+01`). **Always scan the whole extent `[base, base+2048)`, not the base.**
* **Those carry COPYs are the pairwise-merged ones from iter-1**, so they read two buffers with a
  dim-1 `step_elem`. Relocating one member of a pair means recomputing the step
  (`(other - new_first)/4`, int16, fits easily). Keep the dst pattern untouched.
* **An extent scan over `[0x4940, 0x5140)` also catches an unrelated buffer that starts at
  0x5000**, because the compiler aliases 0x4940 and 0x5000 as time-disjoint buffers. Filter by
  lifetime (everything 0x5000 was >400 slots later), do not blindly rewrite.
* **Reordering an engine's stream around an instruction that does *not* update its semaphore is a
  trap.** ACT slot 10 (`dst 0xe340`, `U = NONE`) has no S4 ordinal of its own, so its completion
  is signalled by *the next* S4 increment. Before hoisting slots 11-13 above it I had to prove
  that no waiter on S4 ordinals 7/8/9 needs slot 10: the only waiters are DVE ES 37/42/47, which
  gate the scans reading exactly slots 11/12/13's `dst` (0x5140/0x8140/0xb140), and ordinal 10
  still lands after slot 10. Check this before any such reorder.
* **The ACT `dA` buffer rotation is 6-deep** (0x5140, 0x8140, 0xb140, 0x5940, 0x7140, 0x8940) and
  the compiler's WAR rule for the k-th `dA` writer is `S6 >= ord(the scan that read the (k-6)-th)`
  — verified: ACT slot 13 (3rd buffer) waits S6>=10 = DVE slot 18, which reads 0xb140. Using the
  *later* buffer's wait value for an earlier buffer is therefore always conservative-safe.
* `src_start_addr_addr_immediate` on a DGE DMA is a 64-bit field with the marker in the top byte
  (`0x1000000000004100`); setting it updates the `table_offset` / `var_offset` / `tbl_offs_*`
  aliases correctly and 0x16000 fits (17 bits is fine). Re-decode and assert anyway.

---

## Also shipped: the Cs producer writes straight into y (-3.2 us, the one duration win)

This was the iter-4 review's candidate #1 and it is real. iter-4 turned each group's first y
writer into a `COPY` (`y = Cs_0`). The producer of `Cs_0` is a DVE `TENSOR_TENSOR MULT`
(`Cs = C_psum * state`) 5-8 slots earlier whose `dst` can simply be **retargeted to y**, after
which the `COPY` is dead.

* Sites (iter-4 artifact DVE slot numbers): `COPY cp` / producer `prod` / placeholder MEMSET at
  `prod+1` / Cs / y:
  `105/99/100 0xc140->0x9140`, `192/187/188 0x6800->0xdb40`, `280/274/275 0xb000->0x5000`,
  `369/364/365 0x5000->0xc940`, `457/451/452 0xa000->0x9800`, `547/541/542 0x4000->0xe140`,
  `630/622/623 0x9000->0x6000`, `724/718/719 0xd940->0xe940`, `899/893/894 0xc940->0xe140`,
  `982/974/975 0x7000->0x5000`. All 10 pass; the 2 Pool-side first writers (Pool 8, 303) were
  left alone (their gain is Pool-only, i.e. invisible).
* **The placeholder 1-element MEMSET at `prod+1` writes `y[0] = 0`** (iter-3 shrank it from 512).
  Since the MULT now writes y *before* it, that MEMSET must be repointed to dead scratch
  (0x15400) or it clobbers y[0]. This is the only non-obvious part.
* The `COPY` slot becomes a 1-element MEMSET to 0x15400 keeping its identical events block
  (8 of 10; each has exactly one exact S6 waiter, a Pool y-ADD, and a MEMSET's
  `SEM_INC_COMPLETE` is ordered behind completion — iter-3's U1); the 2 with neither a wait nor
  an S6 update (slots 630, 982) were deleted outright, so no ordinal moved.
* Safety screen that selected the sites (script `xp/check.py`/`check2.py`): (a) no DVE reference
  to the Cs buffer in `(prod, cp)`; (b) no reader of it after `cp` before the next writer — the
  two apparent failures (slots 447, 619) read the address as a **dA** `src0` after ACT has
  rewritten it, so they are false positives and were included; (c) for every non-DVE reader of
  the Cs buffer, map its S6 wait value to a DVE slot and check it does not consume `prod`'s write
  — none did.
* Measured **508.712 -> 505.510** (-3.2 us); DVE busy 491.1 -> 482.0, DVE compute union (no ES)
  495.2 -> 492.2, i.e. ~10 x (416-65) ns of duration as predicted, and the rest of the
  "busy" drop was blocking time that was already hidden.

---

## Refutations added this iteration

### R1. **Reducing DVE instruction *count* by pair-merging the 24 remaining DVE `y+=Cs` ADDs is a
### wash-to-small-loss.** Built, bit-identical, 505.510 -> **505.843 (+0.33, ~1 sigma)**. Closed.
The 24 ADDs are 12 per-group pairs `(a, b)` with the same `y`:
`(89,92) (176,181) (264,267) (352,357) (440,444) (529,534) (629,633) (704,709) (804,809)
(880,885) (980,984) (1052,1054)`. All 12 merge legally at the later slot using iter-4's unlock
(`dst`/`src0` `step_elem[1]=0`, `src1 step_elem[1]=(Cs_b-Cs_a)/4`, `num_elem=[512,2,1]`;
deltas 512 except 1024, -9808, 3584), keeping slot `b`'s wait (slot `a`'s `S6>=v` is
auto-satisfied by position) and putting a 1-element MEMSET on slot `a`.
**DVE busy fell 482.0 -> 476.6 (-5.4 us) and exec went *up*.** So the merged 2-row ADD is
genuinely much cheaper than two 1-row ADDs (~5.4/12 = 450 ns per merge, more than the
157 ns fixed cost I predicted), but 9 of the 12 later slots wait on **ACT's S4** (they are the
`y += D*x` term), so merging drags the first accumulate behind an ACT dependency. Same family as
iter-4 R4. **If anyone retries this, merge only the 3 pairs whose later slot waits on S6
(`(629,633) (804,809) (980,984)`) — worth ~0.3 us, i.e. nothing.**

### R2. **The 4.2 us startup gap is DGE-trigger serialisation plus bandwidth; the ceiling is well
### under 1 us.** Closed by arithmetic with the real numbers (iter-2/iter-3 guessed at this).
Pool issues the 6 group-0 input DMAs back to back at **0.66 us per trigger** (slots 1-6,
t=10.83..14.76, all `W S0>=0`): sem 8 -> SBUF 0x37b00 (128 part x 1024 B = 128 KB, `dt`),
sem 9 -> 0x37700 (128 KB, `x`), sem 10 -> 0x37f00 (4 KB, **A**), sem 11 -> 0x23c08 (128 KB),
sem 12 -> 0x24008 (128 KB), sem 13 -> 0x23c00 (256 B, `D`). DVE pc=5 needs S8>=16 *and* (via the
ES at pc=4) S9>=16; both land ~14.6, i.e. 256 KB in 3.1 us from the first trigger's end = 82 GB/s
with the two queues running concurrently.
* Reordering the triggers cannot help: whichever of sems 8/9 is issued second moves the completion
  out by one 0.66 us trigger slot, and DVE needs both. I worked three permutations through the
  measured model; the best is +0.4 us and the small sem-10 (A) trigger going first costs DVE more
  than it saves ACT.
* Splitting each 128 KB load across two queues adds 0.66 us of trigger issue per extra queue and
  buys ~0.45 us of transfer. Net ~0.
* So the remaining lever would be a *descriptor* restructuring (fewer, larger descriptors so the
  issue cost drops), which is not an instruction edit. **Do not spend an iteration here.**

### R3. **A gap can close without exec moving.** The ACT reorder (row 4 of the table) removed the
0.641 us `pc 36->38` gap outright — it is absent from the post-edit gap list, and DVE compute span
fell 499.317 -> 498.997 — but exec moved only -0.21 us, inside the 0.3 us spread. Below ~0.5 us,
removing DVE idle no longer shows up 1:1 in exec. **Treat 0.5 us as the practical floor for a
single shippable gap edit on this kernel**, and stack several before measuring.

### R4. **Full-bf16 arithmetic is out of scope, not just out of budget.** The 12 `dt*x` MULTs are
the only non-fp32 DVE computes (`BFLOAT16 x BFLOAT16 -> FP32`, 677 ns, same as the 408 all-fp32
ones). Giving them `out_dtype = BFLOAT16` plus `in0 = BFLOAT16` on the 192 `dBx` MULTs that read
`dtx` would, per iter-1's "only 2x on DVE is full bf16", save up to 4 us — but it rounds `dtx`
to 8 mantissa bits, i.e. it **changes what the kernel computes**, which the goal forbids
independently of the 83%-used error budget. Not attempted. (Recorded because the dtype census
makes it look tempting: `TT MULT bf16,bf16->FP32` x12, `FP32,FP32->FP32` x384,
`ADD FP32` x24, `SCAN FP32` x192.)

### R5. **`TENSOR_REDUCE_ARITH_OP` is available on DVE (0x42, 9738 corpus slots, all DVE) but
### cannot pay here.** iter-4's R1 closed it on Pool (NRT_TIMEOUT). On DVE it would work, and an
n-wide reduce costs `n+1` SBUF stream-units against `3(n-1)` for n-1 ADDs, so folding the whole
y-accumulation into reduces cuts SBUF traffic by ~35%. But the y sum lives on **Pool**, where it
costs exec only the 13.2 us of Pool-traffic inflation iter-4 measured; any DVE-side scheme costs
DVE time ~1:1 (DVE is 96% busy), and the cheapest reduce tree I could construct per group
(5 x `[3,512]` -> 1 x `[5,512]` -> 2 ADDs) prices at ~54 us of DVE against the ~34 us it would
save. **Net ~+20 us. Closed by arithmetic.** Also: `ACTIVATION_READ_ACCUMULATOR` (0x24) /
`DVE_READ_ACCUMULATOR` (0x9b) and the `accumulator_cmd` field in `S2S2D2_STT_STRUCT` (the struct
the scan uses, byte 44) are a *free-axis* accumulator producing `[P,1]`, not a `[P,512]`
accumulate, so they cannot host `y` either.

### R6/R7. **The 11.4 us of DVE `TENSOR_TENSOR` duration *excess* over the median is concurrent
### SBUF traffic. Two timing probes close both alternative explanations.** CLOSED.
`TT n=420, median 677, mean 696, sum of (duration - 677)+ = 11.4 us`, and it is concentrated:
the **12 `dt*x` MULTs** carry ~6.5 us of it (durations 677..2073 ns; they are the only DVE
computes that read the `0x1c000-0x31a40` input-staging region and the only ones that wait on a
DGE input queue) and the **4 group-boundary `y+=Cs` ADDs** at pc 633/804/809/984 carry ~4 us
(1512-1702 ns each). `SCAN` excess is only 2.3 us over 192 instructions.
* **R6, address hypothesis: refuted.** All 12 `dt*x` MULTs repointed to `src0=0x16900`,
  `src1=0x16d00` (the quiet free gap; timing only, output wrong): exec **501.508**, DVE 481.7,
  pc 87 still **2060 ns**. Identical to the shipped 501.33/501.50. Consistent with iter-4 R3.
* **R7, hidden-wait hypothesis: refuted.** `events_wait_mode = NONE` on all 12 (so they carry no
  DGE-queue wait at all; the stripped waits were `S8>=16, S11>=16, S15>=16, S18>=16, S22>=16,
  S14>=32, S13>=32, S8>=48, S9>=48, S18>=48, S21>=48, S11>=64` — note the *per-group* input loads
  these name): exec **501.471**, pc 87 still **2073 ns**. So the excess is **not** absorbed wait
  time, and my gap attribution was not missing a stall.
* What is left is global SBUF bandwidth: the group boundary is exactly where the next group's
  256 KB input DMA is in flight *and* Pool runs its densest run of 2.1 us ADDs. The bytes have to
  move either way (iter-4 R1/R2/R3 closed the Pool half), so **this 11.4 us is not addressable.**
  It also means the useful model of this kernel is `exec = DVE duration census + 5.9 us gap`, with
  the census already containing the contention — do not count the 11.4 us as recoverable.

### Race check on the two deleted waits
This iteration removes two real synchronisation edges (DVE 177 `S11>=32`, DVE 96 `S9>=32`), which
is the one class of edit `verify.py` can pass by luck. `verify.py --runs 4`: **all 4 runs
bit-identical**, and the argument is static anyway — after relocation nothing writes the DMA's
source range again, so each wait is vacuous rather than merely unlikely to bite.

---

## Where the 501.3 us goes (shipped NEFF, `prof/Instruction.parquet`, `compiler_pc.notna()`)

| | |
|---|---|
| DVE compute union excl. `EVENT_SEMAPHORE` | **~493 us** |
| DVE compute span | 499.0 us |
| **DVE gap, 17 gaps** | **5.9 us** |
| &nbsp;&nbsp;startup input DMA (pc 3->5, S9/S8 >= 16) | **4.21** (R2: closed) |
| &nbsp;&nbsp;Pool S3>=281 at pc 1050->1052 (drain) | 0.34 |
| &nbsp;&nbsp;12 gaps of 130-162 ns (issue granularity) | ~1.4 |
| exec - DVE span (prologue/drain lapping in) | ~2.4 us |

DVE duration census: `TENSOR_TENSOR` 420 x 677 ns (mean 696) = 292.4 us,
`TENSOR_TENSOR_SCAN` 192 x 1270 = 245.0, `MEMSET` 32 x 59, 410 `EVENT_SEMAPHORE` x 53,
1 `SET_ORDERING_MODE`. Sum 540 vs union 493 => ~74 ns of overlap per compute instruction.
The 420 TT = **384 PSUM MULTs** (immovable: iter-1 #8, iter-2 R1) + 24 `y+=Cs` ADD + 12 `dt*x`.
Per-engine busy: DVE 482.6 (96.3%), Pool 417.0, PE 218.4, ACT 146.0, DMA 104.5, SP 48.4.

**The bound: 384 PSUM MULTs (260 us) + 192 scans (245 us) = 505 us of DVE duration that cannot
leave the engine, cannot merge and is dtype-locked; measured union is 493 us (the difference is
the ~74 ns/instruction overlap). Everything except the 4.2 us startup DMA and ~1.4 us of issue
granularity is now inside that.**

---

## What to try next — the 24 DVE `y+=Cs` ADDs, now priced at 11.6 us

**Measured ceiling (new, do not re-derive).** Timing-only probe, the DVE y-ADDs shrunk to
`num_elem=[1,1,1]` on all three operands (wrong output, nothing else touched — same instruction
count, order, addresses and semaphores, so this prices *exactly* what removing them from DVE
would buy, with their stalls included):

| probe | exec us | delta | DVE busy |
|---|---|---|---|
| shipped | 501.33 | — | 482.6 |
| the 12 `y += D*x` ADDs (the `S4` waiters: slots 92, 181, 267, 357, 444, 534, 633, 709, 809, 885, 984, 1054) | **495.05** | **-6.28** | 479.0 |
| all 24 (the 12 above plus 89, 176, 264, 352, 440, 529, 629, 704, 804, 880, 980, 1052) | **489.74** | **-11.59 (2.3%)** | 472.9 |

That is the whole remaining accessible budget in one block, and it is worth more than the
16.2 us of raw duration suggests per-us because these are the 4 boundary ADDs that also carry
~4 us of contention stall (R6/R7).

1. **Move the 12 `y += D*x` ADDs to Pool, hosting their ACT dependency on DVE.** Ceiling
   **6.28 us (1.25%)**. iter-3 R2 moved the *other* 25 and lost 10 us; it deadlocked when it
   included these 12 because it put the `S4` wait on Pool ("never put an S4 wait on Pool"). The
   untried variant: leave the `S4 >= v` wait on the **1-element MEMSET that replaces the DVE ADD**
   and give the Pool ADD `S6 >= ord(that MEMSET)`. No S4 ever touches Pool, so iter-3's cycle
   (Pool->ACT->Pool) cannot form.
   Two things must be handled, and I ran out of time at the second:
   * The deadline. `nextDVEref` slack for these 12 is `[2, 183, 2, 16, 2, 15, 18, 17, 2, 17, 2, 1]`
     DVE slots. The slack-2 ones are **not** a y dependency — they are the WAR on the `D*x`
     buffer, which the compiler has aliased into the `Cs`/`dBx` rotation and rewrites 2 slots
     later. So either move only the 5 high-slack sites (181, 357, 534, 709, 885 — about 2.6 us) or
     do candidate 2 first and move all 12.
   * ACT's *next* write to the `D*x` buffer currently waits on the DVE ADD's own S6 ordinal. The
     replacement MEMSET satisfies that ordinal before Pool has read the buffer, so the Pool ADD
     must `SEM_INC_COMPLETE S3` and ACT must be re-pointed at it — i.e. full S3 renumbering in all
     five streams (iter-1's `build3.py` does exactly this). Budget a whole iteration.
   **The per-site dependency data, already gathered (`xp/dxdep.py`) — start from this table.**
   For each of the 12: DVE slot / S6 ordinal / y / `D*x` buffer / the ADD's own wait / its single
   exact S6 waiter / what next writes the `D*x` buffer and what gates that:

   | DVE | ord | y | Dx | wait | exact S6 waiter | next Dx writer (gate) |
   |---|---|---|---|---|---|---|
   | 92 | 54 | 0x4940 | 0x8140 | S4>=17 | DVE 93, **Pool 45** | DVE 138 (S6>=81 -> slot 135) |
   | 181 | 109 | 0x9140 | 0xc940 | S4>=33 | **Pool 74** | DVE 440 (S6>=266 -> slot 436) — no ACT writer |
   | 267 | 162 | 0xdb40 | 0x7000 | S4>=49 | DVE 268, **Pool 109** | ACT 44 (**no S6/S3 wait**) |
   | 357 | 217 | 0x5000 | 0xb800 | S4>=65 | **Pool 141** | DVE 402 (-> slot 399); ACT 62 (**S3>=70**) |
   | 444 | 270 | 0xc940 | 0x6000 | S4>=81 | DVE 445, **Pool 176** | DVE 704 (-> slot 700) |
   | 534 | 325 | 0x9800 | 0xa800 | S4>=97 | **Pool 208** | ACT 100 (**S3>=118**) |
   | 633 | 385 | 0xe140 | 0x5000 | S6>=382 | DVE 635, **Pool 243** | DVE 635 (immediately!) |
   | 709 | 431 | 0x6000 | 0x9800 | S4>=130 | **Pool 274** | DVE 756 (-> slot 753); ACT 141 (**S3>=166**) |
   | 809 | 492 | 0xe940 | 0x4000 | S6>=489 | DVE 811, **Pool 307** | DVE 811 (immediately!) |
   | 885 | 538 | 0xa940 | 0x8800 | S4>=162 | **Pool 334** | ACT 61 (**S3>=68**), ACT 181 (S3>=212) |
   | 984 | 598 | 0xe140 | 0xd140 | S6>=595 | DVE 986, **Pool 361** | DVE 986 (immediately!) |
   | 1054 | 640 | 0x5000 | 0x7800 | S4>=194 | **Pool 385** | ACT 102 (**no wait**), ACT 220 (S3>=259) |

   Read it like this. The construction is: insert the ADD's identical 64 bytes on Pool
   **immediately before its exact S6 waiter** (always a Pool slot — the y output DMA or the next
   y-ADD, so Pool's in-order execution covers the `y` side for free) with
   `wait S6 >= ord` and `update = NONE` (no S3 update => **no renumbering at all**), and put a
   1-element MEMSET on the DVE slot keeping its whole events block.
   * The 5 sites whose next `D*x` writer is a **DVE** slot 40-260 slots later (181, 357, 444, 709,
     and 92/534 partially) are the candidates: the `D*x` WAR margin is tens of microseconds
     against a 2.1 us Pool ADD.
   * **But it is still only a scheduling argument, not a semaphore edge.** The hard version needs
     either (a) the ACT/DVE writer of `D*x` re-pointed onto the inserted ADD's S3 ordinal — which
     reintroduces renumbering, or (b) a DVE ES before the next DVE writer given a free extended
     `S3 >=` wait (iter-1 edits 3+4's trick). I checked (c) "insert before the Pool slot holding
     the S3 ordinal ACT already waits on": **it does not work** — e.g. site 357's ACT 62 waits
     S3>=70, but Pool ordinal 70 sits around Pool slot 100 while the insertion point is Pool 141,
     so the inserted ADD would land *after* the edge. Do not take that shortcut.
   * Sites 633, 809, 984 are unmovable: the next writer of their `D*x` buffer is the very next DVE
     slot.
2. **Relocate the 12 `D*x` buffers out of the `Cs`/`dBx` rotation into free SBUF.** Free, by the
   extent-scan recipe above: `0x15280-0x1c000` minus `0x15400` (dead), `0x16000-0x1603f` (carry),
   `0x16100-0x168ff` (relocated state) leaves ~26 KB/partition, and `0x31a40-0x37200` another
   22 KB — a `D*x` tile is 2 KB/partition, so all 12 fit. Worth **0 on its own** (iter-4 R3: no
   bandwidth component) but it converts the slack-2 deadlines above into slack-100+, which is the
   only thing that makes candidates 1 and the other 12 ADDs movable at all. **This is the
   prerequisite everybody since iter-1 has been blocked by without naming it.**
3. **The 4.21 us startup input DMA** is the only gap left above 0.35 us. R2 closes trigger
   reordering and queue-splitting by arithmetic; what is *not* measured is whether 82 GB/s is the
   device aggregate or per-queue. If it is per-queue, building two extra DGE DMA instructions
   (halving the two 128 KB loads by partition) plus splitting DVE pc=5 by `num_active_channels`
   would buy ~1.5 us. That needs new DMA instructions and a spare DGE semaphore (S8..S23 all carry
   64 increments; S0/S1/S2/S7 are untouched but iter-2 flagged them as unproven), so it is the
   riskiest item here for the smallest return.
4. **Nothing else.** Engine rebalancing (iter-1 #8, iter-2 R1/R3, iter-3 R2), instruction merging
   (iter-3 R3, iter-4 R4, R1 here), re-addressing for *bandwidth* (iter-4 R3, R6 here), Pool
   traffic reduction (iter-4 R1/R2, R5 here), dtype (iter-1 #6, R4 here), DVE S6 self-waits
   (iter-4 R5), the `TT` duration excess (R6/R7 here) and the 12 small S6 issue-granularity gaps
   (1.4 us, DVE waiting on itself) are all closed by device measurement on this NEFF.

## Tooling notes

* `tools/profile.py --repeats 5` is ~80 s; spread on this NEFF is 0.04-0.10% (0.2-0.5 us).
  `verify.py` is ~25 s and is a complete oracle for these edits — every mistake I made this
  iteration showed up as a wrong `output_1`, never as a hang.
* Scripts used (in `xp/`, gone after the iteration; each 30-90 lines, all built on iter-1's
  `build3.py` skeleton): `check.py`/`check2.py` (the Cs-retarget safety screen), `build.py`
  (the retarget), `merge.py` (`APPLY=1`, the refuted ADD pair-merge), `gaps.py` (iter-3's gap
  attribution, unchanged), `carry.py`/`rng.py` (extent scans), `reloc.py` (carry array ->
  0x16000 + drop the ES wait), `y0fix2.py` (y/state de-aliasing), `hoist.py`/`actreord.py`
  (the two ACT edits), `cen.py` (census + union of intervals), `dma.py`/`head.py` (DGE DMA
  decode and the startup timeline), `s4w.py` (semaphore-waiter census), `top.py`
  (duration-excess attribution), `probe_stage.py`/`probe_wait.py` (R6/R7), `probe_dx.py`
  (`WHICH=s4|all`, the y-ADD pricing probe — 12 lines, the single most useful thing to re-run
  first next time).
* **A `num_elem -> [1,1,1]` shrink on all three operands is the right way to price "what would
  removing this instruction buy".** It keeps instruction count, order, addresses and every
  semaphore identical, so the delta is the instruction's duration *plus* whatever stalls it
  absorbs, and nothing else. Three of this iteration's conclusions came from that one trick.
* The profiler's reported **"engine busy" is not the union of intervals** — for DVE it read 482.0
  where the union of non-ES rows was 492.2. Use the union (`cen.py`) when comparing against a
  duration census, and read a busy-time change on an engine you did not edit as blocking time.
* `neff_diff.py` against a NEFF with a different slot count mis-reports hundreds of changed slots
  (everything after the deletion shifts). Diff member md5s and sizes instead:
  this iteration `DVE0.bin 67648 -> 67520`, `Pool0.bin` and `Activation0.bin` same size.

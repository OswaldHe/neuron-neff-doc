# iter-2 — selective_scan b1 c768 s1024 n16 bf16

**Result: NO CHANGE SHIPPED. `kernel.neff` is byte-identical to the accepted iter-1 artifact**
(sha256 `67c657b0...`, `neff_diff.py`: "34 identical, 0 changed ... payloads are identical").
Two confirmation measurements of the shipped file: **median 516.908** (spread 0.15%) and
**median 516.430** (min 516.132, max 516.799, spread 0.13%). `verify.py` PASS, output_0
6.245e-02 (83% of limit) / output_1 4.268e-05 — the baseline figures. `launch_rules.py` PASS.
So iter-1's 516.5-516.9 us reproduces and this iteration neither gains nor regresses.

This iteration was spent on the lever iter-1's notes ranked **#1** (stage the PSUM operand into
SBUF on the idle ACT engine, then move a PSUM MULT family to Pool) and on the one open question
about the scan. **Both are now closed by device evidence.** Everything below is a refutation with
the measurement that produced it; read it before spending another iteration on engine rebalancing,
because the remaining headroom is much smaller than iter-1's notes imply.

Read `iter-1.md` first — its calibration table, build scripts and refutation list all still hold.

## The headline refutations

### R1. ACT **can** read PSUM and copy it exactly — but only *once*. Concurrent ACT PSUM reads
### collide. **The whole ACT-staging family of ideas is closed.**

The mechanism was built end to end and works:
* **1 moved `dBx = dtx*B_psum` MULT: exit 0, output bit-identical** (output_0 max_abs_err
  6.245e-02, output_1 4.268e-05 — exactly the baseline figures). So `COPY` (0x46) placed in
  `Activation0.bin` with `src = 0x2000000+off`, `dst` = SBUF scratch, `in/out_dtype = FP32`,
  `num_elem [512,1,1,1]` **executes on the Activation engine and copies PSUM→SBUF bit-exactly**,
  and Pool then multiplies from SBUF bit-exactly. (New fact: ACT `COPY` is enabled in this NEFF's
  opcode table, and no activation table / scale / bias juggling is needed.)
* **4 moved MULTs, same configuration: `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`, nrt 1004, 9.0 s.**
* 12 and 21 moved: same collision (or a 70 s NRT_TIMEOUT once a stronger wait was used, see R2).
* **Control that isolates it:** identical build with the ACT copy's `src` pointed at **SBUF**
  instead of PSUM (`NOPSUM=1`, output deliberately garbage) → **exit 0** with 21 copies, 21 Pool
  MULTs, 21 DVE `EVENT_SEMAPHORE` replacements and the full S3/S4 renumbering in place.
  So the semaphore scheme, the insertions, the renumbering and the SBUF scratch are all correct;
  **the only thing that fails is ACT touching PSUM more than once.**

Conclusion: a *single* non-PE PSUM read from a second engine passes by luck. iter-1's notes claimed
"the enabler is already proven on the device" from exactly such a single-instruction control — that
was a **false positive**. Scale it to 4 and it collides. Treat "move a PSUM-reading instruction to
another engine" (ACT, Pool, DMA) as closed for this kernel.

Corollary: **both DVE MULT families are permanently DVE-only** (iter-1's refuted-item 8 said this
for Pool; it is now also true for ACT). DVE must execute 192 scans + 384 PSUM MULTs ≈ 505 us of
union-busy, and exec is 516.9 us. **There is ~12 us of total slack, see "Where the 516.9 us goes".**

### R2. The compiler's PSUM WAR/RAW guards are *not* conservative enough to re-time a PSUM read.
Chased this hard; three variants, all broken:

| variant | result |
|---|---|
| ACT copy inherits the DVE MULT's own wait (`S5 >= w5`), DVE slot → ES waiting `S4 >= ord(copy)` and still incrementing S6 | PSUM collision at n>=4 |
| ACT copy waits the **exact** S5 ordinal of the matmul that writes its bank (computed by matching PE `MATMUL dst` against the MULT's PSUM operand) | **NRT_TIMEOUT 70 s** (deadlock): the exact ordinal is 1-5 matmuls beyond the compiler's wait, PE needs DVE 1-2 states further, DVE needs ACT's `dA` which ACT cannot emit because it is blocked on the copy. Cycle. |
| PE's WAR re-pointed: extended wait `S4 >= ord(copy)` hosted on the latest PE `EVENT_SEMAPHORE` before the next matmul to that bank, DVE ES left with no wait | PSUM collision again (so either PE `EVENT_SEMAPHORE` *extended* waits are not honoured, or hosting them that far back is not enough — **unresolved, but suspect PE extended waits**) |

Hard data behind this: for 150 of the 192 `dBx` MULTs, the `events_semaphore_value` of the MULT's
`S5` wait is **one less** than the S5 ordinal of the PE matmul that writes the very PSUM bank it
reads (e.g. slot 67 reads `0x2003000` waiting `S5>=21`, but the matmul writing `0x2003000` is #22).
The original NEFF is correct only because DVE reads the bank immediately when its wait clears.
Any re-timing of that read — earlier *or* later — loses.
Script: `/tmp/xp/bmm.py` (recreate: match `MATMUL dst` to the MULT's PSUM src, compare ordinals).

### R3. Pool/GpSimd **cannot execute `TENSOR_TENSOR_SCAN_ARITH` (0xe5)** — closed.
Probe: one DVE scan (slot 639) moved to Pool, fully serialised and provably deadlock-free
(Pool `EVENT_SEMAPHORE` holding `S6 >= s6incl[i-1]` primary + the scan's own original wait as the
extended wait, then the scan's identical 64 bytes with `update S3`; the DVE slot became an ES with
`wait S3 >= ordB, update S6`, so every S6 ordinal is preserved and DVE is already past everything
the Pool side needs). `semaphores.py` clean. Device: **NRT_TIMEOUT, 70 s** — the signature iter-1
established for "the instruction never retires on that engine", not the 9 s PSUM collision and not
a wrong answer. Consistent with `docs/generated/opcodes.md`: opcode 0xe5 has **0 slots on any
engine** in the 14-NEFF corpus, i.e. only DVE's microcode table carries it.
So the 244 us scan half of DVE cannot be offloaded, in addition to being unmergeable (iter-1).
Script: `/tmp/xp/scanprobe.py`.

### R4. Moving the 32 remaining S6-updating `y += Cs` ADDs to Pool is a *structural* loss — do not
### build it.
Not run on the device; refuted by the dependency structure, which is cheap to re-derive:
the ADD's S6 ordinal has exact waiters (iter-1: DVE 26, Pool 12), so the DVE slot must become an
`ES(wait S3 >= ordB, update S6)` **on that slot** (an ES increments S6 at its own stream position,
so the wait cannot be hosted later as in iter-1's edits 3+4). The Pool ADD can only start after
DVE's `Cs` MULT, which is the instruction immediately before. So DVE arrives at the ES and stalls
for the full Pool ADD: **+2.0 us of stall for 0.69 us of saved duration, a net ~1.3 us loss per
ADD.** This is the same mechanism as iter-1's refuted-item 1, now with the reason spelled out.
(DVE's 34 remaining `y+=Cs` ADDs: 32 update S6, 2 do not — only those 2 are free to move the way
iter-1 moved its 170.) The only way to make the other 32 pay is to re-point all ~38 exact waiters
(DVE 26, Pool 12) onto Pool's S3 so the ES needs no wait; that is a much bigger edit than the
~22 us of DVE duration it could recover, and it would push Pool to ~479 us — level with DVE —
so the realistic payoff is a few us at best.

## Where the 516.4 us actually goes (fresh measurement of the shipped NEFF)

**First, a trap to avoid:** `prof/Instruction.parquet` contains a ~10 us runtime **prologue**
(`TENSOR_LOAD` microcode loads, `DRAIN`, `NOTIFY`, `COMPARE_BRANCH`, barrier `EVENT_SEMAPHORE`s)
on every engine before the kernel body. Those rows have **`compiler_pc == NaN`**; filter on
`compiler_pc.notna()` or your gap analysis invents 9 us of phantom idle (I did this once — the
4.7 / 2.2 / 1.0 / 0.7 us "gaps" are prologue, not schedulable). Whole profile span 529.7 us, body
span 517.9 us (t = 10.1 → 528.0), reported `total_exec_time` 516.4 us.

Body only, union-of-intervals over DVE rows excluding `EVENT_SEMAPHORE`:

| | us |
|---|---|
| DVE union busy | **502.0** |
| DVE body span (10.15 → 523.60) | 513.4 |
| DVE gap, **36** gaps | **11.5** |
| &nbsp;&nbsp;startup input DMA (gap before pc=5, `dtx = dt*x`, `W S8>=16`) | **3.9** |
| &nbsp;&nbsp;one PE stall (before pc=205, `W S5>=63`) | 2.0 |
| &nbsp;&nbsp;ACT `dA` late (before pc=111 `W S4>=20`, pc=10 `W S4>=1`) | 1.7 |
| &nbsp;&nbsp;Pool carry `COPY` late (before pc=43, `W S3>=8`) | 1.0 |
| &nbsp;&nbsp;everything else: 1 x 335 ns, then 31 gaps of 132-162 ns | 2.9 |

(This reproduces iter-1's 11.0 us figure, so that accounting was right.)

**The other engines are paced by DVE, not starved by DMA.** Their largest body gaps are all at
group boundaries and all large: PE idles **7.85 us before the `LDWEIGHTS` at each of the 6 channel
tiles** (pc 124/247/370/493/616), ACT idles 10.0-11.1 us before an `ACTIVATE` once per group,
Pool idles 3.2-5.8 us several times. So the four early DVE stalls (pc=10 `S4`, 43 `S3`, 111 `S4`,
205 `S5`) are **pipeline-fill at a group boundary**, not input-DMA starvation: at a boundary all
8 PSUM banks still hold the outgoing group's `B`/`C`, so PE cannot prefetch the next group's
matmuls until DVE drains them, and DVE then waits for the first ones. The fix would be more PSUM
depth, which does not exist (8 banks, 4 for `B` + 4 for `C`). I checked for an input-prefetch
lever and there isn't one.

**The tail is DMA drain, not compute.** DVE's last instruction ends at t=523.60; Pool runs to
528.02, but its tail is `DMA_DIRECT2D` (ends 524.62) and then two `EVENT_SEMAPHORE`s blocked on
the output DMA queues `S18`/`S19` until 527.86. So the 4.4 us after DVE finishes is the output
write draining — **do not try to shorten it by rebalancing the last Pool ADDs** (I checked this
hypothesis; Pool's last `TENSOR_TENSOR` already ends at 521.96, *before* DVE finishes).

Per-engine busy (median repeat): DVE 501.4 (97%), Pool 415.1 (80%), PE 218.7 (42%),
ACT 146.0 (28%), DMA 107.4 (21%), SP 48.2 (9%).
Durations: DVE `TENSOR_TENSOR` 430 x 693.8 ns (p50 677, p99 1685, max 2072; **10.1 us of excess
over the median** sits in ~5 slots), `TENSOR_TENSOR_SCAN` 192 x 1271.2 (p99 1333),
`EVENT_SEMAPHORE` 577 x 57. Pool `TENSOR_TENSOR` 170 x 2008.7, `DMA_DIRECT2D` 60 x 658.3,
`COPY` 115 x 295.9. ACT `ACTIVATE` 204 x 723.2.

**Read this as: exec 516.4 ≈ DVE union 502.0 + 3.9 us startup input DMA + ~4.7 us of in-phase
one-off stalls (PE/ACT/Pool) + ~2.9 us spread over 32 sub-200 ns gaps + a few us of
prologue/drain that laps into the window.** Of the 11.5 us of DVE gap, the 3.9 us startup is a
load latency and the 2.9 us of 132-162 ns gaps are instruction-issue granularity. So **at
constant DVE work the schedulable prize is ~4-5 us (<1%)**. Anything bigger has to remove DVE
instruction *duration*, and R1+R3 say that duration cannot leave the engine.

## Useful facts established along the way (reusable)

* **Compute instructions have exactly ONE wait.** Only `EVENT_SEMAPHORE` (`CTRL_ES`) carries the
  second, *extended* wait block. Any edit needing two waits on a compute slot must insert an ES.
  (iter-1's notes imply otherwise in places.)
* **Free SBUF in this NEFF** (from the union of every compute operand's extent + `imm*_ptr`):
  used `0x004000-0x015280`, `0x01c000-0x031640`, `0x037200-0x038300`; **free gaps
  `0x015280-0x01c000` (28032 B) and `0x031640-0x037200` (23488 B)**. `0x015400` was written and
  read by an inserted ACT copy / Pool MULT with bit-exact output, so gap 1 is genuinely free.
  SBUF partition window is `0x40000`; this kernel uses offsets above `0x38000`, so the
  "corpus never exceeds 0x37FFF" note in chapter 5 is not a limit here.
* **Semaphores in use:** S3 Pool (282 inc), S4 ACT (194 inc), S5 PE (374 inc), S6 DVE (640 inc),
  S8..S23 `dge:Pool` DMA, S24..S39 `dma:qSPIO0`. **S0,S1,S2,S7 are untouched** — I did *not* use
  them (unknown whether the runtime zeroes an undeclared semaphore between the 5 executions of a
  profile capture; reusing S3/S4 and renumbering is safe and not much more work).
* **`semaphores.py` only reads the primary events block.** It reported "balanced" for every
  deadlocking and colliding build in this iteration. It is a necessary, very weak check.
* **Device run times are a reliable error oracle:** ~9-11 s = ran (exit 0, or exit 1 with
  `PSUM_COLLISION` / nrt 1004); ~70 s = NRT_TIMEOUT, i.e. deadlock **or** an instruction the engine
  cannot retire. A `verify.py` round trip is ~25 s wall, `profile.py --repeats 5` ~75 s.
* **Instruction families in this NEFF** (`isa.py` over `work/payload/sg00/`):
  DVE 1216 = 430 `TENSOR_TENSOR_ARITH_OP` + 192 `TENSOR_TENSOR_SCAN_ARITH` + 569 ES + 24 MEMSET;
  Pool 389 = 170 TT + 115 `COPY` + 60 `PSEUDO_DMA_DIRECT2D` + 42 ES; PE 738 = 384 `MATMUL` +
  6 `LDWEIGHTS` + 347 ES; ACT 234 = 204 `ACTIVATE` (192 with `activation_func=7` exp = `dA`,
  12 with `activation_func=128` copy) + 28 ES + 1 table load; SP 647 = 375 `PSEUDO_DMATRIGGER`.
* **Per-state dataflow, by address** (6 channel tiles x 2 chunks = 12 groups x 16 states):
  `dtx` (one SBUF buffer per group, e.g. `0x4140`) = bf16 `dt * x`;
  PE broadcasts `B[n,:]` into one of **4** B banks `{0x2000000,0x2001000,0x2002000,0x2003000}` and
  `C[n,:]` into **4** C banks `{0x2000800,0x2001800,0x2002800,0x2003800}` (so each bank is reused
  every 4 states — that is the WAR window);
  DVE `dBx = dtx * B_psum` -> 3-deep rotation (`0x6140/0x9140/0xc140`, +0x3000);
  DVE `SCAN(src0 = dA, src1 = dBx, imm0 = carry ptr 0x4100+4n)` -> `state`, 3-deep;
  DVE `Cs = C_psum * state` -> 3-deep; Pool `y += Cs` with `src0 = dst = y` (`0x4940`).
  DVE program order per state is `SCAN_n, MULT(dBx_{n+1}), MULT(Cs_n)` — i.e. the compiler already
  software-pipelines `dBx` one state ahead, and its only slack is the 0.69 us `Cs` MULT.
* ACT's 192 `dA` `ACTIVATE`s map 1:1 and in order onto the 192 scans (the k-th `func=7` ACTIVATE
  writes the k-th scan's `src0`); the 192 `dBx` MULTs map 1:1 and in order onto the scans too.

## What I would try next, ranked (honestly, all small)

0. **Also settled:** the iter-1 review's open question "do the new sem-3 waits guard every reader
   of the `0x4940` accumulator?" — `verify.py --runs 3` on the shipped file: **all 3 runs
   bit-identical**, so no race is observable. (Not a proof, but the cheap check it asked for.)
1. **Nothing engine-level is left.** R1+R3 close offload; R4 closes the ADD direction; iter-1
   closed scan merging, bf16, FP32R, Pool-PSUM and the DVE/Pool balance. If a future iteration
   wants >2% it must be the **re-addressing** edit iter-1 ranked #2: deepen the `state`/`Cs`
   rotations from 3 to 4 buffers (free SBUF exists, see above) **and then** pair-merge the
   `Cs` MULTs at the earlier slot. Only the combination pays; the merge alone is blocked
   (iter-1 refuted both merge directions at 3-deep). Budget ~96 pairs x (136 ns fixed overhead
   - 83 ns lost instruction overlap) ~ 5 us, so even this is ~1%. **Probably the kernel is done.**
2. **Pair-merge the `dBx` MULTs.** iter-1 reports both merge directions "built and refuted" for
   the two MULT families in the prior c2560 case study (the earlier-slot direction needs the PSUM
   operand one matmul sooner and costs more than the 136 ns it saves), so this is **probably
   already closed**; I did not re-test it. If someone retries it, the shape is: merge
   `dBx_{n+1}`,`dBx_{n+2}` at the *earlier* slot with `num_elem [512,2]`,
   `src0 step_elem [1,0]` (same `dtx` twice — **step 0 is unverified**),
   `src1 step [1,1024]` (adjacent B banks), `dst step [1,3072]`; raise the merged slot's `S5` wait
   to the later matmul; renumber S6 (640 -> 544 increments, ~1392 waits in 5 streams — waiters on
   either deleted ordinal both map to the merged slot, which is safe because it computes both).
   Ceiling 96 x (136 - 83) ns ~ 5 us.
3. **The 3.9 us startup gap is irreducible by scheduling — closed by arithmetic, don't build it.**
   The idea is to split DVE pc=5 (`dtx = dt*x`, `W S8>=16`) into two halves waiting `S8>=8` and
   `S8>=16` (split by partition with `num_active_channels=64` + start partition 64, which is
   chunk-order-agnostic, so it avoids the "do the 16 DMA chunks arrive in free-dim order?"
   question; make only the second half keep the `SEM_INC_COMPLETE S6` so no S6 renumbering is
   needed). **But the gain is bounded by the work you can hoist, not by the time waited:** half of
   one 512-element bf16 MULT is ~350 ns, minus ~136 ns of extra instruction overhead => ~0.2 us.
   DVE genuinely has nothing else to do until the first input tile lands: every instruction in the
   first group depends on `x`/`dt`. The only real fix is a faster first load — 256 KB in 3.9 us is
   ~66 GB/s, i.e. the first group is spread over too few DMA queues — and that is a DMA-descriptor
   restructuring, not an instruction edit.
4. **Do not** chase: the 4.4 us Pool tail (output DMA drain), the ~10 us runtime prologue, or the
   10.1 us of DVE `TENSOR_TENSOR` duration excess over the p50 (it is ~5 slots at 1.7-2.1 us,
   caused by Pool's concurrent SBUF traffic; Pool is not the critical engine, so there is nothing
   to trade for it).

## Reproducing the refuted builds

Working scripts live in `/tmp/xp/` (gone after the iteration): `build_stage.py` (ACT-stage + move
the `Cs` family), `build_dbx.py` (same for the `dBx` family), `scanprobe.py` (Pool scan),
`bmm.py` (matmul<->MULT bank matching), `gaps2.py` (gap attribution), `recon.py` (semaphore census
+ SBUF free-space map). They are all small variations on iter-1's `build3.py` skeleton:
unpack -> decode all five streams with `isa.py` -> compute insertion positions and the
old->new ordinal maps -> **remap every pre-existing S3/S4 wait first** (primary *and* extended, all
five streams) -> then emit the new streams with the new waits -> `neff.py pack`.
Recreate from iter-1's `build3.py` plus the structure notes above; the only non-obvious parts are
(a) one wait per compute slot, (b) a replaced DVE slot must keep its `SEM_INC_COMPLETE S6` so every
ordinal survives, and (c) the handoff wait can usually be merged into the consuming scan's existing
`S3` wait by taking the max (19-24 of 21-24 cases here).

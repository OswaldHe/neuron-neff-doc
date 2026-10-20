# iter-7 — two dead CASTs, and iter-6's "move an op to the idle engine" lever generalised:
# **ops that increment a semaphore can move too**, and the re-ordering fence belongs on a *third* engine

## Result: **55.1 → 53.25 us (-1.9 us, -3.5%)**, outputs bit-identical

The shipped bytes were measured three separate times today: **53.061 / 53.252 / 53.264 us**.
iter-6's bytes were measured twice: **55.115 / 55.170 us**. Quote 53.25, not 53.06.

Six configurations measured on the device this session, all 7-capture medians of `tools/profile.py`:

| step | edit | metric us | Δ | verdict |
|---|---|---|---|---|
| A | iter-6's shipped NEFF, re-measured at session start | **55.115** | — | anchor |
| B | **two provably dead DVE CASTs removed** (§2) | **54.684** | -0.43 | kept |
| C | **ACT[21]/[23]/[24] executed on DVE** (§3) | **53.854** | -0.83 | kept |
| D | C + ACT[33]→DVE + DVE[63]→ACT + DVE[75]→ACT (§4) | 53.690 | -0.16 | the two DVE→ACT moves dropped |
| F | C + **ACT[33]→DVE** only | **53.535** | -0.32 | kept |
| **G** | F + **ACT[9]→DVE** (§5) | **53.061 / 53.252 / 53.264** | **-0.47** | **SHIPPED** |
| E | F + ACT[9] + **ACT[14]** →DVE | — | — | **verify FAILED, race — §5** |
| H | G + sem-59 staging DGE re-hosted on Activation (§6) | 53.741 | **+0.68** | **REVERTED** |
| I | G + `ACT[8]` rotated into the 759 ns Scalar hole at 17.5 us (§6a) | 53.199 | ~0 | **REVERTED** |

`verify.py --runs 2` PASS and bit-identical after B, C, D, F, G, H; `max_abs_err` identical to the
baseline to every digit (output_0 6.092e-05 = 11 %, output_1 4.494e-04 = 83 %). `launch_rules.py`
PASS. `semaphores.py` no over-wait; **S[4] 61/61 and S[6] 93/93 preserved by every kept edit**.
Slot counts, iter-6 → shipped: `ACT` 121 → 121, `DVE` 136 → 144, `Pool` 37 → 38, `SP` 69 → 70,
`PE` unchanged. Members changed vs iter-6: `sg00/{Activation0.bin, DVE0.bin, Pool0.bin, SP0.bin}`.

**Interleaved A/B at the end of the session** (iter-6's bytes vs the shipped bytes, 7 captures each,
back to back on the same device): **55.170 us (54.670-55.430) vs 53.252 us (52.999-53.708)** →
**-1.92 us, ranges do not overlap**, against a run-to-run spread of ~1.35 %. iter-6's bytes
measured 55.115 at session start and 55.170 at the end, so the machine drifted <0.1 us today —
but keep doing the interleaved pair, it is the only honest number (iter-6 §10).

Shipped configuration = **B + C + ACT[33]→DVE + ACT[9]→DVE**. Total against the iter-0 baseline:
67.525 → 53.1 us, **-21.4 %**. Engine busy at the end: Scalar 31.8 us (59.9 %), Vector 29.9 (56.3),
DMA 19.9 (37.5), Tensor 17.7 (33.4), Pool 14.1 (26.6), Sync 11.9 (22.4) — **Vector has now caught
up with Scalar, which is why §5's "where is the Vector hole" is the binding question for iter-8.**

## 1. Rebuild these two scripts first; they found everything this iteration

`scripts/` is not preserved. Both are short.

* **`d2.py`** — compact stream dump. The field that matters is
  `<role>_mem_pattern_start_addr_addr_immediate`. **Do not mask it with `0xffffff`.**
  The address is `partition<<18 | offset` (partition stride 0x40000 = 262144 — the same number that
  shows up as a `262144` step in `SP0.json`'s DMA descriptors), and PSUM is `>= 0x2000000`.
  So `0x1006480` is **partition 64**, offset 0x6480 — a *different* buffer from `0x6480`
  (partition 0). I masked the top bits on the first pass and produced two false "dead store" hits
  (`DVE[62]` vs `DVE[63]`) that are really the two partition halves of one 128-partition buffer.
  Also `<role>_..._addr_reg_regnum` is **not** a partition: with `addr_reg_marker == 0` it is just
  the low byte of the immediate, so printing it as `pNN` is noise (iter-5/6 notes write `p64+0x...`
  meaning partition 64 — that is the `<<18` field, not `regnum`).
* **`dead2.py`** — ~40 lines and the find of the iteration: read `prof/Instruction.parquet`, regex
  `(src0|src1|src|dst|out|in)=<dtype>@0x<addr>[steps][nelem]` and `channels=N` out of the
  **`operands` column**, and report two instructions on the same engine with an identical
  (opcode, srcs, dsts, channels, op/imm) signature.
  **Build dependency analysis on the profiler's own operand text, not on your own field reads**: it
  prints the full address, expands the pointer operand, and is what the vendor disassembler
  produces. Cross-check each candidate against (a) every other `dst=` in the parquet, (b) every
  `DMA_DIRECT2D`'s `src_addr_imm`/`dst_addr_imm` — **these appear as `0x80200<addr>`, so a naive
  `addr in operands` test misses them** — and (c) every `SP0.json["dma"][*]["desc"][*]["to_off"]`
  with its `to_sizes`/`to_steps` span.

The third thing worth 10 minutes is **union occupancy per 1 us bucket**: merge each engine's
intervals inside the bucket instead of summing durations. Tensor overlaps LDWEIGHTS with MATMUL and
Vector overlaps an `EVENT_SEMAPHORE` with the op before it, so the naive sum reports >1000 ns/us
and hides which engine is actually the wall. The honest table for the iter-6 NEFF:

| window | saturated | note |
|---|---|---|
| 12.3-18 us | nothing | one DGE row + a **2.36 us DRAM→SBUF flight**; closed (iter-6 §2) |
| 14-17 us | **Pool 100 %** | 7 input-load DGE rows back to back — **off the critical path, §7** |
| **18-27 us** | **Scalar 94-98 %** | **8.8 us of back-to-back Activation. This is the lever.** |
| 28-31 us | nothing | the blk5 queued transfer's 3.0 us flight |
| 33-48 us | **Vector 85-96 %** | Tensor 40-70 %, Scalar 10-70 % — but see §4, it will not move |
| 49-66 us | nothing over 65 % | latency-bound hand-offs |

## 2. Edit B — **two of the 136 DVE slots were dead code**: -0.43 us for a 2-slot patch

`dead2.py` reported exactly four same-engine duplicates; two had **no write to the source anywhere
in between, on any engine, by any DMA**:

| slot | op | duplicate of | update |
|---|---|---|---|
| `DVE[56]` | `CAST fp32@0x6480 → bf16@0x5880`, 213 ns | `DVE[48]` | none |
| `DVE[72]` | `CAST fp32@0x1006680 → bf16@0x5b80`, 223 ns | `DVE[65]` | `S[6]++` |

Each recomputes, byte for byte, what an earlier DVE slot already wrote. Checked before packing:
`0x6480` is written only by `DVE[46]` and `0x1006680` only by `DVE[63]`, both *before* the first
CAST; no queued descriptor's `to_off` span reaches into `0x6480..0x66ff`; and every
`PSEUDO_DMA_DIRECT2D` destination in this NEFF is DRAM/var, never SBUF.

* `DVE[56]` → the pure no-op slot (`a0 10` + 62 zero bytes = `EVENT_SEMAPHORE` with everything
  zero, ~24 ns on Vector).
* `DVE[72]` → `EVENT_SEMAPHORE` with `events_update_mode = SEM_INC_COMPLETE`,
  `events_update_idx = 6`, so the `S[6]` count sequence is byte-identical and no waiter moves.
  **The compiler itself emits one of these** (`SP[39]`): `a0 10 00 00 00 00 13 40` = "ES, no wait,
  SEM_INC_COMPLETE on S[64]" — byte 6 is the update mode, byte 7 the update index.

Measured -0.431 against a predicted -0.436. Vector busy 28.75 → 28.40 us.

The other two duplicates (`Scalar[73]` == `Scalar[65]`, `Scalar[75]` == `Scalar[64]`) are **not**
dead: their sources `0x1005600` / `0x1005200` are rewritten in between by seven and ten Vector ops.
**`0x1005200` and `0x1005600` are this kernel's two hot scratch buffers (10 writers / 10 readers
each) — treat any edit touching them as high risk.** Re-ran `dead2.py` on the shipped bytes: no new
duplicates, so this particular well is dry until some future reorder refills it.

## 3. Edit C — three Activation TENSOR_SCALARs on DVE: -0.83 us, and the fence goes on **Sync**

iter-6 §8 continued. `ACT[21]/[23]/[24]` are `TENSOR_SCALAR_ARITH_OP`s (385/384/379 ns) inside the
18-27 us Activation wall, with **no semaphore update**, and all three read the same two regions:

* `src 0x1037b00 / 0x1037c00 / 0x1037d00` — all inside the **one** DGE whose semaphore is **14**
  (`Pool[12]`, `dst_pattern=[256,262144][4,64]` → 4x256 B per partition = `0x37b00..0x37eff`,
  partitions 64-127). **Read the span out of `dst_pattern`, not just the start address** — that one
  fact collapsed three guards into one.
* `imm0_imm_ptr 0x101c000 / 0x101c100 / 0x101c200` — sems **10 / 11 / 12**.

The edit:

1. DVE slots 27..31 inserted (after `DVE[26]`, in the 23.4-25.4 us Vector hole):
   `ES(S[10]>=16, x S[11]>=16)`, `ES(S[12]>=16, x S[14]>=16)`, then the three copies with their own
   wait cleared and `events_update_idx = 70, SEM_INC_COMPLETE`.
2. `ACT[21]`, `ACT[23]`, `ACT[24]` → the pure no-op slot. **No fence on Activation.**
3. One `EVENT_SEMAPHORE wait S[70]>=3` inserted in the SP stream in front of `SP[7]`.

**Step 3 is the whole point and the thing iter-6 did not do.** Each destination (`0x1005880`,
`0x1005a80`, `0x1005c80`) has exactly **one** writer and exactly **one** reader, and that reader is
the DGE `SP[7]/[8]/[9]`. So the fence that re-imposes the ordering goes on **Sync, which is idle**,
instead of on Activation, the engine being relieved. iter-6's recipe step 4 ("replace the vacated
Activation slot with a bare EVENT_SEMAPHORE") would have put the stall straight back: Activation
reaches slot 24 at ~23.2 us but Vector cannot finish three copies before ~24.5, so an Activation
fence is **+90 ns, not -900**. Worked out on paper before spending a device run.

> **Rule for iter-8: find the moved op's consumers, and fence the *consumer's* engine, not the
> engine you are relieving.** Done three times this session (§3, §4 `ACT[33]`→Pool,
> §5 `ACT[9]`), all three kept.

Result: Activation reaches `ACT[25]` 0.92 us earlier, the gain survives the Scalar↔Vector
ping-pong (`ACT[27]`'s `S[6]>=14` and `ACT[32]`'s `S[6]>=17` were already satisfied), the blk5
trigger fires 1.3 us earlier and `Tensor[23]` starts 1.10 us earlier. Metric -0.83.

**A DVE TENSOR_SCALAR of this shape runs 285-295 ns; the same slot on Activation takes 379-387 ns**
(128 elem, 64 channels, bf16→fp32, `POINTER_IMMEDIATE`). iter-6's two moved ops show the same
285/284 vs ~385. The move is not zero-sum — **Vector is ~25 % faster at this op than Activation.**

About 0.6 us of the 1.3 us head start was **eaten by DMA contention**: the five staging DGE
transfers (sems 59-63) now fly at 26.7-29.9 us instead of 28.1-30.0, right on top of blk5, and
blk6's flight went 1.75 → 2.33 us. Expect the next upstream gain to be taxed the same way (and see
§6, which is the same effect from the other side).

## 4. Edit D/F — `ACT[33]`→DVE kept; **Vector→Activation is worth nothing in this kernel**

Three moves in one patch, measured together as -0.16 on the metric but cleanly separable with the
anchor `last_end − Tensor[23].start_ts`, because one is before the anchor and two are after:

* **`ACT[33]` (370 ns, no update, gate `S[4]>=14`) → DVE slot 38** — **-0.32 us** on the metric when
  shipped alone (F). `S[4]>=14` is exactly `ACT[31]`, which writes its source `0x1004b80`; its
  pointer operand `0x1004330` comes from `ACT[5]`, which sits between `S[4]#1` and `#2`, so the one
  existing wait covers both inputs. Its only consumer is the DGE `Pool[18]` (`src 0x4d80`), which
  got an `EVENT_SEMAPHORE wait S[71]>=1` in front of it — fence on an idle engine again.
* **`DVE[63]` (CAST `0x1006480`→`0x4900`) → ACT slot 58** and **`DVE[75]` (CAST `0x6680`→`0x4800`)
  → ACT slot 66**, both into measured ~1 us Scalar holes, with the vacated DVE slot turned into
  `ES wait S[51|52]>=1`: together **+0.09 us post-anchor, i.e. nothing.** Dropped.

**Why the reverse direction fails even though the bucket table calls Vector the busier engine at
36-45 us:** in that window Vector's ops are chained through `S[5]` (Tensor) and `S[6]` waits every
2-3 slots, so removing 220 ns of Vector work just moves the next `evt_wait_time_ns` out by 220 ns.
The 18-27 us window is 8.8 us of *pure Activation program order* (inter-instruction holes of
12-100 ns, `evt_wait` ~0) with nothing to absorb anything.

> **Rule: read `evt_wait_time_ns`. Only move an op whose own wait is ~0 out of a stretch where its
> neighbours' waits are also ~0. A saturated engine whose ops all wait is not the bottleneck.**

By-products, all device-timed and reusable:
* `0x4900`: one writer (`DVE[63]`), one reader (`Tensor[85]` at 40.0 us), rewritten by
  `Scalar[108]` at 57.8. `0x4800`: `DVE[57]` writes → `Tensor[64]` reads → `DVE[75]` writes →
  `Tensor[95]` reads — a strict four-step chain, so a mover must carry a WAR guard.
  `S[5]#35 = Tensor[83]`, so `S[5]>=35` implies `Tensor[64]` has executed: that is how you turn
  "it happens to be ordered" into a guard.
* An `S4D4_TR` `CAST` slot is byte-compatible between Activation and DVE (both
  `common.NEURON_ISA_TPB_S4D4_TR_STRUCT`) — a raw 64-byte copy executes on either, device-verified
  bit-identical. So `CAST` joins `TENSOR_SCALAR_ARITH_OP` on the movable list.
  `ACTIVATE` is still Scalar-only, `TENSOR_TENSOR_ARITH_OP` does not exist on Activation, and
  Pool only runs `TENSOR_SCALAR_AFFINE_SELECT`.

## 5. Edit G/E — **an op that increments S[4] CAN be moved**; the construct is proven, `ACT[14]` is not

iter-6 §8 step 1 restricted the lever to ops with `events_update_mode == NONE`, which is what caps
it at ~4 ops per window. Lifting it needs:

> vacated Activation slot ← `EVENT_SEMAPHORE { wait S[new]>=1, update S[4] SEM_INC_COMPLETE }`

so the copy runs on DVE, increments `S[new]`, and the Activation slot *waits for it and then takes
its place in the `S[4]` count*. `semaphores.py` confirms the bookkeeping (S[4] 61/61, with an
`Activation:EVENT_SEMAPHORE: 1` line) and **the device confirms the semantics**:

* **`ACT[9]` (386 ns, `S[4]#3`, gate `S[9]>=16`) → DVE slot 14: VERIFIED bit-identical, -0.47 us.**
  Guards: `src 0x37b00` ← sem 9, `imm0_imm_ptr 0x101c000` ← sem 10, carried by an inserted
  `ES(S[9]>=16, x S[10]>=16)`. Its destination `0x1004b00` has one writer and one reader
  (`DVE[16]`, iter-6's own moved `ACT[11]` copy) and that reader is later in DVE program order, so
  no extra fence was needed at all. On the device the copy runs **283 ns** at 18.30-18.58 us in the
  Vector hole, and the Activation ES at slot 9 costs 40 ns.
* **`ACT[14]` (387 ns, `S[4]#6`, gate `S[11]>=16`) → DVE slot 18: FAILED.** `output_0`
  7.362e-03 (1363 % of the limit), `output_1` 2.744e-03 (508 %), and **the two runs were not
  bit-identical — a race, not an addressing mistake.** No hang.

So **the wait+update `EVENT_SEMAPHORE` is sound and the restriction is lifted**; `ACT[14]`
specifically has a dependency I could not find. What I verified and ruled out, so iter-8 need not
redo it:

* destination `0x1004d88` has one writer (`ACT[14]`) and its readers are `DVE[18]` (iter-6's moved
  `ACT[16]` copy) at 20.80 us and `GpSimd[17]` at 25.67 us; `Scalar[35]` rewrites the first 256 B
  at 25.28 us. The insertion put the copy at DVE slot 21 and pushed `DVE[18]` to slot 22, so the
  consumer is still *after* the producer in DVE program order;
* `src 0x37c00` is inside sem 9's span and `imm0_imm_ptr 0x101c100` is sem 11, both carried by an
  inserted `ES`;
* the only `S[4]>=3..7` waiters in the whole NEFF are `DVE[16]`, `DVE[18]`, `PE[9]`, `PE[11]`,
  `Pool[14]`, and none can be reached before the corresponding copy;
* the dependency chain DVE-copy → `S[54]` → ACT-fence → `S[4]#6` → DVE-waiter is a round trip, not
  a cycle (consistent with the observed no-hang);
* the op shape is not the problem — `ACT[21]` (kept, §3) and `ACT[9]` (kept) are the same opcode,
  same 128-elem/64-channel bf16→fp32 `TENSOR_SCALAR` with a `POINTER_IMMEDIATE`.

Remaining suspicion: `ACT[14]`'s output is consumed by **iter-6's own moved `ACT[16]` copy**, which
is the only place in this NEFF where a moved op feeds another moved op, with the `S[4]#6` round trip
in between. Something in that two-hop arrangement is not ordered. **`ACT[14]` goes on the
do-not-move list; move the remaining candidates one at a time and verify each (one verify = 1.5 min,
and a failure is loud).**

Remaining `TENSOR_SCALAR` candidates in the window, with the reason each is currently *not* worth
it (all measured on `profG`, which is the shipped NEFF):
`ACT[17]` 331 ns — the only Vector hole (21.89-22.37) starts 0.4 us after Activation reaches the
slot, net negative; `ACT[25]` 384 — Vector is busy 22.98-24.01 with §3's three copies, and inserting
in front of them pushes `S[70]>=3` out; `ACT[31]` 387 and `ACT[34]` 387 — Activation would stall
~1 us waiting for Vector; `ACT[30]` 331 — its destination `0x1004100` has three writers.
**So the 18-27 us window is now hole-limited, not candidate-limited: Vector's idle time there is
down from ~2.9 us to ~1.3 us because this iteration moved five ops into it.** The next gain in that
window has to come from making Vector cheaper there, or from the §7 list.

## 6. Edit H — **REFUTED, and it refutes a whole class**: the staging-DGE burst is flight-bound, not issue-bound

The 909 ns all-engines-idle gap at 28.6-29.5 us is Activation waiting `S[59]>=16 && S[58]>=16`
after six staging DGE rows issue serially on the SP DGE queue (sems 56,57,58,59,60,61) while
Activation is idle 26.3→29.5. So I moved the sem-59 row (`src 0x1005e80`, written only by `ACT[25]`,
so Activation program order at slot 39 is a *stronger* guard than the one it had on SP) from
`SP[12]` onto an inserted Activation slot in front of the `S[59]` fence. `semaphores.py` then
reports sem 59 as `dge:Activation` instead of `dge:SP`.

**It did exactly what it was designed to do and the metric got 0.68 us worse.** Device numbers:

| | G (shipped) | H |
|---|---|---|
| rows | 59@SP 26665-27274, 62@Pool 26792, 60@SP 27319, 63@Pool 27447, 61@SP 27964-**28565** | 59@**ACT** 26395-27052, 60@SP 26457, 62@Pool 26502, 61@SP 27109, 63@Pool 27157-**27770** |
| last row ends | 28565 | **27770 (-795 ns)** |
| `S[59]&S[58]` land | 29581 | **29663 (+82 ns)** |
| blk5 lands (`S[29]>=16`) | 29863 | **30085 (+222 ns)** |
| `Tensor[22]` | 29891 | 30112 |

The rows did go parallel and finished 795 ns earlier — **and the semaphores landed no earlier at
all.** The completion time of that burst is set by DMA bandwidth, not by when the descriptors are
generated, and running five payloads concurrently made them contend with blk5, which landed 222 ns
later. Net regression.

> **Rule: re-hosting a DGE only pays when the *host engine's issue time* is on the path (iter-6 §4,
> a 680 ns row sitting in front of a compute slot, or Pool's 500 ns wake-up). It never pays when
> what you are waiting for is the transfer itself.** Check which of the two you have by comparing
> "row end → semaphore landing" across a candidate pair before building it.

This also explains §3's missing 0.6 us: the same contention, from the other direction.

## 6a. Edit I — **REVERTED, and it says the 18-27 us Activation window is now exhausted**

Activation is idle 17.49-18.25 us (759 ns) waiting `S[6]>=7` for `DVE[11]`, then runs 8.2 us back
to back. `ACT[8]` (ACTIVATE COPY `0x101c000` → `0x1004308`, 291 ns) has **no semaphore update**, so
rotating it is numbering-neutral (iter-6 §5's cheapest edit class): its only gate `S[10]>=16` landed
by ~16.3 us, its destination is 4 bytes read only by `ACT[17]` at 21.7 us, and the adjacent
`0x1004208..0x1004307` block that `ACT[7]`/`DVE[12]`/`DVE[21]` touch does not overlap it. I moved it
from Activation slot 8 to slot 6 (a pure `pop`/`insert`, slot count unchanged).

**Device-verified bit-identical, and it did exactly what it was designed to do locally: `ACT[10]`
moved 19284 → 18856 (-428 ns). The metric did not move (53.199 vs G's 53.061/53.252).**
The chain shows where it went: `Vector[41]` — the `CAST` that produces `S[6]#19`, i.e. the gate of
the blk5 trigger — moved only 26474 → 26376 (**-98 ns of the 428**), so ~330 ns was absorbed
between `ACT[10]` and `Vector[41]`.

**That is the structural news for iter-8: the 18-27 us window is no longer Activation-bound.** This
iteration moved five ops into Vector's holes there, Vector's idle time in the window went ~2.9 →
~1.3 us, and overall Vector busy (56.3 %) has caught up with Scalar (59.9 %). Further Activation
relief in that window is now absorbed by Vector, so **the next edit there has to make Vector
cheaper, not Activation cheaper.**

A `reorder.py`-style scan for more numbering-neutral hoists (update-less op whose gate landed before
its engine's preceding hole) found only one other real candidate and two classes of false positive
worth knowing:

* **"no semaphore wait" does not mean "ready".** `Scalar[3]`/`Scalar[4]` look hoistable over the
  2.5 us hole at 13.9-16.4 us, but they read `0x1037700`/`0x1037704`, which `Vector[9]` does not
  write until 16.37 — they rely purely on Activation program order behind `Scalar[2]`'s `S[6]>=5`.
  Hoisting them would be a race. Always resolve the data dependency separately.
* **Holes computed in `compiler_pc` order are wrong when an `EVENT_SEMAPHORE` is involved.** The scan
  reported a 1103 ns Vector hole at 36144; in time order `Vector[64]` (an ES) runs 36120-36144
  *before* `Vector[63]` (37002-37216), so the "hole" is an artefact. Vector is actually back to back
  36260 → 38619 with 13-30 ns gaps, so there was nowhere to hoist `Vector[66]` into. **Sort by
  `start_ts`, not by `compiler_pc`, when measuring holes.**

## 7. Closed this iteration

* **Pool's 7 back-to-back input-load DGE rows at 14-17.9 us (Pool is 100 % busy) are NOT on the
  critical path.** This was candidate #1 in the iter-6 review, "up to about 2 us". The path through
  that window is `SP[1]`'s DGE row (12.43-13.21) → **2.36 us flight** → `DVE[8]` (15.56) →
  `Tensor[3]` (16.18) → `DVE[9]` (16.37) → `Activation[2]` (16.40) → `Tensor[6]` (17.28) →
  `DVE[10]` → `DVE[11]` (17.97) → `Activation[6]` (18.0). Pool's rows run entirely beside it, and
  the earliest consumer of any of sems 9-15 is `ACT[8]` at 18.24 us with `evt_wait_time_ns == 2` —
  **every input load has already landed before anything wants it.** Do not spend an iteration here.
* The four all-engines-idle gaps are 0.9 / 1.5 / 0.7 / 1.5 us and all four are DMA flight: the
  staging-DGE queue drain (§6, now closed), blk6's transpose, blk31's row→partition scatter, and
  the output stores. Same four as iter-6; `total idle` went 4.0 → 4.7 us as the compute around them
  shrank, i.e. **the kernel is converging on a DMA-flight floor of roughly 4.5-5 us of pure idle.**

## 8. Ranked candidates for iter-8

1. **Cheapen Vector in the 18-27 us window.** §6a is the reason this is now #1: Activation relief
   there is absorbed, Vector is co-limiting, and Vector's idle in the window is down to ~1.3 us.
   Concretely, on the shipped bytes: `DVE[11]` is a **409 ns** `TENSOR_SCALAR MIN` on 64 elements
   that `ACT[6]` waits 759 ns for, while the two *identical-shape* MINs `DVE[22]` and `DVE[24]` take
   **192 and 188 ns** — a 2x outlier worth understanding before anything else (first-of-kind
   pipeline warm-up? a field that differs? compare the three slots byte for byte). There are also
   four `MEMSET`s of 112-165 ns and three 192 ns `CAST`s gated on Pool's `S[3]` in the window.
   Note `op0=MIN` does **not** appear anywhere in `Activation0.bin` (all its TENSOR_SCALARs are
   MULT/BYPASS), so check ALU support before moving a MIN to Activation.
2. **Activation→DVE is still open where a Vector hole exists (§5), restriction lifted.** The
   wait+update `EVENT_SEMAPHORE` construct and the fence-on-the-consumer's-engine pattern are both
   proven on device (three times). What is exhausted is the *hole*, not the candidate list; see §5
   for the ten remaining `TENSOR_SCALAR`s and the measured reason each is currently net-negative.
3. **The tail**, unchanged from iter-6 §9.1 and still untried: 1.53 us of idle waiting on
   `S[20]`/`S[21]`, of which ~0.64 us is the DGE descriptor row on Activation. Convert one output
   store to a queued trigger (append a 34th block to `SP0.json["dma"]`). §6 says do *not* expect the
   flight to shrink — the gain is only the 640 ns row, and only if that row is on the path. All 42
   SB-destination descriptors in `SP0.json` are SB→SB; check whether a `to: "DRAM"` descriptor is
   legal in `qSPSpillReload0` first (blocks 0-4 are the ones to read).
4. **`dead2.py` / the range-accurate dead-store scan on the new bytes** — 2 slots and -0.43 us this
   time. Both came back empty on the shipped bytes, so it is dry *now*, but it is 10 minutes and a
   reorder can refill it.
5. **Diagnose `ACT[14]` (§5)** if the cheap candidates run out: it is the one place where a moved op
   feeds another moved op (iter-6's `ACT[16]` copy consumes it), and understanding it would unlock
   the only op left in the 20-21 us stretch.
6. **Do not retry**: everything in iter-6 §9.5, plus from this iteration — re-hosting the Pool
   input-load DGEs (§7), re-hosting a staging DGE to parallelise its transfer (§6), rotating
   `ACT[8]` into the 17.5 us hole (§6a), Vector→Activation moves in the 36-45 us window (§4), and
   moving `ACT[14]` (§5).

## 9. Mechanics confirmed or learned

* **Address = `partition<<18 | offset`**, PSUM at `>= 0x2000000`; DGE operands print as
  `0x80200<addr>`. Never mask the top bits (§1).
* The profiler's `operands` column is the best decoder available: full addresses, expanded pointer
  operands, `channels=`, `semaphore=`, `src_addr_imm`/`dst_addr_imm`, and `src_pattern`/`dst_pattern`
  for spans. Faster to write against than `isa.py` field names, and it is the vendor's own reading.
* `semaphores.py` does **not** count `events_extended_wait_*`, so a waiter count can read one low;
  that is a reporting gap, not an error. It *does* label a DGE's queue by **host engine**
  (`dge:SP` / `dge:Pool` / `dge:Activation`), and §6 shows the host really does determine which
  queue the payload goes to — the rows went parallel. (iter-4 §1's "the queue follows the semaphore"
  is about something else or is wrong.)
* `EVENT_SEMAPHORE` + `SEM_INC_COMPLETE` **with no wait** replaces a compute op you want to delete
  without renumbering (§2, proven). **With a wait** it replaces a compute op you want to *move*
  without renumbering (§5, proven). Both are device-verified bit-identical this session.
* An `EVENT_SEMAPHORE` can carry two waits (`events_wait_*` + `events_extended_wait_*`); encoding,
  from the compiler's own `ACT[39]`: `a0 10 00 00 | 05 <idx> 00 00 | <thr> 00 00 00 | zeros to
  offset 32 | 05 <xidx> 00 00 | <xthr> 00 00 00`.
* Inserting slots stays free: `DVE` 136 → 144, `SP` 69 → 70, `Pool` 37 → 38 across seven packs, no
  hang. Nothing in the NEFF records an instruction count.
* **Decide with the anchor, report with the metric** (iter-6 §10) earned its keep: edit D was -0.16
  on the metric but decomposes into -0.67 pre-anchor and +0.09 post-anchor, which is what told me to
  keep `ACT[33]` and drop the two Vector→Activation casts — and F then measured -0.32 on its own.
* Device cost this session: 7 verifies (one FAIL), 9 profiles, no hang, no silently wrong output
  that `verify.py` missed. Keeping a `save/kernel_<step>.neff` and a `work<X>/` tree per accepted
  step made every revert a single `cp` — do this, it is what made the §5 bisect cheap.

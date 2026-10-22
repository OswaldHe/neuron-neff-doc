# iter-7 — selective_scan b1 c768 s1024 n16 bf16

**Result: 501.278 -> 497.81/498.02 us (-0.67%). Cumulative from the unedited NEFF
602.637 -> 497.9 (1.2105x).** Numerics bit-identical (output_0 max_abs_err 6.245e-02 = 83% of
limit, cosine 0.999998; output_1 4.268e-05) — the exact baseline figures, and `verify.py --runs 4`
gives the same on every run. `semaphores.py` clean, `launch_rules.py` clean. Two members changed vs
the iter-5 artifact: `sg00/DVE0.bin` and `sg00/Pool0.bin`. Shipped md5 `a00e722e9970b26f5a6940523bf9a87d`.
Two edits, each a 5-repeat median:
**501.278 -> 498.625 / 498.506** (§1, the DVE reorder) **-> 498.021 / 497.811 / 497.753**
(§1b, the last two big Pool COPYs — three independent medians on the shipped bytes). The iter-5
artifact re-measured at the start of this iteration: **501.278** (spread 0.08%, DVE busy 482.0).
All spreads 0.05-0.10%, so 0.3 us is still the right noise floor and both steps clear it.

**Read `iter-6.md` first** (it closes engine rebalancing / instruction merging / DGE trigger
placement / queue re-hosting with device numbers; all of that still holds). This iteration found
the thing those closures were *measuring* without naming it, and that reframes everything:

---

## 0. THE MECHANISM: DVE and Pool share SBUF bandwidth, and only 3-SBUF-stream DVE ops pay

Measure it in one probe (10 min, no verify needed — output is garbage):
set `src0/src1/dst num_elem[0] = 1` on all 168 Pool `TENSOR_TENSOR_ARITH_OP` (512-elem fp32 `y+=Cs`)
ADDs, change nothing else. **exec 501.278 -> 489.87 us (-11.4), DVE busy 482.0 -> 468.5, Pool busy
416.8 -> 246.1.** Then diff the two `Instruction.parquet` per `compiler_pc`:

| DVE family | n | tax removed | median dur before | clean dur |
|---|---|---|---|---|
| `TT` with a PSUM operand (`dBx = dtx*B_psum`, `Cs = C_psum*state`) | 384 | **-1.3 us (none)** | 678 | 678 |
| `TENSOR_TENSOR_SCAN` (dA, dBx, state all SBUF, 2 ALU-ops/elem) | 192 | **+0.6 us (none)** | 1269 | 1269 |
| `TT` bf16 `dtx = dt*x` (3 SBUF streams) | 12 | **+6.66 us** | 1119 | **681** |
| `TT` fp32 `y += ...` (3 SBUF streams) | 24 | **+2.97 us** | 679 | **681** |

So: **every DVE compute is 681 ns (TT) / 1269 ns (scan) when Pool is quiet. The only ones that
inflate are the 36 TTs with all three operands in SBUF, and they inflate by ~0.6-0.9 ns per ns of
concurrent 512-element Pool activity** (up to 2061 ns for a 681-ns op). A DVE TT with one PSUM
operand (2 SBUF streams) is immune. A scan is immune (3 streams spread over 2 cycles/element, so
half the bandwidth per cycle). Tiny Pool ops (the 115 one/two-element carry `COPY`s) do not tax;
only the 168 512-element ADDs and the 2 512-element `COPY`s do.

**Corollaries that correct earlier notes.**
* iter-5 R4's "bf16 DVE arithmetic is slower" is wrong: bf16 x bf16 -> fp32 is **681 ns, identical
  to fp32**. Those 12 slots looked slow only because one sits in every group's Pool-busy window.
* iter-6 M2b ("the contention tax is attached to the window, not the instruction") is *this*
  effect, and it is more specific than M2b states: the window only charges a DVE op that is
  issuing 3 SBUF streams. Put an immune op in the window and the tax is ~0 — that is the lever
  below, and it is worth 1.78-2.8 us measured.
* **11.4 us (2.3%) is the hard ceiling of the whole Pool family**, and no legal traffic reduction
  exists (see §3).

Useful second-order numbers, all measured here:
* Consecutive DVE computes overlap by a flat **77 ns** (median, 611 boundaries, 46.5 us total),
  independent of how many `EVENT_SEMAPHORE` slots sit between them. So the marginal cost of a DVE
  instruction is `duration - 77 ns`, and merging two 512-elem TTs can save at most 126-77 = 49 ns.
  Another nail in iter-6 M2's coffin.
* `exec = DVE all-slot union + ~0`. iter-5 artifact: union 501.70 vs exec 501.278. Shipped
  artifact: union 498.06 vs exec 498.506. **DVE occupancy is 100%; exec is DVE's union, full stop.**
  Decompose it as `compute union + exposed non-compute + holes`:
  iter-5 `491.57 + 1.76 + 7.75`; shipped `487.38 + ~1.5 + 8.83`.
* **The 443 non-compute DVE slots (397 wait-carrying ES, 32 MEMSET) expose only 1.76 us total.**
  Deleting or merging ES slots is worth ~0. Confirms iter-5's 4000-no-op experiment, with a number.
* Pool's own ADD is **990 ns fixed + 2.0 ns/element** (from the 1-element probe: 168 ADDs cost
  167 us at `num_elem=1`). It runs 1263 ns when DVE is quiet and 2158 when not — the contention is
  symmetric, but Pool is not on the critical path so that half does not matter.

---

## 1. WHAT SHIPPED: move each taxed 3-SBUF-stream DVE TT later, past immune neighbours

Per group the DVE stream is
`... SCAN[n] | ES | dtx=dt*x (3-sbuf) | ES | Cs[prev] (PSUM) | dBx[prev] (PSUM) | ES | dBx[0] (reads dtx) ...`
and Pool's 512-element ADD run ends ~1-2 us *into* the group boundary, after which Pool is idle for
6-7 us. So the taxed op sits just before the idle window. Moving it a few slots later — past
PSUM-operand MULTs, which are immune — puts it in the idle window at no cost to the ops that move
up. Measured per site: 1685 -> ~690 ns.

**The transform (this is the reusable part).** To move DVE slot `X` later past `k` slots `Y1..Yk`:
1. Legality: stop before any slot that reads `X.dst`, writes `X.dst`, or writes either `X.src`
   (PSUM operands are never a conflict). Treat PSUM extents as non-aliasing with SBUF.
2. **The only mandatory semaphore fixup is every wait with `S6 == ord(X)` exactly**; set it to
   `ord(X) + (#S6-updating slots among Y1..Yk)`. Waits on the Ys' ordinals can be left alone: after
   the rotation `ord(i)+t` names `Y_{t+1}` instead of `Y_t`, which completes *later*, so stale
   values are conservative, not unsafe. Waits on ordinals below `ord(X)` are untouched.
   (This is why it is far cheaper than iter-6's E3, which re-pointed 157 waits.)
3. `ord(X)`'s waiters are usually **one DVE ES slot** — the RAW guard that stops the first consumer
   from starting inside DVE's 77-ns pipeline overlap — plus sometimes one Pool slot (the `y`-store
   DMA trigger). If a DVE waiter sits at position `p`, clamp `k` to `p - X - 1` so the guard stays
   physically after `X`; then raising its value is safe and cannot deadlock.
4. **Skip any target whose `Y` set contains no `TENSOR_TENSOR*`** (i.e. you would only be hopping
   an `EVENT_SEMAPHORE`). See R1: those lose.
5. **Skip any target whose `Y` set contains another 3-SBUF-stream TT** — swapping tax for tax is
   exactly net zero (verified by hand on three sites).

7 of 36 sites qualified (`pc 610, 633, 785, 804, 809, 960, 984` in iter-5 numbering) and shipped:
**501.278 -> 498.506/498.625**. Residual 3-sbuf tax fell 8.9 -> 5.2 us; DVE compute union
491.57 -> 487.38 us. Note exec gained only 2.8 of the 4.2 us of compute union removed — DVE holes
grew 7.75 -> 8.83 us, i.e. roughly a third of any compute saving at a group boundary reappears as a
DVE wait. Budget for that.

A second and third pass over the shipped artifact (same rule, re-picked from the new profile) find
**nothing**: every remaining taxed slot has `k = 0` or an ES-only / 3-sbuf-only `Y` set. Do not
lower the `MIND` threshold below ~750 ns — the rule then starts "moving" already-clean 681-ns ops,
which can only make them taxed.

## 1b. ALSO SHIPPED: kill the last two 512-element Pool `COPY`s (iter-6 §5.1 / my own candidate #1)

iter-5 retargeted 10 of the 12 group-first-`y` writers (`y = Cs` instead of a Pool COPY) and left 2.
Both are now gone: **498.5 -> 498.021 / 497.811 us**, bit-identical.

* `Pool 8` (`0x7940 -> 0x4940`, 1873 ns) is group 0's. Its data producer is `DVE 14`
  (`ord == 7`, the value the COPY waits on). Nothing else reads `0x7940` before `DVE 26` overwrites
  it, and `0x4940` has no prior content in group 0, so: point `DVE 14`'s `dst` at `0x4940`, done.
* `Pool 303` (`0x8000 -> 0xa940`, 2746 ns) is group 10's and needed one extra step. **The `S6`
  value it waits on (486) is not its producer** — it is the ordinal of `DVE 799`, a *1-element*
  `MEMSET` at `0xa940`, and that MEMSET is the **WAR guard** (`W S9 >= 64`, a DMA semaphore) that
  licenses reuse of `0xa940`; the data producer is `DVE 798` (`ord == 485`). Writing `y` from
  `DVE 798` would put the write *before* the guard, and a TT has no extended wait slot to carry
  `S9 >= 64` alongside its own `S5 >= 287`. Fix: **move `DVE 798` one slot later, after the MEMSET**
  (§1's transform; only waiter on ord 485 was `PE 572`, re-pointed to 486), then point its `dst` at
  `0xa940`. The MEMSET's 1-element zero write is then harmlessly overwritten by the 512-element
  store, so it does not need relocating.
* **Retiring a Pool slot the cheap way:** do not delete it and do not convert the opcode. Leave the
  `COPY` with its events block untouched and repoint `src`/`dst` into a free SBUF hole
  (`0x16040`/`0x16060` and `0x16080`/`0x160a0`) with `num_elem = [1,1,1,1]`. Semaphore counts,
  ordinals and `semaphores.py` are then trivially unchanged — 2746 ns becomes ~190 ns.
* Generalisable lesson: **before retargeting a producer, check whether the consumer's `S6` wait
  names a guard instruction rather than the producer.** iter-5 almost certainly left these two
  because of that MEMSET.

---

## 2. REFUTATIONS (device-measured this iteration)

### R1. Moving a taxed TT past an `EVENT_SEMAPHORE` alone: **+0.35 us per site, net loss. CLOSED.**
The first build moved 11 sites, including `pc 87, 262, 438, 527` whose only `Y` was one ES. Result
**499.499 us** vs **498.625** for the 7-site build. The 4 extra sites each gained ~700 ns of
duration and each created a *new* DVE hole of exactly the same size at exactly its own timestamp
(holes 7.75 -> 12.37 us; the new ones were 1463 ns @55.2, 796 @218.8, 639 @137.3, 415 @260.0).
Mechanism: hopping the ES pulls that ES's wait earlier in the stream, so DVE evaluates it before
the TT instead of concurrently with it. **Rule: never move a compute slot across a wait-carrying ES.**

### R2. Moving the taxed `dtx = dt*x` MULT *earlier* (past the preceding PSUM MULT): **+4.83 us. CLOSED.**
The mirror idea — put the immune op in the Pool-busy slot by moving the taxed one in front of it.
Built for 7 of the 8 blocked sites (the leading input-guard ES has to travel with the MULT, and all
waiters on the skipped op's ordinal must be *raised* by 1, which is the mandatory direction when
moving earlier). Bit-identical, `semaphores.py` clean, **503.456 us** (from 498.625). The ES that
travels with it carries the input DMA wait (`S11/S12/S14/S15 >= 16`), so pulling it earlier makes
DVE block on a DMA that has not landed. **Moving a DMA-guarded op earlier is always a loss on this
kernel.**

### R3. Swapping a taxed 3-SBUF TT with a *clean* 3-SBUF TT: **+1.76 us. CLOSED (measured).**
Each group's boundary has exactly three 3-SBUF-stream TTs (`dtx = dt*x`, and two `y += ...`) and the
first DVE slot after the boundary is always inside Pool's last ADD. Whichever of the three lands
there pays ~700 ns; the other two already sit in Pool's idle window. So I expected a swap to be
exactly net zero, and dropped the `nsbuf == 3` guard on 4 sites (`pc 87, 262, 438, 527`) to check.
**497.81 -> 499.573 us**, bit-identical, `semaphores.py` clean. It is *worse* than net zero: DVE busy
fell to 472.9 (from 475.3) yet exec rose 1.76 us, i.e. the duration saved on the taxed op came back
as DVE holes — the same conservation R1 shows. **Keep the guard.**
The only escape would be putting a *scan* or a PSUM-operand MULT in the busy slot, and every such op
in the neighbourhood either precedes the boundary already or needs `dtx`, which is produced by the
taxed MULT itself. (In the 3 groups where the previous group leaves *two* spare PSUM MULTs after the
boundary — `pc 610/785/960` — that is exactly what §1 exploited; the other 8 groups do not have
them.) **Floor: ~12 x 250-700 ns of tax that instruction reordering cannot reach.**

### R3b. Merging the 115 tiny carry `COPY`s further is worth <=1.45 us and is **blocked by state-buffer reuse**. CLOSED.
Probe (no verify needed, carries break): convert all 117 tiny Pool `COPY`s into Pool
`EVENT_SEMAPHORE`s that keep the same wait and the same `U S3` (the `CTRL_ES` struct has
`events_update_mode`/`idx`, and `semaphores.py` stays clean). That removes ~25 us of Pool
instruction time: **497.81 -> 496.356 us, i.e. the whole family is worth at most 1.45 us**, and a
realistic 4-way merge (115 -> ~35 slots) would reach maybe 1.1 us.
It cannot be built anyway. The 115 slots are 77 pairs (`num_elem=[1,2,1,1]`, `dst` step 1 into the
`0x16000+4k` carry array) and 38 singles, covering 12 x 16 = 192 carries; `src` is
`state_base + 511*4` (e.g. `0x713c` for the `0x6940` state tile). A 4-way merge needs
`num_elem=[1,2,2,1]`, i.e. two pairs with the **same inner stride**. The inner strides cycle
`(-7168, 4096, 3072)` per group, so same-stride pairs are 3 apart — and those pairs have the
**same `src` address** (e.g. slots 51 and 62 both read `0xb93c`), because `state[n]` and `state[n+6]`
share an SBUF tile. A merged read would take one snapshot of a buffer that is rewritten in between.
This is exactly why the compiler stopped at pairs.
Side note: the tiny `COPY`s take ~180 ns, not the 600-1050 ns some rows show — the long ones are
stalled on their own `S6` wait, which the profiler folds into `duration`. And tiny Pool ops do not
tax DVE (verified against the §0 probe), so this family only ever mattered as Pool *occupancy*.

### R4. No `perf_mode` / throughput-mode field exists on DVE compute. CLOSED (docs 14.3 #5).
`NEURON_ISA_TPB_S3S3D3_TT_STRUCT` (TT) and `NEURON_ISA_TPB_S2S2D2_STT_STRUCT` (SCAN) have **no**
`perf_mode` field; only `MATMUL` has `perf_opt`. `NEURON_ISA_TPB_ACCUM_CMD`
(`IDLE/ZERO/ACCUMULATE/ZERO_ACCUMULATE/LOAD_ACCUMULATE`) exists but only on `ACTIVATE`, the
`TENSOR_SCALAR_*` family and `SCAN` — **not** on `TENSOR_TENSOR`, so DVE cannot accumulate into its
destination and the `Cs` MULT cannot absorb the `y +=`. The header's advertised
`v3-2x-4x-dve-perf-mode` is not reachable from an instruction field on this ISA version.

---

## 3. WHY THE 11.4 us POOL CEILING IS UNREACHABLE (arithmetic, not measurement)

Per group (12 groups = 6 channel tiles x 2 seq chunks, 16 states each), SBUF bytes per
(element, 128 partitions): DVE `8 + 16*8 (dBx) + 16*12 (scan) + 16*8 (Cs) + 2*12 = 480`,
Pool `14*12 = 168`. Pool's share is all `y += Cs` and summing 17 arrays needs 16 two-input ops at
3 streams each; there is no cheaper legal form:
* **N-way `TENSOR_REDUCE`** would cut it to `N+1` streams, but the opcode is DVE-only
  (iter-4 R1: Pool -> NRT_TIMEOUT) and on DVE a `[16,512]` reduce costs 7.6 us/group = 91 us of DVE
  against the 11.4 us ceiling. Dead (confirms iter-5 R5 with the new model).
* **Put a `y` or `Cs` stream in PSUM** (would make Pool's ADD 1-stream, or make the 36 taxed DVE TTs
  immune): **all 8 PSUM banks are occupied** — `0x2000000, 0x2000800, ... 0x2003800`, 0x800 = one
  512-fp32 tile each, kept saturated by PE's B and C broadcast matmuls. No spare bank.
  Pool cannot address PSUM anyway (iter-1 #8).
* **bf16 accumulation / bf16 `Cs`** would halve the traffic and changes the arithmetic. Out of scope.
* **Merging Pool ADDs** is invalid, not merely slow: they are a serial in-place chain on one `y`
  (`dst == src0`), so a `[512,2]` merge computes two independent sums, not one. That is the real
  reason iter-4 R4 lost 62 us; do not re-derive it as a cost model.
* **Reordering the 17 addends** to shuffle which op lands in the busy slot would reorder an fp32
  sum. Not attempted: it changes what the kernel computes.

---

## 4. THE HEAD AND THE TAIL, with numbers

The 4.1 us DVE hole at t=10.3 is **not** reorderable. Timeline: the runtime prologue
(`TENSOR_LOAD`/`DRAIN`/`COMPARE_BRANCH`/`NOTIFY` on all five engines) owns t=0.08..10.4 us and is
not in any editable stream. Then `Pool pc1..pc6` are the input `PSEUDO_DMA_DIRECT2D` triggers,
657 ns of Pool each, serial. DVE's first compute (`pc5`, `dtx = dt*x`, `W S8>=16`, srcs 0x37b00 and
0x37700) needs only triggers **#1 (`semaphore 8`) and #2 (`semaphore 9`)**, which are already first;
they push at 11.07 and 11.73 and their semaphores fire at ~14.2 and ~14.86. So the head is
`10.4 (runtime) + 1.3 (two trigger pushes) + ~2.5 (128 KB DMA flight, 51 GB/s)` and the first TT
starts at 14.96. **Trigger reordering is already optimal; the only lever left is the 2.5 us flight,
i.e. splitting one 128 KB transfer across two queues, worth <=1.2 us and requiring a new
descriptor.** Tail: last DVE compute ends ~506.9, output DMA lands ~509; `exec` already excludes
most of it.

---

## 5. WHAT I WOULD TRY NEXT, honestly ranked

The model is now tight: `exec = DVE all-slot union`, and
`compute union = 411648 ALU-op rows x 0.928 ns (382 us, irreducible) + 612 x ~204 ns fixed
- ~46 us pipeline overlap + ~5 us residual Pool tax`, plus ~1.5 us exposed ES and ~9 us of DVE
holes (4.2 of which is the un-editable head). Everything except the tax and the holes is
structural. **I believe 495-497 us is the floor for instruction-level editing, i.e. ~0.5% left**,
and both of this iteration's levers are now exhausted.

Residual 3-SBUF-stream tax on the shipped artifact, from `xp/tax.py prof_m3` (total 7.6 us, of
which 5.0 on the 36 TTs and 2.3 on the 192 scans). The 8 big ones, all blocked by R3:
`pc 87` 1388, `pc 438` 725, `pc 262` 573, `pc 527` 325, `pc 634/810/985` 437 each (these three are
already at `k = 0`), plus two scans, `pc 38` 1110 and `pc 967` 202.

1. **The 4 deterministically slow scans — `pc 38` (2379 ns), `pc 967` (1471), `pc 792` (1468),
   `pc 617` (1436) against 1269 for the other 188 — are worth 1.7 us and I could not explain them.**
   What I established, so you do not repeat it:
   * Their duration is **identical to +-3 ns across all four profiles I captured** (iter-5 artifact,
     the Pool-shrink probe, and both shipped builds), so it is **not** contention — the §0 probe
     that deleted all Pool element work left `pc 38` at 2377 ns.
   * Decoding them against a fast scan (`pc 43`) shows **no field difference except addresses and
     the carry pointer**: same `num_elem [512,1]`, `step_elem [1,1]`, `op0=MULT op1=ADD`, all FP32,
     `num_active_channels=128`, `accumulator_cmd=IDLE`, same wait/update block shape.
   * Incidental discovery while doing this: `pc 617/792/967`'s `imm0_imm_ptr` is **not** in the
     `0x16000` carry array — it is `0x913c / 0xa13c / 0xb13c`, i.e. `state_base + 511*4`, so those
     scans take their carry straight out of the previous chunk's state tile and are the ones that do
     **not** need a Pool carry `COPY`. `pc 38` and the fast scans point into `0x16000+4k` as usual.
     (It also means those three got ~100-170 ns *slower* under my §1 reorder, which is already
     inside the measured net win.)
   * **It is not a hidden wait either.** `pc 38` waits `S3 >= 7` (Pool, the carry `COPY`);
     `SemaphoreUpdate.parquet` has `S[3]` reaching 7 at **t=30706** and `pc 38` starting at
     **30730**, i.e. its wait was already satisfied, and `evt_wait_time_ns` is only 432. So ~1.9 us
     of the 2.38 us is genuine execution of an instruction that is byte-identical (bar addresses) to
     a 1269-ns one.
   * **The carry-pointer hypothesis is refuted.** `pc 38`'s `imm0_ptr` (`0x1601c`) had been written
     by the Pool `COPY` that retired 24 ns earlier, so I probed whether DVE's carry fetch stalls
     behind an in-flight Pool SBUF write: point all four scans' `imm0_imm_ptr` at a long-settled
     free address (`0x15404`), timing only. **Durations unchanged: 2375 / 1443 / 1469 / 1472.**
   * Their `src0/src1/dst` offsets have the *same relative layout* as the fast scans
     (`src1 - src0 = 0x1000`, `dst - src1 = 0x800`), so it is not an obvious bank pattern either.
     Whatever this is, it is not in the 64 bytes of the instruction. Worth 1.7 us if someone cracks
     it; I would not spend a whole iteration on it.
2. **Split the startup 128 KB input DMA across two queues** (§4). Ceiling ~1.2 us, needs a new
   `def.json` queue/descriptor, and `owner` cannot be changed post-compile (iter-6 M4) — so it must
   reuse an existing queue with a different semaphore. Low odds.
3. **Do not retry**: everything in iter-6 §5.4, plus R1 (hop an ES), R2 (move earlier), R3 (rotate
   the three boundary 3-sbuf TTs), R3b (merge the carry COPYs), R4 (perf mode / DVE
   dst-accumulate), the DVE reduce tree, Pool-side traffic reduction of any kind (§3), and
   deleting/merging DVE ES slots (1.76 us total exposure). All 12 group-first-`y` writers are now
   retargeted, so that family is finished too.
4. If a future iteration gets a *source-level* lever: the one big number is that 192 of the 612 DVE
   instructions are scans at 2 ALU-ops/element (245.7 us, half of DVE). A chunked/blocked scan
   (Mamba2-SSD style, matmul on the 43%-idle PE) is the only thing that changes the 382 us of
   irreducible DVE ALU work. Not reachable by editing the binary.

## 6. Tooling (`xp/`, deleted with the worktree — rebuild these two first)

* **`xp/plan_move.py`** — per-target legality and move distance. Needs: exact operand run
  enumeration over pattern dims 1..n (iter-6's `dep.py` lesson; a bounding box false-positives),
  a PSUM test (`*_reg_marker == 2` **or** `immediate >= 0x2000000`), the `S6` ordinal map
  (count slots with `events_update_idx == 6 and events_update_mode != NONE`), and a waiter census
  over all 5 streams on **both** `events_semaphore_value` and `events_extended_semaphore_value`.
* **`xp/move.py` / `move2.py`** — applies the rotation: slice the 64-byte blocks, `blk[1:]+[blk[0]]`,
  rewrite the waiters, re-encode with `Instruction.encode()` padded to 64 bytes. Process targets in
  **descending** pc so indices stay valid. `ONLY=`, `MIND=`, `PROF=` env knobs.
* `xp/tax.py <profdir>` — residual tax per family against the 681/1269 ns clean baselines. This is
  the one number to watch; it was 8.9 us before, 5.2 us after.
* Gotchas: `isa.decode_stream` yields `(index, Instruction)` tuples; `pandas` import breaks if a
  local file is named `struct.py`; `Pool` `COPY` uses the `src_`/`dst_` prefix, not `src0_`.

# iter-6 — selective_scan b1 c768 s1024 n16 bf16

**Result: NO CHANGE SHIPPED. `kernel.neff` is byte-identical to the iter-5 artifact
(md5 `c34c2e233e7177ea0a507b9ac8d96f74`), which I re-measured twice this iteration, before any edit and again after reverting:
`total_exec_time` **501.421 us** (spread 0.05%, min 501.314 max 501.562; DVE busy 482.344,
Pool 416.929, PE 218.495, ACT 146.277, DMA 103.962, SP 48.338) and **501.313 us** (spread 0.06%,
min 501.230 max 501.550; DVE busy 481.741, Pool 416.800). So quote the shipped file as
**501.3-501.4 us, 1.2021x over the unedited NEFF**, and treat 0.3 us as the noise floor.**

I built and device-measured **ten** edits: six that ran correctly and regressed (five of them
badly), three that crashed the host runtime and one that deadlocked. Together they close, with
numbers, the whole "move work off DVE", "merge DVE instructions", "move the DMA triggers" and
"re-host the DGE queue" families — i.e. every ranked candidate iter-1..iter-5 left open, plus the
1024-element fused scan, which I priced and then found to be structurally blocked. **The one-line summary for whoever picks
this up: on this NEFF every microsecond of DVE duration you remove is paid back somewhere else at
roughly 1:1, and the only engine that could take the work (Pool) charges ~1 us of exec per 1 us
of Pool duration added. There is no free capacity left to arbitrage.**

Read `iter-5.md` first (it has the build-script skeleton, the relocation lever and the measured
DVE/Pool calibration), then this file's §1 for the measurement table and §2 for the five
mechanisms (M1, M2, M2b, M3, M4), §3 for a hazard that `verify.py` can pass by luck, §4 for the
tooling worth rebuilding, and §5 for what I think is actually left (not much).

---

## 1. The six measurements (all on the iter-5 artifact, all 5-repeat medians)

| # | edit | exec us | Δ | DVE busy | Pool busy | numerics |
|---|---|---|---|---|---|---|
| — | iter-5 artifact (shipped again) | **501.421** | — | 482.3 | 416.9 | bit-identical |
| E1 | move **all 24** DVE `y+=` ADDs to Pool (+ relocate 12 ACT-produced addends, + S3 renumber, + 12 DVE hand-off guards) | 509.759 | **+8.34** | 481.8 | 443.9 | bit-identical |
| E6 | move only the **12 `y += D*x`** ADDs (the S4 waiters) to Pool, same machinery | 502.037 | **+0.62** | **480.7** | 428.2 | bit-identical |
| E2 | pairwise-merge **93 `dBx = dtx*B_psum` MULT** pairs into `[512,2]`, **at the earlier slot** | 527.562 | **+26.1** | 488.8 | 418.2 | bit-identical (but see §3, this build was *unsafe*) |
| E3 | pairwise-merge **87 `Cs = C_psum*state` MULT** pairs into `[512,2]`, **at the later slot**, 157 cross-engine S6 waits re-pointed | 584.825 | **+83.4** | 519.2 | 387.0 | bit-identical |
| E4 | delay the **15 big (128 KB) Pool DGE input-load triggers** by 2 Pool compute slots | 502.483 | +1.06 | 485.3 | 417.1 | bit-identical |
| E5 | advance the same 17 triggers by 2 Pool compute slots | 513.017 | +11.6 | 495.9 | 407.2 | bit-identical |

Every one of the six passed `verify.py` bit-identically (`output_0` 6.245e-02 = 83% of limit,
`output_1` 4.268e-05 — the baseline figures) except an *intermediate* 96-pair version of E3,
which is how I found the hazard in §3. `semaphores.py` and `launch_rules.py` clean on all of them.

---

## 2. The two mechanisms these six measurements establish

### M1. Pool charges ~1 us of exec per 1 us of Pool duration added. The DVE/Pool arbitrage is over.

iter-5's probe (shrink the 24 `y+=` ADDs to `num_elem=[1,1,1]`, change nothing else) measured
**-11.59 us**, and the iter-5 review's balance model predicted `k ≈ 23` more ops could move. Both
are right about the DVE side and both ignore what Pool charges. Measured:

| | DVE duration removed | Pool duration added | exec |
|---|---|---|---|
| E1 (24 ADDs) | 24 x 0.62 = 14.9 | +27.0 (measured busy) | **+8.3** |
| E6 (12 ADDs) | 12 x 0.62 = 7.4 | +11.3 (measured busy) | **+0.6** |

and the DVE **busy** figure barely moves (482.3 -> 481.8 for E1; -> 480.7 for E6) even though
14.9 / 7.4 us of DVE *duration* is gone. So DVE converts almost all of it into blocking. E6, the
cheapest possible version (one ADD per group, inserted into a measured ~2.5 us Pool idle window
immediately before the group's `y` store DMA), is still +0.6 us — i.e. **exactly break-even at
best**. There is no subset that wins: the per-ADD ledger is DVE -0.62, Pool +0.94, exec +0.05.

Corollary: iter-1 candidate #1 (stage PSUM on ACT, move the `dBx` MULTs to Pool) and the iter-5
review's candidate #1 are both dead for the same reason, independent of the staging mechanics.
Also dead: the iter-5 review's #3 (free 39 us of Pool by re-issuing its DGE loads from SP) — even
with *infinite* free Pool capacity the ceiling is the 11.6 us probe, and E6 shows the realized
fraction is ~0 at the margin.

### M2. Pairwise row-merging a 512-element DVE op saves only 126 ns, and both merge directions break the PSUM pipeline.

**New calibration (measure once, never re-derive):** a `[512,2]` 2-row fp32 `TENSOR_TENSOR` on DVE
takes **1222 ns** against **2 x 674 = 1348 ns** for two 1-row ops (slot 94 of E2's profile, same
operands, same engine). So the per-instruction fixed cost on DVE is **126 ns**, minus 59 ns for
the 1-element MEMSET that has to keep the retired slot's S6 ordinal => **67 ns net per merge**.
iter-1's "~136 ns/pair" was right; iter-5 R1's "450 ns per merge" was DVE *busy* (duration +
blocking), not duration, and must not be used as a duration estimate.

67 ns x 96 pairs = 6.4 us gross per family. Against that:

* **Merging at the earlier slot (E2, the `dBx` family)**: the merged op needs *both* PSUM banks at
  the earlier slot, so it inherits the later member's `S5 >= v` PE wait one-to-three DVE slots
  early. DVE busy +6.5, exec **+26 us**. PE is only 44% busy but it is paced by DVE's PSUM-WAR
  release, so pulling a PSUM read earlier does not make the matmul arrive earlier.
* **Merging at the later slot (E3, the `Cs` family)**: the earlier member's S6 ordinal now retires
  late, so every cross-engine waiter on it (157 of them, mostly PE PSUM-WAR guards) has to be
  re-pointed to the later ordinal. That delays PE by 2-5 DVE slots per pair: PE busy
  218 -> 195.8 (it is starved, not freed) and exec **+83 us**.
* The `dBx` family *cannot* merge at the later slot anyway: only **12 of 96** pairs survive,
  because the `SCAN` that consumes `dBx[s]` sits between the two MULTs. (Script says
  `dstI 0x... vs DVE<scan>/R/src1` for 84 of them.) The `Cs` family cannot merge at the earlier
  slot for the mirror reason (its `src1` is the scan output produced between the two).

Together with iter-4 R4 (pairwise-merging the **Pool** `y+=Cs` ADDs: 62 us loss) and iter-5 R1
(pairwise-merging the 24 DVE `y+=` ADDs: +0.33 us), **instruction merging is now closed on all
four families on both engines.** Do not reopen it.

### M2b. The group boundary carries a ~1-2 us "contention tax" that is attached to the *window*, not to the instruction.

In E1, DVE's group-0 boundary scan (slot 95) went from **1.277 us to 2.226 us** while its own wait
(`S4>=18`, supplied by ACT slot 23 at t=41.3) had been satisfied 8 us earlier. The 1.4 us of ADD
duration I removed from slots 89/92 reappeared verbatim inside the next instruction. Same shape as
the 12 `dt*x` MULTs that run 677..2073 ns (iter-5 R6/R7). So **"shorten the critical instruction
at the group boundary" is not a lever**: whichever DVE instruction occupies the window
45-52 us (mod 42) pays for the 256 KB input DMA and Pool's densest ADD run. This is the
mechanism behind M1's "DVE converts duration savings into blocking".

### M4. The Pool DGE queue cannot be re-hosted on another engine. CLOSED (iter-5 review's #3).

Corpus evidence said this should work: `PSEUDO_DMA_DIRECT2D` has **18062 SP slots and 16216 ACT
slots** (vs 14811 Pool) across the 14 unpacked NEFFs, and `def.json`'s `dma_queue` is seen with
`{"type": "dynamic", "owner": "sp"}` 20 times and `"owner": "act"` 18 times, not only
`"owner": "pool"`. Three device runs close it anyway:

| edit | result |
|---|---|
| move all 60 Pool `PSEUDO_DMA_DIRECT2D` into `SP0.bin` (time-matched insertion, each keeping its own wait; none of them updates S3, so no renumbering) **and** set `def.json dma_queue.qPoolDynamic.owner = "sp"` | **host segfault**, exit 2, ~9 s (register dump from the runtime) |
| same, with `owner = "act"`, slots into `Activation0.bin` | **host segfault**, exit 2, ~9 s |
| `def.json dma_queue.qPoolDynamic.owner = "sp"` **alone**, triggers left on Pool | **host segfault**, exit 2, ~10 s |
| triggers moved to `SP0.bin`, `owner` left as `"pool"` | **NRT_TIMEOUT**, exit 1, 70.0 s |

So the `owner` string is consumed by the runtime for ring/DGE setup and cannot be flipped
post-compile (the compiler must emit more than this field), and a non-owner engine that triggers
the queue simply never retires the instruction — iter-1 #8 / iter-4 R1's 70 s signature. The 39 us
of Pool `PSEUDO_DMA_DIRECT2D` time (60 x 0.657 us) is therefore not recoverable, and by M1 it
would have been worth ~0 anyway.

### M3. The compiler's DGE trigger placement is a local optimum in both directions.

E4/E5 permute the 15-17 Pool `PSEUDO_DMA_DIRECT2D` slots that load 128 KB into SBUF (`dst` marker
`0x10`, `dst_num_elem[0]*dst_elem_size == 131072`), by +-2 Pool compute slots. They carry no S3
update, so the permutation is a pure 64-byte reorder with no renumbering. Both lose:
**later +1.06** (DVE blocks more on the input queues: busy 482 -> 485) and **earlier +11.6**
(the triggers' own `S4`/`S8` waits block Pool, which delays its ADDs, which lands on DVE's
hand-off: Pool busy *drops* 417 -> 407 and DVE busy rises to 496). This closes "peak-shift the
DMA out of the dt*x MULT's window", which was the only untested reading of iter-5 R6/R7.

---

## 3. One real hazard found — read this before merging or moving any DVE instruction

**An `EVENT_SEMAPHORE` slot between two instructions can be carrying the wait that makes the
second one's operands valid. A hazard scan over instruction *operands* alone will not see it.**

E2's first build merged `dBx` pairs at the earlier slot after checking only that nothing in
`(i, j)` touches the operands. It passed `verify.py` **bit-identically** and was still wrong:
e.g. the pair (94, 102) reads PSUM `0x2002000`, whose PE matmul is guarded by
`ES slot 101: W S5>=35` — *between* the two MULTs. Merging at 94 skips that ES entirely. It only
"worked" because PE happened to be ahead on this input. **Collect every wait on every slot in
`[i, j]`, not just the two instructions' own waits.**

The symmetric hazard on the data side cost me four device runs on E3 and is worth a rule:
**a non-DVE engine can write your operand inside the window.** 96 of 96 `Cs` pairs passed a
DVE-only hazard scan; 9 of them were wrong (`output_0` max_abs_err 1.08e+01, 90% of elements still
in tolerance, so it is a *localised* corruption and easy to miss). The culprits are ACT writes to
the alias of a `state`/`Cs` buffer inside the window, e.g. `(168,173)`: `src1_i = 0x7140` is
rewritten by `ACT 44` at t=87.6, between DVE 168 (t=86.4) and DVE 173 (t=88.8). The check that
fixes it, using `start_ts` from the profile as the global order:

```
for every non-DVE access A in the time window [t(i)-0.3, t(j)+2.0]:
    if A is a WRITE overlapping any src of slot i                     -> reject the pair
    if A overlaps dst of slot i and A has no S6 wait v with
       (v == ord(i)  # i.e. one I am about to re-point
        or v >= ord(j))                                              -> reject the pair
```
With it, 87 of 96 `Cs` pairs survive and the build is bit-identical. (The 9 rejects are exactly
the ones at `ACT`'s `dA`/`D*x` rewrite points, one or two per group.)

**Bisection recipe that found it in 6 device runs (~4 min):** make the edit list sliceable with a
`LIMIT=a,b` env var, apply only `pairs[a:b]`, and binary-search on `verify.py`. Cheaper than any
amount of static reasoning, because `verify.py` is a complete oracle for this kernel.

---

## 4. Tooling I built (reproduce these first next time)

All in `xp/`, ~60-180 lines each, built on iter-1's `build3.py` skeleton. They are gone after the
iteration; the two worth rebuilding immediately are `dep.py` and `dep4.py`.

* **`dep.py` — exact-run operand extents.** The single most important fix over previous
  iterations. Earlier scans used `[base, base + num_elem[0]*size)`, which (a) misses the
  dim-1/2/3 offsets and (b) turns a pairwise-merged Pool carry `COPY` (`num_elem=[1,2,1,1]`, two
  4-byte elements 0x10000 apart) into a 64 KB bounding box that false-positives against
  everything. Enumerate the cross product of dims 1..3 and emit one `(lo, hi)` run per
  combination. With it the real SBUF holes are
  **`0xf340-0x15400` (24768 B), `0x15404-0x16000`, `0x16040-0x16100`, `0x16900-0x1c000` (22272 B),
  `0x31640-0x31700`, `0x31704-0x37200` (23292 B)**; `[0, 0x4000)` is the DGE scratch
  (`def.json: dma_queue.qPoolDynamic.dynamic_dma.SB_scratch_partition_size = 16384`) and
  `0x38038` is the runtime's `evtaccel` reservation. `def.json` contains **no** SBUF offsets at
  all (`var` has only name/type/var_id/size, `ops: []`), and the only DMA descriptors with
  addresses are the Pool `PSEUDO_DMA_DIRECT2D` slots (`src_start_addr_addr_immediate` /
  **`dst_start_addr_addr_immediate`** — iter-5's recipe only mentions `src`, you need both; top
  byte `0x10` = SBUF/var, `0x20` = DRAM). The 375 SP `PSEUDO_DMATRIGGER` slots carry **no address
  fields**, so the instruction scan is complete. **This answers the iter-5 review's open question
  "whether def.json / DMA descriptors reference 0x16000-0x1c000": they do not.**
* **`dep4.py` — per-site dependency table.** For each candidate: exact-base producer, next
  overlapping writer, next readers, and the next accesses to `y`, each with the profile
  `start_ts`. This is the table to start from; it is what made E1 and E3 buildable in one pass.
* **`build.py` — the full "move a DVE `y+=` ADD to Pool" pipeline** (`WHICH=all|b|a|[slots]`):
  relocate ACT-produced addends into free SBUF, insert the ADD on Pool at
  `bisect_left(pool S6 wait values, ord(site))` (= latest position still before the `y`
  consumer), replace the DVE slot with a 1-element `MEMSET` that keeps the **entire** events block
  (so the S6 ordinal and its exact waiters survive), renumber all S3 waits in all five streams,
  then place the Pool->DVE hand-off `S3 >=` wait on the latest DVE `EVENT_SEMAPHORE` with a free
  extended wait before the deadline. **This machinery is correct and bit-exact — it is the
  *performance* that fails. Keep it if you need to move anything to Pool.**
  Useful detail: 12 of 12 `y += D*x` addends relocate cleanly (producers `ACT 22, 40, 57, 75, 95,
  113, 134, 154, 173, 194, 213, 233`; each has exactly one consumer), 10 fresh 2 KB tiles at
  `0x16900 + 2048k` plus a wrap for the last two is enough, and the resulting WAR slack goes from
  2 slots to 5-120 us.
* **`merge.py` / `merge2.py`** — pairwise row-merge at the earlier / later slot, with the hazard
  scans of §3, `FAM=dbx|cs`, `LIMIT=a,b`, `APPLY=1`.
* **`dmashift.py`** — permute the big DGE load triggers by `K` Pool compute slots (`K<0` = earlier).
* `isa.py` gotcha: `TENSOR_TENSOR_ARITH_OP`'s struct (`S3S3D3_TT_STRUCT`) has **no
  `events_extended_*` fields at all**, so a generic "copy the events block" helper must swallow
  `KeyError`. Only `CTRL_ES` and the bigger structs have the extended block.
* A `[512,2]` row-merge must keep the original `num_elem`/`step_elem` list **length** (3 for a TT
  `dst`, 2 for a `SCAN`); passing a 4-list where the field is 3 wide raises.

---

## 5. What I would try next, honestly ranked

The accounting has not changed and is now much better bounded:
`exec 501.4 = DVE compute union ~493 + ~5.9 us gap (4.2 of it the startup input DMA) + ~2.4 us
prologue/drain`, and the 493 is `384 PSUM MULTs x 0.677 + 192 SCANs x 1.270 + 36 SBUF TTs`, of
which only the 36 SBUF-only TTs (~25 us) are *structurally* movable — and M1 prices moving them at
zero. **I think this NEFF is within ~1% of the floor for instruction-level editing. With candidate #1
below now also closed, I do not believe there is a >0.5% edit left.**

1. **The 1024-element fused scan (iter-5 review's #2) is CLOSED — by PSUM capacity, not by
   scheduling. I measured both halves of it this iteration; do not spend time on it.**
   * **Calibration (timing-only probe, 10 min, no verify needed):** all 192
     `TENSOR_TENSOR_SCAN_ARITH` slots widened to `num_elem=[512,2]`, `step_elem[1]=512` on
     `src0`/`src1`/`dst` (a true contiguous 1024-element scan; output garbage, nothing else
     touched). exec 705.5 us, and the **scan duration is 2335 ns median (mean 2352) against
     1270 ns at 512 elements**. So the scan's marginal rate is `(2335-1270)/512 = 2.08 ns/element`
     (exactly iter-1's figure) and its **per-instruction fixed cost is 205 ns** — bigger than the
     TT's 126 ns. Fusing all 96 chunk-pairs would be worth `96 x 205 = 19.7 us` of DVE duration
     (~14 us after the 1-element MEMSET placeholders), plus the ~35 us of Pool carry `COPY`s it
     deletes (worth ~0 by M1). That is the largest single number left anywhere in this kernel.
   * **Why it cannot be built as a NEFF edit.** I mapped the carry chain (`xp/carry.py`): the
     16-word carry array at `0x16000+4k` is read exactly twice per state, by the chunk-0 scan and
     by the chunk-1 scan, and the two are **43 us apart** (state 0: `DVE 10 @ t=6.7` and
     `DVE 95 @ t=50.1`; the Pool `COPY` that moves `state[511]` into `0x16000+4k` sits between
     them). So the groups are `(channel tile, chunk)` with chunk minor, and the two scans to fuse
     are in **consecutive groups**, one whole group apart. Fusing them requires chunk 1's `dBu`
     (`= dtx * B_psum`) to exist before chunk 0's scan runs, i.e. PE must have produced *both*
     chunks' `B` matmuls for a state before any scan of that state — on top of the 8 x 2 KB PSUM
     banks the current schedule already keeps saturated (iter-5: the dBx and Cs MULTs alternate
     over 8 banks with the WAR release coming from DVE). There is no spare PSUM to hold a second
     chunk's worth, and M2's two measurements say moving a PSUM-dependent DVE instruction by even
     2-5 slots costs 26-83 us. Fusing is a loop restructure (interleave the two chunks, redo the
     PE/ACT order, the `y` accumulation and the output DMA layout), not an instruction edit.

2. **Accept 501.4 and stop.** Five iterations took 602.6 -> 501.4 (1.202x). Everything in the
   candidate lists of iter-1..iter-5 is now closed by device measurement:
   engine rebalancing (iter-1 #8, iter-2 R1/R3, iter-3 R2, **E1/E6 here**), instruction merging
   (iter-3 R3, iter-4 R4, iter-5 R1, **E2/E3 here**), re-addressing for bandwidth (iter-4 R3,
   iter-5 R6), Pool traffic reduction (iter-4 R1/R2, iter-5 R5), DMA trigger placement
   (iter-5 R2, **E4/E5 here**), dtype (iter-1 #6, iter-5 R4), DVE sync stalls (iter-5 §"Where the
   501.3 us goes"), PSUM access from Pool (iter-1 #8).
3. If something must be tried: the **4.21 us startup input DMA** is still the single largest DVE
   gap and the only one above 0.35 us. iter-5 R2 closed trigger reordering and queue-splitting by
   arithmetic but left open whether the measured 82 GB/s is per-queue or aggregate. That is one
   timing probe (add two more DGE loads splitting one 128 KB transfer by partition and look at
   when its semaphore fires) and the ceiling is ~1.5 us (0.3%).
4. **Do not** retry: moving any `y+=` ADD to Pool (M1), merging any DVE or Pool TT pair (M2),
   permuting the DGE triggers (M3), re-hosting the DGE queue (M4), or hoisting ACT to shorten the group boundary (M2b shows the
   boundary's cost is attached to the window, and ACT's `dA` for the first scan of a group is
   already supplied ~8 us early — ACT is *not* binding; my initial E1 diagnosis that it was is
   wrong, see M2b).

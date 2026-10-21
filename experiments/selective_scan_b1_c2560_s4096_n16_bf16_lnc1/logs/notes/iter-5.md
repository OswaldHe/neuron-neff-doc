# iter-5 — selective_scan b1 c2560 s4096 n16 bf16

**Result: no change shipped. 6604.30 us (median of 5, min 6603.68, max 6604.59) = iter-4's NEFF
byte-for-byte** (`neff_diff` vs iter-4 HEAD: "payloads are identical").

I spent the iteration on measurements instead, and they **close three of the four levers the
iter-4 review ranked, and they pin down the mechanism, the yield and the true cost of the one
lever that is still worth >1%.** Everything here is a device measurement or an exhaustive scan of
this NEFF, not an inference.

### Read this first

1. **`wall = DVE busy + 82 us`, and the 82 us is not addressable.** Proved twice by probes that
   added DVE work and moved the wall 1:1, and once by an edit that relocated 16.9 us of measured
   gap and changed the wall by 0.0 us. **Only DVE duration counts. Stop optimising schedule.**
2. **DVE duration = 0.95 ns/element-op + 203 ns/instruction.** The element-ops (5.24 M = 4 per
   (channel, state, seq)) are the algorithm and cannot be reduced, cannot be offloaded (ACT
   physically cannot do tensor x tensor; Pool cannot read PSUM), and do not respond to dtype or
   operand placement. **The only lever left in this NEFF is instruction count**, worth 203 ns
   each (126 ns net of the lost 77 ns boundary overlap).
3. **The one instruction-count lever that is not already refuted** is merging the 2560 scans
   pairwise along the seq axis (they are the only non-PSUM DVE computes, so iter-3's TT-merge
   refutation does not cover them). Ceiling **1216 merges = 153 us = 2.3%**. Section 4 has the
   schedule, the PE re-banking, the yield census and — the part that was not obvious — the fact
   that it needs a **window-private buffer allocation**, because the compiler's intermediates
   rotate only ~6 deep. Budget a full iteration for it; it touches DVE, PE and Pool.
4. If that is judged too expensive, **this kernel is at its NEFF-editing floor** (6604 us against
   a ~5.0 ms element-op floor plus ~1.0 ms of irreducible per-instruction overhead). Sections 2
   and 4b list every lever that has now been measured shut, so the next agent does not have to
   re-spend a device budget on them.

---

## 1. THE MODEL OF THIS KERNEL — four numbers explain the whole wall clock

```
wall  =  DVE busy  +  82 us          (10 head + 4 tail + ~69 of in-stream gap; the 82 held at
                                      70-83 us across all six NEFFs I measured today, and the
                                      profiler's own DVE-busy figure moves +-10 us run to run,
                                      so treat it as constant and do not chase it)
DVE busy = 0.95 ns x element-ops  +  203 ns x DVE instructions  -  77 ns x boundaries
element-ops = 4 per (channel, state, seq) = 4 x 1.31 M per lane = 5.24 M  -> 5.0 ms  (ALGORITHMIC, fixed)
DVE instructions = 8192 computes -> 1.66 ms of fixed overhead, of which 0.63 ms is given back by overlap
```

Verified two ways:

* **`wall = DVE busy + 82` is 1:1.** A timing probe that widened all 2560 scans to 1024 elements
  moved DVE busy 6523 -> 9391 (+2868) and the wall 6604 -> 9462 (+2858). A second probe that
  widened 1921 C-multiplies to `[512,2]` added +1028 us of DVE duration and +923 us of wall.
  **So any edit is worth exactly the DVE duration it removes, and nothing else.**
* **203 ns fixed / instruction, 0.95 ns per element-op.** Scan 512 elem = 1269 ns, scan 1024 elem
  = 2335 ns  ->  2.082 ns/elem for 2 ops, 203 ns fixed. TT 512 elem = 677 = 203 + 512 x 0.926.
  A `[512,2]` TT reading **two PSUM banks** = 1216 ns = 203 + 2 x 506 — the same model, so a 2-D
  pattern over PSUM is **not** penalised (this is iter-3's 1220 ns, independently confirmed).

Engine busy on the shipped NEFF: DVE 6523 (98.8%), Pool 5616 (85%), PE 2968 (45%), ACT 1903 (29%).
DVE instruction mix: 2560 SCAN (1269 ns), 2560 B-multiply (656), 2560 C-multiply (677),
306 `y+=Cs` fp32 ADD (692), 160 bf16 `dt*x` (676), 46 MEMSET (165).

## 2. CLOSED THIS ITERATION (all by device measurement — do not redo)

### 2a. The 75 us of DVE gap is NOT recoverable by moving waits. Work-conserving. (review cand. 2+3)

I built the full binding-wait attribution (producer ordinal -> supplying instruction's `end_ts`,
primary **and** extended waits, Pool S[3] **and** the 16 DGE-DMA semaphores 8..23) and then the
**general safe transformation**: for the wait `(sem s >= v)` on slot k, compute the region that
loses the guarantee (up to the next slot waiting `s >= v' >= v`), compute every RAW/WAR/WAW of
every slot in the region against the *exact* address ranges of the producers in `(prev_max, v]`,
and move the wait to a fresh bare ES immediately before the **first slot that actually needs it**
(or delete it if nothing needs it). 14 sites applied, covering **16.9 us of measured gap**, plus
3 fresh-buffer re-points. `semaphores.py` balanced, in-order simulator clean, `verify.py`
bit-identical (1.250e-01 / 1.812e-04).

**Measured: 6604.54 us, i.e. zero.** DVE busy 6523 -> 6532. The stall simply reappears at the
next hand-off — exactly what iter-4 saw for its win 1 ("gap budget conserved"). The remaining gap
is a property of the dependency graph, not of where the waits sit. **Stop mining the 75 us.**

Gap census on the shipped NEFF (341 gaps, 75.1 us), for the record:

| binding wait | gaps | us | lateness | reading |
|---|---|---|---|---|
| S[6] (DVE's own, lag-0 RAW) | 208 | 35.2 | 0 | hardware RAW turnaround 118-170 ns |
| DGE DMA, sems 8..23 | 19 | 22.9 | — | 4.5 us of it is the very first input load |
| S[3] primary (Pool) | 109 | 16.9 | -21 ns | Pool is not late |
| S[3] extended (Pool) | 15 | 17.0 | +1061 ns | the only real stalls; the 14 edits above |

(Categories overlap: a gap can have several waits in its window.) **New fact:** the DMA waits
DVE blocks on are the **680 Pool `PSEUDO_DMA_DIRECT2D`** slots (`semaphore` field 8..23,
`sem_increment` 16, `*_start_addr_addr_var_offset` + `*_elem_size` give the exact SBUF range) —
not the 4949 SP DMAs, which use semaphores 24..39 and nothing waits on them. Most of those DVE
waits are a **WAR**: DVE is about to overwrite a 2 KB buffer an output DMA is still reading.

### 2b. `EVENT_SEMAPHORE` slots are free — proved with a null edit

**Inserting 4000 extra no-op ES slots (no wait, no update) into DVE changed nothing**:
6603.70 us, DVE busy 6521.4. This is a zero-risk probe (a no-op ES is semantically invisible) and
it settles iter-4's claim definitively. Corollary: the ~750 removable ES slots in this NEFF
(349 whose wait is already implied by an earlier DVE wait on the same semaphore + 68 no-ops;
221 adjacent ES pairs that fit in one slot's primary+extended fields; 177 whose wait could fold
into the next compute's own wait field) are worth **0 us**. Do not bother.

### 2c. "PSUM src0 is 21 ns slower than PSUM src1" is a profiler artifact, not an operand effect

iter-4 recorded B-multiply (PSUM in src1) 656 ns vs C-multiply (PSUM in src0) 677 ns and listed
it as a hardware property. I swapped the two 16-byte operand blocks (bytes 16..31 <-> 32..47) of
all 2560 C-multiplies — bit-exact, `MULT` is commutative — and measured **6603.87 us, DVE busy
6521: no change**. The real split is by **predecessor slot type**: a PSUM MULT preceded by an
`EVENT_SEMAPHORE` measures 677 ns, one preceded by a compute measures 656 ns, in both NEFFs.
It is the profiler's start timestamp, not work. (So there is no 54 us here, and more generally:
**do not read 20-30 ns duration differences as hardware properties — bucket by predecessor first.**)

### 2d. PE has zero slack against DVE's PSUM-bank releases — iter-3's refutation is mechanical

For all 4994 PE matmuls that wait on S[6], `PE.start - (end of the DVE instruction whose ordinal
it waits for)` is **median 53 ns, p10 52, p90 53, 99.1% under 200 ns**. PE's 3.6 ms of idle is
entirely this wait. PE is a strict follower clocked by DVE's bank releases, and the bank rotation
is **depth 4** (`B(t,n)` -> bank `2*(n%4)`, `C(t,n)` -> bank `2*(n%4)+1`; PE's dst sequence is the
period-8 cycle `0,2,1,4,3,6,5,0,7,...`, i.e. B into the even banks and C into the odd ones).

This explains iter-3's +986 ns of critical path per merged TT exactly: merging two PSUM TTs at the
later slot makes DVE read bank `b` and the bank it needs next *in the same instruction*, so the
chain becomes `DVE read -> PE matmul (585 ns) -> DVE read` with nothing in between to hide it.
**Any edit that keeps the depth-4 rotation while making DVE consume two same-role banks closer
together inserts a PE round trip into the DVE critical path.** The fix is not to avoid merging,
it is to **re-bank** (section 4).

### 2e. Nothing can be offloaded from DVE

* ACT's only op is `ACTIVATE` (`S3D3_AC`): one source, `scale`/`bias` are an immediate or a
  **per-partition** pointer. It physically cannot do an elementwise tensor x tensor product, so
  the 5120 DVE multiplies cannot go to ACT's 4.6 ms of idle.
* Pool cannot read PSUM (iter-2) and 5120 of the 5586 DVE TTs have a PSUM operand.
* The 306 DVE `y+=Cs` ADDs -> Pool (review candidate 1) is **arithmetically a regression**: per
  (group, chunk) the y chain is 17 sequential updates; Pool already runs 35.1 us of the 40.8 us
  that DVE spends on that chunk, and 306 more ADDs at 2164 ns is +4.1 us per chunk -> 39.2 vs 40.8.
  There is no room, and iter-3 *measured* the opposite direction as a 596 ns/ADD **win**.
* DVE's rate does not depend on dtype or on where the operands live (iter-4; re-confirmed here:
  bf16-input TT 676 ns = fp32 TT 677 ns; the 1024-element scan is exactly 2x the elements).

## 3. The loop nest, as measured (iter-6 needs this)

20 channel groups (2560/128 lanes) x 8 seq chunks of 512 x 16 states. Per (group, chunk):
one bf16 `dt_x = dt*x` (shared by all 16 states), then per state n

```
DVE   BMUL  dBx = dt_x * B_psum[bank 2*(n%4)]        656 ns   (PSUM in src1)
DVE   SCAN  h   = dA * h_prev + dBx, carry at 16640+4n   1269 ns  (both operands SBUF, no PSUM)
DVE   CMUL  Cs  = C_psum[bank 2*(n%4)+1] * h         677 ns   (PSUM in src0)
Pool  ADD   y  += Cs           (2260 of them; 306 stayed on DVE; 154 are the group's first COPY)
Pool  COPY  carry(g,n) := h[511]   (1-element, 973 merged pairwise in iter-3)
ACT   ACTIVATE dA = exp(A[c,n] * dt)   (scale = per-partition pointer 229120+4n, func 7)
PE    MATMUL  broadcast B[n,s] / C[n,s] across 128 lanes into PSUM (K=1, 585 ns)
```

**The DVE stream is exactly 160 phases of 51 compute slots, one per (group, chunk), each the
same string** (`X` = `dt*x`, `B`/`S`/`C` = the triple, `A` = a DVE `y+=Cs` ADD, `M` = a MEMSET):

```
X B B A S S C  (B C S)x14  A C          <- 51 slots; phases are delimited by the X
```

and within a phase the n-th `B`, n-th `S`, n-th `C` are state n. **The producer->consumer lag is
2 compute slots in 1868 of 2224 measured (state, phase) pairs** (`S(n)` is 2 slots after `B(n)`,
`C(n)` is 2 slots after `S(n)`; the rest are 3 or 4). That is a deliberate 2-deep software
pipeline and it is why DVE's lag-0 S[6] gaps are only 208 x 170 ns instead of 5120 x 170 ns.
**Any reordering must preserve lag >= 2 or it will add ~130 ns per RAW edge** — see section 4.

* **`carry(g,n)` is ONE address per chain, reused by all 8 chunks** (80 distinct carry addresses,
  32 uses each) and updated in place by the Pool `COPY` between chunks.
* Chunk t and chunk t+1 of the same chain are **~48 DVE compute slots apart** (a whole chunk
  iteration), which is why no two scans of the same chain are ever adjacent.
* SBUF map (`address = partition * 0x40000 + offset`, every instruction here uses offset only with
  `num_active_channels=128`): rotating compute buffers at `16384..60224` (2048-byte slots, three
  families at offsets 0 / 320 / 832 within each 2 KB block — the compiler reuses them across live
  ranges, so the "buffers" genuinely overlap and you must reason about live ranges, not addresses);
  DMA staging from 114688; x/dt tiles at 223808 / 226816 / 227072 / 228096 (two staging slots per
  input, alternating per chunk); A and D at 229120+; runtime reservation 8 B at 229368 (0x37FF8).
* **Free SBUF, counted exactly (not by bounding box — that over-counts 50x):**

  | region | buffers of 2048 B | state |
  |---|---|---|
  | `0 .. 16380` | 7 | **completely unreferenced.** The only thing below 16640 is PE's 20 `LDWEIGHTS` reading `16382..16510` (the bf16 `ones` vector at 16384, backwards: `num_elem=[32,1,4]`, `step=[-1,0,-32]`). |
  | `62272 .. 113472` | 25 | iter-4's win-3 pool. **Exact reference counts are 4-7 per buffer, not the 300+ a bounding-box scan reports** — all 25 are usable with a liveness check (`62272` is the only one Pool also touches). |
  | `113472 .. 114688` | 0 (1216 B) | unreferenced, too small for a 512-fp32 tile. |

## 4. THE ONE LEVER LEFT: merge scan pairs by interleaving chunk pairs. ~153 us, 2.3%

Merging `scan(t,n)` and `scan(t+1,n)` of the **same chain** into one `[512,2]` scan is **bit-exact
by construction**: the carry does not reset across an access-pattern dim (iter-1 proved `[256,2]`
is bit-exact), and chunk t+1's true incoming carry *is* `h(t,n)[511]`, which is exactly what the
merged instruction carries across the dim boundary. Saving = 203 ns of fixed overhead per merge
minus the 77 ns boundary overlap that disappears = **126 ns x 1216 feasible merges = 153 us**, and
by section 1 that is 153 us of wall.

**Read 4a before costing this.** Sections 4/4a were written in the order I learned them, and the
last thing I learned overrides two earlier bullets: the reorder needs a **window-private
reallocation of `dBx`, `h` and `Cs`**, which means Pool *is* touched (it reads `Cs` and `h`). The
bullets below marked (SUPERSEDED) are kept only because they say what I checked.

The scans have **no PSUM operand**, so iter-3's TT-merge refutation does not apply to them. What
does apply is that the merge forces chunk t's C-multiply after the merged scan, and C(t,n) would
then be overwritten by PE. The fix is a **per-state interleave plus a re-bank**:

```
per (group, chunk-pair t,t+1) emit 16 units of 5 slots, KEEPING THE 2-DEEP PIPELINE:

   unit k (k = -2 .. 17, clipped):
      BMUL(t,k+2)   BMUL(t+1,k+2)   SCAN_merged(k)   CMUL(t,k-2)   CMUL(t+1,k-2)

   so SCAN_merged(k) reads dBx produced 2 units (10 slots) earlier, and CMUL(t,k) reads h
   produced 2 units earlier -> every RAW edge keeps lag 2, as in the unedited stream.
   Total 80 compute slots + the 2 X and 4 A slots, vs 96 + 2 X + 4 A before: 16 fewer per pair.

   PE : B(t,n) -> bank 2*(n%2)      B(t+1,n) -> bank 4+2*(n%2)
        C(t,n) -> bank 1+2*(n%2)    C(t+1,n) -> bank 5+2*(n%2)
        emitted in DVE's new consumption order; each matmul's primary wait becomes
        `S[6] >= new_ordinal(last DVE reader of its dst bank)`, which with this banking is
        the same-role, same-chunk read 2 units (10 DVE slots, ~6 us) earlier.
        Carry each matmul's preceding DMA-wait ES slots along with it (PE's structure is
        "ES holds the W24..W39 data wait, MATMUL holds the W6 bank-release wait").
   ACT:  dA(t,n), dA(t+1,n) in the new consumption order                     (reorder only)
```

* **The re-bank is what makes it safe.** Without it `B(t,n)` and `B(t+1,n)` share bank `2*(n%4)`
  and DVE's two adjacent B-multiplies would need a PE matmul between them — that is exactly the
  +986 ns/merge of 2d. Splitting the 8 banks into two disjoint depth-2 rotations (one per chunk of
  the pair) leaves PE ~2 states = ~10 DVE instructions = ~6 us of slack for its 585 ns matmul,
  which 2d shows is 100x what it needs.
* **The merged scan**: start addresses = chunk t's `dA` / `dBx` / `h`; `num_elem = [512,2]`;
  `step_elem = [1, (addr_{t+1} - addr_t)/4]` per operand (signed int16 **in elements**, +-131068 B
  — covers the whole 44 KB buffer region, iter-3 verified this on Pool and iter-1 on DVE);
  `imm0_imm_ptr` = chunk t's carry = `carry(g,n)`, unchanged. Delete chunk t+1's scan.
* **Pool's carry COPYs need no *logical* change** (only an address update if `h` is reallocated,
  which 4a says it must be): the now-redundant `carry := h(t,n)[511]` COPY still precedes the live
  `carry := h(t+1,n)[511]` COPY in Pool's in-order stream, so the final carry value is still the
  right one, and both still read addresses the merged scan writes. **S[3] numbering is untouched**
  — nothing on Pool is added, deleted or reordered, so this is a pure `src`-address edit on the
  carry COPYs and on the `y+=Cs` ADDs whose `Cs` moved. (SUPERSEDED in part: an earlier draft of
  this bullet said "Pool needs NO edit" on the assumption that `h` and `Cs` could stay where they
  are. 4a's order-preservation measurement shows they cannot.)
* **Segment the phases by data flow, not by the `X` slot.** Delimiting phases by the 160 `dt*x`
  slots leaves 20 of them non-canonical (`15/15/14`, `17/17/18` — a state leaks across the
  boundary because of the software pipeline). Instead: **phase j = scans `[16j : 16j+16]`**
  (all 160 phases then have 16 scans with 16 distinct carries), and associate the multiplies by
  data: `B[j][n]` = the latest `B` before `S[j][n]` whose `dst` == `S[j][n].src1`;
  `C[j][n]` = the earliest `C` after `S[j][n]` whose `src1` == `S[j][n].dst`. **That resolves
  2560/2560 with zero misses** and makes the pair census:

  | pairing outcome (80 pairs `(2j, 2j+1)`) | n | fix |
  |---|---|---|
  | all three operand deltas non-zero | **38** | none needed for the scan itself |
  | `dA(t,n)` == `dA(t+1,n)` (same buffer) | 24 | re-point one `ACTIVATE` dst + the scan `src0` |
  | both `dA` and `dBx` collide | 10 | re-point both |
  | `dBx` collides | 4 | re-point the B-multiply dst + the scan `src1` |
  | carry pointers do not match pairwise | 4 | skip (chain-boundary artifact) |

  so **76 of 80 pairs = 1216 merges = 153 us** is the ceiling.
* Phase order is **(group, chunk)**: phase index = 8*group + chunk. Verified, not assumed — the
  Pool carry `COPY` at Pool slot 7 reads `28988` (= `h` buffer `26944` + 2044, the last element of
  phase 0's scan 0) and writes `16640`, which is exactly the `imm0_imm_ptr` of **phase 1's**
  scan 0. So consecutive phases are consecutive seq chunks of the same 16 chains.
* **Renumber S[3], S[4], S[5], S[6]** over the permutation with iter-4's running-max rule
  (`newthr(v) = max(new_ordinal[1..v])`, i.e. "wait for everything you used to wait for"), in
  **all five streams, primary and extended** fields. For deletions (the 1280 scans) a dropped
  ordinal must map to the **surviving merged slot**, not down to the previous survivor
  (iter-3: mapping down gives `PSUM_COLLISION`, nrt 1004, in ~9 s).
* Order of work: do **one group** (4 chunk-pairs, 64 merges) first and check with
  `verify.py --runs 3` — 64 merges is only 8 us so do not try to *measure* it, measure after all 20.

### 4a. The construction that makes a big reorder safe: guarantee-preserving re-emission

Do **not** try to reason about which ES wait protects what when reordering 6400 slots. Use this
instead — it is correct by construction and it is only possible because section 2b proved ES slots
are free:

```python
# OLD pass: for every slot, the per-semaphore guarantee it enjoyed (running max over the OLD order)
need = []                      # need[i][sem] = max wait value on `sem` at or before old slot i
# NEW pass: emit the permuted stream; before each slot, if the accumulated guarantee in the NEW
# order is weaker than that slot's OLD guarantee for any semaphore, insert a bare ES carrying it.
have = defaultdict(int)
for slot in new_order:
    for sem, v in need[old_index_of(slot)].items():
        if have[sem] < v: emit(bare_ES(sem, v)); have[sem] = v
    emit(slot)
    for sem, v in waits(slot): have[sem] = max(have[sem], v)
```
Every slot then holds at least every guarantee it held before, on every semaphore (S[3..6] and
the 16 DMA ones), so no WAR/RAW/WAW can be lost — no address analysis at all. For S[6] (DVE's own
counter) the values must first be remapped by identity:
`newthr(v) = max(new_ordinal[u] for u in 1..v)`.

**The asymmetry that decides how hard this is — I ran out of time at exactly this point, so start
here.** Re-emission is only free for slots that move **later or stay**. For a slot that moves
**earlier** it is actively wrong in two ways:

* it now demands guarantees from producers that are themselves now later, which **deadlocks**
  (PE's matmul for `B(t+1,n)` moves ~50 slots earlier and its old guarantee includes
  `S[6] >= ord(a DVE slot in phase t)`; DVE's reader of that matmul is now before that slot);
* and it is the only direction where a hazard can actually be lost, because the slot now writes
  its destination earlier than whoever last read that buffer expects.

In this edit exactly two kinds of slot move earlier: `X(t+1)` (the `dt*x`) and the 16
`BMUL(t+1,n)`. So the plan is:

* **DVE and PE instructions that move later or stay**: guarantee-preserving re-emission, nothing
  else to check.
* **PE's matmuls**: drop their old `S[6]` waits entirely and recompute from the bank rule
  (`S[6] >= new_ordinal(last DVE reader of that dst bank)`); re-emit only their DMA (24..39)
  guarantees. PE's only other S[6] dependency is the 20 `LDWEIGHTS` (`W6>=1`, the `ones` MEMSET),
  which do not move.
* **`X(t+1)` and the 16 `BMUL(t+1,n)`**: carry only their own waits, and check (read/write sets,
  exact ranges, all five streams) that the buffers they write — `dt_x(t+1)` and `dBx(t+1,n)` —
  have no reader between the new and the old position. (SUPERSEDED: measured below, they *always*
  do, as do several same-phase pairs that merely get stretched apart, so plan on reallocating the
  window's intermediates rather than on a per-slot check.)
* Run the in-order simulator (iter-4's notes, extended waits included) before every pack: a
  deadlock here is a 70 s device timeout but a 1 s simulation.

**Formulate the hazard check as order preservation, not as a WAR search.** I first checked
"does anything still use `dBx(t+1,n)` at the point I move its writer to" using the *window start*
as the new position, and **0 of 12 pairs passed** — with the whole 44 KB buffer region recycling
every ~50 slots, a write moved to the front of the window always clobbers something live. That
check is wrong, not the plan: in the staggered template `BMUL(t+1,n)` lands at unit `n-2`, i.e.
~5n slots into the window, and `dBx(t+1,n)` is live for only 2 units. The right check is

```
for every window, for every pair (i,j) of DVE slots in it whose ranges overlap and at least
one of which writes:   sign(old_pos_i - old_pos_j) == sign(new_pos_i - new_pos_j)
```

which is intra-window only (slots never leave their window, so their order against everything
outside is automatically preserved) and is ~100^2 comparisons per window. Cross-engine, check only
the slots whose guarantees you dropped (the movers), by profile timestamp.

**I ran exactly this check on the 38 free pairs and it is the real cost driver — read this before
estimating the edit.**

| variant of the check | pairs clean | order violations | which roles collide |
|---|---|---|---|
| the staggered new order, all buffers as-is | **0 of 38** | 7503 | B/B 29, B/S 26, B/C 7, X/S 11, X/C 3 |
| same, pretending `dt_x(t+1)`, `dBx(t+1,*)` and `h(t+1,*)` are fresh buffers | **0 of 38** | 4323 | B/C 64, B/S 18, S/C 14, C/B 11, A/C 3 |

The reason is structural: the compiler's intermediate buffers rotate only **~6 deep**, and the
interleave stretches each phase's 51 slots across the 102-slot pair window, so `B(t,n)` ends up
**after** `C(t,m)` for `m` in `[n-4, n-1]` where before it was before them — and those two
genuinely alias, because `Cs` and `dBx` come out of the same small rotation. Freshening only the
instructions that move earlier is not enough.

**So the edit needs a window-private buffer set, not a handful of re-points:** `dBx`, `h` and `Cs`
for both chunks of the pair, allocated out of the 32 free 2 KB buffers (section 3) with a rotation
as deep as the window needs (the live ranges inside a unit are ~2 units, so ~6 of each = 18
buffers, and consecutive windows are disjoint in time so the same 18 serve all 76 pairs). The
consumers to re-point along with them: the merged scan (`src1` = `dBx`), the C-multiply
(`src1` = `h`), the Pool `y+=Cs` ADD/COPY (`src1` = `Cs`, on **Pool**), and the Pool carry COPY
(`src` = `h + 2044`, possibly one half of a dim-1 merged pair from iter-3). Budget the edit as
"reschedule + reallocate a window", not "reschedule a window".

Note also that **rejecting windows that contain a `MEMSET` costs 30 free pairs -> 12**, so handle
the MEMSETs (carry/`ones` inits; the window front is the right place when the window is the
chain's first chunk pair) rather than rejecting.

Keep each phase's `A` (DVE `y+=Cs` ADD) slots attached to the `C` whose `Cs` they read (find it by
matching src1 against the C-multiply's dst) and the `M` (MEMSET) slots at the front of the window.

### 4b. What NOT to do in that reorder (costed, so it does not have to be retried)

* **Do not emit the naive unit `B_t(n) B_{t+1}(n) S_m(n) C_t(n) C_{t+1}(n)`.** It puts the scan
  immediately after its `dBx` producer and the C-multiply immediately after its scan: 2 lag-1 RAW
  edges per unit x 1216 units x ~130 ns = **+316 us**, which more than eats the 153 us. Use the
  `k+2 / k / k-2` staggering above.
* **Do not keep the depth-4 banking.** Section 2d: `B(t,n)` and `B(t+1,n)` share bank `2*(n%4)`,
  so adjacent reads of them force `DVE read -> PE matmul -> DVE read` into the critical path.
  That is iter-3's +986 ns/merge, now explained.
* **Parking `C(t,n)` in SBUF instead** (ACT copies PSUM->SBUF so the bank frees immediately, then
  chunk t's C-multiplies can be deferred as a block and PE needs no re-banking) is *possible* but
  strictly worse: it needs 16 dedicated C buffers **plus** 16 dedicated `dBx` buffers = exactly the
  32 free 2 KB buffers that exist, **plus** 2560 new ACT instructions (+1.85 ms on ACT's 4.6 ms of
  idle, and an S[4] renumbering across every stream), **plus** re-pointing 2560 PE bank-release
  waits from S[6] to S[4] — all to save the same 1216 merges the interleave saves with no extra
  buffers at all. Only fall back to it if the PE reorder turns out to be intractable.

Second-order prize once this lands: with chunk t and t+1 interleaved, `dt_x(t)` and `dt_x(t+1)`
are adjacent too, so the 160 bf16 multiplies merge pairwise for another 10 us, and the two
C-multiplies of a unit read banks 4 KB apart, which section 1 shows is a 1216 ns merge (another
1216 x 126 = 153 us) — but that one *does* delay a PSUM release, so it needs its own re-bank
analysis and should not be bundled in.

## 5. Tooling notes (on top of iter-4's)

* **`compiler_pc` in `prof/Instruction.parquet` is the slot index** (`pc` is not). `compiler_pc`
  is NaN for the runtime's own wrapper instructions. All five streams map 1:1 this way.
* **Bounding-box address ranges give false hazards.** iter-3's merged Pool carry COPYs are
  `num_elem=[1,2]` with `step` up to 26624 B, so a naive `[lo, lo+span]` range covers 26 KB and
  intersects everything. Enumerate the dim>=1 offsets exactly when their product is small
  (`/tmp/w5/lib2.py:rngs` in this iteration did this; 24 bogus "RAW" rejects disappeared).
* The scan's read set must include `imm0_imm_ptr` (the carry) — the `*_start_addr_addr_immediate`
  fields alone miss it.
* A **timing-only probe does not have to be correct**: `profile.py` never checks outputs, so
  widening instructions to measure a rate is free (watch the address bounds: PSUM is 8 x 2048 B
  per partition, so a `+4096` dim-1 step is illegal from bank 7).
* Device cost: `verify.py` ~10 s, `profile.py --repeats 1 --parquet` ~25 s, `--repeats 5` ~80 s.
  **Process-to-process spread is ~1 us** (6603.68 .. 6604.72 for the same bytes), 10x the
  within-run spread, so nothing below ~5 us is separable.
* Keep `/tmp/kernel_iter4.neff` (or equivalent) and a clean `work/` copy; every probe here was
  "copy clean DVE0.bin in, edit, pack, run, copy clean back".

# iter-6 — selective_scan b1 c2560 s4096 n16 bf16

**Result: no change shipped.** `kernel.neff` is byte-identical to iter-5's (md5 `2d4a729e…`;
`verify.py` and `launch_rules.py` PASS, 1.250e-01 / 1.812e-04 as always).

I built the whole "2 DVE computes per state instead of 3" edit, got it **bit-exact on device**, and
measured it as a **+3.5% regression**. Then a control experiment explained why and closed the whole
family of levers: **swapping two adjacent, mutually independent DVE computes — instruction count
and every semaphore guarantee unchanged, output bit-exact — costs 572 ns per swap.** The
compiler's DVE order is a sharp local optimum and must be treated as immutable.
Everything needed to reproduce any of this is in `.autohelix/notes/iter-6-tools/`
(`build6.py`, `sim.py`, `lib.py`, `freescan.py` — see §5).

### Read this first

1. **The measurements that matter — three bit-exact builds, all slower than 6604 us.**
   All on the first **30** canonical phases (the largest prefix that verifies, see §1), all
   deterministic (3 bit-identical runs), all `max_abs_err 1.250e-01` = baseline:

   | build | wall (median of 5) | vs 6604 | DVE busy | slots |
   |---|---|---|---|---|
   | iter-5 (shipped) | 6604.4 | — | 6523 | 14139 |
   | reorder **only**, minimal waits, 0 merges | **6859.9** | **+256 us** | 6718 (+195) | 14139 |
   | reorder + 390 merges, minimal waits | **6837.4** | **+233 us** | 6687 (+164) | 13749 |
   | reorder + 390 merges, conservative wait re-emission (+1150 ES) | **6923.2** | **+319 us** | 6565 | 15917 |
   | **control: swap 2 adjacent independent computes, 420×, nothing else** | **6926.8** | **+322 us** | 6763 (+240) | 14139 |

   Clean attribution: the permutation alone costs **+195 us of DVE duration with the instruction
   count unchanged**, and the 390 merges give back only −31 us (≈ −80 ns each, not the 138 ns the
   fixed-overhead model predicts).
2. **THE GENERAL RESULT — the compiler's DVE instruction order is a sharp local optimum, and
   *any* reordering is ruinously expensive.** The control row above is the important one. It
   swaps `B(k)` with `C(k-1)` — **adjacent, mutually independent** computes (`B` writes `dBx(k)`
   and reads `dt_x`+`B_psum(k)`; `C` writes `Cs(k-1)` and reads `C_psum(k-1)`+`h(k-1)`; disjoint
   ranges, 0 alias-order violations) — 14 times per phase in 30 phases. **Same 14139 slots, same
   instruction mix, every semaphore guarantee preserved, bit-exact output, and +322 us
   (+572 ns per swap, DVE busy +240 us).**
   So the cost is not my schedule, not the merged instruction, and not the sync: **permuting two
   neighbouring DVE computes costs ~0.6 us each.** Every instruction-count lever that needs room
   made for it has to pay that, which is ~5× what the instruction it removes is worth.
   **Treat the compiler's DVE order as immutable.** (Mechanism unknown — the two candidates are
   PE's per-gate PSUM slack and iter-5 §2c's predecessor-type effect on measured duration — but
   for the purpose of editing this NEFF the mechanism does not matter.)
3. **Conservative guarantee re-emission costs ~280 ns of wall per inserted wait-carrying ES
   slot** (rows 3 vs 2 above, ~38 extra ES per phase). iter-5's "ES slots are free" was measured
   with *no-op* ES slots; an ES that carries a wait, placed earlier than the original, is not
   free. Use minimal placement (carry `pre[c]` + the compute's own wait and nothing else — the
   `MINWAIT` path in `build6.py`; bit-exact on 30 phases, costs nothing).
4. **Three hard device facts about editing sync, each of which cost a 70 s timeout to learn**
   (chronology in §2):
   * There are **3567 DVE + 5 ACT *extended* waits on `S[6]`** (`events_extended_wait_mode/_idx/
     _sem_wait_value`), plus 164 DVE extended waits on `S[3]`. **Any `S[6]` renumbering must
     rewrite them too.** Missing them is an unsatisfiable wait → silent 70 s hang with no error
     notification. This was the single cause of every hang I saw (6 of them).
   * **PE's `S[6]` waits must be remapped with the running max** `thr[v]=max(newmap[1..v])`, not
     with `newmap[v]`. They are PSUM-bank releases; exact remapping lets PE overwrite a bank whose
     reader moved later → `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR` / nrt 1004 in 9 s.
     ACT / Pool / DVE must use **exact `newmap[v]`** — prefix-max there closes a
     DVE→Pool→DVE loop through `S[3]` (my simulator catches that one).
   * **You may ADD `EVENT_SEMAPHORE` slots but you may NOT REMOVE them.** An identity rewrite of
     one phase that preserved every per-semaphore guarantee at every one of the 8192 computes
     (mechanically verified: 0 weakened) but pruned 7–10 redundant ES slots ran fine and produced
     `max_abs_err 1.927e+01` (99.56% of elements still in tolerance). Keeping every original ES
     attached to its own compute makes the same rewrite **byte-identical**. The compiler's ES slots
     are pipeline padding as well as waits. (This also kills iter-5's "≈750 removable ES slots".)
5. **There is NO free SBUF.** iter-5's table ("7 free 2 KB buffers at 0..16380, 25 at
   62272..113472") is wrong — do not reuse it. An exhaustive scan (`freescan.py`: every
   `*_start_addr_addr_immediate` with its dim≥1 `num_elem`/`step_elem` offsets enumerated, plus
   every `desc` of every `dma` block in all five `*0.json`) leaves exactly one hole above 16384:
   **113472..114688 (1216 B)**, too small for a 512×fp32 tile. 62272..113472 is fully referenced.
   **0..16383 is a trap**: referenced by nothing, a *single* 2 KB use of 12288 is bit-exact, but
   using it for a whole buffer family **hangs** (3 independent 70 s timeouts; probably
   runtime-owned descriptor/notification memory). Any plan needing a fresh tile is dead unless
   someone does a real liveness analysis and reuses a compiler tile.

---

## 1. The edit, in full (it is correct; it is just not profitable)

iter-5's model stands: `wall ≈ DVE busy + 82 us`, DVE busy = 0.95 ns/element-op + ~203 ns per
instruction, element-ops are algorithmic ⇒ only instruction count is left, at ~120 ns each.

Measured phase structure, confirmed exactly (93 of 160 phases match; `B(n)`=`dBx=dt_x*B_psum(n)`,
`S(n)`=scan, `C(n)`=`Cs=C_psum(n)*h(n)`, `A`=`y+=Cs`, `X`=`dt*x`):

```
X B0 B1 A S0 S1 C0  (B(k) C(k-1) S(k)) k=2..15  A C15          <- 51 computes/phase
B(k)@3k+1   C(m)@3m+5   S(k)@3k+3
PSUM: B_psum(n)=0x2000000+(n%4)*0x1000 ;  C_psum(n)= that +0x800   (B and C share a 4 KB bank)
SBUF per phase, 2048 B tiles, DEPTH-3 rotation per role, roles disjoint within a phase:
  dBx{16384,28672,45888} h{18432,30720,47936} Cs{22528,34816,52032} dA{24576,41792,54080}
  (addresses rotate across phases: phase p+1's dBx can be phase p's Cs — reason about live ranges)
```

**Merge `C(m)` into `B(m+3)`** as one `[512,2]` `TENSOR_TENSOR` MULT. Both members are
FP32/FP32→FP32 `MULT`, 128 channels, word_len 16, identical in every other field; B has PSUM in
`src1` and C in `src0`, so **commute the C row** (MULT is commutative):

```
src0: start=dt_x,      num_elem=[512,2,1], step=[1,(h       - dt_x)/4, 1]
src1: start=B_psum(k), num_elem=[512,2,1], step=[1,(C_psum(k-3)-B_psum(k))/4, 1]  (-2560 or +1536)
dst : start=dBx(k),    num_elem=[512,2,1], step=[1,(Cs(k-3) - dBx(k))/4, 1]
```
13 merges/phase. A 2-row **cross-bank PSUM read with step −2560 elements runs fine** (probe: 40
unmerged C-multiplies widened to `[512,2]` with that step — ran in 9.3 s, only the expected
numeric corruption). So the encoding is not a constraint.

**Why `off=3`** (this algebra is the reusable part). Put `M(k)` at compute index `f(k)=2k+α` and
`S(k)` at `g(k)=2k+β`; 2 computes/state ⇒ `α-β` odd. DVE needs **distance ≥ 3** for a hazard with
no explicit wait (distance 2 works with the lag-1 `S[6]` wait the compiler already emits; distance
1 is a 146 ns stall — iter-3 probe B). With the compiler's depth-3 `dBx` and `h`:

| | constraint | `off=4` | `off=3` |
|---|---|---|---|
| RAW dBx `M(k)→S(k)` | `β-α ≥ 3` | 3..5 | 3 |
| WAR dBx `S(k)→M(k+3)` | `6+α-β ≥ 3` | ✗ (=1) | ✓ (=3) |
| RAW h `S(m)→M(m+off)` | `2·off+α-β ≥ 3` | ✓ | ✓ |
| WAR h `M(m+off)→S(m+3)` | `6-2·off+(β-α)+... ≥ 3` | ✗ | ✓ |

`off=4` (the same-4 KB-bank pair, iter-3's candidate 2, PSUM step a clean +512) is **infeasible
with depth-3 rotations** and also aliases in the *unedited* order (11 violations) — that is why I
first chased a fresh buffer family and hit fact 3. `off=3` with `α-β=-3` satisfies all four at
exactly 3 and **needs no reallocation**:

```
X  B0  B1  A  | M(k) S(k-1) for k=2..15 |  C13  S15  C14  A  C15     <- 38 computes (was 51)
f(k)=2k, g(k)=2k+3.   PE's B_psum(k) and C_psum(k-3) gates both sit 8 computes (~10 us) ahead of
their deadline (12 in the unedited stream); PE needs 585 ns, so PE is not the problem.
```
Full pairwise order-preservation check (`sign(old_i-old_j)==sign(new_i-new_j)` for every pair of
computes in a phase whose exact byte ranges overlap with ≥1 write, including the scan's
`imm0_imm_ptr` carry): **0 violations on all 93 phases**; minimum new distance per aliasing class
≥ 3 except `('S','S',RAW)=2` (safe) and `('C','A',RAW)=1` (carries a re-emitted lag-0 wait).

**Device results, with everything in the "read this first" list fixed:** 1, 2, 12, 24 and
**30 phases bit-exact**; 35 fails `max_abs 1.462e+01`; 93 fails `max_abs 2.379e+01`
(deterministic — 3 bit-identical runs). So there is **one hazard I never found, in canonical
phases 31..35** (`ranges[30..34]`; `LIMIT=30` is the largest passing prefix). My alias check is
**intra-phase only** — the obvious next suspect is a cross-range pair; run the same check over a
window spanning the ±51 computes outside `[lo,hi]`. Given §1's measurements, finding it buys
nothing.

## 2. Chronology (so nothing is retried)

| # | edit | sim | device | lesson |
|---|---|---|---|---|
| 1–2 | `off=4` reorder + merge + dBx→{2048,…}, prefix-max `S[6]` everywhere | deadlock (Pool↔DVE via S[3]) | — | prefix-max is wrong for Pool/ACT/DVE |
| 3 | same, exact `newmap`, B-slot `S[3]` dropped | clean | hang | — |
| 4 | **dBx realloc only**, 2560 pairs, no reorder/merge | n/a | hang | a pure address change cannot deadlock ⇒ low SBUF unusable |
| 5 | one 2 KB use of 12288 | n/a | **PASS bit-exact** | the region decodes fine in isolation |
| 6–7 | dBx realloc with {6144,…}; and canonical phases only | n/a | hang | not address-specific, not boundary-specific |
| 8 | `off=3` reorder+merge, no realloc, 93 phases | clean | hang | — |
| 9–10 | identity rewrite with ES pruned (wait on bare ES / on the compute) | n/a | ran, `1.927e+01` | fact 2 |
| 11 | identity rewrite keeping all ES | — | **byte-identical** | machinery validated |
| 12–13 | merge only (13 merges, then **1** merge), 1 phase | clean | hang | one merge is enough |
| 14 | iter-3's minimal recipe, `off=3`, 1 phase | **deadlock (DVE ES `S[6]>=9`, 7 updaters done)** | hang | forward `newmap` needs the DVE prefix clamp |
| 15 | widen 40 C-multiplies to `[512,2]`, PSUM step −2560 | n/a | ran (expected numeric error) | the merged encoding is legal |
| 16 | **reorder only** (C's kept as separate slots), 1 phase | clean | **PSUM_COLLISION 9 s** | PE needs prefix-max |
| 17 | reorder only + PE prefix-max | clean | **PASS bit-exact** | the schedule is sound |
| 18 | reorder + merge + PE prefix-max | clean | hang | — |
| 19 | **drop 13 `S[6]` increments only** (instructions untouched, no reorder, no merge), both `newmap` directions | clean | hang | isolates the hang to the `S[6]` renumbering ⇒ found the **extended waits** |
| 20 | everything + extended-wait remap | clean | **PASS bit-exact** (1, 2, 12, 24, 30 phases) | — |
| 21 | 30 phases, conservative re-emission, `profile.py --repeats 5` | | **6923.17 us (+4.8%)** | wait-carrying ES slots cost ~280 ns each |
| 22 | 35 / 93 phases | clean | `1.462e+01` / `2.379e+01`, deterministic | one hazard left, in canonical phases 31..35 (my alias check is intra-phase only) |
| 23 | 30 phases, `MINWAIT=1` (no extra ES at all) | clean | **PASS bit-exact**, **6837.40 us (+3.5%)**, DVE 6687 | the sync was not the main cost |
| 24 | 30 phases, `MINWAIT=1 REORDERONLY=1` (0 merges, same slot count) | clean | **PASS bit-exact**, **6859.86 us (+3.9%)**, DVE 6718 | **the permutation alone costs +195 us; the merges give back only 31 us** |
| 25 | **`SWAPONLY=1 MINWAIT=1 LIMIT=30`** — swap `B(k)`↔`C(k-1)` (adjacent, independent), 420 swaps, nothing else | clean | **PASS bit-exact**, **6926.81 us (+4.9%)**, DVE 6763 | **+572 ns per adjacent swap ⇒ the compiler's DVE order is immutable** |

## 3. The verdict on the remaining lever

**Closed by measurement, in both variants.** Per phase the edit removes 13 instructions
(13 × 80 ns measured = 1.0 us, not the 1.56 us the model predicted) and the permutation needed to
make room for them costs **6.5 us**. Even the zero-extra-sync version is +3.5%, and the
permutation alone is +3.9%. Scaling to 93 phases makes it worse, not better.

**And the general closure (read-this-first #2): I ran the falsifier.** Swapping two adjacent,
mutually independent DVE computes — nothing else touched, instruction count identical, every
guarantee preserved, output bit-exact — costs **572 ns per swap**. So it is not my schedule that
is bad; **the compiler's DVE order cannot be perturbed at all without paying more than any
instruction you could remove is worth.** That closes every reorder-based lever in this kernel,
including iter-5 §4's scan-pair interleave (which permutes ~6400 slots) independently of the
SBUF problem that already killed it.

One minor sub-question is still open, and it no longer matters much:
* **Why is a merged `[512,2]` TT worth only 80 ns instead of 138?** iter-5 measured a 2-row PSUM
  TT at 1216 ns = 203 + 2×506, but mine reads two 2 KB regions in *different* 4 KB banks with a
  −2560/+1536-element dim-1 step. Falsifier: the timing probe in §2 row 15 (widen 40 unmerged
  C-multiplies) but with `profile.py --parquet` instead of `verify.py`, and read the widened
  slots' `duration_ns` out of `prof/Instruction.parquet`. Only interesting if someone finds a
  merge that needs **no** reordering — and §2 row 12/13 plus iter-3 say every such pair is
  already merged or refuted.

## 4. Updated closed list

Everything in iter-3/4/5's closed lists, plus:
* **Fresh SBUF buffers** (fact 3) — and therefore iter-5 §4's scan-pair interleave as written,
  which needs a window-private buffer set.
* **Removing ES slots** (fact 2), and iter-5's "750 removable ES slots are worth 0 us" as an edit.
* **`off=4` merges** (`C(m)→B(m+4)`, the same-4 KB-bank pair): infeasible with depth-3 rotations.
* **`off=3` reorder + merge**, in *all* variants: correct and bit-exact, but measured
  **+3.5% with zero extra sync** and **+3.9% for the permutation alone with zero merges**.
  The 2-computes-per-state schedule is dead unless §3's first falsifier shows the permutation
  cost is avoidable.
* **Conservative guarantee re-emission** as a general tool: ~280 ns of wall per inserted
  wait-carrying ES. Fine for a handful of sites, fatal at ~40 per phase. Use `MINWAIT`-style
  minimal placement (carry `pre[c]` + the compute's own wait and nothing else) — it was
  bit-exact here on 30 phases and costs nothing.
* **Reordering DVE at all.** 572 ns per adjacent swap of two independent computes (read-this-first
  #2). This closes, in one measurement, every lever that needs to move DVE instructions:
  iter-5 §4's scan-pair interleave, the `off=3`/`off=4` merge schedules, any "make room for a
  wider instruction" plan, and any attempt to reduce the 75 us of DVE gap by reordering work.
* **Reducing DVE instruction count by merging**, as a lever in general: the realised value of a
  merge measures **~80 ns**, and every remaining merge candidate needs a permutation that costs
  ~0.6 us per slot moved. iter-5's "203 ns per instruction, 126 ns net" is an upper bound the
  required rescheduling more than eats.

**Conclusion: this kernel is at its NEFF-editing floor at ~6604 us** (−16.6% from the 7920 us
baseline). Four iterations of levers (engine offload, sync placement, instruction merging,
rescheduling) are now all measured shut, and the two structural resources an edit would need —
spare SBUF and freedom to reorder DVE — are both measured to be unavailable. A future iteration
should budget very little device time before concluding the same; if it wants one more shot, the
only untried direction is **reducing the 5.24 M DVE element-ops**, which means changing what the
kernel computes (bf16 operands for the scan/multiplies) — and the error budget is already at 83%
of the limit, so that is almost certainly out.

## 5. Tooling left behind — `.autohelix/notes/iter-6-tools/`

* **`lib.py`** — `decs(engine, w=workdir)` → `(raw bytes, [isa.Instruction])` for
  `work/payload/sg00/<Engine>0.bin`.
* **`sim.py`** — the in-order 5-engine semaphore simulator. `python sim.py <path/to/sg00>`.
  Catches real deadlocks in 0.3 s instead of a 70 s device timeout; it caught 3 of my 6 hangs.
  **It cannot see unsatisfiable *extended* waits unless you add them** (it only reads the primary
  wait) — add that first. Increments modelled: `SEM_INC_*` +1, `SEM_ADD_IMM_*` +value,
  `PSEUDO_DMA_DIRECT2D` += `sem_increment`, `PSEUDO_DMATRIGGER` += `num_queues` of the queue in
  `dma_queue_name` (look the block up by `(engine, queue, count_union_block_id)` in
  `<Engine>0.json["dma"]`; `num_queues` comes from `def.json["dma_queue"]` — using 1 makes the
  *unedited* NEFF deadlock in the sim, which is how you know you got it wrong).
  Sanity check: the unedited NEFF completes in 42026 steps.
* **`freescan.py`** — exhaustive SBUF occupancy scan (instructions + DMA descriptors) → free
  2048 B windows. Run it before believing any "free buffer" claim.
* **`build6.py`** — the whole edit, validated to be **byte-identical in identity mode**
  (`IDENT=1 LIMIT=1` then `cmp` the DVE stream — do this after every change to it).
  Env knobs: `LIMIT=n` (first n canonical phases), `OFF=3`, `KRANGE=15`,
  `IDENT=1` / `MERGEONLY=1` / `REORDERONLY=1` / `NOWIDEN=1` (the bisection modes above).
  Its four load-bearing rules, all of which I got wrong at least once:
  1. **Grow the DVE stream freely** (no PC-relative fields; nothing outside references DVE slot
     indices) — build the new stream as a list of 64-byte slots and let `neff.py pack` fix sizes.
  2. **Keep each compute's own wait field, and carry the ES slots that preceded it** (`pre[c]`)
     along with it. Only *add* ES slots (fact 2).
  3. **`S[6]` renumbering**: `oldord` = index among DVE slots with `events_update_mode != 0 and
     events_update_idx == 6` (8022; `SEM_INC_COMPLETE = 19` — *not* 1/2, that typo cost a run).
     `newmap[o] = newo[slot]`, `newmap[ord(absorbed)] = newo[host]`. Rewrite **primary *and*
     extended** `S[6]` waits in **all five** streams (DVE 2262+3567, ACT 2045+5, PE 4994,
     Pool 4161), using `thr` for PE and `newmap` elsewhere (fact 2), then **clamp every DVE `S[6]`
     wait (primary and extended) to the count of `S[6]` updaters strictly before that slot** —
     forward mapping otherwise makes DVE wait on its own future.
  4. `semaphores.py` and `launch_rules.py` catch **none** of this; they stayed green through every
     one of the 6 hangs and both numeric failures.
* Device cost, unchanged: `verify.py` ~10 s, PSUM fault ~9 s, hang ~70 s,
  `profile.py --repeats 5` ~80 s. Process-to-process spread ~1 us.

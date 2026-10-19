# iter-7 — 47.70 → 44.22 us (1.079x on iter-6; 4.93x vs the 218.17 us baseline)

Start-of-iteration re-measure of the iter-6 file: **47.701 us** (5 repeats, spread 9.3% because one
capture hit the slow bimodal mode; the four fast captures were 47.52–47.78).
Shipped file sha256 `f22db5a73e4a5f285eee337291be471ba41b8ef9d055814f2e2491b56ba4b1a9`.
Shipped medians, three separate runs: **44.220** (of 5, spread 1.12%, min 43.950), **44.246**
(of 3, spread 0.49%), **44.474** (of 5, spread 2.07%, min 44.109) — mean ≈ 44.3, and the slow
bimodal mode never appeared, unlike on the parent.
`verify.py` gives **exactly the baseline error triple** (2.999e-03 / 1.928e-03 / 6.033e-04) after
every shipped edit, so all arithmetic is bit-identical; `verify.py --runs 4` reports all 4 runs
bit-identical. `semaphores.py` clean, `launch_rules.py` passes (1 subgraph, num_tpb 1, 11 inputs,
3 output heads, LNC=1, one rank). Members changed vs the original baseline: the same 6 as iter-5/6
(`Activation0.bin` 119→113 slots, `DVE0.bin`, `PE0.bin`, `Pool0.bin` 75→75, `SP0.bin`, `SP0.json`).

| # | edit | median of 5 |
|---|---|---|
| — | iter-6 (HEAD) | **47.701** |
| p1 | **the 8 in_proj tile loads merged into ONE 1024-descriptor DGE DMA, and the 8 tbl-2 tile loads into another** (−6 DMAs off Scalar, −8 off Pool, +8 relay ES on Pool) | 48.320 (+0.6, see §2) |
| p2 | **SP transposes 34+36 merged, 46+50 merged** (4 DGE transfers → 2) | 47.109 |
| p3 | **the T0 `ACT_TABLE_LOAD` hoisted above the bare `ES W[13]>=16` in front of it** | 46.692 |
| p5 | p2's second merge re-gated on `W[77]>=16` (fixes a real race, §4) | 47.167 |
| p6 | **p2's second merge reverted entirely** (SP 46/50 separate again) — correct *and* faster | **46.247** |
| p9 | **Pool's 8-CAST packing run hoisted above three DRAM spill DMAs it does not depend on** (+ p4b) | **44.686** |
| p10 | **the merged tbl-2 DMA split into two 4-tile halves so the tiles DVE needs land ~1.2 us sooner** | **44.220 (shipped)** |
| p4a | merged tbl-2 issued before merged in_proj | 47.949 → **+1.1 REGRESSION** (§5) |
| p4b | Act CASTs 41,42,43 hoisted above the T3 table load | neutral alone (46.816); kept, it is free |
| p8 | tbl-2 split **and** tbl-2-A issued first | 47.175 (+0.9, the §5 rule again) |
| p11 | SP transposes 18+27 merged (one 2-tile transfer, new `W[6]>=26` gate) | 44.562 (+0.34, §5) |
| p12 | Pool[58,59] hoisted above the spill DMAs | 44.850 (+0.6) |
| p13 | three DRAM spill DMAs deferred into Pool's 12.4 us idle window | 44.391 (neutral, §5) |

---

## 0. THE HEADLINE: iter-6 §2's merge rule was WRONG, and an n=8 merge is legal

iter-6 concluded "only the single TRAILING queue can be absorbed, and the surviving instruction
must be the EARLIER one", after `p8-m4` (n=4) failed with a characteristic error triple. **That is
not the rule.** This iteration merged **8 tiles into one DGE DMA twice over**, bit-identical both
times, on the first try:

```
in_proj: 8 tiles of 128x256 B, src tbl0 off 256k -> dst 228864-256k, k=0..7
  base = Activation slot 1 (sem 80, src 0 -> dst 228864); drop Act 2, Pool 2/4/8/10/11/12
  src_num_elem [128,8]  src_step_elem [2048, 256]    src_elem_size 256
  dst_num_elem [128,8]  dst_step_elem [262144, -256] dst_elem_size 256

tbl2:    8 tiles of 128x8 B bf16->fp32, src tbl2 off 1024k -> dst 225280-768k, k=0..7
  base = Activation slot 7 (sem 86, src 0 -> dst 225280); drop Act 8/3/4/5/6, Pool 5/6
  src_num_elem [128,8]  src_step_elem [8, 1024]      src_elem_size 8
  dst_num_elem [128,8]  dst_step_elem [262144, -768] dst_elem_size 16
```

The real rule is just **get the descriptor pattern right**: `num_elem[0]`/`step_elem[0]` is the
per-partition axis (src stride within the tile, dst = 262144 = next partition), `num_elem[1]`/
`step_elem[1]` is the tile axis, steps are in BYTES and signed. `Pool` slot 1 (the conv_weight
load, `[128,8] / [16,-2048] -> [262144,768]`) is an existing compiler-generated proof that n=8
works; I should have copied it from the start. iter-6's n=4 failure was almost certainly bad
arithmetic in its own script, not a hardware limit — **do not trust the "trailing queue" story.**

### The cost model that makes merging worth it (measured on this NEFF)

| | |
|---|---|
| DGE DMA **issue** | ≈ **0.6 us fixed per instruction** + **0.71 ns per descriptor** |
| | (Pool: 128 desc = 0.687 us, 1024 desc = 1.323 us → (1.323−0.687)/896 = 0.71 ns) |
| DGE DMA **completion report** | ≈ **2.8 us fixed** + ~0.6 us per 1024 descriptors, *independent of bytes* |
| compute `SEM_ADD_IMM_COMPLETE` → waiter on another engine | 0.03–0.2 us |
| `ACT_TABLE_LOAD` | 1.283 us |

So 8 DMAs of 128 descriptors cost 8×0.6 + 0.73 = **5.5 us of engine issue time**; merged into one
of 1024 they cost 0.6 + 0.73 = **1.3 us**. That is the whole 4.2 us. **The fixed 0.6 us per
instruction is the thing to attack — descriptor count is nearly free.**

### Semaphore bookkeeping for a merge (the part that takes the time)

Every dropped DMA was a `+16` on its queue semaphore, and later waits use `>=32/48/64`. Replace
each dropped `+16` with an `EVENT_SEMAPHORE W[survivor]>=16 → U[old_sem]+=16` — `events_semaphore_value`
is shared between the wait threshold and the add amount, and both are 16, so one slot does it.
`Pool` slots 13–20 are the compiler's own copies of exactly this idiom (they relay `s80..s87` →
`s20,s21,s22,s23,s8,s9,s10,s11`), so clone one of those as the template. Put the new relays
**after** the engine's remaining real DMAs, otherwise the engine stalls on them and delays its own
transfers. I converted the 8 freed Pool DMA slots into the 8 relays and moved Pool's three
surviving prologue DMAs (sems 13, 9, 16) to the front.

## 1. THE SECOND HEADLINE: compute sitting behind DMAs it does not depend on (−1.56 us, p9)

`Pool` slots 50–57 are eight CASTs that pack four bf16 columns into the 8 contiguous bytes that
`SP` slot 18 (and 34) transpose. The last of them signals `s59`, which gates `SP[16] → SP[18] →
(2.2 us completion hop) → Activation`'s four scale ops. In the compiler's order the run sat
**behind three DRAM spill DMAs (Pool 45/47/49, ~2 us of issue)** that it has no dependency on
(they read SBUF 21760/21248/22272 and write DRAM table 8; the CASTs read 20992/21504/218880/…).
Hoisting the whole block above them deleted Activation's entire 2.155 us wait: **−1.56 us.**

> **The reusable move: for every Scalar/DVE gap, find the semaphore, find its producer, and then
> ask what is in front of the producer *in its own engine's program*. Twice this iteration the
> answer was "DGE DMA issue that nothing in the chain needs". `gaps.py` + `waiters.py` + a decode
> of the producing engine's 20 slots around it is the whole recipe.**

Why the earlier attempts at the same gap failed: hoisting the *Activation* CASTs that feed the Pool
run (`p4b`) was neutral, because Pool could not reach its CAST run until those three DMAs had
issued. The gate was Pool's program position, not the data.

## 2. Why p1 alone was a 0.6 us REGRESSION even though it freed 4.4 us of Scalar and 5.4 us of Pool

The prologue stopped being the critical path and the binder jumped to `SP`'s serial transposes:
`Activation` CAST → `SP` DGE transpose → ~2.8 us completion hop → `Activation`. With four SP
transfers issued back to back (37.9/38.65/39.75/40.53) **their completion reports all landed
together at ~43.7** — the hop stretched from ~1.4 to ~3–5 us. `p2` (merging them pairwise) got
1.2 us of that back. **Contention between in-flight DGE transfers inflates the completion hop; the
number of transfers in flight matters more than their size.** This is also the mechanism behind
iter-6 §10's bimodality.

Lesson for the next iteration: **a win that removes engine busy time can show up as a loss until
the next binder is also fixed. Measure the new profile and follow the binder instead of reverting.**
p1 → p2 → p3 → p9 → p10 was a 4-step chain: p1 alone +0.6, the chain −3.5.

## 3. THE THIRD FINDING: the front is a 2-chain join and in_proj must be issued FIRST

Prologue dependency structure (times from the shipped build):

```
Pool 1  conv_weight DMA (1024 desc) 13.0->14.3 --(3.4 us hop)--> s13 17.4
Act  1  in_proj      DMA (1024 desc) 12.3->13.4 --(3.5 us)--> s80 16.9
                                       -> Pool relays -> s8/s10/s15/s17/s18/s19
                                       -> PE 2..14 LDWEIGHTS+MATMUL (0.93 us total) -> s5
Act  2  tbl2-A (k0..3) 13.4->14.3 --> s86 17.4 -> Pool relay s11/s12 -> DVE 6/7
Act  3  tbl2-B (k4..7) 14.3->15.3 --> s82 18.2 -> Pool relay s22/s23 -> DVE 19/20
Pool 2  tbl1 2-byte DMA --> s9 17.7 -> PE 1
Act  4  T0 ACT_TABLE_LOAD (hoisted above the ES, p3)
Act  5  ES W[13]>=16   <- conv_weight: Act's first ACTIVATE reads its scale from a pointer
                          (scale_value_imm_ptr = 223232/222464/221696, which are conv_weight
                          dst tiles) so this barrier is REAL, do not delete it
Act  6+ ACTIVATEs -> DVE 8..24 -> Scalar
```

* **`s80` (in_proj) is the binding completion.** Delaying in_proj by ~1 us to issue tbl-2 first
  costs ~1 us of wall clock (p4a: +1.1; p8: +0.9). It feeds PE's LDWEIGHTS chain, which feeds
  DVE 12–18, which feeds Scalar. Keep in_proj as Activation slot 1, always.
* **Splitting the merged tbl-2 in two halves is worth 0.47 us** (p10) because `DVE 6/7` only need
  tiles k=2,3: a 512-descriptor half completes ~1.2 us before a 1024-descriptor whole. The split
  reuses the now-free act queue `sem 82` and repoints the four relays that serve tiles k=4..7
  (`Pool 15/16/17/18`) from `W[86]` to `W[82]`. **This is the general shape: merge for issue time,
  then split along the consumer boundary for latency.** A further split into 2+2+4 was computed to
  be a wash and not tried.
* The `T0 ACT_TABLE_LOAD` hoist (p3, −0.42 us) is free money: a bare `EVENT_SEMAPHORE` has no data
  effect, so a table load may always move above one. **Look for table loads that sit immediately
  after a blocking barrier** — the load then runs inside the stall instead of after it.

## 4. THE FOURTH FINDING: a merge can silently drop a *program-order* dependency

`SP`'s four 2-byte transposes read 218624/217856/217088/216320, written by `Act[42]/[43]/[53]/[57]`.
Only two of those writes are covered by a semaphore the DMA itself waits on; the other two were
enforced by **SP's instruction order** (`SP[47]/[48]` wait `W[77]>=16` = Act[57], and they sit
*after* `SP[46]`). Merging 46+50 into slot 46 therefore made the transfer read 216320 ~0.2 us
before `Act[57]` wrote it on the shipped schedule. `verify.py` passed 4/4 bit-identical anyway —
exactly the silent race the review warns about.

> **Before merging two DGE transfers, list every source tile, find its writer, and check that the
> guarantee is a semaphore the *merged* instruction waits on — not a wait that lives in a slot the
> merge moved the read in front of.** An extra wait is usually not expressible: `events_wait` and
> `events_extra_wait` share `events_semaphore_value`, so you can only add a second condition whose
> threshold happens to equal the first (16 vs 21/32 here).

Two fixes measured: re-gating on `W[77]>=16` costs 0.48 us (p5), and **reverting that one merge
costs nothing at all** (p6: 46.247, better than both) because `SP[46]`'s completion was already
the gate and the second transfer's completion gates only a tail `ES`. Shipped: merge 34+36 only
(which *is* properly ordered — `SP[24]` waits s74=Act[42] and `SP[25]/[26]` wait s75=Act[43],
both before slot 34).

## 5. Refuted / closed this iteration (do not redo)

* **Issuing the tbl-2 load before in_proj on Activation: +0.9 to +1.1 us, twice** (p4a, p8). §3.
* **Merging SP[18]+SP[27] into one 2-tile transfer: +0.34 us** (p11). The recipe is sound and
  bit-identical (`src_step [262144, 25216]`, `dst_step [8, -2048]`, and `SP[14]` — a bare no-op ES —
  turned into `W[6]>=26` to order it against `DVE[53..56]`, which write 241024..241031 and are
  followed by `DVE[57]`, the first s6 increment after them). It loses because the merged
  completion is no earlier than SP[18]'s was while now gating both consumers.
* **Hoisting `Pool[58,59]` above the spill DMAs: +0.6 us** (p12), even though `Pool[59]` supplies
  `s18`'s third `+16` which `DVE[89]` waits for at `>=48`. So `DVE[89]` is not the tail's binder.
* **Deferring the three DRAM spill DMAs (sems 13/14/15) out of the 33–34 us window into Pool's
  12.4 us idle window at 41–53: 44.391, neutral/slightly worse** (p13). This was the "fewer
  transfers in flight ⇒ shorter completion hops" (§2) bet applied to a window where the SP
  transposes' reports are contended, and it did not pay — the hops in the shipped file are
  already near the ~2.8 us floor. Bit-identical, and it is safe (each spill source has exactly
  one writer, all retired by ~40 us; nothing in the kernel waits on the spill *data*).
* **`DVE[113]` cannot be hoisted above DVE's 8-op shuffle block** (candidate 2 below, checked
  statically, do not bother): its wait is `W[6]>=52` and `DVE[112]` is itself the 52nd s6
  increment, i.e. the wait *is* the in-order dependency on the shuffles, whose output it reads.
* **Hoisting `Act` CASTs 41/42/43 above the T3 table load: neutral** (p4b, 46.816 vs 46.692, and
  again inside p9). They are bf16→bf16 so §iter-6-3 allows it; it is simply not the gate. Kept in
  the shipped file since it is free and may become useful if Pool's run moves again.
* Everything iter-6 §7 item 6 listed as dead is still dead, **except** the tile merges, which are
  now the single biggest lever of this iteration — see §0.

## 6. Where the time is now (44.22 us; body 12.3 → 56.9, metric ≈ end − 14.7)

`gaps.py prof 0.4` on the shipped file (end 58.98):

| engine | sum of durations | union busy | % of its span |
|---|---|---|---|
| Scalar (Activation) | 45.7 | **40.5** | **80%** |
| Vector (DVE) | 30.3 | 25.9 | 48% |
| GpSimd (Pool) | 22.4 | 22.1 | 43% |
| Tensor (PE) | 21.0 | 16.6 | 33% |
| Sync (SP) | 11.3 | 11.3 | 22% |

Scalar idle is down from 11.5 us to **6.2 us in 8 gaps** — the prologue waits are gone:

| Scalar gap | us | gate |
|---|---|---|
| 19.38 → 20.55 (pc10) | 1.17 | `W[6]>=11` = DVE 19 (needs tbl-2-B's s22 relay + DVE 8..18 in-order) |
| 21.78 → 22.31 (pc17) | 0.53 | `W[6]>=18` = DVE 32 |
| 24.73 → 25.73 (pc22) | 1.00 | `W[6]>=24` = DVE 48 |
| 38.65 → 39.14 (pc66) | 0.49 | SP 34's merged transpose completion (s61) |
| 40.34 → 40.78 (pc73) | 0.44 | SP 46's completion (s71) |
| **44.90 → 46.20 (pc83)** | **1.30** | `W[5]>=22` = PE 52; PE 51 waits `W[6]>=53` = DVE 113, which is behind DVE's 8-op shuffle block (pc105–112) |
| 49.31 / 52.43 (pc98, pc107) | 1.24 | the 4-head tail recurrence (DVE → CAST → PE → Scalar per head) |

Pool is idle 41.0 → 53.5 (12.4 us) and 27.7 → 30.4 (2.7 us, waiting for `Act[37]`'s `s72`).
PE is idle 35.8 → 42.2 (6.3 us) before `pc44 LDWEIGHTS`, which reads the four scale ops' output.

### Ranked candidates for iter-8

1. **Keep applying §1's lens to every remaining gap.** It paid 1.56 us once and the search is
   static (no device time): for each gap, `waiters.py` the semaphore, then decode ±20 slots of the
   producing engine and look for DGE DMA issue or unrelated compute in front of the producer.
   Useful facts gathered for this: Pool's eight DRAM spill DMAs (`sem 13/14/15/19/20/21/22/23`,
   dst table 8 offsets 0..7168) are **pure sinks — nothing in the kernel waits on their data**,
   only queue-reuse guards and Pool's final `NOP W[8]>=64`; each source (21760/21248/22272/22784/
   20224/20736/23296/23808) has exactly one writer, all retired by ~40 us. Deferring three of them
   was neutral (p13) and only the remaining five sit at 38–41, so this particular well is dry;
   the *method* is what to reuse elsewhere.
2. ~~**The 1.30 us `pc83` gap**: hoist `DVE 113` above DVE's 8-op shuffle block~~ — **checked and
   dead, see §5.** The gap is `Scalar (…44.90) → DVE pc105..112 (0.9 us of shuffles) → DVE 113 →
   PE 51/52 → Scalar pc83 (46.20)`, every edge a real dependency, and the last `ACT_TABLE_LOAD`
   is at 42.1–43.4 (*before* the hole) so the p3 trick does not apply either. The only way in is
   to remove DVE work from the chain or find Scalar work that does not depend on `pc83`.
3. **Split the conv_weight DMA (Pool slot 1, 1024 desc, `s13` at 17.4) along its consumer boundary**,
   the way p10 split tbl-2. Activation's first three ACTIVATEs read their scale from tiles k=5,4,3
   (`ptr 223232/222464/221696`); a 4-tile half completes ~1.2 us sooner. The later ACTIVATEs need
   k=0,1,2 and each has its own `ES W[13]>=16` barrier, so the second half needs its own pool queue
   semaphore and those barriers repointed. Now that the front waits are closed this is probably
   worth little — check `gaps.py` first and skip if Scalar's first gap is still at pc10.
4. **The 5-table-load schedule (2.57 us)** — iter-6 §7 item 2's full scoping still applies and is
   still the biggest single arithmetic-free saving. Scalar is 80% busy with only 6.2 us of idle, so
   deleting 2.57 us of its work should now convert almost 1:1. The blocker is unchanged: slots
   52/53 carry `SEM_INC_COMPLETE[4]` so moving them renumbers Activation's instruction counter.
5. **Re-audit for races of the §4 kind in what iter-1..7 have already shipped.** `SP[34]`'s merge is
   clean, but the general pattern "the compiler relies on this engine's instruction order, not on a
   semaphore" is everywhere in `SP0.bin`, and every reorder this loop makes is exposed to it.

## 7. Reproducing

Scripts take the unpacked root as `argv[1]`; `an/p1.py`, `an/p8.py`, `an/p10.py`, `an/p11.py` also
take a *second* root holding the **iter-6** tree (they copy original slot bytes out of it).
From a fresh unpack of the iter-6 NEFF into `work/`, with a pristine copy in `work.bak/`:

```
an/p1.py  work            # the two n=8 tile merges + 8 Pool relays        (+0.6 alone)
an/p2.py  work            # SP 34+36 and 46+50 merged                      (-1.2)
an/p3.py  work            # swap Activation slots 3 and 4 (table load up)  (-0.42)
an/p6.py  work work.bak   # restore SP 46 and 50 as separate DMAs (§4)     (-0.45)
an/p9.py  work            # hoist Pool's CAST run 50..57 above DMAs 45/47/49 (-1.56)
an/p4.py  work b          # hoist Act CASTs 41,42,43 (free, kept)
an/p10.py work work.bak   # split merged tbl-2 into sem 86 (k0-3) + sem 82 (k4-7) (-0.47)
```
then `pack` → `semaphores.py` → `verify.py` → `profile.py --repeats 5`.
Refuted: `p4.py work a` (tbl-2 first), `p8.py` (split + tbl-2 first), `p5.py` (re-gate the racy
merge), `p11.py` (SP 18+27), `p12.py` (Pool 58/59), `p13.py` (defer the spill DMAs).

**The shipped reorders were audited for §4-style races and are clean.** For `p9` (the hoisted Pool
CAST run): sources 20992/21504 are covered by `Pool[50]`'s `W[72]` and `Pool[52]`'s `W[73]`
(Act 37/41, which are after their writers Act 31/33); 218880/218112/217344/216576 are covered by
those same waits plus in-block program order; 18176 is written only by `DVE[16]` (~20 us) and
again by `Act[85]/[86]` (~47 us), and 16384 only by `DVE[17]` (~20) and `Act[75]` (~42), so the
block's new position (~31) reads the same values it read at ~33. The three spill DMAs it moved
above read 21760/21248/22272, which the block never writes. For `p10` (the tbl-2 split) every
consumer's tile is in the half whose semaphore gates it: k=0→s10/DVE[23], k=1→s11/DVE[24],
k=2→s11/DVE[6], k=3→s12/DVE[7] on sem 86; k=4→s22/DVE[19], k=5→s23/DVE[20], k=6→s8/DVE[21],
k=7→s9/DVE[22] on sem 82.

Analysis helpers rewritten this iteration (all in `an/`, all read an unpacked root):
`d.py ROOT ENG [lo hi]` compact per-engine decode with waits/updates;
`full.py ROOT ENG a,b,c` every field of chosen slots, differences starred;
`dma.py ROOT ENG` **every DGE DMA as src table/offset/num/step/size → dst … — this one found both
n=8 merges in about a minute, write it first**;
`waiters.py ROOT s1,s2,..` every wait/update/DGE-sem touching a semaphore set;
`gaps.py prof [thresh]` per-engine union busy and every gap with the next op;
`tl.py prof [ENG|all] [lo hi]` merged timeline with per-engine gaps.

Check order after every edit: `semaphores.py` → `pack` → `verify.py` → `profile.py --repeats 5`.
Keep the last good pair next to the tree (`cp work-good.neff kernel.neff; rm -rf work; cp -r
work.good work`) — I rolled back six times and it cost seconds each time. **No hangs this
iteration**; every edit was either bit-identical or (never) a verify failure, which is what
staying inside "reorder slots within one engine" and "merge/split descriptors" buys you.
`profile.py --parquet` still writes only the LAST capture — check its `end` timestamp
(fast ≈ 58.9–59.5 on the shipped file) before trusting a gap analysis.

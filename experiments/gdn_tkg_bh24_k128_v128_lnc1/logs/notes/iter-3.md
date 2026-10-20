# iter-3 notes — gdn_tkg bh24 k128 v128, LNC=1

## Result

**74.447 us -> 72.704 us, -2.3%** (both numbers measured today on the same machine, 5 repeats).
`verify.py --runs 3/4` PASS with `max_abs_err` **exactly** the baseline digits
(2.322e-02 / 1.633e-03) and bit-identical across runs; `launch_rules.py` PASS; `semaphores.py`
balanced (s25 32/32, everything else unchanged); happens-before graph acyclic.

Final NEFF md5 `b5d2f28ad9ef3c2c10a9afff23b062bb`. Changed members vs iter-2:
`Activation0.bin` (slots reordered, 170 -> 170), `DVE0.bin` (166 -> 170), `Pool0.bin`
(17 -> 18), `PE0.bin`/`SP0.bin` (wait values only).

Apply order, against `work/payload/sg00` after `neff.py unpack` (scripts in `iter-3-scripts/`):

```
applyG.py 4 1,2,3      # split S*=exp_g into 4 chunks of 6 heads, hoist 3 wide casts
applyH.py 2 0          # split the state_in DGE input 2 ways, per-chunk gates
```

| step | edit | total us (5 reps) | verdict |
|---|---|---|---|
| in | iter-2 result, re-measured today | 74.447 | |
| **G** | split the 3.36 us DVE `S*=exp_g` into 4 chunks of 6 heads; move the 3 body-resident wide `S->S_bf` CASTs into the 3.1 us ACT head idle window, each gated on its own chunk | **73.116** | **-1.33 us**, kept |
| **H** | split the 1.5 MB `state_in` DGE input transfer (Pool@2, s25) into 2; gate each exp_g chunk on `s25 >= 16*(g+1)` | **72.70** | **-0.4 us**, kept |
| F1 | 11 of the 24 ACT `ACTIVATE f128` (`w=beta*delta`, 490 ns) moved to DVE as broadcast `TENSOR_TENSOR(MULT)` (~290 ns), ACT slot kept as an `EVENT_SEMAPHORE` waiting on the new DVE op | 75.793 | **+1.35 us REGRESSION, not kept** |
| F2 | same but the ACT `EVENT_SEMAPHORE` keeps the f128's own wait and PE gets a direct `s6` barrier instead | — | **verify FAILED, not kept** |
| I | split the first qPoolDynamic DGE input (Pool@1, s24, the 24x128 bf16 transpose-scatter) 4 ways | 73.176 | **+0.4 us, not kept** (DMA busy +1.6 us) |

Spread of `profile.py --repeats 5` on the final NEFF is 0.16–0.56 % (0.1–0.4 us), better than
iter-2's. The G and H wins are 3–4x the spread; the N=2 vs N=4 vs N=8 choice inside H is **not**
(72.60 / 72.72 / 72.66 over repeated measurements) — N=2 was taken because it adds the fewest DMA
descriptors, not because it measured better.

---

## THE central finding of this iteration (read this before touching anything)

**On-path work is conserved when you relocate it between ACT and DVE. The only edits that pay are
the ones that move on-path work into an engine's *idle window*, or delete it.**

Measured, not reasoned:

* iter-2 ended with ACT 55.5 us busy (74.5 %) and DVE 40.6 (54.6 %), and `cpath.py` showed the
  measured critical path as ACT-occupancy 44.5 + DVE-occupancy 20.7 + PE 5.5 + 10.7 us of gaps.
  That reads like "ACT is the bottleneck, move work off ACT".
* Edit F did exactly that, correctly and bit-identically, for 11 heads: ACT busy fell
  55.5 -> 49.9 (-5.6, as predicted) and DVE rose 40.6 -> 43.8. **The wall went UP 1.35 us.**
  `cpath.py` on the F NEFF: ACT-occupancy 34.9 (-9.6) and DVE-occupancy **31.0 (+10.4)**, gaps
  10.7 -> 12.4. The sum of on-path ACT+DVE busy was 65.1 before and 65.9 after.
* Why: the per-head body chain is
  `PE(S^T k) -> DVE(delta=v-S^Tk) -> ACT(w=beta*delta) -> PE(k (x) w) -> DVE(S+=outer) ->
   ACT(cast S->bf16) -> PE(o=S^T q) -> ACT(f30/READ_ACC/f29) -> DVE(STT out*rstd*silu) -> ...`
  Every link is on the chain. Both ACT and DVE are nearly serial **along that chain**, so a link
  that moves from ACT to DVE just lands in DVE's serial stream and delays DVE's next chain link.
  You pay two extra cross-engine hand-offs (~70-90 ns/head) for nothing.
* Contrast edit G: the 4 wide pre-outer casts are **not** per-head chain links (iter-2 edit A
  already established they are "pure filler" that only needs the `S*=exp_g` barrier). They were
  sitting in ACT *program order* inside the body, which is what put them on the measured path.
  Moving them into the 3.1 us ACT idle hole in the head removed 2.5 us of on-path ACT work and
  converted at ~53 %.

So iter-2's rule ("before removing an ACT op, check whether its output is read by the next link
of the chain; if not, the removal is free but worthless") needs an amendment:

> **Filler that sits in program order on a saturated engine IS on the path.** Removing it is
> worthless; *relocating it into an idle window on the same engine* is worth ~0.5x its duration.
> Conversely, relocating a genuine chain link to another engine is worth ~0 and usually negative.

---

## Edit G: split `S*=exp_g` and hoist the wide casts into the head idle window (-1.33 us)

Measured ACT head of the iter-2 NEFF (all times us):

```
10.34 t? (1283) | 12.78 silu f36 (2762, waits s11=z DMA) | 15.56 t1 | 17.10 f30 (waits s24!)
17.23 f9 | 17.45 t2 | 18.74 f29 | 18.94 t0 | 20.24 f7(A_log) | ...gap 1.14... | 21.60 f7(exp_g)
21.84 t3 (1283, ends 23.12) | ...IDLE 23.12 -> 26.20 (3.08 us)... | 26.20 ES(s6>=10)
26.25 wide cast heads 0-5 (831) | 27.55 f5 (waits s6>=14 = the b shuffles) | body
DVE: 22.81 S*=exp_g (3357 ns, one op, 24 heads) -> 26.17
```

The 3.08 us ACT hole exists because ACT has nothing to do until `S*=exp_g` finishes. The three
other wide casts (`S[6..11]`, `S[12..17]`, `S[18..23]` -> bf16, 831 ns each) were sitting at ACT
slots 50 / 88 / 126 inside the body. Split the exp_g op into 4 chunks of 6 heads and each cast
can be gated on just its own chunk, so all four fit in the hole:

```
after G+H: 21.99 f7(exp_g) | 22.23 t3 | 23.43 ES(chunk0) | 23.52 24.31 25.17 26.04 the 4 casts
           26.87 (end) | ...0.53 idle... | 27.40 f5 | 28.40 body
```

Useful facts:
* the exp_g op is `DVE TENSOR_TENSOR_ARITH_OP(MULT)`, `src0=dst=114688`, `src1=16576` (exp_g
  scalars), `num_elem=[128,24,1]`, `step=[1,128,1]` / src1 `[0,1,1]`, `nch=128`. Chunking is
  `num_elem[1]=PER` plus offsets `114688+512*PER*g` and `16576+4*PER*g`. Only chunk 0 keeps the
  original `s4>=6` wait; the rest rely on DVE program order.
* 4 chunks cost DVE ~+0.6 us of fixed overhead (3.36 -> ~3.95 us total) and that is off the path.
* the 4 wide casts are keyed by `src = 114688 + 3072*g`, `num_elem=[128,6,1,1]`, `nch=128`.
  They are `CAST`s, so **they need no activation table** and can be placed anywhere.
* 4 chunks is the right granularity: the chunk boundaries line up with the 6-head cast groups.
  8 chunks of 3 makes each chunk 0.57 us and the *last* one finish later (more fixed overhead);
  6 chunks of 4 does not divide the 6-head groups and gates groups 2/3 later.
* **Do not merge the casts further.** 2 casts of 12 heads is 0.3 us less ACT work but the second
  one cannot start until chunk 3, so it ends at ~27.9 and pushes the body out.
* retargeting: a wait that resolved to a moved cast must go to the next surviving ACT `s4`
  updater at/after its **original** slot (preserves "ACT progressed this far"); the data is
  strictly earlier, so that is conservative. `applyG.py` does this (3 waits here).

## Edit H: split the `state_in` DGE input transfer (-0.4 us)

After G the gate on the first exp_g chunk was no longer ACT but **`state_in`**: the 1.5 MB fp32
DGE input (`Pool@2`, semaphore 25) only completed at 23.8 us, so chunk 0 could not start until
then. Splitting it into N descriptors on the same queue/semaphore (`sem_increment=16` each) lets
chunk g wait on `s25 >= 16*(g+1)`.

* `Pool@2`: src = DGE-table form, `table_index=1`, `src_step=[65536,512]`, `src_num=[24,128]`,
  `elem_size=512`; dst = VAR form, `dst_start_addr_addr_var_offset=114688`,
  `dst_step=[512,262144]`. Chunk g: `src_start_addr_addr_table_offset = 65536*PER*g`,
  `dst_start_addr_addr_var_offset = 114688 + 512*PER*g`, both `num_elem=[PER,128]`.
  This is the exact mirror of iter-1 edit 9 (state-out) — `apply9.py` is the template.
* s25 has 5 waiters. The DVE ones become per-chunk; the single ACT one (`Activation@12`, a bare
  barrier) was left at the **full** count `16*N` — relaxing it to `16` (chunk 0) measured
  identically (72.697 vs 72.754) so the conservative form was kept.
* **Side effect worth knowing: splitting made the whole transfer land EARLIER**, 23.8 -> ~22.0 us
  un-gated. The DMA engine pipelines separate descriptors better than one big 2-D one. DMA busy
  only rose 19.64 -> 20.45 us.
* N=2/4/8 are all within noise (72.60 / 72.72 / 72.66 over 3 measurements each). N=2 kept.

## Edit F (REFUTED, with the data): ACT `f128` -> DVE broadcast multiply

This is the review's candidate C1 and iter-2's candidate 4, and it is the lever that gave -9 % in
the `gdn_tkg_bh24_k128_v128_lnc2` experiment. **Here it is a regression.** The transform itself is
correct and bit-exact; the schedule does not want it.

What the family is (keyed by `activation_func=128`, `scale_src=POINTER_IMMEDIATE`, 24 of them):
`w[h] = bf16(beta[h] * delta[h])` where `delta = v - S^T k` (written by a DVE `SUBTRACT`) and
`beta[0..23]` is at **16672 + 4h**, produced by the single `ACT@15 ACTIVATE f5` (sigmoid,
`num_elem=[24,1,1]`, `nch=128`, so beta is live on every partition). The dst is read by exactly
one PE `MATMUL` (the moving operand of the `k (x) w` outer product). 16 of the 24 write BFLOAT16
(these are iter-2 edit D's fused ones, dst `0x26000+256n`); the other 8 write FP32 into the
rotating 512-byte intermediates and are read by an ACT `CAST`.

The DVE replacement that verifies bit-identically:
```
TENSOR_TENSOR_ARITH_OP  op=MULT  nch=1
  src0 = f128 src,  step=[1,1,1]  num=[128,1,1]   in0 = FP32
  src1 = 16672+4h,  step=[0,1,1]  num=[128,1,1]   in1 = FP32     <- zero-stride broadcast
  dst  = f128 dst,  step=[1,1,1]  num=[128,1,1]   out = f128's out_dtype (BFLOAT16)
  wait s4 >= count(ACT@15 f5)      <- beta;  src0 comes from DVE program order
  update s6
```
inserted **immediately after the DVE op that wrote src0** (that placement also makes the WAR
against the next writer of the rotating src buffer free), and the ACT slot replaced by an
`EVENT_SEMAPHORE` that keeps the f128's `s4` update (so **zero** `s4` renumbering anywhere) and
waits `s6 >= count(new DVE op)`.

Per-slot verify sweep (one slot at a time, `--runs 1`, all deterministic):

| ACT slot | 18 | 23 | 26 | 27 | 43 | 44 | 59 | 70 | 72 | 81 | 84 | 98 | 107 | 110 | 120 | 122 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| verify | ok | **FAIL** | ok | ok | ok | ok | ok | ok | ok | **FAIL** | ok | **FAIL** | **FAIL** | ok | **FAIL** | ok |

The union of the 11 passing slots verifies bit-identically (`2.322e-02 / 1.633e-03`), so the
failures are per-slot, not pairwise. **The 5 failures are unexplained** and they rhyme with
iter-2 edit D's 5 unexplained fusion failures. They are not explained by "src has ACT writers"
(23 and 98 read `8604672`, which only DVE writes; 43 and 70 read `25183448`, which ACT does
write, and they pass). I then wrote the overlap-aware
scanner (`overlap.py`, see below) that iter-2 asked for, and it **explains 3 of the 5**:

> `overlap.py 25182936:25183448` shows three accesses based at **25182888** — 48 bytes, i.e.
> 12 fp32, BELOW the buffer the f128 reads — each spanning 512 bytes, so each covers 464 of the
> 512 bytes of the "25182936" buffer: `ACT@37 f30` (dst), `ACT@144 f128` (src) and
> `DVE@151 TENSOR_TENSOR` (dst). Likewise `ACT@145 f30` writes [25183400,25183912), covering 464
> bytes of the "25183448" buffer. (Slot numbers are the final iter-3 ACT stream; subtract 3 for
> the iter-2 stream, where `ACT@37` is `ACT@34` — exactly the writer iter-2's notes listed for
> `25182888` and could not place.)

So the partition-96 scratch ring is addressed from **two bases 48 bytes apart** and accesses from
the two families partially overlap. That is an ACT writer my exact-start-address scan did not see,
sitting between the DVE producer and the f128 read — which is precisely the WAR that breaks when
the read moves to DVE. It accounts for the failures at ACT slots **81, 107 and 120** (srcs
25182936 / 25182936 / 25183448), and almost certainly for iter-2 edit D's 5 unexplained pairs,
which live on the same ring.

**Slots 23 and 98 are still unexplained.** Both read `8604672` (partition 32, offset 216064) and
the overlap scan on that range finds no shifted-base access at all — only the 11 exact DVE
writers, the 11 exact ACT f128 readers, and one genuine alias: `ACT@9 f7(A_log)` reads
[8604928,8605024), i.e. the `A_log` **input** array lives inside the delta buffer and is
overwritten by the first delta store. Their dst side is structurally identical to the passing
slots' (`overlap.py 155904:156160 158464:158720` vs `158208:158464` — one f128 writer, one PE
MATMUL reader, nothing else). Whatever breaks those two is not an address-range effect.

Measurement of the 11-slot union: **75.793 us** (ACT 49.86, DVE 43.76). See the central finding
above for why. **This lever is now closed in both directions**, because variant 2 (ACT keeps the
f128's own wait; PE gets a direct `s6 >= count(m)` barrier inserted before the LDWEIGHTS of the
reader MATMUL, so ACT never blocks on DVE) **fails verify** (`max_abs_err` 0.785 / 0.461,
deterministic) even restricted to the 9 slots whose src has no ACT writer. Something besides the
PE MATMUL depends on "ACT has reached the f128 slot" and I did not find it; the `s4` counts and
the single-reader/single-writer analysis say it should be safe, which is a second piece of
evidence that the address scan is missing an overlapping access.

## Edit I (REFUTED): splitting the s24 input transfer

After G+H the head's root gate is **`s24` at ~17.45 us**, which is `Pool@1`: a 24x128 **bf16**
DGE input whose dst pattern is `step=[2,262144] num=[24,128] elem_size=2`, i.e. a transposing
scatter of 3072 two-byte writes, 6 KB that takes ~7 us from kernel start. It gates `ACT@4 f30`,
and `ACT@4` is first in ACT program order among `f30 -> PE@3 -> f29(k_scale) -> ... -> shuffles
-> f7(exp_g) -> S*=exp_g -> the wide casts`, so everything in the head hangs off it.

Splitting it 4 ways (same trick as H) **cost +0.4 us** and raised DMA busy 20.07 -> 21.66 us. The
2-byte-granular scatter does not pipeline the way the 512-byte state_in does. Closed.

What is NOT closed: `ACT@4 f30` could in principle be moved **after** `f7(A_log)` in ACT program
order (`ACT@5 f9` does not read `ACT@4`'s output 16824, and `f30` is in all five activation
tables so it needs no table of its own). That would let `f7(A_log)` run at ~15 us instead of
20.6. But `f29(k_scale)` -> `PE@3` -> `f30` is a real chain and the `b` broadcast shuffles wait
on `f29` via `DVE@6 s4>=4`, so the shuffles — and therefore `f7(exp_g)` — may not actually move.
iter-2 edit E is the closest attempt (it got `f7(A_log)` 20.73 -> 19.49 and measured a wash);
**E is now incompatible with G** because E's permutation parks `t4 + the 2.76 us silu + t3` in
exactly the 21-26.4 us window that G fills with the wide casts.

---

## Where the time is now (72.70 us)

Engines: ACT **55.5 (76.3 %)**, DVE 40.5 (55.7), DMA 20.5, PE 18.7 (25.7), Pool 15.5, SP 3.2.
ACT busy is unchanged from iter-2 — the whole win was moving ACT work out of the body and into
the head's idle.

ACT idle, 15.7 us total, by where:
* 1.38 us at 11.8 -> 12.8 waiting for the `z` input DMA (s11) in front of the merged silu.
* **1.14 us at 20.85 -> 21.99**, the head dependency loop: `f7(A_log) -> DVE@5 -> 4 a-shuffles ->
  f7(exp_g)`. Structural. Nothing can fill it: a `CAST` would need `S*=exp_g` (which needs
  `f7(exp_g)`), and the only table-free ACTIVATEs available (`f30`, `f128`) have no ready input.
  `t3` cannot be loaded there because `f7(exp_g)` still needs `t0`.
* 0.53 us at 26.87 -> 27.40, `f5` waiting for the second shuffle group (`s6>=17`), which DVE runs
  *after* the exp_g chunks. Moving those shuffles ahead of the chunks delays chunk 0 and the
  casts by ~1 us — measured as a loss on paper, not tried.
* **~10 us in the body**, as 24 gaps of 0.4-0.9 us, each ACT waiting on `s6` for that head's
  `DVE S+=outer` before the post-outer `CAST`. This is the per-head cross-engine round trip
  `ACT f128 -> PE k(x)w -> DVE S+=outer -> ACT cast`. Filling it needs ACT to interleave a
  **third** head, i.e. a real reordering of the ACT body stream (the compiler interleaves two).

Per-head ACT body cost (all on the chain, unchanged): `CAST` 298 + `f30` 266 + `READ_ACC` 293 +
`f29` 202 + `f128` 490 = 1.549 us against a ~2.05 us head period, i.e. ACT is ~76 % occupied in
the body and the remaining 24 % is that round trip.

## New tool: `overlap.py` (use this instead of exact-address scans)

`iter-3-scripts/overlap.py <start:end> ...` expands every instruction's mem-patterns into real
(partition, byte-range) extents — `span = sum(|step_i| * (num_i - 1)) * elem_size + elem_size`,
repeated for `p in range(num_active_channels)` at `p * 0x40000` — and reports every access whose
range **overlaps** the query, not just those whose start address equals it. Dtype size comes from
`in_dtype` / `in_bias_dtype_dtype_lo` / `in0_in1_dtype_dtype_lo` for srcs and `out_dtype` for
dsts. Two iterations of mystery failures came from scans that compared start addresses; run this
first on every buffer an edit touches.

## Ranked candidates for iter-4

1. **Interleave a third head in the ACT body stream** (~up to 10 us, the only big block left).
   ACT's 24 body gaps of 0.4-0.9 us are the `ACT -> PE -> DVE -> ACT` round trip for the current
   head. If head h+2's `f30/READ_ACC/f29` (which depend only on head h+2's `o` matmul) were
   scheduled into head h's gap, ACT would run dense. Blockers to check first: there are only
   ~2-5 rotating `delta` buffers and 8 PSUM banks, and `out_p` for every head is PSUM
   **partition 0** bank `5 - (h mod 8)`, so a deeper interleave needs more live buffers. The
   accumulator is also a single per-channel register, so two `f30 ZERO_ACCUMULATE` reductions
   must not interleave between an `f30` and its `READ_ACCUMULATOR`. **That last constraint may
   kill it** — check it before anything else.
2. **Delete `ACTIVATION_READ_ACCUMULATOR` (24 x 293 ns = 7.0 us of on-path ACT, pure fixed cost
   for moving ONE fp32).** There is **no** direct-accumulator-to-`dst` mode — `docs/07` is
   explicit that `accumulator_cmd` results are "read back with `DVE_READ_ACCUMULATOR` /
   `ACTIVATION_READ_ACCUMULATOR`", and `ACTIVATION_READ_ACCUMULATOR` is the only consumer of the
   `D1_RD` struct. So the remaining form is whether two heads can share one `READ_ACC` by
   retargeting the `o` MATMUL's PSUM dst to a second **partition** and using `nch=2` on
   `f30`/`READ_ACC`/`f29` (then one copy to bring partition 1's rstd back to partition 0 for the
   `nch=1` DVE `STT`: net -470 ns per pair, -5.6 us of on-path work). This is the largest block
   of removable work left and it is a *deletion*, which the central finding says is what pays.
   Check first that `f30 ZERO_ACCUMULATE` for head h+1 may legally sit between head h's `f30` and
   its `READ_ACCUMULATOR` — there is one accumulator per channel, so if it may not, this and
   candidate 1 are both dead.
3. **Make `s24` (Pool@1) land earlier** — it is the root of the whole head (17.45 us for 6 KB).
   Splitting it is closed (edit I). What is left: move `ACT@4 f30` later in ACT program order
   (see edit I above) and check with `cpath.py` whether the shuffles actually move; or move the
   transfer onto `qSPIO0` (which finishes all five of its transfers by 12.94 us) as a 6th JSON
   transfer. The latter is a `SP0.json` + trigger edit, same shape as iter-1 edit 11.
4. **Crack the 5 overlapping-access failures** (edit F's and iter-2 edit D's). Not for speed any
   more — F is closed on the measurement — but because an overlap-aware address scanner would
   unblock every future edit on the rotating `delta`/intermediate buffers, which is where both
   iterations' mystery failures live. Cheapest form: extend `xrange.py` to compare *ranges*
   (`start .. start + stride*(num-1) + elem_size`) per partition instead of exact start
   addresses, and re-run it on `8604672`, `25182936`, `25183448`, `25183960`, `25184472`.
5. **The 0.53 us `f5` gap**: move 2 of the 6 `b` shuffles ahead of exp_g chunk 0 only (not all 6),
   so `f5` is ungated at 26.87 while chunk 0 slips only ~0.35 us. Worth one measurement, ~0.3 us.
6. Low value / confirmed noise: the H split count (N=2/4/8 all within 0.12 us), the G chunk count
   (4 is best on paper), re-tuning iter-2 edit B's `k_f` chunk placement.

## Closed, re-confirmed or newly closed

* **Relocating a chain link from ACT to DVE** (`f128`, both wait variants) — regression / verify
  failure. Newly closed, see above. The corollary is that candidate C4 of the review (move the
  remaining `CAST`s to DVE) and candidate 2 of iter-2's list are also closed: they are the same
  relocation of the same kind of chain link. Do not re-run them.
* **Having DVE write `S_bf` directly and deleting the 24 post-outer ACT `CAST`s** — not tried, but
  it is the same relocation: the extra fp32 `S` write would land in DVE's serial stream right
  next to the bf16 one, so on-path work is conserved (580 ns of DVE vs 588 ns of DVE+ACT). Only
  worth it if the fp32 write can be parked in a *DVE* idle window; DVE's on-path gaps are 2.7 us
  total, so at most ~4 of the 24 would fit.
* **Splitting the s24 transposing input transfer** — +0.4 us (edit I).
* **iter-2 edit E (the ACT head permutation)** — now actively incompatible with edit G.
* **5 activation table loads** — still 6, still the floor (iter-1/iter-2 analysis unchanged; `f5`
  is gated at ~27 us so `t3` cannot be hoisted in front of the body's `t2`).
* **iter-2 edit D's 5 failing + 3 excluded fusion pairs are now closed with a reason.** Edit D
  fused `ACTIVATE f128` (fp32 out) with the `CAST` that read it by making the ACTIVATE write bf16
  into a *private* buffer and re-pointing the MATMUL — which stops the fp32 intermediate from
  being written at all. The 8 unfused ones all live on the partition-96 scratch ring, whose two
  48-byte-apart bases mean other ops read/write **overlapping** 512-byte extents of that same
  intermediate (`overlap.py` on `25182936` / `25183448` lists them). Their intermediate is still
  needed, so redirecting the write cannot be correct. Worth ~2.4 us of on-path ACT if anyone finds
  a form that keeps the fp32 store, but the simple fusion is dead.
* **Deep ACT interleaving (candidate 1) is constrained, not dead.** There is one accumulator per
  channel, so another `f30 ZERO_ACCUMULATE` may never land between a head's `f30` and its
  `ACTIVATION_READ_ACCUMULATOR`. `CAST`, `f29` and `f128` from another head may — the current
  stream already puts `f128` and `f29` there. So a third-head interleave has to move that head's
  `CAST`/`f128`/`f29` only, and keep its `f30 -> READ_ACC` pair contiguous.
* Everything iter-1 and iter-2 closed stays closed; nothing in this iteration contradicted them.

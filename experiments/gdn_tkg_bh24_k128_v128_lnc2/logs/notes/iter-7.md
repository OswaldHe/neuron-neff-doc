# iter-7 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 36.805 us (iter-6 NEFF, re-measured here) -> 36.106 us, -1.9 %** (vs the iter-0
baseline 76.054: **-52.5 %**, 2.106x).  Bit-identical: `max_abs_err` exactly the baseline
2.322e-02 on output_0 and 1.633e-03 on output_1, 3 device runs bit-identical to each other.
Final kernel.neff sha256 `bef388609618662899e791ebae5ab9a0e957f5878a9a445397df7710d4c2c7dd`
(md5 `008fc47fabbaec67c3553d488ff6f153`).

| | edit | script | median (repeats) | verdict |
|---|---|---|---|---|
| — | iter-6 NEFF, re-measured | — | 36.805 (5) | |
| A | delete the 12 `out` sinks, precompute `norm_weight*silu_z` wide | `fold.py` | 37.174 (7) | **+0.37, reverted** |
| B | A + ACT `T4+silu` to the front so the precompute hides in DVE's head idle | `head7.py` | 37.728 (5) | **+0.55, reverted** |
| **C** | **4 `out` sinks moved forward into DVE's PE/ACT stalls** | `sink.py` round 1 | **36.261 (7)** | **kept, -0.54** |
| **D** | **a 5th sink moved the same way** | `sink.py` round 2 | **36.055 (7)** | **kept, -0.21** |
| E | D + the `out[h4]`/`out[h7]` sinks moved too | `sink.py` | 35.972 (7) | **VERIFY FAILED** (see below) |

Final metric run (`profile.py --repeats 5`): **36.106 us**, spread 0.84 %, min 36.001 max
36.304.  The 7-repeat run of the same binary read 36.055 (spread 0.48 %).  iter-6's binary
measured 36.805 here and 36.772 in its own iteration, so the -0.70 us separation is ~2.3x
the spread and consistent across two measurements of each binary.

## The lever: a DVE *sink* costs nothing if it sits in front of a stall

`crit.py` on the iter-6 NEFF showed something that should have been obvious earlier: the
twelve `out[h] = out_norm * silu_z[h]` DVE TENSOR_TENSORs are **sinks** — nothing reads
`out[h]` except the final output DMA, which is gated only by the *last* of them — and yet
five of them (DVE slots 98, 101, 104, 111, 113) were sitting in the middle of DVE's program
order **on the critical path**, delaying the real chain behind them purely because DVE is
in-order.

At the same time DVE had four measured stalls in the body where it waits on PE/ACT and has
nothing to do (0.45 us before slot 72, 0.43 before 109, 0.71 before 118, 0.36 before 122).
So: move each movable sink forward to just before the next stalling slot.  The chain ops
that followed it then issue ~0.28 us earlier, and the sink runs for free inside the stall.
Five moved (`out[h0] h1 h5 h8 h9`), total -0.70 us, with **zero** bytes changed except the
order of five 64-byte slots — no arithmetic, no address, no dtype, no wait value (none of
the five updates a semaphore, so no numbering moved at all).

Measured effect on the DVE stream: last DVE slot ends 48.052 -> 46.640 us (-1.41), DVE body
stalls 1.95 -> 1.69 us, DVE busy unchanged (it is the same work).  The metric moved less
than the DVE end did (-0.70 vs -1.41) because the ~3 us runtime drain after the last output
DMA does not scale with it.

**This is a reusable, nearly free lever and the sweep should be re-run after any future
edit**: `stall.py` prints every DVE slot with a gap in front of it together with every op5 /
sink, so the (stall capacity, movable sink) pairs can be read straight off.  Note that
moving work *out of* a stretch that is already followed by a stall gains nothing — only
stretches where DVE is saturated count.

### The hazard that static checking missed (edit E, read this before moving anything)

Moving the `out[h4]` sink (reads 0x4270) and the `out[h7]` sink (reads 0x4170) forward was
measured at **35.972 us** — and was **wrong**: output_0 `max_abs_err` 4.425 (limit 0.027),
92.7 % within tol, i.e. about one head's output garbage.  It passed `semaphores.py` on both
subgraphs, `cycle.py`, `hb.check_psum` **and** the generic reachability check that found
both of iter-6's real hazards.

Why: these compiler scratch tiles **overlap at different element widths**.  `out_norm` is
128 fp32 = 512 B, so op5's 0x4270 buffer is [0x4270,0x4470) — which contains the 0x4370
`delta_bf` tile (128 bf16 = 256 B) that ACT's `cast_delta` writes 2.7 us later, and overlaps
op5's own 0x4170 buffer.  Every check I had compared *start addresses*, so none of them saw
it.  Two lessons:

* **compare byte extents, not start addresses** — `sink.py` now does (`wrange`/`ovl`), and
  the fixed check reproduces the failure as an assertion for exactly the two bad moves and
  passes the five good ones;
* `hb`'s reachability check proves only that *some* happens-before order exists between a
  (reader, writer) pair, **not that it is the right one**.  Pushing a read past a write
  creates a definite — wrong — order and the check goes quiet.  It is a race detector, not a
  dependence checker.

## Why deleting the sinks outright (`fold.py`) does not pay — and the DVE cost model

The obvious bigger move is to delete all 12 sinks.  `out[h] = ((out_p*s[h]) * nw) * silu[h]`
with `nw` = `norm_weight` (0x36400, one 128-fp32 vector shared by all 12 heads) and
`silu[h]` = 0x20000+0x200h.  The op5 STT already does `(src0 op0 scalar) op1 src1`, so if
`nws[h] = nw*silu[h]` is precomputed, op5 can do `(out_p*s) * nws[h]` and write `out[h]`
straight to bf16 — the sink disappears.  `fold.py` does exactly that, and it works:

* a single 1536-element DVE op computes all 12 `nws` (src1 = the same 128-vector with an
  **outer-dim step of 0**, `step [1,0,1] num [128,12,1]` — this encoding works even though
  the compiler only ever emits step 0 on dim 0),
* the STT writes bf16 (set `out_dtype`), dst = `out[h]`,
* the 3 of 12 sinks that updated s6 were deleted by re-pointing their waiters at the
  previous surviving s6 updater (the output doorbell SP0@9 went s6>=80 -> 77),
* **the reassociation is invisible**: `max_abs_err` came back exactly the baseline 2.322e-02.
  ((p*s)*nw)*silu vs (p*s)*(nw*silu) differ by ~1 fp32 ulp and the result is rounded to
  bf16, so a visible flip needs the value within ~1e-7 relative of a bf16 tie.

It still **lost 0.37 us** (36.805 -> 37.174).  `cmp2.py`: DVE busy 27.94 -> 26.43 us
(-1.51 as predicted) but the last DVE slot only moved 48.052 -> 47.959.  The reason is the
DVE cost model:

> **DVE duration ~= 150 ns + 1.0 ns x (elements per channel).**  Verified over the whole
> stream: 1 ch x 128 elem = 278 ns, 128 ch x 128 elem = 288 ns (128 lanes, one per
> partition), 1 ch x 1536 elem = 1740 ns.  A `num_active_channels=1` op therefore uses
> 1/128 of the machine, and ~55 % of a 128-element one is fixed overhead.

So merging twelve 128-element 1-channel ops into one 1536-element 1-channel op saves the 11
fixed overheads (3.42 -> 1.74 us) but turns them into **one unsplittable 1.74 us block**,
and there is nowhere in this kernel to put it: DVE is back-to-back from 19.9 us to the end,
its only idle is 12.6-18.5 us, and the precompute's input `silu(z)` does not exist until
22.8 us.  Placed in front of the body it goes straight onto `crit.py`'s path and costs more
than the twelve scattered 285 ns ops did (which, after edit C/D, mostly sit in stalls).

`head7.py` tried to fix that by making `silu(z)` early: move ACT's `T4 load + silu` to the
front of the ACT stream (it displaces the 0.99 us stall ACT already has in front of
`f9 softplus`, which waits for DVE's `dt`).  The precompute **did** become free — it runs at
15.25-16.99 us inside a 3.5 us DVE idle window — but the total went to **37.728**:
everything after the moved pair shifts ~1.7 us later, so `f7A` (16.52 -> 18.24) and
`f29 k_scale` (18.47 -> 19.91) are late, DVE's broadcast chain hangs off exactly those two,
and ACT's own body start (which waits s6>=17/18 on that chain) ends up 1.1 us later.
**The ACT head is a tight chain that feeds DVE at four points (`f7A`, `f29 k_scale`,
`f7 exp_g`, `f29 q_scale`); nothing can be inserted into it without paying for it twice.**

## The edit is provably a pure permutation (and the two cores stay balanced)

`neff_diff` iter-6 -> final: 87 of 89 members identical, only `sg00/DVE0.bin` and
`sg01/DVE0.bin` changed, same byte length, 124 slots in both, 42 slots differing.  Decoding
both streams and comparing the multiset of encoded slots: **identical multiset, different
order, for DVE0 on both subgraphs; byte-identical in order for ACT, PE, Pool and SP on both
subgraphs.**  So nothing was added, removed or re-encoded -- five 64-byte slots moved.  That
is the strongest guarantee available for a scheduling-only edit, and it is worth re-checking
with the snippet in `iter-7-tools/README.md` after any "pure reorder" claim.

Core balance (the review's open question): the two cores are symmetric.  Busy times agree to
11 ns per engine, and pcore 1 finishes 0.34 us *earlier* than pcore 0, so sg00 sets the time
but only by a third of a microsecond -- there is nothing to win by rebalancing them.

## The 3.1 us after the last DVE slot (next thing to look at after the DVE work)

The last DVE slot ends at 46.64 us but the instruction stream runs to 49.75.  What is there:

| event | t | gated by |
|---|---|---|
| DVE's last `out` sink (s6 #80) | 46.640 | the per-head chain |
| SP@9 `PSEUDO_DMATRIGGER` output_0 | 46.668 | s6>=80 |
| output_0 DMA (3072 B) | 47.387 - 47.651 | **0.72 us of DMA launch latency** |
| Pool@7 state-out DGE, heads 10-11 (128 KB) triggered | 45.107 - 45.916 | s6>=76 (S+= h11) |
| ... that DMA completes (s27 reaches 64) | ~48.0 | 128 KB at ~61 GB/s = 2.1 us |
| Pool@8 `W s16>=16` (output_0 landed) | 47.960 | the output DMA |
| Pool@9 `W s27>=64` (state-out landed) | 48.024 - 48.045 | the state DMA |
| runtime drain | -> 49.745 | not editable |

The two paths are within 0.09 us of each other, so **both** have to be shortened to gain
anything, and each is worth ~0.2-0.3 us:
* split the last state-out DGE (currently heads 10-11 together, 128 KB) into one transfer
  per head: head 10's `S` is final ~1.7 us before head 11's, so the last transfer halves to
  ~1.05 us and lands ~46.9 instead of ~48.0;
* split output_0's DMA into heads 0-10 (triggered as soon as s6 covers them) + head 11
  (256 B), so the 0.72 us launch latency overlaps the tail and only 256 B is left to move.
Both need SP0.json/DGE descriptor edits (iter-6 already split the state-out from 1 into 4, so
there is precedent).  Combined ceiling ~0.3 us; do this only after candidate 1 below.

## Where the remaining 36.1 us is

`profile.py` on the final binary: DVE 95.4 % busy, ACT 90.3 %, PE 52 %, Pool 36 %.
`stall.py` on the final DVE stream (core 0): work runs 12.0 -> 46.64 us, remaining stalls
0.24 (before the slot that waits s5>=17) + 0.15 + 0.15 + 0.23 + 0.17 + 0.19 + 0.36 = 1.69 us
in total, every one of them a wait on PE or ACT.

Per-head DVE body cost is now **1693 ns** in five slots: `v-kv_p` 277 + `*beta_K` 279 +
`S+=outer` 289 + `CAST S->S_bf` 220 + `out_norm` STT 339 + `out` sink 285.  ACT's per-head
body cost is 1056 ns (`cast_delta` 297 + `f30` 264 + `READ_ACC` 293 + `f29` 202).  So DVE is
the per-head rate limiter by ~0.64 us/head and **four of its six ops are
`num_active_channels=1` over 128 elements, i.e. ~55 % fixed overhead and 1/128 of the
datapath**.  That is ~10 us of the 36 that is pure per-instruction overhead and it is the
only big prize left.

## Two things the docs settle (chapter 5.4) that change the search space

* `step_elem` is **signed 16-bit** (max 32767) and **"the partition axis is not part of the
  pattern: it is `num_active_channels` partitions starting at the address's partition"**.  So
  no DVE/ACT access pattern can stride across partitions (that would need a step of 65536
  fp32 elements).  **A strided write cannot replace the 4x`STREAM_SHUFFLE` + 4x
  `LOAD_MASK_SELECT` broadcast groups** (0.70 us each; the `q_scale` one at DVE slots 41-48 is
  on the critical path at the head/body boundary), because STREAM_SHUFFLE only reaches 32
  partitions per op and 4+4 is the floor for 128.  Dead lever, do not spend time on it.
* The same rule is why the per-head 1-channel DVE ops cannot simply be re-laid-out: the only
  ways to get data into many partitions are a **DMA descriptor** (chapter 5.3: descriptors
  "step from one partition to the next" by 0x40000), `STREAM_SHUFFLE` (32 partitions/op), or a
  **matmul output** (output partition = the stationary operand's free index).  That is the
  argument for attacking the input descriptors first in candidate 1.

## Ranked candidates for iter-8

1. **Get the per-head 1-channel DVE ops onto more partitions.**  This is iter-5 candidate 3 /
   iter-6 candidate 4 and it is now clearly the only >1 us lever.  iter-7 adds two concrete
   findings that change how to attack it: (a) the cost model above says the win is ~127/128
   of the data time plus 11 of 12 fixed overheads, (b) **the input DMA descriptors are the
   place to start, not the compute**.  `work/payload/sgNN/SP0.json` has one `desc` per input
   with `to_off`/`to_sizes`/`to_steps`; `z` is `to 0x836600 [6144,1] [1,1]` = 1536 fp32 in a
   single partition, and `norm_weight` is `to 0x36400 [512,1] [1,1]`.  Rewriting those as
   `[512,12]/[1,0x40000]` (scatter z over 12 partitions) and `[512,12]/[1,0x40000]` with
   `from_steps` 0 (broadcast norm_weight) would make ACT's `silu` a 12-channel 128-element op
   (1482 -> ~190 ns, and that 1.3 us is on the ACT head critical chain) and make the whole
   `nws`/`out` tail a 12-channel op.  `out` itself is DMA'd out, so its descriptor would have
   to be gathered from 12 partitions too.  Risk: untested descriptor editing, and whether a
   partition-strided or source-broadcast DGE descriptor is legal.  Cheapest falsifier: change
   `norm_weight`'s descriptor alone to write 12 partitions and check with a 12-channel read
   that the data is there (any wrong value shows up in verify immediately).
2. **Re-run the sink sweep** (`stall.py` + `sink.py`) after any edit.  It is free, it is
   bit-identical, and it is what this iteration shipped.  Two sinks (`out[h4]`, `out[h7]`)
   are currently blocked only by the overlapping-scratch hazard; giving op5 h4/h7 private
   512 B buffers in the free SBUF at 0x28000 (device-confirmed free) would unblock ~0.57 us
   of sink work — but measured against the current stalls it is worth only ~0.15 us, so do it
   only if the stall pattern changes.
3. **The op5 STTs (12 x 339 ns) are also sinks** and could be deferred into stalls the same
   way, which is worth up to ~1 us — but 10 of the 12 update s6 and ACT/PE/Pool waits point
   at those updates, so each move delays a waiter.  Needs the move to carry the update's
   waiters with it (the `qearly.py`/`fold.py` renumbering machinery can, but the semantics
   have to be argued per waiter).
4. Closed / do not redo: `fold.py` and `head7.py` above; `fuse.py` and any ACT<->DVE
   elementwise move (iters 3/5/6); Pool as a host for anything (iter-6); the head table-load
   plan and the input-DMA *order* (iter-6); further wait tightening (`pe4.py` reports zero
   slack).

## Addresses (iter-6's list, plus what iter-7 established)

`S_all` 0x1c000+0x200h (fp32), `S_bf` 0x26000+0x100h, `silu(z)` 0x20000+0x200h (fp32, all 12
heads contiguous in partition 0), `delta` 0x22000+0x200h (fp32, contiguous), `norm_weight`
0x36400 (128 fp32, one vector for all heads), `v` 0x36600+0x200h, `k_f_all` 0x35800 (1536
bf16, partition 0), `K_p` 0x37e00, `Q_p` 0x37f00, `beta_K` 0x4090+4h, `q_scale` 0x40f8,
`k_scale` 0x4128, per-head `out` **0x1806078+0x100h (bf16, partition 96, contiguous; this is
output_0 = 24 heads x 128 bf16 = 6144 B)**.
`out_norm` scratch (op5 dsts, 128 fp32 = **512 B each, and they overlap each other and the
delta_bf tiles**): h0 0x5c70, h1 0x6078, h2 0x5078, h3 0x4570, h4 0x4270, h5 0x5078,
h6 0x4470, h7 0x4170, h8 0x5570, h9 0x5078, h10 0x4470, h11 0x4008.
`delta_bf` (ACT `cast_delta` dsts, 128 bf16 = 256 B): 0x5970, 0x4d70, 0x4170, 0x4470,
0x4170, 0x4270, 0x4170, 0x4670, 0x4370, 0x4470, 0x4170, 0x36800.
Free SBUF (from `sbuf.py`, and **0x28000 confirmed free on device** by the fold experiment):
0x9078-0x1c000, 0x1d800-0x20000, 0x23800-0x26000, 0x27200-0x35500.  SBUF per partition runs
to 0x37ff8 (`def.json: runtime_statebuffer_reservation`).
Input DMA landing times (`DmaPacketAggregated`): dt_bias 13.13, **z 13.58**, a 13.59,
b 13.67, v 14.21, k 14.38, **norm_weight 14.41**, A_log 14.45; output_0 DMA 48.7-48.9.
`addr = partition*0x40000 + offset`; PSUM `0x2000000 + bank*0x800`.

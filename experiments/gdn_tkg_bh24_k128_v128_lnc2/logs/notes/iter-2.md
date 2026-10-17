# iter-2 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 55.601 us (iter-1) -> 44.977 us, -19.1%** (vs the iter-0 baseline 76.054: -40.9%).
Confirmed twice on the final bytes (44.957 / 44.977, min 44.416, spread 1.2-2.3%).
`verify.py --runs 8`: all 8 device runs **bit-identical**, and `max_abs_err` on both outputs
is **identical to baseline** (2.322e-02 = 86% of the limit, 1.633e-03) after every one of the
four accepted edits — so this is pure schedule/placement, zero arithmetic change.
Final sha256 `7677acb82230c74e…`.

## The lever: ACT's scale-pointer ACTIVATEs, and zero-stride broadcast on DVE

**An ACT `ACTIVATE` with a *scale pointer* costs ~490 ns; the same shape with an immediate
scale costs ~310 ns and a `CAST` ~290 ns.** The pointer read is ~180 ns of pure overhead.
There were **36** of them (3 per head x 12 heads) = **17.5 us = a third of the kernel**: every
`nisa.tensor_scalar` with a per-head runtime scalar.

All 36 can be done on **DVE** instead, **bit-exactly, with no new opcode**:

1. DVE already runs `TENSOR_TENSOR_ARITH_OP` with **mixed dtypes**
   (`in0_in1_dtype_dtype_lo` = src0, `..._hi` = src1, plus `out_dtype`), in place, 128
   channels — it is the `K_p *= k_scale` op at DVE slot 22. Clone that slot and retarget it.
2. **`step_elem = 0` broadcasts an element** (docs/05-memory-and-addresses.md:75). A `src1`
   pattern of `step=[0,1,1] num=[128,12,1]` reads `scale[h]` 128 times for each h, so **12
   per-head `tensor_scalar`s collapse into one `TENSOR_TENSOR` over 1536 elements**.
   *Verified on device.* This is the single most useful fact in these notes.
3. ACT's `func(scale*in + bias)` with `func=COPY, bias=0` and a DVE fp32 multiply round
   identically — `max_abs_err` never moved a digit.

A second, equally reusable fact: **`CAST` is a valid DVE *and* Pool opcode** (the doc corpus
has 23900 DVE and 370 Pool instances), so an ACT `CAST` can be moved to DVE **verbatim** —
bit-exact by construction, no need for a multiply-by-1.0 trick.

## Edits applied

| # | edit | ACT slots | ACT busy | total |
|---|---|---|---|---|
| — | iter-1 | 140 | 51.5 us | 55.601 |
| 1 | 12 `k_f *= k_scale[h]` (ch=1, 0x35800+0x100h, scale 0x4128+4h) -> **one** DVE TENSOR_TENSOR, 1536 elem (1741 ns) | 128 | 45.8 | **52.338** |
| 2 | 12 `S *= exp(g_h)` (ch=128, 0x1c000+0x200h, scale 0x4060+4h) -> **one** DVE TENSOR_TENSOR, 128ch x 1536 elem (1757 ns) | 116 | 40.0 | **50.704** |
| 3 | 12 `delta = diff * beta[h]` (ch=1, scale 0x4090+4h) -> **12** DVE TENSOR_TENSORs, each right after the DVE SUBTRACT that makes its `diff`, into **private** delta buffers at 0x22000+0x200h | 104 | 34.0 | **46.140** |
| 4 | 12 line-280 `CAST S -> S_bf` -> **one** 1536-element ACT CAST, with the 12 `S_bf` tiles moved to a contiguous private block at 0x26000+0x100h | 93 | 32.0 | **44.977** |

Scripts (`.autohelix/notes/iter-2-tools/`), run in order on a freshly unpacked iter-1 NEFF,
each with an optional `--dry`:
`neff.py unpack` -> `kf.py` -> `f1.py` -> `f3.py` -> `c280.py --groups 1` -> `neff.py pack`.
`cycle.py` + `semaphores.py` (both subgraphs) + `check.py` after each one; no hang all
iteration. Final semaphore counts: s4 64/64, s5 38/38, s6 66/66, identical in sg00 and sg01.

### The reusable mechanics

* **Semaphore bookkeeping is fully mechanical.** Key every s3/s4/s5/s6 updater by
  `id(instruction)`, record every wait as (waiter *object* -> producer *object*), apply the
  structural edit, recount, rewrite. ~30 lines, never wrong once. Record waits by object,
  **not** `(engine, slot)` — slot indices move when you insert or delete.
* **Deleting an instruction orphans the waits that named it.** Two correct repairs; pick by
  what the dependency actually is:
  - the real producer moved to another engine -> repoint the wait to the new engine's
    semaphore and count. Identify the consumer exactly: match the moved op's dst against
    later ACT CAST srcs *and* cross-check against the semaphore graph. In every case here
    each moved op had **exactly one** waiter, which is a strong correctness signal — assert it.
  - the wait was only incidental ordering -> repoint to the **next surviving s4 updater after
    the deleted slot**, which is strictly stronger than the original (used for 5 PE
    `LDWEIGHTS` in edit 1).
* **One wait per instruction.** Where two conditions are needed (edit 2 needs the `S_all` DGE
  DMA *and* `exp_g_K`), either insert a bare `EVENT_SEMAPHORE` on DVE for one of them (~24 ns,
  DVE already uses them as pure waits) or rely on DVE program order: a condition already
  waited on by an *earlier* slot of the same engine needs no new wait. Two ACT s4 conditions
  collapse into one wait on the larger count, since the counts are monotonic.
* **Inserting slots into the DVE and PE streams works** (12 and 2 insertions here). Only
  *Pool* is fragile (iter-1: inserting between its `PSEUDO_DMA_*` slots hangs).
* **Buffer reuse is the real hazard; private buffers are the answer.** The 12 `delta` buffers
  were shared (0x1035800 by four heads) and the 12 `S_bf` tiles were shared both between the
  two casts of one head *and* across heads (0x5370 by heads 9 and 10, 0x7278 read by two
  different matmuls). Reasoning about those WARs is how you get a silent race; giving each
  head a private buffer removes the hazard instead. Free SBUF (partition-offset, `sbuf.py`):
  **0x9078-0x1c000, 0x1d800-0x20000, 0x21800-0x35500**, minus iter-1's silu block
  0x20000-0x21800, iter-2's delta block 0x22000-0x23800 and S_bf block 0x26000-0x26c00
  (0x24000-0x24c00 is reserved by the unused `c300.py`).
* **When a merge pays.** A merged instruction has to sit at a slot where *all* its members'
  inputs are ready. That is free when the members share one early producer (edit 4: all 12
  line-280 casts wait only on the merged `S *= exp(g)` op, so the merged cast goes at the
  **earliest** member's slot and only ever moves work earlier). It is a trap when the members
  have different producers: the merged op must then sit at the **latest** member's slot and
  delays every earlier head's whole tail. That is why edit 4 worked and why merging the
  line-300 casts (one producer each, spread over the stream) was not attempted.
  Cost model: an ACT op over n elements is `~182 + 0.85*n` ns, so 12 x 128 elem (3.5 us)
  becomes 1.48 us. Measured G=1 (one 1536-elem cast) 44.957 vs G=2 (two 768) 45.063.

## Where the time is now (44.98 us)

| engine | slots | span | busy | idle |
|---|---|---|---|---|
| ACT / Scalar | 93 | 41.76 | **32.02** | 9.74 |
| DVE / Vector | 110 | 42.56 | 24.82 | 17.74 |
| PE / Tensor | 80 | 40.63 | 14.42 | 26.21 |

All three start at t≈12.4 us and end at 53-55; `total_exec_time ≈ DVE span + 2.4`.

ACT body work (32.0 us): 6 `ACT_TABLE_LOAD` 7.70 | 25 `CAST` 8.45 | 33 `ACTIVATE` 9.13
(12 f30 square-reduce 3.7, 12 f29 rsqrt ~2.3, merged silu 1.48, 8 globals) |
12 `ACTIVATION_READ_ACCUMULATOR` 3.51 | 12 `COPY` PSUM->SBUF 3.03.

**The kernel is no longer throughput-bound on one engine — it is bound by the per-head
dependency chain plus a ~10 us serial prologue.** Evidence: moving 3.5 us of ACT work to DVE
(`c300.py`) measured **neutral twice** (46.159 vs 46.140 at the edit-3 state; 44.832 vs
44.957 at the edit-4 state). Perfect overlap would give prologue(~10) + max busy(32) ≈ 42 us,
so there is ~3 us of interleaving headroom left, and beyond that only work reduction helps.

ACT's 9.7 us of idle, by cause: 1.69 the merged silu waiting on the `z` cast DMA (s15);
1.06 `exp(g_line)` waiting on DVE's `g_line` broadcast (s6>=8); 2.4 the first casts and
`sigmoid` waiting on the merged DVE ops and the `b_line` broadcast; **~4.5 spread over the
12 line-300 casts (waiting s6 = DVE's `S += outer`) and the 12 `COPY`s (waiting s5 = PE's
`out_p` matmul)** — i.e. pure chain latency in the per-head tail
`DVE S+=outer -> ACT CAST S_bf -> PE out_p -> ACT COPY -> ACT f30 -> ACT READ_ACC ->
ACT f29 -> DVE xf -> DVE out_sb` (9 steps, ACT appears 4 times, ~2.5 us per head).

## Ranked candidates for iter-3

1. **Shorten the per-head tail by deleting the 12 `COPY` PSUM->SBUF (3.0 us *and* one chain
   step).** Not a *move* (iter-1 proved moves fail: DVE cycles, Pool hangs) but a **deletion**:
   `o_sb` exists only because two consumers read it — the ACT `f30` square-reduce and the DVE
   `scalar_tensor_tensor(xf = o_sb*ssum*nw)`. **Both can read `out_p` from PSUM directly**
   (ACT's COPY already reads PSUM, and DVE reads PSUM for `outer_p` at slot 43), and reading
   the same fp32 values from PSUM instead of SBUF is bit-exact. The blocker to work out: the
   8 `out_p` PSUM tiles are reused across the 12 heads (0x2001800 and 0x2003800 each twice),
   so extending a tile's live range needs a WAR wait on the PE matmul that overwrites it —
   enumerate (PSUM addr -> writers = PE MATMUL dsts, readers) and add `s6` waits on PE. This
   is the best remaining lever and the only one that shortens the chain.
2. **The ~10 us prologue** (t=12.4 to ~22.5: four table loads 5.1 us + silu 1.5 + the s15 and
   s6>=8 stalls 2.8). Everything downstream waits on it because every global scalar
   (`exp_g`, `beta_K`, `k_scale`, `q_scale`) is computed by ACT, and DVE's first real op idles
   **7.2 us** waiting for ACT's 5th s4 update. Two sub-levers:
   - the `z` DMA (`qSPIO0` id 7, s15, a 24 KB `cast bf16->fp32`) is the **last** of the 8
     input transfers in the queue and lands at ~15.1 us, which is what the silu waits for.
     Reordering the queue in `SP0.json` so `z` goes first should remove most of the 1.69 us
     stall (the other small input DMAs then land later — check that `a`/s10 and the K_p DGE
     s24 do not become critical, they already land ~16.5 us). **Not attempted**; ch. 9 has the
     descriptor format and `neff.py` repacks JSON members fine.
   - the 4 prologue table loads are irreducible (see Closed) and the silu is the only
     DMA-independent work available to hide them behind, which is why it is parked there.
3. **Re-interleave the ACT stream** to recover the ~4.5 us of per-head tail stalls (ACT runs
   out of ready work because the 12 head chains are not evenly staggered). This is static
   list-scheduling over the semaphore graph; it is the largest remaining idle block but
   reordering ACT re-orders writes to **reused** buffers, so it needs a full
   writer/reader dependence analysis first (or private buffers everywhere).
4. `c300.py` (move the 12 line-300 casts to DVE) is **written, verified, bit-identical and
   measured neutral** at both states. Keep it in reserve: if a future edit makes ACT binding
   again (ACT was back to 91% busy after edit 4), it buys 3.5 us of ACT work for 3.5 us of
   DVE work. Do not stack it while ACT and DVE are within ~10% of each other, as now.
5. The last untouched ACT blocks are the 12 `f30` square-reduce + 12 `ACTIVATION_READ_
   ACCUMULATOR` (7.2 us). Collapsing them into one 12-channel reduce + one 12-channel
   accumulator read (-6.6 us!) would need the 12 `o_sb` tiles on 12 *different* partitions,
   but all 12 `out_p` PSUM tiles sit on partition 0 and engine ops are partition-preserving
   (that is what `STREAM_SHUFFLE` is for), so it would cost 12 DVE shuffles and a
   cross-partition `ssum` that the DVE `xf` op cannot then read. **Probably closed**, but the
   -6.6 us prize is large enough to re-check if someone finds a cross-partition path.

## Closed this iteration (do not retry)

* **Fewer `ACT_TABLE_LOAD`s via a richer table set.** I enumerated all 22 vendor sets in
  `neuronxcc/pwp/pwp_bin_trainium`: the five functions this kernel needs (`f5` sigmoid, `f7`
  exp, `f9` softplus, `f29` rsqrt, `f36` silu) each live in **exactly one** set and **no set
  holds two of them**; `f30` (square) and `f128` (copy) are in all of them. So 5 loads is the
  floor and iter-1's forced T2 reload makes 6 — **7.7 us of this kernel is unavoidable table
  loading** and that lever is permanently shut. One-liner:
  `{m['func_id'] for m in json.load(open(f))['profile_meta_data']}` over `*.json` there.
* **Filling the head stall with the T1 group.** Moving `[T1 load, f30 square(K_p), f9
  softplus]` in front of `[T4 load, silu]` gave **52.683 us, +2.0 us worse**
  (`head.py`, kept for the record). `f30` waits on the K_p DGE DMA (s24) and `f9` on
  `s6>=4` (which needs the `a` DMA), and both land *later* than `z`, so the stall grew. Any
  fix must use work that waits on nothing, and the only such work is a table load.
* **Placement sweep for the two merged ops** (edits 1-2): both right after DVE@22 = 50.704;
  both just before DVE's first s5 waiter (after every prologue broadcast) = 51.057; f1 early
  + k_f late = 51.137. All within 0.9%, kept the first. `kf.py --late` reproduces variant 2.
  The reasoning that matters: ACT's own early instructions wait on DVE's prologue broadcast
  chain (DVE slots 0-42), so a 1.7 us op inserted in front of it stalls ACT by up to that
  much — but the merged `S *= exp(g)` op cannot start before `exp_g_K` exists (ACT slot 10,
  t≈23 us) anyway, so moving it earlier buys nothing.
* **Merging the line-300 casts** — structurally wrong, see "When a merge pays".

## Method notes

* `profile.py --repeats 5` spread was 0.7-2.3%. Treat <0.5 us as noise; the four accepted
  steps were 1.2-4.6 us each. Re-measure a kept build: I measured the final bytes twice.
* The profiler's `nki_source_location` is **unreliable** after iter-1's reorder. Identify
  instructions by their **fields**: `operands` in `Instruction.parquet` prints the decoded
  form (`scale=[0x4060]` is a scale pointer, `scale=1.000000` is not), and each engine runs
  in order, so the k-th body row (`compiler_pc` not null, sorted by `start_ts`) is the k-th
  NEFF slot. Opcode names differ between the decoder and the profiler
  (`ACTIVATION_TABLE_LOAD` vs `ACT_TABLE_LOAD`).
* The NKI source is on disk and answers every dataflow question in seconds:
  `/home/ubuntu/workspace/gdn-bench/vendor/gdn/gdn_tkg.py` lines 195-340.
* Key addresses, identical in sg00 and sg01: `S_all` 0x1c000+0x200h (128 ch fp32),
  `S_bf` (new) 0x26000+0x100h, `k_f_all` 0x35800+0x100h (1 ch bf16), `K_p` 0x37e00,
  `exp_g_K` 0x4060+4h (128 ch), `beta_K` 0x4090+4h (128 ch), `k_scale` 0x4128+4h,
  `q_scale` 0x40f8+4h, bf16 1.0 (`ones_K`) at 0x40c0, silu block 0x20000+0x200h,
  delta block 0x22000+0x200h. `addr = partition*0x40000 + offset`, PSUM at 0x2000000;
  single-partition tiles are scattered over partitions (0x835800 is partition 32), so the
  partition is `addr >> 18`. Input DMAs: s8 dt_bias, s9 A_log, s10 a, s11 b, s12 v(cast),
  s13 k, s14 norm_weight, s15 z(cast), s16 output; s24-s27 are the Pool-triggered DGE DMAs
  (s25 = the `S_all` bulk load).

## Fingerprint of the final NEFF (for the next review)

`neff_diff.py` against iter-1: 79 of 89 members identical; the 10 that changed are **all five
engine streams of both subgraphs**. `Pool0.bin` and `SP0.bin` change even though no edit
targets them — they carry waits on s4/s6 and the renumbering reaches them, so a correct edit
*must* rewrite waits on all five streams, not just the three it touches.
ACT 140 -> 93 slots (8960 -> 5952 B), DVE 95 -> 110 (6080 -> 7040 B), PE same size
(2 field edits per head), sg00 and sg01 byte-identical in PE.

Reproduction is exact: unpack the iter-1 NEFF, run `kf.py`, `f1.py`, `f3.py`,
`c280.py --groups 1`, repack -> sha256 `7677acb82230c74e0388046082d51fe48bac1372007c81bd53fab4eebe3d5f40`
(checked byte-for-byte against the delivered `kernel.neff`).

# iter-3 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 44.79 us (iter-2, re-measured) -> 40.52 us, -9.5%** (vs the iter-0 baseline 76.054:
**-46.7%**). Four accepted edits, every one pure schedule/placement: `verify.py --runs 4` gives
4 **bit-identical** device runs and `max_abs_err` is **exactly the baseline value** on both
outputs (2.322e-02 = 86% of the limit, 1.633e-03) after every edit. `semaphores.py` clean on
sg00 and sg01 (s4 40/40, s5 38/38, s6 78/78, s3 unused), `cycle.py` acyclic,
`hb.check_psum` 0 unordered PSUM pairs, `check.py` 0 table violations (7 loads/core now).

| # | edit | script | total |
|---|---|---|---|
| — | iter-2 NEFF, re-measured | — | 44.79 (44.36-45.00) |
| 1 | delete the 12 ACT `COPY out_p(PSUM)->o_sb`; both readers read PSUM | `nocopy.py` | **43.07** |
| 2 | make `z` DMA transfer id 1 instead of 7 (swap descs with `A_log`) | `zswap.py --with 1` | **42.21** |
| 3 | 12 line-300 `CAST S->S_bf` from ACT to DVE | `c300b.py` | **41.42** |
| 4 | re-order the ACT head so DVE stops idling 6.4 us | `head3.py` | **40.52** |

The final bytes were measured three times: **40.519** (5 repeats, spread 2.31%),
**40.721** (7 repeats, spread 0.90%, min 40.684) and **40.754** (5 repeats, spread 1.46%) —
call it 40.5-40.8 us. ACT and DVE are now within 0.05 us of each other at 89% busy.
Reproduce: unpack the iter-2 NEFF, run the four scripts in that order, repack.
Final sha256 `4d314f157c0ce724621b121fe31e9442f494ff9b05e7c8abce7a4c5dc55b41cf`.

## The three facts worth carrying forward

### 1. `s4>=v` carries the whole ACT prefix, not just the named producer
This cost me one failed device run and is the single biggest trap in reordering a stream.
A wait `s4>=v` means "ACT completed its first v s4-updating instructions", i.e. (ACT being
in-order) **every ACT instruction up to the v-th updater**. iter-1/iter-2's identity-keyed
renumbering ("resolve the wait to the producer object, recount after the edit") is correct
only for *insertions and deletions*. The moment you **re-order** ACT, the identity rule
silently drops the implicit coverage of everything that used to precede the producer.

Concretely: `DVE@5` (`g = -dt*exp(A_log)`) waits `s4>=5`, whose named producer is
`f7(A_log)` — but its *other* input, the in-place `f9` softplus, was the 3rd updater and was
covered only by ACT program order. Moving `f7(A_log)` to the front and `f9` after it kept
the wait valid by identity and produced garbage (`max_abs_err` 2.57, cosine 0.95).

The fix in `head3.py` is a **prefix closure**: for a wait whose producer is at old index j,
`v' = max over x in act[0..j] of (new count of s4 updaters at-or-before x)`. Two refinements
matter:
* **Exclude `ACTIVATION_TABLE_LOAD` from the closure.** It updates no semaphore and carries
  no data; its only effect is ACT engine state, consumed only by ACT `ACTIVATE`s whose order
  relative to their load you preserve anyway (`check.py` proves it). Without this exclusion
  the closure is needlessly strong and drags waits forward.
* The closure is **conservative**, and 5 of the ~71 s4 waits needed a hand override back to
  the real data dependency, each justified from the operands (see `OVERRIDE` in `head3.py`,
  with the reasoning in comments). One of them (`PE@2`) is not optional: the closure value
  there is a **cycle** (`PE@2` -> silu -> ... -> f29k -> PE@3 -> PE@2) and `cycle.py` catches it.

### 2. PSUM is readable by ACT and DVE, so the `o_sb` COPY is dead weight
`o_sb` existed only so two instructions could read `out_p`: the ACT `activation_reduce(square)`
and the DVE `scalar_tensor_tensor(xf = o_sb*ssum*norm_weight)`. **Both engines read PSUM
directly** (ACT already did, in the COPY itself; DVE reads `kv_p`/`outer_p` from PSUM), and
reading the same fp32 words from PSUM is bit-exact. Deleting the 12 COPYs removed 3.0 us of
ACT work *and* one step from every head's serial tail, for -1.7 us of total.

The feared blocker — the 8 PSUM banks are each reused by 4-5 matmuls, so extending a tile's
live range needs a WAR guard — **turned out to be free**: all 8 constraints were already
implied by the existing happens-before graph (`war.py` proved it by reachability), and 4 of
the 12 tiles are never overwritten at all. The right tool is `hb.py:check_psum`, which checks
*every* (reader, writer) and (writer, writer) pair on every PSUM address for happens-before
ordering; the compiler's own output scores 0 unordered pairs, so it is a clean regression test.
Repairs needed: give each `f30` its COPY's `s5` wait (asserted to resolve exactly to the
producing MATMUL), re-point the DVE `xf` src0 to PSUM, and re-point the 2 incidental PE
`LDWEIGHTS` waits to the next surviving s4 updater.

### 3. Re-ordering `PSEUDO_DMATRIGGER` does nothing; re-order the descriptors
iter-2 parked "reorder the SP queue so `z` goes first" as a candidate. Moving the z trigger
from SP slot 8 to slot 1 **changed nothing at all** (measured: 43.33 vs 43.07, and
`DmaPacketAggregated` shows the transfers still starting in `block_id` order 0..7, ~190 ns
apart). The DMA engine walks the ring in descriptor-**id** order no matter which trigger rang
the doorbell. What works is swapping the `desc` arrays in `SP0.json` between two transfers.
Two gotchas:
* A transfer's semaphore is fixed by `id % 16`, so the two transfers' **waiters must swap
  wait indices** too (`zswap.py` does this for all five streams).
* **Some consumers rely on ring order instead of a semaphore.** Swapping z (id 7) with
  `dt_bias` (id 0, whose semaphore s8 has *zero* waiters) FAILED verify: `dt_bias` is read by
  `DVE@4`, which waits on **s10** (the `a` transfer, id 2) and gets `dt_bias` only because
  id 0 < id 2. So a transfer may only be moved *later* if every reader of it has an explicit
  wait. Swapping z with `A_log` (id 1, explicit waiter `ACT@9`, and `A_log` is not needed
  until much later) works and gave -0.8 us.

## Where the time is now (40.52 us)

| engine | body busy | body idle | body span |
|---|---|---|---|
| ACT / Scalar | 26.31 | 11.86 | 12.29 - 49.00 |
| DVE / Vector | 27.37 | 12.24 | 12.18 - 49.69 |
| PE / Tensor | 14.53 | 27.53 | 12.36 - 48.28 |
| Pool / GpSimd | 5.07 (+ ~4.8 DGE) | — | 13.02 - 54.95 |

(`profile.py`'s per-engine "busy" includes ~7.7 us of non-NEFF runtime prologue on ACT and
~8 us on DVE — `compiler_pc` is NaN for those. Use the body numbers above, from
`Instruction.parquet` filtered on `compiler_pc.notna()`.)

**The kernel is now chain-latency bound with ACT and DVE both ~87% busy and balanced.**
Perfectly packed it would be `12.2 (prologue) + max(26.3, 27.4) ≈ 39.6` of body + the ~1 us
tail, i.e. about 30.5 us of `total_exec` — so ~10 us of pure chain latency is left, spread
over 12 head chains of 11 steps:
`DVE delta -> ACT delta_bf cast -> PE outer_p -> DVE S+=outer -> DVE S_bf cast -> PE out_p ->
ACT f30 square-reduce -> ACT read_acc -> ACT f29 rsqrt -> DVE xf -> DVE out_sb`.

ACT's 11.86 us of body idle: 1.91 + 1.45 before two `delta_bf` casts (waiting s6 = the DVE
delta op); 1.42 + 1.20 + 1.06 before three `f30` square-reduces (waiting s5 = PE `out_p`);
0.90 before `f9` (the `a` input DMA); 0.64 + 0.62 on `EVENT_SEMAPHORE`s.
DVE's 12.24: 1.82 on the `a` DMA, 1.80 on `f7(A_log)`, 1.81 on `f29k`, 0.99 on the S_all DGE.

**The head is now tight and near its floor.** Two independent hard limits:
* `DVE@4` (`dt = a + dt_bias`) cannot run before the `a` transfer lands (~14.5 us), and
  `f9` softplus is in-place on its output.
* `f29k` (k_scale) needs `f30(K_p) -> PE k_ss -> f29`, and `K_p` arrives on the **s24 DGE
  DMA at ~17.7 us**, because **Pool issues the three input DGE descriptors serially at
  ~1.24 us each starting at 13.72** (`GpSimd` pc=1/2/3 in `Instruction.parquet`:
  13.72-14.95 s24, 14.96-16.20 s25 = the 786 KB `S_all` bulk load, 16.20-17.45 s26 = Q_p).
  ACT's head now fills that window almost exactly.

## Ranked candidates for iter-4

1. **Shorten Pool's serial DGE prologue.** s24 (K_p) gates the whole k_scale branch of the
   head, and it cannot land before Pool has spent 1.24 us generating its descriptor starting
   at 13.72. `PSEUDO_DMA_DIRECT2D` is legal on **SP and ACT as well as Pool** (ch. 9, and the
   doc corpus has 18k on SP / 16k on ACT). SP's stream is 9 slots and nearly idle. Moving the
   s25 (`S_all`, 786 KB) and/or s26 (Q_p) descriptor issue to SP would let them overlap
   instead of queueing behind each other — worth ~1.2 us on the Q_p branch, and it is the
   only lever on the 12.2 us prologue I found that is not already tight.
   Unknown: whether a DGE descriptor's `src_table_index` is engine-relative (the four
   descriptors use indices 0..3 of some table) — check that before moving one.
2. **Fill the three 1.0-1.4 us ACT stalls before the `f30` square-reduces** (3.7 us). They
   wait `s5` = the PE `out_p` matmul of their own head, i.e. real chain latency; the fix is
   to interleave the 12 head chains better on ACT (static list-scheduling over the
   happens-before graph) so ACT always has another head's work ready. This is the largest
   single remaining block of idle but needs the prefix-closure machinery of fact 1 applied to
   a full reorder of the ACT body, not just the head. `hb.py` has the closure helper.
3. **The 12 `f30` square-reduce + 12 `ACTIVATION_READ_ACCUMULATOR` (7.2 us of ACT).**
   Still the biggest single ACT block. iter-2 called collapsing them "probably closed"
   because all 12 `out_p` PSUM tiles sit on partition 0. Now that the readers read PSUM
   directly, re-check whether the 12 reduces can be done as one 12-channel reduce by having
   PE write the 12 `out_p` to 12 *different partitions* — `nc_matmul(out_p, q_p, S_bf)` has
   a 1-row output, and the dst partition is chosen by the matmul's dst address, so 12 heads
   could land on 12 partitions of one bank. Then one `ACTIVATE f30` + one
   `READ_ACCUMULATOR` over 12 channels replaces 24 instructions (-6.6 us). The blocker to
   settle: the DVE `xf` op reads `out_p` and `ssum` and writes `out_sb` on partition 0.
4. **Pool as a work engine: open but slow.** iter-1 reported that inserting instructions into
   Pool hangs. **It does not**, if you insert *after* the third `PSEUDO_DMA_DIRECT2D` and
   before the fourth (`pool.py`): the 12 line-300 casts ran there correctly and verified
   bit-exact. iter-1's hang was almost certainly a *semantic* deadlock from inserting before
   an input DGE descriptor (its cycle checker models DMA semaphores as free). But **a Pool
   `CAST` of 128 elements costs ~725 ns against ACT's ~290** (Pool busy went 9.8 -> 18.5 us
   for 12 casts), so Pool is only worth using for work that is genuinely off the critical
   chain. Moving chain-critical casts there cost +1.1 us.

## Closed this iteration (do not retry)

* **Re-ordering `PSEUDO_DMATRIGGER` slots** to re-order the DMA queue: no effect, see fact 3.
* **Swapping the `z` descriptor with `dt_bias` (id 0)**: FAILS verify — `dt_bias` has no
  waiter and `DVE@4` depends on ring order. Use `A_log` (id 1).
* **The 12 line-300 casts on Pool**: runs, bit-exact, but 43.07 -> 44.19 (Pool is 2.5x slower
  per instruction).
* **A 6-table-load ACT head** (`head4.py`, `f29k` before `f7A` so the two `exp` share one T0
  load): 41.41 vs 40.52. Saving a 1.28 us table load is not worth delaying DVE's first real
  instruction by ~3.6 us. The 7-load head in `head3.py` is the better trade.
* **Identity-keyed semaphore renumbering across a reorder**: produces a silent race, fact 1.
* **The 12 `CAST delta -> delta_bf` moved from ACT to DVE** (`deltabf.py`, the same
  transformation as edit 3 one link earlier in the chain): runs, `verify.py --runs 3`
  bit-identical with the baseline `max_abs_err`, but **40.72 -> 41.19**. It takes ACT down to
  74.2% busy (30.56) and pushes DVE to 87.3% (35.97), so the two cross-engine hops it removes
  per head do not pay for the imbalance. **This is the useful negative result of the
  iteration: at 40.5 us ACT (26.31 body) and DVE (27.37 body) are balanced to within 1 us,
  and moving elementwise work either way just makes the receiving engine the bound.**
  Everything that is left has to either remove work outright (candidate 3) or fill existing
  idle without adding any (candidates 1 and 2). `deltabf.py` is kept: its mapping was the
  hard part and is all asserted (the 12 `delta_bf` dsts are shared -- 0x4170 by four heads --
  so each gets a private tile at 0x28000+0x100h; cast->MATMUL pairs by order, both streams
  monotone; 4 of 12 casts have no s4 waiter and 2 have two, where the extra waiter is the
  proxy for *another* pair's cast, leaving exactly 2 PE `EVENT_SEMAPHORE` insertions).
  It also contains the reusable **aliased-scratch rule**: a buffer may legally have several
  unrelated writers, so before re-pointing one writer, check for every other reader of that
  address that *the last writer before it in happens-before order is not the one you are
  moving*. Three DVE `out_sb` ops read `delta_bf` addresses in their `xf` role and would
  have tripped a naive "no other readers" assert.

## Method notes

* `profile.py --repeats 5` spread was 0.45-3.51% this iteration (the 3.51% run was a bad
  outlier; the same bytes re-measured at 0.55%). Treat <0.5 us as noise and re-measure any
  candidate under ~0.8 us. The four accepted steps were 0.8-1.7 us each.
* **Attribute instructions by their decoded fields, never by `nki_source_location`.**
  `Instruction.parquet`'s `operands` column prints the decoded form (addresses, `channels`,
  `FUNC_30`, `scale=`), and each engine runs in program order, so the k-th body row sorted by
  `start_ts` with `compiler_pc` not null is the k-th NEFF slot. `compiler_pc` is reliable and
  equals the slot index.
* The per-engine gap table that found every win this iteration:
  `a = Instruction[(pcore_idx==0)&(engine==E)].sort_values('start_ts')`, then
  `gap = start - end.cummax().shift(1)`, `nlargest(8,'gap')`, restricted to
  `compiler_pc.notna()`. Look at the *waiting* instruction's `operands` to see which
  semaphore it is blocked on.
* `DmaPacketAggregated.parquet` gives per-transfer `block_id`, `variable`, start/end — the
  direct way to see input-DMA ordering. `GpSimd` rows in `Instruction.parquet` give the DGE
  descriptor issue times.
* NKI source: `/home/ubuntu/workspace/gdn-bench/vendor/gdn/gdn_tkg.py` lines 195-340.
* Key addresses (identical in sg00/sg01 unless noted): `S_all` 0x1c000+0x200h,
  `S_bf`(line-280, merged) 0x26000+0x100h, delta 0x22000+0x200h, silu 0x20000+0x200h,
  `k_f_all` 0x35800+0x100h, `K_p` 0x37e00, `Q_p` 0x37f00, `dt` 0x4000, `exp(A_log)` 0x4030,
  `exp_g_K` 0x4060+4h, `beta_K` 0x4090+4h, `ones_K` 0x40c0, `K_p^2` 0x40e0, `Q_p^2` 0x40c8,
  `k_scale` 0x4128+4h, `q_scale` 0x40f8+4h, `q_p` 0x4158+2h, `norm_weight` 0x36400,
  `dt_bias` 0x35500, `a`/`g` 0x35700, `b` 0x35600. `addr = partition*0x40000 + offset`,
  PSUM `0x2000000 + bank*0x800` (8 banks). Input DMAs after `zswap`: id0 dt_bias/s8,
  **id1 z/s9**, id2 a/s10, id3 b/s11, id4 v/s12, id5 k/s13, id6 norm_weight/s14,
  **id7 A_log/s15**, id8 output/s16. DGE: s24 K_p, s25 S_all, s26 Q_p, s27 output state.
* The ACT head order after `head3.py` (19 head slots + a cloned T0 load), with what each
  one is gated on:
  `SET_ORDERING | T1 | f9 softplus (s6: DVE dt) | T0 | f7 exp(A_log) (s15) | T2 |
   f30 K_p^2 (s24) | f29 k_scale (s5: PE k_ss) | T0 again | f7 exp_g (s6: DVE g-bcast) |
   T4 | silu (s9: z) | ES(s25) | wide CAST S->S_bf (s6) | f30 Q_p^2 (s26) | T3 |
   f5 sigmoid->beta_K (s6) | ES(s6) | T2 | f29 q_scale (s5: PE q_ss)` then the 12-head body.

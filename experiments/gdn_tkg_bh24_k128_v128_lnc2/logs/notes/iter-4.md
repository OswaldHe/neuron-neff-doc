# iter-4 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 41.02 us (iter-3 NEFF, re-measured here) -> 38.63 us, -5.8%** (vs the iter-0
baseline 76.054: **-49.2%**). Three accepted edits, all pure schedule/placement:
`verify.py --runs 8` gives 8 **bit-identical** device runs and `max_abs_err` is **exactly the
baseline value** on both outputs (2.322e-02 = 86% of the limit, 1.633e-03). `semaphores.py`
clean on both subgraphs (s4 41/41, s5 38/38, s6 79/79, s27 64/64), `cycle.py` acyclic,
`hb.check_psum` 0 unordered PSUM pairs, `check.py` 0 table-set violations (still 7 loads/core).
Final sha256 `365b6114c86b5de1c09173d8415efb0c3d70e1625cc7ae72e909c51861f0bf3d`.

| # | edit | script | median (repeats) |
|---|---|---|---|
| — | iter-3 NEFF, re-measured | — | 41.020 (5) / 40.74 recorded by iter-3 |
| 1 | split the state-out DGE DMA into 4 head groups | `split.py` | **39.368** (5), 39.210 (7) |
| 2 | DVE reorder + split `S*=exp_g` / `CAST S->S_bf` in halves | `pipe.py` | 39.367 (7) — neutral alone |
| 3 | hoist the ACT T3 load + `f5 sigmoid` ahead of the two casts | `beta.py` | **38.633** (7), 38.631 (5) |

Reproduce: unpack the iter-3 NEFF, run `split.py`, `pipe.py`, `beta.py` in that order, repack.
Edit 2 is worth nothing by itself but edit 3 *cannot be made without it* (see below), and
2+3 together are -0.58 us on top of edit 1.

## The four facts worth carrying forward

### 1. The tail was 2.6 us of state write-out that nothing overlapped (edit 1, -1.65 us)
This is review candidate C1 and it is real. Pool's single `PSEUDO_DMA_DIRECT2D` (slot 4,
sem 27) wrote all 12 heads of `S_all` (786 KB) *after* the last compute instruction; its
`Q0` packets ran 50.18->52.46 on `sg00` and 51.2->53.55 on `sg01`, and they also delayed the
tiny `output_0` queued transfer by ~1.9 us through DMA-engine contention.

Split into 4 groups `[(0,4),(4,4),(8,2),(10,2)]`, each waiting on the DVE s6 count at which
its last head's state tile is final. **I re-derived the counts myself** (`sall.py`, by
scanning every operand address in `Instruction.parquet` for `0x1c000+0x200h`): head *h*'s
last write is DVE `TENSOR_TENSOR ADD` at s6 count
`[21, 28, 30, 32, 39, 42, 48, 53, 62, 64, 70, 74]` (pre-`pipe` numbering; +1 after `pipe`),
identical in both subgraphs, and nothing writes those tiles afterwards — the 12 later
references are the `CAST S->S_bf` reads. Head 11 = 74 = the original wait, which confirms
the mapping.

Field mechanics (`split.py`): add `512*h0` to `src_start_addr_addr_immediate`,
`65536*h0` to `dst_start_addr_addr_immediate` (the `*_table_offset` / `*_var_offset` /
`*_reg_*` fields the decoder prints are **overlapping views of the same bits**, so setting
the immediate is enough), set `src_num_elem[0] = dst_num_elem[0] = n`, and raise the final
`NOP`'s `s27` wait from 16 to `16*N`. `Pool0.json`'s `dma` list is empty, so growing
`Pool0.bin` needs no other bookkeeping.

Two measured corrections to the review's cost model:
* **DGE descriptor generation is ~700 ns fixed + ~0.36 ns/descriptor**, not a flat 1.25 us:
  1536 desc -> 1252 ns, 512 -> 939 ns, 256 -> 816 ns. So splitting is cheap, and Pool is
  idle anyway.
* The chunked write-out costs `sg00` about 1.0 us of extra compute time (DVE's body end went
  49.82 -> 50.86) because the DMA now reads SBUF while DVE is working. It is still a big net
  win, and it made the two cores symmetric (they used to differ by 1 us).
* **Group geometry barely matters**: 4 groups and `[(0,2),(2,10)]`-style variants land within
  0.01 us of each other once the tail is off the critical path. Don't spend time tuning it.

### 2. `beta_K` was the body's real starting gun, not S_bf (edit 3, -0.73 us)
All twelve DVE `TENSOR_TENSOR ... src1=0x4090+4h` ops (the `* beta_K` step of each head)
wait `s4>=10` = ACT's `f5 sigmoid`. That one ACTIVATE sat behind the T3 table load, which
sat behind the two S_bf casts, so beta_K was not ready until 26.49 us and DVE — which had
already reached its first body instruction at 24.35 — stalled 1.9 us.

Moving ACT slots 16 (`ACTIVATION_TABLE_LOAD table_sel=3`) and 17 (`f5`) in front of slot 13
(the first cast) costs the casts ~0.9 us (only PE feels it, 43% busy with slack to the next
head) and buys beta_K 2.4 us earlier. `f30 Q_p^2` then runs with T3 loaded instead of T4 —
legal, see fact 4.

This is an ACT **re-order**, so iter-3's fact 1 (prefix closure, not producer identity)
applies. The updater permutation is 7->8 (cast0), 8->9 (cast1), 9->10 (f30 Q_p^2), 10->7
(f5). The closure value for the twelve old `s4>=10` waiters is 10 (their old prefix contains
the casts); each is **overridden to 7** because its only ACT-side operand is `beta_K`
(asserted in `beta.py`), S_bf is read only by PE (whose own `s4` waits follow the casts), and
`Q_p^2` is read only by PE slot 10, gated by PE slot 9's wait. Verified bit-identical.

### 3. The 12 PE `LDWEIGHTS` are a ready-made per-head hook for splitting the wide cast
The merged `CAST S(fp32,1536) -> S_bf(bf16,1536)` (ACT slot 13) and the merged
`S *= exp_g` (DVE slot 24) are both plain elementwise ops over 12 contiguous per-head tiles
(`S_all` head stride 512 B, `S_bf` head stride 256 B, `exp_g` stride 4 B), so splitting them
at any head boundary is bit-exact: `src/dst` immediate += `512*h0 / 256*h0 / 4*h0`, element
count = `128*n`.

The consumers are trivially re-pointable: **each of the 12 `MATMUL`s that read
`0x26000+0x100h` is immediately preceded by an `LDWEIGHTS` that waits `s4>=7`** (PE slots
5,7,13,15,32,34,44,48,54,58,66,72 -> heads 0..11 in order). That single wait is the entire
S_bf gate, so an N-way split just needs those 12 waits set to the chunk that produces their
head. Nothing else in the NEFF waits on `s4>=7`.

The DVE side of `pipe.py` also moved slot 25 (`k_f_all *= k_scale`, 1.74 us) and slots 26..33
(the `b` broadcast: 4 `LOAD_MASK_SELECT`+`STREAM_SHUFFLE` pairs) ahead of slot 23, the
`EVENT_SEMAPHORE s25>=16` that waits for the 786 KB `S_all` input load. Neither needs `S_all`
nor slot 24's output, and that filled DVE's 2.1 us s25 stall. The s6 renumbering had exactly
one waiter per affected value and each was justified from the operands:
`s6>=12 -> PE[11] LDWEIGHTS src=0x358fe` (k_f_all) -> 11;
`s6>=13 -> DVE[28] LOAD_MASK_SELECT` (inside the moved block) -> 12;
`s6>=14 -> ACT[16] ACTIVATE src=0x35600` (b broadcast) -> 13;
`s6>=11 -> ACT[13] CAST` -> the two halves (14, 15); `>=15` -> +1.

### 4. `square_1p` (f30) is in **all five** activation table sets
From the five `*_and_others.json` `profile_meta_data` lists:
* `f7 exp` -> **T0** only (`natural_log_exp_and_others`)
* `f9 softplus` -> **T1** only
* `f29 reciprocal_sqrt` -> **T2** only
* `f5 sigmoid` -> **T3** only (`tanh`, `erf`, `arctan` are also there)
* `f36 silu` -> **T4** only (`tanh`, `sin` also there)
* `f30 square`, `identity`, `relu`, `leaky_relu`, `prelu`, `sign`, `abs`, `is_finite`,
  `copy`, `memset_zero` -> **every set**.
So the two `f30` ops in the head and the 12 in the body never constrain the table order, and
any ACT re-order only has to respect T0/T1/T2/T3/T4 for the five unique functions.

## Where the time is now (38.63 us)

`total_exec` 38.63 maps onto the `sg00` timeline as: a **~11.9 us non-NEFF runtime prologue**
(two 8-engine `$S[2]` barriers plus `TENSOR_LOAD`s, all `compiler_pc` NaN — not in any stream
we can edit), then body 11.94 -> 49.22, then a 1.1 us `output_0` DMA tail, then the epilogue
barrier.

| engine | body busy | body span | idle in body |
|---|---|---|---|
| DVE / Vector | 27.51 | 11.84 - 49.22 | 12.0 (gap sum) |
| ACT / Scalar | 26.49 | 11.94 - 48.53 | 11.7 |
| PE / Tensor | 14.45 | 12.02 - 47.80 | 27.7 |
| Pool / GpSimd | 7.33 (+ ~6 DGE) | 11.86 - 50.71 | — |

(`profile.py`'s per-engine "busy" includes the runtime prologue. Use
`Instruction.parquet` filtered on `compiler_pc.notna()` — `cmp.py` prints the table above.)

**Head, 11.94 -> 24.77 (DVE's first body instruction).** ACT is ~100% busy the whole way and
is the gate; the exact ACT head, with what each slot waits on:
```
 1 T1 12.008-13.291 | (0.87 idle: DVE dt waits the `a` DMA, 13.97)
 2 f9 softplus 14.162-14.373      | 3 T0 14.390-15.673
 4 f7 exp(A_log) 15.685-15.896 s15| 5 T2 15.914-17.197
 6 f30 K_p^2 17.209-17.421  s24   | 7 f29 k_scale 17.679-17.850 s5
 8 T0 again 17.868-19.151         | 9 f7 exp_g 19.162-19.374 s6>=8
10 T4 19.392-20.675               |11 silu(1536) 20.687-22.169 s9
12 ES s25 20.789                  |13 T3 22.186-23.469
14 f5 sigmoid 23.481-23.693 s6>=13|15 cast0 23.600-24.413 s6>=14
16 cast1 24.321-25.147 s6>=15     |17 f30 Q_p^2 25.054-25.260 s26
18 ES s6>=16                      |19 T2 25.278-26.561
20 f29 q_scale 26.573-26.744 s5   |21.. body
```
7 table loads = **8.98 us of the 14.8 us ACT head**. DVE's head work is only ~5.5 us; its
three remaining >1 us stalls all wait on ACT (`s4>=2` = f7(A_log) at 15.9, `s4>=4` = f29
k_scale at 17.9) or on the `a` input DMA.

**Body, 24.77 -> 49.22 (24.4 us)**, DVE 87% / ACT 83% busy. It is an ACT<->DVE<->PE
ping-pong over 12 independent head chains of 11 steps. ACT's remaining idle is five stalls of
1.0-1.9 us, each waiting `s5` (the PE `out_p` matmul of the head it is about to reduce), i.e.
real chain latency, with another head's work sitting further down ACT's in-order stream.

**Tail, 1.1 us.** Last DVE op 49.22 -> SP doorbell -> `output_0` queued transfer
(3136 B, 16 engines) 49.97-50.31 (`sg00`) / 50.25-50.57 (`sg01`) -> Pool's `s16`/`s27` waits
-> epilogue. The state write-out is no longer on the critical path.

## Measured DMA facts (new this iteration)

* **The 786 KB `S_all` input DGE load reads at ~144 GB/s; the identical 786 KB write-out
  writes at ~393 GB/s.** Per engine (16 of them, `Q0-E64..E79`): 96 descriptors of 512 B in
  5.1-5.5 us on the read (≈53 ns/descriptor) vs 2.0 us on the write (≈21 ns). The read looks
  per-descriptor-latency bound, and the descriptor size is pinned at 512 B by the SBUF
  partition scatter (head *h*, partition *j*: HBM `base+65536h+512j` -> SBUF
  `0x1c000+512h` on partition *j*).
* Pool issues the three input DGEs serially: s24 `K_p` 12.58-13.83, s25 `S_all` 13.84-15.09,
  s26 `Q_p` 15.08-16.33; their transfers then run in ring order, so `K_p` lands ~16.4,
  `S_all` ~21.2 and `Q_p` ~21.4 (`Q_p` is stuck behind the 786 KB).
* `Q0` (Pool's DGE queue) and `qSyncIO0` (SP's queued transfers) use the **same 16 DMA
  engines**, which is why the old monolithic write-out delayed `output_0` by 1.9 us.

## Ranked candidates for iter-5

1. **Static list-scheduling of the ACT body over the happens-before graph (the largest
   single block left: ~7.5 us of ACT idle, probably ~2-3 us recoverable).** ACT's five
   1.0-1.9 us stalls each wait for its own head's `out_p`; the fix is to interleave the 12
   chains so ACT always has another head's `CAST delta_bf` / `f30` ready. `hb.py` has the
   closure helper and `beta.py` is a worked 2-slot example of the override discipline. Do it
   incrementally (move 2-4 slots, measure) — a whole-body permutation is where iter-3 lost a
   device run.
2. **Make the `S_all` input load land earlier (~1.2 us, low risk).** Split the s25 input DGE
   into two 6-head halves and issue them as `S_all_A, K_p, S_all_B, Q_p` (Pool slots 1..4).
   Predicted: `K_p` ~16.7 (unchanged in effect), `S_all` complete ~19.2 instead of 21.2,
   `Q_p` ~19.5 instead of 21.4. Put **both halves on sem 25** (total 32) and raise DVE's
   `EVENT_SEMAPHORE s25>=16` to `>=32`: waiting for the *total* is always safe, whereas
   `>=16` with two DGEs on one semaphore is a race (one engine could signal twice before a
   slow one signals once). **Caveat measured this iteration:** after `pipe` the DVE halves
   start at 21.74 because DVE's own work runs to 21.70, not because of s25 (satisfied at
   21.20) — so this is worth ~0 *unless* combined with candidate 3 or 1. `Q_p` landing 2 us
   earlier does open ACT re-orderings (`f30 Q_p^2` currently cannot run before 21.4).
3. **Get ACT from 7 table loads to 6 (1.28 us of ACT head).** The removable one is the second
   T0 (slot 8), which exists only because `f7 exp_g` must follow `f29 k_scale` in ACT order.
   `f7 exp_g`'s real input is the DVE `g` broadcast (`0x35700` -> `0x835700/0x1035700/
   0x1835700`), whose 4 shuffle pairs are currently *interleaved* with the 4 `k_scale`
   shuffle pairs and therefore gated by `f29 k_scale` (DVE slot 6 waits `s4>=4`). Reorder DVE
   so the four `g` pairs come first (they need only DVE slot 5), then ACT can be
   `T1 f9 | T0 f7A f7exp_g | T2 f30K f29k | T4 silu | T3 f5 | casts | T2 f29q`.
   **I worked the arithmetic and it is close to a wash**: ACT gains 1.28 us but acquires a
   ~0.85 us idle wait for the g broadcast (DVE `g` at 15.93 + 4x0.165), so f5 ends ~23.3
   instead of 23.69 -> net ~-0.35 us. Only do it *together* with candidate 2, which removes
   the other side of the trade.
4. **Collapse the 12 `f30` square-reduce + 12 `ACTIVATION_READ_ACCUMULATOR` (6.6 us of ACT).**
   Still open (iter-3's candidate 3) but now **less** attractive: ACT is at 83% and DVE at
   87%, so removing ACT work does not shorten the span, and merging 12 reduces into one wide
   op *lengthens* every head chain (the wide op cannot start until all 12 `out_p` exist).
   Only worth it if paired with candidate 1 making ACT the bound again.
5. **Split the `output_0` queued transfer (1.1 us tail).** Probably not worth it: the
   transfer is 3136 B over 16 engines and is latency- not size-bound, so halving it buys
   maybe 0.2 us, and adding a transfer id changes the `semaphore_set[id%16]` assignment
   (`semaphore_set[k] = 8+k` here, so a new id 9 would take the free s17) plus a second Pool
   completion wait.

## Closed this iteration (do not retry)

* **Splitting the state-out DMA into different group geometries.** 4 groups
  `[(0,4),(4,4),(8,2),(10,2)]`, and 2-head / 10-head first halves of the cast split, all land
  within 0.01 us. The tail is off the critical path; stop tuning.
* **`pipe.py` on its own**: 39.37 vs 39.21 for `split.py` alone, i.e. neutral-to-slightly-
  worse. It moves DVE's body start 26.13 -> 24.35 but DVE then stalls 1.9 us on `beta_K`, so
  the gain is invisible until `beta.py` lands. Keep them together.
* **Swapping the `z` (id 1, 6208 B) and `b` (id 3, 112 B) input descriptors** so that `a`
  (id 2) is not queued behind z: `bswap.py`. Verifies bit-exact, `semaphores.py` clean,
  measured **38.659 vs 38.633** — no effect. The 0.87 us ACT idle at the head is the `a`
  transfer's own latency, not ring position. (iter-3's `z`<->`A_log` swap is the same lever
  and iter-3 already noted it had become neutral by analysis.)
* **A 2-head first half for the cast/multiply split** (`NH=2 pipe.py`): 38.627 vs 38.633 for
  `NH=6`. `cast0`'s start is gated by ACT finishing `f5`, not by its own length.
* **Reordering the three input DGEs on Pool without splitting** (`S_all` first): analysed,
  not run. `K_p` is tiny but queues behind `S_all`'s 1536 descriptors in each engine's ring,
  so it would land ~19.3 instead of 16.4 and delay the whole `k_scale` branch by more than
  `S_all` gains. The two chains are already balanced to within 0.3 us.

## Method notes

* `profile.py --repeats 7` spread was 0.69-1.78% this iteration; **0.3 us is the noise floor**
  at 5 repeats. Two of my three candidate comparisons (39.37 vs 39.21, 38.63 vs 38.66) were
  inside it and needed 7 repeats to call. Re-measure anything under 0.5 us.
* Check order for every edit, all of which caught nothing this time but are cheap:
  `cycle.py` (unmet waits + deadlock), `semaphores.py` both subgraphs, `hb.py` (PSUM
  ordering), `check.py` (table-set validity), then `verify.py --runs 3`, then `profile.py`.
* `isa.Instruction.set(name, value)` writes exactly the named bitfield; the many
  `*_table_offset` / `*_var_offset` / `*_reg_reg_hi` fields the decoder prints for a DGE
  address are aliases of the one 64-bit immediate, so set the immediate and ignore them.
* The analysis scripts in `iter-4-tools/` that found everything: `cmp.py` (body busy/span per
  core), `gap.py` (top-10 idle gaps per engine with the blocked instruction's operands),
  `win.py TAG lo hi core` (a merged all-engine timeline slice — the single most useful one),
  `sem6.py` (DVE s6 count -> wall-clock time), `sall.py` (every reference to an SBUF address
  range, with its s6 count), `waits.py`/`w2.py`/`w3.py` (who waits on `sN>=v`, and which slot
  produces it), `dma2.py` (per-queue/per-engine DMA packet aggregation), `pre.py`/`tail.py`
  (the runtime prologue/epilogue).
* Addresses unchanged from iter-3, plus: `beta_K` 0x4090+4h, `q_scale` 0x40f8, `K_p^2` 0x40e0,
  `Q_p^2` 0x40c8, `b` 0x35600 (broadcast to 0x835600/0x1035600/0x1835600), `g` 0x35700
  (same pattern), `k_scale` 0x4128, `exp_g` 0x4060+4h, `S_bf` 0x26000+0x100h,
  `S_all` 0x1c000+0x200h, `k_f_all` 0x35800+0x100h.
* After this iteration the stream lengths are ACT 71, DVE 123, PE 80, Pool 10, SP 10 slots
  per subgraph, and `sg00`/`sg01` differ only in the DMA base offsets (as before).

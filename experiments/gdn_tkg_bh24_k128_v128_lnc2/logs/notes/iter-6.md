# iter-6 notes — gdn_tkg bh24 k128 v128 LNC=2

**Result: 38.217 us (iter-5 NEFF, re-measured here) -> 36.772 us, -3.8 %** (vs the iter-0
baseline 76.054: **-51.7 %**). Final sha256
`c3ba6aeabe7746075e8c7677757ec13cd844b5b607107034d7e326c1192c9a46`.

Two kept edits, both pure scheduling, both bit-identical (`max_abs_err` exactly the baseline
2.322e-02 on output_0, 1.633e-03 on output_1, `verify.py --runs 5`):

| # | edit | script | median (repeats) | verdict |
|---|---|---|---|---|
| — | iter-5 NEFF, re-measured | — | 38.217 (5) | |
| A | both wide head casts -> Pool | `pcast.py` | 41.015 (7) | **+2.8, reverted** |
| B | cast2 -> Pool only, + ACT barrier moved | `MODE=pool2 head6.py` | 38.418 (7) | +0.2, reverted |
| **C** | **cast2 -> ACT's own body gap, + ACT barrier moved behind `f29 q_scale`** | `MODE=act2 head6.py` | **37.572 (7)** | **kept, -0.645** |
| **D** | **cast1 split by consumer deadline: heads 0-3 -> DVE, heads 4-5 -> the same ACT gap** | `split6.py` | **36.831 (7)** | **kept, -0.74** |
| E | iter-5's `fuse.py` on top of C+D | `fuse.py` | 36.828 (7) | neutral, reverted |
| F | 4 DVE `out*silu` MULTs -> Pool | `N=4 poolout.py` | — | **VERIFY FAILED, reverted** |

Final confirming run with the metric command (`profile.py --repeats 5`): **36.772 us, spread
0.85 %**, min 36.665 max 36.980. The -1.445 us is ~4.7x the run-to-run spread and was
predicted ahead of each measurement (-0.73 for C, -0.70 for D), so it is not noise.

## The idea that unlocked it: a slot on ACT is not on the critical path *because of its data*

iter-5 established that the kernel is on its critical path essentially everywhere (`crit.py`:
35.26 of the 36.8 us) and that the ACT head (12.3 -> 25.8 us, 93 % busy, 7 x 1.283 us
`ACTIVATION_TABLE_LOAD`) is the critical chain in the head. What iter-5 did **not** check is
whether every ACT slot in that chain *needs* to be there. Two of them did not:

* The two wide `CAST S_all(fp32) -> S_bf(bf16)` (128 channels x 768 elements, 0.813 and
  0.826 us) sat back-to-back in the middle of the head chain, between `f5 sigmoid` and the
  2nd T2 table load.
* **Their readers are 7-10 us later.** `cast2` (heads 6-11, `0x1cc00 -> 0x26600`) is read only
  by PE44/48/54/58/66/72, the first at 35.2 us. `cast1` covers heads 0-5 but is read in two
  waves: PE5/7/13/15 take heads 0-3 at 25.1-26.9 us, PE32/34 take heads 4-5 at 31.2 us.

So both casts were on the chain purely because ACT is in-order. Moving them to places with
slack bought 1.39 us with no work removed from any engine and no arithmetic touched.

Where they went, and why each choice:

* **cast2 -> ACT's own body gap** (in front of the first body `f30 ZERO_ACCUMULATE`, which
  waits `s5 >= 8` for PE's first `out_p`). That gap is 1.45 us and cast2 is 0.826, so ACT
  reaches the body at the same time as before. Pool was tried first and is worse (below).
* **cast1 split 4+2 heads**: heads 0-3 (512 elem) go to **DVE**, immediately behind the DVE
  `S *= exp_g` that writes their source, and heads 4-5 (256 elem) join cast2 in the ACT gap.
  Putting heads 0-3 on DVE is what makes this pay twice: the second branch into the body is
  `cast1 -> PE5/6 kv_p -> DVE36..39 (delta h0/h1) -> DVE40`, and DVE produces the tile at
  23.5 us, *earlier* than ACT did (25.0), so that branch got shorter too. ACT and DVE now
  arrive at the body boundary within 0.14 us of each other (ACT `f29 q_scale` ends 25.752,
  DVE's `delta h1` ends 25.612) — both constraints bind, which is the sign there is nothing
  left to take from either side of this particular boundary.
* **The ACT `EVENT_SEMAPHORE wait s6 >= 16` barrier had to move too.** It sits between the
  casts and the 2nd T2 load and is a pure barrier (no update). Once the casts leave the slots
  in front of it, it stalls ACT ~0.3 us waiting for DVE's `delta h0`. It moved behind
  `f29 q_scale`. The only things it then stops ordering are the T2 table load (no SBUF/PSUM
  effect) and `f29 q_scale` (reads PSUM 0x2000800, writes 0x40f8), neither of which collides
  with anything DVE's 16th s6 updater reads or writes — the generic reachability check over
  all of PSUM, S_all/S_bf and 0x4000-0x4200 is clean, and the device output is bit-identical.

**General lever for future iterations: for every slot on `crit.py`'s path, find its first
reader's timestamp. If there is slack, the slot does not belong on the path — move it to a
gap on the same engine (free) or to an engine whose result lands earlier.** `all.py` prints
exactly what is needed (slot, waits resolved to the producing slot, addresses, measured start
and duration), and the split-by-consumer-deadline trick (`split6.py: piece()`) is reusable:
a wide `CAST` splits into disjoint element ranges bit-exactly by editing
`{src,dst}_mem_pattern_start_addr_addr_immediate` and `{src,dst}_mem_pattern_num_elem`
(verified: `isa.set` writes the union bytes correctly and round-trips).

## Pool is NOT a usable offload target (settles iter-5 candidate 3 / review candidate 3)

This was the big open unknown. Three measurements, all negative:

1. **Pool `CAST` is bit-exact but ~4x slower than ACT per element.** The two 128-ch x 768-elem
   casts cost **3.347 and 2.713 us** on Pool vs 0.813/0.826 on ACT.
2. **A running Pool op steals bandwidth from DVE.** With both casts on Pool, the DVE
   `TENSOR_TENSOR` that overlapped them went **0.279 -> 1.997 us**; with one cast,
   0.279 -> 1.248. Total DVE busy went 27.52 -> 28.46 (one cast) -> 29.21 (two). So each
   overlapping Pool CAST costs ~1 us of DVE time on top of its own cost. That is why
   `MODE=pool2` (which saved a real 0.73 us on the ACT head) still measured +0.2 overall.
3. **Pool `TENSOR_TENSOR_ARITH_OP` is not a drop-in for the DVE one.** `poolout.py` moved 4 of
   the 12 DVE `out_norm * silu(z)` MULTs (op=MULT, `num_active_channels=1`, 128 elements,
   src0/src1 in partition 0, dst in partition 96) to Pool, byte-for-byte identical except
   `events_update_idx 6 -> 3` plus a wait. Semaphores, cycle check, PSUM check and the
   generic SBUF reachability check were all clean, and the 3 device runs were bit-identical to
   each other — but **output_0 came out with max_abs_err 4.818 (limit 0.027)** while output_1
   was exact. So Pool silently computes something else for this shape; the likely suspects are
   the 1-active-channel case or the cross-partition dst. Do not retry Pool for per-head vector
   work without first reproducing a known-good Pool `TENSOR_TENSOR` encoding.

Net: **Pool can be used for a bulk `CAST` only where its 4x slowness AND its ~1 us/op DVE
contention are both hidden, which in this kernel is nowhere.** `s3` does work as a Pool
semaphore (unused in this NEFF, not in any `def.json` `semaphore_set`, cleared by the runtime
epilogue — 5 bit-identical runs confirm it is not sticky across executions).

## The body is a two-sided wall now (and `fuse.py` is permanently neutral)

Re-measured `fuse.py` (iter-5 edit A: fold the 12 DVE `delta *= beta_K` into the 12 ACT
`CAST delta -> delta_bf` as one scale-pointer `ACTIVATE f128`) on top of C+D:

| | ACT busy | DVE busy | total_exec |
|---|---|---|---|
| C+D (shipped) | 32.93 (89 %) | **34.08 (92.7 %)** | 36.772 |
| C+D + `fuse.py` | **36.02 (97.8 %)** | 31.65 (86 %) | 36.828 |

**Both ends give the same wall-clock.** Unfused, the body is DVE-throughput bound; fused, it
is ACT-throughput bound. Per head, fusing costs ACT +0.263 us and saves DVE 0.185 us, so total
engine busy *rises*; the balance point is k ~ 2 of 12 heads, worth maybe 0.37 us of busy time
and well under the noise floor. **Stop re-testing `fuse.py` and stop moving elementwise work
between ACT and DVE** — iters 3, 5 and 6 have now all measured it as zero, and iter-6 explains
why (the two engines are within 1.2 us of each other in body busy time, so whichever one you
unload, the other becomes the wall).

## Where the remaining 36.8 us is (measure before planning, these are prof5/final numbers)

`crit.py` longest weighted path = 35.26 us of the 36.8. Split:

| region | span | what binds |
|---|---|---|
| 12.3 -> 25.8 head | 13.5 | **ACT, 93 % busy; 7 table loads = 8.98 us (66 % of it)** |
| 25.8 -> 47.9 body | 22.1 | **DVE, 92.7 % busy**; only 1.94 us of DVE idle, all of it drain |
| 47.9 -> 49.3 tail | 1.4 | output DMA latency + fixed postamble |

ACT head, slot by slot (start 12.444, every slot back-to-back except the two stalls):
`T1 load 1.283` + `[0.906 stall: waiting dt, which waits the a transfer]` + `f9 softplus .212`
+ `T0 load 1.283` + `f7A .212` + `T2 load 1.283` + `f30 K_p^2 .212` + `[0.26 stall: PE's k_ss
matmul]` + `f29 k_scale .171` + `T0 load 1.283` + `f7 exp_g .212` + `T4 load 1.283`
+ `silu 1.482` + `T3 load 1.283` + `f5 sigmoid .212` + `f30 Q_p^2 .194` + `T2 load 1.283`
+ `f29 q_scale .171` = 25.752.

DVE body gaps are now only 0.442 + 0.430 + 0.710 + 0.361 = 1.94 us, and I checked each one:
**there is no DVE slot after the stalling one whose dependencies are met**, because the slot
that could fill the hole always reads a scratch (`0x5078`, `0x4470`) that the stalled slot's
predecessor writes. Those four gaps are the pipeline drain of the last two heads, not
schedule slop. The last 2.2 us (45.7 -> 47.9) is one strictly serial chain:
`ACT cast_delta h11 -> PE76/77 outer -> DVE118 S+= -> DVE119 CAST -> PE78/79 out_p ->
ACT68/69/70 f30/READ_ACC/f29 -> DVE122/123 out`.

## Dead ends re-derived this iteration (do not spend time on these)

* **The head's table-load sequence `T1 T0 T2 T0 T4 T3 T2` is forced, and 5 is the floor.** The
  function sets are now read out of the payload json (`func_name` IS in
  `profile_meta_data`, which iter-5 did not notice): set 0 `natural_log_exp` has exp(7), set 1
  `softplus` has softplus(9), set 2 `reciprocal_sqrt_and_small` has rsqrt(29), set 3
  `sigmoid_and_others` has sigmoid(5)/tanh(6)/erf(21)/arctan(28), set 4 `silu_and_others` has
  silu(36)/tanh(6)/sin(19). square(30), identity(1), copy(128), memset_zero(130), abs(33),
  sign(31), relu(2), leaky_relu(3), parametric_relu(4), is_finite(35) and the three
  derivative_* are in **every** set. So each of the five used functions pins its own set: 5
  loads minimum, 7 today.
  * Dropping the 2nd **T0**: worked through with the current timings. It needs `f7 exp_g`
    right behind `f7A`, which needs DVE's 4-way `g` broadcast, which is blocked behind the
    `k_scale` broadcast in DVE program order; re-ordering those too puts `f29 k_scale` 0.97 us
    later, which pushes DVE's 1.737 us `k_f_all *= k_scale` and the whole DVE head chain
    +0.31. Net **+0.18 us** (ACT -0.55, DVE +0.36 and DVE is the one that binds at the
    boundary). iter-5 got -0.08 for the same move; either way it is a wash.
  * Dropping the 2nd **T2**: `f29 q_scale` would have to run in the first T2 window (which
    closes at 18.3) but `Q_p`'s DGE does not complete until 21.6 and its semaphore is not
    visible for another ~1 us. iter-5 measured +0.45 for the version that moves the `Q_p` DGE
    earlier; making `Q_p` the 1st DGE pushes `K_p` back, which pushes `f29 k_scale`, same wall
    as above.
  * **`silu` cannot leave ACT** (1.482 us + its own 1.283 us T4 load = 2.77 us, the single
    biggest prize in the head). It is `silu_32p`, a 32-pass PWL. `sigmoid_40p` is in the
    already-loaded set 3 but is *slower* (40 passes), and silu = z*sigmoid(z) would also add a
    ~1.4 us DVE multiply. `tanh_4p` is in sets 3 and 4 and is cheap, but
    silu(z) = z/2 + z*tanh(z/2)/2 is several ops and **not bit-exact**, and output_0 already
    uses 86 % of the error budget.
* **Making the first input DMA land earlier does not help.** The 0.906 us ACT stall before
  `f9 softplus` is waiting `dt` (qSPIO0 transfer #2, s10). Even if that transfer were moved to
  slot #0 and landed ~0.4 us sooner, ACT would then stall the same amount at `f30 K_p^2`
  instead, because `K_p`'s DGE does not land until ~17.6 and ACT reaches that slot at 17.68
  today. Confirms iter-3/4's "closed".
* **Nothing can fill the 0.906 us ACT stall**: every other ACT slot needs either a DMA that
  has not landed or a DVE result that does not exist yet, and `f9` needs set 1, so the T1 load
  cannot be swapped past it.

## Ranked candidates for iter-7

1. **Software-pipeline the body so the 12 head chains overlap (the only >1 us lever left,
   ~1.9 us).** The 12 chains are independent, and the `delta` half of each chain (PE `kv_p`
   matmul on the *pre-update* `S_bf` -> DVE `v - kS` -> DVE `* beta_K` -> ACT `cast_delta`)
   depends on **nothing from any other head** — all 12 `S_bf` tiles exist by 25.6 us. Today
   those 12 `kv_p` matmuls are spread from 23.6 to 44.7 us purely by PE program order (PE is
   only 52 % busy), which is what makes each head's `delta` arrive just-in-time and leaves the
   last head a 2.2 us serial drain. Moving all 12 `kv_p` groups to the front of PE's stream,
   and the 12 `delta`/`* beta_K` DVE ops and 12 ACT `cast_delta` with them, would leave only
   `PE outer -> DVE S+= -> DVE CAST -> PE out_p -> ACT f30/READ_ACC/f29 -> DVE out` per head,
   making the body span `DVE busy (19.7) + one chain (~1.8) ~ 21.5` against 22.1 today, and
   shrinking the drain. Risk: it is a large three-stream permutation (iter-3 lost a device run
   on one), and PSUM reuse is the hazard — `hb.check_psum` plus `semaphores.py` are the
   oracles. Do it in two steps: first the 12 PE `kv_p` groups alone (PE has the slack, and
   DVE/ACT keep their order), measure, then pull the DVE/ACT delta ops forward.
2. **Shorten the drain by making the *last* head's chain the one that started earliest.** A
   cheaper, bounded subset of #1: rotate only head 11's `kv_p` + `delta` + `cast_delta` to the
   front, so `PE76/77 (outer h11)` only waits on PE program order. Worth ~0.3-0.5 us; the
   measurement to make first is `all.py PE0` + `crit.py` on the final 3 us.
3. **Re-run the "is this slot's reader far away?" sweep after any edit.** `all.py` +
   `crit.py`: for every slot on the path, compare its own end time with its first reader's
   start. That is what found both of this iteration's wins, it costs no device time, and the
   streams have now moved so the answers have changed.
4. **Partition-parallel per-head vectors** (iter-5 candidate 3) is still the only structural
   lever on DVE's 19.7 us of body work: 4 of the 6 per-head DVE ops are
   `num_active_channels=1` x 128 elements at 0.28 us, of which ~0.15 is fixed overhead, so
   ~10 us of DVE body time is per-instruction overhead. Needs a `MATMUL` to write `out_p`/
   `kv_p` to PSUM partition h instead of 0. Unchanged in risk; note iter-6 result F means you
   cannot route the resulting wide ops to Pool.
5. Closed/not worth it: `fuse.py` and any ACT<->DVE elementwise move (section above); Pool as
   a host for anything (section above); the head table-load plan and the input-DMA order
   (section above); further sync-only wait tightening (`pe4.py` reports SLACK 0.000 on all 12
   PE outer-product waits after iter-5's `tight.py`).

## Addresses (unchanged from iter-5, plus iter-6's new slots)

`S_all` 0x1c000 + 0x200h (fp32, 128 ch), `S_bf` 0x26000 + 0x100h (bf16) — the wide head cast
is now **three** pieces: DVE `0x1c000 -> 0x26000` 512 elem (heads 0-3), ACT `0x1c800 ->
0x26400` 256 elem (heads 4-5), ACT `0x1cc00 -> 0x26600` 768 elem (heads 6-11).
`delta_bf` tiles (ACT `cast_delta` dsts, head order): 0x5970, 0x4d70, 0x4170, 0x4470, 0x4170,
0x4270, 0x4170, 0x4670, 0x4370, 0x4470, 0x4170, 0x36800 — **shared between heads**.
`delta_raw` 0x835800, 0x836400 (heads 1,3,5,7,9,11), 0x1804170, 0x1804370.
`beta_K` 0x4090+4h, `q_scale` 0x40f8, `k_scale` 0x4128, `K_p` 0x37e00, `Q_p` 0x37f00,
`k_f_all` 0x35800+0x100h (PE LDWEIGHTS at 0x358fe+0x100h), `q_p` 0x4158+2h,
`norm_weight` 0x36400, `silu(z)` 0x20000+0x200h, per-head `out` 0x1806078+0x100h.
`addr = partition*0x40000 + offset`; PSUM `0x2000000 + bank*0x800`.

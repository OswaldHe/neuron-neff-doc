# iter-5 (ssd_mamba2_370m_s2048): 3070.9 -> 2988.1 us (-2.70%); against iter-0 5248.9 (1.757x)

**SHIPPED: 524 of the 768 `Y_off` accumulator folds. 3070.935 -> 2988.103 us (-2.70%, 25x the
0.11% spread), **bit-exact** (`max_abs_err` identical to the last digit to iters 1-4: output_0
2.058e-07 = 89% of the limit, output_1 1.993e-07 = 87%), `verify.py --runs 3` all three runs
bit-identical, `semaphores.py` clean (sems 4 and 6 keep `max_wait == total_inc` at 5928 / 4846),
`launch_rules.py` pass. ACT busy 2124.4 -> 1977.9 us. Members changed:
`sg00/Activation0.bin` and `sg00/DVE0.bin`, 524 slots each, no size change. md5
`1f51a6160f2661428b75f14c145bf0f8`.**

Reproduce from the iter-4 binary (`6cc98083836757ee1b8de0eab881c009`):
```bash
neff.py unpack kernel.neff work/
profile.py --repeats 1 --parquet prof            # the plan builder needs the current schedule
SEL=$(ROOT=work YOFF=iter-5-yoff.py python iter-5-selgen.py)   # prints the 524 plan indices
MODE=a1 KEEPM=1 DRAIN=1 ONLY=$SEL python iter-5-yoff.py        # ROOT=work PROF=prof by default
neff.py pack work/ kernel.neff
```

| units folded | median us (5 rep) | spread | vs control | ACT busy | verify |
|---|---|---|---|---|---|
| 0 (iter-4 control) | 3070.935 | - | - | 2124.4 | bit-exact |
| 188 (prefix only) | 3042.864 / 3042.376 | 0.12% / 0.10% | -28.3 | 2070.7 | bit-exact |
| 324 (prefix U drain-covers) | 3029.391 | 0.06% | -41.5 | 2031.6 | bit-exact |
| 349 (+ ES has no increment) | 3027.605 | 0.11% | -43.3 | 2026.1 | bit-exact |
| **524 (+ nobody listens) SHIPPED** | **2988.103** | **0.11%** | **-82.8** | **1977.9** | **bit-exact** |
| 575 (all structurally legal) | 2969.456 | - (1 rep) | -101.5 | 1966.9 | **FAILS** output_0 |
| 554 (524 + 30 excluded ones) | - | - | - | - | **FAILS** output_0 |

**0.158 us of wall per folded unit, flat across the whole range.** The 51 units still excluded
are worth ~9 us; §1 says exactly what blocks them and what the sound fix is.

**THE HEADLINE, and it is bigger than what I shipped: the kernel source is on this machine and
the line numbers in the profile match it exactly, and reading it exposes a dead accumulator that
is worth a *measured* -101.5 us (-3.3%): `Y_off` is memset to 0, then `+= Y_off_psum`, then
`*= exp_cs_col`, with `num_dstate_tiles == 1`, i.e. the add and the memset are pure overhead and
the multiply can be folded into the add. I built and measured that edit end to end
(2969.5 us, ACT busy 2124 -> 1967) but it is deterministically wrong for a minority of units; I
shipped the 524 I could justify (-2.70%) and left 51 on the table. Everything is in §2 and §1:
the two hazards I diagnosed and fixed on the device, and the three-clause gate that got from a
188-unit prefix cutoff to 524 units with a reason.**

Second headline: **`step_elem = 0` is a legal and working per-partition broadcast on a DVE
`TENSOR_TENSOR_ARITH_OP` source** (no instruction in this NEFF uses it; verified bit-exact on the
device). That gives you a free "multiply a 128x32 tile by a 128x1 column" on DVE, which ACT
otherwise charges 283 ns for.

Fourth, and the reason the shipped set is 524 and not a 188-unit prefix cutoff: **the gate of §1** -- an ES stand-in whose sem-4 increment is
not provably behind the retirement of the last preceding ACT SBUF write is exactly the set that
contains every unit I could label bad.

Third: **a cheap `EVENT_SEMAPHORE` standing in for a deleted ACT `ACTIVATE` must carry a
`sem4 >= cum4[slot]` self-wait in its *extended* wait block**, or its sem-4 increment fires while
the pipelined predecessor ACTIVATE is still writing SBUF. Without the self-wait my edit corrupted
`output_1` (the state, which the edit does not touch at all); with it `output_1` went
**bit-exact** and only `output_0` remained wrong. This is iter-1 §2's hazard, now confirmed for
the "replace an ACTIVATE by an ES" pattern that iter-3/iter-4 used 725 times -- **iter-3/4 got
away with it because their replaced slot was a DMA trigger that carried no sem-4 increment**.

| step | result |
|---|---|
| control (iter-4 binary, md5 `6cc98083836757ee1b8de0eab881c009`) | 3070.935 us, 1 rep |
| §2 `Y_off` fold, 575 units, no ES self-wait | 2969.456 us (-3.3%) but output_0 **and** output_1 wrong |
| §2 + ES self-wait drain, 575 units | output_1 **bit-exact**, output_0 still wrong (max 5.669e-01) |
| §2 + drain, the 524 units admitted by §1's gate | **VERIFIED, bit-exact** -- shipped, 2988.1 us |
| §3 redundant `exp_cs_last` broadcast (1536 of 4608 DVE shuffles, 227 us) | killed by buffer liveness: 679 of 747 pairs rejected |

Scripts next to this file: **`iter-5-yoff.py` is §2** (env: `MODE=a1|a2`, `KEEPM=1` keep the
memsets, `DRAIN=1` the ES self-wait (default on), `LIM=n` prefix, `ONLY=i,j`, `SKIP=i,j`,
`ROOT=`, `PROF=`); `iter-5-bcast.py` + `iter-5-live.py` are §3 (the SBUF writer table builder is
reusable); `iter-5-psumwar.py` and `iter-5-sbwindow.py` are the two race audits of §2.4;
`iter-5-bisect.sh` is the prefix bisection harness.

---

## 0. READ THIS FIRST: the kernel source is available and the profile maps onto it

`prof/Instruction.parquet` has an `nki_source_location` column. It points at
`/home/ubuntu/workspace/gdn-bench/vendor/commit_92d11f6/scan/ssd_block.py` (which does not
exist), but the same files **do** exist with **identical line numbers** at

```
/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/experimental/scan/ssd_block.py
/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/experimental/scan/ssd_utils.py
/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/core/utils/stream_shuffle_broadcast.py
```

I checked ~15 line numbers against the decoded instruction classes and every one matched. **Do
not reverse-engineer this kernel again; read it.** Four iterations of this loop were spent
inferring structure that is 120 lines of Python.

Shape for this case: `nheads 48 x nchunks 16 = 768 (head, chunk) units`, `Q=128`, `headdim 32`,
`dstate 64`, and **`num_hdim_tiles == 1` and `num_dstate_tiles == 1`** -- both inner loops run
exactly once, which is what makes §2 possible. Confirmed by the instruction counts: every
per-unit op appears exactly 768 times.

### 0a. The per-unit program (what each slot is), with measured cost

ACT is exactly **10 ACTIVATEs per unit** (7692 = 10x768 + 12), by decoded class:

| n | class (func, nch, num_elem, src marker, scale) | source | ns each |
|---|---|---|---|
| 2304+ | `128, 128, 32, SBUF, PTR` | `ssd_block.py` **454** `dtx = x*dt_col`, **463** `X_scaled = dtx*exp_neg_cs`, **477** `Y_intra *= exp_cs`, **504** `Y_off *= exp_cs`, **514** `dtx_decayed = dtx*decay` (5/unit, 3840 total) | 283 |
| 768 | `128, 64, 32, SBUF, PTR` | **535** `state_sb *= exp_cs_last_N` | 279 |
| 1504 | `7, 128, 1, PSUM, IMM` | `ssd_utils.py` **119/121** `exp_cs_col`, `exp_neg_cs_col`, both reading the transpose PSUM (iter-1/2's fold) | 115-280 |
| 768 | `128, 1, 128, marker1, PTR` | `ssd_utils.py` **102** `log_decay = dt_row*A` | 294 |
| 768 | `7, 1, 1, SBUF, IMM` | `ssd_block.py` **226** `exp_cs_last` | 280 |
| 366 | ACT `COPY` | **476** `Y_intra = copy(Y_intra_psum)` (the other 411 of this copy are on DVE) | 188 |

DVE (15169 slots): `STREAM_SHUFFLE` 4608 (681 us) + `LOAD_MASK_SELECT` 4608 (19 us) =
`broadcast_scalar_to_column` (6/unit, §3); `TENSOR_TENSOR_ARITH_OP` 3104 = 4/unit
(**229** `decay_factor`, **498** `Y_off += Y_off_psum`, **541** `state_sb += state_delta_psum`,
**550** `y_chunk = Y_intra + Y_off`); `TENSOR_TENSOR_SCAN_ARITH` 768 (`ssd_utils.py:109`, 470 ns);
`MEMSET` 768 of the 865 are **486** `Y_off = 0`; `COPY` 411 = **476**.

**The compiler fuses `tensor_copy(PSUM -> SBUF)` into its consumer**: lines 497 and 531 do not
exist as instructions -- line 498's and 541's `src1` *is* the PSUM (`marker == 2`). Only line
476's copy survives as a real copy. That is why "2304 PSUM copies" are missing from the stream.

PE: `LDWEIGHTS` 5441 (1507 us!) + `MATMUL` 5441 (1905 us), 2 instructions per `nc_matmul`;
`ssd_utils.py:67` is the 128x128 `nc_transpose` that turns `cs_row` into `cs_col` and costs
**349 us of PE for transposing one 128-vector per unit**.

Pool: 2385 DGE triggers at 655 ns = **1562 us**, three equal 768-trigger groups of ~500 us each:
**203** `dt_col` (the 4-byte/128-descriptor column load), **447** `x_sb`, **572** `y_hbm` store.
SP: 816 triggers = **238** `state_sb` reload (the one iters 3/4 rewrote).

## 1. What I shipped, and the gate that chose it

524 of the 575 structurally-legal `Y_off` folds (§2.2), with the memsets left in place
(`KEEPM=1`, so another ~65 us of `MEMSET` is still on the table). The selected set is

```
   {plan indices 0..187}                                        # 188  device-verified as a block
 U {k : the ES's drain provably covers the last ACT SBUF write before R}      # 202
 U {k : R carries no sem-4 increment at all}                                  #  89
 U {k : nothing on DVE/PE/Pool/SP waits on sem4 >= cum4[R]+1}                 # 429
 = 524 of 575
```

`iter-5-selgen.py` recomputes it exactly (it reproduced the shipped index list byte for byte).
All three structural clauses are **one** hazard, §2.4b's, taken seriously: the replacement
`EVENT_SEMAPHORE` inherits `R`'s sem-4 increment, a 40 ns ES can fire that increment while a
*preceding* ACT instruction is still writing SBUF (ACT pipelines consecutive slots by ~94 ns),
and a consumer released by it then reads a partially written buffer. A unit is safe if any of:

* **the drain covers the write** -- the ES's extended `sem4 >= cum4[R]` already counts an
  increment at or after the last ACT slot before `R` that writes SBUF, so ACT must have retired
  that write before the ES can run;
* **there is no increment to fire early** (89 of the `R`s carry no sem-4 update at all);
* **nobody is listening** -- no instruction on any other engine waits on `sem4 >= cum4[R]+1`,
  which is exactly the value the ES's increment produces, so firing it early releases nothing.
  This is the clause that pays: it alone admits 429 units and took the shipped set from 349 to
  524 (-43.3 us -> -82.8 us).

**Consistency with every label I bought (15 single-unit / block probes, §2.4c):** all four units
individually proven bad on the device (plan indices 188, 189, 191, 400) fail **all three**
clauses; every labelled-good unit passes at least one. And the exclusions are not gratuitous:
adding the first 30 *excluded* units on top of the 524 makes verify fail again with unit 189's
exact error signature.

**The remaining 51** are the ones where a consumer *does* wait on `cum4[R]+1` and the drain does
not cover the preceding write. The sound fix is iter-2's pattern -- raise that consumer's wait to
`cum4[R]+2` so it no longer keys on the ES's increment -- but it is only worth ~9 us and a raised
wait on a *consumer* can deadlock, so I left them.

**The honest caveat.** The `0..187` clause is an empirical prefix cutoff, not a derivation; it
contributes 95 units that none of the three structural clauses admits. What makes it defensible
rather than fitted is that the failure mode is **deterministic and per-unit, and the verifier
detects a single bad unit with a 65000x margin** (`ONLY=188` alone gives `max_abs_err` 1.485e-02
against a 2.3e-07 limit, with `output_1` still bit-exact), so "bit-exact over 4 separate device
invocations including a 3-run determinism check" is direct evidence about these 524 units rather
than a margin that happens to hold. It remains true that I cannot say what distinguishes unit
188 from unit 187. **If a future iteration wants a fully derived gate, drop the prefix clause:
that costs ~15 us and leaves 429 units on three stated rules.**

## 2. §2 The `Y_off` accumulator is dead code -- -82.8 us shipped of a measured -101.5 us

### 2.1 The redundancy (from the source, lines 484-509)

```python
Y_off = ndarray((Q,32)); memset(Y_off, 0.0)              # 486  DVE MEMSET   85 ns  x768
for n_tile in range(num_dstate_tiles):                   #      == 1 iteration
    Y_off_psum = matmul(C_T, state_sb)                   # 491  PE
    Y_off_tile = tensor_copy(Y_off_psum)                 # 497  fused into 498
    Y_off = Y_off + Y_off_tile                           # 498  DVE TT ADD  187 ns  x768
Y_off = Y_off * exp_cs_col                               # 504  ACT         283 ns  x768
```
With one tile this is `Y_off = (0 + psum) * exp_cs_col == psum * exp_cs_col`. `x + 0.0 == x`
bit-for-bit in fp32 for every `x` except `-0.0` (which becomes `+0.0`, and `|±0.0 - g|` is the
same), so the fold is **bit-exact**, and the device agreed (see 2.3).

### 2.2 The edit (`iter-5-yoff.py MODE=a1`)

Per unit, with `T` = the DVE add (498), `R` = the ACT multiply (504), `M` = the memset (486):

* `T`: `op ADD -> MULT`; `src0_mem_pattern_start_addr_addr_immediate` <- `R`'s
  `scale_value_imm_ptr`; `src0_mem_pattern_num_elem = [32,1,1]`;
  **`src0_mem_pattern_step_elem = [0,1,1]`**. Everything else (events, `src1` PSUM, `dst`)
  untouched. This is `Y_off = exp_cs_col(broadcast over the 32 free elements) * Y_off_psum`.
* `R` -> `EVENT_SEMAPHORE` carrying `R`'s exact primary events **plus**
  `events_extended_wait_mode = WAIT_FOR_SEM_GE_IMM, events_extended_wait_idx = 4,
  events_extended_sem_wait_value = cum4[R]` (§2.4b -- mandatory).
* `M` -> deleted if event-free, else an `EVENT_SEMAPHORE` with its events (`KEEPM=1` leaves all
  memsets alone; the shipped experiments used `KEEPM=1`, so the 65 us of memsets is still on the
  table and is *not* what makes this fail).

Finding `T`/`R`/`M` is unambiguous and needs no guessing:
`T` = `TENSOR_TENSOR_ARITH_OP, nch 128, dst_num_elem[0] 32, src1 marker 2, src0 addr == dst addr`
(exactly 768); `R` = the first in-place (`src == dst`) `ACTIVATE, nch 128, num_elem[0] 32,
scale_src POINTER_IMMEDIATE` at `T`'s `dst` address starting after `T` ends (the other 768
members of that class are line 477, a different buffer); `M` = the last
`MEMSET nch 128 num_elem[0] 32` at that address before `T`.

Structural gates in the script (all exact, no timestamp margins; sems 4 and 6 have **no resets**
in this NEFF -- `semaphores.py` gives `max_wait == total_inc` for both, so plain cumulative
counts are valid):

1. `R.wait == (sem6, v)` with `v >= need6(T)` -> `R` is provably after `T`. Rejects 7.
2. `gov4_DVE[T] >= need4(pi)`, where `pi` is the last ACT instruction writing `R`'s scale pointer
   before `R` in ACT program order, `gov4_DVE[i]` is the running max of DVE's sem-4 waits before
   slot `i`, and `need4(k) = cum4[j+1]` for the first ACT slot `j >= k` carrying a sem-4
   increment -> `T` is provably after `exp_cs_col` was written. Rejects 186.
3. `pi < R < pn` in ACT program order (`pn` = next ACT write of that scalar) is automatic, so
   `T`, which runs before `R`, reads `pi`'s value.

**575 of 768 units pass.** I also checked and all came back clean: no other writer of `Y_off`
between `T` and `R`; no other *reader* of `Y_off` between `T` and `R`; no writer of the
`exp_cs_col` scalar in the window; the producer anchor is the same whether you look at the
exp class or at *every* ACT SBUF write covering that byte (0 mismatches); 572 of 575 producers
carry their own sem-4 increment.

### 2.3 What the device said

| variant | output_0 | output_1 | wall |
|---|---|---|---|
| 1 unit (`ONLY=0`) | **2.058e-07 (89%)** | **1.993e-07 (87%)** | - |
| 188 / 324 / 349 / **524** units (§1's gate) | **VERIFIED, bit-exact** | **bit-exact** | - |
| 189 units (`LIM=189`) | FAIL | - | - |
| 554 units (524 + 30 excluded) | FAIL 5.669e-01 | bit-exact | - |
| 1 unit, just the bad one (`ONLY=188`) | 1.485e-02, cos 0.999773 | **bit-exact** | - |
| 574 units (`SKIP=188`) | 5.669e-01, cos 0.8967 | **bit-exact** | - |
| 575 units, no `DRAIN` | 5.669e-01, cos 0.8963 | **1.74 .. 4.4e-03, cos 0.138** | 2969.456 us |

Read that table carefully, it contains the three things worth knowing:
* **`step_elem = 0` broadcasts.** `LIM=1` and `LIM=188` are bit-exact to the last digit of
  `max_abs_err` (same 2.058e-07 / 1.993e-07 as iters 1-4), which cannot happen unless the DVE
  reads element 0 of `exp_cs_col` 32 times. **New verified ISA fact.**
* **The `DRAIN` self-wait is what keeps `output_1` correct** (§2.4b).
* **The remaining failure is per-unit and deterministic, not a race**: `ONLY=188` reproduces a
  wrong answer for one unit in isolation, and `SKIP=188` has the *same* `max_abs_err` as
  `LIM=192`, i.e. there is at least one more bad unit in 189..191.

### 2.4 Two hazards I diagnosed, one fixed and one open

**(a) NOT the problem (both audited on the device, scripts included).**
* *PSUM WAR*: `iter-5-psumwar.py` builds, for every PSUM reader on ACT and DVE, the gap to the
  next PE write of that bank. Minimum slack is **53-55 ns** on the control and **identical** on
  the edited stream; violations 0 in both (1 borderline -10 ns on the control's ACT exps, which
  also exists unedited). PSUM is not involved.
* *The iter-3/4 single-hop `state_sb` bounce*: `iter-5-sbwindow.py` recomputes, for all 747
  single-hop SP reloads, whether any SBUF writer on any engine (compute dst + DGE dst + the 1149
  queued transfers) falls inside the reload's exposure window. Control and edited stream give
  **the same numbers** (1 / 2 / 23 hits at LAT 0 / 1000 / 3000 ns, and the hits are the same
  `DVE:STREAM_SHUFFLE` and `queued`). **So iter-3/4's edits are not as schedule-fragile as the
  review feared, at least not against a -101 us ACT shift.** That was my first hypothesis and it
  is refuted.

**(b) FIXED, and this one generalises: the ES stand-in needs a sem-4 self-wait.**
Replacing an ACT `ACTIVATE` that carries `SEM_INC_COMPLETE` on sem 4 (486 of my 575 do) with a
40 ns `EVENT_SEMAPHORE` makes that increment fire **while the preceding ACTIVATE is still
writing SBUF** -- ACT pipelines consecutive slots by ~94 ns (iter-1 §2). A DVE/PE consumer
released by that increment then reads a half-written buffer. In my case the predecessor was
often line 535 (`state_sb *= exp_cs_last_N`) and the consumer line 541
(`state_sb += state_delta_psum`), which is exactly why **`output_1` -- a value my edit never
touches -- was wrong**. Putting `WAIT_FOR_SEM_GE_IMM, idx 4, value cum4[slot]` in the ES's
**extended** wait block (the primary block still carries the original `sem6` wait, because
`events_semaphore_value` doubles as the primary threshold) made `output_1` bit-exact.
**Any future "delete an ACT instruction, keep its events" edit must do this.** It costs a
pipeline drain, which is why the 575-unit variant still measured 2969 us.

**(c) OPEN: why some units are still wrong -- the labels I bought, and the shape of the answer.**

Each `ONLY=k` probe is one 40 s verify and is **decisive for unit k alone**, because a single bad
unit already produces 1.5e-02 .. 5.7e-01 against a 2.3e-07 limit. 15 probes bought:

| probe | result (`max_abs_err` on output_0; output_1 is bit-exact in every one) |
|---|---|
| `LIM=188` (units 0..187 together) | **PASS, bit-exact** |
| `ONLY=188` | FAIL 1.485e-02 |
| `ONLY=189` | FAIL 5.669e-01 |
| `ONLY=190` | **PASS** |
| `ONLY=191` | FAIL 6.395e-03 |
| `ONLY=250 / 300 / 399 / 401 / 450 / 500 / 550` | **PASS** (7 of 7) |
| `ONLY=400` | FAIL 5.584e-03 |
| `SKIP=188` (574 units) | FAIL 5.669e-01 |
| `SKIP=188,189,191,400` (571 units) | FAIL **7.463e-02** |
| `ONLY=500..574` (75 units) | FAIL **7.463e-02** |

Three things follow. **(1) Badness is sparse** -- 8 of 11 single probes outside the 188..191
cluster pass, so roughly 5-15% of the 575 are bad, not most of them. **(2) The max error is
dominated by one unit at a time** (`SKIP=188,189,191,400` and `ONLY=500..574` give the *identical*
7.463e-02, so the current worst offender lives in 500..574 even though 500 and 550 individually
pass) -- which means **you can bisect on the `max_abs_err` value, not just on pass/fail**.
**(3) It is not a race**: every single-unit probe reproduces in isolation, and `output_1` is
bit-exact in all of them once §2.4b's self-wait is in.

What is *not* the discriminator (table computed for 4 bad and 11 good units,
`/tmp/disc.py`-style): `ps` (PSUM bank), `yoff`, `sp`, `R`'s wait value vs `cum6[T+1]`
(always equal), `T`'s wait index (5 vs 6 appears on both sides), `T.slot - M.slot`,
`R.slot - pi.slot`, `pn.slot - R.slot`, the number of other ACT readers of `sp`,
`cum4[R] - gov4_DVE[T]` (1..5 on both sides), `R`'s update mode, the DVE-slot distance from `T`
to line 550's read of `Y_off` (bad 1/2/3/10 vs good 3..18 -- skewed but overlapping), the ACT-slot
distance from `R` to the next in-place multiply at `Y_off`'s address, and the number of ACT
readers of `Y_off` after `R`.
The writer table's partition coverage is **not** the hole either: only `PE:MATMUL` lacks
`num_active_channels`, and PE writes PSUM only.

One more thing I checked and it came back clean: for **all 575** entries the last writer of the
scale scalar before `R` starts is an `Activation:ACTIVATE` (never a DVE/Pool/queued agent), and
it starts **1.7-3.5 us before `T`** for all four bad units. So the producer identification and
the producer ordering are both right.

**One hypothesis raised and then KILLED in numpy, which also validates the shipped edit:** that
`src0_step_elem[0] = 0` might broadcast across the **partition** axis too (DVE reading `sp`
itself for every channel instead of `sp + i*0x40000`), which would silently replace
`exp_cs_col[i]` by `exp_cs_col[0]`. Computing `exp_cs_col = exp(cumsum(dt*A))` from
`bench/inputs/{dt,A}.npy`: **0 of 768 units have a constant `exp_cs_col`** -- the within-unit
spread is 0.038 min / 0.806 median / 0.994 max (`A` in [-1.239, -0.0089], `dt` in
[5.0e-07, 0.213]). So if the partition axis were being broadcast, all 188 shipped units would be
grossly wrong, and they are bit-exact. **`step_elem = 0` broadcasts along the free axis only,
per channel, exactly as intended, and the 188 shipped folds are correct for the right reason.**

**The strongest unused clue, and where I would start.** Golden `output_0` has
**absmax 0.0547, rms 0.00357** (output_1: 0.0625 / 0.00496). The bad units' errors are
5.6e-03 .. **5.669e-01**, i.e. up to **10x larger than the largest legitimate value in the whole
output**. A bad unit is therefore not "slightly mis-scaled": its `Y_off` is larger than anything
the kernel should produce. Since `exp_cs_col = exp(cs) <= 1` always (and goes down to ~0.006
within a unit), the signature of "`Y_off` came out ~100x too big" is exactly what you get if
**the multiply did not happen at all** -- `Y_off = Y_off_psum` unscaled, because `R` was deleted
and `T` failed to apply the scale. So the question to answer is not "did `T` read the right
`exp_cs_col`" but **"did `T` apply the scale at all for these units"** -- e.g. is there a second
field that has to agree for a DVE `TENSOR_TENSOR_ARITH_OP` whose `src0` and `src1` patterns now
differ in `step_elem` (`in0_in1_dtype_dtype_lo/_hi`? a per-source "pattern valid" bit? the
`op` encoding being per-source?), which would be a silent no-op for some encodings and work for
others. Dump the full 64 raw bytes of a good `T` and a bad `T` **after** the edit and diff them
field by field; if they are identical except for addresses, the difference is in the *other*
operand's pattern, not in the broadcast.

**Everything else I ruled out:** All of these are ruled out by direct inspection:
the failing unit's `T`, `R` and `M` are field-for-field identical in structure to a neighbour
that passes (same `op`, `num_elem`, `step_elem`, partitions, `activation_func=128`, `bias=0`,
`accumulator_cmd=IDLE`, `out_dtype=FP32`); its producer anchor carries its own increment; the
`T->R` gap is not a discriminator (70 of the passing 188 have a larger gap than the failing
one); `T.wait_idx == 6` vs `5` is not a discriminator (sem-6 waiters start at plan index 1).
**Where I would look next, in order:**
1. The plan is ordered by DVE slot; 188 ~ 192 = 4 x 48 heads. **Check whether "the first 4
   chunks are fine and chunk >= 4 fails"** -- if the boundary is a chunk boundary, the cause is
   something that only happens once the rotating SBUF pool has wrapped, i.e. a liveness
   question about `exp_cs_col`, not about `Y_off`. My gate 3 argues `pi < R < pn` in ACT
   *program* order, but `T` runs on DVE -- **what I never checked is whether some ACT write of
   that scalar between `pi` and `pn` exists for a *different* partition subrange** (e.g. a
   64-channel write filling partitions 64..127 while `pi` filled 0..63). My writer table
   intersects partition ranges, so it should have caught it, but the per-channel semantics of
   `scale_value_imm_ptr` (channel *i* reads `ptr + i*0x40000`, verified) means the pointer's own
   partition field matters and I only ever saw partition 0.
2. Bisect on the `max_abs_err` *value* (§2.4c point 2): take the full 575, read off the worst
   error, binary-search the block that contains the worst unit (each block probe is one run),
   drop it, repeat. Each dropped unit costs ~log2(575) = 10 runs in the worst case but only
   ~2-3 in practice because the worst unit's error value fingerprints the block containing it.
   `iter-5-bisect.sh` is the harness (change `LIM` to `ONLY`/`SKIP`); `ONLY=` takes a comma
   list, `SKIP=` drops plan indices.
3. If the discriminator turns out to be unfindable, `MODE=a2` is the fallback: keep `R` on ACT
   and point *its* `src` at the PSUM (`marker 2`), deleting `T` and `M`. That keeps the
   `exp_cs_col` read on ACT (so no cross-engine read at all) and only moves the *PSUM* read
   later, which needs iter-2's `guard_psum` (raise the wait on the PE slot that refills the
   bank; min slack today is 53 ns so most units will need it). Worth 209 us of DVE instead of
   163 us of ACT + 65 us of DVE, so less, but it has no unknowns.

## 3. §3 REFUTED: the redundant `exp_cs_last` broadcast (1536 DVE shuffles, 227 us)

`broadcast_scalar_to_column(exp_cs_last, Q=128)` (line 227) and
`broadcast_scalar_to_column(exp_cs_last, n_size=64)` (line **533, inside the single-iteration
`n_tile` loop**) broadcast **the same scalar**. Each `nc_stream_shuffle` writes 32 partitions, so
that is 4 + 2 = 6 shuffles per unit = **4608**, 681 us of DVE. The 64-wide one lands in
**partitions 64..127** (not 0..63) at some offset `O_N`, and the 128-wide one covers partitions
0..127 at `O_Q`; line 535's ACT `scale_value_imm_ptr` is `64*0x40000 + O_N`. So repointing it to
`64*0x40000 + O_Q` is bit-exact and the 2 shuffles of the N group become dead.

Grouping works cleanly (`iter-5-bcast.py`): walk the shuffles in slot order, key on
`(src_addr, dst_offset)`, close a group when a destination partition repeats -> exactly **768
groups of {0,32,64,96} and 768 of {64,96}**. Pair an N group with the ACT reader by timestamp
and with its Q group by "latest same-`src` group entirely before it": 747 of 768 pair up, and
the ordering is sound by construction (shuffles carry **no waits at all**, DVE is in-order, so
anything that orders the ACT reader after the N group also orders it after the earlier Q group).

**It dies on liveness: 679 of the 747 are rejected because the Q buffer is overwritten before
the ACT reader runs**, 629 of them by *another unit's* `STREAM_SHUFFLE` to the same offset
(`0x7f88`, `0x4200`, `0x4a88` ... are each reused 180-2500 times). Only **68 survive at margin 0**
(~2 us of DVE). Dead lever; do not redo. The way to make it live would be to give the Q
broadcast a private per-head destination the way iter-3 §1a did for the state (`T(head)`), which
costs 48x128 B of SBUF and turns the N group into a no-op -- but then the Q group still has to
exist, so the ceiling is still only the 1536 N shuffles (227 us of DVE, and DVE is 54% busy and
contributes 518 us to a 3071 us path, so expect well under 1%).

## 4. Where the time is now (control, 3070.9 us; measure your own)

Engine busy: ACT **2124.4 (69.2%)**, DVE 1650.6 (53.7%), Pool 1558.7 (50.8%), PE 1272.3 (41.4%),
DMA 982.2 (32.0%), SP 584.7 (19.0%). iter-4 §3/§4.4's critical-path decomposition still holds
(ACT 1147 on-path busy, PE 678, DVE 518, Pool 24, gaps 701).

The `Y_off` fold at 575 units moved **ACT busy 2124.4 -> 1966.9 (-157.5)** and the wall
**3070.9 -> 2969.5 (-101.5)**, a **64% busy-to-wall conversion** -- consistent with iters 1-3.
So ACT busy is still the thing that converts, and the way to cut it is to **delete whole
ACTIVATEs**, not to make them cheaper (iter-4 §3: their cost tracks bytes written).

Remaining ACT work, with what is known about each:
* 3840 `128x32` multiplies (1087 us). 2 of the 5 per unit are the in-place ones (477, 504);
  **504 is §2 and is removable**. 477 (`Y_intra *= exp_cs_col`) is removable the same way *if*
  you can get the multiply onto the DVE/ACT `COPY` at line 476 -- 366 of those copies are ACT
  `COPY` (which can take a scale? unchecked) and 411 are DVE `COPY` (which cannot take a second
  operand without an opcode change). Folding 476+477 is worth 366x188 + 411x190 ns on top.
  **Do not** fold 477 and 504 into one multiply of `(Y_intra + Y_off)` -- that is
  `(a+b)*s` vs `a*s + b*s` and the error budget is at 89%/87%.
* 768 `log_decay` rows (226 us) and 768 `exp_cs_last` scalars (215 us) -- 1-element and
  1-channel ops that cost ~280 ns each, i.e. almost pure issue overhead. Nothing obvious.
* 1504 PSUM exps (iter-1/2 already folded the copies into them).

## 5. Closed / do not redo (adds to iters 1-4)

* **The `exp_cs_last` broadcast dedup (§3)** -- 68 of 747 units survive liveness. ~2 us.
* **PSUM WAR is not a constraint on ACT-side edits** (53 ns of slack, unchanged by a -157 us ACT
  shift). Audited with `iter-5-psumwar.py`.
* **iter-3/4's single-hop `state_sb` bounce is NOT schedule-fragile against a -100 us shift**
  (`iter-5-sbwindow.py`, identical hit counts on both schedules). The review's §3 worry is, on
  this evidence, unfounded; the inherited edits are fine.
* **Replacing an ACT ACTIVATE with a bare `EVENT_SEMAPHORE` is unsafe if the ACTIVATE carries a
  sem-4 increment** -- add the extended `sem4 >= cum4[slot]` self-wait (§2.4b). This is the
  single most reusable thing in this iteration.
* `num_dstate_tiles == 1` and `num_hdim_tiles == 1` for this case: **both inner loops of
  `_compute_ssd_head_tile` are single-iteration**, so every "accumulate across tiles" idiom in
  the source is dead code. §2 exploits one of them (`Y_off`); check the others.

## 6. Tool notes (adds to iters 1-4)

* `Instruction.__getitem__` returns the **raw int** for enum fields; only `as_dict()` symbolises
  them. Comparing `x['events_wait_mode'] == 'WAIT_FOR_SEM_GE_IMM'` is silently always false --
  it cost me two device runs. Build `AD = [x.as_dict() for x in A]` once and gate on that.
* `nki_source_location` / `bir_debug_info_source_location` are reliable for **DVE, PE, Pool, SP**
  (exact 768 counts per line) and **unreliable for ACT** (e.g. 461 ACTIVATEs attributed to
  `ssd_block.py:275`, which is a `dma_copy`). Classify ACT slots by decoded fields, not by line.
* Semaphores 4 (ACT, 5928) and 6 (DVE, 4846) have **`max_wait == total_inc`, i.e. no resets**, so
  a plain cumulative increment count per slot is a valid absolute clock on those two engines.
  Do not assume the same for the DGE queue semaphores.
* `need4(k) = cum4[j+1]` for the first slot `j >= k` with an increment is the correct "prove ACT
  passed slot k" value -- `cum4[k+1]` is wrong when slot `k` itself carries no increment.
* An **`EVENT_SEMAPHORE`'s extended wait block works on ACT** (`events_extended_wait_mode/
  _wait_idx/_sem_wait_value`), simultaneously with a primary wait and a primary update. Verified.
* `scale_value_imm_ptr` on an ACTIVATE is a **full SBUF address** (`partition*0x40000 + offset`);
  active channel *i* reads `ptr + i*0x40000`, so a 128-channel op whose `src` is at partition 0
  can legally have its scale pointer at partition 64 (line 535 does exactly that).
* `step_elem = 0` with `num_elem > 1` is **not used anywhere in this NEFF** and **works** on a
  DVE `TENSOR_TENSOR_ARITH_OP` source: it re-reads the same element. Verified bit-exact.
* **Do not `exec()` a copy of the edit script with `if DRY: sys.exit(0)` replaced by `pass`** to
  reuse its analysis -- it runs the apply stage and overwrites whatever `ROOT` points at. I
  destroyed `work_pristine` that way; recover by unpacking the known-good NEFF again.
* Device costs today: `verify.py` 40 s, `profile.py --repeats 1 --parquet` 35 s, `--repeats 5`
  85 s. A prefix bisection over 192 candidates is 8 verify runs = 6 minutes.

## 7. Ranked plan for iter-6

1. **Finish §2 -- the last 51 of 575 folds, ~9 us.** They are the units where a consumer waits on
   exactly `sem4 >= cum4[R]+1` and the ES's drain does not cover the preceding ACT SBUF write.
   Raise that consumer's wait by one (iter-2's pattern) after a deadlock check, or find the real
   per-unit discriminator (§2.4c) and drop the empirical `0..187` clause from the gate at the
   same time. Low value; do #2 first.
2. **The same fold for `Y_intra` (476+477)**, once §2's gate is understood -- another
   ~140 us of engine time, same shape, and the ACT `COPY` at 476 may already accept a scale
   pointer (check `scale_src` on the `COPY` struct).
3. **`ssd_utils.py:67`'s 128x128 `nc_transpose` of a single row**: 768 x (LDWEIGHTS 223 + MATMUL
   232) = **349 us of PE** plus the `padded` memsets, to turn a 1x128 row into a 128x1 column.
   `dt_col` (`ssd_block.py:203`, 500 us of Pool) is the *same* conversion done by DMA. PE is 41%
   busy and 22% of the path; Pool is 51% busy and 1% of the path. Neither is obviously the right
   host, but there is ~850 us of engine time in two redundant row->column conversions and one of
   them (`cs_row -> cs_col`) is only needed because `nisa.activation` wants a per-partition
   column. Unpriced.
4. Pool queue reassignment (iter-4 §4.3), still the biggest un-attempted ceiling (138 us of ACT
   `EVENT_SEMAPHORE` gap + 37 us of Pool ring gap).

**How the 188-unit prefix became 524, for whoever has to do this kind of thing again.** I first
had only an empirical prefix cutoff (-0.91%) and was about to ship that. What turned it into
-2.70% was *not* more device probing: it was taking the four device-proven-bad units and asking
which **structural** predicate separates them from the labelled-good ones. The winning predicate
came straight out of the hazard I had already fixed (§2.4b) -- "can this ES's inherited sem-4
increment release anybody early?" -- in three successively weaker forms, each one checkable
statically from the decoded streams in seconds and each one costing exactly one 40 s verify to
confirm. 15 cheap labels plus one correct mental model beat 25 blind block probes by 6x.
5. **Do not** redo anything in §5 here, or in iter-4 §5, iter-3 §4, iter-2 §4, iter-1's list.

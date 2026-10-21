# iter-5 (ssd_mamba2_1b3_s2048): folded the `cs_col` PSUM->SBUF copy into its two `exp`
# consumers (984 ACT instructions deleted), and found the rule that governs every
# "replace a compute slot with an EVENT_SEMAPHORE" edit on this kernel

**Shipped: 4855.9 -> 4777.2 us (-1.62%). `verify.py` VERIFIED three times, `max_abs_err`
5.234e-07 / 3.334e-07 — byte-identical to iter-4 (the edit changes no arithmetic).
Only member changed: `Activation0.bin` (2952 of 14074 slots). Against iter-0:
7154.9 -> 4777.2, -33.2% (1.498x).**

My 5-repeat control on iter-4's NEFF (md5 `1115f0b0…`): **4855.930 us** (min 4854.7, max 4858.0,
spread 0.07%). ACT 3509.6 us busy (72.3%), DVE 2437.7, Pool 2072.2, PE 1921.8, DMA 1319.2, SP 278.6.
Shipped md5 `ba28564445c1ffdc1359068fd057776a`.

## READ THESE FOUR FIRST

1. **§1 — THE RULE THAT EXPLAINS THREE FAILED DEVICE RUNS.** An `EVENT_SEMAPHORE` executes on the
   **sequencer, in parallel with engine compute**, so a `SEM_INC_COMPLETE` placed on one can
   **overtake an ACTIVATE that is still computing**. A placeholder that inherits a compute slot's
   semaphore update therefore *releases its consumers early* and corrupts data. iter-3 §1a had
   already observed the timing (and lost four device runs to it); **what is new here is the fix** —
   a handshake: the placeholder must **wait** for the increment of the real instruction it must not
   outrun, which is what makes iter-4 §1's recipe usable on *compute* slots rather than only on
   dead DMA triggers. It still cost me three device runs before I read iter-3 §1a properly
   (one `PSUM_COLLISION`, two wrong-value), and a fourth on DVE (§4b).
2. **§3 — THE SIZING PROBE FOR iter-6: deleting the 1024 `exp_cs_last` ACTIVATEs is worth
   -194.8 us (-4.1%)**, 80% conversion, because they sit on the *serial head chain*. That is 2.5x
   this whole iteration. But see §4: the obvious implementation *lengthens* that chain, so the
   realised gain may be much less than the probe. Measure, do not assume.
3. **§5 — Pool/DMA are NOT on the critical path and review candidate #2 is dead.** A destructive
   probe removed 193 us of Pool busy and 519 us of DMA busy and the wall moved **-12.8 us (2%
   conversion)**. Do not spend an iteration on Pool DGE merging.
4. **§4 — `cs_col = exp_cs_last.src + 12` is a COINCIDENCE, not a pairing.** It matches 864 of
   1024 sites and still gives `max_abs_err 1.97e+01`. The multiplicities refute it (75 distinct
   `src+12` values vs 54 real `cs_col` addresses; only 30 counts agree). The `exp_cs_last` pairing
   is still unsolved; §4 has the two joins I tried and why each failed.

## 0. What the shipped edit is

`ssd_utils.py:56-75` (`transpose_row_to_column`) ends with
`nisa.tensor_copy(dst=column[0:Q,0:1], src=transposed_psum[0:Q,0:1])` — an ACT `ACTIVATE`
`func=128, nch=128, scale_src=INSTRUCTION_IMMEDIATE, scale=1.0, bias=0.0, fp32->fp32,
accum=IDLE, src marker 2 (PSUM), num_elem [1,1,1]`. **1020 of them.** Its only consumers are the
two `exp`s right after it (`ssd_utils.py:119/121`, `ACTIVATE func=7 nch=128`, scale `+1.0` and
`-1.0` after iter-1's rewrite):

```
p0  ACT f=128 nch=128 src=PSUM(0x02000000) dst=cs_col(0x9610) scale=+1.0   <- the copy
pa  ACT f=7   nch=128 src=cs_col            dst=0x9618        scale=-1.0   <- exp_neg_cs_col
pb  ACT f=7   nch=128 src=cs_col            dst=cs_col        scale=+1.0   <- exp_cs_col, in place
```

Because the copy's scale is exactly 1.0, `exp(s*(1.0*p)) == exp(s*p)` for `s = ±1`, so pointing
both `exp`s at the PSUM word is **bit-exact** and the copy becomes dead. Source confirms nothing
else reads `cs_col` (`compute_cumulative_decay` returns only `cs_row, exp_cs_col, exp_neg_cs_col`),
and a complete address scan over all five engines confirmed it on the binary (§2).

**Shipped layout** (`iter-5-apply.py`), per site, entirely inside `Activation0.bin`:

| slot | new content | wait | sem-4 update |
|---|---|---|---|
| p0 | `exp_a` (the first `exp`), `src` = the copy's PSUM word | **the copy's wait, verbatim** | none |
| pa | `exp_b` (the second `exp`), `src` = the same PSUM word | `sem4 >= pre` | **takes the increment that was at p0** |
| pb | `EVENT_SEMAPHORE` placeholder | **`sem4 >= V0`** (the handshake, §1) | the 2nd increment when the group had two |

`pre` = sem-4 increments strictly before p0; `V0 = cum4[p0] = pre+1`. Both `exp`s' own
`sem4 >= V0` waits are replaced by `sem4 >= pre`: they were enforcing the ACT->ACT dependency on
the copy's write, which no longer exists, and keeping `>= V0` would deadlock because `V0` is now
produced at `pa`.

Two properties make this safe without touching any other engine:
* **No semaphore threshold anywhere changes.** Each increment stays in the group and moves to an
  equal-or-later slot, so the sorted sequence of increment positions is pointwise >= the original:
  every wait in the program fires at the same time or later, never earlier.
* **The PSUM WAR hazard closes itself.** PE reclaims the transpose's PSUM bank behind a
  `LDWEIGHTS W sem4 >= cum4[p0]` (see §2). Moving that increment from p0 to pa means PE is released
  only after **both** PSUM reads have completed. That is why `pa`, not `pb`, carries it.

984 of 1020 sites applied. Rejections: 34 `PE release before pa` (PE's governing sem-4 wait is
already satisfied *before* the copy runs, so no increment placement can gate it) and 2 copies with
no sem-4 update.

### 0a. Measurement

| NEFF | median us (5 repeats) | min-max | ACT busy | sites |
|---|---|---|---|---|
| iter-4 (control) | 4855.930 | 4854.7-4858.0 | 3509.6 | - |
| contiguous groups only | 4801.250 | 4801.0-4801.5 | 3389.5 | 659 |
| + non-adjacent groups | 4783.052 | 4782.3-4783.6 | 3349.9 | 922 |
| **+ relaxed wait check (shipped)** | **4777.210** | 4775.3-4778.2 | 3343.6 | **984** |

-78.7 us, -1.62%, against a 2-5 us spread. 984 ACT instructions deleted = 166 us of ACT busy =
**79 ns of wall per ACT instruction** (47% conversion). That is *below* iter-4's 184 ns: this copy
is off to the side of the per-unit critical path. §3's probe shows a head-chain ACT instruction is
worth 190 ns, so **where an ACT instruction sits matters more than that it exists.**

## 1. THE RULE: an `EVENT_SEMAPHORE` placeholder must not carry a compute slot's increment
## unless it waits for the increment before it

This is the single most useful thing I learned and it generalises iter-4 §1.

An `EVENT_SEMAPHORE` costs ~64 ns on the **sequencer**, which runs **ahead of / in parallel with**
the engine datapath. An `ACTIVATE`'s `SEM_INC_COMPLETE` fires when its *compute* completes. So if
you swap an `ACTIVATE` for an `ES` and copy its `events_update_*` across, that increment now fires
as soon as the sequencer reaches the slot — **possibly before the previous ACTIVATE has written
SBUF**. Every consumer of that semaphore value is released early.

iter-4's DGE replacement was immune only because its increment stood in for a DMA. iter-3 §1a
("`EVENT_SEMAPHORE` runs on the sequencer and increments *before* the preceding compute instruction
has finished writing") is the same observation and cost that iteration four runs; what is new is
**the fix**, which makes the technique usable on compute slots:

> Give the placeholder a wait on the **immediately preceding increment of the same semaphore**
> (`sem4 >= cum4[slot-1]`), or better, on the increment of the specific instruction it must not
> outrun. `EVENT_SEMAPHORE` has a full `events_extended_wait_*` block (offsets 32/33/36 on
> `CTRL_ES_STRUCT`, with its own `events_extended_sem_wait_value`), so it can carry its original
> wait **and** the handshake.

Evidence (all on the 357-1020 site range, each a device run):

| variant | result |
|---|---|
| repoint both `exp`s at PSUM, **keep the copy and every semaphore byte** (the isolating control) | **VERIFIED bit-identical**, 659 and 1318 refs |
| + copy -> `ES` keeping its wait **and** its update, 659 statically-safe sites | **`PSUM_COLLISION`** |
| + copy -> `ES` keeping its wait **and** its update, 357 statically-safe sites | `max_abs_err` **5.10e-02** |
| rotate payloads + defer increments, placeholder waits `sem4 >= pre` (too weak) | `max_abs_err` **5.48e-01** |
| **rotate payloads + defer increments, placeholder waits `sem4 >= V0`** | **VERIFIED bit-identical** |

The first row is the control that matters: identical data flow, zero sync change, bit-exact. So the
fold itself was never the problem, and three failures in a row were all one bug.

## 2. The static safety machinery (reusable; `iter-5-apply.py`)

Worth keeping because it replaces "measure the slack and hope":

* **Who owns which semaphore.** All waits in this NEFF are `WAIT_FOR_SEM_GE_IMM` (enum **5**, not
  1) and all primary updates are `SEM_INC_COMPLETE` (enum **19**, not 1) — check the raw byte, the
  symbolic name hid this from me for one run. `+1` does not consume `events_semaphore_value`.
  **sem4 = ACT's completion counter** (7360 ACT increments, nothing else touches it),
  **sem5 = PE's**, **sem6 = DVE's**, **40-71 = the DGE semaphores**. Semaphores are monotone, so
  "an earlier wait on the same engine with a >= value" implies a later one.
* **PE's PSUM-bank guard is on the `LDWEIGHTS`, not the `MATMUL`.** The clobbering `MATMUL` has
  `events_wait_mode = NONE`; the `LDWEIGHTS` in front of it carries `W sem4 >= k`. Compute the
  *governing* guard as the **running max of sem-4 wait values down PE's stream** — once PE passes a
  wait it stays satisfied.
* **The release test.** Build the new sem-4 increment vector, `ncum = cumsum`, and require
  `searchsorted(ncum, gov[pe_slot]) >= pa` — the first ACT slot at which sem4 reaches PE's guard
  must be at or after the last PSUM read. This is a *static* test; the timing-slack version
  (`PE write start - last read end`, p50 **528 ns**, negative at 404 of 1020 sites) is only a
  sanity check and 288 of those "overlaps" turned out to be harmless.
* **Deadlock test.** For every ACT slot in the group, `searchsorted(ncum, its sem4 wait) < its own
  index`. Relaxing my first (too strict) "no interleaved increment before `pa`" rule to this
  general test took the site count 922 -> 984 for +5.8 us.
* **Dead-value test — use the COMPLETE field list.** My first scan used
  `{src,src0,src1,src2,dst}_mem_pattern_start_addr_addr_immediate` and silently missed
  `*_start_addr_addr_immediate` (DGE endpoints), `scale_value_imm_ptr`/`bias_imm_ptr`/`imm_imm_ptr`
  (ACT pointer scalars) and `imm0_imm_ptr` (`TENSOR_TENSOR_SCAN_ARITH`). Iterate
  `ins.fields()` and take **every** name ending in `_start_addr_addr_immediate` or `_imm_ptr`.
  Also note `STREAM_SHUFFLE`'s `src` word is at **byte 12** and its `dst` at **byte 44** — not 16/48
  like `ACTIVATE`.

## 3. PROBE FOR iter-6: what each remaining ACT instruction is worth

Destructive, numerically wrong on purpose, ~2.5 min each, run on the shipped NEFF
(`iter-5-probe.py`).

| probe | ACT busy | wall | conversion |
|---|---|---|---|
| shipped | 3343.6 | 4777.2 | - |
| **replace all 1024 `exp_cs_last` (`ACTIVATE func=7 nch=1`) with `ES`** | 3101.3 (-242.3) | **4582.4 (-194.8)** | **80%, 190 ns/instr** |
| (iter-4, for comparison: the `state_sb` scale ACTIVATEs) | -357 | -188.8 | 53% |

ACT is still the top engine: **70.0% busy, 3343.6 us of a 4777 us span**, 9973 `ACTIVATE`s at
335 ns mean. The per-unit ACT population is now 10 instructions (`log_decay` `:100` nch=1;
`exp_cs_last` `:226` nch=1; `exp_cs_col` and `exp_neg_cs_col` `:119/:121`; and the six
pointer-scalar multiplies `dtx :454`, `X_scaled :463`, `Y_intra :477`, `Y_off :504`,
`dtx_decayed :514`, `state_sb :535`). The `:117` copy is gone from 984 of 1024 units.

ACT's gaps: 1742 us over 8535 gaps, **median 34 ns**, mean 204 ns, 1630 us of it sitting in front
of `ACTIVATE`s and distributed across source lines *in proportion to instruction count*. There is
no single stall to attack — which is the positive case for "delete ACT instructions", and the
reason the conversion rate is what it is.

## 4. `exp_cs_last`: still the biggest lever, still unsolved, two joins refuted

`exp_cs_last = exp(cs_row[Q-1])` (`ssd_block.py:226`) and `exp_cs_col[Q-1] = exp(cs_col[Q-1])
= exp(cs_row[Q-1])` are `exp` of the identical fp32 number, so deleting the former and pointing its
broadcast at `exp_cs_col` partition 127 is bit-exact. §3 prices it at **-194.8 us**. Everything
about the encoding is settled (iter-4 §7.1: mask in the `LOAD_MASK_SELECT`'s
`immediate_uint8[0:32]`, offset **32**, count **32**; partition field = bits 20..24 in units of 4,
so read partition 127 as `src_start` field **24** (partition 96) + mask `[31]*32`; shuffle `src`
word = `0x01800000 | (cs_col_offset & 0xFFFFF)`; `mask_enable` stays 0; shuffles have
`num_active_channels = 32`, `dst_start` partitions 0/32/64/96).

**What is missing is still only the pairing, and I refuted two more candidates:**

* **`cs_col = exp_cs_last.src + 12`. REFUTED.** It looked structural — `exp_cs_last.src` is
  `cs_row + 127*4`, and three hand-checked units all had `cs_col = cs_row + 0x208`, so
  `cs_col = src + 12`; it matches **864 of 1024** sites. Shipping 8 sites gave `max_abs_err`
  **1.968e+01** on `y` with `final_state` still perfect. The refutation is static and takes
  10 seconds: there are **75 distinct `src+12` values but only 54 distinct `cs_col` addresses**,
  and only **30** addresses have matching multiplicities. `cs_row` and `cs_col` come from
  *independently rotating* allocation pools, so the offset is a coincidence at 84% of sites.
  *Any* address-arithmetic pairing on this kernel needs this multiplicity check before a device run.
* **Join through the DVE `decay_factor` multiply. REFUTED as written.**
  `decay_factor = exp_neg_cs_col * exp_cs_last_Q` (`:229`) is a DVE `TENSOR_TENSOR_ARITH_OP`
  `op=MULT nch=128` whose two operands are exactly the two things I need to connect, so it should be
  a one-hop join: `src0/src1` -> `exp_neg_cs_col` (= the `-1.0` `exp`'s dst) and ->
  `exp_cs_last_Q` (= the shuffles' dst) -> shuffle `src` = `exp_cs_last`'s dst. It mispairs:
  334 of 1024 units got shuffles with differing `src`, and two different units resolved to the same
  `exp_cs_last_Q` and overlapping shuffle sets. Cause: **there are 6012 `STREAM_SHUFFLE`s, 5.87 per
  unit, not 4** — iter-3 replaced the *second* broadcast at only 545 of 1024 sites, so 479 units
  still have **two** groups of four shuffles reading the same `exp_cs_last`, and a ±30 us time
  window picks members of both. Any future attempt must first partition the shuffles into groups by
  `(dst, contiguous DVE slot run of 4 with its 4 `LOAD_MASK_SELECT`s)` and only then join.

**The join I would build next** (not attempted, no device time spent): go through PSUM, which is
unique per unit at any instant.
`exp_cs_last.src - 508` = `cs_row` base -> the DVE `COPY` with `src == cs_row` base (that is
`ssd_utils.py:67`, `padded[0:1,0:Q] = cs_row`) -> its `dst` = `padded` base -> the PE `LDWEIGHTS`
with `src == padded` base, whose following `MATMUL` has `dst_mem_pattern_num_elem == [1,1,1]`
(the compiler emits the `nc_transpose` with one active row) -> that `MATMUL`'s PSUM `dst` -> **after
this iteration's fold, the two `exp`s of that unit are the only `ACTIVATE func=7 nch=128` slots
whose `src` is that PSUM word** — which is exactly the site list `iter-5-apply.py` already builds.
Disambiguate each hop with `prof/Instruction.parquet` timestamps. Then re-run the 8-site test and
require **bit-identical**.

**And one warning the probe cannot see.** `exp_cs_last` depends only on `cs_row`; `exp_cs_col`
depends on `cs_row -> padded -> PE transpose -> PSUM -> cs_col`. Repointing the broadcast at
`exp_cs_col` therefore makes `decay_factor` (and `state_sb *= exp_cs_last_N`) wait on a **longer**
chain, while the §3 probe measured deletion with the dependency simply removed. So -194.8 us is an
**upper bound** and the realised number could be far smaller, or negative. The cheap way to find out
before building the pairing: take the probe NEFF and, instead of deleting, add the extra wait
(`raise each shuffle group's first `LOAD_MASK_SELECT` to `sem4 >=` the value produced at the
unit's `exp_cs_col`) while keeping `exp_cs_last` alive — if the wall moves back up to ~4777, the
lever is gone and iter-6 should not build it. **Do this probe first.** It is ~3 minutes.
Note that in the shipped layout the covering value is `V0` (carried by `pa`), and for **286 of the
305** sites my join accepted, the `LOAD_MASK_SELECT` wait was **already >= V0**, i.e. no sync change
would have been needed at all — so the raise may be close to free.

## 4b. BUILT, MEASURED, NOT SHIPPED: the redundant `Y_off` zero-init
## (probe -58.4 us, half-edit **+28.9 us regression**, full edit needs a wait retarget)

`num_dstate_tiles == 1` for this shape (`dstate 128`, `dstate_tile = min(128, P_MAX) = 128`), so
`ssd_block.py:486-503` reduces to `Y_off = 0.0 + <one tile>`, and `0.0 + x == x` for every finite
fp32 `x`. **The compiler already fused `Y_off_tile = tensor_copy(Y_off_psum)` into the add**: the
DVE `TENSOR_TENSOR_ARITH_OP op=ADD` has `src0 == dst == Y_off` (SBUF) and **`src1` with marker 2,
i.e. PSUM**. So the live sequence is only

```
DVE MEMSET  dst=Y_off, set_value 0.0            <- dead
DVE ADD     dst=src0=Y_off, src1=Y_off_psum     <- equivalent to a plain COPY
ACT         Y_off *= exp_cs_col  (src=dst=Y_off)
DVE ADD     y_chunk = Y_intra + Y_off
```

**1040 sites identified with zero rejections** (1024 `Y_off` plus 16 `CB_accum` from
`_compute_chunk_shared_projections_tiled`, which has the same `memset`-then-single-tile shape).
`iter-5-yoff-REGRESSION.py` reproduces everything below.

| variant | DVE busy | wall | verdict |
|---|---|---|---|
| shipped | 2426.5 | 4777.2 | - |
| **probe**: both the `MEMSET` *and* the `ADD` -> `ES` (numerically wrong) | 2221.5 (-205.0) | **4718.8 (-58.4)** | **28% conversion — DVE now converts far better than iter-3's 8%** |
| **real half-edit**: `ADD` rewritten as a DVE `COPY` from PSUM in the same slot with its sync bytes verbatim, `MEMSET` -> `ES` + handshake | 2385.2 (-41.3) | **4806.1 (+28.9)** | **VERIFIED bit-identical but a REGRESSION — reverted** |
| same without the handshake on the `MEMSET` placeholder | - | `PSUM_COLLISION` | more evidence for §1 |

**Why the half-edit loses.** Retiring one `MEMSET` (112 ns) costs one `ES` (64 ns) *plus* the
handshake's extended wait, and that wait serialises DVE against its own semaphore and defeats the
sequencer's run-ahead. Net DVE busy only -41 us and the wall moved the **wrong way** by 29 us.
**Generalisable lesson: an `ES` placeholder is not free — it costs ~64 ns and a wait. Replacing a
single cheap instruction is not worth it; the technique pays only when the deleted instruction is
expensive or several go at once.** (The shipped §0 edit retires a 170 ns `ACTIVATE` per group and
wins; this one retires a 112 ns `MEMSET` and loses.)

**The probe's -58 us is UNREACHABLE — treat §4b as closed.** Getting it requires deleting the `ADD`
as well, which means the ACT multiply `Y_off *= exp_cs_col` must read `Y_off_psum` directly, and
**that is exactly what iter-3 §3b already refuted by hardware**: `PSUM_COLLISION` at 1199 and at
562 sites, with the rule *"a PSUM region may only be drained by the engine the compiler assigned to
drain it"*. The `Y_intra`/`Y_off`/`state_delta` matmuls carry `fp32_mode = LOW_HIGH` and
`psum_accumulate_flags = 1` (two `MATMUL` slots accumulating into one bank) and ACT may not read
them at all. So the only shippable form keeps DVE as the PSUM drainer — i.e. the half-edit above,
which loses 29 us. I re-derived this candidate from the source without checking iter-3 §3b first
and spent ~20 minutes on it; **read iter-3 §3b before touching any PSUM read.**

**What my §0 edit adds to iter-3 §3b.** Its stated exception — `transpose_row_to_column`
(`ssd_utils.py:67`/`:117`), whose `MATMUL` has `num_active_rows = 1` and a single PSUM pass — is
exactly the one I folded, and this iteration shows the exception is **stronger than iter-1 thought**:
ACT may read that PSUM word **twice, from two different instructions**, bit-exactly, on 984 units.
iter-1's "a *second* ACT read of a PSUM bank is rejected" does not hold for a single-pass
`num_active_rows = 1` transpose result. The discriminator is the producing matmul, not the number
of readers.

## 5. REFUTED: Pool and DMA are not on the critical path (review candidate #2 is dead)

The probe the review asked for, plus more. Shrinking **all 3169 Pool DGE triggers** to a 1-element
payload (`src_num_elem = dst_num_elem = 1`):

| | Pool busy | DMA busy | wall |
|---|---|---|---|
| shipped | 2072.1 | 1321.3 | 4777.2 |
| probe | 1879.4 (-192.7) | 801.9 (-519.4) | **4764.5 (-12.8)** |

**712 us of Pool+DMA busy removed buys 12.8 us of wall — 1.8% conversion.** Two conclusions:
* The DGE row's host cost is **~594 ns fixed, independent of payload** (193 us / 3169 triggers is
  the payload part of ~655 ns), confirming the figure iter-4 §7.2 imported from other kernels.
* Pool is 43% busy with ~2.9 ms of gaps and converts at ~2%. Merging N heads' `dt_col` into one
  trigger could cut Pool busy to ~2076/N us, i.e. at most ~1.8 ms, which at 2% is **~36 us** for a
  relayout plus new fences that the verifier cannot check. **Not worth an iteration.** Do not redo.

## 6. Ranked for iter-6

Conversion rates measured *this* iteration: a head-chain ACT instruction is **190 ns of wall**, an
off-path one **79 ns**, Pool/DMA busy **~2%**. Re-measure; they have moved every iteration.
Only bit-exact edits are viable (iter-4 §6: `max_abs_err` is at 84% of the limit and the budget was
already 83% spent in iter-0, so there is no headroom to buy back — do not plan on it).

1. **`exp_cs_last` (§4).** -194.8 us by the probe, but run §4's "does the longer dependency eat the
   gain" probe **first** (3 min, could kill it outright). Then build the PSUM-based join, which the
   fold's site list now hands you for free. Falsifier unchanged: 8 sites must be bit-identical.
   Note on the 479 units iter-3's second-broadcast dedup skipped (`STREAM_SHUFFLE` is still DVE's
   largest item: 6012 slots = 5.87/unit, 885 us): **do not chase them for speed** — iter-3 already
   shipped a wider version that removed 402 us of DVE busy and the wall went **up** (5119.6 vs
   5105.3). They are worth touching only to make the shuffle groups 1-per-unit so §4's DVE join
   becomes unambiguous, and even that is better done by grouping statically (4 consecutive
   `LOAD_MASK_SELECT`+`STREAM_SHUFFLE` pairs with a common `dst`) than by editing.
2. **The 36 sites this iteration rejected** (34 `PE release before pa`, 2 no sem-4 update), ~3 us.
   Only worth it as a warm-up; the 34 need PE's `LDWEIGHTS` guard raised, which is a sync change on
   another engine and the first thing §1 says to be careful with.
3. **`log_decay` (`ssd_utils.py:100`, `ACTIVATE f=128 nch=1 scale_src=POINTER_IMMEDIATE`, 1024
   slots)** is the other nch=1 head-chain instruction, so ~190 ns/instr = ~195 us. `log_decay =
   dt_row * A_head` feeds the DVE `TENSOR_TENSOR_SCAN_ARITH`. `A*cumsum(dt)` is not bit-exact, and
   moving the multiply to DVE costs 1 ulp (iter-4 §6). **Open question worth 10 minutes:** does the
   scan's `op0 = MULT` second operand accept a pointer scalar, so the scan could multiply by
   `A_head` itself? If yes it is one ACT instruction per unit for free.
4. **Software-pipeline the serial head** (iter-3 §4, iter-4 §7.3). §3's 80% conversion on the head
   chain is the first direct evidence that this chain is the constraint. Still the largest lever and
   still the most expensive; needs a destructive "drop the head chain's waits" probe first.
5. Do **not** redo: §4b (all of it) and §5 here; iter-3 §3b (ACT may not drain a LOW_HIGH matmul's PSUM); iter-4 §2 (DGE threshold remaps), §5 (ACTIVATE merging), §6 (ACT->DVE
   multiplies); iter-3 §3/§3b; iter-2 §4/§1. And do not "fix" `semaphores.py`'s
   `sems 40-71: 1088 max_wait / 128 total_inc` alarm — it is iter-4's documented false positive
   (the tool does not model `SEM_ADD_IMM_COMPLETE`) and this iteration changed no DMA semaphore.

## 7. Tool notes (additions to iter-4 §8)

* **The nkilib source is on this machine and is readable**:
  `/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/experimental/scan/`
  (`ssd.py`, `ssd_block.py`, `ssd_utils.py`). The loop forbids recompiling, not reading. Reading
  `ssd_utils.compute_cumulative_decay` and `_compute_ssd_head_tile` is what turned the ACTIVATE
  census into a data-flow map in ten minutes, and it is how I knew `cs_col` had exactly two readers.
  The `nki_source_location` column in the parquet names these lines (but its per-line *counts* are
  off by up to 30%, so trust the decoded stream, not the attribution).
* **The six distinct `ACTIVATE` configurations are the whole ACT program.** `isa.py` plus
  `as_dict(symbolic=True)` grouped on
  `(activation_func, num_active_channels, scale_src)` gives: 5841 `f=128 nch=128 ptr` (the six
  per-unit pointer-scalar multiplies, minus iter-2's 303 merges), 2044 `f=7 nch=128 imm` (the two
  `exp`s), 1024 `f=128 nch=1 ptr` (`log_decay`), 1024 `f=7 nch=1 imm` (`exp_cs_last`),
  1020 `f=128 nch=128 imm` (the `cs_col` copy this iteration deleted), 4 stragglers.
  **Do this census first in any future iteration** — it is one minute and it tells you the whole
  per-unit ACT budget.
* `as_dict(symbolic=False)` for enum **byte values** before writing raw bytes:
  `WAIT_FOR_SEM_GE_IMM = 5`, `SEM_INC_COMPLETE = 19`, `NONE = 0`. An assertion against the raw byte
  caught my wrong guess (1) before it reached the device; keep writing those assertions.
* `evt_wait_time_ns` in `Instruction.parquet` is **not usable as wall time** — it sums to 79 ms on
  ACT over a 4.8 ms span. Use `start_ts`/`end_ts` gaps instead.
* `iter-5-apply.py` reproduces the shipped `Activation0.bin` byte for byte. Run order:
  `cp baseline/kernel.neff kernel.neff` reverts to **iter-0**, so save the start first
  (`git show HEAD:kernel.neff > kernel_start.neff`), then
  `neff.py unpack kernel_start.neff work/`, `profile.py --parquet prof`, `iter-5-apply.py`,
  `neff.py pack work/ kernel.neff`. `iter-5-try.sh` does the whole revert->edit->pack->verify loop
  in ~55 s. `iter-5-last-REFUTED.py` is §4's refuted `exp_cs_last` edit (keep it: it has the
  working shuffle/`LOAD_MASK_SELECT` byte surgery and the `SHUFONLY`/`FREEONLY` isolating modes),
  and `iter-5-probe.py` has both §3 and §5 probes.

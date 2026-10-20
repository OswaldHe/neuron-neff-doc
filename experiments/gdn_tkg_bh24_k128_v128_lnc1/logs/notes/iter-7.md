# iter-7 notes — gdn_tkg bh24 k128 v128, LNC=1 (24 heads on ONE core)

## Result

**62.547 -> 60.948 us, -2.6 %** (`profile.py --repeats 5`: 60.942-61.282, spread 0.56 %).
`verify.py` PASS and **bit-identical across 4 + 12 + 12 device runs**; `max_abs_err` unchanged
at 2.322e-2 / 1.633e-3 (83 % / 6 % of the limit), `launch_rules.py` PASS, `semaphores.py`
balanced, `cycle.py` acyclic. Only `DVE0.bin` (213 -> 196 slots) and `Activation0.bin`
(170 -> 156) changed; `PE0.bin` is untouched. md5 `6bfb949777e93ba11ceeea07bd07ee0c`.

### Final stack (rebuild from the iter-6 NEFF with `iter-7-scripts/try3.sh`)

```
applyBD.py work/payload/sg00 noponly          # -0.72  drop the 18 no-op DVE ES
applyBG.py work/payload/sg00 all              # -0.26  clone reorder
applyBF.py work/payload/sg00 Activation all   # -0.17  drop the 14 update-free ACT ES
applyAE.py work/payload/sg00 1 0,1,2,3        # -0.2   head 20's f128
```

I first shipped the variant that also drops 9 of the 18 DVE *self-completion* barriers
(60.50 us, md5 `79e64d0b…`). It passed `--runs 4` and then `--runs 8`, and **failed a second
`--runs 8`** — so it is racy and I backed it out in favour of the no-op-only variant, which is
~0.5 us slower but passed 4 + 12 + 12 runs bit-identical.

## READ THIS FIRST: `verify.py` with one run is NOT enough, and 4 runs is not either

Three builds passed `verify.py` (1 run) and `rows.py` and were **non-deterministic**: `--runs 4`
showed one row differing between runs (`max_abs_err` 2.1 on row 9, or 2.3 on row 7). The iter-6
NEFF **is** deterministic over 4 runs, so this is a regression I introduced, not pre-existing.

Worse: I bisected the 18 DVE self-barriers into two halves, **each half passed `--runs 4`
on its own, and their union failed.** These races are timing-marginal, so a passing
determinism check is evidence, not proof.

And worse still: the variant I was going to ship passed `--runs 4` *and* `--runs 8`, then
failed a **second** `--runs 8`. Per-run flip probability for these races looks like ~5-15 %.

**Rules for the next iteration:**
* every candidate must pass `tools/verify.py --runs 4`, and the one you ship **`--runs 12`,
  twice** (the shipped NEFF here has 28 consecutive bit-identical runs);
* never conclude from passing subsets that their union is safe — re-test the union;
* `scripts/rows.py` (one device run, prints which rows are wrong and the dev/gold ratio) is
  for fast triage only, never for acceptance;
* the metric harness runs `verify.py` once, so a marginally racy NEFF is a coin flip on the
  whole iteration being discarded. Prefer the smaller, deterministic win.

---

## What the body actually is (this corrects iter-6's model)

Measured on the iter-6 NEFF (`profile.py --parquet`, then read the trace):

* body = 24 heads over ~42 us, **1.75 us/head**; DVE 88 % busy, ACT ~52 %.
* **DVE being 88 % busy does not mean DVE-throughput-bound.** Probe: blanking the 21 fp32
  `S` clone ops (3.9 us of DVE work, off the critical chain) moved the median only
  **-0.65 us**. The body is a *latency cycle* that passes through DVE several times.
* The cycle, per 2 heads (~3.4 us), is:
  `ACT f29 -> DVE N(h) -> DVE N(h+1) -> DVE U(h+2) -> PE o(h+2) -> ACT f30 -> RA -> f30 -> RA -> f29`
  i.e. **ACT's [f30,RA,f30,RA,f29] block (1.4-1.5 us) plus DVE's in-order queue
  [N,N,U,U] plus one PE matmul plus ~0.4 us of cross-engine semaphore latency.**
  ACT idles ~1.8 us per 2 heads waiting for DVE, DVE ~1.4 us waiting for ACT.
* Cross-engine hop cost is ~120-250 ns each way, and **each DVE instruction costs ~70-90 ns
  of queue time on top of its duration** (adjacent DVE ops do overlap by ~80 ns, so it is
  dispatch, not serialisation).

After this iteration's edits the body is ~1.3 us/head and DVE is ~92 % busy; re-probing the
clones now gives **-0.86 us**, so the conversion of DVE work into wall time is still only
~0.15 us of wall per us of DVE work. **Do not spend another iteration deleting DVE work.**

---

## Edits that worked

### BD — delete the DVE `EVENT_SEMAPHORE` barrier pairs (-1.6 us, the main win)

Every per-head RMSNorm output op (`N`) is followed by two update-free `EVENT_SEMAPHORE`s:
one waiting `s6 >= rank(the op just before it)` (a *self-completion* barrier) and one with no
wait at all (a pure NOP). Neither updates a semaphore, so **deleting them renumbers nothing**.
46 DVE ES exist; 36 match the pattern.

| dropped (iter-6 DVE slot numbering) | median | determinism |
|---|---|---|
| iter-6 baseline | 62.547 | 4 runs identical |
| all 36 | 60.688 | row 7 wrong in run 0 |
| 18 NOPs only | 61.827 | 4 runs identical |
| 35 (keep `DVE@102`) | 60.811 | **racy** (`--runs 4`) |
| NOPs + `{126,136,139,150,153,162,165,176,201}` | 60.50 | passed `--runs 4` **and** `--runs 8`, then **failed a second `--runs 8`** |
| NOPs + `{70,78,93,95}` | n/m | 4 runs identical |
| NOPs + both of the above sets | ~60.4 | **racy** (`--runs 4`) !! |
| NOPs + `{105,109,112,123}` | n/m | **racy** |

* The 18 **pure NOPs** (`W None, U None` — they cannot be enforcing anything) are worth
  **-0.72 us** and have been stable in every test.
* `DVE@102` is load-bearing *deterministically*: dropping it alone makes row 7 come out
  scaled by 1.285 (a partial sum of squares) every time. Keep it.
* The remaining 9 self-barriers are worth another ~0.6 us but the two safe-looking groups are
  not composable (see the determinism warning above). `{105,109,112,123}` is racy as a group.
* `applyBD.py <sg> <all|noponly|selfonly|only<a-b-c>|keep<a-b>>` does this; the slot list it
  prints also shows which `N` op / output row each barrier follows.

### BG — move the fp32 `S` clone behind the *next* head's bf16 `S_bf` write (-0.26 us)

Per head pair the DVE queue is `Ubf(h) Uf32(h) Ubf(h+1) Uf32(h+1)`. Only the bf16 ops are on
the cycle (PE's `o` matmul waits for `S_bf`); the fp32 clone only feeds the state_out DMA.
Reordering to `Ubf(h) Ubf(h+1) Uf32(h) Uf32(h+1)` lets head h+1's `o` matmul start ~285 ns
earlier. **The clones update no semaphore and the bf16 ops keep their relative order, so no
rank changes at all** — the cheapest kind of reorder there is. 6 quads match.
Pushing the clones further back (`applyBG2.py`, greedy, 2+ passes) is **not** better
(60.57 at best) and 2 passes breaks row 9.

### BF — delete the 14 update-free ACT `EVENT_SEMAPHORE`s (-0.17 us, marginal)

Same trick on ACT. Within spread on its own; kept because it is free and deterministic.

### AE site 4 — head 20's `f128` converts after all (-0.2 us)

iter-6 called head 20 "unfenceable". `applyAE.py <sg> 1 0,1,2,3` converts **only** that site
and it is correct and deterministic. The other four (heads 8, 12, 16, 17) still break.

---

## Refuted this iteration (do not redo)

1. **Taking the ACT hop out of the `o` matmul gate (iter-6's edit AB, redone properly as
   `applyBB.py`): correct but worth exactly 0 us.** Retargeting the PE `o`-matmul gate from
   `s4 >= rank(ACT ES)` to `s6 >= rank(DVE U)` for 19 of 21 heads is bit-identical when the
   previous generation's PSUM readers are fenced (`applyBB` computes and inserts the 7 missing
   `s4` fences itself) — but the median does not move (62.540 vs 62.547). The ACT ES was never
   delaying anything: ACT is idle when the U op finishes, and a semaphore hop costs the same
   ~150 ns whichever engine signals. **Sites 0 and 4 (S heads 3 and 7) are wrong even with the
   fences** — head 7's row comes out scaled by 1.285, i.e. its `READ_ACCUMULATOR` read ~61 %
   of the sum of squares. Inserting an ACT `EVENT_SEMAPHORE` between that head's `f30` and its
   `RA` (`applyBE.py`) does not fix it and perturbs another row.
2. **Software-pipelining the body by swapping each run of `N` ops with the following run of
   `U` ops (`applyBC.py <sg> all`): correct but +3.2 us (62.5 -> 65.7).** The rotation does
   break the ACT/DVE ping-pong, but the `N` op is also what *frees* the head's `o` PSUM bank,
   and the slack between the last read of a bank and its next write is only ~1.65 us — less
   than the ~1.7 us the rotation delays `N` by. PE then stalls on the bank fences.
   Without extra fences it faults with `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`.
   To make this work you would first have to lengthen the bank reuse distance (idea below).
3. **Merging adjacent per-head `f29` (rsqrt) pairs (iter-5's `applyU.py`): +0.12 us.** Only
   one pair is still mergeable, and merging makes head h's `rstd` available later (it has to
   wait for head h+1's `RA`), which delays `N(h)` on the DVE queue by as much as the merge
   saves on ACT. **The whole "re-slot the rstd ring into the 9 free scratch words so more
   f29 pairs become contiguous" plan is therefore dead** — I was about to build it and the
   one-pair measurement killed it. (A merged `f29` must be *contiguous*; a strided ACTIVATE
   only processes element 0 — iter-5.)
4. **The remaining four `f128` sites (heads 8, 12, 16, 17) are a deterministic arithmetic/
   ordering bug, not a race.** The victim rows (6, 10, 14) come out with *bit-identical*
   errors (3.2833 / 2.9272 / 3.6259) under three different fence configurations (0, 20 and 60
   inserted PE PSUM fences) — iter-6's "race" framing is wrong. Converting one site at a time:
   head 8 -> rows 6,9,13 wrong; head 12 -> 10,13; head 16 -> 9,14; head 17 -> 14,16;
   head 20 -> clean. I checked and ruled out: AE picking the wrong `U` ops (the bank-derived
   pairing agrees with AE's `src0 == S_h` pairing on all five sites) and the `v` row being
   shared by two delta ops (every `v_h` has exactly one reader). Still worth ~0.75 us.
5. **PSUM-bank fencing is cheap but buys nothing on its own**: making *every* implicit WAR
   edge on every PSUM bank explicit (20 PE `EVENT_SEMAPHORE`s, `applyBC.py <sg> -1`) costs
   only +0.19 us, and it does not fix any of the failures above.
6. **Deleting the per-head ACT ES placeholder with s4 renumbering (`applyBH.py`): -0.32 us but
   RACY.** It drops 11-13 ACT slots (after BB's retarget there is no waiter left on their s4
   rank) and decrements every higher s4 wait. `verify --runs 4` fails. Possibly salvageable by
   finding which of the 11 is unsafe, but the payoff is small.

## Tools added (`iter-7-scripts/`)

* `rows.py` — one device run, prints the wrong output rows **and the dev/gold ratio**. A
  uniform ratio == wrong `rstd` (the f30/RA sum); `dev == 0` == overflowed sum; a non-uniform
  error == wrong `o` or wrong `S`. This is the fastest classifier available.
* `probe_dve.py <sg> clone|vbeta|N` — blank a whole class of DVE ops (timing only).
* `applyBC.py` — DVE permutation with exact s6-wait remapping **plus** a general
  "fence every PSUM bank against the previous generation's readers" pass
  (`PROF=<dir>` picks the parquet, `FORCE=1` ignores the gate's own wait). The remap rule is
  `new_value = new_rank(the instruction that had the old rank)`; a conservative
  `max{new_rank(Y) : old_rank(Y) <= v}` would keep every old edge but defeats the purpose.
* `applyBB.py` / `audit_ba.py` — the `o`-matmul gate retarget and the per-bank WAR audit.
* `try.sh` / `try2.sh` / `try3.sh` — unpack-from-a-fixed-base, apply a list of edits, check
  `cycle.py`, repack. Always rebuild from a saved base NEFF; never edit in place twice.

## Ranked leads for iter-8

1. **Crack the four `f128` heads (8, 12, 16, 17), worth ~0.75 us.** It is *deterministic*, so
   it can be debugged statically — stop inserting fences. Next step: dump the intermediate
   values. `output_1` (fp32 `S`) is perfect in every failed variant, so `S` and the outer
   product are right; the damage is in `o`/`rstd` of a *different* head. Diff `w` (155648+256n
   ring), the delta scratch (four heads 8/12/16/20 write the **same** scratch address — that
   shared buffer is the prime suspect: converting head 8 makes its delta op write that scratch
   ~500 ns earlier, and heads 12/16/20 read the same address), and `S_bf`.
   `applyAE.py <sg> 1 <skip>` converts one site at a time.
2. **The head is 16 us (26 % of the kernel) and ACT-bound at 83 %**, of which
   **5 `ACT_TABLE_LOAD` x 1.283 us = 6.4 us** are on the ACT critical chain
   (`TL(sel1) -> HARDSWISH -> TL(silu) -> silu 2.76 -> TL(rs) -> f29 x2 -> TL(sp) ->
   SOFTPLUS x2 -> TL(sig) -> 4 CASTs 0.83 -> GELU -> TL(rs again)`). The input DMAs are done
   by 16.5 us, so after that ACT is the only limiter. Removing **one** table load is ~0.5-1.3 us.
   The 6th load (the body's rsqrt table, at 26.3 us) is a *reload* of the 3rd; collapsing it
   needs the two head `f29` ops (ACT pc=6, pc=7, 222+182 ns) moved after the GELU, which
   iter-6 says needs the chain-A DVE shuffles moved too. This is the biggest untouched block
   in the kernel.
3. **Lengthen the `o` PSUM bank reuse distance, then redo the `N`/`U` rotation (lead 2 of the
   refuted list).** Per head PE uses three banks (delta out, outer out, `o` out) and there are
   8, so a bank comes round every ~2.7 heads. The delta bank is dead as soon as that head's
   delta op has read it — *before* the same head's outer product runs — so **the delta and
   outer products of one head could share a bank**, taking it to 2 banks/head and ~4 heads of
   reuse distance. That doubles the slack the rotation needs (and the ordering is automatic:
   the outer product consumes `w`, which the delta op produces). Edit = the outer matmul's
   `dst` and the matching `U` op's `src1`. If it works, the rotation is worth up to ~0.8 us
   per 2 heads.
4. **The 9 remaining DVE self-completion barriers are worth ~0.5 us but every variant that
   drops any of them is racy.** Do not re-try them blind. The failure is always a single
   output row, uniformly scaled, i.e. a wrong `rstd` — the `f30`/`RA` sum-of-squares pair.
   Dropping these barriers lets DVE (and hence ACT, which waits on DVE) run ahead, so the
   suspect is a WAR that is only held by timing today. Two candidate mechanisms, both testable:
   (a) **the rstd ring** — 3 slots for 24 heads, consecutive users only 2-3 heads apart;
   `applyAH.py <sg> <head>:<slot>` re-slots a head into the free words `16396, 16448..16476`
   (9 of them; iter-6 verified the edit is bit-identical, and heads with a merged 3-element
   `f29`, i.e. 6/7, 10/11, 14/15, cannot move). Give heads 8, 12, 16, 18, 20 private slots so
   the minimum distance on slot `16384` goes from 2 to 4, then retry the barrier drops.
   (b) **`ACT f30` reading its PSUM bank before the matmul's writes have drained** — the
   1.285x partial-sum signature on head 7 fits this, and head 7's `o` bank (`0x2002000`) has
   no WAR at all, which rules out a bank conflict for that one. If (b) is real the barriers
   cannot be removed at all and this lead is closed.

## Map / facts (added this iteration)

* `o` PSUM bank lifetime = `o matmul -> ACT f30 -> (RA -> f29) -> DVE N`, about 1.1 us, and
  the gap from the last read to the next write of that bank is only ~1.65 us.
* Output row `h` lives at `25197024 + 256h` (`>= 96*0x40000`); the ops with dst
  25182936/25183448/25183960 in the same partition range are **f30/delta scratch, not output**
  — iter-6's `dst >= 96*0x40000` test for "N op" catches them and that mis-classification
  creates a deadlock cycle if you reorder on it.
* `ACTIVATE f29` = `rsqrt(src * 0.0078125 + 1e-6)`, 1 element; `f30` = square with
  `accumulator_cmd=3`, `dst` is a throwaway; `RA` just moves the accumulator to the rstd slot.
  There is no "accumulator as source" field, so `RA` cannot be folded into `f29`.
* rstd slots by head: `16384` for 0,3,6,8,10,12,14,16,18,20; `16392` for 1,4,7,11,15,19;
  `24024` for 2,5,9,13,17,21 (+`210944` once). Free fp32 scratch: `16396`, `16448..16476`.
* Per-engine busy at ~60.4 us: DVE 46.4 (77 %), ACT 39.4 (65 %), DMA 20.6, PE 18.9, Pool 15.6.
  Pool runs **no compute instruction at all** in this NEFF (only `PSEUDO_DMA_DIRECT2D`), so
  "move work to Pool" means inventing an opcode — I did not try it.

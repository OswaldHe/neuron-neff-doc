# iter-4 — selective_scan b1 c2560 s4096 n16 bf16

**Result: 6711.7 -> 6603.87 us (-1.61% on iter-3, -16.62% on the iter-0 baseline, 1.1993x).**
Numerics **bit-identical to the baseline** (output_0 max_abs_err 1.250e-01 = 83% of limit,
cosine 0.999944; output_1 1.812e-04). `verify.py --runs 3`: all three runs bit-identical (no
race). `semaphores.py` balanced, `launch_rules.py` clean. 5 repeats, median 6603.87,
min 6603.0, max 6604.3, spread 0.02%.

Four edits shipped, each measured on the device:

| # | edit | us | DVE busy | Pool busy |
|---|---|---|---|---|
| — | iter-3 shipped | 6711.7 | 6548 | 5540 |
| 1 | 154 DVE `MEMSET`s (y_accum zero-init) deleted; the group's first Pool ADD becomes a `COPY` | 6693.0 | 6545 | 5600 |
| 2 | 76 stalling DVE `y+=Cs` ADDs hopped one compute slot later, with their S[3] hand-off wait | 6663.3 | 6538 | 5600 |
| 3 | 68 more of the same, unblocked by re-addressing the `Cs` buffer that the next DVE slot recycled | 6613.4 | 6525 | 5605 |
| 4 | 112 lag-0 `S[6]` waits moved off the gating ES to a fresh ES just before their real consumer | **6603.9** | 6521 | 5611 |

The common shape of all four: **DVE is 98.8% busy and every remaining microsecond is a wait,
so the win is always "let DVE do the work it already has before it blocks", never "do less
work"**. Total DVE gap time went 147.0 -> 82.0 us; DVE busy fell 6548 -> 6521 us.

---

## THE BUG THAT COST ME AN HOUR — read this first

### `EVENT_SEMAPHORE` slots carry a SECOND, "extended" wait that `semaphores.py` does not see

`v2.NEURON_ISA_TPB_CTRL_ES_STRUCT` (opcode 0xa0) has, at bytes 32-43, a whole second events
block: `events_extended_wait_mode/_wait_idx` and `events_extended_sem_wait_value` (plus
`_update_mode/_update_idx/_sem_update_value`). In this NEFF:

| stream | ES slots with an **extended** wait | on which semaphore |
|---|---|---|
| DVE | 3679 | **S[6]** (max value 8172) |
| DVE | 308 | S[3] |
| Activation | 5 | S[6] (max 7550) |
| SP / Pool | 5 / 1 | S[5] / S[15] |

`tools/semaphores.py` reads only the primary block (`account()` skips `PSEUDO_EXTENSION` and
never looks at `events_extended_*`), so **it reported "balanced" for a NEFF that deadlocked**.
My first attempt at edit 1 renumbered only the primary S[6] waits; the 3654 stale extended
waits then demanded values above the reduced total (8022) and the model hung
(`Failed to execute model: 10001, error: 5`, then `Model stop timeout (30000 ms) for TPBs`,
~70 s per run). **Any renumbering of a semaphore must rewrite primary *and* extended waits in
all five streams.** The gap-attribution work below also shows the extended wait is very often
the *binding* one, so an analysis that ignores it attributes stalls to the wrong semaphore
(my first pass blamed Activation for 73 us that was really Pool).

A 40-line in-order simulator over the five streams (wait -> execute -> increment, DMA
semaphores 8-39 assumed satisfied) reproduces the deadlock **without a device run** once the
extended waits are included — it is the check to run before every pack. It is reproduced at
the end of these notes.

### An ALU op the compiled NEFF never uses hangs the engine

My first formulation of edit 1 turned the group's first `y += Cs` Pool ADD into
`y = max(Cs, Cs)` (bit-exact, 2 field writes). It **hung the device the same way**. The only
ALU ops in this NEFF are `MULT` (6) and `ADD` (4) on DVE and `ADD` (4) on Pool; `MAX` (8) is
never used. Pool/GpSimd and DVE are driven by the microcode tables shipped in the payload
(`default_opcode_table.bin`, `default_datapath_table.bin`, `default_control_{fast,slow}_table.bin`,
referenced from `DVE0.json`), and the compiler only populates the entries the kernel needs.
**Treat the set of (opcode, alu_op, dtype) triples present in the NEFF as the whole ISA you
may use.** A Pool `COPY` with 512 fp32 elements worked first try, because Pool already has
1573 fp32 `COPY`s (all 1-2 element) — same opcode, same dtype, new length is fine.

---

## WIN 1 — delete the 154 y_accum `MEMSET`s. -18.7 us

160 of the 200 DVE `MEMSET`s are `dst_element_count=512, dtype=FP32, set_value=0,
num_active_channels=128`: the zero-init of one group's `y_accum`. Profiled cost of removing
one (start of the next compute minus end of the previous, plus the 77 ns of overlap that a
removal restores): **median 534 ns, 88.9 us for all 160** (the other 40 memsets are the 20
bf16 `ones` vectors at 16384 and the 20 16-element carry zeros, 13 us).

They are removable because the first instruction that writes `y_accum` after the memset is
always (160/160) a `TENSOR_TENSOR ADD` with `src0 == dst == Y` and identical src0/src1 access
patterns, and **no instruction on any engine reads or writes Y in between** (checked across
all five streams by profile timestamp). So turn that first ADD into a pure overwrite:

```
Pool slot of the first ADD  ->  rebuilt as a COPY (opcode 0x46, S4D4_TR struct) from a
    template Pool COPY: src addr word = ADD's src1 (Cs), dst addr word = ADD's dst (Y),
    src/dst num_elem [512,1,1,1], step [1,1,1,1], in/out dtype FP32, 128 channels,
    events fields copied from the ADD (so S[3] numbering is untouched: 1727 COPY + 2260 ADD
    = the same 3987 increments)
delete the MEMSET slot from DVE, renumber S[6]: newmap[k] = #surviving updaters <= k
    (the deleted ordinal maps DOWN to the previous survivor — correct, because the only
    waiters on it are the 155 first-ADDs, which no longer need it, and 148 DVE ES whose
    guarantee "everything DVE did before the memset" is unchanged)
```
`0 + x == x` exactly in fp32, so this is bit-exact. 154 of the 160 qualified (3 have their
first writer on DVE — a DVE `COPY` would need a microcode entry DVE does not have; 3 memsets
carry a wait of their own, which protects a WAR I did not want to re-derive).

**Only 18.7 us of the predicted 89 arrived, and DVE busy fell by just 3 us.** The gap budget
is conserved: total DVE gap time was 147.0 us before and 147.1 us after — the stalls simply
moved to the next bottleneck (DVE now reaches the Pool hand-offs earlier and waits there
instead). Pool got 60 us busier (the 512-element fp32 Pool COPY is ~2.55 us, the ADD it
replaced 2.17 us) but Pool is 1.1 ms behind DVE, and the binding-supplier analysis below
confirms the new COPYs are *not* on the critical path.

## The measurement that made wins 2 and 3 possible: attribute each gap to its *binding* wait

Earlier iterations attributed a DVE gap to "the wait on the instruction that starts after it".
That is wrong when several slots sit in the gap, each with up to two waits. Correct recipe:

1. Build `supply[(sem, ordinal)] = end_ts` of the instruction that performs that increment
   (walk the owning engine's stream in order: S[3]=Pool, S[4]=ACT, S[5]=PE, S[6]=DVE).
2. For every pair of consecutive DVE *compute* rows with `start_ts > prev end_ts`, look at
   every slot in `(prev_pc, pc]` and every wait on it (**primary and extended**), and take the
   one whose `supply` time is **latest**. That is the binding constraint.

On the iter-3 NEFF (147.0 us of gap in 572 gaps):

| binding wait | gaps | us | median lateness (supply - prev_end) |
|---|---|---|---|
| **S[3] extended (Pool)** | 159 | **85.1** | +403 ns |
| S[3] primary (Pool) | 26 | 7.6 | +208 ns |
| S[6] (DVE's own, lag-0 RAW) | 377 | 43.4 | 0 (serialisation latency 118-170 ns) |
| S[5] (PE) | 4 | 5.2 | — |
| S[4] (ACT) | 1 | 1.2 | — |

So **ACT and PE are not the problem** (the naive attribution said ACT cost 73 us; it costs
1.2 us) and two thirds of the gap is DVE waiting for a Pool `y+=Cs` ADD. For those 167 sites:
Pool is *not* backed up — it starts the ADD a median of **101 ns after its input is ready**
(median Pool idle before it: 38 ns) and the ADD itself takes **1626 ns**. DVE then resumes
116 ns later. The window of DVE work available to hide that 1.84 us round trip is only
~1.3 us, so the stall is 521 ns on average. Moving those ADDs back to DVE costs 590 ns of DVE
work to save 521 ns of stall — **still a net loss, iter-3's threshold rule is confirmed.**

## WIN 2 — hop the stalling DVE ADD one compute slot later. -29.7 us for 76 sites

The 156 worst sites all have the same shape:

```
DVE[j]   EVENT_SEMAPHORE   primary S[6] >= v6 , extended S[3] >= k   <- the stall
DVE[j+1] TENSOR_TENSOR ADD y += Cs   (src0 == dst == Y)              <- needs Pool's ADD #k
DVE[j+2] ...                                                         <- 0.7-1.3 us of work
```
Nothing but the ADD needs `S[3] >= k`, so give DVE the next instruction's worth of work
*before* it blocks:

```
clear the extended wait on DVE[j]   (keep its primary S[6] wait where it is!)
move slot j+1 to just after the next compute slot p, and insert immediately before it a
    bare ES with primary wait (mode 5, idx 3, value k), no update, no extended wait
renumber S[6] for the permutation:  newthr(v) = max(new_ordinal[1..v])   (running max)
```
`newthr` as a running max is the general rule for *any* DVE reordering: it means "wait for
every instruction you used to wait for", so it is correct by construction for moves in both
directions.

**Two wrong-output attempts taught the two checks that matter** (both failed with exactly
`max_abs_err 1.430e+01`, cosine 0.9983, 98.45% of elements still in tolerance — i.e. a
handful of groups corrupted, which is what a missed hand-off looks like):

1. **Do not move the ES along with the ADD.** A wait on an ES gates *every* following slot on
   that engine, not just the next one. Moving it past a scan dropped the scan's carry
   dependency (`imm0_imm_ptr` -> the carry slot at 16640+4n, which Pool writes). Note
   `imm0_imm_ptr` is a real operand address and must be in any read-set you compute: the
   `*_start_addr_addr_immediate` fields alone do not cover the scan's carry.
2. **Check WAR, not only RAW.** For every slot the ADD hops over, reject if any address it
   *writes* is **read** by a Pool instruction with S[3] ordinal in `(prev_s3_guarantee, k]`
   — the hopped slot was relying on the very wait being cleared to know Pool had finished
   reading that buffer (Pool reads 487 distinct scratch addresses as `y_accum` src0).
   Checking only Pool *writes* let 7 bad sites through.

Accepted: 76 of 156. Rejects: 67 "the next slot waits on the ADD's own ordinal", 7 pool-dep,
4 scan, 1 alias, 1 end-of-stream. Measured **390 ns saved per site** (of the 521 ns stall).

## WIN 3 — re-address one `Cs` buffer per site to unblock the 68 remaining sites. -49.9 us

All 68 of the "next slot waits on the ADD's own ordinal" rejects are one pattern:

```
DVE[j]   ES    primary S[6] >= v6, extended S[3] >= k
DVE[j+1] ADD   y += Cs            (reads Y and Cs=0x8140, writes Y)
DVE[j+2] ES    primary S[6] == ordinal(j+1)        <- WAR guard, nothing else
DVE[j+3] TT    dst = Cs (0x8140)                   <- recycles Cs one slot later
```
The hop is blocked by a WAR on `Cs`, which the free SBUF makes unnecessary. **The SBUF map
has a 52416-byte hole at 62272..114688** (no instruction field and no DMA descriptor
references it; the input DMA staging starts at 114688; `def.json`'s only
`runtime_statebuffer_reservation` is 8 bytes at 229368), i.e. 25 spare 2048-byte buffers.
For each site, with `F` taken round-robin from those 25:

```
DVE[j+3].dst            := F
the single consumer of Cs in (j+3, next writer of Cs)  := F   (all 68 are on DVE)
DVE[j].extended wait    := NONE
DVE[j+2] (the WAR guard) := wait NONE        (it exists only for the broken WAR)
move DVE[j+1] to after DVE[j+3], preceded by a fresh bare ES waiting S[3] >= k
renumber S[6] with the same running-max rule
```
Safety: the producer and the only consumer of `F` are both on DVE and thousands of slots
before the next site that reuses the same `F`, so DVE's in-order issue (with its 1-deep
overlap) is the whole guarantee — no new semaphore needed, and `verify.py --runs 3` is
bit-identical. All 68 passed the barrier check "nothing between the cleared guard and the next
slot that re-establishes `S[6] >= ordinal(ADD)` touches Y or Cs".

**-49.9 us for 68 sites = 734 ns each**, more than the 450 ns of gap they showed, because
clearing the lag-0 WAR guard also removes its ~150 ns of serialisation.

## WIN 4 — move a lag-0 `S[6]` wait off the ES onto the slot that actually needs it. -9.7 us

Same idea as win 2 but for DVE's *own* serialisation. 222 of the gaps are a wait on `S[6]`
with **lag 0** (`#updaters before the waiting slot - wait value == 0`, i.e. "my immediate
predecessor must have completed"), 29.6 us at 117-139 ns each.

* **Plain relaxation lag-0 -> lag-1 is dead: all 222 alias** (checked with the read/write sets
  of every un-gated slot up to the next slot that re-establishes `S[6] >= v`, including PSUM
  addresses and the scan's `imm0_imm_ptr`). These are real RAW edges, so this closes
  review/iter-3 candidate "relax the 393 lag-0 sites" **without a device run**.
* **But the waiter is usually an ES, and the instruction that actually consumes the
  predecessor's output is 2 slots further on.** In 112 of the 222 the slot between them does
  not alias at all:
  ```
  DVE[84] ES   primary S[3]>=23 , extended S[6]>=49   <- blocks here
  DVE[85] TT   wr=18752, own wait S[6]>=48            <- does NOT touch DVE[83]'s output
  DVE[86] TT   rd=26944 = DVE[83]'s output            <- the real consumer
  ```
  Edit: clear the `S[6]` wait on the ES, and **insert a fresh bare ES carrying
  `primary (mode 5, idx 6, value v)` immediately before the consumer**. An inserted ES
  increments nothing, so **no renumbering at all** — 112 one-slot insertions, nothing else.
* Sound because the hardware overlap depth is 1 (iter-3): the only instruction that can still
  be running when slot q starts is q's immediate predecessor, so losing "ordinals <= v" for
  the slots in between is the same as losing "the predecessor", which the alias check covers.
* **-9.7 us for 112 sites (86 ns each).** A second pass found 110 lag-0 sites left and **none**
  with a free slot in between: the aliasing consumer is immediately after the wait. Exhausted.

## Where the time is now (measured on the shipped NEFF)

| engine | busy | share |
|---|---|---|
| DVE | 6521 us | **98.8%** |
| Pool | 5611 us | 85.0% |
| PE | 2968 us | 44.9% |
| ACT | 1903 us | 28.8% |

DVE instruction duration: `TENSOR_TENSOR` 5586 x 671 = 3751 us, `TENSOR_TENSOR_SCAN`
2560 x 1298 = 3322 us, `MEMSET` 46 x ~0.2 = 10 us -> 7083 us of duration inside a ~6521 us
union (77 ns of pipeline overlap at every one of 8191 boundaries, uniformly — 593 us; the
hardware depth is 1 and iter-3 proved it is already saturated).

**Total DVE gap is down from 147.0 to 82.0 us** and every lever above is exhausted (the round-3
hop pass found 21 sites worth 18.2 us and could move **none** of them: 8 have a SCAN rather
than a TT as the stalling compute, 7 pool-dep, 4 scan-in-the-way, 1 alias, 1 eof; the round-4
wait-migration pass found 110 sites and could move none). What is left, in 356 gaps:

| binding wait | gaps | us | note |
|---|---|---|---|
| S[6] (DVE's own, lag-0) | 208 | **34.8** | genuine RAW on the immediate predecessor (all 110 remaining big ones alias); 118 ns each |
| S[3] primary | 124 | 24.2 | median lateness **-22 ns**: Pool is *not* late, this is the ~131 ns DVE needs to observe an already-satisfied wait |
| S[3] extended | 15 | 17.0 | the only real stalls left, mean 1.1 us, all blocked by a scan or a pool-dep in the hop window |
| misc (S[4], S[9], S[20]) | 3 | 5.4 | one-off, start/end of the run |

So **~59 us of the 82 is semaphore observation latency (118-131 ns per gated instruction),
not waiting**: 7562 of the 8191 DVE boundaries still get the full 77 ns of overlap, and a gap
appears only when the wait is satisfied within ~130 ns of the previous instruction ending.
That is the floor for the current instruction stream.

## REFUTED (built, ran, bit-exact, measured) — the Pool ADD's access-pattern shape

Pool's 512-element fp32 `TENSOR_TENSOR ADD` costs **1626-2164 ns (≈2.6-3.2 ns/elem) against
DVE's 671 ns (1.31 ns/elem)** for the identical operation, and it is the supplier of every
remaining real stall, so making it cheaper looked like the one >1% lever left. It is not a
pattern-shape effect: rewriting 200 of the 2260 Pool ADDs from `num_elem=[512,1,1]`,
`step=[1,1,1]` to `num_elem=[256,2,1]`, `step=[1,256,1]` (same elements, same order, bit-exact
and verified) changed their duration by **nothing**:

| | n | mean | median | min |
|---|---|---|---|---|
| rewritten, 2 rows of 256 | 200 | 1963 ns | 2163 ns | 1262 ns |
| untouched, 1 row of 512 | 2060 | 1969 ns | 2164 ns | 1256 ns |

and the wall **regressed 6603.9 -> 6665.5 us**, so the shape is not free even when the duration
is identical. Pool's rate is a flat ~2.6 ns/elem on top of ~310 ns of fixed overhead
(iter-3's COPY-merge measurement agrees); treat it as a hardware constant.
`num_active_channels` is the only knob left untested here, and it must stay 128.

## What to try next (ranked) — the scheduling budget is down to ~23 us, so look at duration

1. **Extend win 3's re-addressing to the last 15 S[3]-extended sites (17 us).** Each is blocked
   by a specific thing in the hop window: a SCAN whose carry read (`imm0_imm_ptr` -> 16640+4n)
   is what the cleared wait protects (fix: give the *carry slot* a second buffer from the free
   hole, or hop by two slots), or a pool-dep on a `y_accum` buffer (fix: the same fresh-buffer
   trick as win 3), or the stalling compute is itself a SCAN carrying the S[3] wait (a scan can
   be moved exactly like a TT — my script only refused on a type check, so this is free to
   add). The 25 free 2048 B buffers at 62272 are only ~3x oversubscribed so far.
2. **A fifth of the DVE instruction count is still fixed overhead** (`TENSOR_TENSOR` 671 ns =
   ~533 ns of elements + ~138 ns fixed; `SCAN` 1298 = ~1067 + ~230). iter-3 refuted merging
   adjacent DVE TTs because of the DVE<->PE PSUM lockstep, and that refutation stands. But
   note the arithmetic: 8146 DVE instructions x ~140 ns = **1.14 ms of pure per-instruction
   overhead** against a 6.60 ms wall. Any mechanism that reduces the instruction *count*
   without moving a PSUM-bank release (iter-3's section 2: same-bank pairs, which are not
   adjacent) is worth more than everything else on this list combined.
3. Closed again this iteration: **ACT is not a bottleneck** (1.2 us of binding stall, not the
   73 us a naive attribution suggests); **moving `y+=Cs` ADDs between Pool and DVE is at the
   optimum** (590 ns of DVE work against 521 ns of stall, measured in both directions);
   **lag-0 -> lag-1 relaxation is dead** (all 222 sites are genuine RAW).

## Tooling notes

* `work/payload/sg00/*.bin` have **no `PSEUDO_EXTENSION` (0xda) slots** — every slot is 16
  words. The "extended" waits live inside the `EVENT_SEMAPHORE` slot itself (bytes 32-43), not
  in a following slot. `isa.py` decodes and encodes them by name
  (`events_extended_wait_mode`, `events_extended_wait_idx`, `events_extended_sem_wait_value`).
* Deleting and inserting DVE slots is safe (confirmed again: 14037 -> 13883 -> 14027 slots);
  `neff.py pack` writes the shrunken member size correctly and the unpack round-trips.
  A 2-slot deletion with no renumbering needed (the 2 memsets that increment nothing) was the
  control that separated "deletion is broken" from "renumbering is broken" in one 10 s run.
* Device-run cost: ~10 s pass, ~10 s wrong-output, **~70 s hang** (`exit 1, error: 5` then
  `Model stop timeout`). `profile.py --repeats 5` ~75 s, `--repeats 1 --parquet` ~25 s.
* Keep a verified `.neff` after every accepted step (`/tmp/kernel_*.neff`); every experiment
  then starts from a known-good unpack instead of a half-edited `work/`.

### The deadlock pre-check (run before every pack)

```python
# streams: 5 lists of (name, wait_mode, wait_idx, wait_val, upd_mode, upd_idx,
#                      ext_wait_mode, ext_wait_idx, ext_wait_val, ext_upd_mode, ext_upd_idx)
pc={e:0 for e in ENG}; sem=defaultdict(int); BIG=set(range(8,40))   # DMA sems: assume supplied
while True:
    moved=False
    for e in ENG:
        while pc[e]<len(streams[e]):
            nm,wm,wi,v,um,ui,ew,ei,ev,eu,eui=streams[e][pc[e]]
            if wm and wi not in BIG and sem[wi]<v: break
            if ew and ei not in BIG and sem[ei]<ev: break      # <-- the extended wait
            if um: sem[ui]+=1
            if eu: sem[eui]+=1
            pc[e]+=1; moved=True
    if not moved: break
# every pc[e] must equal len(streams[e]); otherwise print the blocked instruction
```

# iter-2 — moved **all 160** DRAM staging round-trips onto a new queued-DMA queue, then
# neutralised all 224 now-dead ACT DGE saves

## Result: **1205.6 us** — shipped `sha256 68982cd8a8bb400d…`

Three independent profiles of the shipped NEFF: **1207.350 / 1204.209 / 1205.564** (spreads
0.18–0.38%), so call it **1205.7 ± 2**.

| NEFF | median us | note |
|---|---|---|
| iter-1 (= my starting point), re-measured | **1315.9** | band 1314.1–1318.8 (its own notes said 1313.3; drift ≈ 3 us) |
| step 1: 31 Pool-group reloads fused | 1304.5 | −11.4 |
| step 2: + drop the dead save barrier before each | 1300.7 | −3.8 |
| step 3: + neutralise the 94 dead ACT DGE saves | **1273.4** | −27.3 |
| step 4: + fuse the other two staging groups (all 160 reloads, 224 dead saves) | **1205.7** | −67.7 |

**−110 us vs iter-1 (−8.4%), −10.8% vs the iter-0 baseline (1351.4).** Bands never overlap.
`verify.py` PASS at every step with `max_abs_err` **identical to the baseline to every printed
digit** (output_0 1.097e-04 = 10% of limit, output_1 9.081e-04 = 83%). `launch_rules.py` PASS.
Engine busy: ACT 627→521, SP 131→52, DMA 420→342, Pool 255→234; DVE/PE unchanged.
**DVE (534 us, 44%) is now the busiest engine.**

---

## 1. The lever, in one paragraph

All 160 `tmp-buf` DRAM variables are *staging stops for a partition/offset redistribution*
(iter-1 §2). iter-1 fused the 32 whose reader was already a **queued** transfer and concluded the
other 128 were closed, because a **DGE** access pattern allows only **one** partition-crossing
dimension on the SBUF side and every composite needs two or three (iter-1 §4.1, device-refuted
twice). That conclusion was right about DGE and wrong about the kernel: the **queued** path has no
such limit, and you can *add a whole new queue* with free semaphores, so no existing wait value
anywhere has to be renumbered. Every one of the 160 reloads is now a queued SBUF→SBUF transfer on
a new queue `qSPSpillReload1`, the old DGE reload slot is an `EVENT_SEMAPHORE` that repays its
+16, and all 224 ACT DGE saves are dead and have been replaced by 40 ns `EVENT_SEMAPHORE`s.

## 2. Recipe — reuse this verbatim; it is all device-verified here

### 2.1 A new DMA queue costs nothing and unlocks everything
Add to `def.json`:
```json
"qSPSpillReload1": {"type":"data","num_queues":16,"owner":"sp","fabric_path":"main",
                    "semaphore_set":[72,…,87]}
```
* **Transfer `id` is per queue** (verified on the corpus: 00-Attention-B has `qSPIO0` 0–347 *and*
  `qSPSpillReload0` 0–7115). So new ids start at 0 and **no existing `S[24..39]` wait moves**.
* Free semaphores: S[0..2] are the runtime's, this NEFF uses 3–6, 8–23, 24–39, 40–55, 56–71, so
  **72–87 were free**; a NeuronCore has 256 and the runtime epilogue zeroes 3–255 between runs
  (ch.10 §10.1), so a fresh range needs no clearing instruction.
* 5 queues × 16 rings = 80 rings is what the corpus NEFFs already use, so 64→80 is safe. 96 rings
  (two new queues) is **still untested** — I did not need it.
* Transfer `id` → ring `id % num_queues` → `semaphore_set[id % 16]`, `+16` per completion.
  **Ring order is trigger order**, so assign ids in SP-trigger program order and the k-th transfer
  on a ring is waited as `>= 16k`. I put 192 transfers on 16 rings (12 per ring, waits 16…192).

### 2.2 `op:"cast"` in a queued descriptor works and is **bit-exact** vs the DGE cast
This NEFF has no `cast` descriptor at all (576 bf16 transposes, 192 fp32 transposes, 64 plain
`copy`), so it was a guess. It works: `from_dtype:"float32"` → `to_dtype:"bfloat16"`, **`sizes[0]`
is a byte count on each side separately** (512 in, 256 out). Both paths are the same DMA
hardware, so rounding is identical — `max_abs_err` did not move in the last digit. 3-entry
`from_sizes`/`from_steps` are fine (the ch.9 example uses 3; this NEFF's own use 4).

### 2.3 `EVENT_SEMAPHORE` has a **second (extended) events block** — this is the key enabler
`CTRL_ES` carries `events_extended_{wait_mode,wait_idx,update_mode,update_idx}` plus **separate**
`events_extended_sem_wait_value` and `events_extended_sem_update_value`. The compiler already
uses the extended *wait* (e.g. Pool slot 38 waits `S[44]>=16` **and** `S[43]>=16`). That defeats
the "there is only one value field" limit of ch.6.4: one slot can do
**wait `S[a] >= v` (extended) + `S[b] += 16` (primary, value 16)**. Without this every relay
would have needed two slots.

### 2.4 The three composites (all uniform; asserted over all 160 vars)
SBUF byte address = `partition*262144 + offset`; all steps/sizes in **bytes**.

| group | n vars | save(s) | reload(s) |
|---|---|---|---|
| A-Pool | 32 | 2 ACT, `src step[512,262144] num[4,16] elem512`, `dst DRAM step[16384,1024] num[4,16]`, dst_off 0 / 512 | 1 Pool, `src DRAM step[1024,1] num[64,1] elem1024`, `dst SB step[262144,1] num[64,1] elem512`, fp32→**bf16** |
| A-SP | 32 | same | 1 SP, same but `dst elem1024`, fp32→fp32 |
| B | 64 | 1 ACT, `src SB step[262144,1] num[64,1] elem1024`, `dst DRAM step[1024,1] num[64,1]` | 2 SP, `src DRAM step[16384,1024] num[4,16] elem512` at off 0 / 512, `dst SB step[512,262144] num[4,16] elem512` |

Shape **A** → one transfer, two descriptors (`k`=0,1, `dk`=0/512, `run`=E/2 where E=reload
`dst_elem_size`):
```
from SB off = save_k.src_off          sizes [512,  4,16]  steps [1, 512,     262144]
to   SB off = reload.dst_off + dk*run/512   sizes [run, 4,16]  steps [1, 4194304, 262144]
```
Shape **B** → one transfer per reload, one descriptor (`dk` = reload's DRAM src_off):
```
from SB off = save.src_off + dk       sizes [512,4,16]  steps [1, 4194304, 262144]
to   SB off = reload.dst_off          sizes [512,4,16]  steps [1, 512,     262144]
```
`4194304 = 16*262144`: the DRAM block index `i = 16a+b` becomes **two** partition-crossing
dimensions — exactly what a DGE cannot express and a queued descriptor can.

### 2.5 The trigger's wait: two provably-safe conditions
A trigger must be gated on the *producers of the SBUF source*, not on the save's completion
(iter-1: gating on the save's completion is worth **0**).
* **Shape A-Pool**: reuse the saves' own `S[6] >= v`. Verified it is complete: each save's 4
  512-byte chunks are written by DVE instructions with **`num_active_channels = 32`** (partitions
  96–127), i.e. the *same* instructions feed saveA (partitions 96–111) and saveB (112–127), and
  all of them have DVE ordinal ≤ v (31 of 32 vars check out; var 21's saveA has only a ring-reuse
  wait, so it goes in the other bucket).
* **Everything else**: wait on the **save's own semaphore at the save's ordinal**
  (`S[save.semaphore] >= 16*j`). Once the save is an `EVENT_SEMAPHORE` (§2.6) that increment fires
  40 ns after ACT reaches that slot, so the condition is *exactly* "ACT has executed up to and
  including the save" = the save's own precondition, by construction. Uniform, needs no producer
  analysis, and it is why the whole thing could be done in one pass.

### 2.6 Dead saves: `EVENT_SEMAPHORE` + `SEM_ADD_IMM_COMPLETE` is worth **540 ns of ACT each**
Measured per-slot issue cost in this NEFF: ACT `DMA_DIRECT2D` **580 ns**, ACT `EVENT_SEMAPHORE`
**40 ns**; SP DGE 603 ns vs SP ES 27 ns; Pool DGE 702 ns vs Pool ES 19 ns. So neutralising
224 dead ACT saves removes **~121 us of ACT serial time** and 160 SP DGE → **~92 us of SP**.
This is the single biggest contributor (step 3 alone: −27 us for 94 slots; step 4 added the rest).
**iter-1 §4.4/§5.2 said this lever was worth ~1 us — that was true only while 130 of the saves
were still live.** Fusing the reloads is what makes the saves dead, and the two edits together
are worth far more than either alone.
Form: primary `update_mode=21 (SEM_ADD_IMM_COMPLETE)`, `update_idx = own semaphore`,
`events_semaphore_value = 16`; **original wait moved verbatim into the extended wait**.

### 2.7 Counting-semaphore ordering — the one real hazard, and how it is closed
A semaphore only counts, so an early increment from a neutralised (instant) transfer can satisfy
a wait meant for a *still-in-flight live* transfer on the same semaphore. Three cases:
* **ACT S[40..55]**: safe because **all 224** become `EVENT_SEMAPHORE`s, so every increment is in
  ACT program order — same order as the DGE ring gave. (`assert len(deadslots)==n_act_dge`
  in the script guards this. If you leave even one live DGE on a semaphore, you must add an
  extended wait `S[s] >= 16*(j-1)` to every later ES on that semaphore — measured free, the
  previous same-semaphore DGE is ~190 ACT slots back.)
* **SP S[56..71]**: safe because **all 160** become relays, ordered by SP program order.
* **Pool S[8..23]**: only 32 of 320 become relays, so here I *did* need the guard — and it was
  free: the slot immediately before every Pool DGE reload is a pure-wait `EVENT_SEMAPHORE`
  (the "wait for both saves" barrier), so I repurposed it as `S[Psem] >= 16*(j-1)`.

### 2.8 Inserting slots
* **SP**: no SP instruction increments any semaphore, so inserting 192 `PSEUDO_DMATRIGGER`s
  shifts no count. **Placement rule that also proves acyclicity**: insert after the last SP slot
  whose profile `start_ts` is **before** the save's `start_ts`. Then anything the trigger's wait
  transitively needs was already produced before SP reached that point in the baseline, so no
  cross-engine cycle (this is what iter-1's hang was). Assert `trigger_pos <= relay_slot`.
* **ACT**: `EVENT_SEMAPHORE` on ACT does **not** increment S[4] (only `ACTIVATE`,
  `TENSOR_SCALAR`, `CAST` do) and does not change DGE ordinals, so ACT slots can be inserted
  freely too. I did not need to.
* **`count_union_block_id` and `count_union_count` are a union** — writing `count_union_count=0`
  after `count_union_block_id=n` silently zeroes the block id (every trigger then pointed at
  transfer 0). Cost me one debug round; don't set `count_union_count` at all.

## 3. Measurements worth keeping

* New queued transfer, 32 KB written / 64 KB read, 128 runs: `duration` **750 ns**,
  **trigger→completion 1461 ns** (vs the chain it replaced: ACT DGE issue 580 + save transfer
  ~1500 + Pool DGE issue 700 + reload transfer ~1700 ≈ 4.5 us).
* The Pool relay reaches its slot **before** the transfer completes (relay `start_ts` −700 ns
  median vs transfer `end_ts`) and finishes ~380 ns after it, i.e. **the queued transfer is the
  gate** — the chain is tight, nothing to reclaim by moving the trigger.
* **Path attribution is ~16% efficient on this kernel.** Steps 1+2 removed ~3 us × 31 = 93 us of
  dependency chain and bought 15 us. iter-1's backward walk called those 32 Pool reloads + 32 ACT
  saves ~100 us "on the path"; removing them was worth 15. Treat a walk's per-edge us as an
  *upper* bound — a binding edge usually hides a near-equal parallel path. Engine-busy reduction
  (steps 3+4, −121 us ACT / −92 us SP of issue) converted at a **much** better rate.
* `profile.py` spread on this kernel is 0.2–0.45%, so ±3 us; a 1-run `verify.py` is ~50 s,
  `--runs 3` ~2.5 min, `profile.py --repeats 5` ~75 s. I used 7 device cycles.

## 3b. WHERE THE TIME GOES NOW — build on this, it took 4 cheap measurements

### Idle, attributed to the semaphore each engine's next slot waits on
(gap = `start_ts` − previous slot's `end_ts`, summed; engine completion sems are S[3]=Pool,
S[4]=ACT, S[5]=PE, S[6]=DVE. Careful: the "v-th signalling instruction of engine e" map must
count only `update_mode in (19,3) and update_idx == s` — my 224 ACT relays use
`update_mode 21` on sems 40–55 and silently shifted the whole index by +224 on the first try.)

| engine | idle | biggest buckets |
|---|---|---|
| **PE** | 900 us | **qSPSpillReload0 516.5**, S[6] (DVE) 270.1, S[4] (ACT) 112.9 |
| DVE | 697 us | S[5] (PE) 448.3, S[4] 100.9, S[3] 47.0, SP relay 45.0 |
| ACT | 695 us | S[5] (PE) 401.8, S[6] 199.6, qSPSpillReload0 36.1 |

So the loop is **queued DMA → PE → {DVE, ACT} → PE**, and PE is the pivot: true busy 341 us,
idle 900 us, of which **517 us is waiting on the 669 `qSPSpillReload0` transfers**. That is the
one big number left.

### …and here is that 517 us split into its three parts — READ THIS BEFORE ATTACKING DMA
PE's idle is diffuse (top 9% of 1275 gaps carry 52%; top 50% carry 92%) and the biggest single
gaps are all **`LDWEIGHTS` waiting on a queued-DMA semaphore, 5–8 us each** (pc158 S[33]>=32:
7.92 us; pc2334 S[39]>=336: 7.33; pc302 S[26]>=64: 7.29; …). Taking the 148 PE gaps over 1 us
that are gated on S[24..39] (**469.9 us** of idle) and splitting each gap at the producing
transfer's `trigger` and `end_ts`:

| part of the gap | us | share | per gap |
|---|---|---|---|
| PE idle **before the transfer was even triggered** (waiting for the *producer*) | 206.7 | **44%** | 1.40 us |
| transfer latency, trigger → completion | 180.5 | 38% | 1.22 us |
| completion → PE start (semaphore propagation + dispatch) | 82.8 | 18% | 0.56 us |

So **44% of PE's big DMA stalls are not DMA at all** — PE has run ahead of the DVE/ACT producer.
The genuinely DMA-attributable part is ~263 us, and **~110 us of that is the 753 ns flat
queue-start cost × 148 hops**, which iter-1 §4.7 showed is independent of size and ring load.
Conclusion: the ceiling for any DMA-latency work on this path is ~150 us *if transfers became
free*, and most of it is per-hop constants. The 44% producer-wait share is a *scheduling* problem
(overlap batch iterations), which is not reachable by local slot edits.
(Producer of a wait `S[24+r] >= 16k` = `qSPSpillReload0` transfer id `(k-1)*16 + r`; ring order ==
id order within a ring, verified. Script: `scratch/pesplit.py`.)

### The 669 queued transfers: per-hop cost and what their waits mean
Mean `duration` 410 ns, `start_ts − trigger` **753 ns flat**, trigger→completion **1163 ns**.
Classifying all 260 trigger waits that name S[24..39] by SBUF-region overlap between the waiting
transfer and its producer:

| kind | n | meaning |
|---|---|---|
| previous transfer on the same ring, **no** RAW/WAR/WAW overlap | **218** | pure ring throttle |
| RAW (my source ⊂ producer's dest) | 31 | real chain — the only fusable ones |
| WAR (my dest ⊂ producer's source) | 10 | must stay |
| WAW | 1 | must stay |

Trigger waits by group over all **861** triggers: `qSPSpillReload0` 260, ACT dead-save sems 161,
S[4] 155, S[6] 139, S[5] 80, qPool 55, none 9, S[3] 2. The 161 that name the dead saves' sems are
WAR guards against a save's SBUF *read*; because the saves are dead those now resolve ~1.5 us
earlier (a free part of step 4's win).

## 4. Refuted this iteration

* **Queued→queued chains**: only **31** of 669 are real RAW chains (≈36 us of hop latency, and
  both ends usually carry `transpose_op: WZXY`, so composing them is not an affine rewrite).
  My first pass said "0 candidates" using exact (offset, partition-set) keys — that was wrong;
  you need interval overlap of the expanded access patterns. Low value, hard.
* **Removing the 218 stale ring-reuse waits** — measured: the producer transfer ends a **median
  23.2 us (min 14.9 us) before the trigger slot is even reached**. The wait is already satisfied;
  removing it cannot start anything sooner. Closed without a device run.
* **SP trigger slack is nil** — over the 376 triggers whose wait is on S[3..6],
  `trigger.start_ts − wait satisfied` is median 26–63 ns, total 73 us over all of them. SP (now
  4.3% busy) never delays a transfer. Confirms iter-1 §4.5 at full scale; **do not reorder SP**.
* **Dropping the now-meaningless waits on the dead-save relays — 0 us, device-measured.** Of the
  224 ACT relays, 97 have their completion used as a new trigger's data condition and 61 have
  waitless ACT followers, leaving **122** whose extended wait can go; those 122 block ACT for
  **63.8 us**. Dropped all 122 → **1205.0 us** vs 1204.2/1207.4 for keeping them, i.e. inside the
  noise. Reverted (the shipped NEFF keeps the waits: same speed, strictly more ordering).
  Lesson: **ACT blocking is not on the critical path; ACT *issue* time is.** Step 3 removed ACT
  issue time and converted at ~44%; this removed ACT blocking and converted at 0%.
* **Dead-store elimination on DVE — closed, ≤3 us.** Built a full SBUF read/write map
  (`scratch/deadstore.py`, 803k read cells) and asked which of the **776** DVE `MEMSET`/`COPY`
  stores have **no overlapping read after them**: **12**, all adjacent pairs, and all of them are
  A-group producers whose real reader *is* one of my new transfers — i.e. almost certainly map
  false positives. Even if real, 12 × 248 ns = 3 us of DVE issue. There is no dead DVE work.
  The map is the reusable part: address operands are exactly
  `{ACTIVATE,CAST,COPY,MATMUL,TENSOR_SCALAR_*: src+dst}`, `LDWEIGHTS: src`, `MEMSET: dst`,
  `TENSOR_TENSOR_ARITH_OP: src0+src1+dst`, each `<p>_mem_pattern_{start_addr_addr_immediate,
  num_elem[4], step_elem[4]}` with byte stride `step_elem*elem_size` and the partition span given
  by `num_active_channels` (absent ⇒ 128); addresses ≥ `0x2000000` are PSUM, not SBUF.
* `semaphores.py` **cannot see this class of edit**: it only reads the *primary* events block, so
  it reports `max_wait 0` for the new semaphores and under-counts waiters on S[4]/S[6] after
  waits move to the extended block. Totals are still a useful check (all four semaphore groups
  still balance exactly), but it cannot validate extended waits.

## 5. What to try next, in order

Engine busy is now **DVE 534 (44%), ACT 520 (43%), DMA 346, PE 341, Pool 233, SP 52**.
The rule this iteration established: **removing engine *issue* time converts to wall time at
~40%; removing *dependency chain* converts at ~16%; removing *blocking* converted at 0%.** Rank
candidates by issue time removed.

**Which engine's issue time is worth removing:** ACT and SP issue converted well because ACT is
43% busy and 464 us of SP's idle was ACT-bound (SP sat in ACT's shadow). **Pool is not a
candidate** despite having the most expensive slots: Pool idles **980 us of 1205, and 816 us of
that is waiting on S[6] (DVE)** — Pool issue time is already free. Rank by *busy* engine:
DVE 534, ACT 520.

1. **DVE is the top engine (534 us busy, 44%).** 2884 slots: `TENSOR_TENSOR` 1152, `CAST` 672,
   `COPY` 517, `MEMSET` 259, `EVENT_SEMAPHORE` 195; ~185 ns of real issue each, so there is no
   fat in the per-slot cost — only whole slots can go. **Not via dead stores** (§4: 12 of 776,
   and those 12 are map false positives). What is left is *redundant* work: a `COPY` whose
   consumer could read the source directly, or a `MEMSET`+accumulate that could be one op.
   iter-1 §4.2 counted *consecutive identical* slots, a much narrower question.
   One concrete thread I opened and did not finish: the 128 DVE `COPY`s that feed the A-group
   save sources read **PSUM** and write SBUF — they are matmul-result evacuations
   (`src_mem_pattern_start_addr_addr_immediate = 0x2000000 + off`, `addr_reg_marker = 2`; ch.5:
   PSUM is `0x2000000 + offset`, `offset < 0x4000`, and SBUF ends at exactly
   `128*262144 = 0x2000000`). If my fused transfers could read PSUM directly those 128 `COPY`s
   (~32 us of DVE issue) would die. **Two blockers, both unresolved**: (a) a queued descriptor
   names a variable and there is **no PSUM variable type anywhere in the 14-NEFF corpus** (only
   `state-buffer`), so the only hope is that `from:"SB", from_off >= 0x2000000` falls through to
   PSUM — untested; (b) PSUM is 8 banks × 2 KiB per partition, so reading it 1.5 us after the
   matmul almost certainly collides with the next matmul. Also, deleting the `COPY`s means
   renumbering every one of the ~2250 waits on S[6]. Low probability, high payoff.
2. **ACT (520 us busy).** 544 `TENSOR_SCALAR` + 576 `ACTIVATE` + 896 `CAST` ≈ 499 us of that is
   arithmetic at ~248 ns/slot; the 514 `EVENT_SEMAPHORE`s cost only ~21 us of issue. Nothing
   cheap left here — it needs whole operations to disappear.
3. **DMA latency on the PE path — ceiling now measured at ~150 us, mostly per-hop constants**
   (§3b). Only worth it if you can cut the *number* of hops PE waits for: the 669 transfers are
   dominated by DMA transposes (576 bf16 + 192 fp32 `transpose_op: WZXY` descriptors) feeding
   `LDWEIGHTS`, and PE needs them transposed, so they cannot simply go. §2.1's "add a queue with
   free semaphores" makes any DMA restructuring possible without renumbering if you find one.
4. **Delete** the 224 dead ACT `EVENT_SEMAPHORE`s rather than keep them: 40 ns each (~9 us of ACT
   issue) and it needs renumbering every wait on S[40..55]. Low value, do last.
5. **Do not spend budget on**: everything in iter-1 §5.4; everything in §4 above (queued→queued
   fusion, ring-reuse waits, SP reordering, dropping dead-save waits); and the 160 staging
   round-trips — they are gone, all 160 `tmp-buf` vars are dead and all 224 saves are no-ops.

## 6. Residual risks (stated honestly)

* **WAR on the fused transfers' SBUF sources — measured, not hand-waved.** I built the SBUF
  access map (per *partition*, exact intervals — a bounding box over the 64 scattered 512-byte
  runs spans 4 MB and tells you nothing; 285,600 write cells from every compute instruction's
  `dst_mem_pattern_*` + `num_active_channels` and every queued transfer's `to` cells). For each
  of the 192 fused transfers, the gap from its `end_ts` to the **next write to the same
  partition+offset**: **min 1.25 us, p10 2.94 us, median 17.1 us** (5 transfers have no later
  overwrite at all). Tightest cases: blk19/18 at 1.25 us (DVE slot 298 `TENSOR_TENSOR`), blk0 at
  1.41 us (ACT slot 25 `CAST`).
  *Why that is no worse than the compiler's own margin*: the transfer is gated on either
  `S[6]>=v` (A-Pool) or the save's ES, both of which fire at or before the point where ACT issued
  the original save; the transfer then takes 1.46 us, the save's DMA took ~1.5 us. So **the fused
  read finishes no later than the save's read did** — the margin is inherited, not shrunk. Still
  worth re-checking if a future iteration moves a trigger earlier.
  `verify.py --runs 3` is bit-identical and `max_abs_err` matches the baseline exactly.
  Script: `scratch/war2.py`.
* The Pool relays' `+16` can in principle overtake a still-in-flight live Pool DGE on the same
  semaphore; §2.7's repurposed barrier slot closes it for the 32 I touched.
* Scripts are in `scratch/` (reverted between iterations): `fuse_all.py` derives all 128 fusions
  from `work_orig` + a profile and asserts every shape; `apply_all.py` applies them. Both are
  ~120 lines and were written in ~40 min with the facts above in hand.

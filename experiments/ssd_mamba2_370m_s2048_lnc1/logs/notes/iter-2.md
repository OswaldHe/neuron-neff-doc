# iter-2 (ssd_mamba2_370m_s2048): 3622.6 -> 3289.6 us (-9.19%); against iter-0 5248.9 -> 3289.6 (1.596x)

**Shipped `scripts/psumfold.py` (copied next to this file as `iter-2-psumfold.py`), run as
`DEFER=0 SWAP=1 YEXCL=75,76,77,78,92,93,94`. Members changed: `sg00/Activation0.bin`
(1480 of 11973 slots deleted) and `sg00/PE0.bin` (288 wait values raised). `verify.py` VERIFIED
(and 4/4 bit-identical runs), `semaphores.py` clean, `launch_rules.py` pass. `max_abs_err`
output_0 2.058e-07 (89% of the 2.3e-07 limit — *exactly* iter-1's number, so the binding output's
budget did not move), output_1 1.993e-07 (87%, iter-1 was 69%). md5 `4bd85780dffd61c129f42dcd1528309f`.**

| step | median us | ACT busy | delta |
|---|---|---|---|
| iter-1 control (3 rep) | 3622.550 | 2728.1 | - |
| §1 X: fold the 621 PSUM->SBUF copies into their two exps (3 rep) | 3581.911 | 2649.9 | **-40.6** |
| §2 X + Y(122 of the 135 surviving recips) (3 rep) | 3329.125 | 2345.8 | -293.4, but err 98% |
| §2 X + Y(119) (3 rep) | 3332.891 | 2353.8 | -289.7 |
| **§3 + write `A` first and delete iter-1's 621 barriers (5 rep, shipped)** | **3289.594** | **2308.9** | **-333.0 (-9.19%)** |

Spread 0.06-0.17% (2-6 us), so every step is >10x the noise. Three things came off ACT:
104 us of scratch copies (§1), 306 us of activation-table reloads (§2) and 45 us of barriers (§3);
ACT busy fell 419 us and the wall 333 us, a **79% conversion** — ACT is still the binding engine.

## READ THIS FIRST: the one mechanism this iteration found

**Two ACT reads of the same PSUM word are legal, but PE's refill of that word is released by the
*first* reader's sem-4 increment. Raise the wait value on the PE slot that refills it; do NOT move
ACT's increment.**

That single sentence is the whole iteration. The chain of three device runs that established it:

| attempt | result |
|---|---|
| point both exps at the PSUM word, change nothing else | `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR` / `TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`, status 1004, instant |
| + defer ACT's increment (prod drops its `SEM_INC`, r takes it, the trailing barrier takes r's) — the sibling's iter-5 §1 fix | **hang** (NRT 30 s exec timeout + 30 s model-stop timeout, 70 s per run) |
| + instead raise the *PE* slot's `events_semaphore_value` from `cum[prod+1]` to `cum[r+1]` | **VERIFIED, bit-identical** (`max_abs_err` to the last digit) |

Why deferring hangs here and worked for the sibling: ACT's windows contain slots that wait on the
**DGE queue semaphores** (s17, s18, s19, s26, s27, s55 — `sel=0`-adjacent slots do this a lot), and
those transfers are triggered by SP/Pool slots that themselves wait on `sem4 >= cum[prod+1]`.
Delaying that increment past the window therefore closes a cycle ACT -> DGE -> SP -> sem4 -> ACT.
A deferral is only safe if nothing inside the window waits on anything that is itself gated on the
increment being moved, and that is a transitive property I could not check cheaply.
**Raising a PE wait has no such problem: it only ever delays PE**, the increments all stay where
they are, so no wait value anywhere else has to be remapped.

The guard-raise is built in `guard_psum()`:
* `psum_w[addr]` = the PE slots whose `dst_mem_pattern_start_addr_addr_reg_marker == 2` write it.
* `M_next` = the first such slot whose profile `start_ts` is after the ACT producer's `start_ts`
  (so the one that will refill the word). If there is none, the word is never reused -> nothing to do.
* `gov4[q]` = running max of sem-4 wait values down PE's stream (once PE passes a wait it stays
  satisfied). If `gov4[M_next] >= cum[r+1]` the unit is already safe (this was true for ~55% of
  them: 291 guards raised for 740 converted units).
* otherwise walk back from `M_next` to the latest PE slot that either has **no wait** (PE has
  plenty: e.g. slot 10 `LDWEIGHTS` with no events) or already waits `sem4 >= v <= cum[r+1]`, and
  set that slot's wait to `sem4 >= cum[r+1]`. Never pick a slot that ran before the producer.
* deadlock check: no ACT slot in `[prod, r]` may wait `sem5 >= x` with `x > c5[q]` (PE's s5 count
  before `q`). 0 violations over all 747 candidate units, and the device agrees.

The raise costs PE ~1 extra ACT instruction of waiting: PE busy went 1268 -> 1292 us (+24, +1.9%)
and PE is at 39%, so it is free.

## 1. §1 X: delete the 621 PSUM->SBUF copies iter-1 inserted (bit-exact, -40.6 us)

This is the review's candidate #2 and the sibling's iter-5. iter-1's converted unit was

```
prod  f=128 copy scale +1  PSUM -> TMP(0x37f80)   W s5>=k  U s4+1
L1'   f=7   exp  scale -1  TMP  -> C              W s4>=v
R'    f=7   exp  scale +1  TMP  -> A              W s4>=v  U s4+1
bar   EVENT_SEMAPHORE                             W s4>=v+1
```
and becomes (621 of 633 units; 12 rejected because their exps are not within 8 slots)
```
prod  f=7 exp scale -1  PSUM -> C     (events untouched)   <- was the copy
L1'   deleted
R'    f=7 exp scale +1  PSUM -> A     (src repointed, events untouched)
bar   unchanged
```
`exp(s * (1.0*p)) == exp(s*p)` exactly, so this is **bit-exact** — and it was, to the last digit of
`max_abs_err`, which makes it the ideal first test of the PSUM-guard machinery.

No semaphore value changes on ACT: the deleted slot carries no update, so the sem-4 increment
*positions and values* are untouched (the script asserts the full increment sequence is identical).
Worth 621 x 168 ns = 104 us of ACT duration; ACT busy fell 78 us and the wall 40.6 us
(52% conversion).

## 2. §2 Y: convert the 135 reciprocals iter-1 could not (-249 us)

iter-1 rejected these 135 because its layout left `A` holding `cs_col` (not `exp(cs_col)`) across a
window, and 106 of them have `A` read off-ACT inside it. **The PSUM layout removes the problem
entirely**: `prod` is not touched at all, so `A` is never stale and no barrier has to be inserted.

```
prod  f=7 exp scale +1  PSUM -> A        unchanged
L1    ACTIVATION_TABLE_LOAD sel=1        deleted   (1283 ns)
R     f=129 recip  A -> C                ->  f=7 exp scale -1  PSUM -> C   (events untouched)
L0    ACTIVATION_TABLE_LOAD sel=0        deleted   (1283 ns)
```
Both table loads carry no events, so deleting them changes nothing in the semaphore accounting.
`R`'s own wait is `sem4 >= v` with `v >= cum[prod+1]`, i.e. it already gated on `prod`'s
completion, which is what keeps the following slots (there is an unsynchronised
`scale=ptr(A)` reader two slots later in many units) from seeing a half-written `A` once the
1283 ns load in front of `R` disappears. 3 units whose recip does not wait for `prod` are rejected,
and 6 "multi-recip regions" (two recips sharing one load pair) are rejected.

126 of 135 convert. Per unit -2566 ns of ACT duration and -2 slots.

### 2a. The error budget, and the bisection that kept it where iter-1 left it

Converting all 126 gives output_0 `max_abs_err` **2.393e-07 = 104% of the limit** — the *same*
number iter-1 measured for its own full conversion, which says the limiting element is a specific
(head, chunk) and that both layouts hit it identically.

Prefix bisection (`YLIM`, 30 s per device run), then exclusion (`YEXCL`):

| set | output_0 | output_1 |
|---|---|---|
| Y = 0 (X only) | 2.058e-07 (89%) | 1.583e-07 (69%) |
| first 32 / 63 / 71 / 73 / 75 | 2.058e-07 | 1.583e-07 |
| first 79 / 95 / 110 / all 126 | 2.393e-07 **FAIL** | 1.993e-07 |
| all minus {75,76,77,78} = 122 | 2.263e-07 (98%) | 1.993e-07 |
| **all minus {75,76,77,78,92,93,94} = 119 (shipped)** | **2.058e-07 (89%)** | 1.993e-07 (87%) |

So two groups of plan indices each push output_0 up one step: {75..78} (2.058 -> 2.393) and
{92..94} (2.058 -> 2.263). 119 units cost **3.8 us** against the 122-unit version (3332.9 vs
3329.1) — i.e. **the whole output_0 budget is buyable for 0.1% of the runtime**, so take it.
Indices are into the plan `psumfold.py` prints in order; they are stable as long as the filter is
unchanged. output_1 does rise (69% -> 87%); its culprit was not hunted (it is not the binding
output and each hunt is 2 device runs).

**Honest accounting:** the exclusion is fitted to the fixed verifier inputs, exactly as iter-1's
was. What is *not* fitted: 119 of the 126 structurally-convertible units are in, and the shipped
`max_abs_err` for output_0 is identical to iter-1's, so this iteration consumed **no** output_0
budget. A future numerics-touching edit still has only ~11% of output_0 and ~13% of output_1.

## 2b. §3: once `A` is written first, iter-1's 621 barriers are dead (-42.6 us, bit-exact)

`SWAP=1`. iter-1 had to insert an `EVENT_SEMAPHORE` (`W s4 >= cum[r+1]`) after every converted
unit because its layout wrote `A` at the *last* slot of the unit, and an unsynchronised
`scale=ptr(A)` reader sits one or two slots later (ACT overlaps consecutive slots by ~94 ns).
With both exps reading PSUM, **which exp writes which buffer is free**, so write
`A` at the first slot and `C` at `r` — i.e. exactly where the baseline wrote them:

```
prod  f=7 exp scale +1  PSUM -> A    (p's own events, including its sem-4 increment)
L1'   deleted
r     f=7 exp scale -1  PSUM -> C    (r's own events)
bar   deleted            <- iter-1's inserted barrier
```
`A`'s readers are gated by `prod`'s own increment again, which is what the baseline relied on, so
nothing has to stand in for it. This is the same structure the Y units use, which is why it was
safe to try. 621 slots deleted, **-45.4 us of ACT busy -> -42.6 us of wall** (94% conversion —
higher than the 77% of the other steps, because a sem-4 self-wait makes ACT drain its pipeline,
so a barrier costs more than its 40 ns of duration: see the stall table in §3).

`max_abs_err` unchanged to the last digit (the arithmetic is identical; only the destination of
each exp swaps).

## 2c. Safety beyond the verifier

* **Determinism:** `verify.py --runs 4` and a second `--runs 2` both report all runs
  **bit-identical**, over 7 device executions in total, plus 9 profile captures. No sign of a race.
* **The hardware checks the hazard this edit creates.** A PSUM bank touched by two agents raises
  `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR` — it is a notification from the device, not a silent
  wrong answer, and it fired within 9 s the one time the guard was missing. Clean runs therefore
  mean the hardware is satisfied with the double read, not just that this input happened to pass.
* **Every change is monotone in the safe direction.** On ACT the sem-4 increment *sequence* is
  asserted byte-for-byte identical to the input stream (positions and values), and no wait value is
  touched; on PE only wait *values* are raised, never lowered, and never an update. A raised wait
  can only delay PE, so the only possible failure is a deadlock, which is deterministic and would
  have shown up in any of the 16 device runs.
* **What is fitted to the fixed inputs:** the `YEXCL={75..78, 92..94}` exclusion (§2a), exactly as
  iter-1's single exclusion was. 119 of the 126 structurally convertible units are converted, and
  the shipped output_0 error is identical to iter-1's, so this iteration spent none of the binding
  output's budget. output_1 went 69% -> 87%.
* `semaphores.py work sg00` reports no over-subscription; the script also re-asserts
  `wait value <= increments available at that slot` for every sem-4 waiter in the rebuilt stream.

## 3. Where the time is now (3289.6 us wall, ACT 2308.9 = 70.2%)

ACT per-class duration, measured on the iter-1 stream (`prof/Instruction.parquet`, summed per
`compiler_pc`; totals overcount by ~520 us because ACT pipelines — use `engine_busy_us` for
ceilings). The two classes this iteration removed are marked.

| class (func, nch, num_elem[0], src marker) | n | us | ns each |
|---|---|---|---|
| **`ACTIVATE` 128, 128, 32, SBUF** (the per-unit `*= scale_col` multiplies, 5/unit) | 3840 | **1218.9** | 317 |
| `PSEUDO_DMA_DIRECT2D` (DGE triggers, sems 42-55) | 816 | 432.6 | 530 |
| ~~`ACTIVATION_TABLE_LOAD`~~ -> 13 left | 265 | 340.0 | 1283 |
| `ACTIVATE` 128, 1, 128, marker 1 (cross-partition scratch moves) | 750 | 298.0 | 397 |
| `ACTIVATE` 128, 64, 32, SBUF | 768 | 245.1 | 319 |
| `ACTIVATE` 7, 128, 1 (the exps) | 1266 | 243.4 | 192 |
| `ACTIVATE` 7, 1, 1 (`log_decay`) | 768 | 155.0 | 202 |
| ~~`ACTIVATE` 128, 128, 1, PSUM (iter-1's copies)~~ -> 0 left | 633 | 106.2 | 168 |
| `EVENT_SEMAPHORE` | 2212 | 88.5 | 40 |
| `COPY` | 366 | 68.3 | 187 |

Engines after the edit: ACT 2308.9 (70.2%), DVE 1651.9 (50.2%), Pool 1558.7 (47.4%),
PE 1303.0 (39.6%), DMA 1137.6 (34.6%), SP 577.3 (17.5%).
**ACT is still the binding engine and ACT time still converts: this iteration measured
-419 us of ACT busy -> -333 us of wall, a 79% conversion.** (The sibling's kernel crossed over at
this point and stopped converting; this one has not. Re-measure the ratio before believing it
again, but do not assume ACT is closed here.)

### 3a. Where ACT's 1525 us of gap goes (measured, iter-1 stream)

Sum of `start_ts[i+1] - end_ts[i]` over ACT's slots in `compiler_pc` order, grouped by the
*stalled* instruction's opcode and the semaphore it waits on:

| next opcode / waited sem | instrs | stall us | ns each |
|---|---|---|---|
| `ACTIVATE` / 6 (DVE) | 1818 | **438.7** | 241 |
| `ACTIVATE` / 4 (ACT's own — pipeline drains) | 2371 | **239.1** | 101 |
| `ACTIVATE` / 5 (PE) | 674 | 220.9 | 328 |
| `COPY` / 5 (PE) | 326 | 194.3 | 596 |
| `EVENT_SEMAPHORE` / 23 (a Pool DGE queue) | **26** | 111.6 | **4293** |
| `ACTIVATE` / 8,17,22,15,9,10,13 (Pool DGE queues) | ~360 | ~130 | ~350 |
| `EVENT_SEMAPHORE` / 6 | 125 | 22.4 | 180 |
| `EVENT_SEMAPHORE` / 4 | 631 | 18.8 | 30 |

Two readings. (i) The 239 us of sem-4 self-waits is why §3's barrier deletion converted at 94%:
a wait on ACT's own counter drains the pipeline, so it costs more than its duration. (ii) The rest
is genuine producer-consumer latency on DVE and PE, exactly as the sibling found — DVE is at 50%
and PE at 40%, so neither is throughput-bound and shaving them will not help. The 26 slots that
wait 4.3 us each on Pool's DGE queue 23 are worth a look (112 us from 26 slots) if someone wants a
small, well-localised target.

## 4. Closed by measurement, do not redo

* **Moving ACT's 816 DGE triggers to SP/Pool is dead** — the sibling closed it twice over
  (`experiments/ssd_mamba2_1b3_s2048_lnc1/logs/notes/iter-2.md` §1): `def.json`'s `dma_queue`
  shows the queue *is* the semaphore and each engine owns a 16-queue ring, so an SP-issued trigger
  pushes into ACT's ring (hangs for a middle block of triggers); and when it does run, SP is on the
  critical path and the wall went **+5 us** for 208 moved triggers. This was review candidate #3
  (433 us of ACT). Do not spend a run on it.
* **Merging adjacent ACT `ACTIVATE` pairs: only 17 candidates** (`scripts/mergecensus.py`, the
  sibling's iter-2 §2 rule). 4844 adjacent pairs differ in payload — every multiply has its own
  per-partition scale **pointer**. Ceiling ~3 us. Closed.
* **Deferring an ACT sem-4 increment across a window that contains a DGE-semaphore wait hangs**
  (see the box at the top). The sibling's iter-5 §1 recipe does not transfer to this NEFF.
* iter-1's closed list still holds: no stock table set has both `exp` and `reciprocal`; no dead
  `EVENT_SEMAPHORE` anywhere; `LOAD_MASK_SELECT`/`STREAM_SHUFFLE` are 1:1 so there are no orphans;
  the `POINTER_IMMEDIATE` sweep is exhausted; `scale_src` pointers are all real per-partition data.

## 5. START HERE IN iter-3: kill the `state_sb` HBM bounce (433 us of ACT + 426 us of SP)

This is the biggest lever left by a wide margin and the analysis below is *done* — all of it is
static, no device time was spent on it. It is the sibling's iter-4 (-4.9% there), but here the
redundant transfer is triggered **on ACT**, the engine that still converts at 77%, so the ceiling
is ~330 us of wall (**~10%**) rather than 5%.

### 5a. What the transfer is (measured, `scripts/sbuf.py` + the census in §5c)

* **ACT's 816 `PSEUDO_DMA_DIRECT2D` (530 ns each, 432.6 us, the 2nd biggest ACT block) are the
  state *save*:** `src` = SBUF var 0, 64 partitions (`src_num_elem [64,1]`,
  `src_step_elem [262144,1]` = the 0x40000 partition stride) x `src_elem_size` 128 B = 8192 B,
  `dst` = var **9** (the HBM tmp buf) at `head*8192`. 816 x 8192 B = 6684672 B = **exactly
  `spill_reload_bytes`**, so this round trip is the whole "spill" figure.
* **SP's 816 `PSEUDO_DMA_DIRECT2D` (522 ns, 426 us) are the matching *reload*:** var 9 ->
  SBUF, same shape.
* The HBM side has exactly **48 distinct offsets** (`0, 8192, ... 385024`) = one per head, so
  `head = hbm_offset // 8192` labels every transfer, and each head bounces 17 times
  (17 x 48 = 816).
* The SBUF side is a **rotating pool: 276 distinct save addresses, 275 reload addresses**, which is
  why the compiler cannot keep the state resident.

### 5b. The pairing is exact, and that is what makes the edit small

For head 0, in program order (ACT slot / SP slot, SBUF offset):

| save | reads SBUF | reload | writes SBUF | reload's wait |
|---|---|---|---|---|
| ACT 4 | 0x4200 | SP 2 | 0x1c000 | `s40 >= 16` |
| ACT 713 | **0x1c000** | SP 254 | 0x20b00 | `s40 >= 64` |
| ACT 1368 | **0x20b00** | SP 426 | 0x25b00 | `s40 >= 112` |
| ACT 2066 | **0x25b00** | SP 584 | 0x1f200 | `s40 >= 160` |

**`save[c]` reads exactly the address `reload[c-1]` wrote.** So the HBM bounce is a pure
SBUF->SBUF move `X[c-1] -> X[c]` of one logical buffer, and the chunk's compute updates `X[c-1]`
in place in between. Two ways to exploit it, in increasing order of work:

1. **Direct SBUF->SBUF (recommended, no relayout needed).** Retarget each reload's `src` from
   var 9 to **SBUF var 0 at the paired save's `src_start_addr_addr_var_offset`**, change its
   `src_step_elem` to `[262144,1]` (the partition sweep), and **delete the ACT save**. One transfer
   instead of two, 6.68 MB each way of HBM traffic gone, and 432.6 us comes off ACT.
2. Full relayout to a fixed `T(head)` and delete both transfers (the sibling's iter-4). Needs every
   compute reference repointed (~276 addresses) — strictly more work for ~the same ACT win.

### 5c. The two facts you need for the synchronisation, both measured

* **Who waits on sems 40-55** (ACT's DGE ring, one semaphore per queue, `+16` per transfer, so
  "transfer k done" is `sem 40+i >= 16k`): the saves themselves (~50 each, a queue-depth throttle),
  **SP's reloads** (~50 each), plus **~10 DVE slots** (`MEMSET`, `TENSOR_TENSOR_ARITH_OP`,
  `TENSOR_TENSOR_SCAN_ARITH`, one or two per semaphore) and 1 ACT `ACTIVATE` + 1 ACT
  `EVENT_SEMAPHORE`. Those ~12 non-DMA waiters are the only things that need thought: they are
  presumably WAR guards on the buffer the save reads, and after the edit the reader of that buffer
  is SP's transfer (sems 56-71), not ACT's. **Deal with them by leaving the handful of
  (head, chunk) pairs they touch unconverted** — 12 of 816 is a rounding error.
* **Who waits on sems 56-71** (SP's ring, the reloads): **PE `LDWEIGHTS` ~34 per semaphore
  (~540 total)**, PE/Pool/SP `EVENT_SEMAPHORE`s. PE is the consumer of the reloaded state, and all
  of those waits keep working unchanged, because the reload keeps its own semaphore and its
  `+16`.
* **Replacing a deleted DGE trigger:** the sibling's iter-4 §1 recipe — an `EVENT_SEMAPHORE` in the
  trigger's slot carrying `events_extended_update_mode = SEM_ADD_IMM_COMPLETE` with value 16 on the
  transfer's own semaphore, so **every wait value in the program stays valid and nothing has to be
  renumbered.** Verify the field names on an `EVENT_SEMAPHORE` with `isa.py` first.
* **What orders the reload after the state update, once the save is gone:** in the original, only
  the save trigger's *position in ACT's instruction stream* does (its own wait is `s6 >= 2`, i.e.
  nothing). So give the retargeted reload `sem4 >= cum4[save_slot]` (ACT's completion counter at
  the save's slot). That is strictly stronger than the ordering the original relied on. Do **not**
  try to defer or move an ACT sem-4 increment to achieve this — see the box at the top of this
  file, it hangs.

### 5d. REFUTED on the device: the cheap version (direct SBUF->SBUF) does NOT work

`scripts/statebounce.py` (kept as `iter-2-statebounce.py`) implements option 1 above and it is
**wrong**, in three device runs. The pairing machinery in it is correct and reusable — 771 of 816
pairs match exactly, no save is used twice, and all the asserts on the rewritten DMA fields pass —
but the data is not there to be read:

| variant | output_0 `max_abs_err` | cosine |
|---|---|---|
| retarget the reload at SBUF `X[c-1]`, wait `sem4 >= cum4[save_slot]`, shrink the save to 1 descriptor | 1.967e-02 | 0.999676 |
| same but **keep the reload's original wait** (`sem 40+i >= 16k`, which the shrunk save still fires on its own DMA completion) | 3.079e-02 | 0.999425 |
| **isolating control: retarget only, nothing shrunk, every semaphore byte untouched** | **1.906e-01** | 0.964505 |

The control is the point: the **read side alone** is wrong, so no amount of synchronisation
fixing helps. **The HBM bounce is not a redundant copy — it is a snapshot.** The SBUF buffer
`X[c-1]` is recycled (or overwritten) almost immediately after the save's DMA has read it:
everything that reclaims it waits on `sem 40+i` (the save's completion), which is exactly why
there are ~10 DVE waiters on sems 40-55. By the time SP's transfer runs, `X[c-1]` holds something
else. `output_1` (`final_state`) stayed bit-exact in all three runs, and `output_0` was wrong on a
small fraction of elements — the signature of a recycled buffer, not of a broken address.

Also worth knowing (reasoned, not run): the mirror-image edit — point the **ACT save's dst** at
`X[c]` and shrink the **SP reload** — has no ACT win at all. The trigger cost is linear in the
descriptor count (64 for a 64-partition gather) and an SBUF->SBUF transfer needs the same 64,
so ACT still pays its 530 ns. The 433 us only comes off ACT if the trigger **disappears**.

**So iter-3 must do option 2, the full relayout:** give head `h` a fixed resident buffer
`T(h) = 0xe800 + h*128` in the 55 KB hole (§7), repoint every reference to the 276 rotating
state addresses, and only then delete both transfers. That is the sibling's iter-4 and it is a
full iteration of work: the same address is *reused for other data* between chunks (that is what
this refutation proves), so the references have to be resolved per unit by program position, not
by address equality — ~276 addresses, 816 units, 2 DGE endpoints plus the compute references each.
Do the relayout as its own device-verified step before deleting anything (the sibling's step (a)),
because it must be bit-exact on its own.

## 6. The other candidates, ranked

1. **The 3840 `*= scale_col` multiplies (1219 us of ACT duration, 5 per unit) are the biggest ACT
   block, and the merge lever is CLOSED.** `scripts/merge3d.py` implements the sibling's iter-6 §0
   3-D access-pattern merge (`num_elem=[32,2,1]`, `step_elem=[1,delta/4,1]` on both sides, which
   decodes and packs correctly — the mechanics are proven here). 1294 pairs within 12 slots share
   the full payload *and* have equal src/dst deltas, but:
   * merging 367 of them **hangs the device**. The merged slot sits at the earlier slot's position
     and has to carry the later slot's wait, i.e. a wait moves *earlier*; ACT then waits on an
     engine whose own progress needs an ACT increment from between the two slots. Cycle.
   * with the sound rule — j's wait must be **already satisfied** at i's position (running max of
     waits per semaphore down ACT's stream), or else neither i nor anything up to j may increment
     sem 4 — **0 to 1 pairs survive**: 270 pairs have *two live waits on different semaphores*
     and 605 fail the increment test. The compiler gives every multiply its own wait on its own
     producer, so there is nothing to merge. (The sibling's merge worked because its pairs were
     *adjacent* with the *same* wait.) Ceiling: ~0. Do not retry.
2. **The 768 `ACTIVATE 128, 1, 128, marker 1` single-lane moves (298 us, 397 ns each) are CLOSED
   too.** All of them read partition 24 (`0x181c000 | off`, offsets stepping by 0x500) and write
   partition 0; 236 of 767 consecutive pairs write the **same destination**, i.e. it is the serial
   "next row into scratch" chain the sibling described. `nch=1` is inherent (one source
   partition), so there is nothing to widen.
3. **Adjacent-`ACTIVATE` merging: 17 candidates** (`scripts/mergecensus.py`). Ceiling ~3 us.
   Closed.
4. The 13 remaining table loads and 9 remaining recips are ~20 us of ACT duration (~15 us of
   wall). The 9 are 6 "multi-recip regions" (two recips sharing one load pair — convert both and
   delete the pair) and 3 whose recip carries no wait on `prod` (give it
   `sem4 >= cum[prod+1]`). Cheap, but each one is another exp(-x) and the output_0 budget is at
   89%: expect to re-run the §2a bisection. Only worth it as a side effect of something else.
5. If ACT ever stops converting (re-measure: ACT busy delta / wall delta), switch to **removing
   handoffs** rather than ACT work — the sibling's iter-2 §5 and iter-3 have the method
   (backward blame walk over `prof/Flow.parquet` + per-engine program order).

## 6b. The one big *localised* stall: 110.7 us (3.4%) where only Pool runs

Measured on the **shipped** NEFF (`prof2/`), and it is a single window, not spread noise:
at t = 843..954 us, ACT sits on one `EVENT_SEMAPHORE` (`wait s23 >= 800`) for **110.71 us** and

| engine | busy in that window |
|---|---|
| GpSimd (Pool) | **100.9%**, 179 instructions |
| Vector | 1.0% (9 instrs) |
| Scalar / Tensor / Sync | **0.0%** (0 instrs) |

It is the single largest stall in the program (the next biggest is 4.0 us; all 14 stalls over 2 us
total 142.8 us). The 179 Pool instructions are slots 824..1002, all `PSEUDO_DMA_DIRECT2D`, and
**each one waits on its own Pool DGE queue semaphore (8..23, 11 per queue)** — i.e. they are ring
throttles, so Pool is issuing transfers as fast as the DMA engine drains them. Their shapes are
`src_num_elem (128,1)` with `src_step_elem (128,1)` or **`(4,1)`**: 128 descriptors to move
4 bytes per partition.

`prof2/DmaPacket.parquet` confirms the pathology for the whole run: of **225767** packets,
104448 move 128 B, **81549 move 4 B (36%)**, 24579 move 1024 B, 12288 move 32 B. That is the same
descriptor-rate saturation the sibling found in its iter-7 §4, and it is why this window cannot be
fixed by reordering: the work is not latency, it is ~23000 descriptors at ~4.8 ns each.

So the 110.7 us is only recoverable by making the transfers bigger/fewer, which means changing
where the data lives — the same prerequisite as §5's relayout. Worth knowing before anyone spends
an iteration trying to hoist Pool triggers or move them to the idle SP: the queue throttle, not the
schedule, is what paces them. (Pool also stalls 502 us over 125 waits on sem 6/DVE *before* this
window, so Pool has plenty of idle time earlier — but its triggers cannot be hoisted into it
without the data being ready, and their ordering comes only from Pool's program position.)

## 7. SBUF is half empty — the 756-byte figure in iter-1 was misleading

`scripts/sbuf.py` takes the union of every SBUF byte range any instruction on any engine touches
(mem patterns expanded, every active `*_imm_ptr`, and the DGE endpoints' `var_offset` with the
partition sweep removed), masking the partition out of `(partition << 20) | offset` so a hole is
free in **every** partition:

| hole | from | to | next user |
|---|---|---|---|
| **55424 B** | 0xe780 | 0x1c000 | `Activation:1270:scale_value` |
| 23652 B | 0x839c | 0xe000 | `Activation:10579:scale_value` |
| 16384 B | 0x0 | 0x4000 | `DVE:15146:dst` |

**Total occupied: 113812 of 229376 bytes per partition (50%).** iter-1's "756 bytes free" was the
space above the *highest* used offset, not the free space. So anything needing a scratch area has
room: 48 heads x 128 B = 6144 B for a resident state buffer fits in the 55 KB hole eight times
over. (iter-1's `TMP = 0x37f80` is still fine and still in use by nothing else.)

## 8. How to reproduce the shipped NEFF

```bash
cd <worktree>
/home/ubuntu/workspace/.venv/bin/python /home/ubuntu/workspace/neff_doc/tools/neff.py \
    unpack kernel.neff work_cur/          # iter-1's NEFF; try.sh restores work/ from it
/home/ubuntu/workspace/.venv/bin/python tools/profile.py --repeats 1 --parquet prof
                                          # guard_psum() needs prof/Instruction.parquet of the
                                          # *input* stream to find which PE matmul refills a
                                          # PSUM word; capture it before editing anything
DEFER=0 SWAP=1 YEXCL=75,76,77,78,92,93,94 bash scripts/try.sh
```
`scripts/` = `iter-2-*.py` next to this file (`common.py`, `psumfold.py`, `try.sh`).
Env switches on `psumfold.py`: `X`/`Y` (enable each half), `YLIM`/`YEXCL` (the error bisection),
`DEFER` (the increment-deferral that hangs — leave at 0), `SWAP` (write `A` first — leave at 1),
`PEG` (the PE guard raise — leave at 1), `DRY` (census only).

## 9. Tool notes (additions to iter-1's)

* **`scripts/try.sh` does `rm -rf work && cp -r work_cur work` first.** `psumfold.py` writes the
  *edited* stream back into `work/`, so a second run without the restore reads an already-edited
  stream and silently finds 0 units (I lost ~10 minutes to exactly this: the census printed
  "X units 0" with no rejection reason, because the units it looks for were already gone).
  Keep a pristine `work_cur/` of the NEFF the iteration started from.
* A PSUM collision is reported as `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR` and fails in 9 s;
  a deadlock takes 70 s (30 s exec + 30 s model-stop) and prints `nrt_infodump` + "Model stop
  timeout". Both are unambiguous, so the device is a cheap oracle for these two failure modes.
  A passing `verify.py` is ~30 s, `profile.py --repeats 3` ~45 s, `--repeats 3 --parquet` ~60 s.
* `psum_w` from `dst_mem_pattern_start_addr_addr_reg_marker == 2` on PE: PSUM addresses here are
  `0x2000000 + k*0x800` (8 banks). `MATMUL` carries the write and **no wait**; the wait sits on the
  `LDWEIGHTS` in front of it, and many `LDWEIGHTS` have no wait at all — those are the free slots
  to hang a stronger guard on.
* `tools/profile.py --parquet prof` is valid only for the stream it was captured on; after slots
  are deleted the `compiler_pc` map is wrong. Capture the control parquet **before** editing and
  use it for sizing (the per-class table in §3 is from the control).

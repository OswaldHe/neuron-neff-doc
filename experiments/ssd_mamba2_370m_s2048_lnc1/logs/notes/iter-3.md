# iter-3 (ssd_mamba2_370m_s2048): 3291.3 -> 3106.0 us (-5.63%); against iter-0 5248.9 -> 3106.0 (1.690x)

**Shipped three edits, all bit-exact in the sense that matters: `max_abs_err` is *identical to the
last digit* to iter-1's and iter-2's (output_0 2.058e-07 = 89% of the 2.3e-07 limit, output_1
1.993e-07 = 87%), so this iteration spent **none** of the error budget. `verify.py --runs 4` three
times: VERIFIED, all runs bit-identical. `launch_rules.py` pass, `semaphores.py` clean. md5
`48972372faa61507c7b22203d7887d14`. Members changed: `sg00/Activation0.bin` and `sg00/SP0.bin`.**

| step | median us (5 rep) | spread | delta |
|---|---|---|---|
| iter-2 control (3 rep) | 3291.272 | 0.19% | - |
| §1a state bounce staged through resident SBUF instead of HBM | 3277.036 | 0.05% | -14.2 |
| §2 last 8 recips -> exp(-cs), 10 table loads neutralised | 3273.997 | 0.06% | -3.0 |
| §1b 605 of 816 bounces go single-hop; ACT save shrunk to 1 descriptor | 3234.733 | 0.15% | **-39.3** |
| **§1c those 605 ACT saves become a pure `SEM_ADD_IMM` EVENT_SEMAPHORE (shipped)** | **3106.002** | **0.15%** | **-128.7** |
| | | | **total -185.3 (-5.63%)** |

**THE HEADLINE, in one sentence: iter-2's notes declared the `state_sb` HBM bounce dead after two
device refutations, and it was not — it was worth 5.1% once the pair filter was done with a
*correct* DGE address decode (§4.3) and the dead trigger was replaced by a semaphore op instead of
a shrunken DMA (§1c).** §3 (the measured critical path) is the tool that sized it; §4 is five
levers genuinely closed on the device.

Scripts are next to this file: `iter-3-cpath.py` (the critical-path decomposition — the most
reusable thing here), `iter-3-bounce3.py` (§1a), `iter-3-single.py` (§1b+§1c),
`iter-3-lastrecip.py` (§2), `iter-3-recyc.py` (the liveness census that reopened the lever),
`iter-3-holechk.py` (§4.3's corrected DGE decode), `iter-3-mergej.py` (§4.1, refuted),
`iter-3-dcefix.py` (§4.3, refuted), `iter-3-common.py`.

---

## 1. §1: the `state_sb` bounce (-182 us total, bit-exact) — three steps, in the order they must go

### 1a. Stage through a resident per-head SBUF buffer instead of HBM (-14.2 us)

`iter-3-bounce3.py`. Per (head, chunk) the kernel did

```
ACT  PSEUDO_DMA_DIRECT2D  SBUF X[c-1] (64 part x 128 B) -> HBM var9 + head*8192   sem 40+i, +16
SP   PSEUDO_DMA_DIRECT2D  HBM var9 + head*8192 -> SBUF X[c]   wait s40+i>=16k, sem 56+j, +16
```

and now does

```
ACT  PSEUDO_DMA_DIRECT2D  SBUF X[c-1] -> SBUF T(head)     (events, src, shape all untouched)
SP   PSEUDO_DMA_DIRECT2D  SBUF T(head) -> SBUF X[c]       (events, dst, shape all untouched)
```

with `T(head) = 0xe780 + head*128` (48 x 128 B in the 55 KB hole, 64 partitions, partition stride
0x40000). **771 of 816 pairs converted** (17 reloads wait on something other than sems 40-55,
28 have an HBM offset that does not match their save).

**Why this one works when iter-2's §5d and this iteration's mirror (§4.2) both fail.** The
rotating SBUF pool is *tight on both sides*: `X[c-1]` is recycled almost as soon as the save's DMA
has read it (that is iter-2's refutation), and `X[c]` is still holding live data until the reload's
own position in SP's stream (that is §4.2's refutation, measured today). So **neither endpoint of
the move may shift in time.** Staging through a private buffer shifts neither: each transfer keeps
its slot, its events, its semaphore and its `+16`, and the only thing that changes is where the
bytes rest in between. Nothing is renumbered, so no wait value anywhere has to be remapped.

`T(head)` needs no interlock: `save(h,c+1)` reads `X[c]`, which `reload(h,c)` writes, so
`save(h,c+1)` is already ordered after `reload(h,c)` by the existing data dependency — one buffer
per head is enough. (Proved by construction and by the device: bit-identical outputs.)

What it bought: **DMA busy 1137.6 -> 1096.3 us (-3.6%)**, 6.68 MB of HBM writes and 6.68 MB of HBM
reads gone, `spill_reload_bytes` is now SBUF-to-SBUF. Wall -14.2 us — much less than the 165 us of
`LDWEIGHTS` stall suggested, i.e. **the HBM latency was already almost entirely hidden**. On its
own this step is barely worth it; it is here because it is the *safe* intermediate that makes 1b
possible and because the 166 pairs 1b cannot take still use it.

### 1b. 605 of 816 pairs go single-hop; the ACT save shrinks to one descriptor (-39.3 us)

`iter-3-single.py` (default `GUARD=20000`). iter-2 §5d tried exactly this — SP's reload reads
`X[c-1]` straight out of SBUF and the ACT save is reduced to a stub that only fires its semaphore
— and measured output_0 `max_abs_err` 1.9e-01, concluding "the HBM bounce is not a redundant copy,
it is a snapshot: `X[c-1]` is recycled almost immediately". **That conclusion was right about the
mechanism and wrong about the scope.** `iter-3-recyc.py` enumerates, per pair, every instruction
on every engine whose SBUF range overlaps `X[c-1]` and whose profile window falls inside
`[save, reload]` — with the **corrected DGE address decode of §4.3**, which is what iter-2's
analysis was missing:

| margin around the window | pairs with zero toucher (of 799) |
|---|---|
| 0 | **786** |
| 5 us | 707 |
| 20 us | 605 |
| 50 us | 437 |

So only ~13 pairs are *actually* recycled inside the window, which is exactly the "wrong on a small
fraction of elements, cosine 0.9645, output_1 bit-exact" signature iter-2 reported — it converted
all of them. At iter-1's established `GUARD=20000` ns, **605 pairs are clean** and the edit is
bit-exact on the device. The 166 rejected pairs keep 1a's two-hop form (T(head) is per-head and
`save(h,c+1)` is ordered after `reload(h,c)`, so mixing converted and unconverted chunks of the
same head is safe), the other 45 keep the original HBM path.

The filter is **stable across schedules**: re-running it against a profile captured on the
*edited* 3106 us stream gives the same 605 pairs. That is the check that makes a
timestamp-based gate defensible — iter-1's warning about baseline timestamps going stale applies,
so do the re-run.

### 1c. The stub save should be an `EVENT_SEMAPHORE`, not a 1-descriptor DMA (-128.7 us, the big one)

`KILL=1`. Shrinking the save to one 4-byte descriptor moved **ACT busy not at all** (2297.2 ->
2298.0 us): a DGE trigger's cost on the issuing engine is essentially fixed (~390 ns) and only
~2.2 ns/descriptor, so 1b's whole win came from the DMA engine, not from ACT. The trigger's only
remaining job is its `+16`, and ACT's 816 DGE triggers carry **no event-block update at all**
(15 have no events; the rest carry one `WAIT_FOR_SEM_GE_IMM` ring throttle). So replace it with

```
EVENT_SEMAPHORE  events_update_mode        = SEM_ADD_IMM_COMPLETE
                 events_update_idx         = the trigger's `semaphore` (40+i)
                 events_semaphore_value    = the trigger's `sem_increment` (16)
                 events_extended_wait_*    = the trigger's original ring-throttle wait
```

`events_semaphore_value` doubles as the primary wait threshold, so the wait has to go in the
**extended** block — a form the compiler itself uses 138 times here. `SEM_ADD_IMM_COMPLETE` was
untested in this NEFF; **it works**, and `semaphores.py` confirms every ACT DGE semaphore still
gets exactly 816 increments (now `{dge:Activation: ~210, Activation:EVENT_SEMAPHORE: ~600}`).

ACT busy 2298.0 -> **2151.8 us (-146 us)**, DMA 1038.7 -> 995.7, and the ACT DGE trigger's share of
the critical path falls **184.5 -> 45.5 us**. Wall -128.7 us. This is the lever that was worth
having, and the general shape of it is reusable: **a DGE trigger whose transfer is dead but whose
`+16` is load-bearing costs ~40 ns as an `EVENT_SEMAPHORE` and ~390 ns as the smallest possible
DMA.**

## 2. §2: the last recips and the last table loads (-3.0 us, error budget untouched)

`iter-3-lastrecip.py`. 16 `f=129` (reciprocal) ACTIVATEs and 27 `ACTIVATION_TABLE_LOAD`s survived
iter-2 (1 initial `sel=0` + 13 `sel=1`/`sel=0` pairs). Mechanism is iter-2 §2's Y pattern:
rewrite the recip as `f=7, scale=-1` reading the producing exp's **PSUM** word, raise the recip's
own wait to `sem4 >= cum[prod+1]` if it is weaker, and raise the PE guard on the slot that refills
that PSUM word (`guard_psum()` ported verbatim from iter-2; it needs `prof/Instruction.parquet` of
the stream being edited — ACT/PE slot indices are unchanged by §1, so iter-2's parquet is still
valid).

Then the new part: **a table load whose set is already active is a 1283 ns no-op, so once every
`f=129` inside a `sel=1`/`sel=0` region is converted, both loads of that region can be replaced by
an empty `EVENT_SEMAPHORE`** (loads carry no events, so the slot count and the entire sem-4
increment sequence are unchanged; the script asserts `ncum == cum`). It also re-walks the whole
stream asserting that every surviving `f=129` is still under `sel=1` and every `f=7` under `sel=0`.

15 of 16 recips are structurally convertible (1 waits on sem 19 — the same one iter-1 excluded).
**The error budget is what caps it, exactly as in iter-1/iter-2**, and the steps are the same
2.058 / 2.263 / 2.393e-07 ladder:

| set (plan indices) | units | loads killed | output_0 |
|---|---|---|---|
| all 15 | 15 | 24 | 2.393e-07 **104% FAIL** |
| first 8 | 8 | - | 2.393e-07 **FAIL** |
| first 4 | 4 | - | 2.058e-07 (89%) |
| all minus {5,6,7,8} | 11 | 16 | 2.263e-07 (98%) — passes, but spends the budget |
| **all minus {5..11} (shipped)** | **8** | **10** | **2.058e-07 (89%)** |

So `{5,6,7,8}` holds the 104% culprit and `{9,10,11}` holds the 98% one. **I deliberately did not
take the 98% version**: it is worth about 5 us more and it would leave a future iteration ~2% of
output_0's budget on a fitted-to-these-inputs margin. 10 loads x 1243 ns = 12.4 us of ACT duration
-> 3.0 us of wall, a **24%** busy-to-wall conversion.

**That 24% is not a property of ACT, it is a property of these particular slots**, and the
difference matters: §1c removed 146 us of ACT busy and got 128.7 us of wall (**88%**). The table
loads happened to sit mostly off the critical path (26 instances but only 23.7 us of path, 912 ns
each out of 1283 ns), so they converted badly. **Stop pricing ACT edits by `0.8 x ACT busy saved`
and price them by their measured share of the critical path (§3) instead** — that number was
184.5 us for the DGE triggers and it predicted §1c almost exactly (-139 us of path, -128.7 us of
wall).

### 1d. Inventory of the shipped DGE transfers (so the next iteration knows what is left)

```
Activation  166  SB -> SB   (two-hop staging, the GUARD-rejected pairs)   + 605 EVENT_SEMAPHORE
             45  SB -> var9 (the original HBM path, shape/wait mismatches)
SP          771  SB -> SB   (605 single-hop reading X[c-1], 166 reading T(head))
             45  var9 -> SB
Pool        816  SB -> TBL  (writes output_0 / output_1)
            768  TBL -> SB  elem 128   (the 16 KB compute loads)
            768  TBL -> SB  elem 4     (the descriptor storm, §3a)
             33  TBL -> SB  elem 256/512
```
Two independent 5-repeat runs of the shipped binary: 3106.002 and 3106.407 us (spreads 0.15% and
0.12%), so the number is reproducible to under a microsecond.

## 3. READ THIS FIRST: the measured critical path, and how to get it

`iter-3-cpath.py`. This replaces "engine busy %" as the sizing tool and it is cheap (static +
one `profile.py --repeats 1 --parquet prof`).

Method: build the real dependency DAG and walk it backwards from the last-finishing instruction.
* nodes = `prof/Instruction.parquet` rows with `compiler_pc` not null; **`compiler_pc` is exactly
  the slot index** in `work/payload/sg00/<Engine>0.bin` (Scalar=Activation, Vector=DVE,
  Tensor=PE, GpSimd=Pool, Sync=SP).
* predecessors = (a) program order on the same engine, (b) **the semaphore release**: decode each
  slot's `(events_wait_idx, events_semaphore_value)` and find, in `prof/SemaphoreUpdate.parquet`,
  the update at which that semaphore last crossed the threshold *before* the waiter started.
  `SemaphoreUpdate.id` is a string like `S[10] (qGpSimdDynamic)`, so parse the number out, and
  **restrict to updates before the waiter's `start_ts`** — semaphores are reset (`sem_wr_imm`)
  during the run, so a global "first crossing" search collapses the walk to a few hundred nodes.
* `Flow.parquet` alone is **not** enough: it carries data flow only, so the walk jumps over the
  semaphore-only edges and reports 518 us of "path" for a 3287 us run.
* then *clip*: attribute to each path node `min(end_ts, successor.start_ts) - start_ts` and the
  remainder as a gap. The clipped numbers **sum exactly to the wall span**, which is the check
  that the decomposition is real (4954 us unclipped vs 3286.6 us clipped = the wall).

Decomposition of the **final shipped** stream's 3105.4 us wall (the mid-iteration 3271.9 us stream
in brackets, so you can see what each step moved):

| | us | % | |
|---|---|---|---|
| **Scalar/ACT busy** | **1161.4** | **37%** | ACTIVATE 1034.3, EVENT_SEM 50.1, **DGE trigger 45.5 [184.5]**, COPY 16.5, TABLE_LOAD 15.0 [23.7] |
| **Tensor/PE busy** | **685.3** | **22%** | MATMUL 538.3, LDWEIGHTS 147.0 |
| **Vector/DVE busy** | **504.2** | **16%** | TT_SCAN 158.5, TENSOR_TENSOR 143.8, STREAM_SHUFFLE 128.5, COPY 37.1, MEMSET 31.5, LMS 4.8 |
| GpSimd/Pool busy | 24.9 | 1% | |
| **gaps** | **722.7** [779.7] | **23%** | ACTIVATE 206.4, ACT EVENT_SEM 137.3, LDWEIGHTS 115.9, ... |

ACT `ACTIVATE` on the path, by instruction class (`func, nch, num_elem[0], src marker, scale`),
measured on the 3271.9 us stream (the classes themselves did not change in §1b/§1c):

| class | n on path | us on path | ns each |
|---|---|---|---|
| `128, 128, 32, SBUF, PTR` (the `*= scale_col` multiplies) | 2410 | **610.5** | 253 |
| `DMA_DIRECT2D` (state save) — **now 45.5 us after §1c** | 516 | 184.5 | 358 |
| `7, 128, 1, PSUM, IMM` (the exps) | 1083 | 129.7 | 120 |
| `128, 64, 32, SBUF, PTR` | 513 | 106.3 | 207 |
| `128, 1, 128, marker 1, PTR` | 289 | 84.6 | 293 |
| `7, 1, 1, SBUF, IMM` (`log_decay`) | 442 | 78.4 | 177 |
| `ACT_TABLE_LOAD` (16 left, was 26) | 16 | 15.0 | 938 |
| `129, 128, 1` (8 recips left) | 7 | 1.2 | 171 |

**So: ACT is 37% of the wall, not its 69% busy figure; PE is 22% and DVE 16%. No engine owns the
run** (exclusive-busy, i.e. time where only one engine is live, is only Scalar 236 / Pool 187 /
Tensor 95 / Vector 80 us). Use the per-class critical-path column, not `engine_busy`, to size
anything from here.

### 3a. The gaps (786 us), and what each one really is

| waiting instruction | gap us (shipped) | what it is |
|---|---|---|
| Scalar `ACTIVATE` | 212.7 | genuine producer latency on DVE (sem 6) and PE (sem 5) |
| Tensor `LDWEIGHTS` | 158.8 | PE waiting for the state reload (sems 56-71). §1 cut one HBM hop out of this and got 14 us, so **most of it is not HBM latency** |
| Scalar `EVENT_SEMAPHORE` | 134.3 | dominated by the single 110.7 us wait on `s23` (Pool's DGE queue) that iter-2 §6b found |
| Vector `STREAM_SHUFFLE` | 100.0 | DVE waiting on its own program order |
| GpSimd `EVENT_SEMAPHORE` | 57.4 | Pool waiting on its own DGE queues (descriptor-rate throttle) |
| Tensor `EVENT_SEMAPHORE` | ~33 | |

Caveat on the method: an update whose `update_source` is a DMA queue (`qGpSimdDynamic`, ...)
rather than an engine has no instruction to blame, so those edges are dropped and the gap lands in
the `prog` bucket of the waiter. That is why the two DGE-queue waits above show up as
`EVENT_SEMAPHORE` gaps. It does not change the totals, only the attribution.

The Pool descriptor storm is unchanged and still the best-localised gap:
**768 Pool DGE transfers of shape `elem_size 4, num_elem (128,1), src_step (4,1) -> dst_step
(262144,1)`** — 512 B of HBM scattered as 4 B into each of 128 partitions, i.e. **128 descriptors
to move 512 bytes, 98304 descriptors (~25% of all ~400k in the program) for 0.4% of the bytes.**
At the measured ~4.8 ns/descriptor that is ~470 us of DMA descriptor time and it is what paces the
110.7 + 57.8 us of Pool-queue stalls. It cannot be merged (the destination runs are one per
partition, so the descriptor count is fixed by the layout) and it cannot be re-queued
(cross-engine DGE rings hang — iter-2 §4). **The only way out is for the data not to need a
cross-partition scatter**, i.e. load the 128 scalars contiguously into one partition (1
descriptor) and transpose on-chip (PE identity matmul, ~350 ns x 768 = 269 us of PE duration
against ~470 us of descriptor time). That is the biggest single structural inefficiency left and
nobody has priced it on the device.

## 4. Closed on the device this iteration — do not redo

### 4.1 ACT ACTIVATE merging is closed from *both* directions
iter-2 closed "merge the pair at the earlier slot i" (j's wait would move earlier -> deadlock
cycle; the sound rule leaves 0-1 pairs). I tried the other side: **merge at the later slot j and
turn slot i into an `EVENT_SEMAPHORE` carrying i's exact events** (`iter-3-mergej.py`). That form
is *synchronisation-free*: no wait moves earlier, no sem-4 increment moves, both slots keep their
own update, so every one of iter-2's four event rejections disappears. 1646 payload-identical
pairs with equal src/dst deltas exist within 12 slots.

**All of them are rejected by a real data hazard, and I measured why.** The only thing that moves
is i's *write*, so the test is "does anything read i's dst between i and j". Diagnostics on the
first pairs:

```
pair i=20 j=31  src=dst=0x1c000 (in place)  cum[i+1]=10 cum[j+1]=14
   PE slot 25 MATMUL   reads 0x1c000, no wait,          live inside the window
   DVE slot 79 TT_ARITH reads 0x1c000, waits sem4 >= 10 == cum[i+1]  <-- released by i itself
```

i.e. **the compiler emits each multiply with its own producer wait and has its output consumed by
PE/DVE immediately, gated on exactly that multiply's sem-4 increment.** Adding the sound
"provably ordered after j" test (`reader waits sem4 >= cum[j+1]`) rescues nothing. Yield: 0 pairs
at GUARD>=5000 ns, 46 at GUARD=1000, 765 only at GUARD=0 — and GUARD=0 means "no conflict in the
*current* schedule", which is not a safety argument. **The 612.8 us of critical path in the 2418
multiplies is structural. Stop attacking it.**

### 4.2 The mirror state bounce (save writes `X[c]` directly) is REFUTED
`max_abs_err` output_0 **2.863e-02**, cosine 0.998923, output_1 bit-exact. 771 pairs, all asserts
passing, so the addressing is right: `X[c]` is **still holding live data** at the time the save
fires, so writing it early clobbers it. Together with iter-2 §5d this brackets the problem:
*the write of `X[c]` may not move earlier, and the read of `X[c-1]` may only move later for the
pairs where nothing touches it in between* (§1b: 605 of 799 at GUARD=20000). §1a is the form that
needs no liveness argument at all; §1b/§1c are the form that pays, and they need the filter.

### 4.3 "Dead store" elimination is CLOSED — there is exactly 1 dead store, and the field name that hid it
`iter-3-dcefix.py`. A fixpoint dead-code analysis over per-partition SBUF byte ranges said 245
instructions write SBUF that nothing ever reads: 222 DVE `TENSOR_TENSOR_ARITH_OP`
(`op=ADD`, `src0 0x8098 + src1 0x8118 -> 0xe200`, 32 elem x 128 ch, all from
`ssd_block.py:550`), 22 DVE `MEMSET`, 1 ACT `COPY`; 44 us of DVE duration. Replacing them with
`EVENT_SEMAPHORE`s carrying their exact events gave output_0 `max_abs_err` 7.5e-01,
**cosine 0.102**. Adding an extended wait `sem6 >= cum6[k]` to each replacement (to test whether a
cheap `EVENT_SEMAPHORE` advances DVE's completion counter while the pipelined predecessor is still
writing — the DVE analogue of iter-1 §2's ACT overlap) gave an **identical** failure, so it was
not a timing artifact.

**It was a decoder bug, and this is the part to remember.** A DGE endpoint's variable is
`<pre>_start_addr_addr_var_var_id_lo` (+ `_hi`), guarded by
`<pre>_start_addr_addr_var_marker == 16`. There is **no** `<pre>_start_addr_addr_var_id` field, so
the natural-looking `d.get(pre+'_start_addr_addr_var_id') != 0` test returns `None != 0` and
**silently drops every DGE endpoint from the analysis.** With the right name the SBUF-side DGE
endpoints appear:

| engine / side | n | SBUF range |
|---|---|---|
| ACT src (state save) | 816 | 0x4200..0x37e80 |
| Pool src (writes output_0/output_1) | 816 | 0x4000..0x37780 |
| Pool dst (loads) | 1569 | 0x1c200..0x37f04 |
| SP dst (state reload) | 816 | 0xe000..0x37e80 |

and **the dead set collapses from 245 to 1** (the ACT prologue `COPY` at slot 2, 187 ns). Pool's
816 output transfers read 0xe200, which is why removing those 222 stores wrecked `output_0`.
The lever does not exist; the device was right.

Two consequences worth carrying forward:
* `iter-2-sbuf.py` has the same bug, so its hole table is only accidentally right. I re-derived
  the occupancy with the correct field and checked the shipped §1 buffer explicitly:
  **zero pre-existing DGE SBUF endpoints overlap 0xe780..0xff80**, so T(head)'s placement is
  sound. Re-run that check (`iter-3-cpath.py`'s sibling logic, or the snippet in §6) before
  putting anything else in an SBUF "hole".
* §4.1's merge census used the same buggy scan, i.e. it saw *fewer* readers than really exist.
  Adding readers can only reject more pairs, so §4.1's "0 candidates" conclusion stands and in
  fact gets stronger.
* DGE `TBL`-marker endpoints (marker 0x20) are **not** a blind spot: `def.json`'s `DGETable0`
  (var_id 10) has `list = [4, 5, 6, 1, 2, 7, 8]` = B, C, causal_mask, x, dt, output_0, output_1.
  **None of them is SB (var 0)**, so a TBL endpoint never touches SBUF.

### 4.4 Three cheap things that are simply not there
* **No redundant `LDWEIGHTS`**: 0 of 5441 have the same payload as the previous `LDWEIGHTS`
  (checked with only MATMUL/EVENT_SEMAPHORE allowed in between). PE reloads the array every time
  because the stationary operand really does change.
* **No mergeable activation-table regions**: all 13 `sel=1`/`sel=0` pairs are separated by gaps
  that contain `f=7` (exp), and set 1 (`reciprocal_and_small`) has function ids
  `{1,2,3,4,11,12,13,30,31,33,35,128,129,130}` — **no 7 and no 10**. So no pair can be dropped by
  leaving set 1 active across it, and no load is redundant (I walked the active-set sequence: 0
  loads re-select the already-active set).
* **`dma_configs_priority_class`** (a 1-bit field, 0 on all 4017 DGE triggers): setting it to 1 on
  Pool's 768 compute loads measured 3272.1 us and on those plus the 768 4-byte column loads
  3272.4 us, against 3274.0 — **1.6-1.9 us, i.e. 1-2x the 0.03-0.06% spread. Not separable from
  noise; not shipped.** Priority does not create descriptor bandwidth, which is what §3a says the
  Pool stall actually is.

## 5. Where to go next

**Start with #1: it is the direct continuation of §1b/§1c and it is cheap.**

1. **Finish the state bounce.** 211 of 816 pairs still do real work: 166 rejected by
   `GUARD=20000` in §1b, 28 whose save/reload shapes do not match (`save not SB->SB`), 17 whose
   reload waits outside sems 40-55. Each converted pair is worth about the same as the 605 already
   done, i.e. **~0.2 us of wall each, ~40 us for all of them (1.3%)**. Two ways in:
   * the 166 are rejected only by the *timestamp* gate, so a tighter liveness argument (which
     instruction actually touches `X[c-1]`, and can its wait be *raised* to sem 56+j — raising a
     wait is the monotone-safe direction iter-2 proved on PE) converts most of them without
     loosening GUARD. `iter-3-recyc.py` already prints the touchers: at margin 0 only 13 pairs
     have one at all, and they are 9 ACT `ACTIVATE` + 3 DVE + 1 Pool DGE.
   * the 28 + 17 need their own small cases; look at them before writing code.
   **Do not just lower GUARD** — the whole reason this is defensible is that the filter returns the
   same 605 pairs on both the 3274 us and the 3106 us schedules.
2. **The 4-byte Pool column loads (§3a).** 98304 descriptors, ~25% of all DMA descriptors,
   pacing 168 us of Pool-queue stalls (now 137.3 us of ACT `EVENT_SEMAPHORE` gap + 57 us of Pool).
   Replace the cross-partition scatter with a 1-descriptor contiguous load into one partition plus
   an on-chip transpose (PE identity matmul into PSUM, or DVE `STREAM_SHUFFLE`). PE is 22% of the
   path at 41% busy, so it has room. Highest remaining ceiling (>3%) and still unpriced. The
   consumer wants a column (it is an ACT per-partition `scale` pointer), so the transpose is
   mandatory, not optional.
3. **The full `state_sb` relayout** (iter-2 §5, option 2): `T(head)` already exists and is
   resident, so the remaining work is to repoint the 276 rotating pool addresses at `T(head)`
   *per unit by program position* and delete both transfers. After §1c the ACT side is only
   45.5 us of critical path, so the prize is now mostly SP's 816 reloads — and SP contributes
   **0 us** to the critical path. **This is no longer worth an iteration.**
4. **The 98% error-budget variant of §2** is sitting there for ~5 us if a future iteration decides
   the budget is spendable: `EXCL=5,6,7,8` in `iter-3-lastrecip.py` (11 recips, 16 table loads).
5. **The general shape of §1c is worth hunting elsewhere:** any DGE trigger whose transfer is
   redundant but whose `+16` is load-bearing is ~350 ns of pure engine time, recoverable with an
   `EVENT_SEMAPHORE` carrying `SEM_ADD_IMM_COMPLETE` (primary update) + the original wait
   (extended block). Pool has 2385 triggers and SP 816; check whether any of them move data
   nothing reads — with the **corrected** DGE decode of §4.3.
4. **Do not** re-derive: ACT ACTIVATE merging (§4.1, both directions), the two cheap state-bounce
   forms (§4.2 + iter-2 §5d), SBUF dead-store elimination (§4.3: exactly 1 dead store exists),
   LDWEIGHTS dedup, table-region merging, DMA priority, moving DGE triggers between engines
   (iter-2 §4), `POINTER_IMMEDIATE` sweeps (iter-1 §4), `scale_src` pointers (they are all real:
   the queued DMA list shows `dt -> SB 0x181C000` and `A -> SB 0x807F00`, which are exactly the
   `scale_value_imm_ptr` values the nch=1 classes use).

## 6. Tool notes (additions to iter-1's and iter-2's)

* **`SP0.json` has 1149 queued DMA transfers in this NEFF** (768 `dt -> SB`, 753 `A -> SB`), not
  an empty list. They only ever *write* SB, but any SBUF liveness or address analysis has to read
  them. `to_off` is `partition * 262144 + offset`.
* **A DGE endpoint's SBUF range** (the thing every liveness / hole argument needs, and the thing
  that was wrong in §4.3 and in `iter-2-sbuf.py`):
  ```python
  pre = key[:-len('_start_addr_addr_var_offset')]          # 'src' or 'dst'
  if d.get(pre+'_start_addr_addr_var_marker') != 16: continue     # 16 = VAR form
  if d.get(pre+'_start_addr_addr_var_var_id_lo') != 0: continue   # 0 = SB;  NOT '_var_id'
  ne, st, es = d[pre+'_num_elem'], d[pre+'_step_elem'], d[pre+'_elem_size']
  ext = sum((max(n,1)-1)*t for n,t in zip(ne,st) if abs(t) < 0x40000)   # drop the partition sweep
  lo  = d[key] & 0xFFFFF ; hi = lo + ext + es
  ```
  Marker 0x20 (`TBL`) endpoints never touch SBUF in this NEFF (§4.3).
* `prof/SemaphoreUpdate.parquet` is the key to a real critical path (§3). `prof/Flow.parquet` has
  106898 data-flow edges but no semaphore edges.
* `x.set(...)` mutates the `Instruction` object in place, so `new[i].encode() != A[i].encode()` is
  **always false** after an edit — capture `as_dict` / raw bytes *before* editing and assert
  against those (every script here does; the one that did not reported "0 slots changed" on an
  edit that had in fact applied).
* `dma_configs_priority_class`, `dma_configs_dma_transpose_2d_tile` and
  `dma_configs_config_reg_enable` are 1-bit fields, all 0 on all 4017 DGE triggers.
* `EVENT_SEMAPHORE` (`CTRL_ES`) has a **second, extended** event block
  (`events_extended_wait_mode/_wait_idx/_sem_wait_value` and `..._update_mode/_update_idx/
  _sem_update_value`). The compiler uses the extended **wait** 138 times in this NEFF (ACT 112,
  SP 17, Pool 7, DVE 2). **`SEM_ADD_IMM_COMPLETE` is now VERIFIED on the device** (§1c): put it in
  the *primary* update with the addend in `events_semaphore_value`, and move the original wait to
  the extended block, because `events_semaphore_value` doubles as the primary wait threshold.
  `semaphores.py` shows the result as a second source on that semaphore
  (`{dge:Activation: 210, Activation:EVENT_SEMAPHORE: 606}`) with the total unchanged.
* **A DGE trigger costs ~390 ns of issuing-engine time plus ~2.2 ns/descriptor** (fit over ACT
  64-desc 530 ns, SP 64-desc 522 ns, Pool 128-desc 655 ns, and confirmed by §1b: cutting 64
  descriptors to 1 changed ACT busy by 0.8 us over 605 instructions). So shrinking a transfer
  helps the DMA engine, not the issuer; only *deleting* the trigger helps the issuer.
* ACT's 816 DGE triggers carry **no event-block update** — 15 have no events at all and the rest
  carry exactly one `WAIT_FOR_SEM_GE_IMM` ring throttle on their own queue semaphore. The `+16`
  comes from the DGE `semaphore`/`sem_increment` fields, not from the event block.
* Device-run costs today: `verify.py` ~35 s, `--runs 4` ~70 s, `profile.py --repeats 3` ~45 s,
  `--repeats 5` ~75 s. Spread with 5 repeats was 0.05-0.06% (~2 us), with 3 repeats 0.14-0.19%.
  **Use 5 repeats before believing anything under 10 us.**
* Reproduce the shipped NEFF from iter-2's (md5 `4bd85780dffd61c129f42dcd1528309f` ->
  `48972372faa61507c7b22203d7887d14`):
  ```bash
  neff.py unpack kernel.neff work_cur/ ; rm -rf work; cp -r work_cur work
  profile.py --repeats 1 --parquet prof              # guard_psum + the §1b liveness filter need it
  python scripts/bounce3.py                          # §1a  (771 pairs -> T(head) = 0xe780+h*128)
  EXCL=5,6,7,8,9,10,11 python scripts/lastrecip.py   # §2   (8 recips, 10 table loads)
  KILL=1 python scripts/single.py                    # §1b+§1c (605 pairs, GUARD=20000)
  neff.py pack work/ kernel.neff
  ```
  `single.py` must run **after** `bounce3.py` (it looks for staged SB->SB pairs) and it reads the
  streams from `work/`, so keep a pristine `work_cur/` and restore from it between attempts.

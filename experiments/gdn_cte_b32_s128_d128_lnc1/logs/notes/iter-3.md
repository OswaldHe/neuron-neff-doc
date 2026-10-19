# iter-3 — built the critical-path decomposition of the post-iter-2 NEFF, then **measured five
# levers closed**. Net: no shippable win; the engine-issue-time era is over.

## Result

Starting NEFF (= iter-2 shipped): re-measured **1207.688 us** (samples 1206.4–1211.0, spread
0.39%). `baseline/` iter-0 = 1351.4.

| experiment | verify | total_exec_time_us | verdict |
|---|---|---|---|
| start (iter-2 NEFF) | PASS | **1207.688** | reference |
| +224 merged ACT CAST pairs (−10.0 us ACT busy) | PASS, errors bit-identical | **1207.398** | **0% conversion — refuted** |
| +DVE CAST/COPY/MEMSET merges (25) | **HANG** (NRT status 5) | — | see §3b |
| +DVE TENSOR_TENSOR merges (65) | **FAIL** max_abs_err 9.269e-03 (843%) | — | overlapping in-place regions, §3a |
| PSUM-reading queued DMA descriptor (2 encodings) | **NRT_INVALID at load** | — | §4.4, closed |
| control probes: 6th DMA queue, 0x4000 partition step | PASS | — | §4.4, two new verified facts |

So the iteration ships **nothing** (reverted to the iter-2 NEFF, md5 `3088e953…`, byte-identical
to HEAD; `verify.py` and `launch_rules.py` re-confirmed PASS on the restored file). What it
produced is the measurement that kills **five** cheap-looking levers, plus a full critical-path
decomposition and a PSUM-evacuation census to aim the next one.
Read §2 and §5 before choosing anything.

**Noise floor, measured twice on the identical NEFF this session: 1207.688 and 1205.745 us
(5 repeats each, spreads 0.39% and 0.50%).** So the between-measurement floor is ~±2–3 us, not
the ±1 us a single spread figure suggests. **Do not claim anything under ~8 us without two
independent `--repeats 5` runs of each NEFF.** This is why the ACT merge's 0.3 us was not a
result, and why candidates worth <15 us of predicted wall time are not worth a device cycle.

---

## 1. THE NUMBER THAT MATTERS: the critical path is now 65% real instruction execution

Backward dependency walk on the current NEFF (recipe in iter-1 §1; my version is
`scratch/walk.py`, it needs `scratch/graph.py` first). 4111 nodes, path ends at the last
instruction and runs back to `(Pool, 0)`, i.e. it spans the whole run.

| component of the 1222.6 us span | us | share |
|---|---|---|
| **instruction execution on the path** (real work) | **795** | 65% |
| DMA transfer latency on the path (trigger→completion, ~1.04 us × ~220 hops) | ~220 | 18% |
| semaphore-wait blocking charged to `EVENT_SEMAPHORE` slots on the path | ~105 | 9% |
| inter-node gaps (dispatch, propagation) | 98 | 8% |

Path execution by engine and opcode (us on path / us of engine busy):

| engine | opcode | on path | engine total | path share |
|---|---|---|---|---|
| **DVE** | TENSOR_TENSOR | 147.6 | 252.6 | 58% |
| | CAST | 95.7 | 143.8 | 67% |
| | COPY | **82.7** | 128.5 | 64% |
| | MEMSET | 15.4 | 39.3 | 39% |
| | **DVE total** | **348** | 534 | 65% |
| **PE** | MATMUL | 150.5 | 540 (overlapped) | — |
| | LDWEIGHTS | 60.6 | 208 | — |
| | **PE total** | **211** | 340 | — |
| **ACT** | TENSOR_SCALAR | 93.9 | 192.3 | 49% |
| | ACTIVATE | 62.6 | 189.1 | 33% |
| | CAST | **42.8** | 185.9 | **23%** |
| | **ACT total** | **199** | 520 | 38% |
| Pool | TENSOR_SCALAR_AFFINE_SELECT + DGE | 24 | 234 | 10% |
| SP | WRITE (trigger issue) | 13 | 52 | — |

**The single most useful fact here: the path share differs 3x between opcodes.** That is exactly
why iter-2's "ACT issue converts at ~40%" did not generalise (§2.1).

Binding-gate attribution of the 98 us of pure gaps: `prog:DVE` 22.9 (n=1002), `S[6]` 20.4 (560),
`prog:SP` 11.5 (213), `S[5]` 11.3 (488), `S[4]` 9.4 (247), `prog:ACT` 8.4 (476),
`dma:qSPSpillReload0` 6.5 (185), `prog:PE` 2.7 (767), `S[3]` 1.9 (80). Nothing is concentrated.

### 1b. The PE/DMA gap split is stable (same as iter-2, re-measured)
147 PE gaps >1 us gated on `S[24..39]`, **471.4 us** total:
producer wait **205.7 us (44%)**, transfer latency **191.8 us (41%)**, propagation **73.8 (16%)**.

---

## 2. REFUTED THIS ITERATION (each with the measurement that closed it)

### 2.1 Instruction merging on ACT/DVE — mechanism works, payoff is ~0. CLOSED.
I generalised iter-1 §4.2 ("merge consecutive identical slots", which found only 57 slots) to
**affine composition**: two consecutive slots merge into one if every operand's start address
differs by a constant and a free pattern dimension exists, regardless of waits or signalling.
That found **455 removable ACT slots and 580 on DVE** on paper. After the addressing and
dependency filters: **224 ACT merges + 135 DVE merges**, ≈38 us of engine issue time.

**Measured: 224 ACT merges = −10.0 us of ACT busy (520.5 → 510.5) and −0.3 us of wall time
(1207.688 → 1207.398, inside a 0.39% spread). Conversion 0%.**

Why, and this is the reusable lesson: ACT `CAST` is only **23% on the critical path**, and a
merge removes only the *fixed* part of a slot (ACT CAST = 134 ns fixed + 1.26 ns/elem; DVE CAST
165 + 0.36; DVE COPY 145 + 1.0; DVE TT 143 + 1.04 — all measured, see §4.1). So 224 merges
removed 10 us of busy, ~2.3 us of path, ~0 us of wall.
**Rule to replace iter-2's "issue time converts at 40%": multiply the issue time you remove by
that opcode's path share (§1), then expect ~1:1 on the result.** iter-2's win was 121 us of ACT
DGE issue at a high path share; there is no comparable block left.
DVE would be the better target (60–67% path share) but the whole available DVE merge set is only
~21 us of busy ⇒ ~13 us of path ⇒ under 1%, and it deadlocks (§3).

### 2.2 PE's waits on the queued-DMA semaphores are **precise**, not conservative. CLOSED.
Hypothesis: the compiler emits coarse ring-based waits (iter-2 found 218 of 260 *trigger* waits
were stale ring throttles), so PE's `S[24..39]` waits might name a later transfer than PE needs;
lowering the value would start PE earlier. This is the only cheap way at the 471 us of PE gap.
Method: expand every `qSPSpillReload0` transfer's destination to (partition, byte-interval) cells
from `DmaPacketAggregated.write_{offset,steps,shape,sbuf_partitions}`, expand each PE
instruction's SBUF source operands the same way, and ask whether the PE slot reads what the
transfer it waits on writes. Wait value 16k on `S[24+r]` ⇒ `block_id = (k-1)*16 + r`.
**Result: 327 of 329 PE compute waits overlap the named transfer's destination (LDWEIGHTS 295/297
carrying 427.6 us of gap, MATMUL 32/32).** The two non-overlaps carry 0 us of gap. There is no
slack to reclaim. (`scratch/pewait.py`)
Corollary: the 274 PE `EVENT_SEMAPHORE` pre-waits (45.4 us of gap) are *not* redundant either —
removing a wait from a PE ES cannot start anything sooner, because PE is in-order and the
instruction behind it carries the same or a later condition.

### 2.3 `fabric_path` / queue type is not a knob. CLOSED (static).
The 717 ns flat trigger→start cost is ~133 us of the path. `fabric_path` is `"main"` in **all 116
dma_queue entries of the 14-NEFF corpus** — there is no alternative value to try. `type` is
`"data"` for the two SpillReload queues and `"dynamic"` for the DGE ones; a DGE hop was measured
*slower* (iter-1: 1500–1714 ns). Nothing to change.

### 2.4 `SET_ORDERING_MODE` is already `ordering_mode=0` (relaxed) on every engine. No lever.

### 2.5 A profiler artifact to not fall for (this corrects `.autohelix/review.md` §3)
The review says iter-2's fused transfers take "17 to 24 us for casts, growing to about 150 us".
That is wrong. In `DmaPacketAggregated`, `qSyncSpillReload1` rows have
**`start_ts` ~77 us *before* `trigger`** (median `start_ts - trigger` = −77286 ns), so the
`duration_ns` column (end−start) is garbage for that queue: median 78 us for `copy`, 98 us for
`cast`. The real figure is **`end_ts - trigger` = 915 ns median, max 2.3 us** — the fused
transfers are as fast as any other, and the `op:"cast"` ones are not slow.
**Always use `end_ts - trigger`, never `duration_ns`, for a queued transfer's latency.**
(`qSPSpillReload0` for comparison: `end-trigger` 1037 ns median, `start-trigger` 717 ns,
`duration` 312 ns.) Review candidate #3 ("turn the 32 cast transfers into plain copies") is
therefore dead — there is nothing to recover.

### 2.6 Collapsing the ACT `copy`→`exp` ACTIVATE chains — CLOSED, only 2 candidates.
`activation_func` ids here are **128 = `copy_1p` (identity) ×320** and **7 = `exp_400p` ×256**
(`docs/generated/activation_functions.md`; the profiler mislabels 7 as `SOFTPLUS` — trust the
table). `ACTIVATE` computes `out = f(scale*in + bias)` with `scale_src` either an instruction
immediate or a per-partition `POINTER_IMMEDIATE`. So `copy(s1*X+b1)` followed by `exp(1.0*Y+0)`
collapses into one `exp(s1*X+b1)` whenever the second's scale is exactly 1.0 and both biases are
0 — which they are. 576 ACTIVATEs at 292–397 ns = 189 us of ACT busy looked like ~86 us of
removable issue.
**It is not there: of the 320 `copy` ACTIVATEs only 60 are followed by an `exp` within 8 slots,
and 153 are followed by an `exp` that does not read the copy's destination at all. After requiring
a single reader of the intermediate, 2 pairs survive.** The `copy` ACTIVATEs are not stages of a
chain; they are standalone per-partition scales (many with `num_elem = [1,1,1]`, i.e. **one
element over 64 partitions for 292 ns** — pure fixed cost, nothing to merge into).
(`scratch/chain2.py`)

### 2.7 Review candidate #1 (the WAR hole) — not attempted; see §5.

---

## 3. Why DVE merges break and ACT merges do not — two separate causes

`ACT`: 224 merges PASS, bit-identical. `DVE` fails in **two different ways**:

### 3a. DVE `TENSOR_TENSOR` merges give the WRONG ANSWER — I forgot a disjointness check
65 TT merges: runs, verify PASS, `max_abs_err` **9.269e-03 = 843% of the limit**.
Cause: these are in-place accumulations — `op=ADD src0=fp32@0x837700 src1=fp32@PSUM dst=fp32@0x837700`,
i.e. `X += psum`, with **`src0` and `dst` the same address**. Consecutive instances step the
address by less than one block's extent, so block *i+1* reads what block *i* wrote. Sequentially
that is defined; inside one merged instruction the write/read order across the added dimension is
not the same, so the result changes.
**Any merge recipe must require that the L blocks' regions are pairwise disjoint** — i.e.
`|delta| >= extent_bytes` for every operand, *and* no block's dst overlaps another block's src.
I only checked "delta != 0 on dst". ACT CAST happened to satisfy disjointness exactly
(32 bf16 elements = 64 bytes of dst with a 64-byte delta; 128-byte fp32 src with a 2048-byte
delta), which is why ACT passed and DVE did not. Add the check before trusting any merge.

### 3b. The other DVE subsets HANG — the dependency proof is not sufficient on DVE
The 25-merge DVE set (CAST/COPY/MEMSET, strictest filter, only instruction forms that already
occur in the NEFF, every semaphore count byte-identical) **hangs** with
`Failed to execute model: 10001, error: 5` + `Model stop timeout (30000 ms)`.

My acyclicity proof is sound for ACT and insufficient for DVE. The proof: build the whole
dependency graph (nodes = (engine, slot); edges = engine program order + every wait's producer),
compute `dep4[X]`/`dep6[X]` = the greatest ACT/DVE slot index X transitively depends on, by
**fixpoint iteration** over nodes sorted by baseline `start_ts` (3 iterations converge; a single
pass leaves 153 forward references — if you poison those with +inf, the poison propagates and
every candidate is rejected, which cost me one wrong "0 merges" answer). Reject a merge of
`[s, s+L)` if any wait's producer `P` has `dep[P] >= s`.
That is exactly right for ACT. For DVE it is not sufficient, and the most likely hole is that
**`dep6` is under-estimated for the nodes that wait on `S[8..23]`**: those semaphores are fed by
*both* the 288 Pool DGE transfers (+16 at transfer completion, out of program order) and 32 Pool
`EVENT_SEMAPHORE` relays (+16 at slot execution), so my "cumulative value in Pool program order"
producer map can name the wrong producer. DVE merges wait on `S[3]` (Pool) and `S[4]` (ACT) as
well as `S[5]`/`S[6]`; ACT merges wait **only on `S[5]`**, which has a clean 1-increment-per-
MATMUL map — which is why ACT was safe and DVE was not.
**If you ever need this again:** fix the `S[8..23]` producer map first (order Pool DGE completions
by profile timestamp, not program order), or just never relax a wait whose producer chain passes
through `S[8..23]`.

Two mechanism facts that ARE now device-verified (both inside the passing ACT edit):
* **An `EVENT_SEMAPHORE` can increment an engine-completion semaphore.** 11 ACT ES slots with
  `events_update_mode = SEM_ADD_IMM_COMPLETE, events_update_idx = 4, events_semaphore_value = 1`
  kept `S[4]`'s total at exactly 1306 and the kernel ran correctly. iter-2 §2.8's "ACT ES does not
  increment S[4]" is about the *implicit* increment only; an explicit update works on S[4] too.
  (Only slots with `events_update_mode != NONE` ever increment anything — confirmed again here:
  ACT has 2532 slots, 1306 `SEM_INC_COMPLETE[4]`, and 896 CASTs of which only 554 signal.)
* **Merging two compute slots into one is bit-exact**, as expected: `max_abs_err` stayed
  1.097e-04 / 9.081e-04 to every printed digit over the 224-merge ACT edit.

### 3c. The merge recipe, if a future iteration wants it (it is written and tested)
`scratch/{graph,apply}.py` (reverted with the worktree; ~250 lines total, ~45 min to rewrite from
this description).
* Candidate = maximal run of consecutive slots whose **every decoded field except the operand
  start-address immediates** is equal (you must exclude the whole `*_start_addr_*` union — the
  `addr_reg_regnum` / `addr_reg_reserved0` fields overlap the immediate and differ for free;
  keep only `*_marker` in the key).
* Merge = pick the lowest dim `d >= 1` with `num_elem[d] == 1` on every operand, set
  `num_elem[d] = L` and `step_elem[d] = (addr_{i+1} - addr_i) / elem_size`. Element size comes
  most reliably from the profiler's `Instruction.operands` string (`src=fp32@0x...`), because
  several structs encode the input dtype as split `*_dtype_lo`/`_hi` fields.
* **`step_elem` is signed 16-bit** (`width: 2, signed: True`): this alone kills 59 of 103 ACT runs
  and 344 of 425 DVE runs, because the address deltas are 42k–2.1M *elements* (same partition,
  far offset, or a partition crossing that is not a clean multiple of 0x40000). Not fixable.
* Layout: **merged op at the FIRST slot of the group**, carrying the max of the group's waits
  (compute structs have **no extended events block** — one wait slot only; extra wait semaphores
  need a pure-wait ES placed *before* the merged op) and the group's first update; the remaining
  slots become `EVENT_SEMAPHORE`s carrying the remaining updates in order. This is
  monotonically conservative: every `S[4]`/`S[6]` ordinal is reached no earlier than before, and
  all of the group's outputs are complete when the first of them is signalled.
* Use only instruction forms that already occur in the NEFF: there is **no fully empty ES**
  anywhere in this NEFF (every one has a wait or an update), so give an updateless placeholder the
  merged op's (already-satisfied) wait, and give an update-carrying one the iter-2 form
  (extended wait + primary `SEM_ADD_IMM_COMPLETE`).

---

## 4. Measurements worth keeping

### 4.1 Per-slot cost model (median ns, from `Instruction.parquet` joined on `compiler_pc`)
`cost = fixed + slope * num_elem` (independent of `num_active_channels`):

| engine | opcode | fixed ns | ns/elem | 32 el | 128 el |
|---|---|---|---|---|---|
| ACT | CAST | 134 | 1.26 | 174 | 295 |
| ACT | TENSOR_SCALAR | ~330 | 0.4 | — | 380 |
| ACT | ACTIVATE | ~290 | 0.8 | — | 397 |
| DVE | CAST | 165 | 0.36 | — | 211 |
| DVE | COPY | 145 | 1.0 | — | 273 |
| DVE | TENSOR_TENSOR | 143 | 1.04 | 176 | 276 |
| DVE | MEMSET | ~95 | 0.55 | — | 165 |
| ACT/DVE/SP/Pool | EVENT_SEMAPHORE (non-blocking) | 40 / 24 / 32 / 19 | — | — | — |
| Pool | DMA_DIRECT2D (DGE issue) | 687 | — | — | — |

So **the per-slot fixed cost is only 130–330 ns and most ops here are 32–128 elements**: there is
no fat inside a slot, and removing whole slots buys only the fixed part.

### 4.2 Engine busy on the current NEFF (1207.7 us)
DVE 534.2 (44.2%), ACT 520.5 (43.1%), DMA 340.7 (28.2%), PE 340.4 (28.2%), Pool 233.8 (19.4%),
SP 51.7 (4.3%). Note `Instruction.duration_ns` **overlaps within an engine** (PE: MATMUL 540 us +
LDWEIGHTS 208 us against 340 us of busy), so never sum durations to get busy time.

### 4.3 THE BIGGEST COHERENT BLOCK OF WORK IN THE KERNEL: PSUM evacuation
Every compute instruction whose **source is PSUM** (`start_addr >= 0x2000000`), i.e. moving a
matmul result out of PSUM into SBUF. 2272 MATMULs produce them; 1056 instructions evacuate them.

| engine | opcode | n | engine busy | path share (§1) | ⇒ us on path |
|---|---|---|---|---|---|
| ACT | CAST (PSUM→SBUF, fp32→bf16, 32 el × 32 ch) | 640 | 110.4 us | 23% | ~25 |
| DVE | COPY (PSUM→SBUF, same pattern, same dtype) | 288 | 76.2 us | 64% | ~49 |
| ACT | TENSOR_SCALAR (PSUM→SBUF) | 128 | 38.8 us | 49% | ~19 |
| | **total** | **1056** | **225.4 us** | | **~93 us** |

For completeness the non-PSUM halves: ACT CAST 256 / 75.5 us, ACT TS 416 / 153.5 us,
ACT ACTIVATE 576 / 189.1 us, DVE CAST 672 / 143.8 us, DVE COPY 229 / 52.2 us (all SBUF→SBUF
with identical src/dst patterns and dtypes, i.e. pure relocations), DVE TS 64 / 12.5 us,
Pool TSAS 131 / 31.4 us.

**225 us of engine busy and ~93 us of critical path doing nothing but moving data out of PSUM** is
the prize, and the DMA engine is only 28% busy. **I probed whether a queued DMA descriptor can
read PSUM. It cannot — see §4.4. That closes iter-2 §5.1 route (b).**
*Do not* instead move the evacuation to Pool (Pool is 19% busy and only 24 us on the path): the
consumer would have to be released by a Pool→DVE relay, which adds ~0.5 us of semaphore
propagation per instance against the 0.265 us of DVE issue it removes — a net loss of ~68 us
unless you also retarget every consumer's `S[6]` wait, which is positional and covers more than
the one COPY.

### 4.4 **A queued DMA descriptor cannot address PSUM.** CLOSED on the device, 5 runs.
First, the flat TPB address map, from the vendor constants
(`artifacts/reference/hw_constants_v3.json`) — **get these right before guessing**:

| constant | value |
|---|---|
| `STATE_BUF_OFFSET` / `PARTITION_SIZE` / `SZ` | 0 / **0x40000** / **0x2000000** |
| `PSUM_BUF_OFFSET` / `PARTITION_SIZE` / `ACTIVE_SIZE` / `SZ` | **0x2000000** / **0x8000** / 0x4000 / 0x400000 |

So PSUM's flat address is `0x2000000 + partition*0x8000 + offset` — **the partition stride is
0x8000, not the 0x4000 that chapter 5.3's "16 KiB per partition" suggests** (0x4000 is the
*active* size inside a 0x8000 window). The compute-side form `0x2000000 + offset` simply omits the
partition term because each partition's engine sees only its own PSUM.

Probe harness (cheap, reusable, and it works — build it this way): add a **6th** DMA queue to
`def.json`, append **one** transfer with `id: 0` to `SP0.json` on it, and insert **one**
`PSEUDO_DMATRIGGER` into `SP0.bin` after the last SP slot whose profile `start_ts` precedes the
instruction you are shadowing (iter-2 §2.8's acyclicity rule). Because transfer ids are per queue,
nothing existing is renumbered; because no SP slot increments a semaphore, inserting the slot
shifts nothing. `scratch/psum{2,3,4}.py`.

| probe | descriptor | result |
|---|---|---|
| control: 4-byte SB→SB self-copy on the new queue | `from:"SB"`, legal off, step 262144 | **loads, runs, outputs bit-identical** |
| **non-standard partition step** | `from:"SB"`, legal off, `from_steps [1,1,0x4000]` | **loads, runs** (and changes output_0 to 1.513e-04 exactly as a wrong read should ⇒ the transfer really executed) |
| PSUM via an SB offset above the state buffer | `from:"SB"`, `from_off = 0x2000000 + 0x800`, step 262144 | **`NRT_INVALID in nrt_load_util`** — rejected at load |
| PSUM via its own space name | `from:"PSUM"`, `from_off = 96*0x4000 + 0x800` | **`NRT_INVALID in nrt_load_util`** — rejected at load |
| **PSUM via the vendor flat map, in bounds** | `from:"SB"`, `from_off = 0x2000000 + 96*0x8000 + 0x800`, `from_steps [1,1,0x8000]` (extent ends at 0x23F8200, inside PSUM's 0x2000000–0x2400000) | **`NRT_INVALID in nrt_load_util`** — rejected at load |

So the runtime hard-bounds an `SB` offset to the state buffer (`< 0x2000000`) regardless of stride
or extent, and knows no `PSUM` space. Both of iter-2 §5.1's hopes are dead — and note the failure
is at **load**, not execution, so each probe costs only ~12 s and tells you nothing about the
hardware, only about the descriptor validator.
Still untested: whether a **DGE** DMA instruction can read PSUM (its 8-byte address is a union
with `*_var_marker`/`*_table_marker`/`*_reg_marker` — a PSUM marker may exist, and the compute
side uses marker 2 for PSUM in its 4-byte address). Worth one probe, but note the host would have
to be Pool and the consumer release would need a Pool→DVE relay, i.e. the same net loss as above.

**Two new verified facts from this probe, worth more than the probe itself:**
* **A 6th DMA queue works: 96 rings load and run.** iter-2 left "96 rings untested"; the corpus
  maximum is 5 queues / 80 rings. Adding a queue with free semaphores remains the cheapest way to
  add DMA work without renumbering anything.
* **A queued descriptor's SB-side partition step does not have to be a multiple of 262144.**
  `from_steps [1,1,0x4000]` loaded and ran. The "one partition-crossing dimension" limit that
  iter-1 §4.1 found is a **DGE** limit; the queued path accepts arbitrary strides.

### 4.5 Pool is 77% busy in the first 100 us window and 12–23% after
Per-100-us busy (%): GpSimd `77 39 18 20 19 17 18 23 16 15 12 12`; Tensor `59 70 61 65 69 63 65
69 57 70 69 67`; Scalar `52 49 56 ...`; Vector `51 51 45 ...`; Sync `25 17 18 ...`.
48 Pool `DMA_DIRECT2D` (34.5 us of issue) land in the first 60 us — the input load. This is the
only place any engine comes near saturation. But Pool contributes only **24 us to the critical
path in total** (4.7 us of it DGE issue), so the front-load is *not* what delays the start; PE is
already at 59% in window 0. Pool remains a non-candidate (confirms iter-2 §5).

---

## 5. What to try next, ranked by what §1 says is actually there

The budget is now explicit: **795 us of real instruction execution on the critical path, 220 us of
DMA latency, 105 us of semaphore blocking, 98 us of gaps.** Only the first two are big, and the
first one is arithmetic.

1. **PSUM evacuation — 225 us of engine busy, ~93 us of critical path (§4.3).** By far the biggest
   coherent block left. The DMA route is now **closed** (§4.4), so what remains inside it is
   redundancy hunting with no new mechanism: of the 288 DVE PSUM→SBUF COPYs and the 640 ACT
   PSUM→SBUF CASTs, are there pairs that evacuate the **same** PSUM bank to the same destination,
   or a CAST whose fp32→bf16 result duplicates a COPY's fp32 copy of the same bank? Needs the SBUF
   interval map; I did not check it. The one remaining mechanism probe is a **DGE** PSUM read
   (§4.4), which is cheap but whose payoff is doubtful.
2. **DVE `COPY`, SBUF→SBUF half: 229 instructions, 52.2 us, all pure relocations** (src and dst
   access patterns and dtypes identical — verified, `scratch/copies.py`). For each, if the dst's
   only reader could read the src instead, the COPY dies for ~0 risk to numerics. Needs the exact
   SBUF interval map, not the start-address census I used (my census says every dst has 2–157
   "readers" because many instructions share a start address — it is useless at this resolution).
   iter-2 §4 bullet 5 documents exactly how to build the real map. Ceiling ~52 us of busy,
   ~33 us of path.
3. **Reduce the number of DMA hops PE waits for (220 us of path, 192 us of it `qSPSpillReload0`).**
   Hop latency is irreducible (§2.3) and the waits are precise (§2.2), so the only route is fewer
   hops: find pairs of transfers that are *adjacent on the same ring* and both on the path, give
   the first one the second's descriptors as well, make the second a no-op transfer, and retarget
   the second's consumers to the first's semaphore value. Saves ~1.04 us per fused pair. Needs the
   ring-adjacency + path-membership census first (10 min with `scratch/split.py` + `pewait.py`).
   Do not attempt without counting the candidates first.
4. **Do NOT spend budget on:** everything in iter-1 §5.4 and iter-2 §5.5, plus everything in §2
   above — ACT/DVE instruction merging (0% conversion, measured), PE DMA-wait relaxation (waits
   are precise, measured), `fabric_path`/queue type (no alternative exists), ordering mode,
   the `op:"cast"` transfers (not slow — profiler artifact), the ACTIVATE `copy`→`exp` collapse
   (2 candidates), Pool front-load and moving work to Pool (24 us of path total; the relay costs
   more than the work it moves).
   Also: **do not chase ACT issue time at all any more.** ACT is 43% busy, 38% of it on the path,
   and its three big opcodes have path shares of 23% (CAST), 33% (ACTIVATE) and 49% (TS). The
   iter-2-era assumption that engine busy time is fungible with wall time is dead; use §1's path
   shares.
5. **The WAR hole the review flagged (its candidate #1) is still open and still unmeasured.** I did
   not touch it. iter-2 §6 argues the fused read finishes no later than the save's read did, with
   p10 2.94 us / min 1.25 us of margin to the next write; the review wants the 9 non-SP waiters on
   `S[40..55]` retargeted. It is a zero-gain correctness edit, so fold it into whatever iteration
   touches those transfers next rather than spending a whole iteration on it.

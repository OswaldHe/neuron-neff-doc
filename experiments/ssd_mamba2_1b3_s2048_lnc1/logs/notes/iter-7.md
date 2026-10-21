# iter-7 (ssd_mamba2_1b3_s2048): **the big one is a single unused instruction field.**
# Every pointer-scale ACT `ACTIVATE` told the hardware to fetch the activation function's `imm`
# operand from SBUF address 0, which f=128 never reads. Turning that fetch off is **bit-exact and
# worth -8.4%**. Plus three provably-sound slot-deletion passes (dead `EVENT_SEMAPHORE`s and
# `LOAD_MASK_SELECT`s).

**Shipped: 4682.7 -> 4251.2 us (-9.21%). Against iter-0: 7154.9 -> 4251.2, -40.6% (1.683x).
`verify.py` VERIFIED, `max_abs_err` 5.234e-07 / 3.334e-07 — byte-identical to iter-5/iter-6, every
edit is bit-exact. Members changed: `Activation0.bin`, `DVE0.bin`, `PE0.bin`, `Pool0.bin`.
md5 `1c7e8d3000004f1960e36e1be2baf331`. Spread 0.04% (1.6 us) over 5 repeats.**

| step | median us | delta | ACT busy | DVE busy |
|---|---|---|---|---|
| iter-6 (control, 3 rep) | 4682.671 | - | 3112.4 | 2177.0 |
| §1 delete 3119 dead `EVENT_SEMAPHORE` slots (3 rep) | 4645.270 | **-37.4** | 3112.5 | 2167.4 |
| §2 delete 3077 dead `LOAD_MASK_SELECT` slots (3 rep) | 4638.559 | **-6.7** | 3111.0 | 2138.5 |
| §3 + 208 more ES (backward subsumption, 5 rep) | 4638.781 | ~0 | 3110.8 | 2140.1 |
| **§0 `imm_src = INSTRUCTION_IMMEDIATE` on 6157 ACTIVATEs (5 rep, shipped)** | **4251.238** | **-387.5** | **2549.3** | 2133.4 |

## READ THESE FIVE FIRST

0. **§0 — AUDIT EVERY ENUM FIELD OF THE HOT INSTRUCTION FOR A CONFIGURED-BUT-UNUSED OPERAND.**
   All 6157 pointer-scale `ACTIVATE`s carried `imm_src = POINTER_IMMEDIATE` with
   `imm_imm_ptr = 0`. The output was already correct, so `activation_func = 128` provably never
   reads that operand — yet ACT still paid **91 ns per instruction** fetching it. One field flip
   (`imm_src -> INSTRUCTION_IMMEDIATE`, the immediate is already 0.0) is bit-identical and removed
   **561 us of ACT busy -> 387.5 us of wall (-8.4%)**. Seven iterations of instruction-level
   surgery had each bought 0.4-5%; this was a 6-line script. **The generalisable move: for each
   high-count opcode, list every enum field's value distribution and ask whether the expensive
   value is observably needed** (`iter-7-ptrscan.py`, `iter-7-actclasses.py`). The remaining
   `POINTER_IMMEDIATE` fields in this NEFF are now all genuinely used (§0b).
1. **§1 — deleting a dead slot is worth ~12 ns of wall on DVE, and the stream can simply be
   compacted.** 3119 pure no-op `EVENT_SEMAPHORE`s (wait NONE + update NONE, mostly iter-6's
   retired `STREAM_SHUFFLE`s) deleted outright: -37.4 us for only ~10 us of measured busy time, so
   **the win is not in the reported duration** — it is in whatever per-slot cost the profiler does
   not attribute (fetch/issue). Price slot deletion by slot count, not by busy time.
2. **§2 — `STREAM_SHUFFLE`'s mask register is sticky, and 3077 of the 8192 `LOAD_MASK_SELECT`s
   were already dead.** All 8192 LMS payloads are byte-identical; every shuffle has its own LMS
   immediately in front of it; therefore an LMS whose next slot is not a shuffle has no reader.
   Deleting them is bit-exact (confirmed on device) — worth only -6.7 us (2.2 ns/slot, LMS are
   5 ns instructions) but it closes the question.
3. **§4 — the two 57.9 us + 32.1 us *global* stalls are real, reproducible, and only 23 us of them
   is fixable by reordering SP.** Every engine idles together at t≈3.22 ms and t≈4.16 ms. Root
   cause is **DMA descriptor-rate saturation**: `PendingDma` is 830 outstanding normally and
   2316/2973 in the two windows, and **48% of all 165881 DMA packets move 4 bytes** — they are the
   `dt_col` load, 128 four-byte scatters per unit, 0.4% of the bytes and 687 us of DMA-engine time.
4. **§5 — the whole "DMA arrives too late" family is worth 165 us (3.5%), measured.** Nulling
   every wait on a DMA-completion semaphore (sems 8..71) on every engine gives 4638.6 -> 4474.0.
   `iter-7-probe-dma.py` is the probe; re-run it to re-price the family after any edit.
   **It is pure latency, not DMA throughput: shrinking the 1024 `dt_col` DMAs to one element
   removes 123 us of DMA busy and 48% of all DMA packets for only -7.4 us of wall (E1).** So the
   fix has to make a transfer *start earlier*, never make it cheaper — and Pool/DMA work converts
   to wall at only ~6%.

## 0. THE SHIPPED EDIT: `imm_src` (-387.5 us, -8.4%)

`iter-7-imm.py`. Six lines of real work:

```
for every ACT ACTIVATE with activation_func == 128 and imm_src == POINTER_IMMEDIATE:
    assert imm_imm_ptr == 0 and imm_imm_fp32 == 0.0
    set imm_src = INSTRUCTION_IMMEDIATE
```
6157 sites (1024 with `nch=1`, 5133 with `nch=128`). `semaphores.py` unchanged (no sync byte
touched), `verify.py` VERIFIED with `max_abs_err` identical to the last digit.

### 0a. How it was found
`iter-7-actclasses.py` groups the 9265 ACTIVATEs by
`(activation_func, num_active_channels, scale_src, imm_src, bias_src, num_elem[0], imm_imm_ptr,
imm_imm_fp32, bias_imm_fp32, accumulator_cmd)` and prints the mean duration of each class. The
tell was a class of **4** instructions:

| n | mean ns | f | nch | scale_src | imm_src | n0 |
|---|---|---|---|---|---|---|
| 2044 | 162.8 | 7 | 128 | INSTRUCTION_IMMEDIATE | INSTRUCTION_IMMEDIATE | 1 |
| 36 | 161.1 | 128 | 128 | INSTRUCTION_IMMEDIATE | INSTRUCTION_IMMEDIATE | 1 |
| **4** | **294.5** | 129 | 128 | INSTRUCTION_IMMEDIATE | **POINTER_IMMEDIATE** | 1 |

Same element count, same scale source, +133 ns — the only difference is `imm_src`. The soundness
argument is independent of the profile: `imm_imm_ptr == 0` means ACT is being told to read the
operand from SBUF partition 0 offset 0, which holds unrelated kernel data, and the kernel's output
is nevertheless bit-exact against torch — so f=128 cannot consume it.

### 0b. The ACT cost model after the edit (re-measure before using it)
Durations from `prof3`, wall 4251:

| n | mean ns | what it is |
|---|---|---|
| 4405 | 346.3 | `dtx`/`X_scaled`/`dtx_decayed`/`state_sb` — f128 nch128 ptr-scale, 64 elem |
| 1024 | 397.3 | `log_decay = dt_row * A_head` — f128 **nch=1**, 128 elem |
| 2044 | 162.8 | `exp_cs_col`, `exp_neg_cs_col` — f7 nch128, 1 elem |
| 728 | 400.6 | the iter-2/iter-6 merged `Y *= exp_cs_col` — 128 elem |
| 1024 | 201.8 | `exp_cs_last` — f7 **nch=1**, 1 elem |

So: **element cost 0.84 ns/elem; fixed cost 161 ns for an instruction-immediate scale and 293 ns
for a pointer scale.** The 132 ns difference is the per-partition scale-vector fetch and is
**genuinely needed** (there is no "read one scalar and broadcast" mode: `NEURON_ISA_TPB_IMM_SRC` is
only `{INSTRUCTION_IMMEDIATE, POINTER_IMMEDIATE, REG_PTR_IMMEDIATE, REG_VAL_IMMEDIATE}`). 132 ns x
6157 = 813 us of ACT busy that cannot be removed without a different scaling mechanism.
Before the edit the same numbers were 435 / 489 / 163 / 490 / 202 — i.e. the fetch cost 91 ns.

**The `POINTER_IMMEDIATE` sweep is now exhausted** (`iter-7-ptrscan.py` scans every field of every
slot of every engine for a `*POINTER*` enum value and prints the matching `*_imm_ptr`): what is
left is ACT `scale_src` (6157, non-null pointers, used) and DVE
`TENSOR_TENSOR_SCAN_ARITH.imm0_src` (1024, non-null, the scan's `initial` = `zero_scalar`). Nothing
on PE, Pool or SP.

### 0c. Other enum fields checked and left alone (`iter-7-ptrscan.py` + a per-opcode enum census)
* ACT `bias_src` is already `INSTRUCTION_IMMEDIATE` everywhere; `accumulator_cmd` is `IDLE`;
  all dtypes FP32. `ACTIVATE` has no `perf_mode` on this ISA (iter-6 §12).
* `SET_ORDERING_MODE.ordering_mode == 0` ("relaxed") on all five engines already.
* DVE `MEMSET.serialization_mode == SERIAL` on all 1137 (`REG_SERIAL_MODE = {SERIAL, PARALLEL}`).
  **Untested.** Ceiling is small: MEMSET is 63 ns, 72 us of DVE busy in total, so ~10 us of wall
  at best. Cheap probe if someone wants it.
* PE `MATMUL.psum_zero_region == SIZE2048` and `perf_opt == NONE` on all 7233;
  `LDWEIGHTS.perf_opt == NONE`. `PERF_OPT_MODE = {NONE, DOUBLE_ROW, DOUBLE_COLUMN, DOUBLE_PIXEL}`
  are low-precision throughput modes and `fp32_mode` is numerics — both closed (review §4, iter-4).
* DVE `COPY`/`STREAM_SHUFFLE` carry `op_dim = UNUSED`, `negated = 0`, `mask_enable = 0`; nothing
  configured that is not needed.
* PE `row_grp = col_grp = 15` on **all** 14466 LDWEIGHTS/MATMULs, including the 1024
  `num_active_rows = 1` transpose `LDWEIGHTS` — the one place in this NEFF where a field looks
  oversized relative to the real shape (15 reads as "all 16 row/col groups"). Untested, and the
  expected payoff is ~0 because `LDWEIGHTS` (2222 us) runs on a different PE pipe from `MATMUL`
  and has ~1.2 ms of slack. `psum_accumulate_flags` is 1 on the LOW-pass MATMULs and 2 on the
  HIGH-pass ones, so it is the fp32 two-pass accumulation control — numerics, do not touch.

## 1. SLOT DELETION #1: 3119 pure no-op `EVENT_SEMAPHORE`s (-37.4 us)

`iter-7-del.py`. Two provably-dead classes, deleted and the stream compacted (no stream in sg00
has a pc/branch field — iter-1 §3 — and iter-1 already shrank `Activation0.bin`):

* **noop** — `EVENT_SEMAPHORE` with wait NONE and no update, main or extended. Cannot block and
  cannot release. **2829 on DVE** (iter-6 §10's retired `STREAM_SHUFFLE`s), 5 on ACT.
* **redundant (forward)** — the slot's only content is `sem s >= v` and the same engine has
  already *passed* a wait `sem s >= v'` with `v' >= v` earlier in its in-order stream.
  193 ACT, 29 DVE, 52 PE, 11 Pool.
  The enabling fact, which is worth carrying forward: **every semaphore update in this NEFF is
  `SEM_INC_COMPLETE` or `SEM_ADD_IMM_COMPLETE(+16)` — all non-negative** (checked on all five
  engines; there is no `SEM_DEC`/`SEM_WR`/`SEM_SUB`), so the counters are monotone non-decreasing
  and an earlier passed wait implies a later weaker one.

Per-engine: ACT 14074->13876, DVE 24525->21667, PE 14865->14813, Pool 4192->4181, SP unchanged.
Measured busy removed: ~79 us across engines, **wall -37.4 us**. DVE busy only moved 2177->2167,
so most of the win is per-slot cost the profiler does not report. **~12 ns of wall per deleted
DVE slot.**

## 2. SLOT DELETION #2: 3077 dead `LOAD_MASK_SELECT`s (-6.7 us)

`iter-7-lms.py`, census in `iter-7-census-lms.py`. Three facts, all checked as assertions:
1. all 8192 LMS payloads are **byte-identical** with the 8 sync bytes zeroed (`param_ram_offset`
   0, `imm_ptr_select` 0, every immediate array all-zero = the `[0]*32` broadcast mask of
   `nkilib/core/utils/stream_shuffle_broadcast.py`);
2. **all 4818 `STREAM_SHUFFLE`s are immediately preceded by an LMS**;
3. the compiler only ever emits an LMS directly in front of a shuffle or in front of another LMS
   (8192 = 6012 original shuffles + 2180 chained).

So `STREAM_SHUFFLE` is the only reader of the mask register, each shuffle has its own LMS, and an
LMS whose **next surviving slot is not a shuffle** has no reader — the next LMS overwrites the
register with the identical value, or the shuffle that used to follow was retired by iter-3/iter-6.
Deleted 3077 (2200 LMS->LMS chains + 877 orphaned by earlier iterations); 297 skipped because
their successor already waits on a *different* semaphore. 217 waits were **hoisted forward** onto
the successor — moving a wait later only delays, never releases early. DVE 21667 -> 18590.
**Bit-identical on device, so the sticky-mask-register model is confirmed.**

## 3. SLOT DELETION #3: backward subsumption (208 ACT slots, ~0 us)

`iter-7-es2.py`. A pure wait-only `EVENT_SEMAPHORE` whose **next surviving slot** waits on the same
semaphore with a `>=` value is a no-op, because nothing executes between them. 208 on ACT, 0
anywhere else. Worth nothing measurable (4638.56 -> 4638.78, inside the 1.6 us spread) but it is
sound and it is in the shipped NEFF.

### 3a. REFUTED, cheap: folding a wait-only ES into the following instruction
`iter-7-fold.py` moves the ES's wait into the next slot's free wait field and deletes the ES.
**Only 14 of 5443 candidates qualify**: 2027 ACT / 849 DVE / 346 PE / 1000 Pool / 1221 SP are
rejected because the successor **already waits on a different semaphore**. That is exactly *why*
the compiler emitted a separate ES (one wait field per slot, iter-6 §6). **This lever is closed —
do not rebuild it.**

## 4. THE TWO GLOBAL STALLS: 90 us, reproducible, and mostly not SP's fault

`iter-7-actstall.py` (ACT gap census by role and wait-semaphore class). At the iter-6 NEFF, ACT had
**two gaps of 57.9 us and 32.1 us** — 1.9% of the runtime in two events, at t≈3.22 ms and
t≈4.16 ms. They are present identically in the pre-edit profile (57.7 / 32.2 us at the same
timestamps) and **every engine idles in the same window** (Vector 57.5/32.8, Tensor 56.3/31.3,
Sync 72.2/46.0). Not noise.

Chain of evidence:
* SP's 72 us gap sits at a `PSEUDO_DMATRIGGER` waiting `sem 13 >= 2240` (Pool's DGE queue 13,
  transfer #140). SP blocks there, so every *later* SP trigger — including the `dt_row`/`A_head`
  loads ACT needs — is head-of-line blocked, and ACT then stalls on `sem 36 >= 1296` (qSyncIO0).
* **But nulling the waits of all 111 SP slots with a >=10 us gap recovers only 23.1 us**
  (4638.6 -> 4615.4, `iter-7-probe-sp.py`). So SP's stream order is at most a quarter of it.
* The real constraint is the **DMA descriptor rate**. `PendingDma` (outstanding requests) is
  mean 830 in a normal window and **2316 / 2973** in the two stall windows; `DmaUsage` throughput
  is 40-50 GB/s against a 435 GB/s `dma_ddr_bandwidth`, so it is *not* bandwidth.
  `DmaPacket.parquet` grouped by `transfer_bytes`:

  | bytes | packets | DMA-engine ns |
  |---|---|---|
  | **4** | **80287** | **686877** |
  | 2048 | 33792 | 3234465 |
  | 256 | 32768 | 576757 |
  | 32 | 16384 | 246578 |

  **48% of all DMA packets move 4 bytes.** They are the `dt_col` load:
  `dt_col[Q,1] = dt_flat[start:start+Q, 0:1]` is `src (128,1) x 4 B step 4` (contiguous HBM) ->
  `dst (128,1) x 4 B step 262144` (128 separate SBUF partitions), i.e. **128 four-byte scatters
  per unit, 1024 times** — 0.4% of the bytes, 687 us of DMA-engine time, 667 us of Pool busy.

## 5. Where the time is now (wall 4251), and the ranked list for iter-8

| engine | busy us | % of wall | span | idle in span |
|---|---|---|---|---|
| **PE `MATMUL` alone** | **2954** | **69%** | 4152 | — |
| ACT (Scalar) | 3044 (2771 ACTIVATE + 169 ES) | 72% of its 4184 span | 4184 | 1140 |
| PE `LDWEIGHTS` (separate pipe) | 2222 | 52% | — | — |
| DVE | 2445 | 59% of its 4156 span | 4156 | 1711 |
| Pool | 2097 (3169 DMA x 655 ns) | 49% | 4250 | 2153 |
| SP | 471 | 11% | 4210 | 3739 |

The kernel is no longer ACT-dominated. **PE `MATMUL` at 2954 us is now the hard floor** and it is
irreducible: 6144 of the 7233 MATMULs are the three per-unit matmuls x the two fp32 passes
(3088 `LDWEIGHTS LOW` + 3088 `HIGH` + 1057 transposes = 7233; `MATMUL` is always `LOW_HIGH`, so one
logical fp32 matmul = 4 partial products), `num_active_rows = num_active_cols = 128` is exactly the
real shape, and `fp32_mode`/`perf_opt` are numerics. Per-MATMUL fixed cost is ~388 ns against 46 ns
of datapath for 64 moving elements, i.e. PE is issue-bound too, but there is nothing to merge.

**What I would do first in iter-8**, in order: (a) spend the first 20 minutes repeating §0's method
on PE and DVE — dump *every* field of the top opcode, question every value, and ask what a cheaper
value would cost; §0 found 8.4% in six lines and nothing says that seam is closed. (b) If that is
dry, build E2 (the sem-4/sem-5 renumber) on a single site, because it is the only gate on moving
ACT work out of its 1140 us of idle and on prefetching, which is where the remaining 165 us (E3)
and the merged-Y bubble (iter-6 §5) live. **Do not** start with a throughput edit: with PE `MATMUL`
at 2954 us (69%) and no engine above 72% of its span, everything left is overlap and latency.

### E1. Killing the `dt_col` DMA is worth ~7 us. **REFUTED — do not build it.**
This was my top-ranked candidate on the strength of §4 (48% of all DMA packets are the 1024 x 128
four-byte `dt_col` scatters, 687 us of DMA-engine time, 667 us of Pool busy) and the plan was to
replace it with a bit-exact PE transpose of `dt_row`. **The destructive probe kills the idea:**
`iter-7-probe-dtcol.py` shrinks all 1024 of those DMAs to one element — DMA busy 1285 -> 1162
(-123 us), Pool busy 2072 -> 2039, **wall 4251.2 -> 4243.9, only -7.4 us** — and the two global
stalls survive. So the 4-byte descriptor storm is real but **not on the critical path**, and
neither is Pool's 667 us. The same probe says Pool/DMA *work* in general converts at ~6%.
(This also retires, for the same reason, the idea of giving `dtx` its scale from a transposed
`dt_row`: the DMA it would save is worth nothing. The 165 us of §5/E3 is *latency*, not throughput.)

### E2. Build the sem-4 / sem-5 renumbering pass (enabling primitive, still unbuilt)
Unchanged from iter-6 §11.3. sem 4 is purely ACT's completion counter
(7360 increments, 9661 waiters, every wait `GE_IMM`), sem 5 purely PE's (2430 / 2736), sem 6 purely
DVE's (6791 / 8718). For sem 4 and sem 5 the remap is exact because all increments are
interchangeable "instruction k finished" marks; **for sem 6 it is not** (iter-6 §9a: a waiter needs
a *specific* DVE write). Build and validate on one site.

### E3. The 165 us DMA family, properly (needs a prefetch mechanism)
`iter-7-probe-dma.py` prices it at 165 us. The ACT side of it, from `iter-7-actstall.py` on the
iter-6 NEFF (gaps >= 200 ns, total 1515 us):

| us | n | wait | stalling instruction |
|---|---|---|---|
| 384.6 | 493 | sem6 DVE | wait-only `EVENT_SEMAPHORE` (440 are iter-6's merge placeholders — **refuted lever**, iter-6 §6) |
| 288.8 | 363 | sem5 PE | 64-elem ptr multiply (`state_sb` WAR, `dtx_decayed`) |
| 218.1 | 332 | sem6 DVE | merged `Y *= exp_cs_col` |
| 112.2 | 236 | sem6 DVE | `log_decay` |
| **100.5** | **23** | **qSPIO0 DMA** | **`log_decay` — this is the two global stalls** |
| 79.3 | 224 | sem6 | 64-elem ptr multiply |
| 66.5 | 195 | sem6 | `exp_cs_last` |
| 60.9 | 157 | **sem4 ACT** | `exp_cs_col` — ACT waiting on its own completion counter (pipeline drain) |
| 42.8 | 126 | dge Pool | `dtx`, waiting for the `x_sb` load |

### E4. Leftovers worth ~5-10 us each (take only in combination)
* 342 `STREAM_SHUFFLE`s at the iter-6 §10 sites are dead but were kept because they carry a sem-6
  increment. Replacing each with a **1-element `MEMSET` to the same dead buffer** keeps the
  increment firing on compute completion (iter-6 §9) and saves 147-63 = 84 ns x 342 = 29 us of DVE
  busy. Mind iter-6 §9a: `MEMSET` has a separate `dst_element_count` at offset 28.
* The other 380 dead shuffles are at the **95 units iter-6 explicitly rejected** (71 fail the WAR
  check — shipping them gives `max_abs_err 7.9e-02` — and 24 are the worthless timing-only ones).
  **Do not retry the second-broadcast retirement; the remaining sites are exactly the unsafe ones.**
* 230 DVE `MEMSET`s carry no sem-6 update; if any are dead they are deletable slots (~3 us).
* **`MEMSET.serialization_mode SERIAL -> PARALLEL`: BUILT, VERIFIED bit-exact, worth 0.**
  All 1137 sites: 4251.24 -> 4250.91, inside the 1.6 us spread; DVE busy 2133.4 -> 2132.6.
  Not shipped (`iter-7-ser-NOEFFECT.py`). The field is accepted by the hardware and changes
  nothing measurable, so it is a safe but pointless knob.
* **`header_inst_word_len` is 16 on all 61289 slots of all five engines** and
  `header_debug_cmd`/`header_debug_hint` are 0 everywhere — there is no evidence this build
  supports a shorter instruction encoding, so stream compaction beyond deleting whole slots is
  not available.

### E5. Do NOT redo
Everything in iter-6 §11.7, plus: folding a wait-only ES into its successor (§3a, 14 of 5443 sites);
the second `exp_cs_last` broadcast at the remaining 95 units (E4); reordering SP to fix the global
stalls (23 us ceiling, §4); the `POINTER_IMMEDIATE` sweep (§0b, exhausted); ACT merging (iter-6 §2);
`exp_cs_last` recomputation (iter-6 §3, device hang); the merge-placeholder waits (iter-6 §6, -2 us).

## 6. Tool notes (additions to iter-6 §12)
* The whole pipeline this iteration: `neff.py unpack kernel.neff work/`,
  `profile.py --repeats 3 --parquet prof`, run a script from `scripts/` (each loads `work/` and
  rewrites one `*.bin`), `neff.py pack work/ kernel.neff`, `semaphores.py work sg00`,
  `verify.py`, `profile.py`. **Keep a `work_<step>/` copy after every accepted step** — the scripts
  here take a `root=` argument precisely so a later analysis can read an earlier stream while
  `work/` holds the current one.
* **Slot deletion changes `compiler_pc`.** Any analysis joining a stream to `Instruction.parquet`
  must use a profile captured on *that* stream, or the durations are attributed to the wrong slots.
  Deleting 208 ACT slots silently invalidated `prof2`'s ACT mapping.
* `Instruction.parquet` rows with `compiler_pc` NaN are the runtime epilogue
  (`DRAIN`/`NOTIFY`/`COMPARE_BRANCH`/`TENSOR_LOAD`/`EVENT_SEMAPHORE_RANGE_CLEAR`); the whole
  epilogue is ~1 us. Drop them, and drop ACT's `DMA_DIRECT2D` rows before computing gaps (they
  overlap the compute rows and make the gap sum exceed the real idle time).
* Useful tables beyond `Instruction`: **`PendingDma`** (outstanding DMA requests over time — this is
  what exposed §4), `DmaPacket` (group by `transfer_bytes` to find descriptor-rate waste),
  `DmaPacketAggregated` (has a `variable` column naming the tensor: `dt`, `A`, ...),
  `SemaphoreUpdate` (`id`, `value`, `update_source`, timestamp — lets you see exactly when a
  threshold was reached and which engine or AXI port supplied it), `HbmUsage`/`DmaUsage`,
  `Warning` (this build warns that DGE notifications are off, so DMA numbers are approximate).
* Semaphore map for this NEFF: 3 Pool, **4 = ACT completion**, **5 = PE**, **6 = DVE**,
  8-23 = `qGpSimdDynamic` (Pool DGE, +16/transfer), 24-39 = `qSyncIO0` (`dt`/`A`, +16),
  40-55 = `qActDynamicHW`, 56-71 = `qSyncDynamicHW`. The `sems 40-71 wait exceeds increments`
  alarm from `semaphores.py` is still iter-4's documented false positive.
* The kernel source is on this machine:
  `/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/experimental/scan/ssd_block.py`
  (and `ssd_utils.py`). `_compute_ssd_head_tile` is the per-unit body; reading it is much faster
  than inferring the dataflow from the stream. The nine per-unit ACTIVATEs map to
  `log_decay` / `exp_cs_col` / `exp_neg_cs_col` / `exp_cs_last` / `dtx` / `X_scaled` / merged `Y` /
  `dtx_decayed` / `state_sb`.
* A destructive probe (null a class of waits, repack, 2-repeat profile) costs ~4 minutes and prices
  a whole family. `iter-7-probe-sp.py` and `iter-7-probe-dma.py` are templates; nulling a wait can
  never deadlock (it only removes ordering), unlike adding one.

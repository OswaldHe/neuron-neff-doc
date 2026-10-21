# iter-3 — two DGE staging rows re-hosted off the engines that were head-of-line blocking
# the mid-kernel critical path, plus two Vector CASTs executed on idle Scalar.
# Three refutations, and two of the review's candidates closed by measurement.

## Result: **~41.0 → ~40.25 us (−0.75 us, −1.8%)**, outputs bit-identical

| step | edit | median (7 captures) | Δ | verdict |
|---|---|---|---|---|
| — | iter-2's bytes, measured at session start | **41.074** | — | anchor |
| **B** | sem-43 DGE row `ACT[56]` → **Pool**, inserted *before* `Pool[14]` (§2) | **40.822** | −0.25 | **kept** |
| B2 | B + the row's wait split onto a preceding Pool `ES` (§2.1) | 40.794 | ~0 (row 0.3 us *later*) | **REFUTED, reverted** |
| C | B with the row on **Sync** at `SP[22]` instead of Pool (§2.2) | 41.384 | **+0.56** | **REFUTED, reverted** |
| **D** | sem-45 DGE row `SP[26]` → **Activation**, into its 27.1–27.8 hole (§3) | **40.450** | −0.37 | **kept** |
| E | D + the sem-43 row moved from Pool to Activation as well (§3.1) | 40.884 | +0.43 | **REFUTED, reverted** |
| **F** | `DVE[49]` (`CAST`, no wait, no update) executed on idle Scalar (§4) | **40.22**¹ | −0.19¹ | **kept** |
| **G** | `DVE[60]` (`CAST`) too — the case that needed a WAR guard (§4.2) | **40.245**¹ | −0.14¹ | **kept (SHIPPED)** |

¹ F and G are too close to separate with one capture each; their medians and Δs are over several
interleaved pairs (table below), not single 7-capture runs like B–E.

**Every Δ is a mean over interleaved A/B pairs, not a difference of remembered numbers**, because
the device drifted ~0.4 us in *both* directions during the session: iter-2's unmodified bytes read
41.074 / 40.985 / 40.677 / 41.014 at four points. The pairs that decide the result:

| pair, back to back on the same device | A | B | Δ |
|---|---|---|---|
| iter-2's bytes vs **D** | 40.985 | 40.494 | **−0.49** |
| **D** vs **F**, ×2 | 40.385 / 40.453 | 40.224 / 40.233 | −0.16 / −0.22 |
| iter-2's bytes vs **F** | 40.677 | 40.145 | −0.53 |
| **F** vs **G**, ×5 (alternating which went first) | 40.392 / 40.194 / 40.552 / 40.237 / 40.216 | 40.264 / 40.259 / 40.245 / 40.087 / 40.018 | −0.13 / **+0.07** / −0.31 / −0.15 / −0.20 |
| iter-2's bytes vs **G**, final two runs | 41.014 | 40.018 | −1.00 |

Shipped bytes (G) measured **seven** times: 40.018 / 40.062 / 40.087 / 40.245 / 40.259 / 40.264 /
40.316, median **40.245**. iter-2's bytes, four times, median **41.00**. G beat F in 4 of 5 pairs (mean
−0.14) and its *span* is unambiguously better (40.514 vs 40.799, every landmark −0.26 to −0.29),
which is why it ships despite one pair going the other way. **Quote ~40.25 vs ~41.0, i.e. −0.75 us
(−1.8%).** Against iter-0: 48.519 → ~40.25, **−17.0% (1.206x)**.

`verify.py --runs 3` PASS on the shipped bytes, all runs bit-identical, `max_abs_err` **identical
to the baseline to every printed digit** after every kept step (output_0 6.119e-05 = 17%, output_1
2.864e-04 = 82%). `launch_rules.py` PASS. `semaphores.py` shows no semaphore with
`max_wait > total_inc` at any step; `S[4]` stays 41/41, `S[5]` 64/64 and `S[6]` 66/66 throughout.
(`sem 16` reads 0/16 — pre-existing, its only waiter is Pool's final *extended* wait, which
`semaphores.py` does not parse. iter-1 §3 already recorded this.)
`neff_diff` vs iter-2: **4 of 34 members** — `sg00/{Activation0.bin 77→80 slots,
DVE0.bin 100 slots (2 differ), Pool0.bin 23→24, SP0.bin 42→41}`.
sha256 `5cdbc364c8350df899513a1318eaa4f54e4a13badd0c4d61541d1e966b700de3`.
Engine busy on the shipped bytes (`profile.py`'s own report, final run): **Scalar 20.12 (50.2%),
Vector 19.78 (49.4%)**, DMA 13.77 (34.4), Tensor 12.07 (30.1), GpSimd 9.74 (24.3), Sync 8.87 (22.1).
**Scalar is the busiest engine again, by 0.35 us, so the §4 Vector→Scalar lever has run its
course** — the next ~0.2 us of Vector relief would come straight back out of Scalar. iter-4 should
re-read `occ.py` before assuming either direction.

---

## 0. The one number that decides every DGE-row rehost on this kernel

**A DGE descriptor row issues at a different delay after its gate lands depending on the host:**

| host | measured delay gate→row start | row duration | occupies the sequencer? |
|---|---|---|---|
| **Activation** | **+0.02 us** (`ACT[56]` +0.021, `ACT[62]` +0.543¹) | 638–670 ns | yes, but see §3.1 |
| **Sync** | **+0.02 us** (`SP[41]` +0.016) | 606–663 ns | yes |
| **Pool** | **+0.39 us** (`Pool[14]` +0.390, `Pool[17]` +0.405, `Pool[19]` +0.405) | 634–692 ns | yes, strictly serial |

¹ `ACT[62]`'s +0.543 is program-order, not reaction: its gate landed 1.2 us before `ACT[60]` finished.

**+0.39 us is a fixed Pool DGE issue latency, not a semaphore-reaction cost.** Edit B2 tested that
directly: putting the wait on an `EVENT_SEMAPHORE` immediately in front of the row (so the row is
reached by program order, which costs 14 ns between two back-to-back Pool rows — `Pool[7]`→`Pool[8]`)
made the row start **0.30 us later still** (26.451 → 26.753). The `ES` paid a wake-up and the row
paid its latency on top. A Pool *compute* op reacts in +0.05 (`Pool[14]` affine select +0.048).

So: **host a latency-critical DGE row on Activation or Sync, never on Pool**; host a row on Pool
only when you are buying sequencer time on a busier engine and can afford +0.39 on its semaphore.
Both kept edits are exactly that trade, in opposite directions.

## 1. Rebuild these (~30 min; `scripts/` is not preserved)

`scripts/lib.py` (load/save a stream as 64-byte slots, `es(wait=(idx,thr), update=idx)` with an
assert-on-re-decode), `scripts/d2.py` (per-slot waits/updates/extended/addresses/semaphore),
`scripts/an.py` (`load/landing/body/gate` + `occ`, `gaps`, `tl ENG lo hi`, `reorder`),
`scripts/cpath.py` (backward critical path).

Five traps that cost me time, fix them up front:
* **`isa.py`'s "no wait"/"no update" enum name is `NONE`, not `WAIT_NONE`/`SEM_NONE`.**
  `ins.set("events_wait_mode", "NONE")` is how you clear a wait; `"WAIT_NONE"` raises `KeyError`.
  The *set* name for a wait is still `"WAIT_FOR_SEM_GE_IMM"`, and for an update `"SEM_INC_COMPLETE"`.
* **`SemaphoreUpdate.id` is a *string* like `"S[6] (Vector)"`**, not an int — `.str.extract(r"(\d+)")`.
  Same for `DmaPacketAggregated.semaphore_id`, and `str.extract` returns a **DataFrame**, so index
  `[0]` before `.astype(int)`.
* **MATMULs are on `timeline_track == "TensorMatrix"`, not `"Tensor"`.** `"Tensor"` holds only
  `LDWEIGHTS` + sequencer rows, so a critical path that maps PE→`Tensor` dead-ends at every matmul.
  Map **both** tracks to PE. (71 MATMULs, 16.95 us of PE compute, invisible otherwise.)
* **`common.NEURON_ISA_TPB_S4D4_TR_STRUCT` (`CAST`/`COPY`) has no extended events block at all** —
  `as_dict()` contains no `events_extended_*` key — so such a slot can carry exactly **one** wait.
  `EVENT_SEMAPHORE` (`v2.NEURON_ISA_TPB_CTRL_ES_STRUCT`) does have one. If a moved op needs two
  guards, either find a single semaphore that implies both (§4.2) or put an `ES` in front of it.
* **`iter_stream()` yields `(index, Instruction)` tuples**, and a field is a dict with a `"name"`
  key; `ins.name` is the opcode string while `ins["header_opcode"]` is the symbolic enum. Use
  `ins.as_dict()` and read keys — iterating `ins.fields()` for values will bite you.

`an.py gate(S, operands)` = the latest landing of any `S\[(\d+)\][^>=<]*>=\s*(\d+)` in the profiler's
`operands` text. `t0` is normalised to `SP[0] SET_ORDERING_MODE`. **`slack = start − gate` is the
single most useful column**: ~0.02 means the slot is gate-bound (nothing to win), and anything
above ~0.15 means something on its own engine is in front of it.

## 2. Edit B — the sem-43 row off Activation onto Pool: **−0.25 us**

At 26.1 us Activation runs `ACT[56]`, a 661 ns `DMA_DIRECT2D` row (sem 43, src p96..111+0xe000,
dst p64..127+0x1d200), and the whole mid-kernel chain queues behind it:

```
ACT[56] row 26.122-26.783 | ACT[58] TS (gate S[5]>=57 landed 26.187, ran 26.951 → +0.76 BLOCKED)
ACT[58] → S[4]#30 → ACT[60] → S[4]#31 → DVE[83] → S[6]#52 → ACT[62] → ACT[63]
       → S[4]#32 → SP[27] (blk15 trigger) and SP[29] (blk16 trigger)
       → S[39] / S[24]>=32 → ACT[64] / PE[128] → the whole 30-38 us stretch
```

Pool is idle 8.8 → 26.6 us, so the row went to **Pool index 14, before `Pool[14]`** (the affine
select). Vacated `ACT[56]` became `ES{wait S[6]>=49}` — **keep the wait**, it is the guard the rest
of the Activation stream was inheriting, and it costs nothing because `ACT[55]` ends at 26.10.

Safety (all three checks matter):
* the row's own wait `S[6]>=49` is the last writer of its source p96..111+0xe000..0xe7ff — iter-2 §4
  chose these guards deliberately, so the row is self-sufficient off Activation;
* its destination p64..127+0x1d200 has **exactly one reader, blk16**, gated `S[43]→Pool[16]→S[14]
  →SP[29]`. Nothing else in any stream or any `SP0.json` descriptor touches 0x1d200;
* inserting before `Pool[14]` delays the affine select by the row; its only consumer is
  `S[3]#6 → DVE[82]`, which had 1.4 us of slack.

Measured: `S[4]#32` 28.720 → 28.494 (−0.23) but `S[43]` 28.214 → **28.415 (+0.20)** from Pool's
+0.39 issue latency, so blk16 gave back most of what blk15 gained; end to end only −0.15 on the span.
**Kept because it is the precondition for D** (it is what opens Activation's 27.1–27.8 hole).

### 2.1 B2 — REFUTED: splitting a Pool row's wait onto a preceding `ES` makes it *later*
See §0. Row 26.451 → 26.753, `S[43]` 28.415 → 28.797, metric unchanged (40.794 vs 40.822).

### 2.2 C — REFUTED again, harder: **never insert a DGE row in front of Sync's trigger block**
I put the sem-43 row at `SP[22]` (Sync reacts in 20 ns, so `S[43]` should have landed at ~28.15) on
the argument that iter-2 §4.1's +0.56 was about *three* rows and that after B, `ACT[64]` had 0.26 us
of slack against blk14's `S[38]`, so a ~0.19 us push on `SP[23]` would be absorbed.
**Wrong by a factor of five: `S[38]` went 27.828 → 28.811 (+0.98)** and every landmark after it
slipped ~1 us (`S[4]#32` +0.74, `S[39]` +0.97, `S[26]#32` +0.95, `S[17]` +1.01). Metric **+0.56**,
the same number iter-2 measured. One 640 ns row in front of `SP[22]` costs blk14 a full microsecond
— the trigger block is a chain of `ES(ring guard)+trigger` pairs and a row in front of it desyncs
the queue, not just the issue. **This lever is now closed twice. Do not test it a third time.**

## 3. Edit D — the sem-45 row off Sync into Activation's hole: **−0.37 us, the win**

After B, the binding item was a textbook §3-style blockage **on Sync**:

```
SP[25] (sem46 row) ends 28.014
SP[26] (sem45 row) 28.060-28.667          <-- 607 ns in front of two triggers
SP[27] blk15 trigger: gate S[4]>=32 landed 28.494, fired 28.721   (+0.227 blocked)
SP[29] blk16 trigger: gate S[14]>=1  landed 28.462, fired 28.874
   blk16 → S[24]>=32 @30.696 → PE[128] → PE[129] → S[5]#58 → DVE[87] → ... → the tail
   blk15 → S[39] @30.426 → ACT[65]
```

Activation, having lost `ACT[56]` in edit B, is **idle 27.100 → 27.831** — 0.73 us, almost exactly a
row. So the sem-45 row (`S[6]>=50`, src p96..111+0xe800, dst p64..127+0x1d400) was deleted from
`SP[26]` and inserted at **Activation index 62, immediately before the old `ACT[62]`** (the
`ACTIVATE SOFTPLUS` whose gate `S[6]>=52` lands at 27.809).

Measured, exactly as designed:
* row runs 26.777–27.415 on Activation, `S[45]` **30.123 → 28.920 (−1.20)** — a bonus, because
  Activation reacts in 20 ns *and* the hole is earlier than `SP[26]`'s old slot;
* `ACT[63]`/`ACT[64]` (the old `ACT[62]`/`[63]`) did not move: 27.806 and 28.070, unchanged;
* `SP[26]`/`SP[28]` (blk15/blk16 triggers) became **gate-bound**: +0.016 / +0.193 slack;
* `S[39]` −0.33, `S[24]#32` −0.32, `S[26]#32` −0.32, `S[27]#32` −0.34, `S[17]` **40.718 (−0.34)**.

Safety: the row keeps `S[6]>=50`; src p96+0xe800 and dst p*+0x1d400 are disjoint from `ACT[60]`
(reads p32+0x5100, writes p0+0x37900) and from `ACT[63]` (reads p96+0x4318, writes p96+0x4700); the
only reader of p*+0x1d400 is `DVE[87]`, gated by `DVE[86]`'s `S[60]>=1 ← SP[33] ← S[45]`, so issuing
the row 1.2 us earlier cannot expose it. sem 45's queue is named `qScalarDynamicHW` — Activation is
where iter-1 originally put this row, so it is going home.

> **Generalised lever, and it is still live: for every `PSEUDO_DMA_DIRECT2D` row, list the
> `PSEUDO_DMATRIGGER`s that follow it on the same engine and compare each trigger's fire time with
> its own gate landing. Any slack is the row. Then find a 650 ns hole on Activation (never Pool) at
> or before the row's gate.** That is the whole of edit D, and it is the only lever this session
> that paid more than the measurement noise.

### 3.1 E — REFUTED: a second row on Activation *does* block the sequencer
I moved the sem-43 row off Pool onto Activation too (index 57), reasoning from `ACT[60]`
(26.681–27.119) and the sem-45 row at `ACT[62]` (26.777–27.415) **overlapping** in the profile, and
from iter-1 §5.1 ("an ACT DGE row is mostly not exclusive ACT occupancy").
**That overlap is the DGE pipe running behind the sequencer, not free capacity.** With the row in
*front* of compute, `ACT[58]` (the `TENSOR_SCALAR`) went 26.297 → 27.395 (**+1.10**), `S[6]#52`
+0.70, `S[4]#32` 28.469 → **29.195**, metric **+0.43**. The correct reading of iter-1 §5.1 is: a row
placed *after* the op it would otherwise delay overlaps with it; a row placed *before* it does not.
`S[43]` also got *worse* (28.397 → 28.640) despite the faster host, because the row landed later in
Activation's program order than it had in Pool's.

## 4. Edit F — one Vector `CAST` on idle Scalar: **−0.19 us, and the mechanism is now measured**

Vector was the busiest engine after D (20.2 us, 50%) and 85–92% busy from 16 to 26 us while Scalar
sat at 0–50% with a **1.23 us hole at 15.786–17.015**. `TENSOR_TENSOR` does not exist in
`Activation0.bin` so it is not movable, but `CAST` is — and four Vector `CAST`s in that window have
**no wait and no update at all**, which makes the move almost bookkeeping-free.

`DVE[49]` is the one with an airtight address story, and it is worth writing down why:
* `CAST p0+0x4500 fp32 (128 ch) → p0+0x4b08 bf16 (128 ch)`, 206 ns;
* its **source is written exactly once** before it, by `DVE[6] MEMSET(0)` at 1.122–1.287, and not
  again until blk15 at ~29 us — a 27 us stable window;
* its **destination is written by nothing else** and read only by `PE[127] MATMUL` (25.8) and
  `PE[129] MATMUL` (30.4);
* struct is `common.NEURON_ISA_TPB_S4D4_TR_STRUCT` on both engines — byte-identical, so the 64-byte
  slot was copied verbatim and only the two event fields were set.

The move: Scalar gets the slot at **ACT index 36** with `wait S[6]>=4` and
`update S[50] SEM_INC_COMPLETE`; Vector keeps `ES{wait S[50]>=1}` **with no update** in the vacated
slot. The guard needs an argument because `DVE[6]` has no semaphore of its own:
`DVE[6] → DVE[7] → DVE[8]` in Vector program order and `DVE[8]` is the producer of `S[6]#4`, so
**`S[6]>=4` implies `DVE[6]` completed** (and lands at 3.385, 12 us before the insertion point).
The update-less stand-in is what keeps this safe: every later Vector op still implies the CAST
completed, exactly as Vector program order used to, at ~24 ns instead of 206.

> **What this measures, and it is the reason to keep it: the 16–26 us window is only *partly*
> Vector-backlog-bound.** Vector busy fell 19.41 → 19.23 us (−0.18, as designed) but the window's
> last Vector end moved only 15 ns (27.541 → 27.527) and the downstream landmarks only −0.05
> (`S[5]#58` 30.749 → 30.692, `S[24]#32` 30.375 → 30.322, `S[17]` 40.718 → 40.663). The metric
> nonetheless moved −0.19 across three interleaved pairs, and the shipped bytes' spread *tightened*
> to a 0.09 us band. So removing work from Vector here pays about **1 us of metric per 1 us of
> Vector work removed is NOT the right model** — it is closer to 1:1 on the metric but ~0.3:1 on the
> span. Budget ~0.15–0.2 us per movable op, not 0.2+ per 0.2.

### 4.1 Why `DVE[43]`, `DVE[59]`, `DVE[60]` are not *free* — they need a WAR guard as well
All three write into the **0x4300 / 0x4700 / 0x4800 scratch buffers that are rewritten 3–4 times
per kernel with PE matmuls reading in between**, e.g. for 0x4700:
`DVE[46]` writes 17.438–17.657 → **`PE[58] MATMUL` reads it at 17.712, 55 ns later** →
`DVE[59]` rewrites at 20.120 → `PE[96]` reads 21.751 → `PE[125] LDWEIGHTS 0x477e` 25.728 →
`PE[138]` 34.020. Moving `DVE[59]` earlier onto Scalar without a guard would clobber `PE[58]`'s
input, and `PE[58]` has **no semaphore update** (`acc_flags=1`, mid-accumulation). 0x4800 has the
identical shape (`DVE[40]`→`PE[47]`→`ACT[39]`→`PE[68]`→`DVE[60]`→`PE[103]`→`DVE[72]`→`PE[111]`) but
its reader `PE[68]` **does** have an `S[5]` update — which is what made edit G possible.
`DVE[43]`/`DVE[59]` are still open; see §7.1.

### 4.2 Edit G — `DVE[60]` as well: **−0.14 us, and one wait covered both hazards**
`DVE[60]` = `CAST p96+0xe200 fp32 (32 ch) → p0+0x4800 bf16`, 210 ns, no wait, no update, ran 20.266.
It needs two guarantees and **a single semaphore supplies both**:
* **RAW** — its source p96+0xe200 is written by `DVE[55]` (19.218–19.492), the producer of `S[6]#29`
  (lands 19.503);
* **WAR** — the previous occupant of p0..31+0x4800 was written by `ACT[39]` (17.699) and last read
  by `PE[68] MATMUL` (18.770–19.079). `PE[68]` owns `S[5]#32`;
* and **`DVE[55]` itself waits `S[5]>=32`**. So `S[6]>=29` ⟹ `DVE[55]` ran ⟹ `S[5]>=32` held ⟹
  `PE[68]` finished. One wait, both hazards, no extended-wait needed — which matters because
  **the `S4D4_TR` struct has no extended events block at all** (asserted in `editG.py`: no
  `events_extended_*` field exists on a `CAST`, so it can carry exactly one wait).

Attributing `S[5]#32` to `PE[68]` is the step worth copying: **`S[5]` increments land in MATMUL
*end-time* order, ~14 ns after the end** — `PE[67]`→#31 @18.987, `PE[68]`→#32 @19.093,
`PE[70]`→#33 @19.127, `PE[72]`→#34, `PE[74]`→#35 — even though the matmuls overlap heavily and the
profiler prints them out of order. Sort PE by `t1`, not by `pc`, and the mapping is unambiguous.

Scalar is idle 19.437→20.797 (1.36 us), so the copy went at ACT index 44 with `wait S[6]>=29,
update S[51]` and runs ~19.5–19.7; Vector keeps `ES{wait S[51]>=1}`, no update.
Device: `DVE[60]` 210 ns → 24 ns, `DVE[61]` 20.490 → **20.239**, `DVE[62]` −0.25, `S[5]#58`
30.692 → 30.428, `S[17]` 40.663 → **40.378**, span 40.799 → **40.514**. No reader touches
p0+0x4800 between `PE[68]` (18.770) and `PE[103]` (22.350), so writing it 0.8 us early is safe.

## 5. Closed this iteration (do not re-spend budget)

> **Slot indices in §2–§5 are as they appeared in the profile at the time of each edit, and this
> iteration inserted slots.** Against the **shipped** bytes: Activation ≥36 is **+1** and ≥44 is
> **+2** from the numbers in §2/§3/§5 (edits F and G inserted `CAST`s at `ACT[36]` and `ACT[44]`),
> Pool ≥14 is +1, Sync ≥26 is −1.
> `compiler_pc` is always the post-edit slot index, so re-profile and read the indices off the
> parquet rather than trusting any number written here. Addresses and semaphore ids are stable.

* **Swapping the two Activation slots in front of `S[4]#32`** (`ACT[62]` `ACTIVATE SOFTPLUS`,
  update-less, gate 27.78, and `ACT[63]` `TENSOR_SCALAR`, gate `S[38]` 27.81) looks like iter-2
  §5.5's −0.73 pattern and is worth ~0.38 us. **It is a race.** `ACT[62]` writes p96+0x4700 and
  **blk15's second descriptor reads `from_off 25184000` = p96+0x4700** (`SP0.json` id 15), while
  blk15's trigger waits only `S[4]>=32`, i.e. `ACT[63]`. `S[4]>=32` is *also* the WAR guard for
  blk15's `to_off 17664` = p0+0x4500, which `ACT[63]` reads. One semaphore, two duties, both real.
  Giving `ACT[62]` its own semaphore and fencing the trigger on it returns the stall exactly.
  **`semaphores.py` cannot see this. Always resolve a DGE/queued descriptor's `from_off` against
  the instruction you are about to move, in bytes: `p*262144 + offset`.**
* **The review's candidate 1 — "the five Pool input-load DGE rows start too late and are serial"
  — is closed.** They run 1.297→4.880 back to back and land `S[9]`@3.47, `S[10]`@4.31, `S[11]`@5.02,
  `S[12]`@6.42, but **every consumer has 0.48–2.9 us of slack on them**: `SP[3]` +0.57,
  `DVE[14]` +1.91, `ACT[7]` +1.73, `ACT[8]` +2.88, `ACT[12]` +2.42, `DVE[21]` +0.73, `ACT[11]` +0.48.
  The first MATMUL (`PE[3]`, 3.521) is gated by `S[3]>=1` with **+2.85** of slack. The early window
  is Activation/Vector program-order bound (Scalar 93–96% at 6–8 us), not DMA bound. Moving the
  loads to SP/ACT buys nothing.
* **The review's candidate 3 — splitting the final output store** — measure first, then don't.
  Row cost is payload-independent (606–692 ns for 32 KB and 64 KB alike) so splitting cannot shorten
  the row, and the flight is nearly payload-independent too: sem 16 (64 partitions) row-end→first
  increment **1.039 us**, →16th **1.251**; sem 17 (128 partitions) **1.169 / 1.345**. Doubling the
  payload cost 0.13 us, so two halves on two queues can recover **at most ~0.1 us**. The tail's
  1.37 us is a fixed DMA latency with nothing left to overlap.
* **Hoisting the blk16 trigger over the blk15 trigger** — iter-2 §5.3, +0.16, still closed (one
  shared queue). After D both triggers are gate-bound anyway, so there is nothing to hoist.
* **`ACT[71]` (update-less, slack +1.775, parked behind the `ES(S[26]>=32)` fence `ACT[70]`)** and
  **`ACT[72]`/`ACT[73]`** — both reorders are *correct* and both gain **0**: the chain they feed
  (`S[4]#37 → DVE[97] → S[6]#64 → ACT[74] → S[4]#38 →` blk19 trigger) is **Vector**-bound at that
  point, `DVE[94..96]` occupying 34.623–35.375 back to back. Worked out on paper, not run.
* **`ACT[68]`/`ACT[69]` swap** (`ACT[69]`'s gate `S[4]>=34` lands 31.835, blocked to 32.241 by
  `ACT[68]`, and blk18's trigger is gate-bound on `S[4]#36`): arithmetically worth only **0.10 us**
  (both ops total 564 ns either way; only the order of the two ends changes) **and** it renumbers
  `S[4]#35`/`#36`, so `DVE[91]`'s `S[4]>=35` would have to be raised to `>=36`. Not worth it.
* Everything in iter-1 §7 and iter-2 §7 still stands. `dead2.py` was not re-run (iter-2 found it dry
  twice); PE and DVE have **no `PSEUDO_DMA_DIRECT2D` slot anywhere in this NEFF**, so Activation,
  Pool and Sync are the only proven DGE hosts.

## 6. Where the remaining ~40.25 us goes

Span 40.514 on the shipped bytes, **3.76 us of all-engine idle in 6 gaps > 150 ns, every one of
them a DMA flight**, no engine above 51% (Scalar 20.12, Vector 19.78, DMA 13.77, Tensor 12.07,
GpSimd 9.74, Sync 8.87):

| gap us | at | waiting for |
|---|---|---|
| 0.485 | 10.82 | blk3 (var8 reload) — 1.04 us slow scatter, iter-1 §6.1, still a layout change |
| 0.452 | 28.67 | `S[46]` (sem-46 row) — **harmless**, its consumer `DVE[87]` has 1.3 us of slack |
| 0.489 | 29.29 | blk15 `S[39]` → `ACT[67]`; really the 1.6 us trigger→landing of blk15/blk16 |
| 0.384 | 33.34 | blk18 `S[26]>=32` |
| 0.578 | 35.65 | blk19 `S[27]>=32` → `PE[141]` |
| **1.367** | 39.04 | the sem-17 output store: 663 ns row on Sync + **1.35 us flight** → final fence |

Plus the 2.3 us DRAM→SBUF gate load at the head (`SP[1]`, nothing to overlap with).
**Every queued hop is ~1.6–1.75 us trigger→semaphore** (≈0.75 trigger→start, 0.2–0.3 transfer,
0.45–0.68 propagation) and there are five of them on the serial chain (blk15/16, blk18, blk19,
blk20) plus the final DGE store. That is ~8 us of the ~40.2 and it is the real floor unless a hop
can be removed or genuinely overlapped.

The backward critical path through the last 2.8 us, for reference — every hop gate-bound to ±25 ns:
`PE[145] MATMUL` 38.01 → `DVE[99]` 292 ns → `ACT[76]` 297 ns → `SP[41]` row 663 ns → flight 1.348 →
`Pool[22]` fence. Nothing in it is reorderable.

## 7. Ranked candidates for iter-4

1. **`DVE[43]` and `DVE[59]`, the last two wait-less Vector `CAST`s — but Scalar is now the busier
   engine (20.12 vs 19.78), so re-measure the occupancy before spending on this.** `DVE[49]` (§4) and
   `DVE[60]` (§4.2) are done, −0.19 and −0.14. The remaining two:

   | slot | dur | src → dst | RAW guard | WAR guard — the hard part |
   |---|---|---|---|---|
   | `DVE[43]` | 218 | p96+0xe800 → p0+0x4300 bf16, 32 ch | `S[6]` count of `DVE[38]` (writes 0xe800) | the last PE reader of the *previous* 0x4300 write (an `ACT` CAST at ~13.4); `DVE[56]` rewrites 0x4300 at 19.5, so the window is narrow both ways |
   | `DVE[59]` | 218 | p96+0xea00 → p0+0x4700 bf16, 32 ch | `S[6]#28` (`DVE[54]`) | `PE[58]` reads 0x4700 at **17.712, 55 ns** after `DVE[46]` writes it, and `PE[58]` has **no `S[5]` update** — you need the first `S[5]` increment *after* `PE[58]` (sort PE by `t1`, §4.2) and must then check that waiting for it still lets the Scalar copy beat 20.12 |

   Recipe, now device-proven twice (iter-2 §2 in reverse; `CAST`'s `S4D4_TR` struct is
   byte-compatible both ways, confirmed against `ACT[36]`): copy the 64-byte slot to Scalar, set
   **one** wait that covers the RAW *and* the WAR (a `CAST` cannot carry two — no extended events
   block, §4.2; if no single semaphore covers both, put an `ES` carrying the second guard
   immediately in front of it on Scalar) plus an update on a free semaphore, and leave
   `ES{wait S[new]>=1}` **with no update** in the vacated Vector slot. Get the Scalar *insertion
   point* right: its predecessors must all finish before the guard lands, or Vector stalls on
   `S[new]` and you lose — that is exactly how edit E failed. **Budget ~0.15 us per op**, and stop
   when Scalar's busy time passes Vector's. `TENSOR_TENSOR` is **not** available on Scalar (no slot
   of that opcode anywhere in `Activation0.bin`) so the 274–292 ns Vector ops cannot move at all.
2. **The `S[4]>=32` double-duty semaphore (§4, first bullet).** blk15's trigger needs *both*
   `ACT[62]`'s output (RAW on p96+0x4700) and `ACT[63]`'s read (WAR on p0+0x4500). Splitting blk15
   into two blocks in `SP0.json` — descriptor 0 (the p64+0x5100 transpose, needs only the WAR) and
   descriptor 1 (p96+0x4700, needs the RAW) — would let the `ACT[62]`/`ACT[63]` swap go through and
   is worth ~0.38 us on the chain feeding `S[39]`/`S[24]`. Needs a new block id, a new semaphore and
   a second trigger; `json.dumps(J, separators=(",",":"))` reproduces `SP0.json` byte-identically
   (assert that first). Highest ceiling, highest effort.
3. **The 35.98 us blk19 gap (0.59)** is Vector-bound through `DVE[94..97]`; it only opens if
   candidate 1 frees `DVE[96]` (`CAST` p64+0x4108, 222 ns, has an `S[6]` update so it needs the
   stand-in). Combined with hoisting `ACT[71]` over the `ACT[70]` fence and swapping
   `ACT[72]`/`ACT[73]` it is worth ~0.26 us — but only as a 4-slot package, and `DVE[97]`'s sources
   must be checked against `ACT[72]`'s p0+0xe000 output before the swap.
4. **blk3's 1.04 us scatter** (iter-1 §6.1) — still the single slowest queued transfer and still a
   data-layout change.
5. **Converting the sem-17 output-store row to a queued trigger** (iter-2 §8.2): would remove 663 ns
   from the tail's critical path, not the 1.348 us flight. Still blocked on the same unknown — all
   26 descriptors in `SP0.json` are `{"from":"SB","to":"SB"}`, so there is no DRAM-side descriptor
   in this NEFF to copy from. Hang-prone; keep a `cp save/kernel.neff kernel.neff` ready.

## 8. Method notes
* **Three of eight edits were refuted, and every refutation was readable only because the comparison
  config was re-measured in the same few minutes.** `save/kernel_{start,B,B2,C,D,E,F,G}.neff` made
  each revert one `cp` and each refutation cost ~8 minutes.
* **The device drifted ~0.4 us DOWNWARD across this 2 h session** — iter-2's unmodified bytes read
  41.074 at the start, 40.985 in the middle and **40.677** at the end. That is 0.8× the size of the
  whole iteration's win. **At this kernel size, a single 7-capture median is not evidence.** What
  worked: measure A, measure B, measure A again (or A/B/A/B), and only believe a Δ that repeats.
  Edit F (−0.19) was only believable because three interleaved D/F pairs all went the same way.
  **Edit G (−0.14) needed FIVE pairs**: the first two disagreed (−0.13 then +0.07) and the pooled
  medians said G was 0.03 *worse*, because the device drifted up between them. Alternating which
  config goes first in each pair is what made it readable — 4 of 5 pairs favoured G, and G's span
  was −0.29 with every landmark −0.26 to −0.29. **When the metric and the span disagree, add pairs;
  do not pool medians taken minutes apart.**
* **Decide with the landmark table, not with the metric.** `an.py` printing
  `S[43] S[14] S[4]#32 S[39] S[24]#32 S[17]` for two profiles side by side told me *which half of
  an edit worked* every single time — B (+0.20 on one branch, −0.23 on the other), C (+0.98 on
  blk14), E (+1.10 on `ACT[58]`), F (−0.05 on every landmark despite −0.19 on the metric), G (−0.27
  on every landmark). A single metric number would have told me only "better" or "worse".
* **`slack = start − gate` is the whole search.** ~0.02 us ⇒ gate-bound, nothing to win from
  reordering. >0.15 ⇒ something on the same engine is in front of it, and that something is the
  edit. Both kept DGE rehosts and the one refuted swap were found by sorting one column.
* Span vs metric: span −0.49 ↔ metric −0.62 (B→D), span +0.33 ↔ metric +0.43 (E), span −0.06
  ↔ metric −0.19 (D→F), span −0.29 ↔ metric −0.14 (F→G) — the ratio is **not** stable in either
  direction. **Still do not substitute one for the other** — use the span and the
  landmarks to confirm an edit did what you designed, the metric to decide whether to keep it.
* Free semaphores after this iteration: 0, 1, 7, 18–23, 40, 41, 42, **52**–55, 61–63.
  **2 is the runtime barrier — do not touch.** 47/48/49 are iter-2's DVE copies, **50 is edit F's
  and 51 is edit G's**.
* Device cost: 13 verifies (0 failures, every run bit-identical), 25 profiles, no hang, no silently
  wrong output. A verify is ~1 min, a 7-capture profile ~1.5 min, +0.5 min for the parquet ingest.
* Budget shape that worked: 25 min rebuilding tools + reading iter-1/iter-2 notes, then six
  ~12-minute cycles of (landmark table → one edit with asserted decode → `semaphores.py` →
  `verify --runs 2` → `profile --repeats 7 --parquet` → landmark table again), then 20 min of
  interleaved A/B to fix the headline number, then notes. **Write the notes before the last
  experiment, not after** — I wrote them at D and then got F for free in the remaining hour.

# iter-4 (ssd_mamba2_1b3_s2048): deleted the per-chunk `state_sb` HBM round trip
# (1920 of 2176 state DMAs), and measured the conversion rate of an **ACT instruction**

**Shipped: 5107.0 -> 4856.3 us (-4.91%). `verify.py` VERIFIED, `max_abs_err` 5.234e-07 /
3.334e-07 — byte-identical to iter-3 (the edit removes redundant data movement, it changes no
arithmetic). Members changed: `Activation0.bin`, `DVE0.bin`, `PE0.bin`, `SP0.bin`.
Against iter-0: 7154.9 -> 4856.3, -32.1% (1.473x).**

**Read these four first:**
1. §1 — the recipe. A DGE trigger whose transfer is redundant can be replaced **in place** by an
   `EVENT_SEMAPHORE` that carries `events_extended_update_mode = SEM_ADD_IMM_COMPLETE` on the
   transfer's own semaphore with value 16. **Every wait value in the program stays valid, so no
   threshold has to be remapped.** This works on the device (verified bit-identical). It is the
   general way to delete a DMA from this NEFF.
2. §2 — **lowering DGE wait thresholds does not work on this kernel** (`-> max_abs_err 1.5e-01`),
   even when the remap is index-exact. The state-DMA semaphores are load-bearing fences for other
   data. Refuted twice, with the isolating experiment that proves the data flow was fine.
3. §4 — **removing one ACT compute instruction per unit is worth 189 us of wall (53% of the ACT
   busy it removes).** Measured directly. ACT instruction *count* is the lever; ACT instruction
   *size* is nearly free (§4b). This supersedes iter-2's "ACT throughput is a closed lever".
4. §5 — ACTIVATE **merging is nearly exhausted**: only 40 of 10957 pairs survive a correctness
   check. Do not spend another iteration on it.

Start of iteration: iter-3's NEFF (`git show HEAD:kernel.neff`, md5 `1f7d524f...`, saved as
`kernel_start.neff`). My 5-repeat control: **5107.010 us** (min 5103.6, max 5108.8, spread 0.10%).
ACT 3670 us busy (71.9%), DVE 2441, Pool 2072, PE 2012, DMA 1905, SP 805.

## 0. What the edit is

iter-3 §4 item 1 named this and was right: `block:238` loads `state_sb` (128 part x 256 B) from
`state_hbm` and `block:288` stores it back, **every one of the 1024 (head, chunk) units**. 64 heads
x 64 fp32 = 16384 B per partition fits in SBUF, so the round trip is pure waste.

Three steps, each verified separately on the device:

**(a) Relayout.** The compiler allocates `state_sb` from a rotating pool — **408 distinct SBUF
offsets over the 1024 units**, so every reference has to be repointed. Give head `h` the fixed
address `T(h) = 0x0ec00 + h*256` (iter-3's free hole; `scripts/sbuf.py` still reports 54272 B free
at `0x0ec00`, and this edit proves the first 16 KB of it really is free). 8192 field writes:
6 compute references per unit + the 2 DGE endpoints. **Verified on its own, bit-identical** —
do this as a separate device run, it is the gate for everything after it.

**(b) The redundant DMAs.** With `T(h)` persistent, the only transfers still needed are the
**chunk-0 load** (it is what zeroes `state_sb`; `state_hbm` is zero-initialised by 64 prologue ACT
DGEs) and the **chunk-15 store** (it feeds the `final_state` read-back and so `final_state_out`).
The other 960 loads and 960 stores are dead.

**(c) Replace, do not delete.** See §1.

Result: ACT DGE triggers 1088 -> 128, SP 1088 -> 128. Engine busy after:
ACT 3670 -> 3511, SP 805 -> 279, DMA 1905 -> 1317 (60 MB/run of HBM traffic gone), DVE/Pool/PE flat.

### 0a. How the 1024 units are identified (this is the fiddly part)

```
ACT DGEs: 1088 = 64 prologue zero-init + 1024 block:288 stores.  src = SBUF, dst = var 9 (state_hbm).
SP  DGEs: 1088 = 1024 block:238 loads     + 64 final_state read-backs.  src = var 9, dst = SBUF.
head = dst(ACT) // 32768  or  src(SP) // 32768.
```
* **The prologue inits are NOT the first 64 ACT DGEs in slot order** (head 63's sits at slot 897,
  interleaved with chunk-0 stores). Identify them by `src_var_offset == 0x4200 + 256*head` — the
  16 KB zero tile at `0x4200`. Asserting that the 64 found are a permutation of 0..63 catches it.
* **The final_state read-back is the 17th SP load of its head in slot order**, not one of the last
  64 slots (their head order is scrambled). Take `byhead[h][:16]` as the chunks.
* Pair store `c` with load `c` per head and **assert the two SBUF offsets are equal**. That
  assertion passed for all 1024 and is what proves the chunk indexing.
* Then, for unit `(h,c)`, take the window `[load.start_ts, store.end_ts]` from
  `prof/Instruction.parquet` and collect every SBUF reference to `[S, S+256)` inside it.
  **All 1024 units give exactly the same signature**, which is the check that the whole mapping is
  right:
  `ACT src + ACT dst` (`block:535` in-place scale), `DVE src0 + dst` (`block:541` accumulate),
  `PE src x2` (`block:491`, the two fp32 matmul passes read `state_sb` as the moving operand).
* **The window is ~570 us wide, not ~5 us.** The compiler issues the load for unit `(h,c)` about
  64 units *early* and the store about 64 units *late* (head 0 chunk 0: load at 16.4 us, compute at
  23-29 us, store at 328.6 us, next load at 330.4 us). The state is deliberately parked in HBM for
  one whole head sweep. Two consecutive chunks of one head therefore have **overlapping** live
  ranges, which is why the compiler needs 408 buffers and why `T(h)` has to be *persistent* rather
  than double-buffered.

## 1. THE RECIPE THAT WORKED: `SEM_ADD_IMM_COMPLETE` in the extended events block

A DGE trigger's `semaphore`/`sem_increment` (always 16 here) is applied by the **DMA**, not by the
host engine, so simply deleting the slot removes 16 increments and every `>= 16k` threshold behind
it shifts. Instead, replace the trigger **in place** with an `EVENT_SEMAPHORE`
(`v2.NEURON_ISA_TPB_CTRL_ES_STRUCT`, same 64-byte slot) built from any existing `EVENT_SEMAPHORE`
on that engine, with

| field | value |
|---|---|
| `events_wait_mode/_wait_idx/_semaphore_value` | **copied from the trigger** (keeps the upstream ordering the trigger had) |
| `events_update_mode`, `events_update_idx` | 0 |
| `events_extended_wait_*` | 0 |
| `events_extended_update_mode` | **`SEM_ADD_IMM_COMPLETE`** (21) |
| `events_extended_update_idx` | the trigger's `semaphore` |
| `events_extended_sem_update_value` | the trigger's `sem_increment` (16) |

**No wait value anywhere in the program changes.** The slot cost drops from 527-589 ns of engine
*compute* time to ~64 ns on the *sequencer* (which overlaps compute), and the DMA never runs.

* `events_extended_*` exists on `CTRL_ES_STRUCT` (and 7127 other slots) but **not on `ACTIVATE`** —
  an ACTIVATE cannot carry a second wait or an extended update. Checked.
* **`semaphores.py` does not model extended updates.** It prints
  `1088 max_wait / 128 total_inc  <-- wait exceeds increments` for sems 40-71 after this edit.
  That is a **false alarm**; the device runs and verifies. Do not let it talk you out of the edit,
  and do not "fix" it by lowering waits (§2).
* iter-3 §1a's rule still applies and is satisfied here: the increment is not ordering a *compute*
  write, it stands in for a DMA that no longer exists.

## 2. REFUTED TWICE, WITH THE EXPERIMENT THAT ISOLATES IT: do not remap DGE wait thresholds

First attempt: delete the 1920 slots and remap every wait on sems 40-71 by
`new = 16 * (number of kept transfers among the first v/16)` — the index-exact remap. 2226 values
rewritten, `semaphores.py` clean, no hang.

| variant | result |
|---|---|
| delete 960 stores + 960 loads, remap | `max_abs_err` **1.472e-01**, cosine 0.897 |
| delete only the 960 loads, remap | `max_abs_err` **1.279e-01**, cosine 0.978 |
| **keep every slot and every semaphore, only point the 960 redundant loads' `dst` at a scratch SBUF block** | **VERIFIED, bit-identical** |

The third row is the whole point: it has the identical *data flow* to the deleted version
(`T(h)` persists, the loads write nowhere) and changes **no** synchronisation. It passes. So the
data-flow reasoning in §0 is correct and **the failure is caused only by lowering the thresholds**.

Why: sems 40-71 are used by the compiler as general progress fences, not just as "my transfer
landed". 4342 slots wait on them — 1115 SP `EVENT_SEMAPHORE`, 1069 ACT DGE, 1035 SP DGE, 819 PE
`LDWEIGHTS`, 205 PE `EVENT_SEMAPHORE`, 69 Pool, 17 DVE, 7 ACT. Mapping "wait for transfer k" down
to "wait for the latest surviving transfer <= k" moves a fence up to 60 transfers earlier and some
*other* buffer gets reused early. All waits are `WAIT_FOR_SEM_GE_IMM` and all values are multiples
of 16; nothing outside the DGEs ever updates 40-71. **Carry forward: on this kernel, change DGE
semaphore supply only by preserving it exactly (§1), never by renumbering the demand.**

## 3. Measurement of the shipped edit

| | median us | min-max | spread |
|---|---|---|---|
| iter-3 NEFF (control, 5 repeats) | 5107.010 | 5103.6-5108.8 | 0.10% |
| shipped (5 repeats) | **4856.278** | 4854.9-4857.0 | 0.04% |
| shipped (1 repeat, separate session) | 4856.123 | | |

-250.7 us, -4.91%. The spread is 2-5 us, so this is ~50x the noise.

## 4. THE MEASUREMENT THAT MATTERS FOR iter-5: what an ACT instruction costs in wall time

Two destructive probes on the shipped NEFF (numerically wrong on purpose, profiled only; ~2 min
each). These are cheap and they are the only honest way to size an ACT edit before building it.

| probe | ACT busy | wall | conversion |
|---|---|---|---|
| shipped | 3509 | 4856.3 | - |
| **replace the 1024 `state_sb *= exp_cs_last_N` ACTIVATEs by `EVENT_SEMAPHORE`** (1 fewer ACT compute instruction per unit) | 3152 (-357) | **4667.5 (-188.8)** | **53%**, i.e. **184 ns of wall per ACT instruction deleted** |
| set `num_elem[0] = 1` on **all 10957** ACTIVATEs (64 elements -> 1) | 3080 (-429) | 4559.5 (-296.8) | 69% |

### 4a. ACT instruction count is the remaining bottleneck
ACT is 72.3% busy, the highest of six engines (DVE 50.2, Pool 42.7, PE 39.6, DMA 27.1, SP 5.7).
Its compute occupancy is now essentially all `ACTIVATE`: 10957 slots, **348 ns mean**, union of
intervals 3507 us of a 4862 us span, 1741 us of gaps spread over ~10000 gaps of ~170 ns (handoff
latency, not a few big stalls). 10.7 ACTIVATEs per unit; 1024 units; 4.74 us per unit of wall.

### 4b. An ACTIVATE's cost is ~90% fixed per-instruction overhead
Shrinking every ACTIVATE from 64 elements to 1 took the mean from 348 ns to ~310 ns. **Element
count is nearly free; the instruction is not.** So (i) the only way to cut ACT is to *reduce the
number of ACT instructions*, and (ii) any merge of two ACTIVATEs into one double-length one is
worth almost the whole 348 ns — see §5 for why that is nonetheless a dead end.

### 4c. This revises iter-2's and iter-3's conversion table
iter-2 measured "-97 us ACT busy -> -1.6 us wall" and concluded ACT throughput was closed.
In the current NEFF it is 53-69%. The regime changed when the 960 ACT DGE triggers left the
stream. Re-measure a conversion rate every iteration; they are not stable.

## 5. CLOSED: merging adjacent/contiguous ACTIVATEs is exhausted

Scan of the shipped `Activation0.bin` (10957 ACTIVATEs, all `num_elem[0] = 64` except the merged
ones from iter-2). A pair is mergeable only if the config matches on
`activation_func, num_active_channels, in/out_dtype, scale_src, bias_src, accum_op,
scale_value_imm_ptr, bias_imm_ptr, num_elem, step_elem, perf_mode, activation_table_select`
and both `src` and `dst` tiles are contiguous.

| | count |
|---|---|
| adjacent in the stream, same config | 137 (16 forward-contiguous, 62 reverse, 59 non-contiguous) |
| 8046 | adjacent pairs whose **config differs** |
| contiguous, same config, stream distance <= 12 | 627 |
| ... that also pass "nothing on any engine touches the earlier tile between the two slots" | **42** |
| non-overlapping selection | **40** |

40 merges is ~6 us — below the useful threshold. The 585 rejections are real: these ACTIVATEs are
**in place** (`src == dst`), so the earlier tile is read by the PE/DVE chain between the two slots
and the write cannot be postponed. I also shipped the 40 to the device as a check: it **FAILED**
(`max_abs_err 8.2e-03`), so the between-window test is not even sufficient. Do not retry.

The attractive-looking target was `Y_intra *= exp_cs_col` (`block:477`) and
`Y_off *= exp_cs_col` (`block:504`): identical config, one consumer (`block:550`
`y_chunk = Y_intra + Y_off`), and merging them would be bit-exact. iter-2 already merged the 303
units where the two tiles happened to be contiguous. For the rest both tiles would have to be
relocated into one 512 B block (the machinery in `iter-4-apply.py` can do it), and the above says
even then the in-place read between the slots blocks it.

## 6. BUILT AND MEASURED BUT NOT SHIPPABLE: an ACT multiply moved to DVE costs 1 ulp
## and the error budget is already spent

This was going to be iter-5's item 1 and it is now mostly settled, in both directions.

**The mechanism works.** A `TENSOR_SCALAR_PTR_ARITH_OP` (opcode `0x44`,
`v2.NEURON_ISA_TPB_S3D3_TS_STRUCT`) **synthesised from a zeroed 64-byte slot** runs on **DVE**,
reads **PSUM**, and takes a per-partition scalar from `imm0_imm_ptr` with
`imm0_src = POINTER_IMMEDIATE`. This NEFF contains no DVE `TENSOR_SCALAR` to copy from, so it has
to be built field by field; `isa.Instruction(bytearray(64), isa.STRUCT_OF[0x44])` plus `.set()` on
every field is enough, and `neff.py pack` accepts it. Fields that matter:
`header_inst_word_len = 16`, `accumulator_cmd = IDLE`, `op0 = MULT`, `op1 = BYPASS`,
`reverse_operands = NONE`, `imm1_src = INSTRUCTION_IMMEDIATE` with `imm1_imm_arith_fp32 = 0.0`
(raw bytes 44:48), `num_active_channels = 128`, and `step_elem`/`num_elem` are **3-entry** arrays
here where `COPY`'s TR struct has 4 — copy only the first three.
First proof: rewriting 4 (then 599) DVE `COPY`s as `TENSOR_SCALAR_ARITH_OP` (`0x43`) computing
`x * 1.0` verified **bit-identical**.

**The edit.** `Y_intra = copy(Y_intra_psum)` (`block:476`, DVE) followed by
`Y_intra *= exp_cs_col` (`block:477`, ACT `ACTIVATE func=128` with `scale_src=POINTER_IMMEDIATE`,
`bias=0`) is algebraically one multiply. Rewrite the DVE `COPY` as
`TENSOR_SCALAR_PTR(dst=Y_intra, src=Y_intra_psum, imm0_ptr=<the ACT slot's scale_value_imm_ptr>,
op0=MULT)` and replace the ACT slot in place by an `EVENT_SEMAPHORE` that waits
`sem6 >= cum6[k+1]` (the DVE slot's own increment — require the `COPY` to carry
`SEM_INC_COMPLETE` on 6) and keeps the ACT slot's own sem-4 update. **599 of 932 sites qualify**
(303 are iter-2's already-merged double-length ACTIVATEs, 30 have another engine touching the tile
in between). 8 sites: **VERIFIED bit-identical**, so the whole handshake is sound —
no sem-4 renumbering, and `sem6` orders the ACT placeholder's increment after the DVE write, so
iter-3 §1a is respected. The DVE `COPY`'s existing `sem5` wait already covers the new `exp_cs_col`
dependency: the matmul it waits for consumes `X_scaled`, which ACT produces *after* `exp_cs_col`.

**Why it is not shipped: 599 sites give `max_abs_err 1.588e-06` against a 6.2e-07 limit**
(256%, cosine still 1.000000). The deviation is ~0.4 ulp of the y magnitudes, i.e. **ACT's scalar
multiplier and DVE's round the identical fp32 product differently**, and ACT's happens to be the
one that matches torch. 8 sites were below the noise of that; 599 are not.
**So: moving any fp32 multiply from ACT to DVE costs ~1 ulp per moved site.** With iter-1's
`exp(-x)` rewrite already using 84% of the limit there is no room. A subset that fits the budget
is worth ~0 (the gain is linear in the site count).
This closes the whole "rebalance ACT onto DVE" family *unless* a future iteration first buys error
headroom back. If it ever does, the prize is large and the code is written
(`iter-4-fold-NUMERICS.py`, which reproduces the 599-site edit): 599 sites = ~110 us (-2.3%), and
all six per-unit pointer-scalar multiplies (`dtx` `:454`, `X_scaled` `:463`, `Y_intra` `:477`,
`Y_off` `:504`, `dtx_decayed` `:514`, `state_sb` `:535`) = ~1.1 ms, with DVE (50% busy, 2817 us of
gaps) becoming the new constraint after 2-3 moves per unit.

## 7. Ranked for iter-5

Every ceiling below uses the **184 ns of wall per deleted ACT instruction** from §4. §6 means a
candidate is only worth building if it is **bit-exact**; anything that re-rounds an fp32 value is
dead on arrival.

1. **`exp_cs_last` (`block:226`) is `exp_cs_col[127]` — ATTEMPTED, 4 device runs, NOT SOLVED.**
   `exp_cs_col = exp(cs_col)` and `cs_col = nc_transpose(cs_row)` (`ssd_utils.py:55-120`), so
   `exp_cs_col[Q-1]` and `exp_cs_last = exp(cs_row[Q-1])` are `exp` of the identical fp32 number:
   the nch=1 `ACTIVATE` is a recomputation and deleting it is **bit-exact**, unlike §6. Worth
   1024 x (~200 ns these cost) x 0.53 = **~100 us (-2%)**, and it also shortens the serial head.
   The plan: point the broadcast's `STREAM_SHUFFLE`s at `exp_cs_col` partition 127 instead of at
   `exp_cs_last`, then neutralise the ACT slot.
   **Identification is done and solid** (`iter-4-last.py`): the 1024 `ACTIVATE func=7 nch=1
   num_elem=1` are exactly `exp_cs_last`; the 1024 `ACTIVATE func=7 nch=128 num_elem=1
   scale_value_imm_fp32 == 1.0` are exactly `exp_cs_col` (**the field is
   `scale_value_imm_fp32`, not `scale_imm_arith_fp32`**; `bias_imm_fp32`, likewise).
   Pair each `exp_cs_last` with the nearest preceding `exp_cs_col` in ACT slot order ->
   **859 of 1024 sites** pass the full check (154 have no reader left, 5 have a non-shuffle reader,
   5 fail the placeholder's "previous sem-4 increment is at or after `exp_cs_col`" rule).
   **What the 4 runs established about `STREAM_SHUFFLE`, none of which is in `neff_doc`:**
   * The ISA semantics are documented in the installed compiler, not in `neff_doc` — read
     `inspect.getdoc(neuronxcc.nki.isa.nc_stream_shuffle)`. `shuffle_mask[i]` = which input
     partition **within the 32-partition quadrant** output partition `i` copies from; 255 = leave
     unmodified; `num_active_partitions = ceil(max(src_par, dst_par)/32)*32`, and for 32 the
     `src_start`/`dst_start` partitions must each be 0/32/64/96.
     `nkilib/core/utils/stream_shuffle_broadcast.py` shows the kernel calls it with
     `shuffle_mask = [0]*32`, which is why every mask in this NEFF is all-zero.
   * **`shuffle_mask` lives in the `immediate_uint8[0:32]` of the `LOAD_MASK_SELECT`
     (`common.NEURON_ISA_TPB_CTRL_IM_STRUCT`) immediately in front of the shuffle** — one per
     shuffle. **Proved**: writing `[31]*32` there and changing nothing else moves the output
     (`max_abs_err` 3.3e-07 -> 1.8e-02). `mask_enable` on the shuffle is a *different* thing (the
     `mask=` predication parameter) and must stay 0.
   * **The partition field of an SBUF address is 5 bits, bits 20..24, in units of 4 partitions**
     (so partitions 0,4,...,124 only). Bits 25+ are the PSUM/space marker: a field of 96 gives
     `(96<<20)>>25 = 3` and the run dies instantly with
     `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR`. So `field = partition/4`, max 31, and
     **partition 127 cannot be a `src_start`**. iter-3's "units of 4 partitions" is confirmed;
     the 5-bit width and the marker overlap are new.
   * ACT and DVE agree on the encoding: ACT writes `exp_cs_last` to `dst p=8 off=0x9000` and the
     shuffles read `src p=8 off=0x9000`.
   * **The repoint encoding is confirmed correct by the compiler itself.** I compiled a throwaway
     probe kernel (allowed — the loop forbids recompiling *this* kernel, not writing a probe) with
     `nisa.nc_stream_shuffle(src=col[96:128,0:1], dst=out[i*32:i*32+32,0:1],
     shuffle_mask=[31]*32)` x4 and decoded its NEFF: `neuronx-cc` emits
     `src = 0x01837f00` (field 24, offset 0x37f00), `dst` fields 0/8/16/24, mask `[31]*32`,
     `mask_enable = 0` — **byte for byte the form my edit builds**. So "field 24 + mask 31 reads
     partition 127" is right and is not the bug.
     Recipe: `cd /tmp/probe && PATH=/home/ubuntu/workspace/.venv/bin:$PATH python p.py` with
     `@nki.baremetal(save_neff_name=..., artifacts_dir=...)` — **`neuronx-cc` is not on PATH by
     default**, it lives in `/home/ubuntu/workspace/.venv/bin`. Compiling and decoding works;
     *running* the probe returned all zeros even for the kernel's own
     `shuffle_mask=[0]*32` idiom, so the baremetal I/O plumbing needs work before the probe can be
     used as a numeric oracle. Decode-only is enough for encoding questions.

   **WHERE IT STANDS — the remaining bug is the PAIRING, and it is diagnosed.**
   The identity and the encoding are both certain; what is wrong is deciding *which* `exp_cs_col`
   belongs to a given `exp_cs_last`. Measured on the shipped NEFF:
   | pairing | result |
   |---|---|
   | nearest **preceding** `exp_cs_col` in ACT slot order, 8 sites | `max_abs_err` **5.6e-01**, cosine 0.899 |
   | nearest **following**, 8 sites | **hang** (`NRT_TIMEOUT`) — see the guard below |
   | nearest **following** + schedule guard, 8 sites | `max_abs_err` **4.6e-02**, cosine 0.998 |

   * **ACT does not keep the source order.** The source computes `exp_cs_col` *before*
     `exp_cs_last`, but in the stream the `nch=1` `exp_cs_last` often comes *first*
     (slots 7/8/9 = last/col-1.0/col+1.0). So "nearest preceding" picks the previous head's decay.
   * **And neither direction is reliable**: the slot distance from a `last` to the next `col` is
     3-15 slots and the *time* distance is spread 0-5+ us with no gap, while units are only 4.7 us
     apart. The two units' instructions interleave, so **slot order cannot identify the unit.**
     This is the same trap as iter-3 §1b.
   * **The deadlock**: deleting `exp_cs_last` means the broadcast has to wait for `exp_cs_col`
     instead, i.e. raise the sem-4 wait on the group's `LOAD_MASK_SELECT` (the 1st LMS of a group
     carries `wait sem4 >= k`, the other three carry `NONE`; the shuffles themselves carry `NONE`).
     Raising it blocks DVE in front of work ACT needs to reach `exp_cs_col` -> `NRT_TIMEOUT`.
     **The guard that fixes it**: only convert a site where the group's first LMS already starts
     after the `exp_cs_col` ACTIVATE ends in the measured profile
     (`min(TS['DVE'][s-1][0] for s in shuffles) >= TS['Activation'][ci][1]`). With it, no run hung.
     It costs sites: 859 -> 379 before the guard, fewer after.
   * **The pairing to implement (static, no device time)** — join on data flow, not slot order:
     `exp_cs_last.src_off - 127*4` = `cs_row` base -> the PE `MATMUL` (the `nc_transpose`,
     `num_active_rows = 1`) whose moving operand is that `cs_row` -> its PSUM dst -> the ACT
     `prod` (`func=128` copy, PSUM src) reading that PSUM -> `prod.dst` = `cs_col` ->
     the `exp_cs_col` is the `func=7 scale=+1.0` ACTIVATE whose **`src == prod.dst`**
     (true in both of iter-1's variants). Disambiguate the reused addresses with the profile
     timestamps, exactly as `iter-4-apply.py` does for `state_sb`. Then re-run the 8-site test;
     it should come out **bit-identical**, because the two values are `exp` of the same fp32 input.
   * Realistic prize after the guard: maybe 200-300 sites = **40-60 us (-1%)**, which is only
     ~10x the 2-5 us spread. Decide whether that is worth the remaining risk before building it.

2. **Pool's 1024 `dt_col` DGE triggers** (`block:203`, 655 ns each, 671 us of Pool) move 4 bytes
   per partition in 128 descriptors. Prior kernels measured a DGE row's host cost as **fixed
   ~600 ns, independent of payload and of descriptor count**
   (`neff_doc/experiments/gdn_cte_b1_s128_d128_lnc1/logs/notes/iter-6.md` §2, and the 64->32
   descriptor split in `gdn_cte_b1_s256_d128_lnc1` iter-5 §7 that cost *more*). So **merging N
   heads' `dt_col` into one trigger divides the trigger count by N** — relocate `dt_col` so head
   h sits at `BASE + h*4` and use `num_elem [128, N]`, `src_step_elem[1] = seqlen`,
   `dst_step_elem[1] = 1`. Ceiling: Pool compute 2072 -> ~1400 us. **But Pool is only 42.7% busy
   with 2986 us of gaps, so expect little**; measure first with a destructive probe (set
   `num_elem[0] = 1` on all 3169 Pool triggers and profile — if the wall does not move, kill it).
   *This also refutes iter-2's "the trigger cost is linear in the descriptor count", which I could
   not reproduce and which no measurement in this kernel supports (every DGE here has exactly 128
   descriptors).*
3. **Software-pipeline the serial head** (iter-3 §4 item 2). Still the largest lever in principle
   and still the most expensive to build. ~3 us of the 4.74 us unit is the `dt_row -> log_decay ->
   cs_row -> transpose -> cs_col -> 2 exps -> exp_cs_last -> broadcast -> decay_factor` chain,
   which depends only on `dt_row` and `A_head`.
4. **Exactly-duplicate ACTIVATEs: checked, there are none.** 7343 of the 10957 ACTIVATEs fall into
   514 groups with identical `activation_func`/shape/`src`+`dst` address/scale pointer, which looks
   like 6829 deletable recomputations — but consecutive members of a group are **2-8 us apart**,
   i.e. they are *different units* reusing the same SBUF addresses and the same scale-pointer
   address with different values in them (iter-3 §1b's trap, again). Do not chase this without a
   full writer analysis; the only true recomputation in this kernel was the double broadcast
   (iter-3) and `exp_cs_last` (item 1).
5. Do **not** redo: anything in §2, §5 or §6 here, iter-3 §3/§3b, iter-2 §4/§1.

## 8. Tool notes (additions to iter-3 §5)

* `cp baseline/kernel.neff kernel.neff` reverts to **iter-0**. Save the starting NEFF first:
  `git show HEAD:kernel.neff > kernel_start.neff`, and keep each verified win
  (`cp kernel.neff kernel_win1.neff`) so a failed follow-up costs one `cp`.
* A **destructive probe is the cheapest tool in this loop**: break the arithmetic on purpose
  (shrink `num_elem`, swap a compute slot for an `EVENT_SEMAPHORE`), run `profile.py --repeats 3`,
  read the number, restore. ~2 minutes, and it sizes an edit before you build it. Both numbers in
  §4 came from probes; both saved me from building something worthless (§5).
* **`isa.Instruction` API**: `ins.name` (not `.opcode`, which is the raw byte), `ins.fields()` is a
  method, `ins['field']` reads, `ins.set('field', v)` writes (accepts enum *names*),
  `ins.as_dict(symbolic=False)` for raw ints. Array fields read/write as lists.
* SBUF addresses: all the `*_mem_pattern_start_addr_addr_immediate` and DGE
  `*_start_addr_addr_immediate` fields keep the byte offset in the **low 20 bits**; repoint with
  `ins.set(k, (ins[k] & ~0xFFFFF) | new_off)`, which updates every aliased view of the union
  (`_var_offset`, `_table_offset`, `_reg_reg_hi`, ...) consistently. Skip a reference whose
  `*_addr_reg_marker` is non-zero — that is PSUM.
* `prof/Instruction.parquet`'s `compiler_pc` has NaNs; filter with `.notna()` before `int()`.
* Scripts are copied next to this file and reproduce the shipped `kernel.neff` byte for byte
  (`kernel_win1.neff` md5 `1115f0b0c95ba20a6c8d3dec5ecfec49`):
  `iter-4-common.py` (stream loader), `iter-4-apply.py` (**the whole edit**; run with argument
  `nop`), `iter-4-merge-REFUTED.py` (§5), `iter-4-try.sh` (revert -> unpack -> edit -> pack ->
  `semaphores.py` -> `verify.py` in one ~55 s call). Run order: put them in `scripts/` as
  `common.py` / `apply.py` / `merge.py`, `neff.py unpack kernel.neff work/`,
  `profile.py --parquet prof`, then `apply.py nop`, then `neff.py pack work/ kernel.neff`.
  `apply.py` also accepts `scratch` (§2's isolating experiment), `delete`, `noload`, `nostore`
  (the refuted threshold remaps) — keep them, they are the falsifiers.

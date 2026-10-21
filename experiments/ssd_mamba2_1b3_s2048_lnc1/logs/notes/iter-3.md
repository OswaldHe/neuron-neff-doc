# iter-3 (ssd_mamba2_1b3_s2048): the kernel's NKI source is on this machine; killed one of the two
# redundant `exp_cs_last` broadcasts; measured that **DVE work converts to wall at ~8%**

**The four things to read if you read nothing else:**
1. The NKI source is on this machine (section 0) -- with the per-unit instruction table.
2. `EVENT_SEMAPHORE` runs on the **sequencer** and increments *before* the preceding compute
   instruction has finished writing (section 1a). This invalidates ch 10's R2 shortcut and cost
   me four device runs.
3. ACT may **not** read a PE matmul's PSUM output at all, sole reader or not (section 3b).
4. The 35.6 MB/direction of "spill" is the kernel's own per-chunk `state_sb` round-trip through
   HBM, and there are **54272 free bytes per SBUF partition** to hold all 64 heads' states
   instead (section 4 item 1). That is the biggest lever left.

Start of iteration: iter-2's NEFF (`git show HEAD:kernel.neff`, sha256 `1bd995d8...`).
My own 5-repeat control: **5126.314 us** (min 5124.6, max 5131.1, spread 0.13%).
ACT 3669 us busy (71.6%), DVE 2701, Pool 2072, PE 2013, DMA 1875, SP 803.

Shipped: 545 of the 1024 per-unit `broadcast_scalar_to_column(exp_cs_last, 128)` *second* calls
replaced by one 1-element 128-partition `COPY` from the first call's output. 4 `STREAM_SHUFFLE`
-> 1 `COPY` + 3 `EVENT_SEMAPHORE` per site. Only `sg00/DVE0.bin` changed. `verify.py` VERIFIED,
`max_abs_err` 5.234e-07 / 3.334e-07 -- **byte-identical to iter-2** (the two broadcasts write the
same value, so this is exact, not an approximation).

## 0. READ THIS FIRST: the NKI source for this kernel is on the machine

```
/home/ubuntu/workspace/gdn-bench/vendor/commit_92d11f6/scan/ssd_block.py      # the kernel
/home/ubuntu/workspace/gdn-bench/vendor/commit_92d11f6/scan/ssd_utils.py
/home/ubuntu/workspace/.venv/lib/python3.12/site-packages/nkilib/core/utils/stream_shuffle_broadcast.py
```
`prof/Instruction.parquet`'s `nki_source_location` points straight at them. Two iterations were
spent reverse-engineering structure that is 150 lines of readable Python. **Read the source before
profiling.** (Recompiling is still out of scope -- but knowing what the kernel *means* is what
found this iteration's edit, and it is what will find the next one.)

### The per-unit program, from the source (one unit = one (head, chunk), 1024 of them, ~5.0 us each)

`for chunk in range(16): for head in range(64):` -- so consecutive units are **different heads**,
and the state recurrence for one head is 64 units apart. Per unit:

| src line | op | engine | n/unit | cost |
|---|---|---|---|---|
| `block:190,197,203` | dma A_head, dt_row, dt_col | SP/Pool | 3 | |
| `utils:100` | `log_decay = dt_row * A` (1,128) | ACT | 1 | 489 ns, **nch=1** |
| `utils:107` | `cs_row = cumsum` `tensor_tensor_scan` | DVE | 1 | 464 ns, nch=1 |
| `utils:67` | `nc_transpose` cs_row -> cs_col | PE | 1 | LDW+MM, **PE idles 1.44 us after it** |
| `utils:117` | `cs_col = copy(psum)` | ACT | 1 | |
| `utils:119` | `exp_cs_col = exp(cs_col)`, `exp_neg_cs_col` | ACT | 2 | iter-1's rewrite |
| `block:226` | `exp_cs_last = exp(cs_row[-1])` (1,1) | ACT | 1 | |
| `block:227` | `exp_cs_last_Q = broadcast(exp_cs_last,128)` | DVE | **4 shuffles** | 4x147 ns |
| `block:229` | `decay_factor = exp_neg_cs_col * exp_cs_last_Q` | DVE | 1 | 222 ns |
| `block:238` | **`dma state_sb <- state_hbm`  (32 KB)** | SP DGE | 1 | 554 ns issue |
| `block:447` | `dma x_sb <- x_hbm` | Pool DGE | 1 | 667 ns issue |
| `block:454` | `dtx = x * dt_col` | ACT | 1 | 434 ns |
| `block:463` | `X_scaled = dtx * exp_neg_cs_col` | ACT | 1 | 434 ns |
| `block:470` | `Y_intra_psum = CB_masked @ X_scaled` | PE | **2** (fp32 = 2 passes) | |
| `block:476` | `Y_intra = copy(Y_intra_psum)` | DVE COPY | 1 | 213 ns |
| `block:477` | `Y_intra *= exp_cs_col` (in place) | ACT | 1 | 434 ns |
| `block:486` | `memset Y_off = 0` | DVE | 1 | 112 ns |
| `block:491` | `Y_off_psum = C_T @ state_sb` | PE | **2** | |
| `block:498` | `Y_off += Y_off_psum` (reads PSUM directly) | DVE TT | 1 | 222 ns |
| `block:504` | `Y_off *= exp_cs_col` (in place) | ACT | 1 | 434 ns |
| `block:514` | `dtx_decayed = dtx * decay_factor` | ACT | 1 | 434 ns |
| `block:525` | `state_delta_psum = B @ dtx_decayed` | PE | **2** | |
| `block:533` | `exp_cs_last_N = broadcast(exp_cs_last,128)` | DVE | **4 shuffles** | **REDUNDANT** |
| `block:535` | `state_sb *= exp_cs_last_N` (in place) | ACT | 1 | 434 ns |
| `block:541` | `state_sb += state_delta_psum` | DVE TT | 1 | 225 ns |
| `block:550` | `y_chunk = Y_intra + Y_off` | DVE TT | 1 | 222 ns |
| `block:572` | `dma y_hbm <- y_chunk` | Pool DGE | 1 | 678 ns |
| `block:288` | **`dma state_hbm <- state_sb` (32 KB)** | ACT DGE | 1 | 560 ns issue |

`num_hdim_tiles = num_dstate_tiles = 1` for this config (headdim 64, dstate 128), so every loop
in the source body executes once.

### Every DGE transfer in the kernel, by (engine, side, bytes-per-partition, descriptors)

```
('Activation','src',256,128) 1088   state_sb -> state_hbm      (block:288, + 64 final-state)
('SP',        'dst',256,128) 1088   state_hbm -> state_sb      (block:238, + 64 final-state)
('Pool',      'dst',256,128) 1024   x_hbm -> x_sb              (block:447)
('Pool',      'src',256,128) 1088   y_chunk -> y_hbm           (block:572, + 64)
('Pool',      'dst',  4,128) 1024   dt_flat -> dt_col          (block:203)
('Pool',      'dst',512,128)   33   B/C projections            (block:336/341, prologue)
```
Every one is 128 descriptors (one per partition) at 650-680 ns on Pool / 540-560 ns on ACT and SP,
and iter-2 showed the cost is linear in the **descriptor count**, not the bytes. 4353 of the 5377
DGE triggers (**81%**) exist only to move `state_sb` and the x/y tiles, and **2176 of them (the
ACT and SP ones) are the state round-trip alone**.

**Trap for the next iteration:** `state_sb` is `(dstate=128, headdim=64)` and `x_sb` is
`(Q=128, d_size=64)` -- both are 128 partitions x 256 B, so **you cannot tell the state DMAs from
the x/y DMAs by transfer shape**. Separate them by the engine + side table above, or by the HBM
side's `var_id`/`var_offset` against `tensor_map.json`, or by `nki_source_location` in the profile
(reliable for Pool and SP; the ACT DGEs' source attribution is scrambled).

**The 35.6 MB of "spill" traffic each way is not compiler spill.** It is `block:238` +
`block:288`: the kernel writes `state_sb` (128x64 fp32 = 32 KB) to HBM and reads it back **every
chunk**, 1024 times each way. That is what the 1088 ACT `PSEUDO_DMA_DIRECT2D` and the 1024 SP ones
are. Two iterations tried to move those triggers to another engine; nobody noticed they are
avoidable in principle (64 heads x 32 KB = 2 MB would fit in SBUF). See section 4.

## 1. WHAT I SHIPPED: `exp_cs_last` is broadcast to 128 partitions twice per unit

`broadcast_scalar_to_column` (`utils:24`) -> `stream_shuffle_broadcast` -> `nc_stream_shuffle` once
per 32 partitions, so **4 `STREAM_SHUFFLE` + 4 `LOAD_MASK_SELECT` per 128-row broadcast**. The
kernel calls it with the *same* scalar `exp_cs_last` and the *same* 128 rows at `block:227` and
again at `block:533`; the compiler does not CSE across them (separate `nl.ndarray` allocations).
8192 shuffles = 2048 broadcasts = 2 per unit; **1024 of them are recomputation**.

Edit (only `DVE0.bin`): for the second group, replace its 4 shuffles with
* **the last slot** -> `COPY` nch=128, `num_elem=[1,1,1,1]`, src = `(p0 | O1)`, dst = `(p0 | O2)`,
  `wait sem6 >= v`, **keeping the last slot's own events bytes 6:7** (its `SEM_INC_COMPLETE` on 6);
* the other three -> all-zero `EVENT_SEMAPHORE` (`a0 10 00 00` + the slot's original bytes 4:11).

Nothing is repointed, no semaphore count changes, no reader analysis is needed: O2 still ends up
holding the same 128 values. Measured DVE busy 2701 -> 2440 us (**-261 us, -9.7%**).

Wall clock, two independent 5-repeat measurements plus one back-to-back pair in the same session:

| | control (iter-2 NEFF) | 545-pair edit | delta |
|---|---|---|---|
| run A | 5126.314 (spread 0.13%) | 5105.312 (0.13%) | -21.0 |
| run B (paired, same session) | 5128.484 (0.07%) | 5106.287 (0.02%) | **-22.2** |
| run C (final, after cleanup) | - | 5104.743 (0.07%) | - |

Means: control 5127.4, edit 5105.4, so **-22.0 us (-0.43%)**, against a spread of 1-7 us.
Against the iter-0 baseline: 7154.9 -> 5104.7, **-28.65% (1.4016x)**. Real, but small.
`neff_diff`: only `sg00/DVE0.bin`, 2180 of 24525 slots (= 545 x 4).

### 1a. THE TRAP THAT COST ME AN HOUR: `EVENT_SEMAPHORE` runs on the DVE *sequencer*

`prof/Instruction.parquet` has an `execution_unit` column. On Vector:
`COPY / MEMSET / STREAM_SHUFFLE / TENSOR_TENSOR / TENSOR_TENSOR_SCAN` -> **`compute`**;
`EVENT_SEMAPHORE / LOAD_MASK_SELECT / TENSOR_LOAD / COMPARE_BRANCH / SET_ORDERING_MODE` ->
**`sequencer`**.

My first three attempts put the new `COPY` on the group's **first** slot and turned the group's
last slot (the one carrying `SEM_INC_COMPLETE 6`) into an `EVENT_SEMAPHORE`. It ran, it never hung,
and it produced `max_abs_err 8.9e-02`. The reason: **a sequencer `EVENT_SEMAPHORE` increments its
semaphore without waiting for the preceding compute instruction to finish.** So the consumer's
`wait sem6 >= k` was released before the `COPY` had written, every time. Chapter 10.1 rule 5
("instructions complete in program order") is only true *within* an execution unit.

**Rule to carry forward: a semaphore increment that a consumer relies on must sit on a `compute`
instruction. Never move a load-bearing `SEM_INC_COMPLETE` onto an `EVENT_SEMAPHORE`.** This also
means the ch 10 R2 advice "replace the instruction with an `EVENT_SEMAPHORE` that keeps its wait
and its update" is **unsafe when anything depends on that update ordering a compute write**.
(It was safe in iter-1/iter-2 only because those placeholders replaced DGE triggers / waits.)

The same rule constrains the wait side: the `COPY`'s `wait sem6 >= v` must name an increment
carried by a **compute** instruction at or after the first group's last slot. `v` =
`cum6[L+1]` where `L` = the first DVE slot `>= max(G1.slots)` that is in `COMPUTE` and updates
sem 6, and `L` must be `< ` the COPY's slot or it self-deadlocks (that was my first hang,
`NRT_TIMEOUT`). Distribution of `L - max(G1.slots)` over the 545 shipped: 149 at 0, 343 at 1,
46 at 2, 7 at >=3 -- so the wait is tight and costs nothing.

### 1b. How the two groups of a unit are identified (this is the fiddly part)

Destination offsets rotate over ~70 SBUF offsets and the *same* `exp_cs_last` SBUF address is
reused by many units (e.g. `0x809000` for 127 of them), so "two groups with the same src that are
adjacent in DVE slot order" is **wrong** -- it happily pairs unit k with unit k+1.

What works: anchor on the producer.
1. The 1024 ACT slots with `ACTIVATE, activation_func=7, nch=1, num_elem[0]=1` are exactly the
   1024 `exp_cs_last = exp(cs_row[-1])` (`block:226`). Record `(slot, dst_addr, cum4_after)`.
2. For each DVE slot compute `R4[i]` = running max of sem-4 wait values in DVE program order.
3. A broadcast group belongs to unit `k` = the largest producer index with
   `dst == group.src` and `cum4 <= R4[min(group.slots)]`.
   -> 1019 of 1024 units get exactly 2 groups (1 unit got 1, 1 got 3, 1 got 4, 2 groups unmapped).
4. Pair the two groups of a unit, first-in-slot-order = G1.

Filters that mattered, with their counts out of 934 candidate pairs:
* `O2 not an ACT scale pointer` (277): **keep this filter.** Dropping it raises coverage to 816
  pairs and removes 402 us of DVE busy, but the wall went **up** to 5119.6 us (vs 5105.3 for 545).
  The extra 271 pairs are not the `block:227`/`block:533` pair -- they are prologue/other
  broadcasts where the new `COPY`'s sem-6 wait buys a stall instead of saving work. More DVE work
  removed, worse wall: a clean demonstration that *where* you remove DVE work matters more than
  how much.
* `no compute sem6 update between` (111): unreachable with this construction. Would need a new
  increment.
* `O1 rewritten on DVE between G1 and the COPY` (1).

### 1c. Encoding notes
`STREAM_SHUFFLE` and `COPY` share `common.NEURON_ISA_TPB_S4D4_TR_STRUCT`, so the rewrite is
`raw[0] = 0x46` plus three field writes. Field byte offsets in that struct (**different from
ACT's `AC_STRUCT`** -- ACT has src at 16):
`4:12` events (`4` wait_mode, `5` wait_idx, `6` update_mode, `7` update_idx, `8:12` u32 value),
`12:16` src addr, `16:24` src step_elem (4x i16), `24:32` src num_elem (4x u16),
`32` in_dtype, `33` out_dtype, `34` num_active_channels, `35` op_dim,
`44:48` dst addr, `48:56` dst step_elem, `56:64` dst num_elem.
Addresses are `(partition_field << 20) | byte_offset`; for DVE the partition field is in units of
**4 partitions** (a 128-row broadcast is 4 shuffles at fields 0, 8, 16, 24 with `nch=32`).
A (128 partitions, N elements) tile is `nch=128, num_elem=[N,1,1,1]`; a (128,1) column is
`nch=128, num_elem=[1,1,1,1]`.
ACT's per-partition scale operand is `scale_value_imm_ptr` = a bare byte offset (no partition
field) with `scale_src = POINTER_IMMEDIATE`.

## 2. THE MEASUREMENT THAT MATTERS MOST: conversion ratios by engine

"How much wall time does removing 1 us of engine-X work buy?" -- measured on this kernel:

| iteration | what was removed | engine time | wall | conversion |
|---|---|---|---|---|
| iter-1 | 2040 ACT activation-table reloads | -2516 us ACT | -2026 us | **81%** |
| iter-2 | 303 adjacent ACT `ACTIVATE` merges | -97 us ACT | -1.6 us | **1.6%** |
| iter-2 | 208 DGE triggers ACT -> SP | -57 us ACT (+124 SP) | +5.2 us | negative |
| **iter-3** | **545 redundant DVE broadcasts** | **-261 us DVE** | **-21.0 us** | **8%** |
| iter-3 | +271 more DVE broadcasts (816 total) | -402 us DVE | -6.7 us | **1.7%** |

So: ACT ~1.6%, DVE ~8% at the right sites and ~0 elsewhere. **No engine's busy time is the
constraint any more.** 5126 us / 1024 units = 5.006 us per unit; engine busy sums to 12.5 us per
unit over 6 engines, i.e. 2.5 engines busy on average, and every engine idles 30-60%. The thing
that sets 5.0 us is the per-unit *dependency cycle*, and a NEFF edit only converts when it
shortens that cycle.

Issue-interval decomposition (`start_ts[i+1]-start_ts[i]`, sums to each engine's span):

| engine | span | top consumers of the span |
|---|---|---|
| ACT | 5132 | `ACTIVATE` 4038, `DMA_DIRECT2D` 547, `EVENT_SEMAPHORE` 518 |
| PE | 5140 | `MATMUL` 4394 (duration only 2972), `LDWEIGHTS` 506 |
| POOL | 5133 | `DMA_DIRECT2D` 4891 (duration 2077) |
| SP | 5132 | `EVENT_SEMAPHORE` 4096 (duration 237) |
| DVE | 5136 | `STREAM_SHUFFLE` 1435, `TENSOR_TENSOR` 1419, `LOAD_MASK_SELECT` 699, `SCAN` 612 |

Idle (sum of `start_ts[i] - end_ts[i-1]`): ACT 1721, DVE 2894, PE 3019, POOL 3257, SP 4093 us.

## 3. Closed this iteration
* **`broadcast_scalar_to_column` dedup**: done for 545 of 1024. 111 more are blocked on the
  "compute increment" requirement, 277 more are a different kind of pair and make the wall worse.
  Remaining headroom here is ~115 us of DVE busy, i.e. ~10 us of wall. Not worth another pass.
* **Replacing a compute instruction by an `EVENT_SEMAPHORE` that keeps its update**: unsafe
  whenever a consumer's wait on that update orders a compute write. Section 1a. This closes the
  cheapest-looking form of "delete this instruction" for ACT/DVE/PE/Pool compute slots.
* **`O2 not ACT scale pointer` pairs**: refuted as a win (816-pair run, 5119.6 us vs 5105.3).

## 3b. CLOSED BY HARDWARE: ACT may not read a PE matmul's PSUM output

The obvious "merge two chained steps" edit in this kernel is to fold the PSUM->SBUF copy into the
ACT scale that immediately follows it:

| | before | after |
|---|---|---|
| `Y_intra` (`block:476`,`477`) | DVE `COPY` psum -> Y_intra (213 ns); ACT `Y_intra *= exp_cs` | ACT `Y_intra = exp_cs * Y_intra_psum` |
| `Y_off` (`block:486`,`498`,`504`) | DVE `MEMSET` 0; DVE `TT` `Y_off = Y_off + psum` (222 ns); ACT `Y_off *= exp_cs` | ACT `Y_off = exp_cs * Y_off_psum` |

Both are **exact** (`0.0 + x == x` in fp32) and I found 1199 sites that pass a full safety
analysis (562 `COPY`, 637 `TT`): the DVE writer is the latest PSUM-reading writer of the ACT
instruction's buffer, the PSUM operand has step 1 / 64 elements matching ACT's own pattern,
nothing else reads the buffer or the PSUM region in between, and for the `TT` case the other
operand is the dst and the latest previous writer of it is a `MEMSET` with all value fields zero
covering >= 64 elements. To keep the sem-6 accounting and the "increment must sit on a compute
instruction" rule (section 1a), the DVE instruction was kept but shrunk to `num_elem[0] = 1` with
its PSUM operand repointed at its own dst, so **DVE no longer reads the PSUM at all** -- only ACT
does.

Both the 1199-site and the `COPY`-only 562-site version fail immediately with
```
INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR / TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION  error_id=8
```
So iter-1's finding generalises and is stronger than it was written: it is not "a *second* ACT read
of a PSUM bank is rejected", it is **ACT may not read a PSUM region produced by one of these
matmuls at all, even as the sole reader.** Those matmuls carry `fp32_mode = LOW_HIGH` and
`psum_accumulate_flags = 1` (each logical `nc_matmul` is 2 `MATMUL` slots accumulating into one
bank), which is the likely reason -- iter-1's *working* ACT PSUM read is the
`transpose_row_to_column` result (`utils:67`/`:117`), whose matmul has `num_active_rows = 1`, and
that read was already on ACT as the compiler emitted it. Working rule: **a PSUM region may only be
drained by the engine the compiler assigned to drain it.** This closes the whole family of
"move the PSUM read to another engine" edits, and with it the cheapest way to delete a
PE->DVE->ACT hop.

Related fact found while scoping it, worth keeping: **only 2430 of the 7233 `MATMUL`s have
`SEM_INC_COMPLETE` on sem 5** (2.37 per unit for 4 PSUM results per unit), so "the sem-5 count at
which `Y_intra_psum` is complete" is *not* derivable from the matmul that produces it. Any plan
that re-points a consumer's wait from DVE to PE has to first map, per PSUM bank address, which
`MATMUL` slot carries the signal.

## 4. Ranked for iter-4

1. **Keep `state_sb` in SBUF and delete the 2048 state DMAs. THE ADDRESS SPACE IS THERE -- I
   checked.** `block:238` loads 32 KB and `block:288` stores 32 KB per unit; 1024 units x 2 =
   64 MB of HBM traffic, **1088 SP DGE triggers (543 ns each = 591 us of SP) and 1088 ACT DGE
   triggers (560 ns = 609 us of ACT)** -- see the DGE table in section 0; 1024 of each are the
   per-chunk round-trip and 64 are the final-state copy, which has to stay. Plus the DMA engine's
   own time. Because the loop is
   `for chunk: for head:`, head h's state is reloaded 64 units after it was stored, so all 64
   heads' states must live in SBUF at once: 64 heads x 64 fp32 = **16384 B per partition**
   (128 partitions x 16 KB = 2 MB total).

   **Free-space map (`scripts/sbuf.py`, no device time): the per-partition window 0..0x38000
   (229376 B) is 69.2% used, and there is a contiguous free hole of 54272 B at offset
   `0x0ec00`** -- 3.3x what is needed -- plus 16380 B at `0x00004`. The map unions, over all
   five engine streams: every `*_mem_pattern_start_addr_addr_immediate` with its
   `num_elem`/`step_elem` extent and `marker == 0` (marker 2 = PSUM, excluded), every
   `scale/bias/imm_imm_ptr` with `*_src == POINTER_IMMEDIATE`, every DGE
   `*_start_addr_addr_var_offset` whose `step_elem[0] == 0x40000` (that is the SBUF side; the
   partition stride is 0x40000 and `elem_size` is the per-partition byte count), and every
   `desc` with `"to"/"from" == "SB"` in the `dma` arrays of `work/payload/sg00/*0.json`
   (2036 of those). Caveat: register/table-based addressing (`*_addr_reg_regnum` non-zero) is not
   resolved, so treat `0x0ec00` as "very likely free" rather than proven.

   Plan: give head h the SBUF offset `0x0ec00 + h*256`, repoint the instructions per unit that
   touch `state_sb` (`block:491` PE `moving` operand, `block:535` ACT in-place scale src+dst,
   `block:541` DVE TT src+dst), keep the HBM load for chunk 0 and the HBM store for chunk 15 of
   each head (so `final_state_out` still works and the zero init still happens), and delete the
   other 2 x 960 DMAs. Prize: ~1.2 ms of DGE issue time on ACT and SP -- both of which iter-2
   proved are on the critical path -- and it *shortens the cycle* rather than moving work.
   Risk: high, many slots. But the ACT DGE triggers are 537-605 ns each and in the profile they
   run **four back to back with nothing else on ACT** (e.g. ACT pc 6926-6929 occupy 2.5 us of one
   5.0 us unit), so this is the only remaining block of ACT time big enough to matter.
2. **The serial head of each unit is 8 instructions on 1 partition out of 128.**
   `log_decay` (ACT, 489 ns, nch=1) -> `cs_row` scan (DVE, 464 ns, nch=1) -> `nc_transpose`
   (PE; PE then idles 1.44 us) -> `cs_col` copy -> 2 exps -> `exp_cs_last` -> broadcast ->
   `decay_factor`. That is ~3 us of latency per unit to produce three 128-element vectors.
   All of it depends only on `dt_row` and `A_head`, which are loaded by DMA at the top of the
   unit -- **it is independent of everything else in the unit and of all other units**. The win
   is to run it one or more units ahead (software pipelining) so it is never on the critical path.
   At the NEFF level that means moving a contiguous run of slots earlier on each of ACT/DVE/PE and
   re-deriving the waits -- ch 10 R4. Expect this to be the largest remaining lever after (1):
   it is pure latency, and latency is what this kernel is made of.
3. **Pool's 3169 DGE triggers eat 4891 us of Pool's 5133 us span** (655 ns duration each, 1543 ns
   of span each). They are only 3 per unit (`block:203` dt_col, `block:447` x in, `block:572`
   y out) and all three are real I/O, so they cannot be deleted -- but iter-2 established the
   trigger cost is linear in the **descriptor count** (`src_num_elem[0]` = 128, one per
   partition), not in bytes. The only way to cut it is to make one transfer cover several units'
   worth of a tensor (e.g. load x for 2 chunks at once: 128 descriptors of 512 B instead of
   2 x 128 of 256 B, halving the triggers). That needs the SBUF layout of the x tile to become
   2-chunk-wide, i.e. the free hole from item 1. Pair it with item 1 if that one lands.
4. ~~Y_intra / Y_off PSUM fusion~~ -- **REFUTED ON THE DEVICE, see section 3b. Do not retry.**
5. Do **not** redo: anything in iter-2's section 4 or section 1, the 816-pair broadcast variant,
   trimming ACT instruction count, or moving a PSUM read to another engine.

## 5. Tool notes (additions)
* **`cp baseline/kernel.neff kernel.neff` reverts to iter-0, not to the iteration's starting
  point.** The starting NEFF is `git show HEAD:kernel.neff`. I lost ~10 minutes applying an edit
  to iter-0's streams with iter-2's profile timestamps, which silently corrupted every
  timestamp-based check (ACT `compiler_pc` indices differ). Save it first:
  `git show HEAD:kernel.neff > kernel_iter2.neff`.
* `compiler_pc` == static slot index 1:1 on **all five** engines of this NEFF (4192/14074/4061/
  14865/24525 pcs for the same slot counts), so decode <-> profile joins are exact. iter-2's
  "the profile pc does not line up" caveat is wrong; what breaks is `nki_source_location`
  attribution for ACT (`ACTIVATE` counts per line come out as 1477/1466/1315 where the source says
  1024), not `compiler_pc`.
* `isa.decode_stream` yields `(index, Instruction)` tuples, not Instructions.
* A one-line test driver pays for itself; mine (`scripts/try.sh LIMIT OFF`) does revert ->
  unpack -> edit -> pack -> verify -> one `RESULT` line, ~45 s pass / ~75 s hang, and two of them
  fit in one 120 s tool call. `tools/profile.py --repeats 5` takes ~70 s, so **two profiles do not
  fit in one call** -- run them in separate calls or in the background.
* The scripts are copied next to this file (they survive with it):
  `iter-3-common.py` (stream loader), `iter-3-bc.py` (find the 2048 broadcast groups ->
  `scripts/groups.json`), `iter-3-edit4.py` (pair + apply; **verified to reproduce the shipped
  `DVE0.bin` byte-for-byte**, md5 `94560d74`), `iter-3-sbuf.py` (the free-SBUF map),
  `iter-3-fuse-REFUTED.py` (the PSUM fusion of section 3b -- keep it as the worked safety
  analysis, but it hits `PSUM_COLLISION`). Run order: put them in `scripts/` as
  `common.py`/`bc.py`/`edit4.py`/`sbuf.py`, `neff.py unpack kernel.neff work/`,
  `profile.py --parquet prof`, then `bc.py`, then `edit4.py`, then `neff.py pack work/ kernel.neff`.
* `scripts/` also held a 6-line driver (`try.sh LIMIT OFF`) doing revert -> unpack -> edit ->
  pack -> verify -> one `RESULT` line. Rebuild it first; it is what made 12 device
  experiments affordable.

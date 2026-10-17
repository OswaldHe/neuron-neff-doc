# iter-6 — the DMA transposes are MANDATORY (proved), the staging-fusion lever is EXHAUSTED, and iter-5's access map had a hole that invalidated two "candidate" levers

## Result: 14032.7 us, **-0.11% vs my starting NEFF** (14048.6) — i.e. noise. -5.05% vs iter-0.

| NEFF | total_exec_time_us | captures |
|---|---|---|
| iter-5 NEFF = my starting point | **14048.6** (med 3) | spread 0.0x% |
| + 7 provably-dead DGE transfers retired — **shipped** | **14032.7** (med 3) | 14028.7 14032.7 14036.7 |
| the same shipped NEFF, independent second capture set | **14027.7** (med 3) | 14022.5 14027.7 14038.5 |

`verify.py --runs 2`: PASS, bit-identical across runs, `max_abs_err` **identical to baseline to
every printed digit** (1.372e-04 / 6.841e-04). `single_core.py`: PASS. `semaphores.py`: every
semaphore's `max_wait` and `total_inc` **unchanged** (only `dge:*` -> `*:EVENT_SEMAPHORE`
attribution moves).

**The 16 us is below the noise floor; do not count it as a win.** The value of this iteration is
in sections 1-3: two of the four candidates the iter-5 review ranked are now *closed with
evidence*, the one with the big ceiling is closed *structurally*, and section 4 names a new lever
with a real ceiling that is sized and specified.

Scripts are parked beside this file as `iter-6-*.py` (see §5).

---

## 1. THE BUG THAT MATTERS: iter-5's SBUF access map was missing ~42% of all accesses

`iter-5-fuse2.py` built the "every access in the program" map that iter-5's hazard check and the
review's candidates 1 and 3 all rest on. It had **two holes**:

1. **It skipped every operand whose `*_start_addr_addr_reg_marker` was non-zero.**
   That field is just **the top byte of the 4-byte address**, and an SBUF address is
   `partition * 0x40000 + offset`, so **every access to partitions 64..127 has top byte 0x01**
   and every PSUM access has 0x02. The filter therefore threw away **46,315 real SBUF accesses**
   (39,315 based at partition 64, 7,000 at partition 96) while correctly dropping PSUM.
   *Rule: skip `marker >= 2` (PSUM / register), keep 0 and 1 — and note `imm // 0x40000` already
   gives the partition, so no other change is needed.*
2. **It only looked for the operand prefixes `src_mem_pattern` / `dst_mem_pattern`.**
   `DVE TENSOR_TENSOR_ARITH_OP` names its inputs **`src0_mem_pattern` / `src1_mem_pattern`**, so
   **all 11,584 tensor-tensor reads were invisible**. `ACTIVATE` also has a
   `scale_src = POINTER_IMMEDIATE` form whose `scale_value_imm_ptr` is an SBUF address — 3,456
   more reads. The complete prefix list for this NEFF is
   `src_mem_pattern, src0_mem_pattern, src1_mem_pattern, dst_mem_pattern` plus `bias_imm_ptr` /
   `scale_value_imm_ptr` when `bias_src` / `scale_src` is `POINTER_IMMEDIATE`
   (`INSTRUCTION_IMMEDIATE` carries no address). DGE uses the bare `src` / `dst`.

Fixed map: **86,464 -> 149,287 accesses**. `iter-6-acc3.py` is the corrected builder; **use it,
not `iter-5-fuse2.py`**, for anything that reasons about who touches a byte.

iter-5's shipped edit still verifies bit-exact, so the hole did not bite it, but it did make two
levers look much bigger than they are:

## 2. Candidates 1 and 3 from the iter-5 review are CLOSED (both were artefacts of §1)

### (a) Candidate 3 "delete the now-dead staging saves": 204 -> **7**

With iter-5's map, 204 of the 8,196 SBUF-writing DGE transfers looked dead ("no reader whose
`end_ts` reaches this write's `start_ts`"). With the corrected map it is **7**. The 197 others are
read by `TENSOR_TENSOR` operands or by partition-64..127 readers that the old map could not see.
A strict "no reader anywhere in the program" query returns **0** — every destination region is
read at *some* time, because SBUF regions are reused ~30 times over the program.

I shipped the 7 (5 x 256 KB + 2 x 128 KB of SB->SB traffic, Act 3 / SP 4, all in the last 340 us
of the program) using iter-5's proven `ES_a`/`ES_b` replacement. Worth 16 us, i.e. nothing.
**Do not revisit this class.**

### (b) Candidate 1 "finish the fusion on the 132 unfused staging groups": **0 of 132 are fusable**

Re-ran the whole iter-5 pipeline on the current NEFF: it finds exactly the **132** groups iter-5
left (shapes (10,1)x78, (9,1)x40, (11,1)x6, (7,1)x5, (5,1)x2, (6,1)x1 — plus the untouched
(1,8)x39, (2,8)x19, (2,4)x1, (1,4)x1 mirror groups). With the corrected map and a **provenance-
aware** overlap test (two saves may write the same window byte *if both write the same source
byte*, in which case order does not matter):

```
{'save_overlap': 124, 'save_outside': 8}   ->  0 fusable
```

So the 101 "hazard" rejections of iter-5 and its 23 "save overlap" rejections are **the same
124 groups**: every remaining group has two saves writing one window byte from *different*
sources, so at the destination the two writes are unordered (different engines, different
semaphores) and the fusion is unsound. The review's hoped-for "~21 cleared by a finer `crange`"
does not exist — and the reason is worth recording: **compute access patterns in this NEFF are
essentially 1-D** (`num_elem` = `(n,1,1[,1])` for all but `LDWEIGHTS (32,1,4)`, which is
contiguous anyway), so iter-5's bounding-interval `crange` was already exact. The refinement the
review proposed would have found nothing.

**The save/reload fusion lever is exhausted at 188 of 320 groups.** The only way to go further is
to *order* the overlapping saves (give the later one an extra wait on the earlier one's
semaphore), which buys at most the ~23 hazard-free ones = ~25 us. Not worth it.

## 3. THE BIG ONE: the 8,125 SB->SB DMA transposes CANNOT be deleted by swapping matmul operands

This closes `hints.md`'s standing lever and the review's candidate 2 (its only multi-hundred-us
item). Three facts, each checked:

1. **`nc_matmul(stat, mov) = stat^T @ mov`** (the kernel's own comment at `gdn_cte.py:430` says
   so, and the ISA agrees: `out[i,j] = sum_p stat[p,i] * mov[p,j]`). So the stationary operand is
   **already transposed for free**; the compiler's choice of which operand is stationary is
   already exploiting the whole of that lever. There is nothing to swap.
2. **PE contracts the partition axis of *both* operands.** For `C = X @ Y` contracted over K, both
   X and Y must be in SBUF with K on partitions. If X is produced with K on its *free* axis — which
   is what the producers here do — then **a physical transpose of X is unavoidable**, in either
   role and for either output orientation (`C` or `C^T`). I worked the algebra for all four
   placements; each one needs K on partitions for the moving operand.
3. **`transpose_mode = ENABLED` on `LDWEIGHTS` is NOT "load the stationary transposed".** All 1152
   ENABLED slots are FP32/`fp32_mode=LOW` and map to `gdn_cte.py:223`
   `nisa.nc_transpose(cg_cols_p, cg_rows)` — it is **the PE transpose *operation*** (result lands
   in PSUM), not a free re-orientation of a matmul operand.

Device evidence, two runs, both FAIL with NaN (so this is settled, not inferred):
I built the full consumer map of the transposing descriptors (`iter-6-tmap.py`):
3800 `LDWEIGHTS` wait on a transposing transfer; **2949 of them read its destination as the
stationary operand only** (the following MATMUL reads something else), 747 are the
`nc_matmul(k_t, k_t)` Gram case where **one transpose feeds both operands**, 104 feed only the
moving operand. I then took 987 clean single-descriptor square 64x64 bf16 transposes
(`transpose_shape (1,1,64,64)`), whose consumer is a stationary-only `LDWEIGHTS` with
`num_active_rows = num_active_cols = 64`, `num_elem (64,1,1)`, `step_elem (-1,0,0)`, verified
hazard-free, and re-pointed one of them at the transpose's **source** with
`transpose_mode = ENABLED` (a 2-field edit; address convention confirmed from the existing
ENABLED slots: `imm` = the **high** end of the tile, `base + (n-1)*esz`, step -1, which I checked
against the producer `ACTIVATE` that writes exactly `[base, base+256)`).

* source in partitions 64..127 -> 8,192 NaN in output_0.
* source in partitions 0..63 (53 of the 987 qualify) -> 32,768 NaN in output_0 + 16,384 in output_1.

Incidental fact worth keeping: **every one of the 31,840 PE `LDWEIGHTS`/`MATMUL` source reads in
this NEFF is based at partition 0** (11 distinct `(rows, cols, row_grp, col_grp)` classes, all
`imm // 0x40000 == 0`). PE may simply not read partitions >= 64 as an operand base.

**The only remaining way to remove a DMA transpose is to do it on PE instead**
(`nc_transpose`: `LDWEIGHTS` with `transpose_mode=ENABLED` + `MATMUL` -> PSUM, then a PSUM->SBUF
`CAST` back to bf16). That trades ~1.1 us of queued DMA latency for ~2 PE slots + one
Act/DVE slot + a PSUM bank, needs two instructions *inserted* per transpose, and is unproven for
bf16 (every ENABLED slot in this NEFF is FP32). It is a whole-iteration project with a 2.9 ms
ceiling and I would not start it without first probing bf16 `nc_transpose` in isolation.

## 4. Candidate I built and measured and which is ALSO ~0: folding the 7,940 FP32->BF16 `CAST`s away

Worth recording in full because it looked like the best remaining lever and the population is big.
**Conclusion: ~10 of 7,940 are foldable. Do not spend an iteration on it.**

* **7,940** `CAST` slots on Act (1,280) / DVE (6,660) are SBUF->SBUF **FP32 -> BFLOAT16** and
  nothing else: pure format conversion. Activation is the busiest engine (8,111 us, 57.8%) and DVE
  `CAST` is 531 us of iter-5's critical path, so removing a few thousand of them looked like 1-3%.
* The mechanism is device-proved (iter-5 folded 120 FP32->BF16 casts into DGE saves, bit-exact):
  make the *producer* of the FP32 tile write BF16 straight to the CAST's destination, then replace
  the `CAST` slot with an `EVENT_SEMAPHORE` carrying the CAST's wait and its
  `events_update_mode`/`events_update_idx` (ordinary `SEM_INC_COMPLETE` on S[4]/S[6] — no mode-21
  `ADD_IMM` needed, unlike the DGE case). For a *compute* producer the edit is only two fields
  (`out_dtype` + `dst_mem_pattern_start_addr_addr_immediate`) because compute `num_elem`/
  `step_elem` are in **elements**, so halving the element width changes nothing else.
* **Why it does not work.** Per-byte last-writer analysis of every CAST's FP32 source
  (`iter-6-cf4.py` for DMA producers, `iter-6-cf6/7/8.py` for compute producers):
  - **1,440** have a *DGE* writing into the source, but it is always **8 DGEs x (16 partitions x
    2048 B)** covering one CAST's **128 partitions x 512 B** — the N-to-1 staging shape, and the
    DGEs write 4x more bytes than the CAST consumes, so folding would have to move bytes that
    other CASTs consume. Only **10** CASTs survive the full check (8 or 4 contributing DGEs,
    byte ratio exactly 1.0). 10 x ~1 us is noise.
  - **7,448** have a *compute* instruction as the last writer. Of the ones whose producer is
    cleanly identified, the dominant rejection is **`prod_num_mismatch`** (~2,220 overall, and
    192 of the 257 candidates in a clean 901-slot sample): **the producer writes a different tile
    shape than the CAST reads** (e.g. producer `num_elem [128,1,1]` over 64 partitions, CAST reads
    `[64,1,1]`). The FP32 intermediate is produced in one tiling and consumed in another, so the
    cast cannot be absorbed into the producer's `out_dtype` without splitting the producer.
    A further **2,874** are rejected because **something other than the CAST also reads the FP32
    region** while it holds the producer's value — the FP32 copy is genuinely live.
  - Net: **0** foldable via a compute producer, **10** via DGEs.
* If someone does revisit this: the remaining theoretical subset is "producer shape == cast shape
  AND the FP32 region is read only by the cast". `iter-6-cf7.py` computes exactly that but takes
  >9 min over all 7,940 (it is O(partitions x writers) per CAST); `iter-6-cf8.py` is the capped
  sample version. Both use the block index (`WRX`/`RDX`, 2048-byte buckets) which is what makes
  the access queries affordable — reuse it.

## 5. Other measurements made this iteration (use these to size, not to re-measure)

* **Queued-ring flow control costs nothing — do not try to relax it.** Of the 8,285
  `PSEUDO_DMATRIGGER`s on SP: **2,258 wait on their own sub-queue semaphore** `S[24 + id%16]`
  with a depth of exactly **D = 1** (wait for transfer `id-16` to complete, i.e. one outstanding
  descriptor per sub-queue, 16 in flight overall), **296** wait on another ring, and **5,077**
  wait on a producer (`S[4]` Activation 2,285, `S[6]` DVE 1,571, `S[5]` PE 1,221).
  For the 2,258 ring-flow triggers the semaphore was **already satisfied long before** the
  trigger issued (`start_ts - t_reach` sums to 58 ms of slack) and `start_ts - previous SP slot
  end` sums to only **93 us** — they issue back-to-back behind SP's own program order. Raising the
  ring depth would buy **0** and risks ring overflow. (`iter-6-flow.py`)
* **The kernel is producer-bound, and the producers are Activation/DVE/PE compute.** The 5,077
  data-dependent triggers sum **8,012 us** of `start_ts - previous SP slot end` — SP sitting idle
  waiting for a compute producer. No engine is above 58% busy (Act 8,111 / DVE 5,651 / DMA 5,187 /
  PE 3,195 / SP 2,732 / Pool 2,667 us against a 14,033 us span), so the span is a latency chain,
  not an occupancy problem. **This makes iter-1's never-tried lever the other live one:**
  reordering the Activation stream so instructions that gate a trigger run before Activation work
  nothing waits on, via pairwise swaps of two `S[4]`-signalling slots (swap the two slots, then
  swap the wait *values* of their respective waiters — ch 10.3 r08). Ceiling is plausibly >1 ms;
  cost is a real dependence analysis of a 29,935-slot stream.
* `def.json` has **no ring-size / queue-depth field** (`{type, num_queues: 16, owner,
  fabric_path, semaphore_set}`, plus `dynamic_dma.SB_scratch_*` on `qPoolDynamic` only), so the
  hardware ring depth is not editable from the NEFF. The `def.json` DMA-queue axis stays closed.
* `iter-5-cpath2.py` **no longer runs on the current NEFF**: it has a `KeyError: 'SP'` (its
  `sig_order` dict lacks SP — use `.get(eng,-1)`), and after that fix the backward walk
  terminates at step 1 because the last-finishing node is a `Pool EVENT_SEMAPHORE` whose
  `S[23]` wait it cannot attribute. Budget ~20 min to repair it (the Pool DGE producer lookup) if
  you want the critical-path table refreshed; it was 99.3% accounting on the iter-4 binary.

## 6. What I would do next, in order

**Honest summary of the state: every "remove a step from the dependency chain" lever is now used
up.** iters 3 and 5 won by deleting DMA hops (iter-3: reload reads SBUF directly; iter-5: fuse 188
save/reload pairs). I looked for every remaining instance of that shape this iteration — unfused
staging groups (§2b), dead writes (§2a), FP32->BF16 casts (§4), DMA transposes (§3) — and all four
are closed. Exactly two things are left, and both are big projects rather than edits:

1. **Activation reordering** (§5, iter-1's never-tried item 1). The only remaining >=1 ms ceiling.
   The evidence is now quantitative: the 5,077 data-dependent DMA triggers sum **8,012 us** of SP
   idle waiting on `S[4]`/`S[5]`/`S[6]`, no engine exceeds 58% busy, and the span is a latency
   chain. Mechanism: pairwise swap of two `S[4]`-signalling Activation slots, then swap the wait
   *values* of their respective waiters (ch 10.3 r08, device-verified). Needs a real dependence
   analysis of the 29,935-slot Activation stream — use `iter-6-acc3.py` for the access map and the
   `WRX`/`RDX` block index from `iter-6-cf7.py` to make the queries affordable.
2. **bf16 `nc_transpose`** (§3), if you want the 2.9 ms of queued transpose latency. Start with one
   isolated probe: does PE's transpose path work for bf16 at all (every ENABLED slot in this NEFF
   is FP32/`fp32_mode=LOW`)? Only if yes is the "insert `LDWEIGHTS`+`MATMUL`+PSUM->SBUF `CAST` per
   transpose, and find the PSUM banks" work worth planning.
3. **Do not** spend budget on: the save/reload fusion (§2b, exhausted at 188/320), dead DGE writes
   (§2a, 7 existed and are now gone), the FP32->BF16 cast fold (§4, 10 of 7,940), matmul operand
   swapping / `transpose_mode` on a matmul operand (§3, device-refuted twice), queued-ring depth
   (§5), `def.json` DMA queues (§5 + iter-2/3), dominated-`EVENT_SEMAPHORE` deletion
   (iter-5 §3.1, measured 0), DGE adjacency merging (iter-5 §3.3), the consecutive-run instruction
   merge (iter-5 §3.4), PE/DVE perf modes (iter-5 §3.5), compute-for-DMA substitution
   (iter-5 §3.2), the DGE host ratio (iter-2), queued-transfer merging (iter-1 refutations 4 and
   8), `dge_op = DMA_TRANSPOSE` (iter-1 refutation 7).
4. If neither project fits the budget, the right move is to **spend the iteration confirming the
   current number rather than chasing 0.1% edits** — everything cheap left is under the 0.5%
   (70 us) noise floor, and this iteration's own 16 us is an example.

## 7. Tooling notes (additions to iter-5 §5)

* `*_start_addr_addr_reg_marker` is **the top byte of the address**, not a register flag:
  0 = SBUF partitions 0..63, 1 = SBUF partitions 64..127, 2 = PSUM. See §1.
* Operand prefixes differ per opcode; enumerate them with
  `[k for k in d if k.endswith('_start_addr_addr_immediate')]` instead of assuming `src`/`dst`.
* `/home/ubuntu/workspace/gdn-bench/vendor/gdn/gdn_cte.py` **is readable** and
  `prof/Instruction.parquet.nki_source_location` points straight into it. This is by far the
  fastest way to find out what a slot *means* — it is how §3 got settled. (Only `kernel.neff` may
  be edited; reading the source is free.)
* `tools/verify.py` takes `--runs N` and takes ~11 s per run; `tools/profile.py --repeats 3` takes
  ~90 s. Both are cheap enough to use as the primary oracle.

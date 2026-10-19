# iter-1 — 215.2 → 95.3 us (2.26x on my own re-measure; 2.29x vs the recorded 218.17 baseline)

`total_exec_time_us` of the shipped file, three independent 5-repeat runs: **95.284 / 95.451 /
95.591** (spreads 0.48–1.42%) — call it **95.4 us**.
My own re-measure of the *unedited* baseline in this same session: **215.218** (spread 2.86%) —
use that, not 218.17, when comparing.
Shipped `kernel.neff` sha256 `40e89ff150d22d103bdbdc6e75d8487281f55dbac20684f9d1298d9bc23d94fb`.

`verify.py` passes with `max_abs_err` **exactly equal to the baseline's** on all three outputs
(2.999e-03 / 1.928e-03 / 6.033e-04), on two independent runs after every edit: all edits are
arithmetic bit-identical and showed no run-to-run variation. `launch_rules.py` passes.

Changed members: `sg00/SP0.bin`, `sg00/Pool0.bin`, `sg00/Activation0.bin`, `sg00/SP0.json`.
No slot inserted or deleted; `semaphores.py` reports no change. Exactly **one** wait value was
changed in the whole iteration (Activation slot 67, see edit H); everything else is pure
address / access-pattern / dtype field edits.

---

## 1. The finding that carried the whole iteration

The baseline was latency-bound (no engine above 24% busy) with **~127 of its 215 us spent idle
waiting on DMA completion semaphores**. The cause, measured:

> **A DGE DMA that writes 128 narrow (2–8 byte) elements into DRAM takes 17–22 us to signal
> completion — about 135 ns per element. The same bytes as one wide element take ~2 us. Reads
> from DRAM are not affected.**

How to measure it: join `Instruction.parquet` to `SemaphoreUpdate.parquet` filtered to
`update_source == 'AXI'`. A DGE DMA with `sem_increment=16` emits **16 separate +1 updates**, and
a consumer's `>= 16` wait needs the *last* one. Table from the baseline:

| transfer | issue → 16th increment |
|---|---|
| HBM → SBUF, any element size | 1.2 – 3.4 us |
| SBUF → HBM, `elem_size` 512 | 2.0 us |
| SBUF → HBM, `elem_size` 8, **contiguous destination** | 3.8 – 4.6 us |
| SBUF → HBM, `elem_size` 2, `num_elem` 128 | **17 – 22 us** |
| anything queued behind one of those | 17 – 37 us |

`DmaPacket.parquet` shows the data itself lands 0.6 us after the trigger (128 packets, ~8 ns
each) — it is the completion *reporting* that trickles in, and it blocks the whole DMA queue, so
unrelated transfers behind a narrow write inherit the delay. The baseline had ~30 such DMAs in
5–6 serialized generations, each generation gating the next compute burst.

## 2. What those DMAs were for, and the fix

`def.json` has ten DRAM `tmp-buf` variables. The compiler routes small values through them in
writer/reader pairs. Decoding both halves showed that in most cases the pair is **not a
transpose**:

```
writer: src SB+X  step=[262144,1] num=[128,1] elem 2  ->  dst var15+off step=[2,1] num=[128,1] elem 2
reader: src var15+off step=[2,1] num=[128,1] elem 2   ->  dst SB+Y  step=[262144,1] num=[128,1] elem 4 FP32
```

Element *k* goes `SB[k][X] -> DRAM[off+2k] -> SB[k][Y]`: **a per-partition copy, sometimes with a
bf16→fp32 cast, implemented as a 20 us DRAM round trip.** A DGE DMA converts dtypes
(`in_dtype`/`out_dtype`), so one SBUF→SBUF DMA does the whole job.

**The transformation, which needs no semaphore or wait changes at all:**

> Give the **writer** the reader's destination descriptor and give the **reader** the writer's
> source descriptor. Both DMAs then perform the *identical* SBUF→SBUF transfer. The writer's
> completion semaphore now means "the destination is written" — which is what every consumer was
> actually waiting for — and the reader becomes a harmless idempotent repeat that keeps every
> semaphore count and every wait exactly as the compiler emitted them.

## 3. The edits, in the order applied, each measured

| # | chain | what changed | median after |
|---|---|---|---|
| A | `var15` `_mem_6_1`, 8 pairs: SP 5,6,7,8,9,12,17,22 ↔ Pool 34,36,38,40,43,51,53,55 | both halves → SB→SB (per-partition bf16→fp32 cast) | 201.3 |
| B | `var19` `_mem_52_47`, 8 groups: Activation 35,39,40,41,52,55,58,59 ↔ SP 13,14,15,16,24,26,44,46,52,53,55,56. Some writers have **two** readers: the writer adopts reader[0]'s dst, *every* reader adopts the writer's src | both halves → SB→SB | **167.2** |
| C | `var16` `_mem_7_2`: DGE SP 27,32,37,38 + **queued** transfers 5–8 | writers write the final single-partition 256-byte block directly; queued descriptors become `SB -> SB` with `from_sizes=[2,128] from_steps=[1,262144]`, `to_steps=[1,2]` | — |
| D | `var22` `_mem_111_106`: DGE SP 34,36,58,60 + **queued** 13 | writers write `SB 213760 + 512*i` with `dst_step=[4,1] dst_elem_size=4 out_dtype=FP32` (they do the cast themselves); queued 13 → four SB→SB `cast` descriptors | **124.0** |
| E | `var21` `_mem_110_105`: DGE SP 18,19,20,21 + SP 30 + **queued** 4 | writers write the four 256-byte blocks of queued 4's destination (`SB 212224 + 256*i`); **SP 30 gathers the four columns straight out of SBUF with `src_step=[-768, 262144] num=[4,128]`**; queued 4 → four SB→SB descriptors | **111.4** |
| F+G | queued 13 → 4-byte dummy; queued 14 (`cast var23 -> SB`) → four SB→SB `cast` descriptors | | 110.2 (within noise of E; kept) |
| H | `var20` `_mem_109_104`, the last narrow-write chain | **queued 13 repurposed to do the whole 4-column gather as four *ordered* SB→SB descriptors** into `SB 114944 + 2*i`; DGE SP 62,64,66,68 and SP 75 shrunk to `num_elem=[1,1]` (one 2-byte write into the now-dead `var20`) so they still signal at the same counts; **Activation 67** (the only reader of `SB 114944`) re-pointed from `s61>=48` to `s37>=16` | **96.3** |
| I | queued 4 and 9–12 are redundant after E and G | each shrunk to a 2-byte copy into the dead `var21` | **95.3** |

Engine busy barely moved (Scalar 41.8→42.0, Pool 35.8→35.8, DVE 19.2→19.6, PE 7.6→7.6); only
DMA busy fell 52.3 → 35.3 us. **All 120 us came out of idle**, exactly as the model predicts.
After H, **no DGE DMA in the profile exceeds 6.6 us** issue-to-completion (was 37 us).

## 4. Refuted / dangerous — read before touching anything similar

All three failures produced the *same* `max_abs_err` 1.766e-01 on output_0, which is how I
located each cause.

* **Splitting a gather across its writers fails when the per-writer destinations are less than
  a word apart.** First var21 attempt: writer *i* wrote `SB 215808 + 2*i` (2 bytes × 128
  partitions) concurrently. Wrong numbers — interleaved sub-word writes to the same 4-byte word
  clobber each other. The same split **works** when the blocks are 256 bytes apart (edits C, E).
  **Rule: concurrent DMA destinations must be ≥ one word, in practice 256 bytes, apart.** If they
  must interleave, put them as *ordered descriptors inside one queued transfer* (edit H) — that
  is serialized by construction.
* **A DRAM tmp-buf can have a second consumer in `SP0.json`, not just a DGE reader.** `var21` was
  read by DGE SP 30 **and** queued transfer 4; dropping the DRAM writes fed transfer 4 garbage.
  Always check both: `isa.py` for `src_start_addr_addr_var_var_id_lo`, and `SP0.json` for the
  variable *name*.
* **Do not weaken an `EVENT_SEMAPHORE`'s wait just because its producer looks redundant.**
  Re-pointing SP 28/29 from `s67..s70>=16` to `s62..s65>=16` broke the output even though SP 30's
  source data was provably ready — those waits were also what ordered queued transfer 4 (and
  hence Activation 47–50's in-place reads of `SB 212224`).
* **Verified positive, reuse freely:** a DGE DMA reads SBUF correctly and bit-exactly with a
  **negative `src_step_elem`** (`[-768, 262144]`, `num=[4,128]`), and accepts the SBUF partition
  stride in dimension 1 as well as 0. A queued transfer may have **several descriptors, done in
  order**, and `from`/`to` may both be `SB`, with `op: cast`.

## 5. Mechanics (saves an hour next time)

`PSEUDO_DMA_DIRECT2D` byte layout inside the 64-byte slot, so an edit is a byte-block copy:

```
 4..15  events, semaphore, sem_increment, dge_op
16..23  src address   (PSEUDO_ADDR8: u32 offset @16, u16 var_id_lo @20, marker 0x10 @23; var 0 = SB)
24..31  src_step_elem[2]   int32, BYTES (SBUF partition stride 262144; negative works)
32..35  src_num_elem[2]    uint16
36..37  src_elem_size      uint16, bytes
40..47  dst address        48..55 dst_step_elem   56..59 dst_num_elem   60..61 dst_elem_size
62 in_dtype    63 out_dtype
```

`isa.py`: use `Instruction.as_dict()` and `.fields()` (which give every field *and its byte
offset*). **`str(ins)` is only the opcode repr** — it has no fields, which cost me a detour.

Queued DMA (`SP0.json`, chapter 9.2, **all byte units**): for `SB`, `from_off`/`to_off` is
`partition*262144 + offset`; `sizes[0]` is a contiguous byte run and the rest are repeat counts;
`steps[0]` is always 1; for `cast`, `to_sizes[0]` scales with the output dtype. Transfer `id` is
named by the triggering `PSEUDO_DMATRIGGER`'s `count_union_block_id`; semaphore is always
`semaphore_set[id % 16]`.

Trigger → transfer map for this NEFF (SP slot → queued id, with the trigger's wait):
1→0(s6≥8) 2→1(s24) 3→IO0 4→2(s6≥25) 10→3(s26) 11→IO1 31→4(none) 39→5(s58≥32) 40→6(s60≥32)
41→7(s63≥32) 42→8(s64≥32) 47→9(s29) 48→10(s30) 49→11(s31) 50→12(s32) 71→13(s56≥48) 72→14(s36)
76→15(s6≥63) 77→16(s39) 78→17 79→18 80→19 81..84→IO 2..5.

## 6. Where the remaining 95 us goes

Span 110.7 us; still no engine above 44% busy, so it is still latency, not work. Idle per engine:
Tensor 89, Vector 81, Sync 79, GpSimd 71, Scalar 61 us. The end-to-end critical path is now a
**serial chain of ~10 DGE/queued hops, each costing its 2–6 us completion latency**, plus the
tail compute:

| window | gated by |
|---|---|
| Scalar 46.2→58.3 (12.1) | `s59>=32` = DGE SP 30 (the SBUF gather) |
| Vector 42.7→58.9 (16.2) | `s4>=20` = Scalar pc 44, i.e. downstream of the above |
| Vector 60.8→82.1 (21.3) and Scalar 64.3→82.3 (18.0) | `s37`/`s38` = queued 13 / 14 |
| GpSimd 65.4→100.0 (34.6) | `s6>=60` = the tail compute on Vector |
| Sync 79.5→104.1 (24.6) | `s6>=63`, same tail |

Traced precisely for the 82 us barrier: Activation 59 issues at 69.1 → completes 72.9 → Sync
pc 54 passes → Sync pc 55, 56 issue (73.0, 73.2) → SP 56 completes 75.0 → Sync pc 67 passes 77.3
→ DGE SP 60 completes ~77 → **Sync pc 71 triggers queued 13 at 78.0** → `s37` 79.1→82.1 →
Vector resumes. Three completion-latency hops plus Sync's own back-to-back issue time.

### Ranked next steps

1. **Collapse the redundant hops introduced by §2.** After edits A/B every writer/reader pair
   does the *same* SB→SB copy, so the reader is pure latency: any consumer that waits on the
   **reader's** semaphore can be re-pointed to the **writer's** (earlier) semaphore, removing one
   3–5 us hop. Careful: pick the right group index (`>= 16*k` counts transfers on that
   semaphore), and only for destinations the writer actually adopted — where a writer had two
   readers, only reader[0]'s destination is redundant; the second reader's destination is unique.
   For this kernel: `SB 219136` (SP 52) and `SB 57600` (SP 55) are redundant with Activation 58
   and 59; `SB 57344` (SP 53) and `SB 114688` (SP 56) are **not**.
2. **GQA duplication in the var20 gather.** Columns 0,1 of queued 13 are both
   `SB[p][19968]` and columns 2,3 are both `SB[p][20480]` (num_k_heads 2 shared by num_v_heads 4).
   So queued 13 can read `SB 19968` / `SB 20480` directly and depend only on Activation 58/59,
   taking SP 52/53/55/56 off the critical path. Liveness: `SB 19968`/`20480` are overwritten by
   Activation 90/101 at ~95 us, well after queued 13 at ~78 — but re-check if the schedule moves.
3. **Sync pc 71's wait `s56>=48` (DGE SP 60) is an *indirect* dependency** — queued 13's data no
   longer involves `SB 215296`; the wait exists because `s37` is what DVE 87 uses to learn that
   the var22 result `SB 213760..215808` is ready. Re-point DVE 87 (and only it) to the var22
   writers' own semaphores and the queued-13 trigger can fire ~3 us earlier. Needs all four of
   SP 34/36/58/60 covered, and a DVE wait cannot borrow Sync's program order — plan this one
   carefully.
4. Remaining narrow-DRAM writes: **none on the critical path.** The eight `output_1` stores
   (Pool 45,47,49,57,60,62,63,64, `elem 8` → contiguous DRAM) complete in 3.8–4.6 us, so the
   coalescing works there; do not bother staging them through SBUF.
5. Free scratch for any "must signal but must not corrupt" dummy transfer: DRAM tmp-bufs
   `var15` (2048 B), `var16` (1024), `var19` (2048), `var22` (1024), `var23` (1024) are now
   completely unread; `var20` and `var21` hold only dummy bytes.

### Tried at the end and **refuted** (reverted, not in the shipped NEFF)

**Hoisting slots earlier in the SP stream buys nothing.** I moved SP 30 (the `SB 215808` gather)
up to just after SP 21, and the queued-13/14 triggers (SP 71, 72) up above slots 68–70. Both
moves are legal and were **verified bit-exact** — `PSEUDO_DMATRIGGER` signals nothing and no SP
instruction increments a per-engine completion semaphore, so a pure reorder on SP changes **no
semaphore count and no wait**, and there are no `PSEUDO_EXTENSION` slots in this stream to keep
glued. Measured 94.99 vs 95.28 us: **−0.3%, inside the 0.7% spread.** Reverted, because it adds
a new risk class for no gain.

The useful conclusion: **the 82 us barrier is not Sync's issue time.** My estimate that Sync was
issue-bound and that an earlier trigger would pay 1–3 us was wrong. The gate is the *completion*
of DGE SP 60 (`s56>=48`, which the queued-13 trigger waits on) plus the ~3 us queue latency, and
that completion time did not move. So the remaining 10-hop chain is set by DGE/queued completion
latency (2–6 us per hop), not by instruction issue — which also means next steps 1–3 below should
be re-estimated before being trusted: only edits that **remove a hop from the dependency chain**
can pay, and an edit that merely lets an instruction issue sooner cannot.

### Closed by measurement — do not re-open

* **The `ACTIVATION_TABLE_LOAD` tail.** 7 loads × 1.28 us, `act_tbl_sel` alternating 2
  (`silu_and_others`) and 4 (`reciprocal_sqrt_and_small`). Not regroupable: **no shipped table
  contains both `silu` (func 36) and `reciprocal_sqrt` (func 29)**, and the per-head
  `ACTIVATE(square)/READ_ACCUMULATOR/ACTIVATE(rsqrt)` chain forces the alternation. `square` (30)
  and `copy` (128) are in every table so they are free. Ceiling ~5 us and Scalar is 44% busy.
* DMA bandwidth / descriptor count (11k packets, ~3 us busy per engine); PE (7.6 us); the weight
  prefetch (done by 34 us); any arithmetic change (output_0 already uses 83% of the error budget).

## 7. The safety check that justified every edit

For each chain, before editing: (a) decode every DGE DMA in all five streams and every `SP0.json`
descriptor, and confirm the DRAM buffer's complete consumer set; (b) dump every compute
instruction's SBUF range and check that **no instruction reads the destination between the
writer's time and the original reader's time** (writing it earlier with the same value is then
free), and that **the writer's source is not overwritten before the now-later reader reads it**.
In this kernel every affected destination turned out to be read only by instructions that already
wait on the relevant completion semaphore, and every affected source only by tail instructions
running 20–100 us later. Both scripts (`dge.py` over the five `*0.bin`, `scan.py` over every
`*_mem_pattern_start_addr_addr_immediate` with `num_elem`/`step_elem`) are a few lines each — see
§5 for the field names.

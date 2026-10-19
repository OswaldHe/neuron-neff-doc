# iter-4 — 74.56 → 60.0 us (1.24x on top of iter-3; 3.63x vs the 218.17 us baseline)

Start-of-iteration re-measure of the iter-3 file: **74.558 us** (spread 1.54%, 5 repeats).
Use **74.56** as this iteration's reference, not the 74.921 in `history.jsonl`.

Shipped file = **g5** = f1+f2+f3b+f4+f5+f6+f8+f10+g2+g3+g4+g5, sha256
`78910ea86d9100364a0da83508bf2cc3c6518a20883d160376a249b215c5628f`.
Three independent 5-repeat runs of the shipped file: **59.951** (spread 0.17%), **60.156**
(0.54%), **59.884** (0.50%) → call it **60.0 us**. Spread on this device is now well under
1%, so differences of ~0.3 us are readable; anything smaller is not.
`verify.py` passes with `max_abs_err` **exactly the baseline values** (2.999e-03 / 1.928e-03 /
6.033e-04) after every single edit — everything shipped is arithmetic bit-identical.
`semaphores.py` clean on every state. `launch_rules.py` passes (1 subgraph, num_tpb 1,
11 inputs, 3 output heads, LNC=1, one rank).

| state | what | median of 5 |
|---|---|---|
| iter-3 (HEAD) | — | **74.558** |
| f1 | queued blocks 5–8 (duplicate transposes) → 2-byte dummies | 73.996 |
| f2 | swap queued blocks 13 ↔ 14 payloads | 74.255 (neutral) |
| f3b | queued 4-element gather → 4 compute CASTs on Activation | **72.223** |
| f4 | SP 53 / SP 56 (now dead) → bare ESs | **71.943** |
| f5 | queued p0→p32 cast block → 4 **cross-partition** compute CASTs | **69.084** |
| f6 | hoist the two live DGE casts (SP 58/60) inside the SP stream | **67.513** |
| f7 | + run SP 30 ahead of the SP 18–21 transposes | 69.592 (REGRESSION) |
| f7a | + only relax SP 28/29's waits | 71.338 (REGRESSION) |
| f8 | SP 30 (512-descriptor per-partition gather) → 4 compute CASTs on Pool | **65.981** |
| f9 | Activation's `s28` queued-dummy wait → direct waits on the 4 transposes | 66.243 on f8, 65.278 on f10 (REGRESSION both times) |
| f10 | relax the tail's `s56>=48` gate to `s71>=32`, guard DVE 108 instead | **64.717** |
| g1 | probe: `dma_configs_dma_transpose_2d_tile = 1` on SP 18–21 | 64.766 (bit-identical, NO EFFECT) |
| g2 | two remaining SBUF→DRAM→SBUF scratch round trips (blocks 0–3) made direct | **64.419** |
| g3 | **delete 2 of the 7 tail `ACT_TABLE_LOAD`s** by moving one ACTIVATE | **62.853** |
| g4 | delete 2 more by giving one ACTIVATE a private (in-place) destination | **61.143** |
| **g5 (shipped)** | output epilogue's SBUF→DRAM→SBUF round trip made direct | **59.951** |
| g6 | + let the output stores read the pre-staging SBUF directly | 59.994 (neutral, reverted) |
| g7a | pc 100 writes in place, DVE 106 re-pointed (enabler only) | bit-identical, not shipped |
| g7 | + move the three func-36 ACTIVATEs and delete 2 more loads | **verify FAILS** — see §4 |
| g7b | g7 without the load deletions (bisect) | **same failure** — the move is the cause |
| g8 | hoist the sem-19 Pool prologue DMA to issue 2nd instead of 11th | **verify FAILS** — see §4 |

---

## 0. THE HEADLINE FINDING: a compute instruction CAN read one partition and write another

This was listed as *unknown #1* in the iter-3 review and it is the key that unlocked ~3 us.
**It does not need an experiment — the answer is already in this NEFF:**

> `DVE` pc 88: `TENSOR_TENSOR_ARITH_OP`, `num_active_channels = 1`,
> `src0_..._addr_immediate = 213760` (partition 0),
> `dst_..._addr_immediate = 8600832 = 32*262144 + 212224` (partition **32**).

So the compute address immediate is the **full linear SBUF address with the partition encoded**
(`partition * 0x40000 + offset`, chapter 5), for *both* source and destination independently,
and `num_active_channels` is how many partitions from there. A `CAST` with
`num_active_channels=1`, `num_elem=[128,1,1,1]` therefore does a 128-element row copy from
p0 to p32. **Verified bit-identical on the device** (f5). DVE pc 89 writes p96, so 0/32/64/96
all work.

What compute still **cannot** do: a partition→row transpose. The partition axis is not part of
the access pattern (chapter 5), so "128 partitions × 1 element → 1 partition × 128 elements"
is not expressible. Every such transpose has to stay on the DMA engine.

## 0b. THE SECOND FINDING: a redundant `ACT_TABLE_LOAD` can be deleted

iter-2 closed "`ACT_TABLE_LOAD` field edits" (no shipped function set holds both silu and
reciprocal_sqrt) and the last two reviews treated the 13 loads × 1.283 us as fixed. They are
not:

> **Replacing an `ACT_TABLE_LOAD` with a no-op `EVENT_SEMAPHORE` is safe and saves its full
> 1.283 us whenever the table it selects is already resident.** Nothing in the NEFF counts
> table loads (`Activation0.json`'s `activation_function_sets` is a name list and
> `act_semaphore: 12` is unrelated — s12 is a Pool DMA semaphore).

**The rule that makes it safe:** an edit is legal iff **no instruction's resident table
changes**. `CAST`, `COPY`, `ACTIVATION_READ_ACCUMULATOR` and `EVENT_SEMAPHORE` are
table-agnostic, so letting *those* fall under a different table is free — that is all g3 and
g4 do, and both are bit-identical. Mapping evidence: `func 36` occurs only under table 2,
`func 29` only under table 4, and `func 30` / `func 128` occur under **both** in the
compiler's own schedule, so they are probably agnostic too — "probably" because it is not
independently verified and the ids may be indices *within* the selected set.

g3 + g4 moved **two** ACTIVATEs and deleted **four** loads (−3.3 us), taking the tail from
`4 2 4 2 4 2 4` to `4 | 2 2 2 | 4 4 4`.

**What stops the next two loads is not the table (bisected):** g7 tried to move the three
func-36 ACTIVATEs at pc 100/101/102 next to the fourth one at pc 82 so that the LOAD2 at 97
and the LOAD4 at 106 would go, and `output_0` came out 8348% over the limit. **g7b repeated
the move with every table load kept and produced the byte-for-byte same wrong answer**, so
deleting the two loads is not the cause and the func-30/128 agnosticism is not refuted — the
*move* is what breaks. The cause is not identified. The best remaining hypothesis is the
Activation engine's accumulator: the tail has `ACTIVATION_READ_ACCUMULATOR` at pc 95, 104, 116
and 125, the move shifts three ACTIVATEs from the window `(95,104]` into `(…,95]`, and the
READ_ACC at 95 feeds pc 96 — but g3 and g4 each moved a func-36 op *across* a READ_ACC
boundary with no effect, which argues against it. **Whoever picks this up: first establish
experimentally whether `ACTIVATE` accumulates, e.g. insert one extra ACTIVATE (writing to a
dead buffer) just before an existing READ_ACC and see whether the output changes.**

Two enablers made the moves legal, and both are the general tool here:
* an `ACTIVATE` with `events_update_mode = NONE` does **not** increment `s4`, so moving it does
  not shift any `s4>=N` threshold — check this first, it is what makes the edit cheap;
* when a move is blocked by a *destination* collision rather than by data, give the op a
  private destination. g4 wrote **in place on the op's own source**, which is free whenever
  that source has no later reader (and in-place `ACTIVATE` is already used at pc 96/106/117/126).

## 1. THE THIRD FINDING: the DMA engine is a single serial resource, and descriptors are the currency

Three separate experiments (f7, f7a, f9) *correctly* moved work earlier, were bit-identical,
and all **lost time**. The reason is the same every time: between ~34 and ~50 us the DGE engine
is saturated by ~1536 two-byte transpose descriptors (SP 18–21, SP 27/32/37/38, SP 34/36/58/60
— 128 descriptors each, 12 transfers), and it retires them at **~15 ns each**. Pulling any
other transfer into that window just pushes the others' completions out by the same amount.

* **A 128-descriptor DGE transpose takes ~0.6 us to issue and its completion is reported
  ~2.5–3.1 us later; four back-to-back take 5.2 us from the first issue to the last
  completion.** Those numbers were stable across all 11 measurements this iteration.
* Therefore: **the only edits that pay in that window are ones that remove descriptors from the
  DMA engine entirely** (f3b, f5, f8 — all three convert a DMA into compute), not ones that
  re-order or relax waits.
* Corollary, refuted twice: relaxing a wait so a DGE transfer can start sooner is a
  *pessimisation* if the engine is already busy.

## 2. How to tell which DMAs can become compute

Decode the access pattern. For a DGE `PSEUDO_DMA_DIRECT2D` (steps are in **bytes**;
262144 = one partition):

| src_step / dst_step | what it is | movable to compute? |
|---|---|---|
| src `[262144,…]`, dst `[262144,…]` | per-partition copy ("type A") | **yes** — `CAST`, `num_active_channels=128`, `num_elem=1` (iter-3 e1/e4) |
| src `[-768, 262144]` num `[4,128]`, dst `[2, 262144]` num `[4,128]` | per-partition *gather* of 4 strided elements | **yes** — either one CAST with `src_step_elem=[-384]`/`num_elem=[4]`, or 4 single-element CASTs (f8 used 4; steps for compute are in **elements**) |
| src `[262144,…]` num `[128,1]`, dst `[2 or 4, …]` num `[128,1]` | partition→row transpose (optionally with a dtype cast) | **no** |
| contiguous row at p*a* → contiguous row at p*b* | cross-partition row copy/cast | **yes** — `num_active_channels=1`, `num_elem=[128,…]`, address = `part*262144 + off` (§0) |

Same rule for queued descriptors in `*0.json`: `from_steps`/`to_steps` both containing 262144
in the same position ⇒ per-partition ⇒ compute can do it.

## 3. The edits, in order

### f1 — queued blocks 5–8 → dummies (−0.56)
`qSPSpillReload0` blocks 5–8 are byte-identical duplicates of the DGE transposes SP 27/32/37/38
(`SB +18432/18688/18944/19200`, 128 partitions → `SB p0 +218368/217600/216832/216064`), which
iter-3 had already established. Each is 128 two-byte descriptors and the four occupied the
serial queue from 47.8 to 53.4 us, which blocked the *live* blocks 13/14 behind them.
Replaced each `desc` with the same 2-byte write into the dead DRAM tmp-buf that iter-1 left in
blocks 9–12. **Editing `SP0.json` descriptors is a legitimate, tested lever** (iter-1 did it).
Blocks 5–12 now finish by 50.3 instead of 53.6.
Only −0.56 us, because the block-13 trigger (SP 71) is gated by SP program order, not by the
queue — but it is a prerequisite for everything after.

### f2 — swap blocks 13 ↔ 14 (neutral, kept because f3b/f5 build on it)
Block 13 was the expensive one (4 per-partition copies = 512 descriptors, 3.2 us) and block 14
the cheap one (4 contiguous row casts, 0.3 us), and the Activation tail hangs off block 14's
semaphore. Swapping `desc` **and** `semaphore` (so every wait in the program still means "my
data is ready") put the cheap one first. Measured 74.255 vs 73.996 — neutral, because s37
(the other block) immediately became the gate. Keep in mind when reading f3b/f5: after f2,
**block 13 = the casts (sem 38), block 14 = the gather (sem 37)**.

### f3b — the queued gather → 4 compute CASTs (−1.77)
Tracing every producer and reader of the addresses in block 14's four descriptors gave:

```
Act pc 64 -> SB[p]+19968 ;  Act pc 65 -> SB[p]+20480
Act pc 66: CAST 19968 -> 219136        SP 53: DMA 19968 -> 57344
Act pc 67: CAST 20480 -> 57600         SP 56: DMA 20480 -> 114688
gather: 219136->114944, 57344->114946, 57600->114948, 114688->114950
```

i.e. the gather is just `[a,a,b,b]` with `a = SB[p]+19968`, `b = SB[p]+20480`, both already in
SBUF when Activation pc 65 retires. Four per-partition CASTs inserted after Activation pc 67
(Activation is idle there) do the whole thing with no wait.

**The trap, and the one failed verify of this iteration (f3):** raising `s37` as soon as the
gather was done gave `output_2` a 7.2e-2 error (2007% of the limit). `DVE pc 87`'s
`wait s37>=16` has nothing to do with the gather buffer — `DVE 88/89/90` read the fp32 rows
`SB p0 +213760/214272/214784` that the DGE casts SP 34/36/58/60 write, and the *only* thing
that ordered them was that block 13's trigger waited `s56>=48` = SP 60's completion. So
**when you delete a DMA, work out what its completion semaphore was transitively ordering,
not just what its own bytes were** — the compiler's single wait often carries several
guarantees. Fix: two inserted ESs re-impose `s61>=32 & s62>=32` and `s56>=48 & s71>=32`
before a third ES adds 16 to s37.

Note `events_semaphore_value` is one field shared by the wait threshold and the
`SEM_ADD_IMM_COMPLETE` amount — if you need wait value ≠ add value, use two slots.

### f4 — SP 53 and SP 56 are dead (−0.28)
After f3b, `57344` and `114688` have **no reader anywhere in the NEFF** (checked all five
streams and all `*0.json` descriptors). Both DGE DMAs became bare ESs keeping their completion
counts (`s68 += 16`, `s70 += 16`, waited as `>=32`).

### f5 — the p0→p32 cast block → 4 cross-partition compute CASTs (−2.86)
Block 13 (after f2) is four `cast` descriptors `SB p0 +218368/217600/216832/216064`
(128 bf16) → `SB p32 +213760/214272/214784/215296` (128 fp32). The queued route costs
**3.2 us of trigger→completion latency** even though the transfer is 4 descriptors.
Replaced by 4 compute CASTs on Activation (§0), preceded by two ESs re-imposing
`s58>=32 & s60>=32` and `s63>=32 & s64>=32` (the four DGE transposes that write the sources),
followed by an ES adding 16 to s38. Queued block 13 → 2-byte dummy on the **unwaited**
semaphore 33, so it can neither re-write the buffer later nor raise s38 early.

### f6 — hoist SP 58/60 inside the SP stream (−1.57)
The two live DGE casts `217088 -> p0 214784` (s71) and `216320 -> p0 215296` (s56) sat behind
four dummy block triggers and six ESs. SP 44 raises `s65` to 32 right after Activation pc 60
writes 217088 (via s76), and SP 46 raises `s66` to 32 right after Activation pc 63 writes
216320 (via s77) — so the data guarantee allows
`SP 43,44,[57],[58],45,46,[59],[60],47..56`. Pure permutation of slots 43–60.
(This one *did* pay even though it pulls DGE work earlier, because it removes 1.1 us of
pure issue latency without adding descriptors.)

### f8 — SP 30 off the DMA engine (−1.53)
`SP 30` is a **within-partition** 4-element gather
(`218880/218112/217344/216576`, src step −768 B → `215808`, dst step 2 B, 128 partitions =
**512 two-byte descriptors**) and its completion `s59>=32` left Activation idle from 36.1 to
44.6 us. f7/f7a proved the wait in front of it cannot be relaxed (§1). The fix is to stop
using the DMA engine: four compute CASTs on Pool, inserted after Pool pc 52.
They need **no wait at all** — Pool 50 writes 218112 and Pool 52 writes 216576 (program
order), and Pool 50/52's own waits `s72>=16` (Act 43 → 218880) and `s73>=16` (Act 47 → 217344)
cover the other two sources. Last CAST carries `s59 += 16`; SP 30 becomes a no-op.

### f10 — relax the tail's last gate (−1.26)
The inserted ES at Activation pc 80 waited `s56>=48` (SP 50's completion, 52.2 us) before
raising s37. Of the four fp32 rows, only `p0 +215296` comes from SP 50, and its **single
reader in the whole NEFF is DVE pc 108**, ~9 us later. So the Activation gate was weakened to
`s71>=32 & s62>=32` (50.4 us) and an ES waiting `s56>=48` was inserted immediately in front of
DVE 108. General pattern: **when one semaphore guards four buffers with very different
deadlines, split it and push the late guarantee down to its actual consumer.**

### g2 — the last two scratch DRAM round trips (−0.30)
iter-1 rerouted six SBUF→DRAM→SBUF scratch round trips but left four `qSPSpillReload0`
blocks going through DRAM: blk0/blk1 take `SB fp32 +19456` from partitions 0–3 into a 16-byte
DRAM buffer and back out as four contiguous fp32 at `SB p0 +211968`; blk2/blk3 do the same for
`+19712 → p0 +211712`. It is a 4-element partition→row transpose done via DRAM, and the two
hops were 1.04 + 1.18 us of critical path at 22–24.5 us. One direct `SB→SB` descriptor in
blk0/blk2 (`from_sizes [4,4] from_steps [1,262144]` → `to_sizes [4,4] to_steps [1,4]`), blk1/
blk3 → dummy (their semaphore counts are reused as `>=32` by blocks 16–19), and the triggers'
waits on s24/s26 dropped. The gates vanished from the critical path but a dense PE/DVE compute
chain was right behind them, so only 0.3 us surfaced.

### g3 — delete two tail `ACT_TABLE_LOAD`s (−1.57)
**`ACT_TABLE_LOAD` can be NOP'd when the table it selects is already resident** — new, and the
big one. The tail's resident table alternates 4,2,4,2,4,2,4 once per v-head, 1.283 us per
switch. Decoding `act_tbl_sel` against every ACTIVATE's `activation_func` gives the mapping:

| activation_func | needs |
|---|---|
| 36 | table 2 (`silu_and_others`) |
| 29 | table 4 (`reciprocal_sqrt_and_small`) |
| 30, 128 | **neither** — they appear in both regions of the original schedule |
| CAST, COPY, ACT_READ_ACC, EVENT_SEMAPHORE | neither |

pc 108 (`ACTIVATE func 36`, `src SB p32+214784`, `dst 17416`) was the only table-2 consumer in
the pc-107 group, has **no wait and no `SEM_INC_COMPLETE 4`**, its source was written by
Activation pc 76, and 17416 has no writer before it — so it hoists into the previous table-2
region (right after pc 100, the other func 36). That empties the pc-107 group of table-2 work,
so the LOAD2 at 107 **and** the LOAD4 at 111 (table is still 4 from the load at 104) are both
redundant → no-op ESs. Nothing's resident table changes and no s4 count changes.
Verified bit-identical. Nothing in the program depends on the number of table loads
(`Activation0.json`'s `act_semaphore: 12` is unrelated — s12 is a Pool DMA semaphore).

### g4 — delete two more (−1.71)
pc 120 (`ACTIVATE func 36`, `src p32+215296`, `dst p0 17416`, `wait s6>=61`) could not move
because 17416 is also pc 101's destination and `s6>=61` is the anti-dependency letting
DVE 111/112 read pc 101's value first. But **pc 120's own source `p32+215296` is read by
nobody except pc 120**, so pc 120 can write *in place* (128 fp32 either way; in-place ACTIVATE
is already used at pc 96/106/117/126). With a private destination `s6>=61` has nothing left to
order and is dropped, pc 120 joins pc 100/101, and its single consumer DVE pc 114 has `src1`
re-pointed 17416 → 8603904 (DVE 113's `wait s4>=49` still orders it after Activation pc 126).
The LOAD2 at 118 and the LOAD4 at 121 then go. **Tail table loads: 7 → 3.**

### g5 — the output epilogue's DRAM round trip (−1.19)
Block 15 wrote the 1024 contiguous bytes at `SB p32+18184` (the four 256-byte results of
DVE 104/106/112/114) into DRAM `mem_125_120`, and blocks 16–19 read the four quarters straight
back into SBUF, whence `qSPIO0` 2–5 store them to `output_0`. Pointed blocks 16–19 at the SBUF
source directly (`8406792 + 256*k`), made block 15 a dummy, dropped SP 77's `s39>=16`.
(iter-2's "output-0 tail bypass" is a *different* edit and is still closed — see g6.)

### g6 — one hop further: neutral, reverted
Letting `qSPIO0` 2–5 read `8406792+256*k` themselves and dummying blocks 16–19 measured
59.994 vs 59.951. The staging buffers have exactly one writer and one reader each, so the edit
is sound, but the output path is slack (iter-2 said the same). Script in §8 if it is ever
needed.

## 4. Refuted this iteration (do not redo)

* **f7 / f7a — relaxing SP 28/29's waits so SP 30 can run earlier: +2.1 and +3.8 us.**
  SP 28/29 wait on the completions of SP 18–21, which only *read* the same four addresses SP 30
  reads; the waits' only real content was transitive (`s63>=16` = Pool 50 → 218112,
  `s65>=16` = Pool 52 → 216576). Stating those directly is *correct* (bit-identical) and
  *slower*, because SP 30's 512 descriptors then interleave with SP 18–21's. §1.
* **f9 — replacing Activation's `wait s28>=16` (queued dummy block 4, a 2-byte read of
  SB 212224 into dead DRAM) with direct waits on `s67..s70>=16`, one per consumer
  (Activation pc 55/56/57/58 each read one transposed row): +0.26 on f8, +0.56 on f10.**
  It does what it says — the ACT_TABLE_LOAD after it moved 2.6 us earlier — but Activation
  then reaches the *next* DGE gate sooner and the DMA engine is the binding resource, so the
  slack just reappears further down. Script kept in §6 in case the DMA load drops enough to
  make it pay.
* **`dma_configs_dma_transpose_2d_tile = 1` on the DGE transposes SP 18–21 (g1): inert.**
  Bit-identical output and 64.766 vs 64.717 us — no change to the completion time at all. The
  1-bit field exists on `PSEUDO_DMA_DIRECT2D` (`docs/generated/structs.md`) and the DMA engine
  does have a transpose mode (queued descriptors in the attention kernels carry
  `transpose_shape` / `transpose_op: WZXY` / `transpose_element_size`), but setting the
  instruction bit alone does nothing for a `num [128,1]` access pattern. If anyone revisits
  this, the remaining route is a *queued* descriptor written with the `transpose_*` keys
  instead of `op`, with a `[1,1,128,128]`-style shape — a much bigger rewrite.
* **Merging transposes** stays closed (iter-3 e6 hang / e7 regression), and §1 explains why:
  merging does not change the descriptor count, which is what costs.
* **g6** (above): the output staging copies are slack.
* **g7 — moving the three func-36 ACTIVATEs (pc 100/101/102) next to the fourth one at pc 82
  and deleting the LOAD2 at 97 + the LOAD4 at 106: `output_0` off by 8348% of the limit
  (max_abs_err 3.005e-01, cosine 0.949553).** Bisected in two steps:
  * **g7a** = the memory half only (pc 100 writes in place on its own source `p32+214272`,
    DVE pc 106's `src1` re-pointed 16384 → 8602880): **bit-identical**, and a ready-made
    enabler for a future attempt.
  * **g7b** = g7a + the move, with every `ACT_TABLE_LOAD` kept: **the exact same wrong
    numbers**. So the two deleted loads are innocent; the move is what breaks, and I did not
    find out why (see §0b for the accumulator hypothesis and the experiment that would settle
    it). Do not conclude from g7 that func 30/128 need their table.
* **g8 — permuting Pool's eleven no-wait prologue DGE DMAs so the one PE waits for issues
  second instead of last: `verify` FAILS (all three outputs wrong) and it is neutral anyway
  (60.07 vs 59.95).** Pool slot 12 (semaphore 19, a 256-byte-per-partition weight tile) is
  issued last of the group and its completion `s19>=16` is a 1.56 us gate on PE pc 14's
  `LDWEIGHTS` while PE sits idle — so this looked like a free win of iter-3's e5 kind.
  It is not. **iter-3's "permuting no-wait prologue DMAs on distinct queues is free" does not
  generalise past Pool slots 1–6**: moving slot 12 to position 2 (and slots 2–11 one place
  later) corrupts the weights, so some of those eleven transfers have an order requirement
  that is not expressed by any semaphore — overlapping destinations are the obvious
  candidate. If anyone retries this, permute **one pair at a time** and verify each.

## 5. Where the time is now (g5, 60.0 us; trace body 11.9 → 72.4 us)

Engine busy: **Activation 41.9 us (70%)**, Pool 23.8, DMA 22.1, DVE 17.6, SP 12.0, PE 7.3.
Activation's busy time is now back to the *original* 41.8 us (iter-3 had inflated it to 45.9
by hosting moved DMAs; g3/g4 removed 5.1 us of table loads and the compute CASTs added ~2).

Every wait > 0.7 us left on the critical path:

| gate | us | what it is |
|---|---|---|
| Pool 1 / Pool 8 (prologue DMAs) | 0.74 / 1.79 | weight-load DGE issue, 12–18 us |
| PE 14 `S[19]>=16` | 1.56 | Pool slot 12's weight-tile completion, 22.2 us — **g8 tried to hoist it and corrupted the weights (§4)** |
| **SP 28 `S[70]>=16`** | **5.20** | **SP 18–21: four 128-descriptor partition→row transposes, 34.6 → 42.2 us** |
| Activation 54 `S[28]>=16` | 1.38 | queued dummy block 4 — f9's target, refuted twice |
| **Activation 80 `S[71]>=32`** | **2.64** | **SP 46's DGE cast completion, 47.7 → 50.3** |
| Activation 99 (`s5>=23`) | 1.46 | PE |
| SP 83 `S[26]>=32` / Pool 71 `S[45]>=16` | 1.37 / 1.37 | output epilogue (slack — see g6) |

Rough shape: 11.9 → 34.6 prologue+compute, 34.6 → 50.4 **DMA-descriptor bound**,
50.4 → 70 Activation tail (~100% busy, now 3 table loads = 3.8 us), 70 → 72.4 output.

### Ranked candidates for iter-5

1. **The remaining `ACT_TABLE_LOAD`s: 3 in the tail + 6 in the head (9 × 1.283 = 11.5 us, most
   of it on the critical path).** g3/g4 proved the mechanism and g7 found its exact limit
   (§0b, §4). The current layout is
   `… 59:LOAD2 … 82:f36 … 86:LOAD4 … 91:f29 96:f29 … 97:LOAD2 … 100,101,102:f36 103:f30
   105:f128 … 106:LOAD4 … 107:f29 118:f29 126:f29 …` plus head loads `sel 0,1,2,3,4` at
   pc 10/24/35/45/51 (five *different* sets — no redundancy there) and `sel 2` at pc 59.
   Removing another load means moving the three f36 at 100/101/102 next to pc 82 (−2.56 us,
   because the LOAD2 at 97 and the LOAD4 at 106 both become redundant). **The memory side is
   already solved and verified — `g7a`** — but the move itself gives wrong numbers for a
   reason I did not find (§0b, §4). **Start there**: settle whether `ACTIVATE` accumulates,
   then either fix the accumulator bookkeeping or find the real dependency. A cheaper probe
   first: move the three f36 ops one slot at a time and verify after each — that localises
   which of the three (and which neighbour) matters.
2. **Give an Activation instruction a private destination to unlock a move (g4's trick).**
   This is the general tool for candidate 1: most of these ops are blocked by a *destination*
   collision, not by data. Buffers that are now provably **dead** and reusable: `57344`,
   `114688` (f4), `219136`, `57600` (f3b made the gather read 19968/20480 directly) — each one
   bf16-per-partition. For a 512-byte single-partition buffer, prefer in-place on the op's own
   source when the source has no later reader (that is exactly what g4 did).
3. **SP 18–21, the 5.2 us gate (four partition→row transposes, 512 descriptors).** Their four
   p0 rows 212224/212480/212736/212992 are read *only* by the four in-place
   `ACTIVATE func 128` at Activation pc 55–58, i.e. the transpose exists so that Activation can
   reduce 128 per-partition values along its free axis. The only ways out are (a) a real DMA
   transpose mode — the instruction bit is inert (§4), so it would have to be a queued
   descriptor with `transpose_*` keys, or (b) doing the cross-partition reduction on **PE** with
   a ones-vector matmul, which changes where the accumulator lives. Both are multi-hour.
4. **Activation pc 60/63 (`CAST 23040→217088` / `CAST 23552→216320`, bump s76/s77) have no
   wait**, and SP 46/50's DGE casts are gated on them. Hoisting them would pull `s71>=32`
   (the 2.64 us gate) in by up to 2 us — but f9 showed that pulling Activation work earlier
   while the DMA engine is saturated just moves the slack. Retry only after candidate 3.
5. **`S[19]>=16` at PE pc 14 (1.56 us, 22.2 us).** The gate is Pool slot 12's weight-tile
   completion and PE is idle in front of it, so it *looks* free — but g8 showed the eleven
   no-wait Pool prologue DMAs cannot be freely permuted (§4). The remaining route is to move
   that one tile to one of Activation's DGE queues the way iter-3's e2 moved eight others —
   but all of act queues 80–87 are in use, so it would need a reuse guard, and iter-3 found
   that moving *more* than eight regresses. Low expected value.
6. **Do not bother with:** anything that only re-orders or relaxes waits around the DGE
   transposes (§1, §4); permuting the Pool prologue DMAs (g8); merging DMAs (iter-3); the
   output epilogue (g6, iter-2); the `dma_configs_dma_transpose_2d_tile` bit (g1);
   `ACT_TABLE_LOAD` *field* edits (iter-2: no shipped function set holds both silu and rsqrt —
   but **deleting** a redundant load does work, see §0b); PE (12% busy).

## 6. Reproducing — the scripts

All scripts take the unpacked root as `argv[1]` and assert the slot they overwrite.
Build order from a fresh unpack of the iter-3 NEFF (slot indices in each script assume the
preceding ones have run — f5 and g3/g4 in particular depend on f3b's insertion point):

```
f1.py work ; f2.py work ; f3b.py work ; f4.py work ; f5.py work ; f6.py work ; f8.py work
f10.py work ; g2.py work ; g3.py work ; g4.py work ; g5.py work
pack ; verify ; profile
```

`f7.py`, `f7a.py`, `f9.py`, `g1.py`, `g6.py` are the refuted ones.
The analysis helpers (`d.py` compact decoder, `waiters.py` every user of a semaphore set,
`cp.py` backwards critical-path walk) are verbatim from iter-3 §7 and were re-created from
those listings; `cp.py` needs `prof/Instruction.parquet` + `SemaphoreUpdate.parquet` +
`DmaPacketAggregated.parquet` from `profile.py --parquet`.

Two extra one-liners that did most of the work this iteration and are worth rebuilding:

```python
# every instruction and every JSON descriptor that touches an SBUF address range
# -> this is how the f3b dataflow collapse and all the "dead buffer" findings were found
import isa, json
for en in ['PE','Activation','Pool','DVE','SP']:
    for i,x in isa.decode_stream(open(f'work/payload/sg00/{en}0.bin','rb').read()):
        a=x.as_dict()
        for k in a:
            if (k.endswith('addr_table_offset') or k.endswith('addr_immediate')) \
               and isinstance(a[k],int) and LO<=a[k]<=HI:
                print(en,i,x.name,k,a[k])
for b in json.load(open('work/payload/sg00/SP0.json'))['dma']:
    for d in b['desc']:
        for s in ('from','to'):
            if d[s]=='SB' and LO<=d[s+'_off']<=HI: print(b['queue'],b['id'],s,d[s+'_off'])
```

```python
# when a semaphore value is actually reached (prof/SemaphoreUpdate.parquet)
s['sid']=pd.to_numeric(s['id'].str.extract(r'S\[(\d+)\]')[0],errors='coerce')
s[(s.sid==SID)&(s.value>=V)].sort_values('timestamp').iloc[0]
```

### an/f1.py
```python
import json, sys
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
DUMMY=[b['desc'][0] for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==9][0]
for b in d['dma']:
    if b['queue']=='qSPSpillReload0' and b['id'] in (5,6,7,8):
        assert len(b['desc'])==1 and b['desc'][0]['from_sizes']==[2,128]
        b['desc']=[json.loads(json.dumps(DUMMY))]
json.dump(d, open(p,'w'), indent=1)
```

### an/f2.py
```python
import json, sys
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
b13=[b for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==13][0]
b14=[b for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==14][0]
assert b13['semaphore']==37 and b14['semaphore']==38
assert all(x['op']=='copy' for x in b13['desc']) and all(x['op']=='cast' for x in b14['desc'])
b13['desc'], b14['desc'] = b14['desc'], b13['desc']
b13['semaphore'], b14['semaphore'] = 38, 37
json.dump(d, open(p,'w'), indent=1)
```

### an/f3b.py
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
CAST_T=bytes(act[66*64:67*64]); ES_T=bytes(act[78*64:79*64])   # pc 66 = per-partition 1-elem CAST, pc 78 = ES with extended wait
blob=b''
for src,dst in ((19968,114944),(19968,114946),(20480,114948),(20480,114950)):
    c=isa.decode(CAST_T)
    c.set('events_wait_mode','NONE'); c.set('events_wait_idx',0)
    c.set('events_update_mode','NONE'); c.set('events_update_idx',0); c.set('events_semaphore_value',0)
    c.set('src_mem_pattern_start_addr_addr_immediate',src)
    c.set('dst_mem_pattern_start_addr_addr_immediate',dst)
    c.set('in_dtype','BFLOAT16'); c.set('out_dtype','BFLOAT16')
    blob+=c.encode()
for wi,wv,xi,xv in ((61,32,62,32),(56,48,71,32)):      # SP 34/36 then SP 58/60 completions
    x=isa.decode(ES_T)
    x.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_wait_idx',wi); x.set('events_semaphore_value',wv)
    x.set('events_extended_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_extended_wait_idx',xi); x.set('events_extended_sem_wait_value',xv)
    x.set('events_update_mode','NONE'); x.set('events_update_idx',0)
    blob+=x.encode()
x=isa.decode(ES_T)                                      # separate slot: shared value field must be 16 here
x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0)
x.set('events_extended_wait_mode','NONE'); x.set('events_extended_wait_idx',0); x.set('events_extended_sem_wait_value',0)
x.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); x.set('events_update_idx',37); x.set('events_semaphore_value',16)
blob+=x.encode()
act[68*64:68*64]=blob; open(ap,'wb').write(act)
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
DUMMY=[b['desc'][0] for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==9][0]
t=[b for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==14][0]
assert t['semaphore']==37 and len(t['desc'])==4
t['desc']=[json.loads(json.dumps(DUMMY))]; t['semaphore']=33
json.dump(d, open(p,'w'), indent=1)
```

### an/f4.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(p,'rb').read())
ES_T=bytes(d[57*64:58*64])
for slot,sem,src,dst in ((53,68,19968,57344),(56,70,20480,114688)):
    x=isa.decode(bytes(d[slot*64:(slot+1)*64])); a=x.as_dict()
    assert x.name=='PSEUDO_DMA_DIRECT2D' and a['semaphore']==sem \
       and a['src_start_addr_addr_table_offset']==src and a['dst_start_addr_addr_table_offset']==dst
    e=isa.decode(ES_T)
    e.set('events_wait_mode','NONE'); e.set('events_wait_idx',0)
    e.set('events_extended_wait_mode','NONE'); e.set('events_extended_wait_idx',0); e.set('events_extended_sem_wait_value',0)
    e.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); e.set('events_update_idx',sem); e.set('events_semaphore_value',16)
    d[slot*64:(slot+1)*64]=e.encode()
open(p,'wb').write(d)
```

### an/f5.py
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
CAST_T=bytes(act[68*64:69*64]); ES_T=bytes(act[72*64:73*64])    # the f3b group: CASTs at 68..71, ESs at 72..74
blob=b''
for wi,wv,xi,xv in ((58,32,60,32),(63,32,64,32)):               # SP 27/32 then SP 37/38 completions
    x=isa.decode(ES_T)
    x.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_wait_idx',wi); x.set('events_semaphore_value',wv)
    x.set('events_extended_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_extended_wait_idx',xi); x.set('events_extended_sem_wait_value',xv)
    x.set('events_update_mode','NONE'); x.set('events_update_idx',0)
    blob+=x.encode()
for src,doff in ((218368,213760),(217600,214272),(216832,214784),(216064,215296)):
    c=isa.decode(CAST_T)
    c.set('events_wait_mode','NONE'); c.set('events_wait_idx',0)
    c.set('events_update_mode','NONE'); c.set('events_update_idx',0); c.set('events_semaphore_value',0)
    c.set('num_active_channels',1)                              # <-- single partition
    c.set('src_mem_pattern_start_addr_addr_immediate',src)
    c.set('src_mem_pattern_step_elem',[1,1,1,1]); c.set('src_mem_pattern_num_elem',[128,1,1,1])
    c.set('dst_mem_pattern_start_addr_addr_immediate',32*262144+doff)   # <-- partition 32
    c.set('dst_mem_pattern_step_elem',[1,1,1,1]); c.set('dst_mem_pattern_num_elem',[128,1,1,1])
    c.set('in_dtype','BFLOAT16'); c.set('out_dtype','FP32')
    blob+=c.encode()
x=isa.decode(ES_T)
x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0)
x.set('events_extended_wait_mode','NONE'); x.set('events_extended_wait_idx',0); x.set('events_extended_sem_wait_value',0)
x.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); x.set('events_update_idx',38); x.set('events_semaphore_value',16)
blob+=x.encode()
act[72*64:72*64]=blob; open(ap,'wb').write(act)
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
DUMMY=[b['desc'][0] for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==9][0]
t=[b for b in d['dma'] if b['queue']=='qSPSpillReload0' and b['id']==13][0]
assert t['semaphore']==38 and all(y['op']=='cast' for y in t['desc'])
t['desc']=[json.loads(json.dumps(DUMMY))]; t['semaphore']=33
json.dump(d, open(p,'w'), indent=1)
```

### an/f6.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(p,'rb').read())
old={i:bytes(d[i*64:(i+1)*64]) for i in range(43,61)}
NEW=[43,44,57,58,45,46,59,60,47,48,49,50,51,52,53,54,55,56]
for k,src in enumerate(NEW): d[(43+k)*64:(44+k)*64]=old[src]
open(p,'wb').write(d)
```

### an/f8.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
pp=f'{ROOT}/payload/sg00/Pool0.bin'; po=bytearray(open(pp,'rb').read())
T=bytes(po[50*64:51*64])                       # iter-3's per-partition 1-elem Pool CAST
blob=b''
for src,dst,sem in ((218880,215808,None),(218112,215810,None),(217344,215812,None),(216576,215814,59)):
    c=isa.decode(T)
    c.set('events_wait_mode','NONE'); c.set('events_wait_idx',0)
    if sem is None: c.set('events_update_mode','NONE'); c.set('events_update_idx',0); c.set('events_semaphore_value',0)
    else: c.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); c.set('events_update_idx',sem); c.set('events_semaphore_value',16)
    c.set('src_mem_pattern_start_addr_addr_immediate',src)
    c.set('dst_mem_pattern_start_addr_addr_immediate',dst)
    c.set('in_dtype','BFLOAT16'); c.set('out_dtype','BFLOAT16')
    blob+=c.encode()
po[53*64:53*64]=blob; open(pp,'wb').write(po)
sp=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(sp,'rb').read())
a=isa.decode(bytes(d[30*64:31*64])).as_dict()
assert a['semaphore']==59 and a['src_start_addr_addr_table_offset']==218880
E=isa.decode(bytes(d[29*64:30*64]))
for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
            ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
            ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
    E.set(k,v)
d[30*64:31*64]=E.encode(); open(sp,'wb').write(d)
```

### an/f10.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
x=isa.decode(bytes(act[80*64:81*64])); a=x.as_dict()
assert x.name=='EVENT_SEMAPHORE' and a['events_wait_idx']==56 and a['events_semaphore_value']==48
x.set('events_wait_idx',71); x.set('events_semaphore_value',32)
x.set('events_extended_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_extended_wait_idx',62); x.set('events_extended_sem_wait_value',32)
act[80*64:81*64]=x.encode(); open(ap,'wb').write(act)
dp=f'{ROOT}/payload/sg00/DVE0.bin'; dv=bytearray(open(dp,'rb').read())
T=bytes(dv[87*64:88*64])
assert isa.decode(bytes(dv[108*64:109*64]))['src0_mem_pattern_start_addr_addr_immediate']==215296
g=isa.decode(T)
g.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); g.set('events_wait_idx',56); g.set('events_semaphore_value',48)
g.set('events_extended_wait_mode','NONE'); g.set('events_extended_wait_idx',0); g.set('events_extended_sem_wait_value',0)
g.set('events_update_mode','NONE'); g.set('events_update_idx',0)
dv[108*64:108*64]=g.encode(); open(dp,'wb').write(dv)
```

### an/f9.py (refuted, kept for later)
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
sl=lambda i: bytes(act[i*64:(i+1)*64])
assert isa.decode(sl(54))['events_wait_idx']==28
body=[sl(55),sl(56),sl(57),sl(58)]
def guard(sem):
    x=isa.decode(sl(54))
    x.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); x.set('events_wait_idx',sem); x.set('events_semaphore_value',16)
    x.set('events_update_mode','NONE'); x.set('events_update_idx',0)
    return x.encode()
act[54*64:59*64]=b''.join(guard(s)+b for s,b in zip((67,68,69,70),body))
open(ap,'wb').write(act)
```

## 7. Final state

Changed members vs the iter-3 file: `sg00/Activation0.bin` (113 → **127** slots),
`sg00/Pool0.bin` (71 → **75**), `sg00/DVE0.bin` (114 → **115**), `sg00/SP0.bin` (same length,
slots 15–30 and 43–60 permuted, several rewritten), and **`sg00/SP0.json`** (qSPSpillReload0
blocks 0, 1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19 — descriptors plus three semaphore
fields). `SP0.json` is the first JSON this loop has edited since iter-1; `neff.py pack`
handles it fine (it re-serialises and recomputes both MD5s). Final size 132206 bytes;
`neff_diff.py baseline/kernel.neff kernel.neff` reports 5 of 46 members changed
(`Activation0.bin` 6720→8128, `DVE0.bin` 7296→7360, `Pool0.bin` 4544→4800, `SP0.bin`,
`SP0.json`) and no member added or removed.

Check order used after every edit, per iter-3 §4:
`semaphores.py` → `pack` → `verify.py` → `profile.py --repeats 5`. No hang occurred this
iteration (16 edits × 25 executions each) and `verify.py --runs 3` passes on the shipped file.
`verify.py` alone is still not a hang detector.

## 8. The g-edit scripts

### an/g2.py
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
B={b['id']:b for b in d['dma'] if b['queue']=='qSPSpillReload0'}
DUMMY=B[9]['desc'][0]
for src_blk,dst_blk,src_off,dst_off in ((0,1,19456,211968),(2,3,19712,211712)):
    assert B[src_blk]['desc'][0]['from_sizes']==[4,4] and B[dst_blk]['desc'][0]['to_off']==dst_off
    B[src_blk]['desc']=[{"op":"copy","from":"SB","from_dtype":"float32","from_off":src_off,
                         "from_sizes":[4,4],"from_steps":[1,262144],
                         "to":"SB","to_dtype":"float32","to_off":dst_off,
                         "to_sizes":[4,4],"to_steps":[1,4]}]
    B[dst_blk]['desc']=[json.loads(json.dumps(DUMMY))]
json.dump(d, open(p,'w'), indent=1)
sp=f'{ROOT}/payload/sg00/SP0.bin'; s=bytearray(open(sp,'rb').read())
for slot,sem in ((3,24),(10,26)):           # triggers of the now-dummy blocks 1 and 3
    x=isa.decode(bytes(s[slot*64:(slot+1)*64])); a=x.as_dict()
    assert x.name=='PSEUDO_DMATRIGGER' and a['events_wait_idx']==sem
    x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
    s[slot*64:(slot+1)*64]=x.encode()
open(sp,'wb').write(s)
```

### an/g3.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
sl=lambda i: bytes(act[i*64:(i+1)*64])
mv=isa.decode(sl(108)); a=mv.as_dict()
assert mv.name=='ACTIVATE' and a['activation_func']==36 \
   and a['dst_mem_pattern_start_addr_addr_immediate']==17416 \
   and a.get('events_wait_mode') in (None,0,'NONE') and a.get('events_update_mode') in (None,0,'NONE')
assert isa.decode(sl(100)).as_dict()['activation_func']==36
for i,selv in ((97,2),(104,4),(107,2),(111,4)):
    x=isa.decode(sl(i)); assert x.name=='ACTIVATION_TABLE_LOAD' and x['act_tbl_sel']==selv
act = bytearray(act[:101*64] + sl(108) + act[101*64:108*64] + act[109*64:])
NOP=isa.decode(sl(98))
for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
            ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
            ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
    NOP.set(k,v)
nop=NOP.encode()
for i,selv in ((108,2),(111,4)):
    x=isa.decode(bytes(act[i*64:(i+1)*64]))
    assert x.name=='ACTIVATION_TABLE_LOAD' and x['act_tbl_sel']==selv
    act[i*64:(i+1)*64]=nop
assert len(act)//64==127
open(ap,'wb').write(act)
```

### an/g4.py
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
sl=lambda i: bytes(act[i*64:(i+1)*64])
x=isa.decode(sl(120)); a=x.as_dict()
assert a['activation_func']==36 and a['num_active_channels']==1 \
   and a['src_mem_pattern_start_addr_addr_immediate']==8603904 \
   and a['dst_mem_pattern_start_addr_addr_immediate']==17416 \
   and a['events_wait_idx']==6 and a['events_semaphore_value']==61
x.set('dst_mem_pattern_start_addr_addr_immediate',8603904)      # in place on its own source
x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
moved=x.encode()
act = bytearray(act[:102*64] + moved + act[102*64:120*64] + act[121*64:])
NOP=isa.decode(sl(98))
for k,v in (('events_wait_mode','NONE'),('events_wait_idx',0),('events_semaphore_value',0),
            ('events_extended_wait_mode','NONE'),('events_extended_wait_idx',0),
            ('events_extended_sem_wait_value',0),('events_update_mode','NONE'),('events_update_idx',0)):
    NOP.set(k,v)
nop=NOP.encode()
for i,selv in ((119,2),(121,4)):
    y=isa.decode(bytes(act[i*64:(i+1)*64]))
    assert y.name=='ACTIVATION_TABLE_LOAD' and y['act_tbl_sel']==selv
    act[i*64:(i+1)*64]=nop
open(ap,'wb').write(act)
dp=f'{ROOT}/payload/sg00/DVE0.bin'; dv=bytearray(open(dp,'rb').read())
z=isa.decode(bytes(dv[114*64:115*64])); b=z.as_dict()
assert b['src1_mem_pattern_start_addr_addr_immediate']==17416 \
   and b['src0_mem_pattern_start_addr_addr_immediate']==16384
z.set('src1_mem_pattern_start_addr_addr_immediate',8603904)
dv[114*64:115*64]=z.encode(); open(dp,'wb').write(dv)
```

### an/g5.py
```python
import json, sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.json'; d=json.load(open(p))
B={b['id']:b for b in d['dma'] if b['queue']=='qSPSpillReload0'}
DUMMY=B[9]['desc'][0]; src=B[15]['desc'][0]
assert src['from_off']==8406792 and src['from_sizes']==[1024,1]
VAR=src['to']
for k,bid in enumerate((16,17,18,19)):
    de=B[bid]['desc'][0]
    assert de['from']==VAR and de['from_off']==256*k and de['to']=='SB'
    de['from']='SB'; de['from_dtype']='bfloat16'; de['from_off']=8406792+256*k
    de['from_sizes']=[256,1]; de['from_steps']=[1,1]
B[15]['desc']=[json.loads(json.dumps(DUMMY))]
json.dump(d, open(p,'w'), indent=1)
sp=f'{ROOT}/payload/sg00/SP0.bin'; s=bytearray(open(sp,'rb').read())
x=isa.decode(bytes(s[77*64:78*64])); a=x.as_dict()
assert a['events_wait_idx']==39 and a['count_union_block_id']==16
x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
s[77*64:78*64]=x.encode(); open(sp,'wb').write(s)
```

### an/g7a.py — verified bit-identical, not shipped (enabler for candidate 1)
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
x=isa.decode(bytes(act[100*64:101*64])); a=x.as_dict()
assert a['activation_func']==36 and a['src_mem_pattern_start_addr_addr_immediate']==8602880 \
   and a['dst_mem_pattern_start_addr_addr_immediate']==16384
x.set('dst_mem_pattern_start_addr_addr_immediate',8602880)      # in place
act[100*64:101*64]=x.encode(); open(ap,'wb').write(act)
dp=f'{ROOT}/payload/sg00/DVE0.bin'; dv=bytearray(open(dp,'rb').read())
z=isa.decode(bytes(dv[106*64:107*64]))
assert z.as_dict()['src1_mem_pattern_start_addr_addr_immediate']==16384
z.set('src1_mem_pattern_start_addr_addr_immediate',8602880)
dv[106*64:107*64]=z.encode(); open(dp,'wb').write(dv)
```

### an/g1.py (refuted probe), an/g6.py (neutral), an/g7.py (failed)
```python
# g1: isa.decode(slot).set('dma_configs_dma_transpose_2d_tile',1) on SP 18,19,20,21 - inert.
# g6: point qSPIO0 blocks 2-5 at 8406792+256*k and dummy qSPSpillReload0 16-19 - neutral.
# g7:  g7a + move Activation 100,101,102 to 83..85 + NOP the LOAD2 at (new) 100 and the
#      LOAD4 at (new) 106 -> output_0 max_abs_err 3.005e-01, cosine 0.949553.
# g7b: the same move with all table loads kept -> byte-identical failure, so the move is the
#      cause, not the deletions:
#        act = bytearray(bytes(act[:83*64]) + sl(100)+sl(101)+sl(102)
#                        + bytes(act[83*64:100*64]) + bytes(act[103*64:]))
#      When rebuilding a slice-and-splice, capture any template slot (e.g. the ES at 98)
#      *before* reassigning the bytearray - a lambda over `act` sees the new stream.
# g8:  Pool prologue DMA order 1,12,2,3,4,5,6,8,9,10,11 (hoist the sem-19 tile) -> all three
#      outputs wrong.  Permuting those eleven is NOT free.
```

# iter-3 — 83.2 → 74.6 us (1.12x on top of iter-2; 2.93x vs the recorded 218.17 baseline)

Start-of-iteration re-measure of the iter-2 file: **83.02 us** (spread 2.0%), later re-measures
83.20 / 83.31 / 83.32 / 83.27. Use **83.2** as this iteration's reference.
Shipped file: **e11** (= e1+e2+e3+e4+e5+e8+e10+e11), sha256
`9b5cb47b164f04ffa4e45d25a966e1e9dc2a928b438e97bd2825d4efb7e45d81`,
md5 `a994d46e28906f4e1f8043176bbb2b03`, 132210 bytes.
Decisive back-to-back A/B against the previous best (e8): e8 **76.75 / 76.91** vs
e11 **75.33 / 74.72**; e8's own three independent runs were 76.68 / 76.48 / 76.78. Three
independent 5-repeat runs of the shipped file afterwards: **74.610 / 74.542 / 74.914** → call it
**74.6 us**. `verify.py --runs 10`: 10 bit-identical runs, errors exactly the baseline's.

`verify.py` passes with `max_abs_err` **exactly the baseline values** (2.999e-03 / 1.928e-03 /
6.033e-04) after every edit shipped — everything is arithmetic bit-identical. `launch_rules.py`
passes. `semaphores.py` clean (no wait exceeds its supply) on every shipped state.

---

## 0. THE HEADLINE FINDING: **instruction-slot insertion works**

iter-2 listed this as "untested — worth one careful experiment". It is now tested:

> **Inserting 64-byte slots into an engine's `*0.bin` stream works on the device.**
> Test: inserted one no-op `EVENT_SEMAPHORE` at Activation index 1 (105 → 106 slots),
> `neff.py pack`, `verify.py` → bit-identical PASS.

Why it is safe here: nothing records a stream's instruction count (`Pool0.json` is
`{"instr","name","dma"}`; `Activation0.json` adds `act_semaphore` and a *name* list of function
sets; `act_info.json` has no PC references), there are **no branch instructions and no
`PSEUDO_EXTENSION` slots in any of the five streams** (checked: Act = ACTIVATE/CAST/
ACT_TABLE_LOAD/ES/COPY/ACT_READ_ACC/SET_ORDERING only; Pool, SP, DVE, PE likewise), and
`neff.py pack` recomputes member sizes and both MD5s. `compiler_pc` in the profile is simply the
slot index, so after an insertion the profile's pcs shift — remember that when comparing traces.

This unlocks **moving work between engines**, which was the blocked lever of the last two
iterations.

## 1. The second finding: Activation has 8 completely unused DGE queues

`def.json → dma_queue.qActDynamicHW` is `num_queues: 16, semaphore_set: [72..87]`.
The compiler uses only **72–79** (iter-2's eight Activation `CAST`s bump those). **80–87 are
free queues with free semaphores**, and Activation is 0% busy from 12 us to 28 us while Pool
grinds through its 19-DMA weight prologue at 0.66 us each.

So: a Pool prologue DMA can be *moved* to Activation verbatim — copy the 64 bytes, change only
the `semaphore` field to 80+j — and the vacated Pool slot becomes

```
EVENT_SEMAPHORE  wait s(80+j) >= 16   SEM_ADD_IMM_COMPLETE(original Pool sem, 16)
```

which keeps **every downstream wait in the program byte-identical** (no consumer re-pointing at
all) and costs 0.03 us instead of 0.66. That is the cheapest possible way to parallelise an
issue-bound DGE chain. `DGETable0` (var_id 25, `list: [2,1,7,6,4,3,5,11,13,14]`) is global, so a
DRAM-sourced descriptor resolves the same from any engine — W_in_qkv tiles moved to Activation
read correctly.

## 2. The method that found the targets

Rebuilt iter-2's **backwards critical-path walk** (`an/cp.py`, kept in this iteration's worktree
but *not* in the shipped NEFF; rewrite it from the recipe below, it is ~90 lines):

* aggregate `prof/Instruction.parquet` by `(engine, compiler_pc)` → `start=min`, `end=max`
  (a `PSEUDO_DMATRIGGER` emits **two** `WRITE` rows, so dedupe or you get a KeyError);
* 116 rows have `compiler_pc == NaN` — those are the runtime prologue/epilogue, drop them;
* for each instruction, `gate = max(previous pc's end on the same engine, time each wait value
  was reached)` from `SemaphoreUpdate.parquet`, jump to the winner, recurse from the last
  instruction;
* to find the *producer* of a semaphore value, do **not** search by timestamp proximity over all
  instructions — build `sid -> [(engine, pc)]` from the decoded streams (`events_update_idx`,
  `semaphore` + `sem_increment`, `events_extended_update_idx`) and pick the latest one whose end
  is `<= ts`. For queued transfers fall back to `DmaPacketAggregated` (`semaphore_id`, `trigger`).
* engine name map: `Tensor/Scalar/GpSimd/Vector/Sync` = `PE/Activation/Pool/DVE/SP`.

The 83.2 us critical path (span 12.1 → 94.7):

| window | us | gate |
|---|---|---|
| 12.1 → 26.0 | 13.9 | **Pool prologue: 19 serial DGE DMAs, zero waits, 100% busy** |
| 26.0 → 32.5 | 6.5 | S[11]>=32 (last prologue DMA) then DVE/Act compute |
| 32.5 → 40.1 | 7.6 | SP 5–16 DGE issue + S[60] completion |
| 40.1 → 45.4 | 5.3 | **S[65]>=16** = SP 16's SB→SB DMA completion |
| 45.4 → 48.3 | 2.9 | **S[70]>=16** = SP 21's partition→row transpose completion |
| 48.3 → 54.6 | 6.3 | SP 30–37 issue + **S[63]>=32** |
| 54.6 → 61.5 | 6.9 | **S[31]** (queued blk11) + SP 49–60 issue |
| 61.5 → 68.0 | 6.5 | **S[56]>=48** (SP 60) + **S[38]** (queued blk14) |
| 68.0 → 89.1 | 21.1 | **Activation tail, ~100% busy** (6 × `ACT_TABLE_LOAD` = 7.7 us) |
| 89.1 → 94.7 | 5.6 | output stores + epilogue |

**~27 us of the 83 is pure DMA-completion waiting**, ~14 us is Pool DGE issue, ~21 us is
Activation compute. Nothing else matters.

## 3. The edits, in order, each A/B-measured back-to-back

Device drift is still real (iter-2 §0): always alternate `A B A B` in one loop. Spread on a
5-repeat median ran 0.6–6% here.

| # | edit | measured |
|---|---|---|
| e1 | **8 duplicate SP type-A DMAs → Pool compute `CAST`s.** SP 5,6,7,8,9,12,17,22 are `SB[p][X] -> SB[p][Y]` bf16→fp32 per-partition copies whose twins at Pool 34,36,38,40,43,51,53,55 were iter-2 `EVENT_SEMAPHORE`s. Pool slot becomes `CAST` (keeping the wait `s(56+k)>=16`, `SEM_ADD_IMM_COMPLETE(pool_sem,16)` — wait value and add value are both 16, so the shared value field is not a problem); SP slot becomes an ES that only keeps its own count. | **83.20 → 83.20: exactly neutral alone** |
| e2 | **8 Pool prologue DMAs → Activation DGE queues 80–87** (§1). Moved Pool 13,14,15,16,17,18,19,20, inserted as Activation slots 1–8. | 83.3 → **82.4–82.6** (−0.8) |
| e1+e2 | together | **79.64** (spread 0.62%) — i.e. **−3.6**, far more than the sum. e1 only pays once the prologue stops being the long pole. |
| e3 | **Hoisted the `qSPIO0` block-0 trigger (SP slot 3, `dt_bias`+`A_log`, no wait) to SP slot 1** (rotation: new[1,2,3] = old[3,1,2]; a plain swap with slot 1 deadlocks, because slot 2 waits on slot 1's transfer). Its 2.6 us completion was a critical-path gate at 27.4 us. | — |
| e4 | **SP 14 and SP 16 (the last two type-A SB→SB DMAs) → Pool `CAST`s in the vacuous Pool pure-wait slots 50 and 52**, waiting `s72>=16` / `s73>=16` (the Activation `CAST`s that produce their sources) and adding 16 to s63 / s65. SP 14/16 become complete no-ops; the counts stay at 32 because the second producer (SP 37 / SP 44) is untouched. Kills the 2.8 us `S[65]` gate. | e3+e4: **78.6–78.9** (−0.8) |
| e5 | **Issue the merged V3 (`conv_weight`) DMA first on Pool** (permutation of the six no-wait prologue DMAs: Pool 1..6 = old 6,1,2,3,4,5). After e2 the very first gate of the kernel is `S[13]>=16` — that DMA's completion — and five unrelated DMAs were pushing it 3.3 us later. | **78.3** (−0.4) |
| e6 | **Merged the four partition→row transposes SP 18,19,20,21 into one 2-D DGE DMA** (`src 218880 step [262144,-768] num [128,4] elem 2` → `dst 212224 step [2,256] num [128,4] elem 2`) in SP 18 with semaphore 70, `wait s65>=16`; SP 19/20/21 → ES `wait s70>=16, add 16 to s67/s68/s69`. | **HANGS — reverted, see §4** |
| e7 | Same merge for the *other* four transposes, SP 27,32,37,38 (`src 18432 step [262144,256]` → `dst 218368 step [2,-768]`, `num [128,4]`), merged DMA in SP 27 (sem 58), SP 32/37/38 → ES `wait s58>=32, add 32 to s60/s63/s64`. Runs fine, bit-identical. | **80.4 — a 2.3 us REGRESSION, reverted** |
| e8 | **Dropped the `own-semaphore >= 16` queue-guard waits from the four moved DMAs** (ex-Pool 17–20). Those guards exist to stop a DGE queue being re-used while a transfer is outstanding; on act queues 84–87, which are used exactly once, they are vacuous — and one of them (`S[8]>=16`, waiting on the completion of the W_in_qkv tile two slots earlier on Pool) had become the *head* of the whole critical path. | 78.07 → **76.68** (−1.4) |
| e9 | **Moved the inserted block of 8 DMAs from Activation slot 1 to slot 4**, i.e. *after* `ES(s13) / ACT_TABLE_LOAD / ACTIVATE`. Activation slot 3's `ACTIVATE` is the root of the whole kernel's dependency chain (everything hangs off its `S[4]>=1`), so 5 us of DGE issue in front of it was delaying every engine. | see §6 |

| e10 | **Dropped the vestigial waits on SP 47, 48, 49** — the `qSPSpillReload0` triggers of blocks 9, 10, 11, which iter-1 had already reduced to 2-byte dummy writes into the dead DRAM tmp-buf `var21`. Their waits were `s29/s30/s31 >= 16`, i.e. the completions of blocks 5/6/7; **nothing in the program waits on s33/s34/s35**, so the triggers exist only to keep the queue advancing, and the queue is serial so ordering is kept by construction. `S[31]>=16` at SP 49 was a **4.7 us** gate. (Exactly iter-2's edit to SP 50.) | — |
| e11 | **Dropped the waits on SP 39, 40, 41, 42** — the triggers of blocks 5, 6, 7, 8. Decoding `SP0.json` showed these four queued transfers are **byte-identical duplicates of the SP 27/32/37/38 DGE transposes** (`SB +18432/18688/18944/19200, sizes [2,128], steps [1,262144]` → `SB +218368/217600/216832/216064, sizes [2,128], steps [1,2]`), so the queued repeat is idempotent (iter-1's writer/reader argument) and its sources are ready from `s6>=25` onward. `S[63]>=32` at SP 41 was a **2.7 us** gate. | e10+e11: 76.8 → **74.6** (−2.1) |

### Vacuous slots found (free real estate, no insertion needed)

* **Pool 33, 35, 37, 39, 42, 50, 52, 54** — pure-wait ES on `s56/57/58/59/60/61/66/71 >= 16`,
  each immediately followed by another instruction with the *identical* wait. 100% redundant.
  e4 used 50 and 52.
* **Activation 4, 6, 8, 10, 12** — iter-2 re-pointed all of them to `s13 >= 16`, the same
  condition as Activation 1, so 4/6/8/10/12 are redundant. Unused so far.
* **SP 23, 25, 43, 45, 51, 54** — same twin-wait pattern on `s74..s79`. All late in SP's program
  (≥ 40 us), so only useful for late work.

## 4. Refuted / learned the hard way

* **`verify.py` with one run does NOT catch a hang; `profile.py` does.** e6 passed `verify.py`
  (3 outputs, correct to the last bit) and then **deadlocked** under `profile.py --repeats 3`:
  `Failed to schedule execution on XU[1]` → `Model stop timeout (30000 ms) for TPBs`. A single
  cold run can win a race that 5×5 repeated executions lose.
  **New check order: `semaphores.py` → `pack` → `verify.py --runs 3` → `profile.py --repeats 3`,
  and treat "no measurement" as a failure, not a flake.** (It is also worth re-running
  `profile.py` on the *final* file even if nothing changed since the last measurement.)
* **Merging the four partition→row transposes is closed, from both directions.**
  * SP 18/19/20/21 (e6): merged into one `src [262144,-768] × [128,4]` → `dst [2,256] × [128,4]`
    DMA — correct output, then **hangs**. I could not isolate why in the time left; the
    candidates are the 512-descriptor sub-word-strided destination pattern and the three
    forwarding ESs on the merged DMA's own semaphore.
  * SP 27/32/37/38 (e7): the same transformation, with every waiter of s58/s60/s63/s64 audited
    (`an/waiters.py`) — runs, bit-identical, and is a **2.3 us regression** (80.4 vs 78.1). Four
    tiny transposes on four *different* DGE queues complete in parallel; one merged transfer on
    one queue serialises them. This is iter-2's "merging only pays when the tiles are small"
    rule showing up as "merging never pays when the queues were the parallelism".
  **So: do not merge transfers that already sit on distinct DGE queues.** The gain from killing
  a queued/DGE completion hop has to come from moving the work to a *compute* engine (e1/e4) or
  from taking the issue off a busy engine (e2), not from merging.
* **Where an inserted block of DMAs goes inside the host engine's stream matters more than how
  many you move** (e9 vs e2, ~3 us). Put moved transfers *after* the host engine's own
  critical-chain head, and strip their queue guards (e8).
* **Removing SP's SB→SB DMA chain alone buys nothing** (e1 = 0.00 us). The slack simply sits
  elsewhere. Only after the Pool prologue shrank did the same edit become worth ~3 us. Lesson:
  on this kernel, A/B a lever **both alone and in combination with the current best**; a 0.0 us
  reading does not mean the lever is closed.
* **Shortening the *tail* of the Pool prologue is worthless.** After e2 the critical path leaves
  Pool at slot 6 (17.2 us) — Pool slots 7–32 (another 9 us of DGE issue, incl. the 256 KB
  `recurrent_state` load at Pool 29) are entirely off the critical path. Only the DMAs *ahead of
  the first gate* matter, which is why e5 (a permutation) paid and why moving Pool 27–32 to
  Activation would pay nothing. Do not spend further effort there.
* **`neff.py pack` changes the file length** (132170 → 132197 → 132249 bytes) even for pure
  in-place field edits; that is header JSON re-serialisation, not a problem.
* A **bf16 → FP32 compute `CAST` works on Pool** even though no such dtype pair appears anywhere
  in the original NEFF (every compiler-emitted CAST is FP32→bf16 or bf16→bf16). Verified
  bit-identical.
* **Pool's `CAST` opcode byte is 0x47, the same as Activation's**, and the struct is
  `common.NEURON_ISA_TPB_S4D4_TR_STRUCT`, so a CAST slot can be **byte-copied between engines**
  and then patched. Setting `src_mem_pattern_start_addr_addr_immediate` (u32 @12) overwrites
  `..._reg_regnum` (u8 @12), which is exactly how you convert a register-based address to an
  SBUF immediate.

## 5. Where the time is now, and the ranked next steps

Critical path of the **e5 build** (78.06 us, span 12.4 → 91.4), measured with `an/cp.py`:

| window | us | gate |
|---|---|---|
| 12.4 → 15.2 | 2.8 | Pool slots 1–2 (merged V3, then the first W_in_qkv tile) |
| 15.2 → 17.9 | 2.7 | `S[8]>=16` = that W_in_qkv tile's completion, waited on by a **moved** DMA's queue guard — **e8 removes this** |
| 17.9 → 21.3 | 3.4 | the 8 moved DMAs issuing on Activation, in front of Activation's own chain head |
| 21.4 → 23.1 | 1.7 | `ACT_TABLE_LOAD` (1.28) + the first `ACTIVATE` — **this is the root of the whole kernel's dependency graph** |
| 23.1 → 25.3 | 2.2 | DVE 8–14, SP 2, then `S[24]>=16` (spill blk0) |
| 25.4 → 26.7 | 1.4 | `S[25]>=16` (spill blk1) |
| 26.7 → 39.3 | 12.6 | Activation compute (3 × `ACT_TABLE_LOAD` = 3.9 us) + DVE |
| 39.3 → 44.5 | **5.2** | `S[70]>=16` = SP 21's partition→row transpose completion |
| 44.5 → 50.8 | 6.3 | SP 28–41 issue + `S[63]>=32` (SP 37 transpose, 2.7) |
| 50.9 → 55.7 | **4.7** | `S[31]>=16` = queued blk7 completion |
| 55.8 → 60.5 | 4.7 | SP 49–60 issue + `S[56]>=48` (SP 60 transpose, 1.7) |
| 60.6 → 65.0 | **4.4** | `S[38]>=16` = queued blk14 completion |
| 65.0 → ~85 | 20 | **Activation tail, ~100% busy** (6 × `ACT_TABLE_LOAD` = 7.7 us) |
| 85 → 91.4 | 6.4 | output stores + epilogue |

Engine busy at 78 us: Activation 45.8 (59%), DMA 26.6, Pool 23.4, DVE 17.8, SP 13.8, PE 7.6.
**Activation is now clearly the busiest engine and the only one worth attacking as "work".**
Everything else is still latency.

1. **The Activation tail is now ~25% of the runtime and is the single biggest block.** 6
   `ACT_TABLE_LOAD`s (7.7 us) alternate `act_tbl_sel` 2 (silu) / 4 (reciprocal_sqrt); iter-2
   proved no shipped function set holds both, so the loads cannot be *deleted* by a field edit —
   but **reordering the tail so same-table work batches** has never been attempted and is worth
   ~5 us. Now that insertion works you can also *move* tail work to Pool/DVE (both <30% busy
   there) — except that most tail `CAST`/`COPY` slots use **register-based addressing**
   (`S@_reg_regnum=8`, `D@_reg_regnum=8`), and the register file is engine-local, so only the
   immediate-addressed ones can move. Start by decoding the `activation_func` id of every
   `ACTIVATE` between two table loads: that tells you which ones are movable at all.
2. **`S[38] = qSPSpillReload0 block 14`, 4.4 us, gating the start of the Activation tail.**
   Decoded (`SP0.json → dma`, id 14, semaphore 38): **four `cast` descriptors**,
   `SB p0 +218368/217600/216832/216064, 256 B bf16` → `SB p32 +213760/214272/214784/215296,
   512 B fp32`. Its **only waiter is Activation slot 68's `ACTIVATE`**. Four compute `CAST`s
   (`num_active_channels=1`, `num_elem=[128,1,1,1]`, template = Activation 63/65) would do this
   in ~1 us with instant completion. **The one unknown that blocks it: how a compute instruction
   addresses a partition other than 0** — blk14's destination is partition 32
   (`to_off 8602368 = 32*262144 + 213760`), while a compute `..._start_addr_addr_immediate` is
   only the within-partition byte offset. Find that out (ch5) and this is ~4 us. Note the
   trigger (SP 72) cannot simply be fired earlier: `qSPSpillReload0` is one serial queue, so
   blk14 cannot start before blk13.
3. **`S[31] = qSPSpillReload0 block 7`, 4.7 us.** Decoded: a single `copy` descriptor,
   `SB +18944, from_sizes [2,128], from_steps [1,262144]` → `SB +216832, to_sizes [2,128],
   to_steps [1,2]` — i.e. **the exact same partition→row transpose that SP 37's DGE DMA already
   performs**. blk5/6/8 are presumably the duplicates of SP 27/32/38. If so these four queued
   transfers are pure duplicates and their triggers (SP 39–42) can become bare
   `SEM_ADD_IMM_COMPLETE`s on the matching semaphores (iter-2 Finding B). **Verify first that
   SP 27/32/37/38 and blk5–8 really are byte-identical pairs** (iter-1 §4: a tmp-buf can have a
   second consumer you did not see). Worth ~4.7 us plus whatever blk5/6/8 cost.
4. **A `PSEUDO_DMATRIGGER` costs 0.09–0.12 us versus 0.49–0.66 us for a DGE `PSEUDO_DMA_DIRECT2D`**
   (measured on this profile). Converting issue-bound DGE chains into *queued* transfers is a 5x
   saving per instruction, but needs new blocks in `SP0.json → dma` (and `Pool0.json`/
   `Activation0.json` have empty `dma` lists, so a Pool/Act queue would have to be defined from
   scratch). Untried.
5. **Do not bother with:** the output tail (iter-2 §4, it is slack); `ACT_TABLE_LOAD` field edits
   (closed twice); PE (7.6 us busy); merging 32 KB tiles (iter-2's size rule); merging transfers
   that already sit on distinct DGE queues (§4); moving Pool 27–32 off Pool (§4); moving *more*
   prologue DMAs onto Activation (§6 variant eA).

## 6. Final state and the measurement ladder

Shipped = **e1 + e2(8 DMAs at Activation slot 1, queue guards dropped) + e3 + e4 + e5 + e10 +
e11** (= the file labelled `e11`/`eC` above). e6, e7, e9 and eA were built, measured and reverted.

| state | what it is | median (back-to-back A/B) |
|---|---|---|
| iter-2 (HEAD) | — | 83.02 / 83.20 / 83.31 / 83.32 |
| e1 | Pool CASTs only | 83.20 (neutral) |
| e2 | moved DMAs only | 82.41 / 82.60 |
| e1+e2 | | 79.53–80.02 (5 repeats), median 79.64 |
| +e3+e4 | | 78.61 / 78.80 / 78.90 |
| +e5 | | 78.07 / 78.18 / 78.28 / 78.27 |
| **+e8 (shipped)** | guards dropped | **76.38 / 76.61 / 76.68 / 76.78**; final 76.68 / 76.48 / 76.78 |
| +e6 | merged SP 18–21 | **hangs** |
| +e7 | merged SP 27/32/37/38 | 80.37 (regression) |
| **+e10+e11 (shipped)** | vestigial queued-trigger waits dropped on SP 47/48/49 and 39/40/41/42 | **75.33 / 74.72**, then 74.61 / 74.54 / 74.91 |
| e9 | DMA block at Activation slot 4 | 81.04 / 81.17 (regression) |
| eA | only 4 DMAs moved | 78.39 / 78.51 (regression) |

Changed members vs the iter-2 file: `sg00/Activation0.bin` (105 → **113** slots — the only
member whose length changes), `sg00/Pool0.bin` (24 slots differ), `sg00/SP0.bin` (13 slots
differ). No JSON changed. `verify.py --runs 10`: all 10 bit-identical, errors exactly the
baseline's. `launch_rules.py`: 1 subgraph, `num_tpb 1`, 11 inputs, 3 output heads, LNC=1,
one rank.

Reproducing: scripts `an/e1.py … an/e5.py` + `an/cp.py` + `an/waiters.py` + `an/d.py` lived in
the iteration worktree (reverted before judging). The shipped file is
`work_base → e1(all) → e2(13,14,15,16,17,18,19,20 drop) → e3 → e4(all) → e5`; each script is
idempotent-ish and asserts the slot it is about to overwrite, so rebuilding from a fresh unpack
is mechanical. Rewrite them from §0–§3 if they are gone — the per-slot tables above contain
every address, semaphore and value.

## 7. Appendix — the edit scripts, verbatim

Run from the worktree root against an unpacked `work/`, in this order:
`e1.py work all`, `e2.py work 13,14,15,16,17,18,19,20 drop`, `e3.py work`, `e4.py work all`, `e5.py work`.

### an/e1.py
```python
"""Convert the 8 duplicate SP type-A per-partition DMAs into Pool compute CASTs.
usage: e1.py <work_root> <group indices, comma or 'all'>"""
import sys, shutil
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa

ROOT = sys.argv[1]
sel = sys.argv[2]

# (sp_slot, sp_sem, src, dst, pool_slot, pool_sem)
GROUPS = [
    (5,  56, 16896, 223488, 34,  8),
    (6,  57, 17152, 222720, 36,  9),
    (7,  58, 17408, 221952, 38, 10),
    (8,  59, 17664, 221184, 40, 11),
    (9,  60, 17920, 220416, 43, 12),
    (12, 61, 18176, 219648, 51, 16),
    (17, 66, 16384, 225024, 53, 17),
    (22, 71, 16640, 224256, 55, 18),
]
idx = list(range(len(GROUPS))) if sel=='all' else [int(x) for x in sel.split(',')]

def load(eng):
    p=f'{ROOT}/payload/sg00/{eng}0.bin'
    return p, bytearray(open(p,'rb').read())

sp_p, sp = load('SP')
po_p, po = load('Pool')
act = bytearray(open(f'{ROOT}/payload/sg00/Activation0.bin','rb').read())

CAST_TMPL = bytes(act[35*64:36*64])          # immediate-addressed SB->SB CAST (iter-2)
ES_TMPL   = bytes(sp[13*64:14*64])           # SP ES with SEM_ADD_IMM_COMPLETE

pool_cast_op = po[41*64]
print('Pool CAST opcode byte', hex(pool_cast_op), ' Act CAST opcode byte', hex(CAST_TMPL[0]))
assert pool_cast_op == CAST_TMPL[0], 'opcode mismatch between engines'

for g in idx:
    sp_slot, sp_sem, src, dst, po_slot, po_sem = GROUPS[g]
    # --- Pool slot: ES -> CAST
    c = isa.decode(CAST_TMPL)
    c.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM')
    c.set('events_wait_idx', sp_sem)
    c.set('events_update_mode','SEM_ADD_IMM_COMPLETE')
    c.set('events_update_idx', po_sem)
    c.set('events_semaphore_value', 16)
    c.set('src_mem_pattern_start_addr_addr_immediate', src)
    c.set('dst_mem_pattern_start_addr_addr_immediate', dst)
    c.set('src_mem_pattern_step_elem',[1,1,1,1]); c.set('src_mem_pattern_num_elem',[1,1,1,1])
    c.set('dst_mem_pattern_step_elem',[1,1,1,1]); c.set('dst_mem_pattern_num_elem',[1,1,1,1])
    c.set('in_dtype','BFLOAT16'); c.set('out_dtype','FP32')
    c.set('num_active_channels',128)
    po[po_slot*64:(po_slot+1)*64] = c.encode()
    # --- SP slot: DMA -> ES that only keeps the semaphore count
    e = isa.decode(ES_TMPL)
    e.set('events_wait_mode','NONE'); e.set('events_wait_idx',0)
    e.set('events_update_mode','SEM_ADD_IMM_COMPLETE')
    e.set('events_update_idx', sp_sem)
    e.set('events_semaphore_value', 16)
    sp[sp_slot*64:(sp_slot+1)*64] = e.encode()
    print(f'group {g}: Pool {po_slot} = CAST SB+{src}->SB+{dst} bf16->fp32 wait s{sp_sem}>=16 add16 s{po_sem}; SP {sp_slot} = ES add16 s{sp_sem}')

open(sp_p,'wb').write(sp); open(po_p,'wb').write(po)
print('written')
```

### an/e2.py
```python
"""Move N Pool prologue DGE DMAs onto Activation's (idle) DGE queue, using the
unused qActDynamicHW semaphores 80..87.  The vacated Pool slot becomes an
EVENT_SEMAPHORE that waits for the moved transfer and then adds 16 to the
original Pool semaphore, so every downstream wait in the program is untouched.
usage: e2.py <work_root> <pool slots, comma separated>  (max 8)"""
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa

ROOT=sys.argv[1]
slots=[int(x) for x in sys.argv[2].split(',')]
assert len(slots)<=8
DROP_QUEUE_GUARD = 'drop' in sys.argv[3:]
AT = int([a.split('=')[1] for a in sys.argv[3:] if a.startswith('at=')][0]) if any(a.startswith('at=') for a in sys.argv[3:]) else 1

pp=f'{ROOT}/payload/sg00/Pool0.bin'; po=bytearray(open(pp,'rb').read())
ap=f'{ROOT}/payload/sg00/Activation0.bin'; act=bytearray(open(ap,'rb').read())
ES_TMPL=bytes(po[7*64:8*64])    # Pool ES with SEM_ADD_IMM_COMPLETE, no wait

new=[]
for j,s in enumerate(slots):
    sem_new = 80+j
    d = isa.decode(bytes(po[s*64:(s+1)*64]))
    assert d.name=='PSEUDO_DMA_DIRECT2D', (s,d.name)
    sem_old = d['semaphore']
    wi = d['events_wait_idx']; wm = d.as_dict().get('events_wait_mode'); wv = d['events_semaphore_value']
    # the moved copy: identical transfer, new (unused) act DGE queue/semaphore
    m = isa.decode(bytes(po[s*64:(s+1)*64]))
    m.set('semaphore', sem_new)
    if DROP_QUEUE_GUARD:
        m.set('events_wait_mode','NONE'); m.set('events_wait_idx',0); m.set('events_semaphore_value',0)
    new.append(m.encode())
    # the Pool placeholder: wait for it, then keep the original semaphore count
    e = isa.decode(ES_TMPL)
    e.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); e.set('events_wait_idx', sem_new)
    e.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); e.set('events_update_idx', sem_old)
    e.set('events_semaphore_value', 16)
    po[s*64:(s+1)*64]=e.encode()
    print(f'Pool {s:2d} (sem {sem_old}, wait {wm} s{wi}>={wv}) -> Activation slot {AT+j} on sem {sem_new}; Pool {s} = ES wait s{sem_new}>=16 add16 s{sem_old}')

act[AT*64:AT*64] = b''.join(new)
open(pp,'wb').write(po); open(ap,'wb').write(act)
print(f'Activation0.bin {len(act)} bytes = {len(act)//64} slots')
```

### an/e3.py
```python
"""Hoist the qSPIO0 block-0 trigger (SP slot 3, no wait: dt_bias + A_log input load)
to the front of the SP stream so its 2.6 us completion is off the critical path.
Rotation only: new[1]=old[3], new[2]=old[1], new[3]=old[2]."""
import sys
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(p,'rb').read())
s=lambda i: bytes(d[i*64:(i+1)*64])
o1,o2,o3 = s(1),s(2),s(3)
d[1*64:2*64]=o3; d[2*64:3*64]=o1; d[3*64:4*64]=o2
open(p,'wb').write(d)
print('SP slots 1,2,3 = old 3,1,2 (IO0 blk0 trigger hoisted to slot 1)')
```

### an/e4.py
```python
"""Replace the last two type-A SP SB->SB per-partition DMAs (SP 14 -> s63, SP 16 -> s65)
with compute CASTs on Pool, placed in the vacuous pure-wait Pool slots 50 and 52.
The SP slots become complete no-ops; the Pool CAST carries the semaphore increment."""
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
sel = sys.argv[2] if len(sys.argv)>2 else 'all'
# (sp_slot, src, dst, pool_slot, precondition_sem, target_sem)
JOBS=[ (14, 20992, 218112, 50, 72, 63),
       (16, 21504, 216576, 52, 73, 65) ]
idx = range(len(JOBS)) if sel=='all' else [int(x) for x in sel.split(',')]

pp=f'{ROOT}/payload/sg00/Pool0.bin'; po=bytearray(open(pp,'rb').read())
sp_p=f'{ROOT}/payload/sg00/SP0.bin'; sp=bytearray(open(sp_p,'rb').read())
act=open(f'{ROOT}/payload/sg00/Activation0.bin','rb').read()
# an immediate-addressed SB->SB CAST template: find one in Activation (iter-2's)
CAST_TMPL=None
for i,x in isa.decode_stream(act):
    if x.name=='CAST' and x['src_mem_pattern_start_addr_addr_reg_regnum']==0 \
       and x['num_active_channels']==128 and x['src_mem_pattern_num_elem'][0]==1:
        CAST_TMPL=bytes(x.raw); break
assert CAST_TMPL
ES_TMPL=bytes(sp[23*64:24*64])   # SP pure-wait ES

for j in idx:
    sp_slot, src, dst, po_slot, pre, tgt = JOBS[j]
    old=isa.decode(bytes(po[po_slot*64:(po_slot+1)*64]))
    assert old.name=='EVENT_SEMAPHORE' and old.as_dict().get('events_update_mode')=='NONE', (po_slot, old.name)
    c=isa.decode(CAST_TMPL)
    c.set('events_wait_mode','WAIT_FOR_SEM_GE_IMM'); c.set('events_wait_idx',pre)
    c.set('events_update_mode','SEM_ADD_IMM_COMPLETE'); c.set('events_update_idx',tgt)
    c.set('events_semaphore_value',16)
    c.set('src_mem_pattern_start_addr_addr_immediate',src)
    c.set('dst_mem_pattern_start_addr_addr_immediate',dst)
    c.set('src_mem_pattern_step_elem',[1,1,1,1]); c.set('src_mem_pattern_num_elem',[1,1,1,1])
    c.set('dst_mem_pattern_step_elem',[1,1,1,1]); c.set('dst_mem_pattern_num_elem',[1,1,1,1])
    c.set('in_dtype','BFLOAT16'); c.set('out_dtype','BFLOAT16'); c.set('num_active_channels',128)
    po[po_slot*64:(po_slot+1)*64]=c.encode()
    e=isa.decode(ES_TMPL)
    e.set('events_wait_mode','NONE'); e.set('events_wait_idx',0); e.set('events_semaphore_value',0)
    e.set('events_update_mode','NONE'); e.set('events_update_idx',0)
    sp[sp_slot*64:(sp_slot+1)*64]=e.encode()
    print(f'Pool {po_slot} = CAST SB+{src}->SB+{dst} wait s{pre}>=16 add16 s{tgt}; SP {sp_slot} = no-op')
open(pp,'wb').write(po); open(sp_p,'wb').write(sp)
```

### an/e5.py
```python
"""Issue the merged V3 (conv_weight) load first on Pool.  Its completion (S[13]>=16)
is the first gate of the whole kernel; the five no-wait DMAs ahead of it only pushed
it 3.3 us later.  Pure permutation of six no-wait DGE DMAs on distinct queues."""
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/Pool0.bin'; d=bytearray(open(p,'rb').read())
sl=lambda i: bytes(d[i*64:(i+1)*64])
for i in range(1,7):
    x=isa.decode(sl(i))
    assert x.name=='PSEUDO_DMA_DIRECT2D', (i,x.name)
    assert x.as_dict().get('events_wait_mode')=='NONE', (i,'has a wait')
six=sl(6)
rest=[sl(i) for i in range(1,6)]
d[1*64:2*64]=six
for k,b in enumerate(rest):
    d[(2+k)*64:(3+k)*64]=b
open(p,'wb').write(d)
print('Pool slots 1..6 = old 6,1,2,3,4,5  (merged conv_weight DMA now issues first)')
```

### an/cp.py — the backwards critical-path walk (§2)
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa, pandas as pd

ENG = {'Tensor':'PE','Scalar':'Activation','GpSimd':'Pool','Vector':'DVE','Sync':'SP'}
REV = {v:k for k,v in ENG.items()}
ROOT = sys.argv[1] if len(sys.argv)>1 else 'work'
PROF = sys.argv[2] if len(sys.argv)>2 else 'prof'

ins = pd.read_parquet(f'{PROF}/Instruction.parquet')
sem = pd.read_parquet(f'{PROF}/SemaphoreUpdate.parquet')
sem['sid'] = pd.to_numeric(sem['id'].str.extract(r'S\[(\d+)\]')[0], errors='coerce')
sem = sem[sem.sid.notna()].copy(); sem['sid']=sem['sid'].astype(int)
sem = sem.sort_values('timestamp')
t0 = ins['start_ts'].min()
ii = ins[ins.compiler_pc.notna()].copy(); ii['pc']=ii.compiler_pc.astype(int)
agg = ii.groupby(['engine','pc']).agg(start=('start_ts','min'), end=('end_ts','max'),
      dur=('duration_ns','sum'), opcode=('opcode','first')).reset_index()
agg['s']=(agg.start-t0)/1000.0; agg['e']=(agg.end-t0)/1000.0
sem['t']=(sem.timestamp-t0)/1000.0
T={(r.engine,r.pc):r for r in agg.itertuples()}
byeng={e:sorted(g.pc.tolist()) for e,g in agg.groupby('engine')}

streams={}
for en in set(ENG.values()):
    streams[en]={i:x for i,x in isa.decode_stream(open(f'{ROOT}/payload/sg00/{en}0.bin','rb').read())}

# map sid -> list of (engine, pc, kind)
updaters={}
for en,st in streams.items():
    for pc,x in st.items():
        d=x.as_dict()
        if d.get('events_update_mode') not in (None,0,'NONE'):
            updaters.setdefault(d['events_update_idx'],[]).append((REV[en],pc,'ES/'+x.name))
        s=d.get('semaphore')
        if s is not None and d.get('sem_increment',0):
            updaters.setdefault(s,[]).append((REV[en],pc,'DMA/'+x.name))
        if d.get('events_extended_update_mode') not in (None,0,'NONE'):
            updaters.setdefault(d['events_extended_update_idx'],[]).append((REV[en],pc,'ESx'))

def when(sid,v):
    g=sem[(sem.sid==sid)&(sem.value>=v)]
    return None if len(g)==0 else g.iloc[0]
def waits(en,pc):
    x=streams[en].get(pc)
    if x is None: return []
    d=x.as_dict(); out=[]
    if d.get('events_wait_mode') not in (None,0,'NONE'): out.append((d['events_wait_idx'],d['events_semaphore_value']))
    if d.get('events_extended_wait_mode') not in (None,0,'NONE'): out.append((d['events_extended_wait_idx'],d['events_extended_sem_wait_value']))
    return out
dpa = pd.read_parquet(f'{PROF}/DmaPacketAggregated.parquet')
dpa['sid']=pd.to_numeric(dpa['semaphore_id'].str.extract(r'S\[(\d+)\]')[0],errors='coerce')
def qproducer(sid, ts):
    g=dpa[(dpa.sid==sid)&(dpa.end_ts<=ts+400)]
    if len(g)==0: return None
    r=g.sort_values('end_ts').iloc[-1]
    trig=r.trigger
    c=agg[(agg.engine=='Sync')].assign(dd=(agg[agg.engine=='Sync'].end-trig).abs()).sort_values('dd')
    if len(c)==0: return None
    return (c.iloc[0].engine, c.iloc[0].pc, f'queued blk{r.block_id} {r.variable}')

def producer(sid, ts, exclude=None):
    best=None
    for (e,pc,k) in updaters.get(sid,[]):
        r=T.get((e,pc))
        if r is None: continue
        if (e,pc)==exclude: continue
        if r.end<=ts+60:
            if best is None or r.end>T[(best[0],best[1])].end: best=(e,pc,k)
    if best is None: return qproducer(sid, ts)
    return best

if len(sys.argv)>4:
    cur_eng=sys.argv[3]; cur_pc=int(sys.argv[4])
else:
    r=agg.sort_values('end').iloc[-1]; cur_eng=r.engine; cur_pc=r.pc
chain=[]; seen=set()
for _ in range(500):
    row=T[(cur_eng,cur_pc)]; en=ENG[cur_eng]
    order=byeng[cur_eng]; k=order.index(cur_pc)
    prev_pc=order[k-1] if k>0 else None
    cands=[]
    if prev_pc is not None: cands.append(('prog',T[(cur_eng,prev_pc)].e,('prog',prev_pc)))
    for si,sv in waits(en,cur_pc):
        u=when(si,sv)
        cands.append((f'S[{si}]>={sv}',0.0 if u is None else u.t,('sem',si,sv,None if u is None else u.timestamp)))
    if not cands:
        chain.append((cur_eng,cur_pc,row.opcode,row.s,row.e,row.dur,'START',0)); break
    cands.sort(key=lambda x:-x[1]); b=cands[0]
    chain.append((cur_eng,cur_pc,row.opcode,row.s,row.e,row.dur,b[0],b[1]))
    if b[2][0]=='prog': cur_pc=b[2][1]
    else:
        ts=b[2][3]
        if ts is None: break
        p=producer(b[2][1],ts,exclude=(cur_eng,cur_pc))
        if p is None:
            chain.append(('?',-1,f'unknown producer of S[{b[2][1]}]',0,0,0,'STOP',0)); break
        cur_eng,cur_pc=p[0],p[1]
    if (cur_eng,cur_pc) in seen:
        chain.append(('?',-1,f'LOOP at {cur_eng}:{cur_pc}',0,0,0,'STOP',0)); break
    seen.add((cur_eng,cur_pc))

print(f'{"eng":8s} {"pc":>4s} {"opcode":24s} {"start":>7s} {"end":>7s} {"dur":>7s}  gate')
prev_e=None
for c in reversed(chain):
    lat = '' if prev_e is None else f'  (+{c[3]-prev_e:5.2f} wait)'
    print(f'{c[0]:8s} {c[1]:4d} {str(c[2])[:24]:24s} {c[3]:7.2f} {c[4]:7.2f} {c[5]:7.0f}  {c[6]} @{c[7]:.2f}{lat}')
    prev_e=c[4]
```

### an/waiters.py — every user of a set of semaphores
```python
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]; sems=[int(x) for x in sys.argv[2].split(',')]
ENG=['PE','Activation','Pool','DVE','SP']
for en in ENG:
    for i,x in isa.decode_stream(open(f'{ROOT}/payload/sg00/{en}0.bin','rb').read()):
        d=x.as_dict(); out=[]
        if d.get('events_wait_mode') not in (None,0,'NONE') and d['events_wait_idx'] in sems:
            out.append(f"WAIT s{d['events_wait_idx']}>={d['events_semaphore_value']}")
        if d.get('events_extended_wait_mode') not in (None,0,'NONE') and d['events_extended_wait_idx'] in sems:
            out.append(f"XWAIT s{d['events_extended_wait_idx']}>={d['events_extended_sem_wait_value']}")
        if d.get('events_update_mode') not in (None,0,'NONE') and d['events_update_idx'] in sems:
            out.append(f"ADD s{d['events_update_idx']}+={d['events_semaphore_value']}({d['events_update_mode']})")
        if d.get('semaphore') in sems and d.get('sem_increment',0):
            out.append(f"DMA sem{d['semaphore']}+={d['sem_increment']}")
        if out: print(f"{en:11s} {i:4d} {x.name:22s} {' | '.join(out)}")
```

### an/d.py — compact stream decoder
```python
import sys
sys.path.insert(0, '/home/ubuntu/workspace/neff_doc/tools')
import isa

KEEP_PREFIX = ('events_','semaphore','sem_increment','dge_op','src_step_elem','src_num_elem','src_elem_size',
               'dst_step_elem','dst_num_elem','dst_elem_size','in_dtype','out_dtype','num_active_channels',
               'act_tbl_sel','act_func','perf_mode','count_union_block_id','num_elem','step_elem')

def short(ins):
    d = ins.as_dict()
    out = []
    for k,v in d.items():
        if 'reserved' in k or 'marker' in k: continue
        if k.endswith('_addr_immediate') or k.endswith('table_offset') or k.endswith('tbl_offs_offset_immediate') \
           or k.endswith('_reg_reg_hi') or k.endswith('offset_regs') or '_regs_' in k: continue
        if k.endswith('_var_offset'):
            out.append(f'{k[:-11]}=+{v}'); continue
        if isinstance(v,int) and v==0: continue
        if isinstance(v,str) and v=='NONE': continue
        if isinstance(v,list) and all(x==0 for x in v): continue
        out.append(f'{k}={v}')
    return ' '.join(out)

def var_ids(ins):
    d = ins.as_dict()
    r={}
    for k,v in d.items():
        if k.endswith('_var_var_id_lo') or k.endswith('var_id_lo'):
            r[k]=v
    return r

eng = sys.argv[1]
root = sys.argv[2] if len(sys.argv)>2 else 'work'
sel = None
if len(sys.argv)>3:
    sel = set()
    for part in sys.argv[3].split(','):
        if '-' in part:
            a,b = part.split('-'); sel.update(range(int(a),int(b)+1))
        else: sel.add(int(part))
data = open(f'{root}/payload/sg00/{eng}0.bin','rb').read()
for i,ins in isa.decode_stream(data):
    if sel is not None and i not in sel: continue
    print(f'{i:4d} {ins.name:28s} {short(ins)} | VAR {var_ids(ins)}')
```

### 6b. The two late wins (e10, e11) and what they open up

`qSPSpillReload0` holds 20 blocks. After iter-1/iter-2 a large fraction of them are **dead or
duplicated**, and their triggers' waits were still on the critical path:

| block | semaphore | what it does | status |
|---|---|---|---|
| 5, 6, 7, 8 | 29, 30, 31, 32 | per-partition→row transposes `SB +18432/18688/18944/19200` → `SB +218368/217600/216832/216064` | **byte-identical duplicates of the DGE DMAs at SP 27, 32, 37, 38** |
| 9, 10, 11, 12 | 33, 34, 35, 36 | 2-byte copies of `SB +212224` into `var21 +2/+4/+6/+8` | **dummies** left by iter-1; `var21` is dead |
| 13 | 37 | iter-1's 4-column gather into `SB +114944` | live |
| 14 | 38 | 4 × `cast SB p0 +218368/217600/216832/216064 (256 B bf16)` → `SB p32 +213760/214272/214784/215296 (512 B fp32)` | live; sole waiter Activation 68; **4.4 us gate** |

e10/e11 dropped the *waits* on the triggers of 5–11 (seven of the twenty). I deliberately kept
the triggers themselves, because nobody has yet verified that **skipping** a block is safe on a
serial queue — that is the obvious next step and would remove the transfers (and their queue
occupancy) entirely, not just the waits. Candidates: blocks 5–12 are all dead or duplicated, so
SP 39, 40, 41, 42, 47, 48, 49, 50 could in principle become plain no-ops. Change **one** of
them first and check with `profile.py --repeats 3` (not just `verify.py` — see §4).

Also still open on the same queue: block 12's trigger (SP 50) already has no wait, and
`s36 >= 16` (block 12's completion) is what gates **SP 72, the block-14 trigger**. Since block 12
is a dummy, that wait is vestigial too — but block 14 cannot start before block 13 anyway
(serial queue), so dropping it is probably worth little. The real prize is block 14 itself (§5.2).

### an/e10.py
```python
"""Drop the vestigial waits on SP 47,48,49 — the triggers of queued blocks 9,10,11,
which iter-1 reduced to 2-byte dummy writes into the dead DRAM tmp-buf var21.
Their waits (s29/s30/s31 >= 16) are the completions of queued blocks 5/6/7, which are
byte-identical duplicates of the SP 27/32/37 DGE transposes; nothing waits on s33/s34/s35.
qSPSpillReload0 is a single serial queue, so block ordering is kept by construction.
Exactly the edit iter-2 made to SP 50 (trigger of dummy block 12)."""
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(p,'rb').read())
for slot,sem in ((47,29),(48,30),(49,31)):
    x=isa.decode(bytes(d[slot*64:(slot+1)*64])); a=x.as_dict()
    assert x.name=='PSEUDO_DMATRIGGER' and a['events_wait_idx']==sem, (slot,x.name,a.get('events_wait_idx'))
    x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0)
    d[slot*64:(slot+1)*64]=x.encode()
    print(f'SP {slot}: dropped wait s{sem}>=16')
open(p,'wb').write(d)
```

### an/e11.py
```python
"""Drop the waits on SP 39,40,41,42 — the triggers of queued blocks 5,6,7,8, which are
byte-identical duplicates of the SP 27/32/37/38 DGE transposes (verified against SP0.json).
Each wait (s58/s60/s63/s64 >= 32) is the completion of the DGE DMA that already wrote the
same bytes from the same source, so the queued repeat is idempotent (iter-1's writer/reader
argument); the sources (SB 18432..19200) are ready from s6>=25 onward.  After e10 nothing
waits on s29..s32, so the triggers are kept only to keep the queue advancing in order.
S[63]>=32 at SP 41 was a 2.7 us gate on the critical path."""
import sys
sys.path.insert(0,'/home/ubuntu/workspace/neff_doc/tools')
import isa
ROOT=sys.argv[1]
p=f'{ROOT}/payload/sg00/SP0.bin'; d=bytearray(open(p,'rb').read())
for slot,sem in ((39,58),(40,60),(41,63),(42,64)):
    x=isa.decode(bytes(d[slot*64:(slot+1)*64])); a=x.as_dict()
    assert x.name=='PSEUDO_DMATRIGGER' and a['events_wait_idx']==sem and a['events_semaphore_value']==32, (slot,x.name,a.get('events_wait_idx'),a.get('events_semaphore_value'))
    x.set('events_wait_mode','NONE'); x.set('events_wait_idx',0); x.set('events_semaphore_value',0)
    d[slot*64:(slot+1)*64]=x.encode()
    print(f'SP {slot}: dropped wait s{sem}>=32')
open(p,'wb').write(d)
```

Shipped build order: `work_base -> e1(all) -> e2(13,14,15,16,17,18,19,20 drop) -> e3 -> e4(all) -> e5 -> e10 -> e11`.

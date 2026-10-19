# iter-4 — found and fixed **SP head-of-line blocking**: iter-2's 160 relay `EVENT_SEMAPHORE`s
# were stalling the Sync engine in front of ready DMA triggers that PE was waiting for.
# **1207.6 -> 1176.0 us (-2.6%; -12.9% vs the iter-0 baseline)**, outputs bit-identical.

## Result

| NEFF | total_exec_time_us (5 repeats) | samples |
|---|---|---|
| start (= iter-2 shipped, HEAD `f8380ab`) | **1207.620** | 1204.0 1205.6 1207.6 1208.4 1208.7 |
| shipped edit, run 1 | **1174.991** | 1173.8 1173.9 1175.0 1175.7 1178.7 |
| shipped edit, run 2 | **1175.696** | 1173.9 … 1177.9 (spread 0.34%) |
| shipped edit, run 3 | **1176.366** | 1174.0 … 1177.6 (spread 0.31%) |
| shipped edit, run 4 | **1176.638** | 1176.1 … 1178.4 (spread 0.19%) |

Four independent `--repeats 5` runs of the edit land in 1175.0–1176.6 (median of medians 1176.0)
against 1207.6 for the start NEFF (iter-3 measured that same NEFF twice, at 1207.7 and 1205.7).
The ~31 us is ~10x the ±3 us between-measurement floor, and the four runs' bands do not touch the
start's. `verify.py --runs 3`: PASS, **all 3 runs bit-identical**,
`max_abs_err` **identical to the start NEFF to every printed digit** (output_0 1.097e-04 = 10% of
the limit, output_1 9.081e-04 = 83%). `launch_rules.py`: PASS. `semaphores.py`: zero warnings.

**The edit touches `sg00/SP0.bin` only, and only as a permutation of 44 slots** (same file size,
same slot count, every semaphore total unchanged — `neff_diff` vs the iter-2 NEFF shows exactly
one changed member, 148 slots differing because 44 slots moved past up to 8 positions each).
Engine busy barely moved (ACT 520.4->521.2, DVE 535.0->535.0, PE 340.1->340.9, SP 51.7->51.8):
**this is a pure scheduling win, nothing was added or removed.**

---

## 1. The lever: an in-order engine blocks on its *own* program order, and iter-2 put 130 us of
## blocking waits onto the engine that issues every DMA trigger

iter-2 retired 160 DGE reloads onto the new `qSPSpillReload1` queue and relayed each completion
through an `EVENT_SEMAPHORE` on **SP**: `events_extended_wait = S[72+ring] >= 16k`, primary
`update S[56..71] += 16`. Those relays are correct, but SP is **in-order**, so a relay that waits
2 us stalls every slot behind it — and the slots behind it are `PSEUDO_DMATRIGGER`s, i.e. SP is
the only engine that starts a queued DMA.

Measured on the start NEFF (`Instruction.parquet`, Sync engine, `duration_ns` of an ES includes
the time it spends waiting):

* the 160 relay ES slots **blocked SP for 130.3 us** (median 487 ns, p90 1772 ns, max 2981 ns);
* **44 of them had at least one already-unblocked `PSEUDO_DMATRIGGER` immediately behind them,
  carrying 59.8 us of block — 57.7 us of it in front of a `qSPSpillReload0` trigger** (the queue
  that does every `dma_transpose` PE consumes).

Worked example (the one that pointed at this): SP slots 33-37 of the start NEFF.

```
33 EVENT_SEMAPHORE  extwait S[77]>=16  upd S[60]+=16   start=43769  dur=1709   <-- relay, blocks 1.7 us
34 PSEUDO_DMATRIGGER qSPSpillReload0 blk=18  wait S[24]>=32  start=45514 dur=44
35 EVENT_SEMAPHORE                            wait S[27]>=16  start=45598 dur=32
36 PSEUDO_DMATRIGGER qSPSpillReload0 blk=19  wait S[21]>=32  start=45670 dur=44  <-- the W_t transpose
37 EVENT_SEMAPHORE                            wait S[28]>=16  start=45749 dur=29
38 PSEUDO_DMATRIGGER qSPSpillReload0 blk=20  wait S[4]>=36   start=50079 (blocks 4.3 us)
```

Slots 34-37 each started immediately after the previous one, i.e. **their own waits were already
satisfied** — they were held up purely by slot 33's 1.7 us wait. blk 19 is the
`dma_transpose(W_t, W_full)` of line 595 whose completion the line-597 `LDWEIGHTS` waits for, and
line 597 was the single biggest whole-machine-idle site in the kernel (62 us).

### The edit (full recipe — `scratch/` does not survive, this is enough to rebuild it in ~20 min)
For each relay ES, move it **forward** past the maximal run of immediately-following slots that
(a) executed without blocking in the profile, (b) do not wait on a semaphore the relay updates,
and (c) contain at least one `PSEUDO_DMATRIGGER`; stop the run at the first slot that blocked, and
trim trailing non-triggers. 44 relays qualified. Implemented as a permutation of `SP0.bin`.

```python
raw   = open("work/payload/sg00/SP0.bin","rb").read()
slots = [x for i,x in isa.decode_stream(raw)]
sy    = prof_Instruction[engine=="Sync"].drop_duplicates("compiler_pc")   # compiler_pc == slot idx
start, dur = {pc: start_ts}, {pc: duration_ns}
blocked(i) = max(0, start[i] - (start[prev_executed] + dur[prev_executed]))
isrelay(s) = s.name=="EVENT_SEMAPHORE" and s["events_extended_wait_mode"]!=0 \
             and s["events_extended_wait_idx"] >= 72
# per relay at slot i: walk j=i+1.. while (j executed) and blocked(j)<=60 and
#   not (waits(slots[j]) & updates(slots[i])); accept PSEUDO_DMATRIGGER and non-relay ES;
#   then pop trailing non-triggers; require >=1 trigger in the run -> move i to after run[-1]
order = list(range(len(slots)))
for i, last in moves:                      # apply in increasing i
    order.remove(i); order.insert(order.index(last)+1, i)
open(...,"wb").write(b"".join(raw[k*64:(k+1)*64] for k in order))
```
Then `neff.py pack`, `semaphores.py` (must be unchanged), `verify.py --runs 3`, `profile.py`.
Note the planner is **profile-guided**: it needs a `--parquet` profile of the *same* NEFF it is
editing, because `blocked()` comes from measured timestamps.

### Why it is safe (and why "move the relay later" is the right direction)
* The permutation changes no field of any slot, so **every semaphore's total increment is
  unchanged** — `semaphores.py` is byte-for-byte as clean as before.
* A relay **reads no data**; its correctness depends only on its own `S[72+ring]` wait, which
  moves with it. Delaying it can only *delay* the `S[56..71]` update, never advance it, so it
  cannot widen iter-2's WAR window — it narrows it.
* What the moved-past triggers lose is **the relay's wait as an implicit gate**. Audited
  (`scratch/audit.py`): for each moved relay, resolve its `(ring, value)` to the
  `qSyncSpillReload1` `block_id` it relays (`block_id = (value/16 - 1)*16 + ring`; the relay wait
  values really do step 16,32,…,192, so each relay waits for its own transfer), take that
  transfer's `write_offset` set, and intersect it with the `read_offset` set of every trigger moved
  ahead of it. **0 conflicts in 44 moves** — no hoisted transfer reads anything the relay gates.
* The relay is delayed by at most the execution time of the slots it jumps (~40 ns each). If one of
  those triggers now blocks on its own wait, the relay fires at ~the same wall time as before, so
  the edit is monotone, not a gamble.

### What it bought, measured
* whole-machine idle 225.3 -> 195.7 us; **the line-597 site 62.1 -> 33.2 us**.
* relay blocking on SP 130.3 -> **2.4 us** (the lever is now exhausted; re-running the planner on
  the new profile finds nothing).
* the W_t chain, median over the 64 instances:
  `W_full-transfer end -> W_t trigger` **2528 -> 1590 ns**, `W_full end -> LDWEIGHTS start`
  5533 -> 4485 ns (sum over 64: 330 -> 271 us).

---

## 2. REFUTED THIS ITERATION (each with the measurement that closed it)

### 2.1 `LDWEIGHTS transpose_mode=ENABLED` cannot replace a `dma_transpose`. **CLOSED on device.**
The prize looked big: lines 595/607 `dma_transpose` feed lines 597/609 where the transposed tile is
the **stationary** operand, so a transposing weight load would delete 128 DMA hops from the chain
(62 + 17 us of idle). Two probes, both all-64-instances, both **NaN**:

| variant | encoding | result |
|---|---|---|
| A | src = W_full+254, step -1, `num_elem=[128,1,1]`, rows 128 cols 64, `transpose_mode=1` | 262144 NaN in output_0 |
| C | same but rows 64 cols 128 | identical NaN |
| line 609, 1-variable probe | src = `qkt_m_bf16` (the pre-transpose tile), dims unchanged, `transpose_mode=1` | output_0 NaN, **output_1 PASS and bit-identical** |

The third probe is the informative one. It changes only the source address and `transpose_mode`;
output_1 (the final state) is untouched by line 609 and stayed bit-exact, so the kernel ran, and
only the `o_intra` matmul was wrong. **The cause is not `transpose_mode` — it is the partition
base.** `qkt_m_bf16` lives on partitions **64-127** while the moving operand `v_new_bf16` is on
0-63; `W_full` is on 64-127 while `state_bf16` is on 0-127.
**A PE array row is indexed by the absolute SBUF partition of the operand** (confirmed by the
failure pattern), so a stationary operand must sit on the same partitions as the moving operand's
contraction range. Every `dma_transpose` in this kernel conveniently moves the tile from partitions
64-127 to partitions 0-127, which is exactly what makes it load-able.
Facts worth keeping:
* All 1878 `LDWEIGHTS` in this NEFF have `src..._addr_reg_marker = 0`, i.e. no PE load reads above
  partition 63. The `*_addr_reg_marker` field is **just the top byte of the 4-byte immediate**
  (offset 16..19 of the slot), so partition 64+ ⇒ marker 1 automatically; nothing extra to set.
* The only 64 `transpose_mode=ENABLED` slots in the NEFF are all at line 223 =
  `nisa.nc_transpose`, fp32, `fp32_mode=LOW`, square 64x64 — so the field's meaning for a
  non-square, non-`nc_transpose` load remains **unproven**, and the partition rule has to be
  solved first for it to matter. If a future iteration wants it, the missing step is relocating
  `W_all` (written by iter-2's `op:"cast"` transfers at `to_off 0x101d200`, partitions 64-127) to
  partitions 0-63, which needs an SBUF liveness map.
* `LDWEIGHTS` pattern invariant across the whole NEFF: `prod(num_elem) == num_active_cols` and
  `num_active_rows == #partitions read`, with `step_elem[0] = -1` (the free axis is consumed in
  reverse). Useful for building any PE operand by hand.

### 2.2 Review candidate #2, "drop redundant adjacent `LDWEIGHTS`". **CLOSED, statically, 0 candidates.**
Decoded all 1878 `LDWEIGHTS` and compared each consecutive pair on
`(src immediate, step_elem, num_elem, in_dtype, fp32_mode, transpose_mode, perf_opt,
num_active_rows/cols, row_grp, col_grp)`: **0 adjacent identical pairs.** The 2272 MATMULs /
1878 LDWEIGHTS ratio already reflects the compiler's reuse. Do not look again.

### 2.3 Hoisting *triggers* past blocked triggers converts at ~0%. Measured.
Same idea as §1 but from the other side: 188 blocking SP slots have an unblocked trigger behind
them carrying **285.6 us** of block. With the only provably-safe rule — move trigger T ahead of
blocking slot B iff T's own wait is on the *same* semaphore as B's with a `>=` value, B's updates
are not in T's wait set, and (if B is a trigger) B and T are not on the same queue+ring — exactly
**18** moves qualify (36 us of block). Applied, verified, measured: **1175.0 -> 1177.4 us, i.e.
nothing** (spread 0.66%). Reverted. The triggers stuck behind a long `S[4]`/`S[5]` wait are for
transfers nobody is urgently waiting on; the relays won because they sat in front of the
*PE-critical* transposes specifically.

### 2.4 Demoting iter-2's 224 converted ACT `EVENT_SEMAPHORE`s converts at ~0%. Measured.
iter-2 turned 224 dead ACT DGE saves into ES slots that keep the save's producer wait and fake its
`S[40..55]` completion. Those slots **stall ACT for 83.3 us** (of ACT's 87.2 us of total ES stall —
so iter-2 moved cost from ACT issue onto ACT blocking). Identified them by decoding
`Activation0.bin` against `baseline/`'s (224 `PSEUDO_DMA_DIRECT2D -> EVENT_SEMAPHORE`), then moved
79 of them forward past up to 8 unblocked ACT slots each (71 us of stall addressed) with the same
safety argument as §1 (moving them *later* only narrows the WAR window). Applied, verified
bit-identical, measured **1175.0 -> 1176.6 us: nothing.** Reverted.
Lesson, consistent with iter-3 §1: **ACT stall is not on the critical path.** ACT is 44% busy and
38% of its busy time is on the path; blocking it does not cost wall time. Do not spend another
iteration on ACT.
*(Related and still untried: those 224 ES waits are arguably redundant altogether — the fused
transfer that replaced each save carries the same producer wait on SP. Removing the wait would free
83 us of ACT stall but it fires the `S[40..55]` update earlier, which **widens iter-2's known WAR
hole**. Given §2.4's 0% conversion, it is not worth the risk.)*

### 2.5 **A queued DMA descriptor cannot combine a transpose with a dtype cast.** CLOSED on device.
This was candidate #1 (fuse the `Wg_grp --cast--> W_full --transpose--> W_t` two-hop chain into
one). Probed by rewriting **one** W_t transpose's descriptor (q0 block 19, `to_off 0x5400`); each
probe is one `pack` + one `verify.py` (~25 s).

| probe | descriptor | result |
|---|---|---|
| **control: split the existing transpose 4 ways** — 4 descriptors, `transpose_shape [1,1,16,128]`, `transpose_element_size 2`, from `W_full + b*16*262144` `from_sizes [256,16,1,1]`, to `W_t + b*32` `to_sizes [32,128,1,1]` `to_steps [1,262144,33554432,32]`, bf16->bf16 | **loads, runs, output bit-identical** |
| fused: same 4 descriptors but `from_dtype float32`, `from_off = Wg_grp + b*512`, `from_sizes [512,16,1,1]`, `to_dtype bfloat16`, `transpose_element_size 2` | **`NRT_FAILURE in nrt_load_util`** |
| same with `transpose_element_size 4` | **`NRT_FAILURE in nrt_load_util`** |

Two facts to keep from the control probe (both now device-verified):
* **A transposing descriptor's tile does not have to be the compiler's 64x128**: a 16x128 tile
  works, and **one transfer's descriptor count can be changed freely** (1 -> 4) because the ring
  semaphore increments +16 per *transfer*, not per descriptor. That is the tool for reshaping or
  re-sourcing any `dma_transpose` in this kernel.
* The failure is at **load**, not execution, so the descriptor validator rejects transpose+cast
  outright. The only remaining route to fusing this chain is to make `Wg_grp` **bf16 at its
  producer** (the DVE `tensor_copy` of PSUM `yW`, line 466/467) so both hops become pure bf16
  moves — but that halves `Wg_grp`'s row stride (512 -> 256 B) and every reader's addresses move,
  i.e. a layout refactor, not a field edit. (Numerically it should be free: the coupling path at
  line 407 already casts `Wg[bj]` to bf16, and the HBM reload already rounded W to bf16.)

### 2.6 Pool has no head-of-line blocking worth taking. CLOSED, statically.
Pool's 200 `EVENT_SEMAPHORE` slots stall the engine for **53.9 us**, but **0 of them have an
unblocked `PSEUDO_DMA_DIRECT2D` behind them** — every Pool DGE DMA that follows a stalling ES was
itself blocked. The §1 lever is empty on Pool. (Pool gaps total 952 us; 639 us of that is in front
of `TENSOR_SCALAR_AFFINE_SELECT`, i.e. Pool simply has nothing to do.)

### 2.7 **The `qSPSpillReload0` "b-16" ring throttle costs nothing. CLOSED on device, twice.**
This looked like the best lever left after §1, so it is worth recording in full.
**220 of the 669 `qSPSpillReload0` triggers wait on exactly `S[24 + b%16] >= 16*(b//16)`, i.e. "the
transfer 16 blocks earlier on my own ring has completed"** — a ring-depth-1 guard, not a data
dependency. (The other waits: 155 on `S[4]`, 108 on `S[6]`, 80 on `S[5]`, 32 on block `b-1`, 9
none.) **218 of the 220 have no overlap at all** between transfer `b`'s destination cells and
`b-16`'s destination *or* source cells (expanded both to (partition, byte-range) sets from
`from/to_off/sizes/steps`), so they are pure throttles. This is the same 218 iter-2 called "stale
ring throttles".
It matters because the throttle is where the *doubling solve* waits: of the 53 stalled line-441 /
line-447 PE slots (71 us of whole-machine idle, now the biggest site), **every one waits on an
`S[24..39]` ring semaphore — a `dma_transpose`, not on ACT or DVE** (my first pass mis-read this:
`g441.py` reports the *last running* instruction, which is the DVE `TENSOR_TENSOR` of line 443,
but the actual gate is a DMA). And the trigger for that transpose is itself blocked on its own
`S[24..39]` throttle in 27 of 53 cases, worth ~51 us, the rest being genuine `S[6]` data waits
(23.3 us).

| edit | verify | total_exec_time_us |
|---|---|---|
| relax all 218 by one ring generation (`16k -> 16(k-1)`, 6 waits vanish) | PASS, bit-identical | **1174.680** (= no change) |
| remove all 218 waits entirely | PASS, bit-identical | **1178.885** (slightly *worse*) |

So the throttle is free, and removing it is mildly harmful. The explanation that fits both rows:
**the ring serves its transfers FIFO**, so issuing a trigger earlier does not make the transfer
start earlier — and once all the triggers are unthrottled, a critical transfer can queue behind
non-critical ones, i.e. the throttle is acting as a crude priority mechanism. The 51 us attributed
to the throttle is therefore real DMA service latency wearing a ring semaphore's clothes.
**And "give it a ring of its own" is dead too, measured:** for every stalled transfer at the
441/447/597 sites, `start_ts - trigger` is **699-722 ns**, and over all 669 `qSyncSpillReload0`
transfers the median is **717 ns with p90 739 and max 953** — i.e. **there is never a ring
backlog**, so more parallel rings (`num_queues` > 16, or a second transpose queue) cannot help.
The hop is pure fixed latency: 717 ns trigger->start + 292-1104 ns transfer + 384-518 ns
completion->PE start = 1.0-1.8 us, and iter-3 §2.3 already closed `fabric_path`/queue `type` as
knobs on the first term. **The only way to shorten these chains is fewer hops** (candidate #2),
not faster or earlier ones. Note line 441's transfer is the slow one (1104 ns of transfer vs
~300 ns at 447/597) — it is the whole-[GR,GR] block-diagonal transpose of line 396.

### 2.8 Hoisting relays *earlier* (to release their consumers sooner) is not there.
Relay "fire delay" = (relay update time) - (its queue-1 transfer's completion): median 954 ns,
mean 1149, min **440** (the irreducible DMA-completion -> semaphore -> SP-execute propagation),
sum 184 us. Hoisting a relay earlier is also data-safe (it reads nothing), and the right placement
is "just before the first SP slot whose profile `start_ts >= dma_end + 500`" so it never blocks.
Only **38** relays have such a position at all, worth **19.7 us** of earliness in total — below
what is measurable. Not attempted.

---

## 3. Where the remaining 1175 us is (measured on the shipped NEFF, `prof2/`)

Whole-machine idle (no non-semaphore instruction running on **any** engine): **195.7 us** in 1361
gaps; 109.6 us in the 71 gaps over 1 us. Attributed to the first work instruction after the gap:

| site (`gdn-bench/vendor/gdn/gdn_cte.py`) | gaps | us | what it is |
|---|---|---|---|
| PE `LDWEIGHTS` line 441 | 82 | **46.9** | doubling solve `nc_matmul(pMT, PT_stat, MT)` |
| PE `LDWEIGHTS` line 597 | 28 | **33.2** | `nc_matmul(WS_p, W_t, state_bf16)` (was 62.1) |
| PE `MATMUL` line 447 | 29 | **30.2** | doubling solve squaring `nc_matmul(pp, PT_stat, P)` |
| PE `LDWEIGHTS` line 609 | 55 | 16.4 | `nc_matmul(o_intra_p, qkt_m_t, v_new_bf16)` |
| ACT `TENSOR_SCALAR` 563 / 591 | 46 | 19.9 | `state_decayed`, `k_dec_full` |
| everything else | — | ~49 | nothing over 5 us |

Engine busy: DVE 535.0 (45.5%), ACT 521.2 (44.3%), PE 340.9 (29.0%), DMA 339.1 (28.8%),
Pool 233.9 (19.9%), SP 51.8 (4.3%). Still **no saturated engine**; iter-3's decomposition (795 us
of real instruction execution on the critical path) still governs.

### 3.1 The doubling solve is now the biggest idle site (77 us at lines 441+447) and it is **DMA**
Source (per `bi`, `_N_ROUNDS = ceil(log2(_SUBBLK=16)) = 4`):
`441 matmul(pMT, PT_stat, MT)` -> `443 DVE tensor_tensor(MT_new = MT + pMT)` -> next round;
in parallel `447 matmul(pp, PT_stat, P)`, `449 matmul(ppt, P, PT_stat)` -> ACT copies.
`PT_stat = Ndiag_pre[bi]` and `P = Mdiag_pre[bi]` are **pre-built by the `dma_transpose` at line
350 / the single [GR,GR] transpose at line 396**, and that is what PE waits for: of the 53 stalled
line-441/447 slots, **every one's wait is on an `S[24..39]` ring semaphore** and a transfer
completes inside the gap in 52 of 53 cases (median `trigger -> end` 1428 ns, `end -> PE start`
390 ns). Do **not** trust a "what was the last instruction running" attribution here — it names
the DVE `TENSOR_TENSOR` of line 443 / the ACT `CAST` of 456/458, which are coincidental. Also: **0
PE `EVENT_SEMAPHORE` slots sit between the gap and the stalled PE slot**, so ES overhead is not
involved. Only 82/459 and 29/223 instances stall, i.e. roughly once per group (32 groups), so it
is the group-boundary serialisation. §2.7 shows the trigger side of this is already as early as it
usefully can be.

### 3.2 Shape facts for this configuration (worth not re-deriving)
`_CHUNK=64`, `_SUBBLK=16`, `_N_SUBBLK=4`, `_CHUNKS_PER_GROUP=8`, but **S=128 ⇒ cpg=2**, so
`_GROUP_ROWS = 32` and there are **32 groups x 2 chunks = 64 chunks**, one group per batch element,
~36.7 us per group. The 32 groups are **independent** (the recurrent `state` is per batch), so the
whole remaining latency is the compiler's near-sequential schedule of 32 independent 36.7 us
chains; perfect overlap would be DVE-bound at ~535 us. That 2.2x is real but needs a reschedule of
11k instructions, not a field edit.

`W_all` ([64, cpg*D] bf16, partitions 64-127) is written by iter-2's `op:"cast"` transfers
(`from_sizes [512,4,16] fp32` on partitions 96-111 / 112-127 -> `to_sizes [256,4,16] bf16`,
`to_steps [1,4194304,262144]`, i.e. the `(row,bi,d) -> part = bi*16+row` swap) and is read **only**
by the 64 `dma_transpose(W_t, W_full)` of line 595. All 637 data-carrying `qSyncSpillReload0`
transfers are `is_transpose_mode=True` — there are **no pure SBUF->SBUF relocations** on that queue
to delete.

---

## 4. Ranked candidates for iter-5

1. **Targeted trigger hoist at the W_t site only.** 27 of the 64 W_t transpose triggers are
   head-of-line blocked behind a `PSEUDO_DMATRIGGER` waiting on `S[5]` (PE), carrying **19.1 us**
   (`scratch/wt_block.py` has the attribution; the remainder is each trigger's own wait on
   `S[8..21]`). §2.3 says the generic version converts at 0%, but these are the PE-critical ones,
   so this subset should behave like §1 (expect ~10 us).
   **I deliberately did not do it, and it needs a WAR audit before anyone does.** The W_t buffers
   are reused round-robin (`to_off 0x5400` is written by 28 different descriptors), and the gate
   being removed is an `S[5]` (PE-completion) wait — which is exactly the shape of a WAR gate
   ("the PE load that read this buffer has finished"). iter-2 found 218/260 trigger waits are stale
   ring throttles, i.e. the compiler does rely on SP program order for real gates, so you cannot
   assume the W_t trigger's own `S[8..21]` wait covers WAR. The audit: PE `S[5]` ordinal = running
   count of PE slots with `events_update_idx == 5`; find the last PE `LDWEIGHTS` reading that
   buffer before the write, take the ordinal of the `MATMUL` that follows it, and require some slot
   still before the hoist position to wait `S[5] >= that`. ~30 min. **A race here would not
   necessarily fail `verify.py`** — bit-identical output on one input set is not proof.
2. **Making `Wg_grp` bf16 at its producer**, which is what unlocks fusing the W path's two DMA hops
   (see §2.5 — transpose+cast in one descriptor is rejected at load, so the cast has to leave the
   DMA). The line-597 site still carries 33 us of idle and `W_full end -> LDWEIGHTS start` is still
   4.5 us median x 64 = 271 us, so the ceiling is the largest left; the cost is re-addressing every
   `Wg_grp` reader after the row stride halves. Use the §2.5 control-probe encoding
   (`transpose_shape [1,1,16,128]`, 4 descriptors per transfer) to build the fused transpose.
3. **Re-check `prog:`-order blocking after any future edit.** The generic recipe that found this
   iteration's win is cheap and reusable:
   `Instruction.parquet` -> per engine, `block(i) = start_ts[i] - end_ts[i-1]`; classify the
   blocking slot; ask what is immediately behind it and whether that thing was already unblocked.
   It is the only thing in four iterations that converted at >50% (59.8 us of block -> 31 us of
   wall time). Pool and ACT are already checked and empty (§2.4, §2.6); re-run it on SP after any
   edit that adds sync slots there.
4. **The 32 groups are independent** (§3.2) and the schedule is near-sequential: perfect overlap
   would be DVE-bound at ~535 us, a 2.2x. Nobody has attempted a reschedule and it is almost
   certainly out of reach of field edits, but it is where the remaining 640 us lives. If anything
   is going to beat a few percent, it is this.
5. **Do NOT spend budget on:** everything in iter-1 §5.4, iter-2 §5.5 and iter-3 §5.4, plus
   §2.1-§2.8 above. Specifically dead: `LDWEIGHTS transpose_mode` as a transpose replacement
   (partition rule), adjacent-`LDWEIGHTS` removal (0 candidates), generic trigger hoisting (0%),
   ACT ES demotion (0%), Pool ES demotion (0 candidates), relay hoisting (19.7 us, unmeasurable),
   relaxing or removing the 218 `b-16` ring throttles (measured: 0 and -0.3%),
   transpose+cast in one DMA descriptor (rejected at load), ACT/DVE instruction merging (iter-3,
   0%), PE DMA-wait relaxation (iter-3, waits are precise), queued DMA reading PSUM (iter-3,
   rejected at load), `fabric_path`/queue type, ordering mode.

## 5. Tooling notes
* `compiler_pc` in `Instruction.parquet` **is** the 64-byte slot index for every engine in this
  NEFF (checked: 4644/4644 PE opcodes match the decoded stream, 0 mismatches). No realignment
  needed, contrary to chapter 11's general warning.
* `evt_wait_time_ns` is **not** additive wall time (Sync sums to 27 ms over a 1.2 ms span — the
  engines prefetch, so many slots are "waiting" at once). Use `duration_ns` for a blocking ES and
  `start_ts - prev end_ts` for head-of-line blocking instead.
* An ES that blocks reports the wait inside `duration_ns`, not as a gap: the 224 ACT relay ES slots
  show gaps of ~50 ns but durations up to 1.7 us. Check both.
* `DmaPacketAggregated.block_id` == the transfer `id` in `SP0.json` for that queue, and
  `semaphore_id = S[24 + block_id % 16]` for `qSPSpillReload0` / `S[72 + block_id % 16]` for
  `qSPSpillReload1`. A `PSEUDO_DMATRIGGER`'s `count_union_block_id` is that same id, so
  trigger slot <-> transfer <-> profile row is a clean three-way join.
* A consumer's `S[24+r] >= 16k` means "the k-th transfer on ring r completed", so **transfers on
  the same queue+ring must keep their relative trigger order** in any SP permutation. The inverse
  is the workhorse of this iteration: a wait `(idx, value)` with `24<=idx<=39` and `value%16==0`
  names **`block_id = (value/16 - 1)*16 + (idx - 24)`**, which joins straight to
  `DmaPacketAggregated.block_id` and to `SP0.json`'s transfer `id`. Same formula with base 72 for
  `qSPSpillReload1`.
* To ask "does transfer A clobber what transfer B reads", expand a descriptor's
  `{from,to}_{off,sizes,steps}` into (partition, byte-range) cells: entry 0 of `sizes` is a **byte
  count** with step 1; for the rest, recurse `addr += i*steps[k]`; partition = `addr // 262144`,
  offset = `addr % 262144`. 20 lines, and it turned "218 waits look like throttles" into "218 waits
  provably are throttles".
* Mapping a stalled PE slot to the DMA that gated it: take the slot's wait, decode it with the
  formula above, then read that row's `trigger` / `start_ts` / `end_ts`. Do **not** infer the gate
  from "which instruction was running last before the gap" — at the doubling solve that answer
  (DVE line 443) is wrong (§3.1).
* `start_ts - trigger` on a queued transfer is a reliable backlog probe for `qSPSpillReload0`
  (flat 717 ns when the ring is free) but is **garbage for `qSPSpillReload1`** (iter-3 §2.5: it can
  precede the trigger by 77 us). Use `end_ts - trigger` there.
* `verify.py` is ~10 s and `pack` ~15 s, so a one-slot ISA probe costs under a minute. Probe
  semantics you are unsure about instead of reasoning about them (§2.1 cost 3 probes, ~4 minutes,
  and closed a whole branch).

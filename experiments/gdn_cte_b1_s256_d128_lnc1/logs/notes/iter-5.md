# iter-5 — build a real critical-path walker, then fix what it points at

## Result: **59.4 → 57.6 us (-1.65 us, -2.8%)**, outputs bit-identical

Five edits, all kept. All numbers are 7-capture medians of `tools/profile.py`.

| step | edit | median us | Δ |
|---|---|---|---|
| 0 | iter-4's NEFF, measured at session start | **58.991** | A-side |
| 1 | group-B staging DGE (sem 66): real guard `S[6]>=63` + slot swap on Pool (§3) | **58.486** | -0.51 |
| 2 | + sem-65 DGE row off the back of `ACT[91]/[93]`; S[16] fence split 3 ways (§4) | **58.424** | -0.06 |
| 3 | + sem-23 DGE row moved behind the two Sync triggers it was delaying (§5) | **58.132** | -0.29 |
| 4 | + first input DGE (sem 8) hosted on Sync instead of Pool (§6) | **57.968** | -0.16 |
| 5 | + last output store split across ACT and SP so the two rows overlap (§7) | **57.760** | -0.21 |
| 6 | + dead `S[16]` fence no-oped on Pool, store/AFFINE_SELECT swapped (§8) | **57.635** | -0.13 |

**Interleaved A/B, three rounds** (iter-4's bytes vs the shipped bytes, 7 captures each,
alternating): 59.488 / 57.722, 59.362 / 57.735, 59.161 / 57.600 → **-1.77, -1.63, -1.56 us**.
Note iter-4's own NEFF measured **58.991 at the start of the session and 59.16-59.49 three hours
later**: the machine drifts ~0.5 us over a session, so *only interleaved A/B means anything* and
the per-step deltas above (all ≤ 0.5) are individually inside the noise. The sum is not.

`verify.py --runs 3`: PASS after every kept edit, bit-identical across runs, `max_abs_err`
identical to the baseline to every digit (output_0 6.092e-05 = 11%, output_1 4.494e-04 = 83%).
`launch_rules.py` PASS. `semaphores.py` no over-wait. 6 members changed vs baseline:
`sg00/{Activation0.bin, DVE0.bin, PE0.bin, Pool0.bin, SP0.bin, SP0.json}` (SP0.json unchanged
this iteration; Activation0.bin is 121 slots, SP0.bin 67 — see §9).

---

## 1. The tool that made this iteration work: a backward critical-path walker

`scripts/cpath.py` (not preserved — **rewrite it first thing, it is 90 lines**). For every
profiled instruction compute

    binding predecessor = argmax( previous instruction's end on the same engine,
                                  time each of its waits was satisfied )

using `SemaphoreUpdate.parquet` for the satisfaction times (`id` is `"S[29] (qSyncSpillReload0)"`,
so filter on the `S[n]` prefix), then walk back from the globally last `end_ts`. Resolve a
semaphore's producer by searching all engines for (a) an instruction whose `events_update_idx`
is that semaphore and whose end is within ~300 ns, (b) a `PSEUDO_DMA_DIRECT2D` with that
`semaphore`, (c) a `PSEUDO_DMATRIGGER` whose `count_union_block_id` maps to that semaphore via
`SP0.json["dma"][i]["semaphore"]`. Print the path with the gap to the previous path node.

Why it matters: iter-1..4 worked from *per-engine idle gaps*, which mix "this engine is off the
path and idle anyway" (Pool's 17 us) with "the kernel is stopped here". The walker gives the
~130-node chain that actually sets `total_exec_time`, and the gaps on it sum to the whole of the
removable time. Every edit in this iteration came straight off that list. Two traps:

* dedupe the profile rows by `(engine, compiler_pc)` first (`Sync[9]` appears twice) or the walk
  self-loops;
* `compiler_pc` is NaN for some rows — drop them.

Also rebuilt (all small, all worth redoing): `d2.py` compact stream dump (the wait threshold is
`events_semaphore_value`, shared with the update operand; the extended wait exists **only** on
`EVENT_SEMAPHORE` = `CTRL_ES_STRUCT`, fields `events_extended_wait_{mode,idx}` +
`events_extended_sem_wait_value`), `idle.py` per-engine gaps with the blocked slot's wait,
`win.py` time-ordered dump of a window including `value==16/32/48` semaphore landings,
`scan.py` every field pointing into an SBUF window (**see §2**), `qwaits.py` every wait on a
queued-transfer semaphore with the operands and the block's `from_off`/`to_off`.

## 2. **The trap that cost me an edit: a TENSOR_SCALAR's scalar operand can be a POINTER**

`ACT[117]` waits `S[39]>=32` (queued block 31) and idles 1.28 us on it; the whole tail hangs off
it. Its `src`/`dst` (`0x37E00` / `0x4180`) have nothing to do with block 31 (`0x5600 → 0x37300`),
so I moved the fence one slot later. **output_1 went to `max_abs_err` 1.596e-01 = 29551% of the
limit** (deterministic, bit-identical across 3 runs).

The reason is in fields my address scan did not look at:

    imm0_src    = POINTER_IMMEDIATE
    imm0_imm_ptr = 0x37300          <-- exactly block 31's destination

So `ACT[117]`'s *scalar* operand is read from SBUF and it is a genuine RAW on block 31.
**Any "is this wait a real dependency?" analysis must include `imm0_imm_ptr` / `imm1_imm_ptr`
(and `imm0_src`/`imm1_src` to know whether the pointer form is in use).** iter-4 §10.7's whole
candidate list was built without them, which is why it looked more promising than it is. With
them included (`qwaits.py`), **every remaining wait on a queued-transfer semaphore in this NEFF
resolves to a real RAW or WAR** — see §10.

A second reader class the scan misses: an instruction whose start address is *below* the window
but whose extent reaches into it (`num_active_channels` gives the partition count, the
`num_elem[0]` the elements per partition — e.g. `ACT[117]` writes `0x4180..0x427F`, which is how
`PE[164]`'s reverse `LDWEIGHTS` from `0x427E` is fed).

## 3. Edit 1 — the staging DGEs' `S[4]>=42/43` guards are leftovers; the real guard is a DVE signal

The critical path left the var11/var12 staging window through **`S[66]`** (iter-4 §11 predicted
this): `S[66]@54.11 → DVE[98] → DVE[99] → ... → PE[136]`. Chain of facts:

* every one of the 8 staging DGEs reads 4 chunks of 512 B at `0x6480/0x6680/0x6880/0x6A80` from
  16 partitions. Those chunks are written by **8 DVE ops**: `DVE[44,60,78,90]` cover p0-63
  (`num_active_channels=64`) and `DVE[43,61,77,91]` cover p64-127. So **group A depends only on
  `DVE[90]` and group B only on `DVE[91]`**, which end at 49.01 and 49.33 us.
* `ACT[82]`, the one staging DGE whose wait the compiler itself wrote, waits **`S[6]>=62`** — the
  DVE signal immediately after `DVE[90]`. By symmetry group B's correct guard is **`S[6]>=63`**
  (the signal after `DVE[91]`; `DVE[91]` itself does not signal, `DVE[92]` does).
* the `S[4]>=42/43` waits on the other seven are iter-4 §3a's conservative rewrite of what used
  to be ACT program order. `S[4]=42/43` are `ACT[85]`/`ACT[87]`, which write `0x4180`/`0x805600`
  and have nothing to do with the staging sources. They are met at 51.16/51.33 — **2 us late**.
* destination safety: `scan.py` over `0x101CC00..0x101D800` confirms each block still has exactly
  one writer and one reader, so issuing earlier creates no WAR.

**Edit:** swap Pool slots 21 and 22 (so the sem-66 payload issues first — program order on the
host, not just the guard, was holding it) and give the sem-66 slot `S[6]>=63`.

Measured: the row moved 51.85 → 49.76, **`S[66]` 54.11 → 51.69 (-2.43 us)**, `DVE[99]`
-1.43 us... and the metric only -0.51, because the next gate (`DVE[96] ← S[4]>=44`) surfaced
immediately. That is the shape of every edit in this kernel: **a 2.4 us gain on one semaphore
buys 0.5 us of wall clock.**

The other three group-B guards (`sem 67/23/50`) and the three group-A ones can be relaxed the
same way; none of them is on the path today.

## 4. Edit 2 — a DGE row costs the compute slots behind it ~650 ns

New gate after edit 1: `ACT[91] → ACT[93] → S[4]=45 → SP[41]` (trigger of queued block 21).
`ACT[87]` ends at 51.12 but `ACT[91]` only starts at 51.76, because **`ACT[88]`, the sem-65
staging DGE, occupies Activation for 640 ns** and two no-op `EVENT_SEMAPHORE`s follow it. Sem 65's
block is read only by queued block 26, whose trigger fires 5 us later.

Rotated ACT slots 88..93 left by one so the DGE row issues *after* `ACT[93]`. No renumbering was
needed: of those six slots only `ACT[91]` and `ACT[93]` update `S[4]` and their order is
unchanged — **check this before any within-engine rotation; it is the whole safety argument.**

Also split the 4-to-1 `S[16]` fence that iter-4 left: `SP[42]`'s extended wait `S[16]>=1` →
`S[44]>=16` (block 22 reads only sem 44's block), and the other two readers got their own
producer on the free extended wait of the slot in front of their trigger (`SP[50]` ← `S[65]`,
`SP[58]` ← `S[22]`). Both fire 5-7 us after their semaphore lands, so the waits are free, and
they make the later sem-65 row harmless.

Measured: `ACT[91]` -660 ns, `ACT[93]` -617, `DVE[96]` -623 — and the metric -0.06. The gain was
eaten because `SP[41]` turned out to be gated by `SP[40]`'s extended wait `S[6]>=66` and by Sync
program order, not by `S[4]>=45`. Which led to:

## 5. Edit 3 — same lesson on Sync, and this one paid

`SP[41]` (block-21 trigger) fired at 53.44 although its gates were met at 52.07/52.34. In front
of it on Sync: `SP[34]`, the **sem-23 staging DGE row (641 ns)**, then `SP[35..40]`. Sem 23's
block is read by `DVE[120]` at ~61.6 us — 7 us of slack.

Rotated SP slots 34..41 left by one (DGE row to slot 41, behind both triggers). Only `SP[39]`
updates anything there, and it is `S[64]`, which has had no waiter since iter-4.

Measured: trigger **-951 ns**, `DVE[103]` -481 (the transfer's own latency grew 1.56 → 2.06 us
because it now runs inside the DGE flood), last instruction -551, metric **-0.29**.

## 6. Edit 4 — the first input load, and what the startup gap really is

The first node of the critical path is `DVE[8] CAST ← S[8]>=16`, idle **2.3 us**: `Pool[1]` is a
DRAM→SBUF input load (512 B, but a 256-element 2-byte scatter into 64 partitions) and every
engine waits for it. Pool's sequencer also stalls ~625 ns after `SET_ORDERING_MODE` before that
row starts, where Vector's stall is 110 ns and Scalar's 22 ns.

Moved the DGE to **SP slot 1** (inserted; Sync is idle until 16.8 us, and a DGE's queue follows
its semaphore, so the transfer is unchanged). The row now starts 83 ns after Sync's
`SET_ORDERING_MODE` instead of 625 ns after Pool's. Metric -0.16.

**But the 2.3 us itself did not shrink** — `S[8]` landed at the same offset from the stream start.
Supporting data: 64-element load → 2.25 us issue→semaphore; 256-element 512 B load → 2.3-3.0;
256-element 64 KB load → 2.4. So **a DRAM→SBUF DGE is ~2 us of fixed latency plus a small
per-element term**; splitting it or re-patterning it will not help. Treat the startup 2.3 us as
irreducible unless someone finds a way to start the DMA before the engines are up.

## 7. Edit 5 — splitting the last output store (and the hang that taught the host rule)

Tail: `... → ACT[118] → SP[65]` row (sem 20, 64 KB) `→ SP[66]` row (sem 21, 16 KB, 896 ns, and it
waits 246 ns for SP[65]'s row) → `S[21]` 1.37 us later → `Pool[35]` end barrier. The last 2.4 us
is one DGE row plus one DMA latency with nothing overlapping it.

Split the sem-21 store into two 32-partition halves (`src += p*0x40000`, `dst += p*256`,
`src/dst_num_elem [64,1] → [32,1]`), both keeping `semaphore 21, sem_increment 16`, and changed
the end barrier `Pool[35]` to wait **`S[21]>=32`** (its extended wait on `S[20]>=16` is a separate
field, so both fit). Half A appended to the **Activation** stream (idle from 67.98), half B left
in `SP[66]`.

Measured: ACT row 67.27-67.92 and SP row 67.49-68.42 overlap, `S[21]>=32` lands 0.5 us earlier
than `S[21]>=16` used to, kernel end -1.1 us in that profile's own timebase, metric **-0.21**.
Note the SP half still took **934 ns for half the data** (the whole thing took 896): **a DGE row's
cost on the host engine is not proportional to its payload.**

### 7.1 REFUTED, with a device hang: a DGE cannot be hosted on DVE or PE

The first attempt put half A after `DVE[132]` and half B after `PE[167]`. Result: `nrt_execute`
status 5, then `Model stop timeout (30000 ms) for TPBs`. The device recovered on its own (the next
`verify.py` passed; no `rmmod`/`modprobe` needed), total cost ~4 minutes.

The rule is in `def.json`'s `dma_queue` block: the four queues are owned by **act, pool and sp
only**. DVE and PE have no DGE path at all. iter-1..4 only ever moved DGEs among those three, so
this never came up. Combined with iter-4 §1 (*the queue follows the semaphore, not the host*):
**host ∈ {act, pool, sp}, queue = f(semaphore), and the two are independent.**

## 8. Edit 6 — the dead `S[16]` fence was stalling Pool for 1.7 us

After edit 2b nothing waits on `S[16]` (census: `Pool[26]` is its only updater, no waiter on any
engine, primary or extended, and no DGE uses sem 16). But `Pool[24]` still *stalled Pool for
1.70 us* waiting for `S[65]`, so Pool did not reach slot 28 until 54.58 us. `Pool[28]` is an
output store whose semaphore `S[17]` is the **WAR guard of queued block 24** (`0x37900 → 0x4600`,
the transpose `PE[148]` loads from `0x467E`), and `PE[148]` idles 1.50 us on it.

No-oped `Pool[24,25,26]` and swapped `Pool[27]` (AFFINE_SELECT writing p64+`0x4100`, disjoint)
with `Pool[28]`. Measured: `SP[48]` -550 ns, `PE[148]` -380, `PE[151]` -376 — and the **tail only
-45 ns**, metric -0.13 (inside the noise). Kept because it is provably safe and leaves a much
cleaner Pool schedule, but **the `PE[148]`/`PE[151]` branch is not what ends the kernel**: do not
spend more on it without re-running the walker.

## 9. Verified mechanics (new this iteration)

* **Inserting or appending slots to a stream is safe.** Shipped three of them this iteration
  (`SP0.bin` 66→67 slots, `Activation0.bin` 120→121); against the baseline the shipped NEFF now
  has `Pool0.bin` 32→37 slots and `SP0.bin` 62→67, so iterations 1-4 had already grown streams
  without noticing it was a lever. Nothing records an instruction count:
  `def.json` only names the `.bin`, the engine `.json` files carry `instr`/`name`/`dma`, and
  `act_info.json` / `exp_and_others.json` are keyed by **activation function name/id, not slot
  index**. `neff.py pack` recomputes sizes and both MD5s. (The one insertion that broke output was
  §2's, and the cause was a real data dependency, not the insertion.)
* A pure no-op slot is `a010` + 62 zero bytes (`EVENT_SEMAPHORE`, no wait, no update) — copy it
  from `Pool[33]`.
* `isa.Instruction["field"]` returns the **raw int**; enum names only come from
  `.as_dict()`. Assertions like `ins["events_wait_mode"] == "NONE"` silently fail.
* `ins.set("events_wait_mode", "NONE")` plus `set("events_semaphore_value", 0)` is how you clear
  a wait while keeping a `SEM_INC_COMPLETE` update (the value field is shared and `INC` ignores it).
* Writing `dst_start_addr_addr_immediate` (8 bytes) also fixes the `_var_offset` /
  `_table_offset` / `_tbl_offs_*` aliases, which overlay the same bytes — `neff_diff.py` lists
  them all and that is expected, not a bug.
* DRAM addresses in a DGE encode as `0x2000000V0000OOOO` (V = var id, low 32 bits = offset);
  SBUF as `0x1000000000000000 | (partition * 0x40000 + offset)`.
* Cost of a device round trip this session: `verify.py --runs 3` ≈ 2.5 min, `profile.py
  --repeats 7` ≈ 2 min. 10 verifies + 12 profiles, one hang, no driver reload.

## 10. Where the time goes now (shipped NEFF, `prof7`) — the list for iter-6

Critical path, gaps > 250 ns, `cpath.py`. **The sum of these gaps is ~15 us of a 57.6 us
kernel; everything else on the path is back-to-back compute.**

| gap ns | path node | waits for | status |
|---|---|---|---|
| 2357 | `DVE[8] CAST` | `S[8]>=16` | the first input load; **fixed ~2 us DMA latency, closed (§6)** |
| 3166 | `SP[14]` trigger | `S[29]>=16` (block 5, 32 KB) | iter-3 closed every lever (merge = race, split = hang) |
| 1709 | `PE[23] MATMUL` | `S[30]>=16` (block 6) | the second hop of the same chain |
| 1497 | `SP[48]` trigger | `S[17]>=16` | real WAR (§8); already pulled 0.55 us earlier |
| 1501 | `PE[148] LDWEIGHTS` | `S[32]>=32` (block 24 → `0x4600`) | real RAW (`src 0x467E`), and §8 showed its branch is not what ends the kernel |
| 1429 | `PE[151] LDWEIGHTS` | `S[34]>=32` (block 26 → `0x5500`) | real RAW; iter-4 §10.6 closed the transpose route |
| 1278 | `ACT[117]` | `S[39]>=32` (block 31 → `0x37300`) | **real RAW via `imm0_imm_ptr` (§2) — closed** |
| 1252 | `Pool[35]` end barrier | `S[21]>=32` | the output store's DMA latency; §7 took the overlap, ~1.25 us left is fabric |
| 380/485 | `Pool[24]`, `Pool[28]` | `S[65]`, prog | §8 |

Ranked candidates for iter-6:

1. **Re-run `cpath.py` first.** Four of the eight edits I tried moved their own landmark by
   0.4-2.4 us and the metric by ≤0.2 because a different node bound immediately afterwards. The
   path reorders after every edit; working from a stale path wastes a device round trip.
2. **`DVE[103]`'s fence `S[29]>=32` (block 21), 670 ns idle, UNRESOLVED — the one open lever.**
   Block 21 writes `0x5A00..0x5A0F`; the only later reader I can find is `ACT[100]`'s
   `imm0_imm_ptr = 0x5A00`, which is on *another* engine and has no wait of its own, and
   `DVE[109]` (reads `0x5A08`, 2.5 us of slack). I could not work out what the DVE-side fence
   protects, so I did not move it: an unexplained fence is exactly the shape of iter-3's race.
   Settle it by dumping the bytes (repoint a tail output DGE at `0x5A00`), not by guessing.
3. **The three remaining over-conservative staging guards** (§3): `sem 67/23/50` still wait
   `S[4]>=43`, real guard `S[6]>=63`; `sem 44/65/22` wait `S[4]>=42`, real guard `S[6]>=62`.
   Worth nothing today, but if a future edit puts one of them on the path the fix is one field.
4. **Second half of §7**: `SP[65]` (sem 20, 64 KB) is still a single 644 ns row + 1.99 us latency
   and `S[20]` lands at 68.77 vs `S[21]>=32` at 69.65 — ~0.9 us of headroom, so splitting it too
   buys nothing *unless* `S[21]` gets faster first. Check the order before touching it.
5. **The 9 (now 6) no-op slots** iter-4 listed: `ACT[83,84,86]` remain (89/90 moved in §4),
   `SP[14,15]`, `Pool[33,34]`, plus the three new no-ops from §8 and `SP[66]`'s old slot. ~40 ns
   each of sequencer time, all off-path. Only worth deleting as a side effect of something else.
6. **Do not** retry: iter-1 §7, iter-2 §10.4, iter-3 §9.3, iter-4 §5/§10.6/§11.1, and from this
   iteration: `ACT[117]`'s fence (§2), DGEs on DVE/PE (§7.1), re-patterning the startup load (§6).

## 11. The honest model of this kernel after five iterations

67.5 → 57.6 us (-14.7%). What is left is **one fixed DMA latency per hop and ~130 nodes of
dependent compute**. A hop costs ~1.2-1.4 us (DGE row-end → semaphore) or ~1.2-2.1 us (queued
trigger → completion), none of it reducible, and the critical path crosses ~9 of them. Every win
in this iteration came from the same two moves — **issue a transfer earlier (fix an
over-conservative guard, or get it out from behind another engine's row), or overlap two of
them** — and each returned ~25-35% of the semaphore-level gain to the wall clock, because the
next node binds. There is no large structural lever left that I can see: the next 1 us will be
five more 200 ns edits, and each one needs the walker re-run to find.

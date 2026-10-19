# iter-1 — critical-path walk, then fused 32 DRAM staging round-trips into direct SBUF→SBUF transfers

## Result: **1312.2 / 1313.4 us, −2.9% / −2.8%** vs baseline 1351.4 (my own re-measure of the
baseline NEFF: 1350.6, band 1350.0–1352.5)

| NEFF | total_exec_time_us (5 repeats) | captures |
|---|---|---|
| baseline (re-measured) | **1350.597** | 1350.0 1350.4 1350.6 1351.2 1352.5 |
| shipped edit, run 1 | **1312.166** | 1310.2 1311.9 1312.2 1314.4 1314.8 |
| shipped edit, run 2 | **1313.373** | 1309.3 1310.7 1313.4 1314.7 1316.6 |

Ten captures of the edit (1309.3–1316.6) against ten of the baseline (1347.1–1352.5): the bands do
not touch, so the ~38 us is real (noise on this kernel is ±2–3 us, 0.18–0.56% spread).
`verify.py --runs 3`: PASS, all three runs **bit-identical**, `max_abs_err` **identical to the
baseline to every printed digit** (output_0 1.097e-04 = 10% of limit, output_1 9.081e-04 = 83%).
`launch_rules.py`: PASS. `semaphores.py`: no over-wait anywhere (details in §3).
`neff_diff`: 2 of 34 members changed — `sg00/SP0.json` (32 transfers' descriptors) and
`sg00/SP0.bin` (34 slots: 32 trigger waits + 2 follower waits). Nothing else moved.

---

## 1. THE MOST USEFUL THING HERE: a critical-path walk that covers the whole span

Engine-busy tables are useless on this kernel (nothing above 48%). What found the win was a
backward walk over the real dependence graph, which **accounts for 100.0% of the 1351 us span**.
Rebuild it first thing next iteration (≈40 lines, 1 min to run); the ranked table below is what
it prints.

### Recipe (all verified on this NEFF)

1. `neff.py unpack kernel.neff work/`; `profile.py --repeats 3 --parquet prof`.
2. Decode each stream (`isa.decode_stream`), keep per slot: `events_wait_{mode,idx}`,
   `events_semaphore_value`, `events_extended_wait_*`, `events_update_{mode,idx}`,
   `semaphore` (DGE), `dma_queue_name` (trigger).
3. Slot index == `Instruction.parquet.compiler_pc`. Non-null `compiler_pc` counts match the
   stream lengths exactly (Scalar 2532, Vector 2860, Tensor 4644, GpSimd 623, Sync 1315; **Sync
   has 32 duplicate rows — keep the first**). `engine` names: Scalar=ACT, Vector=DVE,
   Tensor=PE, GpSimd=Pool, Sync=SP.
4. `SemaphoreUpdate.parquet`: `id` is sometimes `3` and sometimes `"S[24] (qSyncSpillReload0)"` —
   parse both. Build per-semaphore `t_reach(sid, v)` = first timestamp whose value ≥ v
   (first-reach, **not** a running max: `sem_wr_imm` rows reset to 0).
5. Producer of the v-th increment (all four rules verified by counting on this NEFF):
   * S[3]/S[4]/S[5]/S[6] = Pool/ACT/PE/DVE: the v-th slot of that engine with
     `events_update_mode != NONE` (counts match exactly: 131 / 1306 / 1982 / 2093).
   * DGE sems: S[8..23] `qPoolDynamic`, S[40..55] `qActDynamicHW`, S[56..71] `qSPDynamicHW`.
     Each DGE slot adds **+16** to its own `semaphore`, so the n-th DGE slot naming S completes
     at `t_reach(S, 16n)` (20 slots × 16 = 320 per Pool sem, 14 × 16 = 224 per ACT sem,
     10 × 16 = 160 per SP sem — all exact).
   * Queued `qSPSpillReload0` = S[24..39]: transfer id k signals `S[24 + k%16]` by +16, so
     wait value v on sem s belongs to transfer id `((v+15)//16 - 1)*16 + (s-24)`; transfer id ==
     `PSEUDO_DMATRIGGER` program order on SP == `DmaPacketAggregated.block_id`.
6. For each slot, candidate predecessors = (its own engine's previous slot, time = that slot's
   **start**) ∪ (t_reach of each wait). Binding = the latest. Walk back from the last-ending
   slot; when the binding one is a DMA semaphore, insert a virtual "transfer" node spanning
   [issuing slot's end → t_reach] and continue from the issuing slot.

### What it said about the baseline (1351.1 us walked, 4340 nodes)

| gate | us | n |
|---|---|---|
| DMA transfer latency | **419.7** | 260 |
| same-engine program order | 407.1 | 2029 |
| wait on DVE (S[6]) | 209.6 | 813 |
| wait on PE (S[5]) | 184.4 | 643 |
| wait on ACT (S[4]) | 80.6 | 273 |
| wait on Pool (S[3]) | 13.8 | 61 |

Split into exec vs stall (stall = only 186 us of 1351, so the path is dense, not stalling):

| engine | opcode | n | exec us |
|---|---|---|---|
| SP queued | QUEUED_XFER | 155 | 255.7 |
| DVE | TENSOR_TENSOR | 713 | 145.9 |
| PE | MATMUL | 776 | 106.0 |
| ACT dge | DGE_XFER | 64 | 97.6 |
| DVE | CAST | 417 | 78.1 |
| ACT | TENSOR_SCALAR | 226 | 75.4 |
| ACT | ACTIVATE | 240 | 72.7 |
| ACT | DMA_DIRECT2D (issue) | 130 | 70.8 |
| Pool dge | DGE_XFER | 39 | 62.2 |
| PE | LDWEIGHTS | 708 | 60.8 |
| DVE | COPY | 163 | 43.6 |
| Pool | DMA_DIRECT2D (issue) | 40 | 26.1 |

**31% of the path was DMA transfer latency, ~1.6 us per hop, and 128 of the 260 hops had ≤2
instructions between them** — i.e. the path threads through chains of back-to-back DMAs with no
compute in between. That is the thing to attack on this kernel, not engine occupancy.

### Per-hop cost of a DMA (from `DmaPacketAggregated`, 669 queued transfers)

`start_ts - trigger` = **730 ns ± 55** (min 695, max 1200) — a flat queue-start cost, independent
of size (256 B … 16 KB) and of ring load. Actual `duration_ns` = 379 ns mean. The waiter sees
≈1.65 us (the rest is semaphore propagation). DGE hops measured the same way:
slot-end → completion = **1500 ns** (ACT, n=224), **1714** (Pool, 320), **1683** (SP, 160);
DGE slot issue cost = 581 / 698 / 595 ns. So Pool is the worst DGE host here too, but only by
~120 ns of issue and ~210 ns of latency — worth ≤13 us on the path, not the 1.9% the sibling
`s2304` run got from host rebalancing.

---

## 2. What I shipped: 32 DRAM staging round-trips removed from the critical path

### The structure (read this before any further DMA work on this kernel)

`def.json` has 160 `tmp-buf` DRAM variables. They are **not spills** (same finding as the sibling
`gdn_cte_b32_s2304` run): a save's SBUF source and the matching reload's SBUF destination are
different regions, so DRAM is only a staging stop for a partition/offset redistribution. Four
shapes, by who touches each var:

| n vars | touchers | what |
|---|---|---|
| 64 | 1 ACT DGE write + 2 **SP DGE** reads | DGE-only staging (see §4.1 — closed) |
| 32 | 1 ACT DGE write + 2 ACT DGE writes… (2 ACT writes + 1 **SP DGE** read) | |
| 32 | 2 ACT DGE writes + 1 **Pool DGE** read | biggest remaining path item (§5.1) |
| 32 | 1 ACT DGE write only | **the one I fused**: the reader is a *queued* transfer |

The last group is the fusable one: for each of those 32 vars the chain on the critical path was

```
DVE produces tile  --S[6]>=v-->  ACT DGE save slot (581 ns issue)
     --save transfer SBUF→DRAM, 1.5 us-->  SP PSEUDO_DMATRIGGER (waits S[40+k]>=16)
     --queued transfer DRAM→SBUF, 2.0 us-->  consumer
```

5.7 us of latency to move one 16 KB tile from SBUF to SBUF. All 32 were on the critical path
(65.3 us of it as reload transfers, plus the saves).

### The edit (two parts; both are needed, each alone is worth 0)

**(a) Point the queued reload at the SBUF source instead of the DRAM var** — `SP0.json` only.
The save is always `src=SB off=So step=[S,1] num=[N,1] es=E` → `dst=VAR off=0 step=[E,1]
num=[N,1]`, i.e. DRAM byte `L = i*E + c` is SBUF byte `So + i*S + c`. The reload descriptor's
`from_sizes/from_steps` are **in bytes** for `op:"copy"` (8192 bytes per descriptor here,
2 descriptors per transfer = the 16 KB var). Map each descriptor dim with step `s`:
`s % E == 0` → new step `(s//E)*S`; else require it stays inside one E-byte block → keep `s`.
New `from_off = So + (F//E)*S + (F%E)`, `from = "SB"`. For all 32: sizes `[32,4,4,16]`,
steps `[1,32,4096,256]` → `[1,32,4194304,262144]`, E=256, S=262144, N=64.
Keep `num_tiling_dimensions`, `from_dtype`, and the whole `to` side untouched.
`json.dumps(J, separators=(",",":"))` reproduces `SP0.json` **byte-identically**, so the diff
stays minimal.

**(b) Retarget the trigger's wait from the save's completion to the save's own wait** —
`SP0.bin`, 2 fields per trigger (`events_wait_idx`, `events_semaphore_value`).
Without (b) the reload still waits for the dead save's transfer: measured **1354.955 us, +0.3%,
i.e. nothing** (descriptor-only variant, verified PASS). With (b) the chain becomes
`DVE --S[6]>=v--> trigger --2.0 us--> consumer`, dropping the 581 ns issue + 1.5 us save transfer
per instance.

*Why the save's own wait is the right condition and is safe:*
the DGE save's wait is the compiler's own statement that its source is ready; measured in the
baseline profile, **every one of the 32 saves started 20–21 ns after its wait was satisfied**
(one at 454 ns), so no ACT instruction ran between the wait and the save — the waited-for DVE
instruction is the producer. It also **cannot deadlock**: any condition that was satisfied before
the save executed is satisfied before the trigger ran in the baseline.

*Keep the SP stream's implied ordering:* a wait you remove from slot s was implicitly also the
constraint of every waitless slot after s. Only 2 of the 32 triggers had a waitless follower
(SP slots 45 and 90, both `PSEUDO_DMA_DIRECT2D`); both got the trigger's old wait installed
explicitly (`S[49]>=16`, `S[55]>=16`). Check this for any wait you relax.

### HANG: the obvious retarget deadlocks — do not repeat it

My first attempt pointed the trigger at *the first signalling ACT instruction at or after the
save* (`S[4]>=k`, the condition that also covers ACT program order). It **hung** (NRT_TIMEOUT,
status 5, 30 s model-stop timeout) even though `semaphores.py` showed no over-wait. Cause: those
later ACT slots wait on things SP produces *after* the trigger — e.g. for the first pair, ACT
slots 24/25 wait `S[57]>=16`/`S[56]>=16`, which are SP DGE transfers at SP slots ≥ 6, i.e. behind
the trigger in SP's in-order stream. Cross-engine cycle. **Rule: a retargeted wait must name a
producer that was already satisfied before the instruction you are relaxing ran in the baseline.**
`semaphores.py` cannot see this; it only checks supply ≥ demand.

---

## 3. Safety of the shipped edit

* `semaphores.py` diff vs baseline: identical except (i) S[6] waiters 2220 → 2252 (+32, the
  retargeted triggers), (ii) S[40..55] waiter counts down by 1–3 each, (iii) `max_wait` fell
  224 → 208 on S[42],S[43],S[44],S[49],S[50],S[51] — the last transfer of those ACT DGE groups now
  has no waiter. No `max_wait > total_inc` anywhere, so nothing can hang.
* No slot was added or deleted; no semaphore total changed; no arithmetic instruction touched.
* Residual risk (small, stated honestly): the reload now reads the SBUF source **earlier** than
  before — in parallel with the save's own read instead of after it. Anything that overwrites that
  SBUF region must be ordered after the save's transfer completion (`S[40+k]>=16`), which is
  *later* than the new read start, so the edit is strictly safer than the descriptor-only variant,
  but the read's tail can still overlap the moment such an overwriter is released. I did not build
  a full SBUF access map to rule it out; 3 runs are bit-identical and the error figures match the
  baseline exactly. If a future iteration builds the access map (sibling iter-3's method), confirm
  these 32 regions (SBUF partitions 64–127, one 256-byte column each, `So` in `scratch/plan.json`
  terms = save `src_start_addr_addr_var_offset`).

---

## 4. Refuted this iteration (with the number that closed it)

### 4.1 Fusing the DGE-only staging (64 vars, 2 SP DGE reloads each) — CLOSED, wrong data
Same idea as the shipped edit but the reader is a DGE instruction, not a queued descriptor.
62 of 64 composed cleanly on paper (2 skipped: the save has no wait at all), needing
`src_step_elem = [4194304, 262144]` (16 partitions, 1 partition) instead of `[16384, 1024]`.
* With address + wait both changed: `max_abs_err` 1.155e-02 (1050% of limit), bit-identical over
  3 runs ⇒ deterministic mis-mapping, not a race.
* With **address only** (waits untouched, so correctness depends on nothing but the mapping):
  `max_abs_err` 1.782e+00. So the address mapping itself is wrong.
* Why: **a DGE pattern appears to allow only one partition-crossing dimension on the SBUF side.**
  All 704 SB-side DGE patterns in this NEFF are `part_dims=1, byte_dims=1` (384) or
  `part_dims=1, byte_dims=0` (320); **none** has two dims whose step is a multiple of 262144.
  The queued path has no such limit — the shipped edit's `[1,32,4194304,262144]` works.
* And it cannot be re-expressed: the reload's `src`/`dst` dims are paired positionally, the dst
  (`step=[512,262144] num=[4,16]`) is fixed, so the source must be
  `So + 4194304*i0 + 262144*i1` — two partition dims by construction. Setting the source to one
  partition dim (`num=[64,1] step=[262144,1]`) pairs block *j* with the wrong destination element.
  **Closed unless the reload is converted to a queued transfer** (needs a new ring id → renumbering
  all later `S[24..39]` waits).
* Group "2 ACT saves + 1 Pool reload" (§5.1) is worse: the composite needs 3 dims
  (`512a + 262144b + 4194304k`) plus a 2:1 fp32→bf16 element size, and no two dims merge on both
  sides at once. Same conclusion.

### 4.2 Merging affine runs of identical instructions (the sibling's −1.13% lever) — CLOSED, empty
On `s2304` this deleted 993 slots. Here the whole program is 11974 slots and the maximal runs of
consecutive same-key slots (all fields equal except the two base addresses, follower
non-signalling and waitless) are: **ACT 5, DVE 26, Pool 0, PE 26, SP 0 — 57 removable slots,
~10 us of issue at best**. Not worth a device cycle. (Key must ignore every `*_addr_reg_*` field
and `header_debug*`; `header_inst_word_len == 16`.)

### 4.3 Dedup repeated PE `LDWEIGHTS` (review candidate D) — CLOSED, 0 candidates
1878 `LDWEIGHTS`, 187 distinct full-field keys, one repeated 111×, but **0 of the 1877 consecutive
pairs in the LDWEIGHTS subsequence are identical** (they round-robin over different tiles; same-key
loads are 40+ slots apart). Dedup would require reordering. Also noted: `LDWEIGHTS` never signals
(all 1878 `events_update_mode = NONE`) and 425 have no wait, so deleting one is semaphore-free —
but there is nothing to delete.

### 4.4 Shrinking the 32 now-dead DGE saves does **not** make their slot cheaper — CLOSED, 0 us
After the shipped edit, no descriptor anywhere reads those 32 `tmp-buf` vars, so the saves are
**provably dead** (static, not timing-based: I re-scanned every `SP0.json` descriptor and every
DGE slot). I set `src_num_elem`/`dst_num_elem` from `[64,1]` to `[1,1]` on all 32 (semaphore,
`sem_increment` and `elem_size` untouched).
* **No hang** — this refutes the sibling's "shrinking one hangs" for this case. The DGE's +16 is
  per *slot*, not per block: `semaphores.py` unchanged, `verify.py --runs 2` bit-identical, errors
  identical. 1/32 probe first, then all 32.
* **But no gain**: 1313.709 us (band 1312.1–1317.1) vs 1312.2/1313.4 for the shipped NEFF, and ACT
  busy went *up* 626.5 → 629.0 us. So the ~515–610 ns an ACT DGE slot costs is **fixed per slot**
  and not a function of block count or bytes (measured: 16 KB/64 blocks → 515 ns, 32 KB/64 → 611,
  64 KB/64 → 552). Reverted. The only route to that 55 us of on-path DGE issue is to remove or
  replace the slot (§5.2), not to shrink it.

### 4.5 Moving the retargeted triggers earlier in the SP stream — not worth it, ≤12 us
After the edit, each retargeted trigger starts a mean of **462 ns** after its wait is satisfied
(24 of 32 are at the ~95–100 ns dispatch floor; four outliers of 0.7–2.1 us are behind an SP
`EVENT_SEMAPHORE` that was still running). Total slack 14.8 us, recoverable ≈11 us at best, i.e.
at the noise floor, for a reorder that risks silent races. Skipped.

### 4.6 Engine occupancy is not the limit (confirms the sibling)
ACT 46.4%, DVE 39.5%, DMA 31.1%, PE 25.2%, Pool 18.9%, SP 9.7%. All-engine idle is 215 us of
1365 in 1574 gaps, largest 2.7 us. `evt_wait_time_ns` in `Instruction.parquet` is **not usable**
as a wait measure (Sync `WRITE` sums to 13361 us over a 1365 us span).

### 4.7 Queue contention is not the limit
`qSPSpillReload0` is `num_queues: 16` (16 rings, 42 transfers each); the 730 ns queue-start cost
is flat across rings and sizes, so splitting or re-balancing rings has nothing to recover.

---

## 5. What I would do next, in order

**5.1 The 32 "2 ACT saves + 1 Pool reload" vars = 99.7 us still on the path.**
After my edit the remaining on-path DGE transfers are: 32 ACT saves of this group (49.9 us) +
32 Pool reloads of the same group (49.8 us) back-to-back, 14 group-1 ACT saves (20.6 us), 8
non-tmpbuf Pool DGE (12.1 us), 2 group-1 SP reloads (3.9 us). Removing a hop here is worth
~50 us (3.8%). A 2-D DGE cannot express the composite (§4.1), so the route is to **move the Pool
reload onto the queued path**: build a 4-D `SP0.json` descriptor (which *can* do it —
`512a + 262144b + 4194304k`, fp32→bf16 is just `from_dtype/to_dtype`), append it as a new
transfer id at the end of `qSPSpillReload0` (id 669 → sem `S[24 + 669%16]`, appending leaves every
existing wait untouched — ch.10 R5), add a `PSEUDO_DMATRIGGER` on SP waiting the *saves'* own
wait, and retire the Pool DGE reload (replace with an `EVENT_SEMAPHORE` repaying its +16 via
`SEM_ADD_IMM_COMPLETE`, the sibling's device-verified iter-5 trick). Biggest remaining prize;
budget a whole iteration. Inserting an SP slot is cheap: **no SP instruction increments any
semaphore** (`Sync updates: {}`), so inserting into SP0.bin shifts no count — only trigger ids
(program order) and SP DGE ordinals, so insert *after* the last trigger/DGE you care about or
re-derive both.

**5.2 Replacing the 32 dead ACT DGE saves — measured bound: ~1 us. do not bother.**
I nearly recommended this (replace each dead save with an `EVENT_SEMAPHORE` repaying +16 via
`SEM_ADD_IMM_COMPLETE`, ~485 ns of ACT issue each), then checked it against the post-edit path:
**only 2 of the 32 dead saves are still on the critical path** (0.7 us of the 55.5 us of on-path
ACT DGE issue; the other 94 on-path DGE slots are the *live* saves of the other staging groups).
That is also why §4.4's shrink measured 0. The 55 us of on-path ACT DGE issue is only reachable by
fusing the chains those live saves belong to, i.e. §5.1.

**5.3 Re-run the §1 walk on the shipped NEFF before choosing anything.** It already shifted:
queued transfers 155 → 154 but 255.7 → 268.7 us (the SB→SB reload is a little slower per hop than
the DRAM→SBUF one it replaced — 32-byte contiguous runs either way), ACT DGE transfers 64 → 46
(97.6 → 70.5 us), ACT DGE issue 130 → 96 slots (70.8 → 55.5 us), and SP DGE issue appeared on the
path (2 → 46 slots, 31.2 us) because the triggers now fire early and SP's own stream became the
gate in places. Remaining path: ~405 us DMA transfer latency, ~135 us DGE issue, ~285 us DVE,
~177 us PE, ~160 us ACT compute.

**5.4 Do not spend budget on:** instruction merging (§4.2), LDWEIGHTS dedup (§4.3), engine
occupancy / moving compute between engines (§4.6 and the sibling's closed list — Pool is 17×
slower per element and `ACTIVATE` hangs there), DGE host rebalancing (≤13 us here, §1),
queue/ring depth (§4.7), DGE-side staging fusion (§4.1), shrinking dead DGE transfers (§4.4),
reordering the SP triggers (§4.5), retiring the dead saves (§5.2).

## 6. Tooling notes

* Device-run budget: a `verify.py` run is ~45–70 s, `--runs 3` ~2 min, `profile.py --repeats 5`
  ~70 s. I used 9 device cycles this iteration; budget for ~6 experiments per hour.
* `verify.py --runs 3` requires bit-identical outputs across runs — use it for anything that
  relaxes a wait; it caught nothing here but it is the only race check available.
* A hang shows up as `exit 1` after ~70 s with `Failed to execute model: 10001, error: 5` and
  `Model stop timeout (30000 ms)`. The device recovers; the next run is normal.
* `profile.py --repeats 5` is ~70 s; `--parquet prof` adds the tables. Baseline spread 0.18%,
  edited 0.35–0.56%, so claim nothing under ~7 us.
* `isa.Instruction.set("src_step_elem", [a,b])` takes the whole array. The 8 bytes of a DGE
  address are a union: writing `*_var_offset` (bytes 16–19), `*_var_var_id_lo` (20–21) and
  `*_var_var_id_hi` (22) with `*_var_marker` (23) already 16 reproduces a VAR0 (`SB`) address
  byte-for-byte — verified against a real `SB` address in the same NEFF.
* `neff.py pack` rebuilds the tar, so a member may change size (`SP0.json` shrank 1196 bytes).

# iter-3 — the transfer-5 lever is CLOSED. Five mechanisms refuted, one of them silently.

## Result: **no change shipped** (kernel.neff reverted to iter-2's `c8fefa05…`, 61.1–61.4 us)

I found a 0.7 us win and then **threw it away because it was a race**. The `--runs 3`
determinism check is the only thing that caught it; `verify.py` with no arguments (which is
what the loop's constraint runs) passed it, with `max_abs_err` inside the limit, every time.

| step | edit | median us | verdict |
|---|---|---|---|
| 0 | iter-2 NEFF, re-measured twice (before / after a driver reload) | **61.116 / 61.358** | — |
| 1 | merge transfer 6's transpose descriptor into transfer 5; `PE[23]` wait `S[30]`→`S[29]` | 61.123 | neutral, and **racy** (§3) |
| 4 | same for transfers 6+7+8; `PE[23]`,`PE[38]`,`PE[53]`,`ACT[49]` → `S[29]>=16` | **60.433 / 60.683** | **-0.68 us, but RACY — reverted (§3)** |
| 5 | transfer 5 split: 5 needed windows + transposes in entry 5, full copy in entry 6 **in parallel** | 61.374 | **refuted (§4)** |
| 6 | same split but entry 6 **deferred** to after `S[29]` | NaN output | **refuted (§5)** |
| 2 | split transfer 5 across two rings, `semaphore` field overridden to 29 | NRT hang | **refuted (§6)** |
| 3 | same, half B parked in entry 17 + appended entry 33 | NRT hang | **refuted (§6)** |
| p | probe: overwrite the 96 "padding" bytes of every staging window | NaN output | they are required zeros (§8) |
| o | direct test of intra-transfer descriptor ordering | non-deterministic | **descriptors are unordered (§3)** |

`launch_rules.py` PASS. Shipped NEFF `verify.py --runs 3`: PASS, bit-identical across runs,
`max_abs_err` identical to the baseline (output_0 6.092e-05 = 11%, output_1 4.494e-04 = 83%).
Shipped NEFF measured three times across this iteration: **61.116 / 61.358 / 61.492** (7 captures
each; the 61.116 was before the two driver reloads in §6, the other two after). iter-2 recorded
61.245 for the same bytes, so the 7-capture median of this NEFF drifts over about **±0.25 us**
on this machine — treat anything under ~0.5 us as unmeasured until you have A/B repeats.

---

## 0. READ THIS FIRST: always run `verify.py --runs 3`, not `verify.py`

`tools/verify.py` with no arguments does **one** run and cannot see a race. `--runs 3` adds
`[PASS] all 3 runs produced bit-identical outputs` and prints `FAILED -- this edit must not be
kept` when they differ. The unedited NEFF and iter-1/iter-2's NEFFs are all bit-deterministic,
so **any non-determinism is yours**. My edit-4 passed single-run `verify.py` four separate
times (and would have been accepted by the loop) while being a genuine data race.

Cost me ~25 minutes to find. Make it the first check after every edit.

## 1. The measurement that mattered, and the elasticity map it corrects

iter-2 §9 concluded "the kernel is rigid from `PE[23]` (21.4 us) to the end, slope 1.0, so the
`transfer-5 → transfer-6 → PE[23]` chain is elastic almost 1:1". **The slope is right but the
causality is not.** I moved `PE[23]` 1.33 us earlier (measured: 21580 → 20246) and the finish
did not move at all. What happened instead:

* `PE[26]` waits `S[4]>=19` = Activation's 19th signalling slot (`ACT[43]` CAST), and `ACT`'s
  chain hangs off PE's own earlier matmuls. PE and ACT ping-pong with ~135 ns handoffs.
* Freeing `PE[23]` just exposed the next gate, `ACT[49] ← S[32]` (queued transfer 8).
* Retargeting that one too (edit 4) moved the whole `ACT`/`PE` chain 857 ns earlier and bought
  0.68 us of total time — i.e. **once past `PE[23]` the path is a PE↔ACT↔DVE cycle, and you have
  to free several gates at once to get anything.** A single landmark moving earlier proves
  nothing; `corr(landmark, total) ≈ 1` only means everything shifts together.

Useful slack table (iter-2 NEFF, `prof/`), semaphore-reach vs. the consumer that waits on it:

| sem (transfer) | `>=16` reached | consumer | consumer starts | slack |
|---|---|---|---|---|
| `S[29]` (5) | 19835 | `PE[21]` | 19877 | **0** |
| `S[30]` (6) | 21538 | `PE[23]` | 21580 | **0** |
| `S[31]` (7) | 22254 | `PE[38]` | 23033 | 779 |
| `S[32]` (8) | 22462 | `ACT[49]` | 23009 | 547 |
| `S[33]` (9) | 22669 | `PE[66]` | 26786 | 4117 |
| `S[34]` (10) | 22876 | `PE[80]` | 28106 | 5230 |
| `S[35]` (11) | 23093 | `PE[84]` | 29612 | 6519 |
| `S[36]` (12) | 23295 | `PE[114]` | 33097 | 9802 |

So only transfers 5, 6 and (just) 7/8 are on the path. 9–12 have microseconds of slack: do not
bother with them.

## 2. Anatomy of the one DMA chain that is on the critical path

`DVE[31] CAST` (DVE's 19th signal, writes `p64+0x5780`, the last 128 B of transfer 5's source)
finishes at 16284 → that is a **tight, correct** gate; `SP[9]`'s `S[6]>=19` cannot be lowered.

```
S[6]>=19 @16284 → SP[9] trigger @16300 → DMA start 17006 (+710)
                → 32 KB copy, 2094 ns → end 19102 → S[29]>=16 @19835 (+733)
                → SP[13] trigger @19851 → DMA start 20571 (+710)
                → 8 KB transpose, 234 ns → end 20805 → S[30]>=16 @21538 (+733)
                → PE[23] @21580
```
Fixed per-hop overhead is **~1450 ns** (710 trigger→start, 733 end→semaphore-visible) and it is
not reducible by anything I could find. The only two ways to shorten the chain are to remove a
hop or to speed up the 2094 ns copy. §3 kills the first; §4 kills the second.

## 3. REFUTED, and the trap: descriptors inside one queued transfer are **NOT ordered**

Appending transfer 6's transpose descriptor to transfer 5's `desc` list and pointing `PE[23]` at
`S[29]` removes a whole 1450 ns hop. Chapter 9.1 says a transfer's descriptors are "done in
order", so the transpose (which reads what transfer 5's four copy descriptors write) should be
safe. It passed `verify.py` bit-identically on a single run — and
**`verify.py --runs 3` is not bit-identical, for the minimal one-descriptor version.**

Why: a queued transfer signals its semaphore in **16 unit increments** (`SemaphoreUpdate` shows
`S[29]` going 1,2,…,16; `>=16` is the completion), i.e. the transfer is spread over the queue's
`num_queues: 16` hardware sub-queues and its descriptors run **concurrently**. I ruled out every
other explanation before concluding this:

* not the `ACT[49]`/`PE[53]` retargets — `ACT[49] ACTIVATE` really is a WAR barrier (it *writes*
  `p0+0x37300`, which is transfer 8's second source, so its `S[32]>=16` wait means "transfer 8
  has read it"); but edit-7 made transfers 6/7/8 32-byte self-copy no-ops and restored `PE[53]`,
  and it was *still* non-deterministic (and `output_0`'s error moved 6.092e-05 → 6.364e-05);
* not another writer: window 0's only other writer is `DVE[5] MEMSET`, which finishes at ~5 us
  (§8). I scanned every engine's instruction `dst` (both the `_addr_immediate` and the
  `_var_offset` forms) and every DGE/queued descriptor `to_off` for
  `partition<64, 0x36f00..0x377ff`; nothing writes the windows the transposes read between 19.2
  and 21 us.

**Then I tested it directly** (`scripts/ordtest.py`, one device run): move transfer 5's
descriptor 0 — the only writer of window-0 bytes for partitions 0-15 — into transfer 6, *in front
of* transfer 6's transpose of window 0, and move that data's two consumers (`PE[21]`, and the
triggers of transfers 7-12 via the spare `SP[14]`) from `S[29]` to `S[30]`. If descriptors were
ordered the output would be unchanged. **It is not bit-identical across 3 runs** — while
`max_abs_err` stays at the baseline value, which is exactly what an ordering race looks like
(most of the time the copy does land first). Confirmed, not inferred.

**Rule: never let a descriptor read what a sibling descriptor of the same transfer writes.**
Merging *independent* transfers into one `desc` list is still fine (and iter-1's SB→SB DGE
fusions are unaffected — a DGE is one descriptor).

This closes "fuse queued transfer 5 away" (iter-2 §10.1, the review's #1 candidate): the fusion
needs the transpose to read the copy's output inside one transfer.

## 4. REFUTED: splitting transfer 5 for parallelism. It is **packet-rate bound**

The copy moves 32 KB as **1024 packets of 32 bytes** (`DmaPacket` rows are 32–40 B, 13–19 ns
each) because its destination has 32-byte runs at a 128-byte stride. 32768 B / 2094 ns =
15.6 GB/s, vs 49 GB/s for the 128-byte-run transposes. It looks issue-limited. It is not:

> edit 5 ran a 10 KB slice-copy (entry 5) and the full 32 KB copy (entry 6) on two different
> rings, triggered 240 ns apart. They **shared one throughput budget**: entry 5 took 1720 ns
> instead of ~650, entry 6 took 3794 ns instead of 2094, and entry 5's end→semaphore latency
> blew up from 733 to **1640 ns**. Total 61.374 us, worse than edit 4 and worse than doing
> nothing.

Combined with §3 (the descriptors of one transfer *already* run concurrently on 16 sub-queues),
this says the 2094 ns **is** the fabric's packet rate for that access pattern. Splitting it —
across rings, across descriptors, across hosts — cannot help. iter-2 §6 reached the same
conclusion for the 50–54 us save group; it holds here too.

The only remaining way to make this copy cheaper is to give it a **denser destination** (128-byte
runs → ~520 ns, worth ~1.5 us). That needs the layout of the whole staging region changed and
every consumer in §8 re-addressed. It is mechanical but large, and §8's uninitialised-garbage
problem makes it delicate. That is the one big prize left — see §9.

## 5. REFUTED: deferring the non-critical part of the copy (NaN output)

The four merged transposes only read 5 of the copy's 16 128-byte windows (m = 0,4,5,8,12), so I
tried: entry 5 = just those windows (10 KB) + the transposes; entry 6 = the full copy, triggered
*after* `S[29]`, with a new SP gate `S[30]>=16` in front of the triggers of transfers 7–12.
**Output was all NaN.** Reason: `PE` reads the staging region directly at windows 0, 5, 10, 12,
14, 15 (§8) with no semaphore of its own — it rides on `PE[21]`'s `S[29]>=16` and PE program
order — so the *whole* region has to be complete at `S[29]`. (The source `p64+0x5600..0x57ff` is
also reused for the next chunk, so a deferred copy can read the wrong chunk.)

Useful by-product: the descriptor arithmetic for slicing the copy by window is written down and
device-verified (edit 5 passed `verify.py`). For window set `{d1} × [d2lo, d2lo+nd2)`:

```python
{"op":"copy","num_tiling_dimensions":1,
 "from":"SB","from_dtype":"bfloat16",
 "from_off":(64+16*d2lo)*P + 0x5600 + 128*g + 32*d1,
 "from_sizes":[32,16,nd2,1], "from_steps":[1,P,16*P,1],
 "to":"SB","to_dtype":"bfloat16",
 "to_off":16*g*P + 0x36f00 + 32*g + 128*d1 + 512*d2lo,
 "to_sizes":[32,16,nd2,1], "to_steps":[1,P,512,1]}      # P = 262144, g = 0..3
```
(Check any new descriptor by building the set of (dst,src) byte pairs and asserting it is a
subset of the original's pairs — that validates the dim pairing, not just the address range.)

## 6. REFUTED (two hangs): you cannot add or re-home a queued transfer

Both attempts to get transfer 5 onto two rings deadlocked the device (70 s `NRT_TIMEOUT`).

1. **The JSON `semaphore` field is ignored.** I parked half the copy in the unused-at-that-moment
   entry id 17 and set its `semaphore` to 29 so both halves would signal `S[29]` (reaching 32).
   `semaphores.py` was happy; the device hung. **A queued transfer's completion semaphore is
   derived from its ring, `semaphore_set[id % 16]`, not read from the JSON.** Chapter 9.1's
   "always `semaphore_set[id % 16]`" is a hardware rule, not an observation about the data.
2. **Appending an entry does not work either.** Second attempt: half B in entry 17 signalling its
   natural `S[25]`, entry 17's original 512-byte transpose appended as a new entry id 33 (also
   ring 1, after id 17 in *both* list order and trigger order, so it is correct under either the
   "trigger names the entry" and the "trigger advances the ring" model), a new
   `PSEUDO_DMATRIGGER` for entry 17 placed before the three SP DGEs, the old id-17 trigger
   re-pointed at 33, `ACT[91]`'s `S[25]>=32` bumped to `>=48`, and an `S[25]>=32` gate added on SP
   and PE. Semaphore accounting balanced (24..39 all `max_wait == total_inc`). **Hung anyway**,
   so something else about a 34th transfer is rejected — most likely the 16 sub-queues are loaded
   from the whole `dma` list at load time, so an appended entry's descriptors land after every
   other transfer's in each sub-queue.
   *Changing an existing entry's descriptor count is fine* (edit 4 took entry 5 from 4 to 8
   descriptors and transfer 21, which shares ring 5, still completed normally). It is adding an
   entry that breaks.

**Treat `SP0.json`'s `dma` list as fixed in length, with each entry's `id`, `queue` and
`semaphore` fixed. You may rewrite an entry's descriptors.**

### Device recovery after a hang (you will need this)
A hung NEFF leaves `nd0` wedged: every later run fails with
`tdrv_hbm_scrub_wait: nd0 HBM 2 scrub failed: timeout` → `nrt_init: Failed to initialize
devices, error:1`. Waiting does not help (I waited 3 minutes twice). What works:
`sudo rmmod neuron && sleep 3 && sudo modprobe neuron && sleep 12`, then `neuron-ls`, then
re-run a known-good NEFF to confirm. ~2 minutes; budget ~10 for the whole detour, and note the
machine drifts ~0.25 us across a reload, so **re-measure your A-side after one**.

## 7. What edit 4 actually bought, in case a safe version is ever found

It is worth recording because it is the only thing that moved the number this iteration:
merging the transposes of transfers 6, 7 and 8 into transfer 5 and pointing `PE[23]`, `PE[38]`,
`PE[53]`, `ACT[49]` at `S[29]>=16` cost +203 ns on `S[29]` (19835 → 20038, i.e. 32 KB of extra
transpose work added almost nothing — further evidence for §4's throughput story) and moved
`PE[23]` 21580 → 20158, `ACT[49]` 23009 → 22152, `PE[40]` 23571 → 22714, total
**61.12/61.36 → 60.43/60.68 us (-0.68, -1.1 %)**. So *if* the three round trips could be removed
safely, 0.7 us is there. The blocker is purely §3's ordering.

The only ordering-safe way I can think of: do the transposes with the **PE transpose unit**
(`LDWEIGHTS` has a `transpose_mode` field, `DISABLED` on `PE[21]`) instead of the DMA, so the
0x5900/0x5a00/0x5b00/0x5c00 staging hops disappear. That changes which engine does the work and
is a much deeper edit; I did not attempt it.

## 8. Map of the staging region `partitions 0-63 × 0x36f00..0x377ff` (expensive to rebuild)

It is a 64 × 1024 bf16 array of 16 128-byte "windows" (window `m` at `0x36f00 + 128m`).
**Queued transfer 5 writes only 32 of the 128 bytes per partition per window** — partition `p`
gets bytes `[32*(p//16), +32)`, from `p(64 + p%16 + 16*d2) + 0x5600 + 128g + 32*d1` where
`g = p//16`, `m = d1 + 4*d2`.

**The other 96 bytes are deliberate zeros, and they are live.** `DVE[5] MEMSET` (slot 5, runs at
~5 us) zeroes `0x36f00`, 512 int32 = 2048 bytes, `num_active_channels: 64` — i.e. exactly
partitions 0-63 × windows 0-15, the whole region. So each window is a 64 × 64 bf16 operand whose
only non-zero part is the four 16 × 16 diagonal blocks transfer 5 scatters in; the zeros are the
off-diagonal blocks the matmuls need. I confirmed it with one device run (`scripts/probe.py`):
appending a descriptor to transfer 5 that overwrites bytes 32..127 of every window on partitions
0-15 with arbitrary stable data gives **all-NaN output** (32768 NaN in output_0), bit-identically
across runs. They are not scratch.

Readers (src) — `win` is `m`, `part` is the partition of the base address:

| reader | win |
|---|---|
| `PE[21]`, `PE[25]` | 0 |
| `PE[36]`, `PE[43]` | 5 |
| `PE[57]`, `PE[67]`, `PE[71]` | 10 |
| `PE[145]`, `PE[147]`, `PE[156]`, `PE[158]` | 12 |
| `PE[136]`, `ACT[78]` (part 0/32), `ACT[109]` | 14 |
| `PE[98]`, `PE[101]` | 15 |
| `DVE[104]`, `DVE[115]`, `DVE[125]`, `DVE[132]` | 8 |
| queued 6 / 7 / 8 / 9 / 10 / 11 / 12 (64×64 WZXY transposes) | 0 / 5 / 4+8 / 10 / … |

Writers (dst): queued 5 (all windows, 32 B/partition), queued 31 → win 8, `Pool[13]` DGE →
win 16, `DVE[5] MEMSET` → **all of windows 0-15** (see above), `DVE[84]` → win 14,
`DVE[111]`, `DVE[121]` → win 12,
`DVE[129]` → win 16, `ACT[49]`, `ACT[77]`, `ACT[99]`, `ACT[107]` → win 8 (`0x37300`),
`ACT[72]`, `ACT[98]`, `ACT[106]`, `ACT[113]` → `0x37308`, `ACT[76]` → win 12, `ACT[78]` → win 14.
Several of those `ACT` writes are what the `S[3x]>=16` waits on `ACT` are really guarding (WAR),
**not** RAW — check which before retargeting any wait.

Transfer 5's source is `p64..p127 × 0x5600..0x57ff`, dense, 512 B/partition; `0x5780` is written
by `DVE[31] CAST` (DVE signal #19) and the buffer is reused for the next chunk soon after.

## 9. Ranked candidates for iter-4

1. **~~Dense relayout of the staging region~~ — DEAD, settled this iteration.** The idea was to
   give transfer 5's destination 128-byte runs instead of 32 (2094 ns → ~520 ns, worth ~1.5 us).
   It cannot be done: §8's probe proves the 96 bytes between the real ones are required zeros, so
   the 32-byte scatter stride *is* the operand's block-diagonal structure. Which column a byte
   lands in selects which matmul output it feeds, so no denser arrangement computes the same
   thing. The whole 16.3 → 19.8 us staging copy is therefore irreducible at the NEFF level.
2. **`PE`-side transpose instead of the DMA transposes** (§7, ~0.7 us) — the best remaining idea,
   and the only one I know of that would remove a 1450 ns hop without breaking §3.
   `LDWEIGHTS.transpose_mode` is `DISABLED` on `PE[21]`; if `PE[23]`'s operand can be loaded
   transposed straight from `0x36f00`, queued transfers 6/7/8 and their round trips vanish.
   Needs the PE operand semantics worked out — read chapter 7 on `LDWEIGHTS`/`MATMUL` first, and
   note this *does* change which engine does the arithmetic, so watch `output_1`'s 83 % error
   budget.
3. **Do not retry:** fusing transfer 5 into its consumers (§3), splitting transfer 5 for
   parallelism in any form (§4), deferring any part of the copy (§5), adding or re-homing a
   queued transfer (§6), transfers 9–12 (§1, microseconds of slack), plus everything iter-1 §7
   and iter-2 §10.4 already closed.

### The 40.2 → 44.5 us hole: I traced it, it is closed
4.4 us of `ACT` idle, the second-biggest gap in the profile. The chain, fully resolved:

```
SP[34] DGE row ends 40300 ──(1650 ns of DMA latency)──► S[46]>=16 @~41950
Pool[22] DGE ─► S[45]>=16        ┘ both feed Pool[24] (blocked 40788→42053)
Pool[24] → Pool[25] (S[44]) → Pool[26] (S[43], and it is the only producer of S[16])
   → S[16] releases SP[35] (blocked 40355→42222, extended wait on S[16])
   → SP[36] triggers queued transfer 20 @42263 → S[28]>=32 → PE[134] @43142
```
So the hole is **not** the transfer-20 trigger round trip (that is only the last 880 ns of it);
it is the ~1650 ns DGE-completion latency of the staging saves feeding the `Pool[24..26]` fence,
in exactly the window iter-2 §6 measured as **DMA-fabric limited, not issue limited**. Every
instruction in the chain already issues within ~85 ns of its gate clearing. Moving the transfer
earlier would need the fence weakened, and the fence is a real multi-producer barrier. Nothing to
take here. (`semaphores.py` cannot see `S[16]`'s waiter because `SP[35]` reads it only through the
**extended** events block — scan `events_extended_wait_*` yourself, as iter-1 §4 warns.)

## 10. Tooling (rebuilt in ~20 min; `scripts/` is not preserved)

`d2.py` compact stream dump (`load(engine)` → list of `isa.Instruction`; beware `decode_stream`
yields `(index, ins)` pairs and is a generator), `dq.py` queued descriptors with
`p<part>+0x<off>` addresses, `waits.py` every wait incl. extended grouped by semaphore,
`tl2.py`/`tw.py` per-engine and per-window timelines from `<prof>/Instruction.parquet`,
`rch.py` semaphore reach times for `>=16/32/48`, `gaps.py` idle stretches on one engine,
`edit1..7.py` each edit, self-asserting.

Gotchas that cost time: `DataFrame.sem` is a pandas method, so never name a column `sem`;
`compiler_pc` is a float in the parquet; `Instruction.execution_unit` separates `sequencer` from
`compute`, and an `EVENT_SEMAPHORE` row is *sequencer* time, which is why `PE[53]` appears to run
before `PE[46]` (the sequencer runs ahead of the matrix unit — do not read that as reordering);
`count_union_block_id` and `count_union_count` are the same 4 bytes at offset 48 of a
`PSEUDO_DMATRIGGER`; `MATMUL`/`LDWEIGHTS` have no extended-events fields, so you cannot give them
a second wait — insert an `EVENT_SEMAPHORE` slot instead (copying one and overwriting its wait
works, but assert the template is bare first, iter-2 §4).

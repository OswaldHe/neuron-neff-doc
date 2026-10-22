# iter-1 — selective_scan b1 c768 s1024 n16 bf16

**Result: 602.637 -> 516.5 us (-14.3%, 1.167x). Numerics bit-identical to the baseline**
(output_0 max_abs_err 6.245e-02 = 83% of limit, cosine 0.999998; output_1 4.268e-05 — exactly
the baseline figures). Three independent 5-repeat measurements of the shipped file (sha256
`67c657b0...`): medians **516.469 / 516.692 / 516.933** us, each with spread 0.05-0.07%. The
drift across the three is 0.09%, so quote this as -14.2 to -14.3%.
`verify.py --runs 3`: all three runs bit-identical (no race). `semaphores.py` clean,
`launch_rules.py` clean. 3 of 34 members changed: `sg00/DVE0.bin`, `sg00/Pool0.bin`,
`sg00/Activation0.bin`.

## FIRST: read the prior case study, it is the same kernel

`/home/ubuntu/workspace/neff_doc/experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/`
is **this exact nkilib kernel** at c2560/s4096, optimized 7919.96 -> 6603.96 us (1.199x) over
7 iterations, with a **complete, verified build script** in `logs/notes/iter-1.md` and the
refuted levers in `logs/notes/iter-{1..7}.md`. Everything in this iteration is that work ported
to c768/s1024 plus three new placement wins. Read it before doing anything else; it saved me
about 90 minutes of rediscovery. The iter-0 review's candidate list (C1 scan merging, C2 bf16)
is **already closed there by measurement** — see "Closed" below.

## What shipped (one build, four stacked edits)

Build pipeline, from `baseline/kernel.neff` unpacked into `work/`:

```
ESHOST=1 FRESH="[97,197,295,395,495,597,700,796,900,997,1101,1198]" python -I build3.py
MAXRUN=2 python -I merge_copies.py
neff.py pack work/ kernel.neff
```

Both scripts are reproduced at the end of this file.

| # | edit | us | DVE busy | Pool busy |
|---|---|---|---|---|
| — | baseline | 602.64 | 592.4 | 101.0 |
| 1 | move 170 `y+=Cs` fp32 TT ADDs from DVE to Pool (prior case study's recipe, verbatim) | 524.50 | 497.2 | 435.6 |
| 2 | merge the 192 1-element Pool carry `COPY`s pairwise (77 pairs) | 523.70 | 497.5 | 411.8 |
| 3 | **NEW:** host the S[3] hand-off waits on *original* DVE `EVENT_SEMAPHORE` slots (free extended wait), not only on the freed ADD slots | 517.66 | 499.0 | 411.9 |
| 4 | **NEW:** for the 12 group-boundary ADDs, insert a *fresh* DVE `EVENT_SEMAPHORE` holding the hand-off wait immediately before its deadline slot | **516.47** | 501.5 | 414.9 |

Edits 3 and 4 are the new contribution: **-7.2 us (-1.4%) purely from moving the hand-off wait
later in the DVE stream**, no change to what either engine computes.

## Measured calibration for THIS shape (c768 s1024)

| | |
|---|---|
| DVE TT fp32 512x128 | 672 ns baseline, 687 ns after the move |
| DVE `TENSOR_TENSOR_SCAN` fp32 512 | 1272-1276 ns (dtype-insensitive, see Closed) |
| DVE `EVENT_SEMAPHORE` | 24-54 ns when it does not block |
| DVE MEMSET 512 fp32 | 485 ns; 128 bf16 165 ns |
| **Pool TT fp32 512x128** | **1350 ns probe / 1577 ns loaded / 2009 ns after the COPY merge** (2.0-2.9x DVE) |
| Pool 1-element `COPY` | 308 ns idle, 678 ns loaded, 268 ns after pairwise merge |
| Pool DGE DMA trigger | 656 ns |
| Marginal DVE saving per removed TT | ~0.55 us (0.80 of its duration; neighbours reclaim the rest) |
| Consecutive DVE instructions overlap | ~83 ns, so `DVE_span ~ sum(duration) - 83*n` |

Structure: 6 channel tiles x 2 chunks of 512 = 12 groups x 16 states = 192 scans. Per state DVE
ran MULT(`dtx*B_psum`) -> SCAN -> MULT(`C_psum*state`) -> ADD(`y+=Cs`); the last of those four is
what moved to Pool. DVE 1204 slots, 640 S[6] updaters. Semaphores: S[3]=Pool, S[4]=ACT, S[5]=PE,
S[6]=DVE, S[8..39]=DMA queues.

## Why edits 3 and 4 work (the generalisable finding)

The prior case study placed each Pool hand-off wait at "the latest **freed ADD slot** before the
deadline". That is needlessly early: the baseline DVE stream is full of compiler
`EVENT_SEMAPHORE` barriers, and **most of them have an unused *extended* wait block**
(`events_extended_wait_mode/_wait_idx/_sem_wait_value`, bytes 32-43 of `CTRL_ES`). Any such slot
can host the `S[3] >=` hand-off without disturbing its own primary wait. Example at the first
group boundary:

```
 96 SCAN                    (compute)
 97 ADD y+=Cs[55616]        <- moved to Pool; freed slot, the old wait host.  deadline d = 102
 98 ES  wait S6>=49         ext unused   <- hostable
 99 MULT C_psum*state       (compute, 0.69 us)
100 ES  wait S12>=16        ext unused   <- hostable
101 MULT dt*x               (compute, 0.69 us)
102 ADD y+=Cs               reads 18752  <- the deadline
```

Hosting at 100 instead of 97 buys DVE two compute instructions (~1.4 us) of extra slack per
group; inserting a fresh ES at 102 buys the third. Gap attribution before/after:

| binding wait of each DVE compute gap | after edit 2 | after edits 3+4 |
|---|---|---|
| **S[3] extended (Pool hand-off)** | **14.20 us (12 gaps)** | **0.35 us (1 gap)** |
| none (input DMA queues S[8]/S[9], mostly the 3.7 us startup load) | 4.72 | 4.43 |
| S[6] primary/extended (DVE self-wait, pipeline drain) | 2.54 | 4.68 |
| S[4] (ACT) | 1.68 | 1.54 |
| **total DVE compute gap** | **23.1 us** | **11.0 us** |

**The kernel is now at the DVE compute floor**: DVE compute union-of-intervals 501.3 us inside a
526 us span, 11.0 us idle over 36 gaps, and 3.7 us of that is the startup input DMA. Pool is
414.9 us busy with ~100 us idle. Anything further must *remove DVE instruction duration*.

### How to find the binding wait (reusable recipe)
1. `supply[(sem, ordinal)] = end_ts` of the instruction performing that increment — walk the
   owning engine's stream in order, counting **primary and extended** `SEM_INC_COMPLETE`.
2. Union-of-intervals over DVE rows **excluding `EVENT_SEMAPHORE`** (the ES slots issue in
   parallel with compute; including them hides the real gaps: 520.8 vs 500.7 us union).
3. For each gap, consider every wait (primary **and** extended) on every slot in
   `(prev_pc, pc]` and take the one with the latest `supply` time.
`/tmp/xp/bind.py` and `/tmp/xp/gaps.py` in this iteration did exactly this.

## Refuted here by measurement — do not redo

1. **Keeping the 12 group-boundary ("lock-step") ADDs on DVE is a loss.** These are the ADDs
   whose hand-off wait lands on their own slot because the deadline is only 5 slots away, and
   they are *exactly* the 12 binding X[3] stalls. Excluding them from the move:
   524.50 -> 528.21 us (without the COPY merge) and 523.70 -> 526.03 us (with it). DVE busy rises
   by the expected 6.3 us but the gap only shrinks ~4 us: Pool's backlog at a group boundary is
   **two** ADDs deep, so pulling one back just makes the previous one binding. Edits 3+4 fix the
   same 14 us for 0.9 us of DVE cost instead of 8.3 us.
2. **Inserting the Pool ADDs ahead of the preceding DMA triggers (`SKIPDMA`) does nothing.**
   At each group boundary Pool runs COPY, ADD, 2-3 `PSEUDO_DMA_DIRECT2D` (~2 us), COPY, ADD, and
   DVE waits on that last ADD. Moving the ADD before the DMA triggers: 523.70 -> 523.76 us
   (noise). The ADD's own `S6 >= v` is not satisfied any earlier, so it just blocks earlier and
   pushes the DMAs out. Verified bit-exact, simply pointless.
3. **Offloading tensor-tensor work to Pool/GpSimd at 512 elements costs 2.0-2.9x DVE.** Probe:
   all 192 Pool carry `COPY`s rewritten as 512-element fp32 `TENSOR_TENSOR` ADDs (timing only,
   wrong output): Pool 101 -> 300 us => **1350 ns per op**, and DVE's own TT went 672 -> 879 ns
   (+31%) from the concurrent SBUF traffic, total 602.6 -> 728.4 us. So Pool is only worth using
   while DVE is the longer engine. We are now at the crossover: DVE span 516, Pool span ~520.
4. **The DVE/Pool split is at its optimum.** Model fitted here: `DVE_span ~ 516 - 0.88*x`,
   `Pool_span ~ 415 + 2.9*x` for `x` = us of DVE duration moved. They cross at x ~ 27 us
   (~39 more TT ops, projected ~492 us) but **every remaining DVE TT increments S[6]**, which the
   prior case study measured as a net loss to move (585 ns guard vs 548 ns saving), and all 21 of
   the remaining `y+=Cs` ADDs sit at the group boundary where a 2 us Pool latency lands straight
   on the critical path.
5. **Deleting the 21 S[6]-incrementing `y+=Cs` ADD slots and renumbering S[6] is not free:**
   every one of the 21 has 1-2 *exact* waiters on its own S[6] ordinal (checked in all five
   streams, primary + extended), so the usual "map the deleted ordinal down to the previous
   survivor" rule weakens a real dependency. Each waiter would have to be re-pointed at Pool's
   S[3]. Not attempted.
6. **`FP32R` (DTYPE 11) is not faster on DVE — CLOSED.** Probe: all 418 all-fp32 DVE
   `TENSOR_TENSOR_ARITH_OP` slots patched to `in0=in1=out=FP32R` (timing only, output wrong).
   It does **not** hang — it runs with exit 0 — and `TENSOR_TENSOR` duration is **695 ns vs
   687 ns**, exec 516.38 vs 516.47 us. So on the DVE, FP32R is the same datapath rate as FP32;
   the only 2x on this engine is full bf16, which is structurally blocked (see Closed). Useful
   side fact: an unused DTYPE value did *not* hang the DVE, so the microcode-table rule is about
   (opcode, alu_op) more than dtype.
   **Same for the scan — CLOSED:** all 192 `TENSOR_TENSOR_SCAN_ARITH` slots patched to
   `in0=in1=out=imm0=FP32R` run fine and measure **1273.9 ns vs 1272.6 ns**, exec 516.89. With
   bf16 already closed by the iter-0 review (1268 -> 1268 ns), **the scan runs at 2.08
   ns/element in every dtype the ISA offers**; its 244 us is simply the floor.
7. **Moving a DVE MULT to Pool needs cross-engine wait re-pointing** (census of exact S[6]
   waiters, all five streams, primary + extended, on the pristine baseline):

   | DVE TT family | n | S[6] updaters | with **zero** exact waiters | waiters by stream |
   |---|---|---|---|---|
   | `dBx = dtx*B_psum` | 192 | 192 | **0** | DVE 184, PE 168 |
   | `Cs = C_psum*state` | 192 | 192 | **0** | DVE 171, PE 173, ACT 9 |
   | `y += Cs` | 204 | 32 | **0** | DVE 26, Pool 12 |
   | `dt*x` bf16 etc. | 12 | 12 | **0** | DVE 12 |

   The PE waiters are the PSUM-bank WAR guards. So the cheap "delete the slot and renumber S[6]
   down" route is unavailable for every remaining DVE TT: each one needs its PE (and DVE, ACT)
   waiters re-pointed at Pool's S[3]. PE has 355 `EVENT_SEMAPHORE` slots that could host such a
   wait, so it is possible, just not small.
8. **Pool/GpSimd cannot access PSUM — CLOSED, and this is the hard ceiling on this kernel.**
   The prior case study left this "unknown" (its probe hung for a sync reason). Settled two ways:
   * **Device.** *One* Pool `y+=Cs` ADD, last in the stream, `src1` repointed from SBUF 53568 to
     PSUM `0x2000000`, **nothing else changed** — same wait (`S6>=636`), same update, and the
     PSUM address word is encoded exactly as DVE encodes it (byte 3 = marker 2, which
     `0x2000000` sets for free; checked against DVE slot 14's `src0=0x2000800 marker 2`).
     Result: `Failed to execute model: 10001, error: 5` = **NRT_TIMEOUT**, 70 s — *not* the
     `PSUM_COLLISION` / nrt 1004 that a WAR race produces. No new dependency was introduced, so
     the only explanation is that the instruction never retired: it never incremented S[3] and
     DVE deadlocked behind it.
   * **The control.** The *same* one-field edit applied to an **ACT** slot (`ACTIVATE` slot 115,
     `src_mem_pattern_start_addr_addr_immediate` 197184 -> `0x2000000`, nothing else) **runs with
     exit 0** and does not even change the timing (516.66 us). So the edit shape is sound and the
     NEFF-local microcode is not the obstacle: it is the engine.
   * **Corpus.** Decoding every compute slot of all 14 studied NEFFs and counting operands in
     `[0x2000000, 0x2004000)`: **Pool 0 of 54,042 slots**, Activation 55,760, DVE 48,035,
     PE 714,785. The compiler never emits a Pool PSUM reference. (chapter 5's "PE writes, other
     engines read" is too generous — it should exclude Pool.)

   Consequence: **both MULT families are permanently DVE-only**, because each has one operand in
   a PSUM bank written by a PE matmul (`dBx = dtx * B_psum`, `Cs = C_psum * state`). Together
   with the scan being unmergeable and DVE-only, that is 244 + 264 = ~508 us of DVE *duration*
   that cannot leave the engine. Measured DVE compute union is 501.3 us, so **the shipped NEFF
   is within a few microseconds of the floor reachable without re-addressing**.

## Closed by the prior case study — verified there, do not redo

* **Merging two chunks' scans, or two states' scans, into one instruction (iter-0 review's C1):
  impossible.** `TENSOR_TENSOR_SCAN` does **not** re-seed its carry across access-pattern dim 1
  (falsifier: `[512,1]` -> `[256,2]` over the same contiguous elements stayed bit-exact), so the
  carry would chain across states. Also confirmed here: the per-state buffers rotate over only 3
  addresses at +12288 B, never contiguous. The 244 us scan (49% of DVE) is the hard floor.
* **bf16 for speed (C2): blocked structurally, not just by precision.** The fast path needs *all*
  of in0/in1/out bf16 (bf16-in + fp32-out stays 680 ns), and one operand of both MULT families is
  a **PSUM** bank written by a PE matmul, i.e. fp32. On top of that the baseline already uses 83%
  of the error limit.
* **ACT/Scalar cannot execute `TENSOR_TENSOR_ARITH_OP` (0x41)**: it runs with exit 0 and silently
  drops the term. Pool is the only off-load target for tensor-tensor. SP/Sync has no datapath.
* **Pairwise merging of the two MULT families**: built and refuted in both directions — line 224
  must merge at the earlier slot and then needs its PSUM operand one matmul sooner, which costs
  more than the 136 ns it saves (PE is paced by PSUM availability, which DVE controls); line 267
  must merge at the later slot, which deadlocks against the hand-off wait. Worth ~136 ns/pair.
* **Deleting the y_accum MEMSETs** (turn the group's first ADD into a `COPY`): worth only ~21% of
  its predicted value there; here it is 12 MEMSETs ~ 5.8 us of duration, so ~1 us. Not done.

## Gotchas
* `semaphores.py` only reads the **primary** events block. It reported "balanced" for a NEFF the
  prior case study deadlocked. Any renumbering must rewrite primary **and** extended waits in all
  five streams (`merge_copies.py` below does; 349 fields here).
* `isa.py`: read fields through `as_dict()`; `ins.opcode` is the numeric value, use `ins.name`
  for the mnemonic; the wait value field is `events_semaphore_value`.
* Inserting and deleting slots is safe: no PC-relative fields in these streams, and no member
  records a slot count (`Pool0.json` only names the `.bin`); `neff.py pack` handles the size.
* Device run costs: ~10 s when it works, ~9 s on a PSUM collision, ~70 s on a deadlock.
  `profile.py --repeats 5` is ~65 s. Spread on this kernel is 0.03-0.15%, so trust ~0.3%+.

## What to try next, ranked

The accounting that bounds everything below: exec 516.5 = DVE compute union 501.3 + 11.0 us of
DVE gap (3.7 of which is the unavoidable startup input DMA) + prologue. DVE compute duration is
TT 430x687 = 295.6 us, SCAN 192x1272 = 244.3 us, MEMSET 7.2 us. The scan half is immovable and
unmergeable. Pool is 414.9 us busy with ~100 us idle.

Because Pool cannot touch PSUM (refuted-item 8), **no remaining DVE instruction can simply move
engines**. Both surviving levers therefore have to change *where data lives*, which is a much
bigger edit than anything done so far. Ranked by expected value:

1. **Stage the PSUM operand into SBUF on ACT, then move the `dBx` MULTs to Pool.** ACT *can* read
   PSUM (55,760 slots in the corpus) and is only 28% busy. Insert per state an ACT copy
   `B_psum -> SBUF` (`ACTIVATE` with the identity/copy function — `act_info.json` lists
   `identity` and `copy`, and ACT already runs 204 `ACTIVATE`s, so the microcode entry exists;
   but **ACT silently ignores opcodes it cannot do**, so check the output, not the exit code),
   then the Pool `dBx` MULT reads SBUF only. Budget: Pool has ~100 us idle, Pool TT costs ~2.0 us,
   so only **~50 of the 192** can move; balance model `DVE_span ~ 516 - 0.88x`,
   `Pool_span ~ 415 + 2.9x` crosses at `x ~ 27 us` => **~492 us (-4.7%)**, and ACT grows by
   ~50 x 0.72 = 36 us (fine). Also needs the S[6] renumbering and the re-pointing of each moved
   MULT's exact S[6] waiters (PE 168, DVE 184 — refuted-item 7) onto Pool's S[3], hosted on
   PE/DVE `EVENT_SEMAPHORE` slots with a free extended wait (the trick that won edits 3+4 here).
   It also needs new SBUF scratch for the staged `B`. Expect two or three deadlocks on the way;
   this is a whole iteration, and the first thing to probe is a single staged MULT.
   **The enabler is already proven on the device** (refuted-item 8's control): one `ACTIVATE`
   repointed at `0x2000000` runs with exit 0 and no timing change, even though in the unedited
   NEFF ACT never touches PSUM (all 204 `ACTIVATE`s are SBUF->SBUF; only DVE's 384 MULTs read
   PSUM). What is still unproven is that the copy is numerically correct — ACT silently ignores
   what it cannot do, so check the *output*. Opcode census of the baseline, for reference:
   DVE 600 TT + 192 SCAN + 387 ES + 24 MEMSET; ACT 204 ACTIVATE + 28 ES + 1 table load;
   Pool 192 COPY + 60 `PSEUDO_DMA_DIRECT2D` + 42 ES + 1 NOP.
2. **Re-address the rotating buffer families** so `Cs`/`state` rotate over 4 buffers instead of 3.
   That is the only thing that lengthens the hand-off deadlines further and the only thing that
   would make `Cs`-MULT pair-merging legal at the earlier slot (worth ~136 ns/pair, up to ~13 us).
   Large coherent edit; the prior case study ranked it as its only remaining >2% lever too.
3. **Make Pool cheaper, as an enabler for 1.** Move the 115 remaining Pool carry `COPY`s to ACT
   (~31 us off Pool, and Pool's TTs should drop from 2009 ns back toward 1350 ns). Worth ~0 on
   its own — the hand-off gap is already down to 0.35 us — so only do it as part of 1.
   Needs S[3] and S[4] renumbering.
4. Not worth it: deleting the 12 y MEMSETs (~1 us), triple-merging Pool COPYs, deleting DVE
   self-wait ES slots (4.7 us, but it is removed synchronisation the verifier cannot catch),
   `FP32R` (closed), moving any of the 32 S[6]-incrementing `y+=Cs` ADDs (refuted-item 1 shows
   that direction is already past its optimum).

## Build scripts

### `build3.py` — move the free `y+=Cs` ADDs to Pool, with the new wait placement
Derived from `neff_doc/experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/logs/notes/iter-1.md`;
the `host_es` / `FRESH` blocks are new. Env: `ESHOST=1`, `FRESH=[...]`, `EXCL=[...]`.

```python
import sys, collections, bisect, os, json
sys.path.insert(0, "/home/ubuntu/workspace/neff_doc/tools")
import isa
EXCL = set(json.loads(os.environ.get("EXCL", "[]")))
SG = "work/payload/sg00/"
STREAMS = ["DVE0.bin", "Pool0.bin", "PE0.bin", "Activation0.bin", "SP0.bin"]
S   = {p: bytearray(open(SG+p, "rb").read()) for p in STREAMS}
OBJ = {p: [isa.decode(bytes(S[p][i*64:(i+1)*64])) for i in range(len(S[p])//64)] for p in STREAMS}
D   = {p: [o.as_dict() for o in OBJ[p]] for p in STREAMS}
ADDRK = ("src_mem_pattern_start_addr_addr_immediate", "src0_mem_pattern_start_addr_addr_immediate",
         "src1_mem_pattern_start_addr_addr_immediate", "dst_mem_pattern_start_addr_addr_immediate",
         "src_start_addr_addr_var_offset", "dst_start_addr_addr_var_offset", "imm0_imm_ptr")
refs = lambda d: {d[k] for k in ADDRK if d.get(k) is not None and 0x4000 <= d[k] < 0x38000}

dvd = D["DVE0.bin"]; nd = len(dvd)
s6 = 0; idx_of = [None]; cnt_at = []
for i, d in enumerate(dvd):
    if d.get("events_update_mode") == "SEM_INC_COMPLETE" and d.get("events_update_idx") == 6:
        s6 += 1; idx_of.append(i)
    cnt_at.append(s6)

cand = [dict(i=i, v=d["events_semaphore_value"], k=cnt_at[i],
             Y=d["dst_mem_pattern_start_addr_addr_immediate"],
             CS=d["src1_mem_pattern_start_addr_addr_immediate"])
        for i, d in enumerate(dvd)
        if OBJ["DVE0.bin"][i].name == "TENSOR_TENSOR_ARITH_OP" and d.get("op") == "ADD"
        and d["src0_mem_pattern_start_addr_addr_immediate"] == d["dst_mem_pattern_start_addr_addr_immediate"]
        and d["dst_mem_pattern_num_elem"][0] == 512
        and d.get("events_update_mode") == "NONE"            # the ones that are free to move
        and d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6
        and i not in EXCL]
candset = {c["i"] for c in cand}                              # 170 of 204

nextref = {}                                                  # deadline: next DVE ref to Y or Cs
for i, d in enumerate(dvd):
    if i in candset: continue
    for a in refs(d): nextref.setdefault(a, []).append(i)
cross = {}                                                    # ... and cross-engine refs
for p in ("Activation0.bin", "PE0.bin", "SP0.bin"):
    for d in D[p]:
        r = refs(d)
        if r and d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6:
            for a in r: cross.setdefault(a, []).append(idx_of[d["events_semaphore_value"]])
for a in cross: cross[a].sort()
for c in cand:
    dl = nd
    for a in (c["Y"], c["CS"]):
        for L in (nextref.get(a), cross.get(a)):
            if L:
                j = bisect.bisect_right(L, c["i"])
                if j < len(L): dl = min(dl, L[j])
    c["d"] = dl                                               # slack: min 3, median 14, max 79

pl = D["Pool0.bin"]; npl = len(pl)                            # insert each ADD eagerly on Pool
pw = [(i, d["events_semaphore_value"]) for i, d in enumerate(pl)
      if d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6]
vals = [v for _, v in pw]
ins_at = collections.defaultdict(list)
for c in cand:
    kk = bisect.bisect_right(vals, c["v"])
    pos = (pw[kk-1][0] + 1) if kk else 0
    if os.environ.get("SKIPDMA"):                             # refuted, keep off
        while pos > 0 and OBJ["Pool0.bin"][pos-1].name.startswith("PSEUDO_DMA"): pos -= 1
    add = isa.decode(bytes(S["DVE0.bin"][c["i"]*64:(c["i"]+1)*64]))   # the identical 64 bytes
    add.set("events_wait_mode", "WAIT_FOR_SEM_GE_IMM"); add.set("events_wait_idx", 6)
    add.set("events_semaphore_value", c["v"])
    add.set("events_update_mode", "SEM_INC_COMPLETE"); add.set("events_update_idx", 3)
    ins_at[pos].append((c["i"], add.encode()))
new_pool = bytearray(); old_new = []; cnt = 0; addcnt = {}
for p in range(npl + 1):
    for i, b in ins_at.get(p, []):
        new_pool += b; cnt += 1; addcnt[i] = cnt
    if p < npl:
        new_pool += bytes(S["Pool0.bin"][p*64:(p+1)*64])
        if pl[p].get("events_update_mode") == "SEM_INC_COMPLETE" and pl[p].get("events_update_idx") == 3:
            cnt += 1; old_new.append(cnt)
for c in cand: c["c"] = addcnt[c["i"]]
S["Pool0.bin"] = new_pool

# NEW: hand-off wait hosts = freed ADD slots + original ES slots with an unused extended wait
host_es = set()
if os.environ.get("ESHOST"):
    for i, d in enumerate(dvd):
        if i in candset or OBJ["DVE0.bin"][i].name != "EVENT_SEMAPHORE": continue
        if d.get("events_extended_wait_mode") == "WAIT_FOR_SEM_GE_IMM": continue
        host_es.add(i)
islots = sorted(candset | host_es); done = [False]*len(cand)
FRESH = set(json.loads(os.environ.get("FRESH", "[]")))
byid = {c["i"]: j for j, c in enumerate(cand)}; placed = []; fresh = []
for c in sorted(cand, key=lambda c: (c["d"], c["i"])):        # greedy interval stabbing
    if done[byid[c["i"]]]: continue
    if c["i"] in FRESH:
        fresh.append((c["d"], c["c"])); p = c["d"] - 1; v = c["c"]
    else:
        p = islots[bisect.bisect_left(islots, c["d"]) - 1]; v = c["c"]
        assert p >= c["i"]
        placed.append((p, v))
    for j2, c2 in enumerate(cand):                            # value = exactly this ADD, never higher
        if not done[j2] and c2["c"] <= v and c2["i"] <= p < c2["d"]: done[j2] = True
assert all(done)

for name in STREAMS:                                          # remap every S[3] wait
    buf = S[name]
    for i in range(len(buf)//64):
        ins = isa.decode(bytes(buf[i*64:(i+1)*64])); d = ins.as_dict(); dirty = False
        if d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 3:
            ins.set("events_semaphore_value", old_new[d["events_semaphore_value"]-1]); dirty = True
        if d.get("events_extended_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_extended_wait_idx") == 3:
            ins.set("events_extended_sem_wait_value", old_new[d["events_extended_sem_wait_value"]-1]); dirty = True
        if dirty: buf[i*64:(i+1)*64] = ins.encode()

tmpl = next(bytes(S["DVE0.bin"][i*64:(i+1)*64]) for i in range(nd)
            if OBJ["DVE0.bin"][i].name == "EVENT_SEMAPHORE")
pm = collections.defaultdict(list)
for p, v in placed: pm[p].append(v)
for p_, v_ in placed:                                         # NEW: host on an original ES
    if p_ in candset: continue
    ins = isa.decode(bytes(S["DVE0.bin"][p_*64:(p_+1)*64]))
    ins.set("events_extended_wait_mode", "WAIT_FOR_SEM_GE_IMM")
    ins.set("events_extended_wait_idx", 3)
    ins.set("events_extended_sem_wait_value", max(pm[p_]))
    S["DVE0.bin"][p_*64:(p_+1)*64] = ins.encode()
for c in cand:                                                # freed DVE slot -> ES
    es = isa.decode(tmpl)
    es.set("events_wait_mode", "WAIT_FOR_SEM_GE_IMM"); es.set("events_wait_idx", 6)
    es.set("events_semaphore_value", c["v"])                  # keep the ADD's own dependency
    if pm.get(c["i"]) and c["i"] in candset:
        es.set("events_extended_wait_mode", "WAIT_FOR_SEM_GE_IMM")
        es.set("events_extended_wait_idx", 3)
        es.set("events_extended_sem_wait_value", max(pm[c["i"]]))
    else:
        es.set("events_extended_wait_mode", "NONE")
    es.set("events_update_mode", "NONE"); es.set("events_extended_update_mode", "NONE")
    S["DVE0.bin"][c["i"]*64:(c["i"]+1)*64] = es.encode()
if fresh:                                                     # NEW: fresh ES just before d
    byd = collections.defaultdict(list)
    for d_, v_ in fresh: byd[d_].append(v_)
    buf = S["DVE0.bin"]
    for d_ in sorted(byd, reverse=True):
        es = isa.decode(tmpl)
        es.set("events_wait_mode", "WAIT_FOR_SEM_GE_IMM"); es.set("events_wait_idx", 3)
        es.set("events_semaphore_value", max(byd[d_]))
        es.set("events_extended_wait_mode", "NONE")
        es.set("events_update_mode", "NONE"); es.set("events_extended_update_mode", "NONE")
        buf[d_*64:d_*64] = es.encode()
for p in STREAMS: open(SG+p, "wb").write(bytes(S[p]))
```

`FRESH` is the list of the 12 group-boundary ADD slots, i.e. the ones whose hand-off wait would
otherwise land on their own slot (`p == c["i"]`); print them with
`{cand[v]['i'] for p, v in placed if p == cand[v]['i']}` from a `build3.py` run with `ESHOST`
unset. For this NEFF: `[97,197,295,395,495,597,700,796,900,997,1101,1198]`.

### `merge_copies.py` — merge the 1-element Pool carry COPYs into dim-1 strided copies
`MAXRUN=2` gives 77 pairs here (115 COPYs left of 192). Rejects a merge when a Pool DMA trigger
lies between the two, when the later COPY does not update S[3] while an earlier one does, or when
ACT waits on an exact S[3] ordinal being deleted.

```python
import sys, os
sys.path.insert(0, "/home/ubuntu/workspace/neff_doc/tools")
import isa
MAXRUN = int(os.environ.get("MAXRUN", "2"))
SG = "work/payload/sg00/"
STREAMS = ["DVE0.bin", "Pool0.bin", "PE0.bin", "Activation0.bin", "SP0.bin"]
S = {p: bytearray(open(SG+p, "rb").read()) for p in STREAMS}
OBJ = {p: [isa.decode(bytes(S[p][i*64:(i+1)*64])) for i in range(len(S[p])//64)] for p in STREAMS}
D = {p: [o.as_dict() for o in OBJ[p]] for p in STREAMS}
pool, pd_ = OBJ["Pool0.bin"], D["Pool0.bin"]; npl = len(pd_)
ordn = {}; c = 0
for i, d in enumerate(pd_):
    if d.get("events_update_mode") == "SEM_INC_COMPLETE" and d.get("events_update_idx") == 3:
        c += 1; ordn[i] = c
act_wait = set()
for d in D["Activation0.bin"]:
    if d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 3:
        act_wait.add(d["events_semaphore_value"])
    if d.get("events_extended_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_extended_wait_idx") == 3:
        act_wait.add(d["events_extended_sem_wait_value"])
iscopy = [pool[i].name == "COPY" and pd_[i]["src_mem_pattern_num_elem"][0] == 1 for i in range(npl)]
isdma  = [pool[i].name.startswith("PSEUDO_DMA") for i in range(npl)]
SA = lambda i: pd_[i]["src_mem_pattern_start_addr_addr_immediate"]
DA = lambda i: pd_[i]["dst_mem_pattern_start_addr_addr_immediate"]
upd = lambda i: i in ordn
copies = [i for i in range(npl) if iscopy[i]]
runs = []; k = 0
while k < len(copies):
    run = [copies[k]]
    while len(run) < MAXRUN and k + len(run) < len(copies):
        nxt = copies[k + len(run)]; prev = run[-1]
        if any(isdma[j] for j in range(prev+1, nxt)): break
        ds = SA(nxt) - SA(prev); dd = DA(nxt) - DA(prev)
        if len(run) >= 2 and (ds != SA(run[1])-SA(run[0]) or dd != DA(run[1])-DA(run[0])): break
        if ds % 4 or dd % 4 or abs(ds)//4 > 32767 or abs(dd)//4 > 32767: break
        if not upd(nxt) and any(upd(j) for j in run): break
        if any(upd(j) and ordn[j] in act_wait for j in run): break
        run.append(nxt)
    if len(run) > 1: runs.append(run)
    k += len(run)
drop = set(); merged_of = {}
for r in runs:
    last = r[-1]
    ins = isa.decode(bytes(S["Pool0.bin"][last*64:(last+1)*64]))
    ds = (SA(r[1]) - SA(r[0])) // 4; dd = (DA(r[1]) - DA(r[0])) // 4
    ins.set("src_mem_pattern_start_addr_addr_immediate", SA(r[0]))
    ins.set("dst_mem_pattern_start_addr_addr_immediate", DA(r[0]))
    ins.set("src_mem_pattern_num_elem", [1, len(r), 1, 1])
    ins.set("src_mem_pattern_step_elem", [1, ds, 1, 1])
    ins.set("dst_mem_pattern_num_elem", [1, len(r), 1, 1])
    ins.set("dst_mem_pattern_step_elem", [1, dd, 1, 1])
    S["Pool0.bin"][last*64:(last+1)*64] = ins.encode()
    for j in r[:-1]: drop.add(j); merged_of[j] = last
newmap = {}; newpool = bytearray(); nc = 0; pending = []
for i in range(npl):
    if i in drop: pending.append(i); continue
    newpool += bytes(S["Pool0.bin"][i*64:(i+1)*64])
    if i in ordn:
        nc += 1; newmap[ordn[i]] = nc
        for j in pending:
            if j in ordn and merged_of[j] == i: newmap[ordn[j]] = nc
        pending = [p for p in pending if not (p in ordn and merged_of[p] == i)]
for o in range(1, c+1): assert o in newmap
S["Pool0.bin"] = newpool
for name in STREAMS:
    buf = S[name]
    for i in range(len(buf)//64):
        ins = isa.decode(bytes(buf[i*64:(i+1)*64])); d = ins.as_dict(); dirty = False
        if d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 3:
            ins.set("events_semaphore_value", newmap[d["events_semaphore_value"]]); dirty = True
        if d.get("events_extended_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_extended_wait_idx") == 3:
            ins.set("events_extended_sem_wait_value", newmap[d["events_extended_sem_wait_value"]]); dirty = True
        if dirty: buf[i*64:(i+1)*64] = ins.encode()
for p in STREAMS: open(SG+p, "wb").write(bytes(S[p]))
```

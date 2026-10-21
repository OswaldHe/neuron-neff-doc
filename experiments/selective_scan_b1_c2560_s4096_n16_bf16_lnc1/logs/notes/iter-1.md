# iter-1 — selective_scan b1 c2560 s4096 n16 bf16

**Result: 7919.6 -> 6886.3 us (-13.05%, 1.150x), numerics bit-identical to the baseline**
(max_abs_err stays 1.250e-01 / 83% of limit, cosine 0.999944, output_1 1.812e-04 — exactly the
baseline figures). 5 repeats, spread 0.02% (one capture in ~15 came out 267 us high — a host hiccup, not
reproducible; the metric is the median of 5). `verify.py --runs 3` gives bit-identical outputs
across runs (no race). `semaphores.py` clean, `launch_rules.py` clean.

## The edit

Move **2214 of the 2720 `y_accum += Cs` fp32 512-element `TENSOR_TENSOR_ARITH_OP` ADDs
(NKI line ~275) from DVE to the Pool/GpSimd engine**, with slack-aware semaphore hand-off.
DVE was 99.3% busy and is the only thing that sets the latency; Pool was 15%.

Changed members: `sg00/DVE0.bin` (2214 ADD slots -> `EVENT_SEMAPHORE`, 2227 S[3] wait values
remapped), `sg00/Pool0.bin` (3842 -> 6056 slots), `sg00/Activation0.bin` (252 S[3] wait values
remapped). The build script is at the end of this file.

## The single most important structural fact found

An instruction can only be moved off DVE **for free if it does not increment S[6]**.

* 2214 of the y-ADDs increment nothing (`events_update_mode == NONE`) — pure relocation, no
  semaphore count anywhere changes. These are the ones that pay off.
* For an instruction that *does* increment S[6], the replacement slot has to inherit the
  increment, and then `S6 >= k` must keep meaning "every DVE instruction before slot i has
  completed". An `EVENT_SEMAPHORE` retires out of order ahead of still-running compute
  instructions, so without a guard PE is released early and overwrites a PSUM bank a live DVE
  MULT is reading:
  `INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR / TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION`,
  nrt status 1004, fails in ~9 s. **The fix is to give the replacement ES a self-wait
  `S[6] >= k-1`** (k = the slot's own S[6] ordinal). Verified: with that one field all 352
  S[6]-incrementing y-ADDs move and the output is bit-exact.
* **But the guard costs ~585 ns of DVE pipelining per slot** (measured: moving those 352 removed
  240 us of ADD duration yet DVE busy went *up* 6607 -> 6679 us and the span 6889 -> 7023 us).
  585 ns > the 548 ns marginal saving of removing a TT, so **every S[6]-incrementing instruction
  is a net loss to move.** That closes the remaining 352 ADDs, the 152 Y-zeroing MEMSETs
  (485 ns each) and the 160 line-182 `dt*x` bf16 MULTs (682 ns each) in one shot — all of them
  increment S[6]. Relocating the increment onto the following slot does not help: for 345 of the
  352 the next non-ES slot already increments S[6] itself, and only 1 of 352 has its producer as
  the immediately preceding producer (`k-1 == v`), so the guard is almost never free.

## The second most important fact: where the DVE wait goes

Same Pool stream, three wait placements, measured:

| DVE wait placement | total_exec_time_us |
|---|---|
| at the ADD's own slot, value = that ADD (lock-step) | 8411 (+6.2%, **worse than baseline**) |
| latest legal slot, value = highest ADD issued so far (over-synchronised) | 7377 (-6.9%) |
| latest legal slot, value = *only* the ADD being covered | **6886 (-13.05%)** |

The move is worth ~1.25 ms of DVE time; a naive hand-off throws 1.5 ms of it away in round trips.
Rules that produce the third row:

* **Deadline** `d` for each moved ADD = the first DVE slot after it that references its `Y` or
  `Cs` buffer (other moved ADDs excluded — Pool program order covers them), *and* the first
  ACT/PE/SP slot referencing those buffers, mapped back to the DVE slot producing the S[6] count
  it waits for. Median slack 13 DVE slots, min 3.
* **Greedy interval stabbing**: by increasing deadline, for the first uncovered ADD place one wait
  at the **latest** freed slot `< d` with value = **exactly that ADD's** S[3] ordinal; then mark
  covered every ADD whose ordinal is `<=` that value and whose `[i, d)` contains the slot.
  2174 waits for 2214 ADDs.
* Waiting for a *higher* S[3] value than needed is what cost 500 us — DVE then blocks on Pool
  ADDs it does not depend on yet.

Buffer facts that make the deadlines real: `Cs` rotates over 3 buffers (so the binding constraint
is the line-267 MULT three states later) and `Y`/`Cs`/`dA`/`state` all live in **one shared SBUF
scratch pool** — ACT writes `dA` into the very addresses that are `Y` buffers elsewhere in the
program, so cross-engine WAR has to be in the deadline, not just DVE program order.

## Third fact: Pool insertion position trades Pool efficiency against hand-off latency

Insert each ADD **eagerly** (right after the last Pool slot waiting `S6 <= v`, v = its producer's
count, which puts it right after that state's carry `COPY`) vs **late** (keyed on the DVE S[6]
count already reached at the ADD's own slot, one `COPY` later):

| Pool insert | total us | DVE busy | Pool busy |
|---|---|---|---|
| eager (`key = v_j`) | **6886** | 6607 | 5580 |
| late (`key = cnt_at[i_j]`) | 7424 | 7073 | 3822 |

The late variant makes **Pool much more efficient** (its 1-element carry `COPY` goes 688 -> ~300 ns
and its TT 1551 -> ~1271 ns, 1.76 ms less Pool time) but DVE then waits 470 us more. Eager wins
because DVE is the critical engine and Pool has 1.3 ms idle to burn. Note for later: Pool
instructions get **~2x slower when Pool is loaded and alternating 1-element COPYs with
512-element TTs** — this is why Pool's effective TT cost is ~1550-1975 ns rather than the 1271 ns
measured for a single inserted ADD.

## Measured engine numbers

| | baseline | iter-1 |
|---|---|---|
| total_exec_time | 7919.6 | 6886.3 |
| DVE / Vector busy | 7860 (99.3%) | 6607 (95.9%) |
| Pool / GpSimd busy | 1214 (15.4%) | 5582 (81.0%) |
| PE / ACT / DMA / SP | 2971 / 1902 / 1486 / 609 | 2973 / 1903 / 1447 / 609 |

DVE union-of-intervals busy on the baseline is **7898 us of a 7928 us span — only 29.7 us idle**
over 409 gaps. (The iter-0 review's "2.27 ms of DVE gaps" is wrong: it comes from summing
per-instruction durations, 8965 us, which overlap by ~12%.) On iter-1 DVE is 6653 busy in a
6896 span, 243 us idle over 874 gaps. **DVE is saturated both before and after; only removing
DVE work helps.**

Calibration, all measured on this kernel:
* DVE: TT fp32 512x128 = 663-683 ns; `TENSOR_TENSOR_SCAN` = 1275-1291 ns; MEMSET 512 fp32 =
  485 ns; `EVENT_SEMAPHORE` = 35-39 ns; `DRAIN` = 91 ns.
* Pool: 1-element `COPY` = 299 ns idle / 688 ns loaded; DGE DMA trigger = 655 ns;
  **TT fp32 512x128 = 1271 ns isolated, 1551 ns loaded** (1.86x-2.3x DVE).
* **Marginal DVE saving per removed TT = 548 ns** = 0.80 of its own duration (the rest is
  pipelining that the neighbours reclaim).
* Model that fits every measurement: `DVE_span = 0.881*(8964 - m)`, `Pool_span = 1229 + 2.89*m`,
  m = us of DVE-duration moved. We sit at m = 1512 us. The optimum (m = 1768) is unreachable
  because the only remaining movable work increments S[6].

## Closed by measurement — do not retry

1. **Activation/Scalar cannot run `TENSOR_TENSOR_ARITH_OP` (0x41).** Probe: one ADD moved to ACT
   with correct sync. It runs with **exit 0** but the term is silently dropped — max_abs_err
   1.250e-01 -> 1.271e+01, cosine 0.999838, 99.73% of elements still in tolerance (the exact
   signature of "one chunk's one state missing"). The corpus already said 0x41 appears only on
   DVE and Pool; this confirms ACT ignores it rather than faulting. **Pool is the only off-load
   target for tensor-tensor**, so ACT's 72% idle and SP's 91% idle are unreachable here.
2. **Moving any S[6]-incrementing instruction is a net loss** — see "most important structural
   fact" above. Mechanism understood, fix known (`S6 >= k-1` self-wait), cost measured (585 ns),
   verdict negative. This closes the other 352 y-ADDs, the 152 MEMSETs and the 160 `dt*x` MULTs.
3. **Including the 146 ADDs that wait on S[4] (ACT) in the moved set hangs** (70 s NRT timeout).
   Only 5 of them are also non-S[6]-incrementing, so the whole group is worth <4 us — not worth
   debugging. The hang is probably my Pool insertion key for them (`cnt_at[i]` rather than a real
   S[6] producer) closing a cycle.
4. **Pool reading PSUM is still unknown.** The probe (one line-267 `Cs = C_full*state`, src0 =
   PSUM `0x2000800`, moved to Pool) hung at 70 s, but the hang is fully explained by my own sync:
   I put a `W5>=3` (PE) wait on a Pool `EVENT_SEMAPHORE` placed *before* Pool slots 9-11, which
   are unwaited `PSEUDO_DMA_DIRECT2D` input loads that PE needs -> circular wait.
   `semaphores.py` cannot see this class of cycle. **If retried**, use DVE slot 65 as the
   subject (it is a line-267 MULT whose only wait is `W6>=31`, so the PE dependency is already
   satisfied and no PE wait has to travel to Pool). Low priority: both MULT lines increment
   S[6], so even a positive answer runs into (2).

## DVE fixed per-instruction overhead (measured here, new)

Probe: one `TENSOR_TENSOR` and one `TENSOR_TENSOR_SCAN` rewritten to `num_elem=[64,1]` on all
three operands, then read their `duration_ns` out of `prof/Instruction.parquet` (the output is
wrong, which does not matter for a timing probe).

| opcode | 64 elem | 512 elem | ns/element | **fixed overhead** |
|---|---|---|---|---|
| `TENSOR_TENSOR_ARITH_OP` | 222 ns | 678 ns | 1.018 | **157 ns** (23% of a 512-wide TT) |
| `TENSOR_TENSOR_SCAN_ARITH` | 352 ns | 1273 ns | 2.056 | **220 ns** (17% of a 512-wide scan) |

So "fewer, larger instructions" (playbook 14.3 #4) is worth ~157 ns per merged pair on DVE —
real, but smaller than the second-order costs it incurs here (next section).

**The scan does NOT re-seed its carry across access-pattern dim 1.** Falsifier run: one scan
rewritten from `num_elem=[512,1]` to `[256,2]`, `step_elem=[1,256]` — the same 512 contiguous
elements as two rows. Output stayed **bit-exact** (max_abs_err 1.250e-01, cosine 0.999944), so
the instruction is one linear scan over the whole pattern regardless of row structure.
**Consequence: two different states can never be merged into one scan instruction** — the carry
would chain across them. That kills the biggest-looking idea on the list; the 3.3 ms scan (50% of DVE)
is immovable and unmergeable, which is the hard floor on this kernel.

## Pairwise MULT merging — BUILT AND REFUTED (both directions)

The merge is mechanically sound and was implemented end to end; it does not pay. Keep the facts,
skip the lever.

**It works.** Two consecutive 512-element MULTs become one 2-row instruction because all operands
of consecutive states sit at a constant stride:

| operand | state s | stride to s+1 | 2-D pattern used |
|---|---|---|---|
| line 224 src0 `dtx` | 16704 | 0 | `num_elem=[512,2,1] step_elem=[1,0,1]` (same data twice) |
| line 224 src1 `B_full` (PSUM) | `0x2000000 + b*0x1000` | 0x1000 B | `step_elem=[1,1024,1]` |
| line 224 dst `dtB` | 24896 / 37184 / 49472 | 12288 B | `step_elem=[1,3072,1]` |
| line 267 src0 `C_full` (PSUM) | `0x2000800 + b*0x1000` | 0x1000 B | `step_elem=[1,1024,1]` |
| line 267 src1 `state` | 26944 / 39232 / 51520 | 12288 B | `step_elem=[1,3072,1]` |
| line 267 dst `Cs` | 31040 / 43328 / 55616 | 12288 B | `step_elem=[1,3072,1]` |

PSUM banks of consecutive states are consecutive; the SBUF families rotate over 3 buffers at a
uniform +12288 B, so a pair merges when the second buffer is the higher one — **635 of 1280 pairs
for line 224, 761 for line 267** (the rest would need a negative outer step). Deleting the dropped
slot's S[6] increment and renumbering all **19924** S[6] waits in the five streams works first
time: a 635-pair line-224 build is **bit-exact** (max_abs_err 1.250e-01, cosine 0.999944).
The merged instruction's measured duration is **1190 ns**, exactly `157 + 1024*1.018` — so the
arithmetic saving is real, 136 ns per pair.

**Renumbering recipe (verified):** build the ordered list of surviving S[6] updaters; map every
old count to the new ordinal of its own instruction if it survived, or to the new ordinal of the
instruction it merged into if it was dropped. One rule, and it also fixes PE's PSUM-bank WAR wait
(which named the removed read) and the Pool ADD's producer wait. Apply to
`events_semaphore_value` *and* `events_extended_sem_wait_value` in all five streams. Leave the
dropped slot as an `EVENT_SEMAPHORE` carrying its **original wait** and no update — 35 ns, and it
needs no `S6>=k-1` guard precisely because it no longer increments S[6].

**Why it still loses.** The two lines force opposite merge directions, and each direction breaks
something:

```
27 MULT224 ->dtB[37184]   30 MULT267 ->Cs[31040]   33 MULT224 ->dtB[49472]   36 MULT267 ->Cs[43328]
28 (ADD, moved to Pool)   31 ES                    34 (ADD, moved to Pool)
29 ES                     32 SCAN reads dtB[37184], writes state[39232]
```

* **line 224 must merge at the earlier slot** (the scan at 32 reads slot 27's `dtB` before 33
  runs). That instruction then has to carry slot 33's PE wait, i.e. it needs `B_full` for state
  s+1 one state sooner. **Measured: 7919 -> 7100 us, i.e. 214 us WORSE than the 6886 baseline**
  (DVE busy 6607 -> 6740) even though TT duration fell 90 us. PE is not idle in the way its 43%
  suggests: PE's matmuls are paced by PSUM bank availability, which DVE controls, so DVE and PE
  run in lockstep and pulling the PE dependency one matmul earlier costs more than the merge
  saves. Verified bit-exact, just slower.
* **line 267 must merge at the later slot** (its second `state` operand is written by the scan at
  32, after slot 30). That pushes the `Cs[s]` write later, so the Pool ADD for state s — whose
  producer wait is re-pointed to the merged instruction — now needs a DVE slot that comes *after*
  the DVE slot where iter-1 placed the hand-off wait for that same ADD. **Measured: NRT_TIMEOUT
  (deadlock) at 71 s.** To make it work the ADD move and the merge would have to be planned
  together from the baseline (merge first, then compute the hand-off deadlines), and the merge
  also eats one state of the ADD's slack. Not attempted — given line 224's result the expected
  value is at best small.

## What is left, and what to try next (ranked)

DVE busy 6607 us. Composition of DVE *duration* (sums, pre-overlap) from
`prof/Instruction.parquet` grouped by `opcode` + `nki_source_location`:

| what | n | mean ns | total us | status |
|---|---|---|---|---|
| `TENSOR_TENSOR_SCAN` (line 233) | 2560 | 1291 | 3306 | **DVE-only, and unmergeable** (carry does not reset per row) |
| TT MULT `dtx*B_full` (line 224) | 2560 | 663 | 1698 | cannot move (S[6]); merging measured 214 us worse |
| TT MULT `C_full*state` (line 267) | 2560 | 683 | 1749 | cannot move (S[6]); merging deadlocks against the hand-off |
| TT ADD `y+=Cs` (line 275) | 2560 | 683 | 1749 | **2214 moved**; the other 346 closed (2) |
| TT ADD (line ~297) | 160 | 681 | 109 | increments S[6] -> closed (2) |
| TT MULT `dt*x` bf16 (line ~182) | 160 | 682 | 109 | increments S[6] -> closed (2) |
| MEMSET | 200 | 412 | 82 | 152 increment S[6] -> closed (2) |
| `EVENT_SEMAPHORE` | 7390 | 35 | 257 | 40 of the new ones carry no wait; deletable, ~1.4 us |

1. **Nothing cheap is left.** The two engine-level levers (move work off DVE, merge DVE
   instructions) are both at their measured optimum. The remaining structural ideas all need
   SBUF *re-addressing* — reallocating the 3-deep rotating buffer families so that (a) the `Cs`
   rotation is 4-deep (more hand-off slack, up to the 243 us of DVE idle) or (b) consecutive
   states' operands are contiguous, which would let line 267 merge at the earlier slot and avoid
   the PE coupling that killed line 224. That means re-pointing the ACT `dA` writes, the
   line-224 `dtB` writes and the scan operands together — a large, coherent edit, and the only
   thing left with more than ~2% in it.
2. **Shave the 243 us (3.5%) of DVE hand-off idle.** 874 gaps averaging 278 ns, 229 of them in
   front of a `TENSOR_TENSOR` and 166 in front of an `EVENT_SEMAPHORE`. The binding deadline is
   the 3-deep `Cs` buffer rotation. More slack needs a 4th `Cs` buffer, i.e. re-addressing — same
   blocker as 1. Alternatively reduce Pool's latency: moving Pool's 2560 1-element carry COPYs to
   ACT (`COPY` 0x46 is listed for DVE/ACT/Pool, but **test it first** — see closed item 1, ACT
   silently ignores opcodes it cannot do) would take 1.76 ms off Pool and make the ADDs land
   sooner. Needs renumbering of both S[3] (the scans' WAR waits consume these increments) and
   S[4].
3. **Delete the 40 no-wait `EVENT_SEMAPHORE` slots** left over at freed ADD positions: ~1.4 us.
   Noise; mentioned only for completeness. (Deleting DVE slots is safe PC-wise — see gotchas.)

## Gotchas for whoever edits next

* `semaphores.py` catches unsatisfiable waits only. It did **not** catch either cross-engine
  cycle, nor the PSUM collision, nor the over-synchronisation. Budget one device run per
  structural change; they are 10 s when they work, 9 s when they PSUM-fault, 70 s when they hang.
* **`EVENT_SEMAPHORE` (`CTRL_ES`) carries a second wait**: `events_extended_wait_mode` /
  `events_extended_wait_idx` / `events_extended_sem_wait_value` (the compiler uses it, e.g. Pool
  slot 3839). Compute instructions have only one wait — that is why the baseline is full of bare
  `EVENT_SEMAPHORE` barriers: they are the compiler's way of expressing a second dependency. The
  accepted edit uses the primary wait for the ADD's original dependency and the extended wait for
  the Pool hand-off.
* Neither the DVE nor the Pool stream contains any PC-relative field (no `COMPARE_BRANCH` in the
  NEFF streams; the 2 per engine the profiler reports are runtime prologue/epilogue). **Inserting
  or deleting slots is safe**; `Pool0.json` only names the `.bin` and nothing records a slot
  count, so `neff.py pack` handles the size change.
* `SP0.json`'s 4949 DMA descriptors only *signal* semaphores, never wait — nothing to remap there.
  Queued DMA writes land in SBUF >= 113 k (inputs); the scratch pool is 16384..58176, so they do
  not alias.
* `isa.py`: the wait value field is `events_semaphore_value`, not `events_wait_value`; read fields
  with `ins["name"]` (`__getitem__`), not `.get()`; `as_dict()` hides zero-valued fields.
* Baseline error is already 83% of the limit, so **every edit must be bit-exact**. Everything
  accepted here is a pure instruction *relocation*: the identical 64 bytes with only the events
  block rewritten.

## Build script (reproduces the accepted NEFF from `baseline/kernel.neff`)

Run from the case dir after `neff.py unpack kernel.neff work/`, then `neff.py pack work/ kernel.neff`.

```python
import sys, collections, bisect
sys.path.insert(0, "/home/ubuntu/workspace/neff_doc/tools")
import isa
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
        and d.get("events_update_mode") == "NONE"                      # <- the free ones
        and d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6]
candset = {c["i"] for c in cand}                                       # 2214

nextref = {}
for i, d in enumerate(dvd):
    if i in candset: continue
    for a in refs(d): nextref.setdefault(a, []).append(i)
cross = {}
for p in ("Activation0.bin", "PE0.bin", "SP0.bin"):
    for d in D[p]:
        r = refs(d)
        if r and d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6:
            for a in r: cross.setdefault(a, []).append(idx_of[d["events_semaphore_value"]])
for a in cross: cross[a].sort()
for c in cand:                                                          # deadline
    dl = nd
    for a in (c["Y"], c["CS"]):
        for L in (nextref.get(a), cross.get(a)):
            if L:
                j = bisect.bisect_right(L, c["i"])
                if j < len(L): dl = min(dl, L[j])
    c["d"] = dl

pl = D["Pool0.bin"]; npl = len(pl)
pw = [(i, d["events_semaphore_value"]) for i, d in enumerate(pl)
      if d.get("events_wait_mode") == "WAIT_FOR_SEM_GE_IMM" and d.get("events_wait_idx") == 6]
vals = [v for _, v in pw]
ins_at = collections.defaultdict(list)
for c in cand:                                                          # eager Pool position
    kk = bisect.bisect_right(vals, c["v"])
    pos = (pw[kk-1][0] + 1) if kk else 0
    add = isa.decode(bytes(S["DVE0.bin"][c["i"]*64:(c["i"]+1)*64]))      # identical 64 bytes
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

islots = [c["i"] for c in cand]; done = [False]*len(cand)
byid = {c["i"]: j for j, c in enumerate(cand)}; placed = []
for c in sorted(cand, key=lambda c: (c["d"], c["i"])):                  # greedy stabbing
    if done[byid[c["i"]]]: continue
    p = islots[bisect.bisect_left(islots, c["d"]) - 1]; v = c["c"]       # latest slot, minimal value
    placed.append((p, v))
    for j2, c2 in enumerate(cand):
        if not done[j2] and c2["c"] <= v and c2["i"] <= p < c2["d"]: done[j2] = True
assert all(done)

for name in STREAMS:                                                    # remap every S[3] wait
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
for c in cand:                                                          # DVE slot -> ES
    es = isa.decode(tmpl)
    es.set("events_wait_mode", "WAIT_FOR_SEM_GE_IMM"); es.set("events_wait_idx", 6)
    es.set("events_semaphore_value", c["v"])                            # keep original dependency
    if pm.get(c["i"]):                                                  # Pool hand-off, extended wait
        es.set("events_extended_wait_mode", "WAIT_FOR_SEM_GE_IMM")
        es.set("events_extended_wait_idx", 3)
        es.set("events_extended_sem_wait_value", max(pm[c["i"]]))
    else:
        es.set("events_extended_wait_mode", "NONE")
    es.set("events_update_mode", "NONE"); es.set("events_extended_update_mode", "NONE")
    S["DVE0.bin"][c["i"]*64:(c["i"]+1)*64] = es.encode()
for p in STREAMS: open(SG+p, "wb").write(bytes(S[p]))
```

This script is exactly what built the accepted `kernel.neff` (sha256 of the packed file is
printed by `verify.py`). A variant that puts the S[3] hand-off in the ES *primary* wait and leaves
the extended wait unused — i.e. dropping the ADD's original `S6>=v` dependency — also verifies and
measures 6888.5 us, 2.7 us slower; keeping the original dependency is both faster and safer, so
that is what is shipped.

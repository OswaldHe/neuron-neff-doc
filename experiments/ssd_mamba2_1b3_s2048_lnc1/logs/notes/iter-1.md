# iter-1 (ssd_mamba2_1b3_s2048): delete the ACT activation-table reloads

**Result: 7154.9 -> 5129.9 us (-28.3%, 1.395x). `verify.py` PASS three times, identical numbers
(max_abs_err 5.234e-07 vs baseline 5.122e-07, limit 6.2e-07). Only `sg00/Activation0.bin` changed.**

## 1. Where the time was (measured at iter-1 start)

| ACT opcode | n | total | mean |
|---|---|---|---|
| `ACTIVATE` | 10240 | 3841 us | 375 ns |
| `ACT_TABLE_LOAD` | 2046 | **2625 us** | 1283 ns |
| `DMA_DIRECT2D` (DGE triggers) | 1088 | 609 us | 560 ns |
| `EVENT_SEMAPHORE` | 1930 | 145 us | **75 ns** |

ACT busy 6284 us of 7155 us wall (87.8%). The table reloads alone were 37% of the whole runtime.
Per (head, chunk) unit the ACT stream is exactly:

```
prod : ACTIVATE f=7 (exp)        src = PSUM bank -> A      scale +1, nch=128, num_elem [1,1,1]
 ... : ACTIVATE f=7 (exp)        exp(cs_last), unrelated
L1   : ACTIVATION_TABLE_LOAD sel=1                          <- 1283 ns
R    : ACTIVATE f=129 (recip)    src = A         -> C
L0   : ACTIVATION_TABLE_LOAD sel=0                          <- 1283 ns
```
`natural_log_exp_and_others` (sel 0) holds exp(7)/ln(10)/identity(1)/copy(128);
`reciprocal_and_small` (sel 1) holds reciprocal(129) but **not** exp. No stock set has both
(checked `docs/generated/activation_functions.md`), so the reload pair per unit is unavoidable as
long as 1/x is computed on ACT. `prod` reads one of only 8 PSUM banks, reuse distance ~50 slots.

## 2. REFUTED, with the measurement: read the PSUM twice

The review's Candidate 1 as written -- rewrite `R` as `exp(-1 * cs_col)` reading **the same PSUM
address `prod` reads**, NOP both loads, all 1024 units -- fails on the device immediately:

```
INFER_ERROR_SUBTYPE_SW_PSUM_COLLISION_ERROR / TRAINIUM_NC_ERROR_TYPE_PSUM_COLLISION
nrt_execute_repeat ... status 1004
```

So **a second ACT read of a PSUM bank, even 2 slots after the first and with ~50 slots of bank
reuse distance, is rejected by the hardware.** PE's matmuls into those banks carry no waits at
all, so nothing in the stream protects a re-read. Do not add PSUM reads. (The error is instant
and unambiguous, which makes it a cheap falsifier.)

## 3. WHAT WORKED: one PSUM read, three set-0 ACTIVATEs, no table switch

Three slots are rewritten in place and `L0` is deleted. `A` = `prod`'s dst, `C` = `R`'s dst,
`v` = the sem-4 count after `prod`. Two layouts, depending on which buffer may hold the temp:

**variant 1 (temp in A)** -- used for 951 units:

| slot | after |
|---|---|
| `prod` | `f=128 copy`, scale +1, PSUM -> A      (A = cs_col) |
| `L1` | `f=7 exp`, scale **-1**, A -> C, wait `sem4 >= v`, no update |
| `R` | `f=7 exp`, scale +1, A -> **A** (in place), events untouched |
| `L0` | deleted |

**variant 2 (temp in C)** -- used for 69 more units where a reader of A sits too close after `R`:

| slot | after |
|---|---|
| `prod` | `f=128 copy`, scale +1, PSUM -> **C**   (C = cs_col) |
| `L1` | `f=7 exp`, scale +1, **C** -> A, wait `sem4 >= v`, no update |
| `R` | `f=7 exp`, scale **-1**, **C** -> C (in place), events untouched |
| `L0` | deleted |

1020 of 1024 units converted (the other 4 are structural, see 3c). 2040 of 2046 table loads gone.

### 3a. Why the synchronisation is sound
`prod` keeps its update, `R` keeps its wait and its update, `L1` only *gains* a wait. The script
asserts the whole cumulative sem-4 increment sequence is byte-for-byte identical to the baseline
and that no sem-4 wait exceeds the increments available at its slot. `semaphores.py` output is
unchanged. Within-ACT RAW *does* need a semaphore (ch 10.5), which is why `L1` gets
`wait sem4 >= v` -- `v` is always reachable because `prod < L1`.

### 3b. The real constraint is the stale window, and it has to be measured
Between `prod` and the end of the unit one buffer holds `cs_col` instead of its final value.
Everything hinges on nobody reading it then. What was checked, over **all five engines**, for
every reader field (`src*_start_addr`, plus `scale/bias/imm_imm_ptr` only when the matching
`*_src` is `POINTER_IMMEDIATE`), restricted to the window between `prod` and the next ACT write
of that address:

* **ACT readers**: by program order. The only reader of A between `prod` and the next write of A
  is `R` itself -- which in variant 1 is exactly the instruction that now wants `cs_col`. That is
  what makes variant 1 work at all.
* **cross-engine readers**: accepted if they wait `sem4 >= cum[R+1]`, otherwise checked against
  `prof/Instruction.parquet` start/end timestamps and required to start >= 1500 ns after `R` ends.
* variant 2 additionally requires no reader *or writer* of C overlapping `[prod_start, R_end]`.

SBUF addresses are reused all over the program, so a global address -> reader map gives hundreds
of false positives; the window restriction (program order on ACT, profile timestamps across
engines) is what makes the analysis usable.

**A bug worth remembering:** variant 2 first shipped with the two exps still reading `A` (I built
the new slots from `R`'s bytes and only changed the *destination*). It ran fine and gave
`max_abs_err 2.4e-01`. When moving a temp to a different buffer, patch bytes **16:20 (src addr)**
as well as 48:52 (dst addr) -- both addr8 fields have the same layout, so `nb[16:20]=rb[48:52]`
re-points the source.

### 3c. The table-set state machine
Only convert a reciprocal whose *neighbouring table-load events* are `L1` before and `L0` after.
There are 1023 `L1` and 1023 `L0` for 1024 reciprocals, so one reciprocal shares a neighbour's
load; converting it leaves a `f=129` running with set 0 active. A pass that re-walks the stream
and asserts every `activation_func` is in the active set catches this -- it fired on the first
attempt. Keep that assertion.

### 3d. Accuracy
`A = exp(cs_col)` now goes through a `copy_1p` into SBUF first, so the exp argument is the same
fp32 value; `C = exp_table(-cs_col)` replaces `recip_table(exp_table(cs_col))` (one table
evaluation instead of two, so slightly *better*). `max_abs_err` 5.122e-07 -> 5.234e-07
(83% -> 84% of the 6.2e-07 limit). Headroom is ~0.97e-07: another numerics change needs
re-verification, but this one did not eat the budget.

### 3e. Deleting slots is safe on this NEFF
No stream in `sg00` has any `pc` / `branch_pc` / label field (checked every decoded field of all
5 engines), `def.json` carries no instruction counts, and the profiler's `compiler_pc` matches the
static slot index 1:1 (no pseudo-expansion shift). Deleting the 1020 dead `L0` slots outright
(rather than NOP-ing them) ran and verified; it was worth 13 us on its own
(5280.7 -> 5267.6 us) -- an all-zero `EVENT_SEMAPHORE` costs only ~12-64 ns.

## 4. PARTLY OPEN, and the most valuable thing here: moving ACT's DGE DMA triggers off ACT

ACT's 1088 `PSEUDO_DMA_DIRECT2D` are 606 us (557 ns each) -- 16% of ACT's remaining busy time and
the largest non-`ACTIVATE` block. They are the 35 MB of SBUF spill *saves*: source = 128
partitions x 256 B at a `var_id` 0 (`SB`) byte offset, destination = the 2 MB HBM tmp-buf
`inst__I-10-0:_mem_0_sg0000`. SP is only 15.6% busy and already issues 1088 identical DGE
transfers at 543 ns, so the work fits.

### 4a. The mechanic works
Delete the slot from `Activation0.bin`, insert the same 64 bytes into `SP0.bin`. An SP-issued DGE
DMA **does** signal the semaphore named in its own instruction field, including one belonging to
`qActDynamicHW` (semaphores 40-55, `owner: act`), and shifting all of SP's later DGE queue ids is
harmless. Semaphores 40-55 are targeted by **exactly 68 ACT transfers each, `sem_increment` 16**,
and by nothing else -- so "transfer c on semaphore s has completed" is exactly `sem s >= c*16`.
Slot insertion into another engine's stream is now proven on this NEFF.

### 4b. The two handshakes these triggers need
Decoding their events block is the key fact: **their wait is `sem6 >= 2`** -- a count satisfied
almost at the start of the program. Their events block carries essentially no ordering. Both
directions of the ordering they rely on come from their *position in the ACT instruction stream*:
upstream (the SBUF source already holds the spilled data, produced earlier by ACT) and downstream
(ACT must not refill that SBUF region until the DGE has read it). I confirmed the first half by
placing each trigger at the earliest point its own wait allows: all 256 collapsed onto 2 insertion
points at the very start of SP's stream.

So the move was done with both handshakes restored:
1. the moved trigger on SP waits `sem4 >= v`, `v` = ACT's cumulative sem-4 count at the trigger's
   original ACT slot;
2. the ACT slot is replaced in place by an `EVENT_SEMAPHORE` waiting `sem s >= c*16` (that
   transfer's own completion), so ACT's slot count and all sem-4 counts are untouched.

No cycle: `v` counts increments strictly *before* ACT's placeholder, so ACT reaching the
placeholder already implies `sem4 >= v`, which releases SP, which completes the DMA, which
releases ACT. `semaphores.py` clean every time. Insertion point on SP = the first slot whose
profile `start_ts >= ` the trigger's own baseline `start_ts`.

### 4c. The result: it is not a count limit, it is *specific triggers*
Every run below either `VERIFIED` with numbers identical to the current kernel, or hung with
`NRT_TIMEOUT` (status 5). "hs" = with the two handshakes of 4b; the others are the bare move.

| what was moved | n | result |
|---|---|---|
| trigger 0 only | 1 | **pass** |
| triggers 0..31 | 32 | **pass** |
| triggers 0..63 | 64 | **pass** |
| triggers 0..95, hs | 96 | **pass** |
| triggers 0..127 | 128 | hang |
| triggers 0..127, hs | 128 | hang |
| triggers 0..255 | 256 | hang |
| triggers 0..255, hs | 256 | hang |
| every 4th trigger (spread over the program) | 272 | hang |
| all, inserted at `start_ts + 20 us` | 1088 | hang |
| all, to **Pool** instead of SP | 1088 | hang |
| all, hs | 1088 | hang |
| **swap**: 256 ACT->SP *and* 256 SP->ACT (both engines keep 1088 DGEs, both stream lengths unchanged) | 256 | hang |
| **window 96:128 alone, hs** | **32** | **hang** |
| **window 544:608 alone, hs** | **64** | **pass** |
| **window 1024:1088 alone, hs** | **64** | **hang** |
| **window 1056:1088 alone, hs** | **32** | **pass** |
| triggers 128..1087, hs | 960 | hang |
| triggers 544..1087, hs | 544 | hang |

The last block is the point. A 32-trigger window at 96 hangs while a 64-trigger window at 544
passes and the final 32 pass -- so **there is no count or capacity limit; a minority of specific
triggers cannot be moved and the rest can.** That also explains why the handshakes, the swap, the
Pool target and the later insertion all made no difference: they were all fighting the wrong
thing. `[1024:1088]` hanging while `[1056:1088]` passes localises one bad group to
`[1024:1056)`.

### 4d. How to finish this in iter-2 (this is the open 400-500 us)
1. Scan windows of 32 over all 34 of them with `handshake2.py <lo>:<hi>` + `verify.py`, keeping a
   pass/fail list. Each pass costs ~25 s, each hang ~71 s, so budget ~25 min. **Run it in the
   foreground in 3 chunks and print each result as it comes** -- I lost a first scan by piping it
   through `tail` (output buffered) into a background task that hit a 10-minute limit.
2. Move the union of the passing windows, re-verify, profile. Expect roughly
   `n_moved x 480 ns` off ACT busy and ~0.8x of that off the wall (the ratio this iteration's main
   edit showed), so ~350-450 us (7-9%) if most windows pass.
3. The union is not guaranteed from the parts -- if it hangs, bisect the union, and if a window
   that passed alone fails in company, the cause is interaction and the per-trigger reason is
   worth digging out: compare the bad windows' triggers against the good ones' on
   `src_start_addr_addr_var_offset`, `semaphore`, and which SP slot they land next to.
4. Cheap extra: the ACT placeholder in 4b could be dropped entirely for triggers whose source SBUF
   region is never rewritten later by ACT (that needs the ch 5 address mapping, see below), saving
   another ~75 ns each and removing a stall.

Address mapping needed for step 4, **inferred and not verified**: an instruction address looks
like `(start_partition << 20) | byte_offset` -- every ACT address here fits it (0x4210 -> p0/0x4210,
0x809000 -> p8/0x9000, 0x181c300 -> p24/0x1c300) and per-partition SBUF is 0x38000
(`runtime_statebuffer_reservation` offset 229368), while the DMA's own `src_step_elem[0]` is
0x40000. Check it against ch 5 before relying on it.

## 5. REFUTED: merging adjacent ACT copies
`ACTIVATE` cost on ACT fits `dur ~= 158 ns + 4.3 ns * num_elem` (f=128/nch=128: 162 ns at
num_elem 1, 434 ns at 64), so merging two contiguous 64-element copies saves only ~95 ns. And
there are almost none to merge: of 7168 `f=128` copies only **319** adjacent pairs are
mergeable. The blocker is `scale/bias/imm` (2798 pairs): every copy has its own per-partition
scale **pointer**, and one instruction can only use one. Total ceiling ~30 us. Closed.

## 6. Profile after the edit (median of 5, spread 0.04%)

| Engine | busy | % of 5130 us | was |
|---|---|---|---|
| Activation | 3768 us | 73.4% | 6284 |
| DVE | 2701 us | 52.7% | 2733 |
| Pool | 2072 us | 40.4% | 2072 |
| PE | 2013 us | 39.2% | 1967 |
| DMA | 1898 us | 37.0% | 1879 |
| SP | 801 us | 15.6% | 802 |

Run-to-run spread is 0.04-0.12% (5-13 us), so a real win needs to be >= ~15 us.

Remaining ACT work: ~3000 us of `ACTIVATE` (6144 x `f=128 nch=128 nelem=64` at 434 ns = 2669 us;
1024 x `f=128 nch=1 nelem=128` at 489 ns = 500 us; 1024+1020 exps at ~193 ns),
606 us of DGE triggers, ~8 us of table loads.

## 7. Next, ranked

1. **Finish the DGE-trigger move -- start here.** 606 us of ACT, the largest non-`ACTIVATE`
   block, and section 4 shows it is *not* blocked by a capacity or ordering limit: whole 32- and
   64-trigger windows move, verify and give identical numbers; only a minority of specific
   triggers refuse. All the machinery (slot insertion on another engine, the two handshakes, the
   completion-count formula, the placement rule) is written up in 4b/4d and was exercised on the
   device. What is left is a ~25 minute window scan. Expect 350-450 us (7-9%).
2. **Move ACT copies to Pool/DVE.** The 1024 `f=128 nch=1 nelem=128` copies are 500 us and are
   the worst value on ACT (489 ns to move 512 B on one partition). `COPY` runs on DVE, ACT and
   Pool. Same insertion machinery as item 1 -- do item 1 first, it is the cheaper test of it.
3. **ACT is now only 73% busy, and its idle time is handoff latency, not throughput.** Summing
   the gaps between consecutive ACT instructions in `prof3` and grouping by the semaphore each
   stalled instruction waits on:

   | waited semaphore | stalls | total gap |
   |---|---|---|
   | 6 (DVE completion) | 4349 | **866 us** |
   | 5 (PE / MATMUL) | 1087 | **411 us** |
   | 4 (ACT's own) | 4241 | 120 us |
   | everything else | ~1500 | 230 us |

   DVE is 53% busy and PE 39%, so neither is throughput-bound: these 1.28 ms are producer-consumer
   handoff latency on a serial chain, spread over thousands of ~200-380 ns stalls with no single
   dominant one. The 1087 sem-5 stalls are almost exactly the 1020 `prod` copies waiting for the
   matmul that fills their PSUM bank. Shaving any one engine is now worth less than before;
   reordering ACT so independent work sits in front of a sem-6 wait (ch 14.3 item 2) is the shape
   of the next real win, and it needs the BIR dataflow predecessors (ch 11.2) to be safe.
4. The last 4 unconverted units are worth ~10 us. Not worth the risk.
5. `spill_save_bytes` = `spill_reload_bytes` = 35 MB. This is SBUF pressure from the compiler and
   cannot be fixed by NEFF editing, but it is what all 1088+3169 DGE triggers are for.

## 8. Tool notes
* Control repack of this NEFF is **byte-identical** (`neff.py unpack` then `pack` == baseline), so
  `neff_diff.py` is trustworthy. The payload is compressed: editing bytes without changing any
  member size still changes `payload_size` (it shrank when slots became all-zero). Not a bug.
* `ACTIVATION_TABLE_LOAD` and `NOP` share `CTRL_NO` so a load -> NOP is 2 bytes, but `NOP` never
  appears on Activation in the corpus; an all-zero `EVENT_SEMAPHORE` (byte 0 `0xa0`, byte 1
  `0x10`, rest zero) is proven on this engine, and deleting the slot entirely is better still.
* `ACTIVATE` byte offsets used here: 4:12 events, 12 `imm_src`, 16:20 src addr (+marker at 19),
  20:32 src pattern, 32:35 dtypes/channels, 35 `activation_func`, 36:40 `scale_value_imm_fp32`,
  48:52 dst addr (+marker at 51), 52:64 dst pattern.
* `f=128` (`copy_1p`) does honour `scale` (all 7168 existing ones drive it from a pointer) and its
  `imm` is a don't-care (`imm_imm_ptr` = 0 in all 7168).
* **Do not pipe a long device-test loop through `tail` or into a 10-minute background task.** I
  lost a 17-window scan that way: `tail` buffered every line, the harness stopped the background
  job at its time limit, and all 17 results went with it. Print each result as its own line from
  the foreground, in chunks that each fit a single tool call.
* A hung NEFF costs ~71 s per `verify.py` (30 s exec timeout + 30 s model-stop timeout) and a
  passing one ~25 s; the device recovers fully every time, so budget device tests at ~45 s each.
* The profiler's `Instruction.parquet` `compiler_pc` is NaN for runtime-added instructions
  (`DRAIN`, `NOTIFY`, `COMPARE_BRANCH`, `TENSOR_LOAD`, `EVENT_SEMAPHORE_RANGE_CLEAR`); drop those
  rows before building a pc -> timestamp map. On SP every `PSEUDO_DMATRIGGER` appears as two
  `WRITE` rows carrying the trigger's own `compiler_pc`.

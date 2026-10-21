# iter-5 — **iter-4's §7.1 mystery is SOLVED**: blk18's trigger carries a *hidden WAR* on
# `PE[134]`'s read of `p0+0x4780`, satisfied only by the flat 0.72 us DMA trigger→start latency.
# That closes the whole blk15→blk18 segment for good. Shipped instead: a 4-edit package on the
# **blk20 branch of the tail**, where the same backward walk found 0.26 us of *real* slack.
# Also found, the hard way: **a wait-less update-only `EVENT_SEMAPHORE` on Activation increments
# its semaphore BEFORE the compute op in front of it** — which silently re-numbers `S[4]`.

## Result: **38.281 → 38.138 us (−0.14, −0.37%)** — medians of **9 anchor** and **11 shipped**
## 7-capture runs, and **12 of 12 interleaved A→B comparisons favour the edit**.
## `max_abs_err` identical to the baseline to every printed digit, 3/3 runs bit-identical.

| step | edit | verdict |
|---|---|---|
| — | iter-4's bytes (the anchor), 9 separate 7-capture runs | **38.239 / 38.242 / 38.254 / 38.281 / 38.281 / 38.385 / 38.454 / 38.523 / 38.570** → median **38.281** |
| **M** | `ACT[69]` (388 ns `TENSOR_SCALAR`) on idle Vector + **wait-less** `S[4]` stand-in | **REFUTED: verify FAIL, output_0 9.23e-03, cosine 0.9774** |
| M2 | = M but the stand-in *waits* `S[54]>=1` (so `ACT[70]/[71]` keep their old timing) | PASS, errors = baseline → **the engine move is numerically exact; 100% of M's corruption comes from `ACT[70]/ACT[71]` running 0.5 us earlier** (§1) |
| P | `ACT[12]` (384 ns `TENSOR_SCALAR`, 3.9 us of Scalar backlog) → idle Vector hole at 5.65 | PASS, **38.378 (= 0)**. Landmarks moved as designed (`S[4]#9` −0.08, blk3 −0.07, `S[4]#32` −0.12) but **`S[17]` +0.07** — the head's gain is absorbed mid-kernel. **Dropped.** |
| Q | hoist `ACT[15]` (`ACTIVATE`, wait-less) above `ACT[13]`'s `S[48]` fence | **REFUTED: 2.66e-03, and the two runs were NOT bit-identical** → real race (§2) |
| **R+T+U** | the blk20-branch package (§3) | PASS, but **introduced an unguarded RAW** (§3.1) |
| **+V** | the one-field fix for it | **PASS, SHIPPED.** Measures the same or better than R+T+U — the fix cost nothing |

`launch_rules.py` PASS. `semaphores.py` clean at every step, **including both refutations and the
unguarded RAW** (`S[4]` 41 inc / max_wait 40, `S[5]` 64/64, `S[6]` 66/66, `S[61]` 1/1).
**`semaphores.py` has now missed 7 straight real bugs across iters 4–5. Run `verify.py --runs 2`
before every profile, always.**
`neff_diff` vs iter-4: 7 of 34 members (`sg00/{Activation0.bin 82 slots, DVE0.bin 100→101,
PE0.bin 146, SP0.bin 41}` + 3 `.json` sidecars). md5 `dd0a4e4eb0ad6c965af337be572eee43`.
Engine busy: Scalar 20.2, Vector 19.7, DMA 14.5, PE 12.0, Pool 9.8, Sync 8.9 — nothing over 53%.
Against iter-0: 48.519 → 38.138, **−21.4% (1.27x)**.

---

## 1. THE FINDING: `ACT[69]/ACT[70]/ACT[71]` are unreorderable — iter-4 §7.1 closed

Slot names as of iter-4's bytes:
```
ACT[69] TENSOR_SCALAR  bf16 p0+0x37d00 x ptr p64+0x4500 -> p0+0x4880   388 ns  W S[39]>=16 -> S[4]#33
ACT[70] ACTIVATE COPY  p96+0x4180 x ptr 0x51fc -> p96+0x4b00 (128 fp32) 488 ns  W S[6]>=54  -> S[4]#34
ACT[71] ACTIVATE SOFTPLUS p96+0x4b00 -> p96+0x4400 (128 fp32)           305 ns  W S[4]>=34  -> S[4]#35
SP[35]  blk18 trigger                                                           W S[4]>=35
```
`ACT[70]`'s gate (`S[6]#54` @29.776) lands 0.48 us before it runs (30.277) — parked behind
`ACT[69]`, gate `S[39]#16` @29.840. iter-4 §0's 0.5 us lever, apparently. **It is not:**

> **blk18's descriptor 1 writes `p0..127 + 0x4700..0x47ff` (`SP0.json` id 18, `to_off 18176`,
> `to_sizes [128,128,1,1]`, bf16), and `PE[132]`/`PE[134]` (`MATMUL src=bf16@0x4780`, 128 elem,
> NO semaphore wait of their own) read `0x4780..0x487f`. The overlap is `0x4780..0x47ff`.
> `PE[134]` ends 31.30–31.53; blk18's transfer starts 31.63–31.85. The only thing separating them
> is the flat 0.72 us trigger→start DMA latency.**

So `S[4]>=35` on blk18's trigger does **two** jobs: the RAW on `p96+0x4400` (`ACT[71]`'s output)
*and* "two more Scalar ops of delay so `PE[133]/PE[134]` finish". Any edit that fires that trigger
earlier corrupts `output_0` — iter-4's F (3.3e+04), G and H (byte-identical), and my M
(9.2e-03, milder because the overlap is only partial and `PE[133]` kept its old timing). M2 proves
the engine move itself is bit-exact, so the reorder is the entire cause.

**`PE[131..134]` cannot be pulled earlier either.** `PE[131]`'s gate `S[6]#56` sits on a chain with
≤0.03 slack at every link: `blk16 (S[24]#32) → PE[128/129] → S[5]#58 → DVE TT → S[6]#55 →
DVE CAST (dst p0+0x4780) → S[6]#56`.

**The whole `blk15 → ACT[69] → ACT[70] → ACT[71] → blk18` segment is CLOSED**, and with it iter-4's
candidates 1 and 2 and its edits K and L, which all live downstream of this wall.

> **Find this class in one pass, before building anything:** for every queued block, expand
> `to_off`/`to_sizes`/`to_steps` into an SBUF byte range and grep every stream for
> MATMUL/LDWEIGHTS sources that overlap it. If an overlapping reader has **no wait of its own** and
> finishes less than ~0.8 us before the transfer starts, the trigger's wait is a disguised WAR
> guard and the trigger cannot move. `transfer start − trigger = 0.71–0.72 us flat` for all 21
> blocks (256 B to 16 kB), so "trigger 0.5 earlier" == "the write lands 0.5 earlier".

## 2. Hidden fences: `slack = start − own_gate` is now the WRONG column

* **`SP[23]` (blk14's trigger) shows 1.118 us of "slack"** (own wait `S[4]>=29` lands 25.409, it
  fires 26.527). It is not slack. `SP[22]` in front of it is `ES{W S[6]>=51}` with no update, and
  `S[6]#51` is `DVE[81] TENSOR_TENSOR` whose **`src1` is `p0..63 + 0x4200..0x42ff`** — exactly
  blk14's destination (`to_off 0x4200`, 4 B x 128 partitions). The fence is blk14's WAR guard and
  lands 26.435, 0.09 us before the trigger fires. **blk14 cannot move; the "edit K" class is dead.**
* **`ACT[13]`'s `ES{W S[48]>=1}`** (edit Q): hoisting the wait-less `ACT[15]` over it gave a
  **non-deterministic** wrong `output_0`. `scan.py` over every `*_addr_immediate`/`*_imm_ptr` field
  in all five streams finds only `ACT[15]` (w) and `ACT[18]` (r) on `p64+0x4180`, so the dependency
  is **not visible in the instruction address fields** — suspect a DGE row whose
  `dst_num[0] x dst_elem` spans past its printed offset (iter-4 §7 trap), or Activation-engine
  internal state. Unresolved; do not retry without a new idea.

> **A wait-only `EVENT_SEMAPHORE` (wait, no update) in front of a trigger or a compute op is a
> deliberate fence almost always. Resolve what its semaphore's producer READS, not only what it
> writes.** Compute `gate = max(own wait, every wait-only ES between the previous update and here)`
> and re-rank: on this kernel the five biggest raw "slacks" are all such fences and the sixth is a
> disguised WAR. Everything that survives both filters adds up to about 0.25 us, which is exactly
> what §3 harvested.

## 3. SHIPPED — the blk20 branch of the tail: four edits, each worth 0 alone

The tail has **two branches that bind `ACT[80]` within 10 ns of each other**:
```
blk19 branch (fully gate-bound, nothing to win):
  S[5]#61 @32.895 -> DVE[93] -> S[6]#60 @33.148 -> Pool[19] -> S[3]#7 @33.451 -> DVE[95]
  -> S[6]#62 @33.677 -> blk19 trigger -> S[27]#32 @35.074 -> PE[142/143] -> S[5]#63 @35.509
  -> ACT[79] 35.531-35.789  ----\
blk20 branch (0.26 us of real slack at ACT[77]):                              > ACT[80] 35.815
  ACT[77] (gate S[6]>=60 @33.148, ran 33.452) -> S[4]#37 @33.900 -> DVE[97] ------/
  -> S[6]#64 @34.182 -> ACT[78] -> S[4]#38 @34.467 -> blk20 trigger -> S[28]#32 @35.728
  -> ACT[80] -> PE[144/145] -> S[5]#64 -> DVE[99] -> S[6]#66 @36.888 -> SP[40] row -> S[17] 38.928
```

| | edit | mechanism |
|---|---|---|
| **R** | **swap `ACT[76]` ↔ `ACT[77]`** | `ACT[76]` (`TENSOR_SCALAR` PSUM 0x2001000 x ptr p64+0x4000 → p0+0xe000, gate `S[5]>=62`) has **no `S[4]` update**, so the swap is numbering-neutral. `ACT[77]` (`ACTIVATE` p32+0x5100 x ptr 0x51fc → p0+0x4000, 438 ns) was 0.26 us behind it. *Its own wait `S[6]>=60` **is** the WAR guard against `DVE[93]`, which reads `p0+0x4000` until 33.137 — so 33.17 is the earliest it may legally run, and that is exactly where it lands.* |
| **T** | **swap `DVE[96]` ↔ `DVE[97]`** | `DVE[97]` (`TT` p0+0x4000, p0+0x5100 → p96+0x4308, gate `S[4]>=37`) was behind the 223 ns `DVE[96]` (`CAST` p64+0x4108 → p0+0x4180). Both update `S[6]`, so **#63 and #64 trade**: `ACT[78]` `S[6]>=64 → >=63` (it reads p96+0x4308), `PE[142]` `S[6]>=63 → >=64` (it covers `PE[143]`'s read of p0+0x4180..0x427f — `PE[142]`'s own LDWEIGHTS only reads 0x4100..0x417f, so S[6]>=63 was never *its* guard). Ranges disjoint, and `DVE[96]`'s read p64..127+0x4108..**0x4307** is exactly adjacent to `DVE[97]`'s write p96+**0x4308**..0x4407. |
| **U** | **`ACT[79]` (258 ns `TENSOR_SCALAR` PSUM 0x2002800 x imm 0.0883883 → p0..63+0x4510..0x470f) onto the idle Vector hole 34.17–35.82, with a wait-less `S[4]` stand-in** | After R+T, blk20 lands at 35.16 but `ACT[79]`'s gate `S[5]#63` (the *blk19* branch) is 35.51, so `ACT[79]` would re-bind `ACT[80]` and give back 0.5 us. With it on Vector, `ACT[80]` starts the moment blk20 lands. `DVE[98]` is the only `S[4]>=39` waiter and is re-pointed to the new `S[61]`; the moved op is inserted immediately in front of it so Vector program order still covers it. |
| **V** | **`SP[39]` (blk20's trigger) `S[4]>=38 → >=39`** | the fix for §3.1. |

**Each is worth 0 alone** — R+T only move the blk20 branch, which blk19 then re-binds through
`ACT[79]`; U alone leaves `ACT[80]` waiting for blk20 at its old time.

### 3.1 The bug edit U introduced, and the general lesson

> **A wait-less, update-only `EVENT_SEMAPHORE` on Activation does NOT increment in program order
> with respect to the compute op immediately before it. The Activation sequencer dispatches it
> while that op is still in flight, so the ES's increment lands FIRST.**

Measured: stand-in `ACT[79]` ran 33.893–33.933 and took **`S[4]#38` (@33.936)**; `ACT[78]`
(`ACTIVATE SOFTPLUS → p0+0x5100`) ran 33.931–34.184 and became **`#39`**. `SP[39]`, blk20's
trigger, waits `S[4]>=38` and therefore fired at 33.641 — **before its own source `p0+0x5100` was
written**, leaving only the 0.72 us trigger→start latency (transfer 34.352) as the guard. Output was
bit-identical over 5 runs, so it "worked", but a real RAW guard had been removed.

**Edit V re-points the trigger at `S[4]>=39`, which is correct under either dispatch order**
(whichever of {stand-in, `ACT[78]`} lands second is #39, and both orders put `ACT[78]` at or
before it). Device after V: `S[4]#39` @34.193, trigger 34.21, transfer 34.925 — properly ordered,
and `S[16]`/`S[17]`/the metric are the same or better than before the fix. **The fix was free.**

> **So the iter-2/3/4 stand-in recipe needs splitting in two:**
> * `ES{wait S[new]>=1}` with **no update** — safe, this is what iters 2/3/4 used; it injects no
>   increment and only holds the engine.
> * `ES{no wait, update S[x]}` — **re-orders `S[x]` against the preceding compute op.** After using
>   it, re-derive which count every waiter of `S[x]` now resolves to **from the profile**, not from
>   program order. The `>= n+1` form is the robust one.

**Landmarks, iter-4's bytes → shipped** (single captures, so read the deltas not the absolutes):

| landmark | before | shipped | Δ |
|---|---|---|---|
| `S[4]#37` | 33.900 | 33.554 | −0.35 |
| `S[4]#38/#39` (`ACT[78]` done) | 34.467 | 34.193 | −0.27 |
| `S[28]#32` (blk20) | 35.728 | 35.563 | −0.17 |
| `S[4]#40` (`ACT[80]`) | 36.213 | 35.985 | −0.23 |
| `S[6]#66` (`DVE[99]`) | 36.888 | 36.660 | −0.23 |
| **`S[17]#16` (exit fence)** | **38.928** | **38.707** | **−0.22** |

**Metric, interleaved, alternating which config ran first — 12 of 12 comparisons favour the edit**
(Δ in favour of the edit: 0.273, 0.068, 0.012, 0.129, 0.269, 0.431, 0.104 for R+T+U; then
**0.270, 0.397, 0.466, 0.009, 0.102** for R+T+U+V). Shipped captures (11): 37.985\*, 38.008\*,
38.057, **38.137**, 38.138\*, 38.139\*, 38.184, 38.213\*, 38.230, 38.256\*, 38.269\*
(\* = pre-V bytes, identical chain) → median **38.138**. Anchor (9) → median **38.281**.
**Quote 38.281 → 38.138, −0.14 (−0.37%).** No comparison ever went against the edit, in either
ordering, over 90 minutes of drift (the anchor itself ranged 38.24–38.57).

> **The dependency chain moved 0.22–0.48 us but the metric moved 0.19.** `total_exec_time` comes
> from the vendor summary JSON, not the instruction table, and the trace carries a ~12 us prologue
> and a ~1.8 us multi-engine `DRAIN`/`NOTIFY` epilogue after `S[17]`. Pass-through is roughly 1:1
> to 1:2, **not** 1:1 — and single-capture absolute times drift ±0.4 us between profiles, so
> normalise on an upstream landmark (`S[4]#35` is a good one: it is upstream of everything in §3)
> before claiming a chain delta.

## 4. Where the remaining ~38.1 us goes

**`S[16]#16` (the sem-16 Pool store) is now 38.447, only 0.26 before `S[17]#16` (38.707)**, and
`Pool[22]` (the exit fence, `W S[17]>=16` + extended `sem16>=16`) runs right after `S[17]`.
**So any further cut to `S[17]` is capped at 0.26 us until the sem-16 store moves too** — and that
one is easy: `Pool[20]` is a DGE row on **Pool**, which has a fixed **+0.39 us DGE issue latency**
(iter-3 §0) where Activation and Sync react in +0.02. Activation is idle 35.99→36.66 (only the
dead `ACT[81]` is left there). **Re-host `Pool[20]` at Activation index 81 FIRST in iter-6: worth
0 on its own, but it lifts the cap on everything else in the tail.**

Gaps (≈2.2 us of all-engine idle in 4, plus the ~2.3 us tail store):

| gap us | at | waiting for |
|---|---|---|
| 0.633 | 12.44 | blk4's `S[28]#16` → `PE[15]`. blk4 ends 12.533, the **16th** increment lands 13.035 — 0.50 of pure propagation |
| 0.575 | 34.53 | blk19's `S[27]#32` → `PE[141]` |
| 0.545 | 10.80 | blk3 — the only non-transpose block, **512 packets of 32 B**, 1.08 us |
| 0.460 | 28.87 | `S[46]`'s DGE flight |

**The head (9.0→13.1 us) is 4.0 us of two serial DMA hops and is structural:**
`S[6]#13 @9.02 → blk3 trigger 9.04 → start 9.94 → end 11.02 → S[27]#16 @11.33 → blk4..blk12
triggers 11.35–11.96 → blk4 start 12.06 end 12.53 → S[28]#16 @13.04 → PE[15]`.
blk4..blk12 **all read blk3's destination** (`0x37700..0x37ac0`), so `SP[11]`'s `S[27]>=16` is a
true RAW — and blocks on this queue **do overlap in flight** (blk4 12.058–12.533 overlaps blk5
12.119–12.544), so firing a trigger early there is a real race, not merely a risk.
**blk3 is slow for a findable reason** (`SP0.json` id 3, two `copy` descriptors, no transpose):
`from_sizes [32,4,4,16] / from_steps [1,32,4194304,262144]` → `to_steps [1,64,256,262144]`, i.e.
**32-byte chunks written at stride 64**, 256 packets per descriptor. Edit P proved the head's gain
does **not** reach `S[17]` anyway, so this is low value even if it could be widened.

## 5. Tools (rebuild, ~20 min; `scripts/` is not preserved)

`scripts/d2.py` (load/save/decode a stream as 64-byte slots; per-slot waits/updates/addresses),
`scripts/scan.py` (regex an address over every address-like field in all 5 streams **and**
`SP0.json` `from_off`/`to_off`), `scripts/an.py` (`gaps|occ|slack|tl|land|sem|dma|cols` over the
parquet, normalised to `SP[0] SET_ORDERING_MODE`), `scripts/blk.py` (head-of-line: ops whose gate
landed before the previous op on their engine ended), `scripts/edit{M,P,Q,RTU,V}.py`. **Every edit
script asserts its pre-image fields, re-decodes after `encode()`, and re-derives the `S[4]`/`S[6]`
numbering from the patched stream before saving** — that caught two numbering slips.

Traps on top of iter-3 §1 and iter-4 §1 (all still true):
* **the update field is `events_update_mode` / `events_update_idx`**, not `events_set_*`;
* `ins.struct_name` does not exist — use `ISA.decode(bytes)` + `as_dict()`;
* an SBUF address immediate is `partition*0x40000 + offset`; `*_reg_regnum`, `*_reg_reg_hi`,
  `*_table_offset`, `*_var_offset` are **aliases of the same bytes**, not separate fields;
* `scale_value_imm_ptr` is a union — `0x3f800000` is the float 1.0 (`scale_src =
  INSTRUCTION_IMMEDIATE`); a small value is an SBUF pointer (`POINTER_IMMEDIATE`). Read
  `scale_src`/`imm_src` first. (iter-4's open question about `0x51fc`: it is consistent with
  absolute `p0+0x51fc`, written by blk0 at ~5.0 us and again by `ACT[78]` at ~34.2, and it is not
  the cause of anything here.)
* `DmaPacketAggregated` (`block_id, semaphore_id, trigger, start_ts, end_ts, read_offset,
  write_offset, transfer_bytes, queue_name`) gives the whole DMA story in one `to_string()`;
* `an.py land S:thr` picks the **first** landing of a (sem, value) pair and sems 24–39 reach 16
  twice, so second-round ring `ES` rows appear as 20+ us of fake slack. Filter them.
* **Engine inventory (decides every move):** `Activation0.bin` = `ACTIVATE 18`,
  `TENSOR_SCALAR_ARITH_OP 14`, `CAST 32`, `EVENT_SEMAPHORE 15`, 1 DGE row — **no `TENSOR_TENSOR`**.
  `DVE0.bin` = `TENSOR_TENSOR 36`, `TENSOR_SCALAR 5`, `CAST 17`, `COPY 16`, `MEMSET 11` — **no DGE
  row**. `Pool0.bin` = `TENSOR_SCALAR_AFFINE_SELECT 7`, `MOVE 2`, `NOP`, 8 DGE rows — **no plain
  `TENSOR_SCALAR`**. `PE`/`SP` have no compute. **Activation↔Vector is the only compute-move axis;
  Activation/Pool/Sync are the only DGE hosts.**

## 6. Ranked candidates for iter-6

1. **Re-host `Pool[20]` (the sem-16 output store DGE row) onto Activation index 81** (§4). Worth 0
   alone; it lifts the 0.26 us cap on every further `S[17]` reduction. Pool's +0.39 DGE issue
   latency vs Activation's +0.02 is iter-3 §0, device-measured twice.
2. **Then re-walk the blk20 branch with the §2-corrected gate.** After R+T+U+V it is gate-bound at
   every link: `S[5]#61 → DVE[93] → S[6]#60 → ACT[76] → S[4]#37 → DVE[96] → S[6]#63 → ACT[78]
   → S[4]#39 → blk20 → ACT[80] → PE[144/145] → DVE[99] → SP[40]`.
3. **The 0.37–0.58 us propagation between a block's last packet and its last semaphore
   increment.** `sem_increment = 16` for every block, 2 kB or 16 kB alike; six hops x ~0.5 us is
   the largest repeated cost left. Where does a *queued* block's increment count come from?
   (`SP0.json` has no such field; DGE rows carry `semaphore=NN sem_increment=16` in the slot.)
   One read of `neff_doc` ch. 9 plus the queue descriptor tables (`default_*_table.bin`).
   **Highest remaining ceiling that is not a layout change.**
4. **A second DMA queue** (iter-4 §10.6). All 21 blocks share `queue_idx 3`.
5. **Do NOT retry**: anything that fires blk18's or blk14's trigger earlier (§1, §2); replacing a
   DMA block with compute (iter-4 §3); splitting/re-hosting the sem-17 store (iter-3 §5,
   iter-4 §8); the head reorders `ACT[12]`→Vector and the `ACT[15]` hoist (§P, §Q) — the head's
   gain does not reach `S[17]`, measured.

## 7. Method

* **Build every member of a package, then A/B the package.** R, T, U are each worth 0; together
  they are −0.22 at `S[17]`. The landmark table proved it; the metric alone would have read as noise.
* **Decide with interleaved pairs, never with pooled medians.** The anchor read 38.254 and 38.570
  twenty minutes apart, and drifted *upward* across this session (38.24–38.28 early, 38.45–38.52
  late) — the opposite direction from iter-3's session.
* **Profile the thing you just verified, and then read the semaphore *landings* to check that each
  waiter still resolves to the op you think it does.** That is the only reason §3.1 was caught: the
  design said `S[4]#38 = ACT[78]`, the device said `#38 = the stand-in`.
* Device cost: 10 verifies (2 failures, both caught by `verify.py`, neither by `semaphores.py`),
  19 profiles, no hang.
* Free semaphores after this iteration: 0, 1, 7, 18–23, 40–42, 54, 55, 62, 63 (**61 is edit U's**;
  52/53 iter-4's, 50/51 iter-3's, 47/48/49 iter-2's; **2 is the runtime barrier — do not touch**).

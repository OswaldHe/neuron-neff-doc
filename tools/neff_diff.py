#!/usr/bin/env python3
"""Diff two NEFFs at three levels: header fields, package members, and decoded instructions.

A byte diff of two NEFF files is useless: the payload is gzipped, so one changed byte shifts
every compressed byte after it. This unpacks both files and compares:

  1. header fields (chapter 2.1), noting that data_size / hash / uuid always change with content;
  2. the set of package members, their sizes and their contents;
  3. for each differing engine stream, every instruction slot that changed, as raw bytes and as
     decoded fields (chapter 6);
  4. for differing JSON members, the keys whose values changed.

Usage (workspace/.venv):
    neff_diff.py <a.neff> <b.neff> [--context]
"""

import difflib
import hashlib
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isa  # noqa: E402
import neff  # noqa: E402

SLOT = 64
ENGINE_SUFFIX = ("Activation0.bin", "DVE0.bin", "PE0.bin", "Pool0.bin", "SP0.bin")
ALWAYS_DIFFER = {"payload_size", "payload_md5"}


def members(path):
    """{member name: bytes} of a NEFF payload, plus the parsed header."""
    h, _, payload = neff.read(path)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read()
    return h, out


def diff_header(ha, hb):
    print("== header")
    for k in ha:
        if ha[k] == hb[k]:
            continue
        note = "  (always changes with the payload)" if k in ALWAYS_DIFFER else ""
        print(f"   {k}: {ha[k]} -> {hb[k]}{note}")
    if all(ha[k] == hb[k] for k in ha):
        print("   identical")


def changed_slots(a, b):
    n = max(len(a), len(b)) // SLOT
    for i in range(n):
        sa, sb = a[i * SLOT:(i + 1) * SLOT], b[i * SLOT:(i + 1) * SLOT]
        if sa != sb:
            yield i, sa, sb


def show_slot_diff(name, idx, sa, sb):
    print(f"   slot {idx} of {name}")
    ia = isa.decode(sa) if sa else None
    ib = isa.decode(sb) if sb else None
    for tag, raw in (("  -", sa), ("  +", sb)):
        if not raw:
            print(f"   {tag} (absent)")
            continue
        hexs = " ".join(raw[j:j + 4].hex() for j in range(0, SLOT, 4))
        print(f"   {tag} {hexs}")
    marks = []
    for j in range(SLOT):
        if (sa[j:j + 1] or b"") != (sb[j:j + 1] or b""):
            marks.append(j)
    print(f"     bytes differing: {marks}")
    if ia and ib and ia.struct and ia.struct == ib.struct:
        da, db = ia.as_dict(), ib.as_dict()
        # Several fields can share the same bytes (union members of one address, chapter 9.5), so
        # group the changed fields by the byte range they read and print each range once.
        groups = {}
        for f in ia.fields():
            k = f["name"]
            if da[k] == db[k]:
                continue
            n = f["count"] * (f["stride"] or 1) if f["stride"] else f["width"]
            groups.setdefault((f["offset"], n, f["shift"], f["mask"]), []).append(k)
        for (off, n, shift, mask), names in sorted(groups.items()):
            live = [k for k in names if "reserved" not in k] or names
            pick = min(live, key=len)
            fa, fb = da[pick], db[pick]
            if isinstance(fa, int) and fa > 255:
                fa, fb = f"{fa} ({fa:#x})", f"{fb} ({fb:#x})"
            alias = ("; same bytes as " + ", ".join(k for k in names if k != pick)) if len(names) > 1 else ""
            print(f"     bytes {off}..{off + n - 1}: {pick}: {fa} -> {fb}{alias}")
    elif ia and ib:
        print(f"     opcode {ia.name} [{ia.struct}] -> {ib.name} [{ib.struct}]")


def diff_json(name, a, b):
    ja, jb = json.loads(a), json.loads(b)
    sa = json.dumps(ja, indent=1, sort_keys=True).split("\n")
    sb = json.dumps(jb, indent=1, sort_keys=True).split("\n")
    lines = [l for l in difflib.unified_diff(sa, sb, lineterm="", n=1)][2:]
    print(f"   {len(lines)} differing lines (unified diff of the pretty-printed JSON):")
    for l in lines[:40]:
        print("     " + l)
    if len(lines) > 40:
        print(f"     ... {len(lines) - 40} more")


def main():
    a, b = sys.argv[1], sys.argv[2]
    ha, ma = members(a)
    hb, mb = members(b)
    print(f"A = {a}\nB = {b}\n")
    diff_header(ha, hb)
    only_a, only_b = sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))
    same = [k for k in ma if k in mb and ma[k] == mb[k]]
    diff = [k for k in ma if k in mb and ma[k] != mb[k]]
    print(f"\n== members: {len(ma)} in A, {len(mb)} in B; {len(same)} identical, "
          f"{len(diff)} changed, {len(only_a)} only in A, {len(only_b)} only in B")
    for k in only_a:
        print(f"   only in A: {k}")
    for k in only_b:
        print(f"   only in B: {k}")
    for k in diff:
        print(f"\n== changed member: {k}  ({len(ma[k])} -> {len(mb[k])} bytes, "
              f"md5 {hashlib.md5(ma[k]).hexdigest()[:8]} -> {hashlib.md5(mb[k]).hexdigest()[:8]})")
        if k.endswith(ENGINE_SUFFIX):
            sl = list(changed_slots(ma[k], mb[k]))
            print(f"   {len(ma[k]) // SLOT} slots in A, {len(mb[k]) // SLOT} in B, "
                  f"{len(sl)} slots differ")
            for idx, sa, sb in sl[:20]:
                show_slot_diff(k, idx, sa, sb)
            if len(sl) > 20:
                print(f"   ... {len(sl) - 20} more changed slots")
        elif k.endswith(".json"):
            diff_json(k, ma[k], mb[k])
        else:
            print("   (binary member; not decoded by this tool)")
    if not diff and not only_a and not only_b:
        print("\npayloads are identical")


if __name__ == "__main__":
    main()

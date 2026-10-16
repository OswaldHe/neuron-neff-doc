#!/usr/bin/env python3
"""Read, unpack and repack NEFF files.

A NEFF is a 1024-byte binary header followed by a gzip-compressed tar archive (the payload).
See ../docs/02-container.md for the header layout.

Usage:
    neff.py header  <file.neff>                 print the parsed header as JSON
    neff.py unpack  <file.neff> <out_dir>       write header.json + header.bin + the payload files
    neff.py pack    <unpacked_dir> <out.neff>   rebuild a NEFF from an unpacked directory
    neff.py verify  <file.neff>                 check payload size and MD5 against the header
"""

import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import zlib

HEADER_SIZE = 1024

# Known fields of the 1024-byte header. (offset, struct format, name). Offsets not listed here
# were zero in every NEFF studied (see docs/02-container.md).
HEADER_FIELDS = [
    (0x000, "<Q", "format_version"),
    (0x008, "<Q", "payload_offset"),
    (0x010, "<Q", "payload_size"),
    (0x018, "<Q", "field_0x18"),
    (0x0A8, "<I", "field_0xa8"),
    (0x1DC, "<I", "field_0x1dc"),
    (0x1E0, "<I", "field_0x1e0"),
    (0x220, "<I", "feature_bits_0x220"),
    (0x228, "<Q", "field_0x228"),
]
MD5_OFFSETS = (0x0AC, 0x0CC)
NAME_OFFSET, NAME_END = 0x0DC, 0x1DC


def parse_header(raw: bytes) -> dict:
    if len(raw) < HEADER_SIZE:
        raise ValueError("file is shorter than the 1024-byte NEFF header")
    h = {}
    for off, fmt, name in HEADER_FIELDS:
        h[name] = struct.unpack_from(fmt, raw, off)[0]
    h["payload_md5"] = [raw[o:o + 16].hex() for o in MD5_OFFSETS]
    h["name"] = raw[NAME_OFFSET:NAME_END].split(b"\0", 1)[0].decode("utf-8", "replace")
    h["feature_bits_0x220_hex"] = "0x%08x" % h["feature_bits_0x220"]
    return h


def read(path: str):
    data = open(path, "rb").read()
    return parse_header(data[:HEADER_SIZE]), data[:HEADER_SIZE], data[HEADER_SIZE:]


def verify(path: str) -> list:
    """Return a list of problems (empty when the header matches the payload)."""
    h, _, payload = read(path)
    problems = []
    if h["payload_offset"] != HEADER_SIZE:
        problems.append(f"payload_offset {h['payload_offset']} != {HEADER_SIZE}")
    if h["payload_size"] != len(payload):
        problems.append(f"payload_size {h['payload_size']} != actual {len(payload)}")
    md5 = hashlib.md5(payload).hexdigest()
    for i, m in enumerate(h["payload_md5"]):
        if m != md5:
            problems.append(f"md5 copy {i} {m} != actual {md5}")
    return problems


def unpack(path: str, out_dir: str) -> dict:
    h, raw, payload = read(path)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "header.bin"), "wb") as f:
        f.write(raw)
    with open(os.path.join(out_dir, "header.json"), "w") as f:
        json.dump(h, f, indent=2)
    members = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for m in tar.getmembers():
            members.append(m.name)
            if not m.isfile():
                continue
            # The archive is untrusted input: refuse absolute paths and "..".
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"unsafe member name {m.name!r}")
            dst = os.path.join(out_dir, "payload", m.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with tar.extractfile(m) as src, open(dst, "wb") as f:
                f.write(src.read())
    # Member order matters for a byte-identical repack, so it is recorded.
    with open(os.path.join(out_dir, "members.txt"), "w") as f:
        f.write("\n".join(members) + "\n")
    return h


def _ustar_header(name: str, size: int) -> bytes:
    """One 512-byte POSIX ustar header, formatted exactly as neuronx-cc writes it: mode 0644,
    uid/gid/mtime 0, empty uname/gname, NUL-filled devmajor/devminor, and the checksum as six
    octal digits followed by two NULs."""
    nb = name.encode()
    if len(nb) > 100:
        raise ValueError(f"member name longer than 100 bytes: {name}")
    h = bytearray(512)
    h[0:len(nb)] = nb
    h[100:108] = b"0000644\0"
    h[108:116] = b"0000000\0"
    h[116:124] = b"0000000\0"
    h[124:136] = b"%011o\0" % size
    h[136:148] = b"00000000000\0"
    h[148:156] = b" " * 8                         # checksum is computed over spaces
    h[156:157] = b"0"
    h[257:263] = b"ustar\0"
    h[263:265] = b"00"
    h[148:156] = b"%06o\0\0" % sum(h)
    return bytes(h)


def _gzip_level1(data: bytes) -> bytes:
    """gzip member as neuronx-cc writes it: deflate level 1, memLevel 8, header with
    MTIME=0, XFL=0, OS=3 (Unix)."""
    c = zlib.compressobj(1, zlib.DEFLATED, -15, 8)
    body = c.compress(data) + c.flush()
    head = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"
    tail = struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF)
    return head + body + tail


def build_payload(unpacked_dir: str) -> bytes:
    """Tar + gzip the payload directory in the recorded member order. An unmodified unpack
    rebuilds to the original bytes."""
    order = [l.rstrip("\n") for l in open(os.path.join(unpacked_dir, "members.txt")) if l.strip()]
    base = os.path.join(unpacked_dir, "payload")
    on_disk = []
    for root, _, files in os.walk(base):
        for fn in files:
            on_disk.append(os.path.relpath(os.path.join(root, fn), base))
    extra = sorted(set(on_disk) - set(order))      # files added by hand go at the end
    out = bytearray()
    for name in order + extra:
        p = os.path.join(base, name)
        if not os.path.isfile(p):
            continue                               # files deleted by hand are dropped
        data = open(p, "rb").read()
        out += _ustar_header(name, len(data))
        out += data
        out += b"\0" * (-len(data) % 512)
    out += b"\0" * 1024                            # end-of-archive: two zero blocks
    return _gzip_level1(bytes(out))


def pack(unpacked_dir: str, out_path: str) -> dict:
    raw = bytearray(open(os.path.join(unpacked_dir, "header.bin"), "rb").read())
    payload = build_payload(unpacked_dir)
    struct.pack_into("<Q", raw, 0x010, len(payload))
    md5 = hashlib.md5(payload).digest()
    for o in MD5_OFFSETS:
        raw[o:o + 16] = md5
    with open(out_path, "wb") as f:
        f.write(bytes(raw) + payload)
    return parse_header(bytes(raw))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "header":
        print(json.dumps(read(argv[2])[0], indent=2))
    elif cmd == "unpack":
        print(json.dumps(unpack(argv[2], argv[3]), indent=2))
    elif cmd == "pack":
        print(json.dumps(pack(argv[2], argv[3]), indent=2))
        problems = verify(argv[3])
        print("verify:", "ok" if not problems else problems)
    elif cmd == "verify":
        problems = verify(argv[2])
        print("ok" if not problems else "\n".join(problems))
        return 1 if problems else 0
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

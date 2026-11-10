#!/usr/bin/env python3
"""Extract a SCETlib-AD cache's rule blob to a flat file next to it, for a fast load.

``ScetlibCachedXsecTF.load`` holds the rules blob three times (numpy array,
``.tobytes()``, C++ copy), which dominates peak memory and load time for a large
cache. ``DrellYan.load_bin_rules(path)`` reads a flat file with an ifstream instead.

This writes, beside ``<stem>.npz``:
  <stem>.rules.bin   the raw rule payload (the .npy data, header stripped)
  <stem>.rules.json  what it was extracted from: the zip member's CRC32 and sizes

``ScetlibADXsec`` uses the flat file only while the sidecar matches the cache's
current rules member. zipfile checks the CRC32 at the end of the stream, so a
corrupt read raises instead of writing a bad file.

usage: extract_cache_rules.py <cache.npz> [--out <stem>.rules.bin]
"""

import argparse
import json
import os
import struct
import time
import zipfile

MEMBER = "rules.npy"


def rules_paths(cache_npz):
    stem = cache_npz[:-4] if cache_npz.endswith(".npz") else cache_npz
    return stem + ".rules.bin", stem + ".rules.json"


def member_info(cache_npz):
    with zipfile.ZipFile(cache_npz) as z:
        i = z.getinfo(MEMBER)
        return dict(
            member=MEMBER,
            crc32=i.CRC,
            file_size=i.file_size,
            compress_size=i.compress_size,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", help="path to cache.npz")
    ap.add_argument(
        "--out", default=None, help="default: <stem>.rules.bin beside the cache"
    )
    a = ap.parse_args()
    cache = os.path.abspath(a.cache)
    out, side = rules_paths(cache)
    if a.out:
        out = os.path.abspath(a.out)
        side = out[: -len(".bin")] + ".json" if out.endswith(".bin") else out + ".json"
    info = member_info(cache)
    t0 = time.time()
    done = 0
    with zipfile.ZipFile(cache) as z, z.open(MEMBER) as f, open(
        out + ".part", "wb"
    ) as g:
        magic = f.read(8)
        if magic[:6] != b"\x93NUMPY":
            raise SystemExit(f"{MEMBER} is not an .npy member: {magic!r}")
        hl = (
            struct.unpack("<H", f.read(2))[0]
            if magic[6] == 1
            else struct.unpack("<I", f.read(4))[0]
        )
        header = f.read(hl).decode("latin1")
        if "'descr': '|u1'" not in header or "'fortran_order': False" not in header:
            raise SystemExit(f"unexpected rules array header: {header.strip()}")
        n = int(header.split("'shape': (")[1].split(",")[0])
        while True:
            b = f.read(1 << 26)
            if not b:
                break  # zipfile has now checked the member's CRC32
            g.write(b)
            done += len(b)
            if done % (1 << 34) < (1 << 26):
                print(f"  {done / 2**30:.0f} GiB  {time.time() - t0:.0f} s", flush=True)
    if done != n:
        os.remove(out + ".part")
        raise SystemExit(f"payload is {done} bytes, header says {n}")
    info.update(
        payload_bytes=done, source=cache, extracted=time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    os.replace(out + ".part", out)
    with open(side, "w") as s:
        json.dump(info, s, indent=1)
    print(
        f"wrote {out} ({done} bytes) and {side} in {time.time() - t0:.0f} s", flush=True
    )


if __name__ == "__main__":
    main()

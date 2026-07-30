#!/usr/bin/env python3
"""
Build howru.merges from a byte-level corpus.

The output format is the compact binary layout consumed by howru.c and
train_howru.py:

    uint32 n_merges
    repeated n_merges times: uint32 left, uint32 right, uint32 new_id
"""

import argparse
import collections
import struct


def replace_pair(ids, pair, new_id):
    out = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def build_merges(raw, n_merges):
    ids = list(raw)
    merges = []
    for mi in range(n_merges):
        counts = collections.Counter(zip(ids, ids[1:]))
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < 2:
            break
        new_id = 256 + mi
        merges.append((pair[0], pair[1], new_id))
        ids = replace_pair(ids, pair, new_id)
    return merges


def write_merges(path, merges):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(merges)))
        for a, b, new_id in merges:
            f.write(struct.pack("<III", a, b, new_id))


def main():
    ap = argparse.ArgumentParser(description="Build Howru byte-BPE merges")
    ap.add_argument("corpus", nargs="?", default="howru.txt")
    ap.add_argument("--out", default="howru.merges")
    ap.add_argument("--merges", type=int, default=1024)
    args = ap.parse_args()
    if args.merges < 1 or args.merges > 1024:
        raise SystemExit("--merges must be between 1 and 1024")

    with open(args.corpus, "rb") as f:
        raw = f.read()
    if not raw:
        raise SystemExit("corpus is empty")
    merges = build_merges(raw, args.merges)
    write_merges(args.out, merges)
    print(f"wrote {len(merges)} merges -> {args.out}")


if __name__ == "__main__":
    main()

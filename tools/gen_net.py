#!/usr/bin/env python3
# Quantize a trained f32 nnue.bin into the engine's self-describing net.nnue.
#
# net.nnue layout (little-endian):
#   header: [n_hidden i32][qa i32][qb i32][head_bias_q i32]   (16 bytes)
#   body:   768*N i16 ft weights, N i16 ft bias, 2N i8 head weights
#
# The header carries the quantization params, so swapping nets is a drop-in:
# regenerate net.nnue and rebuild. Only a change in hidden width needs a source
# edit (the N_HIDDEN const in src/eval/params.rs); the loader asserts a match.
#
# Usage (run from the project root):
#   python tools/gen_net.py nets/nnue63m.bin nets/nnue63m.json src/eval/net.nnue

import argparse
import json
import struct
import sys

MAX_ACTIVE = 32  # max pieces on a board; bounds the active feature count


def calibrate_qa(ft_w, ft_b, n):
    worst = 0.0
    for h in range(n):
        col = ft_w[h::n]
        top = sorted((abs(v) for v in col), reverse=True)[:MAX_ACTIVE]
        worst = max(worst, abs(ft_b[h]) + sum(top))
    return 255 if worst == 0.0 else max(1, min(255, int(32000.0 / worst)))


def calibrate_qb(head_w):
    m = max((abs(v) for v in head_w), default=0.0)
    return 64 if m == 0.0 else max(1, min(127, int(127.0 / m)))


def clampi(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin", help="trained f32 weights (nnue.bin)")
    ap.add_argument("meta", help="architecture json (nnue.json)")
    ap.add_argument("out", nargs="?", default="src/eval/net.nnue")
    args = ap.parse_args()

    n = json.load(open(args.meta))["n_hidden"]
    expected = (768 * n + n + 2 * n + 1) * 4
    raw = open(args.bin, "rb").read()[:expected]
    if len(raw) != expected:
        sys.exit(f"{args.bin}: expected {expected} bytes, got {len(raw)}")
    f = list(struct.unpack(f"<{expected // 4}f", raw))

    c = 0
    ft_w = f[c:c + 768 * n]; c += 768 * n
    ft_b = f[c:c + n]; c += n
    head_w = f[c:c + 2 * n]; c += 2 * n
    head_b = f[c]

    qa = calibrate_qa(ft_w, ft_b, n)
    qb = calibrate_qb(head_w)
    head_bias_q = int(round(head_b * qa * qb))

    out = bytearray()
    out += struct.pack("<4i", n, qa, qb, head_bias_q)
    out += struct.pack(f"<{len(ft_w)}h", *[clampi(int(round(v * qa)), -32768, 32767) for v in ft_w])
    out += struct.pack(f"<{n}h", *[clampi(int(round(v * qa)), -32768, 32767) for v in ft_b])
    out += struct.pack(f"<{2 * n}b", *[clampi(int(round(v * qb)), -127, 127) for v in head_w])
    open(args.out, "wb").write(out)

    print(f"wrote {args.out}: n_hidden={n} qa={qa} qb={qb} head_bias_q={head_bias_q}")
    print(f"  {len(out):,} bytes")
    if n != 512:
        print(f"  NOTE: n_hidden={n} differs from the default; set N_HIDDEN={n} "
              f"in src/eval/params.rs before building.")


if __name__ == "__main__":
    main()

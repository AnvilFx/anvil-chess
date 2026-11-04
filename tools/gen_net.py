#!/usr/bin/env python3
# Quantize a trained f32 net into the engine's self-describing net.nnue.
#
# Input (f32 LE), K king buckets, B output buckets (1 for the original nnue63m.bin):
#   ft_weights[768*K*N] ft_bias[N] head_weights[B][2N] head_bias[B]
# This is also bullet's layout when l1w is saved transposed (bullet/examples/anvil.rs).
# The json gives n_hidden, plus optional "buckets" (default 1) and "head_scale"
# (default 1; 400 for bullet nets, whose output is in eval-scale units, not cp).
# The engine applies head_scale after the dot product, so i8 head weights keep
# their precision. "activation" is "crelu" (default) or "screlu".
# "king_buckets" (32 ints, files a-d per rank from rank 1, as BUCKET_LAYOUT in
# anvil.rs; default all 0) and "mirrored" (default false) set the input buckets.
# A single-bucket net is copied into every bucket, so old nets still load.
#
# net.nnue layout (little-endian):
#   header: [n_hidden i32][qa i32][qb i32][n_buckets i32][out_scale i32][activation i32]
#           [king_buckets i32][mirrored i32] then 32 u8 king bucket layout
#           (64 bytes; activation 0 = CReLU, 1 = SCReLU)
#   body:   n_buckets i32 head biases, 768*K*N i16 ft weights, N i16 ft bias,
#           n_buckets*2N i8 head weights
#
# The header carries the quantization params, so swapping nets is a drop-in:
# regenerate net.nnue and rebuild. A change in hidden width or bucket count also
# needs N_HIDDEN / N_BUCKETS in src/eval/params.rs; the loader asserts a match.
#
# Usage (run from the project root):
#   python tools/gen_net.py nets/nnue63m.bin nets/nnue63m.json src/eval/net.nnue

import argparse
import json
import struct
import sys

MAX_ACTIVE = 32  # max pieces on a board; bounds the active feature count
ENGINE_BUCKETS = 8  # N_BUCKETS in src/eval/params.rs
ACTIVATIONS = {"crelu": 0, "screlu": 1}


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
    ap.add_argument("bin", help="trained f32 weights")
    ap.add_argument("meta", help="architecture json")
    ap.add_argument("out", nargs="?", default="src/eval/net.nnue")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    n = meta["n_hidden"]
    buckets = meta.get("buckets", 1)
    head_scale = int(meta.get("head_scale", 1))
    activation = ACTIVATIONS[meta.get("activation", "crelu")]
    layout = meta.get("king_buckets", [0] * 32)
    mirrored = bool(meta.get("mirrored", False))
    kb = max(layout) + 1
    if len(layout) != 32 or sorted(set(layout)) != list(range(kb)):
        sys.exit("king_buckets must be 32 entries numbering buckets 0..K-1")
    if buckets not in (1, ENGINE_BUCKETS):
        sys.exit(f"{buckets} buckets: the engine is built for {ENGINE_BUCKETS}")

    expected = (768 * kb * n + n + buckets * (2 * n + 1)) * 4
    raw = open(args.bin, "rb").read()[:expected]
    if len(raw) != expected:
        sys.exit(f"{args.bin}: expected {expected} bytes, got {len(raw)}")
    f = list(struct.unpack(f"<{expected // 4}f", raw))

    c = 0
    ft_w = f[c:c + 768 * kb * n]; c += 768 * kb * n
    ft_b = f[c:c + n]; c += n
    head_w = f[c:c + buckets * 2 * n]; c += buckets * 2 * n
    head_b = f[c:c + buckets]
    if buckets == 1:
        head_w *= ENGINE_BUCKETS
        head_b *= ENGINE_BUCKETS

    qa = calibrate_qa(ft_w, ft_b, n)
    qb = calibrate_qb(head_w)
    head_bias_q = [int(round(b * qa * qb)) for b in head_b]

    out = bytearray()
    out += struct.pack("<8i", n, qa, qb, ENGINE_BUCKETS, head_scale, activation, kb, int(mirrored))
    out += bytes(layout)
    out += struct.pack(f"<{ENGINE_BUCKETS}i", *head_bias_q)
    out += struct.pack(f"<{len(ft_w)}h", *[clampi(int(round(v * qa)), -32768, 32767) for v in ft_w])
    out += struct.pack(f"<{n}h", *[clampi(int(round(v * qa)), -32768, 32767) for v in ft_b])
    out += struct.pack(f"<{len(head_w)}b", *[clampi(int(round(v * qb)), -127, 127) for v in head_w])
    open(args.out, "wb").write(out)

    print(f"wrote {args.out}: n_hidden={n} buckets={buckets}->{ENGINE_BUCKETS} qa={qa} qb={qb} out_scale={head_scale} "
          f"activation={meta.get('activation', 'crelu')} king_buckets={kb} mirrored={mirrored}")
    print(f"  head_bias_q={head_bias_q}")
    print(f"  {len(out):,} bytes")
    if n != 512:
        print(f"  NOTE: n_hidden={n} differs from the default; set N_HIDDEN={n} "
              f"in src/eval/params.rs before building.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Quantize a trained f32 net into the engine's self-describing net.nnue.
#
# Input (f32 LE), K king buckets, B output buckets:
#   ft_weights[768*K*N] ft_bias[N] head_weights[B][2N] head_bias[B]
# This is also bullet's layout when l1w is saved transposed (tools/train/anvil.rs).
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
# Dense nets ("activation": "pairwise", with "l1" and "l2" sizes; bullet anvil_dense.rs) take
#   ft_weights[768*K*N] ft_bias[N] l1w[B][L1][N] l1b[B*L1] l2w[B][L2][L1] l2b[B*L2] l3w[B][L2] l3b[B]
# and write the same header (activation 2, qa 255, qb 0) followed by
#   [l1 i32][l2 i32], 768*K*N i16 ft weights, N i16 ft bias, B*L1*N i8 l1 weights,
#   then f32: B*L1 l1 row scales, l1 bias, l2 weights, l2 bias, l3 weights, l3 bias.
# Each l1 row gets its own scale (127 / its largest weight): young nets have small l1 weights,
# and a fixed x64 left them a few levels each (~13 cp mean error vs ~2 cp per row).
# The engine feeds L1 with (clamp(a) * clamp(b)) >> 9 of each side's accumulator halves.
#
# The header carries the quantization params, so swapping nets is a drop-in:
# regenerate net.nnue and rebuild. A change in hidden width or bucket count also
# needs N_HIDDEN / N_BUCKETS in src/eval/params.rs; the loader asserts a match.
#
# Usage (run from the project root):
#   python tools/gen_net.py nets/bullet-kb4-984m.bin nets/bullet-kb4.json src/eval/net.nnue

import argparse
import json
import struct
import sys

MAX_ACTIVE = 32  # max pieces on a board; bounds the active feature count
ENGINE_BUCKETS = 8  # N_BUCKETS in src/eval/params.rs
ACTIVATIONS = {"crelu": 0, "screlu": 1, "pairwise": 2}
DENSE_QA = 255  # fixed: the engine's >> 9 after the pairwise product assumes it


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
    ap.add_argument("--permute", metavar="FENS",
                    help="dense nets: reorder hidden neurons by how often they're active on positions from this "
                         "file (FEN first on each line), so zeros cluster into groups the engine skips")
    ap.add_argument("--permute-count", type=int, default=20000, help="positions to sample (every 50th line)")
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

    if activation == ACTIVATIONS["pairwise"]:
        return write_dense(args, meta, n, buckets, head_scale, layout, mirrored, kb)

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


def features(fen, layout, mirrored):
    """Feature indices of both perspectives, as the engine computes them."""
    pieces = []  # (color 0 = white, kind 0..5 PNBRQK, square a1 = 0)
    for r, row in enumerate(fen.split()[0].split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            pieces.append((0 if ch.isupper() else 1, "pnbrqk".index(ch.lower()), (7 - r) * 8 + f))
            f += 1

    def view(ksq):
        flip = 7 if mirrored and ksq % 8 > 3 else 0
        return 768 * layout[(ksq // 8) * 4 + min(ksq % 8, 7 - ksq % 8)], flip

    wb, wf = view(next(s for c, k, s in pieces if c == 0 and k == 5))
    bb, bf = view(next(s for c, k, s in pieces if c == 1 and k == 5) ^ 56)
    return ([wb + (k if c == 0 else k + 6) * 64 + (s ^ wf) for c, k, s in pieces],
            [bb + (k if c == 1 else k + 6) * 64 + (s ^ 56 ^ bf) for c, k, s in pieces])


def permute_neurons(args, ft_w, ft_b, l1w, n, layout, mirrored):
    """Order pairwise neurons by activity; exact, since the engine sums integers either way."""
    import numpy as np
    h = n // 2
    w = np.asarray(ft_w, dtype=np.float64).reshape(-1, n)
    b = np.asarray(ft_b, dtype=np.float64)
    qw, qb = np.round(w * DENSE_QA).astype(np.int32), np.round(b * DENSE_QA).astype(np.int32)
    masks = []
    with open(args.permute) as fh:
        for i, line in enumerate(fh):
            if i % 50:
                continue
            for feats in features(line.split("|")[0].strip(), layout, mirrored):
                acc = np.clip(qb + qw[feats].sum(0), 0, DENSE_QA)
                masks.append((acc[:h] * acc[h:]) >> 9 > 0)
            if len(masks) >= 2 * args.permute_count:
                break
    nz = np.array(masks)
    order = np.argsort(-nz.mean(0), kind="stable")
    group_rate = lambda m: m.reshape(len(m), -1, 4).any(-1).mean()
    print(f"  permute: {len(nz)} accumulators, nonzero bytes {nz.mean():.1%}, "
          f"nonzero groups {group_rate(nz):.1%} -> {group_rate(nz[:, order]):.1%}")
    cols = np.concatenate([order, order + h])
    l1 = np.asarray(l1w, dtype=np.float64).reshape(-1, n)
    return list(w[:, cols].ravel()), list(b[cols]), list(l1[:, cols].ravel())


def write_dense(args, meta, n, buckets, head_scale, layout, mirrored, kb):
    l1, l2 = meta["l1"], meta["l2"]
    if buckets != ENGINE_BUCKETS:
        sys.exit(f"dense nets need {ENGINE_BUCKETS} output buckets")
    sizes = [768 * kb * n, n, buckets * l1 * n, buckets * l1, buckets * l2 * l1, buckets * l2, buckets * l2, buckets]
    expected = sum(sizes) * 4
    raw = open(args.bin, "rb").read()[:expected]
    if len(raw) != expected:
        sys.exit(f"{args.bin}: expected {expected} bytes, got {len(raw)}")
    f = struct.unpack(f"<{expected // 4}f", raw)
    parts, c = [], 0
    for s in sizes:
        parts.append(f[c:c + s])
        c += s
    ft_w, ft_b, l1w, l1b, l2w, l2b, l3w, l3b = parts
    if args.permute:
        ft_w, ft_b, l1w = permute_neurons(args, ft_w, ft_b, l1w, n, layout, mirrored)

    worst =max(abs(ft_b[h]) + sum(sorted((abs(v) for v in ft_w[h::n]), reverse=True)[:MAX_ACTIVE]) for h in range(n))
    if worst * DENSE_QA > 32767:
        print(f"  WARNING: worst-case accumulator {worst * DENSE_QA:.0f} could overflow i16")
    rows = [l1w[r * n:(r + 1) * n] for r in range(buckets * l1)]
    scales = [127.0 / max(max(abs(v) for v in row), 1e-9) for row in rows]
    l1q = [clampi(int(round(v * s)), -127, 127) for row, s in zip(rows, scales) for v in row]

    out = bytearray()
    out += struct.pack("<8i", n, DENSE_QA, 0, ENGINE_BUCKETS, head_scale, ACTIVATIONS["pairwise"], kb, int(mirrored))
    out += bytes(layout)
    out += struct.pack("<2i", l1, l2)
    out += struct.pack(f"<{len(ft_w)}h", *[clampi(int(round(v * DENSE_QA)), -32768, 32767) for v in ft_w])
    out += struct.pack(f"<{n}h", *[clampi(int(round(v * DENSE_QA)), -32768, 32767) for v in ft_b])
    out += struct.pack(f"<{len(l1q)}b", *l1q)
    out += struct.pack(f"<{len(scales)}f", *scales)
    for p in (l1b, l2w, l2b, l3w, l3b):
        out += struct.pack(f"<{len(p)}f", *p)
    open(args.out, "wb").write(out)

    print(f"wrote {args.out}: dense n_hidden={n} l1={l1} l2={l2} buckets={buckets} qa={DENSE_QA} "
          f"out_scale={head_scale} king_buckets={kb} mirrored={mirrored}")
    print(f"  worst-case accumulator {worst * DENSE_QA:.0f}, l1 row scales {min(scales):.0f}..{max(scales):.0f}")
    print(f"  {len(out):,} bytes")


if __name__ == "__main__":
    main()

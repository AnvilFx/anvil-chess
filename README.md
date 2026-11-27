<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg">
    <img src="assets/logo.svg" alt="Anvil logo" width="200">
  </picture>
  <br>
  Anvil
</h1>

<p align="center">
  <a href="#strength"><img src="https://img.shields.io/badge/strength-~3685%20Elo-orange" alt="Strength: about 3685 Elo"></a>
  <a href="#the-net"><img src="https://img.shields.io/badge/self--play%20data-3.6B%20positions-blue" alt="Self-play data: 3.6B positions"></a>
  <a href="#uci"><img src="https://img.shields.io/badge/protocol-UCI-blue" alt="Protocol: UCI"></a>
  <a href="#building"><img src="https://img.shields.io/badge/forged%20in-Rust-orange?logo=rust&logoColor=white" alt="Forged in Rust"></a>
  <a href="https://github.com/AnvilFx/anvil-chess/releases/latest"><img src="https://img.shields.io/github/v/release/AnvilFx/anvil-chess?label=download&logo=github" alt="Download"></a>
</p>

<p align="center"><em>
  Make your move and make it wise, I see through every thin disguise.<br>
  Your boldest trap, your best surprise, your gambits dressed in clever lies,<br>
  I'll watch them with my Anvil Eyes, and with those eyes, I'll analyze.
</em></p>

A UCI chess engine written in Rust, with an NNUE evaluation trained only on its own
self-play games. Anvil has no GUI; use it from any UCI interface (Cute Chess, Arena,
Banksia, fastchess, ...).

## Features

**Search**
- Iterative deepening, principal variation search, aspiration windows; a root fail high
  ends the iteration without a re-search
- Transposition table (4-entry buckets, 64 MB by default), backed by huge pages on Linux
- Null move pruning, razoring, reverse futility pruning, futility pruning, late move pruning,
  SEE pruning, ProbCut
- Null move pruning is skipped when the opponent's last-moved piece can be won, and a null
  search that only just fails searches the node one ply shallower
- Reverse futility pruning needs a larger margin while a piece is attacked by a lesser one
- Late move reductions adjusted by history, by how the static eval changes after the move,
  by how far that eval is from alpha, and less for a move that beats every earlier move's
  eval change
- Singular extensions with multi-cut
- Move ordering by TT move, SEE-scored captures, killers, history and continuation history,
  plus a bonus for quiet checks to safe squares; history is split by whether the destination
  is attacked, and if so whether it is defended
- Pawn-structure correction history for the static eval
- Quiescence search with SEE filtering
- Repetition and fifty-move draw detection

**Evaluation**
- NNUE `(768x4 -> 1024)x2 -> (16 -> 32 -> 1)x8`: four king buckets with horizontal
  mirroring, pairwise CReLU on the accumulator, a sparse int8 first dense layer, SCReLU
  float layers after it, eight output buckets by piece count
- Trained on 3.6B positions from Anvil's own self-play
- Incrementally updated accumulators with a per-king-bucket refresh cache
- A 2 MB eval cache keyed by the full hash, prefetched along with the TT entry
- AVX2, AVX-VNNI and AVX-512 paths selected at runtime, with a scalar fallback

## Strength

Tested at faster time controls because I am impatient.

These estimates are very rough because of the difference in time controls. They are
only meant to give a general idea of the engine's strength.

Matches against engines with a [CCRL Blitz](https://computerchess.org.uk/ccrl/404/)
rating. The estimate is the opponent's rating plus the measured Elo difference, so it
is approximate: these matches run at a different time control and on different
hardware than CCRL.

| Anvil | Opponent | CCRL Blitz | TC | Games | W / D / L | Elo diff | Estimate |
|---|---|---|---|---|---|---|---|
| 3.15.4 | Reckless 0.9.0 | 3767 | 20+0.2 | 1000 | 54 / 450 / 496 | -165 ± 8 | ~3602 |
| 3.15.4 | Hobbes 4.0 | 3742 | 20+0.2 | 1000 | 68 / 467 / 465 | -146 ± 9 | ~3596 |
| 3.15.4 | Halogen 16.0.0 | 3720 | 20+0.2 | 1000 | 161 / 514 / 325 | -58 ± 11 | ~3663 |
| 3.15.4 | Minke 7.0.0 | 3701 | 20+0.2 | 1000 | 160 / 484 / 356 | -69 ± 11 | ~3632 |
| 3.15.4 | Stockfish 12 | 3656 | 20+0.2 | 1000 | 367 / 398 / 235 | +46 ± 12 | ~3702 |
| 3.15.4 | Booot 7.4 | 3651 | 20+0.2 | 1000 | 417 / 423 / 160 | +91 ± 13 | ~3742 |
| 3.15.4 | Cataphract 1.5.1 | 3651 | 20+0.2 | 1000 | 374 / 388 / 238 | +48 ± 12 | ~3699 |
| 3.15.4 | Prune 4.0.1 | 3644 | 20+0.2 | 1000 | 341 / 466 / 193 | +52 ± 12 | ~3696 |
| 3.15.4 | Bread 4.0.0 | 3622 | 20+0.2 | 1000 | 463 / 447 / 90 | +136 ± 12 | ~3758 |
| 3.15.4 | Thrawn 3.2 | 3591 | 20+0.2 | 1000 | 556 / 342 / 102 | +170 ± 15 | ~3761 |
| **Average** | | | | | | | **~3685** |

Single thread for both engines, games played in colour-swapped opening pairs from the
UHO 2024 book on an AMD EPYC 9R14 server running 124 games at a time. Error bars are 95%,
from the results of the opening pairs.

## Downloads

Binaries for Windows and Linux are on the
[releases page](https://github.com/AnvilFx/anvil-chess/releases):

- `x86-64-v3`: CPUs with AVX2 (Intel Haswell / AMD Zen and newer). Use this one if it runs.
- `x86-64-v2`: older x86-64 CPUs.

## Building

Requires a recent stable Rust toolchain.

```sh
cargo build --release
```

The binary is `target/release/anvil-engine` (`.exe` on Windows). The net is embedded,
so the binary is all you need.

`.cargo/config.toml` builds with `target-cpu=native`, which is fastest but may not run
on older CPUs than the build machine. For a portable binary, change it to
`target-cpu=x86-64-v3` (AVX2, most CPUs since 2015) or `x86-64-v2`. It also sets
`--cfg pext`, which uses PEXT for slider attacks. That is fast on Intel and AMD Zen 3 and
newer but very slow on Zen 1 and Zen 2, so remove it when building for those.

## UCI

Supported `go` limits: `wtime`, `btime`, `winc`, `binc`, `movestogo`, `movetime`,
`depth`, `nodes`, `infinite`. Anvil searches with one thread. `nodes N` finishes the
iteration in progress once N nodes are searched and stops mid-iteration only at 8 x N.

| Option | Default | Range | |
|---|---|---|---|
| Hash | 64 | 1-32768 | Transposition table size in MB |
| Move Overhead | 15 | 0-5000 | Milliseconds kept back for GUI and OS latency (at most half of a `movetime`) |
| NormalizeScore | true | | Report normalized scores (see below); `false` gives raw search scores |

The other options (`RFPMargin`, `LMRBase`, `SEEQuietMargin`, ...) are search parameters
exposed for SPSA tuning. The defaults are the tuned values; leave them alone for play.

Scores are normalized like Stockfish's: `cp 100` means the side to move wins about half
its games from here. The conversion depends on the material left on the board and is
fitted on Anvil's own self-play games. An iteration that ends on a root fail high reports
`score cp <x> lowerbound`: the score is at least that.

Non-standard commands:

- `eval` prints the static NNUE eval of the current position.
- `bench [depth]` searches a fixed set of 40 positions and prints `<nodes> nodes <nps> nps`.
  The node count is the same on every machine, so it identifies the search. Running
  `anvil-engine bench` from the command line does the same and exits.

## The net

The net in `src/eval/net.nnue` is trained with
[bullet](https://github.com/jw1912/bullet) on 3.6 billion deduplicated positions from
Anvil's own self-play games, labelled with its own search scores and game results. No
other engine's data or evals are used.

- `tools/train/anvil_dense.rs` is the bullet training config of the current net, trained
  for 800 superbatches with `ANVIL_WDL=0.5` (half game result, half search score).
- `tools/train/anvil.rs` and `nets/bullet-kb4.json` are the older net without dense layers;
  the engine switches between the two with `DENSE` in `src/eval/params.rs`.
- `tools/gen_net.py` quantizes a trained net into `net.nnue`. The header carries the
  quantization and bucket settings, so swapping in a net of the same shape needs no
  code changes.
- `nets/` holds the f32 weights and settings of the current net, so it can be
  regenerated:

```sh
python tools/gen_net.py nets/bullet-kb4-dense1024-3645m-sb800-wdl50.bin nets/bullet-kb4-dense.json src/eval/net.nnue --permute <fens>
cargo build --release
```

`--permute` takes a file of self-play positions (`FEN | score | result` lines) and orders
the hidden neurons by how often they are active, then swaps neurons between groups of four
so that more groups are all zero, which makes the sparse first layer faster.
Any sample gives the same evals; only the same sample gives the same bytes.

A different hidden width or output bucket count also needs `N_HIDDEN` / `N_BUCKETS` in
`src/eval/params.rs`.

## Layout

```
src/
  chess/          the rules: board, move generation, bitboards, zobrist keys
    board.rs      position state, FEN, make/unmake
    movegen.rs    legal and noisy move generation, perft
  eval/           NNUE inference and the embedded net
  search/         the search
    mod.rs        Engine, limits, iterative deepening
    negamax.rs    root, negamax, quiescence, draw detection
    ordering.rs   move scoring and staged move picking
    history.rs    history tables and correction keys
    context.rs    per-search state
    see.rs        static exchange evaluation
    tt.rs         transposition table
    params.rs     tunable parameters and SPSA ranges
    tests.rs      search tests
  timeman.rs      time management
  uci.rs          UCI front end
nets/             f32 weights and settings of the current net
tools/            net quantizer and training config
```

## Testing

```sh
cargo test --release
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.

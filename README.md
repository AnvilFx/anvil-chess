# Anvil

A bitboard chess engine in Rust with an NNUE evaluation and an alpha-beta / PVS
search. Standalone UCI engine, no GUI.

## Layout

```
src/                  Engine crate `anvil-engine` (lib `anvil_engine` + UCI binary)
  core/                 bitboards, board, movegen, zobrist
  eval/                 NNUE: net.nnue (baked in), params, inference
  search.rs             alpha-beta / PVS, TT, move ordering, draw detection
  timeman.rs            time management (soft/hard budgets, stability scaling)
  uci.rs                UCI protocol front-end
nets/                 NNUE weights: trained source (.bin/.json) + old-net backup
tools/                gen_net.py: quantize a trained net into src/eval/net.nnue
```

## Build & run

```sh
# Optimized build -> target/release/anvil-engine(.exe)
cargo build --release

# Run the UCI engine
cargo run --release
```

## Swapping the NNUE net

The net is baked into the binary via `include_bytes!("net.nnue")`. To use a
retrained net, regenerate it and rebuild; the header carries the quantization
params, so no source edits are needed for a same-width retrain:

```sh
python tools/gen_net.py nets/bullet-ob8.bin nets/bullet-ob8.json src/eval/net.nnue
cargo build --release
```

A change in hidden width also needs `N_HIDDEN` updated in `src/eval/params.rs`.

The engine has 8 output buckets chosen by piece count (bullet's
`MaterialCount<8>`). Nets are trained with `bullet/examples/anvil.rs`; give
`gen_net.py` a json with `"n_hidden": 512, "buckets": 8, "head_scale": 400,
"activation": "screlu"` (see `nets/bullet-ob8.json`).
Single-output nets such as `nnue63m.bin` are copied into every bucket.

King input buckets and horizontal mirroring are read from the header too:
`"king_buckets"` (32 entries, files a-d per rank from rank 1, matching
`BUCKET_LAYOUT` in `anvil.rs`) and `"mirrored"` in the json (see
`nets/bullet-kb4.json`). Without them the net is unbucketed, as 3.8's is.

## CPU target

`.cargo/config.toml` sets `target-cpu=native` (AVX2/BMI2 on the build machine).
For a portable binary, change it to `x86-64-v3` or remove the file. The NNUE
itself runtime-detects AVX-VNNI/AVX2, so it stays correct regardless.

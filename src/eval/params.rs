// NNUE parameters: the weight struct and a loader for the self-describing
// `net.nnue` file (a small header + quantized weight arrays).
//
// The weights are baked into the binary with include_bytes!, so the engine is
// a single self-contained executable. To swap in a retrained net, regenerate
// net.nnue (see tools/gen_net.py) and rebuild - the header carries the quantization
// params, so no source edits are needed for a same-width retrain.

use once_cell::sync::Lazy;

// Hidden width is a compile-time constant because the accumulator and weight
// arrays are fixed-size. The net file's header must match it (asserted below).
pub const N_HIDDEN: usize = 512;

pub struct NnueWeights {
    pub ft_weights: Box<[[i16; N_HIDDEN]; 768]>,
    pub ft_bias: [i16; N_HIDDEN],
    pub head_weights: [i8; 2 * N_HIDDEN],
    // Quantization params, read from the net header rather than hardcoded:
    //   activations clamp to [0, qa]; final cp = round(sum_i32 / (qa * qb)).
    pub qa: i32,
    pub qb: i32,
    pub head_bias_q: i32,
}

// net.nnue layout (all little-endian):
//   header: [n_hidden i32][qa i32][qb i32][head_bias_q i32]   (16 bytes)
//   body:   768*N i16 ft weights, N i16 ft bias, 2N i8 head weights
static RAW: &[u8] = include_bytes!("net.nnue");

pub(crate) static WEIGHTS: Lazy<NnueWeights> = Lazy::new(decode_weights);

fn decode_weights() -> NnueWeights {
    let read_i32 = |i: usize| {
        i32::from_le_bytes([RAW[i], RAW[i + 1], RAW[i + 2], RAW[i + 3]])
    };
    let n_hidden = read_i32(0);
    let qa = read_i32(4);
    let qb = read_i32(8);
    let head_bias_q = read_i32(12);
    assert_eq!(
        n_hidden as usize, N_HIDDEN,
        "net.nnue header n_hidden ({}) != compiled N_HIDDEN ({}); rebuild with a matching net",
        n_hidden, N_HIDDEN
    );

    const FT: usize = 768 * N_HIDDEN;
    let expected = 16 + (FT + N_HIDDEN) * 2 + 2 * N_HIDDEN;
    assert_eq!(RAW.len(), expected, "net.nnue: unexpected size");

    let read_i16 = |i: usize| i16::from_le_bytes([RAW[i], RAW[i + 1]]);
    let mut cursor = 16usize;

    let mut ft = Box::new([[0i16; N_HIDDEN]; 768]);
    for feat in 0..768 {
        for h in 0..N_HIDDEN {
            ft[feat][h] = read_i16(cursor);
            cursor += 2;
        }
    }

    let mut ft_bias = [0i16; N_HIDDEN];
    for h in 0..N_HIDDEN {
        ft_bias[h] = read_i16(cursor);
        cursor += 2;
    }

    let mut head_weights = [0i8; 2 * N_HIDDEN];
    for i in 0..2 * N_HIDDEN {
        head_weights[i] = RAW[cursor] as i8;
        cursor += 1;
    }

    NnueWeights { ft_weights: ft, ft_bias, head_weights, qa, qb, head_bias_q }
}

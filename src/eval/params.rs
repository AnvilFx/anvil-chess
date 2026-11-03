use once_cell::sync::Lazy;

pub const N_HIDDEN: usize = 512;
pub const N_BUCKETS: usize = 8;

// Output bucket by piece count, matching bullet's MaterialCount<N_BUCKETS>.
#[inline]
pub fn output_bucket(occupancy: u64) -> usize {
    (occupancy.count_ones() as usize - 2) / 32usize.div_ceil(N_BUCKETS)
}

pub struct NnueWeights {
    pub ft_weights: Box<[[i16; N_HIDDEN]; 768]>,
    pub ft_bias: [i16; N_HIDDEN],
    pub head_weights: [[i8; 2 * N_HIDDEN]; N_BUCKETS],
    // i16 copy for the SCReLU head, which multiplies in 16-bit lanes.
    pub head_weights_i16: [[i16; 2 * N_HIDDEN]; N_BUCKETS],
    // Activations clamp to [0, qa]; output cp = sum * out_scale / (qa * qb).
    pub qa: i32,
    pub qb: i32,
    pub out_scale: i32,
    // Squared clipped ReLU; SCReLU sums are divided by qa to get back to qa * qb.
    pub screlu: bool,
    pub head_bias_q: [i32; N_BUCKETS],
}

static RAW: &[u8] = include_bytes!("net.nnue");

pub(crate) static WEIGHTS: Lazy<NnueWeights> = Lazy::new(decode_weights);

fn decode_weights() -> NnueWeights {
    let read_i32 = |i: usize| {
        i32::from_le_bytes([RAW[i], RAW[i + 1], RAW[i + 2], RAW[i + 3]])
    };
    let n_hidden = read_i32(0);
    let qa = read_i32(4);
    let qb = read_i32(8);
    let n_buckets = read_i32(12);
    let out_scale = read_i32(16);
    let activation = read_i32(20);
    assert_eq!(
        n_hidden as usize, N_HIDDEN,
        "net.nnue header n_hidden ({}) != compiled N_HIDDEN ({}); rebuild with a matching net",
        n_hidden, N_HIDDEN
    );
    assert_eq!(
        n_buckets as usize, N_BUCKETS,
        "net.nnue header n_buckets ({}) != compiled N_BUCKETS ({}); regenerate with tools/gen_net.py",
        n_buckets, N_BUCKETS
    );
    assert!(activation == 0 || activation == 1, "net.nnue: unknown activation {activation}");

    const FT: usize = 768 * N_HIDDEN;
    let expected = 24 + 4 * N_BUCKETS + (FT + N_HIDDEN) * 2 + N_BUCKETS * 2 * N_HIDDEN;
    assert_eq!(RAW.len(), expected, "net.nnue: unexpected size");

    let mut head_bias_q = [0i32; N_BUCKETS];
    for (b, bias) in head_bias_q.iter_mut().enumerate() {
        *bias = read_i32(24 + 4 * b);
    }

    let read_i16 = |i: usize| i16::from_le_bytes([RAW[i], RAW[i + 1]]);
    let mut cursor = 24 + 4 * N_BUCKETS;

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

    let mut head_weights = [[0i8; 2 * N_HIDDEN]; N_BUCKETS];
    for bucket in head_weights.iter_mut() {
        for w in bucket.iter_mut() {
            *w = RAW[cursor] as i8;
            cursor += 1;
        }
    }

    let head_weights_i16 = head_weights.map(|bucket| bucket.map(i16::from));

    NnueWeights {
        ft_weights: ft, ft_bias, head_weights, head_weights_i16,
        qa, qb, out_scale, screlu: activation == 1, head_bias_q,
    }
}
